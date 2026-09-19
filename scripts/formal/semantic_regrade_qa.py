#!/usr/bin/env python3
import argparse, concurrent.futures, glob, json, os, re, time
from pathlib import Path
from urllib.request import Request, urlopen

PROMPT = '''You are a strict QA answer-equivalence judge. Decide whether the predicted answer fully answers the question and is semantically equivalent to at least one gold answer.
Accept harmless wording, punctuation, articles, date prepositions (e.g. "in 2015" vs "2015"), and well-known aliases only when the prediction is complete. Reject empty answers, partial answers, missing entities in a list, wrong qualifiers, extra unrelated answers, and answers that merely overlap.
Examples: gold "Since 2015", pred "2015" => true; gold "Dizzy Dean", pred "Dizzy" => false; gold "A, B, C", pred "A and B" => false; gold "Giotto", pred "" => false; gold "Art Deco-style skyscraper", pred "Art Deco" => false.
Return exactly one line of JSON: {{"match": true}} or {{"match": false}}. Do not explain.

Question: {question}
Gold answer(s): {gold}
Predicted answer: {pred}
'''

def norm(s):
    s = str(s or '').lower()
    s = re.sub(r'[^\w\s]', ' ', s, flags=re.UNICODE)
    return ' '.join(s.split())

def em(pred, gold):
    ps = norm(pred)
    gs = gold if isinstance(gold, list) else [gold]
    return int(any(ps == norm(g) for g in gs))

def f1(pred, gold):
    p = norm(pred).split()
    best = 0.0
    for g in (gold if isinstance(gold, list) else [gold]):
        q = norm(g).split()
        if not p or not q: continue
        common = {}
        for t in p:
            if t in q:
                common[t] = min(p.count(t), q.count(t))
        n = sum(common.values())
        if n:
            pr, rc = n / len(p), n / len(q)
            best = max(best, 2 * pr * rc / (pr + rc))
    return best

def judge(item, endpoint, model, timeout):
    rec = item['record']; task = rec['trajectory']['task']
    gold, pred, question = task.get('reference', ''), rec.get('answer', ''), task.get('prompt', '')
    strict = em(pred, gold)
    payload = {'model': model, 'temperature': 0, 'max_tokens': 256,
               'chat_template_kwargs': {'enable_thinking': False},
               'messages': [{'role': 'user', 'content': PROMPT.format(question=question, gold=gold, pred=pred)}]}
    api_key = (os.environ.get('DEEPSEEK_API_KEY') or os.environ.get('OPENAI_API_KEY') or 'local')
    req = Request(endpoint.rstrip('/') + '/chat/completions', data=json.dumps(payload).encode(),
                  headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + api_key})
    semantic = None; raw = ''
    if strict:
        semantic = 1
    elif not str(pred).strip():
        semantic = 0
    else:
        try:
            with urlopen(req, timeout=timeout) as r:
                data = json.loads(r.read())
            msg = data['choices'][0]['message']
            raw = msg.get('content', '') or msg.get('reasoning_content', '')
            m = re.findall(r'"match"\s*:\s*(true|false)', raw, re.I)
            semantic = int(bool(m and m[-1].lower() == 'true'))
        except Exception as e:
            raw = 'ERROR: ' + repr(e)
    return {'job_key': item['job_key'], 'dataset': item['dataset'], 'example_id': item['example_id'],
            'question': question, 'gold': gold, 'prediction': pred, 'strict_em': strict,
            'f1': f1(pred, gold), 'semantic_match': semantic, 'judge_raw': raw}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--run-dir', required=True)
    ap.add_argument('--output', default='semantic-regrade-qwen35-9b.jsonl')
    ap.add_argument('--endpoint', default=os.environ.get('SPGFS_SEMANTIC_JUDGE_ENDPOINT', 'https://api.deepseek.com/v1'))
    ap.add_argument('--model', default=os.environ.get('SPGFS_SEMANTIC_JUDGE_MODEL', 'deepseek-flash'))
    ap.add_argument('--workers', type=int, default=2); ap.add_argument('--timeout', type=int, default=180)
    args = ap.parse_args(); out = Path(args.run_dir) / args.output
    items = [json.load(open(p)) for p in glob.glob(str(Path(args.run_dir) / 'samples' / '*.json'))]
    items.sort(key=lambda x: x['job_key'])
    done = {}
    if out.exists():
        for line in out.read_text().splitlines():
            if line.strip():
                x = json.loads(line); done[x['job_key']] = x
    todo = [x for x in items if x['job_key'] not in done]
    print(f'items={len(items)} done={len(done)} todo={len(todo)} endpoint={args.endpoint}', flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex, out.open('a') as f:
        futs = [ex.submit(judge, x, args.endpoint, args.model, args.timeout) for x in todo]
        for i, fut in enumerate(concurrent.futures.as_completed(futs), 1):
            x = fut.result(); f.write(json.dumps(x, ensure_ascii=False) + '\n'); f.flush()
            if i % 10 == 0 or i == len(futs): print(f'completed={len(done)+i}/{len(items)}', flush=True)
    rows = list(done.values())
    rows += [json.loads(x) for x in out.read_text().splitlines() if x.strip() and json.loads(x)['job_key'] not in done]
    for ds in sorted(set(x['dataset'] for x in rows)):
        z = [x for x in rows if x['dataset'] == ds]; n=len(z)
        print(ds, n, 'EM', sum(x['strict_em'] for x in z)/n, 'F1', sum(x['f1'] for x in z)/n,
              'semantic', sum(x['semantic_match'] or 0 for x in z)/n, 'judge_errors', sum(x['semantic_match'] is None for x in z), flush=True)

if __name__ == '__main__': main()
