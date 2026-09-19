#!/usr/bin/env python3
import argparse, concurrent.futures, glob, json, re
from pathlib import Path
from urllib.request import Request, urlopen

CURRENT = ('Extract the shortest answer span that directly answers the question. Preserve necessary '
           'units, date ranges, qualifications and precision. Do not add explanations. Return JSON {"answer":"..."}.')
COMPLETE = ('Extract the minimum complete answer that fully answers the question. Preserve all required '
            'entities in a list, full entity names, date ranges and qualifiers, locations, units, numerical '
            'precision, and yes/no polarity. Do not delete information needed for correctness and do not add '
            'unrelated facts or explanations. Return JSON {"answer":"..."}.')
JUDGE = '''Judge whether prediction fully answers the question and is semantically equivalent to a gold answer. Reject empty/partial answers, missing list entities, wrong qualifiers, and mere overlap. Accept harmless punctuation/articles/date prepositions and true aliases. Return only {"match": true} or {"match": false}.
Question: {q}
Gold: {g}
Prediction: {p}'''

def norm(s): return ' '.join(re.sub(r'[^\w\s]', ' ', str(s or '').lower()).split())
def em(p,g): return int(norm(p)==norm(g) if isinstance(g,str) else any(norm(p)==norm(x) for x in g))
def f1(p,g):
    best=0.0
    for x in ([g] if isinstance(g,str) else g):
        a,b=norm(p).split(),norm(x).split(); n=sum(min(a.count(t),b.count(t)) for t in set(a)&set(b))
        if n: best=max(best,2*(n/len(a))*(n/len(b))/((n/len(a))+(n/len(b))))
    return best
def call(endpoint, model, prompt, q, raw, max_tokens=128):
    body={'model':model,'temperature':0,'max_tokens':max_tokens,'chat_template_kwargs':{'enable_thinking':False},'messages':[{'role':'system','content':prompt},{'role':'user','content':json.dumps({'question':q,'raw_answer':raw},ensure_ascii=False)}]}
    req=Request(endpoint.rstrip('/')+'/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json','Authorization':'Bearer local'})
    with urlopen(req,timeout=180) as r: d=json.loads(r.read())
    return d['choices'][0]['message'].get('content','')
def parse(s):
    m=re.findall(r'\{\s*"answer"\s*:\s*"(.*?)"\s*\}',s,re.S)
    return m[-1].strip() if m else str(s).strip()
def judge(endpoint,model,q,g,p):
    if not p.strip(): return 0
    s=call(endpoint,model,JUDGE,q,json.dumps({'gold':g,'prediction':p},ensure_ascii=False),80)
    m=re.findall(r'"match"\s*:\s*(true|false)',s,re.I); return int(m and m[-1].lower()=='true')
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--run-dir',required=True); ap.add_argument('--endpoint',default='http://127.0.0.1:18603/v1'); ap.add_argument('--model',default='Qwen3.5-9B'); ap.add_argument('--workers',type=int,default=2); args=ap.parse_args()
    out=Path(args.run_dir)/'formatter-ablation-qwen35-9b.jsonl'; items=[]
    for p in glob.glob(str(Path(args.run_dir)/'samples/*.json')):
        x=json.load(open(p)); r=x['record']; t=r['trajectory']['task']; a=r['trajectory']['answer_submission']; items.append({'job_key':x['job_key'],'dataset':x['dataset'],'q':t['prompt'],'g':t['reference'],'raw':a['raw_answer']})
    def one(item,kind):
        prompt=CURRENT if kind=='current' else COMPLETE
        pred=parse(call(args.endpoint,args.model,prompt,item['q'],item['raw']))
        return {**item,'kind':kind,'prediction':pred,'em':em(pred,item['g']),'f1':f1(pred,item['g']),'changed':int(norm(pred)!=norm(item['raw']))}
    jobs=[(x,k) for x in items for k in ('current','complete')]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex, out.open('w') as f:
        for i,fu in enumerate([ex.submit(one,x,k) for x,k in jobs],1):
            z=fu.result(); f.write(json.dumps(z,ensure_ascii=False)+'\n'); f.flush()
            if i%50==0: print(f'formatted={i}/{len(jobs)}',flush=True)
    rows=[json.loads(x) for x in out.read_text().splitlines()];
    for kind in ('current','complete'):
        for ds in sorted(set(x['dataset'] for x in rows)):
            z=[x for x in rows if x['kind']==kind and x['dataset']==ds]; n=len(z)
            # Semantic judge only for non-EM candidates, same local Qwen judge.
            sem=sum(x['em'] for x in z)
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
                fs=[ex.submit(judge,args.endpoint,args.model,x['q'],x['g'],x['prediction']) for x in z if not x['em']]
                sem += sum(f.result() for f in fs)
            print(kind,ds,n,'EM',sum(x['em'] for x in z)/n,'F1',sum(x['f1'] for x in z)/n,'semantic',sem/n,'changed',sum(x['changed'] for x in z)/n,'avg_len',sum(len(x['prediction'].split()) for x in z)/n,flush=True)
if __name__=='__main__': main()
