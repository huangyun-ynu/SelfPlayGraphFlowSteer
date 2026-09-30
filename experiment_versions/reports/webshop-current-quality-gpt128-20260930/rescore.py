"""Independently rescore purchased terminal states, after the model run."""
import importlib.util
import json
from pathlib import Path
import random
import re
import sqlite3
import sys
import types

ROOT = Path(__file__).resolve().parents[3]
REPORT = Path(__file__).resolve().parent
RUN = Path((REPORT / 'run-path.txt').read_text().strip())
SOURCE = ROOT / 'assets/webshop/source'
for name, path in [('web_agent_site', SOURCE / 'web_agent_site'), ('web_agent_site.engine', SOURCE / 'web_agent_site/engine')]:
    mod = types.ModuleType(name)
    mod.__path__ = [str(path)]
    sys.modules[name] = mod
from web_agent_site.engine.goal import get_reward

spec = importlib.util.spec_from_file_location('quality', RUN / 'evaluated_source/src/selfplay_graph_flowsteer/webshop_quality.py')
quality = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quality)

goals = [json.loads(line) for line in (RUN / 'goals.jsonl').open()]
random.Random(233).shuffle(goals)
db = sqlite3.connect('file:' + str(ROOT / 'assets/webshop/prepared/products.sqlite3') + '?mode=ro', uri=True)
records = [json.loads(line) for line in (RUN / 'results/records.jsonl').open()]
assert len(records) == 128
rows = []
for record in records:
    env = record['trajectory']['task']['metadata'].get('webshop_environment_result') or {}
    if not env.get('purchased'):
        continue
    page = env['page_text']
    asin = re.search(r'asin \[SEP\] (\w+)', page).group(1)
    options = json.loads(page.split('options [SEP] ')[1].split(' [SEP] ')[0])
    goal = goals[int(record['task_id'].rsplit('-', 1)[1])]
    assert goal['instruction_text'] == record['trajectory']['task']['prompt']
    data, price = db.execute('SELECT product_json,price FROM products WHERE asin=?', (asin,)).fetchone()
    product = json.loads(data)
    revised, details = quality.evaluate(product, goal, price, options)
    official, official_parts = get_reward(product, goal['_official_goal'], price, options, verbose=True)
    assert abs(revised - record['score']) < 1e-9, (record['task_id'], revised, record['score'])
    assert abs(official - env['scoring']['official_reward']) < 1e-9, (record['task_id'], official)
    rows.append({'task_id': record['task_id'], 'asin': asin, 'selected_options': options,
                 'score': revised, 'official_reward_same_purchase': official,
                 'quality_parts': details, 'official_parts': official_parts})
result = {'purchases': len(rows), 'quality_scores_match': True, 'original_label_scores_match': True,
          'note': 'Original-label rewards for these same purchases are not a fresh original-task evaluation.', 'rows': rows}
for path in [RUN / 'rescore.json', REPORT / 'rescore.json']:
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({k: v for k, v in result.items() if k != 'rows'}))
