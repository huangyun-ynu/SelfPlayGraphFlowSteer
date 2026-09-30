"""Offline official scorer: hidden goals enter only after the real run ends."""
import json
import random
import re
import sqlite3
import sys
import types
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
RUN = Path((Path(__file__).resolve().parent/'run-path.txt').read_text().strip())
SOURCE = ROOT/'assets/webshop/source'
for name, path in [('web_agent_site', SOURCE/'web_agent_site'), ('web_agent_site.engine', SOURCE/'web_agent_site/engine')]:
    m=types.ModuleType(name);m.__path__=[str(path)];sys.modules[name]=m
from web_agent_site.engine.goal import get_reward
records=[json.loads(line) for line in (RUN/'results/records.jsonl').open()]
goals=[json.loads(line) for line in (ROOT/'assets/webshop/prepared/goals.jsonl').open()]
random.Random(233).shuffle(goals)
db=sqlite3.connect(f'file:{ROOT}/assets/webshop/prepared/products.sqlite3?mode=ro',uri=True)
rows=[]
for r in records:
    env=r['trajectory']['task']['metadata'].get('webshop_environment_result') or {}
    if not env.get('purchased'):continue
    page=env['page_text']
    asin=re.search(r'asin \[SEP\] (\w+)',page).group(1)
    options=json.loads(page.split('options [SEP] ')[1].split(' [SEP] ')[0])
    goal=goals[int(r['task_id'].rsplit('-',1)[1])]
    assert goal['instruction_text']==r['trajectory']['task']['prompt']
    data,price=db.execute('select product_json,price from products where asin=?',(asin,)).fetchone()
    score,parts=get_reward(json.loads(data),goal,price,options,verbose=True)
    assert abs(score-r['score'])<1e-9,(r['task_id'],score,r['score'])
    rows.append(dict(task_id=r['task_id'],score=score,parts=parts,asin=asin,selected_options=options))
for path in [RUN/'official-rescore.json',Path(__file__).resolve().parent/'official-rescore.json']:
    path.write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'purchases':len(rows),'official_scores_match':True}))
