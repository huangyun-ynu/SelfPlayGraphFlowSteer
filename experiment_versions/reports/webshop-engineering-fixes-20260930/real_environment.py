"""Real official-environment checks using captured public prefixes; no LLM calls.

Post-prefix actions are explicit engineering probes, not fresh model solutions.
"""
import gzip
import json
import sys
from pathlib import Path

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
sys.path.insert(0,str(ROOT/'src'))
from selfplay_graph_flowsteer.webshop_sidecar import OfficialWorker, ProductStore, WebShopSession
from selfplay_graph_flowsteer.webshop_action_reserve import PurchaseReservation, FLEXIBLE_POLICY

CASES=json.loads(gzip.decompress((ROOT/'tests/fixtures/webshop_engineering_128_real_trajectories.json.gz').read_bytes()))['cases']


def main():
    rows=[]
    for number,word in [('00015','terracotta'),('00234','lieutenant'),('00318','cappuccino'),('00358','cantaloupe'),('00365','cottonwood')]:
        worker=OfficialWorker(interpreter=ROOT/'assets/webshop/venv/bin/python',
            worker_script=ROOT/'assets/webshop/private-evaluation/src/skillev_private/benchmarks/official_environment_worker.py',
            source_root=ROOT/'assets/webshop/source',source_revision='64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd',
            store_path=ROOT/'assets/webshop/prepared/products.sqlite3',goals_path=ROOT/'assets/webshop/prepared/goals.jsonl',
            index_path=ROOT/'assets/webshop/source/search_engine/indexes',goal_index=int(number),seed=0,timeout_s=180,
            java_home=ROOT/'assets/java/jdk-11.0.32.1+1')
        session=WebShopSession('engineering-'+number,worker,ProductStore(ROOT/'assets/webshop/prepared/products.sqlite3'),'webshop/goal-'+number)
        try:
            state=session.reset()
            assert session.instruction==CASES[number]['task']
            prefix=0
            for t in CASES[number]['trace']:
                if t['observation']['status']!='ok':continue
                name,args=t['action']['name'],t['action']['arguments']
                state=session.search(args['query']) if name=='webshop_search' else session.click(args['target_id'])
                prefix+=1
                if any(a.get('option_value')==word for a in state['valid_subactions']):break
            selected=next(a for a in state['valid_subactions']if a.get('option_value')==word)
            asin=state['product']['asin']
            assert selected['kind']=='select_option'
            assert not any(a['kind']=='open_product' and a.get('asin')==word.upper()for a in state['valid_subactions'])
            state=session.click(selected['target_id'])
            assert state['product']['asin']==asin and state['selected_options']['color']==word
            manager=PurchaseReservation(policy=FLEXIBLE_POLICY)
            feature=next(a for a in state['valid_subactions']if a['target_id'].startswith('view_features:'))
            error,_=manager.preflight(owner='agent',binding='live',state=state,name='webshop_click',
                arguments={'target_id':feature['target_id'],'completion_plan':{'decision':'reserve','asin':asin,'options':{'color':word}}},
                remaining=16-prefix-1,task_result=True,task=CASES[number]['task'])
            assert error is None
            state=session.click(feature['target_id'])
            manager.observe('agent','live',state)
            assert manager.plan is not None and manager.reserved()==2
            back=next(a for a in state['valid_subactions']if a['target_id'].startswith('previous_page:'))
            assert back['navigation_effect']=='return_to_current_product_page'
            state=session.click(back['target_id'])
            manager.observe('agent','live',state)
            assert manager.reserved()==1 and state['selected_options']['color']==word
            back=next(a for a in state['valid_subactions']if a['target_id'].startswith('previous_page:'))
            assert back['navigation_effect']=='return_to_search_results'
            state=session.click(back['target_id'])
            assert state['page_type']=='search_results' and 'product' not in state and not session.selected_options
            rows.append(dict(task=number,asin=asin,option=word,captured_prefix_steps=prefix,
                extra_probe_steps=4,option_selected=True,section_reservation_retained=True,
                section_return_retains_selection=True,product_return_clears_selection=True,worker_pid=worker._process.pid))
            print(json.dumps(rows[-1]),flush=True)
        finally:
            session.close()
        assert worker._process.poll() is not None
        rows[-1]['owned_worker_stopped']=True
        (OUT/'real-environment.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'checks_passed':len(rows),'model_calls':0,'purchase_calls':0,'all_workers_stopped':all(r['owned_worker_stopped']for r in rows)}),flush=True)


if __name__=='__main__':main()
