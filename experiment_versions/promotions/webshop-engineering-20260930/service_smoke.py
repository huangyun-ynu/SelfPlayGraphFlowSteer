"""Verify the promoted HTTP service with one captured real successful task; no LLM."""
from pathlib import Path
import json,os,socket,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts/formal'))
from selfplay_graph_flowsteer.webshop import WebShopHTTPClient
from check_webshop_official_service import validate_health
from selfplay_graph_flowsteer.webshop_sidecar import implementation_sha256
import hashlib
BASE=ROOT.parent/'SelfPlayGraphFlowSteer-webshop-memory/state/formal-eval/webshop-engineering-affected-128-20260930-164051'
record=next(json.loads(line)for line in (BASE/'results/records.jsonl').open()if json.loads(line)['task_id']=='webshop/goal-00358')
arts={}
def walk(v):
 if isinstance(v,dict):
  if 'artifact_id'in v and 'react_trace'in v:arts[v['artifact_id']]=v
  for x in v.values():walk(x)
 elif isinstance(v,list):
  for x in v:walk(x)
walk(record['trajectory']['events'])
with socket.socket()as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
args=[sys.executable,'-m','selfplay_graph_flowsteer.webshop_sidecar','--host','127.0.0.1','--port',str(port),'--max-sessions','4','--max-initializers','1']
for flag,variable in [('interpreter','INTERPRETER'),('worker-script','WORKER'),('source-root','SOURCE_ROOT'),('source-revision','SOURCE_REVISION'),('store','STORE'),('goals','GOALS'),('index','INDEX'),('java-home','JAVA_HOME')]:args+=['--'+flag,os.environ['SPGFS_WEBSHOP_'+variable]]
log=(OUT/'sidecar-smoke.log').open('w');proc=subprocess.Popen(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONPATH=str(ROOT/'src'),CUDA_VISIBLE_DEVICES=''))
sid=None;client=WebShopHTTPClient(f'http://127.0.0.1:{port}',timeout_s=180)
try:
 deadline=time.monotonic()+30
 while True:
  try:health=client.health();break
  except Exception:
   assert proc.poll()is None and time.monotonic()<deadline
   time.sleep(.2)
 expected={'scorer_version':'official','goals_sha256':hashlib.sha256(Path(os.environ['SPGFS_WEBSHOP_GOALS']).read_bytes()).hexdigest(),'implementation_sha256':implementation_sha256(),'index_path':str(Path(os.environ['SPGFS_WEBSHOP_INDEX']).resolve())}
 validate_health(health,expected)
 state=client.create_session('goal-00358',seed=0);sid=state['session_id'];executed=0
 for a in arts.values():
  for t in a['react_trace']:
   if t['observation']['status']!='ok':continue
   name=t['action']['name'];arguments=t['action']['arguments']
   state=client.search(sid,arguments['query'])if name=='webshop_search'else client.click(sid,arguments['target_id'])
   executed+=1
 assert state['purchased'] and abs(state['reward']-record['score'])<1e-9
 result={'task_id':record['task_id'],'captured_successful_actions':executed,'purchased':state['purchased'],'official_reward':state['reward'],'expected_recorded_reward':record['score'],'scorer_version':health['scorer_version'],'implementation_sha256':health['implementation_sha256'],'model_calls':0,'test_kind':'deterministic public action replay, not a fresh model score'}
finally:
 if sid:client.close_session(sid)
 proc.terminate();proc.wait(timeout=30);log.close()
result['owned_service_stopped']=proc.poll()is not None
(OUT/'service-smoke.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
