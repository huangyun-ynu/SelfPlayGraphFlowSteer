"""One fresh attempt for every task exposed to the repaired engineering defects."""
from pathlib import Path
import hashlib
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import tomllib
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[3]
COUNT = 128
RUN = ROOT / 'state/formal-eval' / (f'webshop-engineering-affected-{COUNT}-' + time.strftime('%Y%m%d-%H%M%S'))
DIRECTOR_URL = 'http://127.0.0.1:18605/v1'
BASELINE = ROOT / 'state/formal-eval/webshop-memory-auto-history-128-20260930-153948'
REPORT = ROOT / 'experiment_versions/reports/webshop-engineering-rerun-20260930'


def save(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def block(text, section, values):
    match = re.search(rf'(?ms)^\[{re.escape(section)}\]\n(.*?)(?=^\[|\Z)', text)
    assert match, section
    body = match.group(1)
    for key, value in values.items():
        line = key + ' = ' + json.dumps(value)
        pattern = rf'(?m)^{re.escape(key)} = .*?$'
        body = re.sub(pattern, lambda _: line, body) if re.search(pattern, body) else line + '\n' + body
    return text[:match.start(1)] + body + text[match.end(1):]


def flat(value, prefix=''):
    result = {}
    for key, item in value.items():
        name = prefix + '.' + key if prefix else key
        if isinstance(item, dict):
            result.update(flat(item, name))
        else:
            result[name] = item
    return result


def status(phase, **extra):
    value = {'phase': phase, 'at': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'run': str(RUN), **extra}
    save(RUN / 'status.json', value)
    print(json.dumps(value), flush=True)


def request(path):
    with urlopen(Request(DIRECTOR_URL.removesuffix('/v1') + path,
                         headers={'Authorization': 'Bearer EMPTY'}), timeout=10) as response:
        return response.read().decode()


def stop(proc):
    if proc is None:
        return True
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=10)
    return proc.poll() is not None


def interrupted(signum, frame):
    raise InterruptedError(f'evaluation launcher signal {signum}')


def main():
    RUN.mkdir(parents=True, exist_ok=False)
    (ROOT / 'state/webshop-engineering-rerun-current.txt').write_text(str(RUN) + '\n')
    shutil.copyfile(__file__, RUN / 'launcher.py')
    (REPORT / 'run-path.txt').write_text(str(RUN) + '\n')
    child = sidecar = None
    shop_port = None
    handles = []
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        assert os.environ.get('DEEPSEEK_API_KEY'), 'Missing DeepSeek credential'
        model = json.loads(request('/v1/models'))['data'][0]
        assert model['id'] == 'Qwen3.5-9B'
        assert Path(model['root']).resolve() == (ROOT.parent / 'models/Qwen3.5-9B').resolve()
        assert model['max_model_len'] == 32768
        request('/health')
        save(RUN / 'director.json', {'base_url': DIRECTOR_URL, 'model': model,
             'physical_gpu': 4, 'owned_by_this_run': False, 'keep_after_run': True})
        gpu_before = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total',
                                              '--format=csv,noheader'], text=True)
        (RUN / 'gpu-before.txt').write_text(gpu_before)
        os.environ.update(SPGFS_ALLOWED_PHYSICAL_GPUS='4', CUDA_VISIBLE_DEVICES='4',
                          SPGFS_RELEASE_PROJECT_GPU_RESERVATIONS='0',
                          TMPDIR=str(ROOT.parent / '.tmp' / ('webshop-v3-' + str(os.getpid()))))
        Path(os.environ['TMPDIR']).mkdir(parents=True, exist_ok=True)
        sources = [p for p in (ROOT / 'src').rglob('*') if p.is_file() and '__pycache__' not in p.parts]
        sources += list((ROOT / 'configs/templates').glob('*.jinja'))
        hashes = {}
        for source in sorted(sources):
            rel = source.relative_to(ROOT)
            target = RUN / 'evaluated_source' / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            hashes[str(rel)] = sha(target)
        frozen_src = str(RUN / 'evaluated_source/src')
        os.environ['PYTHONPATH'] = frozen_src
        os.environ['SPGFS_WEBSHOP_MEMORY_DIR'] = str(RUN / 'memory-store')
        sys.path.insert(0, frozen_src)
        dataset = ROOT / 'data/formal/eval/webshop_official_test_128.jsonl'
        assert sha(dataset) == '35fbda3aef098050f649ca569a8c8e1c8f3a4dbf0e8c800ba3fade74f7335328'
        rows = [json.loads(s) for s in dataset.read_text().splitlines() if s.strip()]
        assert len(rows) == len({r['id'] for r in rows}) == 128
        selection = json.loads((REPORT / 'selection.json').read_text())
        assert selection['source_run'] == BASELINE.name and selection['tasks'] == COUNT
        assert sha(BASELINE / 'results/records.jsonl') == selection['source_records_sha256']
        ids = set(selection['ids'])
        rows = [r for r in rows if r['id'] in ids]
        assert len(rows) == COUNT and {r['id'] for r in rows} == ids
        (RUN / 'tasks.jsonl').write_bytes(dataset.read_bytes())
        save(RUN / 'selection.json', selection)
        shutil.copyfile(BASELINE / 'per-task.json', RUN / 'baseline_current.json')
        prior_hashes = json.loads((BASELINE / 'manifest.json').read_text())['source_sha256']
        changed = sorted(key for key in hashes.keys() | prior_hashes.keys() if hashes.get(key) != prior_hashes.get(key))
        expected = {'src/selfplay_graph_flowsteer/' + n for n in ['runtime.py', 'webshop_action_reserve.py',
            'webshop_sidecar.py', 'webshop_navigation.py', 'webshop_memory.py', 'webshop_memory_projection.py']}
        assert set(changed) == expected, changed
        save(RUN / 'version_diff.json', {'baseline': str(BASELINE), 'changed_sources': changed,
            'previously_committed_memory_guards': ['webshop_memory.py', 'webshop_memory_projection.py'],
            'engineering_commit': 'd662f39', 'runtime_policy': 'completion_reserve_v2'})
        sys.path.insert(0, str(ROOT / 'scripts/formal'))
        from webshop_dataset_index import public_goal_order, validate_records
        assert not validate_records(rows, *public_goal_order(Path(os.environ['SPGFS_WEBSHOP_GOALS'])))
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            shop_port = sock.getsockname()[1]
        formal = ROOT / 'configs/formal_training.toml'
        shutil.copyfile(formal, RUN / 'formal_config.source.toml')
        # Reuse the precise prior evaluated configuration; only run-local destinations change.
        text = (BASELINE / 'config.toml').read_text()
        text = block(text, 'webshop', {'service_url': f'http://127.0.0.1:{shop_port}'})
        text = block(text, 'runtime_routing', {'health_state_path': str(RUN / 'route_health.json')})
        text = block(text, 'canvas', {'submission_journal_dir': str(RUN / 'submissions')})
        text = block(text, 'trace', {'path': str(RUN / 'traces.jsonl')})
        config_path = RUN / 'config.toml'
        config_path.write_text(text)
        from selfplay_graph_flowsteer.application import load_adaptive_config
        config = load_adaptive_config(config_path)
        config.validate()
        assert config.director_prompt_variant == 'v3'
        assert config.canvas.for_dataset('webshop').submission_protocol == 'unified_task_result_v1'
        assert config.canvas.for_dataset('webshop').action_budget_policy == 'shared_total_v1'
        assert 'webshop' not in config.canvas.worker_token_budget_by_dataset
        assert config.webshop.compatibility_profile == 'm02_merged_identity_v1'
        assert config.webshop.worker_guidance_policy == 'merged_checklist_v1'
        assert config.webshop.max_total_calls == 16
        assert config.webshop.purchase_budget_policy == 'completion_reserve_v2'
        assert config.webshop.scheduling_policy == 'bounded_research_v1'
        assert config.runtime_pool()['deepseek'].enable_thinking is False
        assert config.runtime_pool()['deepseek'].max_concurrency == 50
        before_config, after_config = flat(tomllib.loads((BASELINE / 'config.toml').read_text())), flat(tomllib.loads(text))
        config_diff = {k: {'before': before_config.get(k), 'after': after_config.get(k)}
            for k in sorted(before_config.keys() | after_config.keys()) if before_config.get(k) != after_config.get(k)}
        assert set(config_diff) == {'webshop.service_url', 'runtime_routing.health_state_path', 'canvas.submission_journal_dir', 'trace.path'}
        save(RUN / 'comparison-controls.json', {'config_differences': config_diff,
             'tasks_byte_identical': sha(RUN / 'tasks.jsonl') == sha(BASELINE / 'tasks.jsonl')})
        a, b = flat(tomllib.loads(formal.read_text())), flat(tomllib.loads(text))
        overrides = {k: {'formal': a.get(k), 'run': b.get(k)} for k in sorted(a.keys() | b.keys()) if a.get(k) != b.get(k)}
        patch = subprocess.check_output(['git', 'diff', 'HEAD', '--binary'], cwd=ROOT)
        (RUN / 'development.patch').write_bytes(patch)
        save(RUN / 'manifest.json', {'source_commit': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
             'development_patch_sha256': hashlib.sha256(patch).hexdigest(),
             'source_sha256': hashes, 'frozen_source': frozen_src,
             'formal_config_sha256': sha(formal), 'config_sha256': sha(config_path), 'config_overrides': overrides,
             'full_dataset_sha256': sha(dataset), 'dataset_sha256': sha(RUN / 'tasks.jsonl'), 'ids': [r['id'] for r in rows], 'planned': COUNT, 'seed': 0,
             'question_concurrency': 40, 'worker_route': 'deepseek', 'worker_model': 'deepseek-flash',
             'worker_thinking': False, 'route_concurrency': 50, 'director_thinking': True, 'director_context': 32768,
             'skills': 'off', 'tool_budget': 16, 'worker_token_limit': 350000,
             'worker_reported_usage_threshold_enabled': False, 'profile': 'm02_merged_identity_v1',
             'submission_protocol': 'unified_task_result_v1', 'global_director_variant': 'v3',
             'public_goal_mapping_mismatches': 0, 'evaluation_only': True, 'parameter_updates': 0,
             'fresh_attempts_per_task': 1, 'retry_completed_failures': False,
             'purchase_budget_policy': 'completion_reserve_v2', 'scheduling_policy': 'bounded_research_v1',
             'research_action_limit': 8, 'worker_memory_policy': 'factual_memory_v2', 'memory_projection_chars': 6000, 'memory_delivery': 'automatic', 'memory_tool_added': False,
             'memory_store_dir': 'memory-store',
             'prior_fixes_retained': ['candidate_title_label_fallback', 'completion_balance_and_fuse', 'completion_reserve_v2', 'bounded_research_v1', 'action_repair_feedback'],
             'baseline_run': str(BASELINE), 'change_under_test': 'Navigation semantics, section reservation retention, optional plan recovery and option/ASIN parsing; cumulative memory safeguards retained',
             'comparison_note': 'Same 128 tasks, exact prior config except run-local paths and port; one fresh attempt each, no score-based retries. Also report the 59-task direct-issue subgroup.'})
        args = [sys.executable, '-m', 'selfplay_graph_flowsteer.webshop_sidecar', '--host', '127.0.0.1', '--port', str(shop_port)]
        for flag, name in [('interpreter','INTERPRETER'),('worker-script','WORKER'),('source-root','SOURCE_ROOT'),
                           ('source-revision','SOURCE_REVISION'),('store','STORE'),('goals','GOALS'),('index','INDEX'),('java-home','JAVA_HOME')]:
            args += ['--'+flag, os.environ['SPGFS_WEBSHOP_'+name]]
        args += ['--worker-timeout','180','--max-sessions','192','--max-initializers','4']
        log = (RUN / 'sidecar.log').open('w'); handles.append(log)
        sidecar = subprocess.Popen(args,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        save(RUN / 'sidecar.json', {'pid':sidecar.pid,'args':args,'owned_by_this_run':True})
        status('sidecar_starting')
        deadline = time.monotonic()+120
        while True:
            assert sidecar.poll() is None, 'WebShop sidecar exited'
            try:
                with urlopen(f'http://127.0.0.1:{shop_port}/health',timeout=3) as response:
                    health=json.load(response)
                assert health['status'] == 'ok'
                assert health['idempotency_protocol'] == 'webshop-request-v1'
                assert Path(health['index_path']).resolve() == Path(os.environ['SPGFS_WEBSHOP_INDEX']).resolve()
                save(RUN / 'sidecar-health.json',health)
                break
            except OSError:
                assert time.monotonic()<deadline, 'WebShop sidecar readiness timeout'
                time.sleep(2)
        args=[sys.executable,'-m','selfplay_graph_flowsteer','benchmark','--config',str(config_path),
              '--dataset',str(RUN/'tasks.jsonl'),'--output',str(RUN/'results'),'--workers','40','--seed','0',
              '--worker-route','deepseek','--director-thinking','--director-base-url',DIRECTOR_URL,
              '--director-api-key','EMPTY','--director-model','Qwen3.5-9B',
              '--wandb-mode','disabled','--skill-context','off','--disable-swe']
        log=(RUN/'benchmark.log').open('w'); handles.append(log)
        child=subprocess.Popen(args,cwd=ROOT,env=dict(os.environ),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        save(RUN/'child.json',{'pid':child.pid,'args':args})
        status('running',planned=COUNT,completed=0)
        previous=None
        while child.poll() is None:
            time.sleep(15)
            path=RUN/'results/run_state.json'
            state=json.loads(path.read_text()) if path.exists() else {}
            counts=(state.get('completed',0),state.get('failed',0))
            if counts!=previous:
                status('running',planned=COUNT,completed=counts[0],errors=counts[1]); previous=counts
        status('benchmark_exited',returncode=child.returncode)
        save(RUN/'memory-store-manifest.json', {str(p.relative_to(RUN)): sha(p) for p in (RUN/'memory-store').rglob('*.json')})
        save(RUN/'source_integrity.json', {'frozen_source_unchanged':all(sha(RUN/'evaluated_source'/rel)==h for rel,h in hashes.items()),
             'formal_source_still_matches_snapshot':all(sha(ROOT/rel)==h for rel,h in hashes.items())})
        return child.returncode
    except BaseException as exc:
        status('launcher_failed',error=type(exc).__name__,message=str(exc))
        raise
    finally:
        child_stopped=stop(child)
        sidecar_stopped=stop(sidecar)
        for handle in handles: handle.close()
        closed=True
        if shop_port is not None:
            with socket.socket() as sock:
                closed=sock.connect_ex(('127.0.0.1',shop_port)) != 0
        try:
            request('/health'); director_healthy=True
        except Exception:
            director_healthy=False
        save(RUN/'cleanup.json',{'benchmark_stopped':child_stopped,'sidecar_stopped':sidecar_stopped,
             'sidecar_port':shop_port,'sidecar_port_closed':closed,
             'reused_director_retained':True,'director_healthy':director_healthy})


if __name__=='__main__':
    raise SystemExit(main())
