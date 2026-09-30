"""Freeze current cumulative code and run the sidebar's corrected held-out tasks."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import random
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
REPORT = Path(__file__).resolve().parent
VERSION = 'webshop-quality-20260930-v1'
RELEASE = ROOT / 'state/formal-data' / VERSION
DIRECTOR_URL = 'http://127.0.0.1:18605/v1'
RUN = ROOT / 'state/formal-eval' / ('webshop-current-quality-gpt128-' + datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%SZ'))


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


def request(url):
    with urlopen(Request(url, headers={'Authorization': 'Bearer EMPTY'}), timeout=10) as response:
        return response.read().decode()


def status(phase, **extra):
    value = {'phase': phase, 'at': datetime.now(timezone.utc).isoformat(), 'run': str(RUN), **extra}
    save(RUN / 'status.json', value)
    print(json.dumps(value), flush=True)


def stop(proc):
    if proc is not None and proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=10)
    return proc is None or proc.poll() is not None


def interrupted(signum, frame):
    raise InterruptedError(f'launcher signal {signum}')


def main():
    RUN.mkdir(parents=True, exist_ok=False)
    (REPORT / 'run-path.txt').write_text(str(RUN) + '\n')
    (ROOT / 'state/webshop-current-quality-gpt-current.txt').write_text(str(RUN) + '\n')
    shutil.copyfile(__file__, RUN / 'launcher.py')
    child = sidecar = None
    handles = []
    port = None
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        assert os.environ.get('NEXUS_API_KEY'), 'Missing Nexus credential'
        assert os.environ.get('FLOWSTEER_API_KEY'), 'Missing student gateway credential'
        model = json.loads(request(DIRECTOR_URL + '/models'))['data'][0]
        assert model['id'] == 'Qwen3.5-9B' and model['max_model_len'] == 32768
        assert Path(model['root']).resolve() == (ROOT.parent / 'models/Qwen3.5-9B').resolve()
        request(DIRECTOR_URL.removesuffix('/v1') + '/health')
        save(RUN / 'director.json', {'base_url': DIRECTOR_URL, 'model': model, 'owned_by_this_run': False})
        release = json.loads((RELEASE / 'manifest.json').read_text())
        for name, expected in release['files'].items():
            assert sha(RELEASE / name) == expected, name
        dataset = ROOT / 'data/formal/eval/webshop_quality_test_128.jsonl'
        assert sha(dataset) == release['files']['webshop_quality_test_128.jsonl']
        sources = [p for p in (ROOT / 'src').rglob('*') if p.is_file() and '__pycache__' not in p.parts]
        sources += list((ROOT / 'configs/templates').glob('*.jinja'))
        hashes = {}
        for source in sorted(sources):
            rel = source.relative_to(ROOT)
            target = RUN / 'evaluated_source' / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            hashes[str(rel)] = sha(target)
        assert hashes['src/selfplay_graph_flowsteer/webshop_quality.py'] == release['scorer_sha256']
        frozen_src = str(RUN / 'evaluated_source/src')
        os.environ.update(PYTHONPATH=frozen_src, CUDA_VISIBLE_DEVICES='', SPGFS_RELEASE_PROJECT_GPU_RESERVATIONS='0',
                          SPGFS_WEBSHOP_MEMORY_DIR=str(RUN / 'memory-store'))
        sys.path.insert(0, frozen_src)
        shutil.copyfile(dataset, RUN / 'tasks.jsonl')
        shutil.copyfile(RELEASE / 'goals.jsonl', RUN / 'goals.jsonl')
        shutil.copyfile(RELEASE / 'manifest.json', RUN / 'release-manifest.json')
        rows = [json.loads(line) for line in (RUN / 'tasks.jsonl').open()]
        original = {r['id']: r for r in map(json.loads, (ROOT / 'data/formal/eval/webshop_official_test_128.jsonl').open())}
        assert len(rows) == len({r['id'] for r in rows}) == 128
        assert {r['id'] for r in rows} == set(original)
        goals = [json.loads(line) for line in (RUN / 'goals.jsonl').open()]
        random.Random(233).shuffle(goals)
        for task in rows:
            goal = goals[task['metadata']['goal_index']]
            contract = task['metadata']['webshop_quality']
            assert task['prompt'] == goal['instruction_text']
            assert contract['version'] == goal['_quality_contract']['version'] == VERSION
            assert contract['scorer_sha256'] == release['scorer_sha256']
            assert contract['prompt_sha256'] == hashlib.sha256(task['prompt'].encode()).hexdigest()
        prompt_changes = [r['id'] for r in rows if r['prompt'] != original[r['id']]['prompt']]
        assert len(prompt_changes) == 28
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        os.environ.update(SPGFS_WEBSHOP_GOALS=str(RUN / 'goals.jsonl'), SPGFS_WEBSHOP_PORT=str(port))
        formal = ROOT / 'configs/formal_training.toml'
        shutil.copyfile(formal, RUN / 'formal_config.source.toml')
        text = formal.read_text()
        # A run-local TOML resolves relative files against the run directory.
        # Preserve the original formal file locations when relocating it.
        text = re.sub(r'(?m)^(\w+\s*=\s*)"((?:state|assets|models|datasets)/[^"\n]*)"',
                      lambda m: m.group(1) + json.dumps(str(ROOT / m.group(2))), text)
        for section, values in [
            ('webshop', {'service_url': f'http://127.0.0.1:{port}'}),
            ('runtimes.gpt', {'max_concurrency': 5}),
            ('runtimes.gpt_student', {'max_concurrency': 15}),
            ('runtime_routing.endpoint_pools', {'gpt': ['gpt_student', 'gpt']}),
            ('runtime_routing.dataset_worker_routes', {'webshop': ['gpt']}),
            ('resources', {'proposer_gpu_id': 1, 'solver_gpu_id': 1, 'allocated_gpu_ids': [1],
                           'allow_policy_gpu_colocation': True, 'proposer_service_gpu_memory_utilization': 0.32,
                           'solver_service_gpu_memory_utilization': 0.32}),
            ('runtime_routing', {'health_state_path': str(RUN / 'route_health.json')}),
            ('canvas', {'submission_journal_dir': str(RUN / 'submissions')}),
            ('trace', {'path': str(RUN / 'traces.jsonl')})]:
            text = block(text, section, values)
        config_path = RUN / 'config.toml'
        config_path.write_text(text)
        from selfplay_graph_flowsteer.application import load_adaptive_config
        from selfplay_graph_flowsteer.webshop_sidecar import implementation_sha256
        config = load_adaptive_config(config_path)
        config.validate()
        assert config.director_prompt_variant == 'v3'
        assert config.webshop.worker_memory_policy == 'factual_memory_v2'
        assert config.webshop.purchase_budget_policy == 'completion_reserve_v2'
        assert config.webshop.scheduling_policy == 'bounded_research_v1'
        assert config.webshop.max_total_calls == 16
        assert config.runtime_pool()['gpt'].max_concurrency == 5
        assert config.runtime_pool()['gpt_student'].max_concurrency == 15
        assert config.runtime_endpoint_pools['gpt'] == ('gpt_student', 'gpt')
        assert config.dataset_worker_routes['webshop'] == ('gpt',)
        assert config.runtime_pool()['gpt'].reasoning_effort == 'low'
        assert config.runtime_pool()['gpt_student'].reasoning_effort == 'low'
        previous_report = REPORT.parent / 'webshop-current-quality-128-20260930'
        previous_manifest = json.loads((previous_report / 'manifest.json').read_text())
        assert sha(RUN / 'tasks.jsonl') == previous_manifest['dataset_sha256']
        assert sha(RUN / 'goals.jsonl') == previous_manifest['goals_sha256']
        assert release['scorer_sha256'] == previous_manifest['scorer_sha256']
        changed_sources = sorted(k for k in hashes.keys() | previous_manifest['source_sha256'].keys()
                                 if hashes.get(k) != previous_manifest['source_sha256'].get(k))
        save(RUN / 'comparison-controls.json', {'previous_run': str(Path((previous_report / 'run-path.txt').read_text().strip())),
             'tasks_byte_identical': True, 'goals_byte_identical': True, 'scorer_identical': True,
             'source_identical': not changed_sources, 'changed_sources': changed_sources,
             'intentional_changes': ['fixed GPT worker route', '15 question workers', 'student gateway concurrency 15',
                                     'Nexus non-eco concurrency 5', 'eco member excluded from GPT pool']})
        save(RUN / 'manifest.json', {'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
             'source_sha256': hashes, 'formal_config_sha256': sha(formal), 'config_sha256': sha(config_path),
             'dataset_sha256': sha(RUN / 'tasks.jsonl'), 'goals_sha256': sha(RUN / 'goals.jsonl'),
             'scorer_version': VERSION, 'scorer_sha256': release['scorer_sha256'], 'implementation_sha256': implementation_sha256(),
             'planned': 128, 'ids': [r['id'] for r in rows], 'prompt_changed_ids': prompt_changes,
             'question_concurrency': 15, 'route_concurrency': {'gpt_student': 15, 'gpt': 5}, 'worker_route': 'gpt',
             'worker_model': {'gpt_student': 'lab-gpt-5.5-2', 'gpt': 'gpt-5.5'},
             'enabled_pool_members': ['gpt_student', 'gpt'], 'disabled_pool_members': ['gpt_eco'],
             'worker_reasoning_effort': 'low', 'director_thinking': True, 'director_context': 32768,
             'seed': 0, 'skills': 'off', 'tool_budget': 16, 'worker_token_limit': 350000,
             'purchase_budget_policy': 'completion_reserve_v2', 'scheduling_policy': 'bounded_research_v1',
             'worker_memory_policy': 'factual_memory_v2', 'memory_delivery': 'automatic',
             'evaluation_only': True, 'parameter_updates': 0, 'fresh_attempts_per_task': 1, 'retry_completed_failures': False,
             'previous_source_preflight_tests_passed': 104, 'global_formal_configuration_changed': False})
        args = [sys.executable, '-m', 'selfplay_graph_flowsteer.webshop_sidecar', '--host', '127.0.0.1', '--port', str(port)]
        for flag, name in [('interpreter','INTERPRETER'),('worker-script','WORKER'),('source-root','SOURCE_ROOT'),
                           ('source-revision','SOURCE_REVISION'),('store','STORE'),('goals','GOALS'),('index','INDEX'),('java-home','JAVA_HOME')]:
            args += ['--' + flag, os.environ['SPGFS_WEBSHOP_' + name]]
        args += ['--worker-timeout', '180', '--max-sessions', '192', '--max-initializers', '4']
        log = (RUN / 'sidecar.log').open('w'); handles.append(log)
        sidecar = subprocess.Popen(args, cwd=ROOT, env=dict(os.environ), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        save(RUN / 'sidecar.json', {'pid': sidecar.pid, 'args': args, 'owned_by_this_run': True})
        status('sidecar_starting')
        deadline = time.monotonic() + 120
        while True:
            assert sidecar.poll() is None, 'WebShop sidecar exited'
            try:
                health = json.loads(request(f'http://127.0.0.1:{port}/health'))
                assert health['status'] == 'ok'
                for key, expected in [('goals_sha256', sha(RUN / 'goals.jsonl')), ('scorer_version', VERSION),
                                      ('scorer_sha256', release['scorer_sha256']), ('implementation_sha256', implementation_sha256())]:
                    assert health[key] == expected, key
                save(RUN / 'sidecar-health.json', health)
                break
            except OSError:
                assert time.monotonic() < deadline, 'WebShop readiness timeout'
                time.sleep(1)
        args = [sys.executable, '-m', 'selfplay_graph_flowsteer', 'benchmark', '--config', str(config_path),
                '--dataset', str(RUN / 'tasks.jsonl'), '--output', str(RUN / 'results'), '--workers', '15', '--seed', '0',
                '--worker-route', 'gpt', '--director-thinking', '--director-base-url', DIRECTOR_URL,
                '--director-api-key', 'EMPTY', '--director-model', 'Qwen3.5-9B',
                '--wandb-mode', 'disabled', '--skill-context', 'off', '--disable-swe']
        log = (RUN / 'benchmark.log').open('w'); handles.append(log)
        child = subprocess.Popen(args, cwd=ROOT, env=dict(os.environ), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        save(RUN / 'child.json', {'pid': child.pid, 'args': args})
        status('running', completed=0, planned=128)
        previous = None
        while child.poll() is None:
            time.sleep(15)
            path = RUN / 'results/run_state.json'
            state = json.loads(path.read_text()) if path.exists() else {}
            counts = (state.get('completed', 0), state.get('failed', 0))
            if counts != previous:
                status('running', completed=counts[0], errors=counts[1], planned=128)
                previous = counts
        status('benchmark_exited', returncode=child.returncode)
        save(RUN / 'source_integrity.json', {'frozen_source_unchanged': all(sha(RUN / 'evaluated_source' / rel) == h for rel,h in hashes.items()),
             'current_source_matches_snapshot': all(sha(ROOT / rel) == h for rel,h in hashes.items()),
             'frozen_goals_unchanged': sha(RUN / 'goals.jsonl') == release['files']['goals.jsonl'],
             'formal_config_unchanged': sha(formal) == sha(RUN / 'formal_config.source.toml')})
        return child.returncode
    except BaseException as exc:
        status('launcher_failed', error=type(exc).__name__, message=str(exc))
        raise
    finally:
        child_stopped = stop(child)
        sidecar_stopped = stop(sidecar)
        for handle in handles:
            handle.close()
        save(RUN / 'cleanup.json', {'benchmark_stopped': child_stopped, 'sidecar_stopped': sidecar_stopped,
                                  'sidecar_port': port, 'reused_director_retained': True})


if __name__ == '__main__':
    raise SystemExit(main())
