from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import tomllib

import pytest

from selfplay_graph_flowsteer import application as app
from selfplay_graph_flowsteer.aime_formal import VERSION, FormalAIMEApplication, _load_application
from selfplay_graph_flowsteer.llm import MockBackend
from selfplay_graph_flowsteer.graph import MultiAgentGraph
from selfplay_graph_flowsteer.observability import TaskSpec
from selfplay_graph_flowsteer.selfplay_runtime import ByteTokenizer, adaptive_result_to_rollout
from selfplay_graph_flowsteer.submission_contract import OutcomeDecision, receipt_error


def formal_config(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / 'configs/formal_training.toml'
    raw = tomllib.loads(path.read_text())
    monkeypatch.setattr(app, '_load_project_env', lambda _: None)
    monkeypatch.setenv('SPGFS_ALLOWED_PHYSICAL_GPUS', ','.join(map(str, raw['resources']['allocated_gpu_ids'])))
    for runtime in raw['runtimes'].values():
        if runtime.get('api_key_env'):
            monkeypatch.setenv(runtime['api_key_env'], 'synthetic-test-key')
        for name in runtime.get('api_key_env_by_dataset', {}).values():
            monkeypatch.setenv(name, 'synthetic-test-key')
    config = app.load_adaptive_config(path, validate=False)
    return replace(config,
        retrieval=replace(config.retrieval, enabled=False, nq_evidence_mode=None),
        webshop=replace(config.webshop, enabled=False),
        alfworld=replace(config.alfworld, enabled=False),
        swe=replace(config.swe, enabled=False), skillbank_enabled=False,
        trace_path=tmp_path / 'traces.jsonl', route_health_path=tmp_path / 'route-health.json',
        canvas=replace(config.canvas, submission_journal_dir=str(tmp_path / 'submissions')))


def test_formal_config_selects_full_engine_and_binds_all_source_files(tmp_path, monkeypatch):
    config = formal_config(tmp_path, monkeypatch)
    assert config.aime_actions.implementation == VERSION
    assert config.worker_routes_for('aime') == ('gpt', 'grok', 'gemini', 'deepseek', 'minimax')
    assert config.solver_model.enable_thinking is True
    assert config.canvas.worker_usage_policy('aime')['start_threshold'] == 240000
    contract = config.model_manifest()['execution_semantics']['aime_implementation_contract']
    assert contract['module_namespace'] == 'formal_aime'
    assert {'runtime.py', 'llm.py', 'canvas.py', 'math_completion.py', 'final_artifact.py',
            'output_recovery.py', 'aime_protocol.py', 'worker_usage_ledger.py', 'director.py',
            'adaptive.py', 'candidate_feedback.py'} <= set(contract['source_sha256'])


def test_formal_primary_aime_submits_with_canonical_learner_authority(tmp_path, monkeypatch):
    config = formal_config(tmp_path, monkeypatch)
    _load_application()
    gateway = __import__('formal_aime.llm', fromlist=['OpenAICompatibleBackend'])
    gateway_config = __import__('formal_aime.config', fromlist=['ModelGatewayConfig', 'ModelRoleConfig'])
    requests = []

    def create(client, request_config, request, deadline):
        requests.append(request)
        return SimpleNamespace(model='offline-worker', usage=SimpleNamespace(prompt_tokens=20, completion_tokens=30),
            choices=[SimpleNamespace(finish_reason='stop', message=SimpleNamespace(tool_calls=[],
                content=json.dumps({'answer': '4', 'summary': '2 + 2 = 4', 'confidence': 1,
                    'evidence': [], 'unresolved_issues': [], 'tool_summary': []})))])

    monkeypatch.setattr(gateway, '_openai_completion_create', create)
    backend = object.__new__(gateway.OpenAICompatibleBackend)
    backend.config = gateway_config.ModelGatewayConfig(request_profile='generic',
        generation_audit_dir=str(tmp_path / 'generations'),
        roles={'worker': gateway_config.ModelRoleConfig(model='offline-worker', enable_thinking=False)})
    backend._client_for_request = lambda: None
    director = MockBackend([json.dumps(action) for action in [
        {'action': 'add_agent', 'agent_id': 'solver'},
        {'action': 'set_prompt', 'target': 'solver', 'role': 'Calculator',
         'objective': 'Compute 2 + 2', 'scope': 'Solve the public question',
         'expected_output': 'The final integer answer', 'result_scope': 'task_result'},
        {'action': 'set_model', 'target': 'solver', 'runtime_route': 'gpt'},
        {'action': 'finish', 'target': 'solver'},
    ]])
    application = app.create_adaptive_application(config, mock=True,
        director_backend=director, worker_backend=backend)
    try:
        assert isinstance(application, FormalAIMEApplication)
        result = application.solve('What is 2 + 2?', task_id='formal-aime-primary',
            task_type='aime', reference='4', metadata={'dataset': 'aime'})
        assert type(application.solver).__module__ == 'formal_aime.adaptive'
        outcome = result.solver_result.outcome_decision
        assert isinstance(outcome, OutcomeDecision) and outcome.runtime_owned
        assert outcome.status == 'scored', result.to_dict()
        assert outcome.verification.score == 1
        assert requests
        assert 'Do not write Python comments' in requests[0]['messages'][0]['content']
        run = result.solver_result.director_run
        assert receipt_error(run.submission_receipt, run=run, events=result.solver_result.trace.events,
            run_id=result.run_id, dataset='aime') is None
        rollout = adaptive_result_to_rollout(result, ByteTokenizer(), rollout_index=0, seed=0)
        assert rollout.trajectory.metadata['answer_score'] == 1
        assert rollout.trajectory.metadata['outcome_decision']['receipt_ref'] == outcome.receipt_ref
        branch = app.create_adaptive_application(config, mock=True,
            director_backend=MockBackend([]), worker_backend=backend)
        try:
            graph = MultiAgentGraph.from_dict(run.graph)
            evaluated = branch.evaluate_graph(result.task, graph, seed=11, return_verification=True)
            assert evaluated['score'] == 1
            assert type(branch.runtime).__module__ == 'formal_aime.runtime'
            assert branch.last_graph_evaluation['execution_mode'] == 'full_graph_v1'
        finally:
            branch.close()
    finally:
        application.close()


def test_primary_and_counterfactual_select_same_full_aime_application(tmp_path, monkeypatch):
    application = app.create_adaptive_application(formal_config(tmp_path, monkeypatch), mock=True)
    try:
        engine = application._select('aime')
        assert type(engine).__module__ == 'formal_aime.application'
        task = TaskSpec('cf', 'Question', reference='4', metadata={'dataset': 'aime'})
        calls = []
        monkeypatch.setattr(engine, 'evaluate_graph', lambda *args, **kwargs: calls.append((args, kwargs)) or 0.75)
        assert application.evaluate_graph(task, 'graph', seed=3) == 0.75
        assert calls[0][1]['seed'] == 3
        assert application._active is engine
        assert application._select('hotpotqa') is application._base
        assert application._select('aime') is engine
        errors = _load_application()
        failure = errors.GraphEvaluationBackendError({'routes': ['gpt']})
        def failed_replay(*args, **kwargs):
            raise failure
        monkeypatch.setattr(engine, 'evaluate_graph', failed_replay)
        with pytest.raises(app.GraphEvaluationBackendError) as caught:
            application.evaluate_graph(task, 'graph', seed=4)
        assert caught.value.failure == {'routes': ['gpt']}
    finally:
        application.close()


def test_invalid_implementation_cannot_silently_use_old_engine():
    with pytest.raises(ValueError, match='unknown aime_actions.implementation'):
        app.AIMEActionConfig(enabled=True, implementation='missing-version').validate()


def test_aime_and_other_datasets_share_provider_concurrency_and_priority():
    from selfplay_graph_flowsteer import llm

    _load_application()
    gateway = __import__('formal_aime.llm', fromlist=['current_request_priority'])
    assert gateway._REQUEST_GATES is llm._REQUEST_GATES
    assert gateway._REQUEST_GATE_LOCK is llm._REQUEST_GATE_LOCK
    with llm.request_priority('counterfactual'), llm.request_dataset('aime'):
        assert gateway.current_request_priority() == 'counterfactual'
        assert gateway.current_request_dataset() == 'aime'
