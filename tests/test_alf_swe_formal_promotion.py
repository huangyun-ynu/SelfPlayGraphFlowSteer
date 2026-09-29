"""Promoted ALF/SWE contracts in mixed collection and isolated graph branches."""
import copy
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from selfplay_graph_flowsteer.adaptive import AdaptiveWorkflowSolver
from selfplay_graph_flowsteer.alfworld import ALFWorldEnvironmentVerifier
from selfplay_graph_flowsteer.application import (
    AdaptiveApplicationConfig, AdaptiveSolverApplication, GraphEvaluationIncompleteError,
    load_adaptive_config,
)
from selfplay_graph_flowsteer.canvas import GraphCanvas
from selfplay_graph_flowsteer.director import GraphDirector
from selfplay_graph_flowsteer.execution_contract import bind_rollout_contract, validate_training_contract
from selfplay_graph_flowsteer.llm import MockBackend
from selfplay_graph_flowsteer.pats import resolve_scope
from selfplay_graph_flowsteer.pats_refiner import review_system_prompt
from selfplay_graph_flowsteer.rollouts import TrainingBatch, TrainingSample
from selfplay_graph_flowsteer.runtime import MultiAgentRuntime
from selfplay_graph_flowsteer.worker_usage_ledger import WorkerUsageLedger

from .helpers import NumericRecordingExecutor
from .test_alfworld_v3_usage import build, GOAL, ORIGINAL
from .test_unified_submission import prompt


def formal():
    return load_adaptive_config(Path(__file__).resolve().parents[1] / 'configs/formal_training.toml',
                                validate=False)


@pytest.mark.parametrize('dataset', [
    'alfworld', 'swe_bench', 'hotpotqa', 'aime', 'nq_open', 'webshop', 'healthbench_professional',
])
def test_formal_mixed_collection_and_skill_review_choose_the_same_protocol(tmp_path, dataset):
    config = formal()
    canvas = GraphCanvas(task='public task', dataset=dataset,
                         config=replace(config.canvas, submission_journal_dir=str(tmp_path)),
                         runtime=MultiAgentRuntime(NumericRecordingExecutor()))
    director = GraphDirector(canvas=canvas, backend=MockBackend([]), prompt_variant=config.director_prompt_variant)
    variant = 'v3'
    assert canvas.unified is (variant == 'v3')
    assert director.prompt_variant == variant
    scope = resolve_scope('qa', {'dataset': dataset})
    assert config.pats.variant_for_scope(scope) == variant
    if variant == 'v3':
        assert 'There is no SET_OUTPUT action' in review_system_prompt(config.pats.for_scope(scope), 'EXPAND')
    if dataset in {'alfworld', 'swe_bench'}:
        assert config.canvas.worker_usage_policy(dataset)['start_threshold'] == 350000
    else:
        assert config.canvas.worker_usage_policy(dataset) is None
    assert config.alfworld.task_prompt_source == 'environment_reset'
    assert config.canvas.alfworld_terminal_candidate_policy == 'finish_only_v1'
    assert config.worker_routes_for('swe_bench') == ('gpt',)
    assert config.worker_routes_for('alfworld') == config.worker_runtime_routes


def test_mixed_training_accepts_both_v3_datasets_and_rejects_stale_alf_receipt():
    semantics = formal().model_manifest()['execution_semantics']
    assert semantics['director_prompt_variant_by_dataset'] == {
        d: 'v3' for d in ('alfworld', 'swe_bench', 'hotpotqa', 'aime', 'nq_open', 'webshop', 'healthbench_professional')}
    rows = [{'dataset': dataset, 'model_roles': {'execution_semantics': semantics},
             'submission_contract_version': version, 'training_eligible': False}
            for dataset, version in [('alfworld', 'unified_submission_v1'),
                                     ('swe_bench', 'unified_submission_v1'),
                                     ('hotpotqa', 'unified_submission_v1')]]
    rollouts = [SimpleNamespace(trajectory=SimpleNamespace(metadata=row)) for row in rows]
    samples = tuple(TrainingSample(str(i), str(i), (1, 2), (0, 1), 1, 0, metadata=row)
                    for i, row in enumerate(rows))
    batches = bind_rollout_contract((TrainingBatch('proposer', ()), TrainingBatch('solver', samples)), rollouts)
    validate_training_contract(*batches, expected=semantics)
    rows[0]['submission_contract_version'] = semantics['submission_contract_version']
    with pytest.raises(ValueError, match='does not match dataset: alfworld'):
        validate_training_contract(*batches, expected=semantics)


def branch_fixture(tmp_path, monkeypatch):
    life, task, backend, registry, runtime, canvas_config = build(tmp_path, monkeypatch, threshold=15)
    director = MockBackend([json.dumps(action) for action in [
        {'action': 'add_agent', 'agent_id': 'solver'}, prompt(), {'action': 'finish', 'target': 'solver'}]])
    solver = AdaptiveWorkflowSolver(director_backend=director, runtime=runtime,
        action_registry=registry, canvas_config=canvas_config, verifier=ALFWorldEnvironmentVerifier())
    solver.solve(task, run_id='primary')
    primary_usage = dict(task.metadata['worker_usage'])
    app = AdaptiveSolverApplication(config=AdaptiveApplicationConfig(canvas=canvas_config),
        solver=solver, runtime=runtime, skillbank=None, skill_lifecycle=None)
    return app, life, task, backend, solver.active_canvas.graph, primary_usage


def test_graph_branches_use_reset_goal_and_independent_usage_accounts(tmp_path, monkeypatch):
    app, life, task, backend, graph, primary_usage = branch_fixture(tmp_path, monkeypatch)
    original_task, original_graph = copy.deepcopy(task), graph.to_dict()
    attempts = []
    for seed in (0, 1):
        assert app.evaluate_graph(task, graph, seed=seed) == 1.0
        usage = app.last_graph_evaluation['worker_usage']
        assert usage['confirmed_used'] == 20 > 15  # last admitted request may cross the threshold
        assert usage['usage_complete'] and usage['dispatch_policy_valid']
        attempts.append(usage['question_attempt_id'])
        assert app.runtime.worker_usage_ledger is None
    assert len(set(attempts)) == 2 and 'primary' not in attempts
    assert len(backend.calls) == 6  # two requests each for primary and both branches
    assert task == original_task and graph.to_dict() == original_graph
    assert all(GOAL in json.dumps(messages) and ORIGINAL not in json.dumps(messages)
               for messages in backend.messages)
    assert sorted(life.client.closed) == ['1', '2', '3']
    # Reopening the primary proves branch runs have neither charged nor locked it.
    import hashlib
    path = tmp_path / 'journal/worker_usage' / (hashlib.sha256(b'primary').hexdigest() + '.sqlite3')
    ledger = WorkerUsageLedger(path, question_attempt_id='primary', threshold=15)
    try:
        assert ledger.status()['confirmed_used'] == primary_usage['confirmed_used'] == 20
        assert ledger.digest() == primary_usage['digest']
    finally:
        ledger.close()
        app.close()


def test_unknown_branch_usage_cannot_become_training_credit(tmp_path, monkeypatch):
    app, life, task, backend, graph, _ = branch_fixture(tmp_path, monkeypatch)
    settle = WorkerUsageLedger.settle

    def unknown(self, attempt_id, **kwargs):
        return settle(self, attempt_id, **{**kwargs, 'input_tokens': None, 'output_tokens': None})

    monkeypatch.setattr(WorkerUsageLedger, 'settle', unknown)
    with pytest.raises(GraphEvaluationIncompleteError, match='incomplete or invalid Worker usage'):
        app.evaluate_graph(task, graph, seed=0)
    assert not app.last_graph_evaluation['worker_usage']['usage_complete']
    assert app.runtime.worker_usage_ledger is None
    assert sorted(life.client.closed) == ['1', '2']
    app.close()


def test_interrupted_branch_closes_prepared_environment_and_account(tmp_path, monkeypatch):
    app, life, task, backend, graph, _ = branch_fixture(tmp_path, monkeypatch)
    ledgers = []

    def interrupted(**kwargs):
        ledgers.append(app.runtime.worker_usage_ledger)
        raise KeyboardInterrupt('interrupted graph')

    monkeypatch.setattr(app.runtime, 'execute', interrupted)
    with pytest.raises(KeyboardInterrupt):
        app.evaluate_graph(task, graph, seed=0)
    assert ledgers[0]._db is None and ledgers[0]._lock_fd is None
    assert app.runtime.worker_usage_ledger is None and not app.runtime.full_graph_replay
    assert sorted(life.client.closed) == ['1', '2']
    app.close()
