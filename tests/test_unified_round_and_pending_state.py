"""Offline regressions for shared ALFWorld/SWE Director state boundaries."""

import json
from types import SimpleNamespace

import pytest

from selfplay_graph_flowsteer.alfworld import ALFWorldSessionLifecycle
from selfplay_graph_flowsteer.canvas import CanvasState, GraphCanvas
from selfplay_graph_flowsteer.config import CanvasConfig
from selfplay_graph_flowsteer.contracts import CodeArtifactRef
from selfplay_graph_flowsteer.director import GraphDirector
from selfplay_graph_flowsteer.llm import MockBackend
from selfplay_graph_flowsteer.observability import TaskSpec
from selfplay_graph_flowsteer.runtime import MultiAgentRuntime
from selfplay_graph_flowsteer.submission_contract import receipt_error
from .helpers import NumericRecordingExecutor
from .test_alfworld_sessions import Client
from .test_unified_submission import prompt


@pytest.fixture(params=['alfworld', 'swe_bench'])
def canvas_factory(request, tmp_path):
    lifecycles = []

    def make(*, routes=False, complete=True, max_rounds=40):
        executor = NumericRecordingExecutor()
        original = executor.execute
        dataset = request.param
        if dataset == 'alfworld':
            game = tmp_path / 'game.tw-pddl'
            game.write_text('offline fixture')
            life = ALFWorldSessionLifecycle(Client(), tmp_path)
            lifecycles.append(life)
            life.bind_task(TaskSpec('fixture', 'Move mug', metadata={'game_path': str(game)}))
            executor.tools = {'alfworld_step': SimpleNamespace(lifecycle=life)}

        def execute(**kwargs):
            artifact = original(**kwargs)
            if dataset == 'alfworld':
                state = life.begin_execution(agent_id=kwargs['node'].agent_id,
                                             seed=0, revision=kwargs['revision'])
                for _ in range(2 if complete else 1):
                    if not state['done']:
                        state = life.step(state['admissible_actions'][0]['action_id'])
                life.end_execution()
                artifact.environment_result = life.result_for(artifact.agent_id)
            else:
                artifact.code_artifact_ref = CodeArtifactRef(
                    'a' * 64, 'instance', 'repo', 'base', 10, ('code.py',))
                artifact.swe_progress = {
                    'trusted': True, 'commit_ready': complete,
                    'test_after_latest_edit': complete, 'test_failure_count': 0,
                }
            return artifact

        executor.execute = execute
        return GraphCanvas(task='Move mug' if dataset == 'alfworld' else 'Fix the issue',
            dataset=dataset, runtime=MultiAgentRuntime(executor),
            runtime_routes=('deepseek',) if routes else (),
            config=CanvasConfig(submission_protocol='unified_task_result_v1',
                submission_journal_dir=str(tmp_path / 'journal'), max_rounds=max_rounds,
                remaining_time_admission_enabled=False, remaining_token_admission_enabled=False))

    yield make
    for life in lifecycles:
        life.close_all()


def authoritative(canvas, action):
    return canvas.step(json.dumps(action), authoritative_director=True)


def configure(canvas, name='solver'):
    actions = [dict(action='add_agent', agent_id=name), prompt(name)]
    if canvas.runtime_routes:
        actions.append(dict(action='set_model', target=name, runtime_route='deepseek'))
    for action in actions:
        result = authoritative(canvas, action)
        assert result.accepted, result.feedback


@pytest.mark.parametrize('routes', [False, True])
def test_real_director_finishes_on_last_round_without_extra_worker(canvas_factory, routes):
    c = canvas_factory(routes=routes, max_rounds=4 if routes else 3)
    actions = [dict(action='add_agent', agent_id='solver'), prompt()]
    if routes:
        actions.append(dict(action='set_model', target='solver', runtime_route='deepseek'))
    actions.append(dict(action='finish', target='solver'))
    backend = MockBackend([json.dumps(action) for action in actions])
    run = GraphDirector(canvas=c, backend=backend).run()
    assert run.finished, [(t.action.action_type.value, t.rejection_code) for t in c.history]
    assert c.round_index == c.config.max_rounds == len(backend.calls)
    assert len(c.runtime.executor.calls) == 1
    assert c.history[-1].executed_agents == []
    assert not c.active and not c.history[-1].active
    assert c.control_snapshot()['allowed_actions'] == []
    assert receipt_error(c.submission_receipt, run=run, events=c.history,
                         run_id=c.run_id, dataset=c.dataset) is None
    if c.dataset == 'alfworld':
        assert c.submission_receipt.payload['won']
    else:
        assert c.submission_receipt.payload_kind == 'code_patch'


def test_last_round_run_executes_once_and_cannot_open_an_extra_turn(canvas_factory):
    c = canvas_factory(complete=False)
    configure(c)
    c.config.max_rounds = c.round_index + 1
    assert 'run_agent' in c.control_snapshot()['allowed_actions']
    result = authoritative(c, dict(action='run_agent', target='solver'))
    assert result.accepted, result.feedback
    assert len(c.runtime.executor.calls) == 2
    assert c.round_index == c.config.max_rounds
    assert not result.active and result.control_snapshot['allowed_actions'] == []
    assert not c._unified_can_run('solver')
    rejected = authoritative(c, dict(action='run_agent', target='solver'))
    assert rejected.rejection_code == 'canvas_inactive'
    assert c.round_index == c.config.max_rounds and len(c.runtime.executor.calls) == 2


def test_last_round_does_not_bypass_stale_candidate_validation(canvas_factory):
    c = canvas_factory()
    configure(c)
    c.dirty_agents.add('solver')
    c.config.max_rounds = c.round_index + 1
    backend = MockBackend([json.dumps(dict(action='finish', target='solver'))])
    run = GraphDirector(canvas=c, backend=backend).run()
    assert not run.finished and c.submission_receipt is None
    assert any(t.rejection_code == 'director_action_not_allowed' for t in c.history)
    assert len(backend.calls) == 1 and len(c.runtime.executor.calls) == 1
    assert c.round_index == c.config.max_rounds


@pytest.mark.parametrize('awaiting_model', [False, True])
def test_deleting_pending_node_allows_real_director_to_rebuild_and_finish(canvas_factory, awaiting_model):
    c = canvas_factory(routes=True)
    assert authoritative(c, dict(action='add_agent', agent_id='draft')).accepted
    if awaiting_model:
        assert authoritative(c, prompt('draft')).accepted
        assert c.state is CanvasState.AWAITING_MODEL
    deletion = authoritative(c, dict(action='delete_agent', target='draft'))
    assert deletion.accepted, deletion.feedback
    assert c.state is CanvasState.BUILDING and c.pending_agent_id is None
    assert not c.graph.nodes and not c.runtime.executor.calls
    assert deletion.control_snapshot['pending_agent_id'] is None
    assert 'add_agent' in deletion.control_snapshot['allowed_actions']
    actions = [dict(action='add_agent', agent_id='replacement'), prompt('replacement'),
               dict(action='set_model', target='replacement', runtime_route='deepseek'),
               dict(action='finish', target='replacement')]
    run = GraphDirector(canvas=c, backend=MockBackend([json.dumps(a) for a in actions])).run()
    assert run.finished, [t.feedback for t in c.history]
    assert set(c.graph.nodes) == {'replacement'} and len(c.runtime.executor.calls) == 1


@pytest.mark.parametrize('awaiting_model', [False, True])
def test_last_round_pending_delete_clears_state_but_does_not_extend_budget(canvas_factory, awaiting_model):
    c = canvas_factory(routes=True)
    assert authoritative(c, dict(action='add_agent', agent_id='draft')).accepted
    if awaiting_model:
        assert authoritative(c, prompt('draft')).accepted
    c.config.max_rounds = c.round_index + 1
    result = authoritative(c, dict(action='delete_agent', target='draft'))
    assert result.accepted, result.feedback
    assert c.pending_agent_id is None and c.state is CanvasState.BUILDING
    assert not result.active and result.control_snapshot['allowed_actions'] == []
    assert authoritative(c, dict(action='add_agent', agent_id='later')).rejection_code == 'canvas_inactive'
    assert c.round_index == c.config.max_rounds and not c.runtime.executor.calls


def test_pending_model_at_exhausted_round_has_no_advertised_actions(canvas_factory):
    c = canvas_factory(routes=True)
    assert authoritative(c, dict(action='add_agent', agent_id='draft')).accepted
    c.config.max_rounds = c.round_index + 1
    result = authoritative(c, prompt('draft'))
    assert result.accepted and c.state is CanvasState.AWAITING_MODEL
    assert not result.active and result.control_snapshot['allowed_actions'] == []
    assert authoritative(c, dict(action='delete_agent', target='draft')).rejection_code == 'canvas_inactive'
    assert c.pending_agent_id == 'draft' and not c.runtime.executor.calls


@pytest.mark.parametrize('awaiting_model', [False, True])
def test_pending_delete_parameters_match_the_state_gate(canvas_factory, awaiting_model):
    c = canvas_factory(routes=True)
    configure(c)
    assert authoritative(c, dict(action='add_agent', agent_id='draft')).accepted
    if awaiting_model:
        assert authoritative(c, prompt('draft')).accepted
    assert c.control_snapshot()['legal_action_parameters']['delete_agent']['targets'] == ['draft']
    rejected = authoritative(c, dict(action='delete_agent', target='solver'))
    assert not rejected.accepted and set(c.graph.nodes) == {'draft', 'solver'}
    assert c.pending_agent_id == 'draft'
