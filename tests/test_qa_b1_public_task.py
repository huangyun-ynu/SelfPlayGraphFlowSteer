"""B1 integration: trusted question boundaries, role changes, and persisted inputs."""

import copy
import json

import pytest

from selfplay_graph_flowsteer.adaptive import AdaptiveWorkflowSolver
from selfplay_graph_flowsteer.answer_submission import AnswerFinalizer
from selfplay_graph_flowsteer.canvas import GraphCanvas
from selfplay_graph_flowsteer.config import CanvasConfig
from selfplay_graph_flowsteer.dataset_adapters import bind_public_qa_task, public_qa_task_context, solver_task_text
from selfplay_graph_flowsteer.graph import MultiAgentGraph
from selfplay_graph_flowsteer.learning import load_fixed_jsonl
from selfplay_graph_flowsteer.llm import MockBackend
from selfplay_graph_flowsteer.observability import ExecutionTrace, TaskSpec, replay_trace
from selfplay_graph_flowsteer.qa_public_task import QA_PUBLIC_TASK_VERSION, PublicQATask
from selfplay_graph_flowsteer.runtime import ModelAgentExecutor, MultiAgentRuntime, RoutedModelAgentExecutor

from .test_director_relation_audit_context import TokenBackend


ANSWER = json.dumps(dict(evidence=["A public statement"], summary="A local finding",
                         confidence=0.8, unresolved_issues=[], tool_summary=[], answer="Example County"))
QUESTION = "Where did the artist live?"
PUBLIC = "Based on the following passages, answer the question.\n\n[Public source]\nAn artist lived in Example County.\n\nQuestion: " + QUESTION


@pytest.fixture(autouse=True)
def append_only(monkeypatch):
    monkeypatch.setenv('SPGFS_DIRECTOR_CONTEXT_MODE', 'append_only')


def ingest(tmp_path, dataset, *, question=None, prompt=PUBLIC):
    record = dict(id=dataset + "/public/1", dataset=dataset, prompt=prompt,
                  reference="PRIVATE_GOLD_SENTINEL", target_answers=["PRIVATE_ALIAS_SENTINEL"],
                  metadata=dict(dataset=dataset, evidence_mode="provided_context_inline",
                                question_decomposition=["PRIVATE_DECOMPOSITION_SENTINEL"],
                                supporting_facts=["PRIVATE_SUPPORT_SENTINEL"],
                                public_qa_task={"question": "PRIVATE_FORGED_ANCHOR"}))
    if question is not None:
        record['question'] = question
    path = tmp_path / 'examples.jsonl'
    path.write_text(json.dumps(record) + '\n')
    example = load_fixed_jsonl(path)[0]
    # Match the production runner: both datasets execute the Hotpot workflow.
    metadata = {**example.metadata, 'dataset': 'hotpotqa', 'evaluation_dataset': dataset}
    return TaskSpec(task_id=example.example_id, prompt=example.task,
                    reference=example.reference, metadata=metadata)


def config(tmp_path):
    return CanvasConfig(submission_protocol="unified_task_result_v1",
                        submission_journal_dir=str(tmp_path), max_rounds=50)


def delegation(scope="task_result"):
    return dict(action="set_prompt", target="a", role="Analyst",
                objective="Identify the person mentioned in the passages",
                scope="Use supplied evidence", expected_output="Supported local findings",
                result_scope=scope)


def requests(backend):
    return [json.loads(call['messages'][-1]['content']) for call in backend.calls]


def assert_public(request, question=QUESTION, scope="task_result", context=PUBLIC):
    assert request['original_question'] == question
    assert request['public_task_context'] == context
    assert request['result_scope'] == scope
    assert request['qa_public_task_version'] == QA_PUBLIC_TASK_VERSION
    text = json.dumps(request)
    assert 'Submission contract:' not in request['public_task_context']
    assert 'PRIVATE_' not in text
    assert 'person' in request['assigned_task']


@pytest.mark.parametrize('dataset', ['hotpotqa', 'musique'])
def test_full_loader_adaptive_finalizer_chain_is_reference_blind(tmp_path, dataset):
    task = ingest(tmp_path, dataset)
    worker = MockBackend([ANSWER])
    actions = [dict(action='add_agent', agent_id='a'), delegation(), dict(action='finish', target='a')]
    director = TokenBackend(list(map(json.dumps, actions)))
    solver = AdaptiveWorkflowSolver(director_backend=director,
        runtime=MultiAgentRuntime(ModelAgentExecutor(worker)), canvas_config=config(tmp_path),
        answer_finalizer=AnswerFinalizer(), director_prompt_variant='v3',
        director_tokenizer=director.tokenizer, director_observation_schema='compact_factual_v1')
    result = solver.solve(task, run_id='b1-full-chain-' + dataset)
    assert result.director_run.submission_receipt
    assert len(worker.calls) == 1
    assert_public(requests(worker)[0])
    assert 'PRIVATE_' not in json.dumps(director.calls)
    # FINALIZE previously caused solver_task_text to append the contract.
    assert solver_task_text(task, include_submission_contract=True) == PUBLIC
    replay_ids = []
    for _ in range(2):
        replay_worker = MockBackend([ANSWER])
        replay = replay_trace(ExecutionTrace.from_dict(result.trace.to_dict()),
                             runtime=MultiAgentRuntime(ModelAgentExecutor(replay_worker)))
        replay_ids.append(replay.run_id)
        assert replay.submission_receipt
        assert_public(requests(replay_worker)[0])
    assert len(set(replay_ids)) == 2 and result.trace.run_id not in replay_ids


@pytest.mark.parametrize('dataset', ['hotpotqa', 'musique'])
def test_explicit_question_survives_embedded_markers_and_schema_recovery(tmp_path, dataset):
    question = 'What does "Question:" mean?\n\nQuestion: Quote the marker.\n\nSubmission contract: is literal text.'
    public = '[Passage]\nQuestion: is a heading in this document.\n\nQuestion: ' + question
    task = ingest(tmp_path, dataset, question=question, prompt=public)
    worker = MockBackend([json.dumps(dict(answer='Example County', confidence='high')), ANSWER])
    canvas = GraphCanvas(task=solver_task_text(task, include_submission_contract=True),
        public_qa_task=public_qa_task_context(task), runtime=MultiAgentRuntime(ModelAgentExecutor(worker)),
        dataset=task.metadata['dataset'], config=config(tmp_path))
    assert canvas.step(json.dumps(dict(action='add_agent', agent_id='a'))).accepted
    assert canvas.step(json.dumps(delegation('subtask'))).accepted
    assert len(worker.calls) == 2
    for request in requests(worker):
        assert request['original_question'] == question
        assert request['public_task_context'] == public
        assert request['result_scope'] == 'subtask'
        assert 'PRIVATE_' not in json.dumps(request)
    assert canvas.runtime.artifacts['a'].answer == 'Example County'


@pytest.mark.parametrize('dataset', ['hotpotqa', 'musique'])
def test_scope_edit_and_graph_restore_preserve_question_and_invalidate_cache(tmp_path, dataset):
    task = ingest(tmp_path, dataset)
    worker = MockBackend([ANSWER] * 5)
    runtime = MultiAgentRuntime(ModelAgentExecutor(worker))
    canvas = GraphCanvas(task=solver_task_text(task, include_submission_contract=True),
        public_qa_task=public_qa_task_context(task), runtime=runtime, dataset=task.metadata['dataset'], config=config(tmp_path))
    assert canvas.step(json.dumps(dict(action='add_agent', agent_id='a'))).accepted
    assert canvas.step(json.dumps(delegation('subtask'))).accepted
    old_binding = runtime.artifact_input_binding('a')
    assert_public(requests(worker)[0], scope='subtask')
    assert canvas.step(json.dumps(delegation('task_result'))).accepted
    assert_public(requests(worker)[1])
    assert runtime.artifact_input_binding('a')['input_hash'] != old_binding['input_hash']
    assert len(worker.calls) == 2
    stable_id = runtime.artifacts['a'].artifact_id
    runtime.execute(task=canvas.worker_task, graph=canvas.graph, dirty_agents=set())
    assert len(worker.calls) == 2 and runtime.artifacts['a'].artifact_id == stable_id

    # Simulate a persisted B0 Artifact binding without the new QA input version.
    payload = runtime._last_input_payloads[('a', False)]
    payload.pop('qa_public_task_version')
    runtime._artifact_input_bindings['a']['input_hash'] = runtime._cache_key(payload)
    assert not runtime.artifact_matches_current_input_signature('a', task=canvas.worker_task, graph=canvas.graph)
    runtime.cache.clear()
    runtime.execute(task=canvas.worker_task, graph=canvas.graph, dirty_agents=set())
    assert len(worker.calls) == 3

    restored = MultiAgentGraph.from_dict(json.loads(json.dumps(canvas.graph.to_dict())))
    assert restored.nodes['a'].metadata['public_qa_task'] == canvas.public_qa_task
    new_worker = MockBackend([ANSWER])
    fresh = MultiAgentRuntime(ModelAgentExecutor(new_worker))
    fresh.execute(task=canvas.worker_task, graph=restored)
    assert_public(requests(new_worker)[0])

    # Changing the trusted question must never reuse an old input result.
    restored.nodes['a'].metadata['public_qa_task'] = PublicQATask(
        'Which county contains the artist?', 'dataset.question', task.task_id).to_dict()
    assert not fresh.artifact_matches_current_input_signature('a', task=canvas.worker_task, graph=restored)


@pytest.mark.parametrize('dataset', ['hotpotqa', 'musique'])
def test_full_graph_routed_executor_and_trace_replay_have_same_public_anchor(tmp_path, dataset):
    task = ingest(tmp_path, dataset)
    worker = MockBackend([ANSWER])
    canvas = GraphCanvas(task=solver_task_text(task), public_qa_task=public_qa_task_context(task),
        runtime=MultiAgentRuntime(ModelAgentExecutor(worker)), dataset=task.metadata['dataset'], config=config(tmp_path))
    assert canvas.step(json.dumps(dict(action='add_agent', agent_id='a'))).accepted
    assert canvas.step(json.dumps(delegation())).accepted
    graph = MultiAgentGraph.from_dict(copy.deepcopy(canvas.graph.to_dict()))
    graph.output_agent = 'a'
    # Full graph evaluator can also reconstruct provenance from an old TaskSpec.
    graph.nodes['a'].metadata.pop('public_qa_task')
    bind_public_qa_task(task, graph)
    routed_worker = MockBackend([ANSWER])
    graph.nodes['a'].metadata['runtime_route'] = 'qwen'
    runtime = MultiAgentRuntime(RoutedModelAgentExecutor({'qwen': routed_worker}, ('qwen',)))
    runtime.execute(task=solver_task_text(task, include_submission_contract=True), graph=graph)
    assert_public(requests(routed_worker)[0])

    from selfplay_graph_flowsteer.observability import trace_from_canvas
    trace = trace_from_canvas(run_id='b1-replay', task=task, canvas=canvas)
    restored_trace = ExecutionTrace.from_dict(json.loads(json.dumps(trace.to_dict())))
    replay_worker = MockBackend([ANSWER])
    replay_trace(restored_trace, runtime=MultiAgentRuntime(ModelAgentExecutor(replay_worker)))
    assert_public(requests(replay_worker)[0])


def test_legacy_and_custom_inputs_never_infer_question_from_appended_instructions(tmp_path):
    old = TaskSpec(task_id='old', prompt=PUBLIC, metadata={'dataset': 'hotpotqa'})
    assert public_qa_task_context(old)['question'] == QUESTION
    custom = TaskSpec(task_id='custom', prompt='Discuss this custom public task.', metadata={'dataset': 'hotpotqa'})
    assert public_qa_task_context(custom)['question'] is None
    contaminated = TaskSpec(task_id='wrapped', prompt=PUBLIC + '\n\nSubmission contract: do something',
                            metadata={'dataset': 'hotpotqa'})
    assert public_qa_task_context(contaminated)['question'] is None
    nq = TaskSpec(task_id='nq', prompt='Who?', metadata={'dataset': 'nq_open'})
    assert public_qa_task_context(nq) is None
    assert 'Submission contract:' in solver_task_text(nq, include_submission_contract=True)


@pytest.mark.parametrize('dataset', ['hotpotqa', 'musique'])
def test_application_counterfactual_full_graph_uses_clean_input(tmp_path, dataset):
    from dataclasses import replace
    from pathlib import Path
    from selfplay_graph_flowsteer.application import AdaptiveSolverApplication, load_adaptive_config
    from selfplay_graph_flowsteer.observability import FlowSteerQAVerifier

    task = ingest(tmp_path, dataset)
    seed_backend = MockBackend([ANSWER])
    canvas = GraphCanvas(task=solver_task_text(task), public_qa_task=public_qa_task_context(task),
        runtime=MultiAgentRuntime(ModelAgentExecutor(seed_backend)), dataset=task.metadata['dataset'],
        config=config(tmp_path))
    assert canvas.step(json.dumps(dict(action='add_agent', agent_id='a'))).accepted
    assert canvas.step(json.dumps(delegation())).accepted
    graph = MultiAgentGraph.from_dict(canvas.graph.to_dict())
    graph.output_agent = 'a'
    graph.nodes['a'].metadata.pop('public_qa_task')
    branch_worker = MockBackend([ANSWER])
    runtime = MultiAgentRuntime(ModelAgentExecutor(branch_worker))
    solver = AdaptiveWorkflowSolver(director_backend=MockBackend([]), runtime=runtime,
        verifier=FlowSteerQAVerifier(), answer_finalizer=AnswerFinalizer())
    cfg = replace(load_adaptive_config(Path('configs/formal_training.toml'), validate=False),
                  canvas=config(tmp_path), skillbank_enabled=False)
    app = AdaptiveSolverApplication(config=cfg, solver=solver, runtime=runtime,
                                     skillbank=None, skill_lifecycle=None)
    app.evaluate_graph(task, graph, seed=0)
    assert len(branch_worker.calls) == 1
    assert_public(requests(branch_worker)[0])


@pytest.mark.parametrize('present', [False, True])
@pytest.mark.parametrize('dataset', ['hotpotqa', 'musique'])
def test_relation_choice_replay_retains_public_question(tmp_path, dataset, present):
    from selfplay_graph_flowsteer.observability import trace_from_canvas
    task = ingest(tmp_path, dataset)
    worker = MockBackend([ANSWER] * 10)
    canvas = GraphCanvas(task=solver_task_text(task), public_qa_task=public_qa_task_context(task),
        runtime=MultiAgentRuntime(ModelAgentExecutor(worker)), dataset=task.metadata['dataset'],
        config=config(tmp_path), binary_relation_policy=True)
    for agent_id in ['a', 'b']:
        assert canvas.step(json.dumps(dict(action='add_agent', agent_id=agent_id))).accepted
        assert canvas.step(json.dumps({**delegation('subtask'), 'target': agent_id})).accepted
    assert canvas.step(json.dumps(dict(action='consider_relation', source='a', target='b'))).accepted
    assert canvas.resolve_relation_choice('on' if present else 'off').accepted
    trace = trace_from_canvas(run_id='b1-relation-' + dataset, task=task, canvas=canvas)
    replay_worker = MockBackend([ANSWER] * 10)
    replay = replay_trace(ExecutionTrace.from_dict(trace.to_dict()),
                         runtime=MultiAgentRuntime(ModelAgentExecutor(replay_worker)))
    assert bool(replay.graph.bidirectional_edges) == present
    for request in requests(replay_worker):
        assert_public(request, scope='subtask')
