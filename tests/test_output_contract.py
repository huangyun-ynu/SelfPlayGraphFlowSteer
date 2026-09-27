"""Regressions from the 2026-09-25 output-contract consistency audit."""

from __future__ import annotations

import copy
import json
from dataclasses import replace

import pytest

from selfplay_graph_flowsteer.adaptive import AdaptiveWorkflowSolver
from selfplay_graph_flowsteer.application import AdaptiveApplicationConfig
from selfplay_graph_flowsteer.async_cycle import AsyncPolicyLineage, validate_batch_lineage
from selfplay_graph_flowsteer.canvas import CanvasState, GraphCanvas
from selfplay_graph_flowsteer.config import CanvasConfig
from selfplay_graph_flowsteer.contracts import AgentArtifact
from selfplay_graph_flowsteer.counterfactual import RelationDecision, _replay_relation_branch
from selfplay_graph_flowsteer.director import GraphDirector, director_prompt_components
from selfplay_graph_flowsteer.execution_contract import (
    execution_semantics,
    validate_training_contract,
)
from selfplay_graph_flowsteer.llm import MockBackend
from selfplay_graph_flowsteer.observability import ExactMatchVerifier, TaskSpec
from selfplay_graph_flowsteer.rollouts import TrainingBatch, TrainingSample
from selfplay_graph_flowsteer.runtime import ModelAgentExecutor, MultiAgentRuntime


class RoleExecutor:
    version = "output-role-regression-v1"

    def __init__(self):
        self.calls = []
        self.fail_output = False
        self.fail_agent = None

    def execute(self, *, node, revision, **kwargs):
        selected = bool(node.metadata.get("_runtime_is_output_agent"))
        self.calls.append((node.agent_id, selected, revision))
        if (selected and self.fail_output) or node.agent_id == self.fail_agent:
            raise ValueError("simulated execution failure")
        return AgentArtifact(
            artifact_id="pending",
            agent_id=node.agent_id,
            answer="yes" if selected else "local-old",
            summary="visible evidence",
            confidence=1,
            token_in=1,
            token_out=1,
            revision=revision,
        )


def prompt(agent):
    return dict(
        action="set_prompt",
        target=agent,
        role="Evidence analyst",
        objective="Identify the facts required by the public question.",
        scope="Compare the relevant entities using the supplied evidence.",
        expected_output="Return the assigned finding and its evidence.",
    )


def step(canvas, **action):
    result = canvas.step(json.dumps(action), authoritative_director=True)
    assert result.accepted, result.feedback
    return result


def make_canvas(*, routes=()):
    executor = RoleExecutor()
    canvas = GraphCanvas(
        task="Do the entities share the same country?",
        dataset="hotpotqa",
        runtime=MultiAgentRuntime(executor),
        runtime_routes=routes,
        config=CanvasConfig(max_rounds=50, remaining_token_admission_enabled=False),
    )
    return canvas, executor


def add(canvas, agent):
    step(canvas, action="add_agent", agent_id=agent)
    step(canvas, **prompt(agent))


def test_unusable_output_target_is_not_reported_as_already_selected():
    canvas, _ = make_canvas()
    for agent in ("a", "b"):
        add(canvas, agent)
    step(canvas, action="set_relation", source="a", target="b", relation="bidirectional")
    canvas.runtime.artifacts["b"].answer = "WORKER_PROTOCOL_FAILURE"
    assert canvas.graph.output_agent is None
    assert canvas.control_snapshot()["legal_action_parameters"]["set_output"]["targets"] == ["a"]
    rejected = canvas.step('{"action":"set_output","target":"b"}', authoritative_director=True)
    assert not rejected.accepted
    assert rejected.rejection_code == "output_target_not_eligible"
    assert canvas.graph.output_agent is None
    step(canvas, action="set_output", target="a")
    repeated = canvas.step('{"action":"set_output","target":"a"}', authoritative_director=True)
    assert not repeated.accepted
    assert repeated.rejection_code == "output_already_selected"


def test_real_runtime_ledger_accounts_for_both_peer_execution_phases():
    from types import SimpleNamespace
    from selfplay_graph_flowsteer.application import _unique_worker_token_totals

    canvas, _ = make_canvas()
    for agent in ("a", "b"):
        add(canvas, agent)
    report = step(canvas, action="set_relation", source="a", target="b", relation="bidirectional").execution
    assert report.initial_model_calls + report.initial_cache_hits == 2
    assert report.revision_model_calls == 2
    assert len(report.execution_events) == 4
    assert len({e["artifact_id"] for e in report.execution_events}) == 4
    assert sum(a.token_in for a in report.artifacts.values()) < report.token_in
    trace = [SimpleNamespace(payload={"execution":report.to_dict()})] * 2
    clean = canvas.runtime.execute(task=canvas.worker_task, graph=canvas.graph, dirty_agents=set())
    assert not clean.execution_events and clean.token_in == clean.token_out == 0
    trace.append(SimpleNamespace(payload={"execution":clean.to_dict()}))
    assert _unique_worker_token_totals(trace, run_id="peer-runtime") == (report.token_in, report.token_out)


@pytest.mark.parametrize("variant", ["v2", "v2.1", "v2.2"])
def test_director_prompt_variants_import_and_render(variant):
    base, hints = director_prompt_components(variant)
    assert base and hints


def test_early_output_binding_preserves_pending_model_and_first_execution_role():
    canvas, executor = make_canvas(routes=("mock",))
    add(canvas, "a")
    bound = step(canvas, action="set_output", target="a")
    assert bound.execution is None and executor.calls == []
    assert canvas.state is CanvasState.AWAITING_MODEL
    assert canvas.pending_agent_id == "a"
    step(canvas, action="set_model", target="a", runtime_route="mock")
    assert executor.calls == [("a", True, False)]
    assert canvas.selected_output_is_current()
    # A selected output does not automatically end a legal graph.
    assert canvas.active
    backend = MockBackend(['{"action":"finish"}'])
    run = GraphDirector(backend=backend, canvas=canvas, prompt_variant="v2.2").run()
    assert run.finished and run.output == "yes"
    assert len(backend.calls) == len(executor.calls) == 1
    assert not canvas.history[-1].protocol_recovery


def test_role_switch_invalidates_both_peer_component_and_downstream_then_stabilizes():
    canvas, executor = make_canvas()
    for agent in ("a", "b"):
        add(canvas, agent)
    step(canvas, action="set_relation", source="a", target="b", relation="bidirectional")
    add(canvas, "out")
    step(canvas, action="set_layer", target="out", layer=1)
    step(canvas, action="set_relation", source="a", target="out", relation="directed")
    step(canvas, action="set_output", target="a")
    # Runtime must notice identity changes even if a caller thinks everything is clean.
    canvas.graph.set_output("b")
    report = canvas.runtime.execute(task=canvas.worker_task, graph=canvas.graph, dirty_agents=set())
    assert set(report.scheduled_agents) == {"a", "b", "out"}
    assert report.mandatory_revision_calls == 2
    for agent in canvas.graph.nodes:
        assert canvas.runtime.artifact_matches_current_input_signature(
            agent, task=canvas.worker_task, graph=canvas.graph
        )
    binding = canvas.runtime.artifact_input_binding("b")
    assert (
        binding["revision"]
        and binding["peer_artifact_ids"]["a"] != canvas.runtime.artifacts["a"].artifact_id
    )
    assert binding["prior_artifact_id"] != canvas.runtime.artifacts["b"].artifact_id
    count = len(executor.calls)
    report = canvas.runtime.execute(task=canvas.worker_task, graph=canvas.graph, dirty_agents=set())
    assert not report.scheduled_agents and len(executor.calls) == count
    assert report.output == "yes"


def test_failed_output_execution_keeps_old_binding_stale_and_does_not_submit():
    executor = RoleExecutor()
    executor.fail_output = True
    solver = AdaptiveWorkflowSolver(
        director_backend=MockBackend(
            [
                json.dumps(a)
                for a in [
                    dict(action="add_agent", agent_id="a"),
                    prompt("a"),
                    dict(action="set_output", target="a"),
                    dict(action="finish"),
                ]
            ]
        ),
        runtime=MultiAgentRuntime(executor),
        verifier=ExactMatchVerifier(),
        canvas_config=CanvasConfig(max_rounds=50),
        director_prompt_variant="v2.2",
    )
    # Deliberately let the old LOCAL answer match the reference: it must never score.
    task = TaskSpec(
        "stale",
        "Do the entities share the same country?",
        reference="local-old",
        metadata={"dataset": "hotpotqa"},
    )
    result = solver.solve(task, run_id="stale")
    assert not result.director_run.finished and result.director_run.output == ""
    assert not result.answer_submission.valid
    assert result.verification is None
    assert task.metadata["qa_official_metrics"] is None
    from selfplay_graph_flowsteer.application import AdaptiveApplicationResult
    from selfplay_graph_flowsteer.selfplay_runtime import ByteTokenizer, adaptive_result_to_rollout

    rollout = adaptive_result_to_rollout(
        AdaptiveApplicationResult("stale", task, result, (), None, ""),
        ByteTokenizer(),
        rollout_index=0,
        seed=0,
    )
    assert not rollout.trajectory.metadata["training_eligible"]
    assert "output_contract_failure" in rollout.trajectory.metadata["training_exclusion_reasons"]
    assert not rollout.trajectory.metadata["reward_known"]
    assert rollout.trajectory.metadata["typed_policy_failure"] is None
    assert task.metadata["output_contract_failure"]["reason"] == "stale_output_artifact"
    assert not solver.runtime.artifact_matches_current_input_signature("a")
    assert (
        solver.runtime.artifact_input_binding("a")["artifact_id"]
        == solver.runtime.artifacts["a"].artifact_id
    )


def test_upstream_exception_invalidates_downstream_before_it_runs():
    canvas, executor = make_canvas()
    for agent in ("a", "out"):
        add(canvas, agent)
    step(canvas, action="set_layer", target="out", layer=1)
    step(canvas, action="set_relation", source="a", target="out", relation="directed")
    step(canvas, action="set_output", target="out")
    old = canvas.runtime.artifact_input_binding("out")
    executor.fail_agent = "a"
    canvas.graph.nodes["a"].prompt += " Extra public scope."
    with pytest.raises(ValueError, match="simulated"):
        canvas.runtime.execute(task=canvas.worker_task, graph=canvas.graph, dirty_agents={"a"})
    assert canvas.runtime.artifact_input_binding("out") == old
    assert not canvas.selected_output_is_current()
    executor.fail_agent = None
    result = canvas.runtime.execute(task=canvas.worker_task, graph=canvas.graph, dirty_agents=set())
    assert result.output == "yes" and canvas.selected_output_is_current()


def test_selected_recovery_records_consumed_peer_generation_without_rerunning_peer():
    canvas, executor = make_canvas()
    for agent in ("a", "b"):
        add(canvas, agent)
    step(canvas, action="set_relation", source="a", target="b", relation="bidirectional")
    step(canvas, action="set_output", target="a")
    count = len(executor.calls)
    canvas.runtime.cache.clear()
    canvas.runtime.execute(
        task=canvas.worker_task,
        graph=canvas.graph,
        dirty_agents={"a"},
        invalidation_reasons={"a": {"selected_output_recovery_required"}},
    )
    assert executor.calls[count:] == [("a", True, True)]
    clean = canvas.runtime.execute(task=canvas.worker_task, graph=canvas.graph, dirty_agents=set())
    assert clean.scheduled_agents == [] and clean.output == "yes"


@pytest.mark.parametrize("present", [False, True])
def test_counterfactual_suffix_output_binding_replays_only_policy_changes(present):
    canvas, _ = make_canvas()
    for agent in ("a", "b"):
        add(canvas, agent)
    step(canvas, action="set_relation", source="a", target="b", relation="bidirectional")
    prefix = copy.deepcopy(canvas.graph.to_dict())
    step(canvas, action="set_output", target="a")
    final = copy.deepcopy(canvas.graph.to_dict())
    decision = RelationDecision(
        action_index=0,
        source="a",
        target="b",
        probability_present=0.5,
        graph_prefix=prefix,
        chosen_present=True,
        suffix_events=(
            {
                "raw_action": '{"action":"set_output","target":"a"}',
                "graph": final,
                "graph_before": prefix,
            },
        ),
        expected_final_graph=final,
    )
    graph, _, _ = _replay_relation_branch(decision, present=present)
    assert graph.output_agent == "a"
    assert bool(graph.bidirectional_edges) is present
    for node in graph.nodes.values():
        assert node.prompt == canvas.graph.nodes[node.agent_id].prompt
    changed = copy.deepcopy(final)
    changed["nodes"][0]["prompt"] += " hidden policy edit"
    with pytest.raises(ValueError, match="diverged|final graph"):
        _replay_relation_branch(replace(decision, expected_final_graph=changed), present=present)


def test_worker_recovery_keeps_whole_question_and_selected_role():
    backend = MockBackend(['{"answer": null}', '{"answer":"yes","summary":"comparison"}'])
    canvas, _ = make_canvas(routes=("mock",))
    canvas.runtime = MultiAgentRuntime(ModelAgentExecutor(backend))
    add(canvas, "a")
    canvas.graph.nodes["a"].operation_policy_configured = True
    step(canvas, action="set_output", target="a")
    step(canvas, action="set_model", target="a", runtime_route="mock")
    assert len(backend.calls) == 2
    for call in backend.calls:
        assert "selected output Agent" in str(call["messages"])
        assert canvas.worker_task in str(call["messages"])
    context = json.loads(backend.calls[-1]["messages"][-1]["content"])
    assert "original public task" in context["artifact_schema"]["answer"]
    assert canvas.runtime.artifacts["a"].answer == "yes"


def test_prompt_changes_change_executor_bundle_even_without_version_bump(monkeypatch):
    from selfplay_graph_flowsteer import runtime
    from selfplay_graph_flowsteer.selfplay_runtime import _executor_compatibility_signature

    config = AdaptiveApplicationConfig(director_prompt_variant="v2.2")
    before = _executor_compatibility_signature(config.model_manifest())
    original = runtime.selected_output_instruction
    monkeypatch.setattr(
        runtime, "selected_output_instruction", lambda **kw: original(**kw) + " changed"
    )
    assert _executor_compatibility_signature(config.model_manifest()) != before


def test_pats_tracks_actual_director_variant_in_config_and_refiner():
    from selfplay_graph_flowsteer.pats_refiner import review_system_prompt
    from selfplay_graph_flowsteer.pats_semantics import contract_hash, runtime_contract

    config = AdaptiveApplicationConfig(director_prompt_variant="v2.2")
    assert config.pats.director_prompt_variant == "v2.2"
    assert replace(config, director_prompt_variant="v2").pats.director_prompt_variant == "v2"
    assert contract_hash("v2.2") != contract_hash("v2.1")
    assert runtime_contract("v2.2") in review_system_prompt(config.pats, "REVISE")
    assert "before SET_MODEL" in runtime_contract("v2.2")


def test_one_update_policy_lag_cannot_bypass_execution_contract():
    contract = execution_semantics("v2.2")
    lineage = AsyncPolicyLineage(
        target_cycle=2,
        behavior_update_index=1,
        proposer_snapshot="p1",
        solver_snapshot="s1",
        collection_mode="async_one_step_stale",
    ).to_dict()
    sample = TrainingSample(
        "r", "t", (1, 2), (0, 1), 1, 0, metadata={"model_roles": {"execution_semantics": contract}}
    )
    metadata = {"execution_semantics": contract, "policy_lineage": lineage}
    batches = [
        TrainingBatch(role, (sample,), metadata=copy.deepcopy(metadata))
        for role in ("proposer", "solver")
    ]
    kwargs = dict(
        learner_update_index=2, learner_proposer_snapshot="p2", learner_solver_snapshot="s2"
    )
    assert (
        validate_batch_lineage(*batches, expected_execution_semantics=contract, **kwargs)[
            "staleness_updates"
        ]
        == 1
    )
    with pytest.raises(ValueError, match="execution semantics changed"):
        validate_batch_lineage(
            *batches, expected_execution_semantics=execution_semantics("v2.1"), **kwargs
        )
    wrong = copy.deepcopy(batches[1])
    wrong.samples[0].metadata["model_roles"]["execution_semantics"] = None
    with pytest.raises(ValueError, match="execution semantics changed"):
        validate_training_contract(batches[0], wrong)


@pytest.mark.parametrize("change", ["executor", "seed", "environment"])
def test_clean_component_cannot_reuse_old_execution_environment(change):
    canvas, executor = make_canvas()
    add(canvas, "a")
    step(canvas, action="set_output", target="a")
    count = len(executor.calls)
    if change == "executor":
        executor.version = "output-role-regression-v2"
    elif change == "seed":
        canvas.runtime.seed += 1
    else:
        canvas.runtime.environment_fingerprint = "changed-public-evidence"
    assert not canvas.selected_output_is_current()
    report = canvas.runtime.execute(task=canvas.worker_task, graph=canvas.graph, dirty_agents=set())
    assert len(executor.calls) == count + 1 and report.output == "yes"
    assert canvas.selected_output_is_current()


@pytest.mark.parametrize("selected_success,text_failure", [(False, False), (True, True)])
def test_alfworld_scores_selected_episode_and_keeps_success_after_text_failure(
    monkeypatch, selected_success, text_failure
):
    from types import SimpleNamespace

    from selfplay_graph_flowsteer.dataset_actions import DatasetActionAdapter, DatasetActionRegistry
    from selfplay_graph_flowsteer.observability import VerificationResult

    lifecycle = SimpleNamespace(
        bind_task=lambda task: None,
        close_all=lambda: None,
        environment_fingerprint="isolated-fixture",
        result_for=lambda agent: {},
    )
    monkeypatch.setattr(
        "selfplay_graph_flowsteer.adaptive.alfworld_lifecycles", lambda tools: (lifecycle,)
    )
    registry = DatasetActionRegistry(
        (
            DatasetActionAdapter(
                adapter_id="alfworld",
                datasets=("alfworld",),
                action_names=(),
                initial_action_budget=0,
                revision_action_budget=0,
                total_action_budget=0,
            ),
        )
    )

    class EnvironmentExecutor(RoleExecutor):
        def execute(self, **kwargs):
            artifact = super().execute(**kwargs)
            won = selected_success if artifact.agent_id == "chosen" else True
            artifact.environment_result = dict(
                environment_completed=True, done=True, won=won, score=float(won)
            )
            return artifact

    class OfficialVerifier:
        def verify(self, task, answer):
            won = task.metadata["alfworld_environment_result"]["won"]
            return VerificationResult(float(won), won, "offline_environment_fixture")

    def after_director(task, canvas, run):
        if text_failure:
            canvas.runtime.artifacts["chosen"].answer = "WORKER_BACKEND_FAILURE"
            run.output = "WORKER_BACKEND_FAILURE"

    actions = []
    for agent in ("chosen", "peer"):
        actions.extend([dict(action="add_agent", agent_id=agent), prompt(agent)])
    actions.extend(
        [
            dict(action="set_relation", source="chosen", target="peer", relation="bidirectional"),
            dict(action="set_output", target="chosen"),
            dict(action="finish"),
        ]
    )
    solver = AdaptiveWorkflowSolver(
        director_backend=MockBackend([json.dumps(a) for a in actions]),
        runtime=MultiAgentRuntime(EnvironmentExecutor()),
        verifier=OfficialVerifier(),
        action_registry=registry,
        post_director_hook=after_director,
    )
    task = TaskSpec("alf", "Complete the public environment task", metadata={"dataset": "alfworld"})
    result = solver.solve(task, run_id="alf")
    assert result.verification.passed is selected_success
    assert task.metadata["alfworld_environment_result"]["won"] is selected_success
    assert [row["agent_id"] for row in task.metadata["alfworld_unselected_winning_episodes"]] == [
        "peer"
    ]
    assert result.director_run.graph["output_agent"] == "chosen"
    if text_failure:
        assert task.metadata["worker_backend_failure"]


@pytest.mark.parametrize("drift", [False, True])
def test_frontier_transport_override_never_bypasses_output_contract(tmp_path, drift):
    from types import SimpleNamespace

    from selfplay_graph_flowsteer.observability import VerificationResult
    from selfplay_graph_flowsteer.rollouts import TokenizedDirectorTrajectory
    from selfplay_graph_flowsteer.selfplay import AlternatingSnapshots, ProposedTask, SolverRollout
    from selfplay_graph_flowsteer.selfplay_runtime import (
        ByteTokenizer,
        SelfPlayRolloutRunner,
        SelfPlayRunConfig,
    )
    from tests.test_frontier_stability import _graph

    contract = execution_semantics("v2.2")
    current = execution_semantics("v2.1") if drift else contract
    proposal = ProposedTask(
        TaskSpec("q", "question", reference="answer", metadata={"dataset": "hotpotqa"}),
        response="proposal",
        token_ids=(1,),
        action_mask=(1,),
    )
    rows = []
    for index, reward in enumerate((1.0, 0.0)):
        graph = _graph(index + 1)
        rows.append(
            SolverRollout(
                TokenizedDirectorTrajectory(
                    rollout_id=f"q-{index}",
                    task_id="q",
                    token_ids=(1,),
                    action_mask=(1,),
                    reward=reward,
                    graph=graph.to_dict(),
                    seed=0,
                    metadata={
                        "task_reward": reward,
                        "model_roles": {"execution_semantics": contract, "transport": "old"},
                    },
                ),
                graph,
            )
        )
    evaluations = []

    class Application:
        config = SimpleNamespace(
            model_manifest=lambda: {"execution_semantics": current, "transport": "repaired"}
        )
        runtime = SimpleNamespace(seed=0)

        def close(self):
            pass

        def evaluate_graph(self, task, graph, **kwargs):
            evaluations.append(graph.output_agent)
            return dict(prediction="answer", verification=VerificationResult(1.0, True, "offline"))

    (tmp_path / "frontier_execution_override.json").write_text(
        json.dumps({"allow_current_executor": True})
    )
    runner = SelfPlayRolloutRunner(
        proposer=SimpleNamespace(),
        application_factory=lambda _: Application(),
        tokenizer=ByteTokenizer(),
        snapshots=AlternatingSnapshots("p", "s"),
        output_dir=tmp_path,
        config=SelfPlayRunConfig(rollouts_per_task=2, frontier_reverify_fraction=1.0),
    )
    if drift:
        with pytest.raises((RuntimeError, ValueError), match="execution semantics changed"):
            runner._collect_frontier_reverification([proposal], rows)
        assert evaluations == []
    else:
        verified = runner._collect_frontier_reverification([proposal], rows)
        assert len(evaluations) == 2 and verified["q"]["execution_semantics"] == contract
        current = execution_semantics("v2.1")
        with pytest.raises(ValueError, match="execution semantics changed"):
            runner._collect_frontier_reverification([proposal], rows)
        assert len(evaluations) == 2  # A stale journal cannot bypass the gate.
