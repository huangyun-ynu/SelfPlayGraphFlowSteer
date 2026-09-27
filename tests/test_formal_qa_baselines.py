from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import pytest

from selfplay_graph_flowsteer import application as app_module
from selfplay_graph_flowsteer.adaptive import AdaptiveWorkflowSolver
from selfplay_graph_flowsteer.answer_submission import AnswerFinalizer, AnswerSubmissionConfig
from selfplay_graph_flowsteer.application import (
    _gateway_config,
    _runtime_gateway_config,
    create_adaptive_application,
    load_adaptive_config,
)
from selfplay_graph_flowsteer.benchmark_reporting import benchmark_summary
from selfplay_graph_flowsteer.cli import _apply_fresh_route_report
from selfplay_graph_flowsteer.contracts import AgentNode
from selfplay_graph_flowsteer.curriculum import ADSBoundaryScheduler, FixedTaskPool, TSDSRetriever
from selfplay_graph_flowsteer.dataset_actions import default_dataset_action_registry
from selfplay_graph_flowsteer.llm import MockBackend
from selfplay_graph_flowsteer.nq_frozen_context import prepare_pool, validate_frozen_context
from selfplay_graph_flowsteer.observability import FlowSteerQAVerifier, TaskSpec
from selfplay_graph_flowsteer.runtime import (
    ModelAgentExecutor,
    MultiAgentRuntime,
    RoutedModelAgentExecutor,
)
from selfplay_graph_flowsteer.selfplay import FixedPoolQwenProposer, SelfPlaySeed

from .helpers import RecordingExecutor


@pytest.fixture(autouse=True)
def _offline_config_credentials(monkeypatch):
    # These tests use mock backends but load deployment configuration. They
    # must not require real credentials or inherit a developer's API keys.
    for name in (
        "NEXUS_PRO_API_KEY", "NEXUS_API_KEY", "FLOWSTEER_API_KEY", "UUAPI_API_KEY",
        "UUAPI_API_KEY_2", "DEEPSEEK_API_KEY", "MINIMAX_API_KEY",
    ):
        monkeypatch.setenv(name, "unit-test-only")


def _row(dataset="nq_open", question="Who wrote the fictional novel?"):
    return {
        "id": f"{dataset}/train/1", "dataset": dataset, "prompt": question,
        "reference": "PRIVATE_REFERENCE_SENTINEL", "split": "train",
        "metadata": {"dataset": dataset, "split": "train"},
    }


def _fetch(question, *, service_url, top_k):
    assert "PRIVATE_REFERENCE_SENTINEL" not in question
    return [{"id": str(i), "title": "Book", "text": f"Public passage {i}."}
            for i in range(top_k)]


def test_prepare_mixed_pool_preserves_rows_and_reuses_reference_blind_cache(tmp_path):
    rows = [_row("hotpotqa"), _row(), _row("aime")]
    source, output = tmp_path / "source.jsonl", tmp_path / "frozen.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    calls = []

    def fetch(*args, **kwargs):
        calls.append(args)
        return _fetch(*args, **kwargs)

    prepare_pool(source, output, service_url="http://local/retrieve", fetcher=fetch)
    first = output.read_bytes()
    prepared = [json.loads(line) for line in first.splitlines()]
    assert prepared[0] == rows[0] and prepared[2] == rows[2]
    assert prepared[1]["reference"] == rows[1]["reference"]
    assert prepared[1]["id"] == rows[1]["id"]
    assert prepared[1]["split"] == "train"
    validate_frozen_context(prepared[1], top_k=8)
    prepare_pool(source, output, service_url="http://local/retrieve", fetcher=fetch)
    assert len(calls) == 1 and output.read_bytes() == first
    assert all("PRIVATE_REFERENCE_SENTINEL" not in p.read_text()
               for p in (tmp_path / "frozen.jsonl.evidence-cache").glob("*.json"))


def test_failed_evidence_preparation_never_publishes_partial_pool(tmp_path):
    source, output = tmp_path / "source.jsonl", tmp_path / "frozen.jsonl"
    source.write_text(json.dumps(_row()) + "\n")
    output.write_text("previous complete output\n")
    with pytest.raises(ValueError, match="8 frozen inline passages"):
        prepare_pool(source, output, service_url="http://local", fetcher=lambda *a, **k: [])
    assert output.read_text() == "previous complete output\n"
    with pytest.raises(ValueError, match="must differ"):
        prepare_pool(source, source, service_url="http://local")


@pytest.mark.parametrize("anchor_source", ["pool", "seed"])
def test_frozen_nq_proposer_sees_questions_but_solver_keeps_evidence(tmp_path, anchor_source):
    questions = ["Who designed the Silver Observatory?", "When did the Amber Observatory open?"]
    rows = [
        {**_row(question=question), "id": f"nq:{i}", "embedding": [1.0, i / 10]}
        for i, question in enumerate(questions)
    ]
    source, frozen = tmp_path / "source.jsonl", tmp_path / "frozen.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def fetch(question, *, service_url, top_k):
        return [{"id": str(i), "title": "EVIDENCE_ONLY_SENTINEL",
                 "text": "Long public evidence passage. " * 40} for i in range(top_k)]

    prepare_pool(source, frozen, service_url="http://offline", fetcher=fetch)
    pool = FixedTaskPool.from_jsonl([frozen])
    scheduler = ADSBoundaryScheduler(pool, active_clusters=1, mini_cluster_size=2, cooldown=0)
    backend = MockBackend(['{"candidate_id":"nq:1"}'])
    proposer = FixedPoolQwenProposer(
        backend, pool, scheduler, TSDSRetriever(pool, max_k=2, kde_k=2, sigma=0),
        candidate_count=4,
    )
    anchor = pool.tasks["nq:0"].task
    original_prompt = pool.tasks["nq:1"].task.prompt
    original_docs = json.dumps(pool.tasks["nq:1"].task.metadata["context_documents"])
    proposal = proposer.propose(
        SelfPlaySeed(
            anchor.prompt,
            seed_id="nq:0" if anchor_source == "pool" else "external-anchor",
            metadata={"dataset": "nq_open", **(
                {"original_question": questions[0]} if anchor_source == "seed" else {}
            )},
        ),
        task_id="selected",
    )
    payload = json.loads(backend.calls[0]["messages"][1]["content"])
    assert payload["anchor"] == questions[0]
    assert all(candidate["prompt_preview"] == questions[int(candidate["candidate_id"].split(":")[1])]
               for candidate in payload["candidates"])
    actual_messages = json.dumps(backend.calls[0]["messages"])
    assert "EVIDENCE_ONLY_SENTINEL" not in actual_messages
    assert "PRIVATE_REFERENCE_SENTINEL" not in actual_messages
    assert proposal.task.prompt == original_prompt
    assert json.dumps(proposal.task.metadata["context_documents"]) == original_docs
    assert proposal.task.reference == "PRIVATE_REFERENCE_SENTINEL"
    assert pool.tasks["nq:1"].task.prompt == original_prompt
    validate_frozen_context({"id": proposal.task.task_id, "prompt": proposal.task.prompt,
                             "metadata": proposal.task.metadata}, top_k=8)


def test_promoted_action_visibility_is_dataset_specific():
    registry = default_dataset_action_registry(["search"], hotpotqa_search_enabled=True)
    hp = TaskSpec("hp", "question", metadata={"dataset": "hotpotqa"})
    nq = TaskSpec("nq", "question", metadata={
        "dataset": "nq_open", "evidence_mode": "provided_context_inline"})
    assert registry.resolve(hp).action_names == ("search",)
    assert registry.resolve(nq).action_names == ()
    assert default_dataset_action_registry(["search"]).resolve(hp).action_names == ()


@pytest.mark.parametrize("overrides", [{}, {"nq_open": False}])
@pytest.mark.parametrize("dataset", ["hotpotqa", "nq_open", "aime"])
def test_director_thinking_defaults_and_control_overrides_in_actual_calls(dataset, overrides):
    backend = MockBackend([
        '{"action":"add_agent","agent_id":"solver"}',
        '{"action":"set_prompt","target":"solver","role":"Analyst",'
        '"objective":"Solve the task.","scope":"Reason independently.",'
        '"expected_output":"Return a finding."}',
        '{"action":"set_output","target":"solver"}',
        '{"action":"finish"}',
    ])
    solver = AdaptiveWorkflowSolver(
        director_backend=backend, runtime=MultiAgentRuntime(RecordingExecutor()),
        director_enable_thinking=True, director_thinking_by_dataset=overrides,
    )
    solver.solve(TaskSpec("sample", "Find the answer.", metadata={"dataset": dataset}), run_id="test")
    assert backend.calls
    assert all(call["enable_thinking"] is overrides.get(dataset, True) for call in backend.calls)


def test_formal_nq_rejects_online_task_before_any_model_call():
    backend = MockBackend([])
    solver = AdaptiveWorkflowSolver(
        director_backend=backend, runtime=MultiAgentRuntime(RecordingExecutor()),
        required_nq_frozen_top_k=8,
    )
    with pytest.raises(ValueError, match="8 frozen inline passages"):
        solver.solve(TaskSpec("nq", "Question", metadata={"dataset": "nq_open"}), run_id="test")
    assert not backend.calls


@pytest.mark.parametrize("dataset", ["hotpotqa", "nq_open"])
def test_flowsteer_partial_credit_keeps_independent_exact_match_audit(dataset):
    director = MockBackend([
        '{"action":"add_agent","agent_id":"solver"}',
        '{"action":"set_prompt","target":"solver","role":"Answerer",'
        '"objective":"Answer the question.","scope":"Use the available evidence.",'
        '"expected_output":"Return the requested name."}',
        '{"action":"set_output","target":"solver"}', '{"action":"finish"}',
    ])
    worker = MockBackend(['{"answer":"Media Puzzle and Damien Oliver","summary":"Public evidence."}'] * 4)
    solver = AdaptiveWorkflowSolver(
        director_backend=director, runtime=MultiAgentRuntime(ModelAgentExecutor(worker)),
        verifier=FlowSteerQAVerifier(),
        answer_finalizer=AnswerFinalizer(AnswerSubmissionConfig(enabled=True)),
    )
    task = TaskSpec("partial-answer", "Which horse won the cup?", reference="Media Puzzle",
                    metadata={"dataset": dataset})
    result = solver.solve(task, run_id="partial-answer")
    assert result.trace.verification.passed is True
    assert result.trace.verification.score == pytest.approx(0.7)
    metrics = task.metadata["qa_official_metrics"]
    assert metrics["answer_em"] == 0.0
    assert metrics["answer_f1"] == pytest.approx(4 / 7)
    report = benchmark_summary([{
        "task_id": task.task_id, "task_outcome_passed": result.trace.verification.passed,
        "qa_answer_em": metrics["answer_em"], "qa_answer_f1": metrics["answer_f1"],
    }], dataset)
    assert report["metrics"]["success_rate"]["observed"]["mean"] == 1.0
    assert report["metrics"]["answer_em"]["observed"]["mean"] == 0.0
    assert report["metrics"]["answer_f1"]["observed"]["mean"] == pytest.approx(4 / 7)


def test_baseline_formatter_does_not_choose_new_answer_from_evidence():
    task = TaskSpec("nq", "Who wrote the book?", metadata={
        "dataset": "nq_open", "evidence_mode": "provided_context_inline",
        "original_question": "Who wrote the book?",
        "context_documents": [{"id": "p", "text": "Another Person wrote it."}],
    })
    result = AnswerFinalizer(AnswerSubmissionConfig(enabled=True)).finalize(
        task, "Original Person", raw_summary="Another Person wrote it."
    )
    assert result.submitted_answer == "Original Person"
    assert result.method == "qa_deterministic_extraction"


@pytest.mark.parametrize("name", ["formal_training.toml", "formal_training_h200.local.toml", "formal_eval_h200.local.toml"])
def test_formal_configs_keep_director_model_choice_and_qwen_thinking(name, monkeypatch):
    config = load_adaptive_config(Path("configs") / name, validate=False)
    monkeypatch.setenv("SPGFS_ALLOWED_PHYSICAL_GPUS", ",".join(map(str, config.allocated_gpu_ids)))
    # Deployment assets are checked by real environment smoke tests, not this
    # portable configuration regression.
    replace(
        config,
        alfworld=replace(config.alfworld, enabled=False),
        swe=replace(config.swe, enabled=False),
    ).validate()
    assert config.director_prompt_variant == "v2.2"
    assert all(value is True for value in config.director_thinking_by_dataset.values())
    assert config.proposer_model.enable_thinking is True
    assert config.solver_model.enable_thinking is True
    assert config.worker_runtime_routes == ("gpt", "grok", "gemini", "deepseek", "minimax")
    assert config.retrieval.nq_frozen_top_k == 8
    assert not config.retrieval.enabled
    assert not config.retrieval.hotpotqa_search_enabled
    registry = default_dataset_action_registry(())
    for dataset, metadata in (
        ("nq_open", {"evidence_mode": "provided_context_inline"}),
        ("hotpotqa", {}),
    ):
        adapter = registry.resolve(TaskSpec(
            dataset, "Read the supplied evidence.", metadata={"dataset": dataset, **metadata},
        ))
        assert adapter is not None
        assert adapter.action_names == ()
    assert config.answer_submission.enabled
    assert vars(config.answer_submission) == {"enabled": True}
    assert config.skillbank_enabled  # Formal training still learns skills.
    executor = RoutedModelAgentExecutor(
        {route: MockBackend([]) for route in config.runtime_pool()},
        config.worker_runtime_routes, dataset_route_overrides=config.dataset_route_overrides,
    )
    for dataset in ("hotpotqa", "nq_open", "aime"):
        assert dataset not in config.dataset_route_overrides
        for route in config.worker_runtime_routes:
            node = AgentNode("worker", metadata={
                "runtime_route": route, "system_managed_contract": {"dataset": dataset}})
            assert executor.route_for(node) == route


def test_historical_fixed_route_and_nq_no_thinking_are_isolated_to_control(monkeypatch):
    monkeypatch.setenv("SPGFS_ALLOWED_PHYSICAL_GPUS", "0")
    config = load_adaptive_config("configs/qa_best_recorded_eval.toml")
    assert config.worker_runtime_routes == ("deepseek",)
    assert {key: config.director_thinking_by_dataset[key] for key in ("hotpotqa", "nq_open")} == {"hotpotqa": True, "nq_open": False}
    assert not config.skillbank_context_enabled
    assert not config.runtime.enable_thinking
    assert config.retrieval.nq_frozen_top_k == 8
    assert vars(config.answer_submission) == {"enabled": True}


def test_formal_qwen_gateway_enables_both_policy_roles_and_accepts_explicit_override():
    config = load_adaptive_config("configs/formal_training.toml", validate=False)
    proposer = _gateway_config(config.proposer_model, {"proposer": 0.8})
    director = _gateway_config(config.solver_model, {"graph-director": 0.6})
    assert proposer.roles["proposer"].enable_thinking is True
    assert director.roles["graph-director"].enable_thinking is True
    control = _gateway_config(config.solver_model, {"graph-director": 0.6},
                              director_enable_thinking=False)
    assert control.roles["graph-director"].enable_thinking is False


def test_fresh_route_qualification_preserves_and_requires_dataset_targets(tmp_path, monkeypatch):
    config = load_adaptive_config("configs/formal_training.toml", validate=False)
    config = replace(config, alfworld=replace(config.alfworld, enabled=False),
                     swe=replace(config.swe, enabled=False))
    # Explicit control override, never part of the formal configuration.
    config = replace(config, worker_runtime_routes=("gpt",),
                     dataset_route_overrides={"hotpotqa": {"gpt": "deepseek"}})
    monkeypatch.setenv("SPGFS_ALLOWED_PHYSICAL_GPUS", ",".join(map(str, config.allocated_gpu_ids)))
    report = tmp_path / "routes.json"
    requested = list(config.runtime_pool())
    report.write_text(json.dumps({"routes_requested": requested, "usable_routes": requested}))
    args = argparse.Namespace(route_report=report, mock=False, max_route_report_age_s=1800,
                              minimum_selected_routes=1, route_subset="")
    selected, _ = _apply_fresh_route_report(config, args)
    assert "deepseek" in selected.runtime_pool()
    assert selected.dataset_route_overrides["hotpotqa"]["gpt"] == "deepseek"
    report.write_text(json.dumps({"routes_requested": requested,
                                  "usable_routes": [r for r in requested if r != "deepseek"]}))
    with pytest.raises(ValueError, match="override routes must be freshly qualified"):
        _apply_fresh_route_report(config, args)


def test_application_wires_thinking_and_frozen_evidence_and_explicit_override(tmp_path, monkeypatch):
    config = load_adaptive_config("configs/formal_training.toml", validate=False)
    monkeypatch.setenv("SPGFS_ALLOWED_PHYSICAL_GPUS", ",".join(map(str, config.allocated_gpu_ids)))
    config = replace(config, skillbank_enabled=False, route_health_path=tmp_path / "health.json",
                     trace_path=tmp_path / "trace.jsonl", swe=replace(config.swe, enabled=False),
                     alfworld=replace(config.alfworld, enabled=False), webshop=replace(config.webshop, enabled=False))
    app = create_adaptive_application(config, mock=True)
    assert app.solver.director_thinking_by_dataset == config.director_thinking_by_dataset
    assert app.solver.director_enable_thinking is True
    assert app.solver.required_nq_frozen_top_k == 8
    app.close()
    app = create_adaptive_application(config, mock=True, director_enable_thinking=False)
    assert app.solver.director_thinking_by_dataset == {}
    assert app.solver.director_enable_thinking is False
    app.close()


@pytest.mark.parametrize("dataset", ["hotpotqa", "nq_open"])
@pytest.mark.parametrize("chosen", ["grok", "deepseek"])
def test_formal_application_executes_director_qa_choice_with_thinking(dataset, chosen, tmp_path, monkeypatch):
    config = load_adaptive_config("configs/formal_training.toml", validate=False)
    monkeypatch.setenv("SPGFS_ALLOWED_PHYSICAL_GPUS", ",".join(map(str, config.allocated_gpu_ids)))
    config = replace(
        config, skillbank_enabled=False, persist_runtime_updates=False, verifier="none",
        route_health_path=tmp_path / "health.json", trace_path=tmp_path / "traces.jsonl",
        **{name: replace(getattr(config, name), enabled=False) for name in ("swe", "alfworld", "webshop")},
    )
    director = MockBackend([
        '{"action":"add_agent","agent_id":"worker"}',
        json.dumps({"action": "set_prompt", "target": "worker", "role": "Answerer",
                    "objective": "Answer the question.", "scope": "Use the public evidence.",
                    "expected_output": "Return the requested name."}),
        json.dumps({"action": "set_model", "target": "worker", "runtime_route": chosen}),
        '{"action":"set_output","target":"worker"}', '{"action":"finish"}',
    ])
    backends = {route: MockBackend(['{"answer":"Final Answer: A Person","evidence":[]}'] * 4)
                for route in config.runtime_pool()}
    for route, backend in backends.items():
        backend.config = _runtime_gateway_config(config.runtime_pool()[route], {"worker": 0.0}, route_name=route)
    monkeypatch.setattr(app_module, "_create_runtime_backend",
                        lambda runtime, *, route_name: backends[route_name])
    app = create_adaptive_application(config, director_backend=director, distiller_backend=MockBackend([]))
    question = "Who wrote the book?"
    metadata = {"dataset": dataset}
    prompt = question
    if dataset == "nq_open":
        docs = _fetch(question, service_url="unused", top_k=8)
        prompt += "\n" + "\n".join(doc["text"] for doc in docs)
        metadata.update(original_question=question, context_documents=docs,
                        evidence_mode="provided_context_inline")
    try:
        result = app.solve(prompt, task_id="formal-qa-synthetic", metadata=metadata)
        assert app.solver.answer_finalizer is not None
        submission = result.solver_result.answer_submission
        assert submission.raw_answer == "Final Answer: A Person"
        assert submission.submitted_answer == "A Person"
        assert submission.method == "qa_deterministic_extraction"
        assert not any(call["role"] == "answer-formatter"
                       for backend in backends.values() for call in backend.calls)
        assert director.calls and all(call["enable_thinking"] is True for call in director.calls)
        used = {route for route, backend in backends.items() if backend.calls}
        assert used and used <= set(config.runtime_endpoint_pools.get(chosen, (chosen,)))
        assert app.solver.runtime_routes == config.worker_runtime_routes
    finally:
        app.close()
