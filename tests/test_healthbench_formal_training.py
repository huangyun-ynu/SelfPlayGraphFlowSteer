from __future__ import annotations

import json
import re
import tomllib
from dataclasses import replace
from pathlib import Path

import pytest

from selfplay_graph_flowsteer import application as app_module
from selfplay_graph_flowsteer.llm import MockBackend


@pytest.fixture
def formal_config(monkeypatch, tmp_path):
    """Load actual HealthBench settings without unrelated environment assets."""
    path = Path(__file__).resolve().parents[1] / "configs/formal_training.toml"
    monkeypatch.setattr(app_module, "_load_project_env", lambda _: None)
    for name in (
        "SPGFS_ALLOWED_PHYSICAL_GPUS",
        "SPGFS_SWE_CVM_AUTO_START",
        "SPGFS_SWE_CVM_AUTO_STOP",
    ):
        monkeypatch.delenv(name, raising=False)
    payload = tomllib.loads(path.read_text())
    monkeypatch.setenv(
        "SPGFS_ALLOWED_PHYSICAL_GPUS",
        ",".join(str(gpu) for gpu in payload["resources"]["allocated_gpu_ids"]),
    )
    for runtime in [*payload["runtimes"].values(), *payload["models"].values()]:
        if runtime.get("api_key_env"):
            monkeypatch.setenv(runtime["api_key_env"], "synthetic-test-key")
    text = path.read_text()
    for name in ("alfworld", "swe"):
        text = re.sub(rf"(\[{name}\]\s*\nenabled\s*=\s*)true", r"\g<1>false", text)
    isolated = tmp_path / "configs" / "formal_training.toml"
    isolated.parent.mkdir()
    isolated.write_text(text)
    return app_module.load_adaptive_config(isolated)


def test_formal_healthbench_keeps_director_route_choice_and_student_judge(formal_config):
    config = formal_config
    assert config.worker_runtime_routes == ("gpt", "grok", "gemini", "deepseek", "minimax")
    assert config.dataset_worker_routes.get(
        "healthbench_professional", config.worker_runtime_routes
    ) == config.worker_runtime_routes
    assert config.runtime_endpoint_pools["gpt"] == ("gpt", "gpt_eco", "gpt_student")
    assert not config.dataset_route_overrides.get("healthbench_professional")
    assert config.healthbench_judge_runtime_route == "gpt_student"
    assert config.runtime_pool()["gpt_student"].max_concurrency == 5
    for route in config.runtime_endpoint_pools["gpt"]:
        assert config.runtime_pool()[route].reasoning_effort == "low"
    gateway = app_module._gateway_config(config.solver_model, {"graph-director": 0.6})
    assert gateway.roles["graph-director"].enable_thinking is True


def test_formal_application_preserves_history_and_complete_answer(formal_config, tmp_path):
    # Exercise the application factory used by primary training collection.
    # Disable unrelated environment services and skill retrieval in this synthetic run.
    config = replace(
        formal_config,
        skillbank_enabled=False,
        pats=replace(formal_config.pats, enabled=False),
        persist_runtime_updates=False,
        trace_path=tmp_path / "traces.jsonl",
        route_health_path=tmp_path / "route_health.json",
        healthbench_judge_audit_path=tmp_path / "private/judge",
        **{
            name: replace(
                getattr(formal_config, name), enabled=False,
                **({"hotpotqa_search_enabled": False}
                   if name == "retrieval" and hasattr(formal_config.retrieval, "hotpotqa_search_enabled")
                   else {}),
            )
            for name in ("retrieval", "aime_actions", "webshop", "alfworld", "swe")
        },
    )
    director = MockBackend([
        '{"action":"add_agent"}',
        json.dumps({
            "action": "set_prompt", "target": "agent_1", "role": "Responder",
            "objective": "Explain the finding using the public conversation.",
            "scope": "The supplied study and its limitation.",
            "expected_output": "A complete response including uncertainty.",
        }),
        '{"action":"set_model","target":"agent_1","runtime_route":"gpt"}',
        '{"action":"set_output","target":"agent_1"}',
        '{"action":"finish"}',
    ])
    answer = "The association in this synthetic study does not establish causation."
    worker = MockBackend([json.dumps({
        "answer": answer, "summary": "INTERNAL_SUMMARY_ONLY", "confidence": 0.7,
        "evidence": [], "unresolved_issues": [], "tool_summary": [],
    })] * 2)  # Initial execution, then the existing output-role rerun.
    judge = MockBackend([json.dumps({
        "criteria_met": True, "explanation": "The limitation is present.",
    })])
    application = app_module.create_adaptive_application(
        config, mock=True, director_backend=director,
        worker_backend=worker, distiller_backend=judge,
    )
    result = application.solve(
        "What can we conclude?",
        task_id="formal-healthbench-synthetic",
        metadata={
            "dataset": "healthbench_professional", "verifier": "healthbench_rubric",
            "conversation": {"messages": [
                {"role": "user", "content": "PUBLIC_STUDY_CONTEXT: an observational study."},
                {"role": "assistant", "content": "PUBLIC_PRIOR_REPLY: an association was reported."},
                {"role": "user", "content": "What can we conclude?"},
            ]},
        },
        private_verifier_payload={"rubric_items": [
            {"criterion_text": "PRIVATE_RUBRIC: mention uncertainty.", "points": 5},
        ]},
    )
    initial = json.dumps(director.calls[0]["messages"])
    assert "PUBLIC_STUDY_CONTEXT" in initial
    assert "PUBLIC_PRIOR_REPLY" in initial
    for call in worker.calls:
        assert "complete, self-contained result" in call["messages"][0]["content"]
        context = json.loads(call["messages"][1]["content"])["public_task_context"]
        assert "PUBLIC_STUDY_CONTEXT" in context
        assert "complete, self-contained result" in context
        assert "all supporting explanation in summary or evidence" not in context
    assert "PRIVATE_RUBRIC" not in json.dumps([director.calls, worker.calls])
    assert result.solver_result.answer_submission.submitted_answer == answer
    assert len(judge.calls) == 1
    judge_input = json.dumps(judge.calls[0]["messages"])
    assert answer in judge_input
    assert "INTERNAL_SUMMARY_ONLY" not in judge_input
    detail = json.loads(result.solver_result.trace.verification.detail)
    assert detail["criteria_met"] == 1
    assert detail["raw_score"] == 1.0
