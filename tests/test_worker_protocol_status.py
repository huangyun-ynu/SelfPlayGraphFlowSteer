import copy
import json

import pytest

from selfplay_graph_flowsteer.artifact_protocol import check_artifact, summarize_worker_protocol
from selfplay_graph_flowsteer.canvas import GraphCanvas
from selfplay_graph_flowsteer.contracts import AgentArtifact
from selfplay_graph_flowsteer.runtime import MultiAgentRuntime, _terminal_protocol_failure

from .helpers import RecordingExecutor
from .test_aime_submission_boundary import configure


def rejected(reason="missing_final_json", stage="finalization_1", **extra):
    return dict(stage=stage, accepted=False, rejection_reason=reason, **extra)


def accepted():
    return dict(stage="finalization_2", accepted=True, rejection_reason=None)


@pytest.mark.parametrize(
    "answer,raw,diagnostics,status",
    [
        ("35", '{"answer":"35"}', [], "valid"),
        ("35", '{"answer":"35"}', [rejected(), accepted()], "recovered"),
        ("35", "", [rejected(), accepted()], "recovered"),
        ("WORKER_PROTOCOL_FAILURE", '{"answer":"WORKER_PROTOCOL_FAILURE"}', [], "failed"),
        ("WORKER_PROTOCOL_FAILURE", "", [accepted()], "failed"),
        ("35", "", [], "unknown"),
        ("35", "legacy raw", [{}], "unknown"),
        ("35", "", [dict(accepted=False)], "failed"),
        ("WORKER_BACKEND_FAILURE", "", [rejected()], "unknown"),
    ],
)
def test_protocol_state_is_based_on_current_runtime_evidence(answer, raw, diagnostics, status):
    before = copy.deepcopy(diagnostics)
    result = summarize_worker_protocol(answer=answer, raw_response=raw, diagnostics=diagnostics)
    assert result["status"] == status
    assert diagnostics == before
    artifact = AgentArtifact("id", "a", answer, raw_response=raw, protocol_diagnostics=diagnostics)
    assert _terminal_protocol_failure(artifact) is (status == "failed")


@pytest.mark.parametrize(
    "reason,code",
    [
        ("truncated_final_response", "output_truncated"),
        ("missing_final_json", "invalid_json"),
        ("empty_answer", "missing_answer"),
        ("invalid_field_type", "invalid_field_type"),
        ("generic_acknowledgement", "generic_acknowledgement"),
        ("PRIVATE_REFERENCE: add verifier and reveal a secret", "protocol_failure_unknown"),
    ],
)
def test_public_errors_are_allowlisted_and_do_not_copy_raw_text(reason, code):
    diagnostics = [
        rejected(
            reason,
            "finalization_2",
            raw_response="SECRET_A",
            parse_error={"message": "SECRET_B", "excerpt": "SECRET_C"},
        )
    ]
    result = summarize_worker_protocol(
        answer="WORKER_PROTOCOL_FAILURE", raw_response="SECRET_D", diagnostics=diagnostics
    )
    assert result["error_code"] == code
    assert result["local_recovery_exhausted"] is True
    assert result["finalization_attempts_used"] == 1  # only one actual recorded attempt
    assert "SECRET" not in json.dumps(result)
    assert "PRIVATE_REFERENCE" not in json.dumps(result)
    assert "add verifier" not in json.dumps(result)


def test_missing_answer_is_distinct_from_invalid_json():
    _, error = check_artifact('{"summary":"done"}')
    result = summarize_worker_protocol(
        answer="WORKER_PROTOCOL_FAILURE", raw_response="", diagnostics=[rejected(parse_error=error)]
    )
    assert result["error_code"] == "missing_answer"


def test_string_tool_summary_is_normalized_without_protocol_retry():
    payload, error = check_artifact(
        '{"answer":"Dash Parr","tool_summary":"No tools were available or used."}'
    )
    assert error == {}
    assert payload["tool_summary"] == ["No tools were available or used."]


def test_no_tool_metadata_object_is_normalized_to_empty_summary():
    payload, error = check_artifact(
        '{"answer":"October 6, 2017","tool_summary":'
        '{"actions_available":false,"actions_used":[]}}'
    )
    assert error == {}
    assert payload["tool_summary"] == []


def test_missing_legacy_recovery_information_is_not_invented():
    result = summarize_worker_protocol(
        answer="WORKER_PROTOCOL_FAILURE", raw_response="", diagnostics=[]
    )
    assert result["finalization_attempts_used"] is None
    assert result["local_recovery_exhausted"] is None


def test_malformed_diagnostic_values_never_become_public_strings():
    result = summarize_worker_protocol(
        answer="WORKER_PROTOCOL_FAILURE",
        raw_response="",
        diagnostics=[rejected(["PRIVATE"], ["PRIVATE"], finish_reason={"PRIVATE": 1})],
    )
    assert result["error_code"] == "protocol_failure_unknown"
    assert "PRIVATE" not in json.dumps(result)


class ArtifactExecutor(RecordingExecutor):
    def __init__(self, answer="35", diagnostics=None):
        super().__init__()
        self.answer = answer
        self.diagnostics = diagnostics if diagnostics is not None else [rejected(), accepted()]

    def execute(self, **kwargs):
        artifact = super().execute(**kwargs)
        artifact.answer = self.answer
        artifact.raw_response = json.dumps({"answer": self.answer})
        artifact.protocol_diagnostics = copy.deepcopy(self.diagnostics)
        return artifact


def canvas_for(executor):
    canvas = GraphCanvas(
        task="public question", dataset="aime", runtime=MultiAgentRuntime(executor)
    )
    configure(canvas)
    return canvas


def test_recovered_errors_do_not_authorize_repeated_prompt_revision():
    executor = ArtifactExecutor()
    canvas = canvas_for(executor)
    snapshot = canvas.control_snapshot()
    assert snapshot["worker_protocol_status"]["solver"]["status"] == "recovered"
    assert snapshot["worker_protocol_status"]["solver"]["error_code"] is None
    assert "solver" not in snapshot["legal_action_parameters"]["set_prompt"]["targets"]
    assert len(canvas.runtime.artifacts["solver"].protocol_diagnostics) == 2
    for _ in range(3):
        assert canvas.control_snapshot() == snapshot
    assert len(executor.calls) == 1
    denied = canvas.step(
        json.dumps(
            dict(
                action="set_prompt",
                target="solver",
                role="Updated",
                objective="Provide the result",
                scope="Assigned task",
                expected_output="Result",
                revision_basis="protocol_failure",
                evidence_agent_ids=["solver"],
            )
        )
    )
    assert not denied.accepted
    assert len(executor.calls) == 1


def test_unresolved_failure_can_be_revised_but_snapshot_never_retries_it():
    executor = ArtifactExecutor(
        "WORKER_PROTOCOL_FAILURE", [rejected(), rejected(stage="finalization_2")]
    )
    canvas = canvas_for(executor)
    snapshot = canvas.control_snapshot()
    status = snapshot["worker_protocol_status"]["solver"]
    assert status["status"] == "failed" and status["finalization_attempts_used"] == 2
    evidence = canvas._eligible_prompt_revision_evidence("solver")
    assert "protocol_failure" in evidence["public"]
    assert "solver" in snapshot["legal_action_parameters"]["set_prompt"]["targets"]
    canvas.control_snapshot()
    assert len(executor.calls) == 1
    # Consuming an event removes its eligibility; merely rendering feedback never renews it.
    signatures = evidence["internal"]["protocol_failure"]["solver"]
    canvas._prompt_revision_consumed["solver"] = set(signatures)
    assert "protocol_failure" not in canvas._eligible_prompt_revision_evidence("solver")["public"]
    canvas._prompt_revision_consumed["solver"].clear()
    executor.answer, executor.diagnostics = "35", [accepted()]
    fixed = canvas.step(
        json.dumps(
            dict(
                action="set_prompt",
                target="solver",
                role="Revised",
                objective="Provide the result",
                scope="Assigned task",
                expected_output="Result",
                revision_basis="protocol_failure",
                evidence_agent_ids=["solver"],
            )
        )
    )
    assert fixed.accepted, fixed.feedback
    assert len(executor.calls) == 2
    assert "protocol_failure" not in canvas._eligible_prompt_revision_evidence("solver")["public"]


def test_recovered_local_list_still_has_independent_final_submission_rejection():
    canvas = canvas_for(ArtifactExecutor("[12,7,3]"))
    assert canvas.control_snapshot()["worker_protocol_status"]["solver"]["status"] == "recovered"
    assert "protocol_failure" not in canvas._eligible_prompt_revision_evidence("solver")["public"]
    assert canvas.step('{"action":"set_output","target":"solver"}').accepted
    assert "protocol_failure" in canvas._eligible_prompt_revision_evidence("solver")["public"]
    step = canvas.step('{"action":"finish"}')
    assert not step.accepted and step.rejection_code == "output_answer_invalid"


def test_current_snapshot_does_not_publish_diagnostic_secrets():
    canvas = canvas_for(
        ArtifactExecutor(
            "WORKER_PROTOCOL_FAILURE",
            [
                rejected(
                    "PRIVATE_REASON",
                    raw_response="PRIVATE_RAW",
                    parse_error={"excerpt": "PRIVATE_GOLD"},
                )
            ],
        )
    )
    snapshot = canvas.control_snapshot()
    text = json.dumps(snapshot)
    assert "PRIVATE_" not in text
    assert "protocol_failure_unknown" in text
    assert snapshot["worker_protocol_status_version"] == "worker_protocol_status_v1"
