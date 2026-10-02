"""Public Worker reports for Director decisions, with no scoring or model calls."""

import json

from .qa_result_contract import QA_RESULT_CONTRACT_VERSION


def _excerpt(value, limit):
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return {"text": text[:limit], "truncated": len(text) > limit,
            "original_chars": len(text)}


def worker_result_observations(graph, artifacts, assessments, dirty_agents):
    """Publish only artifact fields; never receive a task reference or evaluate support.

    Full answers and identities live in the structured control snapshot, outside the
    free-text feedback truncator. Other fields have explicit, deterministic excerpts.
    A reported quotation remains a Worker claim rather than trusted source text.
    """
    results = {}
    for agent_id, node in sorted(graph.nodes.items()):
        artifact = artifacts.get(agent_id)
        if artifact is None:
            continue
        assessment = assessments.get(agent_id, {})
        evidence = artifact.evidence
        unresolved = artifact.unresolved_issues
        results[agent_id] = {
            "artifact_id": artifact.artifact_id,
            "input_signature": assessment.get("input_signature"),
            "result_scope": node.metadata.get("result_scope"),
            "pending_reexecution": agent_id in dirty_agents,
            "source": "worker_report",
            "qa_result_contract_version": QA_RESULT_CONTRACT_VERSION,
            "correctness": "not_assessed_by_runtime",
            "answer": artifact.answer,
            "summary": _excerpt(artifact.summary, 1600),
            "evidence": [_excerpt(value, 600) for value in evidence[:4]],
            "evidence_count": len(evidence),
            "evidence_omitted_count": max(0, len(evidence) - 4),
            "unresolved_count": len(unresolved),
            "unresolved_issues": [_excerpt(value, 240) for value in unresolved[:4]],
            "unresolved_omitted_count": max(0, len(unresolved) - 4),
            "claimed_confidence": artifact.claimed_confidence,
        }
    return results
