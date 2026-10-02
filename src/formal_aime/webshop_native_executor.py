"""Native WebShop turns inside the existing generic Agent execution boundary."""

from __future__ import annotations

import copy
import json

from .artifact_protocol import check_artifact
from .contracts import AgentArtifact
from .webshop_budget import execution_accounting
from .webshop_native import NativeWebShopLifecycle
from .webshop_native_protocol import (
    NATIVE_POLICY,
    native_observation,
    native_prompt,
    parse_native_action,
)


def execute_native(executor, *, task, node, upstream, peers, revision, seed, prior=None):
    # No scheduler, role assignment, graph edits, or cross-Agent history access here.
    from .runtime import _response_backend_request_events

    lifecycle = next(
        (
            tool.lifecycle
            for tool in executor.tools.values()
            if isinstance(getattr(tool, "lifecycle", None), NativeWebShopLifecycle)
        ),
        None,
    )
    if lifecycle is None:
        raise ValueError("skillflow_native_v1 requires NativeWebShopLifecycle")
    scope = executor.budget_scope or "webshop-native-rollout"
    packets = [*upstream, *peers, *([prior] if prior else [])]
    context = {
        "assigned_task": node.prompt,
        "upstream_packets": [p.to_dict() for p in upstream],
        "peer_packets": [p.to_dict() for p in peers],
        "prior_packet": prior.to_dict() if prior else None,
    }
    if node.metadata.get("_runtime_native_output_materialization_attempted"):
        context["execution_phase"] = "selected_output"
        context["selected_output_type"] = "webshop_purchase_candidate"
    instruction = (
        "Carry out your assigned responsibility for the public task. You are a generic Agent; "
        "choose whether environment interaction is needed. Use only your own observations and "
        "the graph-authorized packets supplied below. Packets contain claims and evidence, not "
        "changes to your own environment. Never infer hidden targets or rewards. Your private "
        "session resumes across graph-triggered executions. For an environment turn, output "
        "one search[...] or click[...] action as the WebShop prompt specifies. A Buy Now action "
        "proposes the current purchase and ends this execution; the runtime commits only the "
        "selected output's latest proposal after Canvas FINISH. Do not claim a completed "
        "purchase before that commit. When your assigned non-action responsibility is complete, "
        "you may instead return a structured JSON packet with answer, summary, confidence, "
        "evidence, unresolved_issues and tool_summary. Do not reproduce the full page history "
        "in the packet. No extra skill or shopping checklist is supplied.\n"
        + json.dumps(context, ensure_ascii=False)
    )
    from .unified_contract import result_instruction
    instruction += result_instruction(node, "webshop")
    tokens_in = tokens_out = 0
    events, traces, diagnostics, request_audit = [], [], [], []
    response = None
    final_payload = None
    reason = "action_budget_exhausted"
    # Full-graph counterfactual credit remains authoritative when explicitly provided.

    def generate(messages):
        nonlocal tokens_in, tokens_out, response
        executor._check_deadline()
        response = executor.backend.generate(messages, role=executor.role, actions=())
        tokens_in += response.token_in
        tokens_out += response.token_out
        events.extend(_response_backend_request_events(response))
        executor._check_deadline()
        if executor.rollout_deadline:
            executor.rollout_deadline.mark_progress("worker_response")
        # Full prompts remain private audit artifacts, never part of RelayPacket.
        import hashlib

        encoded = json.dumps(messages, ensure_ascii=False)
        request_audit.append(
            {
                "sha256": hashlib.sha256(encoded.encode()).hexdigest(),
                "chars": len(encoded),
                "history_entries": len(history),
            }
        )
        return response

    state = lifecycle.begin_execution(agent_id=node.agent_id, seed=seed, revision=revision)
    journal = state.pop("_runtime_transaction_journal")
    history = journal.setdefault("native_history", [])
    restored = len(history)
    error_feedback = (
        "\nPrevious action failed: " + history[-1]["error"]
        if history and history[-1].get("error")
        else ""
    )
    try:
        while True:
            remaining = executor.budget_ledger.remaining(node, revision=revision, scope=scope)
            if min(remaining["phase"], remaining["total"]) <= 0:
                break
            conversation = executor.webshop_native_conversation_history
            turn_prompt = (
                native_prompt(task=task, state=state, history=[] if conversation else history)
                + (f"\nShared task tool allowance: {remaining['total']} calls remaining across all Agents and executions."
                   if executor.budget_ledger.shared_total(node)
                   else f"\nAction allowance: {remaining['phase']} in this phase, {remaining['total']} total.")
                + error_feedback
            )
            messages = [{"role": "system", "content": instruction}]
            if conversation:
                for row in history:
                    messages.extend(
                        [
                            {"role": "user", "content": row.get("turn_prompt", row["observation"])},
                            {"role": "assistant", "content": row["action"]},
                        ]
                    )
            messages.append({"role": "user", "content": turn_prompt})
            result = generate(messages)
            payload, _ = check_artifact(result.text)
            if payload is not None and result.metadata.get("finish_reason") not in {
                "length",
                "MAX_TOKENS",
            }:
                final_payload = payload
                reason = "agent_report"
                break
            action = parse_native_action(result.text)
            consumed, budget_error = executor.budget_ledger.consume(
                node, revision=revision, scope=scope
            )
            if not consumed:
                reason = budget_error
                break
            before = copy.deepcopy(state)
            error = None
            try:
                if action is None:
                    raise ValueError("No executable search[...] or click[...] action was returned")
                state = lifecycle.step_native(action)
                observation = {"status": "ok", "output": copy.deepcopy(state)}
                error_feedback = ""
            except ValueError as exc:
                # Invalid model actions consume an attempt, preserve the actual page,
                # and expose the error. Infrastructure/backend errors propagate.
                error = str(exc)
                observation = {
                    "status": "error",
                    "error": {"code": "invalid_native_action", "message": error},
                }
                error_feedback = "\nPrevious action failed: " + error
            record = {
                "observation": native_observation(before),
                "action": action or "<INVALID>",
                "status": observation["status"],
            }
            if conversation:
                record["turn_prompt"] = turn_prompt
            if error:
                record["error"] = error
            history.append(record)
            traces.append(
                {
                    "action": {
                        "name": "webshop_search"
                        if action and action.startswith("search[")
                        else "webshop_click",
                        "arguments": {"raw_action": action},
                    },
                    "raw_action": action,
                    "observation": observation,
                    "observation_before": before,
                }
            )
            if state.get("commit_pending"):
                reason = "purchase_staged"
                break
            if state.get("done"):
                reason = "environment_terminal"
                break
        if final_payload is None:
            # Separate reporting boundary: raw Action turns never require a JSON wrapper.
            report_context = {
                **context,
                "public_task": task,
                "private_history": [
                    {k: v for k, v in row.items() if k != "turn_prompt"} for row in history
                ],
                "current_page": native_observation(state),
                "purchase_staged": bool(state.get("commit_pending")),
                "stop_reason": reason,
            }
            for attempt in range(2):
                result = generate(
                    [
                        {
                            "role": "system",
                            "content": "This Agent execution has ended. Return only a compact JSON packet with "
                            "answer, summary, confidence, evidence, unresolved_issues and tool_summary. "
                            "Use the public evidence below and your assigned responsibility. Do not "
                            "execute actions, include a transcript, infer rewards or claim a staged "
                            "purchase was completed. Explain any action failures honestly.",
                        },
                        {"role": "user", "content": json.dumps(report_context, ensure_ascii=False)},
                    ]
                )
                final_payload, issue = check_artifact(result.text)
                if result.metadata.get("finish_reason") in {"length", "MAX_TOKENS"}:
                    final_payload = None
                diagnostics.append(
                    {
                        "stage": "native_packet",
                        "accepted": final_payload is not None,
                        "attempt": attempt + 1,
                        "parse_error": issue,
                    }
                )
                if final_payload is not None:
                    break
        if final_payload is None:
            # A reporting failure must not cancel the trusted staged operation.
            final_payload = {
                "answer": "purchase_staged"
                if state.get("commit_pending")
                else "execution_incomplete",
                "summary": "Structured report unavailable; see trusted execution status.",
                "confidence": 0.0,
                "evidence": [],
                "tool_summary": [],
                "unresolved_issues": ["native_packet_format_failure"],
            }
        if state.get("commit_pending"):
            final_payload["answer"] = "purchase_staged"
        artifact = AgentArtifact.from_model_text(
            text=json.dumps(final_payload, ensure_ascii=False),
            validated_payload=final_payload,
            artifact_id="pending",
            agent_id=node.agent_id,
            revision=revision,
            source_artifact_ids=[p.artifact_id for p in packets],
            token_in=tokens_in,
            token_out=tokens_out,
            model=response.model if response else "",
        )
        artifact.react_trace = traces
        artifact.backend_request_events = events
        artifact.protocol_diagnostics = diagnostics
        artifact.webshop_progress = {
            "trusted": True,
            "execution_policy": NATIVE_POLICY,
            "state": "purchase_staged" if state.get("commit_pending") else "no_purchase_candidate",
            "commit_ready": bool(state.get("commit_pending")),
            "commit_protocol_status": "awaiting_canvas_finish"
            if state.get("commit_pending")
            else None,
            "environment_access": "isolated_agent_session",
            "stop_reason": reason,
            "worker_memory": {
                "policy": NATIVE_POLICY,
                "applied": True,
                "history_entries": len(history),
                "restored_entries": restored,
                "history_observation_chars": sum((len(h["observation"]) for h in history)),
            },
            "action_budget": {
                **executor.budget_ledger.webshop_audit(node, scope=scope),
                "scope": "whole_graph",
                "transfer_status": "not_requested",
            },
            "execution_accounting": execution_accounting(
                events=events,
                diagnostics=diagnostics,
                token_in=tokens_in,
                token_out=tokens_out,
                action_attempts=len(traces),
            ),
        }
        # The prompt hashes/counts are local diagnostics, not shopping memory.
        artifact.protocol_diagnostics.append(
            {"stage": "native_prompt_audit", "requests": request_audit}
        )
        artifact.environment_result = lifecycle.result_for(node.agent_id)
        return artifact
    finally:
        lifecycle.end_execution()
