from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

from .canvas import CanvasState, GraphCanvas
from .director_timeline import (
    DELTA_CONTEXT_MODE,
    TIMELINE_CONTEXT_MODES,
    director_context_mode,
    timeline_assistant_content,
    timeline_prefix_audit,
)
from .llm import (
    BinaryChoiceUnavailable,
    ChatBackend,
    DirectorContextExhausted,
    MockBackend,
    director_recovery_budget,
)

# Per-action ceiling. The gateway separately admits the exact templated prompt
# plus completion within the service context; a constant alone cannot ensure fit.
DIRECTOR_ACTION_MAX_TOKENS = 1000
DIRECTOR_PROMPT_MAX_TOKENS = 1000
DIRECTOR_INVALID_ACTION = '{"action":"invalid"}'
DIRECTOR_CONTEXT_SCHEMA = "task_once_action_feedback_history_v1"

DIRECTOR_BASE_PROMPT = """You are the Graph Director. You build and revise a task-adaptive Agent
workflow for the given task. You edit the workflow graph; Workers solve the task.

## 1. Goal

Build a workflow whose Agents and relations reflect the task's actual information, reasoning,
execution, and verification dependencies.

The objective is neither to minimize nor to maximize the number of Agents. Single-Agent and
multi-Agent graphs are both valid outcomes. Let the task, Worker evidence, and execution feedback
determine the structure within the available budget. Use one Agent when the task is genuinely
atomic and no separable responsibility is reasonably likely to change the result. Use multiple
Agents when the task contains separable evidence branches, independent uncertainty, execution and
diagnosis, implementation and validation, conflicting findings, or downstream synthesis.

Do not treat either one-Agent or multi-Agent organization as the default. Every configured Agent
must make a distinct, task-relevant contribution. Agent, token, and relation budgets are execution
constraints, not graph-quality objectives. Do not omit a plausibly useful, non-overlapping
contribution solely to reduce graph size or token cost. Do not add redundant Agents merely to
consume budget. Never solve the task yourself.

## 2. Output Protocol

Return exactly one strict JSON object per turn and no prose. Supported actions are:
{"action":"add_agent"}
{"action":"set_prompt","target":"agent_id","role":"short free-text responsibility","objective":"short objective","scope":"short scope","expected_output":"short output contract"}
{"action":"set_prompt","target":"configured_agent_id","role":"revised responsibility","objective":"revised objective","scope":"revised scope","expected_output":"revised output contract","revision_basis":"upstream_artifact_changed|peer_artifact_changed|tool_error|unresolved_issue|protocol_failure|structural_role_change|controller_repair","evidence_agent_ids":["agent_id"]}
{"action":"set_model","target":"agent_id","runtime_route":"route_id_from_canvas"}
{"action":"set_layer","target":"agent_id","layer":0}
{"action":"consider_relation","source":"agent_a","target":"agent_b"}
{"action":"delete_agent","target":"agent_id"}
{"action":"set_output","target":"agent_id"}
{"action":"finish"}

Never emit more than one action or text before or after the JSON object. Canvas versions are bound
by the controller; omit expected_version. Omit agent_id on ADD_AGENT and let Canvas allocate it.
Never invent an Agent ID. Inside JSON strings, write every literal backslash as `\\`, including
LaTeX commands such as `\\angle`, `\\sqrt`, and `\\frac`.

## 3. Authoritative Canvas Control

The current Canvas snapshot is authoritative. Use only actions listed in allowed_actions and only
targets, relations, layers, revision bases, and evidence Agent IDs listed in
legal_action_parameters. If the snapshot conflicts with an earlier assumption, follow the
snapshot.

After ADD_AGENT, configure the newly allocated Agent with SET_PROMPT, then SET_MODEL.
A responsibility without a selected model is not executable. Select only a route ID listed in the
current Canvas. SET_PROMPT edits preserve the selected model; SET_MODEL edits change it explicitly.
For CONSIDER_RELATION, name only the Agent pair. Canvas infers directed versus bidirectional from
their layers and then requests a separate constrained off/on policy choice. If Canvas reports
STRUCTURAL_REPAIR_REQUIRED, repair only the named structural problem
before unrelated edits. Never escape structural debt by adding new Agents. If the snapshot exposes
only FINISH, finish immediately. Do not repeat SET_OUTPUT for the selected output or switch outputs
without new process evidence.

## 4. Decision Order

Use this order on every turn:
1. Complete the pending SET_PROMPT or SET_MODEL configuration shown by Canvas.
2. Repair a specified structural defect.
3. If time or token consolidation is active, stop expanding and select the best valid existing
   output.
4. Evaluate whether the graph contains all materially useful contributions for this task.
5. Add, connect, revise, or delete an Agent only to address a concrete missing contribution,
   dependency, tool error, unresolved issue, or structural defect.
6. When the graph is sufficient, select the Agent that owns the actual final result.
7. When the output is valid and no useful graph edit remains, FINISH.

Repeated wording, repeated output selection, and unchanged mutations are not progress.

## 5. Graph Design Policy

Choose graph structure from the task's real information, reasoning, execution, and verification
dependencies. Do not begin from a presumption that the graph should be either single-Agent or
multi-Agent. Infer the organization from the task and revise it when factual Worker feedback
reveals a missing or redundant contribution.

Use a directed relation when the target requires the source artifact. Use a bidirectional relation
only for genuine same-level peer refinement. Independent branches that must affect the result must
feed an Agent that owns synthesis, implementation, decision, or final response. Every necessary
Agent must be able to contribute to the selected output. Delete redundant or unreachable Agents.

Before selecting an output, inspect the task and current Worker evidence for an unrepresented
dependency: a separable evidence branch; an independent source of uncertainty; a planning,
execution, or diagnosis dependency; an implementation, testing, or review dependency; conflicting
findings requiring comparison; or multiple contributions requiring synthesis. If one is present
and could materially affect the result, represent it with a distinct Agent and connect it to the
eventual output while budget permits. If none is present, a single-Agent graph is valid. This is a
dependency audit, not a fixed role set or topology template.

## 6. Delegation Contract

SET_PROMPT delegates a responsibility, not a solution. Use role for the kind of contribution the
Agent owns; objective for the concrete task-specific question or outcome; scope for the component,
evidence branch, environment responsibility, or implementation boundary; and expected_output for
the artifact or finding required downstream.

You may repeat task entities, conditions, and the assigned subproblem when needed to define a
concrete contribution. Keep every field concise and specific.
Describe what the Agent should determine, not how to solve it. Do not include a candidate answer,
intermediate calculations,
evidence conclusions, a procedural solution, private references, hidden grader information,
runtime routes, model-selection instructions, Worker Action names, or a prescribed tool sequence.
You must never request, select, or configure Worker Actions. The Worker must decide its own
reasoning and Action sequence from the task and visible Action definitions.

Canvas may append a system-managed task output contract, so do not repeat generic formatting,
exactness, evidence, or uncertainty boilerplate in expected_output. Revise an existing Agent only
when Canvas exposes a valid revision_basis and eligible evidence_agent_ids. Free-text reasoning is
not revision evidence.

## 7. Worker Feedback

Canvas feedback may contain bounded Worker summaries, confidence, evidence counts, unresolved
issues, tool errors, execution status, cache reuse, and remaining budget. These are process signals,
not reference answers or final correctness scores. A fluent textual claim is not proof that a tool
succeeded, an external environment changed, an official terminal state was reached, a repository
was modified, tests passed, or the final answer is correct.

worker_protocol_status describes Artifact acceptance, not task correctness or final submission
validity. recovered means a previous output-protocol error has been resolved; that historical
error alone is not a current reason to revise. unknown means insufficient recorded evidence.
local_recovery_exhausted refers only to the Worker's local finalization allowance, not to the
availability of graph edits. The current Canvas snapshot reports current status; earlier history
records past events. These fields describe facts and do not recommend a next action.

Use feedback only to decide whether a concrete graph edit is useful. Do not expand the graph merely
because confidence is imperfect. If no available edit can address the remaining uncertainty,
select the best valid output and finish.

## 8. Runtime Boundaries

You select each Agent model using SET_MODEL and a Worker route ID exposed by Canvas. Runtime
executes exactly that choice and does not substitute another model. Unknown or unavailable routes
produce factual errors. Use SET_MODEL, not responsibility text, to change models. Dataset
environments determine which Actions Workers can see. Canvas controls legality, version binding,
incremental execution, caching and termination. You control delegation, models, topology and output.

## 9. Termination

Complete the graph within at most 20 Director turns. FINISH when all necessary responsibilities are
represented, required dependencies reach the selected output, the selected output owns the actual
task result, no concrete unresolved issue justifies another graph edit, and Canvas reports FINISH
is legal. Do not edit a sufficient graph merely to use the remaining budget. FINISH ends the
trajectory and no edit is allowed afterward.

You may reason internally, but keep that reasoning brief. The externally visible response must
contain only one JSON action.
"""

# Backward-compatible public name retained for callers that import the old constant.
DIRECTOR_SYSTEM_PROMPT = DIRECTOR_BASE_PROMPT

PROBLEM_TYPE_HINTS: dict[str, str] = {
    "general": """## Problem-Type Guidance: General

This is a general task. Determine its real information, reasoning, execution, and verification
dependencies without presuming either a single-Agent or multi-Agent organization. One Agent may
own a genuinely atomic task; represent distinct analysis, evidence, execution, verification, or
synthesis contributions separately when they could materially affect the result.""",
    "math": """## Problem-Type Guidance: Mathematical Reasoning

This is a mathematical reasoning task. The selected output must determine the exact requested
result with reliable reasoning. Determine the organization from the problem's actual reasoning
dependencies rather than presuming one solver or multiple solvers. A short coherent solution may
be atomic. For a long, case-based, geometry-heavy, ambiguous, or error-prone problem, represent a
distinct derivation, case-analysis, or verification contribution when it could affect the result.
Multiple Agents must contribute different reasoning or checking responsibilities rather than
repeat the same solution.""",
    "retrieval_qa": """## Problem-Type Guidance: Retrieval Question Answering

This is an evidence-grounded retrieval question-answering task. The selected output must provide a
concise answer supported by relevant evidence. Determine whether the answer needs one direct fact
or multiple linked facts. When searches, entities, or evidence chains are meaningfully separable,
assign distinct evidence responsibilities and connect them to an Agent that owns final synthesis.
Do not repeat the same search or paraphrase the same candidate answer. Preserve unresolved
ambiguity instead of treating unsupported text as evidence.""",
    "response": """## Problem-Type Guidance: Context-Grounded Professional Response

This is a context-grounded professional response task. The complete public conversation is the
primary task context. The selected output must provide one coherent response that directly
addresses the user's needs, preserves important uncertainty, and avoids unsupported claims. Do not
assume external search or environment interaction is available, and do not create retrieval
responsibilities when the necessary context is already provided. Use one final response owner;
this output-ownership requirement is not a limit on graph size. A focused request may be atomic.
Represent genuinely distinct concerns, specialized perspectives, or useful safety/review
contributions separately when they could materially affect the response, and feed them to the final
response owner.""",
    "environment": """## Problem-Type Guidance: Stateful Environment Interaction

This is a stateful environment-interaction task. Completion requires the runtime environment to
reach the requested official outcome. Identifying an item, recommending an action, proposing an
action sequence, or claiming success in text is insufficient. Use one Agent as the continuous
state-mutating owner for the selected environment trajectory. This is an environment-ownership
constraint, not a limit on graph size. Other Agents may contribute distinct planning, constraint
tracking, evidence inspection, or failure diagnosis, and their findings must feed the environment
owner. Represent such a contribution when it could materially change the action strategy or detect
a failure. Do not create competing owners of the same mutable environment state. Workers
independently choose visible Actions; never prescribe Action names or a fixed sequence.""",
    "code": """## Problem-Type Guidance: Code and Repository Repair

This is a code task. The selected output must own an executable implementation or runtime-visible
code change with validation status. Analysis, localization, review, or a proposed textual change
alone is not a completed implementation. Use one final commit owner for the selected workspace.
This is a commit-ownership constraint, not a limit on graph size. Distinct Agents may investigate
root cause, inspect separate code paths, analyze tests, or review an implementation, and their
findings must feed the commit owner or a downstream output Agent. Represent such contributions when
they could materially affect the implementation or validation result. Avoid competing final commit
owners. Workers independently choose visible code or repository Actions; never prescribe Action
names or a fixed sequence. If no safe implementation can be produced, preserve the
evidence-grounded failure instead of claiming success.""",
}

DIRECTOR_PROMPT_VARIANTS = frozenset({"v2", "v2.1"})


def _replace_prompt_section(source: str, current: str, legacy: str) -> str:
    if source.count(current) != 1:
        raise RuntimeError("Director prompt section drifted; refusing an inexact V2 reconstruction")
    return source.replace(current, legacy, 1)


# The V2 control arm is reconstructed from the system prompt persisted in the
# 2026-08-31 seven-dataset rollout tokens. Only the three sections changed by
# V2.1 are replaced; the action protocol and all Runtime boundaries stay shared.
DIRECTOR_BASE_PROMPT_V2 = _replace_prompt_section(
    DIRECTOR_BASE_PROMPT,
    "You are the Graph Director. You build and revise a task-adaptive Agent",
    "You are the Graph Director. You build and revise a compact Agent",
)
DIRECTOR_BASE_PROMPT_V2 = _replace_prompt_section(
    DIRECTOR_BASE_PROMPT_V2,
    """Build a workflow whose Agents and relations reflect the task's actual information, reasoning,
execution, and verification dependencies.

The objective is neither to minimize nor to maximize the number of Agents. Single-Agent and
multi-Agent graphs are both valid outcomes. Let the task, Worker evidence, and execution feedback
determine the structure within the available budget. Use one Agent when the task is genuinely
atomic and no separable responsibility is reasonably likely to change the result. Use multiple
Agents when the task contains separable evidence branches, independent uncertainty, execution and
diagnosis, implementation and validation, conflicting findings, or downstream synthesis.

Do not treat either one-Agent or multi-Agent organization as the default. Every configured Agent
must make a distinct, task-relevant contribution. Agent, token, and relation budgets are execution
constraints, not graph-quality objectives. Do not omit a plausibly useful, non-overlapping
contribution solely to reduce graph size or token cost. Do not add redundant Agents merely to
consume budget. Never solve the task yourself.""",
    """Build the smallest workflow that is sufficient for the actual task.

A single Agent is valid when one coherent responsibility can reliably complete the task. Use
multiple Agents when distinct evidence, independent reasoning, stateful execution,
implementation, comparison, verification, or synthesis could materially improve the result.
Do not default every task to the same one-Agent graph. Do not add Agents merely to increase graph
size. Every configured Agent must make a distinct, task-relevant contribution. Never solve the
task yourself.""",
)
DIRECTOR_BASE_PROMPT_V2 = _replace_prompt_section(
    DIRECTOR_BASE_PROMPT_V2,
    """Choose graph structure from the task's real information, reasoning, execution, and verification
dependencies. Do not begin from a presumption that the graph should be either single-Agent or
multi-Agent. Infer the organization from the task and revise it when factual Worker feedback
reveals a missing or redundant contribution.

Use a directed relation when the target requires the source artifact. Use a bidirectional relation
only for genuine same-level peer refinement. Independent branches that must affect the result must
feed an Agent that owns synthesis, implementation, decision, or final response. Every necessary
Agent must be able to contribute to the selected output. Delete redundant or unreachable Agents.

Before selecting an output, inspect the task and current Worker evidence for an unrepresented
dependency: a separable evidence branch; an independent source of uncertainty; a planning,
execution, or diagnosis dependency; an implementation, testing, or review dependency; conflicting
findings requiring comparison; or multiple contributions requiring synthesis. If one is present
and could materially affect the result, represent it with a distinct Agent and connect it to the
eventual output while budget permits. If none is present, a single-Agent graph is valid. This is a
dependency audit, not a fixed role set or topology template.""",
    """Choose graph structure from the task's real information and execution dependencies. Use one Agent
when one coherent contribution is sufficient and another responsibility is unlikely to change the
result. Use multiple Agents for genuinely distinct evidence branches, reasoning or case analysis,
execution and diagnosis, implementation and review, conflicting findings, or final synthesis.

Use a directed relation when the target requires the source artifact. Use a bidirectional relation
only for genuine same-level peer refinement. Independent branches that must affect the result must
feed an Agent that owns synthesis, implementation, decision, or final response. Every necessary
Agent must be able to contribute to the selected output. Delete redundant or unreachable Agents.

Before selecting an output, ask internally whether a Worker with a non-overlapping responsibility
could plausibly discover missing evidence, an unmet environment condition, an implementation
defect, or a material error that would change the result. If yes, add or connect that distinct
contribution while budget permits. If no, a single-Agent graph is acceptable. No fixed role set or
topology is required.""",
)

PROBLEM_TYPE_HINTS_V2: dict[str, str] = {
    **PROBLEM_TYPE_HINTS,
    "math": """## Problem-Type Guidance: Mathematical Reasoning

This is a mathematical reasoning task. The selected output must determine the exact requested
result with reliable reasoning. A single Agent is appropriate for a short coherent solution. For a
long, case-based, geometry-heavy, or error-prone problem, consider a distinct derivation,
case-analysis, or verification contribution. Multiple Agents must contribute different reasoning
or checking responsibilities rather than repeat the same solution.""",
    "response": """## Problem-Type Guidance: Context-Grounded Professional Response

This is a context-grounded professional response task. The complete public conversation is the
primary task context. The selected output must provide one coherent response that directly
addresses the user's needs, preserves important uncertainty, and avoids unsupported claims. Do not
assume external search or environment interaction is available, and do not create retrieval
responsibilities when the necessary context is already provided. A single response Agent is
appropriate for a focused request. Use multiple Agents only for genuinely distinct concerns,
specialized perspectives, or a useful safety/review responsibility, and feed them to one final
response owner.""",
    "environment": """## Problem-Type Guidance: Stateful Environment Interaction

This is a stateful environment-interaction task. Completion requires the runtime environment to
reach the requested official outcome. Identifying an item, recommending an action, proposing an
action sequence, or claiming success in text is insufficient. Prefer one Agent to own continuous
interaction with the mutable environment. Add another Agent only for distinct planning,
constraint-checking, or failure-diagnosis that feeds the environment owner. Do not create competing
owners of the same environment state. Workers independently choose visible Actions; never prescribe
Action names or a fixed sequence.""",
    "code": """## Problem-Type Guidance: Code and Repository Repair

This is a code task. The selected output must own an executable implementation or runtime-visible
code change with validation status. Analysis, localization, review, or a proposed textual change
alone is not a completed implementation. Prefer one implementation owner. Additional Agents may
own root-cause investigation, test analysis, or code review, but their findings must feed the
implementation owner or a downstream output Agent. Avoid competing editing owners. Workers
independently choose visible code or repository Actions; never prescribe Action names or a fixed
sequence. If no safe implementation can be produced, preserve the evidence-grounded failure instead
of claiming success.""",
}


def director_prompt_components(variant: str) -> tuple[str, dict[str, str]]:
    normalized = str(variant).strip().casefold()
    if normalized not in DIRECTOR_PROMPT_VARIANTS:
        raise ValueError(
            f"unsupported Director prompt variant {variant!r}; "
            f"use one of {sorted(DIRECTOR_PROMPT_VARIANTS)}"
        )
    if normalized == "v2":
        return DIRECTOR_BASE_PROMPT_V2, PROBLEM_TYPE_HINTS_V2
    return DIRECTOR_BASE_PROMPT, PROBLEM_TYPE_HINTS


_PROBLEM_TYPE_BY_DATASET = {
    "aime": "math",
    "nq": "retrieval_qa",
    "nq_open": "retrieval_qa",
    "natural_questions": "retrieval_qa",
    "hotpotqa": "retrieval_qa",
    "healthbench_professional": "response",
    "webshop": "environment",
    "alfworld": "environment",
    "swe_bench": "code",
    "swe-bench": "code",
}

_PROBLEM_TYPE_BY_ADAPTER = {
    "aime": "math",
    "retrieval_qa": "retrieval_qa",
    "healthbench_professional": "response",
    "webshop": "environment",
    "alfworld": "environment",
    "swe_bench": "code",
}

_PROBLEM_TYPE_BY_TASK_TYPE = {
    "math": "math",
    "mathematical_reasoning": "math",
    "exact_reasoning": "math",
    "retrieval_qa": "retrieval_qa",
    "open_qa": "retrieval_qa",
    "healthcare": "response",
    "response": "response",
    "professional_response": "response",
    "environment": "environment",
    "stateful_environment": "environment",
    "webshop": "environment",
    "alfworld": "environment",
    "code": "code",
    "repository_repair": "code",
    "swe_bench": "code",
}


def infer_director_problem_type(
    *,
    dataset: str = "",
    task_type: str = "",
    action_adapter_id: str = "",
) -> str:
    """Infer a coarse reusable prompt type without exposing a dataset policy."""

    adapter_key = str(action_adapter_id or "").strip().casefold()
    dataset_key = str(dataset or "").strip().casefold()
    task_key = str(task_type or "").strip().casefold()
    return (
        _PROBLEM_TYPE_BY_ADAPTER.get(adapter_key)
        or _PROBLEM_TYPE_BY_DATASET.get(dataset_key)
        or _PROBLEM_TYPE_BY_TASK_TYPE.get(task_key)
        or "general"
    )


@dataclass
class DirectorTurn:
    round_index: int
    model_action: str
    feedback: str
    accepted: bool
    graph_version: int
    prompt_messages: list[dict[str, str]] = field(default_factory=list)
    trainable: bool = True
    rejection_code: str | None = None
    action_diagnostics: dict[str, Any] = field(default_factory=dict)
    turn_kind: str = "graph_action"
    relation_decision: dict[str, Any] = field(default_factory=dict)
    call_id: str = ""
    raw_reasoning_text: str = ""
    raw_action_text: str = ""
    prompt_token_ids: tuple[int, ...] = ()
    completion_token_ids: tuple[int, ...] = ()
    behavior_log_probs: tuple[float, ...] = ()
    action_character_span: tuple[int, int] | None = None
    model_id: str = ""
    route_name: str = ""
    thinking_requested: bool | None = None
    thinking_effective: bool = False
    token_provenance: str = "unavailable"
    trajectory_schema: str = "director_trajectory_v2_raw_policy_calls"

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_index": self.round_index,
            "model_action": self.model_action,
            "feedback": self.feedback,
            "accepted": self.accepted,
            "graph_version": self.graph_version,
            "prompt_messages": list(self.prompt_messages),
            "trainable": self.trainable,
            "rejection_code": self.rejection_code,
            "action_diagnostics": dict(self.action_diagnostics),
            "turn_kind": self.turn_kind,
            "relation_decision": dict(self.relation_decision),
            "call_id": self.call_id,
            "raw_reasoning_text": self.raw_reasoning_text,
            "raw_action_text": self.raw_action_text,
            "prompt_token_ids": list(self.prompt_token_ids),
            "completion_token_ids": list(self.completion_token_ids),
            "behavior_log_probs": list(self.behavior_log_probs),
            "action_character_span": list(self.action_character_span)
            if self.action_character_span
            else None,
            "model_id": self.model_id,
            "route_name": self.route_name,
            "thinking_requested": self.thinking_requested,
            "thinking_effective": self.thinking_effective,
            "token_provenance": self.token_provenance,
            "trajectory_schema": self.trajectory_schema,
        }


@dataclass
class DirectorRun:
    task: str
    finished: bool
    output: str
    graph: dict[str, Any]
    turns: list[DirectorTurn] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "finished": self.finished,
            "output": self.output,
            "graph": self.graph,
            "turns": [turn.to_dict() for turn in self.turns],
        }


class GraphDirector:
    """Multi-turn Canvas driver adapted from FlowSteer's interactive workflow loop."""

    def __init__(
        self,
        *,
        backend: ChatBackend,
        canvas: GraphCanvas,
        role: str = "graph-director",
        solver_skill_context: str = "",
        prompt_variant: str = "v2.1",
        tokenizer: Any | None = None,
        call_namespace: str = "",
    ) -> None:
        self.context_mode = director_context_mode()
        self.context_schema = (
            "append_only_action_feedback_history_v1"
            if self.context_mode in TIMELINE_CONTEXT_MODES
            else DIRECTOR_CONTEXT_SCHEMA
        )
        if self.context_mode == DELTA_CONTEXT_MODE:
            self.context_schema = "delta_timeline_action_feedback_history_v1"
        self.backend = backend
        self.canvas = canvas
        self.role = role
        self.solver_skill_context = solver_skill_context
        self.prompt_variant = str(prompt_variant).strip().casefold()
        self.relation_token_ids = _binary_relation_token_ids(tokenizer)
        self.relation_tokenizer_attestation = _tokenizer_attestation(tokenizer)
        self.tokenizer = tokenizer
        self.call_namespace = str(call_namespace).strip() or str(self.canvas.runtime.seed)
        if self.prompt_variant not in DIRECTOR_PROMPT_VARIANTS:
            raise ValueError(
                f"unsupported Director prompt variant {prompt_variant!r}; "
                f"use one of {sorted(DIRECTOR_PROMPT_VARIANTS)}"
            )

    def _control_snapshot_text(self, snapshot):
        if self.context_mode == DELTA_CONTEXT_MODE:
            return self._snapshot_codec.encode(snapshot)
        return _snapshot_text(snapshot)

    def run(self) -> DirectorRun:
        from .director_snapshot_delta import SnapshotCodec

        self._snapshot_codec = SnapshotCodec()
        self._completed_turns: list[DirectorTurn] = []
        try:
            return self._run()
        except Exception as exc:
            # Preserve completed policy tokens and factual state, never execute
            # an incomplete/unknown response or convert an exception to success.
            exc.partial_state = {
                **(getattr(exc, "partial_state", None) or {}),
                "director_turns": [turn.to_dict() for turn in self._completed_turns],
                "canvas_state": self.canvas.state.value,
                "round_index": self.canvas.round_index,
                "graph": self.canvas.graph.to_dict(),
                "history": [step.to_dict() for step in self.canvas.history],
                "deadline": (
                    self.canvas.rollout_deadline.diagnostics()
                    if self.canvas.rollout_deadline is not None
                    else None
                ),
            }
            raise

    def _run(self) -> DirectorRun:
        feedback = "Graph is empty."
        turns = self._completed_turns
        problem_type = infer_director_problem_type(
            dataset=self.canvas.dataset,
            task_type=self.canvas.task_type,
            action_adapter_id=(
                self.canvas.action_adapter.adapter_id
                if self.canvas.action_adapter is not None
                else ""
            ),
        )
        base_prompt, problem_type_hints = director_prompt_components(self.prompt_variant)
        base_prompt = base_prompt.replace(
            "at most 20 Director turns",
            f"at most {self.canvas.config.max_rounds} Director turns",
        )
        system_prompt = (
            base_prompt.rstrip() + "\n\n" + problem_type_hints[problem_type].strip() + "\n"
        )
        if self.solver_skill_context:
            # The per-turn conversation is deliberately fresh, so persistent
            # Solver-only orchestration context belongs in the system message.
            system_prompt += (
                "\n## Optional Orchestration Knowledge\n" + self.solver_skill_context + "\n"
            )
        if self.canvas.structural_exploration_required:
            system_prompt += (
                "\n## Bounded Structural Exploration\n\n"
                "This rollout belongs to the bounded structural-exploration stratum. "
                "While budget permits, build a connected graph with at least two "
                "Agents that make distinct task-relevant contributions before selecting the "
                "output or finishing. Choose responsibilities, layers, relation type, and "
                "output dynamically from the task; do not use placeholder roles or a fixed "
                "dataset topology. The authoritative snapshot reports whether this condition "
                "is satisfied or waived by token/time consolidation.\n"
            )
        if self.context_mode == DELTA_CONTEXT_MODE:
            system_prompt += "\nCanvas messages use versioned JSON updates. The first full object defines the state. Each subsequent delta contains set/delete operations with key-array paths, applied in order to the preceding state. A set replaces that entire value (including arrays); delete removes the named field. base/seq are message sequence numbers, distinct from canvas_version. Unchanged fields persist. A later full object replaces the entire state. Reconstruct the latest state before choosing an action; old legal actions may no longer be valid.\n"
        system_message = {"role": "system", "content": system_prompt}
        # Keep the policy history without replaying obsolete control snapshots.
        # Every prior sampled action and its factual environment feedback remain
        # in the conversation; the immutable task and current authoritative
        # snapshot are each supplied exactly once per policy call.
        history_turns: list[tuple[str, str]] = []
        chronological_messages: list[dict[str, str]] = [dict(system_message)]
        previous_policy_ids: tuple[int, ...] = ()
        timeline_chain_valid = True

        def prepare_timeline_turn(messages, prompt_ids, completion_ids):
            nonlocal previous_policy_ids, timeline_chain_valid
            if self.context_mode not in TIMELINE_CONTEXT_MODES:
                return {}
            audit = timeline_prefix_audit(previous_policy_ids, prompt_ids, completion_ids)
            assistant = timeline_assistant_content(self.tokenizer, prompt_ids, completion_ids)
            timeline_chain_valid = timeline_chain_valid and audit["timeline_merge_candidate"]
            audit["timeline_merge_candidate"] = timeline_chain_valid
            chronological_messages[:] = [dict(m) for m in messages]
            chronological_messages.append({"role": "assistant", "content": assistant})
            previous_policy_ids = tuple(prompt_ids) + tuple(completion_ids)
            return audit

        def prompt_messages_for(current_user_content: str) -> list[dict[str, str]]:
            if self.context_mode in TIMELINE_CONTEXT_MODES:
                return [
                    *[dict(m) for m in chronological_messages],
                    {"role": "user", "content": current_user_content},
                ]
            messages = [dict(system_message)]
            if history_turns:
                messages.append({"role": "user", "content": f"Task:\n{self.canvas.director_task}"})
                for index, (raw_action_text, prior_feedback) in enumerate(history_turns):
                    messages.append({"role": "assistant", "content": raw_action_text})
                    if index + 1 < len(history_turns):
                        messages.append(
                            {
                                "role": "user",
                                "content": f"Canvas feedback:\n{prior_feedback}",
                            }
                        )
            return [*messages, {"role": "user", "content": current_user_content}]

        def current_user_prefix() -> str:
            return "" if history_turns else f"Task:\n{self.canvas.director_task}\n\n"

        responsibility_failures = 0
        responsibility_issue: dict[str, Any] = {}
        responsibility_rejections = {
            "responsibility_violation",
            "duplicate_responsibility",
        }
        progress_signature = self.canvas.director_progress_signature()
        observed_turns = 0
        stalled_turns = 0
        while True:
            # Count completed policy calls, including accepted semantic no-ops.
            # Feedback, inference activity and automatic recovery are not progress.
            current_signature = self.canvas.director_progress_signature()
            if len(turns) != observed_turns:
                stalled_turns = stalled_turns + 1 if current_signature == progress_signature else 0
                observed_turns = len(turns)
            progress_signature = current_signature
            trusted_success = self.canvas.recover_trusted_alfworld_success()
            if trusted_success:
                feedback = trusted_success[-1].feedback
                if self.canvas.state in {CanvasState.FINISHED, CanvasState.FAILED}:
                    break
                continue
            # Safe lifecycle closure: this changes only Canvas terminal state and
            # never selects/mutates an Agent, relation, prompt, layer, or output.
            forced_finish = self.canvas.recover_finish_only()
            if forced_finish is not None:
                feedback = forced_finish.feedback
                if self.canvas.state in {CanvasState.FINISHED, CanvasState.FAILED}:
                    break
                if forced_finish.accepted:
                    continue
                # A rejected automatic action is factual feedback for the next
                # bounded Director turn, not permission for a zero-turn retry loop.
            impossible_output = self.canvas.fail_if_no_usable_output_artifact()
            if impossible_output is not None:
                feedback = impossible_output.feedback
                break
            if not self.canvas.active:
                self.canvas.terminate_round_limit_without_graph_repair()
                break
            # Worker execution is synchronous with Canvas.step: by this boundary
            # there are no in-flight Workers to wait for. Pending configuration is
            # represented by legal SET_PROMPT/SET_MODEL/relation actions.
            if stalled_turns >= 4:
                feedback = self.canvas.terminate_director_stall(
                    "director_no_progress_exhausted"
                ).feedback
                break
            if not self.canvas.control_snapshot()["allowed_actions"]:
                feedback = self.canvas.terminate_director_stall(
                    "director_no_legal_continuation"
                ).feedback
                break
            if stalled_turns == 3:
                feedback += (
                    "\nBounded recovery: this is the last call without effective progress. "
                    "Choose one legal action from the snapshot. Keep reasoning concise; "
                    "return a complete JSON action. Workers solve the task."
                )
            if self.canvas.rollout_deadline is not None:
                self.canvas.rollout_deadline.check("director_turn_start")
            if self.canvas.state is CanvasState.AWAITING_RELATION_CHOICE:
                pending = self.canvas.pending_relation_decision
                assert pending is not None
                binary_messages = prompt_messages_for(
                    current_user_prefix()
                    + "Authoritative Canvas control snapshot:\n"
                    + self._control_snapshot_text(self.canvas.control_snapshot())
                    + "\n\nCanvas feedback:\n"
                    + feedback
                    + "\n\nChoose off or on for this relation."
                )
                try:
                    choose_binary = self.backend.choose_binary
                    if self.relation_token_ids is None:
                        raise BinaryChoiceUnavailable(
                            "the active Director tokenizer does not encode off/on as one token each"
                        )
                    binary = choose_binary(
                        binary_messages,
                        role=self.role,
                        token_ids=self.relation_token_ids,
                    )
                except DirectorContextExhausted:
                    self.canvas.terminate_context_limit_without_graph_repair()
                    break
                except (AttributeError, BinaryChoiceUnavailable, TypeError, ValueError) as exc:
                    abandoned = self.canvas.abandon_relation_choice(str(exc))
                    feedback = abandoned.feedback
                    continue
                audit = binary.to_policy_audit()
                audit.update(
                    {
                        "behavior_policy_version": "relation_binary_v1",
                        "tokenizer_attestation": dict(self.relation_tokenizer_attestation),
                        "rollout_seed": int(self.canvas.runtime.seed),
                    }
                )
                timeline_audit = prepare_timeline_turn(
                    binary_messages,
                    binary.prompt_token_ids,
                    (binary.token_ids[binary.choice],),
                )
                step = self.canvas.resolve_relation_choice(binary.choice, policy_audit=audit)
                feedback = step.feedback
                history_turns.append((binary.choice, step.feedback))
                turns.append(
                    DirectorTurn(
                        round_index=step.round_index,
                        model_action=binary.choice,
                        feedback=step.feedback,
                        accepted=step.accepted,
                        graph_version=self.canvas.graph.version,
                        prompt_messages=binary_messages,
                        # The sampled off/on token is a genuine policy action even
                        # when Canvas later rejects the requested graph mutation
                        # (for example, a remaining-time admission gate).  As with
                        # rejected JSON actions below, factual rejection is part of
                        # the environment response; it must not erase the action or
                        # make an otherwise exact trajectory ineligible.
                        trainable=True,
                        rejection_code=step.rejection_code,
                        action_diagnostics={
                            "no_progress_streak_before_call": stalled_turns,
                            "bounded_recovery_call": stalled_turns == 3,
                            "generated_action": True,
                            "director_context_schema": self.context_schema,
                            "timeline_prefix_audit": timeline_audit,
                            "binary_policy_audit": audit,
                            "cleaned_action_chars": len(binary.choice),
                            "backend_request_events": binary.metadata.get(
                                "backend_request_events", []
                            ),
                        },
                        turn_kind="relation_choice",
                        relation_decision=dict(step.relation_decision),
                        call_id=f"{self.call_namespace}:{len(turns)}:relation",
                        raw_action_text=binary.choice,
                        prompt_token_ids=tuple(binary.prompt_token_ids),
                        completion_token_ids=(int(binary.token_ids[binary.choice]),),
                        behavior_log_probs=(float(binary.log_probabilities[binary.choice]),),
                        action_character_span=(0, len(binary.choice)),
                        model_id=binary.model,
                        route_name=str(binary.metadata.get("route", "")),
                        thinking_requested=False,
                        thinking_effective=False,
                        token_provenance=(
                            "mock_text"
                            if isinstance(self.backend, MockBackend)
                            else "provider_prompt_and_binary_token_ids_and_logprobs"
                        ),
                    )
                )
                continue
            if responsibility_failures == 1 and self.canvas.pending_agent_id:
                issue_field = str(responsibility_issue.get("field") or "violating field")
                issue_code = str(responsibility_issue.get("code") or "responsibility_violation")
                if issue_code == "duplicate_responsibility":
                    conflicting_agent = str(
                        responsibility_issue.get("details", {}).get(
                            "conflicting_agent_id", "an existing Agent"
                        )
                    )
                    retry_instruction = (
                        f"The responsibility overlaps {conflicting_agent}. Keep the task "
                        "subject, but make this Agent's contribution genuinely distinct: "
                        "narrow it to a non-overlapping implementation scope or assign "
                        "diagnosis, testing, review, or synthesis with a distinct deliverable. "
                        "Do not merely rename the role or paraphrase the same ownership."
                    )
                else:
                    retry_instruction = (
                        "Preserve the original task subject and intended contribution; rewrite "
                        "only that field and keep the other fields semantically unchanged. "
                        "Task entities and goals are allowed, but answers, procedural solution "
                        "steps, and explicit Action control are not."
                    )
                user_content = (
                    "Your previous SET_PROMPT was rejected. Retry only the currently required "
                    f"SET_PROMPT for {self.canvas.pending_agent_id}. Use the four short fields "
                    "role, objective, scope, and expected_output. The rejection was "
                    f"{issue_code} in {issue_field}. {retry_instruction}\n\n"
                    + current_user_prefix()
                    + "Authoritative Canvas control snapshot:\n"
                    + self._control_snapshot_text(self.canvas.control_snapshot())
                    + "\n\nCanvas feedback:\n"
                    + feedback
                )
            else:
                user_content = (
                    current_user_prefix() + "Authoritative Canvas control snapshot:\n"
                    f"{self._control_snapshot_text(self.canvas.control_snapshot())}\n\n"
                    f"Canvas feedback:\n{feedback}\n\n"
                    "Return the next single JSON action."
                )
            prompt_messages = prompt_messages_for(user_content)
            awaiting_prompt = self.canvas.state is CanvasState.AWAITING_PROMPT
            try:
                with director_recovery_budget(stalled_turns == 3):
                    response = self.backend.generate(
                        prompt_messages,
                        role=self.role,
                        max_tokens=(
                            DIRECTOR_PROMPT_MAX_TOKENS
                            if awaiting_prompt
                            else DIRECTOR_ACTION_MAX_TOKENS
                        ),
                        enable_thinking=None,
                    )
            except DirectorContextExhausted:
                self.canvas.terminate_context_limit_without_graph_repair()
                break
            if self.canvas.rollout_deadline is not None:
                self.canvas.rollout_deadline.check("director_turn_response")
            raw_reasoning = str(response.raw_reasoning_text or "")
            raw_action = str(
                response.raw_action_text or ("" if raw_reasoning else response.text) or ""
            )
            raw_policy_text = raw_reasoning + raw_action
            # The complete reasoning+action completion remains the trainable
            # policy trajectory. The typed-action parser consumes only the
            # provider-authored action channel (or the exact suffix after an
            # inline Qwen </think> tag), so JSON examples mentioned in thought
            # cannot be mistaken for extra Canvas actions.
            parsed = self.canvas.parser.parse_policy_output(raw_action)
            # Canvas receives only the uniquely parsed typed action.  The marker is
            # deliberately invalid and cannot mutate the graph; the authoritative
            # raw policy response remains stored separately below.
            action_text = parsed.action_text or DIRECTOR_INVALID_ACTION
            completion_ids = tuple(int(value) for value in response.completion_token_ids)
            mock_policy = bool(response.metadata.get("mock"))
            if not completion_ids and mock_policy and self.tokenizer is not None:
                chat_encoder = getattr(self.tokenizer, "encode_chat_trajectory", None)
                if callable(chat_encoder):
                    policy_ids, _policy_mask, policy_spans = chat_encoder(
                        [
                            *[dict(message) for message in prompt_messages],
                            {"role": "assistant", "content": raw_policy_text},
                        ]
                    )
                    if not policy_spans:
                        raise ValueError(
                            "rollout-time chat tokenizer did not expose the policy completion"
                        )
                    completion_ids = tuple(
                        int(value) for value in policy_ids[slice(*policy_spans[-1])]
                    )
                else:
                    completion_ids = tuple(
                        int(value)
                        for value in self.tokenizer.encode(
                            raw_policy_text, add_special_tokens=False
                        )
                    )
            behavior_log_probs = tuple(float(value) for value in response.behavior_log_probs)
            exact_behavior = bool(
                completion_ids
                and behavior_log_probs
                and len(completion_ids) == len(behavior_log_probs)
            )
            training_eligible = bool(response.training_eligible and (exact_behavior or mock_policy))
            timeline_audit = prepare_timeline_turn(
                prompt_messages,
                response.prompt_token_ids,
                completion_ids,
            )
            diagnostics = {
                "no_progress_streak_before_call": stalled_turns,
                "bounded_recovery_call": stalled_turns == 3,
                "raw_output_chars": len(raw_policy_text),
                "backend_request_events": response.metadata.get("backend_request_events", []),
                "json_objects_found": parsed.candidate_count,
                "initial_token_out": response.token_out,
                "finish_reason": response.metadata.get("finish_reason"),
                "director_dynamic_budget": response.metadata.get("director_dynamic_budget"),
                "repair_attempted": False,
                "generated_action": True,
                "behavior_logprobs_exact": exact_behavior,
                "trajectory_training_eligible": training_eligible,
                "director_context_schema": self.context_schema,
                "timeline_prefix_audit": timeline_audit,
            }
            parsed_model_action = parsed.action
            diagnostics["model_expected_version"] = parsed_model_action.expected_version
            diagnostics["bound_canvas_version"] = self.canvas.graph.version
            # Pass the already parsed typed action (valid or invalid) so Canvas
            # reports the exact parser failure instead of reparsing a synthetic
            # placeholder. The raw policy response remains separately immutable.
            step = self.canvas.step(parsed.action, authoritative_director=True)
            feedback = step.feedback
            history_turns.append((raw_policy_text, step.feedback))
            turns.append(
                DirectorTurn(
                    round_index=step.round_index,
                    model_action=action_text,
                    feedback=step.feedback,
                    accepted=step.accepted,
                    graph_version=self.canvas.graph.version,
                    prompt_messages=prompt_messages,
                    trainable=training_eligible,
                    rejection_code=step.rejection_code,
                    action_diagnostics=diagnostics,
                    relation_decision=dict(step.relation_decision),
                    call_id=f"{self.call_namespace}:{len(turns)}:action",
                    raw_reasoning_text=raw_reasoning,
                    raw_action_text=raw_action,
                    prompt_token_ids=tuple(response.prompt_token_ids),
                    completion_token_ids=completion_ids,
                    behavior_log_probs=behavior_log_probs,
                    action_character_span=parsed.character_span,
                    model_id=response.model,
                    route_name=str(response.metadata.get("route", "")),
                    thinking_requested=response.metadata.get("enable_thinking"),
                    thinking_effective=bool(
                        raw_reasoning or re.search(r"<think>.*?</think>", raw_action, re.DOTALL)
                    ),
                    token_provenance=response.token_provenance,
                )
            )
            if step.rejection_code in responsibility_rejections:
                responsibility_issue = dict(step.responsibility_issue)
                if self.canvas.pending_agent_id:
                    responsibility_failures += 1
                else:
                    # A rejected rewrite of an already configured Agent must not enter
                    # the mandatory-prompt recovery path: there is no pending Agent to
                    # recover, and referring to it as ``None`` terminates valid graphs.
                    responsibility_failures = 0
                    responsibility_issue = {}
            elif step.accepted:
                responsibility_failures = 0
                responsibility_issue = {}
            if step.topology_edits_frozen:
                continue
            # Loop once more even when this model turn exhausted max_rounds:
            # deterministic FINISH-only or bounded round-limit recovery still
            # has to establish the terminal postcondition.
        output = ""
        if self.canvas.graph.output_agent:
            artifact = self.canvas.runtime.artifacts.get(self.canvas.graph.output_agent)
            if artifact is not None:
                output = artifact.answer
        return DirectorRun(
            task=self.canvas.task,
            finished=self.canvas.state.value == "finished",
            output=output,
            graph=self.canvas.graph.to_dict(),
            turns=turns,
        )


def _snapshot_text(snapshot: dict[str, Any]) -> str:
    import json

    return json.dumps(snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _binary_relation_token_ids(tokenizer: Any | None) -> dict[str, int] | None:
    """Attest at startup that each constrained choice is exactly one policy token."""

    if tokenizer is None:
        return None
    resolved: dict[str, int] = {}
    for choice in ("off", "on"):
        token_ids = list(tokenizer.encode(choice, add_special_tokens=False))
        if len(token_ids) != 1:
            return None
        resolved[choice] = int(token_ids[0])
    if resolved["off"] == resolved["on"]:
        return None
    return resolved


def _tokenizer_attestation(tokenizer: Any | None) -> dict[str, Any]:
    if tokenizer is None:
        return {}
    underlying = getattr(tokenizer, "tokenizer", tokenizer)
    name = str(getattr(underlying, "name_or_path", type(underlying).__qualname__))
    chat_template = str(getattr(underlying, "chat_template", "") or "")
    payload = (
        f"{type(underlying).__module__}.{type(underlying).__qualname__}\n{name}\n{chat_template}"
    )
    return {
        "name_or_path": name,
        "implementation": f"{type(underlying).__module__}.{type(underlying).__qualname__}",
        "chat_template_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }
