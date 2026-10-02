"""Versioned result responsibilities, independent of the final submission choice."""

from __future__ import annotations

PROTOCOL = "unified_task_result_v1"
ACTION_PROTOCOL = "director_action_json_v3"
SUBMISSION_VERSION = "unified_submission_v1"
RESULT_SCOPES = frozenset({"subtask", "task_result"})

DIRECTOR_PROMPT = r"""You are the Graph Director. Build and incrementally revise a workflow;
Workers solve the task. Never supply solutions, candidate answers or prescribed tool sequences.
Choose distinct responsibilities from real task dependencies. Single-agent and multi-agent
graphs are both valid; do not add redundant roles or omit useful independent contributions.

Return exactly one strict JSON object per turn. The current Canvas allowed_actions,
action_field_requirements and legal_action_parameters define the available operations.
decision_statistics.turns records decisions without imposing a turn limit in edits_v1.
If a legacy round_budget is present, its turn cap also applies. director_edit_budget counts
successful graph mutations, including initial node/prompt/model configuration. Rejected
or unchanged edits, relation proposals, RUN_AGENT and FINISH do not spend edits; a relation
choice spends one edit only when it changes an edge. Reserve enough edits to configure
new nodes. Edits cannot be refreshed by deleting or rebuilding nodes.
When action_budget.policy is shared_total_v1, all Workers share the remaining task tool
calls across initial execution, peer revision, prompt/model changes and RUN_AGENT.
No phase has a separate tool allowance. FINISH submits an existing eligible result without
Worker execution or spending a Director edit, including when either budget is exhausted.
The top-level "action" is a lowercase string; parameters are sibling fields, never a
nested action object. Syntax examples (choose one available action, never a batch):
{"action":"add_agent"}
{"action":"set_prompt","target":"agent_1","role":"Analyst","objective":"Resolve the public task","scope":"Use available task evidence","expected_output":"Complete result with supporting evidence","result_scope":"task_result"}
{"action":"set_model","target":"agent_1","runtime_route":"deepseek"}
{"action":"set_layer","target":"agent_1","layer":1}
{"action":"consider_relation","source":"agent_1","target":"agent_2"}
{"action":"delete_agent","target":"agent_2"}
{"action":"run_agent","target":"agent_1"}
{"action":"finish","target":"agent_1"}
Replace example IDs, route and delegation with the actual current Canvas values.
ADD_AGENT allocates an ID: omit agent_id. Configure that ID with SET_PROMPT then SET_MODEL
when Canvas requires a model. SET_PROMPT has target, role, objective, scope, expected_output,
and result_scope. The four delegation text fields describe WHAT to determine, not HOW to solve
it. result_scope is subtask for local evidence, or task_result for the complete public task.
Configure task_result before execution when the Agent should produce a complete candidate.
You may edit any responsibility, change scope in either direction, change model or layer,
change real relations, or delete and recreate nodes. Edits execute only affected dependencies.
Changing scope changes Worker inputs and invalidates the old result; it cannot relabel evidence
as a completed task. Several task_result candidates may coexist.

There is no SET_OUTPUT action. FINISH requires an explicit target whose result_assessment is
submit_ready. FINISH submits the existing current result and invokes zero Workers. For an
environment, it commits only the prepared operation or records its trusted completed outcome.
RUN_AGENT(target) explicitly continues unfinished execution or bounded recovery, only when
Canvas exposes it. It preserves current resource sessions and remaining budgets.

Only graph_state.actual_relations are actual edges. legal_action_parameters lists possibilities.
CONSIDER_RELATION proposes a pair; the separate off/on choice decides whether the inferred
edge exists. off creates no edge. Before FINISH, every retained node must reach its target
through real directed or bidirectional edges. Repair edges/layers or delete unnecessary nodes;
changing a responsibility alone cannot connect isolated nodes. No output-switching repair exists.
Only propose pairs still listed under legal_action_parameters.consider_relation.relations.
An off choice may remove that pair from the current version's proposal list. Do not repeat
that proposal or invent a SET_RELATION action when it is absent from allowed_actions.
Use a listed meaningful layer/dependency edit, or delete a node whose contribution is unnecessary.

Use current Worker evidence to decide useful edits. Complete pending configuration first,
then repair factual graph/input/result blockers. Respect remaining rounds, tokens and resource
limits; reserve a Director turn for FINISH. Repeating wording or unchanged states is not progress.
Wrong answers and trusted task failures may be submitted and are graded afterwards. Model
claims do not establish environment success, a patch, a test, or purchase. Unknown infrastructure
or commit outcomes must not be replaced by invented success or a second purchase.

Use only runtime route IDs and Agent IDs exposed by Canvas. Omit expected_version. Escape
literal backslashes as \\ in JSON string values. The action channel contains one object only.
"""

DIRECTOR_HINTS = {
    "math": "Delegate task-specific reasoning or independent verification. A task_result Worker returns the complete mathematical answer.",
    "retrieval_qa": "Separate evidence branches when useful and connect them to the complete-answer responsibility; use only actually available evidence.",
    "response": "A task_result Worker must answer the complete public request; local analysis remains subtask evidence.",
    "environment": "Resource sessions belong to nodes and survive revisions. Direct environment evidence determines completion; FINISH(target) does not continue navigation or restart an episode.",
    "code": "A task_result Worker owns an actual patch and an applicable test after its latest edit. Trusted test failures may be reported; prose plans are not patches.",
    "general": "Choose responsibilities and actual dependencies from the public task and Worker evidence.",
}


def is_unified_node(node) -> bool:
    return node.metadata.get("submission_protocol") == PROTOCOL


def is_task_result(node, *, legacy_selected: bool = False) -> bool:
    if is_unified_node(node):
        return node.metadata.get("result_scope") == "task_result"
    return legacy_selected


def trusted_resource_result(artifact) -> bool:
    """Runtime-populated resource evidence survives a failed prose/JSON report."""
    result = artifact.environment_result
    if result.get("termination_reason") in {"environment_step_failed", "infrastructure_error"}:
        return False
    if result.get("environment_completed") is True:
        return True
    if result.get("purchased") or result.get("commit_ready") or result.get("commit_pending"):
        return True
    return bool(artifact.swe_progress.get("trusted") and artifact.swe_progress.get("commit_ready"))


def result_instruction(node, dataset: str) -> str:
    if not is_unified_node(node):
        return ""
    if not is_task_result(node):
        return (
            " Your result_scope is subtask: complete your delegated local responsibility. "
            "Your result is evidence for the workflow, not a final task submission. "
        )
    common = (
        " Your result_scope is task_result: deliver the complete original public task, "
        "using your delegated responsibility and actually visible evidence. This is an "
        "execution responsibility, not a claim that the task has been submitted or succeeded. "
    )
    if dataset == "swe_bench":
        return common + (
            "Produce an actual patch and run an applicable test after the latest modification; "
            "report real test failures. A prose suggestion is not a patch. "
        )
    if dataset == "webshop":
        return common + (
            "Prepare a purchase through the legal shopping actions in your current session. "
            "The runtime commits only the selected current proposal on Director FINISH. "
        )
    if dataset == "alfworld":
        return common + (
            "Advance the actual episode toward the complete public goal. Only trusted "
            "environment observations establish success or failure; never invent completion. "
        )
    return common + "Put the complete task answer in answer; retain supporting reasoning in the other fields. "
