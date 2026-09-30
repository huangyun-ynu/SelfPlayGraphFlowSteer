"""Bound local research and repeated no-work executions on one WebShop rollout."""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field

from .contracts import AgentArtifact
from .unified_contract import is_task_result

POLICY = "bounded_research_v1"
GUIDANCE = """
WebShop scheduling: all local subtask research together may spend at most HALF of
this question's action budget (8 of 16). This account never resets on edits,
new nodes or revisions. Return useful evidence earlier when your local work is done.
At the research boundary, report the inspected candidates and unresolved facts;
the Director must explicitly promote an existing session owner to task_result
for purchase work. Runtime does not promote a node or choose a product for you.
A completed local report cannot be RUN_AGENT again on the same responsibility
and evidence. A full-task node gets at most two consecutive executions without
an environment action on unchanged public state/evidence; then report the failure.
A staged purchase needs Director FINISH, not further report generation.
When result_scope is task_result, completing the ORIGINAL shopping task is your
responsibility even if the role label says Reporter. The local research limit no
longer caps this node: current_action_allowance is the shared remaining budget.
Earlier local reports and their research-budget stop reason are historical, not
proof that this full-task node cannot search, inspect, select or Buy.
"""


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,default=str).encode()).hexdigest()


def public_state(state):
    # Physical action counter/version and journal prose are not new shopping evidence.
    return {k:copy.deepcopy(state[k]) for k in (
        "page_type", "page_text", "product", "selected_options", "commit_pending",
        "purchased", "done", "resource_status", "termination_reason") if k in state}


def input_keys(node, state, packets):
    facts = {}
    for p in packets:
        progress=getattr(p,"webshop_progress",{})
        item = {"evidence":getattr(p,"evidence",[]),
                      "inspections":progress.get("product_inspections",[]),
                      "candidates":progress.get("candidate_ledger",[])}
        if any(item.values()):
            facts[digest(item)] = item
    work = digest({"scope":node.metadata.get("result_scope"),
                   "route":node.metadata.get("runtime_route"), "state":public_state(state),
                   "evidence":[facts[key] for key in sorted(facts)]})
    responsibility = digest({"work":work,"prompt":node.prompt})
    return work,responsibility


@dataclass
class SchedulingState:
    entries: dict = field(default_factory=dict)
    no_work: dict = field(default_factory=dict)
    completion_reminder_sent: bool = False

    @staticmethod
    def research_limit(total):
        return total // 2

    def blocker(self,node,state,packets,*,remaining,research_used):
        if remaining <= 0:
            return "webshop_actions_exhausted"
        work,responsibility=input_keys(node,state,packets)
        key=(node.agent_id,node.metadata.get("incarnation_id"),work)
        if not is_task_result(node):
            if research_used >= self.research_limit(node.total_tool_budget):
                return "webshop_research_budget_handoff"
            if self.entries.get(node.agent_id,{}).get("local_responsibility")==responsibility:
                return "webshop_local_report_complete"
        elif self.no_work.get(key,0)>=2:
            return "webshop_no_work_recovery_exhausted"
        return None

    def record(self,node,state,packets,*,actions_used,artifact):
        work,responsibility=input_keys(node,state,packets)
        key=(node.agent_id,node.metadata.get("incarnation_id"),work)
        self.no_work[key]=0 if actions_used else self.no_work.get(key,0)+1
        self.entries[node.agent_id]={
            "scope":node.metadata.get("result_scope"), "actions_last_execution":actions_used,
            "no_action_executions":self.no_work[key],
            "local_responsibility":responsibility if not is_task_result(node) else None,
            "stop_reason":artifact.webshop_progress.get("stop_reason"),
        }

    def snapshot(self,*,total,remaining,research_used):
        return {"policy":POLICY,"research_limit":self.research_limit(total),
                "research_used":research_used,
                "research_remaining":max(0,self.research_limit(total)-research_used),
                "total_remaining":remaining,"reset_on_node_edit":False,
                "nodes":{k:{f:v for f,v in e.items() if f!='local_responsibility'} for k,e in self.entries.items()},
                "next_step":"Assign the ORIGINAL shopping objective to an existing session owner as task_result, including search/selection/Buy when useful; do not assign a report-only job. RUN_AGENT only for new work. FINISH a ready result. At zero actions, delete unrelated nodes or promote an existing owner for a runtime failure receipt; do not create or rerun shopping sessions."}

    def worker_snapshot(self, node, *, total, remaining, research_used):
        result = self.snapshot(total=total, remaining=remaining, research_used=research_used)
        full = is_task_result(node)
        allowance = remaining if full else min(remaining, result["research_remaining"])
        result.update(current_scope="task_result" if full else "subtask",
            local_research_limit_applies=not full, current_action_allowance=allowance)
        if full:
            result["next_step"] = (
                f"You own the complete shopping task and may spend {allowance} more environment actions. "
                "The local research allowance does not restrict you. Use a useful affordable search, inspection, "
                "selection or Buy to complete the original request. FINISH is the Director's step after Buy.")
        return result


def runtime_status(node,state,reason,*,budget,scheduling,prior=None):
    """A new runtime status, never relabel a past model answer as new reasoning."""
    exhausted=reason=="webshop_actions_exhausted"
    failed=reason=="webshop_no_work_recovery_exhausted" and is_task_result(node)
    text=("Shared WebShop action budget is exhausted; no purchase was prepared."
          if exhausted else "Repeated executions produced no new environment work; no purchase was prepared."
          if failed else "Local research is paused. Preserve this session and explicitly assign task_result for completion.")
    raw=json.dumps({"answer":text,"summary":text,"confidence":1.0,
                    "unresolved_issues":[reason],"evidence":[]})
    artifact=AgentArtifact.from_model_text(text=raw,artifact_id="",agent_id=node.agent_id,
        model="runtime-webshop-scheduler",source_artifact_ids=[prior.artifact_id] if prior else [])
    artifact.environment_result=copy.deepcopy(state)
    artifact.webshop_progress={"trusted":True,
        "state":"typed_policy_failure" if failed else "action_budget_exhausted" if exhausted else "research_paused",
        "stop_reason":reason,"runtime_only":True,"worker_requests":0,
        "action_budget":budget,"scheduling":scheduling,
        "policy_failure":{"code":reason,"runtime_terminal":True,"attribution":"model_policy"} if failed else {},
        "product_inspections":copy.deepcopy(prior.webshop_progress.get("product_inspections",[])) if prior else [],
        "candidate_ledger":copy.deepcopy(prior.webshop_progress.get("candidate_ledger",[])) if prior else []}
    artifact.protocol_diagnostics=[{"stage":"webshop_scheduling","accepted":True,
                                    "no_request_dispatched":True,"reason":reason}]
    return artifact
