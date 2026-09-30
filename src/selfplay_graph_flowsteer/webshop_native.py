"""SkillFlow WebShop interaction with graph-owned, revisable purchase candidates.

Reference: SkillFlow 74be52bb6bd9f0e9e68dacb72636b75649197983.
Native text prompts/actions live below the graph scheduler. Agent histories are
private; the only inter-Agent channel remains RelayPacket. No reward is queried
until Canvas FINISH commits the selected Agent's candidate.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from .webshop import WebShopSessionLifecycle
from .webshop_native_protocol import NATIVE_POLICY


@dataclass
class NativeWebShopLifecycle(WebShopSessionLifecycle):
    """Isolated per-Agent sessions; selected candidate commits only on FINISH."""

    execution_policy: str = field(default=NATIVE_POLICY, init=False)
    require_native_actions: bool = True
    _episodes: dict[str, WebShopSessionLifecycle] = field(default_factory=dict)
    _native_active: str | None = None
    _native_committed: str | None = None

    @property
    def owner_agent(self) -> None:
        # No first-executed-Agent privilege. Every generic node is equally capable.
        return None

    def owner_agents(self) -> tuple[str, ...]:
        return tuple(sorted(self._episodes))

    def allows_agent(self, agent_id: str) -> bool:
        return True

    def reserve_full_graph_owner(self, agent_id: str) -> None:
        # Counterfactual replays retain generic per-Agent sessions.
        return None

    def set_committer(self, agent_id: str) -> None:
        self._committer_agent = str(agent_id)

    def begin_execution(self, *, agent_id: str, seed: int, revision: bool) -> dict[str, Any]:
        if self._native_committed is not None:
            raise RuntimeError("cannot execute an Agent after the selected purchase was committed")
        if self._native_active is not None:
            raise RuntimeError("native WebShop lifecycle is already executing")
        child = self._episodes.get(agent_id)
        if child is None:
            child = WebShopSessionLifecycle(
                self.client,
                max_observation_chars=0 if self.require_native_actions else self.max_observation_chars,
                search_observation_mode="legacy" if self.require_native_actions else self.search_observation_mode,
                env_feedback_enabled=self.env_feedback_enabled,
                compatibility_profile=self.compatibility_profile,
                freeze_unknown_mutations=self.freeze_unknown_mutations,
                stage_purchases=True,
                pending_ttl_s=self.pending_ttl_s,
                max_pending_sessions=self.max_pending_sessions,
            )
            if self._task is None:
                raise RuntimeError("native WebShop has no bound task")
            child.bind_task(self._task)
            self._episodes[agent_id] = child
        # A genuine scheduled re-execution invalidates the old proposal, while
        # retaining the exact live page and private history. A cache hit never
        # enters here. No candidate is silently retained after changed inputs.
        pending = child._pending_sessions.pop(agent_id, None)
        if pending is not None:
            child._active_session = pending.session_id
            child._active_agent = agent_id
            child._active_pending_target = None
            child._results[agent_id].update(
                commit_pending=False, commit_ready=False, purchase_executed=False,
                termination_reason="active", environment_completed=False,
            )
        state = child.begin_execution(agent_id=agent_id, seed=seed, revision=revision)
        if self.require_native_actions and not isinstance(state.get("raw_available_actions"), list):
            child.end_execution()
            raise ValueError("native WebShop sidecar capability raw_available_actions is missing")
        self._native_active = agent_id
        return state

    def search(self, query: str) -> dict:
        if self._native_active is None:
            raise RuntimeError("shopping action outside an Agent execution")
        return self._episodes[self._native_active].search(query)

    def click(self, target_id: str, *, purchase_evidence=None, state_version=None) -> dict:
        if self._native_active is None:
            raise RuntimeError("shopping action outside an Agent execution")
        return self._episodes[self._native_active].click(
            target_id, purchase_evidence=purchase_evidence, state_version=state_version
        )

    def preflight_click(self, target_id: object, *, state_version: object = None) -> dict | None:
        if self._native_active is None:
            raise RuntimeError("shopping action outside an Agent execution")
        return self._episodes[self._native_active].preflight_click(
            target_id, state_version=state_version
        )

    def end_execution(self) -> None:
        if self._native_active is not None:
            self._episodes[self._native_active].end_execution()
            self._native_active = None

    def runtime_transaction_journal_for(self, agent_id: str) -> dict | None:
        child = self._episodes.get(agent_id)
        return child.runtime_transaction_journal_for(agent_id) if child else None

    def step_native(self, action: str) -> dict:
        if self._native_active is None:
            raise RuntimeError("native action outside an Agent execution")
        child = self._episodes[self._native_active]
        state = child.result_for(self._native_active)
        if action.startswith("search[") and action.endswith("]"):
            if "search" not in state["raw_available_actions"]:
                raise ValueError("search is not available on this page")
            return child.search(action[7:-1])
        if action not in state["raw_available_actions"]:
            raise ValueError("click is not in the current admissible action list")
        targets = [a for a in state.get("valid_subactions", []) if a.get("raw_action") == action]
        if len(targets) != 1:
            raise ValueError("raw action does not resolve to exactly one current target")
        target = targets[0]
        if target.get("kind") == "purchase":
            child._active_pending_target = target["target_id"]
            staged = copy.deepcopy(state)
            staged.pop("action_effect", None)
            staged.update(
                commit_pending=True,
                commit_ready=True,
                purchase_executed=False,
                termination_reason="purchase_staged",
            )
            child._results[self._native_active] = staged
            return copy.deepcopy(staged)
        return child.click(target["target_id"], state_version=state.get("state_version"))

    def commit_ready_agents(self) -> tuple[str, ...]:
        return tuple(
            sorted(a for a, child in self._episodes.items() if a in child.commit_ready_agents())
        )

    def commit_pending(self, agent_id: str) -> dict:
        if self._native_committed is not None:
            if self._native_committed != agent_id:
                raise RuntimeError("another Agent's candidate was already committed")
            return self.result_for(agent_id)
        child = self._episodes.get(agent_id)
        if child is None:
            raise ValueError("selected Agent has no shopping session")
        result = child.commit_pending(agent_id)
        if result.get("purchased"):
            self._native_committed = agent_id
        return result

    def discard_pending(self, agent_id: str) -> None:
        child = self._episodes.pop(agent_id, None)
        if child:
            child.close_all()

    def session_binding(self, agent_id: str) -> str | None:
        """Read-only real resource identity; never copies another node's session."""
        child = self._episodes.get(agent_id)
        if child is None:
            return None
        pending = child._pending_sessions.get(agent_id)
        return child._active_session or (pending.session_id if pending else None)

    def result_for(self, agent_id: str | None) -> dict:
        child = self._episodes.get(str(agent_id))
        if child:
            return child.result_for(agent_id)
        return {"purchased": False, "reward": 0.0, "termination_reason": "agent_never_executed"}

    def cache_signature(self, agent_id: str) -> str:
        child = self._episodes.get(agent_id)
        payload = {
            "policy": NATIVE_POLICY,
            "agent_id": agent_id,
            "state": child.result_for(agent_id) if child else None,
            "history": self.runtime_transaction_journal_for(agent_id),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()

    def close_all(self) -> None:
        first_error: Exception | None = None
        for child in self._episodes.values():
            try:
                child.close_all()
            except Exception as exc:
                first_error = first_error or exc
        self._episodes.clear()
        self._native_active = None
        self._native_committed = None
        try:
            super().close_all()
        except Exception as exc:
            first_error = first_error or exc
        if first_error is not None:
            raise first_error
