"""Historical MACE state reader/reproducer. Never assembled by production entrypoints."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

FEATURE_DIM = 9
MODEL_FEATURE_DIM = 12
_MACE_SAVE_LOCK = threading.Lock()


@dataclass
class PeerBanditState:
    inverse_design: list[list[float]]
    reward_vector: list[float]
    selections: int = 0
    reward_sum: float = 0.0

    @classmethod
    def initialized(cls, regularization: float) -> PeerBanditState:
        scale = 1.0 / regularization
        return cls(
            inverse_design=[
                [scale if row == column else 0.0 for column in range(FEATURE_DIM)]
                for row in range(FEATURE_DIM)
            ],
            reward_vector=[0.0] * FEATURE_DIM,
        )


@dataclass(frozen=True)
class PeerSelectionRecord:
    agent_id: str
    peer_id: str
    allowed_peers: tuple[str, ...]
    features: tuple[float, ...]
    expected_reward: float
    exploration_bonus: float
    score: float


class MACEPeerSelector:
    """Legacy peer-state reader/math utility; not available in the active runtime.

    The official MACE repository currently contains no implementation. This
    module follows equations 3--5 and 10--20 from the paper appendix.
    """

    def __init__(
        self,
        *,
        alpha: float = 1.0,
        regularization: float = 1.0,
        seed: int = 0,
    ) -> None:
        if regularization <= 0:
            raise ValueError("regularization must be positive")
        self.alpha = float(alpha)
        self.regularization = float(regularization)
        self.random = random.Random(seed)
        self.states: dict[tuple[str, str], PeerBanditState] = {}
        self.records: list[PeerSelectionRecord] = []
        self._pending: dict[tuple[str, str], list[tuple[tuple[float, ...], tuple[str, str]]]] = {}
        self._update_events: list[tuple[str, str, tuple[float, ...], float]] = []
        self.applied_event_ids: set[str] = set()

    def select_peer(
        self,
        *,
        agent_id: str,
        candidates: list[str],
        responses: dict[str, str],
        round_index: int,
        max_rounds: int,
        identities: dict[str, str] | None = None,
    ) -> str:
        allowed = sorted(set(candidates) - {agent_id})
        if not allowed:
            raise ValueError(f"no allowed peers for {agent_id}")
        identities = identities or {}
        agent_identity = identities.get(agent_id, agent_id)
        scored: list[tuple[float, float, float, str, tuple[float, ...]]] = []
        for peer_id in allowed:
            features = relational_features(
                agent_id=agent_id,
                peer_id=peer_id,
                responses=responses,
                historical_performance=self._historical_performance(
                    agent_identity, identities.get(peer_id, peer_id)
                ),
                round_index=round_index,
                max_rounds=max_rounds,
            )
            state_key = (agent_identity, identities.get(peer_id, peer_id))
            state = self._state(*state_key)
            theta = _matrix_vector(state.inverse_design, state.reward_vector)
            expected = _dot(theta, features)
            projected = _matrix_vector(state.inverse_design, features)
            bonus = self.alpha * math.sqrt(max(0.0, _dot(features, projected)))
            scored.append((expected + bonus, expected, bonus, peer_id, features))
        best_score = max(item[0] for item in scored)
        ties = [item for item in scored if math.isclose(item[0], best_score, abs_tol=1e-12)]
        chosen = self.random.choice(ties)
        score, expected, bonus, peer_id, features = chosen
        self._pending.setdefault((agent_id, peer_id), []).append(
            (features, (agent_identity, identities.get(peer_id, peer_id)))
        )
        self.records.append(
            PeerSelectionRecord(
                agent_id=agent_id,
                peer_id=peer_id,
                allowed_peers=tuple(allowed),
                features=features,
                expected_reward=expected,
                exploration_bonus=bonus,
                score=score,
            )
        )
        return peer_id

    def update(
        self,
        *,
        agent_id: str,
        peer_id: str,
        score_before: float,
        score_after: float,
    ) -> float:
        key = (agent_id, peer_id)
        queue = self._pending.get(key, [])
        if not queue:
            raise KeyError(f"no pending MACE selection for {agent_id} -> {peer_id}")
        pending = queue.pop(0)
        if not queue:
            self._pending.pop(key, None)
        features, state_key = pending
        reward = 0.5 * ((float(score_after) - float(score_before)) + float(score_after))
        state = self._state(*state_key)
        _apply_linucb_update(state, features, reward)
        self._update_events.append((*state_key, features, reward))
        return reward

    def discard_pending(self) -> None:
        self._pending.clear()

    def export_update_events(self) -> list[dict[str, object]]:
        return [
            {
                "schema_version": "mace_peer_update_v1",
                "selector_kind": "peer",
                "agent_identity": agent_id,
                "arm": peer_id,
                "features": list(features),
                "reward": reward,
                "alpha": self.alpha,
                "regularization": self.regularization,
                "event_id": hashlib.sha256(
                    json.dumps(
                        [agent_id, peer_id, list(features), reward], separators=(",", ":")
                    ).encode()
                ).hexdigest(),
            }
            for agent_id, peer_id, features, reward in self._update_events
        ]

    @classmethod
    def commit_update_events(
        cls, path: str | Path, events: list[dict[str, object]], *, seed: int = 0
    ) -> str:
        destination = Path(path)
        if destination.exists():
            target = cls.load(destination, seed=seed)
        elif events:
            target = cls(
                alpha=float(events[0]["alpha"]),
                regularization=float(events[0]["regularization"]),
                seed=seed,
            )
        else:
            raise ValueError("cannot initialize MACE peer state from an empty event batch")
        for event in sorted(events, key=_mace_event_sort_key):
            event_id = str(event["event_id"])
            if event_id in target.applied_event_ids:
                continue
            _apply_linucb_update(
                target._state(str(event["agent_identity"]), str(event["arm"])),
                tuple(float(value) for value in event["features"]),
                float(event["reward"]),
            )
            target.applied_event_ids.add(event_id)
        destination.parent.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(destination, target._payload())
        return _payload_hash(target._payload())

    def state_hash(self) -> str:
        return _payload_hash(self._payload())

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with _MACE_SAVE_LOCK:
            target = self
            if destination.exists():
                if not self._update_events:
                    return
                target = self.load(destination)
                for agent_id, peer_id, features, reward in self._update_events:
                    _apply_linucb_update(target._state(agent_id, peer_id), features, reward)
                target.random.setstate(self.random.getstate())
            _write_json_atomic(destination, target._payload())
            self.states = target.states
            self.applied_event_ids = target.applied_event_ids
            self._update_events.clear()

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": 2,
            "alpha": self.alpha,
            "regularization": self.regularization,
            "rng_state_version": 3,
            "rng_state": self.random.getstate(),
            "applied_event_ids": sorted(self.applied_event_ids),
            "states": {
                f"{agent_id}\u0000{peer_id}": asdict(state)
                for (agent_id, peer_id), state in self.states.items()
            },
        }

    @classmethod
    def load(cls, path: str | Path, *, seed: int = 0) -> MACEPeerSelector:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        selector = cls(
            alpha=float(payload["alpha"]),
            regularization=float(payload["regularization"]),
            seed=seed,
        )
        if int(payload.get("schema_version", 0)) != 2:
            return selector
        if payload.get("rng_state") is not None:
            selector.random.setstate(_tuple_tree(payload["rng_state"]))
        selector.applied_event_ids = {str(value) for value in payload.get("applied_event_ids", ())}
        for key, raw in payload.get("states", {}).items():
            agent_id, peer_id = key.split("\u0000", 1)
            selector.states[(agent_id, peer_id)] = PeerBanditState(**raw)
        return selector

    def _state(self, agent_id: str, peer_id: str) -> PeerBanditState:
        return self.states.setdefault(
            (agent_id, peer_id), PeerBanditState.initialized(self.regularization)
        )

    def _historical_performance(self, agent_id: str, peer_id: str) -> float:
        state = self._state(agent_id, peer_id)
        return state.reward_sum / state.selections if state.selections else 0.0


@dataclass
class ModelBanditState:
    inverse_design: list[list[float]]
    reward_vector: list[float]
    selections: int = 0
    reward_sum: float = 0.0

    @classmethod
    def initialized(cls, regularization: float) -> ModelBanditState:
        scale = 1.0 / regularization
        return cls(
            inverse_design=[
                [scale if row == column else 0.0 for column in range(MODEL_FEATURE_DIM)]
                for row in range(MODEL_FEATURE_DIM)
            ],
            reward_vector=[0.0] * MODEL_FEATURE_DIM,
        )


@dataclass(frozen=True)
class ModelSelectionRecord:
    agent_id: str
    model_route: str
    candidates: tuple[str, ...]
    features: tuple[float, ...]
    expected_reward: float
    exploration_bonus: float
    score: float


class MACEModelRouter:
    """MACE-style LinUCB extension that treats runtime models as contextual arms."""

    def __init__(
        self,
        *,
        alpha: float = 1.0,
        regularization: float = 1.0,
        seed: int = 0,
    ) -> None:
        if regularization <= 0:
            raise ValueError("regularization must be positive")
        self.alpha = float(alpha)
        self.regularization = float(regularization)
        self.random = random.Random(seed)
        self.states: dict[str, ModelBanditState] = {}
        self.records: list[ModelSelectionRecord] = []
        self._pending: dict[str, tuple[str, tuple[float, ...]]] = {}
        self._update_events: list[tuple[str, str, tuple[float, ...], float]] = []
        self.applied_event_ids: set[str] = set()

    def select_model(
        self,
        *,
        task: str,
        task_type: str,
        agent_id: str,
        agent_prompt: str,
        candidates: tuple[str, ...],
    ) -> str:
        allowed = tuple(
            dict.fromkeys(str(value).strip() for value in candidates if str(value).strip())
        )
        if not allowed:
            raise ValueError("no runtime model candidates")
        features = model_context_features(task=task, task_type=task_type, agent_prompt=agent_prompt)
        scored: list[tuple[float, float, float, str]] = []
        for route in allowed:
            state = self._state(route)
            theta = _matrix_vector(state.inverse_design, state.reward_vector)
            expected = _dot(theta, features)
            projected = _matrix_vector(state.inverse_design, features)
            bonus = self.alpha * math.sqrt(max(0.0, _dot(features, projected)))
            scored.append((expected + bonus, expected, bonus, route))
        best_score = max(item[0] for item in scored)
        ties = [item for item in scored if math.isclose(item[0], best_score, abs_tol=1e-12)]
        score, expected, bonus, route = self.random.choice(ties)
        self._pending[agent_id] = (route, features)
        self.records.append(
            ModelSelectionRecord(
                agent_id=agent_id,
                model_route=route,
                candidates=allowed,
                features=features,
                expected_reward=expected,
                exploration_bonus=bonus,
                score=score,
            )
        )
        return route

    def update(self, *, agent_id: str, reward: float) -> float:
        pending = self._pending.pop(agent_id, None)
        if pending is None:
            raise KeyError(f"no pending MACE model selection for {agent_id}")
        route, features = pending
        bounded_reward = max(0.0, min(1.0, float(reward)))
        state = self._state(route)
        _apply_linucb_update(state, features, bounded_reward)
        self._update_events.append((agent_id, route, features, bounded_reward))
        return bounded_reward

    def discard_pending(self) -> None:
        self._pending.clear()

    def export_update_events(self) -> list[dict[str, object]]:
        return [
            {
                "schema_version": "mace_model_update_v1",
                "selector_kind": "model",
                "agent_id": agent_id,
                "arm": route,
                "features": list(features),
                "reward": reward,
                "alpha": self.alpha,
                "regularization": self.regularization,
                "event_id": hashlib.sha256(
                    json.dumps(
                        [agent_id, route, list(features), reward], separators=(",", ":")
                    ).encode()
                ).hexdigest(),
            }
            for agent_id, route, features, reward in self._update_events
        ]

    @classmethod
    def commit_update_events(
        cls, path: str | Path, events: list[dict[str, object]], *, seed: int = 0
    ) -> str:
        destination = Path(path)
        if destination.exists():
            target = cls.load(destination, seed=seed)
        elif events:
            target = cls(
                alpha=float(events[0]["alpha"]),
                regularization=float(events[0]["regularization"]),
                seed=seed,
            )
        else:
            raise ValueError("cannot initialize MACE model state from an empty event batch")
        for event in sorted(events, key=_mace_event_sort_key):
            event_id = str(event["event_id"])
            if event_id in target.applied_event_ids:
                continue
            _apply_linucb_update(
                target._state(str(event["arm"])),
                tuple(float(value) for value in event["features"]),
                float(event["reward"]),
            )
            target.applied_event_ids.add(event_id)
        destination.parent.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(destination, target._payload())
        return _payload_hash(target._payload())

    def state_hash(self) -> str:
        return _payload_hash(self._payload())

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with _MACE_SAVE_LOCK:
            target = self
            if destination.exists():
                if not self._update_events:
                    return
                target = self.load(destination)
                for _agent_id, route, features, reward in self._update_events:
                    _apply_linucb_update(target._state(route), features, reward)
                target.random.setstate(self.random.getstate())
            _write_json_atomic(destination, target._payload())
            self.states = target.states
            self.applied_event_ids = target.applied_event_ids
            self._update_events.clear()

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "alpha": self.alpha,
            "regularization": self.regularization,
            "rng_state_version": 3,
            "rng_state": self.random.getstate(),
            "applied_event_ids": sorted(self.applied_event_ids),
            "states": {route: asdict(state) for route, state in self.states.items()},
        }

    @classmethod
    def load(cls, path: str | Path, *, seed: int = 0) -> MACEModelRouter:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        router = cls(
            alpha=float(payload["alpha"]),
            regularization=float(payload["regularization"]),
            seed=seed,
        )
        if int(payload.get("schema_version", 0)) != 1:
            return router
        if payload.get("rng_state") is not None:
            router.random.setstate(_tuple_tree(payload["rng_state"]))
        router.applied_event_ids = {str(value) for value in payload.get("applied_event_ids", ())}
        for route, raw in payload.get("states", {}).items():
            router.states[str(route)] = ModelBanditState(**raw)
        return router

    def _state(self, route: str) -> ModelBanditState:
        return self.states.setdefault(route, ModelBanditState.initialized(self.regularization))


def _apply_linucb_update(
    state: PeerBanditState | ModelBanditState,
    features: tuple[float, ...],
    reward: float,
) -> None:
    projected = _matrix_vector(state.inverse_design, features)
    denominator = 1.0 + _dot(features, projected)
    for row in range(len(features)):
        for column in range(len(features)):
            state.inverse_design[row][column] -= projected[row] * projected[column] / denominator
    for index, value in enumerate(features):
        state.reward_vector[index] += reward * value
    state.selections += 1
    state.reward_sum += reward


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _payload_hash(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _mace_event_sort_key(event: dict[str, object]) -> tuple[object, ...]:
    """Use manifest order when present, otherwise a stable content identity."""

    if "task_order" in event:
        return (
            0,
            int(event["task_order"]),
            int(event.get("rollout_index", 0)),
            str(event.get("agent_identity", event.get("agent_id", ""))),
            int(event.get("interaction_index", 0)),
            str(event["event_id"]),
        )
    return (1, 0, 0, "", 0, str(event["event_id"]))


def _tuple_tree(value: object) -> object:
    if isinstance(value, list):
        return tuple(_tuple_tree(item) for item in value)
    return value


def model_context_features(*, task: str, task_type: str, agent_prompt: str) -> tuple[float, ...]:
    """Interpretable task/role context for model-arm LinUCB selection."""

    combined = f"{task_type} {task} {agent_prompt}".casefold()
    prompt = agent_prompt.casefold()
    roles = (
        _contains(prompt, ("research", "evidence", "retrieve", "search", "检索", "证据")),
        _contains(prompt, ("calculate", "math", "derive", "equation", "计算", "数学", "推导")),
        _contains(prompt, ("verify", "critic", "check", "audit", "验证", "检查", "批判")),
        _contains(prompt, ("synthesize", "final", "integrate", "总结", "综合", "最终")),
        _contains(prompt, ("code", "program", "debug", "implement", "代码", "调试", "实现")),
    )
    tasks = (
        _contains(combined, ("math", "number", "equation", "数学", "计算")),
        _contains(combined, ("qa", "question", "fact", "research", "问答", "事实")),
        _contains(combined, ("code", "program", "software", "代码", "编程")),
        _contains(combined, ("agent", "tool", "webshop", "工具", "购物")),
        _contains(combined, ("reason", "logic", "analysis", "推理", "逻辑", "分析")),
    )
    return tuple(float(value) for value in (*roles, not any(roles), *tasks, 1.0))


def semantic_agent_identity(
    *, prompt: str, layer: int, tools: tuple[str, ...], model_route: str
) -> str:
    role_features = model_context_features(task="", task_type="", agent_prompt=prompt)[:6]
    role_index = max(range(len(role_features)), key=role_features.__getitem__)
    return "\x1f".join(
        (model_route or "unknown-model", f"role-{role_index}", str(layer), ",".join(tools))
    )


def _contains(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def relational_features(
    *,
    agent_id: str,
    peer_id: str,
    responses: dict[str, str],
    historical_performance: float,
    round_index: int,
    max_rounds: int,
) -> tuple[float, ...]:
    own = responses.get(agent_id, "")
    peer = responses.get(peer_id, "")
    diversity = tuple(1.0 - _jaccard_ngrams(own, peer, n) for n in (1, 2, 3))
    divergence: dict[str, list[float]] = {}
    for candidate, text in responses.items():
        divergence[candidate] = [
            sum(
                1.0 - _jaccard_ngrams(text, other, n)
                for other_id, other in responses.items()
                if other_id != candidate
            )
            for n in (1, 2, 3)
        ]
    distinctiveness = []
    for index in range(3):
        total = sum(values[index] for values in divergence.values())
        distinctiveness.append(divergence.get(peer_id, [0.0] * 3)[index] / total if total else 0.0)
    normalized_round = float(round_index) / max(1, int(max_rounds))
    return (
        *diversity,
        *distinctiveness,
        float(historical_performance),
        normalized_round,
        1.0,
    )


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold())


def _jaccard_ngrams(left: str, right: str, n: int) -> float:
    def ngrams(text: str) -> set[tuple[str, ...]]:
        tokens = _tokens(text)
        return {tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)}

    left_set, right_set = ngrams(left), ngrams(right)
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 1.0


def _dot(left: list[float] | tuple[float, ...], right: list[float] | tuple[float, ...]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _matrix_vector(
    matrix: list[list[float]], vector: list[float] | tuple[float, ...]
) -> list[float]:
    return [_dot(row, vector) for row in matrix]
