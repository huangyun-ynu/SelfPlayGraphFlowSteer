from __future__ import annotations

import copy
import hashlib
import json
import re
import threading
import uuid
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Protocol

from .observability import TaskSpec, VerificationResult

# TextWorld's PDDL/Tatsu parser keeps process-global mutable state. Register,
# reset, step, and close must not overlap across thread-parallel rollouts.
_TEXTWORLD_ENGINE_LOCK = threading.Lock()


class ALFWorldClient(Protocol):
    def create_session(self, game_path: str, *, max_steps: int) -> dict[str, Any]: ...

    def step(self, session_id: str, command: str) -> dict[str, Any]: ...

    def close_session(self, session_id: str) -> None: ...


@dataclass
class _LocalSession:
    env: Any
    step: int = 0
    done: bool = False


@dataclass
class LocalALFWorldClient:
    """Small in-process client with one independent TextWorld env per session."""

    _sessions: dict[str, _LocalSession] = field(default_factory=dict)
    _lock: threading.RLock = field(default_factory=threading.RLock)

    def create_session(self, game_path: str, *, max_steps: int) -> dict[str, Any]:
        import textworld
        import textworld.gym
        from alfworld.agents.environment.alfred_tw_env import AlfredDemangler

        path = Path(game_path)
        if not path.is_file():
            raise FileNotFoundError(f"ALFWorld game file does not exist: {path}")
        request_infos = textworld.EnvInfos(
            won=True,
            admissible_commands=True,
            score=True,
            max_score=True,
            description=True,
            inventory=True,
        )
        env = None
        with _TEXTWORLD_ENGINE_LOCK:
            try:
                env_id = textworld.gym.register_game(
                    str(path),
                    request_infos=request_infos,
                    max_episode_steps=int(max_steps),
                    wrappers=[AlfredDemangler()],
                )
                env = textworld.gym.make(env_id)
                observation, infos = env.reset()
            except Exception:
                if env is not None:
                    env.close()
                raise
        session_id = uuid.uuid4().hex
        with self._lock:
            self._sessions[session_id] = _LocalSession(env=env)
        return {
            "session_id": session_id,
            "observation": _first(observation),
            "admissible_commands": _commands(infos),
            "reward": 0.0,
            "score": _scalar(infos.get("score", 0.0)),
            "done": False,
            "won": False,
        }

    def step(self, session_id: str, command: str) -> dict[str, Any]:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise RuntimeError("ALFWorld session is not active")
            if session.done:
                raise RuntimeError("ALFWorld episode is already done")
            with _TEXTWORLD_ENGINE_LOCK:
                observation, reward, done, infos = session.env.step(command)
            session.step += 1
            won = bool(_first(infos.get("won", False)))
            done = bool(_first(done))
            session.done = done
            return {
                "observation": _first(observation),
                "admissible_commands": _commands(infos),
                "reward": float(_scalar(reward)),
                "score": float(_scalar(infos.get("score", reward))),
                "done": done,
                "won": won,
            }

    def close_session(self, session_id: str) -> None:
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is not None:
            with _TEXTWORLD_ENGINE_LOCK:
                session.env.close()


@dataclass
class ALFWorldSessionLifecycle:
    """Keep an independent, continuous episode for each Worker in a bound task."""

    client: ALFWorldClient
    data_root: Path
    max_episode_steps: int = 50
    max_rollout_steps: int = 400
    max_observation_chars: int = 4_000
    adapter_id: str = "alfworld"
    attempt_scoped_budget: bool = True
    _task: TaskSpec | None = None
    _game_path: Path | None = None
    _game_fingerprint: str = ""
    _internal_goal_contract: dict[str, Any] = field(default_factory=dict)
    _active_agent: str | None = None
    _active_session: str | None = None
    _active_state: dict[str, Any] = field(default_factory=dict)
    _action_map: dict[str, str] = field(default_factory=dict)
    _results: dict[str, dict[str, Any]] = field(default_factory=dict)
    _rollout_steps: int = 0
    _attempt_index: int = 0
    _active_attempt_index: int = 0
    _sessions: dict[str, tuple[str, dict[str, Any], dict[str, str], int]] = field(
        default_factory=dict
    )
    _lock: threading.RLock = field(default_factory=threading.RLock)

    @property
    def environment_fingerprint(self) -> str:
        return self._game_fingerprint

    def bind_task(self, task: TaskSpec) -> None:
        with self._lock:
            self.close_all()
            game_path = _trusted_game_path(task, self.data_root)
            self._task = task
            self._game_path = game_path
            self._game_fingerprint = _game_fingerprint(game_path)
            self._internal_goal_contract = _alfworld_internal_goal_contract(task)
            self._results = {}
            self._rollout_steps = 0
            self._attempt_index = 0

    def begin_execution(self, *, agent_id: str, seed: int, revision: bool) -> dict[str, Any]:
        del seed
        with self._lock:
            if self._task is None or self._game_path is None:
                raise RuntimeError("ALFWorld lifecycle has no bound task")
            if self._active_session is not None:
                raise RuntimeError("ALFWorld lifecycle already has an active Worker execution")
            agent_id = str(agent_id)
            saved = self._sessions.get(agent_id)
            if saved is not None:
                if self._results.get(agent_id, {}).get("termination_reason") == "environment_step_failed":
                    raise RuntimeError("ALFWorld session state is uncertain after an environment failure")
                session_id, state, action_map, attempt_index = self._sessions.pop(agent_id)
                self._active_agent = agent_id
                self._active_session = session_id
                self._active_state = state
                self._action_map = action_map
                self._active_attempt_index = attempt_index
                self._results[agent_id]["revision"] = bool(revision)
                if not state.get("done"):
                    self._results[agent_id]["termination_reason"] = "running"
                state["remaining_rollout_env_steps"] = max(
                    0, self.max_rollout_steps - self._rollout_steps
                )
                return dict(state)
            if self._rollout_steps >= self.max_rollout_steps:
                raise RuntimeError("ALFWorld rollout environment-step budget is exhausted")
            payload = self.client.create_session(
                str(self._game_path), max_steps=self.max_episode_steps
            )
            session_id = str(payload.get("session_id", "")).strip()
            if not session_id:
                raise RuntimeError("ALFWorld session creation returned no session_id")
            self._active_agent = str(agent_id)
            self._active_session = session_id
            self._attempt_index += 1
            self._active_attempt_index = self._attempt_index
            state = self._public_state(payload, step=0)
            self._active_state = state
            self._results[self._active_agent] = self._result(
                state,
                revision=revision,
                termination_reason="running",
                environment_completed=True,
            )
            return dict(state)

    def step(self, action_id: str) -> dict[str, Any]:
        with self._lock:
            session_id, agent_id = self._active()
            if self._active_state.get("done"):
                raise RuntimeError("ALFWorld episode is already done")
            command = self._action_map.get(str(action_id))
            if command is None:
                raise ValueError("stale_or_unknown_action_id")
            if self._rollout_steps >= self.max_rollout_steps:
                self._results[agent_id] = self._result(
                    self._active_state,
                    revision=bool(self._results.get(agent_id, {}).get("revision", False)),
                    termination_reason="rollout_environment_step_budget_exhausted",
                    environment_completed=False,
                )
                raise RuntimeError("ALFWorld rollout environment-step budget is exhausted")
            try:
                payload = self.client.step(session_id, command)
            except Exception:
                self._results[agent_id] = self._result(
                    self._active_state,
                    revision=bool(self._results.get(agent_id, {}).get("revision", False)),
                    termination_reason="environment_step_failed",
                    environment_completed=False,
                )
                raise
            self._rollout_steps += 1
            step = int(self._active_state.get("step", 0)) + 1
            state = self._public_state(payload, step=step, executed_command=command)
            self._active_state = state
            reason = (
                "official_success"
                if state["done"] and state["success"]
                else "episode_step_budget_exhausted"
                if state["done"] and step >= self.max_episode_steps
                else "official_done_without_success"
                if state["done"]
                else "running"
            )
            self._results[agent_id] = self._result(
                state,
                revision=bool(self._results.get(agent_id, {}).get("revision", False)),
                termination_reason=reason,
                environment_completed=True,
            )
            return dict(state)

    def result_for(self, agent_id: str | None) -> dict[str, Any]:
        with self._lock:
            if not agent_id:
                return _empty_result("missing_output_agent")
            return dict(self._results.get(agent_id, _empty_result("agent_never_executed")))

    def end_execution(self) -> None:
        with self._lock:
            if self._active_agent and not self._active_state.get("done", False):
                current = self._results.get(self._active_agent, {})
                if current.get("termination_reason") == "running":
                    current.update(
                        {
                            "termination_reason": "worker_final_before_environment_done",
                            "environment_completed": True,
                        }
                    )
                self._results[self._active_agent] = current
            if self._active_agent and self._active_session:
                self._sessions[self._active_agent] = (
                    self._active_session, self._active_state,
                    self._action_map, self._active_attempt_index,
                )
                self._active_session = None
                self._active_agent = None
                self._active_state = {}
                self._action_map = {}

    def close_all(self) -> None:
        with self._lock:
            self._close_active()
            for agent_id, (session_id, _, _, _) in list(self._sessions.items()):
                self.client.close_session(session_id)
                del self._sessions[agent_id]
            self._task = None
            self._game_path = None
            self._game_fingerprint = ""
            self._internal_goal_contract = {}

    def _active(self) -> tuple[str, str]:
        if not self._active_session or not self._active_agent:
            raise RuntimeError("ALFWorld Action called outside an active Worker execution")
        return self._active_session, self._active_agent

    def _close_active(self) -> None:
        if self._active_session:
            try:
                self.client.close_session(self._active_session)
            finally:
                self._active_session = None
                self._active_agent = None
                self._active_state = {}
                self._action_map = {}

    def _public_state(
        self,
        payload: dict[str, Any],
        *,
        step: int,
        executed_command: str = "",
    ) -> dict[str, Any]:
        # TextWorld can return stale admissible commands together with done=True.
        # The trusted terminal result ends the Action surface, not the budget.
        commands = (
            []
            if payload.get("done")
            else [str(value) for value in payload.get("admissible_commands", [])]
        )
        self._action_map = {
            f"s{step:04d}:a{index:03d}": command for index, command in enumerate(commands)
        }
        observation = str(payload.get("observation", ""))
        if step == 0:
            initial_observation = observation[: self.max_observation_chars]
            # Extract before truncation: the public reset goal follows the room description.
            _, marker, public_task = observation.partition("Your task is to:")
            public_task_statement = public_task.strip()[: self.max_observation_chars] if marker else ""
        else:
            initial_observation = str(self._active_state.get("initial_observation", ""))
            public_task_statement = str(self._active_state.get("public_task_statement", ""))
        state: dict[str, Any] = {
            "status": (
                "success"
                if bool(payload.get("done")) and bool(payload.get("won"))
                else "done"
                if bool(payload.get("done"))
                else "running"
            ),
            "step": step,
            "observation": observation[: self.max_observation_chars],
            "initial_observation": initial_observation,
            "public_task_statement": public_task_statement,
            "observation_truncated": len(observation) > self.max_observation_chars,
            "reward": float(payload.get("reward", 0.0)),
            "score": float(payload.get("score", payload.get("reward", 0.0))),
            "done": bool(payload.get("done", False)),
            "success": bool(payload.get("won", False)),
            "remaining_env_steps": max(0, self.max_episode_steps - step),
            "remaining_rollout_env_steps": max(0, self.max_rollout_steps - self._rollout_steps),
            "admissible_actions": [
                {"action_id": action_id, "command": command}
                for action_id, command in self._action_map.items()
            ],
            # Runtime-internal goal interpretation. Model-visible prompt projections
            # remove this unless the explicit legacy replay policy is selected.
            # It contains no demonstration, object location, or solution trajectory.
            "goal_contract": dict(self._internal_goal_contract),
        }
        if executed_command:
            state["executed_command"] = executed_command
        state["factual_memory"] = _update_factual_memory(
            self._active_state.get("factual_memory", {}) if step else {},
            command=executed_command,
            observation=observation[: self.max_observation_chars],
            step=step,
        )
        return state

    def _result(
        self,
        state: dict[str, Any],
        *,
        revision: bool,
        termination_reason: str,
        environment_completed: bool,
    ) -> dict[str, Any]:
        return {
            "adapter": self.adapter_id,
            "game_fingerprint": self._game_fingerprint,
            "attempt_index": self._active_attempt_index,
            "revision": bool(revision),
            "done": bool(state.get("done", False)),
            "won": bool(state.get("success", False)),
            "official_score": float(state.get("score", 0.0)),
            "steps": int(state.get("step", 0)),
            "termination_reason": termination_reason,
            "budget_truncated": "budget_exhausted" in termination_reason,
            "environment_completed": bool(environment_completed),
            "public_task_statement": str(state.get("public_task_statement", "")),
        }


@dataclass(frozen=True)
class ALFWorldStepTool:
    lifecycle: ALFWorldSessionLifecycle
    name: str = "alfworld_step"
    stateful: bool = True
    description: str = (
        "Execute exactly one currently admissible ALFWorld command. Copy action_id from the "
        "latest admissible_actions list; stale or invented ids are rejected."
    )
    error_codes: tuple[str, ...] = (
        "stale_or_unknown_action_id",
        "episode_already_done",
        "episode_not_active",
        "environment_step_budget_exhausted",
    )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"action_id": {"type": "string", "minLength": 1}},
            "required": ["action_id"],
            "additionalProperties": False,
        }

    @property
    def output_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "required": [
                "status",
                "step",
                "observation",
                "reward",
                "done",
                "success",
                "admissible_actions",
            ],
        }

    def execute(self, arguments: dict[str, Any]) -> str:
        action_id = arguments.get("action_id")
        if not isinstance(action_id, str) or not action_id.strip():
            raise ValueError("alfworld_step requires a non-empty action_id")
        return json.dumps(self.lifecycle.step(action_id.strip()), ensure_ascii=False)


class ALFWorldEnvironmentVerifier:
    name = "alfworld_environment"
    supports_intermediate_scoring = False

    def verify(self, task: TaskSpec, prediction: str) -> VerificationResult:
        del prediction
        result = task.metadata.get("alfworld_environment_result")
        if not isinstance(result, dict):
            raise ValueError("ALFWorld verification requires a trusted environment result")
        if not bool(result.get("environment_completed", False)):
            # An incomplete environment result is still a trustworthy runtime
            # observation.  Return a zero-score verification so the rollout can
            # be persisted and the runner can either recover it or, under the
            # narrow model-policy contract, retain the final failed attempt as a
            # negative sample.  Raising here used to discard the full Director
            # trajectory before the trace store could write it.
            diagnostic = {
                key: result.get(key)
                for key in (
                    "done",
                    "won",
                    "official_score",
                    "steps",
                    "termination_reason",
                    "budget_truncated",
                    "attempt_index",
                    "revision",
                )
                if key in result
            }
            return VerificationResult(
                score=0.0,
                passed=False,
                verifier=self.name,
                detail=json.dumps(diagnostic, ensure_ascii=False, sort_keys=True),
            )
        won = bool(result.get("won", False))
        detail = {
            key: result.get(key)
            for key in (
                "done",
                "won",
                "official_score",
                "steps",
                "termination_reason",
                "budget_truncated",
                "game_fingerprint",
                "attempt_index",
                "revision",
            )
            if key in result
        }
        return VerificationResult(
            score=float(won),
            passed=won,
            verifier=self.name,
            detail=json.dumps(detail, ensure_ascii=False),
        )


def alfworld_lifecycles(tools: dict[str, Any]) -> tuple[ALFWorldSessionLifecycle, ...]:
    values: dict[int, ALFWorldSessionLifecycle] = {}
    for tool in tools.values():
        lifecycle = getattr(tool, "lifecycle", None)
        if isinstance(lifecycle, ALFWorldSessionLifecycle):
            values[id(lifecycle)] = lifecycle
    return tuple(values.values())


def _trusted_game_path(task: TaskSpec, data_root: Path) -> Path:
    value = task.metadata.get("game_path")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("ALFWorld task requires trusted metadata.game_path")
    path = Path(value).expanduser().resolve()
    root = Path(data_root).expanduser().resolve()
    if not path.is_relative_to(root):
        raise ValueError("ALFWorld game_path escapes configured data_root")
    if path.name != "game.tw-pddl" or not path.is_file():
        raise ValueError("ALFWorld game_path must reference an existing game.tw-pddl")
    source_split = str(task.metadata.get("source_split", task.metadata.get("split", "")))
    if source_split and source_split not in path.parts:
        raise ValueError("ALFWorld game_path does not match trusted source_split")
    return path


def _game_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    for package in ("alfworld", "textworld"):
        try:
            package_version = version(package)
        except PackageNotFoundError:
            package_version = "unknown"
        digest.update(f"\0{package}:{package_version}".encode())
    return digest.hexdigest()


def _commands(infos: dict[str, Any]) -> list[str]:
    value = infos.get("admissible_commands", [])
    if isinstance(value, (list, tuple)) and value and isinstance(value[0], (list, tuple)):
        value = value[0]
    if not isinstance(value, (list, tuple)):
        return []
    return [str(command) for command in value]


def _alfworld_internal_goal_contract(task: TaskSpec) -> dict[str, Any]:
    """Build the trusted declared-goal shape without reading a solution trajectory."""

    metadata = task.metadata
    task_type = str(metadata.get("alfworld_task_type", "")).strip().casefold()
    raw_params = metadata.get("pddl_params", {})
    params = raw_params if isinstance(raw_params, dict) else {}
    transform = ""
    for candidate in ("heat", "cool", "clean"):
        if candidate in task_type:
            transform = candidate
            break
    return {
        "version": "alfworld-public-goal-v1",
        "task_type": task_type,
        "object_target": str(params.get("object_target", "")).strip(),
        "destination_target": str(params.get("parent_target", "")).strip(),
        "toggle_target": str(params.get("toggle_target", "")).strip(),
        "required_object_count": 2 if "pick_two" in task_type else 1,
        "required_transform": transform,
        "requires_placement": bool(
            "place" in task_type or str(params.get("parent_target", "")).strip()
        ),
        "requires_lit_examination": task_type == "look_at_obj_in_light",
        "requires_slicing": bool(params.get("object_sliced", False)),
    }


def _update_factual_memory(
    previous: dict[str, Any], *, command: str, observation: str, step: int
) -> dict[str, Any]:
    """Keep public observations, never infer task goals or unseen object state."""
    memory = copy.deepcopy(previous) if previous else {
        "current_location": "",
        "visited_locations": [],
        "opened_receptacles": [],
        "location_observations": {},
        "object_interactions": {},
        "recent_interactions": [],
    }
    arrival = re.search(r"You arrive at (.+?)\.", observation)
    if arrival:
        location = arrival.group(1)
        memory["current_location"] = location
        if location not in memory["visited_locations"]:
            memory["visited_locations"].append(location)
    opened = re.search(r"You open the (.+?)\.", observation)
    if opened and opened.group(1) not in memory["opened_receptacles"]:
        memory["opened_receptacles"].append(opened.group(1))
    location = memory["current_location"]
    # These are dated observations, not assertions that objects remain there.
    # Keep inspection evidence separate from subsequent manipulation feedback.
    if location and (arrival or opened or command.startswith("examine ")):
        memory["location_observations"][location] = {
            "step": step, "command": command, "observation": observation,
        }
    if command:
        if command.startswith(("take ", "move ", "clean ", "cool ", "heat ")):
            # Retain each object's dated feedback even after recent history rotates.
            object_name = re.split(r" from | to | with ", command.split(" ", 1)[1])[0]
            memory["object_interactions"].setdefault(object_name, []).append({
                "step": step, "command": command, "observation": observation,
            })
        memory["recent_interactions"].append({
            "step": step, "location": location,
            "command": command, "observation": observation,
        })
        memory["recent_interactions"] = memory["recent_interactions"][-32:]
    return memory


def _first(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return value[0] if value else ""
    return value


def _scalar(value: Any) -> float:
    value = _first(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _empty_result(reason: str) -> dict[str, Any]:
    return {
        "adapter": "alfworld",
        "done": False,
        "won": False,
        "official_score": 0.0,
        "steps": 0,
        "termination_reason": reason,
        "budget_truncated": "budget_exhausted" in reason,
        "environment_completed": False,
    }
