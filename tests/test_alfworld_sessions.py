import pytest

from selfplay_graph_flowsteer.alfworld import ALFWorldSessionLifecycle
from selfplay_graph_flowsteer.observability import TaskSpec


class Client:
    def __init__(self):
        self.sessions = {}
        self.created = 0
        self.closed = []

    def create_session(self, game_path, *, max_steps):
        self.created += 1
        session = str(self.created)
        self.sessions[session] = 0
        return dict(self.state(session), session_id=session)

    def state(self, session):
        done = self.sessions[session] == 2
        return {
            "observation": f"Progress {self.sessions[session]}",
            "admissible_commands": [] if done else ["take mug"],
            "done": done, "won": done, "score": float(done),
        }

    def step(self, session, command):
        self.sessions[session] += 1
        return self.state(session)

    def close_session(self, session):
        self.closed.append(session)


@pytest.fixture
def lifecycle(tmp_path):
    game = tmp_path / "game.tw-pddl"
    game.write_text("test")
    value = ALFWorldSessionLifecycle(Client(), tmp_path)
    value.bind_task(TaskSpec("test", "Move mug", metadata={"game_path": str(game)}))
    yield value
    value.close_all()


def test_revision_preserves_scene_action_ids_and_terminal_result(lifecycle):
    first = lifecycle.begin_execution(agent_id="a", seed=0, revision=False)
    state = lifecycle.step(first["admissible_actions"][0]["action_id"])
    lifecycle.end_execution()
    with pytest.raises(RuntimeError, match="outside"):
        lifecycle.step(state["admissible_actions"][0]["action_id"])
    resumed = lifecycle.begin_execution(agent_id="a", seed=1, revision=True)
    assert resumed == state
    assert lifecycle.client.created == 1
    assert not lifecycle.client.closed
    terminal = lifecycle.step(resumed["admissible_actions"][0]["action_id"])
    assert terminal["success"]
    lifecycle.end_execution()
    assert lifecycle.begin_execution(agent_id="a", seed=2, revision=True)["done"]
    with pytest.raises(RuntimeError, match="already done"):
        lifecycle.step("unused")
    assert lifecycle.result_for("a")["won"]
    assert lifecycle.result_for("a")["attempt_index"] == 1


def test_workers_have_independent_persistent_sessions_and_cleanup(lifecycle):
    first = lifecycle.begin_execution(agent_id="a", seed=0, revision=False)
    lifecycle.step(first["admissible_actions"][0]["action_id"])
    lifecycle.end_execution()
    assert lifecycle.begin_execution(agent_id="b", seed=0, revision=False)["step"] == 0
    lifecycle.end_execution()
    assert lifecycle.begin_execution(agent_id="a", seed=0, revision=True)["step"] == 1
    lifecycle.close_all()
    assert sorted(lifecycle.client.closed) == ["1", "2"]
    assert not lifecycle._sessions


def test_revision_does_not_reset_rollout_budget(lifecycle):
    lifecycle.max_rollout_steps = 1
    first = lifecycle.begin_execution(agent_id="a", seed=0, revision=False)
    lifecycle.step(first["admissible_actions"][0]["action_id"])
    lifecycle.end_execution()
    resumed = lifecycle.begin_execution(agent_id="a", seed=0, revision=True)
    assert resumed["remaining_rollout_env_steps"] == 0
    assert resumed["remaining_env_steps"] == 49
    with pytest.raises(RuntimeError, match="budget is exhausted"):
        lifecycle.step(resumed["admissible_actions"][0]["action_id"])
    assert lifecycle.client.created == 1


def test_failed_environment_is_not_silently_restarted(lifecycle, monkeypatch):
    first = lifecycle.begin_execution(agent_id="a", seed=0, revision=False)

    def fail(*args):
        raise OSError("lost environment response")

    monkeypatch.setattr(lifecycle.client, "step", fail)
    with pytest.raises(OSError):
        lifecycle.step(first["admissible_actions"][0]["action_id"])
    lifecycle.end_execution()
    with pytest.raises(RuntimeError, match="uncertain"):
        lifecycle.begin_execution(agent_id="a", seed=0, revision=True)
    assert lifecycle.client.created == 1


def test_binding_another_task_closes_saved_sessions(lifecycle):
    lifecycle.begin_execution(agent_id="a", seed=0, revision=False)
    lifecycle.end_execution()
    task = lifecycle._task
    lifecycle.bind_task(task)
    assert lifecycle.client.closed == ["1"]
    assert lifecycle.begin_execution(agent_id="a", seed=0, revision=False)["step"] == 0
    assert lifecycle.client.created == 2


def test_factual_memory_survives_navigation_revision_and_isolates_agents(lifecycle, monkeypatch):
    from selfplay_graph_flowsteer.runtime import _alfworld_context_for_prompt

    commands = ["go to fridge 1", "open fridge 1", "go to microwave 1"]
    observations = ["Room", "You arrive at fridge 1. The fridge 1 is closed.",
                    "You open the fridge 1. In it, you see nothing.",
                    "You arrive at microwave 1. The microwave 1 is closed."]

    def state(session):
        index = lifecycle.client.sessions[session]
        return {"observation": observations[index],
                "admissible_commands": commands[index:index + 1],
                "done": False, "won": False}

    monkeypatch.setattr(lifecycle.client, "state", state)
    current = lifecycle.begin_execution(agent_id="a", seed=0, revision=False)
    initial = current
    for _ in commands:
        current = lifecycle.step(current["admissible_actions"][0]["action_id"])
    memory = current["factual_memory"]
    assert memory["current_location"] == "microwave 1"
    assert memory["visited_locations"] == ["fridge 1", "microwave 1"]
    assert memory["opened_receptacles"] == ["fridge 1"]
    assert "nothing" in memory["location_observations"]["fridge 1"]["observation"]
    assert initial["factual_memory"]["visited_locations"] == []
    lifecycle.end_execution()
    other = lifecycle.begin_execution(agent_id="b", seed=0, revision=False)
    assert other["factual_memory"]["visited_locations"] == []
    lifecycle.end_execution()
    resumed = lifecycle.begin_execution(agent_id="a", seed=0, revision=True)
    assert resumed["factual_memory"] == memory
    context = {"action_environment": {"state": resumed, "alfworld_progress": {}}}
    visible = _alfworld_context_for_prompt(context)["action_environment"]
    assert visible["alfworld_progress"]["location_observations"] == memory["location_observations"]
    assert "action_id" not in str(visible["alfworld_progress"])
    assert "goal_contract" not in visible["state"]
    assert "factual_memory" in resumed


@pytest.mark.parametrize("policy", ["factual_memory_v1", "raw_state_v1"])
def test_public_reset_goal_survives_steps_revision_and_prompt_projection(lifecycle, monkeypatch, policy):
    from selfplay_graph_flowsteer.runtime import _alfworld_context_for_prompt

    goal = "clean some spatula and put it in diningtable."
    reset = "Room description. Your task is to: " + goal
    original_state = lifecycle.client.state

    def state(session):
        value = original_state(session)
        if lifecycle.client.sessions[session] == 0:
            value["observation"] = reset if session == "1" else "Another room"
        return value

    monkeypatch.setattr(lifecycle.client, "state", state)
    initial = lifecycle.begin_execution(agent_id="a", seed=0, revision=False)
    current = lifecycle.step(initial["admissible_actions"][0]["action_id"])
    assert current["observation"] != reset
    lifecycle.end_execution()
    other = lifecycle.begin_execution(agent_id="b", seed=0, revision=False)
    assert other["public_task_statement"] == ""
    assert other["initial_observation"] == "Another room"
    lifecycle.end_execution()
    resumed = lifecycle.begin_execution(agent_id="a", seed=1, revision=True)
    context = {"action_environment": {"state": resumed, "alfworld_progress": {}}}
    visible = _alfworld_context_for_prompt(context, guidance_policy=policy)
    public = visible["action_environment"]["state"]
    assert public["initial_observation"] == reset
    assert public["public_task_statement"] == goal
    assert "goal_contract" not in public
    assert lifecycle.result_for("a")["public_task_statement"] == goal
    assert lifecycle._task.prompt == "Move mug"


def test_public_reset_goal_is_extracted_before_room_truncation(lifecycle, monkeypatch):
    goal = "look at vase under the desklamp."
    original_state = lifecycle.client.state

    def state(session):
        value = original_state(session)
        value["observation"] = "Room " * 100 + "Your task is to: " + goal
        return value

    monkeypatch.setattr(lifecycle.client, "state", state)
    lifecycle.max_observation_chars = 100
    initial = lifecycle.begin_execution(agent_id="a", seed=0, revision=False)
    assert initial["observation_truncated"]
    assert len(initial["initial_observation"]) == 100
    assert initial["public_task_statement"] == goal


def test_factual_memory_keeps_object_feedback_without_inventing_success():
    from selfplay_graph_flowsteer.alfworld import _update_factual_memory

    memory = _update_factual_memory({}, command="go to countertop 1",
                                   observation="You arrive at countertop 1. You see a mug 1.", step=1)
    for step, command, observation in [
        (2, "take mug 1 from countertop 1", "You pick up the mug 1 from the countertop 1."),
        (3, "move mug 1 to countertop 1", "Nothing happens."),
    ]:
        memory = _update_factual_memory(memory, command=command, observation=observation, step=step)
    for step in range(4, 40):
        memory = _update_factual_memory(memory, command="look", observation="Room", step=step)
    assert len(memory["recent_interactions"]) == 32
    records = memory["object_interactions"]["mug 1"]
    assert records[0]["step"] == 2
    assert records[1]["observation"] == "Nothing happens."
    assert memory["location_observations"]["countertop 1"]["step"] == 1
