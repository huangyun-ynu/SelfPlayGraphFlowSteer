"""Replay local SOTA actions across Worker revisions in real TextWorld sessions."""

import argparse
import json
from pathlib import Path

from selfplay_graph_flowsteer.alfworld import ALFWorldSessionLifecycle, LocalALFWorldClient
from selfplay_graph_flowsteer.observability import TaskSpec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = []
    with args.records.open() as stream:
        for line in stream:
            record = json.loads(line)
            if not record["task_id"].startswith("alfworld/") or not record["passed"]:
                continue
            trajectory = record["trajectory"]
            artifacts = {
                artifact["artifact_id"]: artifact
                for event in trajectory["events"]
                for artifact in (event["payload"].get("execution") or {}).get("artifacts", {}).values()
            }
            artifact = next((a for a in artifacts.values() if a.get("environment_result", {}).get("won")), None)
            if artifact is None:
                continue
            commands = [
                item.get("observation", {}).get("output", {}).get("executed_command")
                for item in artifact.get("react_trace", [])
            ]
            commands = [command for command in commands if command]
            if not any(command.startswith("take ") for command in commands[:-1]):
                continue
            task_data = trajectory["task"]
            task = TaskSpec(task_data["task_id"], task_data["prompt"], metadata=task_data["metadata"])
            game = Path(task.metadata["game_path"])
            client = LocalALFWorldClient()
            lifecycle = ALFWorldSessionLifecycle(client, game.parent)
            report = {"task_id": task.task_id, "prompt": task.prompt, "events": [], "passed": False}
            try:
                lifecycle.bind_task(task)
                state = lifecycle.begin_execution(agent_id="agent_1", seed=0, revision=False)
                original_session = lifecycle._active_session
                report["initial_state"] = state
                for command in commands:
                    action = next(a for a in state["admissible_actions"] if a["command"] == command)
                    state = lifecycle.step(action["action_id"])
                    report["events"].append({"command": command, "state": state})
                    if state["done"]:
                        break
                    # Every action crosses the same lifecycle boundary as a Worker revision.
                    lifecycle.end_execution()
                    resumed = lifecycle.begin_execution(agent_id="agent_1", seed=0, revision=True)
                    assert lifecycle._active_session == original_session
                    assert resumed == state, "State changed across revision"
                    assert len(client._sessions) == 1
                    report["events"][-1]["revision_state_identical"] = True
                    state = resumed
                lifecycle.end_execution()
                report["environment_result"] = lifecycle.result_for("agent_1")
                assert report["environment_result"]["won"], "Replay did not reach official success"
                assert report["environment_result"]["attempt_index"] == 1
                report["passed"] = True
            except Exception as exc:
                report["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                lifecycle.close_all()
                report["sessions_after_cleanup"] = len(client._sessions)
            results.append(report)
            print(json.dumps({k: v for k, v in report.items() if k not in {"events", "initial_state"}}), flush=True)
            if len(results) == 3:
                break
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"mode": "real_environment_recorded_actions_no_model_calls", "results": results}, indent=2))
    if len(results) != 3 or not all(r["passed"] and r["sessions_after_cleanup"] == 0 for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
