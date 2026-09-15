import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from selfplay_graph_flowsteer.llm import _attach_provider_usage
from selfplay_graph_flowsteer.outcome_metrics import (
    collect_outcome_metrics,
    record_interrupted_collection,
    summarize_requests,
)
from selfplay_graph_flowsteer.wandb_tracking import WandbTracker, cycle_payload


def test_metrics_jsonl_preserves_literal_unicode_separators(tmp_path):
    from selfplay_graph_flowsteer.outcome_metrics import read_rows

    path = tmp_path / "rows.jsonl"
    row = {"text": "a\u2028b\u2029c"}
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    assert read_rows(path) == [row]


def population(root):
    tasks = [
        SimpleNamespace(
            task_id=name, task_type="qa", metadata={"dataset": dataset, "split": "train"}
        )
        for name, dataset in [("n", "nq_open"), ("h", "healthbench_professional"), ("u", "nq_open")]
    ]
    rows, samples = [], []
    for task in tasks:
        for index in range(5):
            if task.task_id == "u" and index == 4:
                continue
            rid = f"{task.task_id}-r{index}"
            reward = 0.37 if task.task_id == "h" else float(index == 0)
            known = not (task.task_id == "u" and index == 3)
            meta = {
                "reward_known": known,
                "task_reward": reward,
                "task_outcome_passed": bool(reward),
                "verification": {"score": reward},
                "reward_admission_reason": "trusted_task_result",
            }
            rows.append(
                {"rollout_id": rid, "task_id": task.task_id, "reward": reward, "metadata": meta}
            )
            if known and task.task_id != "u":
                samples.append(
                    SimpleNamespace(
                        rollout_id=rid, task_id=task.task_id, metadata=meta, advantage=0
                    )
                )
    (root / "solver_rollouts.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    return tasks, SimpleNamespace(samples=samples)


def test_all_slots_scoring_admission_and_healthbench_continuous_are_distinct(tmp_path):
    tasks, batch = population(tmp_path)
    report = collect_outcome_metrics(
        tmp_path, cycle=0, tasks=tasks, solver_batch=batch, k=5, snapshots={}
    )
    assert report["overall"]["planned_rollout_count"] == 15
    assert report["overall"]["recorded_rollout_count"] == 14
    assert report["overall"]["scored_count"] == 13
    assert report["overall"]["unscored_count"] == 2
    assert report["overall"]["training_eligible_count"] == 10
    assert report["overall"]["binary_scored_count"] == 8
    assert report["overall"]["success_count"] == 2
    nq = report["datasets"]["nq_open"]
    assert nq["task_pass_at_k"] == 1
    assert nq["complete_binary_group_count"] == 1
    assert nq["task_success_histogram"]["1"] == 1
    hb = report["datasets"]["healthbench_professional"]
    assert hb["rollout_success_rate"] is None
    assert hb["task_pass_at_k"] is None
    assert hb["training_reward_known"]["mean"] == pytest.approx(0.37)
    assert hb["task_score_raw"]["mean"] == pytest.approx(0.37)
    assert report["overall"]["binary_datasets"] == ["nq_open"]
    saved = [
        json.loads(line)
        for line in (tmp_path / "primary_reward_rows.jsonl").read_text().splitlines()
    ]
    assert saved[-1]["training_reward"] is None
    assert saved[-2]["training_reward"] is None
    assert saved[1]["training_reward"] == 0


def test_uncertain_score_only_does_not_override_recorded_score(tmp_path):
    tasks, batch = population(tmp_path)
    ledger = [
        {
            "rollout_id": rid,
            "reward_known": True,
            "task_reward": 0.0,
            "task_outcome_passed": False,
            "training_eligible": False,
            "reward_admission_reason": "uncertain_attribution_zero",
            "uncertain_attribution_zero": {"attribution": "unknown"},
        }
        for rid in ("u-r4", "n-r0")
    ]
    (tmp_path / "uncertain_failure_outcomes.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in ledger)
    )
    report = collect_outcome_metrics(
        tmp_path, cycle=0, tasks=tasks, solver_batch=batch, k=5, snapshots={}
    )
    assert report["overall"]["uncertain_attribution_zero_count"] == 1
    assert report["overall"]["scored_count"] == 14
    assert report["overall"]["training_eligible_count"] == 10
    rows = {
        row["rollout_id"]: row
        for row in map(
            json.loads, (tmp_path / "primary_reward_rows.jsonl").read_text().splitlines()
        )
    }
    assert rows["n-r0"]["training_reward"] == 1.0
    assert rows["n-r0"]["reward_source"] == "trusted_task_result"
    assert rows["u-r4"]["training_reward"] == 0.0
    assert not rows["u-r4"]["recorded"]
    assert not rows["u-r4"]["training_eligible"]
    assert not rows["u-r4"]["policy_failure_zero"]


def test_conflicting_slot_is_not_chosen_by_highest_reward(tmp_path):
    tasks, batch = population(tmp_path)
    path = tmp_path / "solver_rollouts.jsonl"
    row = json.loads(path.read_text().splitlines()[0])
    row["reward"] = 99
    with path.open("a") as handle:
        handle.write(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="conflicting primary"):
        collect_outcome_metrics(
            tmp_path, cycle=0, tasks=tasks, solver_batch=batch, k=5, snapshots={}
        )


def test_provider_missing_counters_remain_null_and_event_copies_deduplicate():
    events = [{"event": "backend_request_success", "event_id": "one", "request_role": "worker"}]
    _attach_provider_usage(
        events,
        {"prompt_tokens": 12, "completion_tokens": 3},
        surface="chat_completions",
        model="gpt-5.4",
        effort="low",
    )
    assert events[0]["provider_usage"]["cached_tokens"] is None
    report = summarize_requests(events + events)["worker"]
    assert report["success_calls"] == 1
    assert report["tokens"]["input_tokens"]["sum_reported"] == 12
    assert report["tokens"]["reasoning_tokens"]["sum_reported"] is None
    assert report["tokens"]["reasoning_tokens"]["missing_calls"] == 1


def test_wandb_payload_excludes_raw_data_secrets_and_snapshots():
    payload = cycle_payload(
        {
            "cycle": 0,
            "experiment": {"api_key": "secret"},
            "outcomes": {
                "generation_snapshots": {"secret": "secret"},
                "tasks": [{"task_id": "private_task", "answer": "secret"}],
                "overall": {"success_count": 3, "rate": None},
            },
            "policies": {"solver": {"loss": 0.1, "checkpoint": "/private/path"}},
        }
    )
    text = json.dumps(payload)
    assert "secret" not in text and "private" not in text
    assert payload["cycle/overall/success_count"] == 3
    assert "cycle/overall/rate" not in payload


class FakeRun:
    url = "https://wandb.ai/test/run"

    def __init__(self):
        self.rows = []

    def define_metric(self, *args, **kwargs):
        pass

    def log(self, row):
        self.rows.append(row)

    def finish(self, **kwargs):
        pass


def test_wandb_resume_local_dedup_and_outage_preserves_outbox(tmp_path, monkeypatch):
    import sys

    run = FakeRun()
    monkeypatch.setitem(
        sys.modules, "wandb", SimpleNamespace(init=lambda **kw: run, Settings=lambda **kw: kw)
    )
    tracker = WandbTracker(tmp_path, "offline")
    tracker.start({"synthetic": True})
    record = {"cycle": 0, "policies": {}, "outcomes": {"overall": {"success_count": 1}}}
    tracker.log_cycle(record)
    tracker.log_cycle(record)
    assert len(run.rows) == 1
    first_id = tracker.state["run_id"]
    tracker.finish()
    resumed = WandbTracker(tmp_path, "offline")
    resumed.start({"synthetic": True})
    assert resumed.state["run_id"] == first_id
    assert len(run.rows) == 1
    run.log = lambda row: (_ for _ in ()).throw(RuntimeError("secret response body"))
    with pytest.warns(UserWarning):
        resumed.log_cycle({"cycle": 1})
    assert (tmp_path / "wandb_outbox/cycle-000001.json").exists()
    assert "secret" not in (tmp_path / "wandb_error.json").read_text()


def test_disabled_tracking_needs_no_wandb_import_or_files(tmp_path):
    tracker = WandbTracker(tmp_path / "none")
    tracker.start({})
    tracker.log_cycle({"cycle": 0})
    tracker.finish()
    assert not Path(tmp_path / "none").exists()


def test_wandb_conflict_preserves_payload_and_online_can_replay_offline(tmp_path, monkeypatch):
    import sys

    run = FakeRun()
    run.url = "https://example.invalid/synthetic"
    monkeypatch.setitem(
        sys.modules, "wandb", SimpleNamespace(init=lambda **kw: run, Settings=lambda **kw: kw)
    )
    tracker = WandbTracker(tmp_path, "offline")
    tracker.start({"synthetic_data": True})
    tracker.log_cycle({"cycle": 0})
    path = tmp_path / "wandb_outbox/cycle-000000.json"
    original = path.read_bytes()
    with pytest.warns(UserWarning):
        tracker.log_cycle({"cycle": 0, "outcomes": {"overall": {"success_count": 2}}})
    assert path.read_bytes() == original
    online = WandbTracker(tmp_path, "online")
    online.start({"synthetic_data": True})
    assert len(run.rows) == 2
    assert online.state["queued_files_by_mode"]["online"]


def test_interrupted_collection_keeps_unscored_and_admission_pending(tmp_path):
    root = tmp_path / "cycle-0000"
    root.mkdir()
    tasks, _batch = population(root)
    (root / "tasks.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "task": {
                        "task_id": task.task_id,
                        "task_type": task.task_type,
                        "metadata": task.metadata,
                    }
                }
            )
            + "\n"
            for task in tasks
        )
    )
    report = record_interrupted_collection(tmp_path, 5)
    assert report["collection_interrupted"]
    assert report["overall"]["unscored_count"] == 2
    assert report["overall"]["training_eligible_count"] is None
    assert not report["admission_finalized"]


def test_cli_emits_outcomes_and_wandb_with_mock_training(tmp_path, monkeypatch, capsys):
    import sys

    # These are mock configuration IDs, not real GPU allocation.
    monkeypatch.setenv("SPGFS_ALLOWED_PHYSICAL_GPUS", "0,1")

    from selfplay_graph_flowsteer.cli import main

    from .test_training_ops import _write_config

    run = FakeRun()
    monkeypatch.setitem(
        sys.modules, "wandb", SimpleNamespace(init=lambda **kw: run, Settings=lambda **kw: kw)
    )
    config = _write_config(tmp_path)
    seeds = tmp_path / "seeds.jsonl"
    seeds.write_text('{"seed":"synthetic"}\n')
    output = tmp_path / "experiment"
    assert (
        main(
            [
                "selfplay-experiment",
                "--config",
                str(config),
                "--seed-data",
                str(seeds),
                "--output",
                str(output),
                "--cycles",
                "2",
                "--tasks-per-cycle",
                "1",
                "--rollouts",
                "5",
                "--counterfactuals-per-rollout",
                "0",
                "--verifier",
                "none",
                "--mock",
                "--mock-trainer",
                "--wandb-mode",
                "offline",
                "--mini-batch-size",
                "70",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert len(list((output / "provenance").glob("*.json"))) == 1
    assert (output / "cycle-0001/primary_reward_rows.jsonl").is_file()
    assert len([row for row in run.rows if "rollout_step" in row]) == 2
    latest = json.loads((output / "training_metrics_latest.json").read_text())
    assert "outcomes" in latest
    assert latest["experiment"]["optimizer_schedule"]["solver_total_optimizer_steps"] == 2


@pytest.mark.parametrize(
    ("file_mode", "env_mode", "cli_mode", "expected"),
    [
        (None, None, None, "disabled"),
        ("online", None, None, "online"),
        ("online", "offline", None, "offline"),
        ("online", None, "disabled", "disabled"),
        ("invalid", None, "offline", "offline"),
    ],
)
def test_wandb_mode_precedence(tmp_path, monkeypatch, file_mode, env_mode, cli_mode, expected):
    import os

    from selfplay_graph_flowsteer import cli, wandb_tracking

    monkeypatch.setattr(os, "environ", dict(os.environ))
    os.environ.pop("WANDB_MODE", None)
    if env_mode is not None:
        os.environ["WANDB_MODE"] = env_mode
    config = tmp_path / "config.toml"
    config.write_text("")
    if file_mode is not None:
        (tmp_path / ".env").write_text(f"WANDB_MODE={file_mode}\n")
    calls = []
    monkeypatch.setattr(
        wandb_tracking,
        "WandbTracker",
        lambda root, mode: SimpleNamespace(
            mode=mode, finish=lambda **kwargs: calls.append((mode, kwargs["failed"]))
        ),
    )
    monkeypatch.setattr(cli, "_selfplay_experiment", lambda args, tracker: 0)
    args = SimpleNamespace(config=str(config), output=tmp_path / "out", wandb_mode=cli_mode)
    assert cli.selfplay_experiment(args) == 0
    assert calls == [(expected, False)]


def test_wandb_invalid_environment_mode_fails_before_tracker(tmp_path, monkeypatch):
    from selfplay_graph_flowsteer import cli, wandb_tracking

    monkeypatch.setenv("WANDB_MODE", "invalid")
    monkeypatch.setattr(
        wandb_tracking, "WandbTracker", lambda *args: pytest.fail("tracker must not start")
    )
    args = SimpleNamespace(config=str(tmp_path / "config.toml"), output=tmp_path / "out")
    with pytest.raises(ValueError, match="WANDB_MODE must be"):
        cli.selfplay_experiment(args)


def test_wandb_settings_validate_with_installed_sdk(tmp_path, monkeypatch):
    sdk = pytest.importorskip("wandb")
    run = FakeRun()
    settings = []

    def fake_init(**kwargs):
        settings.append(kwargs["settings"])
        return run

    monkeypatch.setattr(sdk, "init", fake_init)
    tracker = WandbTracker(tmp_path, "offline")
    tracker.start({"synthetic_data": True})
    assert tracker.run is run
    assert not tracker.failure
    assert settings[0].console == "off"
    assert settings[0].disable_code
    assert settings[0].disable_git
    assert settings[0].x_disable_meta
    assert settings[0].x_disable_stats
    tracker.finish()
