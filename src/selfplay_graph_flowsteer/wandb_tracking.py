"""Optional, numeric-only W&B sink; durable local telemetry stays authoritative."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import re
import uuid
import warnings
from pathlib import Path

from .outcome_metrics import write_json


def numeric_metrics(value, prefix=""):
    output = {}
    if isinstance(value, dict):
        for key, item in value.items():
            key = str(key)
            if not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", key):
                continue
            if any(
                word in key.lower() for word in ("secret", "api_key", "password", "authorization")
            ):
                continue
            output.update(numeric_metrics(item, f"{prefix}/{key}" if prefix else key))
    elif isinstance(value, (int, float)) and math.isfinite(value):
        output[prefix] = value
    return output


def cycle_payload(record):
    outcomes = record.get("outcomes", {})
    evaluation = bool(record.get("experiment", {}).get("evaluation_only"))
    prefix = "eval" if evaluation else "cycle"
    payload = {"rollout_step": int(record["cycle"]) + 1, "cycle/is_evaluation": int(evaluation)}
    elapsed = record.get("experiment", {}).get("cycle_elapsed_s")
    # Retained outcomes include prior attempts; dividing them by only this
    # resume's wall time would falsely inflate throughput.
    partial_resume = bool(record.get("experiment", {}).get("resumed_partial_cycle"))
    payload["cycle/partial_resume"] = int(partial_resume)
    if isinstance(elapsed, (int, float)) and elapsed > 0 and not partial_resume:
        for key in ("planned_rollout_count", "scored_count", "training_eligible_count"):
            count = outcomes.get("overall", {}).get(key)
            if isinstance(count, (int, float)):
                payload[f"{prefix}/throughput/{key}_per_minute"] = 60 * count / elapsed
    # Do not recursively export full context, snapshots, rows or provider messages.
    for name, section in {
        "datasets": outcomes.get("datasets", {}),
        "overall": outcomes.get("overall", {}),
        "coverage": outcomes.get("field_coverage", {}),
        "extra_executions": outcomes.get("extra_executions", {}),
        "primary_api": outcomes.get("primary_api", {}),
        "advantage": outcomes.get("advantage", {}),
        "source_coverage": outcomes.get("source_coverage", {}),
        "collection_incidents": outcomes.get("collection_incidents", {}),
        "graph_similarity": outcomes.get("graph_similarity", {}),
        "policies": record.get("policies", {}),
        "batches": record.get("batches", {}),
        "topology": record.get("topology_policy", {}),
        "runtime": record.get("runtime", {}),
        "relation": record.get("relation_counterfactual", {}),
        "research": outcomes.get("research", {}),
        "learning_dynamics": record.get("learning_dynamics", {}),
        "curriculum": {
            key: record.get("rollout", {}).get(key, {})
            for key in ("proposal_extraction", "frontier")
        },
        "timing": {
            key: value
            for key, value in record.get("experiment", {}).items()
            if key
            in (
                "cycle_elapsed_s",
                "cycle_target_s",
                "cycle_target_exceeded",
                "collection_elapsed_s",
                "training_and_restore_elapsed_s",
            )
        },
    }.items():
        payload.update(numeric_metrics(section, prefix + "/" + name))
    return payload


class WandbTracker:
    def __init__(self, root, mode="disabled"):
        self.root = Path(root)
        self.mode = mode
        self.run = None
        self.state = {}
        self.failure = False

    def start(self, config):
        if self.mode == "disabled":
            return
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / "wandb_tracking.json"
        self.state = (
            json.loads(path.read_text()) if path.exists() else {"run_id": uuid.uuid4().hex[:16]}
        )
        # Configuration is an explicit non-secret allowlist supplied by the CLI.
        self.state.update(
            project=os.environ.get("WANDB_PROJECT", "selfplay-graph-flowsteer"),
            requested_mode=self.mode,
        )
        write_json(path, self.state)
        try:
            wandb = importlib.import_module("wandb")
            for name in ("NO_PROXY", "no_proxy"):
                os.environ[name] = ",".join(
                    filter(None, [os.environ.get(name), "api.wandb.ai,wandb.ai,.wandb.ai"])
                )
            run_dir = self.root / "wandb"
            run_dir.mkdir(exist_ok=True)
            self.run = wandb.init(
                project=self.state["project"],
                entity=os.environ.get("WANDB_ENTITY") or None,
                id=self.state["run_id"],
                resume="allow" if self.mode == "online" else None,
                name=os.environ.get("WANDB_NAME", self.root.name),
                mode=self.mode,
                dir=str(run_dir),
                config=config,
                settings=wandb.Settings(
                    console="off",
                    disable_code=True,
                    disable_git=True,
                    x_disable_meta=True,
                    x_save_requirements=False,
                    x_disable_stats=bool(config.get("synthetic_data")),
                    x_stats_gpu_device_ids=config.get("allocated_gpu_ids"),
                    init_timeout=30,
                    finish_timeout=30,
                ),
            )
            self.run.define_metric("rollout_step")
            self.run.define_metric("cycle/*", step_metric="rollout_step")
            self.run.define_metric("eval/*", step_metric="rollout_step")
            for role in ("proposer", "solver"):
                self.run.define_metric(f"optimizer_step/{role}")
                self.run.define_metric(f"train/{role}/*", step_metric=f"optimizer_step/{role}")
            self.state.update(mode=self.mode, url=self.run.url if self.mode == "online" else None)
            write_json(path, self.state)
            self._drain()
        except Exception as exc:  # Metrics transport must not change the optimizer outcome.
            self._error("init", exc)

    def _error(self, stage, exc):
        self.failure = True
        write_json(
            self.root / "wandb_error.json",
            {"stage": stage, "exception_type": type(exc).__name__, "local_outbox_preserved": True},
        )
        warnings.warn(
            f"W&B {stage} failed ({type(exc).__name__}); local metrics preserved", stacklevel=2
        )

    def log_cycle(self, record):
        if self.mode == "disabled":
            return
        payload = cycle_payload(record)
        steps = []
        for role in ("proposer", "solver"):
            for row in record.get("policies", {}).get(role, {}).get("step_metrics", []):
                if row.get("role_optimizer_step") is not None:
                    steps.append(
                        {
                            f"optimizer_step/{role}": row["role_optimizer_step"],
                            **numeric_metrics(row, f"train/{role}"),
                        }
                    )
        # Safe payload, idempotent per cycle on local disk; no task/trace attachments.
        path = self.root / "wandb_outbox" / f"cycle-{int(record['cycle']):06d}.json"
        data = {"payload": payload, "optimizer_rows": steps}
        if path.exists() and json.loads(path.read_text()) != data:
            self._error("cycle_conflict", ValueError("cycle payload changed"))
            return
        write_json(path, data)
        self._drain()

    def log_interruption(self, outcomes):
        if self.mode == "disabled":
            return
        payload = {"diagnostic/interrupted": 1}
        if outcomes:
            payload["diagnostic/cycle"] = outcomes["cycle"]
            for key in ("scored_count", "unscored_count", "planned_rollout_count"):
                payload[f"diagnostic/{key}"] = outcomes["overall"][key]
        write_json(self.root / "wandb_interruption.json", payload)
        if self.run is not None:
            try:
                self.run.log(payload)
            except Exception as exc:
                self._error("interruption", exc)

    def _drain(self):
        if self.run is None or self.failure:
            return
        for path in sorted((self.root / "wandb_outbox").glob("cycle-*.json")):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            sent = self.state.setdefault("queued_files_by_mode", {}).setdefault(self.mode, {})
            if sent.get(path.name) == digest:
                continue
            if path.name in sent:
                self._error("cycle_conflict", ValueError("cycle payload changed"))
                return
            data = json.loads(path.read_text())
            try:
                for row in data["optimizer_rows"]:
                    self.run.log(row)
                self.run.log(data["payload"])
                sent[path.name] = digest  # SDK queued, not proof of remote persistence.
                write_json(self.root / "wandb_tracking.json", self.state)
            except Exception as exc:
                self._error("log", exc)
                return

    def finish(self, failed=False):
        if self.run is not None:
            try:
                self.run.finish(exit_code=int(failed))
            except Exception as exc:
                self._error("finish", exc)
