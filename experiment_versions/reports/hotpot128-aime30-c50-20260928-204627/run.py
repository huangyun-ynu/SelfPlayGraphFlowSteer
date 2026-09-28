from __future__ import annotations
import hashlib, json, os, subprocess, sys, time
from pathlib import Path

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
MANIFEST = json.loads((RUN / "manifest.json").read_text())
BASE = [sys.executable, "-m", "selfplay_graph_flowsteer"]

def status(phase, **extra):
    value = {"phase": phase, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **extra}
    tmp = RUN / "status.tmp"
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(RUN / "status.json")
    print(json.dumps(value), flush=True)

def invoke(args, log_name):
    with (RUN / log_name).open("a") as log:
        child = subprocess.Popen(BASE + args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        (RUN / "active-child.json").write_text(json.dumps({"pid": child.pid, "args": args}))
        return child.wait()

def validate_complete(dataset):
    expected = MANIFEST["datasets"][dataset]
    output = RUN / dataset / "results"
    with (output / "records.jsonl").open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    with Path(expected["path"]).open() as handle:
        task_ids = {json.loads(line)["id"] for line in handle if line.strip()}
    assert len(rows) == expected["count"]
    assert {row["task_id"] for row in rows} == task_ids
    state = json.loads((output / "run_state.json").read_text())
    assert not state.get("failed"), state.get("failed")
    return {"completed": len(rows), "passed": sum(bool(r.get("passed")) for r in rows)}

started = False
try:
    status("preflight")
    assert os.environ.get("DEEPSEEK_API_KEY"), "DEEPSEEK_API_KEY is missing"
    from selfplay_graph_flowsteer.application import load_adaptive_config
    config = load_adaptive_config(RUN / "config.toml")
    assert config.worker_runtime_routes == ("deepseek",)
    assert config.runtime_pool()["deepseek"].served_model == "deepseek-flash"
    assert config.runtime_pool()["deepseek"].max_concurrency == 50
    assert config.solver_model.enable_thinking
    for name, info in MANIFEST["datasets"].items():
        assert hashlib.sha256(Path(info["path"]).read_bytes()).hexdigest() == info["sha256"], name
    for relative, expected in MANIFEST["code_sha256"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected, relative
    assert hashlib.sha256((RUN / "config.toml").read_bytes()).hexdigest() == MANIFEST["config_sha256"]
    query = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.total,memory.free,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
    row = next([int(v.strip()) for v in line.split(",")] for line in query.splitlines() if int(line.split(",")[0]) == MANIFEST["gpu_selection"]["selected"])
    assert row[2] > row[1] * .65 and row[3] < 10, "GPU availability changed before launch"
    status("director_starting", gpu=row[0])
    started = True
    rc = invoke(["model-services", "start", "--config", str(RUN / "config.toml"), "--state-dir", str(RUN / "policy-services"), "--role", "solver", "--wait-s", "360"], "service-start.log")
    if rc: raise RuntimeError(f"Director startup failed: exit {rc}")
    completed = {}
    for dataset in MANIFEST["order"]:
        status("running", dataset=dataset, expected=MANIFEST["datasets"][dataset]["count"], previous_completed=completed)
        rc = invoke(["benchmark", "--config", str(RUN / "config.toml"), "--dataset", MANIFEST["datasets"][dataset]["path"], "--output", str(RUN / dataset / "results"), "--workers", "50", "--seed", "0", "--worker-route", "deepseek", "--director-base-url", MANIFEST["director_url"], "--wandb-mode", "disabled", "--skill-context", "off", "--disable-swe"], dataset + ".log")
        if rc: raise RuntimeError(f"{dataset} benchmark exited {rc}; inspect log before resuming")
        completed[dataset] = validate_complete(dataset)
        status("dataset_completed", dataset=dataset, results=completed[dataset])
    status("completed", results=completed)
except BaseException as exc:
    status("failed", error=f"{type(exc).__name__}: {exc}")
    raise
finally:
    if started:
        rc = invoke(["model-services", "stop", "--config", str(RUN / "config.toml"), "--state-dir", str(RUN / "policy-services"), "--role", "solver", "--wait-s", "30"], "service-stop.log")
        (RUN / "cleanup.json").write_text(json.dumps({"exit_code": rc, "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}) + "\n")
