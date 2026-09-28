from collections import Counter
import hashlib
import json
from pathlib import Path
from scripts.formal.analyze_hotpot_paired_eval import worker_events
from selfplay_graph_flowsteer.learning import load_fixed_jsonl
from selfplay_graph_flowsteer.qa_metrics import qa_official_metrics

RUN = Path(__file__).resolve().parent
manifest = json.loads((RUN / "manifest.json").read_text())
summary = {"run_dir": str(RUN), "code_commit": manifest["code_commit"], "datasets": {}}
failures = []
for dataset, info in manifest["datasets"].items():
    result_dir = RUN / dataset / "results"
    with (result_dir / "records.jsonl").open() as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    tasks = {example.example_id: example for example in load_fixed_jsonl(info["path"])}
    assert len(records) == len(tasks) == info["count"]
    assert {record["task_id"] for record in records} == set(tasks)
    events = {event["event_id"]: event for record in records for event in worker_events(record)}
    models = Counter(event.get("provider_model", "unknown") for event in events.values())
    assert set(models) <= {"deepseek-flash", "unknown"}, models
    assert all(event["route"] == "deepseek" for event in events.values())
    missing_provider = [event["event_id"] for event in events.values() if not event.get("provider_model")]
    submitted = 0
    for record in records:
        task = tasks[record["task_id"]]
        bound_task = record["trajectory"]["task"]
        assert bound_task["prompt"] == task.task
        assert bound_task["reference"] == task.reference
        valid = bool(record["trajectory"]["answer_submission"]["valid"])
        submitted += valid
        if dataset == "hotpotqa":
            metric = qa_official_metrics(dataset, record["answer"], task.reference)
            assert metric["answer_em"] == record["answer_metrics"]["answer_em"]
            assert abs(metric["answer_f1"] - record["answer_metrics"]["answer_f1"]) < 1e-9
        correct = (record["answer_metrics"]["answer_em"] == 1) if dataset == "hotpotqa" else bool(record["passed"])
        if not correct:
            failures.append({"dataset": dataset, "task_id": record["task_id"], "question": task.task.rsplit("Question: ", 1)[-1], "reference": task.reference, "answer": record["answer"], "valid_submission": valid, "outcome_status": record.get("outcome_status"), "submission_detail": record["trajectory"]["answer_submission"].get("detail"), "passed": record["passed"]})
    state = json.loads((result_dir / "run_state.json").read_text())
    assert state["failed"] == 0
    passed = sum(bool(record["passed"]) for record in records)
    stats = {"planned": info["count"], "completed": len(records), "execution_errors": state["failed"], "valid_submissions": submitted, "unsubmitted": len(records)-submitted, "verifier_passed": passed, "verifier_pass_rate_all": passed/len(records), "worker_requests": len(events), "worker_models": dict(models), "missing_provider_model_event_ids": missing_provider, "worker_event_outcomes": dict(Counter(event["event"] for event in events.values())), "worker_input_tokens": sum((event.get("completion_usage") or {}).get("token_in", 0) for event in events.values()), "worker_output_tokens": sum((event.get("completion_usage") or {}).get("token_out", 0) for event in events.values()), "dataset_sha256": info["sha256"], "records_sha256": hashlib.sha256((result_dir / "records.jsonl").read_bytes()).hexdigest(), "source_binding_verified": True}
    if dataset == "hotpotqa":
        stats["strict_em_correct"] = sum(record["answer_metrics"]["answer_em"] == 1 for record in records)
        stats["strict_em"] = stats["strict_em_correct"] / len(records)
        stats["answer_f1"] = sum(record["answer_metrics"]["answer_f1"] for record in records) / len(records)
    else:
        stats["numeric_correct"] = passed
        stats["numeric_accuracy_all"] = passed/len(records)
    summary["datasets"][dataset] = stats
summary["shared_config_check"] = json.loads((RUN / "shared-config-check.json").read_text())
(RUN / "audited-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n")
(RUN / "failed-answers.json").write_text(json.dumps(failures, ensure_ascii=False, indent=2)+"\n")
print(json.dumps(summary, ensure_ascii=False, indent=2))
