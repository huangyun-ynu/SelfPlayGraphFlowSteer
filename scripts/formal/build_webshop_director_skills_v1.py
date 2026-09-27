"""Build a Director-only native SkillBank and public provenance, without model calls."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "configs/director_skills/webshop_trajectory_v1"
SOURCE = DEST / "skills.source.json"
RUNS = {
    "merged": ROOT / "state/experiments/webshop-w08-merged-128-20260924-run1/candidate",
    "identity": ROOT / "state/experiments/webshop-merged-identity-128-20260924-run1/candidate",
    "memory_source": ROOT / "state/experiments/webshop-memory-source-128-20260924-run1/candidate",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main() -> None:
    source = json.loads(SOURCE.read_text())
    entries = source["entries"]
    assert 0 < len(entries) < 60
    assert len({e["id"] for e in entries}) == len(entries)
    requested = {f"webshop/goal-{n}" for entry in entries for n in entry["examples"]}
    trace_index: dict[str, dict] = {}
    run_index = {}
    for alias, root in RUNS.items():
        hashes = {}
        ids = set()
        passed = 0
        for path in sorted((root / "trajectories").glob("*.json")):
            trace = json.loads(path.read_text())
            task_id = trace["task"]["task_id"]
            ids.add(task_id)
            passed += bool(trace["verification"]["passed"])
            hashes[str(path.relative_to(ROOT))] = sha(path)
            if task_id not in requested:
                continue
            # Only public environment effects and graph state; never raw reasoning,
            # reference options, hidden attributes, target ASIN or scorer internals.
            actions = []
            tag = task_id.replace("/", "_")
            request_paths = sorted(
                (root / "raw_audit/environment" / tag).glob("*.request.json"),
                key=lambda p: json.loads(p.read_text())["started_unix_s"],
            )
            for request_path in request_paths:
                request = json.loads(request_path.read_text())["request"]
                endpoint = request["path"].split("/")[-1]
                if endpoint not in {"search", "click", "commit"}:
                    continue
                response_path = request_path.with_name(request_path.name.replace(".request.json", ".response.json"))
                response = json.loads(response_path.read_text()).get("response") or {}
                actions.append({
                    "ordinal": len(actions) + 1,
                    "operation": endpoint,
                    "arguments": {k: v for k, v in (request.get("payload") or {}).items() if k in {"query", "target_id"}},
                    "product": response.get("product"),
                    "selected_options": response.get("selected_options"),
                    "page_type": response.get("page_type"),
                    "steps": response.get("steps"),
                    "purchased": response.get("purchased"),
                    "request_path": str(request_path.relative_to(ROOT)),
                })
            artifacts = {}
            for event in trace.get("events", []):
                for artifact in (event.get("payload", {}).get("execution") or {}).get("artifacts", {}).values():
                    artifacts[artifact["artifact_id"]] = artifact
            owner_records = []
            for artifact in artifacts.values():
                progress = artifact.get("webshop_progress") or {}
                if not progress:
                    continue
                owner_records.append({
                    "artifact_id": artifact["artifact_id"],
                    "agent_id": artifact.get("agent_id"),
                    "revision": artifact.get("revision"),
                    "owner": progress.get("environment_owner"),
                    "access": progress.get("environment_access"),
                    "commit_ready": progress.get("commit_ready"),
                    "commit_protocol_status": progress.get("commit_protocol_status"),
                })
            key = f"{alias}:{task_id}"
            trace_index[key] = {
                "trace": str(path.relative_to(ROOT)),
                "sha256": hashes[str(path.relative_to(ROOT))],
                "public_request": trace["task"]["prompt"],
                "public_actions": actions,
                "owner_records": owner_records,
                "output_agent": trace["final_graph"].get("output_agent"),
                "saved_reward": trace["verification"]["score"],
                "passed": bool(trace["verification"]["passed"]),
                "outcome_interpretation": "observational association, not a demonstrated skill effect",
            }
        assert len(ids) == 128, (alias, len(ids))
        run_index[alias] = {"path": str(root.relative_to(ROOT)), "tasks": len(ids), "strict_successes": passed, "trace_sha256": hashes}

    cards = []
    provenance = {}
    table = []
    for entry in entries:
        card = {key: entry[key] for key in ["name", "description", "trigger", "plan", "pitfall", "constraint", "kind"]}
        card.update(
            skill_id=entry["id"], task_types=["shopping"],
            evidence=[f"public trajectory pattern; provenance.json#/cards/{entry['id']}"],
            stats={"usage_count": 0, "helpful_count": 0, "hurt_count": 0, "last_used_step": -1, "creation_step": 0, "is_seed": True},
        )
        cards.append({
            "card": card, "version": 1, "status": "seed",
            "provenance": "trajectory_derived_unvalidated", "parent_version": None,
            "source_cases": [], "required_tools": ["webshop_search", "webshop_click"],
            "excluded_task_types": [],
        })
        refs = [f"{alias}:webshop/goal-{n}" for n in entry["examples"] for alias in RUNS]
        assert all(key in trace_index for key in refs), entry["id"]
        provenance[entry["id"]] = {
            "evidence_level": entry["evidence_level"],
            "interpretation_zh": entry["evidence_note_zh"],
            "public_trace_refs": refs,
            "distinct_source_tasks": len(entry["examples"]),
            "causal_improvement_validated": False,
            "framework_refs": ["src/selfplay_graph_flowsteer/adaptive.py", "src/selfplay_graph_flowsteer/director.py", "src/selfplay_graph_flowsteer/skill_evolution_v2.py"],
        }
        directory = DEST / "cards" / entry["id"]
        directory.mkdir(parents=True, exist_ok=True)
        skill = (
            "---\nname: " + entry["id"] + "\ndescription: " + json.dumps(entry["description"] + " Apply when " + entry["trigger"], ensure_ascii=False)
            + "\nmetadata:\n  target: director\n  dataset: webshop\n  validation: unvalidated\n---\n\n"
            + "# " + entry["name"] + "\n\n"
            + "Use as optional Director orchestration guidance for a WebShop shopping workflow.\n\n"
            + "## When\n\n" + entry["trigger"] + "\n\n"
            + "## Orchestration\n\n" + entry["plan"] + "\n\n"
            + "## Pitfall\n\n" + entry["pitfall"] + "\n\n"
            + "## Boundary\n\n" + entry["constraint"] + "\n"
        )
        (directory / "SKILL.md").write_text(skill)
        table.append(f'|{entry["id"]}|[{entry["name"]}]({directory}/SKILL.md)|{entry["orchestration_zh"]}|{entry["evidence_note_zh"]}|')

    snapshot = {
        "schema": "director_skill_v2",
        "snapshot_id": source["package"] + "-" + sha(SOURCE)[:16],
        "cards": cards,
    }
    dump(DEST / "snapshot.v2.json", snapshot)
    dump(DEST / "provenance.json", {
        "package": source["package"], "source_sha256": sha(SOURCE),
        "authorship": "Codex-authored in response to explicit user request",
        "unique_tasks": 128, "trajectories": sum(x["tasks"] for x in run_index.values()),
        "source_split": "historical official-test-128 already used for development; not held-out validation",
        "runs": run_index, "cards": provenance, "public_cases": trace_index,
        "model_visible_boundary": "Only formatted card name, trigger, plan, pitfall and constraint enter the Director prompt. This provenance file and task identifiers are never concatenated into it.",
    })
    (DEST / "solver_skillbank.fragment.toml").write_text(
        '# Merge these settings into an isolated evaluation config under configs/.\n'
        '# This is a fragment, not a complete runtime configuration.\n'
        '# Keep the frozen source, models, reasoning effort and budgets of the control.\n'
        '[solver_skillbank]\n'
        'enabled = true\nusage = "always"\nmode = "director_skill_v2"\n'
        'path = "configs/director_skills/webshop_trajectory_v1/snapshot.v2.json"\n'
        'prompt_token_budget = 1024\nretrieve_top_k = 3\n'
        'embedding_model_path = "models/e5-base-v2"\n\n'
        '[solver_skillbank.pats]\nenabled = false\n'
    )
    doc = f'''# WebShop Director 轨迹编排技能 v1

共 {len(cards)} 条，来源为合并版、身份修复版、记忆来源修复版各128条轨迹，共384条、同一组128个任务。技能供Director决定职责、依赖、修订与收尾；Worker继续按现有合法动作执行。

此版本是用户明确要求生成的 WebShop 专用 Skill 实验材料。此前无Skill基线保留。status=seed只表示可被现有加载器检索，provenance=trajectory_derived_unvalidated表示尚未通过效果验证；usage/helpful/hurt均为0，没有虚构提分结果。

## 技能目录

|ID|技能|Director的编排动作|轨迹依据与解释边界|
|---|---|---|---|
''' + "\n".join(table) + f'''

## 加载与上下文边界

- 原生加载文件：[snapshot.v2.json]({DEST}/snapshot.v2.json)，兼容项目 DirectorSkillBankV2。作用域是shopping，且运行时须提供WebShop搜索与点击能力；其他任务类型和缺少这些能力的执行器不匹配。
- [配置片段]({DEST}/solver_skillbank.fragment.toml)应合并到独立评测配置。保留原来的模型、low推理强度、Worker路线和预算；CLI显式使用`--skill-context on`。`--director-skill-root`属于另一种单文件静态提示加载器，不用于这个v2包。
- 现有流程只在solve开始时按原始任务检索最多3条，随后注入Director系统上下文，token上限1024。它不会在每个状态变化时重新检索，因此本包没有声称实现按阶段自动切换12条技能。后期预算/收尾技能能否被初始检索覆盖，需要单独观察。
- 技能不是固定图模板，不强制多Agent、不更换Worker路线、不规定固定动作序列，也不把历史题号、ASIN、品牌答案、隐藏选项或评分倍率写入模型可见正文。
- 逐卡SKILL.md用于阅读或其他支持该格式的工具；本项目实际读取JSON字段。目录未安装到全局Codex技能，也没有自动启用到正在使用的基线。

## 证据与验证

[provenance.json]({DEST}/provenance.json)保留源轨迹哈希、公开观察、所有权与结果关联，和模型正文分开。案例是开发材料，不是新独立测试集。轻量评审等卡片是针对已观察缺口提出的编排假设，不能说已经证明有效。

生成脚本为[scripts/formal/build_webshop_director_skills_v1.py]({ROOT}/scripts/formal/build_webshop_director_skills_v1.py)。结构、实际token预算、检索与内容边界见[验证报告]({DEST}/VALIDATION.zh-CN.md)和[机器可读记录]({DEST}/validation.json)。尚未运行新的128题推理测试，也没有训练模型；Director服务保持运行。
'''
    (DEST / "CATALOG.zh-CN.md").write_text(doc)
    print(json.dumps({"skills": len(cards), "source_trajectories": 384, "unique_source_tasks": 128, "snapshot": str(DEST / "snapshot.v2.json")}))


if __name__ == "__main__":
    main()
