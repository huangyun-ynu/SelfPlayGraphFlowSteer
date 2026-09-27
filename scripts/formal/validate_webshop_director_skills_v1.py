"""Offline compatibility, token and retrieval audit; makes no Director/Worker calls."""
from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
import os
import re
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "configs/director_skills/webshop_trajectory_v1"
TOOLS = ("webshop_search", "webshop_click")
SCENARIOS = {
    "ws-public-constraints": "I am assigning a shopping task to an Agent. The customer permits either of two colors but requires a particular size and a price ceiling. Preserve all the original requirements and alternatives in the delegation.",
    "ws-session-owner": "A shopping browser session is held by one Agent. I want to add a helper to the graph without losing who controls that session. How should feedback and execution ownership be connected?",
    "ws-variant-capability": "Search returned a plausible product whose title mentions a different capacity and pack size. We have not inspected its configurable variants. Should the shopping Agent reject it yet?",
    "ws-live-options": "The shopping Agent left a product and reopened it. Its earlier selected size and color may have been cleared. Establish what configuration is actually active now.",
    "ws-targeted-evidence": "A shopping candidate claims to satisfy the request but the evidence checklist omits required compatibility. Determine whether this particular requirement is supported, contradicted or still unknown.",
    "ws-candidate-comparison": "We have several shopping candidates with different known features, unknown properties, prices and return costs. Compare them fairly before replacing the previously viable product with the latest one.",
    "ws-search-revision": "The shopping Agent keeps changing query wording but gets the same results and learns nothing about the missing requirement. Revise the unproductive search objective.",
    "ws-evidence-recovery": "Shopping memory was shortened. A visited flag survives but the earlier product evidence is missing, and another field discourages revisits. Decide what targeted information needs recovering.",
    "ws-completion-budget": "Only a few shopping actions remain. The current page is a detail section, required options are unselected, and purchasing needs more steps. Decide whether another inspection leaves enough budget to finish.",
    "ws-transaction-closure": "A shopping Agent reports a candidate and a staged purchase, while another Agent writes a review. Choose the graph output so the actual owner's transaction can be committed under the existing runtime protocol.",
    "ws-bounded-review": "Should an independent shopping reviewer be added for a specific contradiction? It can only inspect existing evidence. Define a narrow question and useful feedback without duplicating the owner's work.",
    "ws-local-revision": "New feedback identifies one missing shopping prerequisite after useful work. Modify only the necessary responsibility or graph dependency, preserving the owner, evidence and remaining budget.",
}


def main() -> None:
    # CPU-only E5 and tokenization; the resident GPU Director is untouched.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    from transformers import AutoTokenizer
    from selfplay_graph_flowsteer.skill_evolution_v2 import load_bank
    from selfplay_graph_flowsteer.skills import E5SkillEmbedder

    torch.set_num_threads(4)
    config = SimpleNamespace(
        pats=SimpleNamespace(enabled=False), skillbank_path=DEST / "snapshot.v2.json",
        skillbank_retrieve_top_k=3, skillbank_prompt_token_budget=1024,
        skillbank_retrieval_min_score=None,
    )
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "models/Qwen3.5-9B", local_files_only=True)
    embedder = E5SkillEmbedder(ROOT / "models/e5-base-v2")
    bank = load_bank(config, embedder=embedder)
    cards = list(bank.skills.values())
    assert len(cards) == 12 and len(cards) < 60
    quick = Path("/root/.codex/skills/.system/skill-creator/scripts/quick_validate.py")
    spec = importlib.util.spec_from_file_location("skill_quick_validate", quick)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    structure = {}
    for card in cards:
        card.validate()
        valid, message = module.validate_skill(DEST / "cards" / card.skill_id)
        structure[card.skill_id] = {"valid": valid, "message": message}
        assert valid, (card.skill_id, message)
        assert card.stats.usage_count == card.stats.helpful_count == card.stats.hurt_count == 0
        assert bank.records[card.skill_id]["provenance"] == "trajectory_derived_unvalidated"

    texts = [bank.format_prompt_context(cards)] + [
        (DEST / "cards" / c.skill_id / "SKILL.md").read_text() for c in cards
    ]
    # Heuristics complement manual review, and do not claim to detect every leak.
    forbidden = {
        "task_id": r"goal-\d+|\b\d{5}\b",
        "catalog_asin": r"\bB0[A-Z0-9]{8}\b",
        "hidden_rubric_fields": r"gold_answer|reference_answer|rubric_items|r_type|goal_options|target_asin",
        "credential_assignment": r"(?:api_key|access_token|password)\s*[:=]",
        "fixed_action_call": r"(?:webshop_click|webshop_search|click|search)\s*\(",
        "source_paths": r"state/experiments|raw_audit|provenance\.json",
    }
    boundary = {key: not any(re.search(pattern, text, re.I) for text in texts) for key, pattern in forbidden.items()}
    assert all(boundary.values()), boundary

    count = lambda selected: len(tokenizer.encode(bank.format_prompt_context(selected), add_special_tokens=False))
    individual = {c.skill_id: count([c]) for c in cards}
    triples = [count(combination) for combination in itertools.combinations(cards, 3)]
    assert max(triples) <= 1024, max(triples)

    scope_checks = []
    for task_type, tools in [("qa", TOOLS), ("household", TOOLS), ("shopping", ()), ("shopping", TOOLS[:1]), ("shopping", TOOLS[1:])]:
        selected, context, _ = bank.select_context("shopping product candidate purchase", task_type=task_type, tools=tools, tokenizer=tokenizer)
        assert not selected and not context
        scope_checks.append({"task_type": task_type, "tools": list(tools), "selected_count": 0})

    provenance = json.loads((DEST / "provenance.json").read_text())
    # The original task prompt is the production retrieval input. Read only that
    # field; rewards and hidden answers never enter embedding or selection.
    run = ROOT / provenance["runs"]["memory_source"]["path"]
    tasks = []
    for path in sorted((run / "trajectories").glob("*.json")):
        task = json.loads(path.read_text())["task"]
        tasks.append((task["task_id"], task["prompt"]))
    assert len(tasks) == 128
    queries = list(dict.fromkeys(list(SCENARIOS.values()) + [q for _, q in tasks]))
    vectors = dict(zip(queries, embedder.encode(queries, query=True), strict=True))

    class QueryCache:
        def encode(self, texts, *, query):
            assert query
            return [vectors[text] for text in texts]

    bank.embedder = QueryCache()  # Identical E5 vectors, batched once on CPU.
    scenarios = []
    for expected, query in SCENARIOS.items():
        selected, _, manifest = bank.select_context(query, task_type="shopping", tools=TOOLS, tokenizer=tokenizer)
        ids = [c.skill_id for c in selected]
        scenarios.append({"query": query, "expected": expected, "selected": ids, "expected_in_top3": expected in ids, "prompt_tokens": manifest["prompt_tokens"]})
    coverage = Counter({c.skill_id: 0 for c in cards})
    initial = []
    for task_id, query in tasks:
        selected, context, manifest = bank.select_context(query, task_type="shopping", tools=TOOLS, tokenizer=tokenizer)
        ids = [c.skill_id for c in selected]
        assert len(ids) == 3
        coverage.update(ids)
        initial.append({"task_id": task_id, "selected": ids, "prompt_tokens": manifest["prompt_tokens"], "context_sha256": hashlib.sha256(context.encode()).hexdigest()})

    similarities = []
    for left, right in itertools.combinations(cards, 2):
        cosine = sum(a * b for a, b in zip(bank._embeddings[left.skill_id], bank._embeddings[right.skill_id], strict=True))
        similarities.append({"left": left.skill_id, "right": right.skill_id, "cosine": cosine})
    similarities.sort(key=lambda row: -row["cosine"])
    frozen = ROOT / "state/experiments/webshop-memory-source-128-20260924-run1/snapshots/candidate/src/selfplay_graph_flowsteer"
    compatibility = {}
    for name in ["skills.py", "skill_evolution_v2.py", "adaptive.py"]:
        current = ROOT / "src/selfplay_graph_flowsteer" / name
        same = current.read_bytes() == (frozen / name).read_bytes()
        compatibility[name] = {"identical_to_frozen_candidate": same, "sha256": hashlib.sha256(current.read_bytes()).hexdigest()}
        assert same, ("Frozen loader compatibility needs a fresh review", name)
    result = {
        "validation_kind": "offline_structure_token_budget_eligibility_and_E5_retrieval_only",
        "snapshot_id": bank.snapshot_id,
        "snapshot_sha256": hashlib.sha256(config.skillbank_path.read_bytes()).hexdigest(),
        "director_inference_calls": 0, "worker_inference_calls": 0, "model_training": False,
        "skill_count": len(cards), "skill_format_checks": structure,
        "frozen_loader_compatibility": compatibility,
        "model_visible_heuristic_boundary_checks": boundary,
        "tokenizer": "models/Qwen3.5-9B", "individual_context_tokens": individual,
        "all_three_card_combinations": {"count": len(triples), "min_tokens": min(triples), "max_tokens": max(triples), "budget": 1024, "all_fit": True},
        "scope_checks": scope_checks, "embedder": "models/e5-base-v2", "embedding_device": "cpu",
        "synthetic_state_queries": {"purpose": "retrieval routing smoke only; not production phase-trigger validation or a benchmark", "expected_top3_hits": sum(r["expected_in_top3"] for r in scenarios), "total": len(scenarios), "cases": scenarios},
        "initial_128_task_retrieval": {"purpose": "observational coverage of original public requests, not skill-effect evaluation", "task_count": len(initial), "selection_counts": dict(coverage), "never_selected": [key for key, value in coverage.items() if not value], "cases": initial},
        "pairwise_embedding_similarity": {"purpose": "review aid only; shared domain can yield similar vectors without interchangeable behavior", "maximum": similarities[0], "pairs_at_least_0_90": [row for row in similarities if row["cosine"] >= .90]},
        "limitations": ["No new 128-task inference or causal improvement test was run.", "The source corpus repeats the same 128 development-used tasks across three runs.", "Current runtime retrieves once from the initial task; synthetic state queries are not automatically used later.", "No baseline config, model, worker route or inference effort was changed."],
    }
    (DEST / "validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    coverage_rows = "\n".join(f"|{card.name}|{coverage[card.skill_id]}|" for card in cards)
    (DEST / "VALIDATION.zh-CN.md").write_text(f'''# 离线验证记录

这份记录验证可加载性与检索行为，不是新的128题推理测试，没有准确率、F1或reward增益可报告。

- 原生 `load_bank` 成功加载全部12条；12份SKILL.md通过skill-creator格式检查。
- 与记忆来源修复版封存代码相比，skills.py、skill_evolution_v2.py、adaptive.py逐字节一致；未改动这些加载/检索代码。
- Qwen3.5-9B实际tokenizer下，单卡上下文{min(individual.values())}–{max(individual.values())} token；全部220种三卡组合为{min(triples)}–{max(triples)} token，均在1024预算内。
- 其他任务类型、缺少WebShop搜索或点击能力时均不选入。对模型可见正文做了题号、ASIN、隐藏评分字段和来源路径等规则扫描；规则扫描不能替代人工内容审查。
- E5在CPU上完成检索。12条合成状态描述的预期卡均出现在前3条，这仅验证检索区分度；现有运行时不会自动将这些状态描述用于后续检索。
- 全部66组卡片相似度中最高为{similarities[0]["cosine"]:.4f}，对应“具体证据核对”和“候选比较”。人工保留二者：前者核实单个事实缺口，后者在多个候选间作决策，职责不同。

## 原始128题公开请求的检索覆盖

每题从初始公开请求选3条，共384次选择。这里没有执行购物动作或调用Director/Worker模型。

|技能|被选入次数（128题）|
|---|---:|
{coverage_rows}

“搜索停滞修订”未被初始请求检索到；交易收尾、所有权等阶段性技能覆盖也较少。这是当前只在solve开始检索一次的限制，不能声称本包已经让12条技能在对应阶段自动生效。若后续增加按状态检索，应作为独立框架改动和效果实验记录。

技能来自三版共384条轨迹，但只有同一组128个任务，不是384个独立测试样本。没有将观察到的关联写成提分因果，也没有修改模型、推理强度、Worker路线或当前基线配置。使用过的128题属于开发数据，最终效果仍需独立样本验证。

复现命令（在项目根目录执行）：

```bash
.venv/bin/python scripts/formal/build_webshop_director_skills_v1.py
.venv/bin/python scripts/formal/validate_webshop_director_skills_v1.py
```

逐题选择、上下文哈希、token统计、技能包哈希和检查结果均保存于[validation.json]({DEST}/validation.json)。
''')
    print(json.dumps({"skills": len(cards), "triples_max_tokens": max(triples), "synthetic_top3_hits": result["synthetic_state_queries"]["expected_top3_hits"], "initial_coverage": dict(coverage), "max_similarity": similarities[0]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
