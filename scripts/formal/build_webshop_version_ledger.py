"""Build the WebShop-only version ledger from preserved metrics and trajectories.

No inference, environment mutation, or modification of sealed experiment artifacts.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SKILLFLOW = ROOT.parent / "SkillFlow"
OUT = ROOT / "experiment_versions/reports"
DOC = ROOT / "docs/WEBSHOP_EXPERIMENT_LEDGER_2026-09-25.zh-CN.md"
CATALOG = ROOT / "experiment_versions/run_catalog.json"
DATE = "2026-09-25"


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_trace_metrics(directory, *, native=False):
    paths = sorted(Path(directory).glob("*/result.json" if native else "trajectories/*.json"))
    rows = []
    for path in paths:
        t = read(path)
        rows.append((t["task_id"], bool(t["passed"]), float(t["reward"])) if native else
                    (t["task"]["task_id"], bool(t["verification"]["passed"]), float(t["verification"]["score"])))
    assert rows and len({r[0] for r in rows}) == len(rows), directory
    return {"n": len(rows), "correct": sum(r[1] for r in rows),
            "mean_reward": sum(r[2] for r in rows) / len(rows),
            "task_set_sha256": hashlib.sha256("\n".join(sorted(r[0] for r in rows)).encode()).hexdigest()}


def main():
    catalog = read(CATALOG)
    records = []

    def add(id, name, group, parent, changes, guidance, skill, n, correct, reward,
            source, selector, *, run=None, planned=None, status="complete", notes="", trace=None, native=False):
        source = Path(source)
        row = {"id": id, "name": name, "group": group, "parent_or_reference": parent,
               "changes": changes, "worker_guidance": guidance, "director_skill": skill,
               "status": status, "planned": planned if planned is not None else n,
               "completed": n, "correct": correct,
               "em": correct / n if n and correct is not None else None,
               "mean_reward": reward, "mean_score_100": reward * 100 if reward is not None else None,
               "source": str(source), "source_selector": selector, "source_sha256": digest(source),
               "original_run": str(run) if run else None, "notes": notes,
               "verification": "preserved_metric_archive"}
        if trace:
            actual = checked_trace_metrics(trace, native=native)
            assert actual["n"] == n and actual["correct"] == correct, (id, actual, row)
            assert math.isclose(actual["mean_reward"], reward, abs_tol=1e-10), (id, actual, row)
            row["verification"] = "metrics_cross_checked_against_all_completed_trajectories"
            row["task_set_sha256"] = actual["task_set_sha256"]
        records.append(row)
        return row

    historical = [
        ("W01", "fixed", "qwen35-9b-director-webshop-fixed-v1", "目标映射修复", "修正任务与环境目标映射；实际 GPT-5.5 Worker；Director thinking 关；结构化观察4000字符", "无LASER；原公共指令", "static Director skill"),
        ("W02", "no-skill", "qwen35-9b-director-webshop-no-skill-v1", "W01", "移除静态 Director skill，保留 GPT-5.5 与结构化观察", "无LASER；原公共指令", "无"),
        ("W03", "full-index", "webshop-full-index-deepseek-no-skill-v1", "W02", "换完整商品索引和 DeepSeek Worker；双方 thinking 关，含多项同时变化", "无LASER；原公共指令", "无"),
        ("W04", "retain_page_text", "webshop-deepseek-reasoning-noskill-c24-20260918-170329", "W03", "保留原页面文本、lifecycle字符上限设0；Director/Worker thinking 开；另含中间工程修复", "无LASER；原公共指令", "无"),
        ("W05", "legacy", "webshop-legacy-page-only-reasoning-c24-20260918-172320", "W04", "搜索观察改为 legacy，保留内部结构化状态和已选规格", "无LASER；原公共指令", "无"),
        ("W06", "env-feedback", "webshop-env-feedback-c24-20260923", "W05", "增加重复查询、离开商品页面等环境反馈", "无LASER；环境反馈开", "无"),
        ("W07", "LASER checklist", "webshop-laser-checklist-c24-20260923-182224", "W05（不是在W06上累计）", "在 legacy 上加入按页面核对需求、价格、属性、规格的 LASER；环境反馈关闭", "公共指令＋独立LASER", "无"),
        ("W08", "budget-off（历史）", "webshop-budget-off-laser-c24-20260923-191922", "W07", "关闭请求token准入、执行credit与closure预留；累计token和16次动作上限仍保留", "公共指令＋独立LASER，仍开启", "无"),
        ("W09", "identity-fix", "webshop-identity-fix-c24-20260923-202622", "W08", "恢复内部ASIN，修正已访问商品及详情证据的身份判断", "公共指令＋独立LASER，仍开启", "无"),
        ("W10", "detail-unlimited", "webshop-detail-unlimited-c24-20260923-205311", "W09", "取消单段详情1400字符截断；并未取消全部商品缓存和输入裁剪上限", "公共指令＋独立LASER，仍开启", "无"),
        ("W11", "SkillFlow history", "webshop-skillflow-history-c24-20260923-213558", "W10", "完整私有Observation/Action历史，移除重复购物记忆提示；仍走原图工具协议", "LASER仍开启；环境反馈关", "无"),
    ]

    def from_catalog(id, name, suffix, group, parent, changes, guidance, skill, notes=""):
        matches = [r for r in catalog["runs"] if r["run"].endswith(suffix)]
        assert len(matches) == 1, suffix
        r = matches[0]; m = r["metrics"]["webshop"]
        n = m["examples"]; correct = round(n * m["pass_rate"])
        assert math.isclose(correct/n, m["pass_rate"], abs_tol=1e-10)
        return add(id, name, group, parent, changes, guidance, skill, n, correct, m["mean_score"],
                   CATALOG, f"runs[run={r['run']}].metrics.webshop", run=ROOT/r["run"],
                   notes=notes or "迁移前原运行目录当前不可用；依据保留的指标索引及对应历史实验报告。")

    for id, name, suffix, parent, changes, guidance, skill in historical:
        from_catalog(id, name, suffix, "full128", parent, changes, guidance, skill)

    recent = [
        ("W08-R1", "W08原版重新对照", "webshop-w08-merged-128-20260924-run1", "baseline", "W08", "用W08原版重新跑相同128题；这是新一轮采样，不能覆盖历史63/128", "公共指令＋独立LASER", "无"),
        ("M01", "W08提示合并", "webshop-w08-merged-128-20260924-run1", "candidate", "W08-R1（同期对照）", "合并公共Worker指令与LASER，去重并明确核对、探索、购买、预算优先级；Director未改", "merged_checklist_v1；保留清单内容，取消独立重复注入", "无"),
        ("M02", "合并＋身份修复", "webshop-merged-identity-128-20260924-run1", "candidate", "M01", "在合并版移植W09内部ASIN与访问/详情身份修复；不是直接继承W10/W11", "合并清单", "无"),
        ("M03", "合并＋身份＋记忆来源修复", "webshop-memory-source-128-20260924-run1", "candidate", "M02", "访问事实改读裁剪前记录、实时动作标注和匹配商品checkpoint，区分已读未保留；may_add价值推断尚在", "合并清单", "无"),
        ("S01", "Director Skill v1", "webshop-director-skills-128-20260924-run1", "candidate", "M03", "只增加12条Director编排Skill，初始检索top-3、1024 token；Worker及底座源码不变", "合并清单", "12条；每题初始top-3"),
    ]
    for id, name, dirname, arm, parent, changes, guidance, skill in recent:
        run = ROOT/"state/experiments"/dirname
        source = run/"comparison/comparison.json"; m = read(source)[arm]["metrics"]
        add(id, name, "full128", parent, changes, guidance, skill,
            m["examples"], m["strict_successes"], m["mean_reward"], source, f"{arm}.metrics",
            run=run/arm, trace=run/arm,
            notes="仅W08-R1与M01是同期各自重跑；M02/M03/S01的对照复用前轮封存结果。")

    sf = SKILLFLOW/"state/experiments/webshop-director-128-20260924-run1"
    source = sf/"comparison/comparison.json"; m = read(source)["metrics"]
    add("F01", "SkillFlow内部接入Director", "full128", "M03（历史对照，独立架构分支）",
        "直接修改SkillFlow，以reset_react/react_step执行购物，接入项目Director/Canvas；新执行器、独立会话、FINISH提交；未接旧收尾",
        "SkillFlow原始模板＋图职责/报告接口；无旧LASER/合并清单", "无",
        m["completed"], m["strict_successes"], m["mean_reward_all128_errors_as_zero"], source, "metrics",
        run=sf, trace=sf/"candidate", native=True,
        notes="存在自动提前FINISH、候选丢失、报告状态与选项观测缺陷，尚未修复。并发10，历史合并轮并发24；不是单一提示消融。")

    from_catalog("N00", "Native第一轮", "webshop-skillflow-native10-20260923-223240", "dev10", "W11",
                 "项目内适配SkillFlow原生search/click模板，每Agent独立会话、私有完整历史，FINISH提交；后续四项工程修复尚未包含",
                 "SkillFlow原始模板；无旧LASER/购物清单", "无")
    dev = [
        ("N01", "Native before", "webshop-native-iteration-20260924", "before", "N00后工程修复", "修复跨数据集native误用、未闭合think解析、清理中断、提交回写污染；重新跑10题", "原生模板；无旧LASER"),
        ("N02", "Native after", "webshop-native-iteration-20260924", "after", "N01", "增加未购买/候选/共享额度事实反馈，并加入完成购买及职责修订提示", "原生模板＋新增购物策略提示；后来撤回"),
        ("N03", "Native after2", "webshop-native-iteration-20260924", "after2", "N02", "再强调标题规格不等于已选择规格、缺属性证据需看公开详情；历史5/10版本", "原生模板＋规格/详情策略提示；后来撤回"),
        ("N04", "Native m1", "webshop-native-target7-20260924", "m1", "撤回N02/N03策略提示后", "保留纯事实候选/预算反馈，给当前观察及历史补充真实selected_options", "恢复原始策略模板；无旧LASER、无新增策略提示"),
        ("N05", "Native m2-valid", "webshop-native-target7-20260924", "m2-valid", "N04", "适配官方text_rich渲染，保留按钮、选中/访问标记，修正相应动作映射", "同N04；富文本观察"),
        ("N06", "Native m3", "webshop-native-target7-20260924", "m3", "N04机制线（不是累计启用富文本）", "FINISH前对无候选的选中输出Agent增加至多一次同会话收尾，使用原修订额度", "同N04；输出阶段事实字段"),
        ("N07", "Native m4", "webshop-native-target7-20260924", "m4", "N06", "增加同Agent私有多轮对话历史；保持m3收尾机制，模型私有思维不进入历史", "同N04；多轮消息组织"),
    ]
    for id, name, dirname, key, parent, changes, guidance in dev:
        run = ROOT/"state/experiments"/dirname
        source = run/"comparison.json"; m = read(source)[key]
        add(id, name, "dev10", parent, changes, guidance, "无", m["completed"], m["passed"],
            m["mean_score_100"]/100, source, key, run=run/key, trace=run/key,
            notes="同一开发用10题，不能与完整128题直接排名；N02/N03提示后来按用户约束撤回。")

    partial = ROOT/"state/experiments/webshop-native-after2-128-20260924-run1"
    source = partial/"comparison/partial_summary.json"; m = read(source)
    add("N03-P128", "after2扩展128（提前停止）", "partial", "N03",
        "按用户要求恢复历史5/10冻结版本开展128题；低准确率后用户叫停", "after2策略提示；无旧LASER", "无",
        m["completed"], m["successes"], m["mean_reward_on_completed"], source, "root",
        planned=m["planned"], status="stopped_by_user", run=partial, trace=partial/"candidate",
        notes="17/47仅是已完成部分EM；81题未完成，没有完整128题EM或平均分。")

    targeted = [
        ("T01", "closure original", "webshop-closure-ablation-deepseek-v1/original", "原购买提示；历史失败30题"),
        ("T02", "closure removed", "webshop-closure-ablation-deepseek-v1/removed", "在同一30题删除购买收尾提示"),
        ("T03", "closure best_so_far", "webshop-closure-ablation-deepseek-v1/best_so_far", "在同一30题提示预算内购买已见最佳候选"),
        ("T04", "option-fixes", "webshop-option-fixes-deepseek-no-skill-v1", "规格和动作列表修复；21题中20题有后端失败，保留完整故障结果"),
        ("T05", "option-fixes retry", "webshop-option-fixes-deepseek-no-skill-retry-v1", "重试其中20题；仍有1题含后端失败，不能只报告重试成绩"),
        ("T06", "budget-aware", "webshop-budget-aware-deepseek-no-skill-v1", "27题预算与购买收尾策略诊断"),
        ("T07", "price-range", "webshop-price-range-deepseek-goal234-v1", "单题价格区间排障"),
    ]
    for id, name, suffix, changes in targeted:
        from_catalog(id, name, suffix, "targeted", "定向子集", changes,
                     "见对应历史实验；不据名称推断LASER", "见历史档案",
                     notes="迁移前指标档案；定向子集及后端故障会影响结果，不与完整128题排名。")

    excluded = [
        ("X01", "r3", "旧提示方案的后续尝试；0题完成、3次连接失败", ROOT/"state/experiments/webshop-native-target7-20260924/r3/aggregate_summary.json"),
        ("X02", "m2", "富文本初版适配器缺少同目录导入路径，冒烟失败；没有有效完整结果", ROOT/"docs/WEBSHOP_NATIVE_MECHANISM_EXPERIMENT_2026-09-24.zh-CN.md"),
        ("X03", "m2r", "服务端口未释放导致环境连接失败；0题完成、3次失败", ROOT/"state/experiments/webshop-native-target7-20260924/m2r/aggregate_summary.json"),
        ("X04", "m5 high", "改用high违反公平对照要求，已撤回；部分轨迹保留但不作为框架收益或有效成绩", ROOT/"state/experiments/webshop-native-target7-20260924/m5_WITHDRAWN.json"),
        ("X05", "W08复测首次归档中断轮", "模型HTTP正文记录不完整，整轮排除；修复记录器后两组重新开始", ROOT/"state/experiments/webshop-w08-merged-128-20260924-run1/setup-attempt-incomplete-http-audit/reason.json"),
        ("X06", "早期错误目标映射", "prompt与环境目标映射无效，原WebShop成绩废弃，不能纳入有效版本排名", ROOT/"docs/EXPERIMENT_RECORDS.zh-CN.md"),
        ("P01", "09-25旧版/新版修复方案", "已完成工程诊断及修复设计；预算信息、访问价值语义、预算转交、FINISH保护尚未实施和评测", ROOT/"state/audits/webshop-old-repair-plan-20260925/evidence.json"),
    ]
    for id, name, changes, source in excluded:
        add(id, name, "not_evaluated" if id == "P01" else "excluded", "—", changes,
            "未作为有效新版本评测", "—", None, None, None, source, "root",
            status="proposed_not_implemented" if id == "P01" else "excluded_from_comparison")

    full = [r for r in records if r["group"] == "full128"]
    selected = read(ROOT / "configs/webshop_official_baseline.json")
    for row in records:
        row["selected_for_formal_training"] = row["id"] == selected.get("selected_version")
    assert len(full) == 17 and all(r["completed"] == 128 for r in full)
    recent_hashes = {r["task_set_sha256"] for r in full if "task_set_sha256" in r}
    assert len(recent_hashes) == 1
    result = {"schema": "webshop-version-ledger-v1", "as_of": DATE,
              "metrics": {"em": "严格成功比例（官方reward=1；不是文本字符串EM）",
                          "mean_reward": "官方reward均值，0到1", "mean_score_100": "mean_reward乘100",
                          "partial_denominator": "提前停止轮仅对已完成样本报告；没有补算完整128题",
                          "f1": "官方WebShop未定义"},
              "source_boundary": "W01-W11及早期子集按保留指标档案核对；近期6组完整128、7组开发10题、47题中止轮与逐题轨迹交叉核对。",
              "all_recent_full128_task_sets_equal": True, "model_calls": 0, "records": records,
              "formal_training_selection": {
                  "version": selected.get("selected_version"),
                  "source": str(ROOT / "configs/webshop_official_baseline.json"),
                  "source_sha256": digest(ROOT / "configs/webshop_official_baseline.json"),
                  "new_training_or_evaluation_result": False,
              }}
    OUT.mkdir(parents=True, exist_ok=True)
    output_json = OUT/"webshop-version-ledger-20260925.json"
    output_csv = OUT/"webshop-version-ledger-20260925.csv"
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    fields = ["id", "name", "group", "parent_or_reference", "changes", "worker_guidance", "director_skill",
              "status", "selected_for_formal_training", "planned", "completed", "correct", "em", "mean_reward", "mean_score_100", "verification", "source", "notes"]
    with output_csv.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); w.writeheader(); w.writerows(records)

    def table(group):
        lines = ["|版本|基于/对照|主要改动|LASER/Worker提示；Director Skill|正确数 / EM|平均分 /100|", "|---|---|---|---|---:|---:|"]
        for r in records:
            if r["group"] != group: continue
            label = f"[{r['id']} {r['name']}]({r['source']})"
            score = "—" if r["mean_score_100"] is None else f"{r['mean_score_100']:.4f}"
            em = "—" if r["em"] is None else f"{r['correct']}/{r['completed']}，{r['em']*100:.4f}%"
            lines.append(f"|{label}|{r['parent_or_reference']}|{r['changes']}|{r['worker_guidance']}；Skill：{r['director_skill']}|{em}|{score}|")
        return "\n".join(lines)

    document = f"""# WebShop 实验版本总账（更新至 {DATE}）

当前正式训练采用 **{selected.get('selected_version')}**。历史成绩仍属于原评测配置；正式训练保留Director选逻辑模型、程序选接口及Qwen thinking，不等于固定DeepSeek实验复现，也没有新增训练成绩。见[当前正式版本]({ROOT}/docs/WEBSHOP_BASELINE.zh-CN.md)。

本次更新整理17组完整128题结果、8组完整10题开发实验、1组提前停止的128计划，以及定向子集、撤回/失败尝试和未实施方案。仅更新记录，没有运行新推理或训练。W01–W11沿用原记录编号；其他编号是本总账索引，不是发布版本号，也不表示按表格逐行继承。

EM统一指**完全正确数÷样本数**（官方reward=1），不是文本字符串匹配。平均分为**官方reward均值×100**，保留部分得分；JSON/CSV同时保留0–1原始均值。官方WebShop没有F1。

## 1. 完整128题结果

{table('full128')}

原始合并M01的EM最高（66/128，51.5625%）；合并＋身份修复M02的平均分最高（73.5221/100）。这是已有单轮结果，不表示已证明稳定优于其他版本。S01使用同一128题轨迹开发Skill，不是独立泛化评测。

W08历史63/128和W08-R1重新对照58/128必须保留为不同运行。M01的同期对照是W08-R1；M02/M03/S01引用前轮封存对照。F01是直接在SkillFlow内部改造执行流程，与项目内Native适配是不同分支。W01–W04还同时改变模型、索引、thinking或页面观察，不能把差值归因于单项框架改动。

W08关闭的是请求token预算拦截，**没有关闭LASER**。W09、W10、W11也保留LASER。M01–M03、S01将LASER内容合并进公共Worker指令，取消的是重复的独立注入。N系列及F01走原生模板，不使用旧LASER/合并清单；N02/N03另有新增购物策略提示，不能称为无策略提示版。

## 2. Native完整10题开发实验

{table('dev10')}

N00与N01都是1/10，但前者平均37.1667、后者63.0000，且N01包含四项工程修复和新的运行配置，不能合成同一轮。N03的5/10是历史提示版；其策略提示随后被撤回，后来用户指定用该冻结版扩展128题，结果单列如下。历史W11在匹配的这10题为4/10、平均70.8333，仅是W11结果切片，不是新版本或新推理轮。

## 3. 提前停止的128题计划

{table('partial')}

N03-P128只完成47/128，剩81题未完成；EM=17/47=36.1702%，平均55.5851/100均只针对已完成部分。不能写成17/128的完整EM，也不能与完整128题直接排名。原始请求、在途请求、已完成轨迹和停止记录保留。

## 4. 历史定向子集

{table('targeted')}

上述分母各异，尤其closure选的是历史失败题，option-fixes包含大量后端故障。不能只保留重试中的较好结果。其他早期启动/单题smoke记录继续保留在跨数据集历史索引中，不作为新的完整128题版本。

## 5. 无有效新成绩的尝试与方案

{table('excluded')}

{table('not_evaluated')}

破折号表示没有可用于本次版本对照的成绩，不是0%或0分。m5即便留有部分轨迹，也因high推理强度不公平而排除。预算信息和访问价值字段等修复仍是方案，没有新增“修复后EM”。

## 6. 证据边界与复核

- 迁移前W01–W11、N00和早期定向子集读取保留的[指标档案]({CATALOG})及历史实验说明。原运行目录在当前服务器不可直接读取，因此本次没有声称重新审计这些原始轨迹。
- W08-R1、M01、M02、M03、S01、F01共6组完整128题已逐题重算正确数和reward均值，与各自汇总一致；题目ID集合相同。N01–N07共7组10题及中止轮47题也完成同样核对。
- 模型、low推理强度、动作及token预算的公平约束继续保留；F01并发与页面/执行协议不同，属于架构比较。固定配置不保证服务输出逐步一致，不能从单轮成绩作确定因果归因。
- F01存在已确认的自动提前结束、候选取消、报告错误状态和选项观测缺陷。见[诊断]({SKILLFLOW}/state/analyses/webshop-director-128-20260925-diagnosis/diagnosis.zh-CN.md)和[新旧保护核对]({SKILLFLOW}/state/analyses/webshop-merged-protections-20260925/report.zh-CN.md)。旧合并版额外核对见[预算/状态证据]({ROOT}/state/audits/webshop-old-repair-plan-20260925/evidence.json)。分析不构成新评测成绩。
- [机器可读JSON]({output_json})、[CSV]({output_csv})、[重建脚本]({Path(__file__).resolve()})保留来源选择器、指标精度与来源文件SHA256；脚本仅读取封存产物并写总账。
"""
    DOC.write_text(document)
    print(json.dumps({"records": len(records), "full128": len(full), "doc": str(DOC),
                      "json": str(output_json), "csv": str(output_csv)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
