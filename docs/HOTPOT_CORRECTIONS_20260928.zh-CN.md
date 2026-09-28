# HotpotQA 14条数据问题修复（2026-09-28）

## 已生效的改动

用户在审计之后明确要求修复。本次将14条已确认问题应用为本地修订版 `flowsteer-hotpotqa-corrected-v1`，这不是HotpotQA官方勘误。

- 128题固定题目集合及顺序保留；14条内容修订，114条问题和答案保持一致。
- 13处题干修改、3处参考标签修改，交集为2条；`reference`和`target_answers`一致更新。
- 全部128条的段落正文、标题和顺序完全保留，原始`source_id`保留。
- 所有评测`id`加入 `flowsteer-corrected-v1`，并标记`metadata.dataset_version`，防止旧版任务ID触发结果复用。
- 职业集合题要求完整三项，只接受三项的6种排列及有/无末项and两种列举形式，共12个完整答案。使用已有多参考答案EM，评分代码没有改动；旧的两项子集不算严格EM正确。

## 文件与入口

| 用途 | 项目内路径 |
|---|---|
| 新版规范数据 | `data/formal/eval/hotpotqa_flowsteer_corrected_v1_128.jsonl` |
| 当前默认兼容入口，同新版逐字节一致 | `data/formal/eval/hotpotqa_official_test.jsonl` |
| 原始FlowSteer版本，保留供历史比较 | `data/formal/eval/hotpotqa_flowsteer_public_128.jsonl` |
| 原始raw文件，保留 | `data/formal/sources/flowsteer/hotpotqa_eval_128.raw.jsonl` |
| 逐项before/after、裁决理由和证据，只供离线准备使用 | `data/formal/private/hotpotqa_corrections_v1.json` |
| 当前版本/哈希 | `data/formal/eval/static_eval_split_manifest.json` |
| 离线构建统计 | `experiment_versions/reports/hotpot-corrections-20260928/build.json` |

已在Hotpot实验副本及当前主项目同步这些数据修复和准备脚本；主项目的模型、路由、运行时实现没有因本任务变更。现有默认读取 `hotpotqa_official_test.jsonl` 的入口会读取修订版。文件名中的official是历史兼容命名，此128题实际选自HotpotQA train划分。

`scripts/formal/run_qa_best_recorded_eval.sh`是明确的历史对照入口，已显式绑定保留的原始128题，防止历史对照悄悄换数据。`prepare_static_eval.py`重新准备数据时会从冻结原版重新应用修订，不会恢复有问题的旧标签。独立修复命令只写Hotpot相关文件，不重建其他数据集。

原128题的102/128、候选holdout16的14/16均保持原记录。修改题干后需要重新推理才有新版成绩，本次没有重算旧轨迹、调用模型或扩大样本测试。历史对比应使用原版数据，修订版未来单独报告。

## 逐条修订

下面列出实际生效的题干和标签。原问题和原答案完整保存在修订JSON中；审计时恢复的supporting_facts只用于证据核查，没有插入模型输入。

### 1. `5a8346bb55429966c78a6b69`

- 问题：The city of Telmessos was called Telebehi by what Iron Age people?
- 标签：`="Lycians">Lycians` → **Lycians**
- 依据：标签含 ="Lycians"> 这段HTML属性残片。题目问民族，证据指向Lycians。标准HotpotQA镜像也保留相同损坏标签。

### 2. `5ab9a694554299743d22eb8a`

- 问题：Which show featured on Tiny TV first aired in the United States on August 20, 2001?
- 标签：`Tiny TV` → **Oswald**
- 依据：题目问Tiny TV中的节目；Oswald段落第1句给出2001-08-20首播日期，Tiny TV只是包含该节目的节目时段。标签选错实体。原supporting_facts还漏标了Oswald的日期句。
- 裁决：改为Oswald，并明确问该节目的美国首播日期，避免错误暗示Tiny TV在该日期播出。

### 3. `5a8dd3935542995a26add3f2`

- 问题：What is the title of the fourth studio album by French DJ David Guetta that includes "When Love Takes Over", featuring vocals by Kelly Rowland?
- 标签保留：**One Love**
- 依据：题干以Which song提问，Kelly Rowland演唱的歌是When Love Takes Over；One Love是专辑名称，同名歌曲则是Estelle演唱。题干与答案对象类型冲突。
- 裁决：选择问专辑，保留One Love；题干明确album。歌曲题已把When Love Takes Over写入问题，直接把它作为gold会形成题干含答案。

### 4. `5a7e1f3f5542997cc2c47524`

- 问题：Which three occupations do Bernhard Stavenhagen and Franz Liszt share, according to the passages? List all three.
- 标签：`composer and conductor` → **pianist, composer and conductor**
- 依据：两段原文都列出pianist、composer、conductor；参考答案只列后两项。参考列出的职业是真的，但不是唯一充分答案。
- 裁决：明确要求完整三项。只接受三项完整集合的6种顺序，含末项and/无and两种列举形式；共12个完整答案，原两项子集不再接受。使用现有多参考答案严格EM，未改评分代码。

### 5. `5a808d785542996402f6a54e`

- 问题：For which award was the American actor who played Jim Dial on "Murphy Brown" and guest-starred in the 11th episode of the second season of "Family Guy" nominated for his performance on "Murphy Brown"?
- 标签保留：**Emmy Award**
- 依据：题目写award was given，原文写earned a nomination。Television Academy人物页也列1次提名。Emmy Award这个奖项名本身并非错误，授奖前提错误。

### 6. `5a7a7b5155429941d65f2672`

- 问题：When was the performer of "Lil Ghetto Boy", the song sampled in "Things Done Changed", born?
- 标签保留：**February 18, 1965**
- 依据：When was the performer ...缺少born等事件谓语。标签是Dr. Dre出生日期；Things Done Changed明确提到其Lil Ghetto Boy。Donny Hathaway的Little Ghetto Boy是另一首歌，原预测还混入生卒范围。

### 7. `5ac2f46a5542996773102655`

- 问题：According to the passages, which Indian political alliance represented in the 2012 Indian vice-presidential election had Narendra Modi as its leader?
- 标签保留：**National Democratic Alliance**
- 依据：题目问political party，gold NDA在证据中明确是coalition of political parties。BJP是领导联盟的党，两者相关但不是别名。

### 8. `5ae1fec25542997283cd231c`

- 问题：The aircraft involved in the 1950 British Columbia B-36 crash had departed from an Air Force base just southeast of which Alaska town?
- 标签保留：**Moose Creek**
- 依据：问题写en route to阿拉斯加基地；原文是from Eielson(Alaska) to Carswell(Texas)。gold Moose Creek定位的是出发基地。原预测Fairbanks也漏读just southeast对应Moose Creek的限定关系。

### 9. `5a88ffd95542995153361245`

- 问题：Who wrote the screenplay for the 2010 film directed by Paul Greengrass mentioned in the passages?
- 标签保留：**Brian Helgeland**
- 依据：把United 93(2006)的奥斯卡导演提名挂到Green Zone(2010)。Brian Helgeland确实是后者编剧，但前提不成立；此题本轮EM为1。

### 10. `5a76abf65542993569682c7d`

- 问题：After which Scottish mathematician, physicist and astronomer is the Napierian logarithm named?
- 标签保留：**John Napier**
- 依据：Napierian logarithm段落明确写Napier did not introduce this natural logarithmic function；问题却说credited for introducing。按所给原文，题干反转了否定关系。模型复现John Napier获得EM=1。

### 11. `5a80b3a65542992bc0c4a7ba`

- 问题：According to the passages, what city is a tourist destination and is near the site of a battle where combined Austrian losses and French casualties exceeded 10,000?
- 标签保留：**Verona**
- 依据：证据提供奥军losses超过5500、法军至少5000 casualties；题目把合计损失/伤亡改成over 10,000 soldiers died。伤亡不能直接当死亡人数。Verona这个地点仍能匹配，故原EM=1。

### 12. `5abaf81f55429939ce03dd7c`

- 问题：On what date did the American politician who served as Lieutenant Governor of Oklahoma and died in the city that is home to the main campus of Oklahoma State University die?
- 标签保留：**November 22, 1966**
- 依据：James E. Berry原文职务为Lieutenant Governor，题干写Governor。日期标签是该副州长的死亡日期，原EM=1。

### 13. `5ae3642d5542991a06ce99b4`

- 问题：What name is shared by the venue in Prestatyn, Wales that hosted Pro Challenge Series – Event 2 and a company operating holiday parks in the UK?
- 标签保留：**Pontins**
- 依据：来源说比赛在Pontins in Prestatyn举办，Pontins是经营度假园区的公司名称。题干将Pontins表述成a city in Prestatyn，证据不支持这种实体类型。

### 14. `5a83458755429966c78a6b66`

- 问题：What plant growth form do Lodoicea and Nothofagus have in common?
- 标签保留：**trees**
- 依据：问题说两个genus的mutual species，标签却是生长形态trees。tree不是生物学species。模型又把任务改为比较family并回答无共同类别，两边都偏离清楚的提问。
- 裁决：改问生长形态，保留trees。原材料中的palm需要基本植物常识；外部核验只用于裁决，不添加到模型输入。

植物生长形态另经 [Kew Plants of the World Online](https://powo.science.kew.org/taxon/668084-1) 核对：Lodoicea唯一物种L. maldivica被列为tree。该外部核验只进入离线裁决记录；题目材料仍需结合palm这一描述理解生长形态。

此前的原始审计结果保留在 `experiment_versions/reports/hotpot-label-audit-20260928/`，其中`auto_apply=false`描述的是修复前审计快照。当前生效状态以本修订JSON和构建清单为准。

## 保留待核的范围

- `5a7780a655429949eeb29e9b`的Little Mix/EP参与关系仍缺证据，保留原题并继续标为待核，未计入已修复14条。
- 此次没有按旧模型输出追补其他题的别名、删除失分题，或更改独立holdout16/32。
- 训练train512和实际默认训练池此前与14条加1条待核题交集为0；本次复核仍为0，因此这些修复只影响这份评测集。不要把审计过的评测题回流训练。

## 离线复现与检查

在项目根目录使用项目Python环境运行：

```bash
python scripts/formal/hotpot_corrections.py
python scripts/formal/hotpot_corrections.py --check
python -m pytest tests/test_hotpot_corrections.py tests/test_prepare_static_eval.py -q
```

构建器校验原版和raw的SHA256、128个唯一来源ID、恰好14条修订及每条before字段；遇到来源漂移、修订前提不符或不认识的现有数据文件会在写文件之前报错。输出按文件原子替换，重复运行生成相同数据。需要修订未来版本时应建立v2，不覆盖v1。

测试覆盖14条范围、114条内容保留、128条段落保留、证据来自原始上下文、来源ID及版本ID、加载器/严格EM接受完整职业集合且拒绝子集、失效输入零写入、幂等构建、静态准备脚本保持新版、其他数据集清单不变。

验证结果：实验副本与主项目均通过16项针对性离线测试；两处`--check`通过，主项目原有未提交改动、原版数据、训练文件及历史结果文件哈希保持不变。未进行模型推理，新增模型API调用为0。可选ruff检查在现有环境不可用（未安装），未安装新依赖。

## 数据版本哈希

- 原版128：`84db56807e8e8cd98224ba53b124b289f783faf463a55f5a7e6785b2127cbcc2`
- 修订版128：`6e4096785ad5b869b6541c868beb25ad7f157cc64ae8968c0db3cbe2f185e27c`
- 原始raw：`55a74378c02a8fc3d40fa6bbcd967447f10b6021d3a39c707c14e3b5d265f25a`
- 修订规则：`e80de98401ec9aebcccfac32efb2b80961d26ae80362244c96fbaa4695dec045`

来源：[FlowSteer公开副本](https://huggingface.co/datasets/beita6969/FlowSteer-Dataset/blob/main/eval/hotpotqa.jsonl)、[HotpotQA固定revision](https://huggingface.co/datasets/hotpotqa/hotpot_qa/tree/1908d6afbbead072334abe2965f91bd2709910ab)。沿用上游数据的CC BY-SA 4.0许可；新增修订是本项目本地裁决。
