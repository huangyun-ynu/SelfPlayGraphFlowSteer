# Implementation provenance

MBPP+ OOD public execution adapts FlowSteer `src/code_execution.py`, commit
`1c9f2abf55cb9b8ea2ca2e3359cdb91acb9964e9`. The exact upstream file hash,
local diff and compatibility changes are recorded under
`src/mbppplus_ood/_vendor/`; no upstream license file was found in this snapshot.
The OOD graph, budgets and submission protocol use a frozen snapshot of this
project's SWE framework, recorded in `ood/mbppplus/variant.manifest.json`.
Final private-test scoring calls EvalPlus 0.3.1 under its existing installation.

WebShop author V2 reward and its normalizer are vendored unchanged from Princeton's
`princeton-nlp/WebShop` v2 commit `efc76f6474edd7f888b1b170f4f8ec5c7ab3e4da`.
Their academic/research license is retained at
`src/selfplay_graph_flowsteer/_vendor/webshop_v2/LICENSE.md`; per-file hashes and
upstream paths are in its adjacent `provenance.json`.

This project builds on the following designs and implementation patterns:

- **FlowSteer:** progressive canvas editing, one Director action per turn, factual graph feedback, action-masked token probability training and task-group reward normalization.
- **SESA:** separate Proposer/Solver policies, proposal/collection ordering and independent snapshots. Director skill persistence, retrieval, consolidation and distillation are included.
- **MANTA (MIT):** structured agent artifacts, bounded relay packets, information visibility and bounded peer revision. See `third_party/notices/MANTA-LICENSE`.
- **SkillFlow:** Qwen3.5 compatibility patterns, chat-template tokenization, reasoning-content handling and architecture-aware model loading.
- **MACE:** relational features, LinUCB equations and reward blending implemented from the published method.
- **ADS (Apache-2.0):** mini-cluster curriculum state, boundary movement, clustering and difficulty preprocessing adapted to JSONL pools. See `third_party/notices/ADS-LICENSE` and `ADS-NOTICE`.
- **TSDS (MIT):** KNN-KDE probability assignment, adapted to an exact NumPy nearest-neighbor implementation. See `third_party/notices/TSDS-LICENSE`.
- **PATS (Apache-2.0):** policy-aware training scaffolding, grouped rollout evidence, success-rate EMA, pressure-based review modes and bounded skill edits are adapted from [shi-yipeng/PATS](https://github.com/shi-yipeng/PATS), revision `bad468b5c73081c2f5aa74c4e0011c6fb2872dbf`. The adaptation targets Director-only support, the existing cycle snapshots and evidence store, with project-specific evidence checks and edit validation; it does not vendor the upstream training stack or SkillRL seed artifacts. See `docs/PATS.md` and the unchanged upstream license in `third_party/notices/PATS-LICENSE`.

Graph constraints, relation counterfactual integration, checkpoint orchestration, dataset integration and runtime scheduling include project-specific code. Benchmark performance must be established separately; mock tests are not evidence of model quality.

The supplied local snapshots of FlowSteer, SESA and SkillFlow contain no discoverable LICENSE/COPYING file. This record is attribution, not permission to redistribute their adapted code. No repository-wide license grant is asserted by this release preparation. Upstream terms and ownership of project-specific contributions must be established before assigning such a license.
