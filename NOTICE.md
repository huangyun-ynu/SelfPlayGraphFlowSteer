# Implementation provenance

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
