# Director SkillBank

The current implementation uses eight packaged seed cards, a SQLite evidence store and immutable per-cycle snapshots. Skills are injected into the Director only. Existing rollout prompts and PPO probabilities are preserved.

## Configuration

Start from `configs/skillbank.example.toml`, configure Proposer/Solver and Worker services, and set `DEEPSEEK_SKILL_API_KEY` in your environment. The template requests `deepseek-flash` with thinking enabled, high reasoning effort and up to 20 concurrent distillation requests. Provider model availability is deployment-dependent. Install `.[selfplay,openai]` for real E5 retrieval and API use; mock tests need only `.[dev]`.

The original public `graph_features.embedding_model_path` setting is now `solver_skillbank.embedding_model_path`, and `runtime_routing.support` is `runtime_routing.skill_distiller`. Portable mock defaults keep the bank disabled. Enable it explicitly for training. Use an E5 model ID or an accessible local model directory.

## Lifecycle

1. Freeze a snapshot at the beginning of each cycle. Retrieve at most three compatible cards within a 1024-token full-card budget.
2. Collect credible low/high reward contrasts from raw same-task trajectories. Infrastructure failures and unknown scores are excluded from generation evidence.
3. Every 10 cycles, with at least 20 pending cases, start background distillation for up to 20 cases, at most two per source task. Parameter updates need not wait for generation.
4. Validate card structure, type and content before deduplication. Within the same task type, E5 similarity >=0.93 merges evidence; [0.90,0.93) remains pending review. Other task types are not automatically merged.
5. Under `activation_policy="checked"`, accepted cards become active for subsequent snapshots. Paired on/off trials are optional and are not automatically scheduled. Case consumption and publication are transactional; interrupted jobs can resume.
6. Record each injected skill version's usage once per rollout. For ordinary datasets, credible positive rewards count as helpful; credible zero with a valid nonempty answer counts as hurt. Unknown scores and infrastructure failures do not count as feedback. This is outcome association, not causal credit.
7. HealthBench retains its continuous reward: record valid count, reward sum and mean, without helpful/hurt votes or negative-score retirement contributions.
8. Every 10 cycles, non-seed active versions with at least three valid votes and negative net score become deprecated. Seed IDs, including revisions, are protected. Preserve evidence and versions; already frozen snapshots do not change.

Cycle artifacts include `skill_usage_summary.json` and, on maintenance cycles, `skill_usage_retirement.json`. No private banks, cases, trajectory logs or trial results are distributed.
