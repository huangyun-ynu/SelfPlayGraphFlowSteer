# PATS-inspired Director training scaffolding

This integration adapts PATS's training-time scaffold maintenance to the
Director SkillBank. The component summarizes credible rollout groups and
selects bounded skill edits between collection cycles. It is not a separate
trainable policy, a new reward, or a replacement for Proposer task selection.

The source is [PATS: Policy-Aware Training Scaffolding for Agentic Reinforcement
Learning](https://arxiv.org/abs/2607.21419) and the official
[shi-yipeng/PATS repository](https://github.com/shi-yipeng/PATS), inspected at
commit `bad468b5c73081c2f5aa74c4e0011c6fb2872dbf`. The relevant upstream files are
`src/stage1_memory_modes.py`, `src/experience_bank.py`, `src/evidence_card.py`
and `src/group_reviewer.py`. The Apache-2.0 license is retained in
[`third_party/notices/PATS-LICENSE`](../third_party/notices/PATS-LICENSE).
The upstream training stack and its SkillRL seed JSON files are not vendored.

## Configuration and use

Start from [`configs/pats.example.toml`](../configs/pats.example.toml), configure
the existing policy, Worker and skill-distiller services, and choose state
paths for this experiment. The additional settings are:

```toml
[solver_skillbank]
enabled = true
mode = "director_skill_v2"
usage = "training_only"

[solver_skillbank.pats]
enabled = true
ema_alpha = 0.1
revise_threshold = 0.3
compress_threshold = 0.85
max_skills = 30
max_tokens = 2000
min_groups = 2
min_evidence_cards = 2
review_interval = 1
max_edits = 4
max_reviews_per_cycle = 2
max_evidence_groups = 32
max_policy_lag = 1
max_review_input_tokens = 20480
```

The section above is an excerpt, not a complete model/runtime configuration.
PATS capacity limits apply to the entire scoped view. Existing
`solver_skillbank.retrieve_top_k` and `prompt_token_budget` still bound the
subset actually injected into each Director prompt; the example retains
three cards and a 1,024-token prompt budget.
New PATS snapshots persist `selection_revision = "learned_first_v1"`.
Within the selected scope, cards maintained by PATS (including revised seed
cards) rank before inherited seed cards; E5 orders candidates within each
group. Task-type, tool, exclusion and configured score filters still apply.
Selection skips cards that do not fit the actual token budget and continues
through eligible candidates until at most three cards fit. This prevents
question-text similarity alone from keeping process guidance out of training.
Snapshots without a revision retain the original `e5_only_v1` ordering and
manifest, including on resume. Unknown revisions are rejected. New manifests
record the revision, so changing the selection strategy cannot silently
change a frozen collection. Prioritizing a card establishes exposure, not
evidence of better task performance.
PATS is disabled by default. Existing configurations default to
`usage = "always"` so their inference behavior is preserved. `training_only`
injects support during self-play collection and removes the full skill block
for ordinary inference, benchmarks and the experiment's final evaluation.
`usage = "off"` disables injection and maintenance.

`adaptive-solve`, `selfplay-rollout`, `benchmark` and `selfplay-experiment`
accept `--skill-context auto|on|off`:

| Value | Behavior |
| --- | --- |
| `auto` | Use the configured usage policy and the command's training/evaluation role. |
| `on` | Keep enabled SkillBank support, including at evaluation. |
| `off` | Remove support and skip its maintenance without deleting the archive. |

For example, these two offline inference calls exercise the explicit context
switch with mock models:

```bash
spgfs adaptive-solve --config configs/pats.example.toml --mock \
  --task 'What is 6 times 7?' --reference 42 --skill-context off
spgfs adaptive-solve --config configs/pats.example.toml --mock \
  --task 'What is 6 times 7?' --reference 42 --skill-context on
```

Collection and training use the existing `selfplay-rollout`, `train-cycle`
and `selfplay-experiment` commands. No extra PATS service or training policy
is introduced: reviews use the configured `runtime_routing.skill_distiller`
route. In mock collection, the component records evidence and mode decisions
without contacting a refiner or fabricating edits. Real reviews use the
actual Director tokenizer for capacity checks. The asynchronous
`--async-next-cycle-rollouts` experiment option is rejected while PATS is
enabled because the next snapshot must follow the completed review boundary.

## Cycle behavior and durable state

1. Collection freezes the Director bank and PATS views in the existing
   `director_skill_snapshot.v2.json` cycle artifact. The rendered context
   manifest includes `pats_scope`, `pats_snapshot_id` and `context_sha256`.
2. Maintenance reads the raw `solver_rollouts.jsonl`, including credible
   all-failure and all-success groups. Unknown rewards, infrastructure
   failures, final-test examples and groups with inconsistent policy or
   context provenance are excluded.
   Evidence reads the actual nested Canvas `payload` and graph `relations`:
   actions, acceptance/rejection, public feedback, execution summaries and
   selected skill versions. Full-trajectory statistics compare all credible
   siblings; bounded representatives cover the process rather than only its
   first few steps. Reasoning blocks and private verifier payloads are excluded.
3. Evidence is scoped by canonical dataset, task type and fixed task
   difficulty metadata. Difficulty is never inferred from observed reward.
   For each represented scope, the zero-initialized EMA updates once from
   the mean of its per-task group mean rewards. This equals success rate
   for binary rewards; continuous scores retain their continuous values.
   Without explicit `difficulty_bucket` or `difficulty`, the difficulty scope
   is `unspecified`. Continuous ADS NLL values are not separate scopes. Larger
   experiments can supply fixed, documented offline bins; small pools need
   enough distinct tasks per scope to admit reviews.
4. A review requires at least `min_groups` distinct source tasks, follows
   `review_interval`, and consumes at most `max_evidence_groups` recent
   cards. `max_policy_lag` bounds evidence age in collection cycles.
   At most `max_reviews_per_cycle` logical Refiner review calls are made. The
   backend can make multiple provider generations inside one call, for example
   its existing length-recovery path; observed generation attempts are audited
   separately and this limit is not a claim about the total HTTP request count.
   Scopes least
   recently reviewed get priority so a small request budget does not
   permanently exclude later datasets.
   Requests admit complete evidence groups under a 64,000-character bound
   and `max_review_input_tokens`, preferring recent distinct tasks and
   contrasting outcomes. At least two source tasks must still fit; otherwise
   the review is rejected with a diagnostic. The token guard uses the Director
   tokenizer as a proxy for a different Refiner model, and does not include its
   chat template. The default input budget is 20,480 tokens. Configure Refiner
   context for at least this input budget plus its maximum output allowance and
   chat-template overhead. A 32,768-token service accommodates the current
   20,480-token input and 4,096-token output limits with room for the template.
   Each admitted group's request ID is a short local alias (`E1`, `E2`, ...);
   all other evidence contents stay intact. The request lists the allowed IDs,
   and the audit retains their exact mapping to canonical evidence hashes.
5. Each edit must reference at least `min_evidence_cards` different source
   tasks. The whole proposed transaction is validated before publication.
   Rejected edits preserve the previous view. Accepted edits affect only a
   future frozen view for the same scope.

Where supported, the Refiner uses a strict JSON Schema with separate
ADD/UPDATE/DELETE branches, required card fields, allowed card kinds, current
skill IDs, and an evidence-ID enum containing only the supplied short aliases.
The schema restricts output syntax and identifiers; the local transaction
validator still enforces distinct-task evidence, scope, edit budgets and
rendered-token capacity. Exact aliases are mapped back to canonical hashes
before that validation. Text-only compatibility also accepts an exact supplied
canonical ID; prefixes, case changes, whitespace changes and unknown IDs are
never guessed or repaired.

The Refiner also receives the actual Canvas control contract: exposed actions,
legal parameters and budgets govern execution; directed links advance layers,
while bidirectional links connect the same layer. It respects the separate
relation-choice gate where that interface is exposed, model configuration
requirements, and existing tool/access limits. Guidance must distinguish
unresolved task content from execution failures and exhausted budgets, so
collaboration does not require every substantive task issue to be resolved
first. These constraints condition future proposals; they do not rewrite
previously generated cards.
An empty upstream-packet list is normal for an Agent without directed
predecessors and does not establish infrastructure or retrieval failure.
FINISH-only controls and authoritative budget exhaustion override any skill
requesting further repairs. Neither JSON Schema acceptance nor these prompting
constraints establish a card's semantic correctness or effectiveness.

A backend without the structured-generation interface, or one explicitly
reporting unsupported structured generation before sending a request, can use
the plain-text path. Provider/API errors are recorded as failures and do not
trigger an unconstrained fallback or an extra PATS retry.

PATS modifies scope-specific copies of existing Director cards. It does not
delete or rewrite global cards or protected seed versions. An explicitly
empty scoped view represents withdrawn support and does not fall back to
the global bank. Existing usage accounting continues, while legacy skill
generation and negative-use retirement are bypassed when PATS owns
maintenance. This prevents two lifecycle mechanisms from undoing each
other's edits.

State uses `pats_state` and `pats_cycles` tables in the existing SkillStore
SQLite database, derived from `solver_skillbank.cases_path` with suffix
`.v2.sqlite3`. `pats_state` stores scoped views, EMA and recent evidence;
`pats_cycles` records the cycle identity, input digest and review receipt.
An identical replay returns its existing receipt. Changed replay inputs,
configuration changes, a different run identity, or out-of-order cycles
are rejected; use a fresh experiment database when changing the controller
configuration or run location.

`skill_context_contract.json` records whether support is enabled and the
PATS/retrieval settings for a cycle, including cycles with support off.
Resume validates this contract before reusing completed results. Changing
the switch or retrieval settings requires a fresh output cycle. Owner and
cycle-order checks reject incompatible state before collecting new data.
For repeated updates, use `selfplay-experiment`; `selfplay-rollout` is a
single-cycle entry point. The experiment command places skill state under
its own `runtime_state` directory by default, isolating independent runs.

Each maintained cycle writes `pats_review.json` with scope, mode, EMA,
credible-group counts, request counts and review outcomes. Successful
proposals include operations and before/after token counts. The review
audit also records supplied/omitted evidence counts, evidence aliases, returned
evidence IDs, request/response/schema hashes, input token counts and local
validation failures. It retains at most 12,000 characters of public Refiner
response text, stripping reasoning blocks and explicitly named private fields;
malformed JSON remains inspectable instead of being replaced by only a hash.
When the backend supplies generation metadata, the audit records the total
observed generation count and at most eight attempt summaries containing token
counts and finish reasons. JSON surrounded by
one complete outer JSON code fence is accepted; unrelated prose and partial
JSON remain errors. The existing
`skill_usage_summary.json` remains available. A maintenance failure is
reported through `skillbank_v2_maintenance_error.json`; raw evidence remains
available and a valid policy update is not canceled by a refiner failure.

New PATS usage events and outcome records retain `pats_scope`,
`pats_snapshot_id`, and, when supplied, `selection_revision` from the frozen
collection manifest. Usage summaries separate `(skill_id, version, dataset,
pats_scope)`, so independently revised seed copies in different scopes are
not combined. The same scoped card version can accumulate usage across
collection snapshots; the individual records retain each snapshot identity.
Legacy records retain their existing summary fields and grouping. Historical
PATS records without scope stay unscoped until their original raw manifest is
replayed; current state is never used to guess their historical membership.
An identical replay may add these previously absent metadata fields in place,
with the original event key and every original value preserved. Changed
rewards or already recorded scope/snapshot values remain replay conflicts.
This accounting does not consult global card membership or drive the PATS
controller's EMA, edits, or retrieval.

Cycle metrics read that cycle's frozen v2 snapshot even when no legacy live
bank JSON exists. Global seed/card counts, scoped card instances and unique
skill IDs are reported separately; scoped instances are not added to the
global total. PATS review metrics include updated/rejected scopes and applied
operations. Card counts describe the collection-start snapshot, while review
counts describe maintenance after collection; accepted edits normally appear
in the next cycle's card counts. Missing artifacts are marked unavailable.

## Local edit contract

The local API uses `ADD`, `UPDATE` and `DELETE`, with a maximum of four
operations per review by default. ADD and UPDATE require complete Director
cards (`name`, `description`, `trigger`, `plan`, `pitfall`, `constraint`,
`kind`). Update/delete targets must already exist in the scope. Duplicate
edits to one ID, missing evidence, unknown fields, invalid card content and
over-budget resulting views reject the transaction.

| Mode | Local addition budget | Resulting view requirement |
| --- | ---: | --- |
| EXPAND | At most 2 cards | Respect entry and actual rendered-token limits. |
| REVISE | At most 1 card | Respect entry and actual rendered-token limits. |
| COMPRESS | No additions | Strictly reduce actual rendered tokens. |
| FORCED_PRUNE | No additions | Strictly reduce actual rendered tokens and satisfy capacity limits. |

An empty operation list is an explicit no-op. These checks are stricter than
the inspected upstream code. They establish schema, provenance and capacity
boundaries; referenced observations do not prove that a proposed instruction
is causally useful or semantically correct.

## Scope and interpretation

The scaffold supports the Director, which learns graph construction. Worker
parameters remain frozen. Editing a skill can change the Director's future
graph and worker instructions, but the scaffold does not itself supply
worker training gradients. Changes to scaffolding do not change task
verifiers, Solver rewards, Graph-local Frontier, or PPO's action masks.
This change does not add the paper's separate scaffold-conditioned SFT
warmup or reproduce its full training recipe. The ability to interpret the
Director card interface and performance after withdrawal require model-backed
validation.

A low success rate is evidence about the sampled tasks under the current
policy and support. It is not a measurement of unsupported competence.
Proposer/ADS may select harder tasks over time, so a falling rate need not
mean that the policy regressed. Support levels and policy quality should be
compared on fixed training diagnostic tasks, with final test examples kept
out of scaffold maintenance.

PATS's intermediate-success argument explains when binary rollout groups
are more likely to contain both successes and failures. It does not require
every group to target 50% success and does not establish convergence of this
project's coupled task-selection and scaffolding process. Thresholds are
configuration choices, not an additional optimization objective.

Freezing a bank version is only part of reproducibility. All attempts in a
comparison group must share the same actual rendered support. Training must
reuse the prompt and token context captured at rollout time, including that
support, when computing the policy ratio. Newly generated guidance must
only become visible to a later collection snapshot.

Graph relation counterfactuals execute fixed graphs rather than rerunning
the Director. They must retain the original node instructions and executor
configuration while changing only the intended relation. Replanning or
retrieving new guidance during that comparison would change the
intervention being measured.

## Upstream implementation reference

The inspected upstream controller uses a zero-initialized success-rate EMA.
For each task type, it averages the per-task group signal observed in one
PPO step, then updates `ema = alpha * observation + (1 - alpha) * ema` once.
The default alpha is `0.1`. Bank pressure is the larger of entry-count and
estimated-token utilization. Its default bank limits are 30 entries and
2,000 estimated tokens.

Mode precedence is `FORCED_PRUNE` at pressure at least one, `COMPRESS` at or
above the compression threshold, `REVISE` at or above the revision
threshold, and otherwise `EXPAND`. Upstream's controller configuration uses
revision/compression thresholds `0.3/0.7`, while its standalone mode helper
defaults to `0.3/0.85`; callers should specify thresholds explicitly.

Upstream proposal budgets per review are:

| Mode | New skill entries | Of which general | New mistake entries |
| --- | ---: | ---: | ---: |
| EXPAND | 2 | 1 | 1 |
| REVISE | 1 | 1 | 1 |
| COMPRESS | 1 | 1 | 0 |
| FORCED_PRUNE | 0 | 0 | 0 |

Those are source reference values, not a claim that this integration
reproduces the paper's complete training recipe. The upstream implementation
contains several details that should not be mistaken for stronger guarantees:

- New proposals are prompted to cite at least two evidence cards, but the
  upstream entry validator does not check the supplied `evidence_count`.
- Invalid operations are discarded individually; the remaining valid
  mutations are committed together. Atomic writing does not mean that one
  invalid operation rejects the entire reviewer response.
- `COMPRESS` requires a companion update or deletion when adding an entry,
  but upstream does not enforce a non-increasing final token count.
- The fallback prune removes newest entries first. Its forced-shrink flag
  can remove an additional entry after the reviewer has already deleted one.
- Token accounting uses roughly four characters per token. Deduplication
  uses word Jaccard similarity at `0.7`; neither is a model tokenizer or an
  embedding-based semantic guarantee.
- The general-skill renderer includes all dynamically learned entries
  before static entries, so its nominal general top-k is not a hard limit.

Code-path validation and mock runs establish implementation behavior only.
They do not establish better task accuracy, reduced real-model cost,
successful skill internalization, or parity with the published PATS results.
