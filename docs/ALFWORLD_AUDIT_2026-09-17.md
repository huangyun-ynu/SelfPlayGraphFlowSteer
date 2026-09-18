# ALFWorld trajectory audit

Scope: all 128 trajectories from `alfworld-qwen35-9b-no-skill-v1`, all
128 from `alfworld-qwen35-9b-deepseek-no-skill-v1`, and all 7 from
`alfworld-deepseek-memory-targeted-v1`, under `state/sota-20260916`.
This is a read-only diagnosis of execution behavior; no evaluation was rerun.

## Findings

1. Initial public environment task text is not retained after the first action.
   Director starts from the dataset prompt; Worker state is replaced after each
   action. Factual memory currently omits the reset observation/task statement.
   Concrete conflicts: `ce277f7258ce01d843791e6b` omits cleaning a spatula;
   `084d236aaf68d1d8a0da1d2b` requests a yellow disc while the environment asks
   for a vase. Preserve the public reset goal, and audit semantic disagreement;
   do not silently replace benchmark prompts with private PDDL targets.

2. `GraphCanvas.recover_finish_only` automatically finishes structurally valid
   selected-output graphs, without requiring ALFWorld success or environment
   termination. There are 32 failed tasks in the full DeepSeek run with an
   accepted automatic FINISH. The targeted spatula case finishes this way at
   Director round 4, after a Worker fuse at environment step 34. This removes
   further decision opportunities; it does not prove those tasks would recover.
   Restrict automatic closure for active ALFWorld episodes while retaining an
   explicit policy decision to finish and preserving official-success locking.

3. Director feedback lacks structured ALFWorld state. Canvas feedback formats
   SWE/WebShop progress but ALFWorld is reduced to summary and two unresolved
   issues. Expose remaining episode/rollout steps, trusted outcome, recent
   repeated commands, and same-session revision availability, without a plan.

4. Output-selection cycles evade the Director stall detector, which compares
   only successive graph signatures. In the full DeepSeek run, 51 failed tasks
   select outputs at least three times, 52 encounter structural-repair action
   rejections, and 33 consume 24 model turns (overlapping counts). There are
   251 `structural_repair_required` rejections. Track repeated repair states or
   unresolved topology defects rather than counting each output swap as progress.

5. Repair action guidance can omit the correction actually needed. Two
   `prompt_revision_evidence_required` rejections correctly expose eligible
   unresolved-issue evidence, but their concrete `legal_recovery_actions` only
   suggest selecting output. Include a valid grounded revision action template;
   preserve evidence requirements instead of bypassing them.

6. Memory cost and token-budget enforcement need review. The targeted fork
   case (`14351960048e8a9bdf0096c9`) ends with `execution_budget_exceeded` at
   451396/350000 tokens after two episodes (50 and 40 steps). Memory repeats
   observations in location records, object events and recent interactions;
   object histories have no independent character budget. Bound and deduplicate
   visible facts, retain the initial public goal, and enforce remaining token
   allowance at request boundaries. Do not infer improvement from larger context.

7. ALFWorld progress labels every non-fuse artifact `completed`, even when
   `done=false` and `won=false`. Separate Worker-returned from environment-done
   states. Scoring itself uses the official environment result, so this is a
   diagnostics issue, not evidence of false success scores.

## Controls and limitations

- Full DeepSeek run: 20 successes, 33 terminal backend failures; of the other
  75 failures, 72 include a fuse and 3 only exhaust episode steps. Across all
  tasks, 85 encounter a fuse, including tasks later ending with API failure.
- API failures: 17 DNS resolution failures, 15 read timeouts, 1 broken pipe.
- No lost official successes in either post-success-lock DeepSeek run.
- The earlier GPT run has one lost official success, already addressed.
- Same-session revision remains possible, but is rare: only two full DeepSeek
  tasks contain revision artifacts. Creating another Agent deliberately starts
  another environment; this must not be mistaken for same-Agent reset regression.
- Repeated-state fuse is an additional evaluation rule, not an established
  measure of goal progress. Compare with a disabled-fuse arm before attributing
  recoverable failures to model policy.
- Counts of different failure signals overlap. These observations do not prove
  causal success-rate gains; changes need controlled evaluation.
