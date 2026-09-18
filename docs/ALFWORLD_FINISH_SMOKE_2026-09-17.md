# ALFWorld automatic FINISH: real-sample smoke test

## Setup

- Input: `state/sota-20260916/alfworld-finish-targeted-v1.jsonl`.
- Output: `state/sota-20260916/alfworld-deepseek-finish-targeted-v1`.
- Three unchanged real samples selected from the 18 historical premature-closure cases.
- Frozen local Qwen3.5-9B Director; fixed DeepSeek route (`deepseek-flash`).
- Thinking off, no Director dataset Skill, skill context off, seed 0.
- 24 configured slots; episode/cumulative limits 50/200; no-progress threshold 18.
- Direct model requests; W&B disabled; no downloads or automatic reruns.
- Both public-reset-goal retention and the automatic-FINISH guard are present.
  This is not a single-change controlled accuracy comparison.

## Results

| Sample | Historical full-run outcome | Current outcome |
| --- | --- | --- |
| `08e5bb814dcf35aef67b8cd1`, two pans | 20 steps, fuse, automatic FINISH | 50 steps, episode exhausted, explicit Director FINISH |
| `61718bd3d2d20bc29b651f39`, two knives | 24 steps, fuse, automatic FINISH | 43 steps, fuse, explicit Director FINISH |
| `67689d05019f7057ed3b5dc4`, clean soap | 24 steps, fuse, automatic FINISH | DeepSeek request timeout; no verifier result or FINISH-path coverage |

Official success count is 0/3. The summary's `run.failed=0` does not mean no
backend failures: the soap backend failure is recorded as a completed scored
sample with `WORKER_BACKEND_FAILURE`, `request_timeout`, and `APITimeoutError`.

## Control-flow evidence

Both non-API-failure samples executed these Director actions:

1. Configure and execute the Worker.
2. Attempt SET_PROMPT revision, rejected with `prompt_revision_evidence_required`.
3. Select the output with SET_OUTPUT.
4. Receive another actual Director model call, which explicitly chooses FINISH.

No automatic FINISH preempted the Director in either case. This verifies the
changed branch on real samples, but neither Director chose to continue after
output selection. No successful same-session revision was exercised.

Both rejected revisions omitted `revision_basis` and `evidence_agent_ids`.
Feedback exposed eligible unresolved-issue evidence but its concrete
`legal_recovery_actions` suggested only SET_OUTPUT. The existing evidence gate
was not relaxed or changed during this test.

These results establish preserved decision opportunity, not improved task
accuracy. Full trajectories and sample records remain in the output directory.
