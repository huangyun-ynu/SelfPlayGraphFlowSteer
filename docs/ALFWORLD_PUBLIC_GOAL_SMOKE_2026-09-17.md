# ALFWorld public reset goal: real-sample smoke test

## Setup

- Output: `state/sota-20260916/alfworld-deepseek-public-goal-targeted-v1`.
- Input: `state/sota-20260916/alfworld-public-goal-targeted-v1.jsonl`.
- Two unchanged real benchmark records selected from the previous targeted run.
- Frozen local Qwen3.5-9B Director, fixed DeepSeek route (`deepseek-flash`).
- Director and Worker thinking disabled; no dataset Director Skill; skill context off.
- 24 configured rolling slots, two samples; seed 0; no parameter updates.
- Episode limit 50, cumulative environment limit 200, no-progress threshold 18.
- Model requests direct; W&B disabled; no retrieval or downloads started.
- The lightweight project interpreter lacked transformers before evaluation began.
  Execution used the existing `owner/.venvs/spgfs-pats-gpu/bin/python` instead.

## Results

Both samples passed the official ALFWorld environment verifier (`done=true`,
`won=true`, score 1.0). Neither had an API failure or a no-progress fuse.

| Sample | Previous memory-only targeted run | Public-goal fix run |
| --- | --- | --- |
| `084d236aaf68d1d8a0da1d2b` (disc/vase) | Failed; separate artifacts at 48 and 40 steps, second fused | Success in 3 steps |
| `ce277f7258ce01d843791e6b` (spatula) | Failed; fused at 34 steps | Success in 7 steps |

Disc/vase actions: `go to desk 1`, `use desklamp 1`,
`take vase 4 from desk 1`. The original yellow-disc prompt was unchanged.

Spatula actions: visit countertops 1 and 2, take spatula 1, visit sinkbasin 1,
`clean spatula 1 with sinkbasin 1`, visit diningtable 1,
`move spatula 1 to diningtable 1`.

## Context checks and limitations

- All 10 real action outputs retained the initial public observation and reset
  task statement. Reprojection through the actual Worker context function kept
  the statement and removed the internal goal contract.
- Both execution-feedback records included the public reset task for Director.
  Both tasks ended immediately through trusted-success recovery, so there was
  no subsequent Director model call to test strategy revision using that feedback.
- Worker request bodies are not independently captured by this audit; prompt
  delivery evidence combines recorded real states with the runtime projection.
- These are two targeted single runs, not a full benchmark or a controlled
  repeated-seed estimate. The results do not establish a general accuracy gain.
- Complete records, trajectories, summary and manifest remain in the output directory.
