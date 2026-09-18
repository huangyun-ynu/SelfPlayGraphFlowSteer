# ALFWorld DeepSeek thinking-enabled full evaluation

## Setup

- Output: `state/sota-20260916/alfworld-deepseek-thinking-full-v1`.
- Config snapshot: `configs/alfworld_deepseek_thinking_eval.toml`.
- Input: all 128 unchanged records in `data/formal/eval/alfworld_official_test.jsonl`, seed 0.
- Frozen local Qwen3.5-9B Director, Director thinking off.
- Fixed DeepSeek Worker route, served model `deepseek-flash`, thinking on.
- Reasoning effort remains `low`; the only configuration change from the
  current formal config at launch was `runtimes.deepseek.enable_thinking=true`.
- No dataset Director Skill; skill context off; 24 rolling slots.
- Environment limits 50 per episode / 200 cumulative; fuse threshold 18;
  configured task token budget 350000.
- Direct API requests, W&B disabled, no post-run retries or downloads.
- Current code includes factual memory, retained public reset task, success
  locking, and the ALFWorld automatic-FINISH guard. This differs from historical
  thinking-off code and is not a thinking-only controlled ablation.

## Final results

All 128 records completed; the benchmark process exited successfully.

- Official successes: **104/128 (81.25%)**.
- Backend failures: 23 (6 DNS failures, 16 request timeouts, 1 SSL unexpected EOF).
- Other failure: 1, the wet-towel/clean-cloth task, described below.
- All 104 scored successes have a recorded official `done=true, won=true` result.
- Two tasks encountered no-progress fuses, and both ultimately succeeded.
- 1345 of 1457 deduplicated successful backend request events report positive
  reasoning tokens, confirming thinking was actually exercised.

The summary's `run.failed=0` counts orchestration completion, not task success
or freedom from backend failure. API failures remain in the 128-task denominator.

| Task type | Success / total |
| --- | --- |
| Cool then place | 17/22 |
| Clean then place | 25/28 |
| Place two objects | 14/21 |
| Simple placement | 20/24 |
| Heat then place | 13/17 |
| Object under light | 15/16 |

## Historical comparison

The historical fixed-DeepSeek thinking-off run scored 20/128 (15.625%).
Paired outcomes on identical task IDs and prompts:

- Both pass: 18.
- Previously failed, now pass: 86.
- Previously passed, now fail: 2.
- Both fail: 22.

The earlier GPT-5.5 run scored 45/128 (35.15625%). The present improvement
cannot be attributed exclusively to thinking: code fixes and realized API
availability also differ. No current-code thinking-off full run was added.

## Recovery and remaining issue

- `dc4666a93ec30b5c47f183d1`: first Agent fused at 38 steps; a second Agent's
  new session succeeded at 28 steps.
- `a6aad9df6aa71b0ce49acafa`: two Agents fused at 27 and 37 steps, then revision
  artifacts preserved attempt indices 1 and 2 and succeeded at 39 and 49 steps.
  This exercises same-session continuation with real thinking-enabled Workers.
- `679cd6562467268fa8e55717`: original prompt is "Put a wet towel in the cabinet."
  Public reset task is "put a clean cloth in cabinet." Two attempts used 47 and
  50 steps without success. Runtime then reported `execution_budget_exceeded`
  at 592532/350000 tokens. No output Agent was selected, so verification reports
  `missing_output_agent`; its zero step field does not mean no actions ran.
  This exposes delayed token-budget enforcement and is not a cumulative
  200-environment-step exhaustion or a no-progress fuse.

The main formal configuration was not changed by this experiment. No failed
samples were automatically rerun and no W&B upload was performed.
