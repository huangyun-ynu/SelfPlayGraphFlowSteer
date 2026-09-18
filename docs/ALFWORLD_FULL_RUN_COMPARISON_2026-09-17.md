# ALFWorld full-run comparison

Sources under `state/sota-20260916`:

- A: `alfworld-qwen35-9b-no-skill-v1`.
- B: `alfworld-qwen35-9b-deepseek-no-skill-v1`.

Read all 256 sample records and their embedded trajectories. Both runs contain
the same 128 sample keys, identical prompts, seed 0, and matching game
fingerprints wherever both runs recorded environment results. Both used local
Qwen3.5-9B Director without thinking, 24 rolling slots, no dataset Director
Skill, and a 350000-token ALFWorld budget.

## Actual differences

| Setting/result | A | B |
| --- | --- | --- |
| Configured Worker routes | Full pool | DeepSeek only |
| Actual artifact route/model | gpt / gpt-5.5 | deepseek / deepseek-flash |
| Physical successful request routes | gpt and gpt_eco | deepseek |
| Reasoning usage | 2187/2336 successful request events report positive reasoning tokens | thinking off; no positive reported reasoning tokens |
| Episode step limit | 50 | 50 |
| Initial cumulative step allowance | 400 | 200 |
| Observed fuse threshold | 20 | 18 |
| Scored successes | 45/128 | 20/128 |
| Tasks with Worker backend failure sentinel | 62 | 33 |
| Non-API failures with a fuse | 16 | 72 |
| Other non-API failures | 5 | 3 |
| Any fuse, overlapping API/other outcomes | 21 | 85 |
| Failed tasks with accepted automatic FINISH | 12 | 32 |

Artifacts were deduplicated by artifact ID per task. Backend counts classify
failure using the runtime-owned WORKER_BACKEND_FAILURE sentinel; they are not
counts of individual failed network attempts. API-free subsets are selected
post hoc, not unbiased estimates of model capability.

## Paired outcomes

- Both succeed: 17.
- A succeeds, B fails: 28 (9 backend failures, 18 fuse failures, 1 other).
- A fails, B succeeds: 3.
- Both fail: 80.
- Neither run has a backend-failure artifact: 51 matched tasks; A succeeds on
  36, B on 17.

All A scored-success trajectories used at most 57 total recorded environment
steps across attempts. Neither run recorded a zero cumulative remaining-step
allowance in action outputs. Thus the 400-to-200 change does not explain the
observed successes disappearing through actual cumulative-budget exhaustion.
The 20-to-18 fuse change can still affect behavior and needs an ablation.

## Concrete paired traces

- Two knives (`61718bd3d2d20bc29b651f39`): A succeeds in 32 steps. B stops at
  24 steps, including 12 look and 9 examine-drawer actions, without taking a knife.
- Clean soap on toilet (`67689d05019f7057ed3b5dc4`): A succeeds in six steps,
  taking soapbar 1, cleaning it in the sink and placing it on the toilet. B
  repeatedly takes and replaces soapbottle 1, failing at 24 steps.
- Bowl under lamp (`b2593e7449bb9a791fa56449`): A succeeds in 14 steps. B
  repeats take/place/look and fails in separate attempts at 22 and 40 steps.

Both historical runs lack the newly added factual_memory and initial_observation
fields in their recorded action outputs. Shared context and closure defects
therefore predate the later fixes; they are not evidence that those fixes caused
the original 45-to-20 drop. A also has 46 tasks with any official environment
success but only 45 scored successes, consistent with the previously identified
lost-output issue. This was not an inflated 45-success result.

## Interpretation

The main observed change is GPT-5.5 with reported reasoning activity versus
DeepSeek Flash with thinking disabled, interacting with existing context and
recovery weaknesses. The data establish more repetitive unsuccessful behavior
in B, not that one model is intrinsically inferior under matched settings.
API failures decreased and cannot explain the aggregate direction alone.
No exact causal contribution can be assigned without matched configuration,
reasoning mode, code revision and repeated-run controls. The later targeted
three-failure-sample test is not comparable to either full-run accuracy.
