# Formal training pools

This directory contains the seven fixed training pools used by the formal
training configuration. Each JSONL file has 512 records, all marked with
`split=train`.

The pools are copied from the validated ADS task pools generated under the
local training state. Experiment logs, trajectories, checkpoints, retrieval
corpora, model files, virtual environments, and other runtime state are not
included here.

| Dataset | Records |
| --- | ---: |
| AIME | 512 |
| ALFWorld | 512 |
| HealthBench Professional | 512 |
| HotpotQA | 512 |
| NQ-open | 512 |
| SWE-bench Verified | 512 |
| WebShop | 512 |
