# SelfPlayGraphFlowSteer

Separate Proposer and Solver policies for multi-agent graph self-play. The Proposer selects tasks; the Solver's Director builds and executes a graph of workers. Verified rollouts feed Frontier scoring and policy updates.

This distribution includes Director SkillBank generation, retrieval, usage accounting and retirement. It retains graph execution, ADS/TSDS task selection, dataset adapters, relation counterfactuals, Frontier rewards, partial-group training, and the training orchestration code. No credentials, private experiment results, datasets, model weights, or machine-specific launch scripts are included.

## Offline quick start

Python 3.11 or newer is required. Run these commands from this repository's root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'

spgfs --help
spgfs adaptive-solve --task 'What is 6 times 7?' \
  --config configs/mock.toml --mock --reference 42
spgfs dry-run-selfplay --mock --num-tasks 2 --rollouts 5 \
  --output state/demo.json
pytest -q
```

Mock commands do not download models, contact inference providers, or train real parameters. Their scores are synthetic and are not benchmark results. Tests requiring optional model libraries are skipped if those libraries are unavailable.

To exercise collection followed by both mocked policy updates, run `bash examples/mock_cycle.sh`. Choose a fresh directory with `bash examples/mock_cycle.sh state/another-demo`. The verifier is disabled only in this mock recipe; real collection requires a verifier.

## Real inference and training

1. Install the optional dependencies for your use case: `.[openai,selfplay]` for model-backed collection, `.[train]` for training, and `.[serve]` for local vLLM services. Install a PyTorch/CUDA build compatible with your hardware; the package does not force the original machine's CUDA wheel index. ALFWorld, WebShop and ADS have separate optional extras.
2. Copy `configs/adaptive.example.toml` to `configs/adaptive.local.toml` and edit paths, model names, local policy endpoints and runtime endpoints. For files directly inside `configs/`, relative paths resolve from the repository root. For configurations stored elsewhere, they resolve from the configuration file’s directory.
3. Set `WORKER_API_KEY` in the environment. `.env.example` documents this variable; the configuration loader also reads the nearest `.env`, but copying it does not export shell variables to other programs. Never commit real credentials.
4. Configure the physical GPU IDs in `[resources]` and explicitly set `SPGFS_ALLOWED_PHYSICAL_GPUS` to the permitted IDs, for example `0,1,2`. Configure separate Proposer/Solver checkpoints and prepare the required local model and dataset files.
5. Read the available flags before running:

```bash
spgfs selfplay-rollout --help
spgfs train-cycle --help
spgfs selfplay-experiment --help
```

Collection writes durable trajectories and training batches. `train-cycle --run-dir ...` consumes a collected batch; `selfplay-experiment` orchestrates multiple cycles. Use `--config` explicitly for real runs. `configs/adaptive.toml` is a credential-free local example, not the original production deployment configuration.

The dataset adapters cover AIME, NQ Open, HotpotQA, HealthBench, ALFWorld, WebShop and SWE-Bench. Stateful environments and official validators must be supplied separately. This repository does not bundle the original SWE cloud account or automatically provision it.

## Layout

| Path | Contents |
| --- | --- |
| `src/selfplay_graph_flowsteer/` | Graph execution, policies, scoring, collection and training |
| `src/selfplay_graph_flowsteer/legacy/` | Compatibility helpers still used by the implementation |
| `configs/` | Sanitized examples, curriculum profiles and prompt templates |
| `tests/` | Portable algorithm, mock integration and training tests |
| `examples/` | Offline collection/update recipe |
| `docs/PUBLIC_COPY.md` | Release scope, removed functionality and validation limits |
| `third_party/notices/` | Available upstream license texts and notices |

Proposer's dataset EMA baseline is enabled by default with decay `0.9`. Trusted revalidation ties retain 10% of the original graph-pair contribution; EMA state is bound to that reward rule. The supplied runtime configurations disable predicted-time rejection and early consolidation while retaining actual deadlines. See [training policy](docs/TRAINING_POLICY.md) for details and compatibility behavior.

## Provenance and licensing

See [NOTICE.md](NOTICE.md) for implementation provenance. Available MIT/Apache license texts are retained for the corresponding upstream components. The local FlowSteer, SESA and SkillFlow snapshots did not include a discoverable license file; this copy does not invent a license grant for their adapted code. Confirm the relevant upstream redistribution terms before publishing that code under an open-source license.

## Director SkillBank

See [SkillBank setup and lifecycle](docs/SKILLBANK.md). The offline mock configuration keeps SkillBank disabled; `configs/skillbank.example.toml` enables the current implementation with eight seed skills and an environment-variable API credential.

The optional [PATS integration](docs/PATS.md) ([中文说明](docs/PATS.zh-CN.md)) maintains a separate skill view for each dataset, task type and difficulty using observed policy performance. Each collection cycle freezes its context; bounded reviews expand, revise or compress the next cycle's scaffold. `configs/pats.example.toml` enables this component during training and turns skill context off for normal evaluation. Use `--skill-context on` for an explicit supported evaluation or `--skill-context off` for an ablation. The existing SkillBank behavior remains the default. An offline collection check is:

```bash
spgfs selfplay-rollout --config configs/pats.example.toml \
  --mock --seed 'independent evidence verification' \
  --seed 'compare evidence and verify constraints' --rollouts 2 --verifier exact_match \
  --output state/pats-demo/cycle_000
```

Mock mode exercises snapshotting and evidence collection without contacting a refiner or claiming learned-skill gains. Real reviews use the configured frozen `skill_distiller` route. Start independent experiments with separate SkillBank state paths.
