# Public copy scope

This directory is an independent source distribution, not a continuation of a private training run. The original project's services, checkpoints, EMA state, trajectories and working tree are not modified by preparing it.

## Removed functionality

- Skill cards, skill-bank persistence, pending failure queues, retrieval, consolidation and distillation.
- Skill context injection into Director prompts.
- Skill-specific runtime configuration, CLI commands, public exports, result fields and metrics.
- Tests and documentation dedicated to those removed features.

Supporting model calls used by answer formatting and judging remain available through the `support` runtime route. Graph E5 embeddings have an independent loader and use `[graph_features].embedding_model_path`; they no longer require a skill-bank object. These are still necessary parts of graph scoring and verification.

## Distribution boundaries

- Private provider settings are replaced with a generic runtime endpoint and an environment-variable credential.
- The original implicit localhost proxy is removed; set `MODEL_API_PROXY` explicitly when needed.
- Machine-specific SWE cloud provisioning is removed; generic SSH/backend integration remains configurable.
- Experiment archives, internal status documents, downloaded third-party repositories, private task pools, model weights, caches and logs are excluded.
- Curriculum profiles are examples. They do not include or imply access to their original data pools.
- Original CUDA-specific installation overrides are removed. Optional dependencies need installation appropriate to the target hardware. The preparation environment had PyTorch `2.13.0+cu129`, Transformers `5.15.0`, PEFT `0.20.0`, Accelerate `1.14.0` and vLLM `0.27.1+cu129`; this is an observed environment, not a claim that every supported version combination has been tested.

## Validation boundaries

Validation covers source compilation, undefined-name checks, mock collection/execution and portable tests. It does not run paid APIs, download benchmark datasets, perform real model updates or validate multi-GPU performance. The removal changes the policy's available skill context; no unchanged benchmark accuracy is claimed.

Advanced asynchronous and timeline-update modes retain their existing implementation and opt-in flags. Real deployment still requires choosing and validating an appropriate training configuration.

See `VALIDATION.md` for the actual checks performed on this distribution, and the root `NOTICE.md` for licensing limits.
