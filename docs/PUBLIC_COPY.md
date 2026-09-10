# Public copy scope

This directory is an independent source distribution, not a continuation of a private training run. The original project's services, checkpoints, EMA state, trajectories and working tree are not modified by preparing it.

## Included SkillBank functionality

Director skill cards, versioned persistence, pending evidence queues, E5 retrieval, background distillation, prompt integration, usage accounting and retirement are included. Eight initial cards ship in the package. See `SKILLBANK.md` for configuration and lifecycle details.

The support inference role uses `runtime_routing.skill_distiller`; E5 uses `solver_skillbank.embedding_model_path`. The default mock configuration disables the bank; a separate SkillBank template enables it. No private skill databases or experiment cases are included.

## Distribution boundaries

- Private provider settings are replaced with a generic runtime endpoint and an environment-variable credential.
- The original implicit localhost proxy is removed; set `MODEL_API_PROXY` explicitly when needed.
- Machine-specific SWE cloud provisioning is removed; generic SSH/backend integration remains configurable.
- Experiment archives, internal status documents, downloaded third-party repositories, private task pools, model weights, caches and logs are excluded.
- Curriculum profiles are examples. They do not include or imply access to their original data pools.
- Original CUDA-specific installation overrides are removed. Optional dependencies need installation appropriate to the target hardware. The preparation environment had PyTorch `2.13.0+cu129`, Transformers `5.15.0`, PEFT `0.20.0`, Accelerate `1.14.0` and vLLM `0.27.1+cu129`; this is an observed environment, not a claim that every supported version combination has been tested.

## Validation boundaries

Validation covers source compilation, undefined-name checks, mock collection/execution and portable tests. It does not run paid APIs, download benchmark datasets, perform real model updates or validate multi-GPU performance. Enabling SkillBank changes the policy's available context; offline checks do not establish benchmark accuracy.

Advanced asynchronous and timeline-update modes retain their existing implementation and opt-in flags. Real deployment still requires choosing and validating an appropriate training configuration.

See `VALIDATION.md` for the actual checks performed on this distribution, and the root `NOTICE.md` for licensing limits.
