# Offline release validation

## SkillBank restoration (2026-09-10)

The public distribution now includes the current Director SkillBank implementation and eight packaged seed cards. Private runtime state, routes and credentials were not copied.

- Skill lifecycle/integration and release-boundary checks: 41 passed (including the separate distiller-route configuration probe).
- Wheel build: passed; the seed JSON and all three Skill modules are included.
- Source compilation and undefined-name/unused-import checks: passed.
- Public mock collection followed by mock Proposer/Solver updates: passed.
- Private credential value scan: no matches in distributable files.

The broader suite exposed two stale test assumptions: a resume test did not remove the completed-trajectory spool when simulating data loss, and a concurrent abort test assumed a primary log existed even when zero trajectories completed. These test fixtures were corrected; no production behavior was changed to satisfy them. The interrupted broad run completed 523 passing cases and 7 skips before fixture diagnosis. A subsequent 232-case run completed 224 passes, 4 skips and 4 instances of the same stale-spool fixture issue. After correcting all affected fixtures, the focused rerun passed all 6 affected cases (including the earlier two). Thus the latter 232-case coverage is 228 passing cases and 4 optional-dependency skips after corrections; these overlapping runs are not added together. A single fresh full-suite run was not repeated after the fixture-only corrections.

No paid API calls, real GPU training or model updates were run.

## Frontier/EMA and predicted-time admission update (2026-09-10)

The updated copy was checked in the same isolated public-checkout environment:

- Frontier stability, Proposer learning, research metrics, partial training groups and distribution boundaries: **102 passed, 1 skipped**. The skipped test requires the optional PyTorch dependency, absent from this clean environment. This targeted run overlaps the historical suite below; counts are not additive.
- Source compilation, Ruff formatting for source and changed tests, and unused-import/undefined-name checks for source and changed tests: passed.
- All three supplied runtime configurations load with predicted-time admission disabled. Offline probes attach an actual deadline and make either time-estimation function raise if invoked: both prediction paths are bypassed, while an expired real deadline still raises the normal deadline exception. The generic API template used a dummy credential and declared GPU IDs for configuration validation only.

No inference API, real trajectory collection, GPU allocation or parameter update was performed during this synchronization. At that revision, SkillBank was still removed (restored in the update above). These results do not establish benchmark accuracy. The complete historical portable suite was not rerun for this update.

## Initial distribution preparation

Validated on 2026-09-10 using an isolated Python 3.11.15 environment, with only this checkout and its `dev` dependencies installed. No original-project virtual environment or editable package was exposed to that environment.

| Check | Result |
| --- | --- |
| Editable installation, `.[dev]` | Passed |
| Portable test suite | 625 passed, 11 skipped |
| Added distribution boundary tests | 3 passed |
| Focused rerun of deadline, endpoint pool and distribution checks | 25 passed |
| Source compilation | Passed |
| Ruff unused-import / undefined-name checks (`F401,F821,F822,F823`) | Passed |
| Ruff formatting check | Passed |
| CLI mock solve and dry-run batch construction | Passed |
| Mock rollout collection followed by mocked Proposer and Solver updates | Passed |
| `examples/mock_cycle.sh` end-to-end recipe | Passed |
| Wheel build, removed-module exclusion and notice inclusion | Passed |
| Example output paths remain inside the checkout | Passed |
| Optional proxy configuration and local/direct bypass | Passed |
| Private credential values from the source deployment found in distributable files | None found |

The portable suite and three added distribution tests cover 639 collected cases in total: **628 passed and 11 skipped**. Focused reruns overlap those cases and are not added to that total. Skips require optional model libraries or a local tokenizer not included in the clean environment.

Tests tied to removed functionality and private deployment scripts are excluded from this distribution. Public endpoint-pool tests use explicit local fixtures rather than relying on production provider configuration.

No paid inference routes, real benchmark cycles, GPU training or real checkpoint updates were run for this release preparation. These results establish offline integration and package hygiene, not benchmark accuracy or training-speed equivalence.

The GitHub Actions workflow is supplied but has not been executed on GitHub. Python 3.12 is included in its matrix; the local clean-environment run used Python 3.11 only. Source-install workflows are documented in the README; the wheel build does not substitute for preparing external model, prompt/configuration and dataset assets on a new deployment.
