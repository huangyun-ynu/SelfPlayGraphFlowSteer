# Frontier, EMA and execution limits

New independent Proposer batches use trusted revalidation tie discounting (`trusted_reverify_tie_discount_v1`, alpha `0.1`). For each graph pair, let `d0` and `d1` be the primary and revalidation reward differences, and `k` the existing graph kernel:

- Same nonzero direction: retain `k * ((d0 + d1) / 2)^2`.
- Nonzero primary difference and a trusted revalidation tie: retain `0.1 * k * d0^2`.
- Direction reversal or an initially tied pair: contribute zero.

Pair normalization remains unchanged. Missing verification and infrastructure failures cannot be treated as trusted ties. Collection records per-graph reward trust and reward source, and metrics report softened pair counts. Old cached outcomes without explicit trust flags do not receive the new tie contribution.

Proposer advantage is the softened Frontier minus the same dataset's EMA, frozen before selection. The default decay is `0.9`, with zero initialization and no startup bias correction. Trusted zeros enter the history; missing outcomes do not. Current-cycle means affect later batches only. `--proposer-baseline-mode none` remains available.

EMA state and snapshots bind the normalization, decay, Frontier version and alpha. New soft-rule EMA state uses a separate filename and does not silently import hard-gate history. Old frozen batches retain their original scoring and baseline semantics. Incompatible state is rejected rather than reinterpreted. Solver reward, PPO rules and HealthBench reward mapping are unchanged.

## Predicted-time admission

The supplied `configs/adaptive.toml`, `configs/adaptive.example.toml` and `configs/mock.toml` explicitly set:

```toml
[canvas]
remaining_time_admission_enabled = false
```

This bypasses estimated-time checks for adding an Agent and executing subsequent graph edits. Those checks no longer trigger early time-budget consolidation. Actual rollout deadlines and request timeouts still apply; token admission and other structural or environment completion rules remain available.

The library's `CanvasConfig` default remains `true` for compatibility, as in the source project. Custom configurations that omit the flag still enable predicted-time admission. Copy the explicit setting when creating a new deployment configuration. Changing a flag does not retroactively reset a previously saved Canvas that already entered a repair state.
