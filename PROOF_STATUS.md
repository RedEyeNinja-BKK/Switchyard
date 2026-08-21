# REALIGN-1 - Inference Fact Plane + Native Consumption-Seam PROOF (DISPOSABLE)

**Status: ARCHITECTURAL PROOF ONLY. NON-PRODUCTION. DO NOT BUILD/MAKE LIVE.**

## What this branch proves
A generic inference-fact seam can feed a **stock native Switchyard** routing
composition (`FallThrough<State>` + `Processor` + `Classifier`) WITHOUT a custom
routing engine - the central REALIGN-1 question.

Changes are **test-only** (a `#[cfg(test)] mod fact_plane_proof` appended to
`crates/libsy/src/algorithms/fall_through.rs`). No production code path is added
or altered. The module is guarded by `#[cfg(test)]`, so shipping builds are
unaffected.

## Evidence
- Base: pinned upstream `c7beccd4891fa5cfe3a3b94fdd376f5765864507` (pure stock).
- `cargo test -p switchyard-libsy --lib fact_plane_proof` → **4 passed, 0 failed**
  (healthy-preference / factual-exhaustion fail-closed / budget-pressure-as-preference
  / qualification+readiness gating).
- `cargo test -p switchyard-libsy --lib` → **263 passed, 0 failed** (no regression).

## What the proof demonstrates
1. A generic `Processor<FactState>` folds external facts (allowance/balance/readiness)
   into composition state per request - provider-agnostic.
2. A generic `Classifier<FactState>` reads the facts and scores targets via the SAME
   native trait stage/llm_class use.
3. Native `argmax` selects + native `RoutingOutcome` is emitted - no custom router.
4. **Pacing is modeled as preference (pressure), NOT hard eligibility** (scenario C).
5. **Qualification ≠ Readiness** (scenario D): ready-but-unqualified / qualified-but-not-ready
   are excluded; only qualified+ready is selected.

## Deliberately NOT done (REALIGN-1 scope)
- No production fact producer, no live fetch, no routes.toml / unit / credential change.
- No readiness estimator (RE-1) - parked.
- No ResourceRouter change (frozen).
- No provider-specific router ("OpenAiBudgetRouter" etc.) - intentionally absent.

## Base / build / test
- base: `c7beccd4891fa5cfe3a3b94fdd376f5765864507`
- proof branch: `proof/layer8c-fact-plane`
- build: `cargo check -p switchyard-libsy --lib` OK
- test: 4/4 proof PASS; 263/263 libsy PASS

**This is disposable architectural proof. It must NOT be deployed or promoted.
If it outlives its purpose, delete the branch + prune the worktree.**
