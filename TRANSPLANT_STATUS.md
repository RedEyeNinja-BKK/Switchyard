# Layer 8C - Upstream-Alignment Transplant Rehearsal (DISPOSABLE)

**Purpose:** Rehearse porting the LocalClaw Gate-C + extension series onto pinned
upstream HEAD `c7beccd4891fa5cfe3a3b94fdd376f5765864507`.

**Status: INTENTIONALLY NON-BUILDING - transplant evidence only.**
This branch is a **disposable rehearsal artifact**, NOT a deployable target.

## What this branch demonstrates
- The **4 Gate-C Comfy commits cherry-pick cleanly** onto current upstream
  (zero conflicts; `crates/libsy/src/algorithms/comfy.rs` + `switchyard-server/src/comfy.rs`
  compile with 0 errors on the new base). **Gate-C is portable.**
- The **LocalClaw smart-resource-router extension series** (`resource.rs` + `resource_fetcher.rs`)
  requires a **semantic port**, not a mechanical one, because upstream changed the core
  `Algorithm` trait:
  - `route(...) -> Result<Response>` + `driver.decide(Decision)` (v0.2.0-era)
    became `route(...) -> Result<RoutingOutcome>` (+ `RoutingOutcome::route_to`, removed `decide`).
  - Build failure centers on `crates/libsy/src/algorithms/resource.rs`
    (errors E0053 route-trait-mismatch, E0599 decide-not-found, E0432 Decision import, field/method gaps).
- Recurring **intra-series self-conflicts in `resource.rs`** (upstream has no such file) mean the
  LocalClaw routing series is not individually-replayable commit-by-commit onto a new base;
  it would need selective folding/squash + a `RoutingOutcome` re-expression.

## Rehearsal evidence
- Branch base: `c7beccd4891fa5cfe3a3b94fdd376f5765864507` (pinned upstream @ investigation start)
- Transplanted commits: the full 20-commit LocalClaw delta (`a0f1aacc..00899cdd`)
- Build: `cargo check --workspace` fails in `switchyard-libsy` on `resource.rs` (root semantic conflict above)

## NOT for production
No deployment, no routes.toml/unit/credential change, no live switchyard interaction.
Recovery: delete this branch remotely + prune the worktree.
