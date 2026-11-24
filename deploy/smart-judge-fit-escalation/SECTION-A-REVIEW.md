# PR #53 qualification — Section A review (mixed paths, terminal_skipped, negative controls)

Review of commit 72ad384 (plus the A.2 follow-up in this session) against the owner's
narrow review items. All line references are to the worktree at
`/mnt/turnstone-build-ssd/w-fit-esc`, crate `switchyard-runner/src/route.rs` unless noted.

## A.1 — Mixed escalation paths, authorization at every boundary

The walk has two independent claims, and the code keeps them separate:

* `fit_escalation` (new) — the primary pool admitted **nothing** because every candidate
  was excluded on a request-fit fact (`context_window`, `max_input_tokens`,
  `minute_window_too_small`, `output_budget_below_thinking_floor`). Computed once, before
  dispatch, from the pre-dispatch `verdicts` (route.rs:1431).
* `terminal_authorization` (pre-existing) — the primary pool **failed** (dispatched and
  the attempts are capacity outcomes). Re-derived at every boundary from the set the walk
  is leaving (route.rs:1727, 1749, 1786).

The mixed path — primary capacity exhaustion, then a terminal tier that cannot hold the
request — resolves correctly:

1. Pool dispatches, the attempt fails `minute_window`. `fit_escalation` is `None` (the pool
   had eligible candidates), so the entry authorization is
   `terminal_authorization(capable_legs, dispatchable_legs, attempts)` → `primary_pool_exhausted`.
2. Tier 0 is reached. Its targets are excluded on `max_input_tokens`, which lives in
   `capability_gate` (route.rs:1283), so `terminal_capable[0]` is **empty**.
3. The `servable.is_empty()` branch re-derives the next boundary with
   `fit_escalation.filter(tier_cannot_hold)`, which is `None` (no size claim), then
   `terminal_authorization(terminal_capable[0] = ∅, terminal_servable[0] = ∅, …)`.
4. `authorize_terminal` returns `None` on `capable.is_empty()` (route.rs:367). The walk
   stops and the caller receives the primary capacity error. **No silent widening.**

The symmetric case — a size-emptied pool reaching a tier that is excluded for a *non-size*
reason (a rejected credential, a capability) — also stops:

* capability exclusion → `terminal_capable[0]` empty → `capable.is_empty()` → `None`.
* credential exclusion → `terminal_capable[0]` non-empty but `terminal_servable[0]` empty;
  `tier_cannot_hold` is false (code is `credential_unauthorized`, not a fit code), so
  `fit_escalation` drops out, and `authorize_terminal` returns `None` because the excluded
  target is not `quota_blocked` (`quota_blocked` matches only FreeQuota/ModelQuota/
  WholeQuota, route.rs:846, and `CredentialBlockKind::Unauthorized` is deliberately not one
  of them — run.rs:748).

Boundary authorization table:

| boundary being crossed | authorized by | requires |
|---|---|---|
| pool → tier 1 (size) | `fit_escalation` | `route_accepts` AND every primary verdict is a fit code |
| pool → tier 1 (capacity) | `terminal_authorization` | primary attempts all capacity outcomes under the policy |
| tier N → tier N+1 (size) | `fit_escalation` retained | `tier_cannot_hold(terminal_verdicts[N])` — every tier-N target fit-excluded |
| tier N → tier N+1 (capacity) | `terminal_authorization` | tier-N attempts all capacity outcomes; a non-quota block ends it |

Every re-derivation reads the set it is *leaving*, never the union, so "the second tier
looks cheaper" cannot skip the first (route.rs:1722).

## A.2 — `terminal_skipped` cannot be misread as traversal proof

Fixed this session. The field was previously built from the precomputed `terminal_verdicts`
(all tiers, before the walk), which recorded *evaluated* exclusions — a reader could take a
non-empty `terminal_skipped` beside `terminal_crossed: false` as proof a tier was reached
when it was never consulted.

The field is now populated only inside the `servable.is_empty()` branch of the walk
(route.rs:1739), i.e. only for tiers the walk actually reached and found empty. Four states
are now separable:

| state | recorded by |
|---|---|
| evaluated (gate computed) | not recorded |
| reached, found empty | `terminal_skipped` (this target + rule) |
| reached, dispatched | `terminal_crossed` (reason) + `terminal_tier` |
| attempted | `attempts[]` |
| served | `served` |

Two new integration tests pin the distinction from the log the server actually writes:
`the_record_leaves_terminal_skipped_absent_when_tiers_are_never_consulted` and
`the_record_names_every_reached_tier_that_could_not_hold_it`.

## A.3 — Negative controls remain fail-closed

The fit vocabulary is closed to four request-fit codes (`is_request_fit_exclusion`,
route.rs:197); anything else — credential, billing, accounting, policy, capability — ends
the walk. `route_accepts` (route.rs:1425) refuses to escalate a request the route never
promised to serve before any tier is consulted. Malformed bodies fail at request parsing
before route selection.

The pre-existing `smart_judge_terminal` suite already asserts the fail-closed controls, and
my change does not touch that path:
`a_rejected_credential_does_not_spend_the_terminal_leg` (401),
`a_payment_required_does_not_spend_the_terminal_leg` (402),
`an_outage_beside_a_throttle_does_not_spend_the_terminal_leg` (accounting),
`a_healthy_primary_never_spends_the_terminal_leg` (policy),
`both_primaries_unavailable_do_not_spend_the_terminal_leg` (availability).

## A.4 — hash verification

CURRENT release (from PROVENANCE.txt + release.json on switchyard-vm):
- binary sha256 `43a930f52d6697144379bceb5f8879b2e35a2267ec3991d988a2340126944ceb`
- manifest sha256 `f4f7b3b64e7af78165a6a60de8aa6ad304981753274653cc0b48295272cfb97b`
- capabilities sha256 `b5764e3a4179f30b767f9c0b77bbb102d5d51c8d8354d72cf59ec81921109ae2`
- source commit `8c673c13ccb013524e6b6a439cfe1d5d86cc5090`, branch
  `feat/p9-evidence-qualification-20261124`, `source_dirty = false`.

PR #53 HEAD `72ad384`; `git merge-base --is-ancestor 8c673c1 HEAD` is true, and
`git rev-parse 8c673c1` = `8c673c13ccb013524e6b6a439cfe1d5d86cc5090`. PR #53 is based on
exactly the deployed source. Manifest/binary/capabilities hashes are byte-verified against
the deployed `routes.toml` (sha256 f4f7b3b6…).
