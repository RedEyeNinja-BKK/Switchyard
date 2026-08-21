# POLICY-CONSUME-SHADOW — Live Fact → Shadow Policy Composition

**Status: SHADOW / OBSERVATION ONLY. ZERO ROUTING AUTHORITY. NOT deploy-authorized.**
**Base:** FACT-3A `f4db51f83b25715105d512efb20d1d25b1937619` (on the REALIGN-2 lineage).
**Branch:** `feature/layer8c-policy-consume-shadow`

## What this is
The end-to-end (one-way) shadow composition:
```
existing provider fetch → SharedResourceTelemetry → fact projection → OpenAiFacts
  → FACT-3A shadow policy observer → OpenAiDailyTranchePolicy
```
plus four bounded factual/accounting corrections discovered in independent source review.

## Corrections applied (were part of this gate)
1. **TTL source-of-truth** — removed the mirrored `AUTHORITATIVE_REFRESH_TTL_SECONDS`
   constant. `SharedResourceTelemetry` now carries the EFFECTIVE `refresh_ttl` from the
   resource-pool config (`ttl_seconds`); the fact plane reads it as a parameter. Config is
   the authority; no drift. (Proven: changing the effective TTL changes freshness, no code
   change.)
2. **Provider-week continuity** — positively known only when `Some(X)→Some(X)`. Any
   None-involving transition (None→None / None→Some / Some→None) and X→Y are UNKNOWN/CHANGED,
   never silently continuous.
3. **Bangkok day-boundary baseline** — a prior-day observation anchors the new day ONLY when
   it brackets the Bangkok-midnight boundary under a documented near-boundary rule
   (`POLICY_BOUNDARY_BRACKET_SECS`). No observation near midnight → UNKNOWN/UNINITIALIZED
   (never a false 14-pt allocation).
4. **Non-monotonic same-window usage** — `current < prior` in a positively-known week is
   DISCONTINUITY (continuity degraded / UNKNOWN), NOT clamp-to-zero. Invalid/non-finite/
   out-of-range percent → UNKNOWN.
5. **Freshness recovery** — a later fresh continuous observation in the same positively-known
   week + same day + valid baseline + monotonic usage restores accounting to `Ok`.
6. **Terminology** — `OrdinaryCloudPhase` describes POLICY PERMISSION (OpenaiOrdinaryPermitted /
   ConserveOpenaiDeepSeekContinuation), not provider health.
7. **Safe adapter** — `policy_shadow::admit_from_openai_facts(&OpenAiFacts)` derives
   admissibility from the factual contract (health + freshness + valid %), not an arbitrary
   caller bool. `assess()` no longer takes `freshness_ok: bool`.
8. **One-way composition** — `ServerState::shadow_openai_tranche_policy()` + read-only
   `GET /v1/shadow/openai-tranche`. No policy field is written back into facts.
9. **Deduplication** — ObservationId = (last_success_at, weekly_used_percent, provider week);
   an unchanged provider observation is not a new accounting event (repeated polling cannot
   manufacture accounting history).

## Separation (unchanged)
- FACT PLANE ≠ POLICY PLANE (this module consumes facts, never fetches/writes).
- POLICY ≠ ROUTING (zero candidate/selection/fallback authority; unread by routing).
- POLICY ≠ CAPABILITY (thinking/non-thinking untouched).
- Provision of ordinary-CLOUD phase only; local participation invariant preserved; PINNED
  caller intent never silently rewritten.

## Trust / observation boundary
`GET /v1/shadow/openai-tranche` inherits the previously recorded Switchyard-listener trust
boundary and is NOT deploy-authorized as-is (a later protection decision is required before
any deployment). No credentials. Observation identity (dedup) drives accounting, not polling
frequency.

## Tests / build
- `fact_plane` **14/14** · `policy_shadow` **20/20** · full `switchyard-server` **113/113**.
- `cargo check -p switchyard-server` OK.

## NOT included
No RE-1, no native-routing replacement, no ResourceRouter change (only TTL metadata additive),
no provider fetcher added, no GPU workload, no deploy, no restart. `ts-018` PARKED.

**DISPOSABLE-ish: shadow source only. Must NOT be deployed or promoted. Awaiting operator review.**
