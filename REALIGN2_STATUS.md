# REALIGN-2 — Provider/Inference Fact Plane (SHADOW IMPLEMENTATION)

**Status: SHADOW-ONLY / ZERO ROUTING AUTHORITY. NOT a production routing feature.**
**Base:** deployed LocalClaw source `00899cddb323f6ca67a88077ede5748aa39cb08e` (authoritative).
**Branch:** `feature/layer8c-realign2-fact-plane`

## What this is
The DYNAMIC factual half of the future inference-routing architecture, built on the
current deployed lineage, with ZERO routing authority. It establishes **trustworthy facts**
for shadow observation — it does NOT finish routing.

## What it contains
- `crates/switchyard-server/src/fact_plane.rs` (NEW): a generic, shadow-only inference-fact
  plane. Generic provenance/health/freshness envelope + typed provider payloads (OpenAI,
  DeepSeek, Comfy), projected 1:1 from the EXISTING authoritative facts. `UNKNOWN` /
  `Unavailable` / `Stale` are first-class; failed reads FAIL CLOSED (never favorable).
- `crates/switchyard-server/src/lib.rs` (+30): registers `pub mod fact_plane;`, adds the
  read-only `ServerState::fact_plane()` accessor, and adds the read-only
  `GET /v1/fact-plane` observability endpoint.

## Reuse-first (no second fetchers)
- OpenAI + DeepSeek facts = projection of the EXISTING `ResourceSnapshot` (from
  `resource_fetcher.rs` + `ResourceState`), read via the already-published
  `SharedResourceTelemetry`. NO re-fetch.
- Comfy facts + continuity = projection of the EXISTING Gate-C `ComfyResourceState` /
  `ComfyCompositionStatus`. NO second telemetry client.

## Zero routing authority (structural + tested)
- The module imports ONLY factual domain types (no `Candidate`/`WorkClass`/
  `ReasoningPolicy`/`Algorithm`/selection). It has no selection API.
- Explicit invariant test: `fact_refresh_changes_shadow_state_but_not_routing_result`
  proves changing facts changes ONLY the shadow output, never a routing result.
- No route/candidate/fallback/model/alias change. ResourceRouter untouched. Gate C untouched.

## Tests / build
- `cargo test -p switchyard-server --lib fact_plane` → **10 passed, 0 failed**
  (mapping without loss ×3, UNKNOWN/unavailable, fail-closed, stale-vs-healthy,
  field-isolation, continuity, routing-authority invariant ×2).
- `cargo test -p switchyard-server --lib` → **89 passed, 0 failed** (no regression).
- `cargo check -p switchyard-server` → OK.

## Deliberately NOT included (REALIGN-2 scope)
No HTPC producer (schema headroom only), no embedding/reranking, no future policy
(`daily_tranche_open`, `~5 CNY/day` are NOT fact-plane rules), NO daily-spend derivation
from balance deltas, no RE-1 readiness estimator, no cloud-native routing replacement.

**NOT deployed. NOT a routing authority. Awaiting operator review.**
