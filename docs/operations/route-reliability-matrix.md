# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Route reliability matrix

What each routing class promises, and where that promise is tested.

This page exists because a deployment is not validated by a representative route
succeeding. A route is a contract, and a contract is only real if something fails
when it breaks. The table below is deliberately divided into **tested** and **not
tested**, because the second column is the honest one and it is what determines
what may be claimed.

Run the counts with:

```bash
cargo test --workspace --locked
python3 scripts/release/route_audit.py <manifest> --models-url http://127.0.0.1:4000
```

## Classes and their contracts

| Class | What it promises | Tested where |
|---|---|---|
| direct leaf (`direct-*`) | one upstream model, no fallback; declared capabilities are the route's capabilities | `switchyard-server/tests/whole_route.rs` (route resolution, unknown route, capability refusal), `route_audit.py` (advertisement agrees with manifest) |
| leaf with candidates | ordered fallback inside one route; a candidate is only tried when the failure says another target may succeed | `whole_route.rs` (child-internal fallback completes before the parent crosses), `libsy-llm-client` (retry and failure classification) |
| primitive (`smart*`) | ordered target plus candidates; a `-bounded` variant pins reasoning off | `route_audit.py` (bounded contract), `libsy-llm-client` (a bounded route sends `reasoning.enabled=false`) |
| tier (`ST_*`, `Tier_*`) | ordered fallback within one credential | `whole_route.rs` (tier with no eligible child is a request error) |
| composite | children in declared order; a child that cannot serve is skipped without a provider call; a terminal failure does not cross | `whole_route.rs` — pristine request per child, skip without a call, cross and no-cross, composition telemetry, claiming-children conflict |
| any route, both wires | Chat Completions, Responses and Anthropic translate in both directions, streaming and not | `switchyard-translation` — 88 request, 23 response, 61 stream cases |

Capabilities enforced at load and at admission: `video` requires an
`openai_responses` client; audio is refused everywhere; video is admitted only
where the route declares it; `vision` governs images only.

## Not tested, and therefore not claimed

| Contract | State |
|---|---|
| **Cross-provider reasoning continuity** | **Not enforced.** `switchyard_translation::continuity` defines the contract and three fixtures demonstrate the defect; nothing consults it yet. The contract is prepared in #24 and is not deployed. |
| Leaf-route context admission | **Absent from deployed behaviour.** A composite skips a child that cannot hold the request; a leaf is dispatched at whatever size the caller sends. A report-only shadow evaluator is prepared in #23; it is neither merged nor deployed, and no enforcement exists. |
| Session persistence and compaction | Owned by the calling agent, not by this repository. No test here can assert it. |
| Live provider behaviour for every advertised route | Only the budget-guarded harness exists; **no route has been smoke-tested live** under this programme. |
| Bounded contract on `smartfree-bounded` | **Fails today** on the deployed manifest: the route pins `route_reasoning_policy = "inherit"`, so `tier1-bounded` and `tier2-bounded` contain an unbounded leg. The wire test in #22 passes because it asserts the contract; the deployment does not satisfy it. A one-line config correction is prepared and verified, not applied. |

## What is actually deployed

As of 2026-11-08T19:59:51Z, from merged `main` `f8ed776`:

| Item | Value |
|---|---|
| binary | `99b9447e9debd2ef...` (27,618,840 bytes) |
| manifest | `038d3862b32a2f53...` (36,553 bytes, unchanged by the binary release) |
| advertised routes | 48 leaves; the 5 composites are internal and not advertised |
| observability | S0 request correlation is live: `parent_route`, `child_route`, `crossed_child`, `cross_child_reason`, `session_id`, `correlation_id`, `upstream_request_id`, `error_class`, `handling_duration_ms` |

S0 does not fix the generic upstream 400. OpenRouter's generic error response
carries no request identifier, so `upstream_request_id` has nothing to report for
that failure and the original incident remains unattributed by this field. The
observability work makes failures that carry an identifier attributable; it does
not make a provider explain an error it did not identify.

## What "passing" does not mean

A `200` with empty content is not a successful model call. A reasoning-mandatory
model can spend a small output budget entirely on thinking and return nothing a
caller can use, and `scripts/release/live_smoke.py` counts that as a failure
rather than a pass. No test in this repository treats a bare `200` as evidence
that a route works.

## The two known failure shapes

1. **Provider-native reasoning replayed to another provider.** A composite may
   send one provider's opaque reasoning to a different provider, including
   mid-tool-cycle. Demonstrated by `switchyard-translation/tests/provider_continuity.rs`
   (two fixtures currently red by design).
2. **A Responses-native history becoming an invalid Chat request.** A caller
   emitting a reasoning item where the Chat wire expects a message role produces a
   request this router rejects locally. The emitter is not inside this repository;
   the rejection is correct, and the diagnostic fields added in the observability
   work are what will identify the caller.
