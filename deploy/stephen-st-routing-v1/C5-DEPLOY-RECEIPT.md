# C5 receipt — Stephen's Google free lane, and the unified free pool

Deployed 2026-11-22T21:47:31Z by `activate-release.sh deploy` (config-only; binary unchanged).

## What changed

| Artefact | Before | After |
|---|---|---|
| release | `20261122T190705Z-684df16fda5c0fe9-e008aa96` | `20261122T214731Z-684df16fda5c0fe9-0fdabb3b` |
| routes.toml sha256 | `e008aa96…` (111,358 B) | `0fdabb3bc1192c8b400cd5cee33fc4bda819d631b6f1a2450e92f4094be8b95b` (117,408 B) |
| binary sha256 | `684df16f…` | **unchanged** `684df16fda5c0fe95516551892ed14aa6e83b79e7f6bd7565204361ec3e0a487` |
| capabilities | 71 families, 3 subsystems | **identical** (regenerated; see below) |
| PID | — | 512051 |
| LAST-KNOWN-GOOD | `20261122T180436Z-3c3e2589…-862afd23` | unchanged (no `promote`) |

Manifest delta: 195 lines added, 1 replaced. Everything added is inside the Stephen
scope; every pre-existing target block is byte-identical and only `routes.ST_Free`
changes among existing routes.

## The credential gap was a binding gap

`SWITCHYARD_GEMINI_STEPHEN_API_KEY` was already installed
(`/etc/switchyard-gemini-secrets/stephen.env`, root:root 0600, one variable) and was
referenced by **zero** lines of the live manifest. C5 was therefore an unused-binding
and qualification problem, not a missing-secret one. Nothing was requested, created or
replaced.

## Qualification (single bounded session, 2 of 3 requests, no retries)

Wire parity with the deployed `[llm_clients.google_gemini]`: `format = "openai_chat"`,
`POST https://generativelanguage.googleapis.com/v1beta/openai/chat/completions`,
`Authorization: Bearer`, `max_completion_tokens`, `stream = false`.

| attempt | model | outcome | latency |
|---|---|---|---|
| 1 | `gemma-4-26b-a4b-it` | 503 `UNAVAILABLE`, "high demand" | 1,006 ms |
| 2 | `gemini-3.5-flash-lite` | **200**, echo `gemini-3.5-flash-lite`, `finish_reason = stop`, usage 8/1/9, content `ready` | 1,500.6 ms |

A 503 is provider-side demand, not a credential verdict, so the Gemma models are staged
(unreferenced targets) rather than marked `held` — `held` would assert unavailability,
which was not measured.

## Deployed shape

* `[llm_clients.google_gemini_stephen]`, `[llm_clients.google_gemma_31b_stephen]` —
  `api_key_env = "SWITCHYARD_GEMINI_STEPHEN_API_KEY"`,
  `quota_group = "google-ai-studio-free-stephen"`, `max_retries = 0`.
* Ten `[targets."stephen-google-*"]` blocks rendered from the catalogue, so a quirk
  corrected for one profile reaches both. Nine carry no group and no route: inert.
* `[routes.ST_Gemini_Free]` — `passthrough`, `selection = "peer"`,
  `pool_groups = ["ST_Gemini_Free"]`, one member, no candidate list (peer selection
  derives the pool from the group; an explicit empty array would be a second, empty
  membership list).
* `routes.ST_Free` — `pool_groups = ["ST_OR_Free", "ST_Gemini_Free"]` and one appended
  candidate. The unified free pool now spans two providers and two credentials.

## Post-deploy measurements

| check | result |
|---|---|
| `/v1/models` | 58 entries, `ST_Gemini_Free` present |
| `ST_Gemini_Free` call | served `gemini-3.5-flash-lite`, 1,191 ms, `considered` = exactly one member |
| `ST_Free` call | served `gemini-3.5-flash-lite`, 946 ms (pool admitted the new member) |
| `ST_Route` call | served `gemini-3.5-flash-lite`, 894 ms — composite preserved and serving |
| quota isolation | `google-ai-studio-free-stephen-gemini-3.5-flash-lite-a5c177a7…` created as `switchyard:switchyard`, `spent 1 / 500`; the owner's `google-ai-studio-free-…-8dbe02cd…` reads `spent 500 / 500` |
| release verify | binary ok, manifest ok, service healthy |

The quota comparison is the point of the whole change: the owner's lane was **exhausted**
at 500/500 for the day, and the Stephen lane had 499 requests of independent headroom.
Two free-tier accounts are now genuinely additive rather than a second name for one
budget.

## Gate integrity

* `release_validate.py` dry-run with the deployed binary under the unit's four
  `EnvironmentFile`s → validates (`ST_Free, ST_Gemini_Free, ST_Paid, …`).
* Capability manifest regenerated from source `13cefe6a` + the deployed binary: all
  eight dimensions identical to the deployed baseline (57 instruments, 71 families, 16
  always-present, 55 event-driven, 3 subsystems, 8 declared-kind). The deployed
  baseline file was reused as the release artefact because the binary is byte-identical.
  `preflight` PASS both ways; post-deploy `verify-live` PASS (16/16 always-present).
* `_assert_scope` in the applier refuses any change outside `stephen-google-*`, the two
  new clients, `routes.ST_Gemini_Free` and `routes.ST_Free`'s `pool_groups`/`candidates`.

## Verification defects found and fixed

* **P4 could not fail.** `check_qualification_gate` sliced the rendered fragment at
  `"# Staged members"`, a banner near the *top* of the file, so the slice was a few
  comment lines and no member name could appear in it. Mutation PM6 survived and exposed
  it. Rewritten to parse the fragment as TOML and compare the route's `target` and
  `candidates` against the servable set, plus a one-flag flip test.
* **P6 had drifted from production.** Re-anchoring the parity suite to the deployed bytes
  found four real divergences in the catalogue — two `omit_body_fields` sets missing
  `temperature`, an inverted `state_forms` declaration, and a retired `gemini-2.5-flash`
  still listed as an owner candidate. All four corrected against the deployed manifest.
  Rendering a delta from the stale capture would have silently reverted those production
  fixes.

## Observation, not caused by this change

`smart-judge` is failing at a steady ~91% (1,980 of 2,183 records before this deploy;
25 of 26 after). The failure record shows `gemma-4-31b-it` answering `unavailable`,
`gemma-4-26b-a4b-it` hitting `minute_window`, and no terminal leg rescuing the call. The
rate is identical on both sides of the restart, so this deploy neither caused nor
worsened it. Recorded here because it is visible in the same decision log; not touched.
