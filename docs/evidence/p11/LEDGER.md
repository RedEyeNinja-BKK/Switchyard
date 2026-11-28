# P11 engineering ledger — history & media record

This ledger holds the P11 history and media record so the live dashboard's panel 12007 can stay a **short present-tense snapshot** of the current release without pretending an old release is CURRENT.

## Release lineage (routing-relevant)

| Release | Manifest | Note |
|---|---|---|
| `20261126T061331Z-65757ec7aaa51166-737a6887` | `737a6887…` | LAST-KNOWN-GOOD (unchanged, never promoted) |
| `20261127T045138Z-fb12d2881387e338-2a20067b` | `e887392c…` | 53→54 rows (Tencent Hy4 added) |
| `20261127T091816Z-aaf1bec88835acac-c635222b` | `c635222b…` | **Ultra admitted** — `direct-openrouter-nemotron-3-ultra-free` added; `smart-openrouter-free` became 4 leaves (12 = 5/4/1/2) |
| `20261128T002506Z-6a8209f97101563b-18b377cd` | `18b377cd…` (133,736 B) | **P11 functional cleanup** — removed `vc-paid-core`, `VC_Paid`, `tier1-decision`, `smart-nt-bounded-trial`; `google-decision` reconciled into `smart-gemini-decision`; composites 3→1, decision lanes 5→4 |
| `20261128T020526Z-6a8209f97101563b-79286007` | `79286007…` (133,673 B) | **CURRENT — P11 ultra-out**: Ultra removed from automatic dispatch; 11 = 5/3/1/2 |

## What the Ultra removal changed (2026-11-28)

Exactly two manifest edits vs `18b377cd`: `openrouter-nemotron-3-ultra-free` `groups` `["smart-openrouter-free"] → []`, and removal from `[routes.smart-route] candidates`. The direct alias `direct-openrouter-nemotron-3-ultra-free` is untouched and still advertised. No rebuild; binary `6a8209f9…` unchanged; capabilities capsule `987dd039…` reused (byte-identical binary).

## Correction record

- An earlier release id (`20261128T041533Z-…-00870c53`) was asserted without a deployment; it never existed and is retracted.
- The "4/2/0/1" topology was an inspection error (pool `candidates` counted without the pool `target`); the true pre-change topology was 5/4/1/2 = 12.

## Media record

The PNGs under `docs/evidence/p11/` are **synthetic telemetry illustrations** produced with PIL, not captures. Grafana ingress (`https://grafana.bkk.redeyeninja.com`) requires authentication, so no browser screenshots could be produced from this host; that limitation is explicit and standing.

## Dashboard generations

- V7 `turnstone-switchyard-v7`: generation **101** (V7.99) — current.
- V9 `turnstone-switchyard-economics-v9`: version **24** (V9.17) — description updated; dated P11 correction appended to the historical PICTURE-3 record (panel 11007), history preserved.