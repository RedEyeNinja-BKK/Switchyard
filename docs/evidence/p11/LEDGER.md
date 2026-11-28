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
## Gemini Lite withdrawal (2026-11-28)

Owner directive: remove `direct-google-gemini-3.5-flash-lite` and `direct-google-gemini-3.1-flash-lite` from automatic `smart-route` dispatch.

| Release | Manifest | Note |
|---|---|---|
| `20261128T034335Z-6a8209f97101563b-a7ef9174` | `a7ef9174…` (133,545 B) | **CURRENT - gemini-lite-out**: both Lite leaves out of the ladder; `smart-route` now 9 = 3/3/1/2; both direct aliases and `smart-gemini-bounded` membership preserved |

Exactly four semantic lines differ from `79286007`: the `groups` line of each Lite target (`["smart-gemini-bounded", "smart-gemini-chat"]` → `["smart-gemini-bounded"]`) and the two candidate entries removed from `[routes.smart-route]`. The delta was proved by a full parsed-document diff whose differing paths were asserted to be exactly those three keys; any additional path refuses the build.

`smart-gemini-bounded` still has all four members (`direct-google-gemini-3.5-flash-lite`, `direct-google-gemini-3.1-flash-lite`, `direct-google-gemma-4-31b-it`, `direct-google-gemma-4-26b-a4b-it`); the `smart-gemini-chat` pool is `direct-google-gemini-4-hydrogen`, `direct-google-gemini-3.8-flash`, `direct-google-gemini-3.6-flash`. The bounded lane therefore keeps serving the two Lite models; only the automatic chat ladder changed.

Deployment: binary `6a8209f9…` unchanged (no rebuild), capabilities capsule `987dd039…` reused, predecessor `79286007` protected in the store, `LAST-KNOWN-GOOD` `20261126T061331Z-…-737a6887` untouched. P9 gate: `direct=55 qualified=55 failures=0`. `/v1/models` 74 of 74 advertised after restart (PID 879484).

## Incident record - in-place store mutation and repair (2026-11-28)

An earlier attempt copied a candidate manifest directly over `CURRENT/routes.toml` at 03:20:26Z without creating a release. That candidate also carried a defect: both Lite targets were set to `groups = []`, which would have removed them from `smart-gemini-bounded` as well - collateral to the owner's instruction.

What was and was not affected:

- `/etc/switchyard/routes.toml`, the live manifest, was never touched and remained `79286007…` throughout (mtime 02:05:25Z).
- The running process never read the mutated copy: the unit declares no `ExecReload=`, and PID 876990 had started at 02:05:25Z. No reload occurred.
- The store copy was restored byte-exactly from the live manifest at 03:41Z; `activate-release.sh verify` then reported `manifest 79286007… ok`, `service healthy`.
- No release was created by that attempt, and no release id beyond the ones listed above exists.

The defect was corrected before deployment: the shipped candidate keeps `groups = ["smart-gemini-bounded"]`. `activate-release.sh` would in fact have refused the mutated baseline anyway, because its rollback-target integrity check compares the stored manifest against `release.json`.

## Dashboard generations

- V7 `turnstone-switchyard-v7`: generation **102** (title V7.100, revision 105). Panels 12001 (both Lite rows retained), 12002 (topology 9 = 3/3/1/2 and the Lite withdrawal note), 12004 (release reference) and 12007 (present-tense release snapshot, rollback target `020526Z`) updated; all earlier P11 corrections preserved.
- V9 `turnstone-switchyard-economics-v9`: version **25** (title V9.18). A dated `CORRECTION 2026-11-28` block was appended to the historical PICTURE-3 record (panel 11007); the prior text is unedited.

## Generator synchronisation (2026-11-28)

`scripts/economics/v7/correct_dashboard.py` now emits the published V7.100 configuration: title `SWITCHYARD DASHBOARD V7.100`, revision 105, the published description, and the published 12002/12004/12007 texts. The normalisation chain gained the `a7ef9174` token mappings so 12001 lands on the current release, size and read time.

Reproduction check: 64 of 65 panels are byte-identical to `v7-rev105-server-export.json`. The one intentional divergence is panel 12001's Ultra paragraph, which still reads `11 = 5/3/1/2` and `11-candidate` in the published document; the generator renders `9 = 3/3/1/2` and `9-candidate`. The live panel was not corrected because no live dashboard change was authorised in this pass.

The export file is named for the server revision (`v7-rev105-server-export.json`), not for a generation counter, because the revision is the number Grafana assigns and the number the write path checks.
