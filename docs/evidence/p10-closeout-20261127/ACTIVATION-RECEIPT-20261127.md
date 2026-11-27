# ACTIVATION RECEIPT - 2026-11-27 (durable copy)

Copied verbatim from `/var/lib/turnstone/p10-convergence/receipt-20261127.md`; the only edits are
credential-path and hostname sanitisation. Fingerprints are unchanged.

---

## ACTIVATION COMPLETE — peer-recovery successor is live; both dashboards converged

Owner GO received (quoting the controlling review `#issuecomment-7213563322`). Command used was exactly the one posted in `RedEyeNinja-BKK/Switchyard#issuecomment-7213448155` — guarded, serialized, hash-pinned, `--capabilities`; no rebuild, no substitution, no stale manifest, no Ultra admission.

### Deployment receipt

| Item | Value |
|---|---|
| New release | `20261127T083505Z-aaf1bec88835acac-9069fe66` |
| Binary | `aaf1bec88835acaca1d9dbff556fb74cbd1647a719f32ddfd3aafd26b6f01dcf` · 31,227,360 B |
| Manifest | `9069fe66ebfefdb7769fd59619dccae8f4be023ba833ecd543baa693c1dc1ea5` · 133,371 B — byte-identical to the predecessor |
| Capabilities | `5312e9307f68d46bd0969ebf15b096347d6741ad2ef7884ee0623e332a0bf67b` · 25,500 B |
| Source | `088e45b6ce2376ee2a65507e0a7805e77e658fb7` · dirty=false · attested `declared-at-deploy` (not reproducible-claimed) |
| Rollback target | `20261127T063310Z-fb12d2881387e338-9069fe66` — **protected in the store before activation** (6 → 7 ids); not promoted to LKG |
| LKG | `20261126T061331Z-65757ec7aaa51166-737a6887` — unchanged |
| Tool chain | `artefacts match the approved hashes` → `P9 PASS direct=54 qualified=54 failures=0` → `PREFLIGHT PASS` (71 families / 3 subsystems) → `health ok after restart` → `installed bytes verified` → `PASS: every always_present family is exposed` → `ACTIVATION COMPLETE` (exit 0) |

### Post-activation readback (independent reads after the tool exited)

- Installed hashes: all three match `release.json` and the authorized fingerprints.
- Service: user `switchyard`, PID **846870**, `NRestarts=0`, active since `2026-11-27T15:35:04+07`.
- Authoritative endpoint `127.0.0.1:4000`: `/health` ok, `/v1/models` = **74**.
- Strict P9 re-read: `direct=54 qualified=54 failures=0`, `manifest_sha256=9069fe66…`, checker `72838c34…`.
- `smart-route`: 4 groups, **11 leaves**, `5/3/1/2`; **no `deepseek-v4-pro` path** (absent from group, candidates and fallback).
- `smart-deepseek-flash`: singleton on `deepseek-deepseek-flash` (`candidates=[]`).
- NT/judge boundaries intact: `smart-nt` → `[[htpc-…-nt],[deepseek-…-flash-nt]]` (`allowance_only`, then `availability`/`quota_only`); `smart-judge` → `[[htpc-…-mtp-nt]]` (`allowance_only`, ctx 12,307).
- Package D typed services present: 6 decision lanes + embedding capability lanes.
- **Ultra held**: desired free group would be 4 leaves (12 = 5/4/1/2); admitted remains 3 (11 = 5/3/1/2).
- Ordinary traffic only, no probes: 3 `smart-route` requests `status=200` on `deepseek-flash` within 25 s, `reasoning_kinds=reasoning_content+reasoning_item`, ctx ≈ 140 k admitted.

### Dashboard receipts (CAS + raw readback, no redesign)

| Dashboard | Revision | Title | Readback |
|---|---|---|---|
| `turnstone-switchyard-v7` | **89** | V7.88 (`switchyard-dashboard-v7-88`) | 12001/12002/12004/12007 byte-identical; 65 panel ids unchanged |
| `turnstone-switchyard-economics-v9` | **20** | V9.15 (`…-v9-15`) | 1047/1093/11007 byte-identical; 98 panel ids unchanged |

Backups written by the runtime before each mutation (`…-v7-v88-20261127T083617Z.json`, `…-economics-v9-v18-…`, `…-v19-20261127T083710Z.json`). V4 Pro is now described narrowly: **excluded from `smart-route`**, with the direct route, `vc-paid-core` and the `VC_Paid` composite explicitly preserved as deliberate-use paths.

### Headroom — measured, and the exact remaining gap

The activated build **does** export the raw inputs: `provider_balance{provider,currency,period,scope}` and `_balance_component`, `provider_credit`, `provider_usage{…}` (daily/weekly/monthly/total), `account_freshness_seconds`, `allowance_remaining_percent`/`allowance_reset_seconds` (subscription pools). Live: deepseek CNY 44.66 (all topped-up, USD 0); openrouter USD 0.219985 vs 125.00 purchased, usage 124.78001; openrouter_stephen USD 15.67356 vs 25.00; openai primary/secondary 100 % with 17,999 s / 454,256 s resets; freshness 0 s × 4.

**Effective headroom remains underivable**: currency spend exists only as period snapshots (no monotonic counter) and the reconciliation anchor timestamp is not exported, so `balance − usage{daily}` is a lower bound. No gauge was invented; nothing renders absence as zero. Exact delivery needs a code change and therefore a release — outside this activation's authorization.

### Registry / bypass — current census (08:37Z)

**85 rows** (was 90 on 2026-11-25) = **63 Switchyard-bound** (58 advertised + 5 registered-only) + **22 provider-direct bypass**. Cross-checks: 74 advertised − 16 unregistered = 58; 58 + 5 = 63; 63 + 22 = 85. 16 advertised-but-unregistered: 9 direct (8 `direct-google-stephen-*`, `direct-openrouter-tencent-hy4-flash`) and 7 non-direct (`ST_Gemini_Free`, `smart-deepseek`, `smart-deepseek-flash`, `smart-deepseek-nt`, `smart-openai-nt`, `smart-openrouter-free`, `smart-openrouter-paid`). Nothing removed or rebound.

All-node parity: `head` returns 22,920 B; `worker_2/3/4/5` each 22,924 B — a real 4-byte divergence I **cannot** attribute, because the bridge truncates the body and the projection has no per-row hash. Recorded as an open item, not guessed.

### Lineage / paperwork

- PR #57 body now opens with an explicit **superseded** notice: the 55-leaf figures (`b24e76ae…`, `d5ab7cda…`, P9 55) are the original candidate fingerprints; only `aaf1bec8… + 9069fe66… + 5312e930…` is current authorization. Historical receipts retained unedited, no merge performed.
- Consolidated ledger updated: `/var/lib/turnstone/p10-convergence/LEDGER-20261127.md` (Part II: activation, verification, dashboards, headroom, census, corrections, remaining).
- Package C remains blocked at `581e189`. Phase 3 only in a real quiet window; Gate 5 interface only; remote OpenClaw media separate. No credentials, privileges, GPU changes, paid probes or LKG promotion.
