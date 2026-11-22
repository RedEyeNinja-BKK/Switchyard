# F1/F2 — LIVE release receipt (2026-11-22)

Paired release of the Stephen ST routing v1 work onto the R6 production line. One
binary and one manifest, installed together through
`/opt/switchyard/bin/activate-release.sh` under Issue #35's sole-writer procedure.

## Release identity

| Field | Value |
|---|---|
| Release | `20261122T145142Z-abd153b1021592d5-5df130e7` |
| `CURRENT` | `20261122T145142Z-abd153b1021592d5-5df130e7` |
| `LAST-KNOWN-GOOD` | `20261122T145142Z-abd153b1021592d5-5df130e7` (promoted 14:52:02Z) |
| Binary | `abd153b1021592d5a4b089b050011d150345f6315ef5fa073e7070943a9e2dea`, 30,836,712 B |
| Manifest | `5df130e75516106379b724b66493774c6ac2414c5f2b0e58110954013ad6ddbe`, 106,411 B, mode 640 group `switchyard` |
| Capability manifest | `9c9b0fdb144bc77c589f9d980b4b166ec1438dbd5b9aec1f04fcc319829f5e81`, 57 instruments / 71 families (16 always_present, 55 event_driven), 3 subsystems |
| Source | `abaacd8a2283c64eaa4be033ea0f1d251fd70847`, branch `feat/st-routing-v1-on-r6-20261122`, `source_dirty=false` |
| Service | `switchyard-neutral-green.service`, active, MainPID **488636**, `NRestarts=0` |
| Health | `GET /health` → 200; `journalctl -p err` since restart → no entries |

The release binary was built from a clean tree at `abaacd8` (commit 21:06:52 local,
binary 21:30:04 local, `git status` clean at build and now). Provenance is recorded
as `attested / declared-at-deploy`; **no rebuild was compared against these bytes**, so
this is a declared identity, not a reproducible-build verification.

Post-deploy commit `4b2b680` is verifier-only (`deploy/stephen-st-routing-v1/*.py`);
it changes no Rust source, so the released binary above is unaffected.

## Before / after

| | Before | After |
|---|---|---|
| Release | `20261122T055957Z-7952200d6583170b-53fd9a84` | `20261122T145142Z-abd153b1021592d5-5df130e7` |
| Binary | `7952200d6583170b…` | `abd153b1021592d5…` |
| Manifest | `53fd9a8443322ccc…` | `5df130e755161063…` |
| MainPID | 458498 | 488636 |

## Gates, all measured

| Gate | Result |
|---|---|
| Deployed-state regression guard (`verify_no_live_regression.py`) | **6/6 PASS** after the allowlist fix below |
| Negative controls (`mutants_live_regression.py`) | **5/5 caught**, control PASS (M1–M4, M6) |
| Dry-run, candidate binary × live manifest, unit's 4 env files | rc=0 |
| Dry-run, candidate binary × candidate manifest, unit's 4 env files | rc=0 |
| Capability preflight vs deployed baseline | **PASS**, 71 families preserved, 0 removed |
| `always_present` families exposed after restart | 16/16 PASS |
| Coupling | old binary REJECTS the candidate manifest (`unknown variant 'peer'`); candidate binary ACCEPTS the live manifest. Both halves must move together. |

### One gate genuinely failed first, and was fixed rather than waived

`verify_no_live_regression.py` refused L4 with:

```
drifted=["smart-nt:['candidates', 'terminal_fallback']", "smart_gemini:['candidates']"]
```

That is the guard working: its route-key allowlist was written for Batch A/B and
predated F1/F2. The two flagged changes were verified to be exactly the reviewed
edits (`smart-nt` drops the held `gemini-2.5-flash` leaf and moves
`deepseek-flash-nt` to `terminal_fallback`; `smart_gemini` drops the same held
leaf), and the allowlist was widened by exactly those keys. A new mutant **M6**
changes `selection` on `smart-nt` — a key allowed on `ST_Free` but not on
`smart-nt` — and is still refused, proving the widened entry is narrow and not a
blanket permit.

## Live acceptance canaries (bounded, synthetic, free lanes only)

| Route | HTTP | Served by | Policy evidence from `routing.decisions.jsonl` |
|---|---:|---|---|
| `ST_Free` | 200 | `baidu/ernie-5.5-flash-preview:free` | peer selection; 7 dispatched legs, all `*:free`; `terminal=None` |
| `smart-nt` | 200 | `gemini-3.5-flash-lite` | `terminal: ['deepseek-flash']`, `terminal_crossed: None`; DeepSeek absent from `considered` and `dispatched` |
| `smart_gemini` | 200 | `gemini-3.5-flash-lite` | `terminal: None`; owner policy unchanged |
| `/v1/decisions` (`smartlocal-decision`) | 200 | `jev-style-0.8b-decision-v3` via `direct-htpc-jev-style` | typed `noul 0.9601975071103299`; local backend, no provider spend |

`ST_Free` pre-deploy records (03:39Z, 05:09Z) dispatched 4 legs **including**
`openrouter/free`; the post-deploy record (14:48Z) dispatches 7 and **excludes**
it. That is the Batch F1 sanctioned removal of `stephen-openrouter-free`, observed
in live routing rather than inferred from the manifest.

The `smart-nt` canary skipped both Gemma legs on the pre-existing output-budget /
input-limit rules — unchanged behaviour, not a new gate.

**No paid crossing occurred.** Every dispatched leg in the canaries was a free
target; `ST_Paid` was not exercised.

## Paired rollback — drilled live, not inferred

| Step | Result |
|---|---|
| `rollback 20261122T055957Z-7952200d6583170b-53fd9a84` | exit 0, `ROLLBACK VERIFIED` |
| Both halves reverted | binary `7952200d…` **and** manifest `53fd9a84…`, byte-verified |
| Restart | MainPID 488493, health 200 |
| Re-deploy of this release | exit 0, `abd153b1…` / `5df130e7…`, MainPID 488636, health 200 |
| `promote` | LKG = `20261122T145142Z-abd153b1021592d5-5df130e7` |

The rollback restores **both** halves. Rolling back only the binary would have left
an old binary facing `selection="peer"`, which the old binary cannot parse.

### A harness limitation, recorded rather than glossed

`/opt/switchyard/bin/test-release-recovery.py` reports 14/22 against **both** the
deployed script and its pre-fix backup (`8e3aa7d6` and `bak-514226c6`), with
`0 restart(s)` recorded. It is pinned to a purpose-built probe copy at
`/var/tmp/st-probe-a1/activate-release.sh` and to `ENV_FILE_1/2` +
`SWITCHYARD_ENV_STUB` seams that neither production version honours (the env-file
fix replaced them with `systemctl` discovery). Its failures therefore say nothing
about production rollback, and its passes would not have been evidence either. It
should be re-pinned to the current seams before it is used as a release gate again.

## Known state after release

- `routing.decisions.jsonl` gained `terminal`, `terminal_crossed` (and
  `terminal_reason` internally) — all `skip_serializing_if` absent when unused, so
  records for routes without a terminal leg are byte-compatible with before.
- The host copy `/opt/switchyard/bin/capabilities.toml` is one family behind the
  release contract (`switchyard.incomplete_answers` missing). The capability
  manifest for this release was generated with `--contract` pointing at the
  release's own `capabilities.toml`. The deployed **binary already emitted** that
  instrument before this release; the baseline manifest simply never declared it.
  The host copy should be refreshed on the next host-side tooling change.
- The two held Gemini 2.5 CHAT entries are out of every candidate list (ranking
  retired), but their `targets`/`routes` definitions remain. Deleting those
  definitions is a separate manifest change with its own guard implications and
  was deliberately not bundled here.
