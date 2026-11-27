# Resume anchor — Switchyard P10 convergence, session close 2026-11-27T15:5x+07

Purpose: a fresh session must be able to continue from this file without re-reading the transcript.
Pair documents (both already written, both measured):
- `/var/lib/turnstone/p10-convergence/LEDGER-20261127.md` — Part I (pre-activation) + Part II (§10-§16, post-activation).
- `/var/lib/turnstone/p10-convergence/receipt-20261127.md` — deployment receipt for the activation.

## 1. Live state as of this close (all values read, not recalled)

| Item | Value |
|---|---|
| CURRENT | `20261127T083505Z-aaf1bec88835acac-9069fe66` |
| Binary | `aaf1bec88835acaca1d9dbff556fb74cbd1647a719f32ddfd3aafd26b6f01dcf` · 31,227,360 B |
| Manifest | `9069fe66ebfefdb7769fd59619dccae8f4be023ba833ecd543baa693c1dc1ea5` · 133,371 B (byte-identical to predecessor) |
| Capabilities | `5312e9307f68d46bd0969ebf15b096347d6741ad2ef7884ee0623e332a0bf67b` · 25,500 B |
| Source | `088e45b6ce2376ee…` on `feat/nemotron-ultra-peer-recovery-20261127`, dirty=false |
| Service | `switchyard-neutral-green.service`, user `switchyard`, PID 846870, `NRestarts=0` |
| Endpoint | `loopback:4000`, `/health` ok, `/v1/models` 74/74 |
| LKG | `20261126T061331Z-65757ec7aaa51166-737a6887` — unchanged (no promotion, by design) |
| Rollback | `20261127T063310Z-fb12d2881387e338-9069fe66` — protected in store before activation; **not** promoted |
| Dashboards | V7 **r90 / V7.89**, V9 **r21 / V9.16** — closeout corrections, both verified by `raw:true` readback; the owner-accepted baselines were V7 r89/V7.88 and V9 r20/V9.15 |

Acceptance evidence for the activation is §10-§12 of the LEDGER. Do not re-run activation; it is complete.

## 2. Closeout state (2026-11-27T09:0xZ) — what is genuinely still open

The owner's terminal directive (`agent-stack-private-development#35#issuecomment-7213861359`) declared
the production engineering workstream accepted and complete. Detail: `TERMINAL-RECEIPT-20261127.md`
and LEDGER Part III.

1. **CLOSED — `head` vs worker 4-byte difference.** It was the MCP bridge's response envelope: the
   receipt echoes `target`, and `worker_2` is exactly four characters longer than `head`. One shared
   PostgreSQL registry, global settings, byte-identical `node_metadata.models` on all five nodes, and
   an envelope length that tracks receipt-string length one-for-one. Not a configuration difference;
   nothing was synchronised. Reopens only if a body delta survives subtraction of the target-name
   length. Method: `PARITY-20261127.md`.
2. **Registry census.** 91 = 69 bound + 22 bypass (2026-11-27T08:50Z). Dated; re-derive only when a
   decision needs a newer number.
3. **16 advertised-but-unregistered routes.** Not bulk-synced; register one at a time only on a
   specific authorized consumer requirement.
4. **Effective headroom.** Export deferred to a separately authorized telemetry packet; exact
   monetary reconciliation blocked on post-anchor accounting. `balance − usage{daily}` is not a valid
   bound in either direction.
5. **Bypasses with no established purpose.** Recorded UNKNOWN; bindings stay regardless.
6. **Package C.** Blocked, preserved at `581e189`, untouched this session.
7. **Owner actions.** PR #57 ready, unmerged; PRs #54/#55/#56 draft. Review order #54 → #55, #56 in
   parallel, #57 independent. No merge without owner GO.
8. **ERNIE original leg.** Unproven after quota reset; observe ordinary workloads only, no probes.

## 2b. Historical open items from the pre-closeout read (kept for the record)

1. ~~`head` node registry 4 B divergence~~ — closed above.
2. ~~Registry census re-derivation~~ — done 2026-11-27T08:50Z.
3. ~~16 advertised-but-unregistered~~ — decision taken: no sync.
4. Headroom — deferred/blocked as above.
5. Package C — unchanged.
6. PR #57 — unchanged (ready, unmerged).

## 3. Constraints that survive the session

- No automated LKG promotion; no PR merge without owner GO.
- Strict hash pinning on every deployment; no rebuild/substitution.
- No paid probes without authorization.
- Monitoring changes must never interrupt serving work (owner rule).
- ComfyNinja/HTPC model-serving endpoints: non-interference.
- Evidence must be a read from the live service or a verified artefact — never reconstructed prose.

## 4. Traps that cost time this session (keep them)

- Bridge truncation ~22.9 KB: bodies near that size are unusable for diffing; sizes are still exact.
- `GET /v1/api/models/<alias>` → 404. `/v1/api/models` is list-only on these nodes.
- Publish path is `/var/lib/turnstone/bin/publish_dashboard.py` (profile `v7`); suffix derived from
  live+1, CAS on `expected_version`, `folder_uid: null`, text panels need both `options.content` and
  legacy `content`. Verify with `raw:true`, never by a 200.
- Grafana launcher: JSON on stdin, `raw:true` for dashboards, closed verb set.
- Activation driver: run exactly the posted command (guarded, serialized, `--capabilities`); it
  protects the rollback target itself and refuses a stale manifest.
