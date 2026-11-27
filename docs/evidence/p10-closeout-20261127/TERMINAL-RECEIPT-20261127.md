# TERMINAL RECEIPT — Switchyard workstream closeout, 2026-11-27

Controlling directive: `RedEyeNinja-BKK/agent-stack-private-development#35#issuecomment-7213861359`
(2026-11-27T08:44:10Z, owner). This receipt answers it point by point. Every number below was read
from `switchyard-vm`, the local Turnstone database, or a named artefact in this session; nothing is
reconstructed from memory.

## 1. Accepted production baseline — preserved, unchanged by this packet

| Item | Value |
|---|---|
| CURRENT | `20261127T083505Z-aaf1bec88835acac-9069fe66` |
| Binary | `aaf1bec88835acaca1d9dbff556fb74cbd1647a719f32ddfd3aafd26b6f01dcf` · 31,227,360 B |
| Manifest | `9069fe66ebfefdb7769fd59619dccae8f4be023ba833ecd543baa693c1dc1ea5` · 133,371 B |
| Capabilities | `5312e9307f68d46bd0969ebf15b096347d6741ad2ef7884ee0623e332a0bf67b` · 25,500 B |
| Source | `088e45b6ce2376ee2a65507e0a7805e77e658fb7`, `feat/nemotron-ultra-peer-recovery-20261127`, dirty=false |
| Service | `switchyard-neutral-green.service`, PID 846870, `NRestarts=0`, `loopback:4000`, `/health` ok, 74/74 advertised |
| LKG | `20261126T061331Z-65757ec7aaa51166-737a6887` — unchanged, not promoted |
| Rollback | `20261127T063310Z-fb12d2881387e338-9069fe66`, protected in the store before activation |

No build, deployment, selector change, routing change or LKG promotion was performed in this packet.

## 2. Accepted dashboards — and two factual corrections made under the existing grant

| Dashboard | Accepted at closeout | After correction in this packet | Backup written by the runtime |
|---|---|---|---|
| `turnstone-switchyard-v7` | V7.88 / r89 | **V7.89 / r90** | `turnstone-switchyard-v7-v89-20261127T085227Z.json` |
| `turnstone-switchyard-economics-v9` | V9.15 / r20 | **V9.16 / r21** | `turnstone-switchyard-economics-v9-v20-20261127T085244Z.json` |

**V7 12004** (the only panel touched; 4,070 → 4,930 chars, geometry unchanged at h=44 w=12):
the registry census is re-derived from the 2026-11-27T08:50Z reads instead of the stale 2026-11-25
export, and the node-parity statement is replaced by the measured finding in §4. Panel 50007 shows
as "changed" in a raw diff only because it is 12004's parent row and owns it; no other panel changed.

**V9 1047 and 1093**: the phrase asserting `balance − usage{period="daily"}` was a **lower bound** is
removed and replaced with the measured limitation — the balance is already net of spend at
reconciliation, so the subtraction can double-count or omit spend and the direction of the error is
not established. No figures, tables or metrics were altered. `lower bound` now occurs 0 times in both
panels.

Verification for both writes: guarded publisher with CAS on the live revision, then a `raw:true`
re-read — title, description and suffix agree; every panel byte-identical to the payload; both
`options.content` and the legacy `content` mirror equal; panel-id sets unchanged (65 / 98).

> [!NOTE]
> **Version offset, stated rather than hidden.** Live V7 was already `revision 89 / title V7.88`
> before this packet. `publish_dashboard.py` derives the next title suffix from the live *title*
> (`88 + 1 = 89`), so the correction lands as revision 90 titled V7.89 — the one-revision offset is
> inherited, not introduced here. Repairing it would need an out-of-band metadata write or a tool
> change; both are outside this packet's authority. Content identity was therefore verified by raw
> readback, not by the version number.

## 3. Registry preservation decisions (owner, executed as documentation only)

- **22 provider-direct bypass aliases — RETAINED.** Not migrated, removed or rebound. No row was
  touched in this packet. Recorded per alias in `REGISTRY-BYPASS-20261127.md`: provider, host,
  credential class, lifetime calls, pinned-workstream references, and `UNKNOWN` where no evidence
  establishes a purpose.
- **16 advertised-but-unregistered routes — NOT bulk-synced.** Advertisement does not require
  registration. The list is recorded with the rule that a name is registered only when an
  authorized consumer needs it.
- **5 registered-only aliases — PRESERVED**: `ST_Route`, `VC_Paid`, `ST_Rerank`,
  `smartfree-rerank`, `smartlocal-rerank`.
- **Census of record (2026-11-27T08:50Z)**: 91 registered definitions = 69 Switchyard-bound + 22
  bypass. Of the 69: 58 advertised + 5 registered-only + 6 typed-decision lanes. Cross-check:
  74 advertised − 16 unregistered = 58; 58 + 5 + 6 = 69; 69 + 22 = 91.

## 4. Read-only parity comparison — result

The previously reported 4-byte difference (`head` 22,920 B vs `worker_2…5` 22,924 B on
`GET /v1/api/models`) is **not a configuration discrepancy and not a serialization difference in the
registry — it is an artefact of the measurement wrapper.**

Evidence, in the order it was obtained:

1. **Storage is single-source.** All five nodes run the same `TURNSTONE_CONFIG` with one PostgreSQL
   DSN; `model_definitions` is one row set (91 rows) read once per process. There is no per-node
   registry table to diverge. Read with `SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY`.
2. **Settings are global.** All 17 `system_settings` rows carry `node_id = ''`; the five fields the
   endpoint adds beyond the registry (`default_alias`, `channel_default_alias`, `judge_default_alias`,
   `stt_default_alias`, `tts_default_alias`) therefore resolve identically on every node.
3. **Per-node self-reports are byte-identical.** `node_metadata` key `models`: 9,879 B,
   `sha256 cd05d5fcf2a88c4d…`, 91 entries — the same value on `turnstone-head-1` and
   `turnstone-node-2..5`.
4. **The delta is the target name.** The MCP bridge (`/opt/turnstone-integrations/self-admin/server.py`,
   `_emit`) wraps every response as `{"status":…,"receipt":…,"body":…}` and reports `length` =
   `len(envelope)` when it truncates above 18,000 chars. The receipt echoes `target`; `"worker_2"` is
   exactly four characters longer than `"head"` — the observed delta, to the byte.
5. **Control.** The same endpoint with an extended path reproduces the effect one-for-one:
   `head` `/v1/api/_internal/model-status` = 26,790 B; adding `?alias=smart` (12 chars) = 26,802 B;
   adding `?limit=5&offset=0` (17 chars) = 26,807 B. The receipt `path` field is the only difference;
   the body is unchanged (the handler ignores query parameters).
6. **Same pattern on a second endpoint** (`/v1/api/_internal/model-status`): 26,790 B on `head` vs
   26,794 B on `worker_2` — again exactly 4.

**Disposition: no row, no field, and no node was changed.** The exact per-node bodies cannot be
read in full through existing access (bridge truncates at 18,000 chars and returns no partial body;
nodes require the root-only admin token; `GET /v1/api/models/<alias>` does not exist — 404 measured).
That limitation does not affect the conclusion: the shared single-row storage, the identical
self-reports and the arithmetic identity of the delta together establish that there is no
configuration difference to reconcile.

## 5. Headroom — deferred and blocked, as directed

- **Raw telemetry accepted and unchanged**: `provider_balance{provider,currency,period,scope}` and
  `_balance_component`, `provider_credit`, `provider_usage{daily,weekly,monthly,total}`,
  `account_freshness_seconds`, `allowance_remaining_percent` / `allowance_reset_seconds`,
  `calendar_*`.
- **Effective request-headroom export — DEFERRED** until a separately authorized telemetry
  maintenance packet. No candidate was staged; no release was built.
- **Exact monetary reconciliation — BLOCKED** on provable post-anchor accounting: currency spend is
  exported only as period snapshots (no monotonic counter), and the reconciliation anchor timestamp
  is not exported. Additionally the balance is already net of spend, so `balance − usage{daily}` is
  **not a valid bound in either direction**.
- No metric was invented; no unavailable value is rendered as zero. V9 labels were corrected only.

## 6. Durable repository references

Published on branch `docs/p10-closeout-20261127` in `RedEyeNinja-BKK/Switchyard` under
`docs/evidence/p10-closeout-20261127/`:

| File | Content |
|---|---|
| `LEDGER-20261127.md` | Parts I–III: pre-activation, post-activation, terminal closeout |
| `RESUME-20261127-final.md` | Resume anchor: live state, open items, constraints, traps |
| `TERMINAL-RECEIPT-20261127.md` | This receipt |
| `REGISTRY-BYPASS-20261127.md` | The 22 aliases and the 16 unregistered routes, with dispositions |
| `PARITY-20261127.md` | Method and evidence for §4, reproducible from the listed reads |

Pull request: opened for review only. **No merge, no main overwrite, no rebuild.**

## 7. Source consolidation — review order and state

Read from the GitHub API on 2026-11-27T08:5xZ.

| PR | Head | Base | State | Size | Note |
|---|---|---|---|---|---|
| #54 | `integrate/p4-on-640a946` @ `5df8c490` | `fix/tool-replay-eligibility-20261126` | open, **draft** | +10,700 / −202, 32 files | integration branch; mergeable unknown because its base is a topic branch |
| #55 | `tooling/release-protection-20261127` @ `f961c8c6` | `integrate/p4-on-640a946` | open, **draft** | +646 / −42, 2 files | stacks on #54; **must follow #54** |
| #56 | `tooling/p9-catalogue-20261127` @ `5efa9cad` | `feat/p9-evidence-qualification-20261124` | open, **draft**, clean | +10,742 / −0, 5 files | catalogue/evidence tooling; independent of #54/#55 |
| #57 | `feat/nemotron-ultra-peer-recovery-20261127` @ `088e45b6` | `main` | open, **ready**, clean | +75,019 / −771, 223 files | carried the deployed `aaf1bec8` binary; superseding "Current Authorization" notice in the body |

Dependency order for review: **#54 → #55**, in parallel with **#56**; **#57** independent (base
`main`). Remaining conflicts: none reported by GitHub for #56/#57; #54/#55 cannot be assessed until
#54's base settles.

Lineage recorded:

- **Cost-containment manifest** `9069fe66…` ← source `2de27bc7…` on `release/typed-services-d-20261127`,
  committed as `6fe166d4` (tests + `routes.cost-containment-20261127.toml` fixture + release
  evidence). This is the manifest the activated release carries unchanged.
- **Peer-recovery binary** `aaf1bec8…` ← source `088e45b6…` on
  `feat/nemotron-ultra-peer-recovery-20261127` (PR #57).
- **Package D** (typed services): 6 decision lanes present in the deployed manifest and exercised —
  `ST_Decisions`, `google-decision`, `smartfree-decision`, `smartlocal-decision`, `tier1-decision`,
  `smart-decision`, plus the `capabilities` / `capability_clients` embedding lanes.
- **Package C**: preserved at `581e189`, blocked, untouched.

## 8. C1–C8 / D1–D3 reconciliation (latest evidence)

| Item | Disposition | Basis |
|---|---|---|
| **C1** Google collector | **BLOCKED** | No non-human identity, no exported metrics, no scheduled timer. Requires owner-side viewer access + serviceusage grant. No change made. |
| **C2** Copilot owner-only execution | **REGISTERED, NOT PROVEN** | MCP bridge registered; no owner COPILOT credential delivered, no real job, no usage receipt. |
| **C3** Antigravity owner execution | **REGISTERED, NOT PROVEN** | Bridge ready; key/free entitlement unverified; project number unknown. |
| **C4** Stephen Antigravity MCP registration | **NOT VERIFIED** | No install/deny/ledger receipt from the remote host. |
| **C5** Stephen native Google routes | **LIVE (1 qualified member), 6 staged** | `ST_Gemini_Free` admits one qualified member; staged members have no serving receipt. Stephen's project number still unknown (telemetry only). |
| **C6** Grafana refinement | **CLOSED for redesign; factual corrections only** | V7.89/V9.16 accepted in this packet; no layout work performed or requested. |
| **C7** Release/repo reconciliation | **OPEN — review package ready** | §7. No merge, no promotion. |
| **C8** Estate-wide error observation | **OPEN, no action** | Aggregate error counts remain an observed anomaly, not a routing signal; no breakdown or route change performed. |
| **D1** ComfyNinja media / HTPC alignment | **ComfyNinja accepted; HTPC DEFERRED** | Remote OpenClaw multimedia remains a separate workstream. |
| **D2** Smart-route policy revisit | **DEFERRED** | Live topology untouched in this packet; the cost-containment change that produced 11 leaves was separately authorized earlier. |
| **D3** Privacy calibration | **SUPERSEDED** | The broad egress/privacy filter was removed on owner order (2026-11-21); no calibration or re-enable is pending. |

## 9. Finite remaining items — every one with a resumption condition

1. **`head` vs worker registry parity** — CLOSED as a measurement artefact (§4). Reopens only if a
   future read shows a body delta after the target-name length is subtracted.
2. **Registry census freshness** — dated 2026-11-27T08:50Z. Re-derive only when a decision needs a
   newer number; no sync is pending.
3. **16 advertised-but-unregistered routes** — §3. Resume only on a specific authorized consumer
   requirement, one name at a time.
4. **Bypass purposes marked UNKNOWN** — resume only if an owner or an artefact establishes a purpose;
   the bindings stay regardless.
5. **Effective headroom export** — deferred to a separately authorized telemetry packet; exact
   monetary reconciliation blocked on post-anchor accounting.
6. **ERNIE original leg** — unproven after the free-tier quota reset. Observe ordinary workloads only;
   no forced probes.
7. **Package C** — blocked at `581e189`.
8. **Nemotron Ultra** — desired, held: the free group would be 4 leaves (`12 = 5/4/1/2`); the
   admitted ladder is 3 (`11 = 5/3/1/2`).
9. **Phase 3** — deferred unless an actual authorized quiet window exists with a safe guard.
10. **Gate 5** — interface handoff only.
11. **V7 version/revision offset** — revision 90 titled V7.89. Repair needs an owner-approved
    out-of-band metadata write or a tool change; content is verified independently.

## 10. What this packet did not do

No deployment. No GO request. No LKG promotion. No PR merge. No credential, privilege, spend, GPU or
outage action. No paid probe. No routing, selector or manifest change. No dashboard redesign.
