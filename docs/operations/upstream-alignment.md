# Upstream alignment for the local Switchyard fork

**Snapshot:** 2026-11-11 Asia/Bangkok (upstream activity through 2026-11-10 22:04 UTC).  
**Owner and execution record:** [fork issue #35](https://github.com/RedEyeNinja-BKK/Switchyard/issues/35).  
**Separate unresolved Luna incident:** [fork issue #33](https://github.com/RedEyeNinja-BKK/Switchyard/issues/33).

## Baseline: what was actually compared

| Ref | Commit / state |
| --- | --- |
| Official project | [NVIDIA-NeMo/Switchyard](https://github.com/NVIDIA-NeMo/Switchyard) |
| Upstream `main` | `6a924d55e36d65edb74082e2594e1c1364c2cfa9` |
| Fork `main` (before this documentation update) | `85defa031b70f5d313bcbc322f56365975ad4741` |
| Common ancestor | `16cbe5939937f3106bc5f13d9b11cdfa64f3242a` |
| GitHub cross-fork compare | **226 upstream-only / 114 fork-only commits**, **135 changed paths** |
| Most recent upstream release in GitHub API | [v0.4.0a1](https://github.com/NVIDIA-NeMo/Switchyard/releases/tag/v0.4.0a1), **pre-release**, published 2026-10-29 |
| Last stable release listed | [v0.3.0](https://github.com/NVIDIA-NeMo/Switchyard/releases/tag/v0.3.0), published 2026-09-22 |

The 135 paths are a **comparison/change-path inventory**, *not* a count of merge conflicts. The two heads are substantially diverged. Do **not** assume upstream `main`, a pending PR, or a prerelease is the next production release; repeat this comparison against the exact release tag before integration. Numbers are pinned to this snapshot and will change.

The fork contains locally necessary routing, capability, correlation, economics, request-lifecycle and deployment changes; upstream has continued to evolve the same crates. Preserve local semantics deliberately; avoid a blanket merge/reset and do not use an older upstream binary to replace the deployed fork.

## Our work versus the official upstream

| Our work | What is already accomplished | What upstream changes or overlaps | Decision |
| --- | --- | --- | --- |
| [#29 sanitized stream termination](https://github.com/RedEyeNinja-BKK/Switchyard/pull/29) | Merged, classified errors and `consumer_dropped` without logging provider text | [Upstream #1252](https://github.com/NVIDIA-NeMo/Switchyard/pull/1252) merged: native upstream observation **before** routing buffering; [#1221](https://github.com/NVIDIA-NeMo/Switchyard/pull/1221) open: physical attempt lifecycle/cost | Keep our downstream stream semantics, map them to upstream physical attempts, **no duplicate failure/usage accounting** |
| [#30 Responses terminal and correlation](https://github.com/RedEyeNinja-BKK/Switchyard/pull/30) | Merged; real-path decode tests, correlation IDs | [Upstream #1271](https://github.com/NVIDIA-NeMo/Switchyard/pull/1271) merged: Python streaming translation no longer polls after Responses or Anthropic terminal; [#1290](https://github.com/NVIDIA-NeMo/Switchyard/pull/1290) and [#1300](https://github.com/NVIDIA-NeMo/Switchyard/pull/1300) open: incomplete snapshots and refusal stream content | Replay equivalent SSE/translation fixtures on upstream and fork, preserving terminal/event semantics; do not assume Python SDK fix is the Luna root cause |
| [#32 duplicate warning assertion](https://github.com/RedEyeNinja-BKK/Switchyard/pull/32) | Closed, not merged, superseded by mainline correlation tests | No reason to revive it to chase an outdated assertion | Remain closed |
| [#34 SSE HTTP-body marker](https://github.com/RedEyeNinja-BKK/Switchyard/pull/34) | Merged, correlated **INFO** `dropped_before_terminal_frame`; not a transport disconnect signal | [Upstream #1231](https://github.com/NVIDIA-NeMo/Switchyard/pull/1231) **open**: first-frame timing and cancelled pre-answer wait. It edits exactly `crates/switchyard-server/src/{sse.rs,response.rs,redaction.rs,lib.rs}`, which #34 also edits | **DIRECT INTEGRATION CONFLICT RISK.** Incorporate both signals in a typed SSE observation context (correlation ID, start instant, served model) if #1231 lands. No duplicate logging, no false `client_closed` claims |
| [#33 Luna root-cause investigation](https://github.com/RedEyeNinja-BKK/Switchyard/issues/33) | Historical 53 pre-terminal drops identified and inner/request correlation joined; **not resolved** | [Upstream #1168](https://github.com/NVIDIA-NeMo/Switchyard/issues/1168) reports a slower-response cutoff; [#1230](https://github.com/NVIDIA-NeMo/Switchyard/issues/1230) and [#1231](https://github.com/NVIDIA-NeMo/Switchyard/pull/1231) address **pre-answer timing**, not necessarily the same event | Compare actual client/proxy deadlines, upstream terminal delivery and response-body lifetime, with before/after denominators. Never infer root cause from HTTP 200 or a log WARN count |
| Our smart/composed routes | Custom model identities, fallback order and session/routing semantics | [Upstream #1297](https://github.com/NVIDIA-NeMo/Switchyard/pull/1297) merged optional Stage compaction override; [#1204](https://github.com/NVIDIA-NeMo/Switchyard/pull/1204) merged plan/execute via Stage; [#1242](https://github.com/NVIDIA-NeMo/Switchyard/pull/1242) open pool/auto redesign; [#1243](https://github.com/NVIDIA-NeMo/Switchyard/pull/1243) open optional balanced tier | Route and config contract check, but **no automatic adoption of upstream defaults, tiers or model pools** |

Other upstream items relevant to incident diagnosis, **not automatically approved for porting**:

- [#1267 — cancel abandoned Python stream reads](https://github.com/NVIDIA-NeMo/Switchyard/pull/1267), **open**. Only applicable if the actual caller uses the affected Python bridge; it does not demonstrate Rust HTTP server cancellation.
- [#1133 — retry eligible pre-answer SSE 429/503](https://github.com/NVIDIA-NeMo/Switchyard/pull/1133), **open**; [issue #1132](https://github.com/NVIDIA-NeMo/Switchyard/issues/1132). Do not retry or replay after any answer content has escaped.
- [#1215 — diagnostics when stream translation loses content](https://github.com/NVIDIA-NeMo/Switchyard/issues/1215), **open**. Consider if the observed user output is missing content rather than the transport closing.
- [#911 — Responses content-filter completion semantics](https://github.com/NVIDIA-NeMo/Switchyard/issues/911), **open**. Preserve refusal/incomplete versus genuine completed outcomes.
- [#1275 — target label versus provider model ID](https://github.com/NVIDIA-NeMo/Switchyard/pull/1275), **merged**. Check preserved reasoning and correct metric attribution when comparing a labeled route to the actual provider.
- [#819 — response-body and SSE event size bounds](https://github.com/NVIDIA-NeMo/Switchyard/issues/819), **open**. Do not accidentally change memory/error behavior by adopting unrelated streaming patches.
- [#1220 — streaming translation engine](https://github.com/NVIDIA-NeMo/Switchyard/pull/1220) and [#1232 — Python streaming translation bindings](https://github.com/NVIDIA-NeMo/Switchyard/pull/1232), **merged**. These are prerequisites/context for #1271, and may change where tests belong.

### Specific high-risk signature collision

At the pinned fork head, our `frame_stream` takes **fourth argument** `correlation_id: Option<String>` for the #34 HTTP-body marker. Upstream's **open** #1231 proposes a different fourth argument, `first_event_timing: Option<(Instant, String)>`, and additionally adds a `started: Instant` parameter to `into_http_response`. Neither can simply be copied over the other. Design one additive typed context or a compatible signature, pass all call sites (including redaction tests), and test independent first-frame and body-outcome observables. Do not prematurely cherry-pick an open upstream PR. No implicit guarantee that a framed event was delivered over the socket.

### Required upgrade workflow

1. **Inventory before any port:** `git fetch upstream main --tags`; record source tag/SHA, common ancestor, ahead/behind and overlap paths; list upstream *merged* commits since that base. Confirm status of every linked issue/PR at execution time.
2. **Decide for each change:** (a) already equivalent/no action; (b) upstream change to port; (c) local overlay to preserve; (d) direct conflict requiring adaptation; (e) open upstream PR/monitor only; (f) irrelevant. Include exact file/function and test evidence. Do not count commits as conflicts.
3. **Build in a disposable integration branch:** prefer the selected upstream **release tag**, not moving `main`, unless the owner explicitly requests a `main`-tracking canary. Apply the minimum reviewed local overlay and keep changes isolated from the current production branch; keep release-manifest and deployment provenance.
4. **Compatibility tests:** Responses `response.completed`, `response.incomplete` and refusal, Anthropic `message_stop`, Chat `[DONE]` with trailing usage, normal/unpolled/abandoned body, joined request IDs, first-frame wait versus full SSE delivery, pre-answer 429/503 versus post-content failures, correct once-only sanitized warnings, cost/usage once only, selected provider versus route label, smart-lane fallback/session behavior and deployment rollback.
5. **Before real deployment:** provide a migration/rollback diff, a small bounded canary on the **actual consumer** with privacy-safe correlation (no prompt, reasoning or upstream-error body), route-manifest validation, successful CI and explicit production authorization. Do not treat a clean test run alone as live Luna resolution.
6. **Keep divergence visible:** maintain a **read-only upstream drift watch** on a daily schedule or at each upstream release, and a manual dispatch. Report only *newly merged overlapping* upstream PRs, the release/tag, ahead/behind, and a single short change summary in [issue #35](https://github.com/RedEyeNinja-BKK/Switchyard/issues/35). Never auto-merge, force-push or deploy. Suppress duplicate reports when the upstream SHA has not changed.

## Closure evidence

- [Issue #35](https://github.com/RedEyeNinja-BKK/Switchyard/issues/35) needs a one-page upstream/local conflict matrix, an integration-branch ref and tests, plus an implemented low-noise drift-watch mechanism before marking upstream compatibility ready.
- [Issue #33](https://github.com/RedEyeNinja-BKK/Switchyard/issues/33) needs **separate** production proof: where Luna's inner stream was dropped, what the HTTP body and client/proxy did, how many user-facing answers were incomplete, the proven cause, correction, deployed SHA, and before/after lane rates. The historical 53 warnings are not proof of current behavior.
- Keep host-side Turnstone venv/storage deployment status in its own issue; do not use it to justify changing Switchyard routing.

**Policy:** New custom streaming/routing work must begin with the latest upstream release/tag, relevant merged commits, open overlapping PRs and known issues **before** implementation. Upstream source is the baseline; the fork is a small, documented overlay wherever feasible. No speculative fixes, no silent broad rebase, and no “resolved by documentation” for live loss of responses.
