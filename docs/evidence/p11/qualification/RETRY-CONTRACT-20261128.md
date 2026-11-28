# Retry contract: stated interval, honoured wait

2026-11-28. Isolated implementation, compilation and regression testing only.
Production is unchanged: `CURRENT -> 20261128T040916Z-6a8209f97101563b-fc05b640`,
MainPID `880721`, `smart-route` 3/2/1/2, LKG untouched.

## 1. Why

Measured on the Lite lanes (EVIDENCE-20261128 §2, §6): a turn spends ~80k prompt
tokens against `input_tokens_per_minute = 250000`, so about three calls a minute
fit. The fourth is refused with a body naming the wait —

> per-minute input_tokens_per_minute budget for gemini-3.1-flash-lite is full; it
> clears in about 22s, and another candidate may serve this request now

— and a `Retry-After` header that did not exist. Two independent gaps, both now
closed:

| Surface | Before | After |
|---|---|---|
| minute allowance, chat lane | 429, interval in prose only | 429 + `Retry-After: <retry_after_secs>` |
| minute allowance, capability lane | 429, prose only | 429 + `Retry-After` |
| minute allowance, Live lane | 429, prose only | 429 + `Retry-After` |
| armed failure cooldown | 503, no timing at all | 503 + `Retry-After: <remaining>` |
| spent provider day | 429, no header | unchanged: no header |
| daily counter unaccountable | 503 | unchanged: no header |
| request larger than a whole minute | 429, "waiting will not help" | unchanged: no header |
| transport / unavailability with no known end | 4xx/5xx | unchanged: no header |

A repo-wide search had found `RETRY_AFTER` only in `libsy-llm-client`: Switchyard
had never emitted one on any error path, and Turnstone had never read one.

## 2. Switchyard (commit `2564941`, branch `feat/nemotron-ultra-peer-recovery-20261127`)

* `ApiError` carries an optional `retry_after_secs`; `into_response` inserts the
  header when it is set. `error_response_with_retry_after` is the only way to set
  it, and only two arms call it.
* `LlmClientError::CoolingDown { retry_after_secs }` replaces the cooldown's bare
  `TemporarilyUnavailable`. The remainder is measured against the same clock that
  armed the cooldown, rounded up: a still-armed cooldown never advertises `0`.
* Status, body text, error type, error code, `local_error_class`, the metric
  label and `RouteErrorKind` are all identical to the old unavailability, so
  dashboards, log greps and existing clients see no change but the header.
* The router never sleeps on the interval. `CoolingDown` is `Unavailable` for
  routing purposes, so an eligible sibling is still tried immediately.

## 3. Turnstone (commit `dec9e4b`, `workspace/turnstone`)

* `retry_after_delay()` reads `Retry-After` from whichever exception in the cause
  chain carries a response. Delta-seconds only; blank, malformed, HTTP-date, NaN
  and infinite values are not a wait.
* Both bounded ladders use it: the creation ladder (`session.py`, base 1 s
  doubling) and the armed mid-stream re-issue ladder (same base, with the
  drain-retry loop in `model_turn.py` unchanged).
* An interval the turn can afford is the delay, exactly as stated.
* An interval above `RETRY_AFTER_MAX_SECONDS = 60` ends the turn rather than
  being shortened — the mid-stream arm sets the same `terminal` flag it already
  uses, so finalize/discard semantics are unchanged.
* `MAX_COMPLETION_REISSUES = 2` is untouched: this changes how long the ladder
  waits, never how many attempts it makes, and no new path replays a tool call or
  a partially executed side effect.

## 4. SDK-level retries (measured, not assumed)

`openai 3.51.0`, `DEFAULT_MAX_RETRIES = 2`, and `_base_client._calculate_retry_timeout`
honours `retry-after-ms` and `retry-after` (including the HTTP-date form). The chat
client is constructed without `max_retries` (`providers/__init__.py:265-266`), so a
throttled call is retried twice inside the SDK, with the stated wait, before
Turnstone's ladder is reached.

Consequence: the observed "three attempts in ~3 s" applies only after the SDK has
already spent its own two paced attempts. Worst case is now 3 SDK attempts plus 3
ladder attempts, each spaced by at least the advertised interval — bounded, not a
burst. Setting `max_retries=0` would make Turnstone the single pacer and remove the
duplication, but it changes every provider's retry topology and is therefore
proposed, not done.

## 5. Test results (observed, this session)

Switchyard, `CARGO_INCREMENTAL=0 cargo test`:

| Target | Result |
|---|---|
| `switchyard-server --lib` | 161 passed, 0 failed (4 new: header present on minute allowance, header + unchanged body on cooldown, no header on unarmed unavailability, no header on daily/oversized) |
| `switchyard-llm-client --lib` | 235 passed, 0 failed (2 cooldown assertions updated to require the remainder, `250..=300` of a 300 s cooldown and `55..=60` of a 60 s one) |
| `switchyard-runner --lib` | 366 passed, 0 failed |

Turnstone, `testenv-v194`:

| Target | Result |
|---|---|
| `tests/test_retry_after_pacing.py` (new) | 12 passed |
| `test_midstream_retry.py` + `test_completion_recovery.py` + `test_model_turn.py` | 204 passed |
| `test_session.py` | 620 passed |

Two behavioural tests carry the contract: a stream death whose error advertises
7 s sleeps exactly 7.0 s and then succeeds; one advertising 300 s raises with an
empty backoff list, proving the ladder does not retry inside an advertised wait.

## 6. Not proven here

1. **No integrated staging run.** Neither change has been exercised over real HTTP
   through the real client stack, because that needs both candidates running and
   no production activation is authorised. The evidence above is unit and
   session-level with injected responses.
2. **No live 5 s / 19-22 s reproduction.** The two scenarios were measured before
   the change; they have not been re-measured against a candidate binary.
3. **No autonomous multi-window completion.** Still blocked on deployment: it
   requires both halves live plus real allowance.
4. **Build disk.** The build SSD filled during the run (147 G, 0 free). 16 G of my
   own stale test binaries were removed to continue; no other workstream's target
   tree was touched.

## 7. Deployment and rollback proposal

Order matters: the producer first, so the consumer's behaviour can be observed on
its own wire before Turnstone changes.

1. Cut a release branch from the deployed source (`6a5937dd`), cherry-pick
   `2564941`, rebuild and re-run the three Switchyard suites there. The commit
   currently sits on a P11 branch whose base is not the deployed commit; the
   patch is small and file-local, but the build must be verified on the release
   base, not assumed.
2. Stage the candidate in the release store. `--dry-run` against live
   `routes.toml`, then a staging listener on a spare port with a stub target that
   sets `requests_per_minute = 1`: first request served, second must return 429
   with `Retry-After` between 1 and 60, and the capability/Live lanes must match.
3. Production swap of the binary only — no config change, so the release is
   config-identical (`fc05b640` stays). Verify with one probe against a throttled
   target and with `curl -D` showing the header; confirm `smart-route` is still
   3/2/1/2 and that no cooldown is armed by the probe.
4. Activate the Turnstone overlay in a maintenance slot. Verify a workstream that
   receives a 429 waits the stated interval, logs `retry_in=<interval>`, and
   continues without an operator resume.
5. Rollback: Switchyard `CURRENT -> <previous release>` (binary only, config
   unchanged); Turnstone slot revert to the prior venv/overlay. No schema change,
   so no migration and no dump restore is involved.

## 8. Constraints honoured

No model admission, no production swap, no scheduler, no new service, no quota or
accounting change, no privacy work, no P11 reopening. `smart-route` topology,
account boundaries and LKG are untouched.
