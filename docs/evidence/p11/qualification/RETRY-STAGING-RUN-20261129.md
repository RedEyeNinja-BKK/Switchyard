# Retry-contract staging run — 2026-11-29

Integrated HTTP/SDK staging of the retry contract, run locally against the candidate binary from
the retry-contract branch. Production was not contacted, not restarted and not re-pointed; every
endpoint in this run is `127.0.0.1`.

| Item | Value |
|---|---|
| Source | `feat/nemotron-ultra-peer-recovery-20261127` @ `e06ff69ce1fdd1a0e97ff51a51af2959c680c385` |
| Build | `cargo build --bin switchyard-server --offline`, dev profile, `CARGO_INCREMENTAL=0` |
| Candidate binary | `5c79211c32b54fc200b5c07d14eaad6c6c70ef33ca6374b41e55cc6d00e676c9` (327,068,808 B) |
| Switchyard listener | `127.0.0.1:18992`, disposable config, `--routing-log-file` enabled |
| Upstream stub | `127.0.0.1:18991`, scripted per-request responses with arrival timestamps |
| Stub sha256 | `e1c4fe10c640cad36e07bab82928ac5a4dccbb50d9c96fe6ec94a4ea710e9308` (4,118 B) |
| Harness sha256 | `09ff996efd5e9535ec56ae7836508b6a087c601907082e9bdb392c43cfcbe695` (14,645 B) |
| Stage config sha256 | `08bf917f63c503e3d8650726594b21a668e0e80332e064ed5ed418f43c9c1f05` |
| Results | `26/26` checks passed; `results.json` sha256 `927dd5e2c759a5f53ab781f54441a51c11189675a935859d5935d3e10762cdae` |
| Turnstone tree | `/var/tmp/audit-v194/privdev/workspace/turnstone` @ `e78ed9d` (retry pacing `dec9e4b`) |

Reproduce:

```
cd /var/tmp/retry-stage-20261129
PYTHONPATH=/var/tmp/audit-v194/privdev/workspace/turnstone \
  /var/tmp/audit-v194/testenv-v194/bin/python harness.py
```

The harness starts the stub and the candidate, drives the scenarios, then tears both down. It
prints one graded line per assertion; a `NOTE` line records a measured gap that is neither a pass
nor a failure.

## What the scenarios establish

| Scenario | What it exercises | Result |
|---|---|---|
| A1–A3 | Upstream `429` with its own `Retry-After` on a passthrough route | status 429 and the provider's message/code preserved; exactly one outbound attempt |
| A4 | Forwarding of the upstream's own interval to the caller | **not forwarded** — see "Open gap" below |
| A5–A6 | Client-side parse of the upstream interval (`max_retries = 1`) | retry gap 3.0 s against an advertised 3 s; exactly two attempts |
| B1–B6 | A `503` arms a 5 s failure cooldown | next call refused locally `503` + `Retry-After: 5`, zero further upstream dispatches, immediate, established code `temporarily_unavailable` |
| C1–C5 | Local per-minute admission (`requests_per_minute = 1`) | second call `429` `minute_allowance_throttled` + `Retry-After: 59`, one outbound total |
| D1–D5 | Turnstone's real OpenAI SDK client against the staged server | first call served; second raises `RateLimitError`; `retry_after_delay()` returns **59.0**; one outbound; the SDK does not sleep itself |
| F1–F4 | Three advertised windows, paced by the world, then a closing answer | waits `[2.0, 2.0, 2.0]`, closing `stub ok`, four attempts (one per window), observed gaps `[2.06, 2.06, 2.06]` |

Two ownership facts fall out of this and are worth stating plainly:

- **Switchyard states its own intervals and the caller acts on them.** `Retry-After` appears on the
  two refusals Switchyard itself computes (minute window, armed cooldown), Turnstone reads it off the
  wire unclamped, and re-issues exactly one request per window.
- **Turnstone's SDK client does not retry on its own.** `max_retries=0` at construction is visible
  from the outside: one outbound per attempt, and no wall-clock sleeping inside the call.

## Open gap — an upstream's own `Retry-After` is not forwarded

The retry-contract commit `9c46a178` added `upstream_error(status, body, retry_after_secs)` and
wired `LlmClientError::UpstreamHttp { retry_after_secs, .. }` into it. In this build the header does
not reach the caller on a passthrough route, while the parse demonstrably happens one layer down:

- A5: with `max_retries = 1`, the gap between the two upstream attempts is **3.0 s** — exactly what
  the stub advertised. So `retry_after_delay()` in `libsy-llm-client` reads the header.
- A1–A3: with `max_retries = 0`, the caller receives `429` with the provider's message and code but
  **no `Retry-After`**, for `429`, `500` and `503` alike.
- The server's own unit test (`upstream_error(..., Some(7))` → header `7`, `None` → absent) passes,
  so the renderer is not the defect; the interval is lost before or around the render.

This is recorded, not papered over: the observable is a missing header, and the discriminating
measurement is the retry gap. Fixing it is a source change on the retry branch and is **not** part of
this staging run. Until it is fixed, an upstream-stated interval reaches no Turnstone caller, and
Turnstone can only pace on Switchyard's own refusals.

## Turnstone-level semantics (unit, not integration)

`tests/test_retry_after_pacing.py` at `dec9e4b` — **12 passed in 3.02 s**. It pins the parts that a
single-process harness cannot show honestly:

- the header is read from the exception, and through a wrapper's cause chain;
- `None`, blank, words, HTTP-date, `nan` and `inf` are **not** waits;
- an affordable interval is slept exactly (`delays == [7.0]`);
- an interval beyond `RETRY_AFTER_MAX_SECONDS` (60) ends the turn instead of being shortened;
- cancellation during the backoff exits promptly.

## What remains unproven

- `>60 s` end to end through the staged server. It cannot be shown while the upstream header is not
  forwarded, and Switchyard's own refusals cannot exceed a minute window (59 s measured). Unit-level
  only.
- Cancellation during a backoff that is driven by a real socket. Unit-level only.
- The streaming (`text/event-stream`) path. Every scenario here is buffered.
- Anything about production: this run used a disposable config with stub targets. It says nothing
  about the live release, which was not touched.
