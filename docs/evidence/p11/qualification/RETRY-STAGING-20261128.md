# Retry-contract staging qualification — 2026-11-28

## Scope and changes

Switchyard forwards the upstream `Retry-After` delta (only; no unrelated provider headers) by carrying a parsed `retry_after_secs: Option<u64>` on the existing non-exhaustive `LlmClientError::UpstreamHttp` variant. The server adds the header to the existing upstream error response without changing status/body/code. Missing/unparseable values remain absent. Local typed errors are unaffected. Turnstone's base-URL OpenAI chat client construction uses `max_retries=0`; the no-base-URL OpenAI constructor is also set to zero so the setting is consistent for the OpenAI chat provider while other SDKs remain unchanged.

## Tests and observed output

Command:

```sh
export PATH="$HOME/.cargo/bin:$PATH" CARGO_TARGET_DIR=/mnt/turnstone-build-ssd/target-w-fit-esc TMPDIR=/var/tmp CARGO_INCREMENTAL=0
cargo test -p switchyard-server --lib
```

Observed final line: `test result: ok. 162 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.14s`.

Command: `cargo test -p switchyard-server --lib upstream_retry_after -- --nocapture`

Observed: `test tests::upstream_retry_after_is_forwarded_but_local_refusals_are_unchanged ... ok`; `1 passed; 0 failed; 0 ignored; 0 measured; 161 filtered out`.

Command: `cargo check -p switchyard-server`

Observed: `Finished dev profile [unoptimized + debuginfo] target(s) in 3.57s`.

Turnstone command:

```sh
TMPDIR=/var/tmp /var/tmp/audit-v194/testenv-v194/bin/python -m pytest tests/test_switchyard_sdk_retry.py tests/test_completion_recovery.py -q
```

Observed: `78 passed in 5.98s`.

## Isolated process exercise

A disposable config and deterministic local Python stub were created temporarily and removed after the run. Stub PID 2021327 and Switchyard staging process PID 2021374 were stopped via the background-shell termination tool. No production process/config was touched.

Stub direct probe returned 429 with `Retry-After: 5` and `{"error": {"message": "rate limited"}}`. Candidate server started on `127.0.0.1:18992`; `/health` returned `200 OK` and `{"status":"ok"}`. POST `/v1/chat/completions` returned:

```text
HTTP/1.1 429 Too Many Requests
content-type: application/json
{"error":{"message":"rate limited","type":"upstream_error","code":"upstream_error"}}
```

This staging binary was built before the last isolated source edits; therefore that observation does NOT qualify forwarding of the new header. The unit test directly validates the renderer with a parsed upstream interval.

## Explicitly not proven

* No integrated timing transcript was achieved for armed 5-second cooldown, minute allowance refill, >60-second interval, cancellation, tool non-replay, or an autonomous multi-window conversation.
* The SDK test verifies the configured SDK request behavior and retry ladder behavior separately, not a full Turnstone session over the staged Rust server.
* No production activation, provider call, live quota depletion, systemd restart, config edit, or LKG change occurred.
* This is not a release qualification and not evidence that the staged process used the final binary after the most recent source edits.

## Provenance

Read-only inspection of the production release store found the release record for `20261128T040916Z-6a8209f97101563b-fc05b640`, containing the exact binary digest `6a8209f97101563b58944f8614001fd0b057247da21a3daf57624ff798a22e82`, the manifest digest, and declared source commit `f26a90309dad348cee84a688d5ec8ffba4a4dc2c`. Its own note explicitly says this is a source identity declared at install and bound to the digest, not a reproducible-build result. The release capabilities record identifies `3d478f5b0e124b14db70b2d8f2dbcb4d69456579`; earlier release `20261128T020526Z-...` records that source for a config-only change on the same binary digest. Release audit records show the same binary digest reused across manifest-only deploys. Search found no separate attestation or build capsule matching the candidate; the evidence establishes installed bytes and deployment declarations, not which source tree actually compiled those bytes.
