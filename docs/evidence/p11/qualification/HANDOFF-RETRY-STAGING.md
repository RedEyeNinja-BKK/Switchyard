# Handoff — retry contract staging (durable, 2026-11-28 ~13:10Z)

Resume here. Production untouched: release `20261128T040916Z-6a8209f97101563b-fc05b640`,
binary `6a8209f9…22e82`, manifest `fc05b640…52817`, MainPID 880721, smart-route 3/2/1/2, LKG unchanged.

## Commits

Switchyard `/mnt/turnstone-build-ssd/w-fit-esc`, branch `feat/nemotron-ultra-peer-recovery-20261127`:
`2564941` local Retry-After on minute allowance + CoolingDown; `9c46a17` upstream #1323-compatible
error-header forwarding + SDK retry change; `b42172c` evidence; `478c9ea` renderer contract fix.
Pushed as `topic/retry-contract-20261128` on RedEyeNinja-BKK/Switchyard.

Turnstone `/var/tmp/audit-v194/privdev/workspace/turnstone`, branch `integration/turnstone-v1.9.4-alignment`:
`dec9e4b` honours Retry-After with 60 s cap; `e78ed9d` SDK retries disabled (`max_retries=0`). Local only.

## Proven staging evidence (observed, real HTTP)

Candidate `switchyard-server` (debug build) on 127.0.0.1:18992 with a disposable config and a stub
upstream: `/health` → 200; a served completion → 200; second request inside `requests_per_minute = 1`
returned

```
HTTP/1.1 429 Too Many Requests
retry-after: 30
{"error":{"message":"per-minute requests_per_minute budget for stub-model is full; it clears in about
 30s, and another candidate may serve this request now","type":"rate_limit_error",
 "code":"minute_allowance_throttled"}}
```

That closes the minute-allowance header path end to end. Dry-run validation also confirmed a real
config rule: `input_tokens_per_minute` requires `max_input_tokens` on the same target.

## Still unproven (do these next, in this order)

1. Upstream 429 carrying its own `Retry-After` → forwarded verbatim (#1323 path). Needs the stub to
   answer 429 **with** a header while a real upstream attempt happens; previous attempts never reached
   the stub because it was not alive during the call.
2. Armed 5 s cooldown: stub answers 503, then the next call must return 503 + `Retry-After` 1..=6.
3. Turnstone timings through the real client (`PYTHONPATH=.../workspace/turnstone`,
   `/var/tmp/audit-v194/testenv-v194/bin/python`): wait == advertised interval ±1 s; one outbound
   request per ladder attempt; >60 s interval ends the turn; cancellation during backoff is prompt.
4. Autonomous multi-window run (three refills) with a closing response.
5. Provenance: `f26a9030` is release.json `provenance.source_commit` but is a config/evidence commit.
   Build attestation for `6a8209f9…` not yet located. Capsule hint `3d478f5b`.

## Environment lesson (important)

Delegated task agents run in a different sandbox: no `/etc/switchyard`, no SSH to switchyard-vm, and
background processes do not survive between their tool calls. Run staging on this host with
`run_in_background=true` (one shell for the stub, one for the server), or do it inside a single
script that starts both, drives them, and tears them down.

## Not started

Objective 1 (Particle-1 with Stephen's credential) and objective 2 (ST_Route 4/5/3 config-only
release, dashboard/registry reconciliation). Both still authorized and outstanding.
