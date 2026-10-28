# Withdrawn and unstable upstream targets

Measured 2026-10-23 against the provider, not inferred from earlier reports.
This records what is actually unavailable, what was only transiently unavailable,
and the staged remedy. It changes no routing configuration: applying the remedy
is a production change and needs its own approval.

## `minimax/minimax-m3.1-flash:free` — withdrawn by the provider

The `:free` variant is gone permanently. The provider answers `404` with an
explicit statement that the free launch period ended and the variant was
retired, and names the replacement:

```
minimax/minimax-m3.1-flash   list price: $0.15/M input, $0.60/M output
```

This is an upstream retirement, not a Switchyard defect. The route
`direct-openrouter-minimax-m3-1-flash-free` has an empty candidate list, so a
request to it has no fallback and fails immediately. Nothing else in the
deployment references the target: no composite, tier or smart lane composes it.

The error handling is already correct. A `404` is not in the retryable status
set (`408`, `429`, `5xx`), so the client fails fast rather than retrying a
permanently dead model. No code change is warranted; retrying a retirement
forever would be the defect.

## `nvidia/nemotron-3.5-super-vl:free` — available

Answers `200` to a live completion, in about 11 seconds. The earlier failures
were upstream unavailability during a window, not a local routing, quota or
cooldown fault. Nothing is recommended for this target: it is the primary of
`ST_Tier1`, `smartfree` and `smartfree-bounded`, each of which already declares a
candidate fallback.

The latency is worth knowing. At roughly 11 seconds per completion, a target
timeout below that will look like a provider failure; the deployed configuration
declares no `timeout_ms`, so the wait is unbounded and the slow case completes
rather than being cut off.

## Staged remedy (NOT applied)

Two options, and the choice is a policy decision rather than a correctness one:

**Option A — remove the route.**
Delete `direct-openrouter-minimax-m3-1-flash-free` and its target
`openrouter-minimax-m3-1-flash-free`. A route that can only return `404` is not
a capability. The cost is that a caller addressing it by name gets an unknown
route rather than a provider error; both fail, but one fails honestly.

**Option B — repoint to the paid model.**
Change the target's model to `minimax/minimax-m3.1-flash` and rename the route,
because a route whose name says `free` that bills `$0.15/M` input is a trap for
whoever reads it next. This keeps the capability and changes the cost class from
`free` to `metered`, which also changes the economics attribution for that lane.

Option B is only appropriate if the operator wants to pay for MiniMax. Option A
is the smaller change and does not spend money.

Applying either requires editing the production `routes.toml` and reloading, so
neither is part of this change. Whichever is chosen should be validated first
with the candidate binary's `--dry-run` against a copy of the live configuration,
which is the same gate used for the economics work.
