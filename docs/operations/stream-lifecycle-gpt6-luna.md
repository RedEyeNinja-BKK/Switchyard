# Stream lifecycle: gpt-6-luna consumer_dropped warnings

## Status

**UNRESOLVED OPERATIONAL DEFECT (historical baseline).** The original P2
investigation established that an observed inner stream ended before its
terminal event but did not identify who stopped consuming it, whether user
responses were incomplete, or whether the Luna-specific rate persists.
Do not close this issue as "documented" or "no code change required."
Tracking: [Switchyard issue #33](https://github.com/RedEyeNinja-BKK/Switchyard/issues/33).

## Symptom

After the PR #30 stream-correlation deploy, the journal contained a burst of
`LLM stream ended before its terminal event` warnings (`consumer_dropped`)
against `gpt-6-luna`. The concern was that the provider might be silently
dropping mid-stream, which would look like a provider reliability problem.

## Investigation

Two 15-minute observation windows were captured from the journal
(`journalctl -o cat`, one line per entry) and analysed with
`build-disk/pr30-deploy/analyse.py`:

| Window | State | consumer_dropped count | model | correlation_id | Joined to terminal |
|--------|-------|------------------------|-------|----------------|--------------------|
| pre-fix | before PR #30 correlation plumbing | 13 | gpt-6-luna | empty (all) | 0 |
| post-fix | after PR #30 correlation plumbing | 53 | gpt-6-luna | present (all) | 53/53 |

The post-fix window also showed:
- 225 HTTP request completion log lines, **all HTTP 200**, zero synchronous handler failures
- Every warning joined to a terminal line via `correlation_id` (`sy<salt><counter>` shape)
- Terminal fields (message count, tool calls, duration, ctx shadow values) identical between dropped and completed streams
- Zero `LLM stream failed` lines, zero other WARN lines

## Conclusion

The evidence **proves premature stream abandonment before the terminal event** (`MessageStop` / `[done]`) and 53/53 warning-to-request correlation joins. The terminal HTTP 200 entries show that an HTTP response was started/logged, **not** that SSE delivery completed or that the upstream finished successfully. The pattern is **consistent with downstream client or intermediary disconnection**; the available signals do **not** establish which party initiated the close, nor do they establish a provider-side defect. The Luna-versus-DeepSeek difference remains unexplained.

## Recommendation

Do not modify production routing or providers speculatively. The correlation_id
plumbing already on `main` makes historical warnings joinable but does not
prove successful SSE delivery. The new, separate SSE response-body lifecycle
signal in PR linked to issue #33 tracks whether the *HTTP body* was dropped
before its terminal frame was yielded. It is INFO-only and cannot prove
`client_closed`: only caller/proxy transport evidence can identify the initiator.

If the operator wants to distinguish an observed downstream close from other stream termination paths, consider a separately reviewed, bounded `client_closed` marker sourced from the **actual SSE transport disconnect signal**. Preserve an unknown/not-observed state rather than treating absence of evidence as `false`; emit once per stream without logging prompt, reasoning, or response bodies. No production experiment or routing change is authorized by this investigation.

## Evidence files

- `build-disk/pr30-deploy/window.log` — post-fix window (read-only)
- `build-disk/pr30-deploy/window-pre.log` — pre-fix window (read-only)
- `build-disk/pr30-deploy/analyse.py` — analysis script (read-only)

## Completion evidence still required

This file is an evidence record, **not a resolution**. Close [issue #33](https://github.com/RedEyeNinja-BKK/Switchyard/issues/33) only after a bounded, privacy-safe current observation joins the inner observer warning and outer HTTP body outcome to the caller/proxy logs, determines which side initiated cancellation, states whether terminal events were delivered, fixes the proven source, and measures after the fix. Do not substitute a lower warning count for proof of response completeness.

**No production deployment or runtime observation is claimed by the documentation PR.**
