# Stream lifecycle: gpt-6-luna consumer_dropped warnings

## Status

**No code change required.** This document records an evidence-backed
conclusion reached during the Post-PR #30 follow-up (workstream P2), so the
finding is not lost when the investigation branch is closed.

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
- 225 terminal request lines, **all HTTP 200**, zero failures
- Every warning joined to a terminal line via `correlation_id` (`sy<salt><counter>` shape)
- Terminal fields (message count, tool calls, duration, ctx shadow values) identical between dropped and completed streams
- Zero `LLM stream failed` lines, zero other WARN lines

## Conclusion

The evidence **proves premature stream abandonment before the terminal event** (`MessageStop` / `[done]`) and 53/53 warning-to-request correlation joins. The terminal HTTP 200 entries show that an HTTP response was started/logged, **not** that SSE delivery completed or that the upstream finished successfully. The pattern is **consistent with downstream client or intermediary disconnection**; the available signals do **not** establish which party initiated the close, nor do they establish a provider-side defect. The Luna-versus-DeepSeek difference remains unexplained.

## Recommendation

No production routing or config change is warranted. The correlation_id plumbing (PR #30 / branch `fix/p3-duplicate-warn-assertion`) makes these warnings attributable going forward.

If the operator wants to distinguish an observed downstream close from other stream termination paths, consider a separately reviewed, bounded `client_closed` marker sourced from the **actual SSE transport disconnect signal**. Preserve an unknown/not-observed state rather than treating absence of evidence as `false`; emit once per stream without logging prompt, reasoning, or response bodies. No production experiment or routing change is authorized by this investigation.

## Evidence files

- `build-disk/pr30-deploy/window.log` — post-fix window (read-only)
- `build-disk/pr30-deploy/window-pre.log` — pre-fix window (read-only)
- `build-disk/pr30-deploy/analyse.py` — analysis script (read-only)
