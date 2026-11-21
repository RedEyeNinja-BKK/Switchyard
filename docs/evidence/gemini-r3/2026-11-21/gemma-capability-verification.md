# Gemma 4 — direct capability verification, 2026-11-21

Measured, not read. Production key on `switchyard-vm`, read from the deployment's own env
file inside the host so it never left it. Endpoint: `https://generativelanguage.googleapis.com/v1beta/openai`.

**Spend: 25 chat requests and 2 model listings.** Every request was small (`max_tokens` ≤ 400,
prompts ≤ 60 tokens). No 15K or 262K probe was run, per the standing instruction; the TPM
question is settled by the rate limiter, not by burning quota.

## Matrix

| capability | `gemma-4-26b-a4b-it` | `gemma-4-31b-it` |
| --- | --- | --- |
| listed in `/models` | yes | yes |
| text completion | 200, ~2.5 s | 200, 18–55 s, 503 on most attempts |
| `tools` + `tool_choice:"auto"` | 200, `finish_reason:"tool_calls"`, 1 call, `{"location":"Bangkok"}` | 200 on attempt 3 (~18 s), same shape |
| tool round-trip with stripped call (id/type/function only) | 200, correct answer | not run |
| `response_format:{"type":"json_object"}` | 200, but content is `<thought>…</thought>{…}` → not parseable | same |
| `response_format:{"type":"json_schema",…}` | not run | 200, clean JSON, no thought block |
| `reasoning_effort:"none"` | not run | **400** `INVALID_ARGUMENT "Thinking budget is not supported for this model."` |

Raw receipts in `live-probe/`. One record per request, including the failures.

## The three shapes that change the implementation

### 1. Thinking arrives inline in `content`

```
content: "<thought>The user is asking for a single-word reply… </thought>ready"
extra_content: {"google": {"thought": true}}
usage: {"prompt_tokens": 8, "completion_tokens": 1, "total_tokens": 71}
```

`completion_tokens` **excludes** the thought; `total_tokens` includes it. There is no
`completion_tokens_details`, and `reasoning_tokens` is absent, so the only way to count
thinking tokens is `total − prompt − completion` (62 in the example above).

Consequence: Switchyard's extraction reads `reasoning_content` / `reasoning` only, so a Gemma
reply is currently returned with its internal monologue as visible answer text. That is a
defect of the lane, not a preference.

### 2. Tool calls carry a thought signature that turns out not to be required

```json
{"id": "call_1338…", "type": "function",
 "function": {"name": "get_weather", "arguments": "{\"location\":\"Bangkok\"}"},
 "extra_content": {"google": {"thought_signature": "EiYKJGUyNDgzMGE3…"}}}
```

The round-trip was run both ways. Sending the assistant turn back with only
`id`/`type`/`function` — the shape Switchyard's Chat translation produces — returned 200 and
the model answered from the tool result. This differs from Gemini 2.x, where a missing
signature is a 400. The verbatim attempt that would have proved the converse returned 503, so
the only claim made here is the positive one: **the stripped round-trip works.**

### 3. `json_object` and `json_schema` are not the same tool

`json_object` constrains nothing: the model still emits a thought block first, so the body is
not parseable JSON. `json_schema` produced `{"a": 1, "b": 2}` and no thought block at all.
For structured output on this family, `json_schema` is the usable mode and `json_object` is
not.

## Capacity, which is not a capability

31B answered 1 of 5 attempts, taking 18.3 s when it did, and returned
`503 UNAVAILABLE "This model is currently experiencing high demand"` otherwise. Every 503 was
`finish_reason`-less and body-wrapped in a list. 26B answered every attempt in 2–5 s.

So 31B is not unqualified — it is *capacity-starved and slow*. Three consequences:
the bounded route must order 26B first, a 503 must cross to the next leg rather than being
retried in place, and a comparison run has to record 503s as their own outcome instead of
folding them into failure or latency.

## What the docs actually say

`aistudio.google.com/llms.txt` and `docs/models.md` contain **zero** mentions of Gemma. The
function-calling, structured-output and thinking pages document Gemini models only. Gemma's
support on the OpenAI-compatible surface is therefore **undocumented by the vendor** — the
matrix above is the only evidence that exists for it, which is why it is recorded request by
request.

## The leak is closed at the target that declares the marker

`hosted_reasoning = "thought_tag"` on a target is the switch. The target states the marker,
the backend carries it, and both decode paths split on it: the buffered response moves the
block into a `Reasoning` block before the message text is built, and the streamed response
buffers only as much as it needs - enough to tell whether the message opens with the marker,
then the block itself until its close - releasing answer text as soon as the block ends.

A target that does not declare it is untouched. That is not politeness: the two Gemmas and
the Gemini chat models share one credential, and a model that legitimately writes the token
as text must not have it interpreted. The declaration is validated at load, so a typo is a
configuration error rather than a silently disabled filter.

The split refuses rather than repairs, and the refusals carry more weight than the happy
path:

- an unterminated block is a translation error - nothing was ever released as an answer, and
  synthesising one from the truncated thought is the leak the filter exists to prevent;
- a stream that ends inside a block fails the same way instead of emitting an empty answer;
- a `<thought>` after the answer began, or a close with no open, is reported rather than
  stripped, because the visible text is then known to be incomplete in a way this layer
  cannot reconstruct.

Accounting follows: the caller sees only the answer, so its visible text matches
`completion_tokens`, while `total_tokens` still includes the thinking that was moved rather
than discarded - it survives as a reasoning block for a caller that wants it.

Receipts for the change: `switchyard-translation` 67 unit + 196 integration,
`switchyard-llm-client` 167, `switchyard-runner` 274 - 704 tests, 0 failures, plus a
`--dry-run` load of the regenerated manifest on the new binary (49 → 52 routes, the two Gemma
leaves and `smart-gemini-bounded` added, nothing removed). Three mutants - buffered split
disabled, stream filter disabled, end-of-stream refusal disabled - each turned a named test
red; the control case (same body, no marker declared) stayed green, which is what shows the
marker is the switch and not a global strip.


## CORRECTION 2026-11-21 (post-`fd8c61c4`) - the leak had three more channels

The paragraph above says the moved thinking "survives as a reasoning block for a caller that
wants it". **That is withdrawn.** It survives as a reasoning block, and a reasoning block is
not private: the chat encoder writes it back out as `reasoning_content`, so the monologue
reached the caller through a second field while the answer looked clean. Removing the text
from `content` alone was cosmetic.

Three channels were found and closed, each with its own mutant:

| channel | how the text escaped | closure |
|---|---|---|
| `reasoning_content` | the split rebuilt the thought as a `Reasoning` block | the split discards it; a marker-declaring target emits no reasoning field at all |
| retained stream frame | every decoded event carried the raw upstream frame, and the stream encoder replays a frame in preference to the normalized chunks | a marker-declaring stream yields events with **no** preservation (an absent one, not a `Null` one - the encoder replays whatever is present) |
| exact preserved body | the buffered decoder captured the source body for byte-exact replay, and the encoder prefers it over the IR | a marker-declaring target is not eligible for response preservation, which is the only gate that also covers the serving layer, where the reply is re-encoded in the caller's format and the target's policy is out of scope |

The tests now judge the **encoded wire**, not the IR: the label text appears in neither
`content` nor `reasoning_content`, and survives in no block at all. Four mutants - capture
re-enabled, frame retained, split rebuilt as a reasoning block, stream delta forwarding
reasoning - each turned a named test red by name.

## Quota scope: one spent model does not condemn the pool

A spent Free-tier daily quota was treated as account-wide from the code alone, so the first
429 from one model stopped the candidate list. Google allocates RPD **per model** (20/day
Flash, 500/day Flash-Lite, 14.4K Gemma) plus separate feature quotas, so an unscoped
`quota_exceeded` proves an allowance is spent without proving which one.

Scope is now read from the structure the body supplies - a `model` dimension or a
`PerProjectPerModel` id is a model; `PerProject` without `PerModel` is the project; a
feature-named quota is a feature; anything else stays `Unknown` and is never promoted to
project-wide. Only a positively project-wide quota may stop the candidate list. The details
may also widen a bare `RESOURCE_EXHAUSTED` to a daily quota when the quota id says `PerDay`;
the status string alone still may not.

Regression: two models on one project, the first answering `quota_exceeded` per model, the
second serving - and, as the contrast, a project-scoped quota stopping the list after one
call. Restoring the old code-only classification turns that test red.

## Owner-supplied facts recorded

- The AI Studio panel cells (`3.6 Flash 23/20`, `3.8 Flash 22/20`) are a **selected-window
  peak**, not a verified remaining or exhausted balance. Nothing in this lane treats them as
  live quota.
- Google's own guidance is that `/responses` is not offered natively: a caller wanting that
  wire format must go through a translation proxy or Google's native SDK. Switchyard is
  already such a proxy, and the `google-genai` SDK is the reference for the canonical field
  mapping if that lane is built.

## Lane ordering and health

`smart-gemini-bounded` now leads with the 26B: it is an A4B mixture with roughly 4B active
parameters per token against the 31B's 31B, so it moves far fewer weights per request, which
is what matters against a tokens-per-minute allowance. The 31B is the escalation step and
gets its own `llm_client` entry with `failure_cooldown_ms = 300000` and no `cooldown_group` -
a capacity fact about the large model must not cool the lane's default down with it. The
daily counter stays shared because it is one project and one allowance.

Receipts: `switchyard-translation` 67, `switchyard-llm-client` 163 + 4 + 18,
`switchyard-runner` 208, clippy `-D warnings` clean; regenerated manifest 49 -> 52 routes and
`--dry-run` rc=0 on the new binary.
