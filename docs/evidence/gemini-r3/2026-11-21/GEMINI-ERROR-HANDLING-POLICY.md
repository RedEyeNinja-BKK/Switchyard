# Gemini Free-tier error handling — Switchyard engineering contract
2026-11-21 · **applies to ALL 26 actionable Google Free-tier model/service contracts** (Antigravity deferred) · source/test direction, not live execution authority.

## Reference and scope

Owner sources: https://aistudio.google.com/docs/api-errors.md and https://aistudio.google.com/docs/troubleshooting.md. The supplied AI Studio markdown URLs could not be directly fetched by the reviewer, so the publicly accessible official Google Developers **API errors** and **troubleshooting** pages were read at https://ai.google.dev/gemini-api/docs/api-errors (updated 2026-08-05) and https://ai.google.dev/gemini-api/docs/troubleshooting (updated 2026-07-27). **This page explicitly defines Interactions API errors**, including standard HTTP codes, generation-blocked codes, generation-structural error codes, buffered `error` JSON and streaming SSE `event_type=error`. Do NOT assume every Google wire API or the OpenAI-compatible API exposes these *same snake_case codes*. The actual OpenAI-compatible Gemma 4 31B receipt in this PR is **HTTP 503 with Google `UNAVAILABLE` and `error` encoded inside a JSON list**. Normalize known envelopes per transport; preserve unknown originals safely.

Other official Google reference: https://ai.google.dev/gemini-api/docs/troubleshooting (exponential backoff for transient 429/503) and https://ai.google.dev/gemini-api/docs/rate-limits (RPM, input TPM, RPD/feature limits). The owner authenticated **47 model+agent / 23 tool quota table** and all 26 actionable Google Free-tier capabilities are in [OWNER-CONTROLLING-ALL-FREE-CAPABILITIES.md](OWNER-CONTROLLING-ALL-FREE-CAPABILITIES.md). Apply this policy across **chat (10), TTS (4), transcription (1), Decision (1), embedding (2), robotics (1), realtime music (1), Live (6)**. This is NOT a Gemma-only exception or a new model-research phase. Existing deployed R1 remains unchanged until approved release.

## Required error normalization (one provider-aware boundary, not ad hoc fallbacks)

Canonical error evidence record: `provider=google`, account/project quota scope (no secret bytes), native model ID, requested route, actual transport (`interactions` / `generateContent` / `openai_compat` / `embedContent` / `createDecision` / `BidiGenerateContent` / `BidiGenerateMusic` / TTS), HTTP code where present, stable provider `error.code` or `status`, any `error.details` quota/retry hints and `Retry-After`, local physical/context/privacy/admission decision, **model vs project vs feature quota scope**, whether provider was contacted, retry count, final child/physical target, consumption state, and whether response/stream/tool execution had committed. Normalize envelopes:
- **Interactions buffered:** `{"error":{"code":"rate_limit_exceeded","message":"..."}}`.
- **Interactions streaming:** SSE event with `{"event_type":"error","error":{"code":"...", "message":"..."}}` **after HTTP 200**; the event is a failure, not successful completion.
- **Native generative Google:** often `{"error":{"code":503,"status":"UNAVAILABLE","message":"..."}}`; do not confuse numeric error code with Interactions snake_case `code`.
- **OpenAI-compatible Gemini/Gemma:** `error` may be structured in OpenAI form **or returned list-wrapped** (measured 31B 503). Parse carefully rather than collapsing to an opaque generic 500. Unknown or malformed bodies retain HTTP classification with `unknown` subtype; no speculative quota or policy inference.
- **Live/WebSocket and music:** session events/closure/remote cancellations are in-band transport outcomes, not necessarily HTTP statuses. Preserve partial output state and never silently rerun a committed Live session.

**Boundary:** provider errors differ from Switchyard's own `403 privacy_refused`, `payload_not_cleared`, capability mismatch, physical-fit refusal or locally spent RPD. Do NOT relabel a local privacy refusal as a Google `PERMISSION_DENIED` and do not treat any safety/policy refusal as a model-health outage.

## Decision matrix: official Interactions API codes + Switchyard operational policy

| HTTP | Machine `code` (Interactions reference) | Scope and Switchyard action | Automatic retry? | May choose another model/pool? |
| --- | --- | --- | --- | --- |
| 400 | `invalid_request`, `parameter_unknown` | Invalid syntax/unsupported parameter or adapter defect. Do not resend same bytes. Correct adapter/request, mark individual feature unsupported if proven. | **NO** identical request | Only if another target is **known compatible** and request can be normalized without losing requirements; not automatic pay spill |
| 400 | `failed_precondition` | Billing/project/access prerequisite unavailable. Block affected project/operation; never toggle billing or elevate provider account silently. | **NO** | Other eligible, privacy-approved project/provider only; no paid transition without consent |
| 416 | `out_of_range` | Parameter outside allowed bounds. Fix locally; fail closed for unchanged payload. | **NO** | As for unsupported parameter |
| 401 | `authentication` | Credential invalid/missing/expired. Credential-scope breaker, operator alert; never leak key. | **NO** | Only other *independent permitted credential* per explicit route policy; do not test same key across models |
| 403 | `permission_denied` | Permission/project/model/asset denied. Do not bypass with other models on same credential or loosen privacy. | **NO** | Other independently eligible credential only after provider and payload policy check |
| 404 | `model_not_found` | Invalid/retired/unserved exact model slug. Disable/cool down that direct ID and requalify before reuse; do NOT retire whole Google pool. | **NO** same model | **YES**, another already-qualified eligible model, same price/privacy tier |
| 404 | `not_found` | Could be missing file, input asset, session, job, etc.; **not proof model retired**. | **NO** blind resend | Only if known missing-resource recovery; never re-route unchanged bad input as if model failed |
| 409 | `already_exists` | Idempotency/resource conflict, not an availability signal. | **NO** blind retry | No |
| 409 | `aborted` | Concurrency conflict; Google says retry at higher application level. Check idempotency/collision. | **BOUNDED** for proven idempotent request | No provider switch merely to evade consistency |
| 429 | `rate_limit_exceeded` | **Short-window request/token RPM/TPM**. Apply `Retry-After` if present, local pacing/backoff/jitter, increment short-window throttle; do not mark RPD exhausted. | **BOUNDED** and within deadline/attempt cap | **YES** other eligible model/project within same authorized free tier, retaining privacy |
| 429 | `quota_exceeded` | **Daily quota**. Block that exact quota scope until provider reset (project-model or shared project/feature as demonstrated), no immediate retries. Preserve shared group identity. | **NO** until reset | **YES** only another independently funded/eligible free quota; paid only if explicitly allowed |
| 499 | `cancelled` | Client cancelled. End work; no provider hopping, retries or additional token spend. | **NO** | **NO** |
| 500 | `api_error` | Provider/server fault. Short bounded retry if idempotent and pre-commit; else cooldown/reselect. | **BOUNDED** | Eligible free peer; paid only with opt-in |
| 501 | `unimplemented` | Unsupported method or feature. Correct typed adapter or leave direct HELD, not an outage. | **NO** | Known-capable alternative transport only; no silent degraded output |
| 503 | `service_unavailable` | **Capacity/outage**, NOT daily quota. Brief per-model/provider cooldown; fast move to an eligible free peer. Record overload separately. **31B high-demand 503 is measured here.** | **0–1** tiny replay at most, ideally cross to free peer | **YES** within compatible tier; no automatic paid spill |
| 504 | `deadline_exceeded` | Provider deadline expired. Check request size, task deadline, idempotency/stream/tool state; do not treat as RPM/RPD. | Only pre-commit, within remaining deadline | Another free compatible target if uncommitted; no duplicate side effects |

**Source nuance:** Google documents `rate_limit_exceeded` and `quota_exceeded` as two **different 429 codes**. The current Switchyard `failure_class` in `crates/libsy-llm-client/src/run.rs` generally classifies bare 429 as `RateLimited`, and `client.rs::is_account_wide_quota` recognizes specific OpenRouter/OpenAI markers, **not Google's `quota_exceeded` scope**. Extend the Google-specific parser conservatively, preserving existing OpenRouter/OpenAI semantics. Don't condemn the entire Google project from an ambiguous 429; if no stable subtype is available, use short bounded backoff, never infer daily reset from HTTP alone, and prefer pre-admission local quota checks.

## Generation blocked and generation structural errors (official separate tables)

**Generation blocked** codes: `safety`, `recitation`, `language`, `prohibited_content`, `spii`, `blocklist`, `image_safety`, `image_prohibited_content`, `image_recitation`, `image_other`, `content_blocked`.
- Treat as **policy/content outcome**, not an outage, 503 or generic retryable 400. **Do not automatically reformulate the prompt, drop safety restrictions or swap to a more permissive provider to defeat a block**. Report a clear safe explanation to the caller and offer user-directed revision if appropriate.
- `language` may warrant explicit supported-language selection only when the caller permits it; `spii` is also a **privacy** signal, never a reason to try a Free peer.
- Image blocked codes are relevant even if Google's Free image-generation RPD in this owner's snapshot is **0/0**; retain in typed protocol contracts without inventing executable image lanes.

**Generation structural** codes: `malformed_function_call`, `malformed_tool_call`, `unexpected_tool_call`, `no_image`, `too_many_tool_calls`, `missing_thought_signature`.
- Tool shape/signature is a **protocol/model capability** problem, not infrastructure. Validate declared tool names, tool-call JSON, arguments, and continuation state. No automatic tool execution on malformed or undeclared calls.
- `missing_thought_signature` means a signature was required on **that exact transport**. Reconcile against the native protocol; preserve signed state or mark that model/adapter path unsupported. The 26B OpenAI-compatible **positive stripped tool continuation receipt** proves that *one* Gemma path works, NOT that every Google Live/Interactions/native model can omit signatures.
- `too_many_tool_calls` and `unexpected_tool_call`: stop the agent step; never silently increase tool budget or execute undeclared action. `no_image`: do not claim image delivered, mark operation incomplete, enforce typed fallback only to known image-capable model with permission.
- These error codes can arrive **inside a provider response** without an ordinary HTTP error; parse them as first-class outcomes.

## Retry/cascade safety requirements across ALL Google capabilities

1. **Admission first**: provider privacy clearance/provenance for complete assembled bytes, approved cost tier, exact typed operation, model/provider quota+RPM+TPM+feature allowance, input/output/media/timeout fit. Never spend a Google call on a model known to be ineligible. Direct Google uncleared: terminal 403; bounded mixed composite: skip Google and permit only authorized Luna/DeepSeek. Preserve all existing provider behavior.
2. **Same quota identity across all aliases and transports**: one Google project/model budget shared between chat/vision/tools/typed primitives and models reused across routes; project- or feature-wide limits recognized only with evidence. Never create capacity by inventing additional direct alias quotas.
3. **Bounded recovery**: retries count as attempts and may consume quota; no refund after possible send. Prefer selecting another qualified *free* model to hammering repeated 503; observe provider `Retry-After`, capped jitter, cooldown and end-to-end deadline. No automatic paid model crossing solely because of 429, 500, 503 or 504; approved paid policy required.
4. **Idempotency/commit boundary**: no transparent retry/cross after client-visible streaming output, tool side effect, Live session actions, media generation job submission, or unknown-commit timeout. Failure must be observable to caller so it can decide. A 200 response or opening SSE stream does **not** prove successful completion.
5. **Provider-specific breakers**: 401/403 credential/permission-scoped; 404 `model_not_found` exact model ID; 429 short vs daily quota; 503 capacity/health; 400 unsupported mode/parameter; policy and generation-structural terminal. Keep account-wide, model, feature, and live-session state separate.
6. **Observability and hygiene**: record `provider_error_code`, `http_status`, `error_scope`, `retry_after`, `served_model`, `retry_count`, `fallback_from/to`, `last_successful_leg`, `quota_spent`, `committed_stream`, outcome and cancellation, with redacted messages. No credentials, raw user prompts, sensitive attachments, `<thought>` blocks or tool secrets in metrics/logs.
7. **Test fixture matrix**: (a) Interactions JSON `error` codes listed in table, (b) native Google `status` uppercase + numeric code, (c) measured Gemma 31B list-wrapped `UNAVAILABLE` 503, (d) SSE `event_type=error` after successful HTTP 200 and after partial stream, (e) Live/Music session disconnect after media output, (f) policy-blocked and malformed tool/signature, (g) quota scopes and no paid spill, (h) physical-privacy skipped request without Google contact, (i) unknown subtype fallback conservative. Require unit/mutation tests on normalization and recovery decisions, no external request to test any of these.

## Official troubleshooting guide — concrete implementation additions (owner follow-up)

Additional source reviewed directly: **[Google's official Gemini API troubleshooting guide](https://ai.google.dev/gemini-api/docs/troubleshooting)**, current page last-updated 2026-07-27 UTC. The supplied `https://aistudio.google.com/docs/troubleshooting.md` could not be independently fetched, so this section uses the official publicly readable Google developer guide—not a guess about that file's contents. Keep both troubleshooting and [API errors](https://ai.google.dev/gemini-api/docs/api-errors) in the repo reference set.

**1. Retry method and effective attempt budget.** Google recommends exponential backoff **with jitter** for transient **408, 429 and 5xx**, e.g. 1 s, 2 s, 4 s, 8 s; enforce the existing Switchyard request deadline, bounded attempt budget and any valid `Retry-After`. Importantly Google's **Python SDK already retries transient errors automatically (up to four times by default, initially ~1s, max ~60s)**. Switchyard must ensure that whichever SDK/REST or typed adapter is used does not silently multiply retries inside Switchyard's own retry loop. Count **every real upstream attempt** toward quota/request metrics and budgets even when a client library hides the attempt. Google's generic 429 retry guidance **does not override** the API-errors distinction: an explicit `quota_exceeded` **daily** 429 waits for provider-day reset, NOT exponential retries. Unrecognized 429 remains conservatively short-window/unknown, without inventing a daily counter fact.

**2. Request and model parameter failures.** Troubleshooting explicitly calls out matching the actual model ID, **API version** (notably `/v1` versus `/v1beta`, and Beta-only features), candidate count, temperature, top-p and per-model max output tokens. Fail an invalid or unsupported request/parameter **before provider egress** where source config permits; a 400 `parameter_unknown` or 501 `unimplemented` should diagnose the exact adapter field/version, NOT trip provider health or retry identical JSON. The guide tabulates candidate count 1–8, temperature 0–1, top-p 0–1; actual target-specific constraints and model-version docs win over a generic family table. **For Gemini 3.x, Google expressly recommends leaving temperature/top-p/top-k at their defaults**; do not globally lower temperature to “make tools deterministic” or blindly raise it because another model had repetitions.

**3. Thinking, output accounting and quality.** Google documents **Gemini 2.5 Flash/Pro thinking enabled by default**, affecting latency and token use, and provides per-model thinking controls. Do not apply a Gemini 2.5 toggle to Gemma 4 OpenAI-compatible replies: the measured 31B rejected `reasoning_effort:none` with 400. The PR #46 Gemma inline `<thought>` anti-leak guard remains a release blocker. Treat `json_object` with inline thoughts as invalid output even on HTTP 200; use model-qualified `json_schema` and local validation. Troubleshooting calls out output repetitions (Markdown tables/newlines/JSON field ordering), and repeat tool-call loops: classify them as **quality/structural outcomes**, not upstream outage; optionally adjust prompts **only under caller-approved task semantics**, validate tool names/args, and stop on tool budget. Do not log or expose thought content.

**4. Safety, copyright and language blocks.** Google tells developers to review a blocked prompt against request safety settings, and says `BlockedReason.OTHER` can indicate Terms violations or unsupported content. `RECITATION` concerns output resemblance to protected source text; it suggests more unique context/higher temperature. **Switchyard must not automatically weaken safety settings, rewrite an unsafe prompt or hop to another provider to evade a block.** Report the exact class safely and ask caller to revise the task when appropriate. Unsupported-language failures are not evidence of 503 model outage; a known language-capability alternative may be chosen only if no policy block is being circumvented.

**5. Blocked/leaked API keys.** Troubleshooting documents Google proactively **blocking publicly exposed API keys**, with an error text like `Your API key was reported as leaked. Please use another API key.` This is an **operator/security incident**, not a 429, transient 403 or an invitation to try the same key on all Gemini models. Stop egress on that credential scope, alert the owner with redacted evidence, audit exposure, and require owner-approved key rotation. **Never auto-create, rotate, print, replace or broaden access to credentials**. Avoid leaking the raw message if provider text might include secrets.

**6. Streaming and realtime discipline.** Google's HTTP troubleshooting retries apply only where the operation can safely be retried. For Live/transcribe/TTS/Lyria streams, detect midstream errors, cancellation and whether output or any side-effectful work has committed; do not replay a session or generated-media request after partial output. An SDK retry hidden underneath a session transport must not create duplicate media or tool effects.

**Offline fixture acceptance, not a new research phase:** (a) 429 `rate_limit_exceeded` gets jitter/deadline-aware bounded retry while 429 `quota_exceeded` does not; (b) 503 high-demand goes to a compatible free peer without a 31B retry storm; (c) SDK-internal retries cannot bypass the total attempt budget; (d) incorrect v1/v1beta, max-output and unsupported field are terminal adapter diagnostics; (e) blocked/leaked-key text trips credential security alert, no failover with the same key; (f) RECITATION/OTHER/lang and malformed/repetitive tool/JSON are not health outages; (g) midstream SSE/Live errors never replay committed output; (h) all behavior remains privacy-gated and paid-spill gated. Preserve current OpenRouter/DeepSeek behavior. Owner's complete 26-model Free-tier portfolio is in the controlling capabilities dossier; apply these tests to shared transport families, not per-model duplicated handlers.

## Ship-focused implementation steering

- Implement as a **small provider-aware error decoder and policy mapping**, ideally extending `crates/libsy-llm-client/src/run.rs::failure_class` and `client.rs::is_transient_failure / is_account_wide_quota` without changing unrelated OpenRouter recovery; integrate with per-provider transport decoder/SSE/typed sessions. Don't create a second router or a new reporting system. Use `crates/switchyard-runner/src/composition.rs` for the existing privacy/economy-crossing semantics.
- Separate **provider error diagnosis** from **routing decision**: Google says exponential backoff for transient conditions, but it does not authorize retries, provider switching or spending money on the user's behalf. Switchyard decides based on the owner's privacy and economy contract.
- Incorporate this error handling in active PR #46's R3 measured Gemma fixes, then reuse for the **entire 26-actionable Free-tier portfolio** by API family, not 26 bespoke error handlers. Antigravity remains deferred; Stephen `ST_Route` expansion remains future private Agent Stack Issue #58.
- **No calls, deploys or merges are authorized by this document.** Current R1, rollback anchor, strict-NT consumers and Turnstone primary stay unchanged. Deliver a focused code change, offline tests, and one production-ready staged readback; ask for the next owner GO separately, not another documentation/research cycle.

## Implementation record - what is now enforced in code

`crates/libsy-llm-client/src/google.rs` is the one decoder. It recognizes the four envelopes
(Interactions buffered, Interactions streaming `event_type:"error"` after HTTP 200, native
gRPC-style `status`, and the OpenAI-compatible object including the measured list-wrapped
503) and reduces them to one `GoogleCondition`. It only *diagnoses*: every actionable field
lives in `GoogleRecovery`, which the existing declassifier and route read.

Wiring, all inside mechanisms that already existed - no second router:

| seam | behaviour |
| --- | --- |
| `client.rs::is_account_wide_quota` | SUPERSEDED 2026-11-21: a Google `quota_exceeded` is account-wide **only** when the body positively scopes the quota to the project; a per-model or unscoped one retires just that model. `rate_limit_exceeded` is still not account-wide. |
| `client.rs::is_transient_failure` | A condition whose `retry_same_model` is false is not resent, whatever the status says. This is what keeps a daily quota that arrives behind a 5xx from being retried. |
| `run.rs::failure_class` | Conditions map onto the existing classes: `DailyQuota → AccountQuota`, `ShortWindowThrottle → RateLimited`, `Capacity → TemporarilyUnavailable`, `ContentBlocked → ProviderPolicy`, `ModelNotFound → ModelRetired`, credential codes → `Auth`, structural codes → `Request`. |
| `run.rs::fallback_reason` | A condition with `keep_outcome` is not routed around. A policy block, a malformed tool call or a cancelled call is an answer; handing it to another provider would be evading it. |

Deliberate conservatism, all of it test-pinned:

- bare `RESOURCE_EXHAUSTED` on 429 stays a **window throttle** - promoting it to a daily
  quota would stop every candidate until midnight on a condition that clears in a minute;
- an unmapped subtype falls back to the HTTP status, and an unmapped status stays `Unknown`
  with no permissions granted;
- recognition requires positive Google evidence (uppercase gRPC status, `google.rpc.` detail
  type, a documented snake_case code, or the leaked-key sentence), so OpenRouter's
  `limit_source` / masked `server_error` contracts and DeepSeek's body keep their existing
  classification. That separation is itself a test.
- `Retry-After` is parsed and capped by `capped_retry_after`; a date-form value is ignored
  rather than guessed at. Switchyard speaks REST to Google, so there is no SDK retry layer
  underneath to multiply the attempt budget.

Offline coverage (no network in any of these): the decoder's twelve unit tests cover both
429 codes, the list-wrapped 503, the SSE error event, model-vs-resource 404, policy blocks,
malformed tool calls, the leaked-key sentence, the conservative fallbacks and the
other-provider regression guard. Two declassification tests assert the same conditions
through `is_transient_failure` / `is_account_wide_quota` and through
`failure_class` / `fallback_reason`. Four mutants - the account-wide clause, the
retry-same-model gate, the class mapping and the keep-outcome arm - were each removed in turn
and each turned a named test red.
