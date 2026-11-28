# Owner wire-format decision and seven-day Google Free-tier quota context
2026-11-21 · Applies to PR #46 + all actionable Gemini Free-tier model/service work · **documentation only; no provider call or deployment authorized**

## 1. Important owner input: Google's OpenAI-compatible endpoint is NOT the OpenAI Responses API

User-supplied vendor workaround says: if a client absolutely requires OpenAI `/responses`, use a translation proxy (e.g. OpenRouter/LiteLLM) or Google's native `google-genai` SDK rather than assuming the Google OpenAI-compatible endpoint implements it.

**Independent source check (2026-11-21):** Official Google [OpenAI compatibility documentation](https://ai.google.dev/gemini-api/docs/openai) explicitly shows `https://generativelanguage.googleapis.com/v1beta/openai/chat/completions`, including text, function calling, multimodal, streaming and structured output examples; it does NOT document an OpenAI `/responses` path. [Google Developers Forum / Google team reply, 2025-06-23](https://discuss.ai.google.dev/t/openai-responses-api-compatibility/90136/2) explicitly states that the Responses API was not implemented on the OpenAI-compatible endpoint at that time. Do not assume an unsupported endpoint has since appeared unless current Google documentation proves it. This is **OpenAI Responses API compatibility**, not Google's own separate **Interactions API** or Gemini's native GenerateContent/Live APIs.

**Decision for Switchyard:** We already have a request/response normalization layer and multiple external wire encoders. No new OpenRouter/LiteLLM dependency is required merely to supply a *supported subset* of client-facing `/responses` semantics:
- **Client ingress**: OpenAI `/v1/responses` can target Switchyard if its existing Responses decoder/encoder supports the exact used fields.
- **Provider egress**: eligible Gemini/Gemma text/chat target uses **Google's documented `/v1beta/openai/chat/completions`** (or typed native `:generateContent`/Google Interactions when a model or operation needs it). Never send client `/responses` verbatim to `/v1beta/openai/responses` and never interpret that 404 as model retirement.
- **Feature preservation test:** tool-call roundtrip + IDs/args, system instructions, structured outputs, streaming, output text, errors, reasoning privacy, media and final physical model must survive the transformation. **Stateful / Responses-specific features** (e.g. `previous_response_id`, hosted OpenAI built-in tools, encrypted or signed reasoning, background tasks, API-managed tool state) are NOT automatically equivalent to ordinary Chat Completions. If no proven lossless mapping or native equivalent exists for a required feature, refuse *before egress* with a clear unsupported-feature diagnostic. Don't silently drop the feature, invent state or falsely advertise full Responses fidelity.
- **Native provider alternative:** Google's official [Interactions API](https://ai.google.dev/gemini-api/docs/get-started) is available via REST and `google-genai` SDK; the SDK is **not** a drop-in `/responses` endpoint. Prefer a small native Google typed adapter where native features warrant it; keep Free-tier project credential, native model quota, privacy and economics identity intact.
- **Avoid third-party translation by default:** Adding OpenRouter/LiteLLM solely as a bridge introduces unnecessary provider/privacy/availability/billing complexity and can break source-project quota accounting. Consider an external proxy only on the owner's specific instruction after proving an internal/native route cannot satisfy a real need; never automatically reroute Free-tier Google requests through another provider.

**Offline acceptance:** (a) a representative Switchyard `/responses` request → canonical neutral IR → Google Chat Completions payload → back to actual `/responses` output with correct text/tool/JSON attribution, no payload loss; (b) Responses-only unsupported feature refused locally without Google egress; (c) `/v1beta/openai/responses` never constructed; (d) missing vendor endpoint 404 never classified as model lifecycle retirement; (e) schema/streaming/hidden thoughts fail closed; (f) no paid or privacy-ineligible proxy spill. Run offline tests, not fresh provider calls.

## 2. Exact owner Google AI Studio screenshot: seven-day **peak**, not remaining balance

Screenshot text reads **“Rate limits by model”** and **“Peak usage per model compared to its limit over the last 7 days”**. The following is a transcription of the **eight visible rows**, with three independent peak/limit columns:

| Model | RPM peak / limit | Input TPM peak / limit | RPD peak / limit | Eligibility interpretation |
|---|---:|---:|---:|---|
| Gemini 3.6 Flash | 3 / 5 | 53 / 250K | 23 / 20 | Peak exceeded daily denominator in 7-day view; no proof today's remaining is zero |
| Gemini 3.8 Flash | 3 / 5 | 53 / 250K | 22 / 20 | Same |
| Gemini 3.5 Flash Lite | 6 / 15 | 591 / 250K | 10 / 500 | Project allocation confirmed, direct target installed; live Switchyard Lite smoke still gated |
| Gemini 3.1 Flash Lite | 5 / 15 | 396 / 250K | 12 / 500 | Same |
| Gemma 4 31B | 4 / 30 | 175 / 16K | 15 / 14.4K | Per-model 16K input TPM; serving 503 health concern distinct from remaining |
| Gemma 4 26B | 3 / 30 | 477 / 16K | 7 / 14.4K | Same rate limits; 26B observed better serving availability |
| Antigravity | 1 / 60 | 5.45K / 100K | 3 / 100 | Owner explicitly **DEFERRED**, no work |
| Deep Research Pro Preview | 0 / 0 | 0 / 0 | 0 / 0 | Not a project Free-tier allocation |

Rows beyond screenshot and the 23 tool-feature quotas were not shown. Preserve the complete earlier owner-authenticated 47-model/23-feature inventory for those unseen entries. `0/N` denominators are project allowance; `0/0` is no shown allocation. **Peak numerators do not equal current provider-minute/provider-day remaining**. Preserve eligible records and use source-proven model identity, local attempt ledger and provider refusal details for actual admission decisions. No automatic model deactivation from a historical `23/20` peak.

## 3. Source-level production fix still necessary at PR #46 head 44ff4283

`crates/libsy-llm-client/src/client.rs::send_encoded` arms the backend failure cooldown only under `is_transient_failure(error)`. Google `503 service_unavailable/UNAVAILABLE` now correctly sets `GoogleRecovery.retry_same_model = false` (do not retry the same overloaded 31B), causing `is_transient_failure` to return false. **Consequently 31B's configured `failure_cooldown_ms=300000` may never arm on its measured Google overload!** This is a direct mismatch between `retry_same_model` and `cool_down_model`.

**Small fix:** separate the two decisions without adding a subsystem: (1) attempt retry only if the provider condition permits and deadline/budget remain; (2) arm exact-model cooldown when `GoogleRecovery.cool_down_model` is true (including 503) and not when merely a request/policy fault occurs; preserve any preexisting OpenRouter cooldown behavior. No `cooldown_group` for 31B; 26B unaffected. Offline negative test: measured list-wrapped 503 from 31B → zero same-model retry and five-minute 31B-only cooldown; 26B still eligible; no privacy, billing or stream-commit bypass. Keep retry attempts counted.

## 4. Complete current PR #46 and full Free-tier portfolio — no ceremony

- Keep the full [26 actionable Free-tier model/service dossier](OWNER-CONTROLLING-ALL-FREE-CAPABILITIES.md) in control; Antigravity deferred, zero-RPD models preserved only as records. No new broad source research. 
- Finish **rolling RPM and input TPM reservation/pacing** before any sustained automated traffic; until then an opt-in isolated route with owner-capped tiny tests only, **not** active unrestricted consumers. `max_input_tokens` alone is not a rolling meter.
- Correct `bounded_compare.py` default `smart-gemini-bounded,smart-nt`: the second route can spend Luna/DeepSeek. A fake `served_model != smart*` fallback metric is not an actual cascade. Default no-paid and enforce explicit owner-approved budgets for later paid comparison; fix tool roundtrip grading.
- One short owner-authorized Lite test first: two no-egress privacy 403 controls and **at most three** tiny Google calls. Never claim this establishes Gemma 26B vs 31B ranking; that's already separately source-observed.
- After the focused release fixes, deliver typed API families (embedding/Decision/TTS/ASR/Live/music/robotics) from supplied model research **incrementally** and with owner approval, not fake chat/Responses endpoints. Stephen's independent remote OpenClaw `ST_Route` is parked in private Issue #58.
- Current healthy R1 remains unchanged absent explicit release GO. This document is instructions to implement source + offline tests only. No new Gemini or paid wire call, deploy, PR merge, restart or consumer repoint from reading it.
