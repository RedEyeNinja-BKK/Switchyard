> **LATEST PR #46 R3 REVIEW 2026-11-21:** The two commits `d9c77865` and `11bc8b3b` add anti-thought-leak translation and Google envelope decoding; neither is deployed. **Read [two mandatory release corrections + short production checklist](OWNER-R3-11BC-PRODUCTION-CORRECTIONS.md) @ `f8a6078d`**: an Interactions `quota_exceeded` is not automatically **account-wide** when project quotas are per model/feature, and UI 3.6 `23/20` / 3.8 `22/20` peak values are not current-day remaining. Also finish 26B first, shared TPM/RPM, truthful tools/route context, safe benchmark. This remains a **whole Free-tier 26 actionable model/service implementation**, including typed transports; Gemini-only Lite five-step remains owner-GO gated. No new external calls or deployment from this doc.

> **API ERROR HANDLING CONTROL — 2026-11-21:** Google official [Interactions API error reference](https://ai.google.dev/gemini-api/docs/api-errors) has been reviewed and translated into the [all-capabilities Switchyard error handling specification](GEMINI-ERROR-HANDLING-POLICY.md) @ `140bc33eefb461cd0e047d0953d475de122fc601`. This is REQUIRED across the 26 actionable Free-tier model/service contracts (chat, TTS, ASR, Decision, embedding, robotics, realtime music and six Live variants), not a Gemma-only fix. Crucial: distinguish `429 rate_limit_exceeded` (short-window RPM/TPM) from `429 quota_exceeded` (daily quota), 503 overload from quota, 404 exact model-not-found from missing asset, policy blocks from retryable infrastructure, and SSE `event_type=error` after HTTP 200. Preserve native API/OpenAI-compat envelope differences and privacy/economic gates. Turnstone IMPLEMENTS offline normalization tests and scoped fixes; do NOT start new external research or Google calls from this directive.

# OWNER CONTROLLING IMPLEMENTATION DOSSIER — all Google AI Studio Free-tier lanes
2026-11-21 · Switchyard PR #46 source note, **all capability families** · no external-call or deployment authorization.

> **Critical correction:** Turnstone's PAT **can access all owner repositories**, including `RedEyeNinja-BKK/agent-stack-private-development` and `RedEyeNinja-BKK/Switchyard`. Earlier assistant assertions of a PAT permission boundary are **withdrawn**. When a GitHub ref/commit/comment 404s, check the actual repository remote, branch, full SHA, exact file path, GitHub/API URL and credentials used; do NOT infer missing permission. Private Agent Stack Issue #35 remains the production owner thread. This dossier is an implementation reference mirrored in the Switchyard code branch, **not a replacement** for that governance.

## Mission — what to implement, not what to research

The owner personally supplied a complete **47 model/agent + 23 tool quota UI** for the Google AI Studio **Default Gemini Project / Free tier**. Already captured in [the machine-readable inventory on PR #45](https://github.com/RedEyeNinja-BKK/Switchyard/blob/feat/gemini-r2-capability-inventory-20261121/docs/evidence/gemini-r2/2026-11-21/direct-google-capability-inventory.json) and [owner crosswalk in private Agent Stack](https://github.com/RedEyeNinja-BKK/agent-stack-private-development/blob/docs/gemini-r0-responses-handoff-20261120/docs/evidence/gemini-r1/2026-11-21/owner-free-project-all-47-capability-crosswalk.json). **There are 21 finite RPD and six Unlimited RPD = 27 quota-allocated entries. Antigravity (100 RPD) is owner-DEFERRED, leaving exactly 26 actionable capability lanes.** Twenty 0/0 RPD models and 23 tool features are conserved in the inventory but not falsely activatable.

The OWNER/assistant has already researched their **model IDs from the cached project `/v1beta/models` listing, advertised generation methods and input/output token architecture, Google official family API docs, and owner-specific RPM/input TPM/RPD**. This table is the **actual source handoff**; Turnstone should implement and run only *authorized* transport probes, not start a new docs/model-discovery effort or ask for screenshots. Some IDs are explicitly unresolved because the cached listing has a version mismatch; engineering should handle only the narrow mapping at activation rather than guessing.

## All 26 actionable entries — exact account quota, model details, native transport and route family

Table amounts are **RPM / input TPM / RPD limit denominators**, not currently remaining. Architectural input/output limits are from the cached project model listing, not necessarily the effective Free-tier request size. Methods describe the cached listing, not a successful call. URLs point to official exact-model guidance when available, otherwise Google family API contracts.

| # | UI model | Native model ID or exact candidates | RPM / TPM / RPD | Cached max input / output tokens | Cached API methods | Capability primitive | Google developer source | Verified serving evidence |
|---:|---|---|---:|---:|---|---|---|---|
| 1 | Gemini 3.6 Flash | `gemini-3.6-flash` | 5 / 250,000 / 20 | 1,048,576 / 65,536 | `generateContent, countTokens, createCachedContent, batchGenerateContent` | smart-gemini-chat / -bounded / -long-context; vision only if verified | [API evidence](https://ai.google.dev/gemini-api/docs/models) | Project listing / docs only; serving unproven |
| 2 | Gemini 3.8 Flash | `gemini-3.8-flash` | 5 / 250,000 / 20 | 1,048,576 / 65,536 | `generateContent, countTokens, createCachedContent, batchGenerateContent` | smart-gemini-chat / -bounded / -long-context; vision only if verified | [API evidence](https://ai.google.dev/gemini-api/docs/models) | Project listing / docs only; serving unproven |
| 3 | Gemini 3.5 Flash Lite | `gemini-3.5-flash-lite` | 15 / 250,000 / 500 | 1,048,576 / 65,536 | `generateContent, countTokens, createCachedContent, batchGenerateContent` | smart-gemini-chat / -bounded / -long-context; vision only if verified | [API evidence](https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash-lite) | Deployed target, Google wire not tested |
| 4 | Gemini 3.1 Flash Lite | `gemini-3.1-flash-lite` | 15 / 250,000 / 500 | 1,048,576 / 65,536 | `generateContent, countTokens, createCachedContent, batchGenerateContent` | smart-gemini-chat / -bounded / -long-context; vision only if verified | [API evidence](https://ai.google.dev/gemini-api/docs/models) | Deployed target, Google wire not tested |
| 5 | Gemini 2.5 Flash | `gemini-2.5-flash` | 5 / 250,000 / 20 | 1,048,576 / 65,536 | `generateContent, countTokens, createCachedContent, batchGenerateContent` | smart-gemini-chat / -bounded / -long-context; vision only if verified | [API evidence](https://ai.google.dev/gemini-api/docs/models) | Project listing / docs only; serving unproven |
| 6 | Gemini 2.5 Flash Lite | `gemini-2.5-flash-lite` | 10 / 250,000 / 20 | 1,048,576 / 65,536 | `generateContent, countTokens, createCachedContent, batchGenerateContent` | smart-gemini-chat / -bounded / -long-context; vision only if verified | [API evidence](https://ai.google.dev/gemini-api/docs/models) | Historical project 404; hold executable |
| 7 | Gemini 2.5 Flash TTS | `gemini-2.5-flash-preview-tts` | 3 / 10,000 / 10 | 8,192 / 16,384 | `countTokens, generateContent` | smart-gemini-tts | [API evidence](https://ai.google.dev/gemini-api/docs/speech-generation) | Project listing / docs only; serving unproven |
| 8 | Gemini 3 Flash | **ID: unresolved** (gemini-3-flash-preview) | 5 / 250,000 / 20 | — / — | `family method; exact ID needs mapping` | smart-gemini-chat / -bounded / -long-context; vision only if verified | [API evidence](https://ai.google.dev/gemini-api/docs/models) | Project listing / docs only; serving unproven |
| 9 | Gemini 3.1 Flash TTS | `gemini-3.1-flash-tts-preview` | 3 / 10,000 / 10 | 8,192 / 16,384 | `generateContent, countTokens, batchGenerateContent` | smart-gemini-tts | [API evidence](https://ai.google.dev/gemini-api/docs/speech-generation) | Project listing / docs only; serving unproven |
| 10 | Gemini 3.5 Transcribe | `gemini-3.5-transcribe` | 3 / 10,000 / 25 | 98,304 / 32,768 | `generateContent, countTokens` | smart-gemini-transcribe | [API evidence](https://ai.google.dev/gemini-api/docs/audio) | Project listing / docs only; serving unproven |
| 11 | Gemini 3.8 Flash Lite TTS | `gemini-3.8-flash-lite-tts` | 3 / 10,000 / 10 | 8,192 / 16,384 | `generateContent, countTokens, batchGenerateContent` | smart-gemini-tts | [API evidence](https://ai.google.dev/gemini-api/docs/speech-generation) | Project listing / docs only; serving unproven |
| 12 | Gemini 3.8 Flash TTS | `gemini-3.8-flash-tts` | 3 / 10,000 / 10 | 8,192 / 16,384 | `generateContent, countTokens, batchGenerateContent` | smart-gemini-tts | [API evidence](https://ai.google.dev/gemini-api/docs/speech-generation) | Project listing / docs only; serving unproven |
| 13 | Gemini 4 Hydrogen | `gemini-4-hydrogen` | 5 / 250,000 / 20 | 1,048,576 / 65,536 | `generateContent, countTokens, batchGenerateContent` | smart-gemini-chat / -bounded / -long-context; vision only if verified | [API evidence](https://ai.google.dev/gemini-api/docs/models) | Project listing / docs only; serving unproven |
| 14 | Gemini Decision 1 | `gemini-decision-1` | 200 / 50,000 / 10,000 | 262,144 / 1 | `createDecision` | smart-gemini-decision | [API evidence](https://ai.google.dev/gemini-api/docs) | Project listing / docs only; serving unproven |
| 15 | Gemini Embedding 1 | `gemini-embedding-001` | 100 / 30,000 / 1,000 | 2,048 / 1 | `embedContent, countTextTokens, countTokens, asyncBatchEmbedContent` | smart-gemini-embedding | [API evidence](https://ai.google.dev/gemini-api/docs/embeddings) | Project listing / docs only; serving unproven |
| 16 | Gemini Embedding 2 | **ID: unresolved** (gemini-embedding-2 / gemini-embedding-2-preview) | 100 / 30,000 / 1,000 | — / — | `family method; exact ID needs mapping` | smart-gemini-embedding | [API evidence](https://ai.google.dev/gemini-api/docs/models/gemini-embedding-2) | Official stable ID `gemini-embedding-2` documented; project identity untested |
| 17 | Gemini Robotics ER 2 Preview | `gemini-robotics-er-2-preview` | 5 / 250,000 / 20 | 131,072 / 65,536 | `generateContent, countTokens, createCachedContent, batchGenerateContent` | smart-gemini-robotics | [API evidence](https://ai.google.dev/gemini-api/docs/models/gemini-robotics-er-2-preview) | Project listing / docs only; serving unproven |
| 18 | Gemma 4 26B | `gemma-4-26b-a4b-it` | 30 / 16,000 / 14,400 | 262,144 / 32,768 | `generateContent, countTokens` | smart-gemini-chat / -bounded / -long-context; vision only if verified | [API evidence](https://ai.google.dev/gemma/docs/core/gemma_on_gemini_api) | WIRE: text, tools + stripped tool round-trip yes; inline `<thought>`; JSON object broken |
| 19 | Gemma 4 31B | `gemma-4-31b-it` | 30 / 16,000 / 14,400 | 262,144 / 32,768 | `generateContent, countTokens` | smart-gemini-chat / -bounded / -long-context; vision only if verified | [API evidence](https://ai.google.dev/gemma/docs/core/gemma_on_gemini_api) | WIRE: text+tools yes; slow/frequent 503; JSON schema yes; tool continuation unproven; inline `<thought>` |
| 20 | Lyria Realtime | `lyria-realtime-exp` | 15 / - / 1,500 | 1,048,576 / 65,536 | `bidiGenerateMusic` | smart-gemini-music-realtime | [API evidence](https://ai.google.dev/api/live_music) | Project listing / docs only; serving unproven |
| 21 | Gemini 2.5 Flash Native Audio Dialog | **ID: unresolved** (gemini-2.5-flash-native-audio-latest / gemini-2.5-flash-native-audio-preview-12-2025 / gemini-2.5-flash-native-audio-preview-09-2025) | Unlimited / 1,000,000 / Unlimited | — / — | `family method; exact ID needs mapping` | smart-gemini-live / -live-translate / -live-transcribe | [API evidence](https://ai.google.dev/gemini-api/docs/live-api) | Project listing / docs only; serving unproven |
| 22 | Gemini 3 Flash Live | **ID: unresolved** (gemini-3.1-flash-live-preview) | Unlimited / 65,000 / Unlimited | — / — | `family method; exact ID needs mapping` | smart-gemini-live / -live-translate / -live-transcribe | [API evidence](https://ai.google.dev/gemini-api/docs/live-api) | Project listing / docs only; serving unproven |
| 23 | Gemini 3.5 Live Translate | `gemini-3.5-live-translate-preview` | Unlimited / 20,000 / Unlimited | 16,384 / 32,768 | `bidiGenerateContent` | smart-gemini-live / -live-translate / -live-transcribe | [API evidence](https://ai.google.dev/gemini-api/docs/live-api) | Project listing / docs only; serving unproven |
| 24 | Gemini 3.5 Transcribe Live | `gemini-3.5-transcribe-live` | Unlimited / 20,000 / Unlimited | 131,072 / 65,536 | `bidiGenerateContent` | smart-gemini-live / -live-translate / -live-transcribe | [API evidence](https://ai.google.dev/gemini-api/docs/live-api) | Project listing / docs only; serving unproven |
| 25 | Gemini 3.8 Live | `gemini-3.8-live` | Unlimited / 65,000 / Unlimited | 131,072 / 65,536 | `bidiGenerateContent` | smart-gemini-live / -live-translate / -live-transcribe | [API evidence](https://ai.google.dev/gemini-api/docs/live-api) | Project listing / docs only; serving unproven |
| 26 | Gemini 3.8 Live Extended Thinking | `gemini-3.8-live-extended-thinking` | Unlimited / 65,000 / Unlimited | 131,072 / 65,536 | `bidiGenerateContent` | smart-gemini-live / -live-translate / -live-transcribe | [API evidence](https://ai.google.dev/gemini-api/docs/live-api) | Project listing / docs only; serving unproven |

## Group-specific adapter contracts; Turnstone DOES NOT need to research these again

- **10 chat/bounded leaves:** `generateContent` or the actual qualified OpenAI-compatible chat adapter; system, tool calls/continuation, structured JSON, input media and reasoning **per exact model**. Physical context/TPM/RPM/RPD and final payload privacy admission before dispatch. Primitive `smart-gemini-chat`, `-bounded`, `-long-context`, `-vision` only for proven carriers. Canonical direct identity remains one native model, not a hidden fallback. Two Flash-Lite directs are *deployed but still untested against Google*.
- **Four TTS entries:** [official speech generation](https://ai.google.dev/gemini-api/docs/speech-generation); AUDIO output, voice configuration, audio codecs/limits; typed `smart-gemini-tts`, never treated as chat text. Each 10 RPD.
- **One offline audio-to-text transcription:** `gemini-3.5-transcribe`; [Google audio API](https://ai.google.dev/gemini-api/docs/audio); audio carrier + textual response, 25 RPD. `smart-gemini-transcribe`.
- **One Decision:** `gemini-decision-1`, cached listing `createDecision`, 10,000 RPD; typed decision score/protocol, not chat; bind existing Switchyard decision support where possible. No dedicated exact-model official page established in supplied sources; don't spend a sprint re-searching, mock the native method and label exact serving unverified.
- **Two embeddings:** `gemini-embedding-001` via `embedContent`; **official stable** `gemini-embedding-2` [model card](https://ai.google.dev/gemini-api/docs/models/gemini-embedding-2) 8,192-token input, multimodal text/image/video/audio/PDF, selectable 128–3,072 output dimensions. Do not conflate new embedding vectors with existing 1,024-dimensional Qwen index: separate vector space/index/score calibration. 1,000 RPD each. `smart-gemini-embedding`.
- **One robotics:** `gemini-robotics-er-2-preview`, [Google Robotics ER2 model](https://ai.google.dev/gemini-api/docs/models/gemini-robotics-er-2-preview); specialized embodied/vision/physical-state contract under `smart-gemini-robotics`, not generic bounded chat, 20 RPD.
- **One realtime music:** `lyria-realtime-exp`, [Google Live Music API](https://ai.google.dev/api/live_music), WebSocket `BidiGenerateMusic`, separate audio generator and music session limits, 1,500 RPD. `smart-gemini-music-realtime`.
- **Six unlimited-RPD Live variants:** [Google Live API](https://ai.google.dev/gemini-api/docs/live-api), `BidiGenerateContent` stateful sessions, input audio/video/text, output speech and task-specific semantics; four exact IDs cached, two UI-to-version matches pending (2.5 native-audio and 3 Flash Live). **Unlimited daily RPD does not mean unlimited TPM, concurrency or session duration.** Typed `smart-gemini-live`, `-live-translate`, `-live-transcribe`, not OpenAI chat/composite.
- **Antigravity:** preserve the original 100-RPD row, but **no work** until the owner reopens it. No agent or sandbox adapter or cost/credential request.
- **No 0/0 allocations:** Nano Banana image models, Gemini Omni, Genie, Lyria Clip/Pro and other 0/0 rows remain inventory only. Don't create phantom free image/video lanes.

## Known real response quirks, already proven — use these fixtures; do not repeat

[Gemma 4 direct probe receipt](https://github.com/RedEyeNinja-BKK/Switchyard/blob/feat/gemini-r3-adaptive-bounded/docs/evidence/gemini-r3/2026-11-21/gemma-capability-verification.md) and its adjacent `live-probe/*.jsonl` are **25 already-run small chat requests + two listings, not a request for more**.
- 26B A4B: quick successful completions, real function call, **stripped assistant tool-call + result continuation completed**. Promote preferred candidate **after thought-safety fix**.
- 31B: actual text and tool calls but frequent `503 high demand` and slow successful turns; **tool-result continuation still unproven**. Make 31B only eligible as secondary, health/cooldown driven; 503 may move to another eligible free leg but not trigger surprise paid fallback.
- **Blocking anti-leak:** both return literal leading `<thought>…</thought>` in visible `content`. Separate/suppress only confirmed hosted Gemma reasoning blocks in the adapter (including streaming fragments), preserve actual answer/tool/JSON and refuse malformed or ambiguous content rather than leak thought text. Never expose the hidden thought in logs or downstream agent text.
- `json_object` output contains thoughts and is **not valid JSON**; 31B `json_schema` produced clean JSON, 26B schema mode is unverified. `reasoning_effort:none` produced 400 on 31B; no reasoning-off claim. Cached `reasoning=false` flag is not a guarantee it stops internally thinking.
- Official [hosted Gemma API](https://ai.google.dev/gemma/docs/core/gemma_on_gemini_api) documents native model IDs, `:generateContent`, system instructions, function calling and thinking levels **for the native API**. The live receipts establish the distinct OpenAI-compatible wire quirks. Do not misstate Gemma as entirely undocumented.
- Both Gemmas: physical input architecture 262,144 tokens vs **16,000 input TPM project quota**, 30 RPM, 14,400 RPD each. A 16K single-input cap does **not** implement rolling TPM. Separate physical capacity from effective budget; do not advertise 262K usable single turn on a Gemma-only route.

## How ALL capabilities fit Switchyard (not a forced all-chat ladder)

```
project-scoped canonical direct contracts (26 actionable, Antigravity deferred)
    |
    +-- compatible inference target pool (10 chat, context/vision/tools qualified individually)
    |     +-- smart-gemini-chat / smart-gemini-bounded / -long-context / -vision
    |     +-- existing smart_gemini top-level inference composite
    |     +-- opt-in smart-nt-bounded-trial -> Luna-NT -> DeepSeek-Flash-NT
    |
    +-- native typed services, NOT children of a chat composite:
          smart-gemini-tts (4)                   smart-gemini-transcribe (1)
          smart-gemini-decision (1)              smart-gemini-embedding (2)
          smart-gemini-robotics (1)              smart-gemini-music-realtime (1)
          smart-gemini-live/translate/transcribe (6)
```

**Selection intent:** hard privacy approval per final payload and *economic tier* first; within actually compatible available Free models select based on task, tool and media fit, input/output/context, provider RPM/TPM/RPD, health and measured success/latency. A physical model can belong to several primitives but shares ONE canonical project-model quota and must not be retried twice in one request. Existing Switchyard composite children are `[routes]`, not nested `[composites]`. Don't force an artificial minimum number of primitives before shipping a working route. Build typed-family adapters once and attach all qualifying Free endpoints through them, rather than bespoke 26 implementations.

**Privacy:** no free Google outbound of unknown/private/uncleared data; direct Google refuses without actual trusted clearance, mixed `smart-nt` skips Google to Luna then DeepSeek under the configured paid-eligibility policy. Current HMAC `class/scope/day` is not request-body-bound and regex scanner incomplete: do not equate generic client header with permission. Unauthorised provider bypass and surprise paid fallback are both forbidden.

**Context:** Turnstone bounded 65,536 working cap suggested if owner manually sets it, normal short tasks ~8–16K; Switchyard aggregate may advertise **up to 262,144 only when some eligible physical target truly fits**. Individual Gemma cannot use that much with project 16K TPM. No silent consumer repointing.

## Exact engineering handoff: do the builds, not the discovery

1. **Use this complete 26-entry spec as input.** Every one should have canonical direct *capability* definition + typed connector family; statuses must be honest and testable. Run crosswalk conservation checks **47+23, 27 quota-present, 1 explicitly deferred, 26 actionable, 20 zero**, no missing/free invented model. Do NOT re-search all 26 or ask for a new screenshot. Unresolved two Live and one Gemini 3 Flash ID mappings are activation blockers for *those leaves only*; others proceed.
2. **Finish PR #46 immediately:** anti-leak translation/stream safety, 26B preference and 31B 503 policy, correct schema/tool capabilities; real RPM+input TPM pacing (not just per-request ceiling), narrowly scoped measured warmup, paid-safe honest comparison reporting. Test against provided receipts. Stage the compatible 10 chat leaves **truthfully** (currently only 2 Lite active + 2 Gemma wire-proven), don't enable held ones until actual *separate owner-authorized* wire qualification.
3. **Next deliver typed adapters in coherent batches** from the already researched contracts: embedding/Decision, speech TTS + transcription, Live + live translation/transcription, realtime music, robotics. Each batch should take its existing source/native API methods and add direct endpoints plus grouped primitives with realistic usage limits, signed/privacy-aware traffic, tests and reversible activation. **Do not park all non-chat functionality indefinitely behind generic “future use case” phrasing**. Do not fake Chat Completions adapters. Ship completed compatible capabilities incrementally under the same source plan, not 26-model simultaneous GO.
4. **No more external calls without owner GO.** The next small decision is the already-staged Lite 5-step synthetic test (two zero-call privacy controls + at most three tiny Google requests); it remains unrun. Do not expand it into another 20-call package. Later independently authorize narrow typed-operation probes and release(s), with strict call/cost bounds. No unrelated downloads or broad benchmark campaigns.
5. **Production release:** use Issue #35 owner-writer path, verify hashes, existing 49-route R1/rollback, one opt-in reversible release per compatible group and real end-to-end readback. Never reassign existing `smart-nt` or Turnstone primary without separate request. Stephen remote OpenClaw ST_Route is future private Issue #58, not today's work; **reuse this portfolio later**.

**Completion criterion:** all 26 direct contracts accounted for and assigned to concrete typed routes/adapters; actual API family implementation shipped progressively with correct eligibility; privacy and economics enforced and real usage verified. Report what WORKS and the exact next release decision—not another evidence inventory or re-learning Google docs.

## Complete nonallocated and separate tool inventory (unchanged source)
Zero-RPD model names (20, do not activate): Deep Research Pro Preview; Gemini 2 Flash; Gemini 2 Flash Lite; Computer Use Preview; Nano Banana (Gemini 2.5 Flash Preview Image); Gemini 2.5 Pro; Gemini 2.5 Pro TTS; Nano Banana Pro (Gemini 3 Pro Image); Gemini 3.1 Pro; Nano Banana 2 (Gemini 3.1 Flash Image); Nano Banana 2 Lite (Gemini 3.1 Flash Lite Image); Gemini 4 Argon; Gemini 4 Carbon; Gemini 4 Neon; Gemini Nano Banana 2.1; Gemini Omni 1.1 Flash; Gemini Omni 1.2 Flash; Genie 4; Lyria 3 Clip; Lyria 3 Pro.

Tool quota rows (23, never count as model routes):

| Feature | RPD limit | Route status |
|---|---:|---|
| Map grounding / Deep Research Pro Preview | 0 | No free tool allowance |
| Map grounding / Gemini 2 Flash | 0 | No free tool allowance |
| Map grounding / Computer Use Preview | 0 | No free tool allowance |
| Map grounding / Gemini 2.5 Flash | 0 | No free tool allowance |
| Map grounding / Gemini 2.5 Flash Lite | 0 | No free tool allowance |
| Map grounding / Gemini 2.5 Pro | 0 | No free tool allowance |
| Map grounding / Gemini 3 Flash | 0 | No free tool allowance |
| Map grounding / Gemini 3.1 Pro | 0 | No free tool allowance |
| Map grounding / Gemini 3.1 Flash Lite | 0 | No free tool allowance |
| Map grounding / Gemini 3.1 Flash TTS | 0 | No free tool allowance |
| Map grounding / Gemini 3.6 Flash | 0 | No free tool allowance |
| Map grounding / Gemini 3.5 Flash Lite | 0 | No free tool allowance |
| Map grounding / Gemini 3.5 Transcribe | 0 | No free tool allowance |
| Map grounding / Gemini 3.8 Flash | 0 | No free tool allowance |
| Map grounding / Gemini 4 Argon | 0 | No free tool allowance |
| Map grounding / Gemini 4 Carbon | 0 | No free tool allowance |
| Map grounding / Gemini 4 Hydrogen | 0 | No free tool allowance |
| Map grounding / Gemini 4 Neon | 0 | No free tool allowance |
| Map grounding / Gemini Robotics ER 2 Preview | 0 | No free tool allowance |
| Search grounding / Gemini 2 | 1,500 | Available tool quota only, not a model lane |
| Search grounding / Gemini 2.5 | 1,500 | Available tool quota only, not a model lane |
| Search grounding / Gemini 3 and newer | 0 | No free tool allowance |
| Search grounding / Default | 1,500 | Available tool quota only, not a model lane |
