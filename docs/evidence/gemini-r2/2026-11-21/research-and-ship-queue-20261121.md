# Gemini free direct capabilities: targeted source research and shipment queue
2026-11-21 · PR #45 supplementary research · **source-only, no provider call or deployment**

## Contract and evidence

All 27 model/agent entries with owner-attested nonzero/Unlimited RPD appear below. Full 47+23 inventory remains the authoritative machine-readable dataset, including zero-quota/feature rows. Native IDs, methods, architecture input limits come from the 2026-11-20 cached project `/v1beta/models` listing. Official docs linked per family (specific official model card when available) confirm *family capabilities*, not successful responses from this API key. A quota allocation does not prove serving, and the numerator is peak, NOT current-day consumption.

**Antigravity is OWNER-DEFERRED.** Preserve its 100-RPD record but no live probe, no adapter research, no capability primitive, no deployment action or tracking gate. Only reconsider if owner reopens it.

| # | Display | Cached native ID | Native operation | Advertised methods | Cached max input | RPM/TPM/RPD | Official family reference | Queue |
|---:|---|---|---|---|---:|---|---|---|
| 1 | Gemini 3.6 Flash | gemini-3.6-flash | chat.complete | generateContent, countTokens, createCachedContent, batchGenerateContent | 1048576 | 5/250000/20 | [family/exact source](https://ai.google.dev/gemini-api/docs/models) | P2: chat follow-up |
| 2 | Gemini 3.8 Flash | gemini-3.8-flash | chat.complete | generateContent, countTokens, createCachedContent, batchGenerateContent | 1048576 | 5/250000/20 | [family/exact source](https://ai.google.dev/gemini-api/docs/models) | P2: chat follow-up |
| 3 | Gemini 3.5 Flash Lite | gemini-3.5-flash-lite | chat.complete | generateContent, countTokens, createCachedContent, batchGenerateContent | 1048576 | 15/250000/500 | [family/exact source](https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash-lite) | P0: wire test |
| 4 | Gemini 3.1 Flash Lite | gemini-3.1-flash-lite | chat.complete | generateContent, countTokens, createCachedContent, batchGenerateContent | 1048576 | 15/250000/500 | [family/exact source](https://ai.google.dev/gemini-api/docs/models) | P0: wire test |
| 5 | Antigravity | UNRESOLVED | agent.run | unknown | unknown | 60/100000/100 | [family/exact source](https://ai.google.dev/gemini-api/docs/pricing) | DEFERRED owner |
| 10 | Gemini 2.5 Flash | gemini-2.5-flash | chat.complete | generateContent, countTokens, createCachedContent, batchGenerateContent | 1048576 | 5/250000/20 | [family/exact source](https://ai.google.dev/gemini-api/docs/models) | P2: chat follow-up |
| 12 | Gemini 2.5 Flash Lite | gemini-2.5-flash-lite | chat.complete | generateContent, countTokens, createCachedContent, batchGenerateContent | 1048576 | 10/250000/20 | [family/exact source](https://ai.google.dev/gemini-api/docs/models) | P2: chat follow-up |
| 13 | Gemini 2.5 Flash TTS | gemini-2.5-flash-preview-tts | audio.synthesize | countTokens, generateContent | 8192 | 3/10000/10 | [family/exact source](https://ai.google.dev/gemini-api/docs/speech-generation) | P3: typed/use-case |
| 16 | Gemini 3 Flash | UNRESOLVED | chat.complete | unknown | unknown | 5/250000/20 | [family/exact source](https://ai.google.dev/gemini-api/docs/models) | P2: chat follow-up |
| 21 | Gemini 3.1 Flash TTS | gemini-3.1-flash-tts-preview | audio.synthesize | generateContent, countTokens, batchGenerateContent | 8192 | 3/10000/10 | [family/exact source](https://ai.google.dev/gemini-api/docs/speech-generation) | P3: typed/use-case |
| 22 | Gemini 3.5 Transcribe | gemini-3.5-transcribe | audio.transcribe | generateContent, countTokens | 98304 | 3/10000/25 | [family/exact source](https://ai.google.dev/gemini-api/docs/models) | P3: typed/use-case |
| 23 | Gemini 3.8 Flash Lite TTS | gemini-3.8-flash-lite-tts | audio.synthesize | generateContent, countTokens, batchGenerateContent | 8192 | 3/10000/10 | [family/exact source](https://ai.google.dev/gemini-api/docs/speech-generation) | P3: typed/use-case |
| 24 | Gemini 3.8 Flash TTS | gemini-3.8-flash-tts | audio.synthesize | generateContent, countTokens, batchGenerateContent | 8192 | 3/10000/10 | [family/exact source](https://ai.google.dev/gemini-api/docs/speech-generation) | P3: typed/use-case |
| 27 | Gemini 4 Hydrogen | gemini-4-hydrogen | chat.complete | generateContent, countTokens, batchGenerateContent | 1048576 | 5/250000/20 | [family/exact source](https://ai.google.dev/gemini-api/docs/models) | P2: chat follow-up |
| 29 | Gemini Decision 1 | gemini-decision-1 | decision.score | createDecision | 262144 | 200/50000/10000 | [family/exact source](https://ai.google.dev/gemini-api/docs/models) | P3: typed/use-case |
| 30 | Gemini Embedding 1 | gemini-embedding-001 | vector.embed | embedContent, countTextTokens, countTokens, asyncBatchEmbedContent | 2048 | 100/30000/1000 | [family/exact source](https://ai.google.dev/gemini-api/docs/embeddings) | P3: typed/use-case |
| 31 | Gemini Embedding 2 | UNRESOLVED | vector.embed | unknown | unknown | 100/30000/1000 | [family/exact source](https://ai.google.dev/gemini-api/docs/models/gemini-embedding-2) | P3: typed/use-case |
| 35 | Gemini Robotics ER 2 Preview | gemini-robotics-er-2-preview | robotics.perceive | generateContent, countTokens, createCachedContent, batchGenerateContent | 131072 | 5/250000/20 | [family/exact source](https://ai.google.dev/gemini-api/docs/models/gemini-robotics-er-2-preview) | P3: typed/use-case |
| 36 | Gemma 4 26B | gemma-4-26b-a4b-it | chat.complete | generateContent, countTokens | 262144 | 30/16000/14400 | [family/exact source](https://ai.google.dev/gemma/docs/core/gemma_on_gemini_api) | P1: bounded |
| 37 | Gemma 4 31B | gemma-4-31b-it | chat.complete | generateContent, countTokens | 262144 | 30/16000/14400 | [family/exact source](https://ai.google.dev/gemma/docs/core/gemma_on_gemini_api) | P1: bounded |
| 41 | Lyria Realtime | lyria-realtime-exp | music.stream | bidiGenerateMusic | 1048576 | 15/-/1500 | [family/exact source](https://ai.google.dev/api/live_music) | P3: typed/use-case |
| 42 | Gemini 2.5 Flash Native Audio Dialog | UNRESOLVED | session.live | unknown | unknown | Unlimited/1000000/Unlimited | [family/exact source](https://ai.google.dev/gemini-api/docs/live-api) | P3: typed/use-case |
| 43 | Gemini 3 Flash Live | UNRESOLVED | session.live | unknown | unknown | Unlimited/65000/Unlimited | [family/exact source](https://ai.google.dev/gemini-api/docs/live-api) | P3: typed/use-case |
| 44 | Gemini 3.5 Live Translate | gemini-3.5-live-translate-preview | session.live | bidiGenerateContent | 16384 | Unlimited/20000/Unlimited | [family/exact source](https://ai.google.dev/gemini-api/docs/live-api) | P3: typed/use-case |
| 45 | Gemini 3.5 Transcribe Live | gemini-3.5-transcribe-live | session.live | bidiGenerateContent | 131072 | Unlimited/20000/Unlimited | [family/exact source](https://ai.google.dev/gemini-api/docs/live-api) | P3: typed/use-case |
| 46 | Gemini 3.8 Live | gemini-3.8-live | session.live | bidiGenerateContent | 131072 | Unlimited/65000/Unlimited | [family/exact source](https://ai.google.dev/gemini-api/docs/live-api) | P3: typed/use-case |
| 47 | Gemini 3.8 Live Extended Thinking | gemini-3.8-live-extended-thinking | session.live | bidiGenerateContent | 131072 | Unlimited/65000/Unlimited | [family/exact source](https://ai.google.dev/gemini-api/docs/live-api) | P3: typed/use-case |

## Source-confirmed interface notes / limitations

1. Hosted Gemma 4 exact IDs `gemma-4-31b-it` and `gemma-4-26b-a4b-it`; native `generateContent`, multimodal input, tool/function calls and reasoning support documented at https://ai.google.dev/gemma/docs/core/gemma_on_gemini_api . Cached architecture limit 262144 input tokens but owner's Free project gives just 16000 input TPM each. For Turnstone bounded tasks, admit small requests only. Mock actual wire + tool continuation + JSON; then owner-authorized one tiny synthetic request each.
2. Gemini 3.5 Flash-Lite official exact model card https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash-lite covers multimodal inputs and tools, but Switchyard currently truthfully advertises *text only*. Do not silently upgrade vision. 3.1 Flash-Lite has a cached `generateContent` listing and the same no-live-Google proof gap.
3. Embedding 2 official *stable* exact native ID is `gemini-embedding-2`; documentation-level ID ambiguity is resolved. It requires typed embedding transport, separate vector-space/dimensionality from the 1024-d local embedding index, and later use-case-specific work. See https://ai.google.dev/gemini-api/docs/models/gemini-embedding-2 .
4. Decision 1 project list states `gemini-decision-1` and `createDecision`, not chat `generateContent`; existing Switchyard typed decision path should be tested with a mock when needed; public exact-model doc not independently confirmed.
5. TTS is audio output, not a chat text response. Use model-specific speech-generation voice/payload adapter. Docs: https://ai.google.dev/gemini-api/docs/speech-generation . Live has stateful bidirectional `BidiGenerateContent` sessions; Unlimited RPD does not bypass TPM/session controls. Docs: https://ai.google.dev/gemini-api/docs/live-api .
6. Lyria Realtime uses independent WebSocket `BidiGenerateMusic`: https://ai.google.dev/api/live_music ; robotics ER2 is a distinct physical-world task: https://ai.google.dev/gemini-api/docs/models/gemini-robotics-er-2-preview . Neither participates in bounded chat fallback.
7. Gemini 3.6 and 3.8 Flash have unusual owner-documented selected-window peaks 23/20 and 22/20 RPD; preserve as shown and do not claim known current remaining. Gemini 2.5 Flash-Lite had a prior 404 on this project; do not assume serving from quota alone.

## Specific PR #45 corrections — no new discovery project

- The existing R1 trial composite's private/unknown branch **does not bypass Google and reach Luna**. Deployed unsigned `smart-nt-bounded-trial` returned 403. Source `route.rs` returns terminal `PrivacyRefused` when all Google candidates are filtered, and `composition.rs` does not cross that class. Implement a narrow composite *ineligible/skip* outcome with mock proof; direct Gemini leaves must keep 403.
- Multiple *capability primitives* may reference one canonical physical model without duplicating quota. Deduplicate within one task, and share project/model RPM/TPM/RPD. Remove generator's `one_primitive_per_candidate` invariant as a requirement for release and its `two primitive` gate. A useful Gemini composite already exists deployed with two Lite direct routes. Do not create redundant aliases merely to satisfy shape.
- Current Switchyard picks candidates in declared order after eligibility filtering, **not adaptive best-model ranking**. Keep privacy & economic sequence fixed; add the smallest runtime measured selector within eligible free models, using tool/context fit, request feasibility, rolling latency, reliability and quota. No ML router or months-long benchmarking.
- Context: start Turnstone bounded `smart-nt` with **65536 context** if only one setting; prefer 8K–16K routine request input. Switchyard should truthfully support an aggregate **262144** maximum only where *at least one eligible physical target can fit input plus output reserve*, and must apply physical per-target input and project TPM/RPM/RPD guards. Gemma 16K TPM is not a 262144 budget.

## One production queue — avoid ceremonial milestones

**P0** (existing deployment): run *only when owner has authorized live calls* the already-staged 5-step synthetic Lite-only script: 2 no-egress 403 controls; one 3.5 direct, one 3.1 direct, one `smart_gemini` request, zero retries, stop on any anomaly; no paid fallback. Publish exact served IDs/response/counters.
**P1**: native Gemma 31B then 26B support and mocks for text + tools + structured output/continuation; tightly capped authorized synthetic serving proof per model. Record true status, not assumptions.
**P2**: narrow per-target context/input TPM/RPM guard, private→Luna bypass and direct privacy refusal tests; minimal outcome/latency-aware selection among only verified compatible free leaves. Stage one opt-in `smart-nt` trial and compare actual Turnstone bounded tasks vs DeepSeek Flash, preserving current aliases and rollback.
**P3**: expand chat and typed capabilities only as useful workloads justify; leave all 27 catalogued. **No Antigravity work**.

## Change authorization boundaries

This document is source research + task priority; it is NOT an instruction to call Google, merge a PR, restart a service, repoint existing consumers or expose credentials. Live smoke test remains gated unless a separate owner execution instruction is provided. PR #43/#44 review scope and current R1 rollback remain intact.