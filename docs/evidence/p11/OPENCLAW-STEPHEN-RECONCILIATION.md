# OpenClaw ↔ Stephen capability reconciliation (2026-11-29)

Host `openclaw-vm` (`100.107.123.73`), OpenClaw **2026.10.7**, gateway `127.0.0.1:18789`,
config `/home/stephen/.openclaw/openclaw.json`. Stephen's agent is the only tenant changed.

This document reconciles the **eight allowlisted Stephen chat lanes** and the other working
bindings against the **fifteen Stephen specialist capabilities** that release
`20261128T182648Z-6a8209f97101563b-02587e3e` added to Switchyard. It exists because
"8 models allowlisted" and "15 capabilities deployed" are different sets, and conflating them
would overstate what OpenClaw can reach.

## 1. What changed — two additive patches, hot-reloaded

| Patch | Path | Effect |
|---|---|---|
| 1 | `models.providers.switchyard-vm.models` | `ST_Route` kept verbatim; eight Stephen Gemini chat entries appended |
| 2 | `agents.defaults.modelPolicy.allow` | two existing entries kept; the same eight refs appended |

Both returned `requiresRestart: false`. Config hash chain
`hmac-sha256:v1:cirUiGm6…` → `…8Og8IlKI…` → `…yW_r1jeAyEVdjAZJIZKwqLthPhovWSekjbyDbT8vSDg`
(current). `openclaw config validate` as the service account → `{"valid":true,"warnings":[]}`.

**The provider catalogue alone is inert.** `models status` reads `modelPolicy.allow` as the
authoritative `allowed` set; before patch 2 the eight models existed and were unusable. Patch 2
is the gate.

## 2. Exact allowlist (10 entries)

| # | Ref | Origin |
|---|---|---|
| 1 | `kimi/kimi-k2.5` | pre-existing, untouched |
| 2 | `switchyard-vm/ST_Route` | pre-existing, default model, untouched |
| 3 | `switchyard-vm/direct-google-stephen-gemini-3-flash-preview` | added |
| 4 | `switchyard-vm/direct-google-stephen-gemini-3.1-flash-lite` | added |
| 5 | `switchyard-vm/direct-google-stephen-gemini-3.5-flash-lite` | added |
| 6 | `switchyard-vm/direct-google-stephen-gemini-3.6-flash` | added |
| 7 | `switchyard-vm/direct-google-stephen-gemini-3.8-flash` | added |
| 8 | `switchyard-vm/direct-google-stephen-gemini-4-hydrogen` | added |
| 9 | `switchyard-vm/direct-google-stephen-gemma-4-26b-a4b-it` | added |
| 10 | `switchyard-vm/direct-google-stephen-gemma-4-31b-it` | added |

All eight added entries enumerate `available: true` in `models.list`. The ten OpenRouter-keyed
Stephen routes in the manifest are deliberately **not** allowlisted: no additional provider
catalogue entries were added, per the standing scope.

## 3. Other bindings, verified live (preserved, not changed)

| Surface | Binding | Verified |
|---|---|---|
| Speech synthesis | provider `openai` → `https://switchyard-vm.tailc8ef7b.ts.net/v1`, model `stephen-gemini-tts-3.8-flash`, voice `Kore`, `wav`, `auto: tagged` | 200, 159,912 B, 24 kHz mono, 3.33 s, valid RIFF/WAVE |
| Image | `comfyninja/image` (no fallbacks) | config read |
| Video | `comfyninja/video` (no fallbacks) | config read |
| Music | `comfyninja/music` (no fallbacks) | config read |
| Memory search | `switchyard-vm/ST_Embed` (local 4096D contract, batch off) | config read; call refused — see §5 |
| Typed decisions | `switchyard-vm/ST_Decisions` via the `st-decisions` plugin | `models.list` exposes the decision model |
| Default / utility model | `switchyard-vm/ST_Route` | `models status` |

`agents.entries`, channel bindings, secrets and plugins were not edited.

## 4. Reconciliation against the fifteen Stephen specialist capabilities

Release `02587e3e` added fifteen typed capability lanes. OpenClaw 2026.10.7 exposes one speech
binding, one memory/embedding slot, one decision model and one media trio — so only one of the
fifteen is reachable from this agent, by design of the client, not by defect of the lane.

| # | Capability | Operation | OpenClaw surface | Status |
|---|---|---|---|---|
| 1 | `stephen-gemini-tts-3.8-flash` | speech | `tts.providers.openai.model` | **bound, measured 200** |
| 2 | `stephen-gemini-tts-3.8-flash-lite` | speech | none (single TTS slot) | unbound |
| 3 | `stephen-gemini-tts-3.1-flash` | speech | none | unbound |
| 4 | `stephen-gemini-tts-2.5-flash` | speech | none | unbound |
| 5 | `stephen-gemini-embedding` (3072D) | embed | none — the memory slot is `ST_Embed` (4096D) | unbound |
| 6 | `stephen-gemini-embedding-2` (3072D) | embed | none, same reason | unbound |
| 7 | `stephen-gemini-transcribe` | transcription | none — no `stt` key in 2026.10.7 | unbound |
| 8 | `stephen-gemini-live-2-5-flash-native-audio` | live | none — no Live/voice surface | unbound |
| 9 | `stephen-gemini-live-3-flash` | live | none | unbound |
| 10 | `stephen-gemini-live-3-8-flash` | live | none | unbound |
| 11 | `stephen-gemini-live-3-8-flash-extended-thinking` | live | none | unbound |
| 12 | `stephen-gemini-live-3-5-transcribe` | live | none | unbound |
| 13 | `stephen-gemini-live-3-5-translate` | live | none | unbound |
| 14 | `stephen-gemini-music-lyria-realtime` | music | `comfyninja/music` only (owner hardware, not Lyria) | unbound |
| 15 | `stephen-gemini-robotics-er-2-spatial` | robotics / spatial | none | unbound |

"Unbound" is a statement about **this client's configuration surface**, not about the lane. Each
lane is callable from Switchyard directly; §5 records which ones were actually exercised.

## 5. What was measured, and what was not

| Capability | Call | Result |
|---|---|---|
| `stephen-gemini-tts-3.8-flash` | speech, Kore, wav | 200 — 159,912 B, 24,000 Hz, mono, 3.33 s |
| `stephen-gemini-embedding` | embed | 200 — 3072 dims, L2 norm 1.000 |
| `stephen-gemini-embedding-2` | embed | 200 — 3072 dims, L2 norm 1.000 |
| `stephen-gemini-transcribe` | transcribe (typed JSON) | 200 in 3.25 s; returned an ASR paraphrase of the spoken probe |
| six `live-*` lanes, music, robotics | — | **not measured** — no client surface; not exercised |

The transcription body is typed JSON (`{"model","audio":{"mime_type","data"}}`), not multipart;
multipart returns HTTP 400.

**`ST_Embed` refusal.** The memory binding is local 4096D and therefore privacy scope
`third_party_free`; a call from it is refused unless the request declares
`x-switchyard-data-class` of `public` or `synthetic`. The declaration is absent, so
`memory.search` refuses. This is the egress gate working. The privacy decision is deferred:
nothing was relabelled, migrated, relaxed, redirected, and the 4096 index is untouched.

## 6. The eight chat lanes — partial qualification only

One call each through Switchyard `/v1/chat/completions`, prompt `Reply with exactly: integration ok`,
`max_tokens` 700:

| Route | Result |
|---|---|
| `direct-google-stephen-gemini-3-flash-preview` | 200, content `integration ok` |
| `direct-google-stephen-gemini-3.1-flash-lite` | 200, content `integration ok` |
| `direct-google-stephen-gemini-3.5-flash-lite` | 200, content `integration ok` |
| `direct-google-stephen-gemma-4-26b-a4b-it` | 200, content `integration ok` |
| `direct-google-stephen-gemma-4-31b-it` | 200, content `integration ok` |
| `direct-google-stephen-gemini-3.6-flash` | **503 upstream UNAVAILABLE (high demand)** — availability-inconclusive |
| `direct-google-stephen-gemini-3.8-flash` | **503 upstream UNAVAILABLE (high demand)** — availability-inconclusive |
| `direct-google-stephen-gemini-4-hydrogen` | **503 upstream UNAVAILABLE (high demand)** — availability-inconclusive |

The three 503s are **availability-inconclusive, not failures and not qualifications**. They were
returned by Google in 1.0–2.7 s and reported by Switchyard as `upstream_error`; nothing about the
lane's correctness, quota or admission was established. They are not retried here — repeated
probing against a demand-saturated upstream wastes quota and proves nothing.

Five of eight answered with real content. That is **partial qualification**: eight are configured
and allowlisted, five are proven to serve, three remain unproven.

## 7. Measurement limitations that must travel with these numbers

- The gateway's own chat surface rejects model references (`Invalid model. Use openclaw or
  openclaw/<agentId>`), so every model proof above was taken at the provider (Switchyard), not
  through the agent. Agent-level routing of these refs is configured but not separately proven.
- A first pass at `max_tokens: 32` produced "requires a larger max_output_tokens" and
  `empty_answer_zero_completion`. That is the request-fit guard, not a broken lane; 700 tokens
  passes. The wrong reading is recorded here so it is not repeated.
- `openclaw config validate` must run as the service account. The MCP admin adapter fails with
  `spawn /usr/bin/node EACCES`; that is an adapter defect, pre-existing and unrelated to the
  config, which validates cleanly when run as `stephen`.
- No claim is made that "all multimedia works": the only multimedia proven on this agent is
  speech synthesis and the preserved ComfyNinja image/video/music bindings, neither of which was
  re-exercised beyond the speech call and the config read.

## 8. Account and boundary preservation

- Every added chat lane is a **Stephen-keyed Google** target: credential
  `SWITCHYARD_GEMINI_STEPHEN_API_KEY`, quota group `google-ai-studio-free-stephen`, free tier. No
  owner credential was bound to this agent and no paid tier was introduced.
- The owner's OpenRouter catalogue and `smart-route` pool were not touched. No OpenRouter model was
  added to this agent, and no fallback list was written.
- TTS stays on the already-working `stephen-gemini-tts-3.8-flash` (`Kore`, `wav`). The other three
  Stephen TTS lanes exist in Switchyard and were left untouched.
- Image, video and music remain on the owner's `comfyninja/*` hardware providers. Nothing was
  repointed at Google, and the Lyria lane was **not** wired in place of ComfyNinja music.
- `ST_Embed` keeps its privacy restriction. No relabelling, index migration, gate relaxation or
  redirection was performed; the refusal in §5 is the gate behaving correctly.
- Switchyard configuration, the registry, the Stephen credential and the OpenClaw secrets store were
  not modified by this reconciliation. The only OpenClaw mutations were the two additive patches in
  §1.
