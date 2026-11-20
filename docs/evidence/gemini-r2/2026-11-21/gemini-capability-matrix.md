# Gemini free-project inventory and verified/held matrix

Generated from crosswalk `fd8acaf767c5265c1991f10047e035ac4258734b8609908ac58ce6a612ae7abd`
(owner-confirmed 2026-11-21, project Default Gemini Project, tier Free).

- models: 47
- chat-capable: 17
- routable today: 2
- held: 45
- small-bounded-only (per-minute input budget): 2
- feature quotas held as tool quota: 23

Held is a statement about our transport and qualification, never about the
owner's allocation.

| # | model | operation | native id | RPD | state | routable |
|---|---|---|---|---|---|---|
| 1 | Gemini 3.6 Flash | chat.complete | gemini-3.6-flash | 20 | HELD_WIRE_UNQUALIFIED | no |
| 2 | Gemini 3.8 Flash | chat.complete | gemini-3.8-flash | 20 | HELD_WIRE_UNQUALIFIED | no |
| 3 | Gemini 3.5 Flash Lite | chat.complete | gemini-3.5-flash-lite | 500 | EXECUTABLE_AND_DEPLOYED | yes |
| 4 | Gemini 3.1 Flash Lite | chat.complete | gemini-3.1-flash-lite | 500 | EXECUTABLE_AND_DEPLOYED | yes |
| 5 | Antigravity | agent.run | -- unresolved -- | 100 | HELD_NON_CHAT_TRANSPORT | no |
| 6 | Deep Research Pro Preview | agent.run | deep-research-pro-preview-12-2025 | 0 | HELD_NON_CHAT_TRANSPORT | no |
| 7 | Gemini 2 Flash | chat.complete | -- unresolved -- | 0 | HELD_NATIVE_ID_UNRESOLVED | no |
| 8 | Gemini 2 Flash Lite | chat.complete | -- unresolved -- | 0 | HELD_NATIVE_ID_UNRESOLVED | no |
| 9 | Computer Use Preview | unknown | gemini-2.5-computer-use-preview-10-2025 | 0 | HELD_NON_CHAT_TRANSPORT | no |
| 10 | Gemini 2.5 Flash | chat.complete | gemini-2.5-flash | 20 | HELD_WIRE_UNQUALIFIED | no |
| 11 | Nano Banana (Gemini 2.5 Flash Preview Image) | image.generate | gemini-2.5-flash-image | 0 | HELD_NON_CHAT_TRANSPORT | no |
| 12 | Gemini 2.5 Flash Lite | chat.complete | gemini-2.5-flash-lite | 20 | HELD_WIRE_UNQUALIFIED | no |
| 13 | Gemini 2.5 Flash TTS | audio.synthesize | gemini-2.5-flash-preview-tts | 10 | HELD_NON_CHAT_TRANSPORT | no |
| 14 | Gemini 2.5 Pro | chat.complete | gemini-2.5-pro | 0 | HELD_ZERO_RPD_ALLOWANCE_UNSHOWN | no |
| 15 | Gemini 2.5 Pro TTS | audio.synthesize | gemini-2.5-pro-preview-tts | 0 | HELD_NON_CHAT_TRANSPORT | no |
| 16 | Gemini 3 Flash | chat.complete | -- unresolved -- | 20 | HELD_NATIVE_ID_UNRESOLVED | no |
| 17 | Nano Banana Pro (Gemini 3 Pro Image) | image.generate | -- unresolved -- | 0 | HELD_NON_CHAT_TRANSPORT | no |
| 18 | Gemini 3.1 Pro | chat.complete | -- unresolved -- | 0 | HELD_NATIVE_ID_UNRESOLVED | no |
| 19 | Nano Banana 2 (Gemini 3.1 Flash Image) | image.generate | -- unresolved -- | 0 | HELD_NON_CHAT_TRANSPORT | no |
| 20 | Nano Banana 2 Lite (Gemini 3.1 Flash Lite Image) | image.generate | gemini-3.1-flash-lite-image | 0 | HELD_NON_CHAT_TRANSPORT | no |
| 21 | Gemini 3.1 Flash TTS | audio.synthesize | gemini-3.1-flash-tts-preview | 10 | HELD_NON_CHAT_TRANSPORT | no |
| 22 | Gemini 3.5 Transcribe | audio.transcribe | gemini-3.5-transcribe | 25 | HELD_NON_CHAT_TRANSPORT | no |
| 23 | Gemini 3.8 Flash Lite TTS | audio.synthesize | gemini-3.8-flash-lite-tts | 10 | HELD_NON_CHAT_TRANSPORT | no |
| 24 | Gemini 3.8 Flash TTS | audio.synthesize | gemini-3.8-flash-tts | 10 | HELD_NON_CHAT_TRANSPORT | no |
| 25 | Gemini 4 Argon | chat.complete | gemini-4-argon | 0 | HELD_ZERO_RPD_ALLOWANCE_UNSHOWN | no |
| 26 | Gemini 4 Carbon | chat.complete | gemini-4-carbon | 0 | HELD_ZERO_RPD_ALLOWANCE_UNSHOWN | no |
| 27 | Gemini 4 Hydrogen | chat.complete | gemini-4-hydrogen | 20 | HELD_WIRE_UNQUALIFIED | no |
| 28 | Gemini 4 Neon | chat.complete | gemini-4-neon | 0 | HELD_ZERO_RPD_ALLOWANCE_UNSHOWN | no |
| 29 | Gemini Decision 1 | decision.score | gemini-decision-1 | 10000 | HELD_NON_CHAT_TRANSPORT | no |
| 30 | Gemini Embedding 1 | vector.embed | gemini-embedding-001 | 1000 | HELD_NON_CHAT_TRANSPORT | no |
| 31 | Gemini Embedding 2 | vector.embed | -- unresolved -- | 1000 | HELD_NON_CHAT_TRANSPORT | no |
| 32 | Gemini Nano Banana 2.1 | image.generate | gemini-nano-banana-2.1 | 0 | HELD_NON_CHAT_TRANSPORT | no |
| 33 | Gemini Omni 1.1 Flash | multimodal.generate | gemini-omni-1.1-flash | 0 | HELD_NON_CHAT_TRANSPORT | no |
| 34 | Gemini Omni 1.2 Flash | multimodal.generate | gemini-omni-1.2-flash | 0 | HELD_NON_CHAT_TRANSPORT | no |
| 35 | Gemini Robotics ER 2 Preview | robotics.perceive | gemini-robotics-er-2-preview | 20 | HELD_NON_CHAT_TRANSPORT | no |
| 36 | Gemma 4 26B | chat.complete | gemma-4-26b-a4b-it | 14400 | HELD_WIRE_UNQUALIFIED | no |
| 37 | Gemma 4 31B | chat.complete | gemma-4-31b-it | 14400 | HELD_WIRE_UNQUALIFIED | no |
| 38 | Genie 4 | world.generate | genie-4 | 0 | HELD_NON_CHAT_TRANSPORT | no |
| 39 | Lyria 3 Clip | music.stream | lyria-3-clip-preview | 0 | HELD_NON_CHAT_TRANSPORT | no |
| 40 | Lyria 3 Pro | music.stream | lyria-3-pro-preview | 0 | HELD_NON_CHAT_TRANSPORT | no |
| 41 | Lyria Realtime | music.stream | lyria-realtime-exp | 1500 | HELD_NON_CHAT_TRANSPORT | no |
| 42 | Gemini 2.5 Flash Native Audio Dialog | session.live | -- unresolved -- | Unlimited | HELD_NON_CHAT_TRANSPORT | no |
| 43 | Gemini 3 Flash Live | session.live | -- unresolved -- | Unlimited | HELD_NON_CHAT_TRANSPORT | no |
| 44 | Gemini 3.5 Live Translate | session.live | gemini-3.5-live-translate-preview | Unlimited | HELD_NON_CHAT_TRANSPORT | no |
| 45 | Gemini 3.5 Transcribe Live | session.live | gemini-3.5-transcribe-live | Unlimited | HELD_NON_CHAT_TRANSPORT | no |
| 46 | Gemini 3.8 Live | session.live | gemini-3.8-live | Unlimited | HELD_NON_CHAT_TRANSPORT | no |
| 47 | Gemini 3.8 Live Extended Thinking | session.live | gemini-3.8-live-extended-thinking | Unlimited | HELD_NON_CHAT_TRANSPORT | no |
