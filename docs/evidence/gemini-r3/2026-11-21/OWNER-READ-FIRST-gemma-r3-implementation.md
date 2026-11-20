
> **SUPERSEDED FOR OVERALL SCOPE AND REPO ACCESS — 2026-11-21.** Read the [CONTROLLING ALL FREE-TIER CAPABILITIES dossier](https://github.com/RedEyeNinja-BKK/Switchyard/blob/feat/gemini-r3-adaptive-bounded/docs/evidence/gemini-r3/2026-11-21/OWNER-CONTROLLING-ALL-FREE-CAPABILITIES.md) at `2633cbbc797a4076455ab3444ebcdbddecb7b216` FIRST. Owner confirms Turnstone's PAT **can access all repositories**, including private `agent-stack-private-development`; the earlier claim that it cannot was wrong and is retracted. Resolve any 404 as a path/ref/API/credential-context issue, not an established permission boundary. This file is a Gemma-specific *subsection* of the 26-actionable-model portfolio, **not the global work order**. Private Agent Stack Issue #35 remains the production governance thread.

# OWNER READ FIRST — Gemma 4 measured contract and shortest R3 production path
2026-11-21 · source facts and implementation decisions · **not a provider-call or deployment authorization**

**Repository correction:** Owner confirms Turnstone's PAT has access to ALL owner repositories. Use private Agent Stack Issue #35 plus this Switchyard branch. A 404 on a document or comment requires checking repo/path/ref/API/actual auth context; it does not prove a permission limitation. This Gemma-specific note is subordinate to the linked all-Free-tier dossier.

## Measured reality: don't research or probe this again

Exact project-host API surface verified by Turnstone: `https://generativelanguage.googleapis.com/v1beta/openai`, using the already-staged project credential securely on host. Receipts are committed adjacent to this file:
- `docs/evidence/gemini-r3/2026-11-21/gemma-capability-verification.md`
- `docs/evidence/gemini-r3/2026-11-21/live-probe/probe-capabilities.jsonl`
- `docs/evidence/gemini-r3/2026-11-21/live-probe/probe-tools.jsonl`
- `docs/evidence/gemini-r3/2026-11-21/live-probe/probe-31b-tools.jsonl`
- `docs/evidence/gemini-r3/2026-11-21/live-probe/probe-roundtrip.jsonl`

Turnstone reports **25 chat calls + 2 model listings, all tiny; no production deploy**. These are *already completed* live tests, not a request to run 25 more. Do not re-run absent owner GO. The earlier owner approval gate had not explicitly authorized this batch, so **no more discretionary live calls**.

| Claim | 26B A4B | 31B | Shipping consequence |
| --- | --- | --- | --- |
| Exact hosted ID | `gemma-4-26b-a4b-it` | `gemma-4-31b-it` | truthful native direct leaves |
| Listed, text served | 200, successful samples ~2–5 sec | 200 but often 503, successful samples ~18–55 sec | **26B first**; 31B secondary, short health cooldown |
| Function tools auto | 200, correctly typed `tool_calls` and `{location:Bangkok}` | 200 after 503 attempts | tool-call *wire* qualified on both |
| Tool-result continuation with assistant call reduced to id/type/function | 200, correct answer | **not tested** | 26B tool round-trip qualified; 31B not fully qualified for multi-step agent tool cycle |
| `response_format=json_object` | content includes `<thought>…</thought>{…}` | same | output is **not parseable JSON**; do not expose as a guaranteed valid JSON mode |
| `response_format=json_schema` | **not tested** | clean 200 JSON | schema validated on **31B only**, no claim for 26B yet |
| `reasoning_effort:none` | not tested | 400: thinking budget unsupported | **do not send this effort flag**; `reasoning=false` route policy does NOT imply hosted model won't emit thinking |
| Visible hidden-thought block | present in ordinary `content` | present in ordinary `content` | **deployment blocker: prevent reasoning text leakage** |

The concrete reply looked like `"<thought>...reasoning...</thought>ready"`, plus `extra_content.google.thought=true`. Usage `prompt_tokens=8`, `completion_tokens=1`, `total_tokens=71`; difference **62** is *unclassified token accounting gap* compatible with thinking overhead, not an independently defined `reasoning_tokens` field. **Never infer a provider-billable token split or expose the thought text from that arithmetic.**

**Vendor reference independently verified:** [Google's official hosted Gemma 4 guide](https://ai.google.dev/gemma/docs/core/gemma_on_gemini_api) names both exact IDs, native `:generateContent` REST, configurable native `thinkingLevel=minimal/high`, system prompt and function calls. [Google Gemma 4 family overview](https://ai.google.dev/gemma/docs/core) documents model architecture and tool capabilities. This corrects the blanket claim that Gemma is undocumented: **the exact OpenAI-compat translation/inline reasoning/JSON/503 behavior is the part these official pages do not establish**, hence the measured receipts control that behavior. Native thinking-level syntax must NOT be sent as OpenAI `reasoning_effort:none` without endpoint proof.

**Owner's Free-project quota baseline:** each Gemma 4 variant 30 RPM, 16,000 **input TPM**, 14,400 RPD. These are three independent limits. A 16,000 single-request size ceiling is not a minute meter; the 262,144 architecture context is not an effective 262,144 request allowance. For the bounded Gemma-only route advertise a truthful effective input cap <=16,000 including output reserve/headroom, while the physical target retains architecture `262144` metadata and the wider `smart-nt` parent may retain verified aggregate 262144 capacity.

## Finish PR #46: implementation tasks, no extra investigations

**P0. Fix exposed reasoning output FIRST.** In the OpenAI-compat extraction/translation layer, recognize an exact leading complete `<thought>…</thought>` section for these hosted Gemma responses. Keep the actual answer clean; **never include the block in user-facing content, tool-visible text, logs, structured JSON, or streamed answer chunks**. If the block is malformed/incomplete or a streaming parser cannot safely segregate it, fail closed for that response rather than leak it; don't blindly delete arbitrary user text containing these strings. Preserve any genuine tool calls and finish reason. Internal reasoning metadata, if stored at all, must not be accessible to consumers. Add 26B/31B receipt-derived fixtures for text, tool, JSON-object-with-thought, JSON-schema clean, empty/unterminated thought, SSE/streaming boundary split, and multi-turn tool continuation. No call needed to implement.

**P1. Update truthful Gemma candidate definitions.** Use 26B first, 31B secondary; both have *tool calling* demonstrated, but **31B's completed tool-result continuation is unverified**. Enable tool routing only for proven continuation paths; either keep the bounded composite text-first until the 31B continuation is explicitly qualified or make tool request eligibility prefer/require 26B with no unsafe 31B handoff. A `reasoning=false` route claim means no reasoning-feature contract; it is not a model thinking-off switch. Unsupported `reasoning_effort=none` must never reach this endpoint. `json_object` does not meet strict valid-JSON contract; where exact `json_schema` works use that, with local parse/validation and refusal/repair handling, **not a silent fake success**. Propagate 503 `UNAVAILABLE / high demand` as a measured, retry-bounded signal to next *eligible free* child; no blind repeated 31B attempts or unapproved paid cascade. Record 503 separately from user correctness/latency.

**P2. Complete ONLY the three already reviewed engineering fixes:** (1) paid-route safety and true physical fallback/graded-tool success in `bounded_compare.py`; default no paid route, real `--allow-paid` plus hard budget/attempt caps for any comparison; (2) small durable per-project/per-model RPM and input-TPM reservation or strict conservative pacing, distinct from already deployed daily RPD; do not label a per-request ceiling as a TPM ledger; (3) minimal free-only eligibility-aware exploration/warmup of 26B/31B, with correct one-observation-per-upstream-attempt and no mutation of privacy/cost tier. No new ML router or complete redesign.

**P3. Verify source and stage one opt-in candidate.** Re-run targeted tests/mutants and candidate-vs-live dry-run compatibility, avoid unrelated formatter churn. The currently healthy live R1 `CURRENT=20261120T181049Z-a5b9dffc7f198716-f0546288` remains unchanged until owner GO. Produce one candidate binary/manifest hash, a single reversible release procedure through existing `activate-release.sh`, and concise effective route graph. Keep `smart-nt`, Turnstone primary, and existing strict-NT users untouched.

**P4. Ask for one owner decision at a time.** Existing Lite 5-step synthetic test was gated/unrun; the reported Gemma probe results do **not** substitute for testing Lite or `smart_gemini`. Propose the existing 5-step test exactly (two zero-spend controls and at most three tiny Google requests), with no paid fallback, **then stop pending GO**. No new Gemma wire probes, private account calls, deploy, restart, merge, default consumer route repoint, remote OpenClaw/Stephen ST_Route change, Antigravity, or paid comparison authorized by this note.

## Research delegation and forward scope

The owner's complete authenticated Free-project 47-model/23-feature inventory and 27 quota-present *direct capability contracts* are already enumerated in [Switchyard PR #45 source inventory](https://github.com/RedEyeNinja-BKK/Switchyard/pull/45), including its `docs/evidence/gemini-r2/2026-11-21/research-and-ship-queue-20261121.md`. Do not ask Turnstone to repeat external research or claim every quota-present model is callable. The owner/assistant researches missing model-specific documentation and passes exact URLs/methods; Turnstone implements, qualifies and deploys only scoped working additions. Future Stephen `ST_Gemini` + `ST_Free` peer-selector under `ST_Route` remains parked in private issue #58 and is **not part of PR #46**.

**Exit:** patched code with anti-leak fixtures, honest tools/schema/503 routing, small bounded quota and comparison guards, one staged reversible candidate and a clearly named next owner GO. **No more architecture or inventory ceremonies.**
