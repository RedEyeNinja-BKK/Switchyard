# Findings 2026-11-28 — vision estimator correction, Gemini 3.8 localisation, Particle agentic qualification

Read-only measurement plus source-only work. **No binary was activated and no live configuration
was changed in this pass.** Live state is unchanged at
`20261128T080005Z-6a8209f97101563b-f3b743f5`.

---

## 1. Vision estimator correction (supersedes the reviewed candidate)

| Item | Reviewed | Corrected |
|---|---|---|
| commit | `36cdcbcd05bec7ed5350fb5cfd68f29fd706fd7f` | `ab101c0a7567b743876a950579f629d622a7e468` |
| binary | `692539ad…cc082c7` (31,276,296 B) | `60c3556fcc8211769889469501c4c85480376e8afec226ad0ef0431ec8f50a42` (31,277,368 B) |
| capabilities.json | — | `51734a5b2b821249ed55681a31ff275a9ee02708da5732905a1fe1ae9829a149` |
| `request_fit.rs` | — | `7f7c8be6b908de56d0c5d24f57c6169872a38dbce64ff4b8c0650277e104a300` |
| `tests/route.rs` | `d3f3913b…` | `d3f3913b…` (unchanged) |

Two defects were fixed, both from the review at Issue #35 comment `7243104787`:

1. **Bare-string arm removed.** A bare JSON string no longer charges `IMAGE_TOKENS`.
2. **Media accounting is position-aware.** `walk` now carries `Slot::{Root, Message, Part}`; only a
   `Part` reached through a real protocol content position can be media. This mirrors the two-walk
   discipline `preserved_media` already used in the same file. `IMAGE_PART_TYPES` is now the single
   vocabulary consumed by both `is_image_part` and `unsupported_preserved_content`, which removes the
   duplicated `matches!` literal the review also flagged.

One existing assertion was **relaxed, not restored**: `resolution_and_detail_do_not_change_the_charge`
now allows a bounded delta and counts `detail: "low"` as its own 2 tokens, because payload skipping is
no longer unconditional. Counting is the safer direction.

Gates on the corrected candidate:

- `release_validate.py` exit 0; `candidate validates: server OK`.
- Strict P9 **PASS** direct=55 qualified=55 failures=0.
- `request_fit` 49 tests (was 41); `switchyard-runner --lib` 385; every runner suite green.
- Mutation 4/4 killed, restore control green.
- Reproducible build from a fresh clone with an empty target directory → same sha256, same byte count.

Measured effect on the same 800,244-byte body (tool-schema properties named `source` and `image_url`,
each with a 400,000-character description, stub upstream): reviewed binary estimated **4,126** input
tokens, corrected binary estimates **200,018** — a **195,892-token under-count removed**.

Staged at `/var/tmp/vision-release-20261128/` (binary, `routes.toml` `f3b743f5`, `capabilities.json`
`51734a5b`). The manifest is byte-identical to live, so this is a **binary-only** change and the live
Stephen TTS lanes are preserved by construction.

Rollback target is unchanged: `f3b743f5` remains CURRENT, RETAINED, PROTECTED, and
LAST-KNOWN-GOOD `20261126T061331Z-65757ec7aaa51166-737a6887` is untouched.

**Awaiting an owner GO that names `60c3556f…`.**

---

## 2. Gemini 3.8 Flash — localisation and timeout assessment

Earlier this session I reported three `499` results and inferred an upstream stall. That inference is
withdrawn and replaced by the measurement below.

### Population (all attempts on `gemini-3.8-flash`, 2026-11-21T10:14Z → 2026-11-28T17:54Z, n = 6,415)

| outcome | n | share |
|---|---|---|
| `daily_allowance_exhausted` | 6,245 | 97.4% |
| `minute_window` | 76 | 1.2% |
| `http_5xx` | 39 | 0.6% |
| **`served`** | **24** | **0.4%** |
| `unavailable` | 7 | 0.1% |
| `http_400` | 1 | — |
| `http_429` | 1 | — |

### When it is admitted, it answers — slowly

Served latency: n=24, **min 15,132 ms, p50 189,266 ms, p90 482,943 ms, max 564,448 ms.**
Latency is not input-size-driven: the fastest call carried 143,066 input tokens (15 s) and the slowest
carried 80,531 (564 s). It degrades with age — the 2026-11-27 18:53–19:10Z cluster completed in
56–99 s, while the 2026-11-28 08:02–08:43Z cluster needed 15–544 s.

### Why the refusals dominate

Daily allowance counters for day 20785:

| group | model | spent / limit |
|---|---|---|
| `google-ai-studio-free` (owner key) | gemini-3.8-flash | **20 / 20 exhausted** |
| `google-ai-studio-free` (owner key) | gemini-3.6-flash | **20 / 20 exhausted** |
| `google-ai-studio-free-stephen` (Stephen key) | gemini-3.8-flash | **3 / 20 — 17 remaining** |

So the 97.4% refusal rate is correct quota accounting on an exhausted key, not a routing fault.
Stephen's twin of the same model has 17 requests left and is bound to **no composite route** (see §4).

### The three `499` probes

`90,025 / 126,264 / 155,286 ms`, all `client disconnected before a response was written`, all
`selected_model=""`. There are **zero** decision rows for `gemini-3.8` on 2026-11-28 with an empty
attempt list, so those three requests produced **no verdict at all** — the caller deadline expired
before the server resolved an attempt. The durations match my probe timeouts, and the server's own
measured p50 for this model is 189 s. The cancellation is caller-side; the log cannot carry any
stronger claim than that.

The 39 `http_5xx` are separate: p50 3,609 ms, max 330,734 ms, spread across the whole window — genuine
slow upstream failures, not related to the probes.

### Timeout assessment

- **No Google client declares `attempt_timeout_ms`.** Only `llm_clients.openrouter_stephen` declares
  one, at 120,000 ms.
- A 120 s bound on this lane would fail the majority of calls (p50 189 s, p90 483 s, max 564 s).
- Any useful bound must be **≥600 s** to cover p90 plus the observed maximum, and even then the lane
  is not interactive.
- Declaring `vision = false` on the 3.8 targets is correct and was verified: an inline image returns
  `400 unsupported_capability`.

**No timeout change is proposed for deployment on this evidence.** A bound that cannot be met by the
model is a self-inflicted outage; the actionable finding is that `gemini-3.8-flash` is a
minutes-scale lane and should not lead any interactive composite.

---

## 3. Particle six-fixture agentic qualification

Route `direct-openrouter-stephen-particle-1-free` (upstream `particle-ai/particle-1:free`), free
Stephen credential, no spend. Harness `particle_six.py` (sha256 `17a1aa37…`).

| fixture | what it requires | result |
|---|---|---|
| F1 plain_answer | `17*23` answered `391`, no tools | PASS, 571–835 ms |
| F2 single_tool_call | correct tool name + valid JSON args naming Lisbon | PASS |
| F3 two_step_loop | tool call → tool result → grounded answer citing 22 °C | PASS |
| F4 replay_continuity | uses a tool result already in history, does not re-call | PASS, answered `7400`, 0 re-calls |
| F5 structured_output | exact JSON object, keys `id`/`tags`/`ok`, exact types, no prose | PASS |
| F6 determinism | repeated identical tool call | **strict FAIL** in the first run |

**F6 detail, not glossed.** The first run produced
`{"city":"Lisbon","unit":"celsius"}` then `{"city":"Lisbon"}` — same tool, same required argument,
one omitted the optional `unit`. A dedicated characterisation over 6 repeats then produced
6/6 byte-identical signatures (`unit: celsius` every time). Aggregate over 8 observed calls:
**tool name and required arguments stable 8/8; exact argument JSON identical 7/8.**

Verdict: **qualified for single-tool agentic use**, with the stated caveat that presence of an
*optional* argument varies between runs. The fixture is not relaxed — the strict grade stands at 5/6
for the recorded run.

Free-tier throttling was hit mid-run (`429 provider_throttled`, scope `unknown`, then local
`unavailable` benches at 0 ms). The harness paces and retries; fixtures were not scored on a throttled
response.

Vision is declared `false` and the gate refuses inline images correctly — that proves the **gate**, not
model incapability, and is recorded as such.

---

## 4. Stephen capability census — what is bound, what is not

18 Stephen-keyed targets are live. Four Stephen-keyed Google **TTS** lanes were added by the
`20261128T080005Z` release and are working; they are preserved and are not touched here.

**Provisioned but bound to no composite route** (direct alias only):

- `stephen-google-gemini-3.8-flash`
- `stephen-gemma-4-26b-a4b-it-free` (OpenRouter)
- `stephen-gemma-4-31b-it-free` (OpenRouter)

Every sibling Gemini target is in `ST_Free` and `ST_Gemini_Free`; these three are not. Given §2, adding
gemini-3.8-flash to an interactive composite needs a last-position placement and an explicit
acceptance of minutes-scale latency — it is **not** proposed as a drop-in.

**Owner-key-only capabilities with no Stephen twin** (provisioned on
`SWITCHYARD_GEMINI_API_KEY`, not on Stephen's key):

`live-3-5-translate`, `live-3-5-transcribe`, `live-3-8-flash`,
`live-3-8-flash-extended-thinking`, `live-2-5-flash-native-audio`, `live-3-flash`,
`music-lyria-realtime`, `robotics-er-2`, `transcribe-3-5`, plus the owner `speech-*` TTS set,
`embedding-google`, `embedding-google2`.

Stephen twins that **do** exist: `speech-stephen-2-5-flash`, `speech-stephen-3-1-flash`,
`speech-stephen-3-8-flash-lite`, `speech-stephen-3-8-flash`, `embedding-stephen`, `reranker-stephen`.

This is the concrete remainder of "Stephen's native Google free capabilities". Each twin is additive
configuration (a new `[capability_clients.*]` entry plus a `[media_capabilities.*]` entry pointing at
it); none requires a binary change. **None was added in this pass** — this is a census for owner
decision, not a deployment.

---

## 5. Retry staging — confirmed source-only, no overlap

`origin/topic/retry-contract-20261128` at `e06ff69c`. Diff against the deployed source `f26a9030`:
49 files, +14,641/−82. It **does not touch** `crates/switchyard-runner/src/request_fit.rs` or
`crates/switchyard-runner/tests/route.rs`, so it is independent of the vision correction and either
can land without the other. No activation.

---

## 6. Reconciliation — the resume document was one release behind

`RESUME-20261128.md` names `20261128T040916Z-…-fc05b640` (label `ling-out`) as CURRENT. Live is:

| Item | Live value |
|---|---|
| CURRENT | `20261128T080005Z-6a8209f97101563b-f3b743f5` |
| Label | `stephen-google-tts-lanes-recovered-20261128` |
| Binary | `6a8209f9…` (31,281,480 B), unchanged, config-only release |
| Manifest | `f3b743f5…` (137,957 B) |
| Capabilities | `987dd039…` (25,499 B, reused) |
| Predecessor (protected) | `20261128T040916Z-…-fc05b640` |
| LAST-KNOWN-GOOD | `20261126T061331Z-65757ec7aaa51166-737a6887` (not promoted) |
| Protected ids | 18 |
| Service | active, PID 895156 |

A further release `20261128T070126Z-6a8209f97101563b-8325fdca` also exists in the store and is
protected; the skip from `fc05b640` to `8325fdca` to `f3b743f5` is not described in the resume file.

Dashboards: V7 `turnstone-switchyard-v7` title **V7.101**, revision **106**;
V9 `turnstone-switchyard-economics-v9` title **V9.19**, version **29**. The resume's "V9 version 26"
is stale; V9 has advanced three revisions since. No dashboard write was made in this pass.

---

## Correction appended the same day (before review) — §4 membership was wrong

The census in §4 used only `route.target` and `route.candidates` to decide binding. That misses
`targets.*.groups`, which is what fills `routes.*.pool_groups`. Re-derived from the deployed manifest
`f3b743f5`:

| Pool group | Members |
|---|---|
| `ST_Gemini_Free` | 7: `stephen-google-gemini-4-hydrogen`, `-3.6-flash`, `-3.5-flash-lite`, `-3.1-flash-lite`, `-3-flash-preview`, `stephen-google-gemma-4-26b-a4b-it`, `stephen-google-gemma-4-31b-it` |
| `ST_OR_Free` | 5: `stephen-nemotron-3-5-super-vl-free`, `stephen-flint-v1-0-free`, `stephen-ernie-5_5-flash-preview-free`, `stephen-dots-3-note-preview-free`, `stephen-particle-1-free` |
| `ST_OR_Paid` | 3: `stephen-tencent-hy4-flash`, `stephen-paid-glm-5-3-flash`, `stephen-paid-deepseek-v4-1-flash-1017` |

So the two OpenRouter Gemma twins **are** bound, through `ST_Gemini_Free`, and they also appear in
`ST_Free`'s candidate list. The §4 statement that they were "provisioned but bound to no composite
route" is **withdrawn**.

**Exactly one Stephen target remains bound to no pool and no composite:**
`stephen-google-gemini-3.8-flash` (`groups` absent). Given §2 — p50 189 s, p90 483 s, `vision = false` —
leaving it direct-only is defensible and no binding change is proposed.

### What the undocumented release actually did

`20261128T070126Z-6a8209f97101563b-8325fdca` was not the trivial release the resume file implied. A
parsed-document diff against `fc05b640` shows it was the **Stephen pool build-out**:

- created `ST_Gemini_Free` (target moved from `stephen-google-gemini-3.5-flash-lite` to
  `stephen-google-gemini-4-hydrogen`, 6 candidates) and `ST_OR_Paid`;
- gave seven Stephen Google targets `groups = ["ST_Gemini_Free"]`;
- moved the two OpenRouter Gemma twins out of `ST_OR_Free`;
- added `stephen-tencent-hy4-flash` plus its direct route and `ST_OR_Paid` membership;
- grew `ST_Free` from 6 to 10 candidates.

That is the substance of "Stephen's remaining native Google free capabilities" for the chat lanes. It is
already live.

`20261128T080005Z-8325fdca → f3b743f5` was then checked independently and **is** strictly additive: four
`capability_clients.speech-stephen-*` entries, four `capabilities.stephen-gemini-tts-*` entries, nothing
removed, no routing key changed. The release provenance claim holds.

### Still missing (unchanged conclusion, narrower list)

Stephen-keyed twins that do not exist at all — the owner-key clients are present but there is no
`*_stephen` counterpart:

`live-3-5-translate`, `live-3-5-transcribe`, `live-3-8-flash`, `live-3-8-flash-extended-thinking`,
`live-2-5-flash-native-audio`, `live-3-flash`, `music-lyria-realtime`, `robotics-er-2`, `transcribe-3-5`.

Each is one additive `[capability_clients.*]` entry plus a `[capabilities.*]` entry. No binary change.
None added in this pass.

## Dashboard staleness — recorded, not written

V7 `turnstone-switchyard-v7` is at title **V7.101**, revision **106**, and is **stale against live** in
four panels:

- **12001** — describes manifest `a7ef9174…` / 131,100 B and "53 of 53" direct aliases; live is
  `f3b743f5…` / 137,957 B with 55 direct aliases.
- **12002** — `smart-route` described as `8 leaves = 3/2/1/2`; live pools are
  `ST_Gemini_Free` 3 (of 3 legs in `smart-gemini-chat`), `smart-openrouter-free` 2,
  `smart-deepseek-flash` 1, `smart-openrouter-paid` 2, and the two Lite leaves are back in
  `ST_Gemini_Free` after the 070126Z build-out.
- **12004** — cites release `...040916Z-...fc05b640`.
- **12007** — the whole CURRENT snapshot row set names `fc05b640` / PID 880721 / 72 advertised; live is
  `f3b743f5` / PID 895156 / **74 advertised**.

**No dashboard write was made.** Reaching V7.102 requires a new generator stage in
`scripts/economics/v7/correct_dashboard.py` (title, revision 107, the `f3b743f5` token mappings, the new
pool topology, the TTS capability rows) and a guarded write at `expected_version = 106`. I did not
attempt a partial or hand-edited panel because an unverified live dashboard change is worse than a
documented stale one. V9 (`V9.19`, version 29) has advanced three revisions past the resume record and
was not touched.
