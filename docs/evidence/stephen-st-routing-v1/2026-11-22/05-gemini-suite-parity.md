# Gemini Free Suite: one catalogue, two account profiles

**Requirement.** Global `smart_gemini` (owner account) and Stephen `ST_Gemini_Free` (Stephen
account) are two account-scoped profiles of **one** canonical Gemini free suite. Shared model
metadata, adapter quirks, capabilities and corrections apply to both and are regression-tested
in both; only account facts differ, and those differences are declared.

**Built from the deployed R6 implementation.** Nothing is re-derived: the catalogue records
the facts `53fd9a84` already serves, and `project.py --check` proves the owner projection
reproduces them field for field. The manifest is unchanged by this work.

## Mechanism

| file | holds | why there |
|---|---|---|
| `catalogue.json` | native id, window, OpenAI-compatibility quirks (`omit_body_fields`, `unsupported_reasoning_effort`), `<thought>` marker, thinking floor, admission pad, published free-tier rows, provider retirements | facts about a **model**, identical for any account |
| `profile-owner.json` | `google_gemini` / `google_gemma_31b`, owner credential, quota group, per-model serving receipts, `selection = "measured"` pool | facts about an **account** |
| `profile-stephen.json` | `google_gemini_stephen` / `google_gemma_31b_stephen`, Stephen credential, separate quota group, per-model qualification, `selection = "peer"` beside his OpenRouter leaves | facts about **another account** |
| `project.py` | renders both profiles deterministically; `--check` compares the owner projection against a deployed manifest | the projection |
| `verify_parity.py` | P1–P8, bidirectional, named failures | the proof |
| `mutants_parity.py` | six mutations, each must be reported by the check it breaks | the proof that the proof bites |

A model fact in a profile file is the beginning of a second catalogue, so **P8** refuses one.

## What one shared fact looks like on both sides

`gemma-4-31b-it`, rendered for each account (the only differences are `llm_client` and the
target name):

```toml
# owner                                   # stephen
omit_body_fields = ["store", "metadata", "prompt_cache_key"]
unsupported_reasoning_effort = ["none"]
id = "gemma-4-31b-it"
vision = false
video = false
context_window = 262144
free_tier = true
accepts_reasoning_control = false
hosted_reasoning = "thought_tag"
min_output_tokens = 512
admission_margin_permille = 1300
daily_request_limit = 14400
requests_per_minute = 30
input_tokens_per_minute = 16000
max_input_tokens = 12307
llm_client = "google_gemma_31b"           # llm_client = "google_gemma_31b_stephen"
```

Fifteen facts, one home. A correction to the `<thought>` marker, the thinking floor or the
admission pad is made once and appears in both projections; the client name is the account's.

## Verification result

```
PASS P1-deployed-targets-are-catalogued
PASS P2-shared-facts-identical
PASS P3-accounts-are-isolated
PASS P4-qualification-gates-the-pool
PASS P5-route-policies-preserved
PASS P6-owner-projection-matches-deployed
PASS P7-catalogue-names-its-bytes
PASS P8-no-model-facts-in-profiles
parity: 8 checks, 0 failure(s)
```

`project.py --check` against the deployed `53fd9a84`: **0 failures**. That direction matters
because it is the one that loses capability: every deployed field must be known to the
catalogue, not merely every catalogue field present.

Mutation controls on the verifier — all six caught by the named check, tree green after
restore:

| mutation | reported by |
|---|---|
| a catalogue model fact quietly changes (`hosted_reasoning`) | P6 |
| a served model is dropped from the catalogue | P1 |
| a profile starts holding model facts | P8 |
| both accounts are given the owner's credential | P3 |
| the owner's measured policy is replaced by peer selection | P5 |
| the pool ships before qualification | P4 |

## Deliberate differences, declared

| | owner `smart_gemini` | Stephen `ST_Gemini_Free` |
|---|---|---|
| credential | `SWITCHYARD_GEMINI_API_KEY` | `SWITCHYARD_GEMINI_STEPHEN_API_KEY` |
| quota group | `google-ai-studio-free` | `google-ai-studio-free-stephen` |
| ordering | `measured` over one credential at one price | `peer` across providers, joined into `ST_Free` |
| membership | seven deployed leaves; the two 404 models held | the same seven, `unverified` until a receipt exists |
| state | live since R6 | staged; adds to `ST_Free` when qualified |

Neither policy is copied onto the other: the owner's question is *which model of mine answers
best*, Stephen's is *which provider's free leaf can answer at all*.

## The one open gate

Stephen's Gemini members are `unverified`. The credential is a root-only systemd file
(`stephen.env`, mode 600), so a serving receipt needs one root-run measurement. Until then
`render_stephen_staged` emits **no route block** — an empty pool would advertise a capability
that refuses every request — and lists the seven members as staged. Qualifying one member is
a single flag: `project.py` then emits the route with that member as the target and the
remaining qualified members as candidates, and P4 asserts that no unqualified member joins
with it.

This is why the parity work changes **no manifest bytes**: a profile that cannot serve must
not be deployed as if it could.

## Deliberately not done

* No duplicate adapters, endpoints or model discovery: R6 is reused and catalogued.
* No privacy gate (owner deferral, charter §0D, until after the first live deployment).
* No embeddings/TTS/transcribe/Live folding-in: those are separate suites with their own
  contracts; the catalogue records that boundary instead of crossing it.
* No deployment: Agent Stack #35 remains the sole production writer.

## Artifact digests

| artifact | sha256 (first 16) |
|---|---|
| `catalogue.json` | `b96626db75f9f75b` |
| `profile-owner.json` | `2285e94f3c482580` |
| `profile-stephen.json` | `e0fd582bafa3349b` |
| `project.py` | `b47748e34f51e529` |
| `verify_parity.py` | `a180c68de5ce71a3` |
| `mutants_parity.py` | `a8ca72b80a2c981e` |

Rendered fragments are generated, not committed (`rendered/` is ignored): a committed
rendering is a second copy that can be stale, and the verifier re-derives them from the
committed inputs on every run.
