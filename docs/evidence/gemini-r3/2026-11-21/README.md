# Adaptive bounded release, R3 — 2026-11-21

Branch: `feat/gemini-r3-adaptive-bounded`, forked from
`feat/gemini-egress-admission-quota-20261120` at `52b985e7`.

Nothing here is deployed. Production remains
`20261120T181049Z-a5b9dffc7f198716-f0546288` (binary `a5b9dffc…`, manifest `f0546288…`),
`NRestarts=0`, `ActiveEnterTimestamp=Sat 2026-11-21 01:10:48 +07`.

This release has three parts, and only the first is code:

1. **The runner learns two facts and one policy.** A target may state its physical
   `context_window` and a per-request `max_input_tokens`; a route may set
   `selection = "measured"`.
2. **A Gemma 4 bounded lane** is staged as a manifest change against the deployed file.
3. **A comparison harness** measures the bounded lane against the strict non-training
   DeepSeek leg. It has been proven against a local stub; it has made no real call.

## What the code now guarantees

| Property | Where | Test |
| --- | --- | --- |
| An uncleared payload never reaches a Google leg inside a composite, and the composite serves the next leg instead of failing | `composition.rs` skip before the capability gate, capability reported as `privacy_clearance` | `an_uncleared_payload_skips_a_gated_child_and_serves_the_next_leg` |
| The deployed four-child ladder behaves this way in the order the manifest states | same | `the_deployed_bounded_trial_shape_starts_after_google_for_an_uncleared_payload` (+ control: a cleared payload does use leg one) |
| A directly addressed Google leaf still refuses, with zero provider calls | `route.rs` per-model admission filter | `a_directly_addressed_gated_leaf_still_refuses_and_is_never_called` |
| A request above a target's window is not dispatched to it | `route.rs` eligibility filter | `a_request_beyond_a_targets_window_goes_to_the_wider_target` |
| A request above a target's per-request ceiling is not dispatched to it | same | `a_request_beyond_the_input_budget_is_not_sent_to_that_target` |
| When no candidate fits, the refusal names the binding limit and makes no call | `route.rs` refusal reason | `a_request_no_candidate_can_hold_is_refused_without_a_call` |
| Selection reorders only after ≥3 observations per candidate, preserves declared order as the tie-break, and cannot invent a candidate | `selection.rs` | `a_measured_route_prefers_the_leg_that_has_been_completing`, `the_selector_only_ever_reorders_the_candidates_it_was_given` |
| A route may not advertise more context than its widest declared target | `config.rs` `validate_physical_context_claims` | `a_route_cannot_advertise_more_context_than_its_widest_target` |
| A composite may not advertise more context than its widest child | same | `a_composite_cannot_advertise_more_context_than_its_widest_child` |
| A target that states no window keeps loading unchanged | same | `an_undeclared_window_admits_every_request` |

Suite: **772 tests, 0 failures** across `switchyard-runner`, `switchyard-libsy`,
`switchyard-llm-client`. The adaptive group is **16 tests**.

Mutation testing, seven mutants, all caught:

| | Mutation | Caught by |
| --- | --- | --- |
| M1 | `payload_refused` always false | privacy skip test |
| M2 | `physical_ineligibility` always `None` | window/budget skip tests |
| M3 | selector never installed | measured-selection test |
| M4 | physical-context validation removed | over-claim test |
| M5 | output reservation ignored in the size estimate | output-reservation test |
| M6 | composite context claim unchecked | composite over-claim test |
| M7 | physical refusal reason dropped | refusal-message test |

## The manifest change, and why it is code-shaped

`scripts/deploy/switchyard/gemini_r3_stage.py` reads the **deployed** manifest, refuses to run
if any anchor is not matched exactly once, and writes a candidate. It does not carry a copy of
the manifest, so it cannot go stale against production.

Input manifest: `/var/lib/switchyard/releases/CURRENT/routes.toml`,
sha256 `f0546288179e7e40523e900d7fa6a010bc3a0e908cb5e8ad749a3147a0c71d1b`.
The fetched copy used for staging hashes identically, so the splice was made against the same
bytes that are serving traffic.

Three additive edits:

1. The two Gemini Lite targets gain `context_window = 1048576` and
   `max_input_tokens = 250000` — the window is the model's, the ceiling is the project's
   250,000 TPM allowance. The route-level `context_window = 1048576` is left alone: capacity
   and per-request admission are deliberately different numbers.
2. Two Gemma targets are added at native ids `gemma-4-31b-it` and `gemma-4-26b-a4b-it`, with
   the model's 262,144-token window, a 16,000-token per-request ceiling (free-tier TPM), and
   the model's own 14,400 RPD budget.
3. `smart-gemini-bounded` is added as a **passthrough** over both Gemma targets with
   `selection = "measured"`. A passthrough rather than a composite because measured selection
   applies to a route's candidate list; both targets share one credential and one price, so
   reordering them can move success and latency but not cost. It is a new name, so no
   consumer moves.

`smart-nt-bounded-trial` is unchanged; the script asserts that, because it is the lane this
release is compared against.

### Load matrix

`--dry-run` against the rehearsal copy (one substitution: `[quota] state_dir` → a scratch
path, since the production directory is root-owned).

| Manifest | deployed binary `a5b9dffc` | R3 binary |
| --- | --- | --- |
| live (`f0546288…`) | PASS | **PASS** |
| candidate (Gemma lane) | **FAIL** — `unknown field 'selection', expected one of 'target', 'candidates', 'subagents'` | **PASS** (51 models, both Gemma leaves and `smart-gemini-bounded` listed) |

Two consequences, both required by the brief:

* The binary is safe to deploy **alone**: R3 runs the current manifest unchanged, so a
  binary-only rollout is a no-op for every existing route and is reversible by symlink.
* The manifest is **not** safe to deploy alone: the deployed binary refuses it at load, so a
  half-applied release cannot serve traffic in a wrong state. It fails closed.

## Unresolved serving questions (Gemma 4, native)

These are the only things standing between the staged lane and a qualified one. Each is
answerable with one authorised call.

1. **Tool calling.** The cached listing advertises `generateContent` and `countTokens`; it says
   nothing about function calling. The lane therefore declares `tool_calling = false` and a
   tool-bearing request is refused locally. Question: does the OpenAI-compatible endpoint
   accept `tools` for `gemma-4-31b-it`, and does the model emit a well-formed call?
2. **Reasoning.** The Gemma Lite lanes document that thinking cannot be disabled on that
   family. Gemma 4's behaviour is unmeasured, so the lane declares `reasoning = false` rather
   than sending a disable instruction that may be refused upstream. Question: does the family
   emit reasoning tokens, and is the output budget consumed by them (the failure mode already
   recorded for Qwen3.8 with an exhausted reasoning budget)?
3. **The 16,000-token ceiling is a TPM allowance, not a per-request limit.** Whether a single
   16,000-token request is admitted at the start of a minute, or only after the minute's
   counter is empty, decides whether the effective ceiling is lower than declared. Question:
   does a ≥15,000-token prompt pass admission on a cold counter?
4. **262,144 tokens is the listed input limit.** Whether the served context matches the
   listing (rather than being reduced by a smaller output reservation) is unverified.
5. **26B is `a4b`** — a mixture-of-experts model with roughly 4B active parameters. Whether
   its latency is stable enough to sit beside 31B in a measured ranking, or whether its
   variance dominates the EWMA, is a measurement question for the harness.

## Not done, and why

* **No live call of any kind.** No Google, no paid fallback. Deliverable 1 (the five-step
  synthetic test of the two Lite models and `smart_gemini`) is prepared and blocked on explicit
  live-call authorisation; the admission path it exercises is unit-tested, but nothing has been
  sent to Google in this release.
* **No deployment, merge, restart or credential change.** Production state is unchanged and
  verified.
* **The harness has not run against production.** `bounded_compare.py --self-test` passes and
  computes all five axes from a stub. Live mode requires `--authorize LIVE` and an explicit
  base URL, so the default can never be a paid provider.

## Cost

Unknown by construction. DeepSeek Flash and the Gemma free tier both report token counts but
the harness ships no price table, so `cost` is `null` for every route and the unpriced routes
are named in the artifact. Filling that in is a data change, not a code change, once the price
to use is the owner's to state.

## Candidate manifest identity

`routes.candidate.toml` as produced by the script from the deployed bytes:

```
sha256  65dd6d780e025d378f40565ea920c92fe24eb7c06e9d3cdf0a09aaf04896055d
lines   1671   (from 1582 deployed; +89, no lines removed or rewritten)
```

Regenerate with:

```
python3 scripts/deploy/switchyard/gemini_r3_stage.py \
  --live /var/tmp/live-routes.toml --out /var/tmp/routes.candidate.toml
```

The script is deterministic: two runs over the same input produced this same digest.
