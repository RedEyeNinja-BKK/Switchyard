# Adaptive free-pool release, R3 — 2026-11-21

Branch: `feat/gemini-r3-adaptive-bounded`, forked from
`feat/gemini-egress-admission-quota-20261120` at `52b985e7`.

Nothing here is deployed. Production remains
`20261120T181049Z-a5b9dffc7f198716-f0546288` (binary `a5b9dffc…`, manifest `f0546288…`),
`NRestarts=0`, `ActiveEnterTimestamp=Sat 2026-11-21 01:10:48 +07`.

This revision replaces the fixed Lite → Gemma → Luna → DeepSeek chain with selection over the
eligible pool, per the owner's correction (`OWNER-STEERING-ADAPTIVE-FREE-TIER-SELECTION.md`).
Nothing has been sent to any provider.

## What changed

1. **Targets state cost and reasoning-control facts.** `free_tier = true` on the four Google
   targets; `accepts_reasoning_control = false` on the two Gemma targets, whose measured wire
   refuses `reasoning_effort` with HTTP 400.
2. **Selection is adaptive.** A route with `selection = "measured"` orders the candidates that
   survived every hard gate by, in this order: cost tier (free before paid, structural), an
   exhausted allowance last, measured unreliability, input-token pressure against the target's
   own ceiling, remaining daily allowance, measured latency, and finally the declared order.
   A tool request is costed as the conversation it becomes, so the first turn alone cannot make
   a 16,000-token-per-minute leg look appropriate.
3. **Evidence is shared.** One `MeasuredSelector` serves every measured route in the
   deployment, because a physical model behaves the same whichever alias named it. Real
   bounded work informs the ordering instead of training a subsidiary route nothing calls.
4. **Native Google thinking is untouched.** No route-level or target-level reasoning policy,
   effort or dialect exists on any Google leg. The two paid NT targets keep their own
   independent `reasoning_policy = "disabled"`.
5. **`smart-nt` and `smart_gemini` are adaptive** over the eligible Google pool, with the paid
   ladder strictly after free. Both remain `[routes]` passthroughs, so their aliases stay in
   the served model list.

## What the code now guarantees

| Property | Where | Test |
| --- | --- | --- |
| A small bounded task reaches the 14,400-RPD leg without a first-leg failure | `selection.rs` + `route.rs` | `a_small_bounded_task_reaches_the_daily_rich_leg_before_the_scarce_one` |
| A request that fills the small leg's minute goes to the leg with headroom | same | `a_request_that_fills_the_small_legs_minute_goes_to_the_wide_leg` |
| A tool exchange is not placed on the leg that cannot hold its answer turn | same | `a_tool_exchange_is_not_placed_on_the_leg_that_cannot_hold_its_answer` |
| A paid leg never overtakes an eligible free leg, whatever the declared order | same | `a_paid_leg_never_overtakes_an_eligible_free_leg` |
| A request carrying reasoning control skips a leg that cannot accept it | `request_fit.rs` + `route.rs` | `a_reasoning_control_skips_the_leg_that_cannot_accept_it` |
| No reasoning control is invented for a Google leg in either direction | `route.rs` | `a_google_leg_is_never_handed_a_reasoning_control_it_was_not_given` |
| A spent allowance is offered after a leg that still has budget | `selection.rs` | `a_spent_allowance_is_offered_after_a_leg_that_still_has_budget` |
| An uncleared payload never reaches a Google leg inside a composite | `composition.rs` | `an_uncleared_payload_skips_a_gated_child_and_serves_the_next_leg` |
| A directly addressed Google leaf still refuses, with zero provider calls | `route.rs` | `a_directly_addressed_gated_leaf_still_refuses_and_is_never_called` |
| A request above a target's window or per-request ceiling is not dispatched to it | `route.rs` | `a_request_beyond_a_targets_window_goes_to_the_wider_target`, `a_request_beyond_the_input_budget_is_not_sent_to_that_target` |
| When no candidate fits, the refusal names the binding limit and makes no call | `route.rs` | `a_request_no_candidate_can_hold_is_refused_without_a_call` |
| A route may not advertise more context than its widest declared target | `config.rs` | `a_route_cannot_advertise_more_context_than_its_widest_target` |
| A target that states no window keeps loading unchanged | `config.rs` | `an_undeclared_window_admits_every_request` |

Suite: **811 tests, 0 failures** across `switchyard-runner`, `switchyard-libsy` and
`switchyard-llm-client`. The adaptive integration group is **23 tests**; `selection.rs`
carries 12 unit tests.

Mutation controls, ten mutants, all caught:

| | Mutation | Caught by |
| --- | --- | --- |
| M1 | `payload_refused` always false | privacy skip test |
| M2 | `physical_ineligibility` always `None` | window/budget skip tests |
| M3 | selector never installed | measured-selection test |
| M4 | physical-context validation removed | over-claim test |
| M5 | output reservation ignored in the size estimate | output-reservation test |
| M6 | composite context claim unchecked | composite over-claim test |
| M7 | physical refusal reason dropped | refusal-message test |
| M8 | cost tier removed from the sort key | paid-never-first test |
| M9 | daily allowance ignored when building candidates | small-task test |
| M10 | reasoning-control fact ignored in eligibility | reasoning-control skip test |

## The manifest change, and why it is code-shaped

`scripts/deploy/switchyard/gemini_r3_stage.py` reads the **deployed** manifest, refuses to run
if any anchor is not matched exactly once, and writes a candidate. It does not carry a copy of
the manifest, so it cannot go stale against production.

Input manifest: `/var/lib/switchyard/releases/CURRENT/routes.toml`,
sha256 `f0546288179e7e40523e900d7fa6a010bc3a0e908cb5e8ad749a3147a0c71d1b`.

Edits:

1. The two Gemini Lite targets gain `context_window = 1048576`,
   `max_input_tokens = 250000` and `free_tier = true`.
2. Two Gemma targets are added at native ids with the model's 262,144-token window, a
   16,000-token per-request ceiling, a 14,400 RPD budget, `free_tier = true` and
   `accepts_reasoning_control = false`.
3. `smart_gemini` becomes an addressable passthrough route over all four Google targets with
   `selection = "measured"`; `smart-gemini-chat` and `smart-gemini-bounded` are passthroughs
   with the same policy.
4. `smart-nt` keeps its alias and kind, gains the four Google legs ahead of the unchanged paid
   ladder, and sets `selection = "measured"`. Its declared order is now only a tie-break.

`smart-nt-bounded-trial` is unchanged, and the script asserts that the paid targets and the
Luna-NT alias are byte-identical to the live file.

### Load matrix

`--dry-run` against the rehearsal copy (one substitution: `[quota] state_dir` → a scratch
path, since the production directory is root-owned).

| Manifest | deployed binary `a5b9dffc` | R3 binary `1026d953` |
| --- | --- | --- |
| live (`f0546288…`) | PASS | **PASS** |
| candidate (adaptive pool) | **FAIL** — `unknown field 'selection'`, and the new target facts | **PASS** (54 models, both Gemma leaves, `smart-nt`, `smart_gemini`, `smart-gemini-chat`, `smart-gemini-bounded`) |

A binary-only rollout is a no-op for every existing route; the manifest cannot be deployed to
the old binary at all.

## Unresolved serving questions (Gemma 4, native)

1. **Tool calling.** Measured as working on both Gemma 4 models (2026-11-21 capability probe).
   The direct leaves still decline to *promise* it; `smart-nt`/`smart_gemini` rely on the
   measured fact through the target layer.
2. **Reasoning.** Native thinking is preserved and never disabled. `accepts_reasoning_control
   = false` records only that a caller's explicit control must not be forwarded; a request
   that carries one is routed to a leg that can take it.
3. **TPM versus per-request admission.** Whether a 16,000-token request is admitted on a cold
   minute counter is still unverified; the ceiling is treated as a per-request bound.
4. **Served context.** 262,144 tokens is the listing; the served window is unmeasured.
5. **26B latency stability.** Whether the A4B model's latency variance dominates the EWMA is a
   measurement question for the harness.

## Not done, and why

* **No live call of any kind.** No Google, no paid fallback.
* **No deployment, merge, restart or credential change.**
* **The harness has not run against production.** `bounded_compare.py --self-test` passes.

## Cost

Unknown by construction: the harness ships no price table, so `cost` is `null` for every route
and the unpriced routes are named in the artifact.

## Candidate manifest identity

`routes.candidate.toml` as produced by the script from the deployed bytes:

```
sha256  c0b4a24a8bc123c21b31ec8b83889361c031541cf7f1aa7705be0bf020e6cdfb
lines   1781   (from 1582 deployed; +199, no lines removed or rewritten)
```

Regenerate with:

```
python3 scripts/deploy/switchyard/gemini_r3_stage.py \
  --live /var/tmp/live-routes.toml --out /var/tmp/routes.candidate.toml
```

The script is deterministic: two runs over the same input produce the same digest.
