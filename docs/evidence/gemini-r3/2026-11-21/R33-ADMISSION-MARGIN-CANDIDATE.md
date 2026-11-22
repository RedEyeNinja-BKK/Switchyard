# R3.3 — padded free-lane admission, with the measurement that forced it

Status: **candidate built and verified; nothing deployed.**
Production remains R3.2 (`20261120T222915Z-5d8c6562f5a73ef9-308e046d`), which is also
`LAST-KNOWN-GOOD`.

## What was measured

The owner's AI Studio reading records Gemma 26B at **18.05K input tokens against a 16K
per-minute limit**. The request behind it:

| quantity | value |
| --- | --- |
| request characters | 60,809 |
| local estimate | 15,202 tokens |
| provider `prompt_tokens` | 18,054 |
| estimator error | **−18.8 % (under-count)** |

Admission compared 15,202 against the 16,000-token window and admitted the request. The
provider counted 18,054 and refused the minute. Nothing was wrong with the pacing mechanism —
it refused, then admitted, against a number that was too small.

This project holds two measurements of the same estimator and they disagree in **direction**:
one over-counted near 24 %, this one under-counted near 19 %. No fixed accuracy bound is
therefore asserted anywhere in this change. The pad is a margin against measured overshoot,
and the ledger learns the reported count back for the life of each window.

## What changed

### 1. Admission margin

A target may declare `admission_margin_permille`. Every input reservation is multiplied by it
(ceiling division) before the window comparison:

```
reservation = ceil(estimate × margin / 1000)
admitted    = reservation ≤ input_tokens_per_minute
```

The same arithmetic inverted gives the largest estimate a window can admit:

```
safe_ceiling = floor(input_tokens_per_minute × 1000 / margin)
```

Load now refuses three shapes rather than discovering them as a 429 under traffic:

| declaration | verdict | why |
| --- | --- | --- |
| margin with no `input_tokens_per_minute` | refuse | pads nothing; asserts absent protection |
| margin below 1000 | refuse | reserves less than the estimate: an under-count by construction |
| `max_input_tokens` above the padded ceiling | refuse | promises capacity the governor refuses on every large request |

The third rule is what makes the ceiling honest. With a 16,000-token window and a 1,300
margin, the largest admittable estimate is **12,307**, so the manifest must state 12,307 — a
stated ceiling above that is a number no request can ever use.

The route filter and the governor share the derived value (`safe_input_tokens`), so they
cannot disagree about whether a candidate is eligible.

### 2. Reconciliation from provider-reported counts

After a buffered success the provider's own input count is read from every wire shape this
deployment speaks and written back to the minute ledger, raising both the request's own slot
and the model's observed ratio (floor 1.0, never decreasing):

- `usage.prompt_tokens` (OpenAI chat / responses)
- `usage.input_tokens`, plus `cache_read_input_tokens` and `cache_creation_input_tokens`
- `usageMetadata.promptTokenCount` (native Google)

A response carrying no readable count changes nothing — the reservation stands.

### 3. Refusal as its own outcome

A request that cannot fit its window even after the pad is
`InputExceedsMinuteWindow`, surfaced as **429** with code
`minute_allowance_input_exceeds_window`. It is classified for routing as rate-limited, so a
composite crosses to a wider sibling; it is never retried in place, because no amount of
waiting makes a request smaller.

### 4. HTTP 200 is not a completed task

An empty body with a zero completion count (the observed Gemma shape) is now recorded as an
incomplete answer with a reason label on `switchyard.incomplete_answers`:

- `empty_answer_zero_completion`
- `empty_answer_no_usage`
- `empty_answer_reasoning_only`

The response is still returned — this is a measurement, not a refusal. It stops an empty
success from being counted as work.

### 5. Paid spend limits are a separate meter

Google's rolling ten-minute **spend** limits are Free **N/A** and reach this deployment only
as structured paid-tier evidence. A spend-scoped condition cools the model down and does not
retry it in place, and does not stop the candidate list. A bare throttled 429 remains a
short-window throttle, not a spend signal.

## The staged manifest

`scripts/deploy/switchyard/gemini_r33_admission_margin_stage.py` splices the live document
and proves nothing else moved.

| target | `max_input_tokens` | `admission_margin_permille` |
| --- | --- | --- |
| `direct-google-gemma-4-31b-it` | 16,000 → **12,307** | — → **1,300** |
| `direct-google-gemma-4-26b-a4b-it` | 16,000 → **12,307** | — → **1,300** |
| `direct-google-gemini-3.5-flash-lite` | unchanged (250,000) | unchanged (none) |
| `direct-google-gemini-3.1-flash-lite` | unchanged (250,000) | unchanged (none) |

Diff is `+12 / −5` lines (`1799 → 1806`), confined to the two Gemma blocks: two ceiling values, two margin
lines, and one comment that had become false (it stated the ceiling sat at the minute's token
budget — it now sits below it, and a comment contradicting its own field is drift the next
reader trusts).

The Lite lanes are deliberately untouched: their 250,000-token window absorbs the same
estimator error instead of crossing it, and narrowing their ceiling would withdraw capacity
the provider actually grants.

## This is not a config-only release

`admission_margin_permille` is a field the running binary does not know. Measured, with the
deployed R3.2 binary against the candidate manifest:

```
unknown field `admission_margin_permille`, expected one of `id`, `llm_client`, ...
```

So the candidate manifest and the candidate binary ship together. The installer validates the
candidate manifest with the candidate binary before the swap, which is exactly this case.

## Verification

| check | result |
| --- | --- |
| `cargo test --workspace --exclude switchyard-nemo-relay-plugin` | **1,467 passed / 0 failed**, 54 suites |
| touched crates (`llm-client`, `runner`, `server`) | **775 passed / 0 failed**, 27 suites |
| `cargo clippy … --all-targets -D warnings` | clean |
| `cargo fmt --check` (workspace) | clean |
| candidate binary + candidate manifest, `--dry-run` | `server OK`, 54 models |
| candidate binary + live R3.2 manifest, `--dry-run` | `server OK` (backward compatible) |
| R3.2 binary + candidate manifest, `--dry-run` | refused, `unknown field` (pairing proven) |

Excluded: `switchyard-nemo-relay-plugin`'s lib tests do not compile on this branch base
(`missing field hosted_reasoning in initializer of TranslationPolicy`, introduced by
`074b2bec`). The crate's library builds; its test target does not. Untouched by this change.

### New tests

- `the_measured_gemma_overrun_is_refused_before_egress_under_an_admission_margin`
- `the_reported_input_count_is_read_from_each_wire_shape`
- `the_learned_ratio_outlives_the_window_it_was_measured_in`
- `the_declared_ceiling_is_the_inverse_of_the_applied_margin`
- `the_margin_still_admits_ordinary_traffic`
- `a_near_limit_gemma_request_never_reaches_the_padded_window`
- `a_margin_without_a_token_window_refuses_the_deployment`
- `a_margin_below_parity_refuses_the_deployment`
- `a_ceiling_above_the_padded_window_refuses_the_deployment`
- `a_padded_lane_loads_at_its_derived_ceiling`

### Negative control

The unpaced-margin rule was first written *inside* the paced-target loop, where the
`continue` ran before the margin was validated — so a margin naming no window loaded silently
and protected nothing. Moving the check before the `continue` closed it, and the test was
negative-controlled: on the pre-fix ordering it **fails** (`deployment loaded but should have
been refused`); on the fixed ordering it passes. The first version of that test set
`requests_per_minute` and therefore passed on both — which is why the fixture now declares a
margin and nothing else.

## Artifacts

| artifact | value |
| --- | --- |
| commit | `3c469fc6b825d77d909036fdaa1b274d367d4158` |
| branch | `feat/gemini-r3-adaptive-bounded` |
| binary sha256 | `2ee22cd9f6275cce6592558e58757ea70f2e1073c8f4dddc468d62dc61f2b5b3` |
| binary size | 28,194,616 bytes |
| candidate manifest sha256 | `b28bd85cb5be30172d7d990f266ac78fe85138ceab1be7fa10984ec1111d94a9` |
| live (R3.2) manifest sha256 | `308e046d21cfedf26ffbb5612c5c704801a24a566d207c28da3f99757dada7a9` |

Reproduce the manifest byte-for-byte:

```bash
python3 scripts/deploy/switchyard/gemini_r33_admission_margin_stage.py \
    --live  <live routes.toml> \
    --out   /var/tmp/r33-candidate-routes.toml
sha256sum /var/tmp/r33-candidate-routes.toml   # b28bd85c…
```

## Production state, unchanged

- release `20261120T222915Z-5d8c6562f5a73ef9-308e046d`, promoted to `LAST-KNOWN-GOOD`
- `MainPID=390211`, `NRestarts=0`, running since `2026-11-21 05:29:14 +07`
- `/health` ok, 54 models
- `/etc/switchyard/routes.toml` sha256 `308e046d…7a9` — equal to the release manifest

No deploy, restart, merge or promotion was performed for R3.3. Deployment needs explicit
owner GO, and the window must be coordinated because it is a binary plus manifest swap.
