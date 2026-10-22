# Switchyard Metrics Reference

Operational reference for the Prometheus exposition served by a Switchyard
deployment. Pair with [`examples/prometheus/`](../../examples/prometheus/) for
a drop-in scrape config and starter alert rules.

## Endpoint

| Property | Value |
|---|---|
| Path | `GET /metrics` (HTTP path is `/metrics`, **not** `/v1/metrics`) |
| Content-Type | `text/plain; version=0.0.4; charset=utf-8` |
| Format | Prometheus text format 0.0.4 |
| Auth | None |
| Default scrape interval | 15s |

`GET /metrics` is served by the native Rust server.

A JSON summary of the same traffic lives at `GET /v1/stats`.

## Top-line gauges (no labels)

| Metric | Type | Meaning |
|---|---|---|
| `switchyard_total_requests` | gauge | Successful and failed routed model calls since process start. Classifier and judge calls are excluded; a context-window fallback can add another routed call. |
| `switchyard_total_errors` | gauge | Failed routed model calls since process start. |

## Per-endpoint counters

The `model` label is the configured endpoint id (`openai/gpt-5.5`,
`azure_openai/gpt-5.5`, etc.).

The `tier` label is optional. It is present when an algorithm defines a stable
tier for the selected model, such as `strong` or `weak` for a two-tier classifier.

| Metric | Type | Meaning |
|---|---|---|
| `switchyard_requests_total{model}` | counter | Successful routed model calls per endpoint. |
| `switchyard_errors_total{model}` | counter | Failed routed model calls per endpoint. |
| `switchyard_prompt_tokens_total{model}` | counter | Prompt-token billing per endpoint. |
| `switchyard_completion_tokens_total{model}` | counter | Completion-token usage per endpoint. |
| `switchyard_cached_tokens_total{model}` | counter | Cached prompt tokens per endpoint. |
| `switchyard_cache_creation_tokens_total{model}` | counter | Cache-creation tokens per endpoint. |
| `switchyard_reasoning_tokens_total{model}` | counter | Reasoning tokens per endpoint. |

## Per-endpoint latency histograms

Each histogram emits `_bucket`, `_sum`, and `_count` series. Use
`histogram_quantile` in PromQL to calculate a percentile.

| Metric | Type | Meaning |
|---|---|---|
| `switchyard_model_call_latency_ms{model}` | histogram | Successful final routed-call latency. |
| `switchyard_total_latency_ms{model}` | histogram | End-to-end latency for successful routed responses. For streaming responses this is full-turn time, **not** time-to-first-token. |

## Routing overhead (global, no model label)

| Metric | Type | Meaning |
|---|---|---|
| `switchyard_routing_overhead_ms{algorithm}` | histogram | Total run time minus the time spent in successful routed model calls, with overlapping hedged calls counted once. Includes classifier calls, failed routed attempts, target resolution, and decision publication; runs with no successful routed call are not recorded. Measured across the whole run, so it does not reconcile with `switchyard_run_duration_ms`, which times only the algorithm task. |

## In-flight runs

| Metric | Type | Meaning |
|---|---|---|
| `switchyard_algorithms_in_flight{algorithm}` | gauge | Algorithm runs that have started and not yet finished. |

## Classifier fail-open counter

| Metric | Type | Meaning |
|---|---|---|
| `switchyard_classifier_fail_open_total{judge_model,reason}` | counter | Judge failures that made a classifier route without a verdict. The caller's request can still succeed on the fallback target. |

`judge_model` is the configured judge target. `reason` is one of `timeout`, `transport`,
`upstream_5xx`, `upstream_non_5xx`, `invalid_response`, `parse_error`, `client_error`, or
`call_error`. The labels never include request or response text.

## Outcome counters for error-rate ratios

For HTTP-derived responses and attempts, the `outcome` label takes three values:

* `ok` = HTTP 2xx
* `retryable_error` = HTTP 408, 429, any 5xx, or a failure before an HTTP status
* `other_error` = everything else (400, 401, 403, 422, …)

In addition, `switchyard_client_responses_total` records `outcome="client_disconnected"` when a client drops the connection before a response is returned.

| Metric | Type | Meaning |
|---|---|---|
| `switchyard_client_responses_total{outcome}` | counter | HTTP responses returned to clients on the LLM-serving routes (`/v1/chat/completions`, `/v1/messages`, `/v1/responses`). The denominator for the **router-served** error rate. |
| `switchyard_upstream_attempts_total{outcome,code}` | counter | Individual upstream call attempts. One client request can produce N attempts via retry. The denominator for the **direct-to-endpoint** baseline error rate. The `code` label carries the raw upstream HTTP status for plotting the error-code distribution (see below). |
| `switchyard_router_retry_recovered_total` | counter | Upstream operations that succeeded after at least one retry. |

### The `code` label on `switchyard_upstream_attempts_total`

`code` is the raw upstream HTTP status as a string: `"200"`, `"429"`,
`"500"`, `"504"`, etc. Two special values:

* `code="none"`: a non-HTTP failure (network error, connection reset,
  pre-status timeout). The attempt never received a status line, so there
  is no code. These also count as `outcome="retryable_error"`.
* `code="4xx"` / `code="5xx"` / `code="1xx"` / `code="3xx"` / `code="other"`:
  an HTTP code outside the known-codes allowlist, clamped to its class so
  a misbehaving upstream cannot blow up label cardinality.

`outcome` is fully determined by `code`, so adding the label does not
multiply series. You get one series per distinct code either way. The
canonical codes (`200`, `429`, `500`, `504`, `none`) are seeded at `0` so
their time series exist from process start (a `rate()` over a never-seen
counter reads as "no data", not zero).

## Computing the success-criterion ratios

```promql
# Router error rate (the rate clients see)
router_error_rate =
  sum(rate(switchyard_client_responses_total{outcome="retryable_error"}[5m]))
  / sum(rate(switchyard_client_responses_total[5m]))

# Direct-endpoint error rate (what clients would have seen without the router)
direct_error_rate =
  sum(rate(switchyard_upstream_attempts_total{outcome="retryable_error"}[5m]))
  / sum(rate(switchyard_upstream_attempts_total[5m]))

# Headline metric: positive value means the router is reducing client errors
error_rate_reduction = direct_error_rate − router_error_rate

# Traffic share per endpoint
sum by (model) (rate(switchyard_requests_total[5m]))
  / ignoring(model) group_left sum(rate(switchyard_requests_total[5m]))

# Error-code distribution over time (stack the series in a Grafana time-series panel)
sum by (code) (rate(switchyard_upstream_attempts_total{code!="200"}[5m]))

# Same, as a 100%-stacked share rather than absolute rates
sum by (code) (rate(switchyard_upstream_attempts_total{code!="200"}[5m]))
  / ignoring(code) group_left
sum      (rate(switchyard_upstream_attempts_total{code!="200"}[5m]))
```

> **Note:** because `switchyard_upstream_attempts_total` now carries the
> `code` label, always wrap a bare selector in `sum()` (as the ratio
> queries above do) when you want a layer total. Otherwise the selector
> returns one series per code.

The ready-to-deploy alert rules implementing these expressions live in
[`examples/prometheus/switchyard.rules.yaml`](../../examples/prometheus/switchyard.rules.yaml).

## Economics and token accounting

These series attribute usage to the provider, route and cost class the deployment
declares, and report the provider account state Switchyard reads with its own
credentials. Every value comes from the deployment configuration, never from a
model-name guess.

### Token components

| Metric | Type | Meaning |
|---|---|---|
| `switchyard_economics_tokens_total{component,model,provider,route,route_family,cost_class,outcome}` | counter | Tokens consumed, split by component. `component` is one of `prompt`, `cached`, `cache_creation`, `completion`, `reasoning`. |
| `switchyard_economics_requests_total{component="request",...}` | counter | Requests served, with the same attribution labels. |
| `switchyard_economics_provider_cost_total{component="provider_reported",...}` | counter | The cost the **provider itself** reported for the request. |
| `switchyard_economics_estimated_cost_total{component="estimated",...}` | counter | A Switchyard-derived cost estimate. **Reserved and currently unpublished**: Switchyard has no validated rate card, so nothing writes this family and a query against it returns no series. It is published beside the provider-reported cost, never instead of it. |

The provider-reported cost and the Switchyard-derived estimate are deliberately
separate metric families, so a dashboard can never present one as the other.

> **Note:** `cached` and `cache_creation` are *components of* `prompt`, not
> additions to it. `prompt` is the provider's own input total
> (`input + cached + cache_creation`). Adding the components to the prompt total
> double counts the cache. There is deliberately no "all tokens" series.

`reasoning` is a *component of* `completion` in the same way, not an addition to
it. `completion` is the provider's own output total, and on every provider that
reports reasoning separately that reasoning count sits inside the completion count.
Measured 2026-10-21 against the live OpenRouter API: a response reported
`completion_tokens: 650` with `completion_tokens_details.reasoning_tokens: 521` and
`total_tokens: 680`, where `prompt + completion == total_tokens` and
`prompt + completion + reasoning` did not. A total token figure, were one published,
would therefore be `prompt + completion` and nothing else.

`reasoning` is published only when the provider reported it. A provider that does
not report reasoning yields no such series, which reads as absent rather than as a
measured zero.

### Attribution labels

| Label | Values | Meaning |
|---|---|---|
| `provider` | One per configured `llm_client`. | The client the serving target is bound to. |
| `route` | One per configured route. | The route the **caller** addressed. On a composed route this is the parent. |
| `child_route` | One per configured child route; absent on a direct route. | The child that served the request. |
| `route_family` | `direct`, `primitive`, `tier`. | A route's role, derived from its configuration: bound to one target, declares an ordered candidate list, or composes other routes. |
| `cost_class` | `local`, `free`, `subscription`, `metered`. | How the provider bills the lane. |

`cost_class` is a deployment fact. `local` is a deployment the operator hosts,
`free` is a model the provider prices at zero, `subscription` is an
allowance-bearing credential, and `metered` is a metered API key on a paid
provider. A `free` or `subscription` lane must never be given a per-token price.

### Account state

Switchyard reads these provider account surfaces with the credential it already
uses to serve inference, and publishes the result. Nothing downstream of
Switchyard contacts a provider.

| Metric | Type | Meaning |
|---|---|---|
| `switchyard_economics_account_fetch_total{provider,outcome}` | counter | One per account read. `outcome` is `ok`, `unauthorized` or `failed`. |
| `switchyard_economics_account_auth_failures_total{provider}` | counter | Reads rejected with 401 or 403. |
| `switchyard_economics_account_freshness_seconds{provider}` | gauge | Age of the last successful read. |
| `switchyard_economics_account_fetch_duration_seconds{provider}` | gauge | How long the last read took. |
| `switchyard_economics_provider_balance{provider,currency,scope,period}` | gauge | The remaining balance the provider reported. |
| `switchyard_economics_provider_credit{provider,currency,scope,period}` | gauge | The purchased credit the provider reported. |
| `switchyard_economics_provider_usage{provider,currency,scope,period}` | gauge | The usage the provider reported. |
| `switchyard_economics_allowance_remaining_percent{provider,pool}` | gauge | Remaining allowance of a provider-reported window. |
| `switchyard_economics_allowance_reset_seconds{provider,pool}` | gauge | Time until that window resets. |

A provider that exposes no account surface produces no account series at all. It
is not recorded as a zero balance: a provider that exposes nothing has not
reported a balance of zero.

The three amount families carry three separate labels, because a provider figure
has three separate facts attached to it:

| Label | Values | Meaning |
|---|---|---|
| `currency` | `USD`, `CNY`, ... | The currency the amount is denominated in. |
| `scope` | `account`, `key` | Whether the amount covers the whole provider account or the single API key Switchyard holds. |
| `period` | `total`, `daily`, `weekly`, `monthly` | The reporting window the amount covers. |

These are not combined into one label. A currency label that also carried the
scope or the period would make two different amounts indistinguishable without
parsing the label value, and would invite a query to sum series that were never
meant to be summed. A reader selects on `currency`, `scope` and `period`
independently.

The three amount families are three different facts and are not interchanged:

* `provider_credit` is the credit the provider sold.
* `provider_usage` with `scope="account"` is the credit that has been consumed.
* `provider_balance` is the credit that is left, which is the difference between
  the two.

A provider that reports a purchased credit but no consumed credit publishes no
balance, because the remaining amount is not known. A provider that reports a
balance directly, such as DeepSeek, publishes the balance and no credit.

### Bangkok calendar day

| Metric | Type | Meaning |
|---|---|---|
| `switchyard_economics_calendar_tokens{component,...}` | gauge | The same token components, for the current Bangkok day. |
| `switchyard_economics_calendar_requests{...}` | gauge | The same request count, for the current Bangkok day. |
| `switchyard_economics_calendar_day_start` | gauge | Unix timestamp of the current Bangkok midnight. |
| `switchyard_economics_calendar_day_end` | gauge | Unix timestamp of the next Bangkok midnight. |
| `switchyard_economics_calendar_day_elapsed_seconds` | gauge | Seconds since the current Bangkok midnight. |

The day boundary is midnight in `UTC+07:00`, not midnight in UTC and not a
rolling twenty-four hours. Prometheus cannot express this: the `@` modifier
needs a literal timestamp, and a computed one is rejected by the API. Switchyard
therefore publishes the calendar-day totals itself.

> **Note:** these are **gauges**, not counters. The calendar is republished every
> thirty seconds; as counters the published total would grow by the day's total on
> every publish.

## Cardinality

All labels are bounded enums. No per-request or per-user values escape
into label space.

| Label | Values | Where |
|---|---|---|
| `model` | One per configured endpoint, typically 2–6 per deployment. | All per-endpoint metrics. |
| `outcome` | `ok`, `retryable_error`, `other_error`; plus `client_disconnected` on `switchyard_client_responses_total`. | Outcome counters |
| `code` | Bounded: the known-code allowlist (`200`, `400`, `401`, `403`, `404`, `408`, `409`, `422`, `429`, `500`, `502`, `503`, `504`), plus `none` and the per-class buckets `1xx`/`2xx`/`3xx`/`4xx`/`5xx`/`other`. About 20 values max. | `switchyard_upstream_attempts_total` |
| `le` | The configured histogram bucket boundaries. | Histogram buckets |
| `algorithm` | One stable value per configured algorithm. | Routing-overhead histogram, in-flight gauge |
| `tier` | Small enumerated set, optional. | Per-endpoint counters and histograms on algorithms that supply it |
| `judge_model` | One per configured judge target. | Classifier fail-open counter |
| `reason` | Exactly 8 fixed error categories. | Classifier fail-open counter |

## Triage cheatsheet

| Symptom on `/metrics` | Likely cause |
|---|---|
| `model="<unknown>"` rows appear | A routed-call observation did not include a selected model. |
| All counters at 0 after warm-up | Server just started with no traffic, or the scraper is hitting the wrong port. |
| `switchyard_routing_overhead_ms_count` stuck at `0` | No successful algorithm run has recorded a successful routed model call. |
| `switchyard_algorithms_in_flight` stuck above zero with no traffic | Runs are parked on an internal routing call that never returns. Check the classifier or judge target's upstream. |
| `switchyard_classifier_fail_open_total` rising | The judge target is failing or returning a response the classifier cannot parse. Check `judge_model` and `reason`. |
| `switchyard_client_responses_total{outcome="retryable_error"}` rising | Either the upstream is genuinely flaky, or retries are exhausting; compare client responses with retryable upstream attempts. |
