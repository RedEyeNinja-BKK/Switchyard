# Economics telemetry — provenance and known gaps

This is the handoff record for the economics telemetry release. It says where
every published number comes from, which numbers are not native to Switchyard,
and what would be needed to make them native. It is deliberately specific about
what is not known.

Read it before changing a panel. The dashboard is only honest while its
provenance labels stay true.

## 1. Deployment provenance

The dashboard, the binary that feeds it and the source that produced the binary
are three separate artefacts. They are recorded together so a future reader can
reproduce all three rather than trusting a version number.

| Artefact | Identifier |
|---|---|
| Production binary | `1cde1c75a4c5887c3167a19a8b45dffa0306387606da59117fca1ed111652cc4` |
| Production routing config | `42ca3f68a6f16a4b70b6406dae9cff1b7e71713a4268e6dd6a0afa7e55b255fa` |
| systemd unit | `83e6148b2e052c87eb2a1d59246ad28bf0746a580d43867d1db1432b8a6e16c3` |
| Source branch | `integration/economics-on-production-20261023` |
| Source commit | `7fcdc754a5eb920933be4003a9cb3d3e00e36211` |
| Dashboard | `turnstone-switchyard-economics-v8`, version 4 |
| Prometheus job | `switchyard-v7` |

The binary was built from the Rust tree of commit `f76d80d`. That tree is
byte-identical through the branch head: every later commit touches only
`scripts/economics/` and its tests, so the binary and the dashboard generator
can be reasoned about from the same commit even though the binary predates it.
Confirm this before relying on it:

```bash
git rev-parse f76d80d:crates <branch-head>:crates   # must print the same tree
```

The branch is a stack on the production commit, not a fork of the economics
development branch:

```
7c49ac3  production baseline
  ├─ 33e333f  attribution by provider, route family and cost class
  ├─ f76d80d  currency, scope and period as separate labels   <- binary built here
  ├─ 2652acf  docs: partial OpenRouter coverage
  ├─ 6a34cb0  docs: the OpenAI account naming split
  ├─ a279909  account rows in the operator's shape
  ├─ 83ccb67  DeepSeek figures only the older exporter publishes
  └─ 7fcdc75  generator can publish a revision
```

`33e333f` and `f76d80d` are cherry-picks of the economics development branch's
`6da398f` and `67f3dcd`. `git range-diff` shows no content difference beyond the
cherry-pick trailer. Both lineages are preserved outside the branch:

- `refs/preserved/production-7c49ac3`
- `refs/preserved/economics-67f3dcd`
- a complete-history bundle of the production lineage

## 2. Where the numbers come from

### Native — published by Switchyard from its own provider reads

| Metric family | Source |
|---|---|
| `switchyard_economics_tokens_total` | Served request accounting, attributed from deployment configuration. |
| `switchyard_economics_requests_total` | The same requests, counted once. |
| `switchyard_economics_provider_cost_total` | Emitted **only** when the provider reports a cost for the request. |
| `switchyard_economics_calendar_tokens`, `_calendar_requests` | The same usage accumulated into the current Bangkok day. |
| `switchyard_economics_provider_balance` | The provider's own balance endpoint, read with Switchyard's credential. |
| `switchyard_economics_provider_credit`, `_provider_usage` | The provider's own account surface, where one exists. |
| `switchyard_economics_allowance_remaining_percent`, `_allowance_reset_seconds` | Provider-reported allowance windows. |
| `switchyard_economics_account_freshness_seconds`, `_fetch_total`, `_auth_failures_total` | The health of the account reads themselves. |

### Legacy — read from the older localclaw economics exporter

Four DeepSeek panels read `job="localclaw-econ"`, not `job="switchyard-v7"`:

| Panel | Expression |
|---|---|
| DeepSeek Top-Ups (total, CNY) | `localclaw_deepseek_top_up_total_cny` |
| DeepSeek Lifetime Spend (CNY) | `sum(localclaw_deepseek_spend_balance_delta_total_cny)` |
| Cumulative Actual Spend (CNY) | `sum(localclaw_econ_actual_cost_total{currency="CNY",provider="deepseek"})` |
| DeepSeek lifetime spend (CNY) | Both of the above, as a time series |

These are the only non-native figures on the dashboard. Everything else is
Switchyard's own telemetry.

## 3. DeepSeek: a temporary, explicit exception

**These four panels are not native, and the dashboard does not claim they are.**
They are an exception, kept because the figures are real, current and
operationally useful, and because deleting them would remove information an
operator acts on.

### Why Switchyard cannot produce them today

Switchyard's DeepSeek reader calls `GET https://api.deepseek.com/user/balance`.
That endpoint reports a **level**, not a ledger, and it is the only account
surface DeepSeek exposes. From two readings of a level, a top-up and a day of
usage are indistinguishable, so Switchyard cannot derive:

- **cumulative top-ups** — no purchase history exists in the response;
- **lifetime spend** — would require accumulating balance deltas, which is an
  inference rather than a measurement, and is exactly what the legacy exporter
  does;
- **request-derived actual cost** — DeepSeek reports no per-request cost, so
  `provider_cost_total` is never emitted for it. `record_estimated_cost` exists
  but is never called anywhere, and there is no validated rate card, so no
  estimate is published either.

The legacy figures are therefore *not* reclassified as native data, and no
historical value is reconstructed or smoothed.

### What Switchyard now measures natively

DeepSeek's balance response also carries `granted_balance` and
`topped_up_balance` alongside `total_balance`, and Switchyard now reads all
three. They are published as
`switchyard_economics_provider_balance_component{component="granted"|"topped_up"}`,
beside the total and never added to it.

This is remaining credit by kind, not purchase history. `topped_up_balance` in
particular is the topped-up money **still remaining**: an account that bought a
great deal and spent it reports a small remaining figure.

### No historical API exists

Probed 2026-10-23 with the deployment's own credential:

| Endpoint | Result |
|---|---|
| `/user/balance` | 200 |
| `/models` | 200 |
| `/user/transactions` | 404 |
| `/billing/transactions` | 404 |
| `/user/usage` | 404 |
| `/usage` | 404 |
| `/dashboard/billing/usage` | 404 |

DeepSeek publishes a point-in-time balance and a model list. It exposes no
transaction ledger, no usage history and no per-request cost. Because the four
legacy figures cannot be observed, they are not reconstructed: no balance
differencing, and no rate-card estimate presented as a measurement.

A reader added to DeepSeek's own reader would not fix this, and neither would
polling more often: the information is not in the response.

### What full replacement requires

1. **Cumulative top-ups** — either a purchase ledger Switchyard maintains from
   observed balance increases (an inference that must be labelled as one), or a
   provider endpoint that reports purchases. DeepSeek exposes no such endpoint.
2. **Lifetime spend** — the same ledger, accumulated. Not authoritative without
   (1).
3. **Request-derived cost** — a validated DeepSeek rate card and a call site for
   `record_estimated_cost`, or per-request cost from the provider. Neither
   exists. Anything published before a rate card is validated would be a guess
   wearing a unit label.

Until (1)–(3) exist, this exception stands and stays labelled.

### Proposed cutover (not applied)

The legacy exporter must not be shut down and the four panels must not be
deleted without an approved cutover. The proposed shape:

1. **Add** two native balance-component panels to the DeepSeek row. Nothing is
   removed, and the native figures appear beside the legacy ones with no
   ambiguity about which is which.
2. **Freeze** the legacy panels to a fixed historical time range and retitle
   them `(historical, legacy exporter)`, so they read as an archive of a past
   series rather than as a live measurement.
3. **Move** them into a collapsed row that is closed by default, so the live
   row is Switchyard-only.
4. **Keep** the exporter running until the frozen window has been captured in a
   preserved artifact — the dashboard JSON plus a snapshot — because once it
   stops publishing, the historical series leaves Prometheus retention.
5. **Remove** the panels only after that artifact is recorded and the operator
   accepts that the historical figures are no longer queryable.

Steps 1–3 are a dashboard change and need a Grafana GO. Step 4 is an exporter
retirement and needs its own GO. Nothing here is applied.

## 4. OpenRouter: both accounts are now read

Measured 2026-10-23, the two configured OpenRouter clients hold **different
credentials and belong to different financial accounts**:

| Account | Owner identity | Credit pool |
|---|---|---|
| `openrouter` | one `creator_user_id` | its own purchased credit |
| `openrouter_stephen` | a different `creator_user_id` and workspace | a separate credit pool |

`build_account_apis` no longer matches client names against a fixed list. It
decides the account surface from each client's **base URL**, so any configured
OpenRouter client gets a reader, whatever the operator calls it. Clients sharing
one credential collapse to one reader; clients with different credentials are
separate readers published under their own client name.

Where the provider states which account a key belongs to — OpenRouter reports
`creator_user_id` on `/key` — the collector compares those values. Two
credentials resolving to one account are read once and published once, so a
second key cannot double the account's reported money. If the reading label
fails, the other credential takes over rather than the account going unread.

**Still open:** the published V8 (version 4) was built before this change, so its
OpenRouter financial panels still describe the `openrouter` account only. They
remain correctly titled as covering the global key. Publishing a dashboard that
shows both accounts requires a separate Grafana GO.

## 5. Rolling-period accounting

Switchyard publishes three separate views of the same usage, and they answer
different questions:

| View | Series | Question |
|---|---|---|
| Cumulative | `switchyard_economics_tokens_total`, `_requests_total` | What has this process served since it started? |
| Rolling | `switchyard_economics_rolling_tokens{window}`, `_rolling_requests{window}`, `_rolling_provider_cost{window}` | What was served in the last hour or day, ending now? |
| Bangkok day | `switchyard_economics_calendar_tokens`, `_calendar_requests`, `_calendar_provider_cost` | What has been served since Bangkok midnight? |

Cumulative totals are never relabelled as rolling. A cumulative counter resets
when the process restarts; a rolling figure is bounded by the process's uptime,
because nothing before the process started was observed.

The rolling windows are computed from events Switchyard recorded itself, not
from `increase()` over a counter. `increase()` extrapolates from whatever samples
happen to exist, so a window containing a restart, or starting before the
process did, is reported as usage it never held. Keeping the events avoids that:
a one-hour figure means what was observed in the last hour.

### What is still not native

- **Month-to-date** is provider-reported for OpenRouter (the provider's own
  `usage_monthly`), and unavailable for DeepSeek. There is no Switchyard-side
  month accumulator.
- **Billing-account rollup across accounts** is not published as a single
  figure. Account families are per account by design, so a deployment-wide total
  is a query over accounts, not a reported number.
- **Rolling precision is one minute.** Events aggregate into one-minute buckets,
  so a rolling figure can differ from the instant-by-instant truth by at most the
  final partial minute. It is not presented as more precise than that.
- **Restart** resets the rolling windows. A window longer than the uptime reports
  only what the current process observed, which is the honest limit.

A dashboard that shows a rolling figure must label it with its window, and one
that shows a cumulative figure must not call it rolling.

## 6. Upstream model availability — not an economics concern

A withdrawn Minimax endpoint and earlier Nemotron failures are routing
availability problems. They reduce the served-usage series that economics
observes, but they are not caused by, and cannot be fixed by, anything here.
They are tracked as routing issues.

## 7. Definitions that are never interchanged

Five money-shaped facts appear on this dashboard. They are different amounts and
are never summed together or substituted for one another:

| Fact | Meaning | Source |
|---|---|---|
| Balance | Credit still available, as the provider reports it. | Provider balance endpoint. |
| Purchased credit | Credit the provider sold for money. | Provider account surface, where it exists. |
| Provider-reported spend | What the provider says it charged. | `provider_cost_total`, only when reported. |
| Historically accumulated spend | Lifetime inference from balance deltas. | Legacy exporter. |
| Estimated request cost | A Switchyard computation from a rate card. | Not published — no validated rate card. |

A provider that reports nothing produces no series. It is never recorded as a
zero: an unmeasured figure and a measured zero are different claims.

## 8. Verification recipe

Regenerate the dashboard and compare it to what is live:

```bash
python3 scripts/economics/build_economics_dashboard.py \
  --config scripts/economics/economics_dashboard.toml \
  --out /tmp/economics-v8.json
```

The build is reproducible: two runs of the same commit produce byte-identical
output, because panel ids are reset before each build.

When reading a dashboard back through the bounded read-only API, note that the
`dashboard` verb returns a **projection**, not the stored document. It truncates
every expression to 200 characters and omits `fieldConfig`, so units and
descriptions are invisible and any expression longer than 200 characters looks
cut. That truncation is display-only and also appears on earlier versions of the
same dashboard. Validate full expressions against the built request document, or
against the tool's own on-disk backup, never against the projection.

An expression that returns no series is not the same as one that fails. The
auth-failure and failed-account-read panels are legitimately empty when nothing
has failed; a broken query returns an error status instead. Distinguish the two
before reporting a defect.
