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

Switchyard's DeepSeek reader calls `GET https://api.deepseek.com/user/balance`
and keeps only `total_balance`. That endpoint reports a **level**, not a ledger.
From two readings of a level, a top-up and a day of usage are indistinguishable,
so Switchyard cannot derive:

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

### The one improvement that is cheap and native

DeepSeek's balance response also carries `granted_balance` and
`topped_up_balance` alongside `total_balance`. Switchyard discards both.

That means **remaining purchased credit and remaining granted credit are
available today from a call Switchyard already makes**, and are not published
only because the reader does not parse two fields it already receives. This
would be a small reader change — no new credential, no new endpoint, no new
collector.

It would **not** replace the four legacy panels, because a remaining balance is
not a cumulative purchase total. It would add a genuinely native figure next to
them, and it would let the dashboard separate "purchased credit still
available" from "credit purchased in total".

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

## 4. `openrouter_stephen`: served but not accounted

Switchyard attributes served usage to `provider="openrouter_stephen"` and
publishes its tokens and requests. Account-level credit, balance and usage are
**not** collected for that account.

The cause is explicit in `build_account_apis`: clients are matched by name
against `openrouter`, `deepseek` and `openai`/`openai_nt`, and any other name
falls through to no reader. A client named `openrouter_stephen` therefore
produces no account series.

The metric namespace also labels one account per provider, so a second
OpenRouter account cannot share the `openrouter` label. A correct fix gives it
its own label — matching how served usage already labels it — which is a
provider-identity change, not a copy-paste of the existing reader.

Consequence for reporting: the OpenRouter financial panels describe the
**global key only**. They must not be read as covering Stephen's account, and
they are titled to say so. Total OpenRouter spend across the deployment is
higher than the published figure by an unmeasured amount.

## 5. Rolling-period accounting is not complete

The usage panels are cumulative counters, not windowed totals. They show what
the running deployment has served since it started, and they reset when it
restarts. A "24h" figure built as `increase(...[24h])` over a counter is a
rolling estimate and is labelled rolling wherever it appears.

Not yet available:

- validated rolling 1-hour and 24-hour accounting as first-class values;
- a spend figure that survives a restart without a window boundary artefact;
- month-to-date accounting native to Switchyard (the OpenRouter month figure is
  provider-reported, not derived).

Cumulative totals must not be relabelled as rolling, and a rolling figure must
not be presented as cumulative.

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
