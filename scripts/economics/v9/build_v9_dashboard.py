# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build SWITCHYARD ECONOMICS & TOKEN ACCOUNTING V9.0.

WHAT THIS IS
  A second, separately versioned economics dashboard. It queries only families the
  switchyard-server binary emits itself, verified against the live exposition, and
  it refuses to build if any panel names anything else.

WHY IT EXISTS SEPARATELY FROM V8
  V8 was built beside an external exporter and inherited panels that read it. Those
  panels answered a different question — what some other process believed about
  usage — and sitting next to genuinely measured panels made the two look alike.
  V9 answers only what Switchyard itself measures. V8 stays published and untouched
  so an operator can roll back to it, and so the difference between the two is
  inspectable rather than asserted.

WHAT AN OPERATOR SHOULD BE ABLE TO ANSWER FROM THIS DASHBOARD
  * Which provider accounts still have money or allowance, and how stale that answer is.
  * What the deployment is spending, per provider and per model.
  * Where requests actually went — provider, model, route family, cost class.
  * Why upstream calls are failing, at the resolution the binary actually records.

WHAT IT DELIBERATELY DOES NOT DO
  * It does not estimate per-model cost. `record_estimated_cost` is declared in the
    binary and never called, so no estimate exists; provider-REPORTED cost is a
    different number and is labelled as such.
  * It does not present HTTP-status-derived attempt outcomes as semantic provider
    classification. `switchyard_upstream_attempts_total` labels an attempt with the
    status code it received, and its `outcome` is a function of that code alone.
    The semantic signal that does exist is the composition layer's `capability` and
    `reason` labels, and those are shown separately.
  * It does not attribute consumption to an agent or a domain. Switchyard records a
    route and a model, not a caller identity, so such a panel would be invention.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import build_economics_dashboard as bed  # noqa: E402
import native_contract  # noqa: E402
from build_economics_dashboard import (  # noqa: E402
    barchart,
    piechart,
    reset_ids,
    row,
    stat,
    table,
    target,
    text,
    timeseries,
)

# Live native families. Dots become underscores in the exposition.
TOK = "switchyard_economics_tokens_total"
REQ = "switchyard_economics_requests_total"
RTOK = "switchyard_economics_rolling_tokens"
RREQ = "switchyard_economics_rolling_requests"
RCOST = "switchyard_economics_rolling_provider_cost"
PCOST = "switchyard_economics_provider_cost_total"
CCOST = "switchyard_economics_calendar_provider_cost"
CREQ = "switchyard_economics_calendar_requests"
CTOK = "switchyard_economics_calendar_tokens"
CREDIT = "switchyard_economics_provider_credit"
USAGE = "switchyard_economics_provider_usage"
BALANCE = "switchyard_economics_provider_balance"
BALCOMP = "switchyard_economics_provider_balance_component"
ALLOW = "switchyard_economics_allowance_remaining_percent"
RESET = "switchyard_economics_allowance_reset_seconds"
FRESH = "switchyard_economics_account_freshness_seconds"
FETCH = "switchyard_economics_account_fetch_total"
FETCHDUR = "switchyard_economics_account_fetch_duration_seconds"
COVER = "switchyard_economics_rolling_coverage_seconds"
AUTHFAIL = "switchyard_economics_account_auth_failures_total"

ATTEMPTS = "switchyard_upstream_attempts_total"
CLIENTOUT = "switchyard_client_responses_total"
CALLS = "switchyard_llm_calls_total"
COMPATT = "switchyard_composition_attempt_total"
COMPCROSS = "switchyard_composition_cross_total"
RETRYREC = "switchyard_router_retry_recovered_total"
MODELERR = "switchyard_errors_total"
MODELREQ = "switchyard_requests_total"
PTOK = "switchyard_prompt_tokens_total"
CTOKC = "switchyard_completion_tokens_total"
RTOKC = "switchyard_reasoning_tokens_total"
CACHED = "switchyard_cached_tokens_total"
CACHECREATE = "switchyard_cache_creation_tokens_total"
RUNS = "switchyard_runs_total"
DECISIONS = "switchyard_decisions_total"


def build_dashboard(config):
    job = config["job"]
    jf = f'job="{job}"'
    bed.DS = {"type": config["datasource_type"], "uid": config["datasource_uid"]}
    reset_ids()
    panels = []
    y = 0

    def add(panel):
        panels.append(panel)

    def adv(step):
        nonlocal y
        y += step

    # ---------------------------------------------------------------- overview
    add(row("1  EXECUTIVE OVERVIEW", y))
    adv(1)

    add(
        stat(
            "PROVIDER-REPORTED SPEND TODAY",
            f'sum({CCOST}{{{jf},currency="USD",component="provider_reported"}})',
            "What providers reported they charged during the current provider calendar day, in "
            "USD. This is a reported figure, not an estimate: Switchyard records the cost the "
            "provider returns. A provider that reports no per-request cost leaves this sum "
            "lower, not wrong. Cost class `subscription` contributes nothing here by design.",
            0,
            y,
            unit="currencyUSD",
            decimals=4,
        )
    )
    add(
        stat(
            "PROVIDER-REPORTED SPEND, LAST 24H",
            f'sum({RCOST}{{{jf},window="24h",currency="USD",component="provider_reported"}})',
            "Provider-reported charge over Switchyard's own rolling 24h window, USD. The window "
            "is computed by the binary from observed events, so it is not an increase() over "
            "samples and does not undercount around a restart.",
            6,
            y,
            unit="currencyUSD",
            decimals=4,
        )
    )
    add(
        stat(
            "REQUESTS TODAY",
            f'sum({CREQ}{{{jf},component="request"}})',
            "Requests served during the current provider calendar day. A composed request counts "
            "once against the route the caller addressed, not once per child route.",
            12,
            y,
            unit="short",
        )
    )
    add(
        stat(
            "TOKENS TODAY",
            f"sum({CTOK}{{{jf}}})",
            "All token components recorded today, summed. Components are mutually exclusive in "
            "this series — prompt, completion, cached, cache_creation, reasoning — so the sum is "
            "a component total, not a billing figure. Read the token section for the split.",
            18,
            y,
            unit="short",
        )
    )
    adv(5)

    add(
        stat(
            "UPSTREAM ATTEMPT ERROR RATE",
            f'sum({ATTEMPTS}{{{jf},outcome=~"retryable_error|other_error"}}) / '
            f'sum({ATTEMPTS}{{{jf}}})',
            "Share of UPSTREAM ATTEMPTS that did not return 2xx. This is deliberately not a "
            "request failure rate: one client request that retried and then succeeded counts "
            "here as one failed attempt and one served request. Derived from the HTTP status "
            "code, so it cannot distinguish a rate limit from an exhausted balance — the "
            "provider-error section shows what semantic detail the binary does record.",
            0,
            y,
            unit="percentunit",
            decimals=2,
        )
    )
    add(
        stat(
            "FINAL CLIENT ERROR RATE",
            f'sum({CLIENTOUT}{{{jf},outcome=~"retryable_error|other_error"}}) / '
            f"sum({CLIENTOUT}{{{jf}}})",
            "Share of responses delivered to callers that ended in an error, after retries and "
            "fallback have already been applied. This is the number a user experiences. It is "
            "much lower than the attempt error rate precisely when recovery is working.",
            6,
            y,
            unit="percentunit",
            decimals=2,
        )
    )
    add(
        stat(
            "RETRY RECOVERIES",
            f"sum({RETRYREC}{{{jf}}})",
            "Requests that failed an attempt and were served by a later attempt or another "
            "candidate. Every one of these is an upstream error that never reached a caller, "
            "which is why the two rates above differ.",
            12,
            y,
            unit="short",
        )
    )
    add(
        stat(
            "STALEST ACCOUNT READING",
            f"max({FRESH}{{{jf}}})",
            "Age in seconds of the oldest provider-account reading. This bounds how much any "
            "balance on this dashboard should be trusted: a large number means the figures below "
            "describe the past, not the present. It is a measurement about freshness, not about "
            "whether the provider is reachable.",
            18,
            y,
            unit="s",
            decimals=0,
        )
    )
    adv(5)

    add(
        timeseries(
            "PROVIDER-REPORTED SPEND, ROLLING 24H, BY PROVIDER",
            [
                target(
                    f"sum by (provider) ({RCOST}{{{jf},window=\"24h\",currency=\"USD\","
                    'component="provider_reported"})',
                    "{{provider}}",
                )
            ],
            "Each provider's reported charge over the binary's rolling 24h window. Providers are "
            "separate accounts with separate currency, so they are never summed into one line "
            "here; the USD-only selection keeps the axis one currency.",
            0,
            y,
        )
    )
    add(
        timeseries(
            "REQUESTS, ROLLING 24H, BY PROVIDER",
            [
                target(
                    f'sum by (provider) ({RREQ}{{{jf},window="24h",component="request"}})',
                    "{{provider}}",
                )
            ],
            "Request volume per provider over the rolling 24h window. A provider with no line is "
            "not necessarily offline: it may simply not have been routed to. Provider health is a "
            "separate question from provider traffic and is answered by the error section.",
            12,
            y,
        )
    )
    adv(9)

    # ------------------------------------------------------------- accounts
    add(row("2  PROVIDER ACCOUNTS AND FINANCIAL AVAILABILITY", y))
    adv(1)

    add(
        stat(
            "OPENROUTER CREDIT (ACCOUNT)",
            f'sum({CREDIT}{{{jf},provider="openrouter",scope="account",currency="USD"}})',
            "Remaining account-level credit on the primary OpenRouter account, USD. Selective on "
            "`scope=\"account\"` on purpose: the same money is also published under `scope=\"key\"` "
            "for per-key views, and adding both would double count one balance.",
            0,
            y,
            unit="currencyUSD",
            decimals=2,
        )
    )
    add(
        stat(
            "OPENROUTER CREDIT (SECONDARY ACCOUNT)",
            f'sum({CREDIT}{{{jf},provider="openrouter_stephen",scope="account",currency="USD"}})',
            "Remaining credit on the second OpenRouter account. Shown separately rather than "
            "combined because it is a distinct credential with distinct exhaustion behaviour: "
            "one account running dry does not stop the other from serving.",
            6,
            y,
            unit="currencyUSD",
            decimals=2,
        )
    )
    add(
        stat(
            "DEEPSEEK BALANCE (CNY)",
            f'sum({BALANCE}{{{jf},provider="deepseek",scope="account",currency="CNY"}})',
            "DeepSeek reports its balance in CNY and also publishes a zero-valued USD series. The "
            "currency is pinned so the two are never added: one of them is real money and the "
            "other is the absence of a USD balance.",
            12,
            y,
            unit="currencyCNY",
            decimals=2,
        )
    )
    add(
        stat(
            "OPENAI ALLOWANCE REMAINING (PRIMARY POOL)",
            f'max({ALLOW}{{{jf},provider="openai",pool="primary"}})',
            "Remaining percentage of the primary subscription allowance. This is an allowance, "
            "not a balance: running it to zero is a capacity signal, not a billing event, and it "
            "resets on its own schedule rather than when someone pays.",
            18,
            y,
            unit="percent",
            decimals=0,
        )
    )
    adv(5)

    add(
        table(
            "ACCOUNT STATE — ONE ROW PER PROVIDER, CURRENCY AND SCOPE",
            [
                target(
                    f"{CREDIT}{{{jf}}}",
                    "credit {{provider}} {{currency}} {{scope}} {{period}}",
                    instant=True,
                    ref="A",
                ),
                target(
                    f"{BALANCE}{{{jf}}}",
                    "balance {{provider}} {{currency}} {{scope}} {{period}}",
                    instant=True,
                    ref="B",
                ),
                target(
                    f'{USAGE}{{{jf}}}',
                    "usage {{provider}} {{currency}} {{scope}} {{period}}",
                    instant=True,
                    ref="C",
                ),
            ],
            "Every financial figure the binary publishes, with the labels that make it "
            "interpretable. Credit, balance and usage are three different questions and are not "
            "interchangeable: credit is what has been bought and not yet spent, balance is the "
            "provider's own remaining figure, and usage is what has been drawn in a period. Rows "
            "are not summed, because they are in different currencies and describe different "
            "facts.",
            0,
            y,
            h=9,
            unit="currencyUSD",
        )
    )
    add(
        table(
            "ACCOUNT READING HEALTH",
            [
                target(
                    f"sum by (provider, outcome) ({FETCH}{{{jf}}})",
                    "{{provider}} {{outcome}}",
                    instant=True,
                    ref="A",
                ),
                target(
                    f"max by (provider) ({FETCHDUR}{{{jf}}})",
                    "duration {{provider}}",
                    instant=True,
                    ref="B",
                ),
                target(
                    f"max by (provider) ({FRESH}{{{jf}}})",
                    "age {{provider}}",
                    instant=True,
                    ref="C",
                ),
            ],
            "How the balance figures above were obtained. A provider whose reads are failing "
            "still shows its last known balance, which will look current while being stale; this "
            "table is how that is detected. A non-ok outcome here means the financial figures for "
            "that provider are history, not measurement.",
            12,
            y,
            h=9,
            unit="short",
        )
    )
    adv(9)

    add(
        timeseries(
            "ALLOWANCE REMAINING BY POOL",
            [
                target(
                    f"{ALLOW}{{{jf},provider=\"openai\"}}",
                    "{{pool}}",
                )
            ],
            "Subscription allowance remaining, per pool, over time. A falling line inside a fixed "
            "reset period is ordinary consumption; a line that hits zero and stays there is a "
            "capacity wall that routing must route around.",
            0,
            y,
        )
    )
    add(
        timeseries(
            "ACCOUNT READING AGE BY PROVIDER",
            [
                target(
                    f"max by (provider) ({FRESH}{{{jf}}})",
                    "{{provider}}",
                )
            ],
            "Freshness of each provider's financial data. This rises between refreshes and drops "
            "when a read succeeds, so the sawtooth is expected. A line that only rises means the "
            "provider's account reads have stopped, and every balance for it should be treated as "
            "an old observation.",
            12,
            y,
        )
    )
    adv(9)

    add(
        stat(
            "ALLOWANCE RESET (PRIMARY POOL)",
            f'max({RESET}{{{jf},provider="openai",pool="primary"}})',
            "Seconds until the primary allowance resets, as the provider reported it. Useful for "
            "deciding whether a capacity wall is minutes or hours away.",
            0,
            y,
            unit="s",
            decimals=0,
        )
    )
    add(
        stat(
            "ROLLING WINDOW COVERAGE (24H)",
            f'max({COVER}{{{jf},window="24h"}})',
            "Seconds of real observation behind the rolling 24h figures. The windows are computed "
            "from observed events, so after a restart this is smaller than 86400 until the "
            "deployment has been up a full day. While it is smaller, the 24h numbers are a "
            "partial window and should not be compared with a full one.",
            6,
            y,
            unit="s",
            decimals=0,
        )
    )
    add(
        stat(
            "AUTH FAILURES ON ACCOUNT READS",
            f"sum({AUTHFAIL}{{{jf}}})",
            "Times a provider rejected Switchyard's own credential while reading account state. "
            "This family is event driven: it appears only after the first occurrence, so an "
            "absent panel means no such failure has been recorded in this process, not that the "
            "metric is missing. Zero here is the expected state.",
            12,
            y,
            unit="short",
        )
    )
    add(
        stat(
            "ACCOUNT READ FAILURES",
            f'sum({FETCH}{{{jf},outcome!="ok"}})',
            "Account reads that did not succeed, for any reason including a credential rejection. "
            "Distinct from the panel beside it: a network failure and a rejected credential both "
            "land here, and only the latter lands there.",
            18,
            y,
            unit="short",
        )
    )
    adv(5)

    # ---------------------------------------------------------------- tokens
    add(row("3  TOKEN ECONOMICS", y))
    adv(1)

    add(
        stat(
            "PROMPT TOKENS",
            f"sum({PTOK}{{{jf}}})",
            "Input tokens as the provider counted them, since the deployment started. This total "
            "already contains the cached and cache-created tokens, so those components are shown "
            "beside it and never added to it.",
            0,
            y,
            unit="short",
        )
    )
    add(
        stat(
            "COMPLETION TOKENS",
            f"sum({CTOKC}{{{jf}}})",
            "Output tokens as provider-reported, since the deployment started. Reasoning is "
            "published separately by the providers that report it and is not added here: the "
            "recorded totals stay exactly as each provider reported them.",
            6,
            y,
            unit="short",
        )
    )
    add(
        stat(
            "REASONING TOKENS",
            f"sum({RTOKC}{{{jf}}})",
            "Reasoning tokens, where the provider reports them. Its relationship to completion "
            "varies by model — measured across the routing log, some models report reasoning "
            "above completion — so no arithmetic relationship is asserted anywhere on this "
            "dashboard. Absent means the provider did not report it, which is not a measured zero.",
            12,
            y,
            unit="short",
        )
    )
    add(
        stat(
            "CACHED PROMPT TOKENS",
            f"sum({CACHED}{{{jf}}})",
            "Prompt tokens served from the provider's cache. Comparing this with the prompt total "
            "is how much of the input was not re-sent at full price.",
            18,
            y,
            unit="short",
        )
    )
    adv(5)

    add(
        barchart(
            "PROMPT TOKENS BY MODEL",
            [target(f"sum by (model) ({PTOK}{{{jf}}})", "{{model}}")],
            "Input volume per model. This is the cost driver that scales with conversation "
            "length, and the one a routing decision can most directly move.",
            0,
            y,
            unit="short",
        )
    )
    add(
        barchart(
            "OUTPUT TOKENS BY MODEL — COMPLETION AND REASONING",
            [
                target(f'sum by (model) ({CTOKC}{{{jf}}})', "completion {{model}}"),
                target(f'sum by (model) ({RTOKC}{{{jf}}})', "reasoning {{model}}"),
            ],
            "Output volume per model, with reasoning alongside rather than inside completion. "
            "Where both bars exist for a model, the split shows how much of the work was "
            "deliberation; where reasoning is missing, that provider does not report it.",
            12,
            y,
            unit="short",
        )
    )
    adv(9)

    add(
        piechart(
            "TOKEN COMPONENT MIX TODAY",
            [target(f"sum by (component) ({CTOK}{{{jf}}})", "{{component}}")],
            "How today's recorded tokens divide between prompt, completion, cached, cache_creation "
            "and reasoning. Components come from one series so the whole is the sum of its parts "
            "as the binary recorded them.",
            0,
            y,
            unit="short",
        )
    )
    add(
        timeseries(
            "TOKENS, ROLLING 24H, BY COMPONENT",
            [
                target(
                    f'sum by (component) ({RTOK}{{{jf},window="24h"}})',
                    "{{component}}",
                )
            ],
            "Rolling 24h tokens by component. A cached share that grows while prompt stays flat "
            "is the usual sign that prompt reuse is working, and is the difference between an "
            "expensive conversation and a cheap one.",
            8,
            y,
        )
    )
    adv(9)

    # --------------------------------------------------------------- traffic
    add(row("4  OBSERVED TRAFFIC MIX", y))
    adv(1)

    add(
        barchart(
            "REQUESTS BY PROVIDER",
            [target(f'sum by (provider) ({REQ}{{{jf},component="request"}})', "{{provider}}")],
            "Where requests were actually served, since the deployment started. This is observed "
            "routing, not configuration: it shows which provider the routing layer chose, which "
            "can differ from what a route's candidate list might suggest.",
            0,
            y,
            w=8,
        )
    )
    add(
        barchart(
            "REQUESTS BY MODEL",
            [target(f'sum by (model) ({REQ}{{{jf},component="request"}})', "{{model}}")],
            "Requests per model. Read together with the error table below: a model carrying a "
            "large share of traffic and a high error rate is an active routing problem, while the "
            "same error rate on a lightly used model is not.",
            8,
            y,
            w=8,
        )
    )
    add(
        barchart(
            "REQUESTS BY ROUTE FAMILY",
            [
                target(
                    f'sum by (route_family) ({REQ}{{{jf},component="request"}})',
                    "{{route_family}}",
                )
            ],
            "Requests by the kind of route the caller addressed — direct, primitive, tier or a "
            "composite. This distinguishes deliberate direct calls from traffic that arrived "
            "through a routing layer.",
            16,
            y,
            w=8,
        )
    )
    adv(9)

    add(
        piechart(
            "COST CLASS MIX",
            [target(f"sum by (cost_class) ({REQ}{{{jf}}})", "{{cost_class}}")],
            "Requests by whether serving them was metered, free, or drawn from a subscription "
            "allowance. This is the economic character of the traffic: `subscription` requests "
            "cost nothing per call and `metered` ones do, so the mix explains what the spend "
            "figures are made of.",
            0,
            y,
            w=8,
        )
    )
    add(
        timeseries(
            "REQUESTS, ROLLING 24H, BY MODEL",
            [target(f'sum by (model) ({RREQ}{{{jf},window="24h"}})', "{{model}}")],
            "Per-model request volume over the rolling window. Use this to see a shift in routing "
            "as it happens, rather than as a total that hides when it changed.",
            8,
            y,
            w=16,
        )
    )
    adv(9)

    add(
        table(
            "ROUTE × MODEL — VOLUME, TOKENS AND REPORTED COST",
            [
                target(
                    f'sum by (route, model, provider) ({REQ}{{{jf},component="request"}})',
                    "{{route}} | {{model}} | {{provider}}",
                    instant=True,
                    ref="A",
                ),
                target(
                    f'sum by (route, model) ({RTOK}{{{jf},window="24h"}})',
                    "tokens24h {{route}} | {{model}}",
                    instant=True,
                    ref="B",
                ),
                target(
                    f'sum by (route, model) ({RCOST}{{{jf},window="24h",currency="USD",'
                    'component="provider_reported"})',
                    "reported_usd_24h {{route}} | {{model}}",
                    instant=True,
                    ref="C",
                ),
            ],
            "The routing decision and its consequence, joined on route and model. Reported cost "
            "is a 24h window rather than a running total because the cumulative cost counter "
            "carries no currency label, so it cannot be safely filtered to one currency; the "
            "windowed family can. It is absent for subscription traffic, which is a property of "
            "the provider rather than a gap in the data.",
            0,
            y,
            h=10,
        )
    )
    add(
        timeseries(
            "REPORTED COST, ROLLING 24H, BY MODEL",
            [
                target(
                    f'sum by (model) ({RCOST}{{{jf},window="24h",currency="USD",'
                    f'component="provider_reported"}})',
                    "{{model}}",
                )
            ],
            "Which model the money is going to. Only providers that report per-request cost "
            "appear; subscription and free traffic is absent by design, not missing.",
            12,
            y,
            h=10,
        )
    )
    adv(10)

    # ---------------------------------------------------------------- errors
    add(row("5  PROVIDER ERRORS AND ECONOMIC RISK", y))
    adv(1)

    add(
        stat(
            "UPSTREAM ATTEMPTS FAILED",
            f'sum({ATTEMPTS}{{{jf},outcome!="ok"}})',
            "Upstream attempts that did not return 2xx. Attempts, not requests: a request that "
            "retried twice and then succeeded contributes two here and zero to the panel beside "
            "it. Confusing the two is how a healthy deployment looks broken.",
            0,
            y,
        )
    )
    add(
        stat(
            "ATTEMPTS RECOVERED BY RETRY",
            f"sum({RETRYREC}{{{jf}}})",
            "Failed attempts that a later attempt served. This is the count of upstream errors "
            "that never became a caller-visible failure.",
            6,
            y,
        )
    )
    add(
        stat(
            "REQUESTS THAT REACHED A CALLER AS ERRORS",
            f'sum({CLIENTOUT}{{{jf},outcome=~"retryable_error|other_error"}})',
            "Responses delivered to callers that ended in an error, after every recovery "
            "opportunity was used. This is the only error count on this dashboard that a user "
            "would recognise as their own failed request.",
            12,
            y,
        )
    )
    add(
        stat(
            "ACCOUNT-WIDE QUOTA BLOCKS",
            f'sum({COMPATT}{{{jf},capability="account_quota"}})',
            "Times the router found an entire account's allowance exhausted. This is the "
            "semantic signal the binary does record: it is decided from the provider's structured "
            "error, not from the status code, and it is what stops a request being retried "
            "against a credential already known to be empty.",
            18,
            y,
        )
    )
    adv(5)

    add(
        barchart(
            "UPSTREAM ATTEMPTS BY STATUS AND HTTP-DERIVED OUTCOME",
            [target(f"sum by (code, outcome) ({ATTEMPTS}{{{jf}}})", "{{code}} {{outcome}}")],
            "Attempts grouped by the status code received and the outcome the binary derived from "
            "that code alone. Read this as HTTP evidence, not as a diagnosis: `429 retryable_error` "
            "covers both a genuine rate limit and an exhausted balance, because the status alone "
            "cannot tell them apart. The composition panels below carry the semantic distinction.",
            0,
            y,
        )
    )
    add(
        timeseries(
            "UPSTREAM ATTEMPTS BY STATUS OVER TIME",
            [target(f"sum by (code) (increase({ATTEMPTS}{{{jf}}}[15m]))", "{{code}}")],
            "Attempt rate per status code. A steady 429 line is pacing pressure that retries can "
            "absorb; a 401 or 403 line is a credential or policy problem that cannot be retried "
            "away and needs an operator.",
            12,
            y,
        )
    )
    adv(9)

    add(
        barchart(
            "COMPOSITION OUTCOMES BY CAPABILITY",
            [
                target(
                    f"sum by (capability, outcome) ({COMPATT}{{{jf}}})",
                    "{{capability}} {{outcome}}",
                )
            ],
            "What the composition layer decided, classified by the capability it was handling. "
            "This is the binary's own semantic classification and is the most trustworthy error "
            "breakdown available: `credential_quota` means a credential's allowance was known "
            "exhausted before a call was spent, and `account_quota` means an account-wide limit "
            "was hit.",
            0,
            y,
        )
    )
    add(
        table(
            "ROUTE CROSSINGS BY REASON",
            [
                target(
                    f"sum by (parent_route, from_child, to_child, reason) ({COMPCROSS}{{{jf}}})",
                    "{{parent_route}}: {{from_child}} → {{to_child}}",
                    instant=True,
                    ref="A",
                )
            ],
            "Each row is a request that left one child route for another, with the classified "
            "reason. This is the recovery path made visible: a large `account_quota` count against "
            "one child means that child's credential is routinely exhausted and the route is "
            "paying for a crossing it could avoid by ordering candidates differently.",
            12,
            y,
            unit="short",
        )
    )
    adv(9)

    add(
        table(
            "ERROR RATE BY MODEL",
            [
                target(
                    f'sum by (selected_model) ({CALLS}{{{jf},outcome="error"}}) / '
                    f"clamp_min(sum by (selected_model) ({CALLS}{{{jf}}}), 0.000001)",
                    "{{selected_model}}",
                    instant=True,
                    ref="A",
                ),
                target(
                    f"sum by (selected_model) ({CALLS}{{{jf}}})",
                    "calls {{selected_model}}",
                    instant=True,
                    ref="B",
                ),
            ],
            "Per-model failure rate, with its denominator shown so a 100% rate on two calls is not "
            "mistaken for a broken model. `clamp_min` keeps the division defined when a model has "
            "no calls at all; it does not manufacture a zero result for a model that has not been "
            "used.",
            0,
            y,
            h=9,
            unit="percentunit",
        )
    )
    add(
        timeseries(
            "ERROR RATE BY MODEL OVER TIME",
            [
                target(
                    f'sum by (selected_model) (increase({CALLS}{{{jf},outcome="error"}}[1h])) / '
                    f"clamp_min(sum by (selected_model) (increase({CALLS}{{{jf}}}[1h])), 0.000001)",
                    "{{selected_model}}",
                )
            ],
            "The same rate as a trend, which is what distinguishes a model that has always been "
            "unreliable from one that started failing an hour ago. A rate computed over the "
            "cumulative counters would hide the change entirely.",
            12,
            y,
            h=9,
            unit="percentunit",
        )
    )
    adv(9)

    add(
        timeseries(
            "CLIENT-VISIBLE OUTCOMES",
            [
                target(
                    f"sum by (outcome) (increase({CLIENTOUT}{{{jf}}}[15m]))",
                    "{{outcome}}",
                )
            ],
            "What callers actually received. `ok` beside a smaller `retryable_error` is the "
            "signature of recovery working; `other_error` growing without a matching upstream "
            "explanation points at a local failure — a translation or configuration problem — "
            "rather than a provider one.",
            0,
            y,
        )
    )
    add(
        barchart(
            "RUN OUTCOMES BY ALGORITHM",
            [target(f"sum by (algorithm, outcome) ({RUNS}{{{jf}}})", "{{algorithm}} {{outcome}}")],
            "Routing runs and how they ended, grouped by the algorithm that selected. A run can "
            "end differently from its upstream attempts, because the algorithm may fall back.",
            12,
            y,
        )
    )
    adv(9)

    # ----------------------------------------------------------- disclosures
    add(row("6  COVERAGE, MEASUREMENT BASIS AND DISCLOSURES", y))
    adv(1)

    add(
        text(
            "WHAT IS MEASURED, AND WHAT IS NOT",
            "**Measured.** Request counts, token components, provider-reported cost, provider "
            "account balances, credits, usage and allowances, per-attempt HTTP status, "
            "composition decisions and crossings, and client-visible outcomes. These come from "
            "`switchyard-server`'s own Prometheus exposition; the dashboard refuses to query "
            "anything else.\n\n"
            "**Not measured, and therefore absent.** Per-model *estimated* cost. The binary "
            "declares `switchyard.economics_estimated_cost` and never calls it, so no series "
            "exists and no panel presents one. Provider-*reported* cost is shown instead and is "
            "labelled as the provider's own figure.\n\n"
            "**Not attributed.** Per-agent, per-domain or per-caller consumption. Switchyard "
            "records the route, model and provider that served a request; it does not record who "
            "asked. Any such breakdown would be invented, so there is none.",
            0,
            y,
            w=12,
            h=8,
        )
    )
    add(
        text(
            "WHY THE TWO ERROR RATES DIFFER, AND WHY NEITHER IS A DIAGNOSIS",
            "`switchyard_upstream_attempts_total` records one increment per attempt, labelled with "
            "the HTTP status received. Its `outcome` label is a **function of that status alone** — "
            "408, 429 and 5xx become `retryable_error`, everything else `other_error`. It is "
            "honest about what it saw and silent about why.\n\n"
            "That matters for two cases the providers document explicitly. A `429` may be a rate "
            "limit that clears in seconds, or an exhausted balance that no retry can fix; the "
            "status is identical. An OpenRouter `500 server_error` may be masked infrastructure "
            "trouble, or a permanent authentication failure reported only in the top-level "
            "`error_type`. Neither distinction is recoverable from the status code.\n\n"
            "The semantic classification that *does* exist natively is the composition layer: its "
            "`capability` and `reason` labels are decided from the provider's structured error. "
            "Section 5 shows both facts side by side rather than presenting one as the other.",
            12,
            y,
            w=12,
            h=8,
        )
    )
    adv(8)

    add(
        text(
            "ACCURACY OF THE MONEY FIGURES",
            "Three facts about the financial series are worth stating plainly, because each is a "
            "way a cost dashboard misleads without being wrong.\n\n"
            "**Currency is a label, not a constant.** The same provider can publish a real "
            "balance in one currency and a zero-valued series in another — DeepSeek does exactly "
            "that. Every money query here pins a currency, so a zero in an unused currency is "
            "never mistaken for an exhausted account.\n\n"
            "**Scope is a label, and it duplicates.** One balance is published under "
            "`scope=\"account\"` and again under `scope=\"key\"`. They describe the same money "
            "from two vantage points; adding them doubles it. This dashboard pins one scope per "
            "panel, or names the scope in the legend when a panel deliberately shows both.\n\n"
            "**One family cannot be currency-filtered.** The cumulative "
            "`provider_cost_total` carries no currency label, unlike its calendar and rolling "
            "counterparts. It is therefore not used for any money figure here; the windowed "
            "families are, so spend is always attributable to a single currency.",
            0,
            y,
            w=12,
            h=9,
        )
    )

    add(
        text(
            "ABSENCE HAS FOUR DIFFERENT MEANINGS HERE",
            "**Zero.** A counter that exists and has not incremented. The traffic or spend is "
            "genuinely zero.\n\n"
            "**No event observed.** A family that only appears once its event happens — account "
            "authentication failures are the example on this dashboard. An empty panel means the "
            "event has not occurred in this process, not that measurement is broken.\n\n"
            "**Partial window.** The rolling windows are computed by the binary from events it "
            "observed. After a restart, `ROLLING WINDOW COVERAGE` drops below 86400s and the 24h "
            "figures describe less than a day. The coverage panel exists so that is visible "
            "rather than assumed away.\n\n"
            "**Not instrumented.** Where a measurement does not exist at all — estimated cost, "
            "per-caller attribution, partial-stream failure phase — no panel is drawn. An empty "
            "panel would imply the number is available and zero, which is a stronger and falser "
            "claim.",
            12,
            y,
            w=12,
            h=9,
        )
    )
    adv(9)

    add(
        text(
            "READING A PROVIDER CORRECTLY",
            "A provider with **zero calls** is not offline; it simply was not routed to. A "
            "provider with **exhausted credit** is not technically unavailable; it is financially "
            "unavailable, and the difference decides whether the fix is a top-up or an "
            "investigation. A **successfully retried** upstream error is not a failed request. "
            "And a **stale balance** is an old observation, not a current one — the account "
            "reading age panel is what separates the two.\n\n"
            "These are different operational causes with different routing decisions and "
            "different economic consequences, and this dashboard keeps them apart rather than "
            "collapsing them into one error count.",
            12,
            y,
            w=12,
            h=9,
        )
    )
    adv(9)

    dashboard = {
        "annotations": {"list": []},
        "editable": True,
        "graphTooltip": 1,
        "links": [
            {
                "asDropdown": False,
                "icon": "external link",
                "tags": [],
                "targetBlank": False,
                "title": "Switchyard routing (V7)",
                "type": "link",
                "url": f"/d/{config['routing_uid']}",
            },
            {
                "asDropdown": False,
                "icon": "external link",
                "tags": [],
                "targetBlank": False,
                "title": "Economics V8 (previous)",
                "type": "link",
                "url": f"/d/{config['economics_v8_uid']}",
            },
        ],
        "panels": panels,
        "refresh": "1m",
        "schemaVersion": 39,
        "tags": ["switchyard", "economics", "native"],
        "templating": {"list": []},
        "time": {"from": "now-24h", "to": "now"},
        "timezone": "browser",
        "title": config["title"],
        "uid": config["uid"],
        "version": 1,
    }

    check_native_contract(dashboard)
    return dashboard


def check_native_contract(dashboard):
    """Refuses to produce a dashboard that queries anything non-native.

    This runs at build time as well as in the test, so an expression cannot reach a
    published document by way of a path the test does not exercise. The check is on
    the rendered document rather than on the source that produced it: what matters
    is the PromQL that will actually run.
    """
    problems = []
    for panel in dashboard["panels"]:
        for target_ in panel.get("targets", []):
            expr = target_.get("expr", "")
            for problem in native_contract.violations(expr):
                problems.append(f"panel {panel.get('title')!r} [{target_.get('refId')}]: {problem}")
            lowered = expr.lower()
            for marker in ("localclaw_", "job=\"switchyard\"", "hermes", "openclaw", "dsh_"):
                if marker in lowered:
                    problems.append(
                        f"panel {panel.get('title')!r}: names a retired source ({marker})"
                    )
    if problems:
        raise SystemExit(
            "dashboard rejected: non-native metric references\n  " + "\n  ".join(problems)
        )


def build_request(config, expected_version=None):
    request = {
        "op": "create" if expected_version is None else "update",
        "uid": config["uid"],
        "message": config["message"],
        "folder_uid": None,
        "dashboard": build_dashboard(config),
    }
    if expected_version is not None:
        request["expected_version"] = expected_version
    return request


def load_config(path):
    import tomllib

    with open(path, "rb") as handle:
        return tomllib.load(handle)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build the V9.0 economics dashboard.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--expected-version", type=int, default=None)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    request = build_request(config, expected_version=args.expected_version)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(request, handle, indent=1)
    panels = request["dashboard"]["panels"]
    queries = sum(len(panel.get("targets", [])) for panel in panels)
    print(f"wrote {args.out}: {len(panels)} panels, {queries} queries, op={request['op']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
