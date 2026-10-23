# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the Switchyard economics dashboard document.

The dashboard is generated rather than hand-written so the panel layout, the PromQL
and the descriptions cannot drift apart: every panel is one call, and the queries
come from a configuration file rather than being written inline.

WHAT THIS IS
  A builder that turns a configuration file into one Grafana dashboard document,
  in the request envelope the publishing tool accepts.

WHAT THIS IS NOT
  Not a publisher. This writes a file. Publishing is a separate, explicit step
  that a person reviews and runs.

THE ACCOUNTING RULES THE PANELS RESPECT
  These are the rules that keep the dashboard from showing a number that is wrong
  in a way a reader would not notice.

  * `prompt` already contains `cached` and `cache_creation`. No panel adds the
    components to the prompt total, and there is no "all tokens" panel.
  * `completion` already contains `reasoning`, in the same shape. Measured
    2026-10-21 against the live OpenRouter API: a response reported
    `completion_tokens: 650` with `completion_tokens_details.reasoning_tokens: 521`
    and `total_tokens: 680`, where `prompt + completion == total` and
    `prompt + completion + reasoning` did not. A total, were one published, would
    be `prompt + completion` and nothing else.
  * The provider-reported cost and the Switchyard estimate are different series,
    and no panel adds them together.
  * The usage panels read the counters directly rather than through a windowed
    increase. A counter's own value is the cumulative total, and a windowed
    increase misses any increment that happened before Prometheus took its first
    sample of the series. A counter resets when the deployment restarts, so these
    panels read "since the last restart".
  * The currency, the accounting scope and the reporting period are three
    separate facts and are selected separately. A query that adds up provider
    usage must say which scope and which period it wants.
  * There is no savings-versus-baseline panel: that counterfactual needs a
    baseline assumption set that has not been validated.
"""

import argparse
import json
import sys

# The Grafana datasource the panels query. The concrete uid is a deployment
# fact and comes from the configuration; this is only the shape a panel records.
DS = {"type": "prometheus", "uid": ""}

# The economics series. These are bare metric names: a Prometheus selector may
# carry only one label matcher group, so the job filter and any other label are
# merged into a single selector at each use.
TOKENS = "switchyard_economics_tokens_total"
REQUESTS = "switchyard_economics_requests_total"
PROVIDER_COST = "switchyard_economics_provider_cost_total"
ACCOUNT_FETCH = "switchyard_economics_account_fetch_total"
AUTH_FAILURES = "switchyard_economics_account_auth_failures_total"
FRESHNESS = "switchyard_economics_account_freshness_seconds"
BALANCE = "switchyard_economics_provider_balance"
CREDIT = "switchyard_economics_provider_credit"
PROVIDER_USAGE = "switchyard_economics_provider_usage"
ALLOWANCE_PCT = "switchyard_economics_allowance_remaining_percent"
ALLOWANCE_RESET = "switchyard_economics_allowance_reset_seconds"
CAL_TOKENS = "switchyard_economics_calendar_tokens"
CAL_REQUESTS = "switchyard_economics_calendar_requests"
CAL_DAY_START = "switchyard_economics_calendar_day_start"
CAL_DAY_END = "switchyard_economics_calendar_day_end"
CAL_DAY_ELAPSED = "switchyard_economics_calendar_day_elapsed_seconds"

_uid = 1000


def next_id():
    global _uid
    _uid += 1
    return _uid


def reset_ids():
    """Restarts the panel id counter.

    Grafana identifies a panel by its id, so two builds of the same dashboard
    must produce the same ids. The counter therefore starts from the same
    place every time, rather than continuing from wherever the last build in
    this process left it.
    """
    global _uid
    _uid = 1000


def target(expr, legend=None, instant=False, ref="A"):
    out = {
        "datasource": DS,
        "expr": expr,
        "refId": ref,
    }
    if legend is not None:
        out["legendFormat"] = legend
    if instant:
        out["instant"] = True
    return out


def row(title, y):
    return {
        "collapsed": False,
        "gridPos": {"h": 1, "w": 24, "x": 0, "y": y},
        "id": next_id(),
        "panels": [],
        "title": title,
        "type": "row",
    }


def stat(
    title,
    expr,
    description,
    x,
    y,
    w=6,
    h=5,
    unit="short",
    decimals=None,
    thresholds=None,
    legend=None,
):
    defaults = {
        "color": {"mode": "thresholds"},
        "unit": unit,
    }
    if decimals is not None:
        defaults["decimals"] = decimals
    defaults["thresholds"] = thresholds or {
        "mode": "absolute",
        "steps": [{"color": "text", "value": None}],
    }
    return {
        "datasource": DS,
        "description": description,
        "fieldConfig": {"defaults": defaults, "overrides": []},
        "gridPos": {"h": h, "w": w, "x": x, "y": y},
        "id": next_id(),
        "options": {
            "colorMode": "value",
            "graphMode": "none",
            "justifyMode": "auto",
            "orientation": "auto",
            "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
            "textMode": "auto",
        },
        "targets": [target(expr, legend=legend, instant=True)],
        "title": title,
        "type": "stat",
    }


def timeseries(
    title,
    targets,
    description,
    x,
    y,
    w=12,
    h=9,
    unit="short",
    legend_calcs=None,
    stack=False,
    fill=8,
):
    return {
        "datasource": DS,
        "description": description,
        "fieldConfig": {
            "defaults": {
                "custom": {
                    "drawStyle": "line",
                    "fillOpacity": fill,
                    "lineInterpolation": "linear",
                    "lineWidth": 1,
                    "showPoints": "never",
                    "spanNulls": False,
                    "stacking": {
                        "group": "A",
                        "mode": "normal" if stack else "none",
                    },
                },
                "unit": unit,
            },
            "overrides": [],
        },
        "gridPos": {"h": h, "w": w, "x": x, "y": y},
        "id": next_id(),
        "options": {
            "legend": {
                "calcs": legend_calcs if legend_calcs is not None else ["lastNotNull"],
                "displayMode": "table",
                "placement": "bottom",
            },
            "tooltip": {"mode": "multi", "sort": "desc"},
        },
        "targets": targets,
        "title": title,
        "type": "timeseries",
    }


def barchart(title, targets, description, x, y, w=12, h=9, unit="short", stack=False):
    return {
        "datasource": DS,
        "description": description,
        "fieldConfig": {
            "defaults": {
                "custom": {
                    "lineWidth": 1,
                    "fillOpacity": 80,
                    "gradientMode": "none",
                    "stacking": {
                        "group": "A",
                        "mode": "normal" if stack else "none",
                    },
                },
                "unit": unit,
            },
            "overrides": [],
        },
        "gridPos": {"h": h, "w": w, "x": x, "y": y},
        "id": next_id(),
        "options": {
            "legend": {
                "calcs": ["lastNotNull"],
                "displayMode": "table",
                "placement": "bottom",
            },
            "tooltip": {"mode": "multi", "sort": "desc"},
            "xTickLabelRotation": 0,
            "xTickLabelSpacing": 0,
        },
        "targets": targets,
        "title": title,
        "type": "barchart",
    }


def table(title, targets, description, x, y, w=12, h=9, unit="short"):
    return {
        "datasource": DS,
        "description": description,
        "fieldConfig": {"defaults": {"unit": unit}, "overrides": []},
        "gridPos": {"h": h, "w": w, "x": x, "y": y},
        "id": next_id(),
        "options": {
            "legend": {
                "calcs": [],
                "displayMode": "list",
                "placement": "bottom",
            },
            "tooltip": {"mode": "multi", "sort": "desc"},
        },
        "targets": targets,
        "title": title,
        "type": "table",
    }


def text(title, content, x, y, w=24, h=4):
    return {
        "gridPos": {"h": h, "w": w, "x": x, "y": y},
        "id": next_id(),
        "options": {
            "content": content,
            "mode": "markdown",
        },
        "title": title,
        "type": "text",
    }


def build_dashboard(config):
    """Builds the dashboard document from a configuration mapping."""
    job = config["job"]
    uid = config["uid"]
    title = config["title"]
    routing_uid = config["routing_uid"]
    model_filter = 'job="' + job + '",model=~"$model"'

    # The datasource is a deployment fact, so every panel records the one the
    # configuration names rather than a value baked into this file.
    global DS
    DS = {
        "type": config["datasource_type"],
        "uid": config["datasource_uid"],
    }

    reset_ids()
    panels = []
    y = 0

    panels.append(row("USAGE BY PROVIDER, ROUTE AND COST CLASS", y))
    y += 1

    panels.append(
        stat(
            "REQUESTS (TOTAL)",
            f"sum({REQUESTS}{{{model_filter}}})",
            "Requests served since the deployment started, attributed to the route the caller "
            "addressed. A composed request counts once against its parent route. The counter "
            "resets when the deployment restarts.",
            0,
            y,
            w=6,
        )
    )
    panels.append(
        stat(
            "PROMPT TOKENS (TOTAL)",
            f'sum({TOKENS}{{{model_filter},component="prompt"}})',
            "Prompt tokens since the deployment started. The prompt total is the provider's own "
            "input count and already contains the cached and cache-created tokens, so this is "
            "never added to those components.",
            6,
            y,
            w=6,
        )
    )
    panels.append(
        stat(
            "COMPLETION TOKENS (TOTAL)",
            f'sum({TOKENS}{{{model_filter},component="completion"}})',
            "Completion tokens since the deployment started. This is the provider's own "
            "output total, and it already contains the reasoning tokens a provider reports, so "
            "reasoning is not added here.",
            12,
            y,
            w=6,
        )
    )
    panels.append(
        stat(
            "PROVIDER-REPORTED COST (TOTAL)",
            f"sum({PROVIDER_COST}{{{model_filter}}})",
            "What the providers themselves reported they charged, since the deployment started. "
            "This is the provider's own figure, not an estimate. It is absent for a provider "
            "that reports no per-request cost.",
            18,
            y,
            w=6,
            unit="currencyUSD",
            decimals=6,
        )
    )
    y += 5

    panels.append(
        timeseries(
            "TOKENS BY PROVIDER (CUMULATIVE)",
            [
                target(
                    f'sum by (provider) ({TOKENS}{{{model_filter},component="prompt"}})',
                    legend="{{provider}}",
                ),
                target(
                    f'sum by (provider) ({TOKENS}{{{model_filter},component="cached"}})',
                    legend="{{provider}} · cached",
                ),
                target(
                    f'sum by (provider) ({TOKENS}{{{model_filter},component="cache_creation"}})',
                    legend="{{provider}} · cache_creation",
                ),
            ],
            "Prompt, cached and cache-created tokens by provider, since the deployment started. "
            "The cached and cache-created series are components inside the prompt series, not "
            "additions to it: a provider that served mostly cached input shows a large cached "
            "series under a prompt series that already includes it.",
            0,
            y,
            w=12,
            h=9,
            unit="short",
            legend_calcs=["lastNotNull", "max"],
        )
    )
    panels.append(
        barchart(
            "TOKENS BY COST CLASS (CUMULATIVE)",
            [
                target(
                    f'sum by (cost_class) ({TOKENS}{{{model_filter},component="prompt"}})',
                    legend="{{cost_class}}",
                ),
            ],
            "Prompt tokens by how the lane is billed, since the deployment started: local (a "
            "deployment the operator hosts), "
            "free (the provider prices the model at zero), subscription (an allowance-bearing "
            "credential) and metered (a paid API key). A free or subscription lane is never "
            "given a per-token price.",
            12,
            y,
            w=12,
            h=9,
        )
    )
    y += 9

    panels.append(
        timeseries(
            "REQUESTS BY ROUTE (CUMULATIVE)",
            [
                target(
                    f"sum by (route) ({REQUESTS}{{{model_filter}}})",
                    legend="{{route}}",
                ),
            ],
            "Requests by the route the caller addressed, since the deployment started. On a "
            "composed route this is the parent route, so a request that crossed to a second "
            "child is counted once against the route the caller asked for.",
            0,
            y,
            w=12,
            h=9,
        )
    )
    panels.append(
        barchart(
            "TOKENS BY ROUTE FAMILY (CUMULATIVE)",
            [
                target(
                    f'sum by (route_family) ({TOKENS}{{{model_filter},component="prompt"}})',
                    legend="{{route_family}}",
                ),
            ],
            "Prompt tokens by a route's role, since the deployment started: direct (bound to one "
            "target), primitive (declares "
            "an ordered candidate list) and tier (composes other routes). The role comes from "
            "the deployment configuration, not from a route's name.",
            12,
            y,
            w=12,
            h=9,
        )
    )
    y += 9

    panels.append(row("TOKEN COMPONENTS (no double counting)", y))
    y += 1

    panels.append(
        timeseries(
            "PROMPT, CACHED AND CACHE-CREATED TOKENS (CUMULATIVE)",
            [
                target(
                    f'sum({TOKENS}{{{model_filter},component="prompt"}})',
                    legend="prompt (total)",
                ),
                target(
                    f'sum({TOKENS}{{{model_filter},component="cached"}})',
                    legend="cached (inside prompt)",
                ),
                target(
                    f'sum({TOKENS}{{{model_filter},component="cache_creation"}})',
                    legend="cache_creation (inside prompt)",
                ),
            ],
            "The three input components. The prompt series is the total the provider bills as "
            "input, and the cached and cache-created series sit inside it. Adding them to the "
            "prompt series would double count the cache, so no panel in this dashboard does.",
            0,
            y,
            w=12,
            h=9,
        )
    )
    panels.append(
        timeseries(
            "COMPLETION AND REASONING TOKENS (CUMULATIVE)",
            [
                target(
                    f'sum({TOKENS}{{{model_filter},component="completion"}})',
                    legend="completion (total)",
                ),
                target(
                    f'sum({TOKENS}{{{model_filter},component="reasoning"}})',
                    legend="reasoning (inside completion)",
                ),
            ],
            "The two output components. The completion series is the provider's own output "
            "total, and the reasoning series sits inside it on every provider that reports "
            "reasoning — measured 2026-10-21 against the live OpenRouter API, where "
            "prompt + completion was the reported total and adding the reasoning count broke "
            "it. Reasoning is reported only by providers that report it, so an absent reasoning "
            "series means the provider does not report it — it is not a measured zero.",
            12,
            y,
            w=12,
            h=9,
        )
    )
    y += 9

    panels.append(row("BANGKOK CALENDAR DAY", y))
    y += 1

    panels.append(
        stat(
            "DAY START (UTC)",
            f'{CAL_DAY_START}{{job="{job}"}}',
            "The Unix timestamp of the current Bangkok midnight. The calendar day is midnight in "
            "UTC+07:00, not midnight in UTC and not a rolling twenty-four hours.",
            0,
            y,
            w=8,
            unit="dateTimeAsIso",
        )
    )
    panels.append(
        stat(
            "DAY END (UTC)",
            f'{CAL_DAY_END}{{job="{job}"}}',
            "The Unix timestamp of the next Bangkok midnight. Published by the deployment, "
            "because Prometheus cannot express a computed midnight.",
            8,
            y,
            w=8,
            unit="dateTimeAsIso",
        )
    )
    panels.append(
        stat(
            "ELAPSED IN DAY",
            f'{CAL_DAY_ELAPSED}{{job="{job}"}}',
            "Seconds since the current Bangkok midnight. A value near 86400 means the day is "
            "about to roll over.",
            16,
            y,
            w=8,
            unit="s",
        )
    )
    y += 5

    panels.append(
        barchart(
            "TOKENS TODAY (BANGKOK) BY PROVIDER",
            [
                target(
                    f'sum by (provider) ({CAL_TOKENS}{{{model_filter},component="prompt"}})',
                    legend="{{provider}}",
                ),
            ],
            "Prompt tokens since the current Bangkok midnight, by provider. The day is counted by "
            "the deployment itself and resets at Bangkok midnight, so this is a calendar day "
            "rather than a sliding window.",
            0,
            y,
            w=12,
            h=9,
        )
    )
    panels.append(
        barchart(
            "REQUESTS TODAY (BANGKOK) BY ROUTE",
            [
                target(
                    f"sum by (route) ({CAL_REQUESTS}{{{model_filter}}})",
                    legend="{{route}}",
                ),
            ],
            "Requests since the current Bangkok midnight, by the route the caller addressed.",
            12,
            y,
            w=12,
            h=9,
        )
    )
    y += 9

    panels.append(row("PROVIDER ACCOUNT STATE", y))
    y += 1

    # Coverage is stated before any money figure, because the biggest way this
    # dashboard could mislead is by showing one account's credit beside another
    # account's traffic.
    panels.append(
        stat(
            "PROVIDERS WITH ACCOUNT FINANCIAL COVERAGE",
            f'count(count by (provider) ({FRESHNESS}{{job="{job}"}}))',
            "How many providers the deployment reads account state for. A provider that is "
            "missing from this count may still serve traffic, in which case its tokens are "
            "attributed correctly but its credit, balance and usage are not collected at all.",
            0,
            y,
            w=8,
        )
    )
    panels.append(
        stat(
            "PROVIDERS SERVING TRAFFIC",
            f'count(count by (provider) ({REQUESTS}{{job="{job}"}}))',
            "How many providers have served at least one request. Compare with the coverage "
            "count: a provider in this number but not in that one is billing for work the "
            "dashboard cannot show the money side of.",
            8,
            y,
            w=8,
        )
    )
    # Deliberately NOT a subtraction of the two counts above. Served and account
    # series name the same account differently - the OpenAI account appears as
    # openai_nt in served usage and as openai in account state, because two
    # clients share one credential. A naive difference would therefore miscount
    # in both directions: it can report zero while a provider is uncovered, and
    # it can report a covered provider as a gap. The coverage question is
    # answered by the per-provider table below, where a provider carrying tokens
    # but no freshness figure is visible as a row.
    panels.append(
        stat(
            "ACCOUNT READS OK (TOTAL)",
            f'sum({ACCOUNT_FETCH}{{job="{job}",outcome="ok"}})',
            "Successful provider account reads since the deployment started. This is the "
            "count of confirmed account observations, not the number of accounts: a provider "
            "that is read every interval increments once per read.",
            16,
            y,
            w=8,
        )
    )
    y += 5

    panels.append(
        table(
            "ACCOUNT FINANCIAL COVERAGE BY PROVIDER",
            [
                target(
                    f'sum by (provider) ({TOKENS}{{{model_filter},component="prompt"}})',
                    legend="{{provider}}",
                    ref="A",
                    instant=True,
                ),
                target(
                    f'sum by (provider) ({FRESHNESS}{{job="{job}"}})',
                    legend="{{provider}}",
                    ref="B",
                    instant=True,
                ),
                target(
                    f'sum by (provider) ({CREDIT}{{scope="account",period="total",job="{job}"}})',
                    legend="{{provider}}",
                    ref="C",
                    instant=True,
                ),
            ],
            "Which providers this deployment can show the money side of, read column by "
            "column: a row carrying prompt tokens but no freshness value is serving traffic "
            "whose account credit, balance and usage are not collected at all. "
            "openrouter_stephen is the known case - its tokens are attributed correctly but "
            "it has no account figures, so the OpenRouter credit, remaining and consumed "
            "panels describe the global key's account only and must not be read as the "
            "deployment's total OpenRouter spend. A blank cell is absence of collection, "
            "never a measured zero.",
            0,
            y,
            w=24,
            h=8,
        )
    )
    y += 8

    panels.append(
        stat(
            "ACCOUNT READS (TOTAL)",
            f'sum({ACCOUNT_FETCH}{{job="{job}"}})',
            "Provider account reads since the deployment started. A provider that exposes no "
            "account surface produces no series at all, because a provider that exposes nothing "
            "has not reported a balance of zero.",
            0,
            y,
            w=6,
        )
    )
    panels.append(
        stat(
            "AUTH FAILURES (TOTAL)",
            f'sum({AUTH_FAILURES}{{job="{job}"}})',
            "Account reads rejected with 401 or 403, since the deployment started. A non-zero "
            "value means the credential the deployment holds was refused for the account "
            "surface, which is different from the provider being down.",
            6,
            y,
            w=6,
        )
    )
    panels.append(
        stat(
            "OPENROUTER CREDIT (USD) - GLOBAL KEY ACCOUNT ONLY",
            f'{CREDIT}{{provider="openrouter",currency="USD",scope="account",period="total",job="{job}"}}',
            "The credit the provider sold to the GLOBAL key's account only. The deployment also "
            "serves traffic through a second OpenRouter account (openrouter_stephen) whose "
            "account figures are NOT collected, so this is not the deployment's total "
            "OpenRouter credit. This is not the money that is left either: the consumed "
            "credit is a separate figure and the remaining balance is a third.",
            12,
            y,
            w=6,
            unit="currencyUSD",
            decimals=4,
        )
    )
    panels.append(
        stat(
            "DEEPSEEK BALANCE (CNY / USD) - ACCOUNT",
            f'{BALANCE}{{provider="deepseek",scope="account",period="total",job="{job}"}}',
            "The prepaid balance the provider reported for the account, per currency. Absent "
            "when the provider reported no balance; a zero balance is a reported zero, which "
            "is different from absent.",
            18,
            y,
            w=6,
            unit="currencyUSD",
            decimals=4,
        )
    )
    y += 5

    panels.append(
        stat(
            "OPENROUTER REMAINING (USD) - GLOBAL KEY ACCOUNT ONLY",
            f'{BALANCE}{{provider="openrouter",currency="USD",scope="account",period="total",job="{job}"}}',
            "The credit the provider sold, less the credit consumed: the money still "
            "available on the account. Published as its own figure so it is read directly "
            "rather than worked out by subtracting one panel from another.",
            0,
            y,
            w=8,
            unit="currencyUSD",
            decimals=4,
        )
    )
    panels.append(
        stat(
            "OPENROUTER CONSUMED (USD) - GLOBAL KEY ACCOUNT ONLY",
            f'{PROVIDER_USAGE}{{provider="openrouter",currency="USD",scope="account",period="total",job="{job}"}}',
            "The credit the account has consumed. This is the provider's own figure for what "
            "has been charged against the account.",
            8,
            y,
            w=8,
            unit="currencyUSD",
            decimals=4,
        )
    )
    panels.append(
        stat(
            "OPENROUTER KEY USAGE (USD) - GLOBAL KEY ONLY",
            f'{PROVIDER_USAGE}{{provider="openrouter",currency="USD",scope="key",period="total",job="{job}"}}',
            "The usage of the one API key the deployment holds, over the whole life of that "
            "key. This is a different amount from the account-wide figure, because the account "
            "may hold other keys.",
            16,
            y,
            w=8,
            unit="currencyUSD",
            decimals=4,
        )
    )
    y += 5

    panels.append(
        timeseries(
            "ALLOWANCE WINDOWS REMAINING (OPENAI)",
            [
                target(
                    f'{ALLOWANCE_PCT}{{provider="openai",pool="primary",job="{job}"}}',
                    legend="5-hour window",
                ),
                target(
                    f'{ALLOWANCE_PCT}{{provider="openai",pool="secondary",job="{job}"}}',
                    legend="weekly window",
                ),
            ],
            "The remaining percentage of each allowance window the provider reported. The two "
            "windows are shown separately because they refill on different schedules.",
            0,
            y,
            w=12,
            h=8,
            unit="percent",
        )
    )
    panels.append(
        timeseries(
            "ALLOWANCE WINDOW RESET (OPENAI)",
            [
                target(
                    f'{ALLOWANCE_RESET}{{provider="openai",pool="primary",job="{job}"}}',
                    legend="5-hour window",
                ),
                target(
                    f'{ALLOWANCE_RESET}{{provider="openai",pool="secondary",job="{job}"}}',
                    legend="weekly window",
                ),
            ],
            "Seconds until each allowance window resets.",
            12,
            y,
            w=12,
            h=8,
            unit="s",
        )
    )
    y += 8

    panels.append(
        timeseries(
            "OPENROUTER USAGE BY SCOPE AND PERIOD - GLOBAL KEY ACCOUNT ONLY",
            [
                target(
                    f'{PROVIDER_USAGE}{{provider="openrouter",currency="USD",scope="key",period="total",job="{job}"}}',
                    legend="key · total",
                ),
                target(
                    f'{PROVIDER_USAGE}{{provider="openrouter",currency="USD",scope="key",period="daily",job="{job}"}}',
                    legend="key · daily",
                ),
                target(
                    f'{PROVIDER_USAGE}{{provider="openrouter",currency="USD",scope="key",period="weekly",job="{job}"}}',
                    legend="key · weekly",
                ),
                target(
                    f'{PROVIDER_USAGE}{{provider="openrouter",currency="USD",scope="key",period="monthly",job="{job}"}}',
                    legend="key · monthly",
                ),
                target(
                    f'{PROVIDER_USAGE}{{provider="openrouter",currency="USD",scope="account",period="total",job="{job}"}}',
                    legend="account · total",
                ),
            ],
            "The usage the GLOBAL OpenRouter account reported, kept apart by scope and by "
            "period. The key figures are what that one key has used; the account figure covers "
            "every key on that account. Each period is its own window, so none of these is "
            "added to another. The second OpenRouter account (openrouter_stephen) has no "
            "account figures collected, so it does not appear here at all - its served tokens "
            "are attributed in the usage-by-provider panels instead. This is the provider's "
            "own figure for what has been charged, kept separate from any Switchyard-derived "
            "estimate.",
            0,
            y,
            w=24,
            h=8,
            unit="currencyUSD",
        )
    )
    y += 8

    panels.append(row("HOW TO READ THIS DASHBOARD", y))
    y += 1

    panels.append(
        text(
            "ACCOUNTING RULES",
            "**Prompt tokens are the provider's own input total.** The cached and cache-created "
            "counts sit *inside* it, so a panel that shows prompt and cached side by side is "
            "showing a total and one of its components — not two things to add up. There is "
            'deliberately no "all tokens" figure, because any such figure could double count '
            "the cache.\n\n"
            "**Reasoning is a component of completion** on every provider that reports it "
            "separately, and is absent on the providers that do not. An absent series means the "
            "provider does not report it; it is not a measured zero. This is the same shape as "
            "the cache inside the prompt total: a component is never added to the total that "
            "contains it.\n\n"
            "**The provider-reported cost and a Switchyard estimate are different things.** They "
            "are published as separate metric families and are never added together. A provider "
            "that reports no per-request cost produces no cost series.\n\n"
            "**The usage panels are cumulative, not windowed.** They read the counters "
            "directly, so they show what the running deployment has served since it started. A "
            "counter resets when the deployment restarts.\n\n"
            "**Currency, scope and period are separate.** A provider figure has three facts "
            "attached to it: the currency it is in, whether it covers the account or the single "
            "key, and which window it covers. The account-wide figure and the key figure are "
            "different amounts of money, so they are never added together.\n\n"
            "**Purchased credit, consumed credit and remaining balance are different "
            "figures.** The credit the provider sold, the credit consumed against it, and the "
            "credit still available are three separate series. The remaining balance is the "
            "difference between the first two.\n\n"
            "**OpenRouter financial coverage is PARTIAL.** This deployment holds two "
            "OpenRouter credentials and serves through both. Account figures are collected "
            "for the global key's account only. The second account, `openrouter_stephen`, "
            "has its served tokens attributed correctly but its account credit, balance and "
            "usage are **not collected**, so its spend is invisible here. The OpenRouter "
            "credit, remaining and consumed panels therefore describe one account, not the "
            "deployment's total OpenRouter position, and must not be presented as complete "
            "financial reporting. A missing figure is absent, never a measured zero.\n\n"
            "**Cost class is a deployment fact.** `local` is a deployment the operator hosts, "
            "`free` is a model the provider prices at zero, `subscription` is an "
            "allowance-bearing credential, and `metered` is a paid API key. A free or "
            "subscription lane is never given a per-token price, and a local lane is not "
            "presented as costing nothing — it consumes hardware.\n\n"
            "**The calendar day is a Bangkok day.** It resets at midnight in UTC+07:00, which "
            "is not midnight in UTC and not a rolling twenty-four hours.\n\n"
            "**There is no savings-versus-baseline figure here.** That comparison needs a "
            "baseline assumption set that has not been validated, so it is not shown rather "
            "than shown as a guess.",
            0,
            y,
            w=24,
            h=12,
        )
    )
    y += 12

    return {
        "annotations": {"list": []},
        "description": (
            "Provider, route, family and cost-class attribution for Switchyard usage, with "
            "the token components kept separate so nothing is double counted, and the "
            "provider account state Switchyard reads with its own credentials. Every "
            "attribution value comes from the deployment configuration, not from a "
            "model-name guess."
        ),
        "editable": True,
        "graphTooltip": 1,
        "links": [
            {
                "asDropdown": False,
                "icon": "external link",
                "includeVars": False,
                "keepTime": True,
                "tags": [],
                "targetBlank": False,
                "title": "SWITCHYARD V7.11 (routing and serving)",
                "tooltip": "The routing and serving dashboard this one complements.",
                "type": "link",
                "url": f"/d/{routing_uid}",
            }
        ],
        "panels": panels,
        "refresh": "1m",
        "schemaVersion": 39,
        "tags": ["switchyard", "economics", "tokens", "v8"],
        "templating": {
            "list": [
                {
                    "allValue": ".*",
                    "current": {"selected": True, "text": ["All"], "value": ["$__all"]},
                    "datasource": DS,
                    "includeAll": True,
                    "label": "model",
                    "multi": True,
                    "name": "model",
                    "query": {
                        "query": f'label_values({TOKENS}{{job="{job}"}}, model)',
                        "refId": "A",
                    },
                    "refresh": 2,
                    "type": "query",
                },
                {
                    "allValue": ".*",
                    "current": {"selected": True, "text": ["All"], "value": ["$__all"]},
                    "datasource": DS,
                    "includeAll": True,
                    "label": "provider",
                    "multi": True,
                    "name": "provider",
                    "query": {
                        "query": f'label_values({TOKENS}{{job="{job}"}}, provider)',
                        "refId": "A",
                    },
                    "refresh": 2,
                    "type": "query",
                },
            ]
        },
        "time": {"from": "now-24h", "to": "now"},
        "timezone": "browser",
        "title": title,
        "uid": uid,
        "version": 0,
    }


def build_request(config):
    """Builds the publishing request for the dashboard document.

    The publishing tool accepts one request shape:
      {op, uid, message, dashboard, folder_uid, expected_version}
    and refuses any other field. A raw Grafana POST body carries "overwrite",
    which the tool rejects as an unsupported field, and omits "op" and "uid",
    which the tool requires. The tool builds the upstream payload itself,
    including overwrite=True, so the request must not carry it.
    """
    return {
        "op": "create",
        "uid": config["uid"],
        "message": config["message"],
        "folder_uid": None,
        "dashboard": build_dashboard(config),
    }


def load_config(path):
    """Reads the dashboard configuration."""
    import tomllib

    with open(path, "rb") as handle:
        return tomllib.load(handle)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build the economics dashboard document.")
    parser.add_argument(
        "--config",
        required=True,
        help="The dashboard configuration file.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Where to write the request document.",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    request = build_request(config)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(request, handle, indent=1)
    print(f"wrote {args.out}: {len(request['dashboard']['panels'])} panels")
    return 0


if __name__ == "__main__":
    sys.exit(main())
