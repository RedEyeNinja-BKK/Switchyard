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
  * `reasoning` is reported separately, and its relationship to `completion`
    varies by provider and model. Reasoning is never added again when a total is
    calculated: the recorded total is `prompt + completion`. It is NOT true that
    reasoning is always a subset of completion — measured 2026-10-23 across the
    routing log, 17 of 29359 records reported reasoning above completion
    (`z-ai/glm-5.3-flash`, `liquid/lfm-2.5-2.6b:free`), while `prompt + completion
    == total` held in all 29359. An earlier note here asserted the subset
    universally from a single OpenRouter measurement; that was too strong and has
    been corrected. Reasoning is published only when the provider reports it, so
    an absent series means unreported, not a measured zero.
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
CAL_COST = "switchyard_economics_calendar_provider_cost"

# Rolling windows computed by the deployment from the events it observed, rather
# than by `increase()` over a counter. `window` is `1h` or `24h`.
ROLL_TOKENS = "switchyard_economics_rolling_tokens"
ROLL_REQUESTS = "switchyard_economics_rolling_requests"
ROLL_COST = "switchyard_economics_rolling_provider_cost"
ROLL_COVERAGE = "switchyard_economics_rolling_coverage_seconds"

# The parts of a provider's balance, where the provider breaks the total down.
# These are components OF the total, never additional to it.
BALANCE_COMPONENT = "switchyard_economics_provider_balance_component"

# Retired with the localclaw exporter. Switchyard reads DeepSeek's balance
# endpoint, which returns a level rather than a ledger, so a top-up ledger,
# lifetime spend and request-derived cost cannot be reconstructed from it. Those
# figures are not published here at all; the "NOT MEASURABLE HERE" panel says so
# on the dashboard rather than leaving a gap unexplained.

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


def piechart(title, targets, description, x, y, w=8, h=9, unit="short"):
    return {
        "datasource": DS,
        "description": description,
        "fieldConfig": {"defaults": {"unit": unit}, "overrides": []},
        "gridPos": {"h": h, "w": w, "x": x, "y": y},
        "id": next_id(),
        "options": {
            "displayLabels": ["percent"],
            "legend": {
                "calcs": ["sum"],
                "displayMode": "table",
                "placement": "right",
                "showLegend": True,
                "values": ["value", "percent"],
            },
            "pieType": "donut",
            "reduceOptions": {
                "calcs": ["lastNotNull"],
                "fields": "",
                "values": True,
            },
            "tooltip": {"mode": "multi", "sort": "desc"},
        },
        "targets": targets,
        "title": title,
        "type": "piechart",
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
    openrouter_account = config["openrouter_account"]
    openrouter_secondary = config["openrouter_secondary_account"]
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
            "output total. Reasoning is reported separately and is never added to it, so the "
            "provider's figures are shown exactly as reported.",
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
                    legend="reasoning (reported separately)",
                ),
            ],
            "The two output components. The completion series is the provider's own output "
            "total and is never adjusted. The reasoning series is what the provider reported "
            "separately; its relationship to completion varies by model, and reasoning can "
            "exceed completion, so it is shown beside that total rather than as a part of it. "
            "Adding the two would double count whenever reasoning does sit inside completion, "
            "which is why no panel does it. Reasoning is reported only by providers that "
            "report it, so an absent reasoning series means the provider does not report it — "
            "it is not a measured zero.",
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

    panels.append(
        stat(
            "Provider-Reported Cost Today (Bangkok)",
            f'sum by (currency) ({CAL_COST}{{job="{job}"}})',
            "The cost providers themselves reported for requests served since Bangkok midnight, "
            "per currency. This is what the providers say they charged, not a Switchyard "
            "estimate. A currency that appears here is one a provider actually billed in; the "
            "currencies are never added together. The panel is empty when no provider reported "
            "a cost today, which is not the same as a day that cost nothing.",
            0,
            y,
            w=12,
            unit="currencyUSD",
            decimals=6,
        )
    )
    panels.append(
        stat(
            "Requests Today (Bangkok)",
            f'sum({CAL_REQUESTS}{{job="{job}"}})',
            "Every request served since Bangkok midnight, across all routes and providers.",
            12,
            y,
            w=12,
        )
    )
    y += 5

    panels.append(row("ROLLING WINDOWS (1h · 24h — IN-MEMORY)", y))
    y += 1

    # Coverage first, because every other figure in this row is bounded by it. A
    # reader who takes a 24-hour total for a full day when the process restarted an
    # hour ago has been misled by the dashboard, not by the number.
    panels.append(
        stat(
            "1h Window Coverage",
            f'{ROLL_COVERAGE}{{job="{job}",window="1h"}}',
            "How much of the one-hour window this process has actually observed: its uptime, "
            "capped at one hour. At 3600 the window is complete. Below that the rolling "
            "figures cover only the time since the last restart, and everything before it is "
            "outside the window rather than zero.",
            0,
            y,
            w=6,
            unit="s",
            decimals=0,
        )
    )
    panels.append(
        stat(
            "24h Window Coverage",
            f'{ROLL_COVERAGE}{{job="{job}",window="24h"}}',
            "How much of the 24-hour window this process has actually observed: its uptime, "
            "capped at 24 hours. Only at 86400 is a rolling 24-hour total a complete day. Below "
            "that, read every 24h figure on this row as a partial day and say so rather than "
            "comparing it with yesterday.",
            6,
            y,
            w=6,
            unit="s",
            decimals=0,
            thresholds={
                "mode": "absolute",
                "steps": [
                    {"color": "orange", "value": None},
                    {"color": "green", "value": 86400},
                ],
            },
        )
    )
    panels.append(
        stat(
            "Rolling 1h Requests",
            f'sum({ROLL_REQUESTS}{{job="{job}",window="1h"}})',
            "Requests served in the trailing hour, counted from the events this process "
            "observed.",
            12,
            y,
            w=6,
        )
    )
    panels.append(
        stat(
            "Rolling 24h Requests",
            f'sum({ROLL_REQUESTS}{{job="{job}",window="24h"}})',
            "Requests served in the trailing 24 hours, counted from observed events. Bounded by "
            "24h Window Coverage beside it: if that reads less than 86400, this is a partial "
            "day.",
            18,
            y,
            w=6,
        )
    )
    y += 5

    panels.append(
        stat(
            "Rolling 1h Prompt Tokens",
            f'sum({ROLL_TOKENS}{{job="{job}",window="1h",component="prompt"}})',
            "Prompt tokens in the trailing hour. The prompt total includes the cached and "
            "cache-creation components, which are published separately and never added to it.",
            0,
            y,
            w=6,
        )
    )
    panels.append(
        stat(
            "Rolling 24h Prompt Tokens",
            f'sum({ROLL_TOKENS}{{job="{job}",window="24h",component="prompt"}})',
            "Prompt tokens in the trailing 24 hours, subject to the coverage caveat above.",
            6,
            y,
            w=6,
        )
    )
    panels.append(
        stat(
            "Rolling 1h Completion Tokens",
            f'sum({ROLL_TOKENS}{{job="{job}",window="1h",component="completion"}})',
            "Completion tokens in the trailing hour. Reasoning is reported separately and is "
            "not added to this figure.",
            12,
            y,
            w=6,
        )
    )
    panels.append(
        stat(
            "Rolling 24h Completion Tokens",
            f'sum({ROLL_TOKENS}{{job="{job}",window="24h",component="completion"}})',
            "Completion tokens in the trailing 24 hours. Reasoning is reported separately and "
            "is never added to this figure.",
            18,
            y,
            w=6,
        )
    )
    y += 5

    panels.append(
        timeseries(
            "Rolling requests by window",
            [
                target(
                    f'sum({ROLL_REQUESTS}{{job="{job}",window="1h"}})',
                    legend="1h",
                ),
                target(
                    f'sum({ROLL_REQUESTS}{{job="{job}",window="24h"}})',
                    legend="24h",
                    ref="B",
                ),
            ],
            "The two windows over time. They are different questions and different magnitudes, "
            "so they share an axis rather than a total. A step down to zero is a restart, after "
            "which the window repopulates from what the new process observes.",
            0,
            y,
            w=12,
            h=8,
        )
    )
    panels.append(
        timeseries(
            "Rolling 24h tokens by component",
            [
                target(
                    f'sum by (component) ({ROLL_TOKENS}{{job="{job}",window="24h"}})',
                    legend="{{component}}",
                ),
            ],
            "Each token component over the trailing day. Prompt contains cached and "
            "cache-creation, so those are never added to it. Completion and reasoning are shown "
            "as the provider reported them: reasoning is often a component of completion and "
            "occasionally exceeds it, so the two lines are read side by side and never summed "
            "into one total.",
            12,
            y,
            w=12,
            h=8,
        )
    )
    y += 8

    panels.append(
        barchart(
            "Rolling 24h requests by route family",
            [
                target(
                    f'sum by (route_family) ({ROLL_REQUESTS}{{job="{job}",window="24h"}})',
                    legend="{{route_family}}",
                ),
            ],
            "Where the last day of traffic went, by route family: a tier, a routing primitive, "
            "or a directly addressed route.",
            0,
            y,
            w=8,
            h=8,
        )
    )
    panels.append(
        barchart(
            "Rolling 24h requests by provider",
            [
                target(
                    f'sum by (provider) ({ROLL_REQUESTS}{{job="{job}",window="24h"}})',
                    legend="{{provider}}",
                ),
            ],
            "Where the last day of traffic went, by the provider that served it.",
            8,
            y,
            w=8,
            h=8,
        )
    )
    panels.append(
        stat(
            "Rolling 24h Provider-Reported Cost",
            f'sum by (currency) ({ROLL_COST}{{job="{job}",window="24h"}})',
            "The cost providers reported for requests served in the trailing day, per currency. "
            "Empty when no provider reported a cost in the window: a provider that reports no "
            "per-request cost produces no series, which is different from a measured zero.",
            16,
            y,
            w=8,
            unit="currencyUSD",
            decimals=6,
        )
    )
    y += 8

    panels.append(
        text(
            "ROLLING WINDOWS — HOW THESE NUMBERS ARE PRODUCED, AND WHEN THEY ARE INCOMPLETE",
            "**These are computed from events, not from `increase()`.** A PromQL `increase()` "
            "over a counter extrapolates from whichever samples happen to exist, so a window "
            "that begins before the process started, or that contains a restart, is reported as "
            "usage the window never held. Switchyard records the events itself and totals the "
            "ones inside the window, so a one-hour figure means what was observed in the last "
            "hour.\n\n"
            "**They are held in memory and reset on restart.** After a restart the windows begin "
            "empty and repopulate from newly observed traffic. The `Window Coverage` panels "
            "state how much of each window the current process has actually seen, so a partial "
            "day is never presented as a complete one. Compare 24h totals against yesterday "
            "only when 24h coverage reads a full 86400 seconds.\n\n"
            "**They are exact to one minute.** Events aggregate into one-minute buckets and "
            "buckets older than 24 hours are discarded, so memory follows the last day of "
            "activity. A rolling figure can therefore differ from the instant-by-instant truth "
            "by at most the final partial minute; it is not presented as more precise than that.\n\n"
            "**These are not the Bangkok calendar day.** The calendar day resets at midnight in "
            "UTC+07:00; a rolling 24-hour window slides continuously. They answer different "
            "questions and are never substituted for one another.",
            0,
            y,
            w=24,
            h=11,
        )
    )
    y += 11

    # The observed-mix row: the presentation V5.39 used and V8 dropped. It answers
    # "what is the traffic made of" and "where did it go" from the same rolling
    # counters, so it adds a reading and not a measurement.
    panels.append(row("TRAFFIC — OBSERVED MIX (rolling 24h)", y))
    y += 1

    panels.append(
        piechart(
            "REQUESTS BY PROVIDER (24H)",
            [
                target(
                    f'sum by (provider) ({ROLL_REQUESTS}{{job="{job}",window="24h",model=~"$model"}})',
                    legend="{{provider}}",
                    ref="A",
                    instant=True,
                ),
            ],
            "The share of the last day's requests served by each provider. Composition, not "
            "volume: the absolute counts are in the adjacent bar chart.",
            0,
            y,
        )
    )
    panels.append(
        piechart(
            "REQUESTS BY COST CLASS (24H)",
            [
                target(
                    f'sum by (cost_class) ({ROLL_REQUESTS}{{job="{job}",window="24h",model=~"$model"}})',
                    legend="{{cost_class}}",
                    ref="A",
                    instant=True,
                ),
            ],
            "The share of the last day's requests by how they are paid for: metered, "
            "subscription, free or local. This replaces V5.39's reasoning-mode split, which "
            "has no equivalent label in Switchyard's telemetry: reasoning effort is a route "
            "setting, not an attribute recorded per request.",
            8,
            y,
        )
    )
    panels.append(
        piechart(
            "REQUESTS BY MODEL (24H)",
            [
                target(
                    f'sum by (model) ({ROLL_REQUESTS}{{job="{job}",window="24h",model=~"$model"}})',
                    legend="{{model}}",
                    ref="A",
                    instant=True,
                ),
            ],
            "The share of the last day's requests served by each model. Models that were not "
            "addressed in the window are absent rather than zero.",
            16,
            y,
        )
    )
    y += 9

    panels.append(
        barchart(
            "REQUESTS 24H BY MODEL",
            [
                target(
                    f'sum by (model) ({ROLL_REQUESTS}{{job="{job}",window="24h",model=~"$model"}})',
                    legend="{{model}}",
                ),
            ],
            "The absolute request count behind the model pie, ranked so a long tail stays "
            "readable.",
            0,
            y,
            w=8,
            h=8,
        )
    )
    panels.append(
        barchart(
            "COMPLETION TOKENS 24H BY MODEL",
            [
                target(
                    f'sum by (model) ({ROLL_TOKENS}{{job="{job}",window="24h",component="completion",model=~"$model"}})',
                    legend="{{model}}",
                ),
            ],
            "Provider-reported output tokens over the last day, by model. Reasoning is inside "
            "this figure rather than beside it, so it is not added again.",
            8,
            y,
            w=8,
            h=8,
        )
    )
    panels.append(
        barchart(
            "PROMPT TOKENS 24H BY MODEL",
            [
                target(
                    f'sum by (model) ({ROLL_TOKENS}{{job="{job}",window="24h",component="prompt",model=~"$model"}})',
                    legend="{{model}}",
                ),
            ],
            "Provider-reported input tokens over the last day, by model. The cached and "
            "cache-created counts sit inside this figure, so they are not added to it.",
            16,
            y,
            w=8,
            h=8,
        )
    )
    y += 8

    panels.append(
        timeseries(
            "REQUEST RATE BY PROVIDER (rolling 1h)",
            [
                target(
                    f'sum by (provider) ({ROLL_REQUESTS}{{job="{job}",window="1h"}})',
                    legend="{{provider}}",
                ),
            ],
            "The short window, so a provider that has just stopped serving is visible before the "
            "24-hour view moves. The 1h window is in memory and restarts empty.",
            0,
            y,
            w=12,
            h=8,
        )
    )
    panels.append(
        timeseries(
            "TOKEN MIX BY COMPONENT (24H, stacked)",
            [
                target(
                    f'sum by (component) ({ROLL_TOKENS}{{job="{job}",window="24h"}})',
                    legend="{{component}}",
                ),
            ],
            "How the last day's tokens divide between prompt, cached, cache-created, completion "
            "and reasoning. Stacked because the components are disjoint parts of one day's "
            "traffic; the prompt bar is the input side and already contains its cache shares, so "
            "read the stack as a shape and the individual panels for exact figures.",
            12,
            y,
            w=12,
            h=8,
            stack=True,
        )
    )
    y += 8

    panels.append(
        text(
            "TRAFFIC — WHAT THESE WINDOWS MEAN",
            "Every panel in this row reads the same two rolling counters — "
            "`switchyard_economics_rolling_requests` and `…_rolling_tokens` — filtered by "
            "`window`. They are a second *view* of the figures in the rows above, not a second "
            "measurement, so a count here and a count above agree by construction.\n\n"
            "**They are held in memory and reset on restart.** After a deployment restart the "
            "24h window repopulates from newly observed traffic; the "
            "`24h Window Coverage` panel in ROLLING WINDOWS states how much of the window the "
            "current process has actually seen. Do not compare a partial day against a full "
            "one.\n\n"
            "**The composition pies are shares, not volumes.** A pie with three providers tells "
            "you the split and not the total; the bar chart beside it carries the counts.\n\n"
            "**The cost-class pie replaces a panel V5.39 had.** V5.39 split requests by "
            "reasoning mode. Switchyard does not record reasoning effort per request — it is a "
            "route setting, and a route can be reached by several callers with different "
            "intentions — so that split cannot be reconstructed. `cost_class` is the "
            "observability axis the telemetry actually carries.",
            0,
            y,
            w=24,
            h=11,
        )
    )
    y += 11

    panels.append(row("OPENAI ALLOWANCE WINDOWS (primary 5-hr · secondary weekly)", y))
    y += 1

    # V5.39 read these from localclaw-econ's resource exporter. Switchyard now
    # reads the same allowance windows itself, so these are the same figures from
    # the deployment's own credentials.
    #
    # Switchyard publishes the REMAINING percentage and the seconds to reset.
    # Used percent is the complement, so it is computed here rather than stored,
    # and a missing reading stays missing instead of becoming a false 100%.
    panels.append(
        stat(
            "5-hr Remaining",
            f'last_over_time({ALLOWANCE_PCT}{{provider="openai",pool="primary",job="{job}"}}[3h])',
            "Remaining percentage of the primary five-hour window. Read by Switchyard with "
            "the deployment's own OpenAI credential.",
            0,
            y,
            w=4,
            unit="percent",
            decimals=0,
            thresholds={
                "mode": "absolute",
                "steps": [
                    {"color": "red", "value": None},
                    {"color": "orange", "value": 15},
                    {"color": "green", "value": 40},
                ],
            },
        )
    )
    panels.append(
        stat(
            "5-hr Used",
            f'(100 - last_over_time({ALLOWANCE_PCT}{{provider="openai",pool="primary",job="{job}"}}[3h]))',
            "Consumed percentage of the primary five-hour window. The provider reports what is "
            "left, so this is its complement.",
            4,
            y,
            w=4,
            unit="percent",
            decimals=0,
        )
    )
    panels.append(
        stat(
            "Time to 5-hr Reset",
            f'last_over_time({ALLOWANCE_RESET}{{provider="openai",pool="primary",job="{job}"}}[3h]) / 60',
            "Minutes until the five-hour window resets. Switchyard publishes the remaining "
            "seconds, so this is that figure in minutes.",
            8,
            y,
            w=4,
            unit="m",
            decimals=0,
        )
    )
    panels.append(
        stat(
            "Weekly Remaining",
            f'last_over_time({ALLOWANCE_PCT}{{provider="openai",pool="secondary",job="{job}"}}[3h])',
            "Remaining percentage of the secondary weekly window.",
            12,
            y,
            w=4,
            unit="percent",
            decimals=0,
            thresholds={
                "mode": "absolute",
                "steps": [
                    {"color": "red", "value": None},
                    {"color": "orange", "value": 15},
                    {"color": "green", "value": 40},
                ],
            },
        )
    )
    panels.append(
        stat(
            "Weekly Used",
            f'(100 - last_over_time({ALLOWANCE_PCT}{{provider="openai",pool="secondary",job="{job}"}}[3h]))',
            "Consumed percentage of the secondary weekly window.",
            16,
            y,
            w=4,
            unit="percent",
            decimals=0,
        )
    )
    panels.append(
        stat(
            "Time to Weekly Reset",
            f'last_over_time({ALLOWANCE_RESET}{{provider="openai",pool="secondary",job="{job}"}}[3h]) / 3600',
            "Hours until the weekly window resets. Switchyard publishes the remaining seconds, "
            "so this is that figure in hours. The window refreshes on a weekly cycle, not daily.",
            20,
            y,
            w=4,
            unit="h",
            decimals=1,
        )
    )
    y += 5

    panels.append(
        timeseries(
            "5-hr window: used and remaining (Turnstone OpenAI)",
            [
                target(
                    f'{ALLOWANCE_PCT}{{provider="openai",pool="primary",job="{job}"}}',
                    legend="remaining %",
                ),
                target(
                    f'(100 - {ALLOWANCE_PCT}{{provider="openai",pool="primary",job="{job}"}})',
                    legend="used %",
                ),
            ],
            "The primary five-hour window. Used and remaining are complements: together they "
            "always sum to 100, so they cross at the halfway point.",
            0,
            y,
            w=12,
            h=8,
            unit="percent",
        )
    )
    panels.append(
        timeseries(
            "Weekly window: used and remaining (Turnstone OpenAI)",
            [
                target(
                    f'{ALLOWANCE_PCT}{{provider="openai",pool="secondary",job="{job}"}}',
                    legend="remaining %",
                ),
                target(
                    f'(100 - {ALLOWANCE_PCT}{{provider="openai",pool="secondary",job="{job}"}})',
                    legend="used %",
                ),
            ],
            "The secondary weekly window. It refills on a different schedule from the "
            "five-hour window, which is why the two are never combined.",
            12,
            y,
            w=12,
            h=8,
            unit="percent",
        )
    )
    y += 8

    panels.append(row("DEEPSEEK TELEMETRY", y))
    y += 1

    panels.append(
        stat(
            "DeepSeek Status",
            f'last_over_time({FRESHNESS}{{provider="deepseek",job="{job}"}}[3h])',
            "Seconds since Switchyard last read DeepSeek's account successfully. A value near "
            "the polling interval is healthy; a value that keeps climbing means the reads are "
            "failing, which is different from the provider being out of credit.",
            0,
            y,
            w=6,
            unit="s",
            decimals=0,
            thresholds={
                "mode": "absolute",
                "steps": [
                    {"color": "green", "value": None},
                    {"color": "orange", "value": 900},
                    {"color": "red", "value": 3600},
                ],
            },
        )
    )
    panels.append(
        stat(
            "DeepSeek Balance (CNY)",
            f'last_over_time({BALANCE}{{provider="deepseek",currency="CNY",scope="account",period="total",job="{job}"}}[3h])',
            "The prepaid balance DeepSeek reported, in CNY. Thresholds follow the configured "
            "operating floor: above 120 is comfortable, 60 is the floor, below it the lane is "
            "close to refusing requests. A zero here is a reported zero, which is different "
            "from an absent series.",
            6,
            y,
            w=6,
            unit="CNY",
            decimals=2,
            thresholds={
                "mode": "absolute",
                "steps": [
                    {"color": "red", "value": None},
                    {"color": "orange", "value": 60},
                    {"color": "green", "value": 120},
                ],
            },
        )
    )
    panels.append(
        stat(
            "DeepSeek Balance (USD)",
            f'last_over_time({BALANCE}{{provider="deepseek",currency="USD",scope="account",period="total",job="{job}"}}[3h])',
            "The prepaid balance in USD, which DeepSeek reports separately from CNY. Held "
            "beside the CNY figure rather than converted: a conversion would invent an exchange "
            "rate the provider never stated.",
            12,
            y,
            w=6,
            unit="currencyUSD",
            decimals=2,
        )
    )
    panels.append(
        stat(
            "DeepSeek 24h Balance Movement",
            f'clamp_min(max_over_time({BALANCE}{{provider="deepseek",currency="CNY",scope="account",period="total",job="{job}"}}[24h]) '
            f'- min_over_time({BALANCE}{{provider="deepseek",currency="CNY",scope="account",period="total",job="{job}"}}[24h]), 0)',
            "How far the CNY balance moved in the trailing 24 hours. This is a BALANCE move, "
            "not a spend figure: a top-up during the window is invisible in the balance and so "
            "is not subtracted, and a top-up larger than the day's usage would make the true "
            "spend exceed this number. Switchyard does not hold a DeepSeek top-up ledger, so it "
            "cannot separate the two.",
            18,
            y,
            w=6,
            unit="CNY",
            decimals=2,
        )
    )
    y += 5

    # The balance the provider breaks down into the kinds of credit it is made of.
    # These are parts OF the total, so they are shown beside it and never added up
    # with it.
    panels.append(
        stat(
            "DeepSeek Remaining — Topped-Up (CNY)",
            f'last_over_time({BALANCE_COMPONENT}{{provider="deepseek",component="topped_up",currency="CNY",scope="account",period="total",job="{job}"}}[3h])',
            "Credit that was PAID FOR and is still available, as DeepSeek reports it. This is "
            "money left, not money spent: an account that bought a great deal and used it can "
            "show a small remaining figure. It is a component of the balance total, so adding "
            "it to that total would count the same credit twice.",
            0,
            y,
            w=8,
            unit="CNY",
            decimals=2,
        )
    )
    panels.append(
        stat(
            "DeepSeek Remaining — Granted (CNY)",
            f'last_over_time({BALANCE_COMPONENT}{{provider="deepseek",component="granted",currency="CNY",scope="account",period="total",job="{job}"}}[3h])',
            "Credit DeepSeek gave away, still available. A promotional grant and money that was "
            "paid for are different credit and are kept apart, because they run out for "
            "different reasons.",
            8,
            y,
            w=8,
            unit="CNY",
            decimals=2,
        )
    )
    panels.append(
        stat(
            "DeepSeek Remaining — Total (CNY)",
            f'last_over_time({BALANCE}{{provider="deepseek",currency="CNY",scope="account",period="total",job="{job}"}}[3h])',
            "The balance total DeepSeek reports, which is the sum of the two components beside "
            "it. Shown so the components can be checked against the whole rather than trusted "
            "on their own.",
            16,
            y,
            w=8,
            unit="CNY",
            decimals=2,
        )
    )
    y += 5

    # The top-up ledger, lifetime spend and request-derived actual cost that
    # V5.39 read from the localclaw exporter are not reproduced: their inputs are
    # a ledger and a rate card that Switchyard does not have. See the coverage
    # text panel below, which states why rather than leaving the row short.
    panels.append(
        timeseries(
            "DeepSeek balance trend (CNY)",
            [
                target(
                    f'{BALANCE}{{provider="deepseek",currency="CNY",scope="account",period="total",job="{job}"}}',
                    legend="balance ¥",
                ),
            ],
            "The observed balance over time. A step up is a top-up; the slow decline between "
            "steps is metered usage. Because both appear in the same series, the slope alone "
            "cannot be read as a burn rate across a top-up, and the panel cannot separate the "
            "two: a balance level is not a ledger.",
            0,
            y,
            w=24,
            h=8,
            unit="CNY",
        )
    )
    y += 8

    panels.append(
        text(
            "DEEPSEEK COVERAGE - WHERE EACH NUMBER COMES FROM",
            "**Read from Switchyard:** the live balance in both currencies, the 24-hour balance "
            "movement, the remaining-credit breakdown and the age of the last successful "
            "account read. Every figure in this row comes from Switchyard's own credentialed "
            "read of DeepSeek's balance endpoint.\n\n"
            "**Not measurable here:** the top-up ledger, lifetime spend, and request-derived "
            "actual cost. Switchyard reads a balance *level*, not a ledger, so a top-up and a "
            "day of usage are indistinguishable from two readings alone. Those three figures "
            "were computed by the retired localclaw exporter from a ledger and a rate card it "
            "held in its own process; no metric carries their inputs, so they are absent rather "
            "than approximated.\n\n"
            "**Also not measured:** a true metered-spend figure. No available source separates "
            "metered spend from top-ups, so a panel claiming to would be an approximation "
            "dressed as a measurement.",
            0,
            y,
            w=24,
            h=8,
        )
    )
    y += 8

    panels.append(row("OPENROUTER USAGE TELEMETRY", y))
    y += 1

    panels.append(
        stat(
            "OpenRouter Status",
            f'last_over_time({FRESHNESS}{{provider="openrouter",job="{job}"}}[3h])',
            "Seconds since Switchyard last read the primary OpenRouter account successfully. "
            "Each financial panel on this row names the account it covers; the deployment holds "
            "two OpenRouter accounts and neither figure includes the other.",
            0,
            y,
            w=6,
            unit="s",
            decimals=0,
            thresholds={
                "mode": "absolute",
                "steps": [
                    {"color": "green", "value": None},
                    {"color": "orange", "value": 900},
                    {"color": "red", "value": 3600},
                ],
            },
        )
    )
    panels.append(
        stat(
            "OpenRouter Remaining (USD · global key)",
            f'last_over_time({BALANCE}{{provider="openrouter",currency="USD",scope="account",period="total",job="{job}"}}[3h])',
            "Credit left on the GLOBAL key's account: what was purchased less what has been "
            "consumed. Published directly by Switchyard so it is read rather than worked out "
            "from two other panels.",
            6,
            y,
            w=6,
            unit="currencyUSD",
            decimals=4,
            thresholds={
                "mode": "absolute",
                "steps": [
                    {"color": "red", "value": None},
                    {"color": "orange", "value": 10},
                    {"color": "green", "value": 25},
                ],
            },
        )
    )
    panels.append(
        stat(
            "OpenRouter Usage (total, USD · global key)",
            f'last_over_time({PROVIDER_USAGE}{{provider="openrouter",currency="USD",scope="account",period="total",job="{job}"}}[3h])',
            "Lifetime usage the provider reports for the GLOBAL key's account. This is what has "
            "been consumed, not what is left.",
            12,
            y,
            w=6,
            unit="currencyUSD",
            decimals=4,
        )
    )
    panels.append(
        stat(
            "OpenRouter Credits (total, USD · global key)",
            f'last_over_time({CREDIT}{{provider="openrouter",currency="USD",scope="account",period="total",job="{job}"}}[3h])',
            "Credits purchased on the GLOBAL key's account. This is the money put in, not the "
            "money left.",
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
            "OpenRouter 24h Spend (USD · global key)",
            f'clamp_min(max_over_time({PROVIDER_USAGE}{{provider="openrouter",currency="USD",scope="account",period="total",job="{job}"}}[24h]) '
            f'- min_over_time({PROVIDER_USAGE}{{provider="openrouter",currency="USD",scope="account",period="total",job="{job}"}}[24h]), 0)',
            "Spend on the GLOBAL key's account in the trailing 24 hours, measured as the rise "
            "of the provider's own cumulative usage gauge. Unlike a balance, a usage gauge only "
            "rises, so a top-up cannot distort it.",
            0,
            y,
            w=8,
            unit="currencyUSD",
            decimals=4,
        )
    )
    panels.append(
        stat(
            "OpenRouter Key Usage (total, USD · global key)",
            f'last_over_time({PROVIDER_USAGE}{{provider="openrouter",currency="USD",scope="key",period="total",job="{job}"}}[3h])',
            "Lifetime usage of the single GLOBAL key, over the whole life of that key. A "
            "different amount from the account-wide figure whenever the account holds other keys.",
            8,
            y,
            w=8,
            unit="currencyUSD",
            decimals=4,
        )
    )
    panels.append(
        stat(
            "OpenRouter This Month (USD)",
            f'last_over_time({PROVIDER_USAGE}{{provider="openrouter",currency="USD",scope="key",period="monthly",job="{job}"}}[3h])',
            "Usage of the GLOBAL key in the current calendar month, as the provider reports it. "
            "The provider's own window, not a rolling thirty days.",
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
            "OpenRouter usage trend (USD)",
            [
                target(
                    f'{PROVIDER_USAGE}{{provider="openrouter",currency="USD",scope="account",period="total",job="{job}"}}',
                    legend="account usage $",
                ),
                target(
                    f'{CREDIT}{{provider="openrouter",currency="USD",scope="account",period="total",job="{job}"}}',
                    legend="credits purchased $",
                ),
                target(
                    f'{BALANCE}{{provider="openrouter",currency="USD",scope="account",period="total",job="{job}"}}',
                    legend="remaining $",
                ),
            ],
            "Provider-reported cumulative usage, purchased credit and remaining balance for the "
            "GLOBAL key's account. Usage climbing toward the credit line is the burn; the gap "
            "between them is what is left.",
            0,
            y,
            w=12,
            h=8,
            unit="currencyUSD",
        )
    )
    panels.append(
        timeseries(
            "OpenRouter requests 24h by model (rolling)",
            [
                target(
                    f'sum by (model) (increase({REQUESTS}{{provider=~"openrouter.*",job="{job}"}}[24h]))',
                    legend="{{model}}",
                ),
            ],
            "Requests routed to OpenRouter in the trailing 24 hours, by the model that served "
            "them. Covers BOTH OpenRouter credentials the deployment holds, because this is "
            "served traffic rather than account state. Served usage is attributed to the "
            "client that served it, which is a different label from the financial account.",
            12,
            y,
            w=12,
            h=8,
            unit="short",
        )
    )
    y += 8

    panels.append(
        timeseries(
            "OpenRouter tokens 24h by direction (rolling)",
            [
                target(
                    f'sum by (component) (increase({TOKENS}{{provider=~"openrouter.*",job="{job}",component!="cached"}}[24h]))',
                    legend="{{component}}",
                ),
            ],
            "Tokens routed to OpenRouter in the trailing 24 hours, by component. Cached is a "
            "subset of prompt, so it is excluded from this stack to keep the totals true. This "
            "covers both OpenRouter credentials.",
            0,
            y,
            w=24,
            h=8,
            unit="short",
        )
    )
    y += 8

    panels.append(
        text(
            "OPENROUTER — TWO ACCOUNTS, SHOWN SEPARATELY",
            "This deployment holds two OpenRouter credentials belonging to **two different "
            "financial accounts**, each with its own purchased credit. The panels above and "
            "below name the account they cover, and the two accounts are never added together: "
            "their balances are separate money, and a single combined figure would describe no "
            "account that exists.\n\n"
            f"**{openrouter_account}** - the account the client named after the provider"
            " reads.\n"
            f"**{openrouter_secondary}** - a second account read with its own credential, "
            "serving its own traffic.\n\n"
            "Switchyard reads each account with the credential it already serves that "
            "account's traffic with. Where the provider states which account a key belongs "
            "to, the two readings are also checked against each other: if both credentials "
            "turned out to resolve to one account, it is read once and published once, so "
            "the same money is never counted twice.",
            0,
            y,
            w=24,
            h=8,
        )
    )
    y += 8

    # The second account, in the same shape as the first, so the two can be compared
    # directly and neither is hidden behind a sum.
    panels.append(
        stat(
            f"{openrouter_secondary} Status",
            f'last_over_time({FRESHNESS}{{provider="{openrouter_secondary}",job="{job}"}}[3h])',
            "Seconds since Switchyard last read the second OpenRouter account successfully. A "
            "value near the polling interval is healthy; one that keeps climbing means that "
            "account is not being read, and its figures below are stale.",
            0,
            y,
            w=6,
            unit="s",
            decimals=0,
            thresholds={
                "mode": "absolute",
                "steps": [
                    {"color": "green", "value": None},
                    {"color": "orange", "value": 900},
                    {"color": "red", "value": 3600},
                ],
            },
        )
    )
    panels.append(
        stat(
            f"{openrouter_secondary} Remaining (USD)",
            f'last_over_time({BALANCE}{{provider="{openrouter_secondary}",currency="USD",scope="account",period="total",job="{job}"}}[3h])',
            "Credit left on the second OpenRouter account: purchased less consumed. This is "
            "that account's money alone, and is not combined with the other account's balance.",
            6,
            y,
            w=6,
            unit="currencyUSD",
            decimals=4,
        )
    )
    panels.append(
        stat(
            f"{openrouter_secondary} Purchased Credit (USD)",
            f'last_over_time({CREDIT}{{provider="{openrouter_secondary}",currency="USD",scope="account",period="total",job="{job}"}}[3h])',
            "Total credit purchased on the second account, as the provider reports it. The "
            "difference between this and its usage is its remaining balance.",
            12,
            y,
            w=6,
            unit="currencyUSD",
            decimals=4,
        )
    )
    panels.append(
        stat(
            f"{openrouter_secondary} Usage (total, USD)",
            f'last_over_time({PROVIDER_USAGE}{{provider="{openrouter_secondary}",currency="USD",scope="account",period="total",job="{job}"}}[3h])',
            "Lifetime usage on the second account: what has been consumed, not what is left.",
            18,
            y,
            w=6,
            unit="currencyUSD",
            decimals=4,
        )
    )
    y += 5

    panels.append(row("PROVIDER ACCOUNT COVERAGE", y))
    y += 1

    panels.append(
        stat(
            "Account Reads OK (total)",
            f'sum({ACCOUNT_FETCH}{{job="{job}",outcome="ok"}})',
            "Successful provider account reads since the deployment started. One per provider "
            "per polling interval, so this counts observations, not accounts.",
            0,
            y,
            w=6,
        )
    )
    panels.append(
        stat(
            "Auth Failures (total)",
            f'sum({AUTH_FAILURES}{{job="{job}"}})',
            "Account reads rejected with 401 or 403. A non-zero value means the credential was "
            "refused for the account surface, which is a different fault from the provider being "
            "down. This panel is empty when no read has been rejected, which is the healthy state.",
            6,
            y,
            w=6,
            thresholds={
                "mode": "absolute",
                "steps": [
                    {"color": "green", "value": None},
                    {"color": "red", "value": 1},
                ],
            },
        )
    )
    panels.append(
        stat(
            "Account Reads Failed (total)",
            f'sum({ACCOUNT_FETCH}{{job="{job}",outcome="failed"}})',
            "Account reads that reached nothing usable: an unreachable host, an unexpected "
            "status, or a body that carried no account state.",
            12,
            y,
            w=6,
            thresholds={
                "mode": "absolute",
                "steps": [
                    {"color": "green", "value": None},
                    {"color": "orange", "value": 1},
                ],
            },
        )
    )
    panels.append(
        stat(
            "Providers With Account Coverage",
            f'count(count by (provider) ({FRESHNESS}{{job="{job}"}}))',
            "How many providers Switchyard holds account figures for.",
            18,
            y,
            w=6,
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
                    f'sum by (provider) ({CREDIT}{{job="{job}",scope="account",period="total"}})',
                    legend="{{provider}}",
                    ref="C",
                    instant=True,
                ),
            ],
            "Which accounts this deployment can show the money side of, read column by "
            "column: a row carrying prompt tokens but no freshness value is serving traffic "
            "whose account credit, balance and usage are not collected at all. Both OpenRouter "
            "accounts now appear, each under its own label, because each is a separate "
            "financial account. They are never summed: a combined figure would describe no "
            "account that exists.\n\n"
            "One caveat when reading the rows: the OpenAI account appears twice under two "
            "names, because the deployment runs two OpenAI clients on one credential. Served "
            "usage is labelled openai_nt and account state is labelled openai, so one account "
            "legitimately occupies two rows with the figures split between them. That is a "
            "naming split, not a second gap. A blank cell is absence of collection, never a "
            "measured zero.",
            0,
            y,
            w=24,
            h=9,
        )
    )
    y += 9

    # The money side, one table per figure so that currency, scope and period
    # each stay a column of their own. Summing across them would add CNY to USD
    # and an account to a key.
    panels.append(
        table(
            "PROVIDER-REPORTED USAGE BY PROVIDER AND CURRENCY",
            [
                target(
                    f"{PROVIDER_USAGE}{{job=\"{job}\",scope=\"account\",period=\"total\"}}",
                    legend="{{provider}}",
                    ref="A",
                    instant=True,
                ),
            ],
            "What each provider says has been consumed on the account, in that provider's own "
            "currency, as a lifetime figure. This is a provider report, not an estimate. The "
            "currency is a column, so figures in CNY and USD are never added together.",
            0,
            y,
            w=8,
            h=8,
        )
    )
    panels.append(
        table(
            "PURCHASED CREDIT BY PROVIDER AND CURRENCY",
            [
                target(
                    f"{CREDIT}{{job=\"{job}\",scope=\"account\",period=\"total\"}}",
                    legend="{{provider}}",
                    ref="A",
                    instant=True,
                ),
            ],
            "Credit the provider sold to this account, lifetime, in each provider's currency. "
            "This is what was bought, not what remains: see the balance table for what is left.",
            8,
            y,
            w=8,
            h=8,
        )
    )
    panels.append(
        table(
            "REMAINING BALANCE BY PROVIDER AND CURRENCY",
            [
                target(
                    f"{BALANCE}{{job=\"{job}\",scope=\"account\",period=\"total\"}}",
                    legend="{{provider}}",
                    ref="A",
                    instant=True,
                ),
            ],
            "What is left on each account, in that provider's currency. Read against the "
            "purchased-credit table: remaining is not the same figure as credit bought, and "
            "neither is the same as usage reported. A provider that does not publish a balance "
            "simply has no row.\n\n"
            "`scope` is shown because the account and the API-key vantage points describe the "
            "same money. Only `account` rows are selected here; summing account and key rows "
            "would count one balance twice.",
            16,
            y,
            w=8,
            h=8,
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
            "**Reasoning tokens are reported separately** and their relationship to "
            "completion tokens varies by provider and model. They are not added again when "
            "calculating total tokens. Switchyard's recorded total is prompt plus completion. "
            "Provider-reported reasoning may not be a strict subset of completion.\n\n"
            "An absent reasoning series means the provider does not report it; it is not a "
            "measured zero. Unlike the cache inside the prompt total, reasoning is not assumed "
            "to sit inside any other figure — it is shown beside completion and never summed "
            "with it.\n\n"
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
            "**OpenRouter holds two financial accounts, and both are shown.** This deployment "
            "holds two OpenRouter credentials belonging to two separate accounts, each with "
            "its own purchased credit. Each account's balance, credit and usage appear under "
            "their own labels and are never added together, because their money is separate. "
            "Check both before drawing a conclusion about OpenRouter spending. A missing "
            "figure is absent, never a measured zero.\n\n"
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

    panels.append(
        text(
            "NOT MEASURABLE HERE — WHAT THE OLD EXPORTER COMPUTED AND THIS ONE CANNOT",
            "These figures appeared on the predecessor dashboards and are **absent here**, "
            "because their inputs lived inside the retired localclaw economics exporter's "
            "process — a balance ledger and a Python rate card — and no metric carries them. "
            "Each is listed so the gap is stated rather than silently dropped.\n\n"
            "| figure | why it cannot be derived |\n"
            "|---|---|\n"
            "| Savings today vs a pre-smart baseline | needs a validated baseline assumption "
            "set; the exporter held one, Switchyard does not |\n"
            "| Counterfactual cost (upper bound) | same baseline assumption set |\n"
            "| Cost by model, today and lifetime | `switchyard_economics_*` money gauges carry "
            "currency, scope, period and provider — never a model — so no per-model money "
            "figure exists in the telemetry |\n"
            "| DeepSeek top-up ledger, lifetime spend | a balance *level* is not a ledger; a "
            "top-up and a day of usage are indistinguishable from two readings |\n"
            "| Request-derived actual cost | requires the exporter's rate card, which was not a "
            "metric |\n"
            "| Exporter uptime, rate card and feature provenance | the exporter no longer runs |\n"
            "| Domain observability (cloud · local · free · unattributed) | the classifier and "
            "its authority pool lived in the exporter |\n\n"
            "**No substitute is shown.** A panel that looked like the original while measuring "
            "something else would be worse than an absent one. Per-model cost in particular is "
            "not synthesised from token counts: that would be a rate-card *estimate* presented "
            "as provider-reported *actual* cost, and the two are not interchangeable. Any future "
            "per-model cost needs its own instrumentation and must label which of the two it is.",
            0,
            y,
            w=24,
            h=17,
        )
    )
    y += 17

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


def build_request(config, expected_version=None):
    """Builds the publishing request for the dashboard document.

    The publishing tool accepts one request shape:
      {op, uid, message, dashboard, folder_uid, expected_version}
    and refuses any other field. A raw Grafana POST body carries "overwrite",
    which the tool rejects as an unsupported field, and omits "op" and "uid",
    which the tool requires. The tool builds the upstream payload itself,
    including overwrite=True, so the request must not carry it.

    A first publication creates the dashboard, and the tool requires that
    request to omit expected_version. Publishing a revision is an update, which
    the tool refuses unless expected_version names the version currently live -
    that check is what stops a stale request from overwriting someone else's
    change.
    """
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
    parser.add_argument(
        "--expected-version",
        type=int,
        default=None,
        help=(
            "The version currently live, to publish this build as an update. "
            "Omit for a first publication, which creates the dashboard."
        ),
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    request = build_request(config, expected_version=args.expected_version)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(request, handle, indent=1)
    action = request["op"]
    detail = f", expected_version={request['expected_version']}" if action == "update" else ""
    print(f"wrote {args.out}: {len(request['dashboard']['panels'])} panels, op={action}{detail}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
