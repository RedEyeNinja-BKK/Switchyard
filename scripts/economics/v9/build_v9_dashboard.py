# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build SWITCHYARD ECONOMICS & TOKEN ACCOUNTING V9.

WHAT THIS DASHBOARD IS FOR
  Monitoring the provider accounts: what each one still holds, what it has
  consumed, when its allowance refills, and how much money is exposed. Routing
  economics and error classification are on the same dashboard because they
  explain those figures, but they come after the account sections, not before.

WHY IT QUERIES ONLY NATIVE METRICS
  A previous dashboard sat beside an external exporter and inherited panels that
  read it. Those panels answered what another process believed about usage and
  were visually indistinguishable from genuinely measured ones. Every family here
  is emitted by the switchyard-server binary itself and is checked against
  native_contract.ALLOWED at build time; an unreviewed family is a build failure,
  not a blank panel.

THE FOUR THINGS THAT MAKE A PROVIDER PANEL WRONG
  1. CURRENCY. A provider can publish a real balance in one currency and a
     zero-valued series in another. DeepSeek does exactly that: CNY 91.81 is real
     and its USD series is 0 because DeepSeek keeps no USD balance. A panel
     reading the USD series would show a confident, measured-looking "$0.00" for
     an account holding 91.81 CNY. Every money query here pins its currency.
  2. SCOPE. One balance is published under scope="account" and again under
     scope="key". They describe the same money from two vantage points; adding
     them double counts. Panels pin one scope or name the scope per row.
  3. refId. Grafana keys query results by refId. Two targets sharing one refId in
     a single panel collide, and the panel renders "No data" with a warning even
     though both expressions are valid and both have data. This is the defect
     that broke the allowance-history and usage-trend charts in the previous
     dashboard; every target here gets a distinct refId, and the verifier refuses
     a duplicate.
  4. CUMULATIVE SEMANTICS. Account usage and balance are gauges sampled at each
     poll, not counters. A delta over a window is `last_over_time - first_over_time`
     and may legitimately be negative after a top-up or a provider correction.
     Clamping it to zero would turn a replenishment into a measured zero spend.
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

# --- accounting ------------------------------------------------------------
TOK = "switchyard_economics_tokens_total"
REQ = "switchyard_economics_requests_total"
RTOK = "switchyard_economics_rolling_tokens"
RREQ = "switchyard_economics_rolling_requests"
RCOST = "switchyard_economics_rolling_provider_cost"
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

# --- execution -------------------------------------------------------------
ATTEMPTS = "switchyard_upstream_attempts_total"
CLIENTOUT = "switchyard_client_responses_total"
CALLS = "switchyard_llm_calls_total"
COMPATT = "switchyard_composition_attempt_total"
COMPCROSS = "switchyard_composition_cross_total"
RETRYREC = "switchyard_router_retry_recovered_total"
PTOK = "switchyard_prompt_tokens_total"
CTOKC = "switchyard_completion_tokens_total"
RTOKC = "switchyard_reasoning_tokens_total"
CACHED = "switchyard_cached_tokens_total"
CACHECREATE = "switchyard_cache_creation_tokens_total"
RUNS = "switchyard_runs_total"

# Thresholds. Remaining allowance is read the way an operator reads a fuel gauge:
# the number only becomes interesting as it approaches empty.
REMAINING_THRESHOLDS = {
    "mode": "absolute",
    "steps": [
        {"color": "red", "value": None},
        {"color": "orange", "value": 10},
        {"color": "green", "value": 25},
    ],
}
CONSUMED_THRESHOLDS = {
    "mode": "absolute",
    "steps": [
        {"color": "green", "value": None},
        {"color": "orange", "value": 60},
        {"color": "red", "value": 85},
    ],
}
LOW_CREDIT_THRESHOLDS = {
    "mode": "absolute",
    "steps": [
        {"color": "red", "value": None},
        {"color": "orange", "value": 1},
        {"color": "green", "value": 5},
    ],
}


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

    def acc(metric, provider, currency=None, scope="account", period="total", **extra):
        """Builds a provider-account selector.

        Written as a helper rather than inline f-strings because these selectors
        carry five or six label matchers each and a single missing quote produces
        a selector that silently matches nothing. Naming the dimensions also makes
        the two easy mistakes visible at the call site: a money selector with no
        currency matches every currency the provider publishes, and one with no
        scope matches both the account view and the key view of the same balance.
        """
        parts = [jf, f'provider="{provider}"']
        if currency is not None:
            parts.append(f'currency="{currency}"')
        if scope is not None:
            parts.append(f'scope="{scope}"')
        if period is not None:
            parts.append(f'period="{period}"')
        for key, value in extra.items():
            parts.append(f'{key}="{value}"')
        return f"{metric}{{" + ",".join(parts) + "}"

    # ================================================== 1 PROVIDER OVERVIEW
    add(row("1  PROVIDER ACCOUNT OVERVIEW", y))
    adv(1)

    add(
        stat(
            "OPENROUTER REMAINING (GLOBAL ACCOUNT)",
            f"last_over_time({BALANCE}{{{jf},provider=\"openrouter\",currency=\"USD\","
            'scope="account",period="total"}[3h])',
            "What the global OpenRouter account has left to spend, in USD, as OpenRouter "
            "itself reports it. This is the account balance, not a key's share of it. Red "
            "below $1: at that point the next few requests decide whether the account runs "
            "dry, and free-tier slugs would keep working while paid ones start failing.",
            0,
            y,
            unit="currencyUSD",
            decimals=4,
            thresholds=LOW_CREDIT_THRESHOLDS,
        )
    )
    add(
        stat(
            "OPENROUTER REMAINING (STEPHEN ACCOUNT)",
            f"last_over_time({BALANCE}{{{jf},provider=\"openrouter_stephen\",currency=\"USD\","
            'scope="account",period="total"}[3h])',
            "The second OpenRouter account, a separate billing identity with its own credit. "
            "Shown on its own because one account running dry does not stop the other from "
            "serving, and adding the two would hide which one is nearly empty.",
            6,
            y,
            unit="currencyUSD",
            decimals=4,
            thresholds=LOW_CREDIT_THRESHOLDS,
        )
    )
    add(
        stat(
            "DEEPSEEK BALANCE (CNY)",
            f"last_over_time({BALANCE}{{{jf},provider=\"deepseek\",currency=\"CNY\","
            'scope="account",period="total"}[3h])',
            "DeepSeek's balance in CNY, the only currency it holds. DeepSeek also publishes a "
            "USD series that is structurally zero — it keeps no USD balance — so that series "
            "is deliberately not shown as though it were a measured dollar amount.",
            12,
            y,
            unit="currencyCNY",
            decimals=2,
        )
    )
    add(
        stat(
            "OPENAI 5-HR REMAINING",
            f"last_over_time({ALLOW}{{{jf},provider=\"openai\",pool=\"primary\"}}[3h])",
            "Remaining percentage of the primary five-hour allowance. This is a rate window "
            "that refills on a clock, not money: running it low throttles throughput without "
            "costing anything, and waiting restores it.",
            18,
            y,
            unit="percent",
            decimals=0,
            thresholds=REMAINING_THRESHOLDS,
        )
    )
    adv(5)

    add(
        table(
            "EVERY PROVIDER ACCOUNT — BALANCE, CREDIT, USAGE, READ INTERVAL",
            [
                target(
                    f'{BALANCE}{{{jf},scope="account",period="total"}}',
                    "balance {{provider}} {{currency}}",
                    instant=True,
                    ref="A",
                ),
                target(
                    f'{CREDIT}{{{jf},scope="account",period="total"}}',
                    "credit {{provider}} {{currency}}",
                    instant=True,
                    ref="B",
                ),
                target(
                    f'{USAGE}{{{jf},scope="account",period="total"}}',
                    "usage {{provider}} {{currency}}",
                    instant=True,
                    ref="C",
                ),
                target(
                    f"last_over_time({FRESH}{{{jf}}}[3h])",
                    "read interval {{provider}}",
                    instant=True,
                    ref="D",
                ),
            ],
            "One row per provider account per currency. Balance is what the provider says is "
            "left; credit is what has been purchased; usage is cumulative consumption. They are "
            "three different questions and are not summed, because a provider may publish "
            "several currencies and only one of them is real money. The read interval is how "
            "often that provider's account state is refreshed — see section 7 for why it is not "
            "a health signal on its own.",
            0,
            y,
            h=9,
        )
    )
    add(
        table(
            "ACCOUNT READ HEALTH",
            [
                target(
                    f"sum by (provider, outcome) ({FETCH}{{{jf}}})",
                    "{{provider}} {{outcome}}",
                    instant=True,
                    ref="A",
                ),
                target(
                    f"sum by (provider) ({FETCH}{{{jf}}}) - "
                    f'sum by (provider) ({FETCH}{{{jf},outcome="ok"}})',
                    "failed {{provider}}",
                    instant=True,
                    ref="B",
                ),
                target(
                    f"max by (provider) ({FETCHDUR}{{{jf}}})",
                    "fetch seconds {{provider}}",
                    instant=True,
                    ref="C",
                ),
            ],
            "Whether the account figures are still being refreshed. This is the health signal "
            "the read interval cannot provide: a provider whose reads have stopped keeps "
            "publishing its last known balance, which looks current while being an old "
            "observation. Rising `failed` with a flat `ok` is that condition.",
            12,
            y,
            h=9,
        )
    )
    adv(9)

    # ============================================ 2 OPENAI ALLOWANCE WINDOWS
    add(row("2  OPENAI ALLOWANCE WINDOWS  (primary 5-hr · secondary weekly)", y))
    adv(1)

    add(
        stat(
            "5-HR REMAINING",
            f"last_over_time({ALLOW}{{{jf},provider=\"openai\",pool=\"primary\"}}[3h])",
            "Percentage of the five-hour allowance still available.",
            0,
            y,
            w=4,
            unit="percent",
            decimals=0,
            thresholds=REMAINING_THRESHOLDS,
        )
    )
    add(
        stat(
            "5-HR USED",
            f"(100 - last_over_time({ALLOW}{{{jf},provider=\"openai\",pool=\"primary\"}}[3h]))",
            "Percentage of the five-hour allowance consumed. Used and remaining are "
            "complements and always total 100.",
            4,
            y,
            w=4,
            unit="percent",
            decimals=0,
            thresholds=CONSUMED_THRESHOLDS,
        )
    )
    add(
        stat(
            "TIME TO 5-HR RESET",
            f"last_over_time({RESET}{{{jf},provider=\"openai\",pool=\"primary\"}}[3h])",
            "Countdown until the five-hour window refills, as the provider reports it. Rendered "
            "as a duration, so it reads in minutes when close and in hours when not. It falls "
            "toward zero and then jumps back up at the reset.",
            8,
            y,
            w=4,
            unit="s",
            decimals=0,
        )
    )
    add(
        stat(
            "WEEKLY REMAINING",
            f"last_over_time({ALLOW}{{{jf},provider=\"openai\",pool=\"secondary\"}}[3h])",
            "Percentage of the weekly allowance still available. This window refills on a "
            "different schedule from the five-hour one, which is why the two are never combined "
            "into a single figure.",
            12,
            y,
            w=4,
            unit="percent",
            decimals=0,
            thresholds=REMAINING_THRESHOLDS,
        )
    )
    add(
        stat(
            "WEEKLY USED",
            f"(100 - last_over_time({ALLOW}{{{jf},provider=\"openai\",pool=\"secondary\"}}[3h]))",
            "Percentage of the weekly allowance consumed.",
            16,
            y,
            w=4,
            unit="percent",
            decimals=0,
            thresholds=CONSUMED_THRESHOLDS,
        )
    )
    add(
        stat(
            "TIME TO WEEKLY RESET",
            f"last_over_time({RESET}{{{jf},provider=\"openai\",pool=\"secondary\"}}[3h])",
            "Countdown until the weekly window refills. Rendered as a duration, so a figure "
            "near a week reads in days and hours rather than as a raw second count.",
            20,
            y,
            w=4,
            unit="s",
            decimals=0,
        )
    )
    adv(5)

    add(
        timeseries(
            "5-HR WINDOW — USED AND REMAINING",
            [
                target(
                    f'{ALLOW}{{{jf},provider="openai",pool="primary"}}',
                    "remaining %",
                    ref="A",
                ),
                target(
                    f'(100 - {ALLOW}{{{jf},provider="openai",pool="primary"}})',
                    "used %",
                    ref="B",
                ),
            ],
            "The five-hour window over time. Used and remaining are complementary, so the two "
            "lines always total 100 and cross at the halfway point; a rise in one is a fall in "
            "the other, not an independent event. Each downward slope is consumption and each "
            "vertical recovery is the window refilling.",
            0,
            y,
        )
    )
    add(
        timeseries(
            "WEEKLY WINDOW — USED AND REMAINING",
            [
                target(
                    f'{ALLOW}{{{jf},provider="openai",pool="secondary"}}',
                    "remaining %",
                    ref="A",
                ),
                target(
                    f'(100 - {ALLOW}{{{jf},provider="openai",pool="secondary"}})',
                    "used %",
                    ref="B",
                ),
            ],
            "The weekly window over time, on its own refill schedule. A slower, deeper slope "
            "here than in the five-hour window is the expected shape: the same traffic spread "
            "across a much larger budget.",
            12,
            y,
        )
    )
    adv(9)

    add(
        timeseries(
            "RESET COUNTDOWNS OVER TIME",
            [
                target(
                    f'last_over_time({RESET}{{{jf},provider="openai",pool="primary"}}[3h])',
                    "5-hr to reset",
                    ref="A",
                ),
                target(
                    f'last_over_time({RESET}{{{jf},provider="openai",pool="secondary"}}[3h])',
                    "weekly to reset",
                    ref="B",
                ),
            ],
            "Both countdowns on one axis, in seconds. Each is a sawtooth: a long fall as the "
            "window is consumed, then a jump back to the full period at the reset. The jump is "
            "when capacity returns.",
            0,
            y,
        )
    )
    add(
        text(
            "WHY THESE ARE ALLOWANCES AND NOT SPEND",
            "The OpenAI figures on this dashboard are **rate windows**, not money. A five-hour "
            "window at 18% remaining does not mean 82% of a budget has been spent and lost — it "
            "means throughput is currently throttled, and waiting restores it at no cost. That "
            "is why they are not added to the dollar figures anywhere on this dashboard, and "
            "why a low reading is a routing consideration rather than a financial event.\n\n"
            "The two windows refill independently. A request can be refused by a nearly empty "
            "five-hour window while the weekly window is barely touched, so they are shown "
            "separately and never averaged.",
            12,
            y,
            w=12,
        )
    )
    adv(9)

    # ================================================== 3 DEEPSEEK TELEMETRY
    add(row("3  DEEPSEEK TELEMETRY", y))
    adv(1)

    ds_bal = acc(BALANCE, "deepseek", "CNY")
    ds_top = acc(BALCOMP, "deepseek", "CNY", component="topped_up")
    ds_grant = acc(BALCOMP, "deepseek", "CNY", component="granted")

    add(
        stat(
            "ACCOUNT READ INTERVAL",
            f"last_over_time({acc(FRESH, 'deepseek', None, None, None)}[3h])",
            "Seconds between successful DeepSeek account reads - the polling cadence, not a "
            "health status. It is recorded only when a read succeeds, so a stalled provider "
            "leaves the last value standing rather than growing. Read health lives in the "
            "ACCOUNT READ HEALTH table and in section 7.",
            0,
            y,
            unit="s",
            decimals=0,
        )
    )
    add(
        stat(
            "BALANCE (CNY)",
            f"last_over_time({ds_bal}[3h])",
            "DeepSeek's remaining balance in CNY - the only currency this account holds.",
            6,
            y,
            unit="currencyCNY",
            decimals=2,
        )
    )
    add(
        stat(
            "24H NET BALANCE CHANGE (CNY)",
            f"last_over_time({ds_bal}[24h]) - first_over_time({ds_bal}[24h])",
            "Net movement of the balance over 24h, signed. POSITIVE means the balance rose, "
            "which is a top-up exceeding consumption - not spending. NEGATIVE means consumption "
            "exceeded any top-up, which is the reading that represents money actually used. "
            "Deliberately not clamped at zero: clamping would report a top-up as no change and "
            "erase the very event an operator needs to see.",
            12,
            y,
            unit="currencyCNY",
            decimals=2,
        )
    )
    add(
        stat(
            "24H BALANCE RANGE (CNY)",
            f"max_over_time({ds_bal}[24h]) - min_over_time({ds_bal}[24h])",
            "Highest minus lowest balance in the window - how far the balance travelled, "
            "regardless of direction. This is what a top-up followed by spending looks like: a "
            "large range with a small net change. Shown beside the net figure because the two "
            "answer different questions.",
            18,
            y,
            unit="currencyCNY",
            decimals=2,
        )
    )
    adv(5)

    add(
        stat(
            "REMAINING — TOPPED-UP (CNY)",
            f"last_over_time({ds_top}[3h])",
            "The portion of the balance that came from payments. DeepSeek splits its balance "
            "into purchased and granted credit, and the split matters because granted credit "
            "typically expires while purchased credit does not.",
            0,
            y,
            unit="currencyCNY",
            decimals=2,
        )
    )
    add(
        stat(
            "REMAINING — GRANTED (CNY)",
            f"last_over_time({ds_grant}[3h])",
            "The portion of the balance granted rather than purchased. Usually expires, so it "
            "is worth less than the same amount of topped-up credit.",
            6,
            y,
            unit="currencyCNY",
            decimals=2,
        )
    )
    add(
        stat(
            "REMAINING — TOTAL (CNY)",
            f"sum(last_over_time({acc(BALCOMP, 'deepseek', 'CNY')}[3h]))",
            "Topped-up plus granted, computed here from the components rather than read from a "
            "second gauge. It must equal the balance panel; a divergence means the provider "
            "changed how it splits the balance and the components no longer account for it.",
            12,
            y,
            unit="currencyCNY",
            decimals=2,
        )
    )
    add(
        stat(
            "LOWEST BALANCE IN 24H (CNY)",
            f"min_over_time({ds_bal}[24h])",
            "The low-water mark of the balance over 24h. This is the figure that says how close "
            "the account came to empty, which the current reading can hide if a top-up arrived "
            "in the last few minutes.",
            18,
            y,
            unit="currencyCNY",
            decimals=2,
        )
    )
    adv(5)

    add(
        timeseries(
            "BALANCE TREND (CNY)",
            [target(f"{ds_bal}", "balance ¥", ref="A")],
            "The balance as sampled at each account read. Gaps in this line are missing "
            "telemetry, not zero balances: the series simply has no sample there, and the line "
            "is drawn broken rather than interpolated across them. A vertical jump upward is a "
            "top-up; a steady decline is consumption.",
            0,
            y,
        )
    )
    add(
        timeseries(
            "BALANCE COMPONENTS OVER TIME (CNY)",
            [
                target(f"last_over_time({ds_top}[3h])", "topped-up ¥", ref="A"),
                target(f"last_over_time({ds_grant}[3h])", "granted ¥", ref="B"),
            ],
            "How the balance splits between purchased and granted credit. Granted credit "
            "expiring shows as a step down in the granted line that the topped-up line does not "
            "share - which is why the total can fall without any request having been served.",
            12,
            y,
        )
    )
    adv(9)

    add(
        text(
            "DEEPSEEK — WHAT IS MEASURED, AND WHAT THE CURVE DOES NOT SAY",
            "**CNY only.** DeepSeek keeps no dollar balance. It publishes a USD series that is "
            "structurally zero, and a panel reading it would show a confident `$0.00` for an "
            "account holding real money. That series is withheld here rather than displayed, "
            "because a zero that means \"this currency is not used\" is indistinguishable on a "
            "dashboard from a zero that means \"this account is empty\".\n\n"
            "**Movement is not spending.** The 24h net change is signed. A positive value is a "
            "top-up larger than the consumption that followed; only a negative value measures "
            "money actually used. The range panel sits beside it because a top-up and a day of "
            "spending can cancel out to almost no net change while the balance has travelled a "
            "long way.\n\n"
            "**Gaps are gaps.** The trend line is drawn with breaks, not interpolated. A long "
            "horizontal stretch means the account was not being read, which is a telemetry "
            "problem, not a frozen balance.",
            0,
            y,
            w=24,
            h=7,
        )
    )
    adv(7)


    # ============================================ 4 OPENROUTER USAGE TELEMETRY
    add(row("4  OPENROUTER USAGE TELEMETRY", y))
    adv(1)

    add(
        stat(
            "GLOBAL — READ INTERVAL",
            f"last_over_time({FRESH}{{{jf},provider=\"openrouter\"}}[3h])",
            "Polling cadence for the global account. Recorded only on a successful read, so it "
            "does not grow when reads start failing.",
            0,
            y,
            w=4,
            unit="s",
            decimals=0,
        )
    )
    add(
        stat(
            "GLOBAL — REMAINING (USD)",
            f"last_over_time({BALANCE}{{{jf},provider=\"openrouter\",currency=\"USD\","
            'scope="account",period="total"}[3h])',
            "Remaining account balance. Shown to four decimals because the number that matters "
            "here is often a few cents — rounding to two would render a nearly-empty account as "
            "indistinguishable from an empty one.",
            4,
            y,
            w=4,
            unit="currencyUSD",
            decimals=4,
            thresholds=LOW_CREDIT_THRESHOLDS,
        )
    )
    add(
        stat(
            "GLOBAL — TOTAL USAGE (USD)",
            f"last_over_time({USAGE}{{{jf},provider=\"openrouter\",currency=\"USD\","
            'scope="account",period="total"}[3h])',
            "Cumulative consumption on the account since it was created. This is a lifetime "
            "figure and never decreases; it is not this month's spend, which is the panel near "
            "the end of this section.",
            8,
            y,
            w=4,
            unit="currencyUSD",
            decimals=4,
        )
    )
    add(
        stat(
            "GLOBAL — PURCHASED CREDITS (USD)",
            f"last_over_time({CREDIT}{{{jf},provider=\"openrouter\",currency=\"USD\","
            'scope="account",period="total"}[3h])',
            "Total credit ever purchased for the account. Together with remaining balance it "
            "closes the accounting: purchased minus remaining should be the total usage.",
            12,
            y,
            w=4,
            unit="currencyUSD",
            decimals=4,
        )
    )
    add(
        stat(
            "GLOBAL — 24H SPEND (USD)",
            f"last_over_time({USAGE}{{{jf},provider=\"openrouter\",currency=\"USD\","
            'scope="account",period="total"}[24h]) - '
            f"first_over_time({USAGE}{{{jf},provider=\"openrouter\",currency=\"USD\","
            'scope="account",period="total"}[24h])',
            "Cumulative usage now minus cumulative usage 24h ago — what was actually consumed "
            "in the window. Signed on purpose: a provider correction or reset that lowers the "
            "cumulative figure produces a negative value, which is the truth about the series "
            "rather than a fault to clamp away.",
            16,
            y,
            w=4,
            unit="currencyUSD",
            decimals=4,
        )
    )
    add(
        stat(
            "GLOBAL — KEY USAGE (USD, CUMULATIVE)",
            f"last_over_time({USAGE}{{{jf},provider=\"openrouter\",currency=\"USD\","
            'scope="key",period="total"}[3h])',
            "Usage attributed to this key rather than the whole account. Lower than account "
            "usage when other keys also draw on the same credit — which is the normal case and "
            "not a discrepancy.",
            20,
            y,
            w=4,
            unit="currencyUSD",
            decimals=4,
        )
    )
    adv(5)

    add(
        stat(
            "GLOBAL — THIS MONTH (USD)",
            f"last_over_time({USAGE}{{{jf},provider=\"openrouter\",currency=\"USD\","
            'scope="key",period="monthly"}[3h])',
            "Consumption in the current calendar month, as the provider resets it. Distinct "
            "from total usage: this is the figure that matters against a monthly budget.",
            0,
            y,
            w=4,
            unit="currencyUSD",
            decimals=4,
        )
    )
    add(
        stat(
            "GLOBAL — THIS WEEK (USD)",
            f"last_over_time({USAGE}{{{jf},provider=\"openrouter\",currency=\"USD\","
            'scope="key",period="weekly"}[3h])',
            "Consumption in the current week, as the provider resets it. Useful for spotting a "
            "pace change before it shows up in the monthly figure.",
            4,
            y,
            w=4,
            unit="currencyUSD",
            decimals=4,
        )
    )
    add(
        stat(
            "GLOBAL — TODAY (USD)",
            f"last_over_time({USAGE}{{{jf},provider=\"openrouter\",currency=\"USD\","
            'scope="key",period="daily"}[3h])',
            "Consumption so far today. Zero is a real and common reading: it means nothing has "
            "been spent through this key yet in the provider's current day.",
            8,
            y,
            w=4,
            unit="currencyUSD",
            decimals=4,
        )
    )
    add(
        stat(
            "STEPHEN — REMAINING (USD)",
            f"last_over_time({BALANCE}{{{jf},provider=\"openrouter_stephen\",currency=\"USD\","
            'scope="account",period="total"}[3h])',
            "Remaining balance on the second OpenRouter account. A separate billing identity, "
            "not a view of the first: its credit is bought separately and runs out separately.",
            12,
            y,
            w=4,
            unit="currencyUSD",
            decimals=4,
            thresholds=LOW_CREDIT_THRESHOLDS,
        )
    )
    add(
        stat(
            "STEPHEN — PURCHASED CREDITS (USD)",
            f"last_over_time({CREDIT}{{{jf},provider=\"openrouter_stephen\",currency=\"USD\","
            'scope="account",period="total"}[3h])',
            "Credit purchased for the Stephen account.",
            16,
            y,
            w=4,
            unit="currencyUSD",
            decimals=4,
        )
    )
    add(
        stat(
            "STEPHEN — TOTAL USAGE (USD)",
            f"last_over_time({USAGE}{{{jf},provider=\"openrouter_stephen\",currency=\"USD\","
            'scope="account",period="total"}[3h])',
            "Cumulative consumption on the Stephen account.",
            20,
            y,
            w=4,
            unit="currencyUSD",
            decimals=4,
        )
    )
    adv(5)

    add(
        timeseries(
            "GLOBAL ACCOUNT — USAGE, CREDIT AND REMAINING",
            [
                target(
                    f'{USAGE}{{{jf},provider="openrouter",currency="USD",scope="account",'
                    'period="total"}',
                    "total usage $",
                    ref="A",
                ),
                target(
                    f'{CREDIT}{{{jf},provider="openrouter",currency="USD",scope="account",'
                    'period="total"}',
                    "credits purchased $",
                    ref="B",
                ),
                target(
                    f'{BALANCE}{{{jf},provider="openrouter",currency="USD",scope="account",'
                    'period="total"}',
                    "remaining $",
                    ref="C",
                ),
            ],
            "Three views of one account. Usage climbs, credit is a step function that rises on "
            "each purchase, and remaining is bought minus consumed. The vertical gap between "
            "the credit line and the usage line is the money still available — the same number "
            "as the remaining panel, read geometrically. A step up in credit is a top-up; the "
            "remaining line should jump with it.",
            0,
            y,
        )
    )
    add(
        timeseries(
            "STEPHEN ACCOUNT — USAGE, CREDIT AND REMAINING",
            [
                target(
                    f'{USAGE}{{{jf},provider="openrouter_stephen",currency="USD",'
                    'scope="account",period="total"}',
                    "total usage $",
                    ref="A",
                ),
                target(
                    f'{CREDIT}{{{jf},provider="openrouter_stephen",currency="USD",'
                    'scope="account",period="total"}',
                    "credits purchased $",
                    ref="B",
                ),
                target(
                    f'{BALANCE}{{{jf},provider="openrouter_stephen",currency="USD",'
                    'scope="account",period="total"}',
                    "remaining $",
                    ref="C",
                ),
            ],
            "The same three measurements for the Stephen account, kept on their own axis range "
            "because its figures are an order of magnitude smaller and would otherwise look "
            "flat beside the global account.",
            12,
            y,
        )
    )
    adv(9)

    add(
        timeseries(
            "REQUESTS, ROLLING 24H, BY MODEL",
            [
                target(
                    f'sum by (model) ({RREQ}{{{jf},window="24h",component="request",'
                    'provider=~"openrouter.*"})',
                    "{{model}}",
                    ref="A",
                )
            ],
            "What the OpenRouter accounts actually served over the rolling window, per model. "
            "Combined across both accounts because the question is which models the traffic "
            "went to; the per-account split is in the panels above.",
            0,
            y,
        )
    )
    add(
        timeseries(
            "SPEND, ROLLING 24H, BY MODEL (USD)",
            [
                target(
                    f'sum by (model) ({RCOST}{{{jf},window="24h",currency="USD",'
                    'component="provider_reported",provider=~"openrouter.*"})',
                    "{{model}}",
                    ref="A",
                )
            ],
            "Which models are consuming the credit. Only metered traffic appears: free and "
            "subscription-class requests cost nothing per call and are absent by design rather "
            "than missing.",
            12,
            y,
        )
    )
    adv(9)

    add(
        text(
            "TWO ACCOUNTS, FOUR MEASUREMENTS, NONE INTERCHANGEABLE",
            "Both OpenRouter accounts are shown because **they are separate billing identities**. "
            "`openrouter` and `openrouter_stephen` hold separately purchased credit; one running "
            "out does not affect the other, so merging them would hide which one is close to "
            "empty and would misstate the exposure.\n\n"
            "Within one account, four figures are easy to confuse and are deliberately kept "
            "apart: **remaining** (what is left), **purchased credit** (what was ever bought), "
            "**total usage** (lifetime consumption) and **this month** (consumption inside the "
            "provider's current month). Remaining is not the difference between the other two "
            "unless no top-up has ever been partial, and this month is not a fraction of total "
            "usage in any fixed way.\n\n"
            "**Scope is not a second account.** The same balance is published under "
            "`scope=\"account\"` and `scope=\"key\"`. They describe the same money from two "
            "vantage points, so adding them would double count it. Every panel here names the "
            "scope it reads.",
            0,
            y,
            w=24,
            h=7,
        )
    )
    adv(7)

    # ============================================== 5 TOKEN ACCOUNTING
    add(row("5  TOKEN ACCOUNTING AND OBSERVED USAGE", y))
    adv(1)

    add(
        stat(
            "PROMPT TOKENS",
            f"sum({PTOK}{{{jf}}})",
            "Input tokens as provider-counted, since the deployment started. This total already "
            "contains the cached and cache-created tokens, so those are shown beside it and "
            "never added to it.",
            0,
            y,
        )
    )
    add(
        stat(
            "COMPLETION TOKENS",
            f"sum({CTOKC}{{{jf}}})",
            "Output tokens as provider-reported. Reasoning is published separately by the "
            "providers that report it and is not added here, so each provider's own figures are "
            "preserved exactly.",
            6,
            y,
        )
    )
    add(
        stat(
            "REASONING TOKENS",
            f"sum({RTOKC}{{{jf}}})",
            "Reasoning tokens where reported. Its relationship to completion varies by model — "
            "measured across the routing log, some models report reasoning above completion — so "
            "no arithmetic relationship is asserted. Absent means unreported, not measured zero.",
            12,
            y,
        )
    )
    add(
        stat(
            "CACHED PROMPT TOKENS",
            f"sum({CACHED}{{{jf}}})",
            "Prompt tokens served from the provider's cache. Compared with the prompt total, "
            "this is how much input was not re-sent at full price.",
            18,
            y,
        )
    )
    adv(5)

    add(
        barchart(
            "PROMPT TOKENS BY MODEL",
            [target(f"sum by (model) ({PTOK}{{{jf}}})", "{{model}}", ref="A")],
            "Input volume per model — the cost driver that scales with conversation length and "
            "the one routing can most directly move.",
            0,
            y,
        )
    )
    add(
        barchart(
            "OUTPUT TOKENS BY MODEL — COMPLETION AND REASONING",
            [
                target(f'sum by (model) ({CTOKC}{{{jf}}})', "completion {{model}}", ref="A"),
                target(f'sum by (model) ({RTOKC}{{{jf}}})', "reasoning {{model}}", ref="B"),
            ],
            "Output volume per model with reasoning alongside rather than inside completion. "
            "Where both bars exist, the split shows how much of the work was deliberation.",
            12,
            y,
        )
    )
    adv(9)

    add(
        piechart(
            "TOKEN COMPONENT MIX TODAY",
            [target(f"sum by (component) ({CTOK}{{{jf}}})", "{{component}}", ref="A")],
            "Today's tokens split by component, from one series so the whole is the sum of its "
            "parts as the binary recorded them.",
            0,
            y,
            w=8,
        )
    )
    add(
        timeseries(
            "TOKENS, ROLLING 24H, BY COMPONENT",
            [
                target(
                    f'sum by (component) ({RTOK}{{{jf},window="24h"}})',
                    "{{component}}",
                    ref="A",
                )
            ],
            "Rolling 24h tokens by component. A cached share growing while prompt stays flat is "
            "the usual sign that prompt reuse is working.",
            8,
            y,
            w=16,
        )
    )
    adv(9)

    add(
        barchart(
            "REQUESTS BY PROVIDER",
            [target(f'sum by (provider) ({REQ}{{{jf},component="request"}})', "{{provider}}", ref="A")],
            "Where requests were actually served. Observed routing, not configuration: this is "
            "the provider the routing layer chose.",
            0,
            y,
            w=8,
        )
    )
    add(
        barchart(
            "REQUESTS BY MODEL",
            [target(f'sum by (model) ({REQ}{{{jf},component="request"}})', "{{model}}", ref="A")],
            "Requests per model. Read with the error table in section 6: a model carrying a "
            "large share of traffic and a high error rate is an active problem.",
            8,
            y,
            w=8,
        )
    )
    add(
        piechart(
            "COST CLASS MIX",
            [
                target(
                    f'sum by (cost_class) ({REQ}{{{jf}}})',
                    "{{cost_class}}",
                    ref="A",
                )
            ],
            "Requests by economic character. `subscription` costs nothing per call, `metered` "
            "does, and free traffic is neither — so this explains what the spend figures are "
            "made of.",
            16,
            y,
            w=8,
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
            "The routing decision and its consequence, joined on route and model. Cost is a 24h "
            "window rather than a lifetime total because the cumulative cost counter carries no "
            "currency label and cannot be safely filtered to one currency; the windowed family "
            "can. It is absent for subscription traffic, which is a property of the provider "
            "rather than a gap.",
            0,
            y,
            h=10,
        )
    )
    add(
        timeseries(
            "REQUESTS, ROLLING 24H, BY ROUTE FAMILY",
            [
                target(
                    f'sum by (route_family) ({RREQ}{{{jf},window="24h",component="request"}})',
                    "{{route_family}}",
                    ref="A",
                )
            ],
            "Requests by the kind of route addressed — direct, primitive, tier or composite. "
            "Separates deliberate direct calls from traffic that arrived through a routing "
            "layer.",
            12,
            y,
            h=10,
        )
    )
    adv(10)

    # ================================== 6 PROVIDER ERRORS AND ROUTING ECONOMICS
    add(row("6  PROVIDER ERRORS AND ROUTING ECONOMICS", y))
    adv(1)

    add(
        stat(
            "UPSTREAM ATTEMPTS FAILED",
            f'sum({ATTEMPTS}{{{jf},outcome!="ok"}})',
            "Upstream attempts that did not return 2xx. Attempts, not requests: a request that "
            "retried twice and then succeeded contributes two here and nothing to the panel "
            "beside it. Confusing the two makes a healthy deployment look broken.",
            0,
            y,
        )
    )
    add(
        stat(
            "ATTEMPTS RECOVERED BY RETRY",
            f"sum({RETRYREC}{{{jf}}})",
            "Failed attempts that a later attempt served. Upstream errors that never became a "
            "caller-visible failure.",
            6,
            y,
        )
    )
    add(
        stat(
            "REQUESTS THAT REACHED A CALLER AS ERRORS",
            f'sum({CLIENTOUT}{{{jf},outcome=~"retryable_error|other_error"}})',
            "Responses delivered to callers that ended in an error, after every recovery "
            "opportunity was used. The only error count here a user would recognise as their "
            "own failed request.",
            12,
            y,
        )
    )
    add(
        stat(
            "ACCOUNT-WIDE QUOTA BLOCKS",
            f'sum({COMPATT}{{{jf},capability="account_quota"}})',
            "Times the router found an entire account's allowance exhausted. Decided from the "
            "provider's structured error, not from the status code — the semantic signal that "
            "stops a request being retried against a credential already known to be empty.",
            18,
            y,
        )
    )
    adv(5)

    add(
        barchart(
            "UPSTREAM ATTEMPTS BY STATUS AND HTTP-DERIVED OUTCOME",
            [
                target(
                    f"sum by (code, outcome) ({ATTEMPTS}{{{jf}}})",
                    "{{code}} {{outcome}}",
                    ref="A",
                )
            ],
            "Attempts grouped by the status received and the outcome the binary derived from "
            "that code alone. HTTP evidence, not a diagnosis: `429 retryable_error` covers both "
            "a genuine rate limit and an exhausted balance, because the status cannot tell them "
            "apart. The composition panels carry that distinction.",
            0,
            y,
        )
    )
    add(
        timeseries(
            "UPSTREAM ATTEMPTS BY STATUS OVER TIME",
            [
                target(
                    f"sum by (code) (increase({ATTEMPTS}{{{jf}}}[15m]))",
                    "{{code}}",
                    ref="A",
                )
            ],
            "Attempt rate per status. A steady 429 line is pacing pressure retries can absorb; "
            "a 401 or 403 line is a credential or policy problem that cannot be retried away.",
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
                    ref="A",
                )
            ],
            "What the composition layer decided, by the capability it handled. This is the "
            "binary's own semantic classification and the most trustworthy error breakdown "
            "available: `credential_quota` means an allowance was known exhausted before a call "
            "was spent; `account_quota` means an account-wide limit was hit.",
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
            "Requests that left one child route for another, with the classified reason. A "
            "large `account_quota` count against one child means that child's credential is "
            "routinely exhausted and the route is paying for a crossing it could avoid by "
            "reordering candidates.",
            12,
            y,
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
            "Per-model failure rate with its denominator shown, so a 100% rate on two calls is "
            "not mistaken for a broken model. `clamp_min` keeps the division defined when a "
            "model has no calls; it does not manufacture a zero for an unused model.",
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
                    ref="A",
                )
            ],
            "The same rate as a trend, which distinguishes a model that has always been "
            "unreliable from one that started failing an hour ago — a change a cumulative rate "
            "would hide.",
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
                    ref="A",
                )
            ],
            "What callers actually received. `ok` beside a smaller `retryable_error` is recovery "
            "working; `other_error` growing without a matching upstream explanation points at a "
            "local failure — a translation or configuration problem — rather than a provider.",
            0,
            y,
        )
    )
    add(
        barchart(
            "RUN OUTCOMES BY ALGORITHM",
            [
                target(
                    f"sum by (algorithm, outcome) ({RUNS}{{{jf}}})",
                    "{{algorithm}} {{outcome}}",
                    ref="A",
                )
            ],
            "Routing runs and how they ended, by the algorithm that selected. A run can end "
            "differently from its upstream attempts because the algorithm may fall back.",
            12,
            y,
        )
    )
    adv(9)

    # ================================ 7 DATA QUALITY AND DISCLOSURES
    add(row("7  DATA QUALITY, MEASUREMENT BASIS AND DISCLOSURES", y))
    adv(1)

    add(
        stat(
            "ACCOUNT READS OK",
            f'sum({FETCH}{{{jf},outcome="ok"}})',
            "Successful provider account reads since this process started.",
            0,
            y,
        )
    )
    add(
        stat(
            "ACCOUNT READS FAILED",
            f"sum({FETCH}{{{jf}}}) - sum({FETCH}{{{jf},outcome=\"ok\"}})",
            "Account reads that did not succeed, for any reason. Computed as total reads minus "
            "successful reads rather than by filtering to a failing outcome, so a healthy "
            "deployment reads a real 0 instead of No Data, while a vanished metric family still "
            "leaves the panel empty. A rising count here means the balance figures above are "
            "ageing.",
            6,
            y,
        )
    )
    add(
        stat(
            "AUTH FAILURES ON ACCOUNT READS",
            f"sum({AUTHFAIL}{{{jf}}})",
            "Times a provider rejected our own credential while reading account state. Event "
            "driven: the family appears only after the first occurrence, so an absent panel "
            "means none has been recorded in this process, not that measurement is broken.",
            12,
            y,
        )
    )
    add(
        stat(
            "PROVIDERS WITH ACCOUNT COVERAGE",
            f"count(count by (provider) ({FRESH}{{{jf}}}))",
            "How many provider accounts are being read at all. A provider missing here has no "
            "native financial instrumentation, so its absence from the account sections is a "
            "coverage gap rather than a zero balance.",
            18,
            y,
        )
    )
    adv(5)

    add(
        stat(
            "ROLLING 24H COVERAGE",
            f'max({COVER}{{{jf},window="24h"}})',
            "Seconds of real observation behind every rolling 24h figure on this dashboard. "
            "After a restart this is below 86400 until the process has been up a full day; "
            "while it is, the 24h numbers describe a partial window and should not be compared "
            "with a full one.",
            0,
            y,
            w=12,
        )
    )
    add(
        stat(
            "PROVIDERS CURRENTLY REPORTING",
            f"count(count by (provider) ({FETCH}{{{jf}}}))",
            "Distinct providers that have completed at least one account read in this process. "
            "A provider absent from the account sections entirely is a coverage gap, not a zero "
            "balance: it has no native financial instrumentation, so nothing is known about its "
            "money rather than everything being zero.",
            12,
            y,
            w=12,
        )
    )
    adv(5)

    add(
        text(
            "WHAT IS MEASURED, AND WHAT IS NOT",
            "**Measured.** Request counts, token components, provider-reported cost, provider "
            "account balances, credits, usage and allowances, per-attempt HTTP status, "
            "composition decisions and crossings, and client-visible outcomes. All of it comes "
            "from `switchyard-server`'s own Prometheus exposition; the generator refuses to "
            "build a panel on any family outside that contract.\n\n"
            "**Not measured.** Per-model *estimated* cost. The binary declares "
            "`switchyard.economics_estimated_cost` and never calls it, so no series exists and "
            "no panel presents one. Provider-*reported* cost is a different number and is "
            "labelled as the provider's own figure.\n\n"
            "**Not attributed.** Per-agent, per-domain or per-caller consumption. Switchyard "
            "records the route, model and provider that served a request; it does not record "
            "who asked. Any such breakdown would be invented, so there is none.\n\n"
            "**DeepSeek in USD.** Not shown. DeepSeek publishes a structurally zero USD series "
            "because it keeps no dollar balance; displaying it would put a measured-looking "
            "`$0.00` beside an account holding CNY 91.81.",
            0,
            y,
            w=12,
            h=9,
        )
    )
    add(
        text(
            "THE ACCOUNT READ INTERVAL IS NOT A HEALTH STATUS",
            "`switchyard.economics_account_freshness_seconds` is recorded **only when a read "
            "succeeds**, and its value is the elapsed time since the previous success. Two "
            "consequences follow, and both matter for reading this dashboard.\n\n"
            "It reports the **polling cadence** — roughly five minutes for a healthy provider — "
            "not the age of the balance on screen. A panel labelled \"Status\" showing `5 mins` "
            "was reporting the cadence, not saying the account was healthy five minutes ago.\n\n"
            "It **cannot reveal staleness**. When reads start failing, no new value is recorded, "
            "so the gauge keeps its last interval and the panel goes on showing a healthy-looking "
            "number while the balance it describes gets older. That is why read health is shown "
            "from `account_fetch_total` by outcome instead: a flat `ok` count beside a rising "
            "failure count is the actual signal, and it is what section 1's ACCOUNT READ HEALTH "
            "table displays.",
            12,
            y,
            w=12,
            h=9,
        )
    )
    adv(9)

    add(
        text(
            "WHY THE TWO ERROR RATES DIFFER, AND WHY NEITHER IS A DIAGNOSIS",
            "`switchyard_upstream_attempts_total` records one increment per attempt, labelled "
            "with the HTTP status received. Its `outcome` label is a **function of that status "
            "alone** — 408, 429 and 5xx become `retryable_error`, everything else `other_error`. "
            "It is honest about what it saw and silent about why.\n\n"
            "Two cases the providers document explicitly are invisible in that label. A `429` "
            "may be a rate limit that clears in seconds or an exhausted balance that no retry "
            "can fix. An OpenRouter `500 server_error` may be masked infrastructure trouble or a "
            "permanent authentication failure reported only in the top-level `error_type`. "
            "Neither is recoverable from the status code.\n\n"
            "The semantic classification that does exist natively is the composition layer: its "
            "`capability` and `reason` labels are decided from the provider's structured error. "
            "Section 6 shows both facts side by side rather than presenting one as the other.",
            0,
            y,
            w=12,
            h=9,
        )
    )
    add(
        text(
            "ABSENCE HAS FOUR DIFFERENT MEANINGS HERE",
            "**Zero.** A series that exists and reads zero. The traffic, spend or balance is "
            "genuinely zero — `GLOBAL — TODAY` showing 0.00 is a real reading.\n\n"
            "**No event observed.** A family that only appears once its event happens. Account "
            "authentication failures are the example: an empty panel means the event has not "
            "occurred in this process, not that measurement is broken.\n\n"
            "**Partial window.** Rolling windows are computed by the binary from events it "
            "observed. After a restart the coverage panel drops below 86400s and the 24h figures "
            "describe less than a day.\n\n"
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
                "title": "Switchyard routing (V7.21)",
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
        "tags": ["switchyard", "economics", "native", "provider-accounts"],
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

    Runs at build time as well as in the test, so an expression cannot reach a
    published document by way of a path the test does not exercise. The check is
    on the rendered document, because what matters is the PromQL that will run.
    """
    problems = []
    for panel in dashboard["panels"]:
        refs = {}
        for tgt in panel.get("targets") or []:
            ref = tgt.get("refId")
            refs[ref] = refs.get(ref, 0) + 1
            expr = tgt.get("expr", "")
            for problem in native_contract.violations(expr):
                problems.append(f"panel {panel.get('title')!r} [{ref}]: {problem}")
            lowered = expr.lower()
            for marker in ("localclaw_", 'job="switchyard"', "hermes", "openclaw", "dsh_"):
                if marker in lowered:
                    problems.append(f"panel {panel.get('title')!r}: retired source ({marker})")
        for ref, count in refs.items():
            if count > 1:
                problems.append(
                    f"panel {panel.get('title')!r}: {count} targets share refId {ref!r} — "
                    "Grafana keys results by refId, so duplicates render as No Data"
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
    parser = argparse.ArgumentParser(description="Build the V9 economics dashboard.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--expected-version", type=int, default=None)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    request = build_request(config, expected_version=args.expected_version)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(request, handle, indent=1)
    panels = request["dashboard"]["panels"]
    queries = sum(len(p.get("targets") or []) for p in panels)
    print(f"wrote {args.out}: {len(panels)} panels, {queries} queries, op={request['op']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
