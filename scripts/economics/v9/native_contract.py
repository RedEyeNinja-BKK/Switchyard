# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The native Switchyard metric contract the V9.0 dashboard is allowed to query.

WHAT THIS IS
  A positive allowlist. Every family named here is emitted by the Switchyard
  binary's own Prometheus exposition, captured from the live deployment on
  2026-10-28. The dashboard generator asserts its own output against this list at
  build time, and a test asserts it again over the generated document.

WHY A POSITIVE LIST RATHER THAN A DENYLIST
  A denylist can only reject the sources someone remembered. The V8 family grew up
  alongside an external exporter and inherited panels that read it; removing those
  panels one by one leaves nothing that prevents the next such panel from being
  added. A positive list fails closed: an expression naming a family that is not
  here cannot be generated, whether the family is retired, external, or simply new
  and unreviewed.

THE DISTINCTION THIS ENCODES
  "Native" means the switchyard-server binary emits it on /metrics. Provider
  account balances, credits and usage count as native: they are Switchyard's own
  reads of its own credentials, not telemetry about some other system. A metric
  produced by a separate exporter process, by an agent runtime, or by a retired
  scrape job does not count, however useful it looks.

FAMILIES THAT MAY LEGITIMATELY BE ABSENT
  Some families are counters that only increment when their event occurs. They are
  in the contract, but a given scrape may not contain them. A panel on such a
  family must explain the absence rather than render a zero, because "no auth
  failure has happened" and "the auth-failure counter is not being emitted" are
  different facts and only one of them is reassuring.

DELIBERATELY ABSENT
  `switchyard.economics_estimated_cost` is declared in
  crates/switchyard-server/src/economics/metrics.rs and never called anywhere in
  the tree, so no series for it is ever produced. It is excluded here so that no
  panel can present an estimate that the binary does not compute. Per-model
  provider-REPORTED cost is a different thing and is available.
"""

import re

# Every family the live deployment exposed on /metrics, captured 2026-10-28.
# Histograms are listed under their base name: the exposition adds _bucket, _sum
# and _count, and a query may legitimately name any of the three.
LIVE_FAMILIES = frozenset(
    {
        "switchyard_algorithms_in_flight",
        "switchyard_build_info",
        "switchyard_cache_creation_tokens_total",
        "switchyard_cached_tokens_total",
        "switchyard_client_responses_total",
        "switchyard_completion_tokens_total",
        "switchyard_composition_attempt_total",
        "switchyard_composition_cross_total",
        "switchyard_decisions_total",
        "switchyard_economics_account_fetch_duration_seconds",
        "switchyard_economics_account_fetch_total",
        "switchyard_economics_account_freshness_seconds",
        "switchyard_economics_allowance_remaining_percent",
        "switchyard_economics_allowance_reset_seconds",
        "switchyard_economics_calendar_day_elapsed_seconds",
        "switchyard_economics_calendar_day_end",
        "switchyard_economics_calendar_day_start",
        "switchyard_economics_calendar_provider_cost",
        "switchyard_economics_calendar_requests",
        "switchyard_economics_calendar_tokens",
        "switchyard_economics_provider_balance",
        "switchyard_economics_provider_balance_component",
        "switchyard_economics_provider_cost_total",
        "switchyard_economics_provider_credit",
        "switchyard_economics_provider_usage",
        "switchyard_economics_requests_total",
        "switchyard_economics_rolling_coverage_seconds",
        "switchyard_economics_rolling_provider_cost",
        "switchyard_economics_rolling_requests",
        "switchyard_economics_rolling_tokens",
        "switchyard_economics_tokens_total",
        "switchyard_errors_total",
        "switchyard_llm_call_duration_ms",
        "switchyard_llm_calls_total",
        "switchyard_model_call_latency_ms",
        "switchyard_prompt_tokens_total",
        "switchyard_reasoning_tokens_total",
        "switchyard_requests_total",
        "switchyard_router_retry_recovered_total",
        "switchyard_routing_overhead_ms",
        "switchyard_run_duration_ms",
        "switchyard_runs_total",
        "switchyard_total_errors",
        "switchyard_total_latency_ms",
        "switchyard_total_requests",
        "switchyard_upstream_attempts_total",
    }
)

# Declared and emitted, but only once their event has occurred at least once. A
# scrape taken before the first occurrence contains no series at all.
EVENT_DRIVEN = frozenset(
    {
        "switchyard_economics_account_auth_failures_total",
    }
)

ALLOWED = LIVE_FAMILIES | EVENT_DRIVEN

# Names that must never appear in a generated expression, with the reason a reader
# can act on. Absence from ALLOWED would already reject these; naming them gives a
# precise diagnostic instead of "unrecognized family".
FORBIDDEN = {
    "switchyard_economics_estimated_cost": (
        "declared in economics/metrics.rs but never called: no series exists, so any panel "
        "on it would present an estimate the binary does not compute"
    ),
}

_HISTOGRAM_SUFFIXES = ("_bucket", "_sum", "_count")

# A PromQL metric reference: a dotted-stem family followed by an optional histogram
# suffix. Label values are quoted strings and are matched separately, because a
# legend value such as "switchyard-v7" is not a metric reference.
_METRIC_RE = re.compile(r"\bswitchyard_[a-z0-9_]+", re.IGNORECASE)


def _base(name):
    for suffix in _HISTOGRAM_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def referenced_metrics(expr):
    """Returns the Switchyard metric families an expression names.

    Only bare identifiers are considered. A quoted label value is skipped, so
    `job="switchyard-v7"` does not look like a metric reference and a legend
    template does not either.
    """
    # Drop quoted strings first: they carry label values, which are data, not
    # metric names.
    without_strings = re.sub(r'"(?:[^"\\]|\\.)*"', '""', expr)
    found = set()
    for match in _METRIC_RE.finditer(without_strings):
        found.add(_base(match.group(0)))
    return found


def violations(expr):
    """Returns a list of reasons this expression may not be used, empty when clean.

    Checked in order so the most actionable reason is the one reported: a forbidden
    family gets its explanation, an unknown family gets named, and any remaining
    non-native source is caught by the caller's own scan of the same expression.
    """
    problems = []
    for name in sorted(referenced_metrics(expr)):
        if name in FORBIDDEN:
            problems.append(f"{name}: {FORBIDDEN[name]}")
        elif name not in ALLOWED:
            problems.append(
                f"{name}: not in the native contract; either the family is not emitted by "
                "switchyard-server or it has not been reviewed for this dashboard"
            )
    return problems
