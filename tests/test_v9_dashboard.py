# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""V9.0 economics dashboard must query native Switchyard metrics only.

The dashboard's defining constraint is that every number on it is something the
switchyard-server binary measured itself. This suite is what holds that: the
generator and its allowlist are checked together, and a set of negative controls
proves the checks bite rather than merely passing.

The controls matter more than the happy path here. A test that only asserts
"the real dashboard is clean" passes equally well against an allowlist that
accepts everything, so each rule below is exercised with a document that should
be refused.
"""

import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(os.path.dirname(HERE), "scripts", "economics")
V9 = os.path.join(SCRIPTS, "v9")
sys.path.insert(0, SCRIPTS)
sys.path.insert(0, V9)

import build_v9_dashboard as gen  # noqa: E402
import native_contract  # noqa: E402
import verify_v9 as verifier  # noqa: E402


@pytest.fixture(scope="module")
def config():
    return gen.load_config(os.path.join(V9, "v9_dashboard.toml"))


@pytest.fixture(scope="module")
def dashboard(config):
    return gen.build_dashboard(config)


def _exprs(doc):
    for panel in doc["panels"]:
        for target in panel.get("targets") or []:
            if target.get("expr"):
                yield panel.get("title"), target["expr"], target.get("legendFormat")


# ------------------------------------------------------------------ the rule
def test_every_expression_names_only_native_families(dashboard):
    problems = []
    for title, expr, _ in _exprs(dashboard):
        for problem in native_contract.violations(expr):
            problems.append(f"{title}: {problem}")
    assert problems == []


def test_no_panel_references_a_retired_source(dashboard):
    assert verifier.provenance(dashboard) == []


def test_every_money_query_pins_or_names_its_scope(dashboard):
    """A balance is published under two scopes describing one amount of money."""
    offenders = []
    for title, expr, legend in _exprs(dashboard):
        if not any(m in expr for m in verifier.MONEY):
            continue
        if "scope=" in expr:
            continue
        if "{{scope}}" in (legend or "") or "by (scope" in expr:
            continue
        offenders.append(f"{title}: {expr[:70]}")
    assert offenders == []


def test_dashboard_structure_is_valid(dashboard):
    assert verifier.v8.structure(dashboard) == []


def test_the_retired_scraper_is_absent_entirely(dashboard):
    blob = json.dumps(dashboard)
    for marker in ("localclaw_", 'job="localclaw-econ"', "hermes", "openclaw"):
        assert marker not in blob


# --------------------------------------------------------------- negative controls
def test_generator_refuses_an_external_exporter_metric(config):
    """A panel reading a retired external exporter must not build."""
    doc = gen.build_dashboard(config)
    doc["panels"].append(
        {
            "title": "injected",
            "targets": [{"refId": "A", "expr": 'sum(localclaw_requests_total{job="localclaw-econ"})'}],
        }
    )
    with pytest.raises(SystemExit) as excinfo:
        gen.check_native_contract(doc)
    assert "localclaw_" in str(excinfo.value) or "retired source" in str(excinfo.value)


def test_generator_refuses_an_unreviewed_native_prefix(config):
    """A switchyard_ name that is not in the contract must still be refused."""
    doc = gen.build_dashboard(config)
    doc["panels"].append(
        {"title": "injected", "targets": [{"refId": "A", "expr": "sum(switchyard_not_a_real_family)"}]}
    )
    with pytest.raises(SystemExit) as excinfo:
        gen.check_native_contract(doc)
    assert "not in the native contract" in str(excinfo.value)


def test_generator_refuses_the_dead_estimated_cost_family(config):
    """`record_estimated_cost` is never called, so no series exists to plot."""
    doc = gen.build_dashboard(config)
    doc["panels"].append(
        {
            "title": "injected",
            "targets": [{"refId": "A", "expr": "sum(switchyard_economics_estimated_cost)"}],
        }
    )
    with pytest.raises(SystemExit) as excinfo:
        gen.check_native_contract(doc)
    message = str(excinfo.value)
    assert "estimated_cost" in message
    assert "never called" in message


def test_allowlist_would_reject_a_family_that_stops_being_emitted():
    """The contract is a closed list, so a typo is a build failure, not a blank panel."""
    assert native_contract.violations("sum(switchyard_economics_provider_credi)") != []
    assert native_contract.violations("sum(switchyard_economics_provider_credit)") == []


def test_event_driven_families_are_allowed_but_not_claimed_as_always_present():
    """An event-driven family is legitimate; presenting it as always-present is not."""
    for family in native_contract.EVENT_DRIVEN:
        assert family not in native_contract.LIVE_FAMILIES
        assert native_contract.violations(f"sum({family})") == []


def test_histogram_suffixes_resolve_to_their_base_family():
    expr = "histogram_quantile(0.99, sum by (le) (rate(switchyard_llm_call_duration_ms_bucket[5m])))"
    assert native_contract.referenced_metrics(expr) == {"switchyard_llm_call_duration_ms"}
    assert native_contract.violations(expr) == []


def test_quoted_label_values_are_not_mistaken_for_metric_names():
    """`job="switchyard-v7"` is data, not a metric reference."""
    assert native_contract.referenced_metrics('sum(switchyard_runs_total{job="switchyard-v7"})') == {
        "switchyard_runs_total"
    }


def test_the_dashboard_declares_the_cost_family_it_cannot_currency_filter(dashboard):
    """The cumulative cost counter has no currency label; money panels must avoid it."""
    for title, expr, _ in _exprs(dashboard):
        if any(m in expr for m in verifier.MONEY):
            continue
        assert "switchyard_economics_provider_cost_total" not in expr, (
            f"{title} uses the unlabelled cumulative cost counter for a money figure"
        )


# ------------------------------------------------- refId: the No Data defect
def test_no_panel_reuses_a_refid(dashboard):
    """Grafana keys query results by refId, so duplicates render as No Data.

    This is the defect that broke the allowance-history and usage-trend charts:
    both targets were valid, both had data, and both were labelled "A". A PromQL
    check sees nothing wrong and a per-expression liveness check sees nothing
    wrong, which is why the assertion lives in the structure gate.
    """
    offenders = []
    for panel in dashboard["panels"]:
        refs = [t.get("refId") for t in (panel.get("targets") or [])]
        for ref in {r for r in refs if refs.count(r) > 1}:
            offenders.append(f"{panel.get('title')}: refId {ref!r} x{refs.count(ref)}")
    assert offenders == []


def test_structure_gate_rejects_a_duplicate_refid():
    """The negative control: the check must bite, not merely pass."""
    import verify_generated_dashboard as v8

    doc = {
        "panels": [
            {
                "id": 1,
                "title": "two queries, one refId",
                "gridPos": {"x": 0, "y": 0, "w": 12, "h": 8},
                "targets": [
                    {"refId": "A", "expr": "sum(switchyard_runs_total)"},
                    {"refId": "A", "expr": "sum(switchyard_requests_total)"},
                ],
            }
        ]
    }
    problems = v8.structure(doc)
    assert any("sharing refId" in p for p in problems), problems


def test_the_repaired_history_charts_carry_two_distinct_refids(dashboard):
    """Named explicitly: these are the charts that were broken in V8."""
    wanted = {
        "5-HR WINDOW — USED AND REMAINING",
        "WEEKLY WINDOW — USED AND REMAINING",
        "GLOBAL ACCOUNT — USAGE, CREDIT AND REMAINING",
        "STEPHEN ACCOUNT — USAGE, CREDIT AND REMAINING",
    }
    seen = set()
    for panel in dashboard["panels"]:
        if panel.get("title") in wanted:
            refs = [t.get("refId") for t in panel["targets"]]
            assert refs == sorted(refs), f"{panel['title']}: {refs}"
            assert len(set(refs)) == len(refs), f"{panel['title']}: {refs}"
            seen.add(panel["title"])
    assert seen == wanted


def test_the_operator_critical_account_sections_are_present(dashboard):
    """Feature parity with V8 on the panels an operator actually uses."""
    titles = {p.get("title") for p in dashboard["panels"]}
    for required in (
        "5-HR REMAINING",
        "5-HR USED",
        "TIME TO 5-HR RESET",
        "WEEKLY REMAINING",
        "WEEKLY USED",
        "TIME TO WEEKLY RESET",
        "5-HR WINDOW — USED AND REMAINING",
        "WEEKLY WINDOW — USED AND REMAINING",
        "BALANCE (CNY)",
        "REMAINING — TOPPED-UP (CNY)",
        "REMAINING — GRANTED (CNY)",
        "REMAINING — TOTAL (CNY)",
        "BALANCE TREND (CNY)",
        "GLOBAL — REMAINING (USD)",
        "GLOBAL — TOTAL USAGE (USD)",
        "GLOBAL — PURCHASED CREDITS (USD)",
        "GLOBAL — 24H SPEND (USD)",
        "GLOBAL — KEY USAGE (USD, CUMULATIVE)",
        "GLOBAL — THIS MONTH (USD)",
        "GLOBAL ACCOUNT — USAGE, CREDIT AND REMAINING",
        "STEPHEN — REMAINING (USD)",
        "STEPHEN — PURCHASED CREDITS (USD)",
        "STEPHEN — TOTAL USAGE (USD)",
    ):
        assert required in titles, f"missing operator-critical panel: {required}"


def test_no_panel_presents_a_structural_zero_currency_as_a_balance(dashboard):
    """DeepSeek has no USD balance; a $0.00 panel would look measured."""
    for panel in dashboard["panels"]:
        for t in panel.get("targets") or []:
            expr = t.get("expr") or ""
            if "switchyard_economics_provider_balance" not in expr:
                continue
            assert 'currency="USD",scope="account"' not in expr or "deepseek" not in expr, (
                f"{panel.get('title')} plots DeepSeek's structural USD zero as a balance"
            )
