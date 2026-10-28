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
