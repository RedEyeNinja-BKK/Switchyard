# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the economics dashboard generator.

The dashboard is published from a document this generator builds, so the
document must be reproducible from the repository. These tests pin the parts of
it that a reader depends on: the accounting rules the panels encode, the label
schema the provider account queries use, and the shape the publishing tool
accepts.
"""

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "scripts" / "economics" / "economics_dashboard.toml"
GENERATOR = REPO_ROOT / "scripts" / "economics" / "build_economics_dashboard.py"

sys.path.insert(0, str(REPO_ROOT / "scripts" / "economics"))

import build_economics_dashboard as builder  # noqa: E402


def _config():
    return builder.load_config(CONFIG)


def _request():
    return builder.build_request(_config())


def _expressions():
    document = _request()["dashboard"]
    return [
        (panel.get("title"), target.get("expr"))
        for panel in document.get("panels", [])
        for target in panel.get("targets", [])
        if target.get("expr")
    ]


def test_the_request_has_the_shape_the_publishing_tool_accepts() -> None:
    request = _request()
    # The tool refuses any field outside this set.
    assert set(request) == {
        "op",
        "uid",
        "message",
        "folder_uid",
        "dashboard",
    }
    assert request["op"] == "create"
    assert request["uid"] == _config()["uid"]
    assert request["folder_uid"] is None
    assert request["message"].strip()


def test_the_dashboard_declares_the_uid_the_tool_will_write() -> None:
    request = _request()
    # The tool writes the document under the request uid, so the two must agree.
    assert request["dashboard"]["uid"] == request["uid"]


def test_the_build_is_reproducible() -> None:
    first = builder.build_request(builder.load_config(CONFIG))
    second = builder.build_request(builder.load_config(CONFIG))
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_the_generator_runs_from_the_command_line(tmp_path) -> None:
    out = tmp_path / "economics-v8.json"
    result = subprocess.run(
        [
            sys.executable,
            str(GENERATOR),
            "--config",
            str(CONFIG),
            "--out",
            str(out),
        ],
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr.decode()
    written = json.loads(out.read_text())
    assert written["op"] == "create"
    assert written["dashboard"]["panels"]


def test_no_panel_uses_a_windowed_increase() -> None:
    # A windowed increase misses any increment that happened before Prometheus
    # took its first sample of the series, so a real request can read as zero.
    for title, expression in _expressions():
        assert "increase(" not in expression, title


def test_the_token_components_are_never_added_to_each_other() -> None:
    # The prompt series contains the cached and cache-created series, and the
    # completion series contains the reasoning series. A panel that selects two
    # components in one expression would add a component to the total that
    # contains it.
    components = ("prompt", "cached", "cache_creation", "completion", "reasoning")
    for title, expression in _expressions():
        selected = [
            component for component in components if f'component="{component}"' in expression
        ]
        assert len(selected) <= 1, f"{title} selects {selected}"


def test_no_panel_adds_the_reasoning_to_the_completion() -> None:
    # Reasoning is inside completion. A panel that sums the two would double
    # count the reasoning.
    for title, expression in _expressions():
        if 'component="completion"' in expression:
            assert 'component="reasoning"' not in expression, title


def test_no_panel_adds_the_cache_to_the_prompt() -> None:
    # The cached and cache-created counts are inside the prompt total.
    for title, expression in _expressions():
        if 'component="prompt"' in expression:
            assert 'component="cached"' not in expression, title
            assert 'component="cache_creation"' not in expression, title


def test_the_provider_account_queries_name_the_scope_and_the_period() -> None:
    # The currency, the accounting scope and the reporting period are three
    # separate labels. A query that leaves one of them open would match every
    # scope and every period at once.
    families = (
        "switchyard_economics_provider_usage",
        "switchyard_economics_provider_credit",
        "switchyard_economics_provider_balance",
    )
    checked = 0
    for title, expression in _expressions():
        if not any(family in expression for family in families):
            continue
        checked += 1
        assert "scope=" in expression, title
        assert "period=" in expression, title
    assert checked >= 5, "the provider account queries must be checked"


def test_no_query_uses_a_composite_currency_label() -> None:
    # A currency label that also carried the scope or the period would make two
    # different amounts indistinguishable without parsing the label value.
    for title, expression in _expressions():
        assert 'currency="USD_' not in expression, title


def test_the_credit_the_consumed_credit_and_the_balance_are_different_panels() -> None:
    titles = {title for title, _ in _expressions()}
    assert "OPENROUTER CREDIT (USD) - GLOBAL KEY ACCOUNT ONLY" in titles
    assert "OPENROUTER CONSUMED (USD) - GLOBAL KEY ACCOUNT ONLY" in titles
    assert "OPENROUTER REMAINING (USD) - GLOBAL KEY ACCOUNT ONLY" in titles


def test_the_dashboard_links_back_to_the_routing_dashboard() -> None:
    links = _request()["dashboard"]["links"]
    urls = {link.get("url") for link in links}
    assert f"/d/{_config()['routing_uid']}" in urls


def test_the_deployment_facts_come_from_the_configuration() -> None:
    # The datasource, the job and the routing uid are deployment facts, so the
    # generator must take them from the configuration rather than hardcoding
    # them. Changing the configuration must change the document.
    config = _config()
    baseline = builder.build_request(config)["dashboard"]
    datasource = baseline["panels"][1]["datasource"]
    assert datasource["uid"] == config["datasource_uid"]
    assert datasource["type"] == config["datasource_type"]

    changed = dict(config)
    changed["datasource_uid"] = "a-different-datasource"
    changed["routing_uid"] = "a-different-routing-dashboard"
    document = builder.build_request(changed)["dashboard"]
    assert document["panels"][1]["datasource"]["uid"] == "a-different-datasource"
    assert f"/d/{changed['routing_uid']}" in {link["url"] for link in document["links"]}


def test_every_panel_has_a_unique_id() -> None:
    # Grafana identifies a panel by its id, so two panels sharing one would make
    # the dashboard ambiguous.
    panels = _request()["dashboard"]["panels"]
    ids = [panel["id"] for panel in panels]
    assert len(ids) == len(set(ids))


def test_every_panel_fits_the_grid() -> None:
    # Grafana's grid is 24 columns wide; a panel that starts inside it but runs
    # past the edge is laid out unpredictably.
    for panel in _request()["dashboard"]["panels"]:
        grid = panel["gridPos"]
        assert grid["x"] + grid["w"] <= 24, panel.get("title")


def test_the_dashboard_discloses_partial_openrouter_coverage() -> None:
    # The deployment holds two OpenRouter credentials. Account figures are
    # collected for the global key only, so the money panels describe one
    # account. Publishing that as complete financial coverage would misstate
    # real spend, so the limitation is stated on the dashboard itself.
    document = _request()["dashboard"]
    rules = [
        panel["options"]["content"]
        for panel in document["panels"]
        if panel.get("type") == "text" and "ACCOUNTING" in panel.get("title", "")
    ]
    assert rules, "the accounting-rules panel must exist"
    text = "\n".join(rules)
    assert "PARTIAL" in text
    assert "openrouter_stephen" in text
    assert "not collected" in text


def test_the_money_panels_name_the_account_they_cover() -> None:
    # A panel titled only "OPENROUTER CREDIT" would read as the deployment's
    # total OpenRouter credit. It is one account's credit.
    titles = [panel.get("title", "") for panel in _request()["dashboard"]["panels"]]
    money = [
        title
        for title in titles
        if "OPENROUTER" in title
        and any(word in title for word in ("CREDIT", "REMAINING", "CONSUMED", "USAGE"))
    ]
    assert money, "the OpenRouter money panels must exist"
    for title in money:
        assert "GLOBAL KEY" in title, f"{title} must name the account it covers"


def test_coverage_is_shown_per_provider_not_as_a_subtraction() -> None:
    # Served usage and account state name the same account differently: the
    # OpenAI account appears as openai_nt in served usage and as openai in
    # account state, because two clients share one credential. Subtracting the
    # provider counts would therefore be wrong in both directions - it can read
    # zero while a provider is uncovered, and it can report a covered provider
    # as a gap. Coverage is answered per row instead, where a provider carrying
    # tokens but no freshness value is visible.
    document = _request()["dashboard"]
    titles = [panel.get("title", "") for panel in document["panels"]]
    assert "ACCOUNT FINANCIAL COVERAGE BY PROVIDER" in titles

    # No panel may subtract the two distinct-provider counts.
    for panel in document["panels"]:
        for target in panel.get("targets", []):
            expression = target.get("expr", "")
            assert not (
                "account_freshness_seconds" in expression and "requests_total" in expression
            ), f"{panel.get('title')} derives coverage by subtraction"
