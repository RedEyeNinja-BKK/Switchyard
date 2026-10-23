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


def test_cumulative_panels_never_use_a_windowed_increase() -> None:
    # A windowed increase on a cumulative panel misses any increment that
    # happened before Prometheus took its first sample of the series, so a real
    # request can read as zero. Cumulative panels therefore read the counter
    # directly.
    #
    # A ROLLING panel is the opposite case: a trailing window is exactly what
    # increase() computes, and it is what the historical dashboard showed. The
    # rule is therefore that the two shapes stay distinct - a panel may use
    # increase() only if it says it is a rolling window, and a panel that claims
    # to be cumulative must not.
    for title, expression in _expressions():
        if "increase(" not in expression:
            continue
        assert "rolling" in title.lower(), (
            f"{title} uses increase() but is not labelled as a rolling window"
        )

    for title, expression in _expressions():
        if "cumulative" not in title.lower():
            continue
        assert "increase(" not in expression, (
            f"{title} claims to be cumulative but uses a windowed increase"
        )


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
    assert "OpenRouter Credits (total, USD · global key)" in titles
    assert "OpenRouter Usage (total, USD · global key)" in titles
    assert "OpenRouter Remaining (USD · global key)" in titles


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


def test_the_dashboard_states_that_both_openrouter_accounts_are_separate() -> None:
    # Replaces an earlier test that asserted OpenRouter coverage was PARTIAL,
    # because the second account was not collected. That is no longer true: both
    # accounts now have readers. The requirement never was "say it is partial" - it
    # was "never let one account's money stand for the deployment's position". The
    # assertion follows the requirement into its new form.
    document = _request()["dashboard"]
    rules = [
        panel["options"]["content"]
        for panel in document["panels"]
        if panel.get("type") == "text" and "ACCOUNTING" in panel.get("title", "")
    ]
    assert rules, "the accounting-rules panel must exist"
    text = "\n".join(rules)
    assert "two financial accounts" in text
    assert "never added together" in text
    assert "PARTIAL" not in text, "the coverage gap is closed; the caveat must go"


def test_the_money_panels_name_the_account_they_cover() -> None:
    # A panel titled only "OpenRouter Credits" would read as the deployment's
    # total OpenRouter credit. It is one account's credit, so each money panel
    # names the account it covers.
    titles = [panel.get("title", "") for panel in _request()["dashboard"]["panels"]]
    money = [
        title
        for title in titles
        if "OpenRouter" in title
        and any(word in title for word in ("Credit", "Remaining", "Usage", "Spend"))
    ]
    assert money, "the OpenRouter money panels must exist"
    for title in money:
        assert "global key" in title, f"{title} must name the account it covers"


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


def test_the_coverage_table_explains_the_openai_naming_split() -> None:
    # The OpenAI account legitimately occupies two rows: served usage is labelled
    # openai_nt and account state is labelled openai, because two clients share
    # one credential. Without explaining that, the table reads as a second
    # uncovered provider.
    document = _request()["dashboard"]
    table = [
        panel
        for panel in document["panels"]
        if panel.get("title") == "ACCOUNT FINANCIAL COVERAGE BY PROVIDER"
    ]
    assert table, "the coverage table must exist"
    description = table[0]["description"]
    assert "openai_nt" in description
    assert "one credential" in description


def test_the_deepseek_row_shows_every_figure_the_previous_dashboard_showed() -> None:
    # The row the operator reads must not lose information in the move to
    # Switchyard telemetry. Switchyard publishes the balance and its movement,
    # but not the top-up ledger, lifetime spend, or request-derived actual cost -
    # those live only in the older localclaw exporter. They are still current,
    # so the row carries them rather than presenting a thinner row.
    document = _request()["dashboard"]
    titles = {panel.get("title", "") for panel in document["panels"]}
    required = {
        "DeepSeek Status",
        "DeepSeek Balance (CNY)",
        "DeepSeek 24h Balance Movement",
        "DeepSeek Top-Ups (total, CNY)",
        "DeepSeek Lifetime Spend (CNY)",
        "Cumulative Actual Spend (CNY)",
        "DeepSeek balance trend (CNY)",
        "DeepSeek lifetime spend (CNY)",
    }
    assert required <= titles, f"missing from the DeepSeek row: {sorted(required - titles)}"


def test_every_legacy_deepseek_figure_is_attributed_to_its_source() -> None:
    # Three figures cannot be derived from Switchyard's balance readings. The
    # dashboard must say where they actually come from, or an operator reading
    # them as Switchyard output will trust provenance that does not exist.
    config = _config()
    document = _request()["dashboard"]
    legacy_job = config.get("legacy_job")
    assert legacy_job, "the configuration must name the exporter that publishes these figures"
    legacy_panels = [
        panel
        for panel in document["panels"]
        if any(legacy_job in target.get("expr", "") for target in panel.get("targets", []))
    ]
    assert legacy_panels, "the legacy-sourced panels must exist"

    coverage = [
        panel["options"]["content"]
        for panel in document["panels"]
        if panel.get("type") == "text" and "DEEPSEEK COVERAGE" in panel.get("title", "")
    ]
    assert coverage, "the DeepSeek coverage note must exist"
    text = "\n".join(coverage)
    assert "localclaw" in text, "the coverage note must name the exporter it relies on"
    assert "Read from Switchyard" in text, "the note must separate the two sources"


def test_a_first_publication_creates_and_carries_no_expected_version() -> None:
    # The publishing tool rejects a create that names expected_version, so the
    # create request must leave the field out entirely rather than send null.
    request = builder.build_request(_config())
    assert request["op"] == "create"
    assert "expected_version" not in request


def test_publishing_a_revision_updates_and_names_the_live_version() -> None:
    # An update without expected_version would be a blind overwrite. The tool
    # requires the version it expects to replace, which is what refuses a stale
    # request and protects a concurrent change.
    request = builder.build_request(_config(), expected_version=3)
    assert request["op"] == "update"
    assert request["expected_version"] == 3


def test_both_request_shapes_carry_only_the_fields_the_tool_accepts() -> None:
    allowed = {"op", "uid", "message", "folder_uid", "dashboard", "expected_version"}
    for request in (
        builder.build_request(_config()),
        builder.build_request(_config(), expected_version=7),
    ):
        extra = set(request) - allowed
        assert not extra, f"the publishing tool refuses unknown fields: {sorted(extra)}"
        assert "overwrite" not in request, "the tool builds overwrite itself"


def test_both_openrouter_accounts_are_shown_and_never_combined() -> None:
    # The deployment holds two OpenRouter credentials belonging to two separate
    # financial accounts. One account's balance must never be published as though it
    # were the deployment's OpenRouter position, and the two must never be added.
    document = _request()["dashboard"]
    config = _config()
    secondary = config["openrouter_secondary_account"]
    titles = [panel.get("title", "") for panel in document["panels"]]
    for field in ("Status", "Remaining (USD)", "Purchased Credit (USD)", "Usage (total, USD)"):
        assert f"{secondary} {field}" in titles, f"the second account must show its {field}"

    # Served traffic may be aggregated across both credentials: it is a count of
    # requests, not money. Money may not be. No financial query may span both
    # accounts, because the two balances are separate credit.
    financial_families = (
        "switchyard_economics_provider_balance",
        "switchyard_economics_provider_credit",
        "switchyard_economics_provider_usage",
        "switchyard_economics_provider_balance_component",
    )
    for panel in document["panels"]:
        for target in panel.get("targets", []):
            expression = target.get("expr", "")
            if not any(family in expression for family in financial_families):
                continue
            spans_both = 'provider=~"openrouter.*"' in expression or (
                'provider="openrouter"' in expression and secondary in expression
            )
            assert not spans_both, (
                f"{panel.get('title')} aggregates a financial figure across both "
                "OpenRouter accounts"
            )

    # Each account's money panels select that account explicitly or by its own name.
    money = [
        panel
        for panel in document["panels"]
        if any(word in panel.get("title", "") for word in ("Remaining (", "Purchased Credit"))
        and "OpenRouter" in panel.get("title", "") or secondary in panel.get("title", "")
    ]
    assert money
    for panel in money:
        joined = " ".join(t.get("expr", "") for t in panel.get("targets", []))
        assert 'provider="openrouter"' in joined or f'provider="{secondary}"' in joined, (
            f"{panel.get('title')} does not name the account it reads"
        )


def test_the_rolling_row_states_its_coverage_and_its_limits() -> None:
    # A 24-hour rolling total is only a day if the process has been up for a day.
    # Without a coverage figure an operator reads a partial window as a quiet one.
    document = _request()["dashboard"]
    titles = [panel.get("title", "") for panel in document["panels"]]
    assert "1h Window Coverage" in titles
    assert "24h Window Coverage" in titles

    body = "\n".join(
        panel["options"]["content"]
        for panel in document["panels"]
        if panel.get("type") == "text" and "ROLLING WINDOWS" in panel.get("title", "")
    )
    assert body, "the rolling row must disclose how its numbers are produced"
    for claim in ("in memory", "restart", "one minute", "Bangkok calendar day"):
        assert claim.lower() in body.lower(), f"the disclosure must mention {claim!r}"


def test_the_rolling_figures_come_from_events_and_not_from_increase() -> None:
    # increase() extrapolates from whichever samples exist, so a window containing a
    # restart reports usage it never held. The rolling families are the fix for that
    # and must not themselves be derived with increase().
    document = _request()["dashboard"]
    rolling = [
        panel
        for panel in document["panels"]
        if any("switchyard_economics_rolling_" in t.get("expr", "")
               for t in panel.get("targets", []))
    ]
    assert rolling, "the rolling panels must exist"
    for panel in rolling:
        for target in panel.get("targets", []):
            assert "increase(" not in target.get("expr", ""), (
                f"{panel.get('title')} derives a rolling figure with increase()"
            )


def test_the_deepseek_balance_parts_are_not_added_to_the_total() -> None:
    # Granted and topped-up credit are parts of the reported balance, not additions
    # to it. A query that summed them with the total would count the balance twice.
    document = _request()["dashboard"]
    parts = [
        panel
        for panel in document["panels"]
        if "provider_balance_component" in json.dumps(panel)
    ]
    assert parts, "the balance-component panels must exist"
    for panel in parts:
        for target in panel.get("targets", []):
            expression = target.get("expr", "")
            assert "provider_balance_component" in expression
            # A component query may not also select the total and sum the two.
            assert not (
                "provider_balance{" in expression and "sum(" in expression
            ), f"{panel.get('title')} combines the components with the total"


def test_no_panel_still_claims_openrouter_coverage_is_partial() -> None:
    # The second account is now read. Text that still says it is not collected would
    # contradict the panels beside it.
    document = _request()["dashboard"]
    for panel in document["panels"]:
        blob = json.dumps(panel)
        assert "has no account figures" not in blob
        assert "not collected**, so its spend is invisible" not in blob


def test_acceptance_checks_ignore_placeholder_series() -> None:
    # The read tool answers an unknown metric family with a series that carries no
    # points rather than with an empty result. Counting series would therefore
    # report an absent family as present - the exact inverse of the truth - so the
    # acceptance check filters on points, not on series count.
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "economics"))
    import acceptance_checks

    scrape = acceptance_checks.parse_scrape(
        'switchyard_economics_rolling_requests{window="1h"} 3\n'
        'switchyard_economics_provider_balance{provider="deepseek",currency="CNY"} 74.36\n'
    )
    assert "switchyard_economics_rolling_requests" in scrape
    assert len(scrape["switchyard_economics_rolling_requests"]) == 1
    assert scrape["switchyard_economics_rolling_requests"][0][1] == 3.0
    # A family that is absent from the scrape is absent, not a zero series.
    assert "switchyard_economics_provider_balance_component" not in scrape


def test_cost_panels_can_show_a_sub_cent_amount() -> None:
    # A provider-reported cost is frequently far below one cent. With four
    # decimals and a non-currency unit, a real cost of 8.248e-06 renders as
    # "0.0000" - visually a measured zero, which is the one thing a cost panel
    # must never look like when the figure is non-zero.
    document = _request()["dashboard"]
    cost_panels = [
        panel
        for panel in document["panels"]
        if panel.get("type") != "row"
        and "Cost" in panel.get("title", "")
        and any(
            "provider_cost" in t.get("expr", "") for t in panel.get("targets", [])
        )
    ]
    assert cost_panels, "the cost panels must exist"
    for panel in cost_panels:
        defaults = panel.get("fieldConfig", {}).get("defaults", {})
        unit = defaults.get("unit")
        decimals = defaults.get("decimals")
        assert unit and unit.startswith("currency"), (
            f"{panel.get('title')} shows money with unit {unit!r}"
        )
        assert decimals is not None and decimals >= 6, (
            f"{panel.get('title')} renders sub-cent costs as zero (decimals={decimals})"
        )
