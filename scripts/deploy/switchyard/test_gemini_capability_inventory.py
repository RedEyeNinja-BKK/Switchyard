#!/usr/bin/env python3
"""Tests and negative controls for the Gemini capability inventory and verifier.

Every check in the verifier has a control here that makes it fail, because a
check that cannot fail is not a check. The suite is stdlib-only and runs itself:
the host that operates this deployment has no pytest, and a missing test
framework must not be the reason a staging document ships unverified.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gemini_capability_inventory as inventory_module
import gemini_capability_verify as verify_module


# --------------------------------------------------------------------------
# A crosswalk fragment shaped like the owner's evidence, small enough to reason
# about by hand and wide enough to exercise every classification branch.
# --------------------------------------------------------------------------

def _model(index, name, contract, status, native, *, executable=False, rpm=5, tpm=250_000, rpd=20,
           in_limit=1_048_576, out_limit=65_536):
    return {
        "index": index,
        "display_name": name,
        "ai_studio_category": "text",
        "quota_display": {
            "rpm": {"limit": rpm, "peak_usage_selected_1d": 0},
            "tpm": {"limit": tpm, "peak_usage_selected_1d": 0},
            "rpd": {"limit": rpd, "peak_usage_selected_1d": 0},
        },
        "project_quota_status": status,
        "native_model_id": native,
        "native_model_id_candidates": [native] if native else [],
        "native_id_evidence": "TEST_FIXTURE",
        "cached_input_token_limit": in_limit,
        "cached_output_token_limit": out_limit,
        "cached_supported_generation_methods": ["generateContent"],
        "operation_contract": contract,
        "capability_route_candidates": ["smart-gemini-chat"],
        "proposed_direct_contract_name": f"direct-google-{name.lower().replace(' ', '-')}" if native else None,
        "direct_contract_is_executable": executable,
        "route_status": "EXECUTABLE" if executable else "HELD",
        "qualification_notes": None,
    }


def fixture_crosswalk() -> dict:
    models = [
        _model(1, "Live Chat", "chat.complete", "POSITIVE_FINITE_RPD", "chat-live", executable=True),
        _model(2, "Held Chat", "chat.complete", "POSITIVE_FINITE_RPD", "chat-held"),
        _model(3, "Unnamed Chat", "chat.complete", "POSITIVE_FINITE_RPD", None),
        _model(4, "Zero Chat", "chat.complete", "ZERO_RPD", "chat-zero", rpm=0, tpm=0, rpd=0),
        _model(5, "Live Session", "session.live", "UNLIMITED_RPD", "session-live-1", rpm=None, rpd=None),
        _model(6, "Speech", "audio.synthesize", "POSITIVE_FINITE_RPD", "speech-1"),
        _model(7, "Small Chat", "chat.complete", "POSITIVE_FINITE_RPD", "chat-small",
               executable=True, tpm=16_000, rpd=14_400, in_limit=262_144),
        _model(8, "Second Chat", "chat.complete", "POSITIVE_FINITE_RPD", "chat-second",
               executable=True),
    ]
    return {
        "schema": 1,
        "kind": "fixture",
        "owner_confirmed_date": "2026-11-21",
        "project": "fixture",
        "tier": "Free",
        "source_official_doc": "fixture",
        "current_switchyard_release": "fixture",
        "totals": {"all_models": len(models), "feature_rows": 1},
        "models": models,
        "feature_quotas": [
            {
                "index": 1,
                "display_name": "Search grounding",
                "rpd": {"limit": 1500, "peak_usage_selected_1d": 0},
                "project_feature_status": "POSITIVE_FEATURE_QUOTA",
                "is_model_direct_route": False,
                "routing_role": "HELD",
                "notes": None,
            }
        ],
    }


def build_fixture(tmp_path: Path):
    crosswalk = fixture_crosswalk()
    path = tmp_path / "crosswalk.json"
    path.write_text(json.dumps(crosswalk))
    inventory = inventory_module.build(path)
    graph = inventory_module.route_graph(inventory)
    # Parsed from the generator's own output, so the document under test is the
    # document that ships rather than a second implementation of it.
    import tomllib

    fragment = tomllib.loads(inventory_module.staged_toml(inventory, graph))
    return crosswalk, inventory, graph, fragment


def codes(ledger) -> set[str]:
    return {finding["code"] for finding in ledger.findings}


def run_verify(inventory, graph, fragment, crosswalk=None):
    ledger = verify_module.Ledger()
    verify_module.verify_inventory(inventory, crosswalk, ledger)
    verify_module.verify_graph(graph, ledger)
    verify_module.verify_fragment(fragment, inventory, graph, ledger)
    return ledger


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------

def test_a_chat_operation_is_classified_by_wire_qualification(tmp_path):
    _, inventory, _, _ = build_fixture(tmp_path)
    states = {row["display_name"]: row["state"] for row in inventory["models"]}
    assert states["Live Chat"] == "EXECUTABLE_AND_DEPLOYED", states
    assert states["Held Chat"] == "HELD_WIRE_UNQUALIFIED", states
    assert states["Unnamed Chat"] == "HELD_NATIVE_ID_UNRESOLVED", states
    assert states["Zero Chat"] == "HELD_ZERO_RPD_ALLOWANCE_UNSHOWN", states


def test_a_non_chat_operation_is_never_classified_as_chat(tmp_path):
    _, inventory, _, _ = build_fixture(tmp_path)
    for row in inventory["models"]:
        if row["operation_contract"] != "chat.complete":
            assert not row["chat_capable"], row
            assert row["state"] == "HELD_NON_CHAT_TRANSPORT", row
            assert row["adapter_requirement"], row


def test_a_zero_rpd_row_is_retained_and_never_routed(tmp_path):
    crosswalk, inventory, _, _ = build_fixture(tmp_path)
    zero = [row for row in inventory["models"] if row["display_name"] == "Zero Chat"]
    assert len(zero) == 1, "a zero observed peak must not delete the row"
    assert zero[0]["routable_today"] is False, zero
    assert inventory["totals"]["models"] == len(crosswalk["models"])


def test_a_small_per_minute_budget_qualifies_for_bounded_work_only(tmp_path):
    _, inventory, _, _ = build_fixture(tmp_path)
    small = [row for row in inventory["models"] if row["display_name"] == "Small Chat"][0]
    assert small["small_bounded_only"] is True, small
    assert "small bounded agent tasks" in small["bounded_note"], small
    live = [row for row in inventory["models"] if row["display_name"] == "Live Chat"][0]
    assert live["small_bounded_only"] is False, live


def test_a_feature_quota_is_never_recorded_as_a_model_endpoint(tmp_path):
    _, inventory, _, _ = build_fixture(tmp_path)
    for row in inventory["feature_quotas"]:
        assert row["is_model_direct_route"] is False, row
        assert row["routing_role"] == "HELD_TOOL_QUOTA_NOT_A_MODEL_ENDPOINT", row


# --------------------------------------------------------------------------
# Quota partition
# --------------------------------------------------------------------------

def test_two_legs_over_one_model_are_refused(tmp_path):
    _, inventory, graph, _ = build_fixture(tmp_path)
    doubled = copy.deepcopy(graph)
    doubled["primitives"]["smart-gemini-long-context"]["candidates_today"] = list(
        doubled["primitives"]["smart-gemini-chat"]["candidates_today"]
    )
    doubled["quota_partition"] = inventory_module.route_graph.__globals__["route_graph"](
        inventory
    )["quota_partition"]
    ledger = verify_module.Ledger()
    verify_module.verify_graph(doubled, ledger)
    # The graph alone cannot see a hand-edited candidate list; the document check can.
    fragment = {
        "routes": {
            "smart-gemini-chat": {"candidates": ["direct-google-live-chat"]},
            "smart-gemini-long-context": {"candidates": ["direct-google-live-chat"]},
        },
        "composites": {},
    }
    ledger = verify_module.Ledger()
    verify_module.verify_fragment(fragment, inventory, doubled, ledger)
    assert "QTA2-fragment-one-leg-per-model" in codes(ledger), codes(ledger)


def test_the_generator_assigns_each_candidate_to_one_primitive(tmp_path):
    _, inventory, graph, _ = build_fixture(tmp_path)
    assert graph["quota_partition"]["one_primitive_per_candidate"] is True, graph
    claim = {}
    for name, primitive in graph["primitives"].items():
        for candidate in primitive["candidates_today"]:
            assert candidate not in claim, (candidate, claim[candidate], name)
            claim[candidate] = name


# --------------------------------------------------------------------------
# Transport and claim controls
# --------------------------------------------------------------------------

def test_a_live_model_in_a_chat_route_is_refused(tmp_path):
    _, inventory, graph, fragment = build_fixture(tmp_path)
    bad = copy.deepcopy(fragment)
    bad["routes"]["smart-gemini-chat"]["candidates"].append("direct-google-live-session")
    ledger = verify_module.Ledger()
    verify_module.verify_fragment(bad, inventory, graph, ledger)
    assert "CND2-candidate-is-routable" in codes(ledger), codes(ledger)


def test_a_route_wider_than_its_smallest_candidate_is_refused(tmp_path):
    _, inventory, graph, fragment = build_fixture(tmp_path)
    bad = copy.deepcopy(fragment)
    bad["routes"]["smart-gemini-chat"]["candidates"] = ["direct-google-small-chat"]
    bad["routes"]["smart-gemini-chat"]["context_window"] = 1_048_576
    ledger = verify_module.Ledger()
    verify_module.verify_fragment(bad, inventory, graph, ledger)
    assert "CLM1-context-within-candidates" in codes(ledger), codes(ledger)


def test_a_nested_composite_is_refused(tmp_path):
    _, inventory, graph, fragment = build_fixture(tmp_path)
    bad = copy.deepcopy(fragment)
    bad["composites"]["inner"] = {"children": ["smart-gemini-chat"]}
    bad["composites"]["smart_gemini"]["children"] = ["inner"]
    ledger = verify_module.Ledger()
    verify_module.verify_fragment(bad, inventory, graph, ledger)
    assert "GRP4-child-is-not-a-composite" in codes(ledger), codes(ledger)


def test_a_held_primitive_that_was_emitted_is_refused(tmp_path):
    _, inventory, graph, fragment = build_fixture(tmp_path)
    bad = copy.deepcopy(fragment)
    bad["routes"]["smart-gemini-vision"] = {"candidates": [], "context_window": 1_048_576}
    ledger = verify_module.Ledger()
    verify_module.verify_fragment(bad, inventory, graph, ledger)
    assert "GRP6-held-is-absent" in codes(ledger), codes(ledger)


def test_a_clean_document_passes(tmp_path):
    crosswalk, inventory, graph, fragment = build_fixture(tmp_path)
    ledger = run_verify(inventory, graph, fragment, crosswalk)
    assert ledger.ok, ledger.findings


# --------------------------------------------------------------------------
# Inventory controls
# --------------------------------------------------------------------------

def test_an_invented_native_id_is_refused(tmp_path):
    crosswalk, inventory, graph, fragment = build_fixture(tmp_path)
    bad = copy.deepcopy(inventory)
    for row in bad["models"]:
        if row["display_name"] == "Unnamed Chat":
            row["native_model_id"] = "chat-invented"
            row["state"] = "HELD_WIRE_UNQUALIFIED"
    ledger = verify_module.Ledger()
    verify_module.verify_inventory(bad, crosswalk, ledger)
    assert "INV3-native-id-fidelity" in codes(ledger), codes(ledger)


def test_a_held_row_without_a_gate_is_refused(tmp_path):
    crosswalk, inventory, graph, fragment = build_fixture(tmp_path)
    bad = copy.deepcopy(inventory)
    for row in bad["models"]:
        if row["display_name"] == "Live Session":
            row["adapter_requirement"] = None
    ledger = verify_module.Ledger()
    verify_module.verify_inventory(bad, crosswalk, ledger)
    assert "TRN3-held-states-a-reason" in codes(ledger), codes(ledger)


def test_a_dropped_model_row_is_refused(tmp_path):
    crosswalk, inventory, graph, fragment = build_fixture(tmp_path)
    bad = copy.deepcopy(inventory)
    bad["models"] = bad["models"][:-1]
    bad["totals"]["models"] = len(bad["models"])
    ledger = verify_module.Ledger()
    verify_module.verify_inventory(bad, crosswalk, ledger)
    assert "INV0-evidence-coverage" in codes(ledger), codes(ledger)


def test_one_native_model_behind_two_contracts_is_refused(tmp_path):
    crosswalk = fixture_crosswalk()
    crosswalk["models"].append(
        _model(9, "Chat Twin", "chat.complete", "POSITIVE_FINITE_RPD", "chat-live")
    )
    crosswalk["totals"]["all_models"] = len(crosswalk["models"])
    path = tmp_path / "dupe.json"
    path.write_text(json.dumps(crosswalk))
    inventory = inventory_module.build(path)
    assert inventory["invariants"]["one_quota_counter_per_native_model"] is False, inventory


# --------------------------------------------------------------------------

def _run_standalone(argv):
    import inspect
    import tempfile
    import traceback

    selected = argv or [name for name, fn in sorted(globals().items())
                        if name.startswith("test_") and callable(fn)]
    failures = []
    for name in selected:
        function = globals().get(name)
        if function is None:
            print(f"no such test: {name}", file=sys.stderr)
            return 2
        parameters = list(inspect.signature(function).parameters)
        with tempfile.TemporaryDirectory(prefix="geminicap-") as tmp:
            try:
                function(**{"tmp_path": Path(tmp)} if parameters else {})
                print(f"PASS {name}")
            except Exception:
                failures.append(name)
                print(f"FAIL {name}")
                traceback.print_exc()
    print(f"\n{len(selected) - len(failures)} passed, {len(failures)} failed")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(_run_standalone(sys.argv[1:]))
