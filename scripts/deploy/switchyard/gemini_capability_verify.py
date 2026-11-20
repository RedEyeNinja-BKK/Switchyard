#!/usr/bin/env python3
"""Offline proof matrix for the Gemini free-project staging fragment.

Every check here answers one question about a document, and every check has a
negative control in the test module that proves it can fail. The verifier reads
files; it never contacts a provider, never touches a running deployment, and
never writes to one.

Checks are grouped by what they protect:

  inventory  the enumeration covers the evidence and keeps its allocation facts
  transport  nothing non-chat is represented as a chat leaf
  quota      no provider-model quota is reachable through two legs
  graph      composite children are routes, not composites
  claim      a route advertises no more context than its candidates can serve
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path

ROUTABLE_STATE = "EXECUTABLE_AND_DEPLOYED"
CHAT_OPERATIONS = frozenset({"chat.complete"})


class Ledger:
    """Collects findings so one run reports every failure, not the first."""

    def __init__(self) -> None:
        self.findings: list[dict] = []
        self.checks = 0

    def check(self, code: str, description: str, ok: bool, detail: str = "") -> None:
        self.checks += 1
        if not ok:
            self.findings.append({"code": code, "description": description, "detail": detail})

    @property
    def ok(self) -> bool:
        return not self.findings


def _routes(document: dict) -> dict:
    return document.get("routes") or {}


def _candidates(route: dict) -> list[str]:
    listed = route.get("candidates")
    if isinstance(listed, list):
        return [str(name) for name in listed]
    single = route.get("target")
    return [str(single)] if single else []


def verify_inventory(inventory: dict, crosswalk: dict | None, ledger: Ledger) -> None:
    models = inventory["models"]
    totals = inventory["totals"]

    ledger.check(
        "INV1-self-consistent",
        "the stated total is the number of rows enumerated",
        totals["models"] == len(models),
        f"{totals['models']} declared, {len(models)} enumerated",
    )
    if crosswalk is not None:
        # A row that silently vanishes passes INV1 whenever the total was
        # recomputed from the surviving rows, so coverage is proved against the
        # evidence itself: same indices, none missing, none invented.
        evidence_indices = sorted(row["index"] for row in crosswalk["models"])
        inventory_indices = sorted(model["index"] for model in models)
        ledger.check(
            "INV0-evidence-coverage",
            "the inventory covers exactly the rows the evidence contains",
            evidence_indices == inventory_indices,
            f"missing {sorted(set(evidence_indices) - set(inventory_indices))}, "
            f"extra {sorted(set(inventory_indices) - set(evidence_indices))}",
        )
        by_index_evidence = {row["index"]: row for row in crosswalk["models"]}
        lost: list[int] = []
        for model in models:
            evidence = by_index_evidence.get(model["index"])
            if evidence is None:
                continue
            display = evidence.get("quota_display") or {}
            for window in ("rpm", "tpm", "rpd"):
                source = display.get(window) or {}
                carried = model["quota"][window]
                if (
                    carried.get("limit") != source.get("limit")
                    or carried.get("peak_usage_selected_1d")
                    != source.get("peak_usage_selected_1d")
                ):
                    lost.append(model["index"])
                    break
        ledger.check(
            "INV2-selected-window-preserved",
            "every displayed peak/limit pair survives transcription unchanged",
            not lost,
            f"rows {sorted(set(lost))} differ from the evidence",
        )
        by_index = {model["index"]: model for model in models}
        mismatched = [
            row["index"]
            for row in crosswalk["models"]
            if row["index"] not in by_index
            or by_index[row["index"]]["native_model_id"] != row.get("native_model_id")
        ]
        ledger.check(
            "INV3-native-id-fidelity",
            "no native model id was rewritten",
            not mismatched,
            f"rows {mismatched} differ from the evidence",
        )
        invented = [
            model["index"]
            for model in models
            if model["native_model_id"] is None
            and model["state"] not in ("HELD_NATIVE_ID_UNRESOLVED", "HELD_NON_CHAT_TRANSPORT")
        ]
        ledger.check(
            "INV4-no-invented-id",
            "a row without a resolved native id is never treated as definable",
            not invented,
            f"rows {invented} claim a native id the evidence does not contain",
        )

    chat_leaf = [
        model["index"]
        for model in models
        if model["routable_today"] and model["operation_contract"] not in CHAT_OPERATIONS
    ]
    ledger.check(
        "TRN1-chat-is-chat",
        "no non-chat operation is routable as a chat leaf",
        not chat_leaf,
        f"rows {chat_leaf} would be served over a chat completion",
    )
    unqualified = [
        model["index"] for model in models if model["routable_today"] and model["state"] != ROUTABLE_STATE
    ]
    ledger.check(
        "TRN2-routable-is-qualified",
        "every routable row is marked executable",
        not unqualified,
        f"rows {unqualified} are routable without wire qualification",
    )
    silent = [
        model["index"]
        for model in models
        if not model["routable_today"] and not model["adapter_requirement"]
    ]
    ledger.check(
        "TRN3-held-states-a-reason",
        "every held row states what would release it",
        not silent,
        f"rows {silent} are held with no gate",
    )


def verify_graph(graph: dict, ledger: Ledger) -> None:
    partition = graph["quota_partition"]
    ledger.check(
        "QTA1-one-leg-per-model",
        "no candidate is reachable through two primitives",
        partition["one_primitive_per_candidate"],
        json.dumps(partition["duplicates"]),
    )
    primitives = graph["primitives"]
    composite = graph["composite"]["smart_gemini"]
    crossing = [child for child in composite["children"] if child not in primitives]
    ledger.check(
        "GRP1-children-are-primitives",
        "every composite child names a primitive route",
        not crossing,
        f"unknown children {crossing}",
    )
    ledger.check(
        "GRP2-children-are-not-composites",
        "no composite child is itself a composite",
        not [child for child in composite["children"] if child in graph["composite"]],
        "nested composites are refused at load",
    )


def verify_fragment(fragment: dict, inventory: dict, graph: dict, ledger: Ledger) -> None:
    by_name = {
        model["proposed_direct_contract_name"]: model
        for model in inventory["models"]
        if model["proposed_direct_contract_name"]
    }
    routes = _routes(fragment)
    composites = fragment.get("composites") or {}

    claimed: dict[str, str] = {}
    seen_in: dict[str, list[str]] = {}
    for name, route in routes.items():
        for candidate in _candidates(route):
            seen_in.setdefault(candidate, []).append(name)
    duplicated = {c: names for c, names in seen_in.items() if len(names) > 1}
    ledger.check(
        "QTA2-fragment-one-leg-per-model",
        "a candidate appears in one route only",
        not duplicated,
        json.dumps(duplicated),
    )

    for name, route in routes.items():
        for candidate in _candidates(route):
            model = by_name.get(candidate)
            if model is None:
                ledger.check(
                    "CND1-candidate-is-enumerated",
                    f"{name} names {candidate}, which the inventory does not define",
                    False,
                    "a route may only reference an enumerated contract",
                )
                continue
            ledger.check(
                "CND2-candidate-is-routable",
                f"{name} candidate {candidate} is wire-qualified",
                model["routable_today"],
                f"state is {model['state']}",
            )
            ledger.check(
                "CND3-candidate-is-chat",
                f"{name} candidate {candidate} serves a chat completion",
                model["operation_contract"] in CHAT_OPERATIONS,
                f"operation is {model['operation_contract']}",
            )
            claimed[candidate] = name

        window = route.get("context_window")
        if window is not None:
            # A route must not advertise more context than its smallest candidate
            # can take; the promise is made to callers before any target is chosen.
            limits = [
                by_name[c]["native_input_limit"]
                for c in _candidates(route)
                if c in by_name and by_name[c].get("native_input_limit")
            ]
            if limits:
                ledger.check(
                    "CLM1-context-within-candidates",
                    f"{name} advertises at most its smallest candidate's input limit",
                    int(window) <= min(limits),
                    f"route {window} vs smallest candidate {min(limits)}",
                )

    for name, composite in composites.items():
        for child in composite.get("children") or []:
            ledger.check(
                "GRP3-child-is-a-route",
                f"composite {name} child {child} is a route in this document",
                child in routes,
                "a composite child must be a route",
            )
            ledger.check(
                "GRP4-child-is-not-a-composite",
                f"composite {name} child {child} is not another composite",
                child not in composites,
                "nested composites are refused at load",
            )

    apply_today = graph["composite"]["smart_gemini"]["emittable"]
    for name, primitive in graph["primitives"].items():
        should_be_present = primitive["emittable"] and apply_today
        if should_be_present:
            ledger.check(
                "GRP5-emitted-is-intended",
                f"emitted route {name} is recorded as emittable in the graph",
                name in routes,
                "the graph and the document disagree about what is live",
            )
        else:
            ledger.check(
                "GRP6-held-is-absent",
                f"held primitive {name} is absent from the document",
                name not in routes,
                "a held primitive was emitted",
            )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--graph", required=True, type=Path)
    parser.add_argument("--fragment", required=True, type=Path)
    parser.add_argument("--crosswalk", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)

    inventory = json.loads(args.inventory.read_text())
    graph = json.loads(args.graph.read_text())
    crosswalk = json.loads(args.crosswalk.read_text()) if args.crosswalk else None
    try:
        fragment = tomllib.loads(args.fragment.read_text())
    except tomllib.TOMLDecodeError as error:
        print(f"fragment does not parse: {error}", file=sys.stderr)
        return 2

    ledger = Ledger()
    verify_inventory(inventory, crosswalk, ledger)
    verify_graph(graph, ledger)
    verify_fragment(fragment, inventory, graph, ledger)

    report = {
        "checks": ledger.checks,
        "failed": len(ledger.findings),
        "findings": ledger.findings,
        "verdict": "PASS" if ledger.ok else "FAIL",
    }
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n")
    for finding in ledger.findings:
        print(f"FAIL {finding['code']}: {finding['description']}")
        if finding["detail"]:
            print(f"     {finding['detail']}")
    print(f"{ledger.checks - len(ledger.findings)}/{ledger.checks} checks passed -- {report['verdict']}")
    return 0 if ledger.ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
