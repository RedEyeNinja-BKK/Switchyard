#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate the V9.0 economics dashboard before and after publication.

Four gates, all of which must pass:

  1. STRUCTURE   grid geometry, unique ids, every panel titled. Reused from the V8
                 verifier rather than reimplemented, so the two dashboards are held
                 to one definition of a well-formed layout.
  2. NATIVE      every metric reference belongs to the native contract. This is the
                 positive allowlist gate — it fails closed on an unfamiliar family
                 rather than on a remembered list of bad ones.
  3. PROVENANCE  no retired source is named anywhere in the document, and every
                 money selector pins `scope` so one balance is not counted twice.
  4. LIVE        every expression parses in Prometheus and returns series. A panel
                 that is legitimately empty must be declared here with the reason,
                 so an unexpectedly empty panel fails instead of blending in.

Usage:
    python3 verify_v9.py <generated.json> [--published <readback.json>]
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import native_contract  # noqa: E402
import verify_generated_dashboard as v8  # noqa: E402

# Panels with no series whenever their trigger has not occurred in this process.
# Every entry is a real measurement whose series materialises on an event; none of
# them is a broken query, and each says why it may be absent.
EXPECTED_EMPTY = {
    "AUTH FAILURES ON ACCOUNT READS": (
        "event-driven counter: appears only after a provider rejects our credential"
    ),
    "ACCOUNT READ FAILURES": "no account read has failed since this process started",
    "UPSTREAM ATTEMPTS BY STATUS OVER TIME": (
        "increase() over a 15m window with no attempts in that window"
    ),
    # Composition counters are recorded per child attempt of a WHOLE-ROUTE composite.
    # The live manifest routes every model through a passthrough route whose candidates
    # are targets, not routes, so no composite runs and the series does not exist. The
    # panels are kept because a composite route is the one place these facts are
    # observable, and an absent series is not a failed query.
    "ACCOUNT-WIDE QUOTA BLOCKS": (
        "no whole-route composite has run since this process started"
    ),
    "COMPOSITION OUTCOMES BY CAPABILITY": (
        "no whole-route composite has run since this process started"
    ),
    "ROUTE CROSSINGS BY REASON": (
        "no composite child has failed over to a later child since this process started"
    ),
}

MONEY = (
    "switchyard_economics_provider_balance",
    "switchyard_economics_provider_credit",
    "switchyard_economics_provider_usage",
)

RETIRED = ("localclaw_", 'job="localclaw-econ"', 'job="localclaw"', 'job="switchyard"', "hermes", "dsh_")


def native(doc: dict) -> list[str]:
    """Every metric reference must be in the native contract."""
    bad = []
    for panel in doc["panels"]:
        for t in panel.get("targets") or []:
            expr = t.get("expr") or ""
            for problem in native_contract.violations(expr):
                bad.append(f"'{panel.get('title')}' [{t.get('refId')}]: {problem}")
    return bad


def provenance(doc: dict) -> list[str]:
    bad = []
    for panel in doc["panels"]:
        blob = json.dumps(panel)
        for token in RETIRED:
            if token in blob:
                bad.append(f"'{panel.get('title')}' mentions a retired source: {token}")
        for t in panel.get("targets") or []:
            expr = t.get("expr") or ""
            if not any(m in expr for m in MONEY):
                continue
            if "scope=" in expr:
                continue
            # A panel may deliberately show both vantage points, but then it must
            # name the scope in each row so a reader can tell the rows apart and
            # does not add two views of one balance together.
            if "{{scope}}" in (t.get("legendFormat") or "") or "by (scope" in expr:
                continue
            bad.append(f"'{panel.get('title')}' money query does not pin scope: {expr[:70]}")
    return bad


def live(doc: dict):
    reported, empty, broken = [], [], []
    for panel in doc["panels"]:
        for t in panel.get("targets") or []:
            expr = t.get("expr")
            if not expr:
                continue
            state, value = v8.ask(expr)
            if state == "error":
                broken.append((panel.get("title"), expr[:90], value))
            elif value == 0:
                empty.append((panel.get("title"), expr[:90]))
            else:
                reported.append(panel.get("title"))
    return reported, empty, broken


def main() -> int:
    path = sys.argv[1]
    raw = json.load(open(path))
    doc = raw.get("dashboard") or raw.get("document") or raw
    rc = 0
    queries = sum(len(p.get("targets") or []) for p in doc["panels"])

    print(f"== structure ({len(doc['panels'])} panels, {queries} queries) ==")
    bad = v8.structure(doc)
    print("   FAIL" if bad else "   ok")
    for b in bad[:10]:
        print("     -", b)
    rc |= 1 if bad else 0

    print("== native contract ==")
    bad = native(doc)
    print("   FAIL" if bad else f"   ok: every reference is one of {len(native_contract.ALLOWED)} native families")
    for b in bad[:10]:
        print("     -", b)
    rc |= 1 if bad else 0

    print("== provenance ==")
    bad = provenance(doc)
    print("   FAIL" if bad else "   ok: no retired source, every money query pins scope")
    for b in bad[:10]:
        print("     -", b)
    rc |= 1 if bad else 0

    print("== live against Prometheus ==")
    reported, empty, broken = live(doc)
    print(f"   reported: {len(reported)}   empty: {len(empty)}   broken: {len(broken)}")
    for title, expr, why in broken[:10]:
        print(f"     BROKEN {title} :: {expr} :: {why}")
    rc |= 1 if broken else 0

    undeclared = [(t, e) for t, e in empty if t not in EXPECTED_EMPTY]
    for title, _expr in empty:
        reason = EXPECTED_EMPTY.get(title, "UNDECLARED")
        print(f"     EMPTY {title} — {reason}")
    if undeclared:
        print("   FAIL: panels empty for no declared reason:")
        for title, expr in undeclared:
            print(f"     - {title} :: {expr}")
        rc |= 1
    else:
        print("   ok: every empty panel is a declared event-driven absence")

    print("RESULT:", "PASS" if rc == 0 else "FAIL")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
