#!/usr/bin/env python3
"""Validate a generated economics dashboard before it is published.

Three independent gates, all of which must pass:

  1. STRUCTURE   no overlapping panels, no panel off the 24-column grid, ids
                 unique, every panel has a title.
  2. PROVENANCE  no expression and no text mentions a retired family or a
                 retired scrape job; every money selector pins scope="account".
  3. LIVE        every expression parses in Prometheus and is reported, with
                 template variables widened to `.*`; panels that read zero are
                 listed as LAZY and must be declared in EXPECTED_LAZY, so an
                 unexpectedly empty panel fails rather than blends in.

Usage:
    python3 verify_generated_dashboard.py <generated.json>
"""
from __future__ import annotations

import json
import re
import sys
import urllib.parse
import urllib.request

BASE = "http://10.10.20.33:9090/api/v1"

# Panels that are empty whenever their trigger has not fired since the process
# started. Each is a real measurement whose series materialises on an event.
EXPECTED_LAZY = {
    "PROVIDER-REPORTED COST (TOTAL)": "no provider reported a per-request cost since restart",
    "Provider-Reported Cost Today (Bangkok)": "same",
    "Rolling 24h Provider-Reported Cost": "same",
    "Auth Failures (total)": "no account read has been rejected since restart",
    "Account Reads Failed (total)": "no account read has failed since restart",
}

BANNED = ("localclaw_", 'job="localclaw-econ"', 'job="localclaw"', 'job="switchyard"')
MONEY = ("switchyard_economics_provider_balance",
         "switchyard_economics_provider_credit",
         "switchyard_economics_provider_usage")


def widen(expr: str) -> str:
    expr = re.sub(r"\$\{[^}:]+(?::[^}]*)?\}", ".*", expr)
    return re.sub(r"\$[A-Za-z_][A-Za-z0-9_]*", ".*", expr)


def ask(expr: str):
    url = f"{BASE}/query?" + urllib.parse.urlencode({"query": widen(expr)})
    try:
        body = json.load(urllib.request.urlopen(url, timeout=25))
    except Exception as exc:  # transport, not a verdict
        return "error", str(exc)
    if body.get("status") != "success":
        return "error", body.get("error", "unknown")
    return "ok", len(body["data"]["result"])


def structure(doc: dict) -> list[str]:
    bad, seen, ids = [], [], set()
    for p in doc["panels"]:
        pid = p.get("id")
        if pid in ids:
            bad.append(f"duplicate panel id {pid}")
        ids.add(pid)
        if not p.get("title"):
            bad.append(f"panel {pid} has no title")
        gp = p.get("gridPos") or {}
        x, y, w, h = gp.get("x"), gp.get("y"), gp.get("w"), gp.get("h")
        if None in (x, y, w, h):
            bad.append(f"panel {pid} has an incomplete gridPos")
            continue
        if x + w > 24:
            bad.append(f"panel {pid} '{p['title']}' overflows the grid ({x}+{w})")
        for (ox, oy, ow, oh, otitle) in seen:
            if x < ox + ow and ox < x + w and y < oy + oh and oy < y + h:
                bad.append(f"panel {pid} '{p['title']}' overlaps '{otitle}'")
        seen.append((x, y, w, h, p["title"]))
    return bad


def provenance(doc: dict) -> list[str]:
    bad = []
    for p in doc["panels"]:
        blob = json.dumps(p)
        for token in BANNED:
            if token in blob:
                bad.append(f"panel {p.get('id')} '{p.get('title')}' mentions {token}")
        for t in p.get("targets") or []:
            e = t.get("expr") or ""
            if any(m in e for m in MONEY) and "scope=" not in e:
                # an unpinned money selector matches the account AND the key
                # vantage point of the same balance and counts it twice.
                bad.append(
                    f"panel {p.get('id')} money query does not pin scope: {e[:70]}")
    return bad


def live(doc: dict):
    live, lazy, broken = [], [], []
    for p in doc["panels"]:
        for t in p.get("targets") or []:
            e = t.get("expr")
            if not e:
                continue
            state, val = ask(e)
            if state == "error":
                broken.append((p.get("title"), e[:80], val))
            elif val == 0:
                lazy.append((p.get("title"), e[:80]))
            else:
                live.append(p.get("title"))
    return live, lazy, broken


def main() -> int:
    path = sys.argv[1]
    raw = json.load(open(path))
    doc = raw.get("dashboard") or raw.get("document") or raw
    rc = 0

    print(f"== structure ({len(doc['panels'])} panels) ==")
    bad = structure(doc)
    print("   FAIL" if bad else "   ok")
    for b in bad[:10]:
        print("     -", b)
    rc |= 1 if bad else 0

    print("== provenance ==")
    bad = provenance(doc)
    print("   FAIL" if bad else "   ok: no retired family or scrape job referenced")
    for b in bad[:10]:
        print("     -", b)
    rc |= 1 if bad else 0

    print("== live against Prometheus ==")
    ok, empty, broken = live(doc)
    print(f"   reported: {len(ok)}   empty: {len(empty)}   broken: {len(broken)}")
    for t, e, why in broken[:10]:
        print(f"     BROKEN {t} :: {e} :: {why}")
    rc |= 1 if broken else 0
    undeclared = []
    for t, e in empty:
        if t not in EXPECTED_LAZY:
            undeclared.append((t, e))
    for t, r in EXPECTED_LAZY.items():
        hits = [x for x in empty if x[0] == t]
        print(f"     LAZY  {t}: {len(hits)} query(s) empty — {r}")
    if undeclared:
        print("   FAIL: panels empty for no declared reason:")
        for t, e in undeclared:
            print(f"     - {t} :: {e}")
        rc |= 1
    else:
        print("   ok: every empty panel is a declared lazy series")

    print("RESULT:", "PASS" if rc == 0 else "FAIL")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
