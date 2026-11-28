#!/usr/bin/env python3
"""Widen a direct alias's tool admission, but only from recorded evidence.

A ``direct-*`` route is a 1:1 passthrough, so its tool admission is a claim
about one physical target.  Some leaves were generated with
``tool_calling = false`` as a conservative placeholder while the model was
unqualified.  Once the target is proven to answer tool calls, the route
declaration is simply wrong, and the P9 contract refuses a route that
under-declares against a qualified target (DIRECT_SUPPRESSED_CAPABILITY).

This applier performs exactly that correction, and nothing else:

- it never invents a route, a target or a capability;
- it requires a recorded basis and receipt per route;
- it refuses when the target's own declaration contradicts the widening;
- it refuses when the route is not a clean 1:1 passthrough;
- it proves the only delta in the document is the widened key.

The opposite direction (true -> false) is never performed here.

Usage:
  python3 apply_route_tool_admission.py --baseline CURRENT/routes.toml \
      --evidence route_admission_evidence.json --out candidate.toml
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from pathlib import Path

import direct_capability_gate as gate

BASES = frozenset({"LIVE_TOOL_CALL", "SHARED_MODEL_SPEC", "PRIOR_MEASUREMENT"})


class Refused(ValueError):
    pass


def headings(text: str) -> dict[str, tuple[int, int]]:
    lines = text.splitlines(keepends=True)
    found = [(i, line.strip()) for i, line in enumerate(lines)
             if line.startswith("[") and line.rstrip().endswith("]")]
    sections = {}
    for index, (start, head) in enumerate(found):
        end = found[index + 1][0] if index + 1 < len(found) else len(lines)
        sections[head] = (start, end)
    return sections


def heading_for(kind: str, name: str, sections: dict[str, tuple[int, int]]) -> str:
    matches = [head for head in (f"[{kind}.{name}]", f'[{kind}."{name}"]')
               if head in sections]
    if len(matches) != 1:
        raise Refused(f"REFUSED_AMBIGUOUS_HEADING {kind}.{name}: {matches}")
    return matches[0]


def plan(document: dict, evidence: dict) -> dict[str, str]:
    """Return {route: receipt} for the routes that may be widened."""
    routes = document.get("routes") or {}
    targets = document.get("targets") or {}
    allowed: dict[str, str] = {}
    for route, record in sorted(evidence.items()):
        basis = record.get("basis")
        receipt = record.get("receipt")
        if basis not in BASES or not receipt:
            raise Refused(f"REFUSED_EVIDENCE_INCOMPLETE {route}: "
                          f"basis={basis!r} receipt={receipt!r}")
        if route not in routes:
            raise Refused(f"REFUSED_UNKNOWN_ROUTE {route}")
        declared = routes[route]
        if not route.startswith("direct-"):
            raise Refused(f"REFUSED_NOT_A_DIRECT_ALIAS {route}")
        if declared.get("type") != "passthrough":
            raise Refused(f"REFUSED_NOT_PASSTHROUGH {route}")
        if declared.get("candidates") or declared.get("terminal_fallback") \
                or declared.get("escalation_target") or declared.get("pool_groups"):
            raise Refused(f"REFUSED_HIDDEN_SUBSTITUTION {route}")
        current = declared.get("tool_calling")
        if current is not False:
            raise Refused(f"REFUSED_NOT_A_PLACEHOLDER {route}: "
                          f"tool_calling={current!r} (expected explicit false)")
        target_name = declared.get("target")
        target = targets.get(target_name)
        if target is None:
            raise Refused(f"REFUSED_TARGET_MISSING {route} -> {target_name}")
        if target.get("tool_calling") is False:
            raise Refused(f"REFUSED_CONTRADICTED_BY_TARGET {target_name}: "
                          "the target itself declares tools unsupported")
        if target.get("held"):
            raise Refused(f"REFUSED_TARGET_NOT_SERVABLE {target_name}")
        allowed[route] = receipt
    return allowed


def apply(text: str, evidence: dict) -> tuple[str, list[str]]:
    document = tomllib.loads(text)
    allowed = plan(document, evidence)
    lines = text.splitlines(keepends=True)
    sections = headings(text)
    # Rewrite the single placeholder line per route, from the end so earlier
    # spans stay valid.
    order = sorted(allowed, key=lambda name: sections[
        heading_for("routes", name, sections)][0], reverse=True)
    for route in order:
        head = heading_for("routes", route, sections)
        start, end = sections[head]
        hits = [i for i in range(start + 1, end)
                if re.match(r"^tool_calling\s*=\s*false\s*$", lines[i].rstrip() + "\n")]
        if len(hits) != 1:
            raise Refused(f"REFUSED_PLACEHOLDER_NOT_FOUND {route}: {len(hits)} matches")
        lines[hits[0]] = "tool_calling = true\n"
    out = "".join(lines)

    parsed = tomllib.loads(out)
    if set(parsed) != set(document):
        raise Refused("REFUSED_TOP_LEVEL_SECTION_CHANGED")
    for kind in ("routes", "targets", "llm_clients"):
        before, after = document.get(kind, {}), parsed.get(kind, {})
        if set(before) != set(after):
            raise Refused(f"REFUSED_{kind.upper()}_NAMES_CHANGED")
        for name in before:
            for key, value in before[name].items():
                changed = after[name].get(key)
                if changed == value:
                    continue
                if kind == "routes" and name in allowed \
                        and key == "tool_calling" and value is False and changed is True:
                    continue
                raise Refused(f"REFUSED_VALUE_CHANGED {kind}.{name}.{key}")
            extra = set(after[name]) - set(before[name])
            if extra:
                raise Refused(f"REFUSED_KEYSET_CHANGED {kind}.{name}: {sorted(extra)}")
    return out, sorted(allowed)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.baseline.resolve() == args.out.resolve():
        print("REFUSED: cannot overwrite baseline", file=sys.stderr)
        return 3
    try:
        evidence = json.loads(args.evidence.read_text())
        out, widened = apply(args.baseline.read_text(), evidence)
        args.out.write_text(out)
    except (OSError, ValueError, json.JSONDecodeError, tomllib.TOMLDecodeError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 3
    print(f"WIDENED {len(widened)} route(s) from explicit false to true: "
          + ", ".join(widened))
    return 0


if __name__ == "__main__":
    sys.exit(main())
