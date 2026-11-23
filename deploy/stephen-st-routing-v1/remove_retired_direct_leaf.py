#!/usr/bin/env python3
"""Remove one retired direct leaf and its target, after auditing every consumer.

A direct alias whose upstream provider no longer serves the model cannot be
qualified, cannot serve, and cannot stay in the candidate: P9 refuses an
unqualified direct alias, and leaving it in place would keep a permanently
4xx route in production.  Removal is still a change to a served surface, so it
happens only when the audit proves it is safe:

- the route is a 1:1 passthrough to exactly the named target;
- no other route names either the target or the route;
- the provider receipt says the model is retired (404, quoting the provider);
- the document keeps every other byte: the only delta is the two sections.

Usage:
  python3 remove_retired_direct_leaf.py --baseline candidate.toml \
      --evidence retired_direct_leaf_evidence.json --out candidate2.toml
"""
from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path

TARGET_FIELDS = ("target", "terminal_fallback", "escalation_target")


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


def references(document: dict) -> list[tuple[str, str, str]]:
    """Every (route, field, value) that names a target or route identifier.

    Values are compared as whole identifiers.  A substring test reads
    ``direct-google-gpt-6-luna-nt`` as a reference to ``openai-gpt-6-luna`` and
    refuses a legitimate removal, so each reference keeps its field and value.
    """
    found = []
    routes = document.get("routes") or {}
    for route_name, route in routes.items():
        for field in TARGET_FIELDS:
            raw = route.get(field)
            values = raw if isinstance(raw, list) else ([raw] if raw else [])
            for value in values:
                found.append((route_name, field, value))
        for candidate in route.get("candidates") or []:
            found.append((route_name, "candidates", candidate))
        groups = route.get("pool_groups")
        # pool_groups is a list of group names in this manifest, but a mapping
        # in older shapes; audit both so a reference can never hide.
        if isinstance(groups, list):
            for item in groups:
                found.append((route_name, "pool_groups", item))
        elif isinstance(groups, dict):
            for group_name, group in groups.items():
                if isinstance(group, list):
                    for item in group:
                        found.append((route_name, f"pool_groups.{group_name}", item))
    return found


def plan(document: dict, evidence: dict) -> tuple[str, str]:
    route_name = evidence.get("route")
    target_name = evidence.get("target")
    if not route_name or not target_name:
        raise Refused("REFUSED_EVIDENCE_INCOMPLETE: route and target are required")
    if evidence.get("provider_status") != 404:
        raise Refused(f"REFUSED_NOT_RETIRED: provider_status="
                      f"{evidence.get('provider_status')!r}")
    message = evidence.get("provider_message") or ""
    if "no longer available" not in message:
        raise Refused("REFUSED_NOT_RETIRED: provider message does not say the "
                      "model is gone")
    if not evidence.get("receipt"):
        raise Refused("REFUSED_EVIDENCE_INCOMPLETE: receipt is required")

    routes = document.get("routes") or {}
    targets = document.get("targets") or {}
    route = routes.get(route_name)
    if route is None:
        raise Refused(f"REFUSED_UNKNOWN_ROUTE {route_name}")
    if route.get("type") != "passthrough":
        raise Refused(f"REFUSED_NOT_PASSTHROUGH {route_name}")
    if route.get("target") != target_name:
        raise Refused(f"REFUSED_TARGET_MISMATCH {route_name}: "
                      f"{route.get('target')!r} != {target_name!r}")
    if route.get("candidates"):
        raise Refused(f"REFUSED_ROUTE_HAS_CANDIDATES {route_name}")
    if target_name not in targets:
        raise Refused(f"REFUSED_UNKNOWN_TARGET {target_name}")

    for route_other, where, value in references(document):
        if route_other == route_name:
            continue
        if value == target_name:
            raise Refused(f"REFUSED_CONSUMER_EXISTS {target_name}: "
                          f"{route_other} {where}={value}")
        if value == route_name:
            raise Refused(f"REFUSED_ROUTE_BEING_REFERENCED {route_name}: "
                          f"{route_other} {where}={value}")
    return route_name, target_name


def apply(text: str, evidence: dict) -> tuple[str, list[str]]:
    document = tomllib.loads(text)
    route_name, target_name = plan(document, evidence)
    lines = text.splitlines(keepends=True)
    sections = headings(text)
    spans = [sections[heading_for("routes", route_name, sections)],
             sections[heading_for("targets", target_name, sections)]]
    # Remove from the end so earlier spans stay valid.
    for start, end in sorted(spans, reverse=True):
        del lines[start:end]
    out = "".join(lines)

    parsed = tomllib.loads(out)
    if route_name in (parsed.get("routes") or {}):
        raise Refused("REFUSED_ROUTE_SURVIVED")
    if target_name in (parsed.get("targets") or {}):
        raise Refused("REFUSED_TARGET_SURVIVED")
    for kind in ("routes", "targets", "llm_clients"):
        before = set((document.get(kind) or {}))
        after = set((parsed.get(kind) or {}))
        expected = {route_name} if kind == "routes" else (
            {target_name} if kind == "targets" else set())
        if before - after != expected:
            raise Refused(f"REFUSED_UNEXPECTED_REMOVAL {kind}: "
                          f"{sorted(before - after)}")
        if after - before:
            raise Refused(f"REFUSED_UNEXPECTED_ADDITION {kind}: {sorted(after - before)}")
        for name in before & after:
            if document[kind][name] != parsed[kind][name]:
                raise Refused(f"REFUSED_VALUE_CHANGED {kind}.{name}")
    return out, [route_name, target_name]


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
        out, removed = apply(args.baseline.read_text(), evidence)
        args.out.write_text(out)
    except (OSError, ValueError, json.JSONDecodeError, tomllib.TOMLDecodeError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 3
    print("REMOVED retired leaf: " + ", ".join(removed))
    return 0


if __name__ == "__main__":
    sys.exit(main())
