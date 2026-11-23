#!/usr/bin/env python3
"""Declare a target's own tool support as unsupported, from a provider receipt.

Why a separate applier from apply_target_tool_qualification.py: that tool is
deliberately one-way (it will only ever write ``true``) because a ``false`` it
invented would start refusing work that is served today.  This tool is the
narrow exception, and it is safe only under conditions it can check:

- the provider itself refused a tool-enabled request, and the receipt carries
  the provider's own message (not our inference);
- the target has exactly ONE referencing route in the manifest;
- that route already declares ``tool_calling = false``, so no request carrying
  tools can reach this target through it.

Under those conditions the declaration cannot change behaviour; it makes the
manifest state the fact the provider already enforces.  If any condition fails
the tool refuses and writes nothing.

Usage:
  python3 declare_target_tool_refusal.py --baseline candidate.toml \
      --evidence target_tool_refusal_evidence.json --out candidate2.toml
"""
from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path

BASIS = "PROVIDER_REFUSES_TOOLS"
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


def referencing_routes(document: dict, target_name: str) -> list[str]:
    found = []
    for route_name, route in (document.get("routes") or {}).items():
        refs = [route.get(field) for field in TARGET_FIELDS]
        refs.extend(route.get("candidates") or [])
        groups = route.get("pool_groups")
        if isinstance(groups, list):
            refs.extend(groups)
        elif isinstance(groups, dict):
            for group in groups.values():
                if isinstance(group, list):
                    refs.extend(group)
        if target_name in refs:
            found.append(route_name)
    return sorted(found)


def plan(document: dict, evidence: dict) -> dict[str, str]:
    targets = document.get("targets") or {}
    routes = document.get("routes") or {}
    allowed: dict[str, str] = {}
    for target_name, record in sorted(evidence.items()):
        if record.get("basis") != BASIS:
            raise Refused(f"REFUSED_BASIS {target_name}: {record.get('basis')!r}")
        receipt = record.get("receipt")
        message = record.get("provider_message")
        if not receipt or not message:
            raise Refused(f"REFUSED_EVIDENCE_INCOMPLETE {target_name}")
        if record.get("provider_status") not in (400, 501, 503):
            raise Refused(f"REFUSED_STATUS {target_name}: "
                          f"{record.get('provider_status')!r}")
        target = targets.get(target_name)
        if target is None:
            raise Refused(f"REFUSED_UNKNOWN_TARGET {target_name}")
        if "tool_calling" in target:
            raise Refused(f"REFUSED_KEY_ALREADY_PRESENT {target_name}")
        users = referencing_routes(document, target_name)
        if len(users) != 1:
            raise Refused(f"REFUSED_NOT_SOLE_CONSUMER {target_name}: {users}")
        route = routes[users[0]]
        if route.get("tool_calling") is not False:
            raise Refused(f"REFUSED_ROUTE_STILL_ADMITS_TOOLS {users[0]}")
        if route.get("candidates") or route.get("terminal_fallback") \
                or route.get("escalation_target") or route.get("pool_groups"):
            raise Refused(f"REFUSED_ROUTE_HAS_SUBSTITUTION {users[0]}")
        allowed[target_name] = receipt
    return allowed


def apply(text: str, evidence: dict) -> tuple[str, list[str]]:
    document = tomllib.loads(text)
    allowed = plan(document, evidence)
    lines = text.splitlines(keepends=True)
    sections = headings(text)
    order = sorted(allowed, key=lambda name: sections[
        heading_for("targets", name, sections)][0], reverse=True)
    for target_name in order:
        start, _ = sections[heading_for("targets", target_name, sections)]
        lines.insert(start + 1, "tool_calling = false\n")
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
                if after[name].get(key) != value:
                    raise Refused(f"REFUSED_VALUE_CHANGED {kind}.{name}.{key}")
            extra = set(after[name]) - set(before[name])
            expected = {"tool_calling"} if (kind == "targets" and name in allowed) else set()
            if extra != expected:
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
        out, declared = apply(args.baseline.read_text(), evidence)
        args.out.write_text(out)
    except (OSError, ValueError, json.JSONDecodeError, tomllib.TOMLDecodeError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 3
    print(f"DECLARED {len(declared)} target(s) tool_calling=false on provider evidence: "
          + ", ".join(declared))
    return 0


if __name__ == "__main__":
    sys.exit(main())
