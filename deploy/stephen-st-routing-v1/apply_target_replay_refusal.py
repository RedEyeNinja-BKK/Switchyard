#!/usr/bin/env python3
"""Declare that one target's wire cannot accept a replayed tool call.

Why a separate applier: the declaration is a measured wire fact about one provider,
not a routing preference, and a ``false`` written in the wrong place silently removes
a working leg from every conversation that carries tool history.  So this tool writes
only under conditions it can check from its own inputs:

- the evidence names one target, one route and one provider receipt;
- the receipt is a provider refusal of a REPLAYED tool call, with the provider's own
  message text (not our inference);
- the target is genuinely reachable from that route, as a direct target, a candidate,
  or a member of one of the route's ``pool_groups``;
- the target is not already declared, so a second run is a refusal rather than a
  silent rewrite.

The declaration is placed immediately after the target's ``tool_calling`` line, which
keeps every other byte of the manifest untouched. Unrelated routes that happen to use
the same target are unaffected: the rule only fires for requests carrying tool lineage.

Usage:
  python3 apply_target_replay_refusal.py --baseline routes.toml \
      --evidence replay_refusal_evidence.json --out routes2.toml
"""
from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path

BASIS = "PROVIDER_REFUSES_REPLAY"
FIELD = "tool_call_replay"
REFERENCE_FIELDS = ("target", "terminal_fallback", "escalation_target")
# The provider's own words, from the refusal this applier exists to record.
REQUIRED_PHRASE = "thought_signature"


class Refused(ValueError):
    pass


def headings(text: str) -> dict[str, tuple[int, int]]:
    lines = text.splitlines(keepends=True)
    found = [
        (index, line.strip())
        for index, line in enumerate(lines)
        if line.startswith("[") and line.rstrip().endswith("]")
    ]
    sections: dict[str, tuple[int, int]] = {}
    for position, (start, head) in enumerate(found):
        end = found[position + 1][0] if position + 1 < len(found) else len(lines)
        sections[head] = (start, end)
    return sections


def heading_for(name: str, sections: dict[str, tuple[int, int]]) -> str:
    matches = [head for head in (f"[targets.{name}]", f'[targets."{name}"]')
               if head in sections]
    if len(matches) != 1:
        raise Refused(f"REFUSED_AMBIGUOUS_HEADING targets.{name}: {matches}")
    return matches[0]


def route_reaches_target(document: dict, route_name: str, target_name: str) -> bool:
    """Whether a candidate or pool group of this route names the target.

    Groups are labels carried by targets, not their own sections: a route's
    ``pool_groups`` names labels, and membership is whatever targets declare them.
    """
    routes = document.get("routes") or {}
    route = routes.get(route_name)
    if route is None:
        return False
    if target_name in [route.get(field) for field in REFERENCE_FIELDS]:
        return True
    if target_name in (route.get("candidates") or []):
        return True
    pool_groups = route.get("pool_groups") or []
    if not pool_groups:
        return False
    record = (document.get("targets") or {}).get(target_name) or {}
    return any(group in pool_groups for group in (record.get("groups") or []))


def validate(document: dict, evidence: dict) -> tuple[str, str]:
    if evidence.get("basis") != BASIS:
        raise Refused(f"REFUSED_BASIS {evidence.get('basis')!r} is not {BASIS}")

    target_name = evidence.get("target")
    route_name = evidence.get("route")
    if not isinstance(target_name, str) or not target_name:
        raise Refused("REFUSED_EVIDENCE target must name one target")
    if not isinstance(route_name, str) or not route_name:
        raise Refused("REFUSED_EVIDENCE route must name one route")

    receipt = evidence.get("receipt") or {}
    status = receipt.get("status")
    message = str(receipt.get("message") or "")
    if status != 400:
        raise Refused(f"REFUSED_RECEIPT status {status!r} is not the provider's 400")
    if REQUIRED_PHRASE not in message:
        raise Refused(
            f"REFUSED_RECEIPT the provider message does not name {REQUIRED_PHRASE!r}"
        )
    if not evidence.get("replayed_tool_call") is True:
        raise Refused("REFUSED_EVIDENCE the receipt must be of a REPLAYED tool call")

    targets = document.get("targets") or {}
    record = targets.get(target_name)
    if record is None:
        raise Refused(f"REFUSED_UNKNOWN_TARGET {target_name}")
    if FIELD in record:
        raise Refused(f"REFUSED_ALREADY_DECLARED {target_name} already has {FIELD}")
    if record.get("tool_calling") is False:
        raise Refused(
            f"REFUSED_NO_EFFECT {target_name} already refuses every tool request"
        )
    if not route_reaches_target(document, route_name, target_name):
        raise Refused(
            f"REFUSED_UNREACHABLE {target_name} is not reachable from route {route_name}"
        )
    return target_name, route_name


def render(text: str, target_name: str) -> str:
    sections = headings(text)
    head = heading_for(target_name, sections)
    start, end = sections[head]
    lines = text.splitlines(keepends=True)
    anchor = None
    for index in range(start + 1, end):
        if lines[index].strip().startswith("tool_calling"):
            anchor = index
            break
    if anchor is None:
        raise Refused(
            f"REFUSED_NO_ANCHOR {target_name} declares no tool_calling line to sit beside"
        )
    if not lines[anchor].endswith("\n"):
        lines[anchor] += "\n"
    indent = lines[anchor][: len(lines[anchor]) - len(lines[anchor].lstrip())]
    lines.insert(
        anchor + 1,
        f"{indent}{FIELD} = false\n",
    )
    return "".join(lines)


def apply(baseline_text: str, evidence: dict) -> tuple[str, str, str]:
    document = tomllib.loads(baseline_text)
    target_name, route_name = validate(document, evidence)
    candidate = render(baseline_text, target_name)

    after = tomllib.loads(candidate)
    record = (after.get("targets") or {}).get(target_name) or {}
    if record.get(FIELD) is not False:
        raise Refused(f"REFUSED_NOT_WRITTEN {target_name}.{FIELD} did not land as false")
    # Everything except the one declaration must be identical.
    before_without = dict(document)
    after_without = dict(after)
    after_without["targets"] = dict(after["targets"])
    after_without["targets"][target_name] = {
        key: value for key, value in record.items() if key != FIELD
    }
    if before_without != after_without:
        raise Refused("REFUSED_COLLATERAL the edit changed something besides the declaration")
    return candidate, target_name, route_name


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    baseline_text = args.baseline.read_text(encoding="utf-8")
    evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
    try:
        candidate, target_name, route_name = apply(baseline_text, evidence)
    except Refused as refusal:
        print(f"{refusal}", file=sys.stderr)
        return 2
    if not args.dry_run:
        args.out.write_text(candidate, encoding="utf-8")
    print(f"declared {target_name}.{FIELD} = false (route {route_name})")
    print(f"bytes {len(baseline_text)} -> {len(candidate)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
