#!/usr/bin/env python3
"""Minimal, candidate-ONLY normalization of direct capability admission metadata.

Safe derivations only:
- video absent => video=false (Switchyard already refuses video unless true);
- route image/tools absent, target explicitly false => route=false (target
  already refused; move refusal to ingress and make /v1/models honest);
- route image/tools absent, target explicitly true => route=true (the route
  previously admitted that medium; advertise the already-qualified target);
- route carrier absent, target explicitly narrow => narrow route to carrier,
  where route medium is already true (avoid advertising unreachable carrier).

NEVER upgrades false to true; NEVER fills a target's unknown tool/image truth;
NEVER treats Google vendor documentation as a serving wire receipt. The
all-provider direct_capability_gate remains the mandatory follow-on.
"""
from __future__ import annotations

import argparse
import difflib
from pathlib import Path
import sys
import tomllib

import direct_capability_gate as gate


class Refused(ValueError):
    pass


def blocks(text: str):
    lines = text.splitlines(keepends=True)
    headings = []
    for i, line in enumerate(lines):
        if line.startswith("[") and line.rstrip().endswith("]"):
            headings.append((i, line.strip()))
    sections = {}
    for j, (start, heading) in enumerate(headings):
        end = headings[j + 1][0] if j+1 < len(headings) else len(lines)
        sections[heading] = (start, end)
    return lines, sections


def insert_fields(text: str, changes: dict[str, dict]) -> str:
    lines, sections = blocks(text)
    for heading in changes:
        if heading not in sections:
            raise Refused(f"section {heading} is absent")
    # Insert from the end so all earlier spans remain valid.
    for heading in sorted(changes, key=lambda h: sections[h][0], reverse=True):
        start, end = sections[heading]
        body = lines[start+1:end]
        expected = changes[heading]
        existing = set()
        for line in body:
            key = line.partition("=")[0].strip()
            existing.add(key)
        if existing & set(expected):
            raise Refused(f"attempted rewrite instead of additive correction: {heading}")
        position = start + 1
        # TOML: append immediately below current block title, before sub-tables.
        for field, value in expected.items():
            if type(value) is bool:
                rendered = str(value).lower()
            elif isinstance(value, str) and value in gate.CARRIERS:
                rendered = f'"{value}"'
            else:
                raise Refused(f"unsupported new value for {heading}.{field}")
            lines.insert(position, f"{field} = {rendered}\n")
            position += 1
    out = "".join(lines)
    tomllib.loads(out)  # syntax gate; the source is not overwritten.
    return out


def plan(document: dict) -> dict[str, dict]:
    edits = {}
    for name in gate._direct_names(document):
        route, target_name, target = gate._one_target(document, name)
        rhead = f'[routes."{name}"]' if "." in name or "-" in name else f"[routes.{name}]"
        thead = (f'[targets."{target_name}"]'
                 if "." in target_name or "-" in target_name else f"[targets.{target_name}]")
        # Some manifests use unquoted table names with '-' legal in TOML. Actual
        # heading spelling is discovered from text during application; this
        # function emits logical table keys, not guessed headings.
        redit, tedit = {}, {}
        if "video" not in target:
            tedit["video"] = False
        if "video" not in route:
            redit["video"] = False
        for medium in ("vision", "tool_calling"):
            if medium not in route and type(target.get(medium)) is bool:
                redit[medium] = target[medium]
        for medium, carrier in (("vision", "image_carrier"), ("video", "video_carrier")):
            if route.get(medium) is True and carrier not in route:
                if target.get(carrier) in ("inline", "remote"):
                    redit[carrier] = target[carrier]
        if redit:
            edits["routes", name] = redit
        if tedit:
            edits["targets", target_name] = tedit
    return edits


def resolve_heads(text: str, edits: dict[tuple[str, str], dict]) -> dict[str, dict]:
    _, section = blocks(text)
    resolved = {}
    for (kind, name), changes in edits.items():
        possible = (f"[{kind}.{name}]", f'[{kind}."{name}"]')
        matched = [head for head in possible if head in section]
        if len(matched) != 1:
            raise Refused(f"{kind}.{name}: expected exactly one TOML heading, found {matched}")
        resolved[matched[0]] = changes
    return resolved


def apply(text: str) -> tuple[str, dict]:
    old = tomllib.loads(text)
    edits = plan(old)
    out = insert_fields(text, resolve_heads(text, edits))
    new = tomllib.loads(out)
    changed = {key for key in set(old) | set(new) if old.get(key) != new.get(key)}
    if changed - {"routes", "targets"}:
        raise Refused(f"non-route section changed: {sorted(changed)}")
    for kind in ("routes", "targets"):
        if set(old.get(kind, {})) != set(new.get(kind, {})):
            raise Refused(f"{kind} names changed")
        for name, before in old.get(kind, {}).items():
            after = new[kind][name]
            allowed = edits.get((kind, name), {})
            if {key: after[key] for key in after if before.get(key) != after[key]} != allowed:
                raise Refused(f"unauthorized metadata mutation {kind}.{name}")
            for key in before:
                if before[key] != after.get(key):
                    raise Refused(f"existing value overwritten {kind}.{name}.{key}")
    return out, edits


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--diff", action="store_true")
    a = p.parse_args(argv)
    if a.baseline.resolve() == a.out.resolve():
        print("REFUSED: cannot overwrite baseline", file=sys.stderr)
        return 3
    try:
        original = a.baseline.read_text()
        out, edits = apply(original)
        a.out.write_text(out)
        report = gate.analyze(tomllib.loads(out))
    except (OSError, ValueError, tomllib.TOMLDecodeError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 3
    if a.diff:
        print("".join(difflib.unified_diff(original.splitlines(keepends=True),
                                           out.splitlines(keepends=True), n=2,
                                           fromfile=str(a.baseline),
                                           tofile=str(a.out))))
    print(f"CANDIDATE_ONLY: inserted {sum(map(len, edits.values()))} safe declarations; "
          f"unresolved={len(report['failures'])}; verifier exit remains blocking until "
          "existing qualification receipts back every remaining field")
    for issue in report["failures"]:
        print("UNRESOLVED", issue)
    return 0 if not report["failures"] else 2


if __name__ == "__main__":
    sys.exit(main())
