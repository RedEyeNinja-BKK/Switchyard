#!/usr/bin/env python3
"""Materialise target-level tool_calling from RECORDED evidence, or refuse.

Why this exists
---------------
An absent ``target.tool_calling`` is permissive in the runner
(``config.rs``: ``target.tool_calling != Some(false)``), so for a 1:1
passthrough a route declaring ``tool_calling = true`` can only load when its
single target is ``true`` or absent.  Absent therefore carries no information
the route does not already carry -- but it also cannot be *audited*, which is
why the capability gate reports it as TARGET_CAPABILITY_UNQUALIFIED.

The safe direction is one-way.  ``false`` is NOT an option: it would start
refusing tool requests that are served today, which is a downgrade, and it is
also unloadable against a route that claims ``true``.  So the only repair is a
source-backed ``true``.

This tool never decides that a target supports tools.  It reads an evidence
file, and it will only write ``true`` for a target named there with a recorded
basis and receipt.  Anything else is left alone and reported as a blocker.

Evidence bases accepted
-----------------------
LIVE_TOOL_CALL      A real tool call was answered on the direct alias, which is
                    a 1:1 passthrough with no candidates, so the answer is
                    evidence about that physical target and nothing else.
SHARED_MODEL_SPEC   Owner and Stephen Google routes front the SAME upstream
                    model.  A tool call proven on one account is evidence about
                    the model; the other account's serving is separate evidence
                    and is recorded separately.
PRIOR_MEASUREMENT   A dated measurement campaign on the provider endpoint, with
                    artifact receipts, that exercised tools directly.

Refusals carry tokens; nothing is inferred.
"""
from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path

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


def target_heading(name: str) -> list[str]:
    return [f"[targets.{name}]", f'[targets."{name}"]']


def apply(text: str, evidence: dict, manifest: dict) -> tuple[str, list[str], list[str]]:
    sections = headings(text)
    lines = text.splitlines(keepends=True)
    routes = manifest.get("routes") or {}
    targets = manifest.get("targets") or {}

    writes: dict[str, str] = {}
    skipped: list[str] = []
    for route, record in sorted(evidence.items()):
        basis = record.get("basis")
        receipt = record.get("receipt")
        if basis not in BASES or not receipt:
            raise Refused(f"REFUSED_EVIDENCE_INCOMPLETE {route}: basis={basis!r} receipt={receipt!r}")
        if route not in routes:
            raise Refused(f"REFUSED_UNKNOWN_ROUTE {route}")
        target_name = routes[route].get("target")
        target = targets.get(target_name)
        if target is None:
            raise Refused(f"REFUSED_TARGET_MISSING {route} -> {target_name}")
        if "tool_calling" in target:
            if target["tool_calling"] is not True:
                raise Refused(f"REFUSED_WOULD_OVERWRITE {target_name}.tool_calling={target['tool_calling']!r}")
            skipped.append(f"{target_name}: already declared true")
            continue
        if target.get("held") or not target.get("id"):
            raise Refused(f"REFUSED_TARGET_NOT_SERVABLE {target_name}")
        # A route that refuses tools must not gain an explicit target claim from
        # another route's evidence: the route-level refusal is a separate policy.
        if routes[route].get("tool_calling") is not True:
            skipped.append(f"{target_name}: route {route} does not declare tool_calling=true")
            continue
        writes[target_name] = receipt

    # Insert from the end so earlier spans stay valid.
    order = sorted(writes, key=lambda name: sections[_heading(sections, name)][0], reverse=True)
    for target_name in order:
        head = _heading(sections, target_name)
        start, _ = sections[head]
        body = "".join(lines[start + 1:sections[head][1]])
        if _declares(body, "tool_calling"):
            raise Refused(f"REFUSED_KEY_ALREADY_PRESENT {target_name}")
        lines.insert(start + 1, "tool_calling = true\n")

    out = "".join(lines)
    parsed = tomllib.loads(out)
    # Prove the only delta is the added key.
    if set(parsed) != set(manifest):
        raise Refused("REFUSED_TOP_LEVEL_SECTION_CHANGED")
    for kind in ("routes", "targets"):
        if set(parsed.get(kind, {})) != set(manifest.get(kind, {})):
            raise Refused(f"REFUSED_{kind.upper()}_NAMES_CHANGED")
        for name, before in manifest.get(kind, {}).items():
            after = parsed[kind][name]
            if set(before) != set(after) - ({"tool_calling"} if kind == "targets" and name in writes else set()):
                raise Refused(f"REFUSED_KEYSET_CHANGED {kind}.{name}")
            for key, value in before.items():
                if after.get(key) != value:
                    raise Refused(f"REFUSED_VALUE_CHANGED {kind}.{name}.{key}")
    return out, sorted(writes), skipped


def _declares(body: str, key: str) -> bool:
    """Whether the block really assigns this key, not merely mentions it.

    Manifests document measured facts in comments. A substring search reads
    ``# tool_calling = true was called, not inferred`` as an existing
    declaration and refuses a write that is still needed, so only a
    non-comment assignment line counts. The authoritative check remains the
    parsed table above; this one only guards the textual insert.
    """
    for line in body.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        if stripped.partition("=")[0].strip() == key:
            return True
    return False


def _heading(sections: dict, name: str) -> str:
    matches = [head for head in target_heading(name) if head in sections]
    if len(matches) != 1:
        raise Refused(f"REFUSED_AMBIGUOUS_HEADING targets.{name}: {matches}")
    return matches[0]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args(argv)
    if arguments.baseline.resolve() == arguments.out.resolve():
        print("REFUSED: cannot overwrite baseline", file=sys.stderr)
        return 3
    try:
        text = arguments.baseline.read_text()
        manifest = tomllib.loads(text)
        evidence = json.loads(arguments.evidence.read_text()).get("routes", {})
        out, written, skipped = apply(text, evidence, manifest)
        arguments.out.write_text(out)
    except (OSError, ValueError, KeyError, tomllib.TOMLDecodeError) as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 3
    print(f"applied tool_calling = true to {len(written)} target(s)")
    for name in written:
        print(f"  + {name}")
    for note in skipped:
        print(f"  = {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
