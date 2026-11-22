#!/usr/bin/env python3
"""Batch A — Stephen ST direct-leaf truth and removal of the dynamic free selector.

The production route manifest is not stored in this repository (it embeds internal
topology), so Batch A ships as a deterministic edit applied to the exact deployed
bytes. Every edit is anchored: a missing or doubled anchor aborts the run instead of
producing a partially edited manifest.

Edits, all inside the Stephen scope:

  1. delete the `stephen-openrouter-free` target (OpenRouter's dynamic Free Models
     Router) and the `direct-openrouter-stephen-openrouter-free` leaf that named it;
  2. widen the three mis-declared direct leaves to their provider-published context
     windows (ERNIE 1,024,000; Ling 1,048,576; Nemotron 1,000,000). The previous
     262,144 was the route policy promise, not an upstream limit;
  3. replace ST_Free's dynamic third candidate with the four explicit free leaves so
     no ordered leg depends on a model chosen at request time by the provider;
  4. declare `free_tier = true` on the seven explicit free leaves, so the selection
     tier recorded for a Stephen free leg is a declared fact rather than a guess from
     a slug.

Nothing outside `stephen-*`, `direct-openrouter-stephen-*` and `ST_Free` is touched:
owner `smart*`, `smartfree`, `VC_Paid`, the Google lanes, `ST_Paid`, `ST_Route`,
`ST_Embed`, `ST_Rerank` and `ST_Decisions` are byte-identical in the output.

Usage:
    apply_batch_a.py --baseline routes.toml --out candidate.toml
    apply_batch_a.py --baseline routes.toml --out candidate.toml --diff

The baseline is read only. The output is written to a new path; the tool refuses to
overwrite its input.
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
import tomllib

SECTION = re.compile(r"^\[([^\]]+)\]\s*$")

FREE_LEAVES = [
    "stephen-ernie-5_5-flash-preview-free",
    "stephen-flint-v1-0-free",
    "stephen-dots-3-note-preview-free",
    "stephen-gemma-4-26b-a4b-it-free",
    "stephen-gemma-4-31b-it-free",
    "stephen-ling-3_2-flash-free",
    "stephen-nemotron-3-5-super-vl-free",
]

ST_FREE_ORDER = [
    "stephen-ernie-5_5-flash-preview-free",
    "stephen-ling-3_2-flash-free",
    "stephen-nemotron-3-5-super-vl-free",
    "stephen-flint-v1-0-free",
    "stephen-dots-3-note-preview-free",
    "stephen-gemma-4-26b-a4b-it-free",
    "stephen-gemma-4-31b-it-free",
]

# Provider-published context windows, captured from the OpenRouter model catalogue
# on 2026-11-22 (see or-model-facts-20261122.json). Only the three leaves whose
# declared promise disagreed with the provider are changed.
CONTEXT_TRUTH = {
    "direct-openrouter-stephen-ernie-5_5-flash-preview-free": 1024000,
    "direct-openrouter-stephen-ling-3_2-flash-free": 1048576,
    "direct-openrouter-stephen-nemotron-3-5-super-vl-free": 1000000,
}

DYNAMIC_TARGET = "targets.stephen-openrouter-free"
DYNAMIC_ROUTE = "routes.direct-openrouter-stephen-openrouter-free"

# The ST_Free comment block states a reason that stops being true once the dynamic
# router is removed. Replaced verbatim; the applier asserts it matched once.
ST_FREE_COMMENT_OLD = """# The three named legs are declared at 262144, so that is what this route promises.
# The R4 Free Models Router declares up to 200k; it is the last leg and a bonus
# rather than the guarantee, which is why the route number follows the primary legs
# and the escalation threshold below follows the router's ceiling.
"""
ST_FREE_COMMENT_NEW = """# 262144 is this route's promise, not a limit any leg imposes: the leaf routes each
# carry their provider-published window (ERNIE 1,024,000; Ling 1,048,576; Nemotron
# 1,000,000). Raising this number is a separate, qualified decision about what ST_Free
# callers may be admitted with.
"""
ESCALATION_COMMENT_OLD = """# The measured smartfree learning carries over: Ling took ~120 s on a 430k-token
# request while Nemotron handled it in ~50 s, so above the router's ceiling the
# wider leg is offered first instead of the narrow one being tried and failing.
"""
ESCALATION_COMMENT_NEW = """# The measured learning carries over: Ling took ~120 s on a 430k-token request while
# Nemotron handled it in ~50 s, so above 200k tokens the proven wide-context leg is
# offered first instead of the narrow one being tried and failing.
"""


class EditError(RuntimeError):
    """An anchored edit did not match exactly once."""


def sections(lines: list[str]) -> list[tuple[str, int, int]]:
    """Return (header, first_line, end_line) for every top-level table.

    `end_line` is exclusive. Text before the first header is not part of any section.
    """
    bounds = [
        (m.group(1), idx)
        for idx, line in enumerate(lines)
        if (m := SECTION.match(line))
    ]
    out = []
    for pos, (header, start) in enumerate(bounds):
        end = bounds[pos + 1][1] if pos + 1 < len(bounds) else len(lines)
        out.append((header, start, end))
    return out


def section_span(lines: list[str], header: str) -> tuple[int, int]:
    matches = [(s, e) for h, s, e in sections(lines) if h == header]
    if len(matches) != 1:
        raise EditError(f"section [{header}] matched {len(matches)} times, expected 1")
    return matches[0]


def key_span(lines: list[str], start: int, end: int, key: str) -> tuple[int, int]:
    """Return the inclusive line span of `key = ...`, following a multi-line array."""
    prefix = f"{key} ="
    hits = [i for i in range(start + 1, end) if lines[i].startswith(prefix)]
    if len(hits) != 1:
        raise EditError(f"{key!r} matched {len(hits)} times in lines {start}-{end}")
    first = hits[0]
    depth = 0
    last = first
    for i in range(first, end):
        depth += lines[i].count("[") - lines[i].count("]")
        last = i
        if depth == 0 and i >= first:
            break
    return first, last


def delete_section(lines: list[str], header: str) -> None:
    """Delete a table and its keys, keeping any comment block that follows it.

    A section runs from its header to the next header, so trailing comments belong to
    the table textually even when they are the banner for the next table. Deleting the
    whole span would silently drop that banner; only the header and the key/value lines
    are removed here.
    """
    start, end = section_span(lines, header)
    body = [
        i
        for i in range(start, end)
        if lines[i].strip() and not lines[i].lstrip().startswith("#")
    ]
    if not body:
        raise EditError(f"section [{header}] has no keys")
    del lines[start : body[-1] + 1]
    # The blank line that separated the table from the next banner is now adjacent to
    # the blank line the deleted table's last key left behind.
    if (
        0 < start < len(lines)
        and not lines[start - 1].strip()
        and not lines[start].strip()
    ):
        del lines[start]


def set_scalar(lines: list[str], header: str, key: str, value: str) -> None:
    start, end = section_span(lines, header)
    first, last = key_span(lines, start, end, key)
    if first != last:
        raise EditError(f"{key!r} in [{header}] is not a scalar")
    lines[first] = f"{key} = {value}\n"


def set_list(lines: list[str], header: str, key: str, values: list[str]) -> None:
    start, end = section_span(lines, header)
    first, last = key_span(lines, start, end, key)
    replacement = [f'{key} = [\n'] + [f'  "{v}",\n' for v in values] + [']\n']
    lines[first : last + 1] = replacement


def insert_key(lines: list[str], header: str, key: str, value: str, after: str) -> None:
    start, end = section_span(lines, header)
    first, last = key_span(lines, start, end, after)
    if any(lines[i].startswith(f"{key} =") for i in range(start + 1, end)):
        raise EditError(f"{key!r} already present in [{header}]")
    lines.insert(last + 1, f"{key} = {value}\n")


def replace_once(lines: list[str], old: str, new: str) -> None:
    text = "".join(lines)
    count = text.count(old)
    if count != 1:
        raise EditError(f"comment block matched {count} times, expected 1")
    lines[:] = text.replace(old, new, 1).splitlines(keepends=True)


def apply(text: str) -> str:
    lines = text.splitlines(keepends=True)
    before = tomllib.loads(text)

    delete_section(lines, DYNAMIC_TARGET)
    delete_section(lines, DYNAMIC_ROUTE)
    replace_once(lines, ST_FREE_COMMENT_OLD, ST_FREE_COMMENT_NEW)
    replace_once(lines, ESCALATION_COMMENT_OLD, ESCALATION_COMMENT_NEW)
    set_list(lines, "routes.ST_Free", "candidates", ST_FREE_ORDER[1:])
    for name, window in CONTEXT_TRUTH.items():
        set_scalar(lines, f"routes.{name}", "context_window", str(window))
    for leaf in FREE_LEAVES:
        insert_key(
            lines,
            f"targets.{leaf}",
            "free_tier",
            "true",
            after="llm_client",
        )

    out = "".join(lines)
    after = tomllib.loads(out)

    # Fail closed on scope: exactly the intended objects changed.
    _assert_scope(before, after)
    return out


def _assert_scope(before: dict, after: dict) -> None:
    removed_targets = set(before["targets"]) - set(after["targets"])
    removed_routes = set(before["routes"]) - set(after["routes"])
    if removed_targets != {"stephen-openrouter-free"}:
        raise EditError(f"unexpected targets removed: {sorted(removed_targets)}")
    if removed_routes != {"direct-openrouter-stephen-openrouter-free"}:
        raise EditError(f"unexpected routes removed: {sorted(removed_routes)}")
    if set(after["targets"]) - set(before["targets"]):
        raise EditError("the edit must not add targets")
    if set(after["routes"]) - set(before["routes"]):
        raise EditError("the edit must not add routes")

    for key in ("composites", "capabilities", "decision_lanes", "quota"):
        if before[key] != after[key]:
            raise EditError(f"[{key}] changed; Batch A must not touch it")

    for name, cfg in before["targets"].items():
        if name in {"stephen-openrouter-free"}:
            continue
        if name in FREE_LEAVES:
            want = dict(cfg, free_tier=True)
        else:
            want = cfg
        if after["targets"][name] != want:
            raise EditError(f"target {name} changed outside the declared delta")

    for name, cfg in before["routes"].items():
        if name in {"direct-openrouter-stephen-openrouter-free", "ST_Free"}:
            continue
        want = dict(cfg)
        if name in CONTEXT_TRUTH:
            want["context_window"] = CONTEXT_TRUTH[name]
        if after["routes"][name] != want:
            raise EditError(f"route {name} changed outside the declared delta")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", required=True, help="deployed manifest to read")
    ap.add_argument("--out", required=True, help="candidate path to write")
    ap.add_argument("--diff", action="store_true", help="print a unified diff")
    args = ap.parse_args(argv)

    with open(args.baseline, "rb") as handle:
        text = handle.read().decode()
    try:
        out = apply(text)
    except EditError as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 2

    with open(args.baseline) as handle:
        original = handle.read()
    if out == original:
        print("REFUSED: the edit produced no change", file=sys.stderr)
        return 2

    with open(args.out, "w") as handle:
        handle.write(out)

    if args.diff:
        sys.stdout.writelines(
            difflib.unified_diff(
                original.splitlines(keepends=True),
                out.splitlines(keepends=True),
                fromfile=args.baseline,
                tofile=args.out,
                n=2,
            )
        )
    print(f"wrote {args.out} ({len(out.encode())} bytes)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
