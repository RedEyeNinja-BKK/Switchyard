#!/usr/bin/env python3
"""P0, OFFLINE-ONLY: remove Gemini 2.5 Flash/Flash-Lite owner+Stephen
native Google targets AND their direct aliases from a live manifest copy.

Does not deploy, write credentials or call a provider. The gate REFUSES to drop
a 2.5 target still referenced by a non-direct route. The manifest publisher
must use an independent CURRENT read/CAS/rollout/verify.
"""
from __future__ import annotations

import argparse
import difflib
import re
import sys
import tomllib
from pathlib import Path

REMOVED_IDS = {"gemini-2.5-flash", "gemini-2.5-flash-lite"}
HEADER = re.compile(
    r'^\s*\[(routes|targets)\.(?:"([^"]+)"|([\w-]+))\]\s*(?:#.*)?$'
)
ALL_HEADER = re.compile(r'^\s*\[\[?[^\]]+\]\]?\s*(?:#.*)?$')


def target_id(model: dict) -> str | None:
    v = model.get("id")
    if isinstance(v, str):
        return v
    return None


def nested_contains(value: object, names: set[str]) -> bool:
    if isinstance(value, str):
        return value in names
    if isinstance(value, dict):
        return any(nested_contains(v, names) for v in value.values())
    if isinstance(value, list):
        return any(nested_contains(v, names) for v in value)
    return False


def remove_only(text: str) -> tuple[str, list[str]]:
    old = tomllib.loads(text)
    routes, targets = old["routes"], old["targets"]
    removed_routes = {
        n for n in routes
        if n.startswith(("direct-google-", "direct-google-stephen-"))
        and any(n.endswith("-" + model) for model in REMOVED_IDS)
    }
    removed_targets = {
        n for n, cfg in targets.items()
        if target_id(cfg) in REMOVED_IDS and
        (n.startswith("stephen-google-") or "google" in str(cfg.get("llm_client", "")))
    }
    expected_routes = {
        prefix + suffix
        for prefix in ("direct-google-", "direct-google-stephen-")
        for suffix in REMOVED_IDS
    }
    if removed_routes != expected_routes or len(removed_targets) != 4:
        raise ValueError("expected exactly 4 defined direct native 2.5 routes "
                         "and exactly 4 native owner/Stephen targets; "
                         f"got routes={sorted(removed_routes)}, "
                         f"targets={sorted(removed_targets)}; STOP")
    identifiers = removed_routes | removed_targets
    for kind, entries, removed in (
        ("routes", routes, removed_routes),
        ("targets", targets, removed_targets)
    ):
        for name, row in entries.items():
            if name not in removed and nested_contains(row, identifiers):
                raise ValueError(f"still referenced: {kind}.{name}; refuse")
    # Only remove the exact TOML tables. No global find/replace on 2.5 tokens;
    # historical comments and unrelated image-generation models stay untouched.
    lines = text.splitlines(keepends=True)
    spans = []
    current_name = None
    current_start = None
    for i, line in enumerate(lines):
        match = HEADER.match(line)
        if ALL_HEADER.match(line):
            if current_name is not None:
                spans.append((current_start, i, current_name))
            current_start = i if match else None
            current_name = (match.group(1), match.group(2) or match.group(3)) if match else None
    if current_name is not None:
        spans.append((current_start, len(lines), current_name))
    wanted = {("routes", x) for x in removed_routes} | {
        ("targets", x) for x in removed_targets
    }
    hit = {n for _, _, n in spans if n in wanted}
    if hit != wanted:
        raise ValueError(f"could not find unique native TOML tables: {wanted-hit}")
    discard = set()
    for first, last, name in spans:
        if name in wanted:
            discard.update(range(first, last))
    out = "".join(line for i, line in enumerate(lines) if i not in discard)
    new = tomllib.loads(out)
    assert set(old["routes"]) - set(new["routes"]) == removed_routes
    assert set(old["targets"]) - set(new["targets"]) == removed_targets
    assert all(old["routes"][n] == new["routes"][n] for n in new["routes"])
    assert all(old["targets"][n] == new["targets"][n] for n in new["targets"])
    for k in set(old) | set(new):
        if k not in ("routes", "targets") and old.get(k) != new.get(k):
            raise ValueError(f"other manifest section changed: {k}")
    for name, cfg in new["targets"].items():
        if cfg.get("id") in REMOVED_IDS and name.startswith("stephen-google-"):
            raise ValueError(f"retired target persisted: {name}")
    return out, sorted(f"{kind}.{name}" for kind, name in wanted)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--diff", action="store_true")
    args = parser.parse_args()
    if args.baseline.resolve() == args.out.resolve():
        print("REFUSED: never modify baseline in-place", file=sys.stderr)
        return 3
    try:
        original = args.baseline.read_text()
        out, removed = remove_only(original)
    except (ValueError, OSError, KeyError, tomllib.TOMLDecodeError) as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 3
    args.out.write_text(out)
    if args.diff:
        sys.stdout.writelines(difflib.unified_diff(
            original.splitlines(keepends=True),
            out.splitlines(keepends=True),
            fromfile=str(args.baseline), tofile=str(args.out), n=3,
        ))
    print("OFFLINE candidate only; removed EXACTLY:", ", ".join(removed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
