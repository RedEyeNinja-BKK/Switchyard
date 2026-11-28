#!/usr/bin/env python3
"""Batch C — Stephen's own Google free lane, and the unified ST_Free pool.

Batches A and B are already in the live manifest, so this batch asserts their
post-conditions and adds only the pieces that need a qualified credential:

  1. two `[llm_clients.*]` blocks on `SWITCHYARD_GEMINI_STEPHEN_API_KEY`, with their own
     `quota_group` — the account, not the provider, is the unit of free-tier accounting;
  2. the ten `stephen-google-*` targets, taken from the rendered fragment rather than
     hand-typed, so a quirk corrected in the catalogue reaches both profiles;
  3. `[routes.ST_Gemini_Free]`, the same Gemini free suite under Stephen's account;
  4. `groups = ["ST_Gemini_Free"]` on the ONE member that has a serving receipt, and
     `ST_Gemini_Free` added to ST_Free's pool — so ST_Free becomes one free pool across
     two providers and two credentials.

What is deliberately NOT done:

  * Only a member with a receipt joins the group. A staged member is a target with no
    group and no route, which is inert: putting nine unmeasured models in a pool spends
    the first request after every deploy discovering that they do not answer.
  * `gemma-4-26b-a4b-it` answered 503 UNAVAILABLE (provider-side demand) on 2026-11-22.
    That is a transient condition, not a credential verdict, so it is staged rather than
    marked `held` — `held` would claim the model is unavailable, which was not measured.
  * No candidate list on ST_Gemini_Free: peer selection derives the pool from
    `pool_groups`, and an explicit empty array would be a second, empty membership list.

Usage:
    apply_gemini_free.py --baseline routes.toml --out candidate.toml [--diff]

The baseline is read only, and the tool refuses to overwrite its input.
"""

from __future__ import annotations

import argparse
import difflib
import sys
import tomllib
from pathlib import Path

import apply_batch_a

RENDERED = Path(__file__).resolve().parent / "gemini-suite" / "rendered"

GEMINI_GROUP = "ST_Gemini_Free"
QUALIFIED = "stephen-google-gemini-3.5-flash-lite"

NEW_CLIENTS = ["google_gemini_stephen", "google_gemma_31b_stephen"]
ST_FREE = "routes.ST_Free"


def insert_at_table_end(lines: list[str], header: str, text: str) -> None:
    """Insert `text` after the last key line of `header`, before any trailing banner.

    A table's span runs to the next header, so appending at `end` would place text after
    the comment banner that introduces the following table. Land on the last non-blank,
    non-comment line instead.
    """
    start, end = apply_batch_a.section_span(lines, header)
    body = [
        i
        for i in range(start + 1, end)
        if lines[i].strip() and not lines[i].lstrip().startswith("#")
    ]
    if not body:
        raise apply_batch_a.EditError(f"[{header}] has no keys")
    lines[body[-1] + 1 : body[-1] + 1] = text.splitlines(keepends=True)


def insert_before_header(lines: list[str], header: str, text: str) -> None:
    start, _ = apply_batch_a.section_span(lines, header)
    lines[start:start] = text.splitlines(keepends=True)


def _fragment(name: str) -> str:
    path = RENDERED / name
    if not path.is_file():
        raise apply_batch_a.EditError(
            f"missing {path}; run gemini-suite/project.py first"
        )
    return path.read_text()


def _targets_fragment() -> str:
    """The rendered targets, plus a group label on the single qualified member only."""
    text = _fragment("stephen-targets.toml")
    marker = f'[targets."{QUALIFIED}"]'
    if text.count(marker) != 1:
        raise apply_batch_a.EditError(f"rendered targets: {marker} matched {text.count(marker)} times")
    start = text.index(marker)
    end = text.find("\n[targets.", start + 1)
    block = text[start : end if end != -1 else len(text)]
    if f'groups = ["{GEMINI_GROUP}"]' in block:
        raise apply_batch_a.EditError("qualified target already carries the group label")
    injected = block.rstrip("\n") + f'\ngroups = ["{GEMINI_GROUP}"]\n'
    return text[:start] + injected + (text[end:] if end != -1 else "")


def _route_fragment() -> str:
    """The rendered ST_Gemini_Free route, minus its (empty) candidate list."""
    text = _fragment("stephen-routes.toml")
    if text.count("candidates = []") != 1:
        raise apply_batch_a.EditError("rendered route: expected exactly one empty candidate list")
    return text.replace("candidates = []\n", "")


def require_batches_ab(parsed: dict) -> None:
    """Batches A and B are already in the live manifest; assert that, do not re-apply them.

    Re-applying them would either fail on their own anchors or, worse, silently duplicate
    a block. Their post-conditions are what this batch depends on, so check those.
    """
    route = parsed["routes"].get("ST_Free")
    if route is None:
        raise apply_batch_a.EditError("baseline has no [routes.ST_Free]")
    if route.get("selection") != "peer":
        raise apply_batch_a.EditError("baseline ST_Free is not a peer pool")
    if route.get("pool_groups") != ["ST_OR_Free"]:
        raise apply_batch_a.EditError(f"baseline ST_Free pool_groups is {route.get('pool_groups')}")
    unlabelled = [
        name
        for name in apply_batch_a.FREE_LEAVES
        if parsed["targets"].get(name, {}).get("groups") != ["ST_OR_Free"]
    ]
    if unlabelled:
        raise apply_batch_a.EditError(f"baseline leaves lack the free group label: {unlabelled}")
    for name in NEW_CLIENTS:
        if name in parsed["llm_clients"]:
            raise apply_batch_a.EditError(f"[{name}] is already present; nothing to add")
    if GEMINI_GROUP in parsed["routes"]:
        raise apply_batch_a.EditError(f"[{GEMINI_GROUP}] is already present; nothing to add")


def apply(text: str) -> str:
    require_batches_ab(tomllib.loads(text))
    lines = text.splitlines(keepends=True)

    # 1. The two clients, beside the existing Stephen client.
    insert_at_table_end(lines, "llm_clients.openrouter_stephen", _fragment("stephen-clients.toml"))

    # 2/3. Targets and the route, immediately before the [quota] table that closes the file.
    insert_before_header(lines, "quota", _targets_fragment() + "\n" + _route_fragment() + "\n" + _fragment("stephen-leaves.toml") + "\n")

    # 4. ST_Free becomes the unified pool.
    start, end = apply_batch_a.section_span(lines, ST_FREE)
    first, last = apply_batch_a.key_span(lines, start, end, "pool_groups")
    if first != last:
        raise apply_batch_a.EditError("ST_Free pool_groups is not a scalar")
    if 'ST_OR_Free' not in lines[first]:
        raise apply_batch_a.EditError(f"ST_Free pool_groups is {lines[first].strip()!r}")
    if GEMINI_GROUP in lines[first]:
        raise apply_batch_a.EditError("ST_Free already pools the Gemini group")
    lines[first] = f'pool_groups = ["ST_OR_Free", "{GEMINI_GROUP}"]\n'

    # Append the qualified member to the candidate list without rewriting the others.
    start, end = apply_batch_a.section_span(lines, ST_FREE)
    first, last = apply_batch_a.key_span(lines, start, end, "candidates")
    if not lines[last].strip().startswith("]"):
        raise apply_batch_a.EditError("ST_Free candidates is not a bracketed list")
    for index in range(first, last):
        if f'"{QUALIFIED}"' in lines[index]:
            raise apply_batch_a.EditError(f"{QUALIFIED} is already a candidate")
    indent = "  "
    for index in range(first + 1, last):
        if lines[index].strip():
            indent = lines[index][: len(lines[index]) - len(lines[index].lstrip())]
            break
    lines.insert(last, f'{indent}"{QUALIFIED}",\n')

    out = "".join(lines)
    _assert_scope(text, out)
    return out


def _assert_scope(before: str, after: str) -> None:
    old = tomllib.loads(before)
    new = tomllib.loads(after)

    added_clients = set(new["llm_clients"]) - set(old["llm_clients"])
    if added_clients != set(NEW_CLIENTS):
        raise apply_batch_a.EditError(f"client delta is {sorted(added_clients)}")
    for name in NEW_CLIENTS:
        block = new["llm_clients"][name]
        if block.get("api_key_env") != "SWITCHYARD_GEMINI_STEPHEN_API_KEY":
            raise apply_batch_a.EditError(f"client {name} is not bound to Stephen's key")
        if block.get("quota_group") != "google-ai-studio-free-stephen":
            raise apply_batch_a.EditError(f"client {name} shares another account's quota group")

    added_targets = set(new["targets"]) - set(old["targets"])
    if not added_targets or any(not n.startswith("stephen-google-") for n in added_targets):
        raise apply_batch_a.EditError(f"target delta is {sorted(added_targets)}")
    grouped = {
        n for n in added_targets if new["targets"][n].get("groups") == [GEMINI_GROUP]
    }
    if grouped != {QUALIFIED}:
        raise apply_batch_a.EditError(f"pool members are {sorted(grouped)}, expected only {QUALIFIED}")

    added_routes = set(new["routes"]) - set(old["routes"])
    expected_direct = {
        f"direct-google-stephen-{key}" for key in (
            "gemini-3-flash-preview",
            "gemini-3.1-flash-lite",
            "gemini-3.5-flash-lite",
            "gemini-3.6-flash",
            "gemini-3.8-flash",
            "gemini-4-hydrogen",
            "gemma-4-26b-a4b-it",
            "gemma-4-31b-it",
        )
    }
    if added_routes != {GEMINI_GROUP} | expected_direct:
        raise apply_batch_a.EditError(f"route delta is {sorted(added_routes)}")
    for direct in expected_direct:
        block = new["routes"][direct]
        if block.get("type") != "passthrough" or not block.get("target", "").startswith("stephen-google-"):
            raise apply_batch_a.EditError(f"{direct}: bad Stephen direct target binding")
    if any("gemini-2.5-" in route for route in added_routes):
        raise apply_batch_a.EditError("retired Gemini 2.5 must not be generated")

    for name in set(old["targets"]):
        if old["targets"][name] != new["targets"][name]:
            raise apply_batch_a.EditError(f"existing target {name} changed")

    touched = {n for n in old["routes"] if old["routes"][n] != new["routes"][n]}
    if touched != {"ST_Free"}:
        raise apply_batch_a.EditError(f"changed routes are {sorted(touched)}")
    delta = {
        key
        for key in set(old["routes"]["ST_Free"]) | set(new["routes"]["ST_Free"])
        if old["routes"]["ST_Free"].get(key) != new["routes"]["ST_Free"].get(key)
    }
    if delta != {"pool_groups", "candidates"}:
        raise apply_batch_a.EditError(f"ST_Free changed keys {sorted(delta)}")
    free = new["routes"]["ST_Free"]
    if free["pool_groups"] != ["ST_OR_Free", GEMINI_GROUP]:
        raise apply_batch_a.EditError(f"ST_Free pool_groups is {free['pool_groups']}")
    if free["candidates"] != list(old["routes"]["ST_Free"]["candidates"]) + [QUALIFIED]:
        raise apply_batch_a.EditError(f"ST_Free candidates are {free['candidates']}")
    if free["selection"] != "peer" or free.get("escalation_target") != old["routes"]["ST_Free"].get("escalation_target"):
        raise apply_batch_a.EditError("ST_Free lost its selection mode or its escalation leg")

    for key in set(old) | set(new):
        if key in {"llm_clients", "targets", "routes"}:
            continue
        if old.get(key) != new.get(key):
            raise apply_batch_a.EditError(f"top-level section {key!r} changed")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--diff", action="store_true")
    args = parser.parse_args(argv)

    if args.baseline == args.out:
        print("REFUSED: refusing to overwrite the baseline", file=sys.stderr)
        return 2

    original = Path(args.baseline).read_text()
    try:
        out = apply(original)
    except apply_batch_a.EditError as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 2

    Path(args.out).write_text(out)
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
