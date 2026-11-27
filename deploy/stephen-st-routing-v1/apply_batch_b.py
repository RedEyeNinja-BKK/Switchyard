#!/usr/bin/env python3
"""Batch B — one provider-neutral ST_Free pool, declared by group membership.

Batch A is applied first (it is imported, not copied), then Batch B adds only what a
peer pool needs. The production manifest is not in this repository, so this ships as a
deterministic edit against exact bytes with every anchor asserted to match once.

Edits, all inside the Stephen scope:

  1. `groups = ["ST_OR_Free"]` on the seven free Stephen chat leaves, so the pool's
     membership is a label on the target rather than a second hand-kept list;
  2. `groups = ["ST_OR_Paid"]` on the two metered leaves, for the same reason. The label
     is catalogue metadata: `ST_Paid` is deliberately NOT a pool route, because a pool
     admits only positively-free candidates;
  3. `selection = "peer"` and `pool_groups = ["ST_OR_Free"]` on ST_Free.

`ST_Gemini_Free` is intentionally absent from `pool_groups`. Stephen's Google key is
staged and loaded but has no qualified chat target yet, and an empty group would be a
phantom pool member. Adding it later is a two-line manifest change (label the qualified
Gemini targets, add the name to the list) with no code change; the selection machinery is
already proven against a second provider by tests/peer_pool.rs.

Nothing outside `stephen-*`, `direct-openrouter-stephen-*` and `ST_Free` is touched:
owner `smart*`, `smartfree`, `VC_Paid`, the Google lanes, `ST_Paid`, `ST_Route`,
`ST_Embed`, `ST_Rerank` and `ST_Decisions` are byte-identical in the output.

Usage:
    apply_batch_b.py --baseline routes.toml --out candidate.toml
    apply_batch_b.py --baseline routes.toml --out candidate.toml --diff

The baseline is read only. The output is written to a new path; the tool refuses to
overwrite its input.
"""

from __future__ import annotations

import argparse
import difflib
import sys
import tomllib

import apply_batch_a

FREE_GROUP = "ST_OR_Free"
PAID_GROUP = "ST_OR_Paid"

# The two metered Stephen leaves. Named explicitly so the applier cannot silently pick a
# different set if the manifest changes underneath it.
PAID_LEAVES = [
    "stephen-paid-deepseek-v4-1-flash-1017",
    "stephen-paid-glm-5-3-flash",
]

ST_FREE = "routes.ST_Free"


def section_bounds(lines: list[str], header: str) -> tuple[int, int]:
    """Offsets of the table body, exclusive of the header line and the next header."""
    return apply_batch_a.section_span(lines, header)


def insert_after_line(lines: list[str], header: str, needle: str, text: str) -> None:
    """Insert `text` immediately after the single line equal to `needle` in `header`.

    Scoped to one table so a key that appears in every target cannot be matched in the
    wrong one.
    """
    start, end = section_bounds(lines, header)
    hits = [
        index
        for index in range(start + 1, end)
        if lines[index].strip() == needle.strip()
    ]
    if len(hits) != 1:
        raise apply_batch_a.EditError(
            f"anchor {needle!r} matched {len(hits)} times in [{header}]"
        )
    lines.insert(hits[0] + 1, text)


def apply(text: str) -> str:
    """Apply Batch A, then Batch B, to `text`."""
    staged = apply_batch_a.apply(text)
    lines = staged.splitlines(keepends=True)

    # 1/2. Membership labels. Every free leaf must already declare itself free — Batch A
    # is what establishes that — so this asserts the dependency rather than assuming it.
    for name in apply_batch_a.FREE_LEAVES:
        header = f"targets.{name}"
        insert_after_line(lines, header, "free_tier = true", f'groups = ["{FREE_GROUP}"]\n')
    for name in PAID_LEAVES:
        header = f"targets.{name}"
        # A metered leaf must NOT declare free_tier. Assert the shape this relies on.
        start, end = section_bounds(lines, header)
        body = "".join(lines[start + 1 : end])
        if "free_tier" in body:
            raise apply_batch_a.EditError(
                f"[{header}] declares a tariff but is listed as a metered leaf"
            )
        insert_after_line(
            lines, header, 'llm_client = "openrouter_stephen"', f'groups = ["{PAID_GROUP}"]\n'
        )

    # 3. The route becomes a peer pool over the free group.
    insert_after_line(lines, ST_FREE, 'type = "passthrough"', 'selection = "peer"\n')
    insert_after_line(lines, ST_FREE, 'selection = "peer"', f'pool_groups = ["{FREE_GROUP}"]\n')

    out = "".join(lines)
    _assert_scope(staged, out)
    return out


def _assert_scope(before: str, after: str) -> None:
    """Refuse if the edit changed anything outside the declared Stephen delta.

    Compares parsed objects, so a change in formatting or key order is invisible and a
    change in meaning is not.
    """
    old = tomllib.loads(before)
    new = tomllib.loads(after)

    before_targets = set(old["targets"])
    after_targets = set(new["targets"])
    if before_targets != after_targets:
        raise apply_batch_a.EditError(
            f"target set changed: added={sorted(after_targets - before_targets)} "
            f"removed={sorted(before_targets - after_targets)}"
        )
    before_routes = set(old["routes"])
    after_routes = set(new["routes"])
    if before_routes != after_routes:
        raise apply_batch_a.EditError(
            f"route set changed: added={sorted(after_routes - before_routes)} "
            f"removed={sorted(before_routes - after_routes)}"
        )

    # Targets: only the nine Stephen leaves may differ, and only by gaining a group label.
    allowed_targets = set(apply_batch_a.FREE_LEAVES) | set(PAID_LEAVES)
    touched: set[str] = set()
    for name in before_targets:
        if old["targets"][name] != new["targets"][name]:
            touched.add(name)
            if name not in allowed_targets:
                raise apply_batch_a.EditError(f"target {name} changed but is out of scope")
            delta = {
                key
                for key in set(old["targets"][name]) | set(new["targets"][name])
                if old["targets"][name].get(key) != new["targets"][name].get(key)
            }
            if delta - {"groups"}:
                raise apply_batch_a.EditError(
                    f"target {name} changed keys {sorted(delta)}; only `groups` may change"
                )
    if touched != allowed_targets:
        raise apply_batch_a.EditError(
            f"changed target set is {sorted(touched)}, expected {sorted(allowed_targets)}"
        )

    # Routes: ST_Free only.
    touched_routes = {
        name
        for name in before_routes
        if old["routes"][name] != new["routes"][name]
    }
    if touched_routes != {"ST_Free"}:
        raise apply_batch_a.EditError(
            f"changed route set is {sorted(touched_routes)}, expected ['ST_Free']"
        )
    delta = {
        key
        for key in set(old["routes"]["ST_Free"]) | set(new["routes"]["ST_Free"])
        if old["routes"]["ST_Free"].get(key) != new["routes"]["ST_Free"].get(key)
    }
    if delta != {"selection", "pool_groups"}:
        raise apply_batch_a.EditError(
            f"ST_Free changed keys {sorted(delta)}; only selection/pool_groups may change"
        )
    if new["routes"]["ST_Free"]["selection"] != "peer":
        raise apply_batch_a.EditError("ST_Free selection is not `peer`")
    if new["routes"]["ST_Free"]["pool_groups"] != [FREE_GROUP]:
        raise apply_batch_a.EditError("ST_Free pool_groups is not exactly [ST_OR_Free]")

    # Everything else in the document must be identical.
    for key in set(old) | set(new):
        if key in {"targets", "routes"}:
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

    with open(args.baseline) as handle:
        original = handle.read()
    try:
        out = apply(original)
    except apply_batch_a.EditError as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 2

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
