#!/usr/bin/env python3
"""Structural verifier for the Batch B candidate.

Checks the candidate against the live baseline it was derived from. Every check is a
statement about the *candidate*, not about the applier: a defect in `apply_batch_b.py`
that still produced a plausible file would be caught here.

Exit 0 when every check passes, 1 otherwise. Nothing is written.
"""

from __future__ import annotations

import argparse
import sys
import tomllib

FREE_GROUP = "ST_OR_Free"
PAID_GROUP = "ST_OR_Paid"

FREE_LEAVES = [
    "stephen-ernie-5_5-flash-preview-free",
    "stephen-flint-v1-0-free",
    "stephen-dots-3-note-preview-free",
    "stephen-gemma-4-26b-a4b-it-free",
    "stephen-gemma-4-31b-it-free",
    "stephen-ling-3_2-flash-free",
    "stephen-nemotron-3-5-super-vl-free",
]

PAID_LEAVES = [
    "stephen-paid-deepseek-v4-1-flash-1017",
    "stephen-paid-glm-5-3-flash",
]

RESULTS: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> None:
    RESULTS.append((ok, name, detail))


def legs(route: dict) -> list[str]:
    """Every leg a passthrough route can call: `target` first, then `candidates`."""
    out = []
    if route.get("target"):
        out.append(route["target"])
    out.extend(route.get("candidates") or [])
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    args = parser.parse_args()

    with open(args.baseline, "rb") as handle:
        base = tomllib.load(handle)
    with open(args.candidate, "rb") as handle:
        cand = tomllib.load(handle)

    targets, routes = cand["targets"], cand["routes"]

    # B1 — the free group is exactly the seven free leaves, each positively free.
    free_members = sorted(
        name for name, target in targets.items() if FREE_GROUP in (target.get("groups") or [])
    )
    check(
        free_members == sorted(FREE_LEAVES),
        f"B1  {FREE_GROUP} membership is exactly the seven free leaves",
        f"n={len(free_members)}",
    )
    unfree = [
        name
        for name in free_members
        if targets[name].get("free_tier") is not True
    ]
    check(
        not unfree,
        f"B2  every {FREE_GROUP} member declares free_tier = true",
        f"members={len(free_members)} offenders={unfree}",
    )

    # B3 — the paid group is exactly the two metered leaves, none of which claims free.
    paid_members = sorted(
        name for name, target in targets.items() if PAID_GROUP in (target.get("groups") or [])
    )
    check(
        paid_members == sorted(PAID_LEAVES),
        f"B3  {PAID_GROUP} membership is exactly the two metered leaves",
        f"n={len(paid_members)}",
    )
    check(
        all(targets[name].get("free_tier") is not True for name in paid_members),
        f"B4  no {PAID_GROUP} member claims a free tariff",
        f"members={len(paid_members)}",
    )

    # B5/B6 — ST_Free is a peer pool over the free group and nothing else.
    st_free = routes["ST_Free"]
    check(
        st_free.get("selection") == "peer",
        "B5  ST_Free selects with `peer`",
        f"selection={st_free.get('selection')!r}",
    )
    check(
        st_free.get("pool_groups") == [FREE_GROUP],
        "B6  ST_Free pool_groups is exactly [ST_OR_Free]",
        f"pool_groups={st_free.get('pool_groups')!r}",
    )
    check(
        sorted(legs(st_free)) == sorted(free_members),
        "B7  ST_Free legs are exactly the pool members",
        f"legs={len(legs(st_free))} members={len(free_members)}",
    )
    check(
        st_free.get("context_window") == 262144,
        "B8  ST_Free keeps the 262144 policy promise",
        f"context_window={st_free.get('context_window')}",
    )

    # B9 — a group with no members must not be named, or the pool would not load.
    named = st_free.get("pool_groups") or []
    empty = [
        group
        for group in named
        if not any(group in (target.get("groups") or []) for target in targets.values())
    ]
    check(not empty, "B9  no pool group is empty", f"empty={empty}")

    # B10 — ST_Paid must not be a pool: a free pool excludes every metered leg, so making
    # ST_Paid one would leave it with no candidates at all.
    check(
        "pool_groups" not in routes["ST_Paid"],
        "B10 ST_Paid is not a pool route",
        f"pool_groups={routes['ST_Paid'].get('pool_groups')!r}",
    )
    check(
        sorted(legs(routes["ST_Paid"])) == sorted(PAID_LEAVES),
        "B11 ST_Paid legs are exactly the metered leaves",
        f"legs={sorted(legs(routes['ST_Paid']))}",
    )

    # B12 — Gemini membership is truthfully absent: no target claims it yet.
    gemini = sorted(
        name for name, target in targets.items() if "ST_Gemini_Free" in (target.get("groups") or [])
    )
    check(
        gemini == [],
        "B12 no target claims ST_Gemini_Free, so naming it would be a phantom",
        f"members={gemini}",
    )

    # B13 — scope: no owner target gained or lost a group label.
    owner_labelled = sorted(
        name
        for name, target in targets.items()
        if (target.get("groups") or [])
        and name not in set(FREE_LEAVES) | set(PAID_LEAVES)
    )
    check(
        not owner_labelled,
        "B13 only Stephen leaves carry group labels",
        f"unexpected={owner_labelled}",
    )

    # B14 — owner and typed surfaces are byte-identical to the baseline.
    for name in (
        "ST_Route",
        "ST_Paid",
        "ST_Embed",
        "ST_Rerank",
        "ST_Decisions",
        "smartfree",
        "smartlocal",
        "smartlocal-nt",
        "VC_Paid",
    ):
        if name in base["routes"]:
            check(
                base["routes"][name] == routes[name],
                f"B14 route {name} is unchanged",
                f"changed_keys={sorted(k for k in set(base['routes'][name]) | set(routes[name]) if base['routes'][name].get(k) != routes[name].get(k))}",
            )
        if name in base.get("composites", {}):
            check(
                base["composites"][name] == cand["composites"].get(name),
                f"B15 composite {name} is unchanged",
            )

    for section in ("capabilities", "decision_lanes", "decision_lane_composition", "quota"):
        check(
            base.get(section) == cand.get(section),
            f"B16 section [{section}] is unchanged",
        )

    # B17 — owner targets unchanged.
    changed_owner = sorted(
        name
        for name in base["targets"]
        if name not in set(FREE_LEAVES) | set(PAID_LEAVES)
        and base["targets"][name] != targets.get(name)
    )
    check(not changed_owner, "B17 every non-Stephen target is unchanged", f"changed={changed_owner}")

    # B18 — no dangling reference anywhere.
    dangling: list[str] = []
    for rid, route in routes.items():
        for leg in legs(route):
            if leg not in targets:
                dangling.append(f"{rid}->{leg}")
    for name, children in (cand.get("composites") or {}).items():
        for child in children.get("children") or []:
            if child not in routes:
                dangling.append(f"composite {name}->{child}")
    check(not dangling, "B18 no dangling target or route reference", f"dangling={dangling}")

    # B19 — no nested composite (the engine rejects it; a pool must not introduce one).
    composite_children = {
        child
        for entry in (cand.get("composites") or {}).values()
        for child in (entry.get("children") or [])
    }
    nested = sorted(name for name in (cand.get("composites") or {}) if name in composite_children)
    check(not nested, "B19 no nested composite", f"nested={nested}")

    # B20 — the pool route's members are all callable by it (a label with no leg, or a leg
    # with no label, would be a pool that disagrees with its own manifest).
    stranded = sorted(
        name for name in free_members if name not in legs(st_free)
    )
    check(not stranded, "B20 no pool member is unreachable from ST_Free", f"stranded={stranded}")

    failed = [entry for entry in RESULTS if not entry[0]]
    for ok, name, detail in RESULTS:
        status = "PASS" if ok else "FAIL"
        print(f"{status}  {name:72s} {detail}")
    print()
    print(f"{len(RESULTS)} checks, {len(failed)} failed")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
