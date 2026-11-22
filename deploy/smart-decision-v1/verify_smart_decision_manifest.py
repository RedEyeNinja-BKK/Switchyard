#!/usr/bin/env python3
"""Prove a candidate manifest declares a capability-first, provider-neutral
`smart-decision` without weakening any existing decision lane.

The policy itself lives in source (`LaneSelection::CapabilityFirst`); this guard
proves the deployment that is about to be loaded actually uses it, and that the
pools it reaches satisfy the owner's constraints. It is anchored on the manifest
under test and on the currently deployed manifest, so a candidate that quietly
drops a lane or widens the OpenRouter pool is refused rather than shipped.

Checks:

  R1  `decision_lane_composition.smart-decision` exists and declares
      `selection = "capability_first"`.
  R2  Its children are exactly the named constituents: `smartlocal-decision`,
      `smartfree-decision`, `google-decision`.
  R3  Every child resolves to a leaf lane that owns at least one target, and
      every reachable target declares a decision transport.
  R4  Exactly ONE OpenRouter decision target is reachable from the pool, and it
      is named `direct-openrouter-span-01-lite`.
  R5  No reachable target authenticates with a Stephen credential.
  R6  `tier1-decision` still exists, still selects in declared order, and still
      names its original children.
  R7  The pre-existing lanes (`smartlocal-decision`, `smartfree-decision`,
      `google-decision`, `ST_Decisions`) and `default_decision_lane` are byte-for-
      byte what is deployed.
  R8  The pool is not a renamed two-provider pair: every named child contributes
      at least one physical target.

Exit 0 only when every check ran and passed. A check that cannot evaluate its
subject is a failure, never a skip: a guard that verified nothing and exited 0
is how a broken deployment gets a green light.
"""

from __future__ import annotations

import argparse
import sys
import tomllib

SMART = "smart-decision"
REQUIRED_CHILDREN = ("smartlocal-decision", "smartfree-decision", "google-decision")
SEPARATE_LANES = ("smartlocal-decision", "smartfree-decision", "google-decision", "ST_Decisions")
LEGACY_TIER = "tier1-decision"
LEGACY_TIER_CHILDREN = ["smartfree-decision", "smartlocal-decision"]
SPAN = "direct-openrouter-span-01-lite"
OR_TRANSPORT = "open_router_alpha"
STEPHEN_MARKER = "STEPHEN"


def load(path: str) -> dict:
    with open(path, "rb") as handle:
        return tomllib.load(handle)


def reachable_targets(manifest: dict, lane: str) -> list[str]:
    """Physical targets a composed lane can reach, following child lanes once."""
    composition = manifest.get("decision_lane_composition", {})
    lanes = manifest.get("decision_lanes", {})
    entry = composition.get(lane)
    if entry is None:
        raise KeyError(f"{lane} is not a composed lane")
    found: list[str] = []
    for child in entry.get("children", []):
        leaf = lanes.get(child)
        if leaf is None:
            continue
        for target in leaf.get("targets", []):
            if target not in found:
                found.append(target)
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="candidate routes manifest")
    parser.add_argument("--deployed", required=True, help="currently deployed manifest")
    args = parser.parse_args()

    candidate = load(args.manifest)
    deployed = load(args.deployed)
    targets = candidate.get("targets", {})
    composition = candidate.get("decision_lane_composition", {})
    lanes = candidate.get("decision_lanes", {})

    failures: list[str] = []
    ran: list[str] = []

    def check(name: str, condition: bool, detail: str) -> None:
        ran.append(name)
        if not condition:
            failures.append(f"{name}: {detail}")

    # R1 -- the policy is declared, not inferred from the lane's name.
    entry = composition.get(SMART)
    check(
        "R1",
        entry is not None and entry.get("selection") == "capability_first",
        f"{SMART} must declare selection = \"capability_first\"; found "
        f"{None if entry is None else entry.get('selection')!r}",
    )

    # R2 -- the named constituents are present, and nothing else is smuggled in.
    children = list(entry.get("children", [])) if entry else []
    check(
        "R2",
        sorted(children) == sorted(REQUIRED_CHILDREN),
        f"children must be exactly {list(REQUIRED_CHILDREN)}; found {children}",
    )

    # R4 needs the pool; compute it once R2 has established the shape.
    pool: list[str] = []
    try:
        pool = reachable_targets(candidate, SMART)
    except KeyError as error:
        failures.append(f"R4: cannot resolve the pool ({error})")

    # R3 -- every child is a leaf lane with at least one real target.
    missing: list[str] = []
    for child in children:
        leaf = lanes.get(child)
        if leaf is None or not leaf.get("targets"):
            missing.append(child)
            continue
        for target in leaf.get("targets", []):
            declared = targets.get(target)
            if declared is None:
                missing.append(f"{child}->{target} (no such target)")
            elif not declared.get("decision_transport"):
                missing.append(f"{child}->{target} (no decision_transport)")
    check("R3", not missing, f"unresolvable pool members: {missing}")

    # R4 -- one OpenRouter leaf, and it is the allowlisted one.
    openrouter = [
        name
        for name in pool
        if targets.get(name, {}).get("decision_transport") == OR_TRANSPORT
    ]
    check(
        "R4",
        openrouter == [SPAN],
        f"exactly one OpenRouter decision leaf is permitted ({SPAN}); found {openrouter}",
    )

    # R5 -- the owner pool must not reach across into Stephen's account.
    stephen: list[str] = []
    for name in pool:
        declared = targets.get(name, {})
        credential = str(declared.get("decision_api_key_env", ""))
        client = str(declared.get("llm_client", ""))
        if STEPHEN_MARKER in credential.upper():
            stephen.append(f"{name} uses {credential}")
        if STEPHEN_MARKER in client.upper():
            stephen.append(f"{name} uses llm_client {client}")
    check("R5", not stephen, f"Stephen's account is not part of this pool: {stephen}")

    # R6 -- the ordered legacy tier keeps its meaning.
    legacy = composition.get(LEGACY_TIER)
    legacy_children = list(legacy.get("children", [])) if legacy else None
    check(
        "R6",
        legacy is not None
        and legacy.get("selection", "ordered") == "ordered"
        and legacy_children == LEGACY_TIER_CHILDREN,
        f"{LEGACY_TIER} must stay an ordered composition of {LEGACY_TIER_CHILDREN}; "
        f"found selection={None if legacy is None else legacy.get('selection')!r} "
        f"children={legacy_children}",
    )

    # R7 -- existing lanes and the default are untouched.
    drift: list[str] = []
    for lane in SEPARATE_LANES:
        before = deployed.get("decision_lanes", {}).get(lane)
        after = lanes.get(lane)
        if before != after:
            drift.append(lane)
    if deployed.get("default_decision_lane") != candidate.get("default_decision_lane"):
        drift.append("default_decision_lane")
    check("R7", not drift, f"existing decision lanes must not change: {drift}")

    # R8 -- each named child contributes at least one physical target.
    empty = [child for child in children if not lanes.get(child, {}).get("targets")]
    check("R8", not empty and len(pool) >= len(REQUIRED_CHILDREN), f"empty children: {empty}")

    # A guard that evaluated nothing must not report success.
    if len(ran) != 8:
        failures.append(f"only {len(ran)} of 8 checks ran: {ran}")

    if failures:
        for failure in failures:
            print(f"FAIL {failure}", file=sys.stderr)
        return 1
    print(f"PASS {len(ran)} checks: " + ", ".join(ran))
    print(
        "pool: "
        + ", ".join(
            f"{name}[{targets[name]['decision_transport']}]" for name in pool
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
