#!/usr/bin/env python3
"""Prove a candidate manifest cannot regress the DEPLOYED manifest.

This is the guard the stacking mistake teaches: a candidate generated from a
hand-captured snapshot silently reverts everything deployed after that snapshot.
The Batch A/B verifiers each check their own delta; neither of them can see a
section that the other workstream added to production between capture and now.

Anchored on the deployed bytes, this check refuses:

  * a deployed target or route that the candidate drops, unless it is named in
    SANCTIONED_REMOVALS (with the reason it was approved);
  * a changed key on a deployed object that is not in ALLOWED_TARGET_KEYS or
    ALLOWED_ROUTE_KEYS for that object;
  * a top-level section that loses a key, or a new top-level section.

Exit 0 only when every deployed object survives the candidate.

Negative controls are run by mutants_live_regression.py, which injects a
deployed-only target and a deployed-only key and requires this script to fail.
"""

from __future__ import annotations

import argparse
import sys
import tomllib

# Removals approved in Batch A, with the evidence that made them safe. Anything
# not listed here is a regression.
SANCTIONED_REMOVALS = {
    "targets": {
        "stephen-openrouter-free": (
            "dynamic openrouter/free router target; 0 requests naming it in the "
            "last 30 days of routing log (Batch A, check A25)"
        ),
    },
    "routes": {
        "direct-openrouter-stephen-openrouter-free": (
            "leaf route of the removed dynamic target; same zero-reference evidence"
        ),
    },
}

# Keys a candidate may change on a deployed object. Anything else changing is a
# silent edit to production behaviour nobody reviewed.
ALLOWED_TARGET_KEYS = {
    # Batch A: explicit tariff declaration replaces the old free_tier spelling.
    # Batch B: pool membership labels.
    "*": {"free_tier", "groups"},
}
ALLOWED_ROUTE_KEYS = {
    # Batch A replaces the dynamic router leg with the six individually qualified
    # free leaves; Batch B makes ST_Free a peer pool. Both are reviewed.
    "ST_Free": {"selection", "pool_groups", "legs", "candidates", "context_window"},
    # Batch A corrects these three windows from captured provider facts
    # (ERNIE 1,024,000 / Ling 1,048,576 / Nemotron 1,000,000).
    "direct-openrouter-stephen-ernie-5_5-flash-preview-free": {"context_window"},
    "direct-openrouter-stephen-ling-3_2-flash-free": {"context_window"},
    "direct-openrouter-stephen-nemotron-3-5-super-vl-free": {"context_window"},
    # Batch F1/F2. smart-nt's candidate list drops the HELD 2.5-flash leaf (a held
    # route answers 404) and moves DeepSeek NT out of the ranked pool and into the
    # exhaustion-only terminal leg. Deliberately narrow: `selection`, windows and
    # every other key on smart-nt stay unreviewed, and M6 proves that still fails.
    "smart-nt": {"candidates", "terminal_fallback"},
    # Batch F1, the same held-404 cleanup applied to the owner Gemini profile.
    "smart_gemini": {"candidates"},
    "*": set(),
}


def load(path: str) -> dict:
    with open(path, "rb") as handle:
        return tomllib.load(handle)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", required=True, help="deployed manifest bytes")
    parser.add_argument("--candidate", required=True, help="candidate manifest")
    args = parser.parse_args(argv)

    live = load(args.live)
    cand = load(args.candidate)
    failures: list[str] = []
    judgements = 0

    def check(ok: bool, label: str, detail: str = "") -> None:
        nonlocal judgements
        judgements += 1
        status = "PASS" if ok else "FAIL"
        print(f"{status}  {label}" + (f"   {detail}" if detail else ""))
        if not ok:
            failures.append(label)

    # L1/L2 — nothing deployed disappears except by explicit sanction.
    for section in ("targets", "routes"):
        dropped = sorted(set(live.get(section, {})) - set(cand.get(section, {})))
        sanctioned = set(SANCTIONED_REMOVALS[section])
        unexpected = [name for name in dropped if name not in sanctioned]
        check(
            not unexpected,
            f"L{1 if section == 'targets' else 2} no deployed {section[:-1]} is dropped",
            f"dropped={dropped} sanctioned={sorted(sanctioned & set(dropped))} "
            f"unexpected={unexpected}",
        )

    # L3 — every deployed target survives with only reviewed keys changed.
    drifted: list[str] = []
    for name, body in live.get("targets", {}).items():
        if name not in cand.get("targets", {}):
            continue  # covered by L1
        other = cand["targets"][name]
        allowed = ALLOWED_TARGET_KEYS["*"]
        changed = {k for k in set(body) | set(other) if body.get(k) != other.get(k)}
        if changed - allowed:
            drifted.append(f"{name}:{sorted(changed - allowed)}")
    check(not drifted, "L3 deployed targets change only in reviewed keys", f"drifted={drifted}")

    # L4 — every deployed route survives with only reviewed keys changed.
    drifted_routes: list[str] = []
    for name, body in live.get("routes", {}).items():
        if name not in cand.get("routes", {}):
            continue  # covered by L2
        other = cand["routes"][name]
        allowed = ALLOWED_ROUTE_KEYS.get(name, ALLOWED_ROUTE_KEYS["*"])
        changed = {k for k in set(body) | set(other) if body.get(k) != other.get(k)}
        if changed - allowed:
            drifted_routes.append(f"{name}:{sorted(changed - allowed)}")
    check(
        not drifted_routes,
        "L4 deployed routes change only in reviewed keys",
        f"drifted={drifted_routes}",
    )

    # L5 — a top-level section may be added, never lose a key.
    lost: list[str] = []
    for section, body in live.items():
        if section in {"targets", "routes"}:
            continue
        other = cand.get(section)
        if other is None:
            lost.append(f"{section}: whole section dropped")
            continue
        if isinstance(body, dict):
            for key in body:
                if key not in other:
                    lost.append(f"{section}.{key}")
                elif body[key] != other[key]:
                    lost.append(f"{section}.{key} changed")
    check(not lost, "L5 deployed non-target sections survive unchanged", f"lost={lost}")

    # L6 — the live object census is accounted for exactly.
    live_targets = set(live.get("targets", {}))
    cand_targets = set(cand.get("targets", {}))
    check(
        live_targets <= cand_targets | set(SANCTIONED_REMOVALS["targets"]),
        "L6 every deployed target is present or explicitly sanctioned for removal",
        f"unaccounted={sorted(live_targets - cand_targets - set(SANCTIONED_REMOVALS['targets']))}",
    )

    print(f"\n{judgements} checks, {len(failures)} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
