#!/usr/bin/env python3
"""Declare `fit_escalation` on `smart-judge`.

The production manifest is not stored in this tree: it names internal hosts. This applier
reads deployed bytes and writes a candidate beside them. It never writes to production and
never restarts anything.

One declaration is added, and nothing else about the route changes:

    [routes.smart-judge]
    fit_escalation = true

What that buys, measured 2026-11-24: the route's two Gemma legs admit at most 12,307
estimated tokens each, because 16,000 tokens per minute padded by the 1,300-per-mille
admission margin is what the window leaves. A judge turn larger than that was refused with
`no_eligible_candidate` and `dispatched: []` even though both declared tiers were wide enough
to hold it -- the primary pool is filtered before the tiers are constructed, so an empty pool
never reached the terminal leg at all. The same route served a 4,946-token request 54 ms
earlier, so the refusal was a size fact, not capacity.

The option is a separate axis from `terminal_policy` on purpose. `terminal_policy` decides
when a pool that failed may spend the next tier; `fit_escalation` decides whether a pool that
never dispatched anything may hand the request up because the request is larger than the pool
can hold. Each tier is still judged on its own gates: a tier that cannot hold it either ends
the walk with the refusal the pool gave.

Ordering constraint, and the reason the applier names the binary: the deployed binary does
not know this field, and the loader refuses unknown route options. The manifest and a binary
that parses it must land together, or the service does not start.

Usage:
  python3 apply_fit_escalation.py --baseline deployed.toml --out candidate.toml [--diff]
"""
from __future__ import annotations

import argparse
import difflib
import sys
import tomllib

# The smart-judge declaration as deployed in release 20261124T021742Z (routes.toml sha256
# f4f7b3b6...). `terminal_policy = "availability"` alone appears more than once in the
# manifest, so the anchor carries the two-tier declaration that makes it unique.
ANCHOR = (
    'terminal_fallback = [["openai-gpt-6-luna-nt"], ["htpc-qwen3_5-9b-mtp-nt"]]\n'
    'terminal_policy = "availability"\n'
)
DECLARATION = "fit_escalation = true\n"
ROUTE = "smart-judge"


def apply(text: str) -> str:
    """Return the candidate manifest, or raise ValueError naming the refusal."""
    if "fit_escalation" in text:
        raise ValueError("refused: the manifest already declares fit_escalation")
    hits = text.count(ANCHOR)
    if hits != 1:
        raise ValueError(
            f"refused: the smart-judge anchor matched {hits} times, expected exactly 1"
        )
    candidate = text.replace(ANCHOR, ANCHOR + DECLARATION)

    before = tomllib.loads(text)
    after = tomllib.loads(candidate)
    route_before = before["routes"][ROUTE]
    route_after = after["routes"][ROUTE]
    if route_after.get("fit_escalation") is not True:
        raise ValueError("refused: the candidate does not declare fit_escalation = true")
    if {key: value for key, value in route_after.items() if key != "fit_escalation"} != route_before:
        raise ValueError("refused: the candidate changed a field beside fit_escalation")

    # Everything outside the route must be byte-identical as parsed objects; a regex or
    # block-anchor edit that spilled into a neighbouring table would otherwise pass the
    # route-level check above.
    rest_before = {key: value for key, value in before.items() if key != "routes"}
    rest_after = {key: value for key, value in after.items() if key != "routes"}
    if rest_before != rest_after:
        raise ValueError("refused: the candidate changed a section outside [routes.*]")
    routes_before = {key: value for key, value in before["routes"].items() if key != ROUTE}
    routes_after = {key: value for key, value in after["routes"].items() if key != ROUTE}
    if routes_before != routes_after:
        raise ValueError("refused: the candidate changed another route")

    added = [
        line
        for line in difflib.unified_diff(text.splitlines(), candidate.splitlines(), n=0)
        if line.startswith("+") and not line.startswith("+++")
    ]
    if added != ["+" + DECLARATION.rstrip("\n")]:
        raise ValueError(f"refused: the edit is not one added line ({added})")
    return candidate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, help="deployed routes.toml")
    parser.add_argument("--out", required=True, help="candidate path to write")
    parser.add_argument("--diff", action="store_true", help="print the change")
    arguments = parser.parse_args()

    with open(arguments.baseline, encoding="utf-8") as handle:
        baseline = handle.read()
    try:
        candidate = apply(baseline)
    except ValueError as refusal:
        print(refusal, file=sys.stderr)
        return 1
    with open(arguments.out, "w", encoding="utf-8") as handle:
        handle.write(candidate)
    if arguments.diff:
        sys.stdout.writelines(
            difflib.unified_diff(
                baseline.splitlines(keepends=True),
                candidate.splitlines(keepends=True),
                fromfile=arguments.baseline,
                tofile=arguments.out,
            )
        )
    print(f"wrote {arguments.out}; the candidate needs a binary that parses {ROUTE}.fit_escalation")
    return 0


if __name__ == "__main__":
    sys.exit(main())
