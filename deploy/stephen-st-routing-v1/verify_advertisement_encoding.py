#!/usr/bin/env python3
"""Prove the gate's expected-on-wire rule against the running binary.

`direct_capability_gate.py` compares a manifest's declarations with a saved
`GET /v1/models` payload. That comparison is only meaningful if the rule used to
predict the wire value is the rule the binary actually applies, so this script
states the rule as code and checks it against a live pair.

The rule, read from `RouteConfig::capabilities()` and `model_entry_json()`:

  context_length   <- route.context_window       (absent -> JSON null)
  tool_calling     <- route.tool_calling         (absent -> JSON null)
  vision           <- route.vision               (absent -> JSON null)
  video            <- route.video                (absent -> JSON null)
  video_carriers   <- ["inline"]/["remote"] only when route.video is true AND
                      route.video_carrier names one carrier, else null
  image_carriers   <- the same rule over route.vision / route.image_carrier

Nothing here reads the target: `/v1/models` advertises what the *route* declares,
which is why an undeclared route field reaches the wire as null rather than
inheriting the target's fact.

Exit 0 when every advertised route agrees; 3 on disagreement, naming each route
and field; 2 when the pair cannot be compared at all (a route the payload does
not carry, or a payload route the manifest does not define).

Usage:
  python3 verify_advertisement_encoding.py --manifest CURRENT/routes.toml \
      --models live-models.json [--quiet]
"""
from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path

MEDIA = {"inline": ["inline"], "remote": ["remote"], "both": None}


def expected_route_values(route: dict) -> dict:
    """The wire values the binary derives from one route's own declarations."""
    def carriers(medium: str, field: str):
        if route.get(medium) is True:
            return MEDIA.get(route.get(field), None)
        return None

    return {
        "context_length": route.get("context_window"),
        "tool_calling": route.get("tool_calling"),
        "vision": route.get("vision"),
        "video": route.get("video"),
        "video_carriers": carriers("video", "video_carrier"),
        "image_carriers": carriers("vision", "image_carrier"),
    }


def observed_route_values(entry: dict) -> dict:
    capabilities = entry.get("capabilities")
    if not isinstance(capabilities, dict):
        raise ValueError(f"{entry.get('id')}: capabilities is not an object")
    values = {"context_length": entry.get("context_length")}
    for key in ("tool_calling", "vision", "video", "video_carriers", "image_carriers"):
        values[key] = capabilities.get(key)
    return values


def compare(manifest: dict, models: dict) -> tuple[list[str], list[str]]:
    advertised = models["data"] if isinstance(models, dict) and "data" in models else models
    if not isinstance(advertised, list):
        raise ValueError("model payload has no list of models")
    live = {entry["id"]: entry for entry in advertised}
    routes = manifest.get("routes") or {}
    # A held alias is registered but not advertised, so it is not comparable here.
    named = {name: route for name, route in routes.items() if not route.get("held")}
    problems: list[str] = []
    for name in sorted(set(named) ^ set(live)):
        where = "advertised but not defined" if name in live else "defined but not advertised"
        problems.append(f"ROUTE_SET_MISMATCH {name}: {where}")
    checked = 0
    for name in sorted(set(named) & set(live)):
        expected = expected_route_values(named[name])
        try:
            observed = observed_route_values(live[name])
        except ValueError as exc:
            problems.append(f"PAYLOAD_SHAPE {exc}")
            continue
        for field, want in expected.items():
            if observed[field] != want:
                problems.append(f"ENCODING_MISMATCH {name}.{field}: "
                                f"rule={want!r}, GET={observed[field]!r}")
        checked += 1
    return problems, [f"checked {checked} advertised routes against the live payload",
                      f"manifest routes={len(named)} payload routes={len(live)}"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    try:
        manifest = tomllib.loads(args.manifest.read_text())
        models = json.loads(args.models.read_text())
        problems, summary = compare(manifest, models)
    except (OSError, ValueError, KeyError, json.JSONDecodeError,
            tomllib.TOMLDecodeError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    for line in summary:
        print(line)
    for problem in problems:
        print(problem)
    if problems:
        print(f"ENCODING_NOT_PROVEN: {len(problems)} disagreement(s)")
        return 3
    print("ENCODING_PROVEN: the gate's expected-on-wire rule reproduces the payload")
    return 0


if __name__ == "__main__":
    sys.exit(main())
