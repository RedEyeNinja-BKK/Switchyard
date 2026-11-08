#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Route manifest contract audit.

Checks a Switchyard manifest against the model list the server actually
advertises, and enforces the routing invariants that load-time validation cannot
express. Exit 0 when every invariant holds, exit 1 with one line per violation.

WHAT THIS ADDS OVER CONFIG LOAD
    Config load rejects a manifest that cannot be served. It cannot tell you that
    an internal composite is being advertised, that a leaf is unreachable by name,
    that a bounded composite contains a child with no bounded contract, or that the
    advertisement disagrees with the manifest it was built from. Those are the
    mistakes that stay invisible until a route is called.

THE INVARIANTS
    1. Advertised set == leaf route set. Composites are internal routing objects:
       callable by id, never listed. A leaf missing from the advertisement is a
       route nobody can call; a composite in it advertises an alias whose children
       are an internal detail.
    2. Every advertised entry agrees with the manifest on context window and on
       each declared capability.
    3. Every passthrough target exists and names a configured client.
    4. `video = true` only appears on an `openai_responses` client, because the
       Chat codec has no faithful video representation.
    5. Every composite child exists, is a leaf, and no composite nests another.
    6. A `-bounded` route pins `route_reasoning_policy = "none"`, and a `-bounded`
       composite names only children that do the same.
    7. No advertised id is absent from the manifest.

USAGE
    route_audit.py <manifest.toml> <models.json>
    route_audit.py <manifest.toml> --models-url http://127.0.0.1:4000

    Capture the advertisement from the running server so the audit compares the
    deployment to the file it was built from. In CI, start the built binary on a
    sanitized fixture and audit that.
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
import urllib.request
from pathlib import Path

# Clients whose endpoint is not reached over the public internet. Used only to
# label a route local or remote in the inventory; it never affects a check.
LOCAL_CLIENTS = {"local_a", "local_b", "comfyninja", "htpc"}

# Credentials that must never be mixed inside one route, supplied by the caller:
# which accounts may not share a route is deployment knowledge, so it is an
# argument rather than a constant in a public repository. A role-to-role fallback
# across ordinary providers is designed; serving a caller from an account it does
# not belong to is a boundary violation. See --exclusive-credentials.


def is_bounded(name: str) -> bool:
    """True for a route or composite that carries the bounded contract.

    A suffix test, not a substring one: `prims-unbounded` contains the word
    "bounded" and is the opposite of a bounded route.
    """
    return name.endswith("-bounded") or name.endswith("_bounded")


def load(manifest: Path, models: Path) -> tuple[dict, dict]:
    config = tomllib.loads(manifest.read_text(encoding="utf-8"))
    payload = json.loads(models.read_text(encoding="utf-8"))
    return config, {row["id"]: row for row in payload["data"]}


def fetch_models(base: str) -> dict:
    # Accept a base URL or the full endpoint, so a caller can paste either.
    url = base if base.rstrip("/").endswith("/v1/models") else base.rstrip("/") + "/v1/models"
    with urllib.request.urlopen(url, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return {row["id"]: row for row in payload["data"]}


def classify(name: str, kind: str) -> str:
    if kind == "composite":
        return "composite"
    if name.startswith("direct-"):
        return "stephen-direct" if "stephen" in name else "global-direct"
    if name.startswith("ST_"):
        return "tier"
    return "primitive"


def audit(
    config: dict,
    advertised: dict,
    exclusive_credentials: tuple[set[str], ...] = (),
) -> tuple[list[dict], list[str]]:
    targets = config["targets"]
    routes = config["routes"]
    composites = config.get("composites", {})
    clients = config["llm_clients"]
    failures: list[str] = []
    inventory: list[dict] = []

    for name, route in routes.items():
        target = route.get("target")
        resolved = targets.get(target or "", {})
        client = resolved.get("llm_client")
        inventory.append(
            {
                "route": name,
                "class": classify(name, route.get("type", "passthrough")),
                "kind": route.get("type"),
                "target": target,
                "model": resolved.get("id", ""),
                "client": client,
                "locality": "local" if client in LOCAL_CLIENTS else "remote",
                "context_window": route.get("context_window"),
                "vision": route.get("vision"),
                "video": route.get("video"),
                "tools": route.get("tool_calling"),
                "reasoning": route.get("reasoning"),
                "policy": route.get("route_reasoning_policy", "(caller)"),
                "candidates": route.get("candidates", []),
            }
        )

        entry = advertised.get(name)
        if entry is None:
            failures.append(f"{name}: leaf route is not advertised")
        else:
            if entry.get("context_length") != route.get("context_window"):
                failures.append(
                    f"{name}: context_window manifest={route.get('context_window')} "
                    f"advertised={entry.get('context_length')}"
                )
            capabilities = entry.get("capabilities") or {}
            for field in ("vision", "tool_calling", "video"):
                declared = route.get(field)
                if declared is not None and capabilities.get(field) != declared:
                    failures.append(
                        f"{name}: {field} manifest={declared} advertised={capabilities.get(field)}"
                    )

        if not resolved:
            failures.append(f"{name}: target {target!r} is not defined")
            continue
        if client not in clients:
            failures.append(f"{name}: target client {client!r} is not configured")
            continue
        if resolved.get("video") is True and clients[client]["format"] != "openai_responses":
            failures.append(f"{name}: video target on non-responses client {client}")

        # Candidate fallback ACROSS providers is designed: a route may legitimately
        # fall from a local or one provider's model to another's. What is never
        # legitimate is mixing two credentials that must stay separate, which serves
        # a caller from an account it does not belong to.
        credential_clients = {client}
        for candidate in route.get("candidates", []):
            if candidate not in targets:
                failures.append(f"{name}: candidate {candidate!r} is not defined")
            else:
                credential_clients.add(targets[candidate].get("llm_client"))
        for pair in exclusive_credentials:
            if pair <= credential_clients:
                failures.append(
                    f"{name}: mixes credentials {sorted(pair)}; a route must stay on one"
                )

        if is_bounded(name) and route.get("route_reasoning_policy") != "none":
            failures.append(
                f"{name}: bounded route must pin route_reasoning_policy = \"none\", "
                f"found {route.get('route_reasoning_policy')!r}"
            )

    for name, composite in composites.items():
        for child in composite.get("children", []):
            if child in composites:
                failures.append(f"{name}: composite child {child!r} is itself a composite")
            elif child not in routes:
                failures.append(f"{name}: unknown child {child!r}")
            elif is_bounded(name) and not is_bounded(child):
                failures.append(
                    f"{name}: bounded composite names unbounded child {child!r}"
                )
        inventory.append(
            {
                "route": name,
                "class": "composite",
                "kind": "composite",
                "target": None,
                "model": "",
                "client": ",".join(
                    sorted(
                        {
                            targets[routes[child]["target"]]["llm_client"]
                            for child in composite.get("children", [])
                            if child in routes
                        }
                    )
                ),
                "locality": "mixed",
                "context_window": composite.get("context_window"),
                "vision": composite.get("vision"),
                "video": composite.get("video"),
                "tools": composite.get("tool_calling"),
                "reasoning": composite.get("reasoning"),
                "policy": "per-child",
                "candidates": composite.get("children", []),
            }
        )

    for advertised_id in advertised:
        if advertised_id in composites:
            failures.append(f"{advertised_id}: composite is advertised; only leaves may be")
        elif advertised_id not in routes:
            failures.append(f"{advertised_id}: advertised but absent from the manifest")

    return inventory, failures


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("manifest", type=Path)
    parser.add_argument("models", nargs="?", type=Path, help="captured /v1/models payload")
    parser.add_argument("--models-url", help="fetch /v1/models from a running server")
    parser.add_argument("--inventory", type=Path, help="write the route inventory as JSON")
    parser.add_argument("--quiet", action="store_true", help="print only the verdict")
    parser.add_argument(
        "--exclusive-credentials",
        action="append",
        default=[],
        metavar="A,B",
        help="two client names that must never appear in one route (repeatable)",
    )
    args = parser.parse_args(argv)

    if args.models_url:
        advertised = fetch_models(args.models_url)
        config = tomllib.loads(args.manifest.read_text(encoding="utf-8"))
    elif args.models:
        config, advertised = load(args.manifest, args.models)
    else:
        parser.error("provide a captured models payload or --models-url")

    exclusive = tuple(
        set(pair.split(","))
        for pair in args.exclusive_credentials
        if len(pair.split(",")) == 2
    )
    for pair in args.exclusive_credentials:
        if len(pair.split(",")) != 2:
            parser.error(f"--exclusive-credentials expects A,B; got {pair!r}")

    inventory, failures = audit(config, advertised, exclusive)

    if args.inventory:
        args.inventory.write_text(json.dumps(inventory, indent=1), encoding="utf-8")

    if not args.quiet:
        leaves = sum(1 for row in inventory if row["kind"] != "composite")
        print(
            f"routes audited: {len(inventory)} ({leaves} leaves, "
            f"{len(inventory) - leaves} composites)  advertised: {len(advertised)}"
        )
        for row in sorted(inventory, key=lambda r: (r["class"], r["route"])):
            flags = "".join(
                [
                    "V" if row["vision"] else "-",
                    "T" if row["tools"] else "-",
                    "R" if row["reasoning"] else "-",
                    "v" if row["video"] else "-",
                ]
            )
            print(
                f"  {row['class']:14s} {row['route']:52s} {str(row['client'])[:18]:18s} "
                f"{row['locality']:6s} ctx={str(row['context_window']):>8s} {flags}"
            )

    if failures:
        print("\nFAIL", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print("\nPASS: every route satisfies the manifest contract", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
