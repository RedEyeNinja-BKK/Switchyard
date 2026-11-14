#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 RedEyeNinja
# SPDX-License-Identifier: Apache-2.0
"""Check a Switchyard production manifest against the published contract.

The manifest itself stays on the host: it carries private topology, so what is published is a
redacted copy (see ``sanitize_manifest.py``) plus the digests in ``README.md``. That split
creates an obvious question -- is the file that is deployed still the one that was reviewed?
This script answers it, in two layers:

* **On the host**, against the real file: the byte digest must equal the published digest, and
  the redaction of the real file must equal the published redacted file, byte for byte.
* **Anywhere**, against the redacted file: redaction changes endpoint coordinates only, so all
  routing facts survive and can be asserted from the public copy.

Every assertion is a fact about behaviour, not about a file having the right length: the exact
route and target inventory, the context each route advertises, the branch settings on the
local lane, and the absence of every retired profile. A manifest that merely parses is not
accepted; ``--dry-run`` already proves that, and the failure this contract exists to prevent
was a manifest that parsed perfectly and named four models that no longer existed.

Usage:
    manifest_contract.py <manifest.toml> [--profile candidate|recovery]
                         [--sha256 <hex>] [--redacted <published.toml>]

Exit status: 0 contract holds, 1 a fact is false, 2 usage.
"""
from __future__ import annotations

import hashlib
import pathlib
import re
import sys

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    sys.exit("needs python >= 3.11 for tomllib")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import sanitize_manifest  # noqa: E402  (local tool, not a package)

# Values that were measured and accepted; they are not tunable here. Changing one means
# re-deriving the manifest and re-running that measurement, not editing this table.
COMFYNINJA_TEXT = "comfyninja-qwen3_8-27b-q4-mtp"
COMFYNINJA_VISION = "comfyninja-qwen3_8-27b-q4-mtp-vl"
HTPC_PRIMARY = "htpc-qwen3_5-9b-mtp"

# Profiles deleted from the Studio registry. A route naming any of these makes Studio
# re-download a deleted model, so their absence is a safety fact, not tidiness. Listed as
# complete ids rather than a substring so the check cannot drift into matching something
# legitimate.
RETIRED_IDS = (
    "comfyninja-qwen3_8-27b-q3",
    "comfyninja-qwen3_8-27b-q3-mtp",
    "comfyninja-qwen3_8-27b-q3-mtp-vl",
    "comfyninja-qwen3_8-27b-q4",
    "direct-comfyninja-qwen3_8-27b-q3",
    "direct-comfyninja-qwen3_8-27b-q3-mtp",
    "direct-comfyninja-qwen3_8-27b-q3-mtp-vl",
    "direct-comfyninja-qwen3_8-27b-q4",
)

# The composite context that must never be advertised: it exceeds what either candidate can
# serve, so advertising it would promise capacity no leg can deliver.
FORBIDDEN_CONTEXT = 163840

CONTRACTS = {
    "candidate": {
        "counts": {
            "llm_clients": 8, "targets": 36, "routes": 42, "composites": 5,
            "decision_lanes": 3, "capability_clients": 6, "capabilities": 6,
        },
        "routes": {
            COMFYNINJA_TEXT: {"direct-comfyninja-qwen3_8-27b-q4-mtp": {
                "context_window": 179712, "tool_calling": True, "reasoning": True, "vision": False}},
            COMFYNINJA_VISION: {"direct-comfyninja-qwen3_8-27b-q4-mtp-vl": {
                "context_window": 152832, "tool_calling": True, "reasoning": True, "vision": True}},
        },
        "smartlocal": {
            "target": HTPC_PRIMARY,
            "escalation_target": COMFYNINJA_VISION,
            "escalation_max_input_tokens": 61440,
            "context_window": 81920,
        },
        "target_ids": {COMFYNINJA_TEXT: "comfyninja/qwen3.8-27b-q4-mtp",
                       COMFYNINJA_VISION: "comfyninja/qwen3.8-27b-q4-mtp-vl"},
        "forbid_targets": [],
        "composites": {
            "tier2": {"children": ["smartfree", "smartlocal"], "context_window": 225280},
            "tier2-bounded": {"children": ["smartfree-bounded", "smartlocal-bounded"],
                              "context_window": 225280},
        },
    },
    "recovery": {
        "counts": {
            "llm_clients": 7, "targets": 34, "routes": 38, "composites": 5,
            "decision_lanes": 3, "capability_clients": 6, "capabilities": 6,
        },
        "routes": {},
        "smartlocal": None,
        # The recovery manifest exists precisely to remove every local-inference lane, so a
        # surviving one is the failure it was built to prevent.
        "target_ids": {},
        "forbid_targets": [COMFYNINJA_TEXT, COMFYNINJA_VISION],
        "composites": {
            "tier2": {"children": ["smartfree"], "context_window": 1000000},
            "tier2-bounded": {"children": ["smartfree-bounded"], "context_window": 262144},
        },
    },
}


def check(doc: dict, profile: str) -> list[str]:
    fails: list[str] = []
    contract = CONTRACTS[profile]

    for key, expected in contract["counts"].items():
        actual = len(doc.get(key) or {})
        if actual != expected:
            fails.append("%s: expected %d, found %d" % (key, expected, actual))

    targets, routes, composites = (doc.get("targets") or {}, doc.get("routes") or {},
                                   doc.get("composites") or {})

    # No retired id anywhere in the document, not just in the tables that name models: an
    # escalation_target or candidate pointer is the same download hazard. Matched with a
    # token boundary, because every retired id is also a prefix of a retained one
    # (``...-q4`` vs ``...-q4-mtp``) and a prefix match would fail on healthy config.
    blob = repr(doc)
    for retired in RETIRED_IDS:
        if re.search(re.escape(retired) + r"(?![A-Za-z0-9_-])", blob):
            fails.append("retired profile reference survives: %s" % retired)

    for target, model_id in contract["target_ids"].items():
        if target not in targets:
            fails.append("target missing: %s" % target)
        elif targets[target].get("id") != model_id:
            fails.append("target %s: expected id %s, found %s"
                         % (target, model_id, targets[target].get("id")))

    for target in contract["forbid_targets"]:
        if target in targets:
            fails.append("target must not exist: %s" % target)
        if any(r.get("target") == target or target in (r.get("candidates") or [])
               for r in routes.values()):
            fails.append("route still references removed target: %s" % target)
        if any(target in (c.get("children") or []) for c in composites.values()):
            fails.append("composite still references removed target: %s" % target)

    for expected_routes in contract["routes"].values():
        for name, expected in expected_routes.items():
            route = routes.get(name)
            if route is None:
                fails.append("route missing: %s" % name)
                continue
            for key, value in expected.items():
                if route.get(key) != value:
                    fails.append("%s.%s: expected %r, found %r"
                                 % (name, key, value, route.get(key)))

    local = contract["smartlocal"]
    if local:
        for name in ("smartlocal", "smartlocal-bounded"):
            route = routes.get(name)
            if route is None:
                fails.append("route missing: %s" % name)
                continue
            for key, value in local.items():
                if route.get(key) != value:
                    fails.append("%s.%s: expected %r, found %r"
                                 % (name, key, value, route.get(key)))
            if route.get("context_window") == FORBIDDEN_CONTEXT:
                fails.append("%s advertises the forbidden %d" % (name, FORBIDDEN_CONTEXT))

    for name, expected in contract["composites"].items():
        composite = composites.get(name)
        if composite is None:
            fails.append("composite missing: %s" % name)
            continue
        for key, value in expected.items():
            if composite.get(key) != value:
                fails.append("%s.%s: expected %r, found %r"
                             % (name, key, value, composite.get(key)))

    # Referential integrity: a route or composite that names a missing target fails at load,
    # and a target nothing names is dead configuration that will drift.
    referenced: set[str] = set()
    for name, route in routes.items():
        for key in ("target", "escalation_target"):
            value = route.get(key)
            if value:
                referenced.add(value)
                if value not in targets:
                    fails.append("%s.%s names unknown target %s" % (name, key, value))
        for value in route.get("candidates") or []:
            referenced.add(value)
            if value not in targets:
                fails.append("%s.candidates names unknown target %s" % (name, value))
    for name, composite in composites.items():
        for child in composite.get("children") or []:
            if child not in routes and child not in composites:
                fails.append("composite %s names unknown child %s" % (name, child))
    for lane, body in (doc.get("decision_lanes") or {}).items():
        for target in body.get("targets") or []:
            referenced.add(target)
            if target not in targets:
                fails.append("decision lane %s names unknown target %s" % (lane, target))
    for target in targets:
        if target not in referenced:
            fails.append("target is referenced by nothing: %s" % target)

    # The embedding and rerank services on the local host are separate processes, not GGUF
    # profiles loaded by Studio. They must survive every manifest revision.
    for client in ("embedding-local", "reranker-local"):
        if client not in (doc.get("capability_clients") or {}):
            fails.append("capability client missing: %s" % client)

    return fails


def main(argv: list[str]) -> int:
    positional: list[str] = []
    options: dict[str, str] = {}
    index = 1
    while index < len(argv):
        item = argv[index]
        if item in {"--profile", "--sha256", "--redacted"}:
            if index + 1 >= len(argv):
                print("%s needs a value" % item)
                return 2
            options[item] = argv[index + 1]
            index += 2
        elif item.startswith("--"):
            print("unknown option %s" % item)
            return 2
        else:
            positional.append(item)
            index += 1
    if len(positional) != 1:
        print(__doc__.strip().splitlines()[-1])
        return 2

    path = positional[0]
    profile = options.get("--profile", "candidate")
    if profile not in CONTRACTS:
        print("unknown profile %s (expected candidate or recovery)" % profile)
        return 2

    raw = open(path, "rb").read()
    digest = hashlib.sha256(raw).hexdigest()
    doc = tomllib.loads(raw.decode())

    fails = check(doc, profile)

    if "--sha256" in options and options["--sha256"].lower() != digest:
        fails.append("digest: expected %s, found %s" % (options["--sha256"].lower(), digest))

    if "--redacted" in options:
        try:
            redacted, _ = sanitize_manifest.sanitize(path)
        except ValueError as exc:
            fails.append("redaction failed: %s" % exc)
        else:
            published = open(options["--redacted"], "r", encoding="utf-8", newline="").read()
            if redacted != published:
                fails.append("redacted projection differs from %s (regenerate it)"
                             % options["--redacted"])

    if fails:
        print("CONTRACT FAILED -- %s (%s)" % (path, profile))
        for problem in fails:
            print("  - %s" % problem)
        return 1

    routes = doc["routes"]
    print("contract holds -- %s (%s)" % (path, profile))
    print("  sha256   : %s" % digest)
    print("  inventory: %d routes, %d targets, %d composites, %d decision lanes"
          % (len(routes), len(doc["targets"]), len(doc["composites"]),
             len(doc["decision_lanes"])))
    for name in ("smartlocal", "smartlocal-bounded"):
        if name in routes:
            route = routes[name]
            print("  %-18s ctx %d  escalate >%d -> %s"
                  % (name, route["context_window"], route.get("escalation_max_input_tokens", 0),
                     route.get("escalation_target")))
    for name in ("direct-comfyninja-qwen3_8-27b-q4-mtp",
                 "direct-comfyninja-qwen3_8-27b-q4-mtp-vl"):
        if name in routes:
            route = routes[name]
            print("  %-42s ctx %d vision=%s"
                  % (name, route["context_window"], route.get("vision")))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
