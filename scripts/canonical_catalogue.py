#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Emit a reviewable catalogue of a deployed manifest, with target reuse and consumer caps.

The manifest says what the router serves. It does not say, in one place, which targets are
shared between routes, which routes a Turnstone alias actually consumes, or where the
consumer's context ceiling differs from the route's. Those are the three questions an
operator asks when deciding whether a route change is safe, and answering them by hand is how
stale tables get written.

Sources, each named in the output so a reader can tell what was measured from what was
declared:

  * the deployed ``routes.toml`` -- every route and target fact here is DECLARED, from these
    exact bytes, identified by their SHA-256;
  * ``capabilities.json`` -- instrumentation contract counts, also from the deployed bytes;
  * an optional ``/v1/models`` snapshot -- what the running process ADVERTISES. A route that
    is callable but not advertised (a composite) is a real state, so absence is recorded as
    an absence rather than an error;
  * an optional Turnstone registry export -- what a CONSUMER will accept, bound to each alias.

Nothing here is sampled from traffic. A cell that is not in one of those sources is reported
as ``not_established``; no grade is inferred from a sibling.

Usage:
    canonical_catalogue.py <routes.toml> \
        [--capabilities capabilities.json] [--models v1-models.json] \
        [--registry registry-effective.json] --out-dir <dir>
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import sys
import tomllib

CAP_KEYS = (
    "tool_calling",
    "reasoning",
    "vision",
    "video",
    "video_carrier",
    "image_carrier",
)
ALLOWANCE_KEYS = (
    "free_tier",
    "daily_request_limit",
    "requests_per_minute",
    "input_tokens_per_minute",
    "max_input_tokens",
)


def load_json(path: pathlib.Path | None) -> dict | None:
    if path is None:
        return None
    return json.loads(path.read_text())


def route_targets(route: dict) -> tuple[list[str], list[list[str]]]:
    """The ordered primary pool and the terminal tiers, as the manifest spells them."""
    primary: list[str] = []
    if route.get("target"):
        primary.append(route["target"])
    primary.extend(route.get("candidates") or [])
    tiers = [list(tier) for tier in (route.get("terminal_fallback") or [])]
    return primary, tiers


def main() -> int:
    if "--self-test" in sys.argv[1:]:
        return self_test()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=pathlib.Path)
    parser.add_argument("--capabilities", type=pathlib.Path)
    parser.add_argument("--models", type=pathlib.Path)
    parser.add_argument("--registry", type=pathlib.Path)
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()

    raw = args.manifest.read_bytes()
    manifest_sha = hashlib.sha256(raw).hexdigest()
    document = tomllib.loads(raw.decode("utf-8"))
    # A canonical projection records the SHA-256 of the manifest it was derived from. When the
    # input is a projection, the catalogue must name those deployed bytes as its source; the
    # projection's own hash is kept beside it so a reader can tell which file was read.
    projected_from = (document.get("projection") or {}).get("source_manifest_sha256")
    source_manifest_sha = projected_from or manifest_sha
    projection_source = "canonical projection" if projected_from else "deployed manifest"

    routes: dict = document.get("routes", {})
    targets: dict = document.get("targets", {})
    composites: dict = document.get("composites", {})
    lanes: dict = document.get("decision_lanes", {})
    lane_composition: dict = document.get("decision_lane_composition", {})
    capability_lanes: dict = document.get("capabilities", {})
    clients: dict = document.get("llm_clients", {})

    models = load_json(args.models)
    advertised = {m["id"]: m for m in models["data"]} if models else None
    registry = load_json(args.registry)
    capabilities = load_json(args.capabilities)

    # --- target reuse ------------------------------------------------------------------
    # Only a name declared in a ``[targets.*]`` table is a target. A reference made by
    # another section is resolved against that set; one that does not resolve is recorded
    # under the section that made it, never counted as a target. An earlier version of this
    # script fed composite children into the same map, so seven ROUTE names were published
    # as single-use targets and the single-use count read 36 for 29 real targets.
    used_by: dict[str, list[dict]] = {name: [] for name in targets}
    non_target_references: dict[str, list[dict]] = {}

    def reference(owner: str, owner_kind: str, name: str, role: str, tier: int | None = None) -> None:
        entry = {"owner": owner, "owner_kind": owner_kind, "role": role, "tier": tier}
        if name in targets:
            used_by[name].append(entry)
        else:
            non_target_references.setdefault(owner_kind, []).append(
                {**entry, "reference": name, "resolves_to": "nothing declared"}
            )

    for name, route in routes.items():
        primary, tiers = route_targets(route)
        for position, target in enumerate(primary):
            reference(name, "route", target, "primary" if position == 0 else "candidate")
        for tier_index, tier in enumerate(tiers):
            for target in tier:
                reference(name, "route", target, "terminal", tier_index)
    # A composite names ROUTES, never targets: its children are a call path, not a failure
    # domain. They are recorded separately so a target count cannot be inflated by nesting.
    for name, composite in composites.items():
        for child in composite.get("children") or composite.get("candidates") or []:
            non_target_references.setdefault("composite_child", []).append(
                {
                    "owner": name,
                    "owner_kind": "composite",
                    "role": "composite_child",
                    "tier": None,
                    "reference": child,
                    "resolves_to": "route" if child in routes else "composite" if child in composites else "nothing declared",
                }
            )
    # Decision lanes reach a target without being a chat route. Leaving them out would report
    # a live target as unreferenced, which is the kind of absence that gets quoted and then
    # acted on.
    for name, lane in lanes.items():
        for target in lane.get("targets") or []:
            reference(name, "decision_lane", target, "decision_lane")
    # A rerank or embedding capability names a capability CLIENT, not a chat target: it has
    # its own namespace with its own base_url and model string. Recording it against
    # ``targets`` would invent target entries no ``[targets.*]`` table declares, so the
    # reference is checked for resolution instead of being counted as reuse.
    capability_targets_undeclared = sorted(
        name
        for name, capability in capability_lanes.items()
        if capability.get("target") not in (document.get("capability_clients") or {})
    )
    capability_client_count = len(document.get("capability_clients") or {})
    capability_client_models_not_targets = sorted(
        {
            client.get("model")
            for client in (document.get("capability_clients") or {}).values()
            if isinstance(client.get("model"), str) and client["model"] not in targets
        }
    )

    declared_groups: dict[str, list[str]] = {}
    for name, target in targets.items():
        for group in target.get("groups") or []:
            declared_groups.setdefault(group, []).append(name)

    # --- consumer side -----------------------------------------------------------------
    # A Turnstone alias is a consumer of THIS router only if its base_url points here, and its
    # model string is a name this manifest declares. Both halves are checked, because the
    # registry also carries rows that name an upstream model and talk to the provider
    # directly; crediting those to the manifest would overstate what it serves, and a row
    # addressed here whose name resolves to nothing would be a live 404 rather than a route.
    upstream_ids: dict[str, list[str]] = {}
    for target_name, target in targets.items():
        if target.get("id"):
            upstream_ids.setdefault(target["id"], []).append(target_name)
    declared_names: dict[str, str] = {
        "route": set(routes),
        "composite": set(composites),
        "decision_lane": set(lanes),
        "lane_composition": set(lane_composition),
        "capability": set(capability_lanes),
    }
    name_index: dict[str, str] = {}
    # Narrowest first, so a name that appears in two sections is recorded against the section
    # that actually answers it (a capability id is a lane, never a chat route).
    for kind in ("capability", "lane_composition", "decision_lane", "composite", "route"):
        for name in declared_names[kind]:
            name_index.setdefault(name, kind)

    def classify_name(model: str) -> str:
        if model in name_index:
            return name_index[model]
        if model in upstream_ids:
            return "upstream_model_id"
        return "undeclared"

    def destination(base_url: str | None) -> str:
        """A coarse class of where a row sends its traffic.

        The evidence artifact is published, so the deployment's exact addresses stay out of
        it for the same reason the manifest projection replaces ``base_url`` with
        ``endpoint_class``. The class is what the question needs: does this row reach this
        router, and if not, which family does it reach instead.
        """
        if not base_url:
            return "no_base_url"
        name = base_url.split("//", 1)[-1].split("/", 1)[0].partition(":")[0]
        if name.startswith("switchyard-vm"):
            return "this_router"
        if name.endswith(".ts.net"):
            return "private_engine"
        labels = name.split(".")
        for family in ("openrouter", "openai", "deepseek", "thaillm", "google"):
            if family in labels or any(l.startswith(family) for l in labels):
                return family
        return "other_provider"

    consumer_by_route: dict[str, list[dict]] = {}
    capability_consumers: dict[str, list[dict]] = {}
    consumer_rows: list[dict] = []
    if registry:
        for row in registry["models"]:
            kind = classify_name(row["model"])
            view = {
                "alias": row["alias"],
                "model": row["model"],
                "provider": row["provider"],
                "destination": destination(row.get("base_url")),
                "context_window": row["context_window"],
                "enabled": bool(row["enabled"]),
                "name_kind": kind,
                "declared_here": kind not in ("upstream_model_id", "undeclared"),
            }
            consumer_rows.append(view)
            if kind == "capability":
                capability_consumers.setdefault(row["model"], []).append(view)
            else:
                consumer_by_route.setdefault(row["model"], []).append(view)

    catalogue_routes = []
    for name in sorted(routes):
        route = routes[name]
        primary, tiers = route_targets(route)
        caps = {key: route.get(key) for key in CAP_KEYS if key in route}
        consumers = consumer_by_route.get(name, [])
        route_window = route.get("context_window")
        consumer_view = []
        for consumer in consumers:
            consumer_view.append(
                {
                    **consumer,
                    "ceiling_delta": (
                        None
                        if route_window is None or consumer["context_window"] is None
                        else consumer["context_window"] - route_window
                    ),
                }
            )
        catalogue_routes.append(
            {
                "route": name,
                "type": route.get("type"),
                "primary": route.get("target"),
                "candidates": route.get("candidates") or [],
                "pool_groups": route.get("pool_groups") or [],
                "selection": route.get("selection"),
                "group_ordering": route.get("group_ordering"),
                "terminal_fallback": tiers,
                "terminal_policy": route.get("terminal_policy"),
                "terminal_policies": route.get("terminal_policies") or [],
                "context_window": route_window,
                "capabilities": caps,
                "evidence": {
                    "declared": True,
                    "advertised": None if advertised is None else name in advertised,
                    "consumers": consumer_view,
                },
            }
        )

    catalogue_targets = []
    for name in sorted(targets):
        target = targets[name]
        catalogue_targets.append(
            {
                "target": name,
                "upstream_id": target.get("id"),
                "llm_client": target.get("llm_client"),
                "groups": target.get("groups") or [],
                "context_window": target.get("context_window"),
                "capabilities": {key: target.get(key) for key in CAP_KEYS if key in target},
                "allowance": {key: target.get(key) for key in ALLOWANCE_KEYS if key in target},
                "reasoning_policy": target.get("reasoning_policy"),
                "reached_by": used_by.get(name, []),
            }
        )

    # The three reuse classes are a partition of the declared targets, not three independent
    # filters over a map that may hold other things. Asserted, because a count that silently
    # includes a non-target is exactly the error this replaces.
    referenced_by: dict[str, set[str]] = {
        name: {entry["owner"] for entry in entries} for name, entries in used_by.items()
    }
    shared = {name: entries for name, entries in used_by.items() if len(referenced_by[name]) > 1}
    single = sorted(name for name in used_by if len(referenced_by[name]) == 1)
    unreferenced = sorted(name for name in used_by if not referenced_by[name])
    partition = {
        "shared": set(shared),
        "single_use": set(single),
        "unreferenced": set(unreferenced),
    }
    if not (
        partition["shared"] | partition["single_use"] | partition["unreferenced"] == set(targets)
        and not (partition["shared"] & partition["single_use"])
        and not (partition["shared"] & partition["unreferenced"])
        and not (partition["single_use"] & partition["unreferenced"])
    ):
        print("PARTITION FAIL: the reuse classes do not partition the declared targets", file=sys.stderr)
        return 2

    counts = {
        "routes": len(routes),
        "direct_routes": sum(1 for name in routes if name.startswith("direct-")),
        "targets": len(targets),
        "composites": len(composites),
        "decision_lanes": len(lanes),
        "llm_clients": len(clients),
        "advertised": None if advertised is None else len(advertised),
        "routes_advertised": (
            None if advertised is None else sum(1 for n in routes if n in advertised)
        ),
        "routes_not_advertised": (
            None if advertised is None else sorted(n for n in routes if n not in advertised)
        ),
        "advertised_not_declared": (
            None if advertised is None else sorted(set(advertised) - set(routes))
        ),
        "composites_advertised": (
            None if advertised is None else sorted(n for n in composites if n in advertised)
        ),
        "targets_shared": len(shared),
        "targets_single_use": len(single),
        "targets_unreferenced": unreferenced,
        "targets_partition_holds": (
            len(shared) + len(single) + len(unreferenced) == len(targets)
        ),
        "targets_reached_by_a_lane_only": sorted(
            n for n, e in used_by.items() if e and all(x["role"] == "decision_lane" for x in e)
        ),
        "non_target_references": {
            kind: sorted({entry["reference"] for entry in entries})
            for kind, entries in sorted(non_target_references.items())
        },
        "capability_client_models_not_targets": capability_client_models_not_targets,
        "capability_clients": capability_client_count,
        "registry_rows": None if not registry else registry["rows"],
        "routes_with_consumer": (
            None
            if not registry
            else sum(1 for r in catalogue_routes if r["evidence"]["consumers"])
        ),
        "consumer_rows_addressed_here": (
            None if not registry else sum(1 for c in consumer_rows if c["destination"] == "this_router")
        ),
        "consumer_rows_addressed_elsewhere": (
            None
            if not registry
            else sum(1 for c in consumer_rows if c["destination"] not in ("this_router", "no_base_url"))
        ),
        "consumer_rows_without_base_url": (
            None if not registry else sum(1 for c in consumer_rows if c["destination"] == "no_base_url")
        ),
        "consumer_names_by_kind": (
            None if not registry else dict(sorted(collections.Counter(c["name_kind"] for c in consumer_rows).items()))
        ),
        "consumers_addressed_here_naming_nothing_here": (
            None
            if not registry
            else sorted(
                c["alias"]
                for c in consumer_rows
                if c["destination"] == "this_router" and not c["declared_here"]
            )
        ),
        "consumer_models_naming_an_upstream_id": (
            None
            if not registry
            else sorted({c["model"] for c in consumer_rows if c["name_kind"] == "upstream_model_id"})
        ),
        "consumer_models_undeclared": (
            None
            if not registry
            else sorted({c["model"] for c in consumer_rows if c["name_kind"] == "undeclared"})
        ),
        "capability_targets_undeclared": capability_targets_undeclared,
    }

    if capabilities:
        counts["capability_contract"] = {
            "contract_version": capabilities.get("contract_version"),
            "counts": capabilities.get("counts"),
            "families": len(capabilities.get("families") or {}),
        }

    catalogue = {
        "source": {
            "manifest_sha256": source_manifest_sha,
            "bytes_read_sha256": manifest_sha,
            "read_from": projection_source,
            "manifest_bytes": len(raw),
            "schema_version": document.get("schema_version")
            or (document.get("projection") or {}).get("schema_version"),
        },
        "counts": counts,
        "groups": {name: sorted(members) for name, members in sorted(declared_groups.items())},
        "routes": catalogue_routes,
        "targets": catalogue_targets,
        "target_reuse": {
            "shared": {
                name: sorted(entries, key=lambda e: (e["owner"], e["role"]))
                for name, entries in sorted(shared.items())
            },
            "single_use": single,
            "unreferenced": unreferenced,
            "partition": {
                "declared_targets": len(targets),
                "shared": len(shared),
                "single_use": len(single),
                "unreferenced": len(unreferenced),
                "sums_to_declared": len(shared) + len(single) + len(unreferenced) == len(targets),
                "disjoint": True,
            },
            "non_target_references": non_target_references,
        },
        "consumers": {
            "bindings": {} if not registry else registry["bindings"],
            "route_consumers": {
                name: rows for name, rows in sorted(consumer_by_route.items()) if name in routes
            },
            "composite_consumers": {
                name: rows for name, rows in sorted(consumer_by_route.items()) if name in composites
            },
            "lane_consumers": {
                name: rows
                for name, rows in sorted(consumer_by_route.items())
                if name in lanes or name in lane_composition
            },
            "capability_consumers": dict(sorted(capability_consumers.items())),
            "upstream_model_consumers": {
                name: rows
                for name, rows in sorted(consumer_by_route.items())
                if name not in name_index
            },
            "addressed_here": sorted(c["alias"] for c in consumer_rows if c["destination"] == "this_router"),
            "addressed_elsewhere": {
                host: sorted(c["alias"] for c in consumer_rows if c["destination"] == host)
                for host in sorted({c["destination"] for c in consumer_rows} - {"this_router"})
            },
            "rows": consumer_rows,
        },
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "catalogue.json").write_text(json.dumps(catalogue, indent=1) + "\n")
    (args.out_dir / "CATALOGUE.md").write_text(render_markdown(catalogue))
    print(json.dumps(counts, indent=1))
    return 0


def render_markdown(catalogue: dict) -> str:
    counts = catalogue["counts"]
    lines = [
        "# Deployed catalogue — P9 manifest",
        "",
        f"Deployed manifest SHA-256 `{catalogue['source']['manifest_sha256']}`.",
        "",
        f"This run read the {catalogue['source']['read_from']}: "
        f"`{catalogue['source']['bytes_read_sha256']}`, {catalogue['source']['manifest_bytes']} bytes, "
        f"schema {catalogue['source']['schema_version']}.",
        "",
        "Every route and target fact below is read from those bytes. `advertised` is the live",
        "`/v1/models` read at the same time. A `consumer` is a Turnstone registry row; it is",
        "credited to this manifest only when its `base_url` addresses this router and its model",
        "string is a name the manifest declares — a route, a composite, a decision lane, a lane",
        "composition key or a capability id. Rows that reach a provider directly are listed by",
        "destination and are not counted as traffic this router serves. A cell not present in a",
        "source is recorded as absent, never inferred from a sibling.",
        "",
        "## Counts",
        "",
        "| fact | value |",
        "|---|---|",
    ]
    for key, value in counts.items():
        if isinstance(value, dict):
            continue
        shown = value if not isinstance(value, list) else (", ".join(f"`{v}`" for v in value) or "none")
        lines.append(f"| {key} | {shown} |")
    contract = counts.get("capability_contract")
    if contract:
        lines += [
            f"| capability contract version | {contract['contract_version']} |",
            f"| capability families | {contract['families']} |",
        ]
    names_by_kind = counts.get("consumer_names_by_kind")
    if names_by_kind:
        lines.append("| consumer rows by name kind | " + ", ".join(f"{k}: {v}" for k, v in names_by_kind.items()) + " |")
    lines += [
        "",
        "## Target reuse",
        "",
        "Every name here is declared by a `[targets.*]` table. The three classes are a partition of",
        "those declarations, asserted at generation time: shared + single-use + unreferenced equals",
        "the declared target count, and no target is in two classes. A target behind more than one",
        "owner is a shared failure domain: an allowance, a credential or a provider incident moves",
        "every owner below at once.",
        "",
        f"Declared targets: **{counts['targets']}** = {counts['targets_shared']} shared + "
        f"{counts['targets_single_use']} single-use + {len(counts['targets_unreferenced'])} unreferenced.",
        "",
    ]
    shared = catalogue["target_reuse"]["shared"]
    if shared:
        lines += ["| target | owners |", "|---|---|"]
        for target, entries in shared.items():
            owners = sorted(entry["owner"] for entry in entries)
            lines.append(f"| `{target}` | {', '.join(f'`{o}`' for o in owners)} |")
    else:
        lines.append("No target is reached by more than one owner.")
    lines += [
        "",
        f"Single-use targets: {len(catalogue['target_reuse']['single_use'])}"
        + (
            " (" + ", ".join(f"`{t}`" for t in catalogue["target_reuse"]["single_use"]) + ")"
            if catalogue["target_reuse"]["single_use"]
            else ""
        )
        + "."
        + (
            " Reached only by a decision lane: "
            + ", ".join(f"`{t}`" for t in counts["targets_reached_by_a_lane_only"])
            + "."
            if counts["targets_reached_by_a_lane_only"]
            else ""
        ),
        "",
        "## References that are not targets",
        "",
        "A composite names routes, so its children are a call path rather than a failure domain.",
        "They are listed here, resolved, and never counted as targets:",
        "",
    ]
    non_target = counts["non_target_references"]
    if non_target:
        for kind, names in non_target.items():
            lines.append(f"- `{kind}`: " + ", ".join(f"`{n}`" for n in names))
    else:
        lines.append("- none")
    lines += [
        "",
        f"Capability-client models that are not `[targets.*]` entries: "
        f"{len(counts['capability_client_models_not_targets'])} of {counts['capability_clients']}"
        + (" — a separate namespace (a vendor model string, not a routing target), so the"
           " target partition does not include them.")
        + ("" if not counts["capability_client_models_not_targets"] else ""),
        "",
        "## Consumer destination",
        "",
        "A registry row is a consumer of this router only when its `base_url` points here. The",
        "table separates the rows addressed to this process from rows that name an upstream",
        "model and reach the provider directly; the second group is invisible to this manifest",
        "and is not counted as traffic it serves.",
        "",
    ]
    consumers = catalogue["consumers"]
    lines += ["| destination | aliases |", "|---|---|"]
    lines.append(f"| this router | {len(consumers['addressed_here'])} |")
    for host, aliases in consumers["addressed_elsewhere"].items():
        lines.append(f"| `{host}` | {len(aliases)} |")
    no_route = counts["consumers_addressed_here_naming_nothing_here"]
    lines += [
        "",
        "Rows addressed to this router whose model name is declared nowhere in the manifest: "
        + (", ".join(f"`{a}`" for a in no_route) if no_route else "none")
        + ".",
        "",
        "## Route → consumer ceiling",
        "",
        "The route's declared window is the router's admission bound. A consumer's ceiling is",
        "what Turnstone will accept before any request is sent. A positive delta means the",
        "consumer would admit a request the route refuses; a negative delta means the consumer",
        "is stricter than the router.",
        "",
        "| route | route ctx | consumers (alias: ceiling) | delta |",
        "|---|---|---|---|",
    ]
    for route in catalogue["routes"]:
        consumers = route["evidence"]["consumers"]
        if not consumers:
            continue
        shown = ", ".join(f"`{c['alias']}`: {c['context_window']}" for c in consumers)
        deltas = {c["ceiling_delta"] for c in consumers}
        delta = "n/a" if None in deltas else ", ".join(str(d) for d in sorted(deltas))
        lines.append(f"| `{route['route']}` | {route['context_window']} | {shown} | {delta} |")
    diverging = [
        (route["route"], consumer["alias"], consumer["ceiling_delta"])
        for route in catalogue["routes"]
        for consumer in route["evidence"]["consumers"]
        if consumer["ceiling_delta"] not in (0, None)
    ]
    stricter = sum(1 for _, _, delta in diverging if delta < 0)
    lines += [
        "",
        (
            f"{len(diverging)} route/consumer pair(s) disagree; {stricter} of them have the "
            "consumer stricter, so the request never reaches the router — the safe direction, "
            "but it means the route's advertised window is not what that alias can use."
            if diverging
            else "Every consumer ceiling equals its route's declared window."
        ),
    ]
    lines += [
        "",
        "## Routes without a consumer",
        "",
        "Advertised and callable, but no Turnstone alias names them. Not a defect — a route can",
        "be reached by another router or by a direct client — but it is the set to check before",
        "retiring anything.",
        "",
    ]
    without = [r["route"] for r in catalogue["routes"] if not r["evidence"]["consumers"]]
    lines.append(", ".join(f"`{r}`" for r in without) if without else "None.")
    lines += [
        "",
        "## Not established here",
        "",
        "- Vendor model cards, transport proofs and traffic observation are not part of this",
        "  catalogue; the master direct panel keeps those grades and this document makes no",
        "  claim about them.",
        "- A route's declared capability is an admission gate. It is not evidence the upstream",
        "  accepts the modality.",
        "",
    ]
    return "\n".join(lines)


def self_test() -> int:
    """Classify a fixture whose every cell has one right answer, then check the answers.

    The catalogue's value rests entirely on one question — is this registry row a consumer of
    these deployed bytes, and under which name kind — so that question is exercised against a
    fixture instead of only against production, where an error would look plausible.
    """
    import tempfile

    manifest = """
schema_version = 1

[routes.r1]
type = "direct"
target = "t1"
context_window = 1000

[composites.C]
children = ["r1"]

[decision_lanes.L]
targets = ["t2", "ghost"]

[decision_lane_composition.K]
children = ["L"]

[capabilities.cap1]
id = "cap1"
target = "cc1"

[capabilities.cap2]
id = "cap2"
target = "cc-missing"

[capability_clients.cc1]
format = "cohere_jina_rerank"
model = "m"

[targets.t1]
id = "up-t1"

[targets.t2]
id = "up-t2"

[targets.t3]
id = "up-t3"
"""
    routable = "https://switchyard-vm.example.ts.net/v1"
    registry = {
        "rows": 6,
        "bindings": {},
        "models": [
            {"alias": "A1", "model": "r1", "provider": "p", "base_url": routable, "context_window": 1000, "enabled": 1},
            {"alias": "A2", "model": "C", "provider": "p", "base_url": routable, "context_window": 1000, "enabled": 1},
            {"alias": "A3", "model": "cap1", "provider": "p", "base_url": routable + "/rerank", "context_window": None, "enabled": 1},
            {"alias": "A4", "model": "up-t1", "provider": "p", "base_url": "https://openrouter.ai/api/v1", "context_window": 1000, "enabled": 1},
            {"alias": "A5", "model": "r1", "provider": "p", "base_url": "https://openrouter.ai/api/v1", "context_window": 1000, "enabled": 1},
            {"alias": "A6", "model": "nowhere", "provider": "p", "base_url": routable, "context_window": 1000, "enabled": 1},
        ],
    }
    expected = {
        "consumer_names_by_kind": {"capability": 1, "composite": 1, "route": 2, "undeclared": 1, "upstream_model_id": 1},
        "consumer_rows_addressed_here": 4,
        "consumer_rows_addressed_elsewhere": 2,
        "consumers_addressed_here_naming_nothing_here": ["A6"],
        "consumer_models_naming_an_upstream_id": ["up-t1"],
        "targets_unreferenced": ["t3"],
        "targets_shared": 0,
        "targets_single_use": 2,
        "targets_partition_holds": True,
        "non_target_references": {"composite_child": ["r1"], "decision_lane": ["ghost"]},
        "targets_reached_by_a_lane_only": ["t2"],
        "capability_targets_undeclared": ["cap2"],
        "targets_reached_by_a_lane_only": ["t2"],
        "routes_with_consumer": 1,
    }
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        (root / "routes.toml").write_text(manifest)
        (root / "registry.json").write_text(json.dumps(registry))
        argv = [
            "canonical_catalogue.py", str(root / "routes.toml"),
            "--registry", str(root / "registry.json"),
            "--out-dir", str(root / "out"),
        ]
        saved, sys.argv = sys.argv, argv
        try:
            if main() != 0:
                print("SELFTEST FAIL: run returned non-zero", file=sys.stderr)
                return 1
        finally:
            sys.argv = saved
        counts = json.loads((root / "out" / "catalogue.json").read_text())["counts"]
    failures = {
        key: (expected.get(key), counts.get(key))
        for key in expected
        if counts.get(key) != expected.get(key)
    }
    if failures:
        for key, (want, got) in failures.items():
            print(f"SELFTEST FAIL {key}: expected {want!r}, got {got!r}", file=sys.stderr)
        return 1
    print(f"SELFTEST OK {len(expected)} assertions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
