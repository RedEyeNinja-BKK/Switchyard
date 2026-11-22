#!/usr/bin/env python3
"""Offline acceptance for Batch A of the Stephen ST routing work.

Reads a baseline manifest, the candidate the applier produced, the per-leaf truth
table and the captured provider catalogue subset. Every check is structural: no
provider is contacted and no credential is read.

The verifier fails closed. It refuses to report success when it examined nothing, and
it names each check that ran together with how many objects that check judged, so a
pass cannot hide a vacuous comparison.

Exit codes: 0 all checks passed; 1 at least one failed; 2 bad input; 3 nothing checked.
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib

FREE_SUFFIX = ":free"
DYNAMIC = "openrouter/free"
DYNAMIC_TARGET = "stephen-openrouter-free"
DYNAMIC_ROUTE = "direct-openrouter-stephen-openrouter-free"
ST_POLICY_CAP = 262144
STEPHEN_CLIENT = "openrouter_stephen"

ST_FREE_EXPECTED = [
    "stephen-ernie-5_5-flash-preview-free",
    "stephen-ling-3_2-flash-free",
    "stephen-nemotron-3-5-super-vl-free",
    "stephen-flint-v1-0-free",
    "stephen-dots-3-note-preview-free",
    "stephen-gemma-4-26b-a4b-it-free",
    "stephen-gemma-4-31b-it-free",
]


class Verdict:
    def __init__(self) -> None:
        self.rows: list[tuple[str, bool, int, str]] = []

    def check(self, name: str, ok: bool, inspected: int, detail: str = "") -> None:
        self.rows.append((name, ok, inspected, detail))

    def report(self) -> int:
        width = max(len(name) for name, *_ in self.rows) if self.rows else 0
        for name, ok, inspected, detail in self.rows:
            flag = "PASS" if ok else "FAIL"
            suffix = f"  {detail}" if detail else ""
            print(f"{flag}  {name:<{width}}  n={inspected:<3}{suffix}")
        failed = [row for row in self.rows if not row[1]]
        total = sum(row[2] for row in self.rows)
        print(f"\n{len(self.rows)} checks, {total} object judgements, {len(failed)} failed")
        if not self.rows or total == 0:
            print("REFUSED: the verifier judged nothing", file=sys.stderr)
            return 3
        return 1 if failed else 0


def free_scope(model: str) -> bool:
    return model.endswith(FREE_SUFFIX) or model == DYNAMIC


def routes_of(config: dict) -> dict:
    return config["routes"]


def targets_of(config: dict) -> dict:
    return config["targets"]


def route_models(config: dict, route: dict) -> list[str | None]:
    """Upstream ids for a route's legs, `None` where the manifest names a missing one.

    A missing leg is a finding for A18 to report, not an exception for this helper to
    raise: a verifier that dies while judging is indistinguishable from one that judged
    and failed, and the difference decides whether the defect was seen at all.
    """
    names = [route["target"]] + list(route.get("candidates") or [])
    return [config["targets"].get(name, {}).get("id") for name in names]


def reference_edges(config: dict) -> list[tuple[str, str, str]]:
    """(kind, owner, referenced name) for every intra-manifest reference."""
    edges = []
    for name, route in config["routes"].items():
        if route.get("target"):
            edges.append(("route.target", name, route["target"]))
        for key in ("capable_target", "efficient_target"):
            if route.get(key):
                edges.append((f"route.{key}", name, route[key]))
        for candidate in route.get("candidates") or []:
            edges.append(("route.candidate", name, candidate))
    for name, composite in config["composites"].items():
        for child in composite.get("children") or []:
            edges.append(("composite.child", name, child))
    for name, lane in config["decision_lanes"].items():
        for target in lane.get("targets") or []:
            edges.append(("lane.target", name, target))
    for name, capability in config["capabilities"].items():
        if capability.get("target"):
            edges.append(("capability.target", name, capability["target"]))
        for target in capability.get("targets") or []:
            edges.append(("capability.targets", name, target))
    return edges


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--facts", required=True)
    args = ap.parse_args(argv)

    try:
        baseline = tomllib.load(open(args.baseline, "rb"))
        candidate = tomllib.load(open(args.candidate, "rb"))
        truth = json.load(open(args.truth))
        facts = json.load(open(args.facts))["models"]
    except (OSError, tomllib.TOMLDecodeError, json.JSONDecodeError) as error:
        print(f"REFUSED: cannot read inputs: {error}", file=sys.stderr)
        return 2

    v = Verdict()
    bt, ct = targets_of(baseline), targets_of(candidate)
    br, cr = routes_of(baseline), routes_of(candidate)

    # A1 — the only objects removed are the dynamic selector and its leaf.
    removed_targets = set(bt) - set(ct)
    removed_routes = set(br) - set(cr)
    v.check(
        "A1 removed objects are exactly the dynamic selector",
        removed_targets == {DYNAMIC_TARGET} and removed_routes == {DYNAMIC_ROUTE},
        len(removed_targets) + len(removed_routes),
        f"targets={sorted(removed_targets)} routes={sorted(removed_routes)}",
    )
    v.check(
        "A2 nothing added",
        (set(ct) - set(bt)) == set() and (set(cr) - set(br)) == set(),
        len(ct) + len(cr),
        f"targets={len(bt)}->{len(ct)} routes={len(br)}->{len(cr)}",
    )

    # A3 — every changed object is inside the Stephen scope.
    changed_targets = {
        name for name in bt if name in ct and bt[name] != ct[name]
    }
    changed_routes = {name for name in br if name in cr and br[name] != cr[name]}
    expected_changed_targets = {
        name for name, cfg in ct.items() if "stephen" in name and cfg.get("free_tier")
    }
    v.check(
        "A3 changed targets are exactly the declared free leaves",
        changed_targets == expected_changed_targets,
        len(changed_targets),
        ",".join(sorted(changed_targets)),
    )
    expected_changed_routes = {
        "ST_Free",
        "direct-openrouter-stephen-ernie-5_5-flash-preview-free",
        "direct-openrouter-stephen-ling-3_2-flash-free",
        "direct-openrouter-stephen-nemotron-3-5-super-vl-free",
    }
    v.check(
        "A4 changed routes are exactly ST_Free and the three context corrections",
        changed_routes == expected_changed_routes,
        len(changed_routes),
        ",".join(sorted(changed_routes)),
    )

    # A5 — owner surfaces are byte-identical.
    owner_sections = ["composites", "capabilities", "decision_lanes", "quota",
                      "decision_lane_composition", "capability_clients", "llm_clients"]
    same_sections = [key for key in owner_sections if baseline[key] == candidate[key]]
    v.check(
        "A5 owner sections unchanged",
        len(same_sections) == len(owner_sections),
        len(owner_sections),
        f"identical={len(same_sections)}/{len(owner_sections)}",
    )
    owner_targets = [
        name for name in bt if "stephen" not in name and name in ct and bt[name] == ct[name]
    ]
    v.check(
        "A6 owner targets unchanged",
        len(owner_targets) == len([name for name in bt if "stephen" not in name]),
        len(owner_targets),
    )
    owner_routes = [
        name
        for name in br
        if "stephen" not in name and name != "ST_Free" and name in cr and br[name] == cr[name]
    ]
    v.check(
        "A7 owner routes unchanged",
        len(owner_routes)
        == len([name for name in br if "stephen" not in name and name != "ST_Free"]),
        len(owner_routes),
    )

    # A8 — ST_Free is the explicit free pool, in the declared order, with no router.
    st_free = cr["ST_Free"]
    members = [st_free["target"]] + list(st_free.get("candidates") or [])
    v.check(
        "A8 ST_Free candidate list is exactly the seven explicit free leaves",
        members == ST_FREE_EXPECTED,
        len(members),
        ",".join(members),
    )
    free_models = route_models(candidate, st_free)
    v.check(
        "A9 every ST_Free leg is free-scope and none is the dynamic router",
        all(model is not None and free_scope(model) for model in free_models)
        and DYNAMIC not in free_models,
        len(free_models),
    )

    # A10 — ST_Paid stays metered.
    st_paid = cr["ST_Paid"]
    paid_models = route_models(candidate, st_paid)
    v.check(
        "A10 ST_Paid carries only metered slugs",
        all(model is not None and not free_scope(model) for model in paid_models),
        len(paid_models),
        ",".join(paid_models),
    )

    # A11 — one credential, named once.
    stephen_targets = {name: cfg for name, cfg in ct.items() if "stephen" in name}
    wrong_client = [
        name for name, cfg in stephen_targets.items() if cfg.get("llm_client") != STEPHEN_CLIENT
    ]
    v.check(
        "A11 every Stephen target resolves to the Stephen client",
        not wrong_client,
        len(stephen_targets),
        f"exceptions={wrong_client}",
    )
    client = candidate["llm_clients"][STEPHEN_CLIENT]
    v.check(
        "A12 the Stephen client names one credential environment variable",
        client.get("api_key_env") == truth["policy"]["credential_env_name"],
        1,
        client.get("api_key_env", ""),
    )

    # A13 — direct leaves are deterministic single-target routes with unique upstreams.
    stephen_routes = {name: cfg for name, cfg in cr.items() if "stephen" in name}
    multi = [
        name
        for name, cfg in stephen_routes.items()
        if cfg.get("candidates") or cfg.get("type") != "passthrough"
    ]
    v.check(
        "A13 every Stephen direct leaf is one passthrough target and zero candidates",
        not multi,
        len(stephen_routes),
        f"exceptions={multi}",
    )
    upstream_ids = [ct[cfg["target"]]["id"] for cfg in stephen_routes.values()]
    v.check(
        "A14 upstream ids are unique across Stephen direct leaves",
        len(upstream_ids) == len(set(upstream_ids)),
        len(upstream_ids),
        f"unique={len(set(upstream_ids))}",
    )

    # A15 — context truth comes from the provider, not from the parent policy cap.
    context_rows = 0
    context_ok = True
    detail = []
    for leaf in truth["leaves"]:
        if leaf["disposition"] == "REMOVED":
            continue
        route = stephen_routes[leaf["route"]]
        provider = (facts.get(leaf["upstream_id"]) or {}).get("context_length")
        declared = route.get("context_window")
        context_rows += 1
        if provider is not None and declared != provider:
            context_ok = False
            detail.append(f"{leaf['route']}={declared}(provider {provider})")
        if provider is not None and provider > ST_POLICY_CAP and declared == ST_POLICY_CAP:
            context_ok = False
            detail.append(f"{leaf['route']} inherited the parent policy cap")
    v.check(
        "A15 direct-leaf context equals the provider window, never the parent cap",
        context_ok,
        context_rows,
        "; ".join(detail),
    )

    # A16 — free means provider-priced at zero; paid means priced above zero.
    price_rows = 0
    price_ok = True
    price_detail = []
    for leaf in truth["leaves"]:
        if leaf["disposition"] == "REMOVED":
            continue
        row = facts.get(leaf["upstream_id"])
        if not row or not row.get("listed"):
            continue
        price_rows += 1
        prompt = float(row["pricing_per_token"]["prompt"])
        completion = float(row["pricing_per_token"]["completion"])
        if leaf["free"] and not (prompt == 0 and completion == 0):
            price_ok = False
            price_detail.append(f"{leaf['upstream_id']} is declared free but priced")
        if not leaf["free"] and not (prompt > 0 or completion > 0):
            price_ok = False
            price_detail.append(f"{leaf['upstream_id']} is declared paid but free")
    v.check(
        "A16 declared tariff matches the provider price",
        price_ok,
        price_rows,
        "; ".join(price_detail),
    )

    # A17 — free_tier declarations follow the tariff.
    declared_free = {
        name for name, cfg in stephen_targets.items() if cfg.get("free_tier")
    }
    truth_free = {
        leaf["target"]
        for leaf in truth["leaves"]
        if leaf["free"] and leaf["disposition"] != "REMOVED"
    }
    v.check(
        "A17 free_tier is declared on exactly the free leaves",
        declared_free == truth_free,
        len(stephen_targets),
        f"declared={len(declared_free)} truth={len(truth_free)}",
    )

    # A18 — no reference survives to the removed objects.
    edges = reference_edges(candidate)
    resolvable = set(ct) | set(cr) | set(candidate["capability_clients"])
    dangling = [edge for edge in edges if edge[2] not in resolvable]
    baseline_edges = reference_edges(baseline)
    baseline_resolvable = (
        set(bt) | set(br) | set(baseline["capability_clients"])
    )
    baseline_dangling = [
        edge for edge in baseline_edges if edge[2] not in baseline_resolvable
    ]
    v.check(
        "A18 no dangling reference, and no new one introduced",
        not dangling and len(dangling) <= len(baseline_dangling),
        len(edges),
        f"candidate={dangling[:4]} baseline={len(baseline_dangling)}",
    )
    text = open(args.candidate).read()
    v.check(
        "A19 the removed dynamic target is named nowhere in the candidate",
        text.count(DYNAMIC_TARGET) == 0 and text.count(DYNAMIC_ROUTE) == 0,
        2,
        f"occurrences={text.count(DYNAMIC_TARGET) + text.count(DYNAMIC_ROUTE)}",
    )

    # A20 — no composite child is a composite.
    composites = set(candidate["composites"])
    nested = [
        (parent, child)
        for parent, cfg in candidate["composites"].items()
        for child in cfg.get("children") or []
        if child in composites
    ]
    v.check(
        "A20 no nested composite",
        not nested,
        sum(len(cfg.get("children") or []) for cfg in candidate["composites"].values()),
        f"nested={nested}",
    )

    # A21 — ST_Route keeps its consumer contract and refuses paid crossing on 429/5xx.
    st_route = candidate["composites"]["ST_Route"]
    v.check(
        "A21 ST_Route keeps children, credential-only crossing and the 262144 cap",
        st_route["children"] == ["ST_Free", "ST_Paid"]
        and st_route["cross_on"] == "credential_block"
        and st_route["context_window"] == ST_POLICY_CAP,
        3,
        f"children={st_route['children']} cross_on={st_route['cross_on']} ctx={st_route['context_window']}",
    )

    # A22 — typed surfaces keep their identity.
    embeds = candidate["capabilities"]["ST_Embed"]
    rerank = candidate["capabilities"]["ST_Rerank"]
    decision_lane = candidate["decision_lanes"]["ST_Decisions"]
    v.check(
        "A22 typed surfaces unchanged (4096-d embed, Nemotron rerank, plural decisions)",
        embeds == baseline["capabilities"]["ST_Embed"]
        and rerank == baseline["capabilities"]["ST_Rerank"]
        and decision_lane == baseline["decision_lanes"]["ST_Decisions"]
        and embeds["dimensions"] == 4096
        and decision_lane["targets"] == ["direct-openrouter-stephen-span-01-lite"],
        3,
        f"dims={embeds['dimensions']} lane_targets={decision_lane['targets']}",
    )

    # A23 — the truth table covers every Stephen object, and its numbers match the config.
    truth_routes = {leaf["route"] for leaf in truth["leaves"]}
    config_stephen_routes = set(stephen_routes) | {DYNAMIC_ROUTE}
    v.check(
        "A23 every Stephen route appears in the truth table",
        truth_routes == config_stephen_routes,
        len(config_stephen_routes),
        f"missing={sorted(config_stephen_routes - truth_routes)} extra={sorted(truth_routes - config_stephen_routes)}",
    )
    drift = []
    for leaf in truth["leaves"]:
        if leaf["disposition"] == "REMOVED":
            continue
        if stephen_routes[leaf["route"]]["context_window"] != leaf["declared_route_context"]:
            drift.append(leaf["route"])
        provider = (facts.get(leaf["upstream_id"]) or {}).get("context_length")
        if provider is not None and leaf["provider_context"] != provider:
            drift.append(f"{leaf['route']}:provider")
    v.check(
        "A24 truth table matches both the candidate and the captured provider facts",
        not drift,
        len(truth["leaves"]),
        f"drift={drift}",
    )

    # A25 — the removed selector was safe to remove: nothing named it in 30 days.
    v.check(
        "A25 removal has explicit zero-reference evidence",
        truth["leaves"][-1]["disposition"] == "REMOVED"
        and "0 requests" in truth["leaves"][-1]["changed_in_batch_a"],
        1,
        truth["leaves"][-1]["changed_in_batch_a"],
    )

    return v.report()


if __name__ == "__main__":
    raise SystemExit(main())
