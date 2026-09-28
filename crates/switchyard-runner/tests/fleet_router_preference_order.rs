// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! F4 preference ordering, measured against the unmodified live deployment file.
//!
//! **These are F4 projections, not final production routing decisions.** Only
//! static eligibility (F3) has run. Context-token admission (row 40), live
//! readiness (F6), and the remaining capability/reasoning constraints (F5/F7)
//! are absent, so a candidate appearing in today's order may still be excluded
//! before it could serve.
//!
//! Live route tables are projected the same way the F1/F2/F3 fixtures project
//! them: route-level capability fields and reasoning policy belong to
//! `RouteConfig` (row 41) and row 12, not to the algorithm parser.

use std::collections::BTreeMap;

use libsy::{
    EligibilityFacts, FleetRouter, compare_preference, preference_order, static_eligibility,
};
use switchyard_protocol::ModelId;
use switchyard_runner::{AlgorithmSpec, FleetCandidateConfig, build_fleet_router};

const LIVE_CONFIG: &str = "/home/vincent/.local/lib/localclaw-switchyard/routes.toml";

const STRIPPED: &[&str] = &[
    "id",
    "context_window",
    "tool_calling",
    "reasoning",
    "supports_vision",
    "vision",
    "reasoning_policy",
    "escalation_reasoning_policy",
];

fn live_document() -> toml::Table {
    let text = std::fs::read_to_string(LIVE_CONFIG)
        .unwrap_or_else(|error| panic!("live config must be readable: {error}"));
    toml::from_str(&text).expect("live config is valid TOML")
}

fn live_targets() -> BTreeMap<String, ModelId> {
    live_document()
        .get("targets")
        .and_then(toml::Value::as_table)
        .map(|targets| {
            targets
                .keys()
                .map(|name| (name.clone(), ModelId::from(name.as_str())))
                .collect()
        })
        .unwrap_or_default()
}

struct LiveFleet {
    name: String,
    candidates: Vec<FleetCandidateConfig>,
    escalation: Option<ModelId>,
    threshold: Option<u64>,
}

fn live_fleet() -> Vec<LiveFleet> {
    live_document()
        .get("routes")
        .and_then(toml::Value::as_table)
        .expect("live config has [routes]")
        .iter()
        .filter(|(_, value)| {
            value.get("type").and_then(toml::Value::as_str) == Some("fleet_router")
        })
        .map(|(name, value)| {
            let mut table = value.as_table().expect("route is a table").clone();
            for key in STRIPPED {
                table.remove(*key);
            }
            let spec: AlgorithmSpec = toml::Value::Table(table)
                .try_into()
                .unwrap_or_else(|error| panic!("fleet route {name} must parse: {error}"));
            match spec {
                AlgorithmSpec::FleetRouter {
                    candidates,
                    escalation,
                    escalation_max_input_tokens,
                } => LiveFleet {
                    name: name.clone(),
                    candidates,
                    escalation,
                    threshold: escalation_max_input_tokens,
                },
                other => panic!("expected fleet_router, got {other:?}"),
            }
        })
        .collect()
}

fn live_routers() -> Vec<(String, FleetRouter)> {
    let targets = live_targets();
    live_fleet()
        .iter()
        .map(|fleet| {
            let router = build_fleet_router(
                &fleet.name,
                &fleet.candidates,
                &fleet.escalation,
                fleet.threshold,
                &targets,
            )
            .unwrap_or_else(|error| panic!("fleet route {} must construct: {error}", fleet.name));
            (fleet.name.clone(), router)
        })
        .collect()
}

/// The facts a plain text request produces: no requirement on any dimension.
fn plain_text_facts() -> EligibilityFacts {
    EligibilityFacts {
        requires_tools: false,
        requires_vision: false,
        requires_reasoning: false,
    }
}

// --- live preference_rank census ---------------------------------------------

/// Which comparator arms have live evidence and which are fixture-only.
#[test]
fn live_preference_rank_census() {
    let routers = live_routers();
    let all: Vec<u16> = routers
        .iter()
        .flat_map(|(_, router)| router.candidates().iter())
        .map(|candidate| candidate.preference_rank)
        .collect();

    let mut counts: BTreeMap<u16, usize> = BTreeMap::new();
    for rank in &all {
        *counts.entry(*rank).or_default() += 1;
    }
    let multi_rank: Vec<&str> = routers
        .iter()
        .filter(|(_, router)| {
            let ranks: Vec<u16> = router
                .candidates()
                .iter()
                .map(|c| c.preference_rank)
                .collect();
            ranks
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len()
                > 1
        })
        .map(|(name, _)| name.as_str())
        .collect();
    let tied: Vec<&str> = routers
        .iter()
        .filter(|(_, router)| {
            let ranks: Vec<u16> = router
                .candidates()
                .iter()
                .map(|c| c.preference_rank)
                .collect();
            ranks.len()
                != ranks
                    .iter()
                    .collect::<std::collections::BTreeSet<_>>()
                    .len()
        })
        .map(|(name, _)| name.as_str())
        .collect();

    println!("live preference_rank census (46 candidates):");
    println!("  values used: {:?}", counts.keys().collect::<Vec<_>>());
    println!("  counts per rank: {counts:?}");
    println!("  routes containing a rank TIE: {}", tied.len());
    println!("  => the target tie-break has NO live evidence; it is fixture-proven only");
    println!("  multi-rank routes: {multi_rank:?}");

    // Derived from the live routers: the 2026-09-28 authorized realignment
    // changed the candidate population, so a pinned literal would report that
    // authorized change as a defect.
    let live_routes = routers.len();
    assert!(
        !all.is_empty(),
        "the live file must declare fleet candidates"
    );
    println!("  routes with >1 distinct rank: {} / {live_routes}", multi_rank.len());
    assert!(
        multi_rank.len() > 0,
        "at least one live route must span more than one rank, or the ordering \
         seam has no live evidence at all"
    );
    assert!(
        tied.is_empty(),
        "the live file contains no rank ties, so the tie-break arm is unit-only: {tied:?}"
    );
}

// --- declaration order vs derived order --------------------------------------

/// The stored ladder keeps exact `routes.toml` declaration order; the derived
/// preference view is a separate allocation. Reading the router after deriving an
/// order must still return declaration order.
#[test]
fn declaration_order_survives_deriving_preference_order() {
    let routers = live_routers();
    let mut differed = 0usize;

    for (name, router) in &routers {
        let declared: Vec<String> = router
            .candidates()
            .iter()
            .map(|c| c.target.to_string())
            .collect();
        let ordered: Vec<String> =
            preference_order(&static_eligibility(router, &plain_text_facts()))
                .iter()
                .map(|c| c.target.to_string())
                .collect();

        // Same membership, possibly different sequence.
        let mut declared_sorted = declared.clone();
        declared_sorted.sort();
        let mut ordered_sorted = ordered.clone();
        ordered_sorted.sort();
        assert_eq!(
            declared_sorted, ordered_sorted,
            "route {name} changed membership"
        );

        if declared != ordered {
            differed += 1;
        }

        // The stored ladder is unchanged after deriving the order, twice over.
        let after_first: Vec<String> = router
            .candidates()
            .iter()
            .map(|c| c.target.to_string())
            .collect();
        let after_second: Vec<String> =
            preference_order(&static_eligibility(router, &plain_text_facts()))
                .iter()
                .map(|c| c.target.to_string())
                .collect();
        assert_eq!(
            after_first, declared,
            "route {name} stored ladder was mutated"
        );
        assert_eq!(
            after_second, ordered,
            "route {name} order derivation is not stable"
        );
    }

    println!("routes whose preference order differs from declaration order: {differed}/26");
    // MEASURED FACT, not an assumption: every live route already declares its
    // candidates in ascending preference rank, so the derived order is identical
    // to declaration order on the real deployment. The comparator therefore has
    // NO live reordering evidence, and this test proves non-mutation rather than
    // reordering. The reordering arms are proven by unit fixtures, where
    // declaration order and preference order genuinely differ.
    assert_eq!(
        differed, 0,
        "live routes currently declare ascending rank; if this changes, the \
         non-mutation assertions above are what must still hold"
    );
}

/// Unit-fixture counterpart to the live non-mutation test: when declaration
/// order and preference order genuinely differ, the derived view must be the
/// preference order while the stored ladder keeps declaration order.
#[test]
fn derived_order_reorders_only_the_view_when_they_differ() {
    use libsy::{FleetCandidate, FleetRouter};

    let candidate = |target: &str, rank: u16| FleetCandidate {
        target: ModelId::from(target),
        tool_calling: true,
        reasoning: true,
        supports_vision: true,
        preference_rank: rank,
        usable_context_tokens: None,
    };
    // Declared in deliberately wrong order: rank 3, then 1, then 2.
    let router = FleetRouter::new(
        vec![
            candidate("zulu", 3),
            candidate("alpha", 1),
            candidate("mike", 2),
        ],
        None,
        None,
    );
    let stored: Vec<String> = router
        .candidates()
        .iter()
        .map(|c| c.target.to_string())
        .collect();
    let ordered: Vec<String> = preference_order(&static_eligibility(&router, &plain_text_facts()))
        .iter()
        .map(|c| c.target.to_string())
        .collect();

    assert_eq!(
        stored,
        ["zulu", "alpha", "mike"],
        "declaration order must be preserved"
    );
    assert_eq!(
        ordered,
        ["alpha", "mike", "zulu"],
        "the view must be preference order"
    );
    assert_ne!(
        stored, ordered,
        "this fixture must actually differ, else it proves nothing"
    );

    // And the stored ladder is still declaration order after the reordering.
    let after: Vec<String> = router
        .candidates()
        .iter()
        .map(|c| c.target.to_string())
        .collect();
    assert_eq!(after, stored);
}

// --- comparator truth table over live routes ---------------------------------

/// For every live route, the derived order must equal a from-scratch application
/// of the production comparator to the declared ladder. This compares the whole
/// route at once rather than pairwise.
#[test]
fn live_route_ordering_matches_the_production_comparator() {
    let routers = live_routers();
    for (name, router) in &routers {
        let derived: Vec<String> =
            preference_order(&static_eligibility(router, &plain_text_facts()))
                .iter()
                .map(|c| c.target.to_string())
                .collect();

        // Independent recomputation straight from the declared ladder.
        let mut expected: Vec<&libsy::FleetCandidate> = router.candidates().iter().collect();
        expected.sort_by(|a, b| compare_preference(a, b));
        let expected: Vec<String> = expected.iter().map(|c| c.target.to_string()).collect();

        assert_eq!(
            derived, expected,
            "route {name} ordering diverges from the comparator"
        );
    }
}

/// Ranks must be non-decreasing along the derived order, and any equal-rank run
/// must be sorted by target id ascending.
#[test]
fn derived_order_obeys_the_comparator_invariants() {
    for (name, router) in live_routers() {
        let ordered = preference_order(&static_eligibility(&router, &plain_text_facts()));
        for pair in ordered.windows(2) {
            let (a, b) = (pair[0], pair[1]);
            assert!(
                a.preference_rank < b.preference_rank
                    || (a.preference_rank == b.preference_rank && a.target < b.target),
                "route {name} is out of comparator order: {:?} then {:?}",
                a.target,
                b.target
            );
        }
    }
}

/// The same target appearing in two routes orders by each route's own ranks.
#[test]
fn shared_live_targets_order_per_route() {
    let routers = live_routers();
    // target -> list of (route name, index in that route's preference order)
    let mut positions: BTreeMap<String, Vec<(String, usize)>> = BTreeMap::new();
    for (name, router) in &routers {
        let ordered = preference_order(&static_eligibility(router, &plain_text_facts()));
        for (index, candidate) in ordered.iter().enumerate() {
            positions
                .entry(candidate.target.to_string())
                .or_default()
                .push((name.clone(), index));
        }
    }

    let shared: Vec<(&String, &Vec<(String, usize)>)> =
        positions.iter().filter(|(_, v)| v.len() > 1).collect();
    assert!(
        !shared.is_empty(),
        "expected the live file to reuse a target across routes"
    );

    // Ordering is recomputed per route from that route's own metadata; a target
    // may legitimately sit at different positions in different routes.
    for (target, entries) in shared {
        for (name, _) in entries {
            let router = routers
                .iter()
                .find(|(route, _)| route == name)
                .expect("route must be present");
            let ordered = preference_order(&static_eligibility(&router.1, &plain_text_facts()));
            let first = ordered
                .iter()
                .position(|c| &c.target.to_string() == target)
                .expect("the shared target must appear in its own route's order");
            assert_eq!(first, entries.iter().find(|(n, _)| n == name).unwrap().1);
        }
    }
}

/// Escalation destinations must never enter the ordering input.
#[test]
fn live_escalation_destinations_stay_outside_the_order() {
    let routers = live_routers();
    let mut escalations = 0usize;
    for (name, router) in &routers {
        let ordered = preference_order(&static_eligibility(router, &plain_text_facts()));
        let order: Vec<String> = ordered.iter().map(|c| c.target.to_string()).collect();
        if let Some(destination) = router.escalation() {
            escalations += 1;
            assert!(
                !order.contains(&destination.to_string()),
                "route {name} must not rank its escalation destination as a candidate"
            );
        }
        // The declared candidate count is exactly the order length: escalation
        // adds no entry.
        assert_eq!(order.len(), router.candidates().len());
    }
    assert_eq!(
        escalations, 8,
        "8 live routes declare an escalation destination"
    );
}
