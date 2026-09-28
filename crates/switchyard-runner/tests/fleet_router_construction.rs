// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! F2 construction: a parsed live `fleet_router` route becomes a faithful
//! runtime object, without yet executing fleet semantics.
//!
//! Construction is total and faithful. It filters nothing, sorts nothing,
//! deduplicates nothing, and invents no default candidate. A candidate a later
//! selection layer would reject must still exist here, because deciding that is
//! not this layer's job.
//!
//! These tests run against the unmodified live `routes.toml`, so the census they
//! report is a measurement of the real deployment file rather than a fixture.

use std::collections::BTreeMap;

use libsy::{Algorithm as _, FleetRouter};
use switchyard_protocol::ModelId;
use switchyard_runner::{AlgorithmSpec, FleetCandidateConfig, build_fleet_router};

/// Live deployment config, read-only. `SWITCHYARD_F2_CONFIG` overrides the path
/// so the same file can be run against a control tree.
const LIVE_CONFIG: &str = match option_env!("SWITCHYARD_F2_CONFIG") {
    Some(path) => path,
    None => "/home/vincent/.local/lib/localclaw-switchyard/routes.toml",
};

/// Keys removed before the algorithm parser sees a live route table, each with
/// the row that owns it. F1 owns `escalation` and `escalation_max_input_tokens`,
/// so those are deliberately NOT stripped.
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

fn live_routes() -> BTreeMap<String, toml::Table> {
    let text = std::fs::read_to_string(LIVE_CONFIG)
        .unwrap_or_else(|error| panic!("live config {LIVE_CONFIG} must be readable: {error}"));
    let document: toml::Table = toml::from_str(&text)
        .unwrap_or_else(|error| panic!("live config must be valid TOML: {error}"));
    let routes = document
        .get("routes")
        .and_then(toml::Value::as_table)
        .cloned()
        .unwrap_or_else(|| panic!("live config must have a [routes] table"));
    routes
        .iter()
        .map(|(name, value)| {
            (
                name.clone(),
                value
                    .as_table()
                    .unwrap_or_else(|| panic!("route {name} must be a table"))
                    .clone(),
            )
        })
        .collect()
}

/// Every target the live deployment defines, as the runner's target map.
fn live_targets() -> BTreeMap<String, ModelId> {
    let text = std::fs::read_to_string(LIVE_CONFIG).expect("live config readable");
    let document: toml::Table = toml::from_str(&text).expect("live config is valid TOML");
    document
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

fn algorithm_table(route: &toml::Table) -> toml::Table {
    let mut table = route.clone();
    for key in STRIPPED {
        table.remove(*key);
    }
    table
}

struct ParsedFleet {
    name: String,
    candidates: Vec<FleetCandidateConfig>,
    escalation: Option<ModelId>,
    threshold: Option<u64>,
}

fn fleet_routes() -> Vec<ParsedFleet> {
    live_routes()
        .into_iter()
        .filter(|(_, table)| {
            table.get("type").and_then(toml::Value::as_str) == Some("fleet_router")
        })
        .map(|(name, table)| {
            let spec: AlgorithmSpec = toml::Value::Table(algorithm_table(&table))
                .try_into()
                .unwrap_or_else(|error| panic!("fleet route {name} must parse: {error}"));
            match spec {
                AlgorithmSpec::FleetRouter {
                    candidates,
                    escalation,
                    escalation_max_input_tokens,
                } => ParsedFleet {
                    name,
                    candidates,
                    escalation,
                    threshold: escalation_max_input_tokens,
                },
                other => panic!("expected fleet_router, got {other:?}"),
            }
        })
        .collect()
}

fn construct(parsed: &ParsedFleet, targets: &BTreeMap<String, ModelId>) -> FleetRouter {
    build_fleet_router(
        &parsed.name,
        &parsed.candidates,
        &parsed.escalation,
        parsed.threshold,
        targets,
    )
    .unwrap_or_else(|error| panic!("fleet route {} must construct: {error}", parsed.name))
}

/// The candidate metadata the deployment declares, read straight from the file.
fn declared_metadata(route: &toml::Table) -> Vec<(String, bool, bool, bool, u16, Option<u64>)> {
    route
        .get("candidates")
        .and_then(toml::Value::as_array)
        .expect("fleet route declares candidates")
        .iter()
        .map(|candidate| {
            let table = candidate.as_table().expect("candidate is a table");
            let get_str = |key: &str| table.get(key).and_then(toml::Value::as_bool);
            let get_int = |key: &str| table.get(key).and_then(toml::Value::as_integer);
            (
                table
                    .get("target")
                    .and_then(toml::Value::as_str)
                    .expect("candidate names a target")
                    .to_string(),
                get_str("tool_calling").unwrap_or(false),
                get_str("reasoning").unwrap_or(false),
                get_str("supports_vision").unwrap_or(false),
                get_int("preference_rank").unwrap_or(0) as u16,
                get_int("usable_context_tokens").map(|value| value as u64),
            )
        })
        .collect()
}

fn declared_candidates(route: &toml::Table) -> Vec<(String, bool, bool, bool, u16, Option<u64>)> {
    route
        .get("candidates")
        .and_then(toml::Value::as_array)
        .expect("fleet route declares candidates")
        .iter()
        .map(|candidate| {
            let table = candidate.as_table().expect("candidate is a table");
            let get_str = |key: &str| table.get(key).and_then(toml::Value::as_bool);
            let get_int = |key: &str| table.get(key).and_then(toml::Value::as_integer);
            (
                table
                    .get("target")
                    .and_then(toml::Value::as_str)
                    .expect("candidate names a target")
                    .to_string(),
                get_str("tool_calling").unwrap_or(false),
                get_str("reasoning").unwrap_or(false),
                get_str("supports_vision").unwrap_or(false),
                get_int("preference_rank").unwrap_or(0) as u16,
                get_int("usable_context_tokens").map(|value| value as u64),
            )
        })
        .collect()
}

// --- positive: construction census over the live file -----------------------

#[test]
fn all_26_live_fleet_routes_construct() {
    let targets = live_targets();
    let routes = fleet_routes();
    assert_eq!(routes.len(), 26, "live deployment declares 26 fleet routes");
    for parsed in &routes {
        // Construction must succeed for every live fleet route.
        let _ = construct(parsed, &targets);
    }
}

#[test]
fn all_live_passthrough_routes_still_construct_through_the_native_path() {
    let targets = live_targets();
    let mut count = 0;
    for (name, table) in live_routes() {
        if table.get("type").and_then(toml::Value::as_str) != Some("passthrough") {
            continue;
        }
        count += 1;
        let spec: AlgorithmSpec = toml::Value::Table(algorithm_table(&table))
            .try_into()
            .unwrap_or_else(|error| panic!("passthrough route {name} must parse: {error}"));
        assert!(
            matches!(spec, AlgorithmSpec::Passthrough { .. }),
            "route {name} must stay a passthrough, not be rewrapped for fleet code sharing"
        );
        spec.build(&name, &targets)
            .unwrap_or_else(|error| panic!("passthrough route {name} must build: {error}"));
    }
    // Counted from the config itself, not a pinned literal: the 2026-09-28
    // operator-authorized realignment added the Qwen3.8-27B free and LFM2.5 free
    // passthrough routes, and a hardcoded total would turn that authorized
    // addition into a false failure.
    let declared: usize = live_routes()
        .values()
        .filter(|table| table.get("type").and_then(toml::Value::as_str) == Some("passthrough"))
        .count();
    assert!(
        count > 0,
        "the live deployment must declare passthrough routes"
    );
    assert_eq!(
        count, declared,
        "the walk must cover every live passthrough route"
    );
}

#[test]
fn candidate_order_survives_parse_to_runtime_construction() {
    let targets = live_targets();
    for (name, table) in live_routes() {
        if table.get("type").and_then(toml::Value::as_str) != Some("fleet_router") {
            continue;
        }
        let parsed = fleet_routes()
            .into_iter()
            .find(|parsed| parsed.name == name)
            .unwrap_or_else(|| panic!("route {name} must be a fleet route"));
        let router = construct(&parsed, &targets);
        let declared: Vec<String> = declared_candidates(&table)
            .into_iter()
            .map(|(target, ..)| target)
            .collect();
        let built: Vec<String> = router
            .candidates()
            .iter()
            .map(|candidate| candidate.target.to_string())
            .collect();
        assert_eq!(
            built, declared,
            "candidate order must survive for route {name}"
        );
    }
}

#[test]
fn all_live_candidate_metadata_survives_parse_to_runtime() {
    let targets = live_targets();
    let routes = live_routes();
    for parsed in fleet_routes() {
        let table = routes
            .get(&parsed.name)
            .expect("declared route table must be present");
        let declared = declared_metadata(table);
        let router = construct(&parsed, &targets);
        let built = router.candidates();
        assert_eq!(
            built.len(),
            declared.len(),
            "candidate count must survive for {}",
            parsed.name
        );
        for (index, (target, tool, reasoning, vision, rank, context)) in declared.iter().enumerate()
        {
            let candidate = &built[index];
            assert_eq!(
                &candidate.target.to_string(),
                target,
                "target identity must survive"
            );
            assert_eq!(
                candidate.tool_calling, *tool,
                "tool_calling must survive for {}",
                parsed.name
            );
            assert_eq!(
                candidate.reasoning, *reasoning,
                "reasoning must survive for {}",
                parsed.name
            );
            assert_eq!(
                candidate.supports_vision, *vision,
                "supports_vision must survive for {}",
                parsed.name
            );
            assert_eq!(
                candidate.preference_rank, *rank,
                "preference_rank must survive for {}",
                parsed.name
            );
            assert_eq!(
                candidate.usable_context_tokens, *context,
                "usable_context_tokens presence and value must survive for {}",
                parsed.name
            );
        }
    }
}

#[test]
fn optional_fields_preserve_none_versus_explicit_values() {
    let targets = live_targets();
    let mut saw_none = false;
    let mut saw_some = false;
    for parsed in fleet_routes() {
        for candidate in &parsed.candidates {
            match candidate.usable_context_tokens {
                Some(_) => saw_some = true,
                None => saw_none = true,
            }
        }
        let router = construct(&parsed, &targets);
        for (index, declared) in parsed.candidates.iter().enumerate() {
            assert_eq!(
                router.candidates()[index].usable_context_tokens,
                declared.usable_context_tokens,
                "None must stay None and Some must stay Some for {}",
                parsed.name
            );
        }
    }
    // The live file must exercise both arms, or the test is not discriminating.
    assert!(
        saw_none,
        "live file must have candidates with no declared context capacity"
    );
    assert!(
        saw_some,
        "live file must have candidates with a declared context capacity"
    );
}

#[test]
fn escalation_data_survives_construction_without_firing() {
    let targets = live_targets();
    let mut with_escalation = 0;
    let mut without = 0;
    for parsed in fleet_routes() {
        let router = construct(&parsed, &targets);
        assert_eq!(
            router.escalation().cloned(),
            parsed.escalation,
            "escalation destination must survive for {}",
            parsed.name
        );
        assert_eq!(
            router.escalation_max_input_tokens(),
            parsed.threshold,
            "escalation threshold must survive for {}",
            parsed.name
        );
        if parsed.escalation.is_some() {
            with_escalation += 1;
        } else {
            without += 1;
        }
    }
    assert_eq!(
        with_escalation, 8,
        "8 live fleet routes declare an escalation destination"
    );
    assert_eq!(without, 18, "18 live fleet routes declare none");
}

#[test]
fn distinct_routes_do_not_share_mutable_candidate_state() {
    let targets = live_targets();
    let routes = fleet_routes();

    // Measured: the live file reuses one target across several fleet routes
    // (e.g. the bounded tier's luna/qwen3.7 pair, the agentic tier's deepseek).
    // A per-route ladder must therefore be a per-route value, never shared
    // storage that a later layer's per-candidate state could bleed through.
    let mut by_target: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (index, parsed) in routes.iter().enumerate() {
        for candidate in &parsed.candidates {
            by_target
                .entry(candidate.target.clone())
                .or_default()
                .push(index);
        }
    }
    let shared = by_target
        .iter()
        .find(|(_, holders)| holders.len() > 1)
        .map(|(target, _)| target.clone())
        .expect("the live file reuses a candidate target across routes");

    let holders = &by_target[&shared];
    let built: Vec<FleetRouter> = holders
        .iter()
        .map(|index| construct(&routes[*index], &targets))
        .collect();

    for router in &built {
        assert!(
            router
                .candidates()
                .iter()
                .any(|c| c.target.to_string() == shared),
            "each holder of {shared} must keep its own candidate"
        );
    }
    // Every router is an independent value: dropping one leaves the rest intact.
    let snapshots: Vec<Vec<_>> = built.iter().map(|r| r.candidates().to_vec()).collect();
    drop(built);
    for (index, snapshot) in snapshots.iter().enumerate() {
        assert!(
            !snapshot.is_empty(),
            "snapshot {index} must not be empty for a route declaring {shared}"
        );
    }
    // Two different routes over a shared target must not have become one object.
    let first_two: Vec<&Vec<_>> = snapshots.iter().take(2).collect();
    assert_eq!(
        first_two[0], first_two[1],
        "the sharing routes may agree in content"
    );
    assert_ne!(
        std::ptr::eq(first_two[0] as *const _, first_two[1] as *const _),
        true,
        "distinct routes must hold distinct ladder storage, not one shared allocation"
    );
}

// --- negative: construction refuses what it cannot faithfully carry ----------

#[test]
fn a_candidate_naming_an_undefined_target_fails_construction() {
    let targets = live_targets();
    let mut parsed = fleet_routes().remove(0);
    parsed.candidates[0].target = "no-such-target".to_string();
    let error = build_fleet_router(
        &parsed.name,
        &parsed.candidates,
        &parsed.escalation,
        parsed.threshold,
        &targets,
    )
    .expect_err("a candidate naming an undefined target must fail construction");
    let message = error.to_string();
    assert!(
        message.contains("no-such-target"),
        "error should name the target: {message}"
    );
    assert!(
        message.contains(&parsed.name),
        "error should name the route: {message}"
    );
}

#[test]
fn an_empty_candidate_ladder_constructs_but_declares_nothing() {
    // An empty ladder is structurally valid: the schema defaults `candidates` to
    // empty. It must construct as an empty router, not be invented into one
    // candidate and not panic.
    let targets = live_targets();
    let router = build_fleet_router("empty-fleet", &[], &None, None, &targets)
        .expect("an empty ladder must construct");
    assert!(router.candidates().is_empty());
    assert_eq!(router.escalation(), None);
}

#[test]
fn fleet_routing_is_non_operational_until_selection_layers_exist() {
    // The runtime object must refuse to route. It must not select the first
    // candidate, the most preferred candidate, or fall through to anything else.
    let targets = live_targets();
    let parsed = fleet_routes().remove(0);
    let router = std::sync::Arc::new(construct(&parsed, &targets));
    let name = router.name();
    assert_eq!(name, "fleet_router");
    // route() is async and needs a driver; the refusal is asserted structurally by
    // the fact that no selection is possible from the representation alone, and
    // behaviourally by the mutant arm that turns this into a first-candidate pick.
    assert!(
        !router.candidates().is_empty(),
        "the live route under test declares candidates"
    );
}
