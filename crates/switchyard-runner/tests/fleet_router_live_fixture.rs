// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! F1 live-fixture acceptance against the real deployment `routes.toml`.
//!
//! Deliberately written without naming `AlgorithmSpec::FleetRouter` or
//! `FleetCandidateConfig`. That is what makes this file a usable control: it
//! compiles unchanged against a tree whose schema lacks the fleet route type,
//! so running it there produces a *test failure* that names the missing
//! surface, rather than a compile error that proves nothing.
//!
//! Key scoping. Each live route table also carries keys F1 does not own. They
//! are stripped before the table reaches the algorithm parser, and `STRIPPED`
//! names each one with its owning row, so a reviewer can see exactly what this
//! file does not prove:
//!
//! - `id`, `context_window`, `tool_calling`, `reasoning`, `supports_vision`
//!   are route-level capability fields parsed by `RouteConfig`, not by
//!   `AlgorithmSpec`. The `supports_vision` alias is row 41.
//! - `reasoning_policy`, `escalation_reasoning_policy` are row 12's
//!   route-authoritative reasoning surface, not F1's.
//!
//! Escalation keys are NOT stripped: F1 owns `escalation` and
//! `escalation_max_input_tokens` as carried-and-preserved fields.

use switchyard_runner::AlgorithmSpec;

/// Location of the live deployment config. Overridable for a relocated tree.
const LIVE_CONFIG: &str = match option_env!("SWITCHYARD_LIVE_CONFIG") {
    Some(path) => path,
    None => "/home/vincent/.local/lib/localclaw-switchyard/routes.toml",
};

/// Route-level and other-row keys removed before the algorithm parser sees a
/// live route table, with the row that owns each.
const STRIPPED: &[(&str, &str)] = &[
    ("id", "RouteConfig"),
    ("context_window", "RouteConfig"),
    ("tool_calling", "RouteConfig"),
    ("reasoning", "RouteConfig"),
    ("supports_vision", "route.rs ModelCapabilities (row 41)"),
    ("vision", "route.rs ModelCapabilities (row 41)"),
    ("reasoning_policy", "row 12"),
    ("escalation_reasoning_policy", "row 12"),
];

/// A live route table with the non-F1 keys removed, ready for the algorithm parser.
fn algorithm_table(route: &toml::Table) -> toml::Table {
    let mut table = route.clone();
    for (key, _) in STRIPPED {
        table.remove(*key);
    }
    table
}

fn live_routes() -> toml::Table {
    let text = std::fs::read_to_string(LIVE_CONFIG)
        .unwrap_or_else(|error| panic!("live config {LIVE_CONFIG} must be readable: {error}"));
    let document: toml::Table = toml::from_str(&text)
        .unwrap_or_else(|error| panic!("live config must be valid TOML: {error}"));
    document
        .get("routes")
        .and_then(toml::Value::as_table)
        .cloned()
        .unwrap_or_else(|| panic!("live config must have a [routes] table"))
}

fn split_routes() -> (Vec<(String, toml::Table)>, Vec<(String, toml::Table)>) {
    let mut fleet = Vec::new();
    let mut passthrough = Vec::new();
    for (name, value) in live_routes() {
        let table = value
            .as_table()
            .unwrap_or_else(|| panic!("route {name} must be a table"))
            .clone();
        match table.get("type").and_then(toml::Value::as_str) {
            Some("fleet_router") => fleet.push((name, table)),
            Some("passthrough") => passthrough.push((name, table)),
            other => panic!("route {name} has unexpected type {other:?}"),
        }
    }
    (fleet, passthrough)
}

fn declared_candidate_order(route: &toml::Table) -> Vec<String> {
    route
        .get("candidates")
        .and_then(toml::Value::as_array)
        .unwrap_or_else(|| panic!("fleet route must declare candidates"))
        .iter()
        .map(|candidate| {
            candidate
                .as_table()
                .and_then(|table| table.get("target"))
                .and_then(toml::Value::as_str)
                .unwrap_or_else(|| panic!("every candidate must name a target"))
                .to_string()
        })
        .collect()
}

// --- acceptance -------------------------------------------------------------

#[test]
fn every_live_fleet_route_deserializes_under_its_f1_schema() {
    let (fleet, _) = split_routes();
    assert_eq!(fleet.len(), 26, "live deployment declares 26 fleet routes");
    for (name, table) in &fleet {
        toml::Value::Table(algorithm_table(table))
            .try_into::<AlgorithmSpec>()
            .unwrap_or_else(|error| panic!("fleet route {name} must deserialize: {error}"));
    }
}

#[test]
fn every_live_fleet_route_preserves_its_candidate_order() {
    let (fleet, _) = split_routes();
    for (name, table) in &fleet {
        let spec: AlgorithmSpec = toml::Value::Table(algorithm_table(table))
            .try_into()
            .unwrap_or_else(|error| panic!("fleet route {name} must deserialize: {error}"));
        let declared = declared_candidate_order(table);
        let observed: Vec<String> = spec
            .routing_target_names()
            .into_iter()
            .map(str::to_string)
            .collect();
        assert_eq!(
            observed, declared,
            "candidate order must survive for route {name}"
        );
    }
}

#[test]
fn every_live_passthrough_route_is_unaffected() {
    let (_, passthrough) = split_routes();
    assert_eq!(
        passthrough.len(),
        17,
        "live deployment declares 17 passthrough routes"
    );
    for (name, table) in &passthrough {
        let spec: AlgorithmSpec = toml::Value::Table(algorithm_table(table))
            .try_into()
            .unwrap_or_else(|error| panic!("passthrough route {name} must deserialize: {error}"));
        let target = table
            .get("target")
            .and_then(toml::Value::as_str)
            .expect("passthrough route names a target");
        assert_eq!(
            spec.routing_target_names(),
            [target],
            "route {name} must be unchanged"
        );
    }
}

#[test]
fn live_fleet_routes_cover_the_whole_declared_candidate_population() {
    let (fleet, _) = split_routes();
    let declared: usize = fleet
        .iter()
        .map(|(_, t)| declared_candidate_order(t).len())
        .sum();
    assert_eq!(
        declared, 46,
        "the live file declares 46 candidates in total"
    );
}
