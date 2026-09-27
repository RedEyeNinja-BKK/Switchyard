// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! F1 schema admission for the `fleet_router` route type.
//!
//! Scope is the live deployment surface only: the route keys and candidate keys
//! a live `routes.toml` actually sets, preserved exactly and in order. No
//! runtime behavior is asserted here, because none is implemented - see the
//! `boundary_*` tests, which pin that a fleet route cannot be built yet.
//!
//! "Schema admission" in these names means the table deserializes into the
//! declared shape. It deliberately does not mean the route is runnable.

use switchyard_runner::AlgorithmSpec;

/// A live-shaped fleet route: one candidate, no escalation.
const LIVE_SINGLE_CANDIDATE: &str = r#"
type = "fleet_router"
candidates = [
  { target = "comfyninja-qwen3_8-27b-q3", tool_calling = true, reasoning = true, supports_vision = true, preference_rank = 1 },
]
"#;

/// A live-shaped fleet route exercising every key F1 carries, including the
/// escalation pair and a declared context capacity.
const LIVE_ALL_F1_KEYS: &str = r#"
type = "fleet_router"
candidates = [
  { target = "alpha", tool_calling = true, reasoning = true, supports_vision = true, preference_rank = 1, usable_context_tokens = 262144 },
  { target = "beta", tool_calling = false, reasoning = false, supports_vision = false, preference_rank = 2 },
  { target = "gamma", preference_rank = 3 },
]
escalation = "sw-comfyninja-qwen3_8-27b-q3"
escalation_max_input_tokens = 61440
"#;

fn parse(table: &str) -> Result<AlgorithmSpec, toml::de::Error> {
    toml::from_str::<AlgorithmSpec>(table)
}

fn fleet(
    spec: &AlgorithmSpec,
) -> (
    &[switchyard_runner::FleetCandidateConfig],
    Option<&str>,
    Option<u64>,
) {
    match spec {
        AlgorithmSpec::FleetRouter {
            candidates,
            escalation,
            escalation_max_input_tokens,
        } => (
            candidates,
            escalation.as_deref(),
            *escalation_max_input_tokens,
        ),
        other => panic!("expected fleet_router, got {other:?}"),
    }
}

fn parse_err(table: &str) -> String {
    parse(table)
        .expect_err("table must be rejected")
        .to_string()
}

// --- schema admission: live surface accepted and preserved -------------------

#[test]
fn live_fleet_route_table_deserializes() {
    let spec = parse(LIVE_SINGLE_CANDIDATE).expect("live-shaped fleet route must deserialize");
    let (candidates, _, _) = fleet(&spec);
    assert_eq!(candidates.len(), 1);
    assert_eq!(candidates[0].target, "comfyninja-qwen3_8-27b-q3");
}

#[test]
fn all_f1_route_keys_are_retained() {
    let spec = parse(LIVE_ALL_F1_KEYS).expect("fleet route with every F1 key must deserialize");
    let (_, escalation, max_input) = fleet(&spec);
    assert_eq!(escalation, Some("sw-comfyninja-qwen3_8-27b-q3"));
    assert_eq!(max_input, Some(61_440));
}

#[test]
fn candidate_order_is_preserved() {
    let spec = parse(LIVE_ALL_F1_KEYS).unwrap();
    let (candidates, _, _) = fleet(&spec);
    let order: Vec<&str> = candidates.iter().map(|c| c.target.as_str()).collect();
    assert_eq!(
        order,
        ["alpha", "beta", "gamma"],
        "declared order must survive deserialization"
    );
}

#[test]
fn candidate_metadata_is_preserved() {
    let spec = parse(LIVE_ALL_F1_KEYS).unwrap();
    let (candidates, _, _) = fleet(&spec);

    assert!(candidates[0].tool_calling);
    assert!(candidates[0].reasoning);
    assert!(candidates[0].supports_vision);
    assert_eq!(candidates[0].preference_rank, 1);
    assert_eq!(candidates[0].usable_context_tokens, Some(262_144));

    // Absent booleans take the owner's default of false rather than erroring.
    assert!(!candidates[1].tool_calling);
    assert!(!candidates[1].reasoning);
    assert!(!candidates[1].supports_vision);
    assert_eq!(candidates[1].preference_rank, 2);
    assert_eq!(candidates[1].usable_context_tokens, None);

    // Every key is optional except `target`; a bare candidate is valid.
    assert_eq!(candidates[2].target, "gamma");
    assert_eq!(candidates[2].preference_rank, 3);
    assert_eq!(candidates[2].usable_context_tokens, None);
}

#[test]
fn routing_targets_follow_declared_candidate_order() {
    let spec = parse(LIVE_ALL_F1_KEYS).unwrap();
    assert_eq!(spec.routing_target_names(), ["alpha", "beta", "gamma"]);
}

#[test]
fn routing_targets_are_never_sorted_even_when_declared_order_is_not_alphabetical() {
    // `alpha, beta, gamma` is already alphabetical, so a sort would be invisible
    // there. This fixture declares a genuinely non-alphabetical ladder, so any
    // sort by target id reorders it and this assertion can observe that.
    let spec = parse(
        r#"type = "fleet_router"
candidates = [
    { target = "zeta", preference_rank = 1 },
    { target = "alpha", preference_rank = 2 },
    { target = "mike", preference_rank = 3 },
]"#,
    )
    .expect("a fleet route must deserialize");
    assert_eq!(
        spec.routing_target_names(),
        ["zeta", "alpha", "mike"],
        "the declared ladder must survive verbatim; no layer may sort it"
    );
}

#[test]
fn absent_candidates_and_escalation_are_allowed() {
    let spec = parse(r#"type = "fleet_router""#).expect("an empty fleet route must deserialize");
    let (candidates, escalation, max_input) = fleet(&spec);
    assert!(candidates.is_empty());
    assert_eq!(escalation, None);
    assert_eq!(max_input, None);
}

// --- negative: everything outside the live surface is rejected, not ignored --

#[test]
fn unknown_candidate_key_fails_closed() {
    let error = parse_err(
        r#"
type = "fleet_router"
candidates = [ { target = "alpha", unknown_key = 1 } ]
"#,
    );
    assert!(
        error.contains("unknown_key"),
        "error should name the key: {error}"
    );
}

#[test]
fn context_policy_fails_closed() {
    let error = parse_err(
        r#"
type = "fleet_router"
candidates = [ { target = "alpha", context_policy = { kind = "unmanaged" } } ]
"#,
    );
    assert!(
        error.contains("context_policy"),
        "error should name the key: {error}"
    );
}

#[test]
fn work_shape_fails_closed() {
    let error = parse_err(
        r#"
type = "fleet_router"
candidates = [ { target = "alpha", work_shape = "bounded" } ]
"#,
    );
    assert!(
        error.contains("work_shape"),
        "error should name the key: {error}"
    );
}

#[test]
fn work_shape_source_fails_closed() {
    let error = parse_err(
        r#"
type = "fleet_router"
work_shape_source = "request"
"#,
    );
    assert!(
        error.contains("work_shape_source"),
        "error should name the key: {error}"
    );
}

#[test]
fn malformed_candidate_fails_closed() {
    // Not a table at all.
    let error = parse_err(
        r#"
type = "fleet_router"
candidates = [ "alpha" ]
"#,
    );
    assert!(
        error.contains("invalid type") || error.contains("expected"),
        "{error}"
    );

    // A candidate with no `target` cannot name a target to select.
    let error = parse_err(
        r#"
type = "fleet_router"
candidates = [ { tool_calling = true } ]
"#,
    );
    assert!(
        error.contains("target"),
        "error should name the missing field: {error}"
    );
}

#[test]
fn malformed_escalation_fails_closed() {
    // Destination must be a string route id.
    let error = parse_err(
        r#"
type = "fleet_router"
escalation = 42
"#,
    );
    assert!(
        error.contains("invalid type") || error.contains("expected"),
        "{error}"
    );

    // Threshold must be a non-negative integer token count.
    let error = parse_err(
        r#"
type = "fleet_router"
escalation_max_input_tokens = "61440"
"#,
    );
    assert!(
        error.contains("invalid type") || error.contains("expected"),
        "{error}"
    );

    // A negative threshold is not a representable token count.
    let error = parse_err(
        r#"
type = "fleet_router"
escalation_max_input_tokens = -1
"#,
    );
    assert!(
        error.contains("invalid value") || error.contains("expected"),
        "{error}"
    );
}

#[test]
fn unknown_fleet_route_key_fails_closed() {
    let error = parse_err(
        r#"
type = "fleet_router"
candidate = []
"#,
    );
    assert!(
        error.contains("candidate"),
        "error should name the key: {error}"
    );
}

// --- boundary: a fleet route is not yet runnable ---------------------------
//
// These pin the deliberate state of this layer. The schema is accepted; the
// runtime is absent, and a caller that tries to build one gets a legible
// configuration error rather than a panic, a silent fallback, or a false
// success. If a future runtime layer lands, these fail and must be revisited -
// that is the intended signal, not a stale test to be relaxed.

#[test]
fn boundary_constructing_a_fleet_route_succeeds_and_preserves_the_candidate() {
    use std::collections::BTreeMap;

    // F2: construction is faithful and total. The route becomes a runtime object.
    let spec = parse(LIVE_SINGLE_CANDIDATE).unwrap();
    let (candidates, escalation, threshold) = match &spec {
        AlgorithmSpec::FleetRouter {
            candidates,
            escalation,
            escalation_max_input_tokens,
        } => (candidates, escalation.clone(), *escalation_max_input_tokens),
        other => panic!("expected fleet_router, got {other:?}"),
    };
    let targets = BTreeMap::from([(
        "comfyninja-qwen3_8-27b-q3".to_string(),
        switchyard_protocol::ModelId::from("comfyninja-qwen3_8-27b-q3"),
    )]);
    let router = switchyard_runner::build_fleet_router(
        "sw-comfyninja-qwen3_8-27b-q3",
        &candidates,
        &escalation,
        threshold,
        &targets,
    )
    .expect("F2 must construct a parsed fleet route");

    let built = router.candidates();
    assert_eq!(built.len(), 1);
    assert_eq!(
        built[0].target,
        switchyard_protocol::ModelId::from("comfyninja-qwen3_8-27b-q3")
    );
    assert!(built[0].tool_calling);
    assert!(built[0].reasoning);
    assert!(built[0].supports_vision);
    assert_eq!(built[0].preference_rank, 1);
    assert_eq!(built[0].usable_context_tokens, None);
    assert_eq!(router.escalation(), None);
    assert_eq!(router.escalation_max_input_tokens(), None);
}

#[test]
fn boundary_other_algorithm_types_still_build() {
    use std::collections::BTreeMap;

    // The FleetRouter arm must not have displaced or broken any other arm.
    let spec = parse(
        r#"type = "passthrough"
target = "alpha""#,
    )
    .unwrap();
    let targets = BTreeMap::from([(
        "alpha".to_string(),
        switchyard_protocol::ModelId::from("alpha"),
    )]);
    assert!(spec.build("passthrough-route", &targets).is_ok());
}
