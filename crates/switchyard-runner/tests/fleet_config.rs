// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Fleet-route configuration tests: the `AlgorithmSpec::FleetRouter` variant
//! parses the live-config schema (candidates, escalation, context policies)
//! and the load-time validations enforce truthful advertisement and bounded
//! escalation.

use switchyard_runner::Runner;

/// Minimal deployment in the exact shape the live `routes.toml` uses for
/// `fleet_router` routes (candidates as TOML array-of-tables, inline
/// `context_policy`, escalation fields at route level).
const FLEET_DEPLOYMENT: &str = r#"
schema_version = 1

[llm_clients.a]
format = "openai_chat"
base_url = "http://127.0.0.1:1/v1"
api_key_env = "TEST_KEY_A"

[llm_clients.b]
format = "openai_chat"
base_url = "http://127.0.0.1:2/v1"
api_key_env = "TEST_KEY_B"

[targets.primary]
id = "vendor/primary"
llm_client = "a"
[targets.secondary]
id = "vendor/secondary"
llm_client = "b"

[targets.primary-route]
id = "vendor/primary-served"
llm_client = "a"
[targets.secondary-route]
id = "vendor/secondary-served"
llm_client = "b"

[routes.served-primary]
id = "vendor/primary-served"
type = "passthrough"
target = "primary"

[routes.served-secondary]
id = "vendor/secondary-served"
type = "passthrough"
target = "secondary"

[routes.fleet]
id = "fleet/smart"
type = "fleet_router"
context_window = 266000
tool_calling = true
reasoning = false
escalation = "vendor/secondary-served"

[[routes.fleet.candidates]]
target = "primary-route"
tool_calling = true
reasoning = false
preference_rank = 1
usable_context_tokens = 266000
supports_vision = true

[[routes.fleet.candidates]]
target = "secondary-route"
tool_calling = true
reasoning = false
preference_rank = 2
usable_context_tokens = 1000000
context_policy = { kind = "bounded", usable_context_tokens = 1000000, input_token_source = "openai_chat_input_tokens" }

[fleet_readiness]
observe_interval_seconds = 30
ready = ["vendor/primary-served", "vendor/secondary-served"]
"#;

fn deployment_with_fleet_body(body: &str) -> String {
    // Replace the fleet route table with `body` for negative variants.
    let start = FLEET_DEPLOYMENT
        .find("[routes.fleet]")
        .expect("fleet route marker");
    let end = FLEET_DEPLOYMENT
        .find("[fleet_readiness]")
        .expect("readiness marker");
    let mut out = String::new();
    out.push_str(&FLEET_DEPLOYMENT[..start]);
    out.push_str(body);
    out.push('\n');
    out.push_str(&FLEET_DEPLOYMENT[end..]);
    out
}

/// The runner's fail-loud credential guard reads `api_key_env` at build time;
/// test deployments point at throwaway env vars set once here.
fn ensure_test_credentials() {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        for key in ["TEST_KEY_A", "TEST_KEY_B"] {
            // Test-process-local; no concurrent writes after the Once.
            unsafe { std::env::set_var(key, "test-credential-placeholder") };
        }
    });
}

/// Asserts the deployment fails to load and returns the configuration error.
fn expect_config_error(source: &str) -> switchyard_runner::RunnerError {
    ensure_test_credentials();
    match Runner::from_toml(source) {
        Err(error) => error,
        Ok(_) => panic!("expected configuration failure"),
    }
}

#[test]
fn live_shape_fleet_route_parses_and_builds() {
    ensure_test_credentials();
    let runner = Runner::from_toml(FLEET_DEPLOYMENT).expect("live-shape fleet deployment parses");
    // The fleet route registers, its candidates are the routing targets, and
    // the advertised envelope derives truthfully from the candidate set.
    let route = runner.route("fleet/smart").expect("fleet route registered");
    assert_eq!(route.algorithm_name(), "fleet_router");
    let caps = route.capabilities();
    assert_eq!(caps.context_window, Some(266000));
    assert_eq!(caps.tool_calling, Some(true));
    assert_eq!(caps.reasoning, Some(false));
    assert_eq!(caps.supports_vision, Some(true));
    // Escalation metadata is carried for the server-side runtime.
    assert_eq!(
        route.escalation().map(ModelId::as_str),
        Some("vendor/secondary-served")
    );
    assert!(route.escalation_max_input_tokens().is_none());
    // The deployment-wide fleet-state handle exists for the monitor.
    assert!(runner.fleet_state().is_some());
    assert!(runner.fleet_readiness().is_some());
}

use switchyard_protocol::ModelId;

#[test]
fn unregistered_escalation_destination_is_rejected() {
    let source = deployment_with_fleet_body(
        r#"[routes.fleet]
id = "fleet/smart"
type = "fleet_router"
escalation = "vendor/missing"
candidates = []
"#,
    );
    let error = expect_config_error(&source); // unregistered destination must fail
    assert!(
        error.to_string().contains("is not a registered route id"),
        "unexpected error: {error}"
    );
}

#[test]
fn self_escalation_is_rejected() {
    let source = deployment_with_fleet_body(
        r#"[routes.fleet]
id = "fleet/smart"
type = "fleet_router"
escalation = "fleet/smart"
candidates = []
"#,
    );
    let error = expect_config_error(&source); // self escalation must fail
    assert!(
        error.to_string().contains("cannot escalate to itself"),
        "unexpected error: {error}"
    );
}

#[test]
fn escalation_chains_are_rejected() {
    let source = FLEET_DEPLOYMENT.replace(
        r#"[routes.served-secondary]
id = "vendor/secondary-served"
type = "passthrough"
target = "secondary"
"#,
        r#"[routes.served-secondary]
id = "vendor/secondary-served"
type = "fleet_router"
escalation = "vendor/primary-served"
candidates = []
"#,
    );
    let error = expect_config_error(&source); // escalation chains must fail
    assert!(
        error
            .to_string()
            .contains("must not itself declare an escalation"),
        "unexpected error: {error}"
    );
}

#[test]
fn escalation_threshold_without_destination_is_rejected() {
    let source = deployment_with_fleet_body(
        r#"[routes.fleet]
id = "fleet/smart"
type = "fleet_router"
escalation_max_input_tokens = 1000
candidates = []
"#,
    );
    let error = expect_config_error(&source);
    assert!(
        error
            .to_string()
            .contains("declares escalation_max_input_tokens without an escalation destination"),
        "unexpected error: {error}"
    );
}

#[test]
fn untruthful_capability_advertisement_is_rejected() {
    let source = FLEET_DEPLOYMENT.replace(
        r#"tool_calling = true
reasoning = false
escalation = "vendor/secondary-served""#,
        r#"tool_calling = true
reasoning = true
escalation = "vendor/secondary-served""#,
    );
    let error = expect_config_error(&source);
    assert!(
        error
            .to_string()
            .contains("advertises reasoning=true but no"),
        "unexpected error: {error}"
    );
}

#[test]
fn empty_candidate_fleet_is_rejected() {
    let source = deployment_with_fleet_body(
        r#"[routes.fleet]
id = "fleet/smart"
type = "fleet_router"
candidates = []
"#,
    );
    let error = expect_config_error(&source); // empty candidates must fail
    assert!(
        error
            .to_string()
            .contains("requires at least one candidate profile"),
        "unexpected error: {error}"
    );
}

#[test]
fn readiness_gated_list_requires_its_source() {
    let source = FLEET_DEPLOYMENT.replace(
        r#"[fleet_readiness]
observe_interval_seconds = 30
ready = ["vendor/primary-served", "vendor/secondary-served"]
"#,
        r#"[fleet_readiness]
observe_interval_seconds = 30
ready = ["vendor/primary-served"]

[fleet_readiness.resource]
openai_gated = ["gpt-5.6-luna"]
"#,
    );
    let error = expect_config_error(&source);
    assert!(
        error
            .to_string()
            .contains("openai_gated candidates require openai_url and openai_auth_token_env"),
        "unexpected error: {error}"
    );
}
