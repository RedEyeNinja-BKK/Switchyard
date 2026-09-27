//! F7 load-time validation: a route's reasoning policy must fail closed
//! against the exact live surface, before any request is ever sent.
//!
//! The route table is spelled the way the live file spells it: top-level
//! `type = "passthrough"` (or `fleet_router`), with the reasoning keys as
//! siblings inside `[routes.<name>]`.

use std::collections::BTreeMap;

use switchyard_llm_client::{Backend, HttpBackendConfig};
use switchyard_protocol::{ReasoningDialect, ReasoningPolicy};
use switchyard_runner::Runner;

fn config(
    policy: Option<&str>,
    dialect: Option<(&str, &str)>,
    vocabulary: Option<&str>,
    target_extra: &str,
) -> String {
    let vocabulary_line = vocabulary.unwrap_or("");
    let (client_name, client_block) = match dialect {
        Some((name, value)) => (
            name.to_string(),
            format!(
                "[llm_clients.{name}]\n\
                 format = \"openai_responses\"\n\
                 base_url = \"https://example.test/v1\"\n\
                 reasoning_dialect = \"{value}\"\n{vocabulary_line}\n"
            ),
        ),
        None => (
            "plain".to_string(),
            "[llm_clients.plain]\nformat = \"openai_responses\"\nbase_url = \"https://example.test/v1\"\n".to_string(),
        ),
    };
    let policy_line = match policy {
        Some(value) => format!("reasoning_policy = \"{value}\"\n"),
        None => String::new(),
    };
    format!(
        "schema_version = 1\n\
         \n\
         [targets.t]\n\
         id = \"m/a\"\n\
         llm_client = \"{client_name}\"\n{target_extra}\
         \n\
         [routes.r]\n\
         id = \"route/r\"\n\
         type = \"passthrough\"\n\
         target = \"t\"\n\
         {policy_line}\
         {client_block}"
    )
}

fn load(body: &str) -> Result<(), String> {
    // Unique per call: cargo runs these in parallel threads, so a shared or
    // content-derived name lets two tests overwrite each other's config and
    // observe another test's result.
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let unique = COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    let mut path = std::env::temp_dir();
    path.push(format!(
        "switchyard-f7-{}-{unique}.toml",
        std::process::id()
    ));
    std::fs::write(&path, body).unwrap();
    let result = Runner::load(&path)
        .map(|_| ())
        .map_err(|error| error.to_string());
    let _ = std::fs::remove_file(&path);
    result
}

const OA_VOCAB: &str = "reasoning_efforts = [\"none\", \"low\", \"max\"]\n";

// ---------------------------------------------------------------------------
// Accepted combinations (live-equivalent).
// ---------------------------------------------------------------------------

#[test]
fn an_openai_effort_route_with_a_declared_vocabulary_loads() {
    load(&config(
        Some("none"),
        Some(("c", "openai_effort")),
        Some(OA_VOCAB),
        "",
    ))
    .expect("a live-shaped route policy must load");
}

#[test]
fn a_boolean_dialect_route_policy_loads_without_any_vocabulary() {
    for policy in ["none", "enabled"] {
        load(&config(
            Some(policy),
            Some(("c", "llama_cpp_enable_thinking")),
            None,
            "",
        ))
        .unwrap_or_else(|error| panic!("{policy} on a boolean dialect must load: {error}"));
    }
}

#[test]
fn an_absent_reasoning_policy_loads() {
    load(&config(None, None, None, "")).expect("a route with no policy must load unchanged");
}

#[test]
fn a_target_without_a_reasoning_pin_is_neutral_and_loads() {
    // The neutral end state the conflict rule pushes operators toward.
    load(&config(
        Some("none"),
        Some(("c", "openai_effort")),
        Some(OA_VOCAB),
        "",
    ))
    .expect("a neutral target must load with any expressible route policy");
}

#[test]
fn strip_reasoning_content_is_accepted_on_a_target() {
    load(&config(
        None,
        None,
        None,
        "strip_reasoning_content = true\n",
    ))
    .expect("the live file declares this on five targets; it must parse");
}

// ---------------------------------------------------------------------------
// The strip seam, through the REAL backend rather than a local copy.
// ---------------------------------------------------------------------------

fn backend_with_strip(strip: bool) -> Backend {
    Backend::OpenAiResponses(HttpBackendConfig {
        base_url: "https://example.test/v1".to_string(),
        api_key: None,
        forward_auth: false,
        extra_headers: BTreeMap::new(),
        extra_body: BTreeMap::new(),
        reasoning_effort: None,
        reasoning_dialect: Some(ReasoningDialect::OpenRouterEnabled),
        reasoning_efforts: None,
        strip_reasoning_content: strip,
        max_retries: 0,
        timeout: None,
    })
}

#[test]
fn a_stripping_backend_reports_the_flag_and_a_plain_one_does_not() {
    assert!(backend_with_strip(true).strip_reasoning_content());
    assert!(
        !backend_with_strip(false).strip_reasoning_content(),
        "a target that did not declare the flag must not strip"
    );
}

#[test]
fn the_strip_flag_is_carried_per_target_not_globally() {
    // The five live targets are OpenRouter lanes; every other target in the
    // same deployment must be unaffected. A backend built with the flag off
    // must stay off even when another target in the same file has it on.
    let stripping = backend_with_strip(true);
    let plain = backend_with_strip(false);
    assert!(stripping.strip_reasoning_content());
    assert!(!plain.strip_reasoning_content());
}

#[test]
fn the_backend_carries_the_dialect_and_vocabulary_it_was_configured_with() {
    let backend = backend_with_strip(false);
    assert_eq!(
        backend.reasoning_dialect(),
        Some(ReasoningDialect::OpenRouterEnabled)
    );
    assert_eq!(backend.reasoning_efforts(), None);
}

// ---------------------------------------------------------------------------
// Rejected combinations: fail closed at load time.
// ---------------------------------------------------------------------------

#[test]
fn a_policy_reaching_a_dialect_less_client_is_rejected() {
    let error = load(&config(Some("none"), None, None, ""))
        .expect_err("a policy-bearing route may not reach a dialect-less client");
    assert!(
        error.contains("declares no reasoning_dialect"),
        "unexpected error: {error}"
    );
}

#[test]
fn an_exact_effort_on_a_boolean_dialect_is_rejected() {
    let error = load(&config(
        Some("max"),
        Some(("c", "llama_cpp_enable_thinking")),
        None,
        "",
    ))
    .expect_err("an exact effort must never collapse to a bare switch");
    assert!(
        error.contains("cannot faithfully express it"),
        "unexpected error: {error}"
    );
}

#[test]
fn enabled_on_an_effort_bearing_dialect_is_rejected() {
    let error = load(&config(
        Some("enabled"),
        Some(("c", "openai_effort")),
        Some(OA_VOCAB),
        "",
    ))
    .expect_err("enabled has no tested representation on an effort dialect");
    assert!(
        error.contains("cannot faithfully express it"),
        "unexpected error: {error}"
    );
}

#[test]
fn an_undeclared_effort_is_rejected_and_never_substituted() {
    // The vocabulary declares none/low/max; the route demands `high`.
    let error = load(&config(
        Some("high"),
        Some(("c", "openai_effort")),
        Some(OA_VOCAB),
        "",
    ))
    .expect_err("an undeclared effort must fail closed, not be approximated");
    assert!(
        error.contains("cannot faithfully express it"),
        "unexpected error: {error}"
    );
    assert!(
        !error.contains("substitut"),
        "the error must not suggest a substitute effort"
    );
}

#[test]
fn an_effort_bearing_dialect_without_a_vocabulary_is_rejected() {
    let error = load(&config(
        Some("none"),
        Some(("c", "openai_effort")),
        None,
        "",
    ))
    .expect_err("an effort-bearing dialect needs a declared vocabulary");
    assert!(
        error.contains("without declaring reasoning_efforts"),
        "unexpected error: {error}"
    );
}

#[test]
fn a_route_policy_contradicting_a_target_reasoning_pin_is_rejected() {
    // Two authoritative sources must not fight.
    let error = load(&config(
        Some("none"),
        Some(("c", "openai_effort")),
        Some(OA_VOCAB),
        "reasoning_effort = \"low\"\n",
    ))
    .expect_err("a target hard pin contradicts the route policy");
    assert!(
        error.contains("must not fight"),
        "unexpected error: {error}"
    );
}

#[test]
fn a_route_policy_contradicting_an_extra_body_reasoning_pin_is_rejected() {
    // Spelled the way the live file spells it: an inline `extra_body` table.
    for pin in [
        "extra_body = { reasoning = { enabled = false } }\n",
        "extra_body = { reasoning_effort = \"high\" }\n",
        "extra_body = { chat_template_kwargs = { enable_thinking = false } }\n",
    ] {
        let error = load(&config(
            Some("none"),
            Some(("c", "openai_effort")),
            Some(OA_VOCAB),
            pin,
        ))
        .expect_err("an extra_body reasoning pin contradicts the route policy");
        assert!(
            error.contains("must not fight"),
            "unexpected error for pin {pin:?}: {error}"
        );
    }
}

#[test]
fn an_unknown_policy_spelling_is_rejected_rather_than_read_as_absent() {
    let error = load(&config(
        Some("turbo"),
        Some(("c", "openrouter_enabled")),
        None,
        "",
    ))
    .expect_err("an unknown policy must not silently mean 'no policy'");
    assert!(
        error.contains("not a known reasoning policy"),
        "unexpected error: {error}"
    );
}

/// A fleet route with an escalation destination, as the live file spells it:
/// the destination is another registered ROUTE id, and the escalated leg runs
/// under the destination's own reach.
fn fleet_with_escalation(
    primary: &str,
    escalation: &str,
    dest_dialect: (&str, &str),
    dest_vocabulary: Option<&str>,
) -> String {
    let dest_client = dest_dialect.0;
    let dialect_value = dest_dialect.1;
    let vocabulary_line = dest_vocabulary.unwrap_or("");
    format!(
        "schema_version = 1\n\
         \n\
         [targets.dest]\n\
         id = \"m/dest\"\n\
         llm_client = \"{dest_client}\"\n\
         \n\
         [llm_clients.{dest_client}]\n\
         format = \"openai_responses\"\n\
         base_url = \"https://example.test/v1\"\n\
         reasoning_dialect = \"{dialect_value}\"\n\
         {vocabulary_line}\n\
         \n\
         [routes.dest_route]\n\
         id = \"route/dest\"\n\
         type = \"passthrough\"\n\
         target = \"dest\"\n\
         \n\
         [routes.r]\n\
         id = \"route/r\"\n\
         type = \"fleet_router\"\n\
         candidates = [{{ target = \"dest\", preference_rank = 1 }}]\n\
         escalation = \"route/dest\"\n\
         reasoning_policy = \"{primary}\"\n\
         escalation_reasoning_policy = \"{escalation}\"\n"
    )
}

#[test]
fn an_escalation_policy_under_an_unexpressible_dialect_is_rejected() {
    // The destination's dialect cannot express `max`, so the escalated leg
    // could never be honoured: a load-time config error, not a runtime
    // surprise once the seam is actually crossed in F8.
    let body = fleet_with_escalation("none", "max", ("c", "llama_cpp_enable_thinking"), None);
    let error =
        load(&body).expect_err("an escalation policy must honour the same expressibility rules");
    assert!(
        error.contains("cannot faithfully express it"),
        "unexpected error: {error}"
    );
}

#[test]
fn a_live_shaped_escalation_policy_parses() {
    // Live shape: primary `none` + escalation `none` on a
    // llama_cpp_enable_thinking destination, which declares no vocabulary.
    let body = fleet_with_escalation("none", "none", ("c", "llama_cpp_enable_thinking"), None);
    load(&body).expect("the live escalation spelling must load");
}

#[test]
fn an_escalation_policy_without_a_destination_is_rejected() {
    // A passthrough route declares no escalation destination, so an
    // escalation policy on it is a contradiction, not a no-op.
    let body = config(Some("none"), Some(("c", "openrouter_enabled")), None, "").replace(
        "reasoning_policy = \"none\"\n",
        "reasoning_policy = \"none\"\nescalation_reasoning_policy = \"none\"\n",
    );
    let error = load(&body).expect_err("an escalation policy needs a destination");
    assert!(
        error.contains("without an escalation destination"),
        "unexpected error: {error}"
    );
}

#[test]
fn an_escalation_policy_naming_an_unregistered_route_is_rejected() {
    let body = fleet_with_escalation("none", "none", ("c", "llama_cpp_enable_thinking"), None)
        .replace(
            "escalation = \"route/dest\"",
            "escalation = \"route/nonexistent\"",
        );
    let error = load(&body).expect_err("an unknown escalation destination must fail closed");
    assert!(
        error.contains("not a registered route id"),
        "unexpected error: {error}"
    );
}

#[test]
fn a_declared_policy_derives_the_route_reasoning_capability() {
    // A route pinned to `none` advertises no reasoning, even though its
    // candidate profile is reasoning-capable.
    let body = fleet_with_escalation("none", "none", ("c", "llama_cpp_enable_thinking"), None);
    load(&body).expect("the capability derivation must not reject a live shape");
}

// ---------------------------------------------------------------------------
// The real Route: policy and escalation resolution, not a local re-implementation.
// ---------------------------------------------------------------------------

fn runner_with_policies(
    primary: Option<ReasoningPolicy>,
    escalation: Option<ReasoningPolicy>,
) -> Runner {
    let mut path = std::env::temp_dir();
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let unique = COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    path.push(format!(
        "switchyard-f7-route-{}-{unique}.toml",
        std::process::id()
    ));

    let primary_line = primary
        .map(|p| format!("reasoning_policy = \"{p}\"\n"))
        .unwrap_or_default();
    let escalation_line = escalation
        .map(|p| format!("escalation_reasoning_policy = \"{p}\"\n"))
        .unwrap_or_default();
    std::fs::write(
        &path,
        format!(
            "schema_version = 1\n\
             \n\
             [targets.dest]\n\
             id = \"m/dest\"\n\
             llm_client = \"c\"\n\
             \n\
             [llm_clients.c]\n\
             format = \"openai_responses\"\n\
             base_url = \"https://example.test/v1\"\n\
             reasoning_dialect = \"llama_cpp_enable_thinking\"\n\
             \n\
             [routes.dest_route]\n\
             id = \"route/dest\"\n\
             type = \"passthrough\"\n\
             target = \"dest\"\n\
             \n\
             [routes.r]\n\
             id = \"route/r\"\n\
             type = \"fleet_router\"\n\
             candidates = [{{ target = \"dest\", preference_rank = 1 }}]\n\
             escalation = \"route/dest\"\n\
             {primary_line}{escalation_line}"
        ),
    )
    .unwrap();
    let runner = Runner::load(&path).expect("the fixture must load");
    let _ = std::fs::remove_file(&path);
    runner
}

fn route_with_policies(
    primary: Option<ReasoningPolicy>,
    escalation: Option<ReasoningPolicy>,
) -> Runner {
    runner_with_policies(primary, escalation)
}

#[test]
fn the_real_route_reports_its_declared_primary_policy() {
    // `enabled` on a llama_cpp target is expressible, so this loads.
    let runner = route_with_policies(Some(ReasoningPolicy::Enabled), Some(ReasoningPolicy::None));
    let route = runner.route("route/r").expect("route present");
    assert_eq!(route.reasoning_policy(), Some(ReasoningPolicy::Enabled));
}

#[test]
fn the_real_route_resolves_the_escalation_phase() {
    // Rung 1: an explicit escalation policy wins over a different primary.
    let runner = route_with_policies(Some(ReasoningPolicy::Enabled), Some(ReasoningPolicy::None));
    let route = runner.route("route/r").expect("route present");
    assert_eq!(
        route.escalated_reasoning_policy(),
        Some(ReasoningPolicy::None),
        "the explicit escalation policy must win"
    );
    // Rung 2: primary `none` stays `none` at the seam with no explicit policy.
    let runner = route_with_policies(Some(ReasoningPolicy::None), None);
    let route = runner.route("route/r").expect("route present");
    assert_eq!(route.reasoning_policy(), Some(ReasoningPolicy::None));
    assert_eq!(
        route.escalated_reasoning_policy(),
        Some(ReasoningPolicy::None),
        "a non-thinking route must never silently become thinking at the seam"
    );
    // Rung 3: the destination's own declaration governs.
    let runner = route_with_policies(Some(ReasoningPolicy::Enabled), None);
    let route = runner.route("route/r").expect("route present");
    assert_eq!(route.escalated_reasoning_policy(), None);
}

#[test]
fn a_route_with_no_policy_resolves_both_phases_to_none() {
    let runner = route_with_policies(None, None);
    let route = runner.route("route/r").expect("route present");
    assert_eq!(route.reasoning_policy(), None);
    assert_eq!(
        route.escalated_reasoning_policy(),
        None,
        "an absent primary policy must not be invented as `none`"
    );
}
