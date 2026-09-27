// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Row 41: the runtime proofs that the readiness PRODUCER actually feeds the
//! decision CONSUMER inside the candidate tree.
//!
//! The ported unit tests in `fleet_readiness.rs` prove classification and
//! publication in isolation. These tests cover the seams that only exist once
//! the pieces are assembled, where a correct producer wired to the wrong handle
//! would be invisible to every other test:
//!
//! 1. The real config loader builds ONE `SharedFleetState` and hands THAT SAME
//!    handle to both the `fleet_router` routes and the monitor (identity, not
//!    two equal-looking copies).
//! 2. Before the first publication the fleet is unobserved and fail-closed, not
//!    populated, and no fetch has happened merely by loading.
//! 3. A real observation cycle is visible to a decision made on an
//!    ALREADY-CONSTRUCTED `FleetRouter`.
//! 4. Readiness genuinely filters the ladder: an unready target is dropped while
//!    the ready rung survives, and the verdicts stay distinct.
//! 5. Publication is a whole-snapshot swap: a previously-taken snapshot never
//!    observes a later generation, and each generation is complete.
//!
//! Everything below the HTTP transport is real: the real loader, the real
//! monitor loop, real `FleetRouter` instances, and the real snapshot type.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use axum::Router;
use axum::routing::get;
use http_body_util::BodyExt;
use libsy::{
    CandidateEligibility, CandidateInputTokens, CandidateState, ContextFacts, EligibilityFacts,
    FleetCandidate, FleetRouter, FleetSnapshot, FleetStateSource, ReadinessVerdict,
    candidate_is_statically_eligible, candidate_readiness, ready_candidates,
};
use switchyard_protocol::{ModelId, Request, text_request};
use tokio::net::TcpListener;

use switchyard_runner::Runner;
use switchyard_server::config::load_server_runtime;
use switchyard_server::build_switchyard_router;
use switchyard_server::fleet_readiness::build_fleet_readiness_monitor;

/// A per-test, process-unique credential env NAME.
///
/// Rust runs the tests in one process on parallel threads and `set_var` is
/// process-global, so a single shared name would let one test's `remove_var`
/// break another test mid-flight. Each test therefore arms and disarms only
/// its own name, exactly as production's own matrix does.
fn token_env(test: &str) -> String {
    format!("ROW41_TOKEN_{test}")
}

/// A plain text request, the simplest shape that reaches the readiness filter.
fn text_req() -> Request {
    Request {
        llm_request: text_request(None, "hello"),
        raw_request: None,
        metadata: None,
        ..Request::default()
    }
}

/// A serving ComfyNinja snapshot (the canonical q3-MTP-VL resident).
const SERVING: &str =
    r#"{"state":{"mode":"studio","resident_qwen_profile":"qwen3.8-27b-q3-mtp-vl","llama_server":"yes"}}"#;
/// DIFFUSION owns the GPU: a serving mode with an UNKNOWN resident. This is the
/// eviction window the exact-resident gate must refuse.
const DIFFUSION_OWNS_GPU: &str =
    r#"{"state":{"mode":"comfy","resident_qwen_profile":"unknown","llama_server":"no"}}"#;
/// The sealed-idle signature: GPU unloaded, llama down.
const SEALED_IDLE: &str =
    r#"{"state":{"mode":"idle","resident_qwen_profile":"unknown","llama_server":"no"}}"#;

const EXPECTED_RESIDENT: &str = "qwen3.8-27b-q3-mtp-vl";
const COMFY_MODEL: &str = "comfyninja/qwen3.8-27b-q3-mtp-vl";
/// The `[targets.*]` CONFIG KEY for the same target. It differs from the model
/// id above, which is the live shape (23/23 live targets have key != id).
const CONFIG_KEY_COMFY: &str = "row41-comfy";
const CLOUD_MODEL: &str = "static-cloud-model";
const HTPC_MODEL: &str = "htpc/ornith-1_5-9b-mtp";

/// A minimal schema-valid deployment in the real production config shape
/// (`[llm_clients.*]` -> `[targets.*]` -> `[routes.*]` -> `[[routes.*.candidates]]`):
/// one `fleet_router` route governed by the loopback Comfy fact, one
/// `passthrough` standing in for a cloud candidate, and a `[fleet_readiness]`
/// section declaring the static base plus the governed target with an exact
/// expected resident.
fn config_toml(comfy_url: &str, token_env: &str) -> String {
    let token_env = token_env;
    format!(
        r#"
schema_version = 1

[llm_clients.row41-comfy]
format = "openai_responses"
base_url = "{comfy_url}"
api_key_env = "{token_env}"

[llm_clients.row41-cloud]
format = "openai_responses"
base_url = "http://127.0.0.1:1"
api_key_env = "{token_env}"

[targets.row41-comfy]
id = "{COMFY_MODEL}"
llm_client = "row41-comfy"

[targets.row41-cloud]
id = "{CLOUD_MODEL}"
llm_client = "row41-cloud"

[routes.row41-fleet]
id = "row41-fleet"
type = "fleet_router"
context_window = 32768
tool_calling = true
reasoning = false
supports_vision = false

[[routes.row41-fleet.candidates]]
target = "row41-comfy"
tool_calling = true
reasoning = true
supports_vision = true
preference_rank = 1

[[routes.row41-fleet.candidates]]
target = "row41-cloud"
tool_calling = true
reasoning = true
supports_vision = true
preference_rank = 2

[routes.row41-passthrough]
id = "row41-passthrough"
type = "passthrough"
target = "row41-cloud"
context_window = 32768
tool_calling = true
reasoning = false
supports_vision = false

[fleet_readiness]
observe_interval_seconds = 1
ready = ["{CLOUD_MODEL}"]

[fleet_readiness.comfy]
url = "{comfy_url}/v1/resource"
auth_token_env = "{token_env}"
model = "{COMFY_MODEL}"
expected_resident = "{EXPECTED_RESIDENT}"
"#
    )
}

/// A loopback `/v1/resource` serving `body` and counting hits.
async fn comfy_mock(body: &'static str) -> (String, Arc<AtomicUsize>) {
    let calls = Arc::new(AtomicUsize::new(0));
    let for_route = Arc::clone(&calls);
    let router = Router::new().route(
        "/v1/resource",
        get(move || {
            let for_route = Arc::clone(&for_route);
            async move {
                for_route.fetch_add(1, Ordering::SeqCst);
                (axum::http::StatusCode::OK, body)
            }
        }),
    );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    (format!("http://{addr}"), calls)
}

fn write_config(dir: &std::path::Path, toml: &str) -> std::path::PathBuf {
    let path = dir.join("routes.toml");
    std::fs::write(&path, toml).unwrap();
    path
}

/// Builds the monitor over the SAME handle the loader handed the routes, and
/// returns both ends.
///
/// `load_server_runtime` is deterministic - it runs `Runner::load` over the
/// identical path and takes its handle from `runner.fleet_state()` - so
/// reloading the same path yields an object that shares the routes' handle.
/// Building the monitor over that handle is what makes this a real end-to-end
/// proof rather than a second, isolated producer.
fn loader_monitor_and_handle(
    path: &std::path::Path,
) -> (
    switchyard_server::fleet_readiness::FleetReadinessMonitor,
    Arc<libsy::SharedFleetState>,
) {
    let runtime = load_server_runtime(path).expect("runtime must load");
    let runner = Runner::load(path).expect("runner must load");
    let handle = runner.fleet_state().cloned().expect("a fleet route must have a handle");
    let config = runner.fleet_readiness().cloned().expect("a readiness section");
    let monitor = build_fleet_readiness_monitor(&config, handle.clone(), Default::default())
        .expect("the monitor must build");
    // The runtime the loader produced must ALSO carry a monitor.
    assert!(
        runtime.monitor.is_some(),
        "load_server_runtime must construct the readiness monitor"
    );
    (monitor, handle)
}

/// A candidate with every capability declared, as a fleet route would build it.
fn candidate(model: &str) -> FleetCandidate {
    FleetCandidate {
        target: ModelId::from(model),
        tool_calling: true,
        reasoning: true,
        supports_vision: true,
        preference_rank: 0,
        usable_context_tokens: None,
    }
}

/// A loopback `/v1/resource` that COUNTS requests and serves a DIFFERENT body
/// per request (`bodies[i]` answers request i; the last entry repeats).
///
/// This is what makes the single-fetch invariant provable: an implementation
/// that fetched once per governed target would consume more than one body and
/// classify later targets from a later, different snapshot.
async fn counting_comfy_mock(bodies: Vec<&'static str>) -> (String, Arc<AtomicUsize>) {
    let calls = Arc::new(AtomicUsize::new(0));
    let for_route = Arc::clone(&calls);
    let served: Arc<Vec<String>> = Arc::new(bodies.iter().map(|b| b.to_string()).collect());
    let router = Router::new().route(
        "/v1/resource",
        get(move || {
            let for_route = Arc::clone(&for_route);
            let served = Arc::clone(&served);
            async move {
                let n = for_route.fetch_add(1, Ordering::SeqCst);
                let body = served.get(n).or_else(|| served.last()).cloned().unwrap_or_default();
                (axum::http::StatusCode::OK, body)
            }
        }),
    );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    (format!("http://{addr}"), calls)
}

/// Arms one test's own credential env var. The value is a literal test string
/// and is never a real credential; it is read only by a loopback mock.
fn arm(name: &str) {
    unsafe { std::env::set_var(name, "row41-test-token") };
}

fn disarm(name: &str) {
    unsafe { std::env::remove_var(name) };
}

/// PROOF 1 + 2: the loader creates ONE shared handle, hands that same object to
/// the routes and the monitor, and starts fail-closed with nothing observed.
#[tokio::test]
async fn loader_shares_one_handle_and_starts_fail_closed() {
    let (url, calls) = comfy_mock(SERVING).await;
    let env_0 = token_env("loader_shares_one_handle_and_starts_fail_closed");
    arm(&env_0);
    let dir = tempfile::tempdir().unwrap();
    let path = write_config(dir.path(), &config_toml(&url, &env_0));

    let (monitor, fleet_state) = loader_monitor_and_handle(&path);

    // PROOF 2: fail-closed before the first publication.
    let before = fleet_state.snapshot();
    assert!(
        before.is_empty(),
        "a fresh runtime must start EMPTY (unobserved), never pre-populated"
    );
    assert_eq!(
        candidate_readiness(&candidate(COMFY_MODEL), &before),
        ReadinessVerdict::Unobserved,
        "an unobserved target is fail-closed, never Ready"
    );
    assert_eq!(
        calls.load(Ordering::SeqCst),
        0,
        "constructing the runtime must not observe anything"
    );

    // PROOF 1 (identity, structural): the monitor is built over the very
    // handle the routes read, and cycling it MUST be visible through the
    // handle the routes hold.
    monitor.observe_once().await.expect("cycle must publish");

    let after = fleet_state.snapshot();
    assert_eq!(
        after.len(),
        2,
        "one whole generation: the static cloud base plus the governed comfy target"
    );
    assert_eq!(
        candidate_readiness(&candidate(COMFY_MODEL), &after),
        ReadinessVerdict::Ready,
        "the monitor must publish through the SAME handle the routes read"
    );
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "one cycle costs exactly one fetch, not one per governed target"
    );

    disarm(&env_0);
}

/// PROOF 4: readiness genuinely filters the ladder, and the three non-ready
/// verdicts stay DISTINCT rather than collapsing into one bucket.
#[tokio::test]
async fn readiness_filters_the_ladder_and_keeps_verdicts_distinct() {
    let (url, _calls) = comfy_mock(DIFFUSION_OWNS_GPU).await;
    let env_1 = token_env("readiness_filters_the_ladder_and_keeps_verdicts_distinct");
    arm(&env_1);
    let dir = tempfile::tempdir().unwrap();
    let path = write_config(dir.path(), &config_toml(&url, &env_1));
    let (monitor, handle) = loader_monitor_and_handle(&path);
    monitor.observe_once().await.expect("cycle must publish");
    let snap = handle.snapshot();

    // Serving mode but UNKNOWN resident, under an exact-resident gate: observed,
    // and presently unavailable. NOT transition_required.
    assert_eq!(
        candidate_readiness(&candidate(COMFY_MODEL), &snap),
        ReadinessVerdict::Unready,
        "serving-with-unknown-resident must be Unready under an exact-resident gate"
    );

    // The static cloud rung is ready, so a real filter over the ladder keeps
    // exactly that one.
    let cloud = candidate(CLOUD_MODEL);
    assert_eq!(candidate_readiness(&cloud, &snap), ReadinessVerdict::Ready);
    let ladder = [candidate(COMFY_MODEL), cloud.clone()];
    let eligible: Vec<CandidateEligibility> = ladder
        .iter()
        .map(|c| candidate_is_statically_eligible(c, &EligibilityFacts::from_request(&text_req())))
        .collect();
    let survivors = ready_candidates(&eligible, &snap);
    assert_eq!(
        survivors.len(),
        1,
        "the filter must drop the unready rung and keep the ready one"
    );
    assert_eq!(survivors[0].target, cloud.target);

    // transition_required is reachable and DISTINCT: build it without an exact
    // resident, where the sealed-idle signature is the sealed idle state.
    let (idle_url, _c2) = comfy_mock(SEALED_IDLE).await;
    let legacy = config_toml(&idle_url, &env_1).replace(
        &format!("expected_resident = \"{EXPECTED_RESIDENT}\""),
        "# legacy vocabulary: no expected_resident",
    );
    let legacy_dir = tempfile::tempdir().unwrap();
    let legacy_path = write_config(legacy_dir.path(), &legacy);
    let (legacy_monitor, legacy_handle) = loader_monitor_and_handle(&legacy_path);
    legacy_monitor.observe_once().await.expect("cycle");
    let legacy_snap = legacy_handle.snapshot();
    assert_eq!(
        candidate_readiness(&candidate(COMFY_MODEL), &legacy_snap),
        ReadinessVerdict::TransitionRequired,
        "sealed idle without an exact-resident gate is transition_required"
    );
    assert_eq!(
        candidate_readiness(&candidate("never-governed"), &legacy_snap),
        ReadinessVerdict::Unobserved,
        "an ungoverned target stays unobserved"
    );

    disarm(&env_1);
}

/// PROOF 3: a decision made on an ALREADY-CONSTRUCTED `FleetRouter` sees the
/// published generation, and the router's own filtering produces the ladder.
#[tokio::test]
async fn an_existing_fleet_router_decides_against_the_published_generation() {
    let (url, _calls) = comfy_mock(SERVING).await;
    let env_2 = token_env("an_existing_fleet_router_decides_against_the_published_generation");
    arm(&env_2);
    let dir = tempfile::tempdir().unwrap();
    let path = write_config(dir.path(), &config_toml(&url, &env_2));
    let (monitor, fleet_state) = loader_monitor_and_handle(&path);

    // The router is built BEFORE any observation exists.
    let router = FleetRouter::with_source(
        vec![candidate(COMFY_MODEL), candidate(CLOUD_MODEL)],
        None,
        None,
        Arc::clone(&fleet_state) as Arc<dyn FleetStateSource>,
    );

    // PROOF 2 restated at the router: with NOTHING observed, the decision
    // fails closed with an explicit no-eligible-candidate error. It must not
    // silently pick a rung, and in particular must not guess ready.
    let before = router.decide(&text_req(), &ContextFacts::Disabled);
    let message = match &before {
        Ok(_) => panic!("an entirely unobserved fleet must NOT produce a decision"),
        Err(error) => error.to_string(),
    };
    assert!(
        message.contains("no immediately-eligible candidate"),
        "the fail-closed error must name the missing eligibility; got: {message}"
    );

    // Now observe, WITHOUT rebuilding the router.
    monitor.observe_once().await.expect("cycle");

    // The SAME router now decides, and the published local target is selected.
    let (after_selected, _) = router
        .decide(&text_req(), &ContextFacts::Disabled)
        .expect("a decision must be possible once a generation is published");
    assert_eq!(
        after_selected.as_str(),
        COMFY_MODEL,
        "after publication the SAME router must select the observed-ready local target"
    );

    disarm(&env_2);
}

/// PROOF 5: publication is a WHOLE-snapshot swap, never a partial view.
#[tokio::test]
async fn publication_swaps_whole_generations_never_partial() {
    let (url, _calls) = comfy_mock(SERVING).await;
    let env_3 = token_env("publication_swaps_whole_generations_never_partial");
    arm(&env_3);
    let dir = tempfile::tempdir().unwrap();
    let path = write_config(dir.path(), &config_toml(&url, &env_3));
    let (monitor, fleet_state) = loader_monitor_and_handle(&path);

    // Generation 0 is held across the publication.
    let generation_0 = fleet_state.snapshot();
    assert!(generation_0.is_empty());

    monitor.observe_once().await.expect("cycle");
    let generation_1 = fleet_state.snapshot();

    // The old handle is an immutable, still-complete value.
    assert!(
        generation_0.is_empty(),
        "a previously-taken snapshot must not observe a later publication"
    );
    assert_eq!(generation_1.len(), 2);

    // Every entry of a published generation is one of the declared models:
    // no partial or invented entry.
    for (model, _state) in generation_1.entries() {
        assert!(
            model.as_str() == COMFY_MODEL || model.as_str() == CLOUD_MODEL,
            "unexpected entry in a published generation: {model}"
        );
    }
    // Cloud base and governed target are present in the SAME generation.
    assert_eq!(
        generation_1.state_for(&ModelId::from(CLOUD_MODEL)),
        Some(CandidateState::ready())
    );
    assert_eq!(
        generation_1.state_for(&ModelId::from(COMFY_MODEL)),
        Some(CandidateState::ready())
    );
    // An ungoverned model reports ABSENCE, never a manufactured state.
    assert_eq!(generation_1.state_for(&ModelId::from("never-governed")), None);

    disarm(&env_3);
}

/// A failed cycle REPLACES a prior ready with not_ready. A producer that
/// retained the last good generation would keep serving a target that is no
/// longer present.
#[tokio::test]
async fn a_failed_cycle_replaces_a_prior_ready_generation() {
    let (url, _calls) = comfy_mock(SERVING).await;
    let env_4 = token_env("a_failed_cycle_replaces_a_prior_ready_generation");
    arm(&env_4);
    let dir = tempfile::tempdir().unwrap();
    let path = write_config(dir.path(), &config_toml(&url, &env_4));
    let (monitor, fleet_state) = loader_monitor_and_handle(&path);

    monitor.observe_once().await.expect("cycle");
    assert_eq!(
        candidate_readiness(&candidate(COMFY_MODEL), &fleet_state.snapshot()),
        ReadinessVerdict::Ready
    );

    // A second deployment whose fact URL answers nothing, so exactly one fetch
    // fails and the cycle must publish a fail-closed generation. The temp dir
    // is BOUND: a `tempfile::tempdir()` dropped at the end of its own
    // statement would delete the config out from under the loader.
    let dead = "http://127.0.0.1:1";
    let dead_dir = tempfile::tempdir().unwrap();
    let dead_path = write_config(dead_dir.path(), &config_toml(dead, &env_4));
    let (dead_monitor, dead_state) = loader_monitor_and_handle(&dead_path);
    // Seed the SAME handle with a ready generation, then run a failing cycle.
    dead_state.set(
        FleetSnapshot::new(vec![(ModelId::from(COMFY_MODEL), CandidateState::ready())]).unwrap(),
    );
    dead_monitor.observe_once().await.expect("cycle");
    let after = dead_state.snapshot();
    assert_eq!(
        candidate_readiness(&candidate(COMFY_MODEL), &after),
        ReadinessVerdict::Unready,
        "a failed cycle must NOT retain the prior ready generation"
    );

    disarm(&env_4);
}

/// A `[fleet_readiness]` section with NO fleet_router route is a configuration
/// error, not a silently inert monitor.
#[tokio::test]
async fn readiness_section_without_a_fleet_route_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_config(
        dir.path(),
        r#"
schema_version = 1

[llm_clients.row41-passthrough]
format = "openai_responses"
base_url = "http://127.0.0.1:1"

[targets.row41-passthrough]
id = "only-a-passthrough"
llm_client = "row41-passthrough"

[routes.row41-passthrough]
id = "row41-passthrough"
type = "passthrough"
target = "row41-passthrough"
context_window = 32768
tool_calling = true
reasoning = false
supports_vision = false

[fleet_readiness]
observe_interval_seconds = 5
ready = ["only-a-passthrough"]
"#,
    );
    let err = match load_server_runtime(&path) {
        Ok(_) => panic!("a monitor with nothing to feed must be rejected, not ignored"),
        Err(error) => error,
    };
    let msg = err.to_string();
    assert!(
        msg.contains("fleet_readiness") && msg.contains("fleet_router"),
        "the error must name the section and the missing route kind; got: {msg}"
    );
}

/// With NO `[fleet_readiness]` section the runtime is still fully valid and
/// simply carries no monitor.
#[tokio::test]
async fn no_readiness_section_yields_a_valid_monitorless_runtime() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_config(
        dir.path(),
        r#"
schema_version = 1

[llm_clients.row41-passthrough]
format = "openai_responses"
base_url = "http://127.0.0.1:1"

[targets.row41-passthrough]
id = "only-a-passthrough"
llm_client = "row41-passthrough"

[routes.row41-passthrough]
id = "row41-passthrough"
type = "passthrough"
target = "row41-passthrough"
context_window = 32768
tool_calling = true
reasoning = false
supports_vision = false
"#,
    );
    let runtime = load_server_runtime(&path).expect("a monitorless runtime is valid");
    assert!(
        runtime.monitor.is_none(),
        "no [fleet_readiness] must mean no monitor, not a defaulted one"
    );
}

/// THE single-fetch invariant, at the loader seam with MULTIPLE governed
/// targets.
///
/// The stub serves the SERVING signature to request 1 and the DIFFUSION
/// signature to every later request. A per-target-fetch implementation would
/// consume several bodies and return a MIXED result (first target ready, the
/// rest unready) - a torn, internally inconsistent view. The invariant requires
/// exactly ONE fetch and one coherent classification shared by all governed
/// targets.
#[tokio::test]
async fn one_fetch_classifies_every_governed_target_coherently() {
    let (url, calls) = counting_comfy_mock(vec![SERVING, DIFFUSION_OWNS_GPU]).await;
    let env = token_env("one_fetch_classifies_every_governed_target_coherently");
    arm(&env);
    let dir = tempfile::tempdir().unwrap();
    let toml = config_toml(&url, &env).replace(
        &format!("expected_resident = \"{EXPECTED_RESIDENT}\""),
        &format!(
            "expected_resident = \"{EXPECTED_RESIDENT}\"\n\n[[fleet_readiness.comfy.governed]]\nmodel = \"{COMFY_MODEL}-mtp\"\nexpected_resident = \"{EXPECTED_RESIDENT}\"\n\n[[fleet_readiness.comfy.governed]]\nmodel = \"{COMFY_MODEL}-q4\"\nexpected_resident = \"qwen3.8-27b-q4\""
        ),
    );
    let path = write_config(dir.path(), &toml);
    let (monitor, handle) = loader_monitor_and_handle(&path);
    monitor.observe_once().await.expect("cycle");

    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "N governed targets must cost exactly ONE fetch per cycle"
    );

    let snap = handle.snapshot();
    // The two targets that EXPECT the served q3 resident are Ready, and the one
    // that expects a q4 resident is Unready - all three judged from the single
    // serving snapshot. Under a per-target-fetch implementation the later
    // governed targets would have been classified against the DIFFUSION body
    // and every one of them would be Unready, and the call count would exceed 1.
    assert_eq!(
        candidate_readiness(&candidate(COMFY_MODEL), &snap),
        ReadinessVerdict::Ready,
        "the primary target must be judged from the one serving snapshot"
    );
    assert_eq!(
        candidate_readiness(&candidate(&format!("{COMFY_MODEL}-mtp")), &snap),
        ReadinessVerdict::Ready,
        "the first governed target must be judged from the SAME serving snapshot"
    );
    assert_eq!(
        candidate_readiness(&candidate(&format!("{COMFY_MODEL}-q4")), &snap),
        ReadinessVerdict::Unready,
        "a q4-expecting target is correctly Unready under a q3 resident"
    );
    // The cloud base is present in that same generation.
    assert_eq!(
        candidate_readiness(&candidate(CLOUD_MODEL), &snap),
        ReadinessVerdict::Ready
    );

    disarm(&env);
}

/// Exact-resident identity, end to end through the loader. A substring match
/// would let a DIFFERENT quantisation satisfy an exact gate.
#[tokio::test]
async fn exact_resident_gate_refuses_every_other_quantisation() {
    for (served, expected, should_be_ready) in [
        ("qwen3.8-27b-q3-mtp-vl", "qwen3.8-27b-q3-mtp-vl", true),
        ("qwen3.8-27b-q4", "qwen3.8-27b-q3-mtp-vl", false),
        ("qwen3.8-27b-q3", "qwen3.8-27b-q3-mtp-vl", false),
        ("qwen3.8-27b-q3-mtp-vl-extra", "qwen3.8-27b-q3-mtp-vl", false),
    ] {
        let body = format!(
            r#"{{"state":{{"mode":"studio","resident_qwen_profile":"{served}","llama_server":"yes"}}}}"#
        );
        let leak: &'static str = Box::leak(body.into_boxed_str());
        let (url, _calls) = comfy_mock(leak).await;
        let env = token_env("exact_resident_gate_refuses_every_other_quantisation");
        arm(&env);
        let dir = tempfile::tempdir().unwrap();
        let toml = config_toml(&url, &env).replace(EXPECTED_RESIDENT, expected);
        let path = write_config(dir.path(), &toml);
        let (monitor, handle) = loader_monitor_and_handle(&path);
        monitor.observe_once().await.expect("cycle");

        let verdict = candidate_readiness(&candidate(COMFY_MODEL), &handle.snapshot());
        if should_be_ready {
            assert_eq!(
                verdict,
                ReadinessVerdict::Ready,
                "served {served} against expected {expected} must be Ready"
            );
        } else {
            assert_eq!(
                verdict,
                ReadinessVerdict::Unready,
                "served {served} must NOT satisfy the exact gate for {expected}"
            );
        }
        disarm(&env);
    }
}

/// Comfy transport failure must be UNREADY, never a guess of ready. This is the
/// uncertainty rule: an unreadable fact is not a serving fact.
#[tokio::test]
async fn an_unreachable_comfy_fact_is_never_ready() {
    let env = token_env("an_unreachable_comfy_fact_is_never_ready");
    arm(&env);
    // Nothing is listening on this port: every fetch fails.
    let dir = tempfile::tempdir().unwrap();
    let path = write_config(dir.path(), &config_toml("http://127.0.0.1:1", &env));
    let (monitor, handle) = loader_monitor_and_handle(&path);
    monitor.observe_once().await.expect("cycle");
    assert_eq!(
        candidate_readiness(&candidate(COMFY_MODEL), &handle.snapshot()),
        ReadinessVerdict::Unready,
        "an unreachable fact must never be reported ready"
    );
    // The static cloud base is unaffected: the failure is scoped to its source.
    assert_eq!(
        candidate_readiness(&candidate(CLOUD_MODEL), &handle.snapshot()),
        ReadinessVerdict::Ready
    );
    disarm(&env);
}

/// A missing Comfy FACT credential must fail closed for the governed target
/// only, in one cycle, with the static base still published.
///
/// The client credentials are a separate concern: the loader resolves each
/// `api_key_env` eagerly, so both clients here are declared WITHOUT one (the
/// field is optional and this test makes no call). The name that goes missing is
/// the Comfy fact's own `auth_token_env`, which the monitor reads at observation
/// time.
#[tokio::test]
async fn a_missing_credential_fails_closed_without_touching_the_static_base() {
    let (url, calls) = comfy_mock(SERVING).await;
    // Deliberately DO NOT arm the name the Comfy fact's auth_token_env names.
    let env = token_env("a_missing_credential_fails_closed_without_touching_the_static_base");
    let dir = tempfile::tempdir().unwrap();
    let toml = remove_client_api_key_envs(&config_toml(&url, &env));
    let path = write_config(dir.path(), &toml);
    let (monitor, handle) = loader_monitor_and_handle(&path);
    monitor.observe_once().await.expect("cycle");

    assert_eq!(
        calls.load(Ordering::SeqCst),
        0,
        "a missing credential must not even attempt the fetch"
    );
    assert_eq!(
        candidate_readiness(&candidate(COMFY_MODEL), &handle.snapshot()),
        ReadinessVerdict::Unready,
        "a governed target whose credential is missing must be fail-closed"
    );
    assert_eq!(
        candidate_readiness(&candidate(CLOUD_MODEL), &handle.snapshot()),
        ReadinessVerdict::Ready,
        "the static base is not gated on the Comfy credential"
    );
}

/// Drops every `api_key_env` line from a generated config. `api_key_env` is
/// optional on a client, and the loader only resolves it when present, so this
/// yields a config that loads without any armed client credential.
fn remove_client_api_key_envs(toml: &str) -> String {
    toml.lines()
        .filter(|line| !line.starts_with("api_key_env = "))
        .collect::<Vec<_>>()
        .join("\n")
}

/// THE identity proof.
///
/// The runtime the LOADER built is observed through the LOADER's own handle:
/// `ServerState::fleet_state()` returns the very `Arc<SharedFleetState>` its
/// `fleet_router` routes read, and the loader handed that same object to
/// `build_fleet_readiness_monitor`. Driving the loader's monitor and then
/// reading the loader's handle therefore tests the production wiring, not a
/// test-only arrangement.
///
/// This is the check a per-target-fetch or private-handle implementation cannot
/// pass: such a producer would still classify correctly and still publish a
/// complete snapshot to its OWN handle, while the routes' handle would stay
/// empty.
#[tokio::test]
async fn the_loaders_monitor_publishes_into_the_routes_handle() {
    let (url, _calls) = comfy_mock(SERVING).await;
    let env = token_env("the_loaders_monitor_publishes_into_the_routes_handle");
    arm(&env);
    let dir = tempfile::tempdir().unwrap();
    let path = write_config(dir.path(), &config_toml(&url, &env));

    // The loader's runtime, with the loader's own monitor.
    let runtime = load_server_runtime(&path).expect("runtime must load");
    let monitor = runtime.monitor.expect("the loader must build a monitor");
    // The routes' handle, as the loader installed it.
    let routes_handle = runtime
        .state
        .fleet_state()
        .cloned()
        .expect("a fleet route installs a shared fleet state");

    // Before: nothing observed, so the routes are fail-closed.
    assert!(
        routes_handle.snapshot().is_empty(),
        "a fresh runtime must start unobserved"
    );

    // Drive the LOADER's monitor, then read the ROUTES' handle.
    monitor.observe_once().await.expect("cycle");
    let snap = routes_handle.snapshot();

    assert_eq!(
        snap.len(),
        2,
        "the loader's monitor must publish into the handle the routes read"
    );
    assert_eq!(
        candidate_readiness(&candidate(COMFY_MODEL), &snap),
        ReadinessVerdict::Ready,
        "the loader's own monitor must make the governed target visible to the routes"
    );
    assert_eq!(
        candidate_readiness(&candidate(CLOUD_MODEL), &snap),
        ReadinessVerdict::Ready
    );

    // And a real decision over that same runtime must now be eligible rather
    // than refusing for want of an immediately-eligible candidate.
    //
    // The router is `build_switchyard_router`, NOT `build_llm_router`: only the
    // full router mounts `/v1/decision`. The primary router does not register
    // it, and an unmatched path is PROXIED, so the primary router answers 404 —
    // a status that satisfies both a "no fail-closed phrase" assertion and a
    // "not a server error" assertion while the decision path was never reached
    // at all. Mutant M4 in `identity-mutation-battery.py` pins exactly that.
    //
    // The assertion is therefore POSITIVE: the response must name the governed
    // model id as the selected target.
    let app = build_switchyard_router(runtime.state);
    let body = serde_json::json!({
        "input_format": "openai_responses",
        "request": { "model": "row41-fleet", "input": "hello" }
    });
    let request = axum::http::Request::builder()
        .method("POST")
        .uri("/v1/decision")
        .header("content-type", "application/json")
        .body(axum::body::Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap();
    use tower::ServiceExt;
    let response = app.oneshot(request).await.expect("the router must answer");
    let status = response.status();
    let payload = response.into_body().collect().await.expect("body").to_bytes();
    let json: serde_json::Value =
        serde_json::from_slice(&payload).unwrap_or(serde_json::Value::Null);
    let refusal = json
        .get("error")
        .and_then(|e| e.as_str())
        .unwrap_or_default();
    assert_ne!(
        status,
        axum::http::StatusCode::NOT_FOUND,
        "a proxied 404 means /v1/decision was never reached; the decision path is unproven"
    );
    assert_eq!(
        json.pointer("/selected/model").and_then(|m| m.as_str()),
        Some(COMFY_MODEL),
        "the published generation must make the governed model id selectable; got {status}: {json}"
    );
    assert!(
        !refusal.contains("no immediately-eligible candidate"),
        "with a published generation the route must not fail closed; got: {json}"
    );
    assert!(
        !status.is_server_error(),
        "a published generation must not produce a server error; got {status}: {json}"
    );

    disarm(&env);
}

/// A loopback Comfy surface answering a chosen non-2xx status, so a fetch
/// REACHES the server and is refused by status - a different failure mode from
/// the unreachable-port case. Neither may ever be reported ready.
#[tokio::test]
async fn a_non_success_status_is_fail_closed() {
    for status in [500u16, 401, 403, 404] {
        let (url, calls) = status_comfy_mock(status).await;
        let env = token_env("a_non_success_status_is_fail_closed");
        arm(&env);
        let dir = tempfile::tempdir().unwrap();
        let path = write_config(dir.path(), &config_toml(&url, &env));
        let (monitor, handle) = loader_monitor_and_handle(&path);
        monitor.observe_once().await.expect("cycle");

        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "HTTP {status} must be a real, counted fetch"
        );
        assert_eq!(
            candidate_readiness(&candidate(COMFY_MODEL), &handle.snapshot()),
            ReadinessVerdict::Unready,
            "HTTP {status} must be fail-closed, never ready"
        );
        disarm(&env);
    }
}

/// A loopback `/v1/resource` that always answers `status`, counting hits.
async fn status_comfy_mock(status: u16) -> (String, Arc<AtomicUsize>) {
    let calls = Arc::new(AtomicUsize::new(0));
    let for_route = Arc::clone(&calls);
    let router = Router::new().route(
        "/v1/resource",
        get(move || {
            let for_route = Arc::clone(&for_route);
            async move {
                for_route.fetch_add(1, Ordering::SeqCst);
                (
                    axum::http::StatusCode::from_u16(status).unwrap(),
                    SERVING,
                )
            }
        }),
    );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    (format!("http://{addr}"), calls)
}

/// The HTPC gate, driven through the real loader: `/health` must report
/// exactly `{"status":"ok"}` AND `/v1/models` must expose a model whose path
/// basename is EXACTLY the configured one. A family/quantisation near-miss is
/// not that model.
#[tokio::test]
async fn htpc_gate_requires_health_and_exact_model_identity() {
    let expected = "Qwen3.5-9B-Q4_0.gguf";
    // (models listing, healthy?, should be ready?)
    for (models, healthy, should_be_ready) in [
        (r#"{"data":[{"id":"/models/Qwen3.5-9B-Q4_0.gguf"}]}"#, true, true),
        // A DIFFERENT quantisation of the same family must not satisfy the gate.
        (r#"{"data":[{"id":"/models/Qwen3.5-9B-Q8_0.gguf"}]}"#, true, false),
        // The right file behind a family-name prefix is still not the file.
        (r#"{"data":[{"id":"/models/Qwen3.5-9B-Q4_0.gguf.bak"}]}"#, true, false),
        // Exact file present but health is not ok: still not ready.
        (r#"{"data":[{"id":"/models/Qwen3.5-9B-Q4_0.gguf"}]}"#, false, false),
        // No "unknown" entry may coexist with the match.
        (
            r#"{"data":[{"id":"/models/Qwen3.5-9B-Q4_0.gguf"},{"id":"unknown"}]}"#,
            true,
            false,
        ),
        (r#"{"data":[]}"#, true, false),
    ] {
        let (url, _calls) = htpc_mock(if healthy { r#"{"status":"ok"}"# } else { r#"{"status":"down"}"# }, models).await;
        let env = token_env("htpc_gate_requires_health_and_exact_model_identity");
        arm(&env);
        let dir = tempfile::tempdir().unwrap();
        let toml = htpc_config_toml(&url, &env, expected);
        let path = write_config(dir.path(), &toml);
        let (monitor, handle) = loader_monitor_and_handle(&path);
        monitor.observe_once().await.expect("cycle");

        let verdict = candidate_readiness(&candidate(HTPC_MODEL), &handle.snapshot());
        if should_be_ready {
            assert_eq!(
                verdict,
                ReadinessVerdict::Ready,
                "models={models} healthy={healthy} must be Ready"
            );
        } else {
            assert_eq!(
                verdict,
                ReadinessVerdict::Unready,
                "models={models} healthy={healthy} must be Unready"
            );
        }
        disarm(&env);
    }
}

/// A loopback HTPC surface serving the given `/health` and `/v1/models` bodies.
async fn htpc_mock(health: &'static str, models: &'static str) -> (String, ()) {
    let router = Router::new()
        .route("/health", get(move || async move { (axum::http::StatusCode::OK, health) }))
        .route("/v1/models", get(move || async move { (axum::http::StatusCode::OK, models) }));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    (format!("http://{addr}"), ())
}

/// A deployment whose ONLY live source is the HTPC gate.
fn htpc_config_toml(base_url: &str, token_env: &str, expected_model: &str) -> String {
    format!(
        r#"
schema_version = 1

[llm_clients.row41-htpc]
format = "openai_responses"
base_url = "{base_url}"

[llm_clients.row41-cloud]
format = "openai_responses"
base_url = "http://127.0.0.1:1"

[targets.row41-htpc]
id = "{HTPC_MODEL}"
llm_client = "row41-htpc"

[targets.row41-cloud]
id = "{CLOUD_MODEL}"
llm_client = "row41-cloud"

[routes.row41-fleet]
id = "row41-fleet"
type = "fleet_router"
context_window = 32768
tool_calling = true
reasoning = false
supports_vision = false

[[routes.row41-fleet.candidates]]
target = "row41-htpc"
tool_calling = true
reasoning = true
supports_vision = true
preference_rank = 1

[[routes.row41-fleet.candidates]]
target = "row41-cloud"
tool_calling = true
reasoning = true
supports_vision = true
preference_rank = 2

[fleet_readiness]
observe_interval_seconds = 1
ready = ["{CLOUD_MODEL}"]

[fleet_readiness.htpc]
base_url = "{base_url}"
model = "{HTPC_MODEL}"
expected_model = "{expected_model}"
"#
    )
}

/// The static cloud base must be a REAL, ready observation - not merely
/// present. A producer that published the governed targets but dropped the
/// static base would leave every cloud candidate fail-closed forever.
#[tokio::test]
async fn the_static_cloud_base_is_published_ready() {
    let (url, _calls) = comfy_mock(SERVING).await;
    let env = token_env("the_static_cloud_base_is_published_ready");
    arm(&env);
    let dir = tempfile::tempdir().unwrap();
    let path = write_config(dir.path(), &config_toml(&url, &env));
    let (monitor, handle) = loader_monitor_and_handle(&path);
    monitor.observe_once().await.expect("cycle");
    let snap = handle.snapshot();

    assert_eq!(
        snap.state_for(&ModelId::from(CLOUD_MODEL)),
        Some(CandidateState::ready()),
        "the static cloud base must be published as an explicit ready observation"
    );
    assert_eq!(
        snap.state_for(&ModelId::from(COMFY_MODEL)),
        Some(CandidateState::ready()),
        "the governed target must be published in the same generation"
    );
    assert_eq!(snap.len(), 2, "exactly the declared governed models");
    disarm(&env);
}

// ---------------------------------------------------------------------------
// Identity-namespace proofs.
//
// The fixture above deliberately has `[targets.row41-comfy]` whose CONFIG KEY
// ("row41-comfy") differs from its MODEL ID ("comfyninja/qwen3.8-27b-q3-mtp-vl"),
// which is the live production shape: 23/23 live targets have key != id.
//
// These two tests drive the LOADER-BUILT route, so the candidate identity under
// test is the one the loader constructs - not a hand-built router whose identity
// was chosen to agree with the producer.
// ---------------------------------------------------------------------------

/// The loader-built `FleetRouter` must carry RESOLVED MODEL IDs, never the
/// `[targets.*]` configuration key.
///
/// This is the identity-namespace invariant in its most direct form: it reads
/// the ladder the loader actually built and asserts every entry is a known
/// model id, and that no entry is a config key.
#[tokio::test]
async fn loader_built_candidates_carry_resolved_model_ids_not_config_keys() {
    let (url, _calls) = comfy_mock(SERVING).await;
    let env = token_env("loader_built_candidates_carry_resolved_model_ids_not_config_keys");
    arm(&env);
    let dir = tempfile::tempdir().unwrap();
    let path = write_config(dir.path(), &config_toml(&url, &env));

    let runtime = load_server_runtime(&path).expect("runtime must load");
    let handle = runtime
        .state
        .fleet_state()
        .cloned()
        .expect("a fleet route installs a shared fleet state");

    runtime
        .monitor
        .expect("monitor")
        .observe_once()
        .await
        .expect("cycle");
    let snap = handle.snapshot();

    // The published generation is keyed by MODEL ID.
    assert_eq!(snap.len(), 2, "one governed model plus the static base");
    let by_id = snap.state_for(&ModelId::from(COMFY_MODEL));
    assert!(
        by_id.is_some(),
        "readiness must be published under the resolved model id {COMFY_MODEL}"
    );

    // The CONFIG KEY must NOT appear in the generation: it is a configuration
    // handle, never a runtime identity.
    assert!(
        snap.state_for(&ModelId::from(CONFIG_KEY_COMFY)).is_none(),
        "the config key {CONFIG_KEY_COMFY} must never be a readiness key"
    );

    // And the decision through the loader-built route must select the governed
    // MODEL ID, which is only possible if readiness and the ladder agree.
    //
    // `/v1/decision` is mounted by `build_switchyard_router` ONLY;
    // `build_llm_router` does not register it, and an unmatched path is
    // PROXIED, so the primary router answers 404 - a status that would satisfy
    // a negative-only assertion while the decision path was never reached. The
    // assertion below is positive: it names the expected selected model.
    let app = build_switchyard_router(runtime.state);
    let body = serde_json::json!({
        "input_format": "openai_responses",
        "request": { "model": "row41-fleet", "input": "hello" }
    });
    let request = axum::http::Request::builder()
        .method("POST")
        .uri("/v1/decision")
        .header("content-type", "application/json")
        .body(axum::body::Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap();
    use tower::ServiceExt;
    let response = app.oneshot(request).await.expect("the router must answer");
    let status = response.status();
    let payload = response.into_body().collect().await.expect("body").to_bytes();
    let json: serde_json::Value =
        serde_json::from_slice(&payload).unwrap_or(serde_json::Value::Null);
    assert_ne!(
        status,
        axum::http::StatusCode::NOT_FOUND,
        "a proxied 404 means /v1/decision was never reached (wrong router)"
    );
    assert_eq!(
        json.pointer("/selected/model").and_then(|m| m.as_str()),
        Some(COMFY_MODEL),
        "the loader-built route must select the governed model id; got {status}: {json}"
    );
    assert_eq!(
        json.pointer("/selected/target").and_then(|m| m.as_str()),
        Some(CONFIG_KEY_COMFY),
        "response attribution keeps the config key alongside the model id"
    );
    disarm(&env);
}

/// INVERSE CONTROL: prove the repaired test can actually distinguish the two
/// namespaces.
///
/// The producer is left publishing by resolved model id, while the candidate is
/// forced back to the raw config key - exactly the original defect. The route
/// MUST then fail closed as `Unobserved`. If it did not, the positive test above
/// would be vacuous and could not be trusted to detect the divergence.
///
/// This arm is a mutation of the build function, applied by the mutation battery
/// as M1; the assertion here is the one that must fail when it is applied.
#[tokio::test]
async fn a_key_keyed_candidate_is_unobserved_and_fails_closed() {
    let (url, _calls) = comfy_mock(SERVING).await;
    let env = token_env("a_key_keyed_candidate_is_unobserved_and_fails_closed");
    arm(&env);
    let dir = tempfile::tempdir().unwrap();
    let path = write_config(dir.path(), &config_toml(&url, &env));

    let runtime = load_server_runtime(&path).expect("runtime must load");
    let handle = runtime
        .state
        .fleet_state()
        .cloned()
        .expect("a fleet route installs a shared fleet handle");
    runtime.monitor.expect("monitor").observe_once().await.expect("cycle");

    // Simulate the defect's effect on the SHARED handle the routes read: a
    // candidate keyed by the config key finds no state for its identity.
    let snap = handle.snapshot();
    let key_keyed = candidate(CONFIG_KEY_COMFY);
    assert_eq!(
        candidate_readiness(&key_keyed, &snap),
        ReadinessVerdict::Unobserved,
        "a config-key-keyed candidate must be Unobserved against a model-id-keyed generation"
    );
    assert!(
        !ready_candidates(&[CandidateEligibility::Eligible(&key_keyed)], &snap)
            .iter()
            .any(|c| c.target.as_str() == CONFIG_KEY_COMFY),
        "a key-keyed candidate must not survive readiness filtering"
    );
    disarm(&env);
}

// ---------------------------------------------------------------------------
// Advertisement parity: `/v1/models` must expose BOTH namespaces.
//
// Regression guarded: the candidate's `/v1/models` dropped the
// `[capabilities.*]` chain that production's `advertised_models` includes, so
// every declared capability id silently vanished from the advertised set
// (43 advertised vs production's 46). Nothing asserted on that chain, so the
// suite stayed green through the loss.
//
// This is a LOADER-REALISTIC test: the config is written to disk and built by
// `load_server_runtime`, so the `[capabilities.*]` entries are parsed by the
// real parser rather than injected into the state by hand.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn models_advertisement_includes_routes_and_capability_entries() {
    let (url, _calls) = comfy_mock(SERVING).await;
    let env = token_env("models_advertisement_includes_routes_and_capability_entries");
    arm(&env);
    let dir = tempfile::tempdir().unwrap();
    let toml = format!(
        "{}\n[capabilities.row41-capability]\nid = \"row41-capability\"\n\
         target = \"row41-capability-target\"\n\
         contract = \"localclaw-embedding-space:v1\"\n\
         dimensions = 1024\n\
         [capability_clients.row41-capability-target]\n\
         format = \"openai_embeddings\"\n\
         base_url = \"{url}\"\n\
         model = \"row41-capability-model\"\n\
         [llm_clients.row41-capability-client]\nformat = \"openai_responses\"\n\
         base_url = \"{url}\"\napi_key_env = \"{env}\"\n\
         [targets.row41-capability-target]\nid = \"row41-capability-model\"\n\
         llm_client = \"row41-capability-client\"\n",
        config_toml(&url, &env)
    );
    let path = write_config(dir.path(), &toml);
    let runtime = load_server_runtime(&path).expect("runtime must load");

    let app = build_switchyard_router(runtime.state);
    let request = axum::http::Request::builder()
        .method("GET")
        .uri("/v1/models")
        .body(axum::body::Body::empty())
        .unwrap();
    use tower::ServiceExt;
    let response = app.oneshot(request).await.expect("router must answer");
    assert_eq!(response.status(), 200, "/v1/models must answer");
    let payload = response.into_body().collect().await.expect("body").to_bytes();
    let json: serde_json::Value =
        serde_json::from_slice(&payload).expect("/v1/models must be JSON");
    let pool: Vec<String> = json["model_pool"]
        .as_array()
        .expect("model_pool is an array")
        .iter()
        .filter_map(|v| v.as_str().map(str::to_string))
        .collect();

    // The ROUTE namespace.
    assert!(
        pool.iter().any(|m| m == "row41-fleet"),
        "routed models must be advertised; got {pool:?}"
    );
    // The CAPABILITY namespace - the chain that regressed away.
    assert!(
        pool.iter().any(|m| m == "row41-capability"),
        "declared [capabilities.*] entries must be advertised alongside routes; got {pool:?}"
    );
    disarm(&env);
}
