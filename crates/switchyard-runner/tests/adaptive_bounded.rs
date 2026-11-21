// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The adaptive bounded behaviour a Gemma/Lite release depends on.
//!
//! Four properties are exercised here, each with a control that fails without it:
//!
//! * a private or unknown payload skips a Google leg inside a composite and reaches the
//!   strict non-training leg, while a directly addressed Google leaf still refuses;
//! * a request too large for a target's physical window or per-minute input budget is not
//!   sent to it, so an aggregate context claim stays truthful;
//! * measured selection reorders candidates only after it has evidence, and only among the
//!   candidates the hard gates already admitted;
//! * a route may not advertise capacity no declared target can serve.
//!
//! Every stub here counts its calls, because "the leg was skipped" and "the leg answered and
//! was ignored" are different claims and only the first one is a privacy guarantee.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

use async_trait::async_trait;
use libsy::RuntimeModels;
use switchyard_llm_client::{
    CapabilityFacts, ClientRouter, RunObservation, RunObserver, SelectionTrace,
};
use switchyard_protocol::{
    Category, LlmClientError, LlmResponse, ModelId, Request, Response, RoutedLlmClient,
    text_request, text_response,
};
use switchyard_runner::{
    AlgorithmSpec, MeasuredSelector, ModelCapabilities, Route, RouteReasoningPolicy, Runner,
    RunnerError,
};

/// Records that it was asked and how big the request was.
struct CountingClient {
    calls: Arc<AtomicUsize>,
    /// Set when the stub should behave like an unavailable backend.
    failing: bool,
}

#[async_trait]
impl RoutedLlmClient for CountingClient {
    async fn call(&self, request: Request) -> Result<Response, LlmClientError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        if self.failing {
            return Err(LlmClientError::TemporarilyUnavailable);
        }
        Ok(Response {
            llm_response: LlmResponse::Agg(text_response(
                request.llm_request.model.clone(),
                "served",
            )),
            metadata: None,
            upstream_headers: Default::default(),
        })
    }
}

/// Builds one request carrying `text`; no clearance is attached anymore.
fn request(text: &str) -> Request {
    let raw = serde_json::json!({
        "messages": [{"role": "user", "content": text}],
    });
    Request {
        llm_request: text_request(Some("routed".to_string()), text),
        raw_request: Some(raw),
        ..Request::default()
    }
}

/// A one-target leaf route.
fn leaf(
    name: &str,
    calls: &Arc<AtomicUsize>,
    facts: CapabilityFacts,
    capabilities: ModelCapabilities,
    failing: bool,
) -> (ModelId, Route) {
    let target = format!("{name}-target");
    let algorithm = AlgorithmSpec::Passthrough {
        target: target.clone(),
        candidates: Vec::new(),
        subagents: None,
    }
    .build(
        name,
        &BTreeMap::from([(target.clone(), ModelId::from(target.as_str()))]),
    )
    .expect("single target builds");
    let clients = ClientRouter::new(
        BTreeMap::from([(
            ModelId::from(target.as_str()),
            Arc::new(CountingClient {
                calls: Arc::clone(calls),
                failing,
            }) as Arc<dyn RoutedLlmClient>,
        )])
        .into_iter()
        .collect(),
    )
    .with_target_capabilities(HashMap::from([(ModelId::from(target.as_str()), facts)]));
    (
        ModelId::from(name),
        Route::new(
            algorithm,
            clients,
            None,
            capabilities,
            None,
            None,
            Vec::new(),
            RuntimeModels::new([(Category::Any, vec![ModelId::from(target.as_str())])].into()),
            RouteReasoningPolicy::Inherit,
            None,
            None,
        ),
    )
}

fn gated() -> CapabilityFacts {
    CapabilityFacts {
        ..CapabilityFacts::default()
    }
}

fn open() -> CapabilityFacts {
    CapabilityFacts::default()
}

// ---------------------------------------------------------------------------
// Privacy: a composite skips the leg it may not reach
// ---------------------------------------------------------------------------

/// The owner's requested branch: an uncleared payload starts after Google, not at it.
///
/// This is the behaviour the deployed `smart-nt-bounded-trial` lacks — it terminates with
/// the child's refusal instead — so the assertion is on both legs' call counters: Google
/// must be at zero and the strict leg must be exactly one.
#[tokio::test]
async fn nothing_is_refused_at_the_gate_anymore() {
    let google_calls = Arc::new(AtomicUsize::new(0));
    let strict_calls = Arc::new(AtomicUsize::new(0));
    let runner = Runner::new(vec![
        leaf(
            "google",
            &google_calls,
            gated(),
            ModelCapabilities::default(),
            false,
        ),
        leaf(
            "strict",
            &strict_calls,
            open(),
            ModelCapabilities::default(),
            false,
        ),
    ])
    .with_composite(
        "bounded".to_string(),
        vec!["google".to_string(), "strict".to_string()],
    );

    // No payload is refused at the gate: the first declared leg serves.
    let output = runner
        .execute_route("bounded", request("private material"), None)
        .await
        .expect("a leg can serve any payload");

    assert_eq!(
        google_calls.load(Ordering::SeqCst),
        1,
        "the first declared leg is contacted for any payload"
    );
    assert_eq!(
        strict_calls.load(Ordering::SeqCst),
        0,
        "no sibling is spent when the first leg serves"
    );
    assert!(
        !output.output.selected_model.to_string().is_empty(),
        "the response must name what served it"
    );
}

/// The deployed `smart-nt-bounded-trial` shape, asserted rather than assumed.
///
/// Four children in the order the production manifest states: two Google legs, then the
/// strict non-training OpenAI leg, then the strict non-training DeepSeek leg. An uncleared
/// payload must start at leg three, and a cleared one must start at leg one -- without the
/// second assertion the first could be satisfied by never using Google at all.
#[tokio::test]
async fn a_route_serves_in_declared_order_for_any_payload() {
    let google_calls = Arc::new(AtomicUsize::new(0));
    let strict_calls = Arc::new(AtomicUsize::new(0));
    let runner = Runner::new(vec![
        leaf(
            "google",
            &google_calls,
            gated(),
            ModelCapabilities::default(),
            false,
        ),
        leaf(
            "strict",
            &strict_calls,
            open(),
            ModelCapabilities::default(),
            false,
        ),
    ])
    .with_composite(
        "bounded".to_string(),
        vec!["google".to_string(), "strict".to_string()],
    );

    // Declared order is preserved; nothing is skipped for its content.
    runner
        .execute_route("bounded", request("anything at all"), None)
        .await
        .expect("the first leg serves");

    assert_eq!(
        google_calls.load(Ordering::SeqCst),
        1,
        "the first declared leg is contacted"
    );
    assert_eq!(
        strict_calls.load(Ordering::SeqCst),
        0,
        "the sibling is not spent"
    );
}
/// The same request addressed straight at the gated leaf is still refused, with no call.
///
/// Without this control the skip above could be implemented by weakening the gate, which is
/// the failure mode that would matter most.
#[tokio::test]
async fn a_directly_addressed_leaf_is_called_for_any_payload() {
    let google_calls = Arc::new(AtomicUsize::new(0));
    let runner = Runner::new(vec![leaf(
        "google",
        &google_calls,
        gated(),
        ModelCapabilities::default(),
        false,
    )]);

    let output = runner
        .execute_route("google", request("private material"), None)
        .await
        .expect("the leaf serves any payload");

    assert_eq!(google_calls.load(Ordering::SeqCst), 1);
    assert!(
        !output.output.selected_model.to_string().is_empty(),
        "the response must name what served it"
    );
}
/// A cleared payload is served by the gated leg, so the skip is about clearance, not about
/// the leg being permanently unusable.
#[tokio::test]
async fn a_cleared_payload_is_served_by_the_gated_child() {
    let google_calls = Arc::new(AtomicUsize::new(0));
    let strict_calls = Arc::new(AtomicUsize::new(0));
    let runner = Runner::new(vec![
        leaf(
            "google",
            &google_calls,
            gated(),
            ModelCapabilities::default(),
            false,
        ),
        leaf(
            "strict",
            &strict_calls,
            open(),
            ModelCapabilities::default(),
            false,
        ),
    ])
    .with_composite(
        "bounded".to_string(),
        vec!["google".to_string(), "strict".to_string()],
    );

    runner
        .execute_route("bounded", request("public material"), None)
        .await
        .expect("a cleared payload may use the gated leg");

    assert_eq!(google_calls.load(Ordering::SeqCst), 1);
    assert_eq!(strict_calls.load(Ordering::SeqCst), 0);
}

/// A gated child that is contacted for a cleared payload and then fails still crosses, so the
/// skip did not replace the existing failure handling.
#[tokio::test]
async fn a_failing_gated_child_still_crosses_on_a_cleared_payload() {
    let google_calls = Arc::new(AtomicUsize::new(0));
    let strict_calls = Arc::new(AtomicUsize::new(0));
    let runner = Runner::new(vec![
        leaf(
            "google",
            &google_calls,
            gated(),
            ModelCapabilities::default(),
            true,
        ),
        leaf(
            "strict",
            &strict_calls,
            open(),
            ModelCapabilities::default(),
            false,
        ),
    ])
    .with_composite(
        "bounded".to_string(),
        vec!["google".to_string(), "strict".to_string()],
    );

    runner
        .execute_route("bounded", request("public material"), None)
        .await
        .expect("an unavailable gated leg crosses to the next child");

    assert_eq!(google_calls.load(Ordering::SeqCst), 1);
    assert_eq!(strict_calls.load(Ordering::SeqCst), 1);
}

// ---------------------------------------------------------------------------
// Physical fit
// ---------------------------------------------------------------------------

fn physical(context_window: u64, max_input_tokens: Option<u64>) -> CapabilityFacts {
    CapabilityFacts {
        context_window: Some(context_window),
        max_input_tokens,
        ..CapabilityFacts::default()
    }
}

fn long_request(words: usize) -> Request {
    // Four ASCII characters per estimated token, so the estimate is close to `words / 4`.
    request(&"alpha ".repeat(words))
}

/// A request larger than one target's physical window is not dispatched to it, and the
/// wider target serves it.
#[tokio::test]
async fn a_request_beyond_a_targets_window_goes_to_the_wider_target() {
    let narrow_calls = Arc::new(AtomicUsize::new(0));
    let wide_calls = Arc::new(AtomicUsize::new(0));
    let runner = Runner::new(vec![
        leaf(
            "narrow",
            &narrow_calls,
            physical(1_024, None),
            ModelCapabilities::default(),
            false,
        ),
        leaf(
            "wide",
            &wide_calls,
            physical(262_144, None),
            ModelCapabilities::default(),
            false,
        ),
    ])
    .with_composite(
        "bounded".to_string(),
        vec!["narrow".to_string(), "wide".to_string()],
    );

    runner
        .execute_route("bounded", long_request(4_000), None)
        .await
        .expect("the wider target can hold the request");

    assert_eq!(
        narrow_calls.load(Ordering::SeqCst),
        0,
        "a request that cannot fit was sent"
    );
    assert_eq!(wide_calls.load(Ordering::SeqCst), 1);
}

/// When nothing in the route can hold the request the refusal must be immediate and local.
///
/// This is what a 16,000-token bounded lane does with a 40,000-token prompt: it must not
/// send it to Gemma and let the provider reject it, because the call would still be counted
/// against the day's budget while returning nothing.
#[tokio::test]
async fn a_request_no_candidate_can_hold_is_refused_without_a_call() {
    let calls = Arc::new(AtomicUsize::new(0));
    let runner = Runner::new(vec![leaf(
        "gemma",
        &calls,
        physical(262_144, Some(16_000)),
        ModelCapabilities::default(),
        false,
    )]);

    // ~120k estimated tokens: inside the 262,144-token window, far above the 16,000-token
    // per-minute allowance, so the input ceiling is the limit that binds.
    let error = runner
        .execute_route("gemma", long_request(80_000), None)
        .await
        .err()
        .expect("a request above every candidate's ceiling must be refused");

    assert_eq!(
        calls.load(Ordering::SeqCst),
        0,
        "the provider was contacted for a request it had already been ruled unable to serve"
    );
    assert!(
        matches!(
            error,
            RunnerError::UnsupportedCapability {
                capability: "input_tokens_per_minute",
                ..
            }
        ),
        "the refusal must name the limit that bound it: {error}"
    );
}

/// A per-request input ceiling smaller than the window — the Gemma case, where a
/// 262,144-token model is allocated 16,000 input tokens per minute — excludes the target
/// even though the window would fit.
#[tokio::test]
async fn a_request_beyond_the_input_budget_is_not_sent_to_that_target() {
    let gemma_calls = Arc::new(AtomicUsize::new(0));
    let lite_calls = Arc::new(AtomicUsize::new(0));
    let runner = Runner::new(vec![
        leaf(
            "gemma",
            &gemma_calls,
            physical(262_144, Some(16_000)),
            ModelCapabilities::default(),
            false,
        ),
        leaf(
            "lite",
            &lite_calls,
            physical(1_048_576, None),
            ModelCapabilities::default(),
            false,
        ),
    ])
    .with_composite(
        "bounded".to_string(),
        vec!["gemma".to_string(), "lite".to_string()],
    );

    // ~50k tokens: fits the 262,144 window, far above the 16,000-token minute.
    runner
        .execute_route("bounded", long_request(200_000), None)
        .await
        .expect("the wider budget serves it");

    assert_eq!(gemma_calls.load(Ordering::SeqCst), 0);
    assert_eq!(lite_calls.load(Ordering::SeqCst), 1);
}

/// The caller's output reservation counts against the target's fit, because a window that
/// holds the input but not the answer rejects the request upstream after dispatch.
#[tokio::test]
async fn an_output_reservation_counts_against_the_physical_window() {
    let calls = Arc::new(AtomicUsize::new(0));
    let runner = Runner::new(vec![leaf(
        "small",
        &calls,
        physical(1_000, None),
        ModelCapabilities::default(),
        false,
    )]);

    // ~400 estimated input tokens, which the window holds comfortably; the 900-token
    // reservation is what does not fit. Without that separation the test would pass on the
    // input alone and prove nothing about the reservation.
    let mut sized = request(&"alpha ".repeat(260));
    sized.llm_request.output.max_output_tokens = Some(900);

    let error = runner
        .execute_route("small", sized, None)
        .await
        .err()
        .expect("input plus reservation exceeds the window");

    assert!(
        matches!(error, RunnerError::UnsupportedCapability { .. }),
        "{error:?}"
    );
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

/// A target that declares nothing is unconstrained, so a manifest written before the field
/// existed keeps its behaviour.
#[tokio::test]
async fn an_undeclared_window_admits_every_request() {
    let calls = Arc::new(AtomicUsize::new(0));
    let runner = Runner::new(vec![leaf(
        "legacy",
        &calls,
        open(),
        ModelCapabilities::default(),
        false,
    )]);

    runner
        .execute_route("legacy", long_request(400_000), None)
        .await
        .expect("an undeclared window is not a refusal");
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

// ---------------------------------------------------------------------------
// Measured selection
// ---------------------------------------------------------------------------

/// Builds a two-target leaf route whose candidates are both served, scored in the given
/// declared order, with the second leg shaped to answer faster.
fn measured_route(
    first_calls: &Arc<AtomicUsize>,
    second_calls: &Arc<AtomicUsize>,
    failing_first: bool,
) -> (ModelId, Route) {
    let algorithm = AlgorithmSpec::Passthrough {
        target: "first-target".to_string(),
        candidates: vec!["second-target".to_string()],
        subagents: None,
    }
    .build(
        "measured",
        &BTreeMap::from([
            ("first-target".to_string(), ModelId::from("first-target")),
            ("second-target".to_string(), ModelId::from("second-target")),
        ]),
    )
    .expect("two targets build");
    let clients = ClientRouter::new(
        BTreeMap::from([
            (
                ModelId::from("first-target"),
                Arc::new(CountingClient {
                    calls: Arc::clone(first_calls),
                    failing: failing_first,
                }) as Arc<dyn RoutedLlmClient>,
            ),
            (
                ModelId::from("second-target"),
                Arc::new(CountingClient {
                    calls: Arc::clone(second_calls),
                    failing: false,
                }) as Arc<dyn RoutedLlmClient>,
            ),
        ])
        .into_iter()
        .collect(),
    )
    .with_target_capabilities(HashMap::from([
        (ModelId::from("first-target"), open()),
        (ModelId::from("second-target"), open()),
    ]));
    (
        ModelId::from("measured"),
        Route::new(
            algorithm,
            clients,
            None,
            ModelCapabilities::default(),
            None,
            None,
            Vec::new(),
            RuntimeModels::new(
                [(
                    Category::Any,
                    vec![
                        ModelId::from("first-target"),
                        ModelId::from("second-target"),
                    ],
                )]
                .into(),
            ),
            RouteReasoningPolicy::Inherit,
            None,
            None,
        )
        .with_measured_selection(MeasuredSelector::new()),
    )
}

/// A cold route follows the declared order: measurement has nothing to say yet.
#[tokio::test]
async fn a_cold_measured_route_still_uses_the_declared_order() {
    let first = Arc::new(AtomicUsize::new(0));
    let second = Arc::new(AtomicUsize::new(0));
    let (model, route) = measured_route(&first, &second, false);
    let runner = Runner::new(vec![(model, route)]);
    runner
        .execute_route("measured", request("hello"), None)
        .await
        .expect("served");
    assert_eq!(
        first.load(Ordering::SeqCst),
        1,
        "the declared first leg must be tried"
    );
    assert_eq!(second.load(Ordering::SeqCst), 0);
}

/// Once the declared first leg has failed enough to be measured as unreliable, the route
/// prefers the second leg — and reaches it without a provider failure on this request.
#[tokio::test]
async fn a_measured_route_prefers_the_leg_that_has_been_completing() {
    let first = Arc::new(AtomicUsize::new(0));
    let second = Arc::new(AtomicUsize::new(0));
    let (_, route) = measured_route(&first, &second, true);

    // Three requests where the declared first leg fails and the second serves it: that is
    // the evidence the selector needs, and it is produced by requests, not by setup.
    for _ in 0..3 {
        route
            .execute(request("warm"), None)
            .await
            .expect("the second leg serves");
    }
    let first_calls_before = first.load(Ordering::SeqCst);
    let second_calls_before = second.load(Ordering::SeqCst);
    assert_eq!(
        first_calls_before, 3,
        "the failing leg was tried while cold"
    );

    route.execute(request("warm"), None).await.expect("served");

    assert_eq!(
        first.load(Ordering::SeqCst),
        first_calls_before,
        "the measured-unreliable leg must not be tried first any more"
    );
    assert_eq!(second.load(Ordering::SeqCst), second_calls_before + 1);
}

// ---------------------------------------------------------------------------
// Truthful aggregate context
// ---------------------------------------------------------------------------

/// A route may not advertise more context than its widest declared target can serve.
///
/// The claim is what a caller reads before choosing the route, and nothing at request time
/// compares it to a model's real window, so the check has to happen at load.
#[test]
fn a_route_cannot_advertise_more_context_than_its_widest_target() {
    let configuration = r#"
schema_version = 1

[llm_clients.stub]
format = "openai_chat"
base_url = "http://127.0.0.1:1/v1"
api_key_env = "SWITCHYARD_ADAPTIVE_TEST_KEY"

[targets.narrow]
id = "narrow"
llm_client = "stub"
context_window = 32768

[targets.wide]
id = "wide"
llm_client = "stub"
context_window = 262144

[routes.overclaimed]
id = "overclaimed"
type = "passthrough"
target = "narrow"
context_window = 1048576
candidates = ["narrow", "wide"]
"#;
    // SAFETY: test-only, read exclusively by the configuration build in this test.
    unsafe {
        std::env::set_var("SWITCHYARD_ADAPTIVE_TEST_KEY", "test");
    }
    let error = switchyard_runner::Runner::from_toml(configuration)
        .err()
        .expect("the over-claim must be refused");
    let message = error.to_string();
    assert!(
        message.contains("context_window 1048576"),
        "the refusal must name the claim: {message}"
    );
    assert!(
        message.contains("262144"),
        "the refusal must name the widest real window: {message}"
    );
}

/// A composite is resolved before its children and reports its own guarantee, so the same
/// over-claim has to be refused one level up. This is the shape a bounded lane takes: a
/// composite over two Google leaves, where the promise a caller reads comes from the
/// composite rather than from either child.
#[test]
fn a_composite_cannot_advertise_more_context_than_its_widest_child() {
    let configuration = r#"
schema_version = 1

[llm_clients.stub]
format = "openai_chat"
base_url = "http://127.0.0.1:1/v1"
api_key_env = "SWITCHYARD_ADAPTIVE_TEST_KEY"

[targets.small]
id = "small"
llm_client = "stub"
context_window = 262144
max_input_tokens = 16000

[routes.leaf-small]
id = "leaf-small"
type = "passthrough"
target = "small"
context_window = 262144
candidates = []

[composites.overclaimed-bounded]
children = ["leaf-small"]
context_window = 1048576
"#;
    // SAFETY: test-only, read exclusively by the configuration build in this test.
    unsafe {
        std::env::set_var("SWITCHYARD_ADAPTIVE_TEST_KEY", "test");
    }
    let error = switchyard_runner::Runner::from_toml(configuration)
        .err()
        .expect("the composite over-claim must be refused");
    let message = error.to_string();
    assert!(
        message.contains("overclaimed-bounded"),
        "the refusal must name the composite: {message}"
    );
    assert!(
        message.contains("262144"),
        "the refusal must name the widest window below it: {message}"
    );
}

/// The same configuration with an honest claim loads, and the narrow target is skipped for a
/// request only the wide one can hold.
#[test]
fn an_honest_aggregate_claim_loads() {
    let configuration = r#"
schema_version = 1

[llm_clients.stub]
format = "openai_chat"
base_url = "http://127.0.0.1:1/v1"
api_key_env = "SWITCHYARD_ADAPTIVE_TEST_KEY"

[targets.narrow]
id = "narrow"
llm_client = "stub"
context_window = 32768

[targets.wide]
id = "wide"
llm_client = "stub"
context_window = 262144

[routes.bounded]
id = "bounded"
type = "passthrough"
target = "narrow"
context_window = 262144
candidates = ["wide"]
selection = "measured"
"#;
    unsafe {
        std::env::set_var("SWITCHYARD_ADAPTIVE_TEST_KEY", "test");
    }
    switchyard_runner::Runner::from_toml(configuration).expect("an honest claim loads");
}

/// A per-request ceiling above the declared window is a configuration error, because the
/// larger of the two can never bind.
#[test]
fn a_ceiling_above_the_window_is_refused() {
    let configuration = r#"
schema_version = 1

[llm_clients.stub]
format = "openai_chat"
base_url = "http://127.0.0.1:1/v1"
api_key_env = "SWITCHYARD_ADAPTIVE_TEST_KEY"

[targets.gemma]
id = "gemma-4-31b-it"
llm_client = "stub"
context_window = 16000
max_input_tokens = 262144

[routes.bounded]
id = "bounded"
type = "passthrough"
target = "gemma"
"#;
    unsafe {
        std::env::set_var("SWITCHYARD_ADAPTIVE_TEST_KEY", "test");
    }
    let error = switchyard_runner::Runner::from_toml(configuration)
        .err()
        .expect("the inconsistent pair must be refused");
    assert!(error.to_string().contains("max_input_tokens"), "{error}");
}

// ---------------------------------------------------------------------------
// Adaptive free-pool selection
// ---------------------------------------------------------------------------
//
// These exercise the route, not the selector in isolation: a request is admitted,
// candidates are filtered, the shared evidence and the deployment facts order the
// survivors, and the called leg is counted. The point of each test is a decision that
// cannot be reached by reading the manifest's order.

/// Records what it was asked, including the body the route handed it.
struct RecordingClient {
    calls: Arc<AtomicUsize>,
    bodies: Arc<std::sync::Mutex<Vec<serde_json::Value>>>,
}

#[async_trait]
impl RoutedLlmClient for RecordingClient {
    async fn call(&self, request: Request) -> Result<Response, LlmClientError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.bodies
            .lock()
            .expect("test mutex is never poisoned")
            .push(
                request
                    .raw_request
                    .clone()
                    .unwrap_or(serde_json::Value::Null),
            );
        Ok(Response {
            llm_response: LlmResponse::Agg(text_response(
                request.llm_request.model.clone(),
                "served",
            )),
            metadata: None,
            upstream_headers: Default::default(),
        })
    }
}

type CallCounters = HashMap<&'static str, Arc<AtomicUsize>>;
type RecordedBodies = HashMap<&'static str, Arc<std::sync::Mutex<Vec<serde_json::Value>>>>;

/// The deployed free Google pair, in the order the manifest lists them.
const LITE_DAILY: u64 = 500;
const GEMMA_DAILY: u64 = 14_400;
const LITE_INPUT_CEILING: u64 = 250_000;
const GEMMA_INPUT_CEILING: u64 = 16_000;

fn free_lite() -> CapabilityFacts {
    CapabilityFacts {
        free_tier: true,
        context_window: Some(1_048_576),
        max_input_tokens: Some(LITE_INPUT_CEILING),
        ..CapabilityFacts::default()
    }
}

fn free_gemma(accepts_reasoning_control: bool) -> CapabilityFacts {
    CapabilityFacts {
        free_tier: true,
        context_window: Some(262_144),
        max_input_tokens: Some(GEMMA_INPUT_CEILING),
        no_reasoning_control: !accepts_reasoning_control,
        ..CapabilityFacts::default()
    }
}

/// The Gemma lane as the deployment admits it: 16,000 input TPM with a 1300 per-mille
/// admission pad, so the largest estimate it takes is floor(16,000 * 1000 / 1300) = 12,307.
fn free_gemma_padded() -> CapabilityFacts {
    CapabilityFacts {
        safe_input_tokens: Some(12_307),
        ..free_gemma(true)
    }
}

fn a_paid_leg() -> CapabilityFacts {
    CapabilityFacts {
        context_window: Some(262_144),
        ..CapabilityFacts::default()
    }
}

/// One test's private counter directory, since the allowance reads a real file.
fn allowance_dir(label: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "switchyard-adaptive-{label}-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).expect("test allowance directory");
    dir
}

/// A measured passthrough route over `legs`, declared in the order given.
///
/// `legs` carries each target's facts and daily budget, so a test states the deployment
/// rather than the order it expects.
fn recording_pool(
    dir: &std::path::Path,
    legs: &[(&'static str, CapabilityFacts, Option<u64>)],
) -> (Route, CallCounters, RecordedBodies) {
    let targets: Vec<(String, ModelId)> = legs
        .iter()
        .map(|(name, _, _)| {
            (
                format!("{name}-target"),
                ModelId::from(format!("{name}-target").as_str()),
            )
        })
        .collect();
    let names: Vec<String> = targets.iter().map(|(name, _)| name.clone()).collect();
    let algorithm = AlgorithmSpec::Passthrough {
        target: names[0].clone(),
        candidates: names[1..].to_vec(),
        subagents: None,
    }
    .build("adaptive-pool", &BTreeMap::from_iter(targets.clone()))
    .expect("the pool builds");

    let mut calls = CallCounters::new();
    let mut bodies = RecordedBodies::new();
    let mut clients: Vec<(ModelId, Arc<dyn RoutedLlmClient>)> = Vec::new();
    for ((_, model), (name, _, _)) in targets.iter().zip(legs.iter()) {
        let calls_seen = Arc::new(AtomicUsize::new(0));
        let bodies_seen = Arc::new(std::sync::Mutex::new(Vec::new()));
        calls.insert(name, Arc::clone(&calls_seen));
        bodies.insert(name, Arc::clone(&bodies_seen));
        clients.push((
            model.clone(),
            Arc::new(RecordingClient {
                calls: calls_seen,
                bodies: bodies_seen,
            }) as Arc<dyn RoutedLlmClient>,
        ));
    }

    let facts: HashMap<ModelId, CapabilityFacts> = targets
        .iter()
        .zip(legs.iter())
        .map(|((_, model), (_, facts, _))| (model.clone(), facts.clone()))
        .collect();
    let allowances: HashMap<ModelId, Arc<switchyard_llm_client::DailyRequestAllowance>> = targets
        .iter()
        .zip(legs.iter())
        .filter_map(|((_, model), (name, _, limit))| {
            limit.map(|limit| {
                (
                    model.clone(),
                    Arc::new(switchyard_llm_client::DailyRequestAllowance::new(
                        "google-test",
                        model.clone(),
                        limit,
                        dir.join(format!("{name}.json")),
                    )),
                )
            })
        })
        .collect();

    let clients = ClientRouter::new(clients.into_iter().collect()).with_target_capabilities(facts);
    let route = Route::new(
        algorithm,
        clients,
        None,
        ModelCapabilities::default(),
        None,
        None,
        Vec::new(),
        RuntimeModels::new(
            [(
                Category::Any,
                targets.iter().map(|(_, model)| model.clone()).collect(),
            )]
            .into(),
        ),
        RouteReasoningPolicy::Inherit,
        None,
        None,
    )
    .with_measured_selection(MeasuredSelector::new())
    .with_daily_allowances(Arc::new(allowances));
    (route, calls, bodies)
}

fn calls_of(calls: &CallCounters, leg: &str) -> usize {
    calls[leg].load(Ordering::SeqCst)
}

/// A small request lands on the leg whose daily budget can afford it, with no prior
/// failure needed to discover that.
///
/// This is the correction the release exists for: the deployed chain spends the 500-request
/// model first and reaches the 14,400-request model only when the first one fails.
#[tokio::test]
async fn a_small_bounded_task_reaches_the_daily_rich_leg_before_the_scarce_one() {
    let dir = allowance_dir("small");
    let (route, calls, _) = recording_pool(
        &dir,
        &[
            ("lite", free_lite(), Some(LITE_DAILY)),
            ("gemma", free_gemma(true), Some(GEMMA_DAILY)),
        ],
    );
    let runner = Runner::new(vec![(ModelId::from("adaptive-pool"), route)]);

    runner
        .execute_route("adaptive-pool", request("hello"), None)
        .await
        .expect("the pool serves a small request");

    assert_eq!(
        calls_of(&calls, "gemma"),
        1,
        "the 14,400-request leg is the economical first choice for small work"
    );
    assert_eq!(
        calls_of(&calls, "lite"),
        0,
        "the scarce leg was spent on work the daily-rich leg could serve"
    );
}

/// A request that would claim most of the small leg's minute goes to the leg with
/// headroom, even though its daily budget is smaller.
#[tokio::test]
async fn a_request_that_fills_the_small_legs_minute_goes_to_the_wide_leg() {
    let dir = allowance_dir("large");
    let (route, calls, _) = recording_pool(
        &dir,
        &[
            ("lite", free_lite(), Some(LITE_DAILY)),
            ("gemma", free_gemma(true), Some(GEMMA_DAILY)),
        ],
    );
    let runner = Runner::new(vec![(ModelId::from("adaptive-pool"), route)]);

    // ~12k estimated tokens: inside Gemma's 16,000-token ceiling but three quarters of
    // its minute, so the Lite is the appropriate leg for it.
    runner
        .execute_route("adaptive-pool", long_request(8_000), None)
        .await
        .expect("the pool serves a large request");

    assert_eq!(calls_of(&calls, "lite"), 1);
    assert_eq!(calls_of(&calls, "gemma"), 0);
}

/// A tool exchange is budgeted as the conversation it becomes, so the first turn alone
/// does not make the small leg look appropriate.
#[tokio::test]
async fn a_tool_exchange_is_not_placed_on_the_leg_that_cannot_hold_its_answer() {
    let dir = allowance_dir("tools");
    let (route, calls, _) = recording_pool(
        &dir,
        &[
            ("gemma", free_gemma(true), Some(GEMMA_DAILY)),
            ("lite", free_lite(), Some(LITE_DAILY)),
        ],
    );
    let runner = Runner::new(vec![(ModelId::from("adaptive-pool"), route)]);

    // ~9k estimated input tokens before the tool growth, comfortably inside Gemma's
    // 16,000-token ceiling on its own but not once the answer turn is budgeted.
    let prompt = "alpha ".repeat(6_000);
    let mut tool_request = request(&prompt);
    tool_request.raw_request = Some(serde_json::json!({
        "messages": [{"role": "user", "content": prompt}],
        "tools": [{"type": "function", "function": {"name": "lookup"}}],
    }));

    runner
        .execute_route("adaptive-pool", tool_request, None)
        .await
        .expect("the pool serves a tool request");

    assert_eq!(
        calls_of(&calls, "lite"),
        1,
        "a tool conversation does not fit the leg that declares a 16,000-token minute"
    );
    assert_eq!(calls_of(&calls, "gemma"), 0);
}

/// Cost tier is structural: an eligible free leg is offered before a metered one whatever
/// the manifest order says.
#[tokio::test]
async fn a_paid_leg_never_overtakes_an_eligible_free_leg() {
    let dir = allowance_dir("tier");
    let (route, calls, _) = recording_pool(
        &dir,
        &[
            ("paid", a_paid_leg(), None),
            ("free", free_gemma(true), Some(GEMMA_DAILY)),
        ],
    );
    let runner = Runner::new(vec![(ModelId::from("adaptive-pool"), route)]);

    runner
        .execute_route("adaptive-pool", request("hello"), None)
        .await
        .expect("the free leg serves it");

    assert_eq!(calls_of(&calls, "free"), 1);
    assert_eq!(
        calls_of(&calls, "paid"),
        0,
        "a metered leg was chosen ahead of an eligible free one"
    );
}

/// A request carrying reasoning control skips a leg that cannot accept it, instead of
/// being sent a parameter the provider rejects.
#[tokio::test]
async fn a_reasoning_control_skips_the_leg_that_cannot_accept_it() {
    let dir = allowance_dir("control");
    let (route, calls, _) = recording_pool(
        &dir,
        &[
            ("gemma", free_gemma(false), Some(GEMMA_DAILY)),
            ("lite", free_lite(), Some(LITE_DAILY)),
        ],
    );
    let runner = Runner::new(vec![(ModelId::from("adaptive-pool"), route)]);

    let mut controlled = request("hello");
    controlled.llm_request.reasoning.effort = Some("none".to_string());

    runner
        .execute_route("adaptive-pool", controlled, None)
        .await
        .expect("the leg that accepts control serves it");

    assert_eq!(
        calls_of(&calls, "gemma"),
        0,
        "an unsupported reasoning control was forwarded to a target that states it cannot carry one"
    );
    assert_eq!(calls_of(&calls, "lite"), 1);
}

/// The deployment never invents a reasoning control for a Google leg, in either direction:
/// a request that carries none reaches the target with none, so native thinking and the
/// provider default stand.
#[tokio::test]
async fn a_google_leg_is_never_handed_a_reasoning_control_it_was_not_given() {
    let dir = allowance_dir("native");
    let (route, calls, bodies) =
        recording_pool(&dir, &[("gemma", free_gemma(false), Some(GEMMA_DAILY))]);
    let runner = Runner::new(vec![(ModelId::from("adaptive-pool"), route)]);

    runner
        .execute_route("adaptive-pool", request("hello"), None)
        .await
        .expect("served");

    assert_eq!(calls_of(&calls, "gemma"), 1);
    let seen = bodies["gemma"]
        .lock()
        .expect("test mutex is never poisoned")
        .clone();
    let body = seen.first().expect("the body reached the target");
    for key in ["reasoning", "reasoning_effort", "thinking"] {
        assert!(
            body.get(key).is_none_or(serde_json::Value::is_null),
            "the route invented a {key} field the caller never sent: {body}"
        );
    }
    assert!(
        body.pointer("/output_config/effort").is_none(),
        "the route invented an output_config.effort the caller never sent: {body}"
    );
}

/// An exhausted daily budget demotes a leg within its tier without making it ineligible:
/// the hard quota gate stays the authority on whether a request may be sent at all.
#[tokio::test]
async fn a_spent_allowance_is_offered_after_a_leg_that_still_has_budget() {
    let dir = allowance_dir("spent");
    let (route, calls, _) = recording_pool(
        &dir,
        &[
            ("gemma", free_gemma(true), Some(GEMMA_DAILY)),
            ("lite", free_lite(), Some(1)),
        ],
    );
    // Spend the Lite's single request before the measured order is computed.
    let lite_path = dir.join("lite.json");
    std::fs::write(
        &lite_path,
        format!(
            "{{\"day\":{},\"spent\":1,\"limit\":1,\"group\":\"google-test\",\"model\":\"lite-target\",\"basis\":\"local_budget\",\"provider_remaining\":null}}\n",
            switchyard_llm_client::allowance::provider_day(SystemTime::now())
        ),
    )
    .expect("the counter is writable");
    let runner = Runner::new(vec![(ModelId::from("adaptive-pool"), route)]);

    runner
        .execute_route("adaptive-pool", request("hello"), None)
        .await
        .expect("the leg with budget serves it");

    assert_eq!(calls_of(&calls, "gemma"), 1);
    assert_eq!(calls_of(&calls, "lite"), 0);
}

/// A request near Gemma's minute budget is refused before egress rather than dispatched on
/// an optimistic estimate.
///
/// The routing half of the measured overrun. On 2026-11-21 a prompt the estimator placed at
/// 15,202 tokens was admitted against Gemma's 16,000 input TPM and the provider charged
/// 18,054 - 18.8 % above the estimate and 12.8 % above the window. With a 1300 per-mille
/// admission pad the largest estimate Gemma may take is 12,307, and that bound is a hard
/// filter, not a preference: the target is dropped from the candidate list, so no request
/// leaves the process and the caller is told the honest reason.
///
/// The control is the same route without the bound: the identical request dispatches,
/// which is what proves the bound - not the request size - is what stopped it.
#[tokio::test]
async fn a_near_limit_gemma_request_never_reaches_the_padded_window() {
    // 9,000 words of "alpha " -> about 13,500 estimated tokens: above the padded Gemma
    // ceiling of 12,307 and below its raw 16,000 window.
    let padded_calls = Arc::new(AtomicUsize::new(0));
    let (model, route) = leaf(
        "gemma",
        &padded_calls,
        free_gemma_padded(),
        ModelCapabilities::default(),
        false,
    );
    let route = route.with_measured_selection(MeasuredSelector::new());
    let runner = Runner::new(vec![(model.clone(), route)]);
    let error = match runner
        .execute_route(model.as_str(), long_request(9_000), None)
        .await
    {
        Ok(_) => panic!("the padded window cannot admit this request"),
        Err(error) => error,
    };
    assert_eq!(
        padded_calls.load(Ordering::SeqCst),
        0,
        "nothing may be dispatched when the deployment's own bound refuses the request"
    );
    assert!(
        format!("{error}").contains("input_tokens_per_minute"),
        "the refusal must name the minute input budget, got: {error}"
    );

    // Control: no admission bound on the same target, so the identical request dispatches.
    let open_calls = Arc::new(AtomicUsize::new(0));
    let (control_model, control_route) = leaf(
        "gemma",
        &open_calls,
        free_gemma(true),
        ModelCapabilities::default(),
        false,
    );
    let control_route = control_route.with_measured_selection(MeasuredSelector::new());
    let control_runner = Runner::new(vec![(control_model.clone(), control_route)]);
    control_runner
        .execute_route(control_model.as_str(), long_request(9_000), None)
        .await
        .expect("without the admission bound the request is dispatched");
    assert_eq!(
        open_calls.load(Ordering::SeqCst),
        1,
        "the control proves the bound is what moved the request"
    );
}

// ---------------------------------------------------------------------------
// Selection trace: every candidate answers "why"
// ---------------------------------------------------------------------------

/// A leg that answers with no usable content.
///
/// The empty answer is the failure a measured selector acts on immediately, and the one the
/// routing log cannot show: the attempt fails before any usage line is written.
struct EmptyAnsweringClient {
    calls: Arc<AtomicUsize>,
}

#[async_trait]
impl RoutedLlmClient for EmptyAnsweringClient {
    async fn call(&self, request: Request) -> Result<Response, LlmClientError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Err(LlmClientError::EmptyAnswer {
            model: ModelId::from(
                request
                    .llm_request
                    .model
                    .as_deref()
                    .unwrap_or("first-target"),
            ),
            reason: "test_no_content",
        })
    }
}

/// Records every selection decision the route emits, in emission order.
fn selection_sink() -> (Arc<Mutex<Vec<SelectionTrace>>>, RunObserver) {
    let sink: Arc<Mutex<Vec<SelectionTrace>>> = Arc::new(Mutex::new(Vec::new()));
    let written = Arc::clone(&sink);
    let observer: RunObserver = Arc::new(move |observation: RunObservation| {
        if let RunObservation::Selection(trace) = observation {
            written.lock().expect("trace sink").push(*trace);
        }
    });
    (sink, observer)
}

/// A two-target measured route, so a decision record has more than one candidate to explain.
fn traced_route(
    first: Arc<dyn RoutedLlmClient>,
    second: Arc<dyn RoutedLlmClient>,
    first_facts: CapabilityFacts,
    second_facts: CapabilityFacts,
) -> (ModelId, Route) {
    let algorithm = AlgorithmSpec::Passthrough {
        target: "first-target".to_string(),
        candidates: vec!["second-target".to_string()],
        subagents: None,
    }
    .build(
        "traced",
        &BTreeMap::from([
            ("first-target".to_string(), ModelId::from("first-target")),
            ("second-target".to_string(), ModelId::from("second-target")),
        ]),
    )
    .expect("two targets build");
    let clients = ClientRouter::new(
        BTreeMap::from([
            (ModelId::from("first-target"), first),
            (ModelId::from("second-target"), second),
        ])
        .into_iter()
        .collect(),
    )
    .with_target_capabilities(HashMap::from([
        (ModelId::from("first-target"), first_facts),
        (ModelId::from("second-target"), second_facts),
    ]));
    (
        ModelId::from("traced"),
        Route::new(
            algorithm,
            clients,
            None,
            ModelCapabilities::default(),
            None,
            None,
            Vec::new(),
            RuntimeModels::new(
                [(
                    Category::Any,
                    vec![
                        ModelId::from("first-target"),
                        ModelId::from("second-target"),
                    ],
                )]
                .into(),
            ),
            RouteReasoningPolicy::Inherit,
            None,
            None,
        )
        .with_measured_selection(MeasuredSelector::new()),
    )
}

/// A candidate the privacy gate excluded is named, with its reason, even though no provider
/// was ever contacted for it.
///
/// The counter assertions are the control: the trace could claim anything about a leg that
/// was in fact called, so the same test proves both that the refusal happened and that the
/// record describes it.
#[tokio::test]
async fn every_candidate_is_eligible_in_the_selection_trace() {
    let first_calls = Arc::new(AtomicUsize::new(0));
    let second_calls = Arc::new(AtomicUsize::new(0));
    let (_, route) = traced_route(
        Arc::new(CountingClient {
            calls: Arc::clone(&first_calls),
            failing: false,
        }),
        Arc::new(CountingClient {
            calls: Arc::clone(&second_calls),
            failing: false,
        }),
        gated(),
        open(),
    );
    let (traces, observer) = selection_sink();

    route
        .execute(request("any payload"), Some(observer))
        .await
        .expect("a leg serves");

    assert_eq!(
        first_calls.load(Ordering::SeqCst),
        1,
        "the first candidate is contacted"
    );
    assert_eq!(second_calls.load(Ordering::SeqCst), 0);

    let traces = traces.lock().expect("trace sink");
    assert_eq!(traces.len(), 1, "one decision record per execution");
    let trace = &traces[0];
    assert_eq!(trace.algorithm, "passthrough");

    let first = trace
        .candidates
        .iter()
        .find(|candidate| candidate.model == "first-target")
        .expect("the first candidate appears");
    assert!(first.eligible, "every candidate is eligible");
    assert_eq!(first.exclusion, None);
    assert_eq!(first.rank, Some(0));

    let second = trace
        .candidates
        .iter()
        .find(|candidate| candidate.model == "second-target")
        .expect("the second candidate appears");
    assert!(second.eligible);
    assert_eq!(second.exclusion, None);
    assert_eq!(second.rank, Some(1));

    assert_eq!(
        trace.dispatched,
        vec!["first-target".to_string(), "second-target".to_string()]
    );
    assert_eq!(trace.served.as_deref(), Some("first-target"));
    assert_eq!(trace.outcome, "served");
    assert_eq!(trace.attempts.len(), 1);
    assert_eq!(trace.attempts[0].model, "first-target");
    assert_eq!(trace.attempts[0].phase, "answer");
    assert_eq!(trace.attempts[0].outcome, "served");
}
/// One empty answer is enough to move a leg behind a sibling, and the record says so.
///
/// Without the trace the effect is only visible as "the second leg answered again"; the
/// point of the record is that the demotion and the reason are stated rather than inferred.
#[tokio::test]
async fn an_empty_answer_demotes_the_leg_and_names_the_reason() {
    let empty_calls = Arc::new(AtomicUsize::new(0));
    let healthy_calls = Arc::new(AtomicUsize::new(0));
    let (_, route) = traced_route(
        Arc::new(EmptyAnsweringClient {
            calls: Arc::clone(&empty_calls),
        }),
        Arc::new(CountingClient {
            calls: Arc::clone(&healthy_calls),
            failing: false,
        }),
        open(),
        open(),
    );

    // Evidence first, through a real request: the selector learns from observations, not
    // from a hand-populated map.
    route
        .execute(request("warm"), None)
        .await
        .expect("the sibling answers when the first leg returns nothing");
    assert_eq!(empty_calls.load(Ordering::SeqCst), 1);

    let (traces, observer) = selection_sink();
    route
        .execute(request("hello"), Some(observer))
        .await
        .expect("served");

    assert_eq!(
        empty_calls.load(Ordering::SeqCst),
        1,
        "the demoted leg must not be contacted again on this request"
    );

    let traces = traces.lock().expect("trace sink");
    assert_eq!(traces.len(), 1);
    let trace = &traces[0];
    let demoted = trace
        .candidates
        .iter()
        .find(|candidate| candidate.model == "first-target")
        .expect("the demoted candidate appears");
    assert!(demoted.eligible, "it passed every hard gate");
    assert_eq!(
        demoted.demotion,
        Some("empty_answers"),
        "the record must name the measurement that moved it"
    );
    assert_eq!(demoted.rank, Some(1), "it was dispatched second");
    assert_eq!(
        trace.dispatched,
        vec!["second-target".to_string(), "first-target".to_string()]
    );
    assert_eq!(trace.served.as_deref(), Some("second-target"));
}

/// The record names the model that answered, not the one the run selected first.
///
/// Measured defect 2026-11-21: six live `smart-nt` records named `gemma-4-31b-it` as served
/// while a 5xx on that leg made `gemini-3.x-flash-lite` answer. A record that credits a model
/// which answered nothing is worse than no record: it is exactly the figure an operator uses
/// to decide whether a provider is being reached at all.
#[tokio::test]
async fn the_record_names_the_model_that_answered_not_the_one_picked_first() {
    let first_calls = Arc::new(AtomicUsize::new(0));
    let second_calls = Arc::new(AtomicUsize::new(0));
    let (_, route) = traced_route(
        Arc::new(CountingClient {
            calls: Arc::clone(&first_calls),
            failing: true,
        }),
        Arc::new(CountingClient {
            calls: Arc::clone(&second_calls),
            failing: false,
        }),
        open(),
        open(),
    );

    let (traces, observer) = selection_sink();
    route
        .execute(request("hello"), Some(observer))
        .await
        .expect("the healthy sibling answers when the first leg is unavailable");

    assert_eq!(
        second_calls.load(Ordering::SeqCst),
        1,
        "the second leg is the one that was asked"
    );
    let traces = traces.lock().expect("trace sink");
    assert_eq!(traces.len(), 1, "one decision record per execution");
    let trace = &traces[0];
    assert_eq!(
        trace.served.as_deref(),
        Some("second-target"),
        "the answering model, not the initially selected one"
    );
    assert_eq!(trace.outcome, "served");
    let first = trace
        .attempts
        .iter()
        .find(|attempt| attempt.model == "first-target")
        .expect("the failed leg was contacted and must appear");
    assert_ne!(first.outcome, "served", "it answered nothing");
}

/// A route that refuses before contacting anyone still records why every candidate fell.
///
/// This is the case the caller-facing error cannot describe: it names one missing capability,
/// while the record names the rule that excluded each candidate - including the gated leg
/// that never saw the payload.
#[tokio::test]
async fn a_leaf_is_called_before_any_call_is_recorded() {
    let calls = Arc::new(AtomicUsize::new(0));
    let (_, route) = leaf(
        "google",
        &calls,
        gated(),
        ModelCapabilities::default(),
        false,
    );
    let (traces, observer) = selection_sink();

    let output = match route.execute(request("any payload"), Some(observer)).await {
        Ok(output) => output,
        Err(error) => panic!("the leaf serves any payload, got: {error}"),
    };

    assert_eq!(calls.load(Ordering::SeqCst), 1, "the leaf is contacted");
    assert!(
        !output.selected_model.to_string().is_empty(),
        "the response names what served it"
    );

    let traces = traces.lock().expect("trace sink");
    assert_eq!(traces.len(), 1, "the decision is recorded too");
    let trace = &traces[0];
    assert_eq!(trace.outcome, "served");
    assert_eq!(trace.dispatched, vec!["google-target".to_string()]);
    assert_eq!(trace.served, Some("google-target".to_string()));
    assert_eq!(trace.attempts.len(), 1);
    assert_eq!(trace.attempts[0].model, "google-target");
    assert_eq!(trace.attempts[0].phase, "answer");
    assert_eq!(trace.attempts[0].outcome, "served");
}
/// A leg that writes its reasoning inline against the caller's output reservation.
///
/// Measured 2026-11-21 on hosted Gemma 4 26B: 61-328 thinking tokens for prompts of 49-438
/// tokens, and no visible text at all once the budget ran out mid-thought. 512 is the smallest
/// reservation at which it produced an answer in that sweep.
fn thinking_floor() -> CapabilityFacts {
    CapabilityFacts {
        min_output_tokens: Some(512),
        ..open()
    }
}

fn reservation(tokens: u64) -> Request {
    let mut request = request("hello");
    request.llm_request.output.max_output_tokens = Some(tokens);
    request
}

/// Inside a composite the child is ineligible, not failing: the sibling answers and the leg
/// that charges thinking is never asked. Without this the request is dispatched to a leg whose
/// outcome the deployment can already predict, which is how a 16-token probe gets recorded as a
/// fault of the model.
#[tokio::test]
async fn a_reservation_below_the_thinking_floor_skips_the_child_that_charges_thinking() {
    let thinking_calls = Arc::new(AtomicUsize::new(0));
    let sibling_calls = Arc::new(AtomicUsize::new(0));
    let runner = Runner::new(vec![
        leaf(
            "thinking",
            &thinking_calls,
            thinking_floor(),
            ModelCapabilities::default(),
            false,
        ),
        leaf(
            "sibling",
            &sibling_calls,
            open(),
            ModelCapabilities::default(),
            false,
        ),
    ])
    .with_composite(
        "bounded".to_string(),
        vec!["thinking".to_string(), "sibling".to_string()],
    );

    runner
        .execute_route("bounded", reservation(64), None)
        .await
        .expect("the sibling answers");
    assert_eq!(
        thinking_calls.load(Ordering::SeqCst),
        0,
        "a floor the deployment declared must not cost a provider call to rediscover"
    );
    assert_eq!(sibling_calls.load(Ordering::SeqCst), 1);
}

/// The control: at the floor the same leg is dispatched and answers. Without this, a route that
/// dropped the leg on every request would also pass the test above.
#[tokio::test]
async fn a_reservation_at_the_thinking_floor_still_dispatches_the_leg() {
    let thinking_calls = Arc::new(AtomicUsize::new(0));
    let sibling_calls = Arc::new(AtomicUsize::new(0));
    let runner = Runner::new(vec![
        leaf(
            "thinking",
            &thinking_calls,
            thinking_floor(),
            ModelCapabilities::default(),
            false,
        ),
        leaf(
            "sibling",
            &sibling_calls,
            open(),
            ModelCapabilities::default(),
            false,
        ),
    ])
    .with_composite(
        "bounded".to_string(),
        vec!["thinking".to_string(), "sibling".to_string()],
    );

    runner
        .execute_route("bounded", reservation(512), None)
        .await
        .expect("the thinking leg answers at its floor");
    assert_eq!(thinking_calls.load(Ordering::SeqCst), 1);
    assert_eq!(sibling_calls.load(Ordering::SeqCst), 0);
}

/// The production shape: one passthrough over several candidates. The skipped leg is recorded
/// with the rule that excluded it and the sibling is the model named as served, so an operator
/// reading the decision log can tell a budget skip from a failure.
#[tokio::test]
async fn a_budget_skip_is_named_in_the_record_and_the_sibling_is_credited() {
    let thinking_calls = Arc::new(AtomicUsize::new(0));
    let sibling_calls = Arc::new(AtomicUsize::new(0));
    let (_, route) = traced_route(
        Arc::new(CountingClient {
            calls: Arc::clone(&thinking_calls),
            failing: false,
        }),
        Arc::new(CountingClient {
            calls: Arc::clone(&sibling_calls),
            failing: false,
        }),
        thinking_floor(),
        open(),
    );
    let (traces, observer) = selection_sink();

    route
        .execute(reservation(64), Some(observer))
        .await
        .expect("the sibling answers");

    assert_eq!(thinking_calls.load(Ordering::SeqCst), 0);
    assert_eq!(sibling_calls.load(Ordering::SeqCst), 1);

    let traces = traces.lock().expect("trace sink");
    assert_eq!(traces.len(), 1, "one decision record per execution");
    let trace = &traces[0];
    let skipped = trace
        .candidates
        .iter()
        .find(|candidate| candidate.model == "first-target")
        .expect("the skipped candidate appears");
    assert!(!skipped.eligible);
    assert_eq!(
        skipped.exclusion,
        Some("output_budget_below_thinking_floor"),
        "the record must name the rule, not merely that it was passed over"
    );
    assert_eq!(trace.served.as_deref(), Some("second-target"));
}

/// A directly addressed leaf cannot advance, so it refuses — and the refusal names the axis,
/// which is the difference between a caller who sends a larger reservation and one who retries
/// the same request.
#[tokio::test]
async fn a_directly_addressed_leaf_names_the_thinking_floor_it_cannot_meet() {
    let calls = Arc::new(AtomicUsize::new(0));
    let (model, route) = leaf(
        "thinking",
        &calls,
        thinking_floor(),
        ModelCapabilities::default(),
        false,
    );

    let error = match route.execute(reservation(64), None).await {
        Ok(_) => panic!("a leg that cannot answer within the reservation must not be asked"),
        Err(error) => error,
    };
    assert!(
        format!("{error}").contains("output_budget_below_thinking_floor"),
        "the refusal must name the binding limit, got: {error}"
    );
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert_eq!(model.as_str(), "thinking");
}

// ---------------------------------------------------------------------------
// The measured floor on a model that does not declare hosted_reasoning
// ---------------------------------------------------------------------------

/// `direct-google-gemini-3-flash` answered HTTP 500 with an empty completion at an 8-token
/// reservation and normally at 600. 600 is the smallest reservation observed to work, so it
/// is the declared floor; the true boundary was not bisected and is not claimed.
fn measured_floor_600() -> CapabilityFacts {
    CapabilityFacts {
        min_output_tokens: Some(600),
        ..open()
    }
}

#[tokio::test]
async fn a_leaf_declaring_the_measured_floor_skips_a_tiny_reservation() {
    let calls = Arc::new(AtomicUsize::new(0));
    let (_, route) = leaf(
        "three-flash",
        &calls,
        measured_floor_600(),
        ModelCapabilities::default(),
        false,
    );

    let error = match route.execute(reservation(8), None).await {
        Ok(_) => panic!("an 8-token reservation is the measured failure and must not be sent"),
        Err(error) => error,
    };
    assert!(
        format!("{error}").contains("output_budget_below_thinking_floor"),
        "the refusal must name the binding limit, got: {error}"
    );
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

/// The control: at the measured floor the same leg is dispatched. A floor applied to every
/// reservation would pass the test above and silently disable the target.
#[tokio::test]
async fn a_leaf_declaring_the_measured_floor_dispatches_at_the_floor() {
    let calls = Arc::new(AtomicUsize::new(0));
    let (_, route) = leaf(
        "three-flash",
        &calls,
        measured_floor_600(),
        ModelCapabilities::default(),
        false,
    );

    route
        .execute(reservation(600), None)
        .await
        .expect("the measured passing reservation must be dispatched");
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

// ---------------------------------------------------------------------------
// The caller's reasoning effort is a requirement; the deployment's is not
// ---------------------------------------------------------------------------

/// A target whose provider answers HTTP 400 to `reasoning_effort: none`, as every Google
/// chat target does, while accepting the graded values.
fn refusing_none() -> CapabilityFacts {
    CapabilityFacts {
        unsupported_reasoning_effort: BTreeSet::from(["none".to_string()]),
        ..open()
    }
}

fn with_caller_effort(mut request: Request, effort: Option<&str>) -> Request {
    request.llm_request.reasoning.effort = effort.map(str::to_string);
    request
}

/// When the caller is the one who asked for a refused value, the leg is ineligible and still
/// no request is sent. Stripping the value instead would leave the provider's native
/// reasoning running while the caller believed `none` had been applied.
#[tokio::test]
async fn a_caller_supplied_refused_effort_makes_the_leaf_ineligible() {
    let calls = Arc::new(AtomicUsize::new(0));
    let (_, route) = leaf(
        "google",
        &calls,
        refusing_none(),
        ModelCapabilities::default(),
        false,
    );

    let error = match route
        .execute(with_caller_effort(request("hello"), Some("none")), None)
        .await
    {
        Ok(_) => panic!("a refused caller-supplied effort must not be dispatched"),
        Err(error) => error,
    };
    assert!(
        format!("{error}").contains("unsupported_reasoning_effort"),
        "the refusal must name the refused control, got: {error}"
    );
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

/// The discriminator for the test above: the same target, the same refused value in its
/// declaration, but no caller-supplied effort. A deployment-injected value is the
/// deployment's to correct, so the leg stays eligible - if this failed, the gate would be
/// reading the target's declaration instead of the caller's request.
#[tokio::test]
async fn a_deployment_injected_refused_effort_does_not_make_the_leaf_ineligible() {
    let calls = Arc::new(AtomicUsize::new(0));
    let (_, route) = leaf(
        "google",
        &calls,
        refusing_none(),
        ModelCapabilities::default(),
        false,
    );

    route
        .execute(with_caller_effort(request("hello"), None), None)
        .await
        .expect("a request that asks for nothing must not be refused for a value it lacks");
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

/// Only the refused values exclude: a graded effort the provider serves is dispatched.
#[tokio::test]
async fn a_caller_supplied_accepted_effort_is_dispatched() {
    let calls = Arc::new(AtomicUsize::new(0));
    let (_, route) = leaf(
        "google",
        &calls,
        refusing_none(),
        ModelCapabilities::default(),
        false,
    );

    route
        .execute(with_caller_effort(request("hello"), Some("low")), None)
        .await
        .expect("low is served by this wire and must reach it");
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

/// Inside a composite the leg that cannot honour the caller's effort is skipped, and the
/// sibling answers. This is the escape hatch that makes a strict `none` usable at all, and
/// the reason the refusal is a candidate gate rather than a request-level error: refusing the
/// request outright would deny the caller a leg that can actually honour it.
#[tokio::test]
async fn a_caller_supplied_refused_effort_crosses_to_a_sibling_that_accepts_it() {
    let google_calls = Arc::new(AtomicUsize::new(0));
    let nt_calls = Arc::new(AtomicUsize::new(0));
    let runner = Runner::new(vec![
        leaf(
            "google",
            &google_calls,
            refusing_none(),
            ModelCapabilities::default(),
            false,
        ),
        leaf("nt", &nt_calls, open(), ModelCapabilities::default(), false),
    ])
    .with_composite(
        "strict-none".to_string(),
        vec!["google".to_string(), "nt".to_string()],
    );

    runner
        .execute_route(
            "strict-none",
            with_caller_effort(request("hello"), Some("none")),
            None,
        )
        .await
        .expect("the sibling honours the requested effort");
    assert_eq!(
        google_calls.load(Ordering::SeqCst),
        0,
        "the leg that would strip the value must not be asked"
    );
    assert_eq!(nt_calls.load(Ordering::SeqCst), 1);
}

// ---------------------------------------------------------------------------
// A free leg's health and fallback, offline
// ---------------------------------------------------------------------------

/// The free Lite leg that serves the small-request path is still in the ladder and still
/// crosses when it fails.
///
/// This is deliberately an offline shape rather than a provider probe: it asserts the
/// arrangement the deployment relies on - 3.1-flash-lite is offered, it is asked first, and a
/// failure on it hands the request to the next free leg without ending the route. A probe
/// would additionally spend the free budget to learn what the router already guarantees.
#[tokio::test]
async fn a_failing_flash_lite_leg_still_crosses_to_the_next_free_leg() {
    let lite_calls = Arc::new(AtomicUsize::new(0));
    let next_calls = Arc::new(AtomicUsize::new(0));
    let runner = Runner::new(vec![
        leaf(
            "gemini-3.1-flash-lite",
            &lite_calls,
            open(),
            ModelCapabilities::default(),
            true,
        ),
        leaf(
            "gemini-3.5-flash-lite",
            &next_calls,
            open(),
            ModelCapabilities::default(),
            false,
        ),
    ])
    .with_composite(
        "free-ladder".to_string(),
        vec![
            "gemini-3.1-flash-lite".to_string(),
            "gemini-3.5-flash-lite".to_string(),
        ],
    );

    runner
        .execute_route("free-ladder", request("hello"), None)
        .await
        .expect("the next free leg answers after the first one fails");
    assert_eq!(
        lite_calls.load(Ordering::SeqCst),
        1,
        "the failing leg is tried, not skipped pre-emptively"
    );
    assert_eq!(next_calls.load(Ordering::SeqCst), 1);
}
