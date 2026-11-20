// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pre-egress admission at the route boundary.
//!
//! These exercise the gate where it is enforced — the per-candidate filter inside
//! `Route::execute` — rather than through the parser underneath it. A gate that refuses in
//! a unit test but lets a directly addressed leaf through is not a gate, so the assertions
//! include how many times each stub client was called.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use libsy::RuntimeModels;
use switchyard_llm_client::{CapabilityFacts, ClientRouter};
use switchyard_protocol::{
    Category, LlmClientError, LlmResponse, Message, Metadata, ModelId, Request, Response, Role,
    RoutedLlmClient, text_request, text_response,
};
use switchyard_runner::{AlgorithmSpec, Route, RunnerError};

/// A client that records that it was asked, and answers.
struct CountingClient {
    calls: Arc<AtomicUsize>,
}

#[async_trait]
impl RoutedLlmClient for CountingClient {
    async fn call(&self, request: Request) -> Result<Response, LlmClientError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
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

/// The Pacific civil date of the current provider day, as the clearance header spells it.
///
/// Derived here independently of the parser under test: the gate is only useful if a
/// caller following the documentation can produce a value it accepts.
fn today_header() -> String {
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock is after the epoch")
        .as_secs();
    let days = ((seconds - 8 * 3_600) / 86_400) as i64;
    // Days since the epoch → civil date (Howard Hinnant's civil_from_days).
    let shifted = days + 719_468;
    let era = shifted.div_euclid(146_097);
    let day_of_era = shifted - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_prime = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_prime + 2) / 5 + 1;
    let month = if month_prime < 10 {
        month_prime + 3
    } else {
        month_prime - 9
    };
    let year = if month <= 2 { year + 1 } else { year };
    format!("v1;class=public;scope=assembled;day={year:04}-{month:02}-{day:02}")
}

fn request_with_clearance(clearance: Option<&str>, text: &str) -> Request {
    let mut request = Request {
        llm_request: text_request(Some("routed".to_string()), text),
        ..Request::default()
    };
    if let Some(clearance) = clearance {
        let mut headers = http::HeaderMap::new();
        headers.insert(
            switchyard_runner::CLEARANCE_HEADER,
            http::HeaderValue::from_str(clearance).expect("valid test header"),
        );
        request.metadata = Some(Metadata {
            http_headers: Some(headers),
            ..Metadata::default()
        });
    }
    request
}

/// A route over `targets` in the order given, with each target's facts.
fn route_over(targets: &[(&str, bool)], calls: &[Arc<AtomicUsize>]) -> Route {
    let names: Vec<String> = targets.iter().map(|(name, _)| name.to_string()).collect();
    let spec = AlgorithmSpec::Passthrough {
        target: names[0].clone(),
        candidates: names[1..].to_vec(),
        subagents: None,
    };
    let target_map = targets
        .iter()
        .map(|(name, _)| (name.to_string(), ModelId::from(*name)))
        .collect::<BTreeMap<_, _>>();
    let algorithm = spec
        .build("switchyard", &target_map)
        .expect("a leaf route over known targets builds");
    let clients: HashMap<ModelId, Arc<dyn RoutedLlmClient>> = targets
        .iter()
        .zip(calls)
        .map(|((name, _), calls)| {
            (
                ModelId::from(*name),
                Arc::new(CountingClient {
                    calls: Arc::clone(calls),
                }) as Arc<dyn RoutedLlmClient>,
            )
        })
        .collect();
    let facts: HashMap<ModelId, CapabilityFacts> = targets
        .iter()
        .map(|(name, gated)| {
            (
                ModelId::from(*name),
                CapabilityFacts {
                    requires_public_payload: *gated,
                    ..CapabilityFacts::default()
                },
            )
        })
        .collect();
    let models = RuntimeModels::new(
        [(
            Category::Any,
            targets
                .iter()
                .map(|(name, _)| ModelId::from(*name))
                .collect(),
        )]
        .into(),
    );
    Route::new(
        algorithm,
        ClientRouter::new(clients).with_target_capabilities(facts),
        None,
        switchyard_runner::ModelCapabilities::default(),
        None,
        None,
        Vec::new(),
        models,
        switchyard_runner::RouteReasoningPolicy::Inherit,
        None,
        None,
    )
}

/// A directly addressed gated target refuses an uncleared payload and is never contacted.
#[tokio::test]
async fn a_direct_address_to_a_gated_target_is_refused_without_egress() {
    let calls = Arc::new(AtomicUsize::new(0));
    let route = route_over(&[("gemini", true)], &[Arc::clone(&calls)]);

    let error = match route
        .execute(request_with_clearance(None, "hello"), None)
        .await
    {
        Ok(_) => panic!("an uncleared payload must not leave"),
        Err(error) => error,
    };

    match error {
        RunnerError::PrivacyRefused { route, refusals } => {
            // The route's algorithm name, so an operator can tell which gate refused.
            assert_eq!(route, "passthrough");
            let codes: Vec<&str> = refusals.iter().map(|refusal| refusal.as_str()).collect();
            assert_eq!(codes, vec!["no_clearance"]);
        }
        other => panic!("expected PrivacyRefused, got {other:?}"),
    }
    assert_eq!(
        calls.load(Ordering::SeqCst),
        0,
        "a refused payload must not reach the provider"
    );
}

/// The same target serves a request that carries a current clearance.
#[tokio::test]
async fn a_cleared_request_reaches_the_gated_target() {
    let calls = Arc::new(AtomicUsize::new(0));
    let route = route_over(&[("gemini", true)], &[Arc::clone(&calls)]);
    let clearance = today_header();

    let output = route
        .execute(request_with_clearance(Some(&clearance), "hello"), None)
        .await
        .expect("a cleared payload may leave");

    assert_eq!(output.selected_model, ModelId::from("gemini"));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

/// A gated candidate is skipped and the next, ungated leg serves the request.
#[tokio::test]
async fn a_refusal_falls_through_to_an_ungated_candidate() {
    let gated_calls = Arc::new(AtomicUsize::new(0));
    let plain_calls = Arc::new(AtomicUsize::new(0));
    let route = route_over(
        &[("gemini", true), ("luna", false)],
        &[Arc::clone(&gated_calls), Arc::clone(&plain_calls)],
    );

    let output = route
        .execute(request_with_clearance(None, "hello"), None)
        .await
        .expect("the ungated leg must serve");

    assert_eq!(output.selected_model, ModelId::from("luna"));
    assert_eq!(gated_calls.load(Ordering::SeqCst), 0);
    assert_eq!(plain_calls.load(Ordering::SeqCst), 1);
}

/// A clearance does not override the content inspection: the bytes are still checked.
#[tokio::test]
async fn sensitive_content_is_refused_even_when_cleared() {
    let calls = Arc::new(AtomicUsize::new(0));
    let route = route_over(&[("gemini", true)], &[Arc::clone(&calls)]);
    let clearance = today_header();

    let error = match route
        .execute(
            request_with_clearance(Some(&clearance), "token sk-abcdef0123456789abcdef"),
            None,
        )
        .await
    {
        Ok(_) => panic!("a credential in the payload must not leave"),
        Err(error) => error,
    };

    match error {
        RunnerError::PrivacyRefused { refusals, .. } => {
            let codes: Vec<&str> = refusals.iter().map(|refusal| refusal.as_str()).collect();
            assert_eq!(codes, vec!["sensitive_content_api_key"]);
        }
        other => panic!("expected PrivacyRefused, got {other:?}"),
    }
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

/// An ungated deployment is unaffected: no clearance, no inspection, normal service.
#[tokio::test]
async fn an_ungated_route_is_unchanged() {
    let calls = Arc::new(AtomicUsize::new(0));
    let route = route_over(&[("plain", false)], &[Arc::clone(&calls)]);

    let output = route
        .execute(
            request_with_clearance(None, "token sk-abcdef0123456789abcdef"),
            None,
        )
        .await
        .expect("an ungated target receives anything its caller assembles");

    assert_eq!(output.selected_model, ModelId::from("plain"));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

/// The gate reads the assistant's own prior content too: a tool result is what usually
/// carries a secret, and continuation replays it to the provider.
#[tokio::test]
async fn a_refusal_sees_content_from_earlier_turns() {
    let calls = Arc::new(AtomicUsize::new(0));
    let route = route_over(&[("gemini", true)], &[Arc::clone(&calls)]);
    let mut request = request_with_clearance(Some(&today_header()), "continue please");
    request.llm_request.messages.push(Message::text(
        Role::User,
        "found it in /srv/turnstone/credentials/api/key",
    ));

    let error = match route.execute(request, None).await {
        Ok(_) => panic!("a credential path must not leave"),
        Err(error) => error,
    };

    match error {
        RunnerError::PrivacyRefused { refusals, .. } => {
            let codes: Vec<&str> = refusals.iter().map(|refusal| refusal.as_str()).collect();
            assert_eq!(codes, vec!["sensitive_content_credential_path"]);
        }
        other => panic!("expected PrivacyRefused, got {other:?}"),
    }
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

/// A stale clearance is refused rather than silently accepted, and the reason says so.
#[tokio::test]
async fn a_stale_clearance_is_refused() {
    let calls = Arc::new(AtomicUsize::new(0));
    let route = route_over(&[("gemini", true)], &[Arc::clone(&calls)]);
    let error = match route
        .execute(
            request_with_clearance(Some("v1;class=public;scope=assembled;day=2020-01-01"), "hi"),
            None,
        )
        .await
    {
        Ok(_) => panic!("a clearance from another day must not be honoured"),
        Err(error) => error,
    };

    match error {
        RunnerError::PrivacyRefused { refusals, .. } => {
            let codes: Vec<&str> = refusals.iter().map(|refusal| refusal.as_str()).collect();
            assert_eq!(codes, vec!["clearance_expired"]);
        }
        other => panic!("expected PrivacyRefused, got {other:?}"),
    }
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

/// The format the documentation tells a caller to send is the format the gate accepts.
///
/// Written from the documented grammar rather than from the parser's internals, so a
/// change that quietly narrows the accepted shape fails here rather than at a deployment.
#[test]
fn a_caller_following_the_documented_format_is_admitted() {
    let request = request_with_clearance(Some(&today_header()), "summarise this");
    let admission = switchyard_runner::egress::evaluate(&request);
    assert!(admission.allowed(), "{:?}", admission.refusals());
}

// ---------------------------------------------------------------------------
// End to end: the declarations in a deployment file, not hand-built facts.
//
// The gate's usefulness depends on configuration reaching the filter, so these build a
// real `Runner` from TOML and execute through the same entry point HTTP ingress uses.
// ---------------------------------------------------------------------------

/// One gated deployment: a client declaring the requirement, one target, one route.
fn gated_toml(base_url: &str, extra: &str) -> String {
    format!(
        r#"
schema_version = 1

[llm_clients.google]
format = "openai_chat"
base_url = "{base_url}/v1"
requires_public_payload = true

[targets.gemini]
id = "google/gemini-test"
llm_client = "google"

[routes.r]
id = "r"
type = "passthrough"
target = "gemini"
candidates = []
{extra}
"#
    )
}

fn scratch_dir(label: &str) -> std::path::PathBuf {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    std::env::temp_dir()
        .join("switchyard-egress-config-tests")
        .join(format!(
            "{}-{}-{label}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::SeqCst)
        ))
}

/// The declaration on the client alone is enough to gate its target: no clearance, no call.
#[tokio::test]
async fn a_client_declaration_gates_its_target_end_to_end() {
    let server = wiremock::MockServer::start().await;
    wiremock::Mock::given(wiremock::matchers::method("POST"))
        .and(wiremock::matchers::path("/v1/chat/completions"))
        .respond_with(
            wiremock::ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "chatcmpl-1",
                "model": "google/gemini-test",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "served"},
                    "finish_reason": "stop"
                }],
                "usage": {}
            })),
        )
        .mount(&server)
        .await;

    let runner = switchyard_runner::Runner::from_toml(&gated_toml(&server.uri(), ""))
        .expect("the deployment parses");

    let error = match runner
        .execute_route("r", request_with_clearance(None, "hello"), None)
        .await
    {
        Ok(_) => panic!("an uncleared payload must not leave"),
        Err(error) => error,
    };
    assert!(
        matches!(error, RunnerError::PrivacyRefused { .. }),
        "expected PrivacyRefused, got {error:?}"
    );
    assert_eq!(
        server.received_requests().await.unwrap_or_default().len(),
        0,
        "the provider must not have been contacted"
    );

    // The same deployment serves the request once it is cleared, so the gate refuses
    // rather than closing the route.
    let clearance = today_header();
    let execution = runner
        .execute_route("r", request_with_clearance(Some(&clearance), "hello"), None)
        .await
        .expect("a cleared payload is served");
    assert_eq!(
        execution.output.selected_model,
        ModelId::from("google/gemini-test")
    );
    assert_eq!(
        server.received_requests().await.unwrap_or_default().len(),
        1
    );
}

/// A budget with nowhere to count it is a configuration error, not a live cap that is
/// silently not enforced.
#[test]
fn a_daily_limit_without_a_state_directory_is_refused() {
    let toml = gated_toml("http://127.0.0.1:1", "").replace(
        "llm_client = \"google\"",
        "llm_client = \"google\"\ndaily_request_limit = 20",
    );
    let error = switchyard_runner::Runner::from_toml(&toml)
        .err()
        .expect("a limit without state must be refused");
    let message = error.to_string();
    assert!(
        message.contains("[quota]"),
        "the refusal names the missing section: {message}"
    );
}

/// A zero limit would refuse every request forever, which is never what a deployment means.
#[test]
fn a_zero_daily_limit_is_refused() {
    let directory = scratch_dir("zero");
    let toml = gated_toml(
        "http://127.0.0.1:1",
        &format!("\n[quota]\nstate_dir = \"{}\"", directory.display()),
    )
    .replace(
        "llm_client = \"google\"",
        "llm_client = \"google\"\ndaily_request_limit = 0",
    );
    let error = switchyard_runner::Runner::from_toml(&toml)
        .err()
        .expect("a zero limit must be refused");
    assert!(
        error.to_string().contains("at least 1"),
        "the refusal explains the floor: {error}"
    );
}

/// A declared limit with a usable directory loads, and the directory is proven writable at
/// load rather than at the first request.
#[test]
fn a_declared_limit_creates_and_probes_its_state_directory() {
    let directory = scratch_dir("probe");
    assert!(!directory.exists());
    let toml = gated_toml(
        "http://127.0.0.1:1",
        &format!("\n[quota]\nstate_dir = \"{}\"", directory.display()),
    )
    .replace(
        "llm_client = \"google\"",
        "llm_client = \"google\"\ndaily_request_limit = 20",
    );
    switchyard_runner::Runner::from_toml(&toml).expect("a declared, countable limit loads");
    assert!(directory.is_dir(), "load creates the state directory");
    assert_eq!(
        std::fs::read_dir(&directory).expect("readable").count(),
        0,
        "the write probe leaves no file behind"
    );
}
