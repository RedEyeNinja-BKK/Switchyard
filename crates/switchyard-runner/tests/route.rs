// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures_util::StreamExt;
use libsy::RuntimeModels;
use switchyard_llm_client::{ClientRouter, RunObservation};
use switchyard_protocol::{
    Category, LlmClientError, LlmResponse, ModelId, Request, Response, RoutedLlmClient,
    text_request, text_response,
};
use switchyard_runner::{
    AlgorithmSpec, ModelCapabilities, Route,
    request_fit::estimate_input_tokens,
};

struct StubClient;

#[async_trait]
impl RoutedLlmClient for StubClient {
    async fn call(&self, request: Request) -> Result<Response, LlmClientError> {
        Ok(Response {
            llm_response: LlmResponse::Agg(text_response(
                request.llm_request.model.clone(),
                "plugin response",
            )),
            metadata: None,
            upstream_headers: Default::default(),
        })
    }
}

fn plugin_route(client: Arc<dyn RoutedLlmClient>) -> Route {
    let spec = AlgorithmSpec::Passthrough {
        target: "semantic-target".to_string(),
        candidates: Vec::new(),
        subagents: None,
    };
    let targets = BTreeMap::from([(
        "semantic-target".to_string(),
        ModelId::from("semantic-target"),
    )]);
    let algorithm = spec
        .build("switchyard", &targets)
        .expect("identity target map should build");
    let clients = ClientRouter::new(
        BTreeMap::from([(ModelId::from("semantic-target"), client)])
            .into_iter()
            .collect(),
    );
    Route::new(
        algorithm,
        clients,
        None,
        ModelCapabilities::default(),
        None,
        None,
        Vec::new(),
        RuntimeModels::new([(Category::Any, vec![ModelId::from("semantic-target")])].into()),
        switchyard_runner::RouteReasoningPolicy::Inherit,
        None,
        None,
    )
}

#[tokio::test]
async fn plugin_shaped_route_executes_without_runner_model_or_toml() {
    let route = plugin_route(Arc::new(StubClient));
    let observations = Arc::new(Mutex::new(Vec::new()));
    let observer = {
        let observations = Arc::clone(&observations);
        Arc::new(move |observation| observations.lock().unwrap().push(observation))
    };
    let request = Request {
        llm_request: text_request(Some("arbitrary-upstream-model".to_string()), "hello"),
        ..Request::default()
    };
    let output = route
        .execute(request, Some(observer))
        .await
        .expect("route should execute");

    assert_eq!(output.selected_model, "semantic-target");
    assert_eq!(
        output
            .response
            .llm_response
            .as_agg()
            .unwrap()
            .model
            .as_deref(),
        Some("semantic-target")
    );
    assert!(
        observations
            .lock()
            .unwrap()
            .iter()
            .any(|observation| { matches!(observation, RunObservation::AnswerCall(_)) })
    );
    assert!(
        observations
            .lock()
            .unwrap()
            .iter()
            .any(|observation| { matches!(observation, RunObservation::RoutingOverhead(_)) })
    );
}

struct LazyStreamClient {
    polls: Arc<AtomicUsize>,
}

#[async_trait]
impl RoutedLlmClient for LazyStreamClient {
    async fn call(&self, _request: Request) -> Result<Response, LlmClientError> {
        let polls = Arc::clone(&self.polls);
        let stream = futures_util::stream::poll_fn(move |_| {
            polls.fetch_add(1, Ordering::SeqCst);
            std::task::Poll::Ready(None)
        })
        .boxed();
        Ok(Response {
            llm_response: LlmResponse::Stream(stream),
            metadata: None,
            upstream_headers: Default::default(),
        })
    }
}

#[tokio::test]
async fn route_returns_stream_without_polling_it() {
    let polls = Arc::new(AtomicUsize::new(0));
    let route = plugin_route(Arc::new(LazyStreamClient {
        polls: Arc::clone(&polls),
    }));
    let request = Request {
        llm_request: text_request(None, "hello"),
        ..Request::default()
    };

    let output = route
        .execute(request, None)
        .await
        .expect("stream handle should be returned");

    assert!(matches!(
        output.response.llm_response,
        LlmResponse::Stream(_)
    ));
    assert_eq!(polls.load(Ordering::SeqCst), 0);
}

/// Two distinct clients so the selected target is observable from which one answered,
/// not merely from a stored field.
struct EchoClient {
    model: &'static str,
    calls: Arc<Mutex<Vec<&'static str>>>,
}

#[async_trait]
impl RoutedLlmClient for EchoClient {
    async fn call(&self, request: Request) -> Result<Response, LlmClientError> {
        self.calls.lock().unwrap().push(self.model);
        if self.model == "refused" {
            // Classified as "another candidate may serve this", which is what makes the
            // walk continue to the legs behind it.
            return Err(LlmClientError::TemporarilyUnavailable);
        }
        Ok(Response {
            llm_response: LlmResponse::Agg(text_response(
                request.llm_request.model.clone(),
                "ok",
            )),
            metadata: None,
            upstream_headers: Default::default(),
        })
    }
}

/// A local route whose narrow leg is primary and whose wide leg is the escalation target.
///
/// A third, always-failing leg sits behind the wide one. It is what makes the tail of the
/// candidate list observable: if promotion dropped or reordered the candidates behind the
/// promoted target, this leg would be skipped and the fallback would stop short.
fn escalating_route(
    threshold: Option<u64>,
    escalation_target: Option<&str>,
    calls: &Arc<Mutex<Vec<&'static str>>>,
) -> Route {
    let spec = AlgorithmSpec::Passthrough {
        target: "narrow".to_string(),
        candidates: vec!["wide".to_string(), "refused".to_string()],
        subagents: None,
    };
    let targets = BTreeMap::from([
        ("narrow".to_string(), ModelId::from("narrow")),
        ("wide".to_string(), ModelId::from("wide")),
        ("refused".to_string(), ModelId::from("refused")),
    ]);
    let algorithm = spec.build("switchyard", &targets).expect("algorithm builds");
    let clients = ClientRouter::new(
        BTreeMap::from([
            (
                ModelId::from("narrow"),
                Arc::new(EchoClient {
                    model: "narrow",
                    calls: Arc::clone(calls),
                }) as Arc<dyn RoutedLlmClient>,
            ),
            (
                ModelId::from("wide"),
                Arc::new(EchoClient {
                    model: "wide",
                    calls: Arc::clone(calls),
                }) as Arc<dyn RoutedLlmClient>,
            ),
            (
                ModelId::from("refused"),
                Arc::new(EchoClient {
                    model: "refused",
                    calls: Arc::clone(calls),
                }) as Arc<dyn RoutedLlmClient>,
            ),
        ])
        .into_iter()
        .collect(),
    );
    Route::new(
        algorithm,
        clients,
        None,
        ModelCapabilities::default(),
        None,
        None,
        Vec::new(),
        RuntimeModels::new(HashMap::from([(
            Category::Any,
            vec![
                ModelId::from("narrow"),
                ModelId::from("wide"),
                ModelId::from("refused"),
            ],
        )])),
        switchyard_runner::RouteReasoningPolicy::Inherit,
        threshold,
        escalation_target.map(ModelId::from),
    )
}

/// A body whose recovered estimate is exactly `tokens`.
///
/// The estimator charges ASCII `div_ceil(4)` per string plus one per string, plus 8
/// structural tokens. Two strings are fixed here ("m" and "user"), contributing 2 + 2, so
/// the repeated text must contribute `tokens - 12` and each of its 4-char groups is worth
/// one token: n = (tokens - 13) * 4. Asserting the estimate keeps the fixture and the
/// comparator in step, so a fixture cannot silently drift off the boundary.
fn body_of_estimated_tokens(tokens: u64) -> serde_json::Value {
    let body = serde_json::json!({
        "model": "m",
        "messages": [{"role": "user", "content": "q".repeat(((tokens - 13) * 4) as usize)}],
    });
    assert_eq!(
        estimate_input_tokens(&body),
        tokens,
        "fixture must land on the exact boundary"
    );
    body
}

/// The historical comparator is strictly greater-than: 61,440 stays on the narrow primary
/// and only 61,441 moves to the declared escalation leg before any provider is called.
#[tokio::test]
async fn context_pressure_reorders_the_escalation_target_before_dispatch() {
    let at_threshold = Arc::new(Mutex::new(Vec::new()));
    let route = escalating_route(Some(61_440), Some("wide"), &at_threshold);
    let output = route
        .execute(
            Request {
                llm_request: text_request(None, "hello"),
                raw_request: Some(body_of_estimated_tokens(61_440)),
                ..Request::default()
            },
            None,
        )
        .await
        .expect("request at the threshold is served");
    assert_eq!(output.selected_model, "narrow", "61,440 must not escalate");
    assert_eq!(*at_threshold.lock().unwrap(), ["narrow"]);

    let above = Arc::new(Mutex::new(Vec::new()));
    let route = escalating_route(Some(61_440), Some("wide"), &above);
    let output = route
        .execute(
            Request {
                llm_request: text_request(None, "hello"),
                raw_request: Some(body_of_estimated_tokens(61_441)),
                ..Request::default()
            },
            None,
        )
        .await
        .expect("request above the threshold is served");
    assert_eq!(output.selected_model, "wide", "61,441 must escalate");
    assert_eq!(
        *above.lock().unwrap(),
        ["wide"],
        "the narrow leg must receive no provider call"
    );
}

/// A route that declares no threshold, or a threshold with no escalation target, keeps its
/// declared order. This is the property that stops the reorder leaking into other routes.
#[tokio::test]
async fn no_escalation_contract_leaves_the_declared_order_intact() {
    for (threshold, escalation_target) in
        [(None, Some("wide")), (Some(61_440), None), (None, None)]
    {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let route = escalating_route(threshold, escalation_target, &calls);
        let output = route
            .execute(
                Request {
                    llm_request: text_request(None, "hello"),
                    raw_request: Some(body_of_estimated_tokens(400_000)),
                    ..Request::default()
                },
                None,
            )
            .await
            .expect("request is served");
        assert_eq!(
            output.selected_model, "narrow",
            "threshold {threshold:?} with target {escalation_target:?} must not reorder"
        );
        assert_eq!(*calls.lock().unwrap(), ["narrow"]);
    }
}

/// Promotion reorders; it must not shorten the list. With the promoted leg forced to fail,
/// the candidates behind it are still reached in their declared relative order.
#[tokio::test]
async fn promotion_keeps_the_rest_of_the_candidate_list() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let route = escalating_route(Some(61_440), Some("refused"), &calls);
    let output = route
        .execute(
            Request {
                llm_request: text_request(None, "hello"),
                raw_request: Some(body_of_estimated_tokens(61_441)),
                ..Request::default()
            },
            None,
        )
        .await
        .expect("the request is served by a later leg");
    // `selected_model` is the first model attempted, so the promoted leg is expected there
    // even though it failed. The client log is the authority on which legs were actually
    // walked: the response model only echoes the rewritten request, so it cannot show this.
    assert_eq!(
        output.selected_model, "refused",
        "the promoted leg is attempted first"
    );
    assert_eq!(
        *calls.lock().unwrap(),
        ["refused", "narrow"],
        "the promoted leg is tried first, then the remaining legs in declared order until \
         one serves the request"
    );
}

// Preserve checkpoint target order and explicit TOML overrides.
#[test]
fn prefill_router_config_preserves_target_order_and_overrides() {
    let spec: AlgorithmSpec = toml::from_str(
        r#"
type = "prefill_router"
targets = ["fast", "strong"]
checkpoint = "/models/router.pt"
device = "cuda:1"
max_length = 4096
batch_size = 8
"#,
    )
    .expect("prefill router config should parse");

    let AlgorithmSpec::PrefillRouter {
        targets,
        checkpoint,
        device,
        cache_dir,
        max_length,
        batch_size,
    } = spec
    else {
        panic!("expected prefill router config");
    };
    assert_eq!(targets, ["fast", "strong"]);
    assert_eq!(checkpoint.to_string_lossy(), "/models/router.pt");
    assert_eq!(device.as_deref(), Some("cuda:1"));
    assert_eq!(cache_dir, None);
    assert_eq!(max_length, Some(4096));
    assert_eq!(batch_size, Some(8));
}
