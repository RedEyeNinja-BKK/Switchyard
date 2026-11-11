// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What a composite reports when nothing answered.
//!
//! Two ways reach that end and they are not the same diagnosis. If every child was skipped,
//! the request asks for something no child declares it can serve. If a child was contacted
//! and failed, the request was routable and the provider refused it — reporting the skip
//! instead would tell the caller the opposite of what happened.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use async_trait::async_trait;
use libsy::RuntimeModels;
use serde_json::json;
use switchyard_llm_client::{CapabilityFacts, ClientRouter};
use switchyard_protocol::{
    Category, LlmClientError, LlmResponse, ModelId, Request, Response, RoutedLlmClient,
    text_response,
};
use switchyard_runner::{
    AlgorithmSpec, MediaCarrier, ModelCapabilities, Route, Runner, RunnerError,
    RouteReasoningPolicy,
};
use switchyard_translation::{WireFormat, decode_request};

/// Fails the way an unavailable backend does, which is the class another child may recover
/// from and therefore the class that lets the composite reach its terminal path.
struct FailingClient;

#[async_trait]
impl RoutedLlmClient for FailingClient {
    async fn call(&self, _request: Request) -> Result<Response, LlmClientError> {
        Err(LlmClientError::TemporarilyUnavailable)
    }
}

struct ServingClient;

#[async_trait]
impl RoutedLlmClient for ServingClient {
    async fn call(&self, request: Request) -> Result<Response, LlmClientError> {
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

fn route(
    name: &str,
    capabilities: ModelCapabilities,
    client: Arc<dyn RoutedLlmClient>,
) -> (ModelId, Route) {
    let target = format!("{name}-target");
    let algorithm = AlgorithmSpec::Passthrough {
        target: target.clone(),
        candidates: Vec::new(),
        subagents: None,
    }
    .build(name, &BTreeMap::from([(target.clone(), ModelId::from(target.as_str()))]))
    .expect("single target builds");
    // The eligibility filter reads capability *facts* on the client, not the route's
    // declaration, and video is refused to an uncharacterised target by design. Declaring
    // the facts here is what lets the stub actually be called.
    let clients = ClientRouter::new(
        BTreeMap::from([(ModelId::from(target.as_str()), client)])
            .into_iter()
            .collect(),
    )
    .with_target_capabilities(HashMap::from([(
        ModelId::from(target.as_str()),
        CapabilityFacts {
            no_video: capabilities.video != Some(true),
            no_remote_video: capabilities.video_carrier != Some(MediaCarrier::Remote)
                && capabilities.video_carrier != Some(MediaCarrier::Both),
            no_inline_video: capabilities.video_carrier != Some(MediaCarrier::Inline)
                && capabilities.video_carrier != Some(MediaCarrier::Both),
            ..CapabilityFacts::default()
        },
    )]));
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
            RuntimeModels::new(
                [(Category::Any, vec![ModelId::from(target.as_str())])].into(),
            ),
            RouteReasoningPolicy::Inherit,
            None,
            None,
        ),
    )
}

/// A request carrying inline video, decoded the way ingress decodes it so the media check
/// sees a `ContentBlock::Video` rather than an unparsed body.
fn video_request() -> Request {
    let body = json!({
        "messages": [{"role": "user", "content": [
            {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AAAA"}}
        ]}]
    });
    let llm_request =
        decode_request(WireFormat::OpenAiChat, &body).expect("chat video body decodes");
    Request {
        llm_request,
        raw_request: Some(body),
        ..Request::default()
    }
}

fn video_capable() -> ModelCapabilities {
    ModelCapabilities {
        video: Some(true),
        video_carrier: Some(MediaCarrier::Inline),
        ..ModelCapabilities::default()
    }
}

/// A child that was called and refused is the answer. The sibling that was skipped for its
/// capabilities says nothing about why nothing was served, so it must not be reported.
#[tokio::test]
async fn a_contacted_child_failure_is_reported_instead_of_a_siblings_skip() {
    let runner = Runner::new(vec![
        route("text_only", ModelCapabilities::default(), Arc::new(ServingClient)),
        route("video_child", video_capable(), Arc::new(FailingClient)),
    ])
    .with_composite(
        "composite",
        vec!["text_only".to_string(), "video_child".to_string()],
    );

    let error = runner
        .execute_route("composite", video_request(), None)
        .await
        .err()
        .expect("the only eligible child failed, so the composite fails");

    assert!(
        !matches!(error, RunnerError::UnsupportedCapability { .. }),
        "a request that reached a provider was routable: {error}"
    );
    assert!(
        error.to_string().contains("temporarily unavailable"),
        "the provider's own failure must survive: {error}"
    );
}

/// With every child skipped nothing was contacted, so the caller is told what the request
/// needed. This is the branch the test above must not take.
#[tokio::test]
async fn every_child_skipped_still_names_the_missing_capability() {
    let runner = Runner::new(vec![
        route("first_text_only", ModelCapabilities::default(), Arc::new(ServingClient)),
        route("second_text_only", ModelCapabilities::default(), Arc::new(ServingClient)),
    ])
    .with_composite(
        "composite",
        vec!["first_text_only".to_string(), "second_text_only".to_string()],
    );

    let error = runner
        .execute_route("composite", video_request(), None)
        .await
        .err()
        .expect("no child declares video, so the composite refuses it");

    match error {
        RunnerError::UnsupportedCapability { model, capability } => {
            assert_eq!(model, "composite");
            assert_eq!(capability, "video");
        }
        other => panic!("expected the capability refusal, got {other}"),
    }
}
