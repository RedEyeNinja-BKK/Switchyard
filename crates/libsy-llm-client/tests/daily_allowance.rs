// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Daily request allowances enforced at the send path.
//!
//! The unit tests in `allowance` prove the counter arithmetic. These prove the part that
//! matters operationally: that a spent allowance stops the request *before* it leaves, that
//! a retry cannot spend a second unit it does not have, and that an unusable counter refuses
//! rather than sending unaccounted.

use std::error::Error;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use serde_json::json;
use switchyard_llm_client::{
    Backend, DailyRequestAllowance, HttpBackendConfig, LlmClientError, ModelConfig,
    TranslatingLlmClient, allowance_path,
};
use switchyard_protocol::{Request, text_request};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn scratch(label: &str) -> std::path::PathBuf {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    std::env::temp_dir()
        .join("switchyard-allowance-integration")
        .join(format!(
            "{}-{}-{label}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ))
}

fn completion() -> ResponseTemplate {
    ResponseTemplate::new(200).set_body_json(json!({
        "id": "chatcmpl-1",
        "model": "gpt",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "ok"},
            "finish_reason": "stop"
        }],
        "usage": {}
    }))
}

fn client_for(
    base_url: &str,
    max_retries: u32,
    allowance: Option<Arc<DailyRequestAllowance>>,
) -> Result<TranslatingLlmClient, Box<dyn Error + Send + Sync>> {
    let backend = HttpBackendConfig {
        base_url: base_url.to_string(),
        api_key: Some("secret".to_string()),
        forward_auth: false,
        extra_headers: Default::default(),
        extra_body: Default::default(),
        omit_body_fields: Default::default(),
        reasoning_effort: None,
        reasoning_policy: None,
        reasoning_dialect: Default::default(),
        max_retries,
        failure_cooldown: Duration::ZERO,
        cooldown_group: None,
        timeout: None,
        attempt_timeout: None,
        stream_idle_timeout: None,
        requires_list_input: false,
        allowance,
    };
    Ok(TranslatingLlmClient::new(&[ModelConfig::new(
        "gpt",
        Backend::OpenAiChat(backend),
        None,
    )])?)
}

fn allowance_at(directory: &std::path::Path, limit: u64) -> Arc<DailyRequestAllowance> {
    Arc::new(DailyRequestAllowance::new(
        "google-free",
        "google/gemini-test".into(),
        limit,
        allowance_path(directory, "google-free", "google/gemini-test"),
    ))
}

fn request() -> Request {
    Request {
        llm_request: text_request(Some("gpt".to_string()), "hi"),
        ..Request::default()
    }
}

/// The request after the limit is spent is refused locally, and the provider never sees it.
#[tokio::test]
async fn a_spent_allowance_stops_the_request_before_it_is_sent()
-> Result<(), Box<dyn Error + Send + Sync>> {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(completion())
        .expect(1)
        .mount(&server)
        .await;
    let directory = scratch("spent");
    let client = client_for(&server.uri(), 0, Some(allowance_at(&directory, 1)))?;

    client.call_rewrite_model(request(), None).await?;
    let refused = client.call_rewrite_model(request(), None).await;

    match refused {
        Err(LlmClientError::DailyRequestAllowanceExhausted { limit, model }) => {
            assert_eq!(limit, 1);
            assert_eq!(model.as_str(), "gpt");
        }
        Ok(_) => panic!("expected the allowance to refuse the second request"),
        Err(other) => panic!("expected the allowance, got {other:?}"),
    }
    // `expect(1)` above fails the test if the mock saw a second request.
    Ok(())
}

/// A retry cannot spend a unit the allowance does not have: the first attempt keeps its
/// reservation even though it failed, and the retry is refused rather than resent.
#[tokio::test]
async fn a_retry_does_not_resend_after_the_allowance_is_spent()
-> Result<(), Box<dyn Error + Send + Sync>> {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(500).set_body_string("upstream is unwell"))
        // One attempt only: a second would mean the reservation was refunded.
        .expect(1)
        .mount(&server)
        .await;
    let directory = scratch("retry");
    let client = client_for(&server.uri(), 3, Some(allowance_at(&directory, 1)))?;

    let result = client.call_rewrite_model(request(), None).await;

    match result {
        Err(LlmClientError::DailyRequestAllowanceExhausted { .. }) => {}
        Ok(_) => panic!("expected the allowance, not a success, to end the call"),
        Err(other) => panic!("expected the allowance, not the 500, to end the call: {other:?}"),
    }
    Ok(())
}

/// An unusable counter refuses the request instead of spending budget nobody is counting.
#[tokio::test]
async fn an_unwritable_counter_refuses_rather_than_sending()
-> Result<(), Box<dyn Error + Send + Sync>> {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(completion())
        .expect(0)
        .mount(&server)
        .await;
    // A regular file where a directory is needed: creating the parent must fail.
    let blocker = scratch("blocked");
    std::fs::create_dir_all(blocker.parent().expect("scratch has a parent"))?;
    std::fs::write(&blocker, b"not a directory")?;
    let path = blocker.join("counter.json");
    let allowance = Arc::new(DailyRequestAllowance::new(
        "google-free",
        "google/gemini-test".into(),
        5,
        path,
    ));
    let client = client_for(&server.uri(), 0, Some(allowance))?;

    let result = client.call_rewrite_model(request(), None).await;

    match result {
        Err(LlmClientError::DailyRequestAllowanceUnavailable { reason, .. }) => {
            // The reason names the file and the fault. Depending on the state it is a
            // read, a create or a write that failed; all three are the same operational
            // story, and the request must not be sent in any of them.
            assert!(
                reason.contains("counter.json") && reason.contains("cannot be"),
                "the reason names the fault: {reason}"
            );
        }
        Ok(_) => panic!("expected the accounting fault to refuse the request"),
        Err(other) => panic!("expected the accounting fault, got {other:?}"),
    }
    Ok(())
}

/// Without a declared allowance nothing changes: the same call repeats freely.
#[tokio::test]
async fn an_undeclared_allowance_changes_nothing() -> Result<(), Box<dyn Error + Send + Sync>> {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(completion())
        .expect(3)
        .mount(&server)
        .await;
    let client = client_for(&server.uri(), 0, None)?;

    for _ in 0..3 {
        client.call_rewrite_model(request(), None).await?;
    }
    Ok(())
}
