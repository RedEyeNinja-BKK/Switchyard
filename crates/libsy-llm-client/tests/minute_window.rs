// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-minute admission enforced at the send path.
//!
//! The unit tests in `allowance` prove the window arithmetic. These prove the three things
//! that matter operationally: a full window stops the request *before* it leaves, a refusal
//! is not retried on the same leg (which is how one throttle becomes a burst), and every
//! name for one physical model draws on one budget so aliases cannot double-spend a minute.

use std::error::Error;
use std::sync::Arc;
use std::time::Duration;

use serde_json::json;
use switchyard_llm_client::{
    Backend, DailyRequestAllowance, GovernorKey, HttpBackendConfig, LlmClientError, ModelConfig,
    PacedQuota, QuotaGovernor, TranslatingLlmClient, WindowLimits, allowance_path,
};
use switchyard_protocol::{ModelId, Request, text_request};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const GEMMA: &str = "google/gemma-4-26b-a4b-it";
const GEMMA_31B: &str = "google/gemma-4-31b-it";
const GROUP: &str = "google-free";

fn scratch(label: &str) -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    std::env::temp_dir()
        .join("switchyard-minute-window")
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

fn backend(
    base_url: &str,
    max_retries: u32,
    allowance: Option<Arc<DailyRequestAllowance>>,
    pacing: Option<Arc<PacedQuota>>,
) -> Backend {
    Backend::OpenAiChat(HttpBackendConfig {
        tool_signature_placeholder: None,
        base_url: base_url.to_string(),
        api_key: Some("secret".to_string()),
        forward_auth: false,
        extra_headers: Default::default(),
        extra_body: Default::default(),
        omit_body_fields: Default::default(),
        unsupported_reasoning_effort: Default::default(),
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
        pacing,
        hosted_reasoning: None,
    })
}

fn client_for(
    base_url: &str,
    max_retries: u32,
    allowance: Option<Arc<DailyRequestAllowance>>,
    pacing: Option<Arc<PacedQuota>>,
) -> Result<TranslatingLlmClient, Box<dyn Error + Send + Sync>> {
    Ok(TranslatingLlmClient::new(&[ModelConfig::new(
        "gpt",
        backend(base_url, max_retries, allowance, pacing),
        None,
    )])?)
}

fn paced(governor: &Arc<QuotaGovernor>, model: &str, limits: WindowLimits) -> Arc<PacedQuota> {
    Arc::new(PacedQuota::new(
        Arc::clone(governor),
        GovernorKey::new(GROUP, ModelId::from(model), "generate"),
        limits,
        switchyard_llm_client::AdmissionMargin::NONE,
    ))
}

fn request() -> Request {
    Request {
        llm_request: text_request(Some("gpt".to_string()), "hi"),
        ..Request::default()
    }
}

fn long_request(model: &str, characters: usize) -> Request {
    Request {
        llm_request: text_request(Some(model.to_string()), "a".repeat(characters)),
        ..Request::default()
    }
}

/// The second request inside a one-request window is refused locally, the provider never
/// sees it, and the client does not retry into the window it just exhausted.
#[tokio::test]
async fn a_full_minute_window_refuses_before_the_request_leaves()
-> Result<(), Box<dyn Error + Send + Sync>> {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(completion())
        // One request total: three retries were configured and none may be used here.
        .expect(1)
        .mount(&server)
        .await;
    let governor = Arc::new(QuotaGovernor::new());
    let limits = WindowLimits {
        requests_per_minute: Some(1),
        input_tokens_per_minute: None,
    };
    let client = client_for(
        &server.uri(),
        3,
        None,
        Some(paced(&governor, GEMMA, limits)),
    )?;

    client.call_rewrite_model(request(), None).await?;
    let refused = client.call_rewrite_model(request(), None).await;

    match refused {
        Err(LlmClientError::MinuteAllowanceThrottled {
            model,
            scope,
            retry_after_secs,
        }) => {
            assert_eq!(model.as_str(), "gpt");
            assert_eq!(scope, "requests_per_minute");
            assert!((1..=60).contains(&retry_after_secs), "{retry_after_secs}");
        }
        Ok(_) => panic!("expected the window to refuse the second request"),
        Err(other) => panic!("expected a minute throttle, got {other:?}"),
    }
    Ok(())
}

/// The input-token axis refuses before a large prompt leaves, even when requests remain.
#[tokio::test]
async fn the_input_token_axis_refuses_a_prompt_that_would_overfill_the_window()
-> Result<(), Box<dyn Error + Send + Sync>> {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(completion())
        // Three prompts of about 5,001 tokens fit a 16,000-token window; the fourth
        // cannot, and is refused before it is sent.
        .expect(3)
        .mount(&server)
        .await;
    let governor = Arc::new(QuotaGovernor::new());
    let limits = WindowLimits {
        requests_per_minute: Some(100),
        input_tokens_per_minute: Some(16_000),
    };
    let client = client_for(
        &server.uri(),
        0,
        None,
        Some(paced(&governor, GEMMA, limits)),
    )?;

    for _ in 0..3 {
        client
            .call_rewrite_model(long_request("gpt", 20_000), None)
            .await?;
    }
    match client
        .call_rewrite_model(long_request("gpt", 20_000), None)
        .await
    {
        Err(LlmClientError::MinuteAllowanceThrottled { scope, .. }) => {
            assert_eq!(scope, "input_tokens_per_minute");
        }
        Ok(_) => panic!("expected the token window to refuse the fourth prompt"),
        Err(other) => panic!("expected a token throttle, got {other:?}"),
    }
    Ok(())
}

/// Two names for one physical model draw on one window: the second name cannot spend a
/// minute the first already used, which is what stops aliases multiplying the budget.
#[tokio::test]
async fn aliases_of_one_physical_model_cannot_double_spend()
-> Result<(), Box<dyn Error + Send + Sync>> {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(completion())
        .expect(1)
        .mount(&server)
        .await;
    let governor = Arc::new(QuotaGovernor::new());
    let limits = WindowLimits {
        requests_per_minute: Some(1),
        input_tokens_per_minute: None,
    };
    let shared = paced(&governor, GEMMA, limits);
    let client = TranslatingLlmClient::new(&[
        ModelConfig::new(
            "direct-gemma",
            backend(&server.uri(), 0, None, Some(Arc::clone(&shared))),
            None,
        ),
        ModelConfig::new(
            "gemma-alias",
            backend(&server.uri(), 0, None, Some(Arc::clone(&shared))),
            None,
        ),
    ])?;

    client
        .call_rewrite_model(long_request("direct-gemma", 10), None)
        .await?;
    match client
        .call_rewrite_model(long_request("gemma-alias", 10), None)
        .await
    {
        Err(LlmClientError::MinuteAllowanceThrottled { model, .. }) => {
            assert_eq!(model.as_str(), "gemma-alias");
        }
        Ok(_) => panic!("the second name must not get a second window"),
        Err(other) => panic!("expected a minute throttle, got {other:?}"),
    }
    Ok(())
}

/// A different physical model keeps its own window.
#[tokio::test]
async fn a_sibling_model_is_not_paced_out_by_another_models_use()
-> Result<(), Box<dyn Error + Send + Sync>> {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(completion())
        .expect(2)
        .mount(&server)
        .await;
    let governor = Arc::new(QuotaGovernor::new());
    let limits = WindowLimits {
        requests_per_minute: Some(1),
        input_tokens_per_minute: None,
    };
    let client = TranslatingLlmClient::new(&[
        ModelConfig::new(
            "gemma-26b",
            backend(
                &server.uri(),
                0,
                None,
                Some(paced(&governor, GEMMA, limits)),
            ),
            None,
        ),
        ModelConfig::new(
            "gemma-31b",
            backend(
                &server.uri(),
                0,
                None,
                Some(paced(&governor, GEMMA_31B, limits)),
            ),
            None,
        ),
    ])?;

    client
        .call_rewrite_model(long_request("gemma-26b", 10), None)
        .await?;
    client
        .call_rewrite_model(long_request("gemma-31b", 10), None)
        .await?;
    Ok(())
}

/// A request refused by the daily budget must give its minute-window slot back: nothing
/// reached the provider, and spending the window on it would burst the day's traffic.
#[tokio::test]
async fn a_daily_refusal_returns_its_minute_window_slot() -> Result<(), Box<dyn Error + Send + Sync>>
{
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(completion())
        .expect(0)
        .mount(&server)
        .await;
    let directory = scratch("daily-refusal");
    // A zero daily budget is spent by definition; the point of the test is what happens to
    // the minute slot it was granted first.
    let allowance = Arc::new(DailyRequestAllowance::new(
        GROUP,
        ModelId::from(GEMMA),
        0,
        allowance_path(&directory, GROUP, GEMMA),
    ));
    let governor = Arc::new(QuotaGovernor::new());
    let limits = WindowLimits {
        requests_per_minute: Some(1),
        input_tokens_per_minute: None,
    };
    let quota = paced(&governor, GEMMA, limits);
    let client = client_for(&server.uri(), 0, Some(allowance), Some(Arc::clone(&quota)))?;

    for _ in 0..2 {
        match client.call_rewrite_model(request(), None).await {
            // If the window slot were kept, the second call would report a minute throttle
            // instead of the daily budget it is actually blocked by.
            Err(LlmClientError::DailyRequestAllowanceExhausted { .. }) => {}
            Ok(_) => panic!("a zero daily budget cannot serve a request"),
            Err(other) => panic!("expected the daily budget, got {other:?}"),
        }
    }
    assert_eq!(
        governor.used(quota.key(), std::time::SystemTime::now()),
        (0, 0),
        "a request that was never sent must not hold the window"
    );
    Ok(())
}

/// A target that declares no window is unaffected, which keeps every pre-existing manifest
/// behaving exactly as it did.
#[tokio::test]
async fn a_target_without_a_declared_window_is_unchanged()
-> Result<(), Box<dyn Error + Send + Sync>> {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(completion())
        .expect(3)
        .mount(&server)
        .await;
    let client = client_for(&server.uri(), 0, None, None)?;
    for _ in 0..3 {
        client.call_rewrite_model(request(), None).await?;
    }
    Ok(())
}
