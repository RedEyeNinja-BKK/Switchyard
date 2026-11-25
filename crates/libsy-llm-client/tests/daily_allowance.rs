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
    Backend, DailyRequestAllowance, DayBoundary, HttpBackendConfig, LlmClientError, ModelConfig,
    ProviderReading, ProviderState, ReserveError, TranslatingLlmClient, account_allowance_path,
    allowance_path,
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
        pacing: None,
        hosted_reasoning: None,
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

fn at(seconds: u64) -> std::time::SystemTime {
    std::time::UNIX_EPOCH + Duration::from_secs(seconds)
}

// Fixed instants around the two candidate midnights. 2026-01-15 is mid-winter, so
// Pacific is a clean UTC-8 with no daylight-saving transition nearby:
//   04:00Z = 2026-01-14 20:00 PST  - Pacific and UTC disagree about the date
//   23:00Z = 2026-01-15 15:00 PST  - last hour of the UTC day
//   00:30Z = 2026-01-15 16:30 PST  - next UTC day, same Pacific day
const PACIFIC_DATE_BEHIND: u64 = 1_768_449_600;
const LATE_UTC_DAY: u64 = 1_768_518_000;
const EARLY_NEXT_UTC_DAY: u64 = 1_768_523_400;

/// The two boundaries really are different days for the same instant; otherwise the
/// reset tests below could pass while the boundary was ignored.
#[test]
fn the_boundaries_disagree_about_which_day_an_instant_belongs_to() {
    let now = at(PACIFIC_DATE_BEHIND);
    let utc = DayBoundary::Utc.day(now);
    let pacific = DayBoundary::Pacific.day(now);
    assert_eq!(
        utc,
        pacific + 1,
        "04:00Z is the next UTC day but the same Pacific evening day"
    );
}

/// A UTC-scoped allowance turns over at 00:00Z: a request after that instant is served
/// from the new day's budget, while the same instant leaves a Pacific-scoped allowance
/// still inside yesterday's spent budget.
#[test]
fn utc_boundary_resets_at_midnight_utc_and_pacific_does_not() {
    let directory = scratch("utc-reset");
    let utc = DailyRequestAllowance::new_on(
        "openrouter-free",
        "vendor/free-a".into(),
        1,
        account_allowance_path(&directory, "openrouter-free"),
        DayBoundary::Utc,
    );

    utc.try_reserve_at(at(LATE_UTC_DAY))
        .expect("first request of the UTC day");
    assert!(
        matches!(
            utc.try_reserve_at(at(LATE_UTC_DAY)),
            Err(ReserveError::Exhausted)
        ),
        "the UTC day's single unit must be spent"
    );
    utc.try_reserve_at(at(EARLY_NEXT_UTC_DAY))
        .expect("00:30Z is a new UTC day, so the budget is restored");

    let directory = scratch("pacific-reset");
    let pacific = DailyRequestAllowance::new_on(
        "google-free",
        "google/gemini-test".into(),
        1,
        allowance_path(&directory, "google-free", "google/gemini-test"),
        DayBoundary::Pacific,
    );

    pacific
        .try_reserve_at(at(LATE_UTC_DAY))
        .expect("first request of the Pacific day");
    assert!(
        matches!(
            pacific.try_reserve_at(at(EARLY_NEXT_UTC_DAY)),
            Err(ReserveError::Exhausted)
        ),
        "00:30Z is still 16:30 PST, so a Pacific allowance must not reset"
    );
}

/// The declared default is Pacific, so an allowance built with `new` keeps the historical
/// behaviour and a config that names no boundary does not silently change when it resets.
#[test]
fn new_still_defaults_to_the_pacific_boundary() {
    let allowance = DailyRequestAllowance::new(
        "google-free",
        "google/gemini-test".into(),
        1,
        allowance_path(
            &scratch("default-boundary"),
            "google-free",
            "google/gemini-test",
        ),
    );
    assert_eq!(allowance.boundary(), DayBoundary::Pacific);
}

/// An account-scoped allowance is one counter for every model that draws on it: two
/// different models on the same account path share the budget, so the second model is
/// refused once the first model has spent it.
#[test]
fn an_account_scope_shares_one_budget_across_models() {
    let directory = scratch("account-scope");
    let path = account_allowance_path(&directory, "openrouter-free");

    let a = DailyRequestAllowance::new_on(
        "openrouter-free",
        "vendor/free-a".into(),
        2,
        path.clone(),
        DayBoundary::Utc,
    );
    let b = DailyRequestAllowance::new_on(
        "openrouter-free",
        "vendor/free-b".into(),
        2,
        path,
        DayBoundary::Utc,
    );

    a.try_reserve_at(at(LATE_UTC_DAY)).expect("a's first");
    a.try_reserve_at(at(LATE_UTC_DAY)).expect("a's second");
    assert!(
        matches!(
            b.try_reserve_at(at(LATE_UTC_DAY)),
            Err(ReserveError::Exhausted)
        ),
        "b must see a's spend, because the account budget is shared"
    );
}

/// The contrast case: without the account sentinel, the same two models on their own
/// paths each get their own budget. This is the behaviour the account scope replaces, and
/// it must stay available for providers that cap per model.
#[test]
fn per_model_paths_keep_separate_budgets() {
    let directory = scratch("per-model");
    let a = DailyRequestAllowance::new_on(
        "openrouter-free",
        "vendor/free-a".into(),
        1,
        allowance_path(&directory, "openrouter-free", "vendor/free-a"),
        DayBoundary::Utc,
    );
    let b = DailyRequestAllowance::new_on(
        "openrouter-free",
        "vendor/free-b".into(),
        1,
        allowance_path(&directory, "openrouter-free", "vendor/free-b"),
        DayBoundary::Utc,
    );

    a.try_reserve_at(at(LATE_UTC_DAY)).expect("a's own unit");
    b.try_reserve_at(at(LATE_UTC_DAY))
        .expect("b has its own unit when the counter is keyed per model");
}

// ---------------------------------------------------------------------------
// Provider-sourced reconciliation
//
// A local counter sees only this deployment's traffic. The provider sees the whole
// account, so only its reading can confirm the account is empty -- and only while the
// reading is current. These cases pin the three states apart, because collapsing any two
// of them produces a wrong admission: an unknown reading must not refuse, and a stale one
// must not confirm.
// ---------------------------------------------------------------------------

fn provider(used: u64, limit: u64, remaining: u64, observed: u64) -> ProviderReading {
    ProviderReading {
        used,
        limit,
        remaining,
        observed_at: at(observed),
        reset_at: None,
    }
}

fn account_allowance(directory: &std::path::Path, limit: u64) -> DailyRequestAllowance {
    DailyRequestAllowance::new_on(
        "openrouter-free",
        "vendor/free-a".into(),
        limit,
        account_allowance_path(directory, "openrouter-free"),
        DayBoundary::Utc,
    )
}

/// A provider that reports no headroom refuses the request, and the refusal is reported as
/// the provider's, not as this deployment's own budget running out.
#[test]
fn a_provider_confirmed_exhaustion_refuses_with_its_own_reason() {
    let allowance = account_allowance(&scratch("provider-exhausted"), 1000);
    allowance
        .reconcile(provider(1000, 1000, 0, LATE_UTC_DAY))
        .expect("the reading is durable");

    match allowance.try_reserve_at(at(LATE_UTC_DAY)) {
        Err(ReserveError::ProviderExhausted { limit }) => assert_eq!(limit, 1000),
        other => panic!("expected the provider's refusal, got {other:?}"),
    }
    // The local budget is untouched by a refused request.
    assert_eq!(allowance.remaining_at(at(LATE_UTC_DAY)), 0);
}

/// The same reading, older than the tolerance, is not evidence: the account may have
/// recovered, so the local budget governs and the request is admitted.
#[test]
fn a_stale_reading_does_not_confirm_exhaustion() {
    let allowance = account_allowance(&scratch("provider-stale"), 10)
        .with_provider_tolerance(Duration::from_secs(60));
    allowance
        .reconcile(provider(1000, 1000, 0, LATE_UTC_DAY))
        .expect("the reading is durable");

    let much_later = LATE_UTC_DAY + 3600;
    assert_eq!(
        allowance.provider_state_at(at(much_later)),
        ProviderState::Unknown,
        "an hour-old reading cannot say the account is empty now"
    );
    allowance
        .try_reserve_at(at(much_later))
        .expect("a stale reading must not refuse traffic the account may now permit");
}

/// A reading from a previous provider day describes a window that has ended, however
/// recently it was taken. It cannot constrain the new day.
#[test]
fn a_reading_from_yesterday_cannot_refuse_today() {
    let allowance = account_allowance(&scratch("provider-yesterday"), 10);
    allowance
        .reconcile(provider(1000, 1000, 0, LATE_UTC_DAY))
        .expect("the reading is durable");

    // 00:30Z the next day: a fresh clock, a new UTC day, one hour after the reading.
    let next_day = EARLY_NEXT_UTC_DAY;
    assert_eq!(
        allowance.provider_state_at(at(next_day)),
        ProviderState::Unknown
    );
    allowance
        .try_reserve_at(at(next_day))
        .expect("yesterday's exhaustion must not refuse today's request");
}

/// Reconcile adopts the reading's day, and the local spend must not follow it across.
/// Yesterday's three sends were spent against a window that has closed; carrying them
/// into today would silently shrink today's budget by a number the provider has already
/// forgiven. The provider reading here is deliberately generous so the local figure is
/// the only thing that can move.
#[test]
fn yesterdays_spend_does_not_consume_todays_budget() {
    let directory = scratch("provider-carry");
    let allowance = account_allowance(&directory, 10);
    for _ in 0..3 {
        allowance
            .try_reserve_at(at(LATE_UTC_DAY))
            .expect("yesterday has room");
    }
    assert_eq!(allowance.remaining_at(at(LATE_UTC_DAY)), 7);

    allowance
        .reconcile(provider(0, 1000, 1000, EARLY_NEXT_UTC_DAY))
        .expect("the reading is durable");

    assert_eq!(
        allowance.remaining_at(at(EARLY_NEXT_UTC_DAY)),
        10,
        "a new provider day gives back the whole local budget"
    );
}

/// No reading at all is Unknown, which is not exhaustion. A deployment that never
/// collected from the provider must keep serving on its own budget.
#[test]
fn no_reading_is_unknown_and_admits() {
    let allowance = account_allowance(&scratch("provider-none"), 2);
    assert_eq!(
        allowance.provider_state_at(at(LATE_UTC_DAY)),
        ProviderState::Unknown
    );
    allowance
        .try_reserve_at(at(LATE_UTC_DAY))
        .expect("with no reading, the local budget governs");
}

/// When the provider reports headroom, the reported remainder bounds admission even though
/// the local budget has more room: another client may have spent most of the account.
#[test]
fn provider_headroom_bounds_the_local_budget() {
    let allowance = account_allowance(&scratch("provider-bounds"), 1000);
    allowance
        .reconcile(provider(997, 1000, 3, LATE_UTC_DAY))
        .expect("the reading is durable");

    assert_eq!(
        allowance.remaining_at(at(LATE_UTC_DAY)),
        3,
        "the provider's remainder is smaller than the local budget and must win"
    );
    for _ in 0..3 {
        allowance
            .try_reserve_at(at(LATE_UTC_DAY))
            .expect("within both");
    }
    // Three sends carry the anchor from 0 spent to the provider's reported remainder of 3,
    // so the refusal is the provider's allowance running out - not the local budget, which
    // still has 997. Classifying it as Exhausted would blame the wrong limit and, worse,
    // would be indistinguishable from a local wall for routing purposes.
    assert!(matches!(
        allowance.try_reserve_at(at(LATE_UTC_DAY)),
        Err(ReserveError::ProviderExhausted { limit: 1000 })
    ));
}

/// The reading survives a restart: a new object over the same path sees what the provider
/// said, because it is in the counter file rather than in process memory.
#[test]
fn the_reading_is_durable_across_a_restart() {
    let directory = scratch("provider-restart");
    account_allowance(&directory, 1000)
        .reconcile(provider(1000, 1000, 0, LATE_UTC_DAY))
        .expect("the reading is durable");

    let after_restart = account_allowance(&directory, 1000);
    assert_eq!(
        after_restart.provider_state_at(at(LATE_UTC_DAY)),
        ProviderState::Exhausted { limit: 1000 }
    );
    assert!(matches!(
        after_restart.try_reserve_at(at(LATE_UTC_DAY)),
        Err(ReserveError::ProviderExhausted { .. })
    ));
}

/// A provider reading is account-scoped: two models sharing the account path must both see
/// it, because it describes the account, not either model.
#[test]
fn a_reading_applies_to_every_model_on_the_account() {
    let directory = scratch("provider-account-share");
    let path = account_allowance_path(&directory, "openrouter-free");
    let a = DailyRequestAllowance::new_on(
        "openrouter-free",
        "vendor/free-a".into(),
        1000,
        path.clone(),
        DayBoundary::Utc,
    );
    let b = DailyRequestAllowance::new_on(
        "openrouter-free",
        "vendor/free-b".into(),
        1000,
        path,
        DayBoundary::Utc,
    );

    a.reconcile(provider(1000, 1000, 0, LATE_UTC_DAY))
        .expect("the reading is durable");
    assert!(
        matches!(
            b.try_reserve_at(at(LATE_UTC_DAY)),
            Err(ReserveError::ProviderExhausted { .. })
        ),
        "the account is empty for every model on it, not just the one that read the count"
    );
}

/// The counter file states which authority admission rests on, so an operator can tell a
/// locally-budgeted counter from a reconciled one without reading the code.
#[test]
fn the_counter_records_its_basis_and_the_provider_figures() {
    let directory = scratch("provider-basis");
    let allowance = account_allowance(&directory, 1000);
    allowance
        .reconcile(provider(997, 1000, 3, LATE_UTC_DAY))
        .expect("the reading is durable");

    let body = std::fs::read_to_string(account_allowance_path(&directory, "openrouter-free"))
        .expect("the counter exists");
    assert!(
        body.contains("\"basis\":\"local_budget+provider_reconciled\""),
        "{body}"
    );
    assert!(body.contains("\"used\":997"), "{body}");
    assert!(body.contains("\"remaining\":3"), "{body}");
}
