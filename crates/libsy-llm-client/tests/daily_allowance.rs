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
    Backend, DailyRequestAllowance, DayBoundary, ExhaustionBasis, HttpBackendConfig,
    LlmClientError, ModelConfig, ProviderReading, ProviderState, ReserveError,
    TranslatingLlmClient, account_allowance_path, allowance_path,
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
    // A tolerance wide enough to outlive the day boundary on purpose. Otherwise the reading
    // would be Unknown for being stale, and this case would pass without ever exercising
    // the rule it names -- the day the reading belongs to.
    let allowance = account_allowance(&scratch("provider-yesterday"), 10)
        .with_provider_tolerance(Duration::from_secs(2 * 86_400));
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
        ProviderState::Exhausted {
            limit: 1000,
            basis: ExhaustionBasis::Confirmed
        }
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

// ---------------------------------------------------------------------------
// A reading is data, not an instruction: ordering, freshness and the difference
// between a provider's raw figure and the headroom this counter may actually use.
// ---------------------------------------------------------------------------

/// A reading that arrives after the provider day it describes has ended must not be
/// applied. Adopting it would move the counter back a day, discard the spend this
/// deployment has already made today, and hand back a budget the provider has already
/// closed the window on.
#[test]
fn a_delayed_previous_day_reading_does_not_reset_todays_spend() {
    let allowance = account_allowance(&scratch("order-prev-day"), 10);
    for _ in 0..3 {
        allowance
            .try_reserve_at(at(LATE_UTC_DAY))
            .expect("today has room");
    }
    assert_eq!(allowance.remaining_at(at(LATE_UTC_DAY)), 7);

    // Read yesterday at the same wall-clock time; applied now.
    let yesterday = LATE_UTC_DAY - 86_400;
    let ticket = allowance
        .begin_observation_at(at(yesterday))
        .expect("a ticket is taken before the read");
    let refused =
        allowance.reconcile_at(provider(0, 1000, 1000, yesterday), ticket, at(LATE_UTC_DAY));

    assert!(
        matches!(
            refused,
            Err(ReserveError::ObservationRejected { reason })
                if reason.contains("ended")
        ),
        "a reading for a closed window must be refused, got {refused:?}"
    );
    assert_eq!(
        allowance.remaining_at(at(LATE_UTC_DAY)),
        7,
        "today's three sends must survive a delayed reading from yesterday"
    );
    assert_eq!(
        allowance.spent_at(at(LATE_UTC_DAY)),
        3,
        "the counter must not adopt the reading's earlier day"
    );
}

/// An older observation of the same day must not replace a newer one. The older figure
/// was true before requests were made since; applying it restores headroom that has
/// already been spent.
#[test]
fn an_older_same_day_reading_does_not_restore_headroom() {
    let allowance = account_allowance(&scratch("order-reversed"), 1000);
    allowance
        .reconcile(provider(997, 1000, 3, LATE_UTC_DAY))
        .expect("the newer reading is applied");
    for _ in 0..3 {
        allowance
            .try_reserve_at(at(LATE_UTC_DAY))
            .expect("the provider said three remain");
    }
    assert_eq!(allowance.remaining_at(at(LATE_UTC_DAY)), 0);

    let older = LATE_UTC_DAY - 600;
    let ticket = allowance
        .begin_observation_at(at(older))
        .expect("a ticket is taken before the read");
    let refused = allowance.reconcile_at(provider(897, 1000, 100, older), ticket, at(LATE_UTC_DAY));

    assert!(
        matches!(
            refused,
            Err(ReserveError::ObservationRejected { reason })
                if reason.contains("later")
        ),
        "an out-of-order reading must be refused, got {refused:?}"
    );
    assert_eq!(
        allowance.remaining_at(at(LATE_UTC_DAY)),
        0,
        "a stale figure must not restore headroom that has been spent"
    );
}

/// The same observation delivered twice changes nothing. Re-applying it must not reset the
/// anchor either, which would forgive every request made since the first application.
#[test]
fn a_duplicate_reading_is_not_applied_twice() {
    let allowance = account_allowance(&scratch("order-duplicate"), 1000);
    allowance
        .reconcile(provider(900, 1000, 100, LATE_UTC_DAY))
        .expect("the first delivery is applied");
    for _ in 0..5 {
        allowance
            .try_reserve_at(at(LATE_UTC_DAY))
            .expect("within the reported headroom");
    }
    assert_eq!(allowance.remaining_at(at(LATE_UTC_DAY)), 95);

    let ticket = allowance
        .begin_observation_at(at(LATE_UTC_DAY))
        .expect("a ticket is taken before the read");
    let refused = allowance.reconcile_at(
        provider(900, 1000, 100, LATE_UTC_DAY),
        ticket,
        at(LATE_UTC_DAY),
    );

    assert!(
        matches!(refused, Err(ReserveError::ObservationRejected { .. })),
        "the same observation delivered again must be refused, got {refused:?}"
    );
    assert_eq!(
        allowance.remaining_at(at(LATE_UTC_DAY)),
        95,
        "a duplicate must not move the anchor and forgive the five sends"
    );
}

/// A reading stamped in the future cannot have been taken, so it describes nothing.
#[test]
fn a_future_reading_is_refused() {
    let allowance = account_allowance(&scratch("order-future"), 1000);
    let ticket = allowance
        .begin_observation_at(at(LATE_UTC_DAY))
        .expect("a ticket is taken before the read");
    let refused = allowance.reconcile_at(
        provider(0, 1000, 1000, LATE_UTC_DAY + 3_600),
        ticket,
        at(LATE_UTC_DAY),
    );

    assert!(
        matches!(
            refused,
            Err(ReserveError::ObservationRejected { reason })
                if reason.contains("future")
        ),
        "a future observation must be refused, got {refused:?}"
    );
    assert_eq!(allowance.remaining_at(at(LATE_UTC_DAY)), 1000);
}

/// Requests sent while the account read was outstanding are not in the provider's figure,
/// so they must still be subtracted. The ticket is taken before the read for exactly this
/// reason: anchoring at application time would forgive them.
#[test]
fn an_inflight_reservation_is_not_forgiven_by_the_application() {
    let allowance = account_allowance(&scratch("order-inflight"), 1000);
    // The collector starts its read here, before the five requests below are sent.
    let ticket = allowance
        .begin_observation_at(at(LATE_UTC_DAY))
        .expect("a ticket is taken before the read");
    for _ in 0..5 {
        allowance
            .try_reserve_at(at(LATE_UTC_DAY))
            .expect("the local budget has room");
    }

    allowance
        .reconcile_at(
            provider(10, 1000, 990, LATE_UTC_DAY),
            ticket,
            at(LATE_UTC_DAY),
        )
        .expect("the reading is newer than anything held");

    assert_eq!(
        allowance.remaining_at(at(LATE_UTC_DAY)),
        985,
        "the five sends made during the read are not in the provider's 990"
    );
}

/// A read that straddles the day boundary is refused rather than applied to either day: the
/// counter cannot tell which requests the provider counted in which window.
#[test]
fn a_read_that_straddles_the_day_boundary_is_refused() {
    let allowance = account_allowance(&scratch("order-rollover"), 10);
    // Started at 23:00Z, the day rolls over while the read is outstanding.
    let ticket = allowance
        .begin_observation_at(at(LATE_UTC_DAY))
        .expect("a ticket is taken before the read");
    allowance
        .try_reserve_at(at(EARLY_NEXT_UTC_DAY))
        .expect("the new day has room");

    let refused = allowance.reconcile_at(
        provider(1000, 1000, 0, LATE_UTC_DAY),
        ticket,
        at(EARLY_NEXT_UTC_DAY),
    );

    assert!(
        matches!(
            refused,
            Err(ReserveError::ObservationRejected { reason })
                if reason.contains("ended")
        ),
        "a reading from the previous day must not be applied after rollover, got {refused:?}"
    );
    assert_eq!(
        allowance.remaining_at(at(EARLY_NEXT_UTC_DAY)),
        9,
        "the new day's spend must survive"
    );
}

/// A known exhaustion keeps refusing after the reading ages out, so a collector that stops
/// answering does not turn a 429 into a retry storm. The retention is bounded and ends.
#[test]
fn a_known_exhaustion_still_refuses_after_the_reading_goes_stale() {
    let allowance = account_allowance(&scratch("cooldown-hold"), 1000)
        .with_provider_tolerance(Duration::from_secs(60));
    allowance
        .reconcile(provider(1000, 1000, 0, LATE_UTC_DAY))
        .expect("the reading is applied");

    // Older than the tolerance, so the reading is no longer fresh, but the exhaustion is
    // still within its bounded retention.
    let stale = LATE_UTC_DAY + 120;
    assert!(matches!(
        allowance.provider_state_at(at(stale)),
        ProviderState::Exhausted {
            limit: 1000,
            basis: ExhaustionBasis::Retained
        }
    ));
    assert!(
        matches!(
            allowance.try_reserve_at(at(stale)),
            Err(ReserveError::ProviderExhausted { limit: 1000 })
        ),
        "an account known to be empty must not be re-probed just because the reading aged out"
    );
}

/// A retained exhaustion refuses a send; it does not assert that the allowance is spent.
///
/// The two questions are answered by different readings of the same counter, and the
/// difference decides where the next request goes: `try_reserve` answers "may I send", while
/// `spent_today` answers "is there affirmative evidence this free domain is empty" and is
/// what authorises spending the local tier. A reading that has stopped being current
/// describes the past, so it may refuse the send and must not license the spend.
#[test]
fn a_retained_exhaustion_refuses_the_send_without_claiming_the_allowance_is_spent() {
    let allowance = account_allowance(&scratch("retained-not-spent"), 1000)
        .with_provider_tolerance(Duration::from_secs(60));
    allowance
        .reconcile(provider(1000, 1000, 0, LATE_UTC_DAY))
        .expect("the reading is applied");

    assert_eq!(
        allowance.spent_today_at(at(LATE_UTC_DAY)),
        Some(true),
        "a current reading with no headroom is affirmative evidence"
    );

    let stale = LATE_UTC_DAY + 120;
    assert!(
        matches!(
            allowance.try_reserve_at(at(stale)),
            Err(ReserveError::ProviderExhausted { limit: 1000 })
        ),
        "the send must still be refused: dispatch behaviour is not what this changes"
    );
    assert_eq!(
        allowance.spent_today_at(at(stale)),
        None,
        "a retained refusal describes the past, so it cannot be quoted as spent today"
    );

    let recovered = LATE_UTC_DAY + 3600;
    assert_eq!(
        allowance.spent_today_at(at(recovered)),
        Some(false),
        "past the retention the budget governs again, and it has room"
    );
}

/// Retention never masks a local budget that is genuinely spent: that reading is current and
/// belongs to this deployment, so it stays affirmative while the retention is alive.
#[test]
fn a_spent_local_budget_stays_affirmative_under_a_retention() {
    let allowance = account_allowance(&scratch("retained-local-spent"), 1)
        .with_provider_tolerance(Duration::from_secs(60));
    allowance
        .try_reserve_at(at(LATE_UTC_DAY))
        .expect("the day's only request");
    allowance
        .reconcile(provider(1000, 1000, 0, LATE_UTC_DAY))
        .expect("the reading is applied and starts a retention");

    let stale = LATE_UTC_DAY + 120;
    assert!(matches!(
        allowance.provider_state_at(at(stale)),
        ProviderState::Exhausted {
            basis: ExhaustionBasis::Retained,
            ..
        }
    ));
    assert_eq!(
        allowance.spent_today_at(at(stale)),
        Some(true),
        "this deployment's own counter is current, and it says the day is spent"
    );
}

/// The retention ends, so a recovered account serves again without requiring a new reading
/// to arrive first. Bounded in both directions: no storm, no permanent wall.
#[test]
fn the_exhaustion_retention_ends_and_recovery_admits() {
    let allowance = account_allowance(&scratch("cooldown-expiry"), 10);
    allowance
        .reconcile(provider(1000, 1000, 0, LATE_UTC_DAY))
        .expect("the reading is applied");

    let after_retention = LATE_UTC_DAY + 3600;
    assert_eq!(
        allowance.provider_state_at(at(after_retention)),
        ProviderState::Unknown,
        "past the bounded retention the provider has told us nothing current"
    );
    allowance
        .try_reserve_at(at(after_retention))
        .expect("a recovered account must serve on the local budget");
}

/// A fresh reading that reports headroom clears the retention immediately.
#[test]
fn a_recovered_reading_clears_the_retention() {
    let allowance = account_allowance(&scratch("cooldown-clear"), 1000)
        .with_provider_tolerance(Duration::from_secs(600));
    allowance
        .reconcile(provider(1000, 1000, 0, LATE_UTC_DAY))
        .expect("the exhausted reading is applied");

    allowance
        .reconcile(provider(0, 1000, 1000, LATE_UTC_DAY + 60))
        .expect("the recovered reading is applied");

    assert!(matches!(
        allowance.provider_state_at(at(LATE_UTC_DAY + 60)),
        ProviderState::Available { .. }
    ));
    allowance
        .try_reserve_at(at(LATE_UTC_DAY + 60))
        .expect("a recovered account admits again");
}

/// The retention does not survive the provider day it belongs to: yesterday's exhaustion
/// says nothing about today.
#[test]
fn the_exhaustion_retention_does_not_survive_the_day_boundary() {
    let allowance = account_allowance(&scratch("cooldown-day"), 10)
        .with_provider_tolerance(Duration::from_secs(2 * 86_400));
    allowance
        .reconcile(provider(1000, 1000, 0, LATE_UTC_DAY))
        .expect("the reading is applied");

    assert_eq!(
        allowance.provider_state_at(at(EARLY_NEXT_UTC_DAY)),
        ProviderState::Unknown
    );
    allowance
        .try_reserve_at(at(EARLY_NEXT_UTC_DAY))
        .expect("yesterday's exhaustion must not refuse today's request");
}

/// Production defaults: freshness and retention are both 600 s, so a retention counted from
/// apply time expires at the same instant the reading does and protects nothing against the
/// failure it exists for -- one collection round that never arrives. The retention is
/// measured from the end of freshness, so it still refuses for a full window after the
/// reading stops being current.
#[test]
fn the_retention_outlives_the_default_freshness_window() {
    let allowance = account_allowance(&scratch("retention-default"), 1000);
    assert_eq!(
        allowance.provider_tolerance(),
        Duration::from_secs(600),
        "this test is about the deployed defaults, so it must not set a tolerance"
    );
    allowance
        .reconcile(provider(1000, 1000, 0, LATE_UTC_DAY))
        .expect("the reading is applied");

    // Past the 600 s freshness window, inside the 600 s retention that follows it.
    let after_freshness = LATE_UTC_DAY + 700;
    assert!(
        matches!(
            allowance.provider_state_at(at(after_freshness)),
            ProviderState::Exhausted {
                basis: ExhaustionBasis::Retained,
                ..
            }
        ),
        "one missed round must not turn an empty account into admitted traffic"
    );
    assert!(matches!(
        allowance.try_reserve_at(at(after_freshness)),
        Err(ReserveError::ProviderExhausted { .. })
    ));

    // And the hold ends: a recovered account serves without waiting for a new reading.
    let after_hold = LATE_UTC_DAY + 1_300;
    assert_eq!(
        allowance.provider_state_at(at(after_hold)),
        ProviderState::Unknown
    );
    allowance
        .try_reserve_at(at(after_hold))
        .expect("the retention is bounded in both directions");
}

/// The pool is usually emptied by this deployment's own sends, not by a provider that
/// volunteers a zero. The reservation that spends the last unit arms the same retention, so
/// the next request after the reading ages out is refused instead of admitted.
#[test]
fn the_retention_is_armed_by_the_reservation_that_empties_the_pool() {
    let allowance = account_allowance(&scratch("retention-effective"), 1000);
    allowance
        .reconcile(provider(997, 1000, 3, LATE_UTC_DAY))
        .expect("the reading is applied");
    for _ in 0..3 {
        allowance
            .try_reserve_at(at(LATE_UTC_DAY))
            .expect("within the reported headroom");
    }
    assert!(matches!(
        allowance.provider_state_at(at(LATE_UTC_DAY)),
        ProviderState::Exhausted {
            basis: ExhaustionBasis::Confirmed,
            ..
        }
    ));

    let after_freshness = LATE_UTC_DAY + 700;
    assert!(
        matches!(
            allowance.try_reserve_at(at(after_freshness)),
            Err(ReserveError::ProviderExhausted { .. })
        ),
        "the last unit was spent locally; admitting the next request invents a request the \
         account does not have"
    );
}

/// The retention never outlives the provider's stated reset: a window that has ended cannot
/// refuse the next one, even if the bounded hold still has time on it.
#[test]
fn the_retention_never_outlives_the_provider_stated_reset() {
    let allowance = account_allowance(&scratch("retention-reset"), 1000);
    allowance
        .reconcile(ProviderReading {
            used: 1000,
            limit: 1000,
            remaining: 0,
            observed_at: at(LATE_UTC_DAY),
            reset_at: Some(at(LATE_UTC_DAY + 700)),
        })
        .expect("the reading is applied");

    assert!(matches!(
        allowance.provider_state_at(at(LATE_UTC_DAY + 650)),
        ProviderState::Exhausted { .. }
    ));
    assert_eq!(
        allowance.provider_state_at(at(LATE_UTC_DAY + 750)),
        ProviderState::Unknown,
        "past the provider's own reset the statement is over, whatever the bounded hold says"
    );
}

/// The reported state is the headroom this counter may use, not the provider's raw figure.
/// Preselection reads the state, so answering with the raw remainder would advertise
/// requests that have already been sent since the reading.
#[test]
fn the_reported_state_is_effective_headroom_not_the_raw_reading() {
    let allowance = account_allowance(&scratch("state-effective"), 1000);
    allowance
        .reconcile(provider(990, 1000, 10, LATE_UTC_DAY))
        .expect("the reading is applied");
    for _ in 0..4 {
        allowance
            .try_reserve_at(at(LATE_UTC_DAY))
            .expect("within the reported headroom");
    }

    assert_eq!(
        allowance.provider_state_at(at(LATE_UTC_DAY)),
        ProviderState::Available { remaining: 6 },
        "four sends since the reading leave six of the reported ten"
    );
    assert_eq!(
        allowance.provider_reading_at(at(LATE_UTC_DAY)),
        Some(provider(990, 1000, 10, LATE_UTC_DAY)),
        "the raw observation stays available for telemetry"
    );
}
