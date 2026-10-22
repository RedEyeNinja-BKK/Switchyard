// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tests for the provider account readers and the collection that drives them.
//!
//! The account surface is Switchyard's own: it reads provider state with the
//! credential Switchyard already holds. These tests pin the outcomes that matter to
//! an operator — a provider that answers, a provider that rejects the credential, a
//! provider that answers with something unusable, and a provider that cannot be
//! reached — because a collection that reports a failure as a success would show a
//! healthy balance that does not exist.

use std::time::Duration;

use serde_json::json;
use switchyard_runner::{ClientFacts, DeploymentFacts};
use switchyard_server::economics::account_apis::{
    CodexAllowanceAccount, DeepSeekAccount, OpenRouterAccount,
};
use switchyard_server::economics::collector::AccountCollector;
use switchyard_server::economics::providers::{
    AccountAmount, AccountApi, AccountObservation, AccountScope, FetchOutcome, ReportingPeriod,
};
use switchyard_server::economics::runtime::build_account_apis;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// Builds a deployment whose clients are the ones named, each holding the
/// environment variable named.
fn facts(clients: &[(&str, &str)]) -> DeploymentFacts {
    let mut deployment = DeploymentFacts::default();
    for (name, key_env) in clients {
        deployment.clients.insert(
            (*name).to_string(),
            ClientFacts {
                name: (*name).to_string(),
                base_url: "https://example.invalid/v1".to_string(),
                api_key_env: Some((*key_env).to_string()),
                billing: None,
            },
        );
    }
    deployment
}

/// The environment a test declares: every named variable holds its own name.
fn env_holding(names: &[&str]) -> impl Fn(&str) -> Option<String> {
    move |name: &str| {
        names
            .iter()
            .find(|candidate| **candidate == name)
            .map(|value| value.to_string())
    }
}

/// Two clients that name the same provider account produce one reader, not two.
///
/// The deployed configuration names an `openai` and an `openai_nt` client that share
/// one credential and one base URL and differ only in the reasoning dialect they
/// speak. They describe one account. Two readers would read that account twice per
/// interval and report two account reads for one account — a doubled count of
/// something that happened once.
#[test]
fn two_clients_for_one_provider_produce_one_reader() {
    let deployment = facts(&[
        ("openai", "SY_OPENAI_OAUTH_TOKEN"),
        ("openai_nt", "SY_OPENAI_OAUTH_TOKEN"),
    ]);
    let apis = build_account_apis(&deployment, &env_holding(&["SY_OPENAI_OAUTH_TOKEN"]));
    assert_eq!(
        apis.len(),
        1,
        "one account must produce one reader, got {}",
        apis.len()
    );
    assert_eq!(apis[0].provider(), "openai");
}

/// Clients that name different providers each produce their own reader.
#[test]
fn clients_for_different_providers_each_produce_a_reader() {
    let deployment = facts(&[
        ("openrouter", "SWITCHYARD_OPENROUTER_API_KEY"),
        ("deepseek", "SWITCHYARD_DEEPSEEK_API_KEY"),
    ]);
    let apis = build_account_apis(
        &deployment,
        &env_holding(&[
            "SWITCHYARD_OPENROUTER_API_KEY",
            "SWITCHYARD_DEEPSEEK_API_KEY",
        ]),
    );
    assert_eq!(apis.len(), 2, "two providers must produce two readers");
}

/// A client whose credential the environment does not hold produces no reader.
///
/// Switchyard cannot read an account it holds no credential for, so it publishes
/// nothing for that provider rather than publishing a zero.
#[test]
fn a_client_without_a_credential_produces_no_reader() {
    let deployment = facts(&[("openrouter", "SWITCHYARD_OPENROUTER_API_KEY")]);
    let apis = build_account_apis(&deployment, &env_holding(&[]));
    assert!(
        apis.is_empty(),
        "a missing credential must produce no reader"
    );
}

/// A client whose provider exposes no account surface produces no reader.
#[test]
fn a_provider_without_an_account_surface_produces_no_reader() {
    let deployment = facts(&[("thaillm", "SWITCHYARD_THAILLM_API_KEY")]);
    let apis = build_account_apis(&deployment, &env_holding(&["SWITCHYARD_THAILLM_API_KEY"]));
    assert!(
        apis.is_empty(),
        "a provider with no account API must produce no reader, not a zero balance"
    );
}

/// A provider that answers with a valid account state.
#[tokio::test]
async fn the_openrouter_reader_reports_the_credit_and_the_usage() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": {"usage": 12.5, "usage_monthly": 40.0}
        })))
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/credits"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": {"total_credits": 125.0, "total_usage": 124.232404098}
        })))
        .mount(&server)
        .await;

    let reader = OpenRouterAccount::new("test-key").with_base_url(server.uri());
    let outcome = reader.fetch().await;
    let FetchOutcome::Ok(observation) = outcome else {
        panic!("expected a successful reading, got {outcome:?}");
    };
    assert_eq!(observation.credits.len(), 1);
    let credit = &observation.credits[0];
    assert_eq!(credit.currency, "USD");
    assert_eq!(credit.scope, AccountScope::Account);
    assert_eq!(credit.period, ReportingPeriod::Total);
    assert_eq!(credit.value, 125.0);

    assert_eq!(observation.usage.len(), 3);
    let account_usage = observation
        .usage
        .iter()
        .find(|amount| amount.scope == AccountScope::Account)
        .expect("an account-wide usage figure");
    assert_eq!(account_usage.value, 124.232404098);
    assert_eq!(account_usage.period, ReportingPeriod::Total);

    let key_total = observation
        .usage
        .iter()
        .find(|amount| amount.scope == AccountScope::Key && amount.period == ReportingPeriod::Total)
        .expect("a key usage figure for the whole life of the key");
    assert_eq!(key_total.value, 12.5);

    let key_monthly = observation
        .usage
        .iter()
        .find(|amount| {
            amount.scope == AccountScope::Key && amount.period == ReportingPeriod::Monthly
        })
        .expect("a key usage figure for the current month");
    assert_eq!(key_monthly.value, 40.0);
}

/// The currency, the accounting scope and the reporting period are separate facts.
///
/// The key endpoint and the credits endpoint both report usage in US dollars, but one
/// covers the key and the other the whole account, so they are different amounts.
/// A single label that carried the currency together with the scope or the period
/// would make the two indistinguishable without parsing the label value, and a
/// reader could sum series that were never meant to be summed.
#[tokio::test]
async fn the_currency_the_scope_and_the_period_are_separate_labels() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": {
                "usage": 1.0,
                "usage_daily": 2.0,
                "usage_weekly": 3.0,
                "usage_monthly": 4.0
            }
        })))
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/credits"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": {"total_credits": 10.0, "total_usage": 6.0}
        })))
        .mount(&server)
        .await;

    let reader = OpenRouterAccount::new("test-key").with_base_url(server.uri());
    let outcome = reader.fetch().await;
    let FetchOutcome::Ok(observation) = outcome else {
        panic!("expected a successful reading, got {outcome:?}");
    };

    // Every figure carries the same currency, and differs in exactly one of scope
    // and period.
    assert!(
        observation
            .usage
            .iter()
            .all(|amount| amount.currency == "USD")
    );

    let key_figures: Vec<(&str, f64)> = observation
        .usage
        .iter()
        .filter(|amount| amount.scope == AccountScope::Key)
        .map(|amount| (amount.period.as_str(), amount.value))
        .collect();
    assert_eq!(
        key_figures,
        vec![
            ("total", 1.0),
            ("daily", 2.0),
            ("weekly", 3.0),
            ("monthly", 4.0),
        ],
        "each reporting window is its own figure"
    );

    let account_figures: Vec<(&str, f64)> = observation
        .usage
        .iter()
        .filter(|amount| amount.scope == AccountScope::Account)
        .map(|amount| (amount.period.as_str(), amount.value))
        .collect();
    assert_eq!(
        account_figures,
        vec![("total", 6.0)],
        "the account-wide usage is a different amount from the key's usage"
    );
}

/// Total credits, consumed credits and remaining balance are three different facts.
///
/// The provider reports the credit it sold and the usage drawn against it. What is
/// left is the difference. Publishing all three under their own labels means an
/// operator reads the remaining balance directly instead of subtracting one series
/// from another, and a reader cannot mistake the purchased credit for the money that
/// is left.
#[tokio::test]
async fn the_credit_the_consumed_credit_and_the_remaining_balance_are_separate() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": {"usage": 1.0}
        })))
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/credits"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": {"total_credits": 125.0, "total_usage": 124.232622134}
        })))
        .mount(&server)
        .await;

    let reader = OpenRouterAccount::new("test-key").with_base_url(server.uri());
    let FetchOutcome::Ok(observation) = reader.fetch().await else {
        panic!("expected a successful reading");
    };

    assert_eq!(observation.credits.len(), 1, "one purchased credit figure");
    assert_eq!(observation.credits[0].value, 125.0);

    let consumed = observation
        .usage
        .iter()
        .find(|amount| amount.scope == AccountScope::Account)
        .expect("the consumed credit");
    assert_eq!(consumed.value, 124.232622134);

    assert_eq!(
        observation.balances.len(),
        1,
        "one remaining balance figure"
    );
    let remaining = &observation.balances[0];
    assert_eq!(remaining.scope, AccountScope::Account);
    assert_eq!(remaining.period, ReportingPeriod::Total);
    assert!(
        (remaining.value - (125.0 - 124.232622134)).abs() < 1e-9,
        "the remaining balance is the purchased credit less the consumed credit, got {}",
        remaining.value
    );
}

/// A provider that rejects the credential is reported as unauthorized, not as a
/// provider that reported no account state. The two are different facts and only
/// one of them is actionable.
#[tokio::test]
async fn the_openrouter_reader_reports_a_rejected_credential() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/key"))
        .respond_with(ResponseTemplate::new(401).set_body_json(json!({"error": "no"})))
        .mount(&server)
        .await;

    let reader = OpenRouterAccount::new("test-key").with_base_url(server.uri());
    assert_eq!(reader.fetch().await, FetchOutcome::Unauthorized);
}

/// A provider that answers with something unusable is a failure, not an empty
/// reading. An empty reading would be published as a balance of nothing.
#[tokio::test]
async fn the_openrouter_reader_reports_an_unusable_answer_as_a_failure() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"data": {}})))
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/credits"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"data": {}})))
        .mount(&server)
        .await;

    let reader = OpenRouterAccount::new("test-key").with_base_url(server.uri());
    let outcome = reader.fetch().await;
    assert!(
        matches!(outcome, FetchOutcome::Failed(_)),
        "expected a failure, got {outcome:?}"
    );
}

/// A provider that cannot be reached is a failure. The reader must not block
/// forever: it carries its own deadline.
#[tokio::test]
async fn the_deepseek_reader_reports_an_unreachable_provider_as_a_failure() {
    // A port nothing listens on, so the connection is refused rather than hanging.
    let reader = DeepSeekAccount::new("test-key").with_base_url("http://127.0.0.1:1");
    let outcome = reader.fetch().await;
    assert!(
        matches!(outcome, FetchOutcome::Failed(_)),
        "expected a failure, got {outcome:?}"
    );
}

/// The DeepSeek reader reports the balance the provider states.
#[tokio::test]
async fn the_deepseek_reader_reports_the_balance() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/user/balance"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "balance_infos": [{"currency": "CNY", "total_balance": "12.34"}]
        })))
        .mount(&server)
        .await;

    let reader = DeepSeekAccount::new("test-key").with_base_url(server.uri());
    let outcome = reader.fetch().await;
    let FetchOutcome::Ok(observation) = outcome else {
        panic!("expected a successful reading, got {outcome:?}");
    };
    assert_eq!(observation.balances.len(), 1);
    let balance = &observation.balances[0];
    assert_eq!(balance.currency, "CNY");
    assert_eq!(balance.scope, AccountScope::Account);
    assert_eq!(balance.period, ReportingPeriod::Total);
    assert_eq!(balance.value, 12.34);
}

/// A provider may report an amount as a decimal string rather than as a JSON
/// number. The live DeepSeek balance endpoint does exactly that, so a reader that
/// accepted only numbers would find no balance on a funded account and report the
/// lane as unmeasured.
#[tokio::test]
async fn a_balance_reported_as_a_string_is_read_as_an_amount() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/user/balance"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "is_available": true,
            "balance_infos": [
                {"currency": "USD", "total_balance": "7.50", "granted_balance": "7.50"},
                {"currency": "CNY", "total_balance": "-0.00"}
            ]
        })))
        .mount(&server)
        .await;

    let reader = DeepSeekAccount::new("test-key").with_base_url(server.uri());
    let outcome = reader.fetch().await;
    let FetchOutcome::Ok(observation) = outcome else {
        panic!("expected a successful reading, got {outcome:?}");
    };
    let usd = observation
        .balances
        .iter()
        .find(|amount| amount.currency == "USD")
        .expect("a US dollar balance");
    assert_eq!(usd.value, 7.5);
    let cny = observation
        .balances
        .iter()
        .find(|amount| amount.currency == "CNY")
        .expect("a renminbi balance");
    assert_eq!(
        cny.value, -0.0,
        "a negative zero balance is still a balance the provider reported"
    );
}

/// A value that is neither a number nor a decimal string is not an amount, and is
/// left absent rather than read as zero.
#[tokio::test]
async fn a_balance_that_is_not_an_amount_is_left_absent() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/user/balance"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "balance_infos": [{"currency": "USD", "total_balance": "not-a-number"}]
        })))
        .mount(&server)
        .await;

    let reader = DeepSeekAccount::new("test-key").with_base_url(server.uri());
    let outcome = reader.fetch().await;
    assert!(
        matches!(outcome, FetchOutcome::Failed(_)),
        "an unreadable balance must not be reported as a zero balance, got {outcome:?}"
    );
}

/// The allowance reader reports the remaining percentage of each window, so the
/// dashboard can show the five-hour window and the weekly window separately.
#[tokio::test]
async fn the_allowance_reader_reports_each_window() {
    let server = MockServer::start().await;
    let reset_at = 1_794_556_200_f64;
    Mock::given(method("GET"))
        .and(path("/wham/usage"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "rate_limit": {
                "primary_window": {"used_percent": 25.0, "reset_at": reset_at},
                "secondary_window": {"used_percent": 80.0, "reset_at": reset_at}
            }
        })))
        .mount(&server)
        .await;

    let reader = CodexAllowanceAccount::new("token", Some("account-1".to_string()))
        .with_base_url(server.uri());
    let outcome = reader.fetch().await;
    let FetchOutcome::Ok(observation) = outcome else {
        panic!("expected a successful reading, got {outcome:?}");
    };
    assert_eq!(
        observation.allowance_remaining_percent.get("primary"),
        Some(&75.0)
    );
    assert_eq!(
        observation.allowance_remaining_percent.get("secondary"),
        Some(&20.0)
    );
    assert!(
        observation.allowance_reset_seconds.contains_key("primary"),
        "the reset time must be reported alongside the remaining allowance"
    );
}

/// A provider that exposes no allowance reports a failure rather than a zero
/// allowance, which would read as a provider with nothing left.
#[tokio::test]
async fn the_allowance_reader_reports_a_missing_allowance_as_a_failure() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/wham/usage"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"other": 1})))
        .mount(&server)
        .await;

    let reader = CodexAllowanceAccount::new("token", None).with_base_url(server.uri());
    let outcome = reader.fetch().await;
    assert!(
        matches!(outcome, FetchOutcome::Failed(_)),
        "expected a failure, got {outcome:?}"
    );
}

/// The collection counts the providers that answered, and a failing provider does
/// not stop the others from being read.
#[tokio::test]
async fn the_collection_counts_only_the_providers_that_answered() {
    struct Stub {
        provider: &'static str,
        outcome: FetchOutcome,
    }

    #[async_trait::async_trait]
    impl AccountApi for Stub {
        fn provider(&self) -> &str {
            self.provider
        }

        async fn fetch(&self) -> FetchOutcome {
            self.outcome.clone()
        }
    }

    let mut collector = AccountCollector::new(vec![
        std::sync::Arc::new(Stub {
            provider: "ok",
            outcome: FetchOutcome::Ok(AccountObservation {
                balances: vec![AccountAmount::new(
                    "USD",
                    AccountScope::Account,
                    ReportingPeriod::Total,
                    1.0,
                )],
                ..AccountObservation::default()
            }),
        }),
        std::sync::Arc::new(Stub {
            provider: "denied",
            outcome: FetchOutcome::Unauthorized,
        }),
        std::sync::Arc::new(Stub {
            provider: "broken",
            outcome: FetchOutcome::Failed("unreachable".to_string()),
        }),
    ])
    .with_interval(Duration::from_secs(1));

    assert_eq!(collector.collect_once().await, 1);
}
