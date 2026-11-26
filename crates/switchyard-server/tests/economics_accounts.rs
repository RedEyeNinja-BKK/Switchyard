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

use std::sync::Arc;
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
/// environment variable named and pointing at the base URL given.
///
/// The base URL decides which account surface a client reads. It is supplied by the
/// test rather than inferred from the client's name, because the name is only a
/// label: a client called anything at all still points at a provider.
fn facts(clients: &[(&str, &str, &str)]) -> DeploymentFacts {
    let mut deployment = DeploymentFacts::default();
    for (name, key_env, base_url) in clients {
        deployment.clients.insert(
            (*name).to_string(),
            ClientFacts {
                name: (*name).to_string(),
                base_url: (*base_url).to_string(),
                api_key_env: Some((*key_env).to_string()),
                billing: None,
                quota_group: Some((*name).to_string()),
                quota_provider: None,
            },
        );
    }
    deployment
}

const OPENROUTER_URL: &str = "https://openrouter.ai/api/v1";
const DEEPSEEK_URL: &str = "https://api.deepseek.com";
const CODEX_URL: &str = "https://chatgpt.com/backend-api/codex";

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
        ("openai", "SY_OPENAI_OAUTH_TOKEN", CODEX_URL),
        ("openai_nt", "SY_OPENAI_OAUTH_TOKEN", CODEX_URL),
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
        (
            "openrouter",
            "SWITCHYARD_OPENROUTER_API_KEY",
            OPENROUTER_URL,
        ),
        ("deepseek", "SWITCHYARD_DEEPSEEK_API_KEY", DEEPSEEK_URL),
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
    let deployment = facts(&[(
        "openrouter",
        "SWITCHYARD_OPENROUTER_API_KEY",
        OPENROUTER_URL,
    )]);
    let apis = build_account_apis(&deployment, &env_holding(&[]));
    assert!(
        apis.is_empty(),
        "a missing credential must produce no reader"
    );
}

/// A client whose provider exposes no account surface produces no reader.
#[test]
fn a_provider_without_an_account_surface_produces_no_reader() {
    let deployment = facts(&[(
        "thaillm",
        "SWITCHYARD_THAILLM_API_KEY",
        "https://thaillm.or.th/api/v1",
    )]);
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
            outcome: FetchOutcome::Ok(Box::new(AccountObservation {
                balances: vec![AccountAmount::new(
                    "USD",
                    AccountScope::Account,
                    ReportingPeriod::Total,
                    1.0,
                )],
                ..AccountObservation::default()
            })),
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

/// Two OpenRouter clients holding different credentials are two accounts, and each
/// gets its own label.
///
/// The deployed configuration names `openrouter` and `openrouter_stephen`. They are
/// separate accounts with separate credit, so a single shared label would make one
/// account's balance silently overwrite the other's.
#[test]
fn two_openrouter_credentials_produce_two_labelled_readers() {
    let deployment = facts(&[
        (
            "openrouter",
            "SWITCHYARD_OPENROUTER_API_KEY",
            OPENROUTER_URL,
        ),
        (
            "openrouter_stephen",
            "SWITCHYARD_STEPHEN_OPENROUTER_API_KEY",
            OPENROUTER_URL,
        ),
    ]);
    let apis = build_account_apis(
        &deployment,
        &env_holding(&[
            "SWITCHYARD_OPENROUTER_API_KEY",
            "SWITCHYARD_STEPHEN_OPENROUTER_API_KEY",
        ]),
    );
    let mut labels: Vec<&str> = apis.iter().map(|api| api.provider()).collect();
    labels.sort();
    assert_eq!(labels, vec!["openrouter", "openrouter_stephen"]);
}

/// The account surface is decided by the base URL, so a client named something new
/// still gets a reader. A deployment that adds a second OpenRouter client called
/// anything at all must not silently get no account telemetry.
#[test]
fn a_newly_named_openrouter_client_still_gets_a_reader() {
    let deployment = facts(&[(
        "openrouter_third",
        "SWITCHYARD_THIRD_OPENROUTER_API_KEY",
        OPENROUTER_URL,
    )]);
    let apis = build_account_apis(
        &deployment,
        &env_holding(&["SWITCHYARD_THIRD_OPENROUTER_API_KEY"]),
    );
    assert_eq!(apis.len(), 1);
    assert_eq!(apis[0].provider(), "openrouter_third");
}

/// A reader that reports an account already covered by another label is not
/// published, because its figures are the same money.
///
/// Two keys on one OpenRouter account both answer `/key`, both report the same
/// `creator_user_id`, and both report the same account-wide credit. Publishing both
/// would double the account's reported balance.
#[tokio::test]
async fn two_keys_for_one_account_are_published_once() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": {"usage": 1.0, "creator_user_id": "user_shared"}
        })))
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/credits"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": {"total_credits": 25.0, "total_usage": 1.0}
        })))
        .mount(&server)
        .await;

    let first = std::sync::Arc::new(OpenRouterAccount::new("key-a").with_base_url(server.uri()));
    let second = std::sync::Arc::new(OpenRouterAccount::new("key-b").with_base_url(server.uri()));
    let mut collector = AccountCollector::new(vec![
        std::sync::Arc::new(
            switchyard_server::economics::providers::LabeledAccount::new("openrouter", first),
        ),
        std::sync::Arc::new(
            switchyard_server::economics::providers::LabeledAccount::new(
                "openrouter_stephen",
                second,
            ),
        ),
    ]);

    collector.collect_once().await;
    // Both answered, but the second is recognised as the same account and is
    // dropped from the poll set rather than counted twice every interval.
    assert_eq!(
        collector.polled_labels(),
        vec!["openrouter".to_string()],
        "one account must be polled under one label"
    );
}

/// Two keys on two different accounts are both published.
#[tokio::test]
async fn keys_for_different_accounts_are_both_published() {
    let first = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": {"usage": 1.0, "creator_user_id": "user_one"}
        })))
        .mount(&first)
        .await;
    Mock::given(method("GET"))
        .and(path("/credits"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": {"total_credits": 125.0, "total_usage": 1.0}
        })))
        .mount(&first)
        .await;

    let second = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": {"usage": 0.5, "creator_user_id": "user_two"}
        })))
        .mount(&second)
        .await;
    Mock::given(method("GET"))
        .and(path("/credits"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": {"total_credits": 25.0, "total_usage": 0.5}
        })))
        .mount(&second)
        .await;

    let mut collector = AccountCollector::new(vec![
        std::sync::Arc::new(
            switchyard_server::economics::providers::LabeledAccount::new(
                "openrouter",
                std::sync::Arc::new(OpenRouterAccount::new("a").with_base_url(first.uri())),
            ),
        ),
        std::sync::Arc::new(
            switchyard_server::economics::providers::LabeledAccount::new(
                "openrouter_stephen",
                std::sync::Arc::new(OpenRouterAccount::new("b").with_base_url(second.uri())),
            ),
        ),
    ]);

    collector.collect_once().await;
    let mut labels = collector.polled_labels();
    labels.sort();
    assert_eq!(
        labels,
        vec!["openrouter".to_string(), "openrouter_stephen".to_string()],
        "two distinct accounts must both be polled"
    );
}

/// One account's rejected credential does not stop the other account being read.
#[tokio::test]
async fn a_rejected_credential_does_not_starve_the_other_account() {
    let rejected = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/key"))
        .respond_with(ResponseTemplate::new(401))
        .mount(&rejected)
        .await;

    let healthy = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": {"usage": 2.0, "creator_user_id": "user_healthy"}
        })))
        .mount(&healthy)
        .await;
    Mock::given(method("GET"))
        .and(path("/credits"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": {"total_credits": 10.0, "total_usage": 2.0}
        })))
        .mount(&healthy)
        .await;

    let mut collector = AccountCollector::new(vec![
        std::sync::Arc::new(
            switchyard_server::economics::providers::LabeledAccount::new(
                "openrouter",
                std::sync::Arc::new(OpenRouterAccount::new("bad").with_base_url(rejected.uri())),
            ),
        ),
        std::sync::Arc::new(
            switchyard_server::economics::providers::LabeledAccount::new(
                "openrouter_stephen",
                std::sync::Arc::new(OpenRouterAccount::new("good").with_base_url(healthy.uri())),
            ),
        ),
    ]);

    let successful = collector.collect_once().await;
    assert_eq!(successful, 1, "the healthy account must still be read");
}

/// A DeepSeek balance that reports its parts keeps them separate from the total.
#[tokio::test]
async fn the_deepseek_reader_reports_the_balance_and_its_parts() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/user/balance"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "is_available": true,
            "balance_infos": [{
                "currency": "CNY",
                "total_balance": "81.76",
                "granted_balance": "0.00",
                "topped_up_balance": "81.76"
            }]
        })))
        .mount(&server)
        .await;

    let reader = DeepSeekAccount::new("key").with_base_url(server.uri());
    let FetchOutcome::Ok(observation) = reader.fetch().await else {
        panic!("expected a successful reading");
    };
    assert_eq!(observation.balances.len(), 1);
    assert_eq!(observation.balances[0].value, 81.76);

    let parts: Vec<(&str, f64)> = observation
        .balance_components
        .iter()
        .map(|part| (part.component.as_str(), part.amount.value))
        .collect();
    assert_eq!(parts, vec![("granted", 0.0), ("topped_up", 81.76)]);
    for part in &observation.balance_components {
        assert_eq!(
            part.amount.period,
            ReportingPeriod::Total,
            "a remaining balance is not a purchase total and carries no window"
        );
    }
}

// ---------------------------------------------------------------------------
// Provider allowance reconciliation, end to end.
//
// These cases drive the whole path a real deployment uses: a configuration that
// declares a provider pool, the readers built from its credentials, the counters bound
// to those readers, and finally admission. A unit test on any one link would not catch
// the failure that actually matters -- a deployment that reads the provider correctly
// and never consults it.
// ---------------------------------------------------------------------------

use switchyard_llm_client::{DayBoundary, ReserveError};
use switchyard_server::economics::reconcile::Reconcilers;
use switchyard_server::economics::runtime::{build_account_readers, build_reconcilers};

/// A deployment with one OpenRouter client declaring a provider pool and an account counter.
fn pool_deployment(
    client: &str,
    key_env: &str,
    group: &str,
    pool: Option<switchyard_runner::QuotaProvider>,
) -> (DeploymentFacts, std::path::PathBuf) {
    let mut deployment = DeploymentFacts::default();
    deployment.clients.insert(
        client.to_string(),
        ClientFacts {
            name: client.to_string(),
            base_url: OPENROUTER_URL.to_string(),
            api_key_env: Some(key_env.to_string()),
            billing: None,
            quota_group: Some(group.to_string()),
            quota_provider: pool,
        },
    );
    let directory = std::env::temp_dir()
        .join("switchyard-pool-deployment")
        .join(format!("{client}-{}", std::process::id()));
    (deployment, directory)
}

/// Builds the account-scoped counter the loader builds, for the test's state directory.
fn account_counter(
    directory: &std::path::Path,
    group: &str,
    limit: u64,
) -> std::sync::Arc<switchyard_llm_client::DailyRequestAllowance> {
    std::sync::Arc::new(switchyard_llm_client::DailyRequestAllowance::new_on(
        group,
        switchyard_llm_client::ACCOUNT_SCOPE_MODEL.into(),
        limit,
        switchyard_llm_client::account_allowance_path(directory, group),
        DayBoundary::Utc,
    ))
}

fn at(seconds: u64) -> std::time::SystemTime {
    std::time::UNIX_EPOCH + Duration::from_secs(seconds)
}

/// 2026-01-15T23:00:00Z
const LATE_UTC_DAY: u64 = 1_768_518_000;

/// The configuration binds a counter to the account it was read from, and a reading then
/// refuses admission under that account.
#[test]
fn a_declared_pool_is_bound_and_refuses_admission() {
    let (deployment, directory) = pool_deployment(
        "openrouter-free",
        "SY_OPENROUTER_STEPHEN_KEY",
        "openrouter-free-stephen",
        Some(switchyard_runner::QuotaProvider::OpenRouterFreeModels),
    );
    let env = env_holding(&["SY_OPENROUTER_STEPHEN_KEY"]);
    let readers = build_account_readers(&deployment, &env);
    assert_eq!(readers.len(), 1, "one credential, one reader");

    let counter = account_counter(&directory, "openrouter-free-stephen", 1000);
    let mut allowances = std::collections::HashMap::new();
    allowances.insert(
        (
            "openrouter-free-stephen".to_string(),
            switchyard_llm_client::ACCOUNT_SCOPE_MODEL.into(),
        ),
        std::sync::Arc::clone(&counter),
    );

    let reconcilers = build_reconcilers(&readers, &deployment, &allowances);
    assert_eq!(
        reconcilers.bindings(),
        vec![(
            "openrouter-free".to_string(),
            "free_model_daily_requests".to_string()
        )],
        "the binding names the account label and the provider's own pool"
    );

    // The provider says the account has spent its free-model allowance.
    let mut observation = AccountObservation::default();
    observation.request_allowances.push(
        switchyard_server::economics::providers::RequestAllowanceReading::new(
            "free_model_daily_requests",
            1000,
            1000,
            0,
            at(LATE_UTC_DAY),
            None,
        ),
    );
    let applied = reconcilers.apply("openrouter-free", &observation, at(LATE_UTC_DAY));
    assert_eq!(applied.written, 1);

    assert!(matches!(
        counter.try_reserve_at(at(LATE_UTC_DAY)),
        Err(ReserveError::ProviderExhausted { limit: 1000 })
    ));
}

/// A client that declares no pool produces no binding, and its counter keeps admitting on
/// its own budget. Without this, every deployment would be reconciled against whatever
/// account happened to be configured.
#[test]
fn a_client_without_a_pool_binds_nothing() {
    let (deployment, directory) =
        pool_deployment("openrouter-free", "SY_OPENROUTER_STEPHEN_KEY", "g", None);
    let env = env_holding(&["SY_OPENROUTER_STEPHEN_KEY"]);
    let readers = build_account_readers(&deployment, &env);
    let counter = account_counter(&directory, "g", 3);
    let mut allowances = std::collections::HashMap::new();
    allowances.insert(
        (
            "g".to_string(),
            switchyard_llm_client::ACCOUNT_SCOPE_MODEL.into(),
        ),
        std::sync::Arc::clone(&counter),
    );

    let reconcilers = build_reconcilers(&readers, &deployment, &allowances);
    assert!(reconcilers.is_empty());
    counter
        .try_reserve_at(at(LATE_UTC_DAY))
        .expect("an unreconciled counter admits on its own budget");
}

/// Two accounts on one provider are two bindings with two labels. A reading that arrives
/// from one of them cannot touch the other's counter -- this is the isolation boundary
/// that stops one account's exhaustion from refusing another account's traffic.
#[test]
fn two_accounts_on_one_provider_are_bound_apart() {
    let (mut owner, owner_dir) = pool_deployment(
        "openrouter-free-owner",
        "SY_OPENROUTER_KEY",
        "openrouter-free-owner",
        Some(switchyard_runner::QuotaProvider::OpenRouterFreeModels),
    );
    let (stephen, _stephen_dir) = pool_deployment(
        "openrouter-free-stephen",
        "SY_OPENROUTER_STEPHEN_KEY",
        "openrouter-free-stephen",
        Some(switchyard_runner::QuotaProvider::OpenRouterFreeModels),
    );
    for (name, client) in stephen.clients {
        owner.clients.insert(name, client);
    }
    let env = env_holding(&["SY_OPENROUTER_KEY", "SY_OPENROUTER_STEPHEN_KEY"]);
    let readers = build_account_readers(&owner, &env);
    assert_eq!(readers.len(), 2, "two credentials, two accounts");

    let owner_counter = account_counter(&owner_dir, "openrouter-free-owner", 1000);
    let stephen_counter = account_counter(&owner_dir, "openrouter-free-stephen", 1000);
    let mut allowances = std::collections::HashMap::new();
    for (group, counter) in [
        ("openrouter-free-owner", &owner_counter),
        ("openrouter-free-stephen", &stephen_counter),
    ] {
        allowances.insert(
            (
                group.to_string(),
                switchyard_llm_client::ACCOUNT_SCOPE_MODEL.into(),
            ),
            std::sync::Arc::clone(counter),
        );
    }

    let reconcilers = build_reconcilers(&readers, &owner, &allowances);
    assert_eq!(reconcilers.len(), 2);

    // Exhaustion arrives under one account's label only.
    let mut observation = AccountObservation::default();
    observation.request_allowances.push(
        switchyard_server::economics::providers::RequestAllowanceReading::new(
            "free_model_daily_requests",
            1000,
            1000,
            0,
            at(LATE_UTC_DAY),
            None,
        ),
    );
    for (account, _) in reconcilers.bindings() {
        if account.contains("owner") {
            reconcilers.apply(&account, &observation, at(LATE_UTC_DAY));
        }
    }

    owner_counter
        .try_reserve_at(at(LATE_UTC_DAY))
        .expect_err("the account that reported exhaustion must refuse");
    stephen_counter
        .try_reserve_at(at(LATE_UTC_DAY))
        .expect("the other account must keep serving");
}

/// The collector applies a reading during its round, so no separate call is needed for a
/// running deployment. This is the link a unit test on the reconciler alone would miss.
#[test]
fn the_collector_applies_the_reading_during_its_round() {
    let directory = std::env::temp_dir()
        .join("switchyard-collector-reconcile")
        .join(format!("{}", std::process::id()));
    let counter = account_counter(&directory, "openrouter-free", 1000);
    let reconcilers = Reconcilers::new(vec![
        switchyard_server::economics::reconcile::AllowanceReconciler::new(
            "openrouter-free",
            "free_model_daily_requests",
            std::sync::Arc::clone(&counter),
        ),
    ]);

    let mut observation = AccountObservation::default();
    observation.request_allowances.push(
        switchyard_server::economics::providers::RequestAllowanceReading::new(
            "free_model_daily_requests",
            1000,
            1000,
            0,
            std::time::SystemTime::now(),
            None,
        ),
    );
    let api = Arc::new(StaticAccount::new("openrouter-free", observation));
    let mut collector = AccountCollector::new(vec![api]).with_reconcilers(reconcilers);
    assert_eq!(collector.reconciled_counters(), 1);

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime");
    let successful = runtime.block_on(collector.collect_once());
    assert_eq!(successful, 1);

    assert!(matches!(
        counter.try_reserve_at(std::time::SystemTime::now()),
        Err(ReserveError::ProviderExhausted { limit: 1000 })
    ));
}

/// An account reader that answers with a fixed observation, so a collection round can be
/// driven without a provider.
struct StaticAccount {
    label: String,
    observation: AccountObservation,
}

impl StaticAccount {
    fn new(label: &str, observation: AccountObservation) -> Self {
        Self {
            label: label.to_string(),
            observation,
        }
    }
}

#[async_trait::async_trait]
impl AccountApi for StaticAccount {
    fn provider(&self) -> &str {
        &self.label
    }

    async fn fetch(&self) -> FetchOutcome {
        FetchOutcome::Ok(Box::new(self.observation.clone()))
    }
}
