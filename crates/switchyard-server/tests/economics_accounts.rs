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
use switchyard_server::economics::providers::{AccountApi, AccountObservation, FetchOutcome};
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
    assert_eq!(observation.credits.get("USD"), Some(&125.0));
    assert_eq!(observation.usage.get("USD_account"), Some(&124.232404098));
    assert_eq!(observation.usage.get("USD"), Some(&12.5));
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
    assert_eq!(observation.balances.get("CNY"), Some(&12.34));
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
    assert_eq!(observation.balances.get("USD"), Some(&7.5));
    assert_eq!(
        observation.balances.get("CNY"),
        Some(&-0.0),
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
                balances: [("USD".to_string(), 1.0)].into_iter().collect(),
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
