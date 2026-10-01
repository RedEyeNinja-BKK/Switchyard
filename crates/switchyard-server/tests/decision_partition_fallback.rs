// SPDX-FileCopyrightText: Copyright (c) 2026 LocalClaw
// SPDX-License-Identifier: Apache-2.0

//! The per-question PARTITION of the one-hop decision fallback.
//!
//! The claim under test is narrow and falsifiable: when a capability declares a
//! `question_partition`, an ELIGIBLE primary failure fans the fallback out across
//! the declared executors, each receiving EXACTLY its assigned questions, and the
//! answers are recombined into one contract-shaped response in the caller's order.
//!
//! Three independent loopback executors on three ports count their own
//! dispatches AND record the question keys they were actually asked, so the
//! partition is observed rather than inferred. Each executor answers with a
//! DIFFERENT `type`, so the recombined body itself names which executor produced
//! each answer: a silent single-leg fallback could not produce that body.
//!
//! Every test builds the REAL router through the REAL loader from a REAL config.
//! Nothing here touches production, and nothing here observes a real provider.

use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use axum::Router;
use axum::body::Body;
use axum::http::Request;
use http_body_util::BodyExt;
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tower::ServiceExt;

use switchyard_server::build_switchyard_router;
use switchyard_server::config::load_server_runtime;

/// How an executor answers.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Reply {
    Ok,
    Http(u16),
    /// A 200 whose `answers` omits every question asked.
    Answerless,
}

fn status_reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        400 => "Bad Request",
        500 => "Internal Server Error",
        502 => "Bad Gateway",
        503 => "Service Unavailable",
        504 => "Gateway Timeout",
        _ => "Programmed",
    }
}

/// One executor's view of the request: the questions it was actually asked.
#[derive(Clone, Debug, Default)]
struct Seen {
    paths: Vec<String>,
    questions: Vec<Vec<String>>,
}

/// A loopback decision executor that answers EXACTLY the questions it is asked,
/// tagged with `answer_type` so the caller can tell the legs apart, and records
/// both its request path and the question keys it received.
async fn spawn_executor(
    answer_type: &'static str,
    reply: Reply,
    hits: Arc<AtomicUsize>,
    seen: Arc<Mutex<Seen>>,
) -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr: SocketAddr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        loop {
            let Ok((mut socket, _)) = listener.accept().await else {
                return;
            };
            hits.fetch_add(1, Ordering::SeqCst);
            tokio::time::timeout(
                std::time::Duration::from_secs(5),
                async {
                    let mut buffer = vec![0u8; 256 * 1024];
                    let read = socket.read(&mut buffer).await.unwrap_or(0);
                    let request = String::from_utf8_lossy(&buffer[..read]);
                    let path = request
                        .split_whitespace()
                        .nth(1)
                        .unwrap_or_default()
                        .to_string();
                    // The body follows the blank line. Only a typed decision
                    // leg carries a `questions` object, and every executor here
                    // is typed, so this is the exact set it was assigned.
                    let asked: Vec<String> = request
                        .split_once("\r\n\r\n")
                        .and_then(|(_, body)| serde_json::from_str::<Value>(body).ok())
                        .and_then(|value| {
                            value
                                .get("questions")
                                .and_then(Value::as_object)
                                .map(|questions| questions.keys().cloned().collect())
                        })
                        .unwrap_or_default();
                    {
                        let mut seen = seen.lock().unwrap();
                        seen.paths.push(path);
                        seen.questions.push(asked.clone());
                    }
                    let mut answers = serde_json::Map::new();
                    if reply != Reply::Answerless {
                        for question in &asked {
                            answers.insert(
                                question.clone(),
                                json!({"type": answer_type, "noul": 0.5, "score": 0.5}),
                            );
                        }
                    }
                    let body = match reply {
                        Reply::Ok => json!({
                            "model": format!("{answer_type}-exec"),
                            "answers": Value::Object(answers),
                            "usage": {"input_tokens": 10, "output_tokens": 0},
                        })
                        .to_string(),
                        Reply::Answerless => json!({
                            "model": format!("{answer_type}-exec"),
                            "answers": {},
                        })
                        .to_string(),
                        Reply::Http(status) => {
                            json!({"error": {"message": "programmed", "code": status}}).to_string()
                        }
                    };
                    let (status, reason) = match reply {
                        Reply::Http(status) => (status, status_reason(status)),
                        _ => (200u16, "OK"),
                    };
                    let response = format!(
                        "HTTP/1.1 {status} {reason}\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{body}",
                        body.len()
                    );
                    let _ = socket.write_all(response.as_bytes()).await;
                    let _ = socket.flush().await;
                },
            )
            .await;
        }
    });
    addr.port()
}

/// Six questions, deliberately supplied out of partition order so the
/// recombination order can be checked against the CALLER's order.
const SIX_QUESTIONS: &[(&str, &str)] = &[
    ("task_complexity", "a"),
    ("reasoning_demand", "b"),
    ("tool_dependency", "c"),
    ("agentic_complexity", "d"),
    ("route_sufficiency", "e"),
    ("local_suitability", "f"),
];

fn decide_body() -> Value {
    let mut questions = serde_json::Map::new();
    for (name, marker) in SIX_QUESTIONS {
        questions.insert(
            (*name).to_string(),
            json!({
                "type": "noul",
                "instructions": format!("assess {marker}"),
                // A noul question takes criteria keyed true/false.
                "criteria": {"true": "yes", "false": "no"},
            }),
        );
    }
    json!({
        "model": "decision-primary",
        "state": "the state under assessment",
        "questions": Value::Object(questions),
    })
}

/// A three-executor deployment: the primary, the DEFAULT fallback, and a
/// partition target. Each answers with a distinct type, so a response names the
/// executor that produced it.
fn partition_config(
    primary: u16,
    fallback: u16,
    partition_target: u16,
    partition_table: &str,
) -> String {
    partition_config_with(
        primary,
        fallback,
        partition_target,
        partition_table,
        "switchyard-decision:v1",
        "",
    )
}

/// The same deployment, with the PARTITION TARGET's contract and optional own
/// fallback varied, so every admission rule can be provoked through the REAL
/// loader rather than by appending a duplicate TOML table.
fn partition_config_with(
    primary: u16,
    fallback: u16,
    partition_target: u16,
    partition_table: &str,
    partition_contract: &str,
    partition_extra: &str,
) -> String {
    format!(
        r#"
schema_version = 1

[targets.primary-target]
id = "primary-target"
llm_client = "primary-chat"

[llm_clients.primary-chat]
format = "openai_responses"
base_url = "http://127.0.0.1:{primary}"

[capability_clients.primary-exec]
format = "openrouter_alpha_decisions"
base_url = "http://127.0.0.1:{primary}"
model = "primary-exec-model"
timeout_seconds = 5

[capability_clients.fallback-exec]
format = "openrouter_alpha_decisions"
base_url = "http://127.0.0.1:{fallback}"
model = "fallback-exec-model"
endpoint_path = "/v1/systemone"
timeout_seconds = 5

[capability_clients.partition-exec]
format = "openrouter_alpha_decisions"
base_url = "http://127.0.0.1:{partition_target}"
model = "partition-exec-model"
endpoint_path = "/v1/systemone-partition"
timeout_seconds = 5

[capabilities.decision-primary]
id = "decision-primary"
target = "primary-exec"
decision_contract = "switchyard-decision:v1"
max_questions = 8
max_state_chars = 4000
fallback_target = "decision-fallback"
{partition_table}

[capabilities.decision-fallback]
id = "decision-fallback"
target = "fallback-exec"
decision_contract = "switchyard-decision:v1"
max_questions = 8
max_state_chars = 4000

[capabilities.decision-partition]
id = "decision-partition"
target = "partition-exec"
decision_contract = "{partition_contract}"
max_questions = 8
max_state_chars = 4000
{partition_extra}

[routes.decision-primary]
id = "decision-primary"
type = "passthrough"
target = "primary-target"
"#
    )
}

/// The mixed policy under test: one question to the partition target, the rest to
/// the default fallback.
const MIXED_TABLE: &str = r#"question_partition = [{ target = "decision-partition", question_ids = ["reasoning_demand"] }]"#;

struct PartitionHarness {
    app: Router,
    _dir: std::path::PathBuf,
    primary_hits: Arc<AtomicUsize>,
    fallback_hits: Arc<AtomicUsize>,
    partition_hits: Arc<AtomicUsize>,
    primary_seen: Arc<Mutex<Seen>>,
    fallback_seen: Arc<Mutex<Seen>>,
    partition_seen: Arc<Mutex<Seen>>,
}

impl PartitionHarness {
    async fn with(primary: Reply, table: &str) -> Self {
        Self::with_all(primary, Reply::Ok, Reply::Ok, table).await
    }

    /// A harness where the PARTITION TARGET can also be made to fail, which is
    /// the only way to reach the "an assigned batch failed" branch.
    async fn with_all(primary: Reply, fallback: Reply, partition: Reply, table: &str) -> Self {
        let dir = std::env::temp_dir().join(format!(
            "decision-partition-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let (primary_hits, fallback_hits, partition_hits) = (
            Arc::new(AtomicUsize::new(0)),
            Arc::new(AtomicUsize::new(0)),
            Arc::new(AtomicUsize::new(0)),
        );
        let (primary_seen, fallback_seen, partition_seen) = (
            Arc::new(Mutex::new(Seen::default())),
            Arc::new(Mutex::new(Seen::default())),
            Arc::new(Mutex::new(Seen::default())),
        );
        let primary_port = spawn_executor(
            "noul",
            primary,
            Arc::clone(&primary_hits),
            Arc::clone(&primary_seen),
        )
        .await;
        let fallback_port = spawn_executor(
            "choice",
            fallback,
            Arc::clone(&fallback_hits),
            Arc::clone(&fallback_seen),
        )
        .await;
        let partition_port = spawn_executor(
            "score",
            partition,
            Arc::clone(&partition_hits),
            Arc::clone(&partition_seen),
        )
        .await;
        let config = dir.join("routes.toml");
        std::fs::write(
            &config,
            partition_config(primary_port, fallback_port, partition_port, table),
        )
        .expect("write config");
        let runtime = load_server_runtime(&config).expect("the runtime must load");
        let app = build_switchyard_router(runtime.state);
        Self {
            app,
            _dir: dir,
            primary_hits,
            fallback_hits,
            partition_hits,
            primary_seen,
            fallback_seen,
            partition_seen,
        }
    }

    async fn decide(&self) -> (u16, Value) {
        self.decide_with_questions(None).await
    }

    /// `subset` restricts the caller's question set, so a test can prove the
    /// behaviour of a partition target that claims none of what was asked.
    async fn decide_with_questions(&self, subset: Option<&[&str]>) -> (u16, Value) {
        let request = Request::builder()
            .method("POST")
            .uri("/v1/decisions")
            .header("content-type", "application/json")
            .body(Body::from(decide_body_for(subset).to_string()))
            .unwrap();
        let response = self
            .app
            .clone()
            .oneshot(request)
            .await
            .expect("the router must answer");
        let status = response.status().as_u16();
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        let parsed = if bytes.is_empty() {
            Value::Null
        } else {
            serde_json::from_slice(&bytes).unwrap_or(Value::Null)
        };
        tokio::time::sleep(std::time::Duration::from_millis(80)).await;
        (status, parsed)
    }

    /// The question keys one executor was actually asked, in order.
    fn asked(&self, executor: &str) -> Vec<String> {
        let seen = match executor {
            "primary" => &self.primary_seen,
            "fallback" => &self.fallback_seen,
            "partition" => &self.partition_seen,
            _ => panic!("unknown executor"),
        };
        let seen = seen.lock().unwrap();
        seen.questions.first().cloned().unwrap_or_default()
    }

    fn asked_count(&self, executor: &str) -> usize {
        let seen = match executor {
            "primary" => &self.primary_seen,
            "fallback" => &self.fallback_seen,
            "partition" => &self.partition_seen,
            _ => panic!("unknown executor"),
        };
        seen.lock().unwrap().questions.len()
    }
}

/// The caller body, optionally restricted to a subset of the six questions.
fn decide_body_for(subset: Option<&[&str]>) -> Value {
    let mut body = decide_body();
    if let Some(subset) = subset {
        let questions = body
            .get_mut("questions")
            .and_then(Value::as_object_mut)
            .expect("the body must carry questions");
        questions.retain(|key, _| subset.contains(&key.as_str()));
    }
    body
}

fn answer_type<'a>(body: &'a Value, question: &str) -> Option<&'a str> {
    body.get("answers")
        .and_then(|answers| answers.get(question))
        .and_then(|answer| answer.get("type"))
        .and_then(Value::as_str)
}

fn partitioned_questions(body: &Value) -> Vec<String> {
    body.get("answers")
        .and_then(Value::as_object)
        .map(|answers| answers.keys().cloned().collect())
        .unwrap_or_default()
}

// ---------------------------------------------------------------------------
// The historical contract must survive untouched.
// ---------------------------------------------------------------------------

/// With no partition declared, the fallback is ONE leg receiving ALL questions:
/// the partition target is never even contacted.
#[tokio::test]
async fn without_a_partition_the_fallback_is_one_leg_and_the_target_is_untouched() {
    let h = PartitionHarness::with(Reply::Http(503), "").await;
    let (status, body) = h.decide().await;
    assert_eq!(status, 200, "body: {body}");
    assert_eq!(h.primary_hits.load(Ordering::SeqCst), 1);
    assert_eq!(h.fallback_hits.load(Ordering::SeqCst), 1);
    assert_eq!(
        h.partition_hits.load(Ordering::SeqCst),
        0,
        "an unconfigured partition target must never be called"
    );
    assert_eq!(h.asked("fallback").len(), SIX_QUESTIONS.len());
    assert_eq!(
        body.get("fallback_partition"),
        None,
        "an un-partitioned fallback must not claim a partition: {body}"
    );
    assert_eq!(body["fallback_model"], "decision-fallback");
    assert_eq!(body["fallback_used"], true);
}

/// A DECLARED-BUT-EMPTY table must behave exactly like none, so an empty list can
/// never change behaviour by accident.
#[tokio::test]
async fn an_empty_partition_table_cannot_change_behaviour() {
    let h = PartitionHarness::with(Reply::Http(503), "question_partition = []").await;
    let (status, body) = h.decide().await;
    assert_eq!(status, 200, "body: {body}");
    assert_eq!(h.fallback_hits.load(Ordering::SeqCst), 1);
    assert_eq!(h.partition_hits.load(Ordering::SeqCst), 0);
    assert_eq!(h.asked("fallback").len(), SIX_QUESTIONS.len());
    assert_eq!(body.get("fallback_partition"), None, "body: {body}");
}

// ---------------------------------------------------------------------------
// The healthy primary path must be completely untouched.
// ---------------------------------------------------------------------------

/// The whole point of scoping partitioning to the eligible-fallback branch: a
/// healthy primary costs ONE call and contacts no partition target.
#[tokio::test]
async fn a_healthy_primary_costs_one_call_and_touches_no_partition_target() {
    let h = PartitionHarness::with(Reply::Ok, MIXED_TABLE).await;
    let (status, body) = h.decide().await;
    assert_eq!(status, 200, "body: {body}");
    assert_eq!(h.primary_hits.load(Ordering::SeqCst), 1);
    assert_eq!(h.fallback_hits.load(Ordering::SeqCst), 0);
    assert_eq!(h.partition_hits.load(Ordering::SeqCst), 0);
    assert_eq!(
        h.asked_count("partition"),
        0,
        "the target must not be contacted"
    );
    // The primary's own answer type is served unchanged.
    for (question, _) in SIX_QUESTIONS {
        assert_eq!(
            answer_type(&body, question),
            Some("noul"),
            "{question}: {body}"
        );
    }
}

// ---------------------------------------------------------------------------
// The partition itself.
// ---------------------------------------------------------------------------

/// The core claim: an eligible failure fans out, and each executor receives
/// EXACTLY its assigned questions.
#[tokio::test]
async fn an_eligible_failure_fans_out_one_call_per_assigned_executor() {
    let h = PartitionHarness::with(Reply::Http(503), MIXED_TABLE).await;
    let (status, body) = h.decide().await;
    assert_eq!(status, 200, "body: {body}");

    // ONE primary call, then ONE call per assigned executor: physical calls are
    // 1 + N while ROUTING STAGES remain two.
    assert_eq!(h.primary_hits.load(Ordering::SeqCst), 1);
    assert_eq!(h.partition_hits.load(Ordering::SeqCst), 1);
    assert_eq!(h.fallback_hits.load(Ordering::SeqCst), 1);

    assert_eq!(h.asked("partition"), vec!["reasoning_demand".to_string()]);
    let fallback_asked = h.asked("fallback");
    assert_eq!(
        fallback_asked.len(),
        5,
        "the other five: {fallback_asked:?}"
    );
    assert!(
        !fallback_asked.contains(&"reasoning_demand".to_string()),
        "the assigned question must not also go to the default executor"
    );
    for (question, _) in SIX_QUESTIONS {
        if *question != "reasoning_demand" {
            assert!(
                fallback_asked.contains(&question.to_string()),
                "{question} must be answered by the default executor"
            );
        }
    }

    // Each leg keeps its OWN endpoint path, so "the right executor ran" is
    // proved at the transport, not just inferred from an answer value.
    assert_eq!(
        h.partition_seen.lock().unwrap().paths.as_slice(),
        ["/v1/systemone-partition".to_string()],
        "the partition target must be called on its own path"
    );
    assert_eq!(
        h.fallback_seen.lock().unwrap().paths.as_slice(),
        ["/v1/systemone".to_string()],
        "the default executor must be called on its own path"
    );
}

/// The recombined body names the executor behind every answer, and every
/// question appears exactly once in the CALLER's order.
#[tokio::test]
async fn answers_are_recombined_in_the_callers_order_with_per_question_attribution() {
    let h = PartitionHarness::with(Reply::Http(503), MIXED_TABLE).await;
    let (status, body) = h.decide().await;
    assert_eq!(status, 200, "body: {body}");
    assert_eq!(answer_type(&body, "reasoning_demand"), Some("score"));
    for (question, _) in SIX_QUESTIONS {
        if *question != "reasoning_demand" {
            assert_eq!(answer_type(&body, question), Some("choice"), "{question}");
        }
    }
    // `preserve_order` is enabled, so this is the real key order.
    assert_eq!(
        partitioned_questions(&body),
        SIX_QUESTIONS
            .iter()
            .map(|(q, _)| q.to_string())
            .collect::<Vec<_>>()
    );
    assert_eq!(body["fallback_used"], true);
    // Two executors answered, so there is NO truthful single `fallback_model`.
    // `fallback_partition` is the authoritative per-question record.
    assert!(
        body.get("fallback_model").is_none(),
        "a mixed result must not claim one executor: {body}"
    );
    assert_eq!(body["served_model"], "decision-primary");
    assert_eq!(body["contract"], "switchyard-decision:v1");
    // The attribution map groups questions by the executor that answered them.
    let partition_map = body["fallback_partition"]["decision-partition"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    assert_eq!(partition_map, vec![json!("reasoning_demand")]);
    assert_eq!(
        body["fallback_partition"]["decision-fallback"]
            .as_array()
            .map(|v| v.len()),
        Some(5)
    );
}

/// A question the caller sends that no entry names goes to the DEFAULT executor,
/// which is the historical behaviour. A newly added question is therefore never
/// dropped and never silently unanswered.
#[tokio::test]
async fn an_unmapped_question_falls_to_the_default_executor() {
    let h = PartitionHarness::with(Reply::Http(503), MIXED_TABLE).await;
    let (status, body) = h.decide().await;
    assert_eq!(status, 200, "body: {body}");
    let asked = h.asked("fallback");
    assert!(
        asked.contains(&"local_suitability".to_string())
            && asked.contains(&"tool_dependency".to_string()),
        "unmapped questions must reach the default executor: {asked:?}"
    );
    assert_eq!(answer_type(&body, "local_suitability"), Some("choice"));
}

/// A partition entry naming a question the caller never sent must not create a
/// phantom request: that question simply is not routed.
#[tokio::test]
async fn a_partition_question_the_caller_never_sent_is_not_routed() {
    let h = PartitionHarness::with(
        Reply::Http(503),
        r#"question_partition = [{ target = "decision-partition", question_ids = ["reasoning_demand", "never_asked"] }]"#,
    )
    .await;
    let (status, body) = h.decide().await;
    assert_eq!(status, 200, "body: {body}");
    assert_eq!(h.asked("partition"), vec!["reasoning_demand".to_string()]);
    assert!(body["answers"].get("never_asked").is_none(), "body: {body}");
    assert_eq!(partitioned_questions(&body).len(), SIX_QUESTIONS.len());
}

/// Every availability class must still reach the partition; the fallback
/// classifier itself is unchanged.
#[tokio::test]
async fn every_eligible_failure_class_fans_out_exactly_once() {
    for status in [429u16, 500, 502, 503, 504] {
        let h = PartitionHarness::with(Reply::Http(status), MIXED_TABLE).await;
        let (code, body) = h.decide().await;
        assert_eq!(code, 200, "HTTP {status} must degrade: {body}");
        assert_eq!(h.primary_hits.load(Ordering::SeqCst), 1, "HTTP {status}");
        assert_eq!(h.partition_hits.load(Ordering::SeqCst), 1, "HTTP {status}");
        assert_eq!(h.fallback_hits.load(Ordering::SeqCst), 1, "HTTP {status}");
    }
}

/// A caller/auth failure is NOT availability, so no fan-out may occur.
#[tokio::test]
async fn an_ineligible_failure_never_fans_out() {
    let h = PartitionHarness::with(Reply::Http(400), MIXED_TABLE).await;
    let (status, body) = h.decide().await;
    assert_eq!(status, 502, "a 400 must surface, not degrade: {body}");
    assert_eq!(h.fallback_hits.load(Ordering::SeqCst), 0);
    assert_eq!(h.partition_hits.load(Ordering::SeqCst), 0);
}

// ---------------------------------------------------------------------------
// Fail closed. Never synthesise an answer that no executor produced.
// ---------------------------------------------------------------------------

/// A partition target that omits a question it was assigned fails the WHOLE leg.
/// There is no third stage and no invented 0/false/neutral stand-in.
#[tokio::test]
async fn an_executor_that_answers_nothing_fails_the_whole_leg_closed() {
    let dir = std::env::temp_dir().join(format!(
        "decision-partition-answerless-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let primary_hits = Arc::new(AtomicUsize::new(0));
    let fallback_hits = Arc::new(AtomicUsize::new(0));
    let partition_hits = Arc::new(AtomicUsize::new(0));
    let blank = || Arc::new(Mutex::new(Seen::default()));
    let primary_port =
        spawn_executor("noul", Reply::Http(503), Arc::clone(&primary_hits), blank()).await;
    let fallback_port =
        spawn_executor("choice", Reply::Ok, Arc::clone(&fallback_hits), blank()).await;
    let partition_port = spawn_executor(
        "score",
        Reply::Answerless,
        Arc::clone(&partition_hits),
        blank(),
    )
    .await;
    let config = dir.join("routes.toml");
    std::fs::write(
        &config,
        partition_config(primary_port, fallback_port, partition_port, MIXED_TABLE),
    )
    .unwrap();
    let runtime = load_server_runtime(&config).unwrap();
    let app = build_switchyard_router(runtime.state);
    let request = Request::builder()
        .method("POST")
        .uri("/v1/decisions")
        .header("content-type", "application/json")
        .body(Body::from(decide_body().to_string()))
        .unwrap();
    let response = app.oneshot(request).await.unwrap();
    assert_eq!(
        response.status().as_u16(),
        502,
        "a missing answer must fail visibly, never be synthesised"
    );
    assert_eq!(primary_hits.load(Ordering::SeqCst), 1);
    assert_eq!(partition_hits.load(Ordering::SeqCst), 1);
    assert_eq!(fallback_hits.load(Ordering::SeqCst), 1);
    let _ = std::fs::remove_dir_all(&dir);
}

/// A partition target that FAILS fails the whole leg, and is NOT rescued by the
/// default executor: there is no third stage and no fallback-of-fallback, so a
/// question is never silently re-routed to a different backend than the operator
/// declared for it. Every assigned executor is still contacted exactly once.
#[tokio::test]
async fn a_failing_partition_target_fails_the_whole_leg_without_a_third_stage() {
    for status in [500u16, 503, 429] {
        let h = PartitionHarness::with_all(
            Reply::Http(503),
            Reply::Ok,
            Reply::Http(status),
            MIXED_TABLE,
        )
        .await;
        let (code, body) = h.decide().await;
        assert_eq!(
            code, 502,
            "a failed partition target must fail visibly, never degrade to the default \
             executor: HTTP {status} gave {body}"
        );
        assert_eq!(h.primary_hits.load(Ordering::SeqCst), 1, "HTTP {status}");
        assert_eq!(
            h.partition_hits.load(Ordering::SeqCst),
            1,
            "HTTP {status}: the assigned executor runs exactly once, with no retry"
        );
        assert_eq!(
            h.fallback_hits.load(Ordering::SeqCst),
            1,
            "HTTP {status}: no third stage, so the default executor is never revisited"
        );
    }
}

/// The default executor failing alongside a healthy partition target still fails
/// the leg: a PARTIAL answer set is never served, because a missing answer would
/// silently change the consumer's decision.
#[tokio::test]
async fn a_failing_default_executor_also_fails_the_whole_leg() {
    let h = PartitionHarness::with_all(Reply::Http(503), Reply::Http(500), Reply::Ok, MIXED_TABLE)
        .await;
    let (code, body) = h.decide().await;
    assert_eq!(
        code, 502,
        "a partial answer set must never be served: {body}"
    );
    assert_eq!(h.partition_hits.load(Ordering::SeqCst), 1);
    assert_eq!(h.fallback_hits.load(Ordering::SeqCst), 1);
}

/// FINDING 1, the zero-hit regression. A configured partition target that the
/// caller gave NO matching question must not be contacted at all -- not called,
/// not a failure dependency, not attributed, not timed. The target here would
/// answer 500 if it were ever contacted, so a regression that called it would
/// fail the whole leg and turn this test red.
#[tokio::test]
async fn a_configured_target_with_zero_matching_questions_is_never_contacted() {
    let h = PartitionHarness::with_all(
        Reply::Http(503), // primary fails, forcing the R2 stage
        Reply::Ok,        // default executor healthy
        Reply::Http(500), // the partition target would FAIL if contacted
        MIXED_TABLE,
    )
    .await;
    // The caller asks for the FIVE signals that are NOT reasoning_demand, so the
    // configured Jev-side target claims nothing at all.
    let (status, body) = h
        .decide_with_questions(Some(&[
            "task_complexity",
            "tool_dependency",
            "agentic_complexity",
            "route_sufficiency",
            "local_suitability",
        ]))
        .await;
    assert_eq!(
        status, 200,
        "an unexercised target must not be able to sink the request: {body}"
    );
    assert_eq!(
        h.partition_hits.load(Ordering::SeqCst),
        0,
        "a target claiming zero questions must never be dialled"
    );
    assert_eq!(
        h.asked_count("partition"),
        0,
        "no request body was built for it"
    );
    // The default executor answered every question that WAS asked, truthfully
    // attributed: one executor answered, so the single-model identity is true.
    assert_eq!(h.fallback_hits.load(Ordering::SeqCst), 1);
    assert_eq!(answer_type(&body, "task_complexity"), Some("choice"));
    assert_eq!(body["fallback_model"], "decision-fallback");
    assert_eq!(partitioned_questions(&body).len(), 5);
    // A partition WAS declared and executed here (it simply claimed nothing), so
    // the attribution key is present and must name ONLY the real responder: the
    // unexercised target must not appear anywhere in it.
    let attributed = body["fallback_partition"].as_object().cloned().unwrap_or_default();
    assert!(
        !attributed.contains_key("decision-partition"),
        "an unexercised target must not be attributed: {body}"
    );
    assert_eq!(attributed.keys().collect::<Vec<_>>(), vec!["decision-fallback"]);
}

/// A partition that claims ONLY a question the caller never sends leaves the
/// whole request on the default executor, and the partition key must not appear.
#[tokio::test]
async fn a_partition_entirely_unmatched_degrades_to_the_default_executor() {
    let h = PartitionHarness::with_all(
        Reply::Http(503),
        Reply::Ok,
        Reply::Http(500),
        r#"question_partition = [{ target = "decision-partition", question_ids = ["not_asked_by_any_caller"] }]"#,
    )
    .await;
    let (status, body) = h.decide().await;
    assert_eq!(status, 200, "body: {body}");
    assert_eq!(h.partition_hits.load(Ordering::SeqCst), 0);
    assert_eq!(h.fallback_hits.load(Ordering::SeqCst), 1);
    assert_eq!(h.asked("fallback").len(), SIX_QUESTIONS.len());
    // One executor answered, so the historical single-model identity is truthful.
    assert_eq!(body["fallback_model"], "decision-fallback");
}

/// FINDING 2: when exactly ONE executor answers a partitioned fallback, the
/// single `fallback_model` identity IS truthful and must be present.
#[tokio::test]
async fn a_single_answering_executor_may_name_itself_in_fallback_model() {
    // EVERY question is assigned to the partition target, so it alone answers and
    // the default executor is never dialled.
    let all_questions: Vec<String> = SIX_QUESTIONS
        .iter()
        .map(|(question, _)| format!("\"{question}\""))
        .collect();
    let h = PartitionHarness::with_all(
        Reply::Http(503),
        Reply::Ok,
        Reply::Ok,
        &format!(
            "question_partition = [{{ target = \"decision-partition\", \
             question_ids = [{}] }}]",
            all_questions.join(", ")
        ),
    )
    .await;
    let (status, body) = h.decide().await;
    assert_eq!(status, 200, "body: {body}");
    assert_eq!(h.partition_hits.load(Ordering::SeqCst), 1);
    assert_eq!(
        h.fallback_hits.load(Ordering::SeqCst),
        0,
        "the default claimed nothing"
    );
    assert_eq!(body["fallback_model"], "decision-partition");
    assert_eq!(answer_type(&body, "reasoning_demand"), Some("score"));
}

// ---------------------------------------------------------------------------
// Admission: every partition defect must be a LOAD-time rejection.
// ---------------------------------------------------------------------------

/// Builds a config whose partition table is the caller's, and reports whether the
/// REAL loader accepted it.
async fn loads_with_partition_table(table: &str) -> Result<(), String> {
    loads_with(table, "switchyard-decision:v1", "").await
}

async fn loads_with(
    table: &str,
    partition_contract: &str,
    partition_extra: &str,
) -> Result<(), String> {
    let dir = std::env::temp_dir().join(format!(
        "decision-partition-admission-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    // Ports are never dialled: a rejected config fails before any request.
    let config = dir.join("routes.toml");
    std::fs::write(
        &config,
        partition_config_with(
            65500,
            65501,
            65502,
            table,
            partition_contract,
            partition_extra,
        ),
    )
    .unwrap();
    let outcome = load_server_runtime(&config)
        .map(|_| ())
        .map_err(|error| error.to_string());
    let _ = std::fs::remove_dir_all(&dir);
    outcome
}

#[tokio::test]
async fn a_partition_to_self_is_rejected_at_load() {
    let error = loads_with_partition_table(
        r#"question_partition = [{ target = "decision-primary", question_ids = ["reasoning_demand"] }]"#,
    )
    .await
    .expect_err("self-partition must be rejected");
    assert!(error.contains("partitions to itself"), "{error}");
}

#[tokio::test]
async fn a_partition_to_an_undeclared_capability_is_rejected_at_load() {
    let error = loads_with_partition_table(
        r#"question_partition = [{ target = "not-declared", question_ids = ["reasoning_demand"] }]"#,
    )
    .await
    .expect_err("an undeclared partition target must be rejected");
    assert!(error.contains("not declared"), "{error}");
}

#[tokio::test]
async fn a_partition_to_a_non_decisions_capability_is_rejected_at_load() {
    let error = loads_with_partition_table(
        r#"question_partition = [{ target = "embed-route", question_ids = ["reasoning_demand"] }]

[capabilities.embed-route]
id = "embed-route"
target = "primary-exec"
contract = "localclaw-embedding-space:v1"
dimensions = 4
"#,
    )
    .await
    .expect_err("a non-decisions partition target must be rejected");
    assert!(error.contains("not a decisions capability"), "{error}");
}

#[tokio::test]
async fn a_partition_target_on_a_different_contract_is_rejected_at_load() {
    let error = loads_with(MIXED_TABLE, "a-different-contract:v1", "")
        .await
        .expect_err("a contract mismatch must be rejected");
    assert!(error.contains("serves contract"), "{error}");
}

#[tokio::test]
async fn a_partition_target_with_its_own_fallback_is_rejected_at_load() {
    let error = loads_with(
        MIXED_TABLE,
        "switchyard-decision:v1",
        "fallback_target = \"decision-fallback\"",
    )
    .await
    .expect_err("fallback-of-fallback inside a partition must be rejected");
    assert!(error.contains("only one hop"), "{error}");
}

#[tokio::test]
async fn an_empty_question_list_is_rejected_at_load() {
    let error = loads_with_partition_table(
        r#"question_partition = [{ target = "decision-partition", question_ids = [] }]"#,
    )
    .await
    .expect_err("an empty question list must be rejected");
    assert!(error.contains("empty question list"), "{error}");
}

#[tokio::test]
async fn one_question_named_by_two_executors_is_rejected_at_load() {
    let error = loads_with_partition_table(
        r#"question_partition = [{ target = "decision-partition", question_ids = ["reasoning_demand"] }, { target = "decision-partition", question_ids = ["reasoning_demand"] }]"#,
    )
    .await
    .expect_err("a question routed to two executors must be rejected");
    assert!(error.contains("routes question"), "{error}");
}

/// A well-formed mixed policy must load: admission must not be vacuously strict.
#[tokio::test]
async fn a_well_formed_partition_loads() {
    loads_with_partition_table(MIXED_TABLE)
        .await
        .expect("a valid mixed policy must load");
}

// ---------------------------------------------------------------------------
// FINDING 4: a partition that can never execute must fail closed at load.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn a_partition_without_a_fallback_target_is_rejected_at_load() {
    // Partition declared, but no default executor to carry unmatched questions.
    let dir = std::env::temp_dir().join(format!(
        "decision-partition-no-fallback-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let config = dir.join("routes.toml");
    std::fs::write(
        &config,
        format!(
            r#"
schema_version = 1

[targets.aux-target]
id = "aux-target"
llm_client = "probe-chat"

[llm_clients.probe-chat]
format = "openai_responses"
base_url = "http://127.0.0.1:9"

[capability_clients.primary-exec]
format = "openrouter_alpha_decisions"
base_url = "http://127.0.0.1:9"
model = "p"
timeout_seconds = 1

[capabilities.decision-primary]
id = "decision-primary"
target = "primary-exec"
decision_contract = "switchyard-decision:v1"
question_partition = [{{ target = "decision-partition", question_ids = ["reasoning_demand"] }}]

[capabilities.decision-partition]
id = "decision-partition"
target = "primary-exec"
decision_contract = "switchyard-decision:v1"

[routes.decision-primary]
id = "decision-primary"
type = "passthrough"
target = "aux-target"
"#
        ),
    )
    .unwrap();
    let error = load_server_runtime(&config)
        .err()
        .map(|e| e.to_string())
        .expect("a partition with no default executor must be rejected");
    assert!(error.contains("no fallback_target"), "{error}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn a_blank_question_id_is_rejected_at_load() {
    let error = loads_with_partition_table(
        r#"question_partition = [{ target = "decision-partition", question_ids = ["   "] }]"#,
    )
    .await
    .expect_err("a whitespace-only question id must be rejected");
    assert!(error.contains("blank question id"), "{error}");
}

#[tokio::test]
async fn an_empty_partition_target_is_rejected_at_load() {
    let error = loads_with_partition_table(
        r#"question_partition = [{ target = "  ", question_ids = ["reasoning_demand"] }]"#,
    )
    .await
    .expect_err("an empty partition target must be rejected");
    assert!(error.contains("empty target"), "{error}");
}

#[tokio::test]
async fn too_many_partition_entries_are_rejected_at_load() {
    // The bound is the capability's own max_questions (8 in this fixture).
    let many: Vec<String> = (0..9).map(|i| format!("\"q{i}\"")).collect();
    let entries: Vec<String> = many
        .iter()
        .map(|q| {
            format!("{{ target = \"decision-partition\", question_ids = [{q}] }}")
        })
        .collect();
    let table = format!("question_partition = [{}]", entries.join(", "));
    let error = loads_with_partition_table(&table)
        .await
        .expect_err("an unbounded partition entry count must be rejected");
    assert!(error.contains("max_questions"), "{error}");
}
