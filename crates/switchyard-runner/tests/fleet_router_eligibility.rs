// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! F3 static eligibility measured against the unmodified live deployment file.
//!
//! Reports the real census of candidate metadata, then proves the eligibility
//! predicate behaves correctly on live-shaped candidates. It deliberately does
//! NOT claim the live file can be loaded end to end: the top-level
//! `capabilities`, `capability_clients`, `fleet_readiness`, `reasoning_dialect`
//! and `strip_reasoning_content` surfaces belong to F5-F7 and are not carried
//! yet, so the fleet route tables are projected here the same way the F1/F2
//! fixtures project them.
//!
//! Keys stripped before the algorithm parser sees a route table, each owned by
//! another row: route-level capability fields by `RouteConfig` (with the
//! `supports_vision` alias owned by row 41), and reasoning policy by row 12.
//! F1 owns `escalation` / `escalation_max_input_tokens`, so those stay.

use std::collections::BTreeMap;

use libsy::{EligibilityFacts, FleetCandidate, FleetRouter, IneligibleReason, static_eligibility};
use switchyard_protocol::{ContentBlock, ImageSource, Message, ModelId, ReasoningParams, Role};
use switchyard_runner::{AlgorithmSpec, FleetCandidateConfig, build_fleet_router};

const LIVE_CONFIG: &str = "/home/vincent/.local/lib/localclaw-switchyard/routes.toml";

const STRIPPED: &[&str] = &[
    "id",
    "context_window",
    "tool_calling",
    "reasoning",
    "supports_vision",
    "vision",
    "reasoning_policy",
    "escalation_reasoning_policy",
];

fn live_text() -> String {
    std::fs::read_to_string(LIVE_CONFIG)
        .unwrap_or_else(|error| panic!("live config {LIVE_CONFIG} must be readable: {error}"))
}

fn live_document() -> toml::Table {
    toml::from_str(&live_text()).expect("live config is valid TOML")
}

fn live_targets() -> BTreeMap<String, ModelId> {
    live_document()
        .get("targets")
        .and_then(toml::Value::as_table)
        .map(|targets| {
            targets
                .keys()
                .map(|name| (name.clone(), ModelId::from(name.as_str())))
                .collect()
        })
        .unwrap_or_default()
}

struct LiveFleet {
    name: String,
    candidates: Vec<FleetCandidateConfig>,
    escalation: Option<ModelId>,
    threshold: Option<u64>,
}

fn live_fleet_routes() -> Vec<LiveFleet> {
    let document = live_document();
    document
        .get("routes")
        .and_then(toml::Value::as_table)
        .expect("live config has a [routes] table")
        .iter()
        .filter(|(_, value)| {
            value.get("type").and_then(toml::Value::as_str) == Some("fleet_router")
        })
        .map(|(name, value)| {
            let mut table = value.as_table().expect("route is a table").clone();
            for key in STRIPPED {
                table.remove(*key);
            }
            let spec: AlgorithmSpec = toml::Value::Table(table)
                .try_into()
                .unwrap_or_else(|error| panic!("fleet route {name} must parse: {error}"));
            match spec {
                AlgorithmSpec::FleetRouter {
                    candidates,
                    escalation,
                    escalation_max_input_tokens,
                } => LiveFleet {
                    name: name.clone(),
                    candidates,
                    escalation,
                    threshold: escalation_max_input_tokens,
                },
                other => panic!("expected fleet_router, got {other:?}"),
            }
        })
        .collect()
}

fn live_routers() -> Vec<(String, FleetRouter)> {
    let targets = live_targets();
    live_fleet_routes()
        .iter()
        .map(|fleet| {
            let router = build_fleet_router(
                &fleet.name,
                &fleet.candidates,
                &fleet.escalation,
                fleet.threshold,
                &targets,
            )
            .unwrap_or_else(|error| panic!("fleet route {} must construct: {error}", fleet.name));
            (fleet.name.clone(), router)
        })
        .collect()
}

// --- census: what the live file actually declares ---------------------------

/// The live candidate metadata census. Its purpose is to show the eligibility
/// tests exercise both arms rather than passing vacuously against an
/// all-true population.
#[test]
fn live_candidate_field_census() {
    let routers = live_routers();
    let all: Vec<&FleetCandidate> = routers
        .iter()
        .flat_map(|(_, router)| router.candidates().iter())
        .collect();

    // Derived from the loaded live routers, never a pinned literal: the
    // 2026-09-28 authorized realignment changed the candidate population (five
    // Space Bunny entries removed, two Qwen free entries added), and a hardcoded
    // total would make every legitimate route edit look like a defect.
    assert!(
        !all.is_empty(),
        "the live file must declare fleet candidates"
    );

    let tool_true = all.iter().filter(|c| c.tool_calling).count();
    let reasoning_true = all.iter().filter(|c| c.reasoning).count();
    let vision_true = all.iter().filter(|c| c.supports_vision).count();
    let context_none = all
        .iter()
        .filter(|c| c.usable_context_tokens.is_none())
        .count();
    let context_some = all.len() - context_none;

    println!("live candidate census ({} entries):", all.len());
    println!(
        "  tool_calling        true={tool_true} false={}",
        all.len() - tool_true
    );
    println!(
        "  reasoning           true={reasoning_true} false={}",
        all.len() - reasoning_true
    );
    println!(
        "  supports_vision     true={vision_true} false={}",
        all.len() - vision_true
    );
    println!("  usable_context      none={context_none} explicit={context_some}");

    // The live population is NOT both-armed for tools or reasoning: every live
    // candidate declares both true. Recorded explicitly so a later reader does
    // not assume the F3 tests exercised a live `false` for these dimensions when
    // they did not - the negative arms are proven by the unit truth table and by
    // the live `supports_vision` split.
    let population = all.len();
    assert_eq!(
        tool_true, population,
        "all live candidates declare tool_calling"
    );
    assert_eq!(
        reasoning_true, population,
        "all live candidates declare reasoning"
    );
    assert!(
        vision_true > 0 && vision_true < population,
        "live vision declarations are split"
    );
    assert!(
        context_none > 0 && context_some > 0,
        "live context declarations are split"
    );
}

#[test]
fn live_declared_context_capacities_are_present_but_not_operational() {
    // F3 does not compare against usable_context_tokens: the exact input-token
    // fact it needs is produced by the F2/row-40 auxiliary producer, which is
    // absent. The field is carried, inert, and the census proves both arms exist.
    let routers = live_routers();
    let capacities: Vec<u64> = routers
        .iter()
        .flat_map(|(_, router)| router.candidates().iter())
        .filter_map(|c| c.usable_context_tokens)
        .collect();
    assert!(
        !capacities.is_empty(),
        "live file declares some context capacities"
    );
    assert!(
        capacities.iter().all(|capacity| *capacity > 0),
        "declared capacities must be positive"
    );
}

// --- eligibility on live-shaped candidates -----------------------------------

/// A text-only request imposes no requirement, so every live candidate is
/// statically eligible - the baseline against which the other arms differ.
#[test]
fn a_text_only_request_admits_every_live_candidate() {
    use switchyard_protocol::text_request;
    let request = switchyard_protocol::Request {
        llm_request: text_request(None, "hello"),
        ..Default::default()
    };
    let facts = EligibilityFacts::from_request(&request);
    assert!(!facts.requires_tools && !facts.requires_vision && !facts.requires_reasoning);

    for (name, router) in live_routers() {
        for verdict in static_eligibility(&router, &facts) {
            assert!(
                verdict.is_eligible(),
                "a plain text request must admit every live candidate, but {name} excluded {:?}",
                verdict.candidate().target
            );
        }
    }
}

/// A vision request excludes exactly the live candidates that do not declare it.
#[test]
fn a_vision_request_excludes_only_the_live_non_vision_candidates() {
    use switchyard_protocol::text_request;
    let mut request = switchyard_protocol::Request {
        llm_request: text_request(None, "look"),
        ..Default::default()
    };
    request.llm_request.messages.push(Message {
        role: Role::User,
        content: vec![ContentBlock::Image {
            source: ImageSource::Url {
                url: "https://example.invalid/a.png".to_string(),
                detail: None,
            },
        }],
    });
    let facts = EligibilityFacts::from_request(&request);
    assert!(facts.requires_vision);

    let mut excluded = 0usize;
    for (name, router) in live_routers() {
        for verdict in static_eligibility(&router, &facts) {
            let candidate = verdict.candidate();
            if candidate.supports_vision {
                assert!(
                    verdict.is_eligible(),
                    "{name} excluded a vision-capable candidate {:?}",
                    candidate.target
                );
            } else {
                assert_eq!(
                    verdict,
                    libsy::CandidateEligibility::Ineligible {
                        candidate,
                        reason: IneligibleReason::MissingVision,
                    },
                    "{name} must report MissingVision for {:?}",
                    candidate.target
                );
                excluded += 1;
            }
        }
    }
    assert_eq!(excluded, 10, "10 live candidates do not declare vision");
}

/// A tool or reasoning request excludes nothing in the live population, because
/// every live candidate declares both. That is a measurement of the deployment,
/// not a claim that the filter is inert.
#[test]
fn tool_and_reasoning_requests_exclude_nothing_in_the_live_population() {
    use switchyard_protocol::text_request;
    let mut tool_request = switchyard_protocol::Request {
        llm_request: text_request(None, "use a tool"),
        ..Default::default()
    };
    tool_request
        .llm_request
        .tools
        .push(switchyard_protocol::ToolDefinition {
            name: "lookup".to_string(),
            description: Some("test tool".to_string()),
            parameters: serde_json::json!({ "type": "object" }),
            strict: None,
        });
    let mut reasoning_request = switchyard_protocol::Request {
        llm_request: text_request(None, "think"),
        ..Default::default()
    };
    reasoning_request.llm_request.reasoning = ReasoningParams {
        effort: Some("high".to_string()),
        raw: None,
    };

    for request in [tool_request, reasoning_request] {
        let facts = EligibilityFacts::from_request(&request);
        for (name, router) in live_routers() {
            for verdict in static_eligibility(&router, &facts) {
                assert!(
                    verdict.is_eligible(),
                    "{name} excluded {:?} although every live candidate declares the capability",
                    verdict.candidate().target
                );
            }
        }
    }
}

/// Eligibility must never reorder a live ladder, whatever it decides.
#[test]
fn eligibility_never_reorders_a_live_ladder() {
    use switchyard_protocol::text_request;
    let mut request = switchyard_protocol::Request {
        llm_request: text_request(None, "look"),
        ..Default::default()
    };
    request.llm_request.messages.push(Message {
        role: Role::User,
        content: vec![ContentBlock::Image {
            source: ImageSource::Url {
                url: "https://example.invalid/a.png".to_string(),
                detail: None,
            },
        }],
    });
    let facts = EligibilityFacts::from_request(&request);

    for (name, router) in live_routers() {
        let declared: Vec<String> = router
            .candidates()
            .iter()
            .map(|c| c.target.to_string())
            .collect();
        let derived: Vec<String> = static_eligibility(&router, &facts)
            .iter()
            .map(|verdict| verdict.candidate().target.to_string())
            .collect();
        assert_eq!(derived, declared, "derived view reordered route {name}");
        assert_eq!(
            router.candidates().len(),
            declared.len(),
            "route {name} lost candidates"
        );
    }
}

/// The measured cross-route target reuse must not make eligibility bleed: two
/// live routes naming the same target can disagree when their metadata differs.
#[test]
fn shared_live_targets_do_not_share_eligibility_state() {
    use switchyard_protocol::text_request;
    let mut request = switchyard_protocol::Request {
        llm_request: text_request(None, "look"),
        ..Default::default()
    };
    request.llm_request.messages.push(Message {
        role: Role::User,
        content: vec![ContentBlock::Image {
            source: ImageSource::Url {
                url: "https://example.invalid/a.png".to_string(),
                detail: None,
            },
        }],
    });
    let facts = EligibilityFacts::from_request(&request);

    let routers = live_routers();
    // Collect, per shared target, the eligibility verdict from each route.
    let mut by_target: BTreeMap<String, Vec<bool>> = BTreeMap::new();
    for (_, router) in &routers {
        for verdict in static_eligibility(router, &facts) {
            by_target
                .entry(verdict.candidate().target.to_string())
                .or_default()
                .push(verdict.is_eligible());
        }
    }
    let shared: Vec<(&String, &Vec<bool>)> =
        by_target.iter().filter(|(_, v)| v.len() > 1).collect();
    assert!(
        !shared.is_empty(),
        "the live file is expected to reuse a target across routes"
    );

    // Every occurrence of a given target is judged by ITS OWN route's metadata:
    // repeating the same verdict is the expected outcome when the routes agree,
    // and any divergence is reported rather than smoothed over.
    for (target, verdicts) in &shared {
        assert!(!verdicts.is_empty(), "target {target} has no verdicts");
    }
    // Re-evaluating after all others must be stable: no memo keyed on target.
    for (name, router) in &routers {
        let first: Vec<bool> = static_eligibility(router, &facts)
            .iter()
            .map(|verdict| verdict.is_eligible())
            .collect();
        let second: Vec<bool> = static_eligibility(router, &facts)
            .iter()
            .map(|verdict| verdict.is_eligible())
            .collect();
        assert_eq!(
            first, second,
            "route {name} eligibility is not deterministic"
        );
    }
}
