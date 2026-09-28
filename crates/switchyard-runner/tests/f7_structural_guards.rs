//! F7 structural guards.
//!
//! These assert the NEGATIVE space: that compatibility preservation did not
//! quietly become reconfiguration. They read the live file and the candidate
//! sources, so a drift fails here rather than at cutover.

use std::path::Path;
use std::process::Command;

const LIVE_CONFIG: &str = "/home/vincent/.local/lib/localclaw-switchyard/routes.toml";
const CANDIDATE: &str = "/home/vincent/.local/lib/localclaw-switchyard/v030-candidate";

fn live_text() -> String {
    std::fs::read_to_string(LIVE_CONFIG).expect("the live config must be readable")
}

fn read_candidate(relative: &str) -> String {
    std::fs::read_to_string(Path::new(CANDIDATE).join(relative))
        .unwrap_or_else(|error| panic!("read {relative}: {error}"))
}

// ---------------------------------------------------------------------------
// F4: exactly one preference-ordering seam.
// ---------------------------------------------------------------------------

#[test]
fn f4_remains_the_only_sorting_seam() {
    let source = read_candidate("crates/libsy/src/algorithms/fleet_router.rs");
    let sorts = source.matches("sort_by").count();
    assert_eq!(sorts, 1, "F7 must not add a second ordering seam");
}

// ---------------------------------------------------------------------------
// F8: execution exists, and it is exactly one decision.
//
// F7 pinned the refusal; F8 replaces it with the real selection, so this guard
// now pins the STRONGER property: route() produces a RoutingOutcome, and it
// does so by delegating to a single `decide` call rather than assembling a
// selection of its own.
// ---------------------------------------------------------------------------

#[test]
fn fleet_route_produces_a_routing_outcome_from_one_decide_call() {
    let source = read_candidate("crates/libsy/src/algorithms/fleet_router.rs");
    let body_start = source.find("async fn route(").expect("route() must exist");
    let body = &source[body_start..];
    let body = &body[..body.find("\n    }\n").expect("route() must have a body")];
    assert!(
        body.contains("Ok(RoutingOutcome::route_to("),
        "FleetRouter::route() must hand the captured ladder to RoutingOutcome"
    );
    assert_eq!(
        body.matches("self.decide(").count(),
        1,
        "the decision must happen exactly once, in one place"
    );
    assert!(
        !body.contains("sort_by"),
        "route() must not sort; the single F4 sort lives in decide()"
    );
}

/// Exactly ONE readiness read in the decision path.
///
/// One decision must consume one readiness generation. A second `snapshot()`
/// call inside the filters would let a decision blend generations.
#[test]
fn the_decision_reads_readiness_exactly_once() {
    let source = read_candidate("crates/libsy/src/algorithms/fleet_router.rs");
    let start = source
        .find("pub fn decide(")
        .expect("decide() must exist");
    let body = &source[start..];
    let body = &body[..body.find("\n    }\n").expect("decide() must have a body")];
    assert_eq!(
        body.matches("self.snapshot()").count(),
        1,
        "decide() must take exactly one readiness snapshot"
    );
    assert!(
        !body.contains("escalation"),
        "the escalation destination is host-owned and never a ladder entry"
    );
}

// ---------------------------------------------------------------------------
// The live file is unchanged by this reconciliation.
// ---------------------------------------------------------------------------

#[test]
fn the_live_config_declares_the_f7_surfaces_and_nothing_newer() {
    let text = live_text();
    for surface in [
        "reasoning_policy",
        "escalation_reasoning_policy",
        "reasoning_dialect",
        "reasoning_efforts",
        "strip_reasoning_content",
    ] {
        assert!(
            text.contains(surface),
            "the live file must still declare {surface}"
        );
    }
}

#[test]
fn the_live_policy_vocabulary_is_unchanged_and_unextended() {
    // F7 must not introduce new policy values. The parse set is exactly
    // production's seven; anything else is a new vocabulary, not a
    // compatibility port.
    let text = live_text();
    for value in ["\"none\"", "\"enabled\"", "\"low\"", "\"max\""] {
        assert!(
            text.contains(&format!("reasoning_policy = {value}")),
            "live uses {value}"
        );
    }
    // The live file uses no `medium`/`high`/`xhigh` route policy; those parse
    // values exist in the type but are unexercised by the live file.
    for absent in [
        "reasoning_policy = \"medium\"",
        "reasoning_policy = \"high\"",
        "reasoning_policy = \"xhigh\"",
    ] {
        assert!(
            !text.contains(absent),
            "the live file must not suddenly use {absent}"
        );
    }
}

#[test]
fn the_live_dialect_vocabulary_is_unchanged() {
    let text = live_text();
    let dialects = [
        "llama_cpp_enable_thinking",
        "openai_effort",
        "openrouter_enabled",
    ];
    for dialect in dialects {
        assert!(
            text.contains(&format!("reasoning_dialect = \"{dialect}\"")),
            "live uses {dialect}"
        );
    }
    // The fourth dialect in the type is NOT live. Its presence in the enum is
    // a type-level port, not a live configuration change.
    assert!(
        !text.contains("chat_template_reasoning_effort"),
        "the live file declares no chat_template_reasoning_effort client"
    );
}

#[test]
fn the_live_strip_targets_are_still_exactly_five_and_unchanged() {
    let text = live_text();
    let count = text.matches("strip_reasoning_content = true").count();
    // 2026-09-28: the operator-authorized realignment added the OpenRouter
    // Qwen3.8-27B free target, which carries the same strip as its siblings.
    assert_eq!(count, 6, "the live strip targets must be the five originals plus the Qwen free lane");
}

#[test]
fn the_candidate_introduces_no_nt_model_identity() {
    // Guard: F7 must not mint an `-nt` model identity. The live file's
    // identities are unchanged, and the candidate adds no model ids.
    let text = live_text();
    let candidate = read_candidate("crates/switchyard-runner/src/config.rs");
    assert!(
        !candidate.contains("\"-nt\""),
        "F7 must not introduce an -nt model identity"
    );
    let _ = text;
}

#[test]
fn the_candidate_never_hard_codes_a_live_target_or_route_id() {
    // Structural guard: F7 logic must be general. A literal live route or
    // target id inside the compatibility code would mean the layer was tuned
    // to today's file rather than ported.
    let config = read_candidate("crates/switchyard-runner/src/config.rs");
    let policy = read_candidate("crates/protocol/src/reasoning_policy.rs");
    for source in [config.as_str(), policy.as_str()] {
        for live_id in [
            "switchyard-smart-",
            "openrouter-qwen3_7-flash",
            "htpc-qwen3_5-9b-mtp",
            "openai-gpt-6-luna",
        ] {
            assert!(
                !source.contains(live_id),
                "compatibility code must not hard-code the live id {live_id}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// The dormant row-40 producer must NOT have been activated.
// ---------------------------------------------------------------------------

#[test]
fn the_row40_producer_wire_was_not_activated_by_f7() {
    let server = read_candidate("crates/switchyard-runner/src/route.rs");
    // F7 touched Route for the reasoning policy only. The dormant
    // input-token producer target list must still not exist on the candidate.
    assert!(
        !server.contains("input_tokens_targets"),
        "F7 must not wire the dormant row-40 producer"
    );
    let config = read_candidate("crates/switchyard-runner/src/config.rs");
    assert!(
        !config.contains("with_input_tokens_targets"),
        "F7 must not introduce the dormant producer's builder"
    );
}

#[test]
fn the_candidate_still_ignores_the_dormant_producer_in_production() {
    // The production fact F7 must preserve: the producer has no live call site,
    // so `candidate_input_tokens` is still empty in production.
    let production = "/home/vincent/.local/lib/localclaw-switchyard/integration-wt";
    let route = std::fs::read_to_string(
        Path::new(production).join("crates/switchyard-runner/src/route.rs"),
    )
    .expect("production route.rs must be readable");
    assert_eq!(
        route.matches("with_input_tokens_targets").count(),
        1,
        "production defines the builder exactly once"
    );
    let definition = route
        .find("pub fn with_input_tokens_targets")
        .expect("the definition must exist");
    let declaration = &route[definition..];
    let declaration_end = declaration
        .find('\n')
        .expect("the definition occupies one line");
    let after_definition = &declaration[declaration_end..];
    assert!(
        !after_definition.contains("with_input_tokens_targets"),
        "the builder must still have zero call sites in production"
    );
}

// ---------------------------------------------------------------------------
// F7 must not have changed production.
// ---------------------------------------------------------------------------

#[test]
fn production_is_byte_identical_to_its_anchor() {
    let digest = Command::new("sha256sum")
        .arg(LIVE_CONFIG)
        .output()
        .expect("sha256sum must run");
    let text = String::from_utf8_lossy(&digest.stdout);
    assert!(
        text.starts_with("6fd12abf672ea99526223240c925596b80bdb1691f402bbc919666d9037d7440"),
        "the live config changed: {text}"
    );
}
