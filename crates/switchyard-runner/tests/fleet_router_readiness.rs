// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! F6 acceptance: operational readiness over otherwise-compatible candidates.
//!
//! F6 answers exactly one question — is this candidate *presently available*
//! under production readiness policy? It does not decide context admission
//! (row 40), reasoning/dialect expressibility (F7), preference (F4), or whether
//! anything is executed (F8).
//!
//! The four production distinctions are all preserved rather than collapsed:
//! statically eligible, ready, unready, and unobserved.

use std::collections::{BTreeMap, BTreeSet};

use libsy::{
    CandidateEligibility, CandidateState, EligibilityFacts, FleetCandidate, FleetSnapshot,
    ReadinessVerdict, candidate_readiness, readiness_filter, ready_candidates,
};
use switchyard_protocol::ModelId;
use switchyard_runner::facts_config::FleetReadinessConfig;

/// Location of the live deployment config. Overridable for a relocated tree.
const LIVE_CONFIG: &str = match option_env!("SWITCHYARD_LIVE_CONFIG") {
    Some(path) => path,
    None => "/home/vincent/.local/lib/localclaw-switchyard/routes.toml",
};

fn model(id: &str) -> ModelId {
    ModelId::from(id.to_string())
}

fn candidate(target: &str, rank: u16) -> FleetCandidate {
    FleetCandidate {
        target: model(target),
        tool_calling: true,
        reasoning: true,
        supports_vision: true,
        preference_rank: rank,
        usable_context_tokens: None,
    }
}

/// The production deployment, read-only.
fn live_document() -> toml::Table {
    let text = std::fs::read_to_string(LIVE_CONFIG)
        .unwrap_or_else(|error| panic!("live config {LIVE_CONFIG} must be readable: {error}"));
    toml::from_str(&text).unwrap_or_else(|error| panic!("live config must be valid TOML: {error}"))
}

fn live_readiness() -> FleetReadinessConfig {
    let table = live_document()
        .get("fleet_readiness")
        .cloned()
        .expect("live config must have a [fleet_readiness] table");
    table
        .try_into()
        .expect("the live [fleet_readiness] table must deserialize through the carried schema")
}

/// An all-eligible verdict list, so F6 is exercised over an input that F3 has
/// already admitted. Readiness is never allowed to be the reason a candidate
/// was excluded statically.
fn all_eligible(candidates: &[FleetCandidate]) -> Vec<CandidateEligibility<'_>> {
    let facts = EligibilityFacts {
        requires_tools: false,
        requires_vision: false,
        requires_reasoning: false,
    };
    candidates
        .iter()
        .map(|c| CandidateEligibility::Eligible(c))
        .map(|v| {
            let _ = &facts;
            v
        })
        .collect()
}

// ---------------------------------------------------------------------------
// The four-state truth table, in production terms.
// ---------------------------------------------------------------------------

#[test]
fn unobserved_is_fail_closed_and_distinct_from_observed_unready() {
    let candidates = vec![candidate("m/a", 1)];
    let snapshot = FleetSnapshot::empty();
    assert_eq!(
        candidate_readiness(&candidates[0], &snapshot),
        ReadinessVerdict::Unobserved,
        "a target with no observation must be unobserved, never implicitly ready"
    );
    assert!(
        !snapshot.state_for(&model("m/a")).is_some(),
        "an empty snapshot records no state at all"
    );

    let observed = FleetSnapshot::new(vec![(model("m/a"), CandidateState::not_ready())]).unwrap();
    assert_eq!(
        candidate_readiness(&candidates[0], &observed),
        ReadinessVerdict::Unready,
        "an observed, unavailable target is unready"
    );
    // The distinction is load-bearing: both fail closed, but for different
    // reasons, and only one of them means a producer has spoken.
    assert_ne!(
        candidate_readiness(&candidates[0], &snapshot),
        candidate_readiness(&candidates[0], &observed)
    );
}

#[test]
fn governed_and_ready_survives_the_filter() {
    let candidates = vec![candidate("m/a", 1), candidate("m/b", 2)];
    let snapshot = FleetSnapshot::new(vec![(model("m/a"), CandidateState::ready())]).unwrap();
    let ready: Vec<&FleetCandidate> = ready_candidates(&all_eligible(&candidates), &snapshot);
    assert_eq!(ready.len(), 1, "only the observed-ready candidate survives");
    assert_eq!(ready[0].target, model("m/a"));
}

#[test]
fn governed_and_explicitly_unready_is_filtered() {
    let candidates = vec![candidate("m/a", 1)];
    let snapshot = FleetSnapshot::new(vec![(model("m/a"), CandidateState::not_ready())]).unwrap();
    let ready = ready_candidates(&all_eligible(&candidates), &snapshot);
    assert!(
        ready.is_empty(),
        "an observed-unready candidate must not survive"
    );
}

#[test]
fn transition_required_is_valid_but_not_immediately_eligible() {
    // The distinct case production preserves: the candidate is capable and
    // configured, it simply needs an external transition first. The router
    // never performs that transition.
    let candidates = vec![candidate("m/a", 1)];
    let snapshot =
        FleetSnapshot::new(vec![(model("m/a"), CandidateState::transition_required())]).unwrap();
    assert_eq!(
        candidate_readiness(&candidates[0], &snapshot),
        ReadinessVerdict::TransitionRequired
    );
    assert!(ready_candidates(&all_eligible(&candidates), &snapshot).is_empty());
}

#[test]
fn an_ungoverned_candidate_is_exactly_as_unavailable_as_an_unobserved_one() {
    // Production fails closed for both. F6 must not make an ungoverned
    // candidate ready merely because no rule governs it.
    let candidates = vec![candidate("m/unknown", 1)];
    let snapshot = FleetSnapshot::new(vec![(model("m/other"), CandidateState::ready())]).unwrap();
    assert_eq!(
        candidate_readiness(&candidates[0], &snapshot),
        ReadinessVerdict::Unobserved,
        "an ungoverned target is unobserved, not exempt"
    );
}

// ---------------------------------------------------------------------------
// Readiness identity: shared across aliases, never broader than production.
// ---------------------------------------------------------------------------

#[test]
fn two_targets_sharing_one_model_id_share_one_observation() {
    // Production keys readiness on the model id, so sibling/alias targets that
    // resolve to the same model deliberately share one state.
    let candidates = vec![candidate("m/shared", 1), candidate("m/shared", 2)];
    let snapshot = FleetSnapshot::new(vec![(model("m/shared"), CandidateState::ready())]).unwrap();
    let ready = ready_candidates(&all_eligible(&candidates), &snapshot);
    assert_eq!(
        ready.len(),
        2,
        "both candidates resolve to the same model id and share its single observation"
    );
    // Keyed too narrowly (by target name) would leave one candidate unobserved.
    assert!(candidates[0].target == candidates[1].target);
}

#[test]
fn a_distinct_model_id_never_inherits_another_models_observation() {
    let candidates = vec![candidate("m/one", 1), candidate("m/two", 2)];
    let snapshot = FleetSnapshot::new(vec![(model("m/one"), CandidateState::ready())]).unwrap();
    let ready = ready_candidates(&all_eligible(&candidates), &snapshot);
    assert_eq!(ready.len(), 1, "only the observed model id is ready");
    assert_eq!(ready[0].target, model("m/one"));
    assert_eq!(
        candidate_readiness(&candidates[1], &snapshot),
        ReadinessVerdict::Unobserved,
        "a different model id must not inherit the observation"
    );
}

#[test]
fn a_duplicate_snapshot_key_is_rejected() {
    let duplicate = FleetSnapshot::new(vec![
        (model("m/a"), CandidateState::ready()),
        (model("m/a"), CandidateState::not_ready()),
    ]);
    assert!(
        duplicate.is_err(),
        "two observations for one model id would make a decision ambiguous"
    );
}

// ---------------------------------------------------------------------------
// Ordering purity: F6 filters, F4 is still the only ordering seam.
// ---------------------------------------------------------------------------

#[test]
fn the_readiness_filter_never_reorders_its_input() {
    // Declared rank order 3,1,2 — deliberately NOT already sorted. F6 must
    // return the survivors in exactly the order they arrived.
    let candidates = vec![
        candidate("m/c", 3),
        candidate("m/a", 1),
        candidate("m/b", 2),
    ];
    let snapshot = FleetSnapshot::new(vec![
        (model("m/a"), CandidateState::ready()),
        (model("m/b"), CandidateState::ready()),
        (model("m/c"), CandidateState::ready()),
    ])
    .unwrap();
    let ready: Vec<ModelId> = ready_candidates(&all_eligible(&candidates), &snapshot)
        .into_iter()
        .map(|c| c.target.clone())
        .collect();
    assert_eq!(
        ready,
        vec![model("m/c"), model("m/a"), model("m/b")],
        "F6 must preserve input order, not sort by preference_rank"
    );
}

#[test]
fn readiness_ignores_preference_rank_entirely() {
    // Two candidates, identical readiness, opposite ranks: the rank-1 candidate
    // being unready must not promote the rank-2 one any differently, and rank
    // must never rescue an unready candidate.
    let candidates = vec![candidate("m/first", 1), candidate("m/second", 2)];
    let snapshot = FleetSnapshot::new(vec![(model("m/second"), CandidateState::ready())]).unwrap();
    let ready = ready_candidates(&all_eligible(&candidates), &snapshot);
    assert_eq!(ready.len(), 1);
    assert_eq!(ready[0].target, model("m/second"));
    assert_eq!(
        candidate_readiness(&candidates[0], &snapshot),
        ReadinessVerdict::Unobserved,
        "the highest-preference candidate is not made ready by its rank"
    );
}

#[test]
fn the_filter_marks_rather_than_deletes() {
    // Every input candidate is accounted for, so a caller can still report why
    // each one did not survive.
    let candidates = vec![
        candidate("m/a", 1),
        candidate("m/b", 2),
        candidate("m/c", 3),
    ];
    let snapshot = FleetSnapshot::new(vec![(model("m/a"), CandidateState::ready())]).unwrap();
    let marked = readiness_filter(&all_eligible(&candidates), &snapshot);
    assert_eq!(
        marked.len(),
        3,
        "the filter marks every input; it drops nothing"
    );
    assert!(marked[0].is_ready());
    assert!(!marked[1].is_ready());
    assert!(!marked[2].is_ready());
}

#[test]
fn a_state_change_alters_the_next_decision_without_touching_metadata() {
    // A candidate's static profile is immutable across readiness generations.
    let candidates = vec![candidate("m/a", 4)];
    let before: FleetCandidate = candidates[0].clone();
    let ready_gen = FleetSnapshot::new(vec![(model("m/a"), CandidateState::ready())]).unwrap();
    let unready_gen =
        FleetSnapshot::new(vec![(model("m/a"), CandidateState::not_ready())]).unwrap();

    assert!(!ready_candidates(&all_eligible(&candidates), &ready_gen).is_empty());
    assert!(ready_candidates(&all_eligible(&candidates), &unready_gen).is_empty());

    let after = &candidates[0];
    assert_eq!(before.target, after.target);
    assert_eq!(before.preference_rank, after.preference_rank);
    assert_eq!(before.tool_calling, after.tool_calling);
    assert_eq!(before.reasoning, after.reasoning);
    assert_eq!(before.supports_vision, after.supports_vision);
    assert_eq!(before.usable_context_tokens, after.usable_context_tokens);
}

// ---------------------------------------------------------------------------
// Live production configuration.
// ---------------------------------------------------------------------------

/// Every model id the live `[fleet_readiness.*]` declares as governed.
fn live_governed_ids() -> BTreeSet<String> {
    let readiness = live_readiness();
    let mut governed: BTreeSet<String> = readiness.ready.iter().cloned().collect();
    governed.extend(readiness.transition_required.iter().cloned());
    if let Some(comfy) = &readiness.comfy {
        governed.insert(comfy.model.clone());
        governed.extend(comfy.governed.iter().map(|g| g.model.clone()));
    }
    if let Some(htpc) = &readiness.htpc {
        governed.insert(htpc.model.clone());
    }
    if let Some(resource) = &readiness.resource {
        governed.extend(resource.openai_gated.iter().cloned());
        governed.extend(resource.openclaw_openai_gated.iter().cloned());
        governed.extend(resource.deepseek_gated.iter().cloned());
    }
    governed
}

#[test]
fn the_live_readiness_table_deserializes_and_validates() {
    // The whole point of F6's config half: the exact live table loads.
    let readiness = live_readiness();
    assert_eq!(
        readiness.observe_interval_seconds, 5,
        "the live observe interval is 5 seconds"
    );
    assert!(
        readiness.comfy.is_some(),
        "live config declares a ComfyNinja readiness source"
    );
    assert!(
        readiness.htpc.is_some(),
        "live config declares an HTPC readiness source"
    );
    assert!(
        !readiness.ready.is_empty(),
        "live config declares static cloud base entries"
    );
    assert!(
        readiness.transition_required.is_empty(),
        "the live surface exercises no static transition_required entry"
    );
}

#[test]
fn the_live_comfy_gate_uses_exact_expected_resident_for_every_governed_target() {
    let comfy = live_readiness().comfy.expect("live comfy source");
    let expected = comfy
        .expected_resident
        .as_deref()
        .expect("the live comfy gate declares an exact expected_resident");
    assert!(
        !comfy.governed.is_empty(),
        "the live comfy source governs additional targets"
    );
    // Every governed target pins its OWN exact resident basename. This is what
    // stops a q3 candidate being marked ready while a q4 backing is resident:
    // the gate is an exact match, never a family/wildcard/substring match.
    // Note the expected value is the resident *basename*
    // (`qwen3.8-27b-q3`), while the governed model id is the full target id
    // (`comfyninja/qwen3.8-27b-q3`) - they are deliberately different strings,
    // and production compares the observed resident against the basename.
    for governed in &comfy.governed {
        let basename = governed.model.rsplit('/').next().unwrap_or(&governed.model);
        assert_eq!(
            governed.expected_resident, basename,
            "governed target {} must pin its own exact resident basename",
            governed.model
        );
    }
    let primary_basename = comfy.model.rsplit('/').next().unwrap_or(&comfy.model);
    assert_eq!(
        expected, primary_basename,
        "the primary gate pins its own resident too"
    );

    // No two governed targets may share a resident, or the exact gate could not
    // discriminate them.
    let mut residents: BTreeSet<&str> = comfy
        .governed
        .iter()
        .map(|g| g.expected_resident.as_str())
        .collect();
    residents.insert(expected);
    assert_eq!(
        residents.len(),
        comfy.governed.len() + 1,
        "each governed ComfyNinja target must pin a DISTINCT expected resident"
    );
}

#[test]
fn every_live_fleet_candidate_model_id_is_covered_by_a_readiness_entry() {
    // 100% coverage on the live file: an uncovered candidate would be
    // permanently fail-closed.
    let document = live_document();
    let targets: BTreeMap<String, String> = document
        .get("targets")
        .and_then(toml::Value::as_table)
        .expect("live config must have a [targets] table")
        .iter()
        .map(|(name, value)| {
            let table = value.as_table().expect("each [targets.*] must be a table");
            (
                name.clone(),
                table
                    .get("id")
                    .and_then(toml::Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
            )
        })
        .collect();
    let governed = live_governed_ids();

    let routes = document
        .get("routes")
        .and_then(toml::Value::as_table)
        .cloned()
        .expect("live config must have a [routes] table");

    let mut checked = 0usize;
    let mut uncovered = Vec::new();
    for (route_name, value) in &routes {
        let table = value.as_table().expect("each [routes.*] must be a table");
        if table.get("type").and_then(toml::Value::as_str) != Some("fleet_router") {
            continue;
        }
        for entry in table
            .get("candidates")
            .and_then(toml::Value::as_array)
            .expect("a fleet route must declare candidates")
        {
            let entry = entry.as_table().expect("each candidate must be a table");
            let target_name = entry
                .get("target")
                .and_then(toml::Value::as_str)
                .expect("each candidate must name a target");
            let model_id = targets.get(target_name).unwrap_or_else(|| {
                panic!("candidate target {target_name} must resolve in [targets.*]")
            });
            checked += 1;
            if !governed.contains(model_id) {
                uncovered.push(format!("{route_name} -> {target_name} ({model_id})"));
            }
        }
    }
    assert!(
        checked > 0,
        "the live config must exercise fleet candidates"
    );
    assert!(
        uncovered.is_empty(),
        "every live fleet candidate must be governed by some readiness entry, uncovered: {uncovered:?}"
    );
}

#[test]
fn live_cross_route_reuse_shares_one_readiness_identity() {
    // The TRUE live structure, which a correct census must not overstate: no
    // model id is reachable through more than one distinct candidate TARGET.
    // What the live file does exercise is one target name reused by several
    // routes, so a single observation governs that model everywhere it appears.
    let document = live_document();
    let targets: BTreeMap<String, String> = document
        .get("targets")
        .and_then(toml::Value::as_table)
        .expect("[targets] table")
        .iter()
        .map(|(name, value)| {
            let table = value.as_table().expect("target table");
            (
                name.clone(),
                table
                    .get("id")
                    .and_then(toml::Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
            )
        })
        .collect();
    let routes = document
        .get("routes")
        .and_then(toml::Value::as_table)
        .cloned()
        .expect("[routes] table");

    // model id -> the distinct target names that reach it
    let mut by_model: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    // model id -> the distinct ROUTES that reference it
    let mut model_routes: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for (route_name, value) in &routes {
        let table = value.as_table().expect("route table");
        if table.get("type").and_then(toml::Value::as_str) != Some("fleet_router") {
            continue;
        }
        for entry in table
            .get("candidates")
            .and_then(toml::Value::as_array)
            .unwrap_or(&Vec::new())
        {
            if let Some(entry) = entry.as_table() {
                if let Some(name) = entry.get("target").and_then(toml::Value::as_str) {
                    if let Some(id) = targets.get(name) {
                        by_model
                            .entry(id.clone())
                            .or_default()
                            .insert(name.to_string());
                        model_routes
                            .entry(id.clone())
                            .or_default()
                            .insert(route_name.clone());
                    }
                }
            }
        }
    }

    // The honest live fact: no model id is reached by two different target names.
    for (id, names) in &by_model {
        assert_eq!(
            names.len(),
            1,
            "live model id {id} unexpectedly resolves through multiple targets {names:?}; \
             if the live topology changed, readiness keying must be re-examined"
        );
    }

    // The reuse that IS live: one model referenced by several routes.
    let reused: Vec<(&String, &BTreeSet<String>)> =
        model_routes.iter().filter(|(_, rs)| rs.len() > 1).collect();
    assert!(
        !reused.is_empty(),
        "the live surface must exercise a model referenced by several routes"
    );

    for (model_id, route_names) in reused {
        // Two DISTINCT routes, one model, one shared observation.
        let shared: Vec<FleetCandidate> = (1..=2)
            .map(|i| {
                let mut c = candidate(model_id, i as u16);
                c.target = model(model_id);
                c
            })
            .collect();
        let snapshot =
            FleetSnapshot::new(vec![(model(model_id), CandidateState::ready())]).unwrap();
        let ready = ready_candidates(&all_eligible(&shared), &snapshot);
        assert_eq!(
            ready.len(),
            2,
            "one observation for {model_id} must govern it in every route that uses it \
             (routes: {route_names:?})"
        );
    }
}

#[test]
fn readiness_is_not_keyed_on_route_or_target_name() {
    // Two different routes referencing targets that resolve to DIFFERENT model
    // ids must not share state, and a target name must never be the key.
    let mut a = candidate("alias/one", 1);
    let mut b = candidate("alias/two", 2);
    a.target = model("real/one");
    b.target = model("real/two");
    let snapshot = FleetSnapshot::new(vec![(model("real/one"), CandidateState::ready())]).unwrap();
    let pair = [a, b];
    let ready = ready_candidates(&all_eligible(&pair), &snapshot);
    assert_eq!(ready.len(), 1);
    assert_eq!(ready[0].target, model("real/one"));
}

// ---------------------------------------------------------------------------
// Discriminating arms for the semantics the first mutant battery missed.
// ---------------------------------------------------------------------------

#[test]
fn a_state_that_is_both_ready_and_transition_required_is_still_not_eligible() {
    // No constructor produces this combination, so the `&&` conjunct in
    // `is_immediately_eligible` looks redundant until it is tested directly.
    // Production's rule is the conjunction, so the conjunction is the contract:
    // a candidate claiming ready while also demanding a transition is NOT
    // immediately available. Dropping the conjunct would mark it ready.
    let impossible = CandidateState {
        ready: true,
        transition_required: true,
    };
    assert!(
        !impossible.is_immediately_eligible(),
        "ready=true must not override an outstanding transition requirement"
    );
    let candidates = vec![candidate("m/a", 1)];
    let snapshot = FleetSnapshot::new(vec![(model("m/a"), impossible)]).unwrap();
    assert_eq!(
        candidate_readiness(&candidates[0], &snapshot),
        ReadinessVerdict::TransitionRequired,
        "the transition requirement wins over the ready flag"
    );
    assert!(
        ready_candidates(&all_eligible(&candidates), &snapshot).is_empty(),
        "such a candidate must not be selectable for an immediate request"
    );
}

#[test]
fn a_static_transition_required_entry_is_never_immediately_eligible() {
    // The live file declares none today, so the branch is proven on a fixture
    // with the same shape the producer would publish for a sealed-idle model.
    let doc = format!(
        r#"
schema_version = 1

[fleet_readiness]
observe_interval_seconds = 30
transition_required = ["sealed/idle-model"]

[llm_clients.local]
format = "openai_chat"
base_url = "https://example.invalid"

[targets.t1]
id = "m1"
llm_client = "local"

[routes."fleet/test"]
id = "fleet/test"
type = "fleet_router"
tool_calling = false
reasoning = false
candidates = [
  {{ target = "t1", tool_calling = false, reasoning = false, supports_vision = false }},
]
"#
    );
    let runner =
        switchyard_runner::Runner::from_toml(&doc).expect("a sealed-idle declaration must load");
    let readiness = runner
        .fleet_readiness()
        .expect("the runner must carry the declaration");
    assert_eq!(
        readiness.transition_required,
        vec!["sealed/idle-model".to_string()],
        "the static sealed-idle entry must be carried verbatim"
    );
    // And it is not ready, matching its name and production's semantics.
    assert!(
        !CandidateState::transition_required().is_immediately_eligible(),
        "a sealed-idle model is valid but not immediately available"
    );
}

#[test]
fn a_duplicate_model_across_two_readiness_sources_is_refused_at_load() {
    // A duplicate key would make every snapshot cycle ambiguous, so production
    // fails loudly at load rather than stranding the fleet fail-closed.
    let doc = format!(
        r#"
schema_version = 1

[fleet_readiness]
ready = ["shared/model"]

[llm_clients.local]
format = "openai_chat"
base_url = "https://example.invalid"

[targets.t1]
id = "m1"
llm_client = "local"

[routes."fleet/test"]
id = "fleet/test"
type = "passthrough"
target = "t1"
"#
    );
    assert!(
        switchyard_runner::Runner::from_toml(&doc).is_ok(),
        "a disjoint readiness declaration must load (control arm)"
    );

    // `htpc` is a sub-table of [fleet_readiness], so it is appended inside it.
    let duplicate = doc.replace(
        "[llm_clients.local]",
        "[fleet_readiness.htpc]\nbase_url = \"https://htpc.invalid\"\nexpected_model = \"m1\"\nmodel = \"shared/model\"\n\n[llm_clients.local]",
    );
    let message = match switchyard_runner::Runner::from_toml(&duplicate) {
        Ok(_) => panic!("a model declared in two readiness sources must be refused"),
        Err(error) => error.to_string(),
    };
    assert!(
        message.contains("shared/model") && message.contains("appears in both"),
        "the refusal must name the duplicated model and both sources, got: {message}"
    );
}

#[test]
fn a_gated_list_without_its_observation_source_is_refused_at_load() {
    // Otherwise that candidate is silently excluded every cycle, which is the
    // silent-failure mode the invariant exists to prevent.
    let base = format!(
        r#"
schema_version = 1

[llm_clients.local]
format = "openai_chat"
base_url = "https://example.invalid"

[targets.t1]
id = "m1"
llm_client = "local"

[routes."fleet/test"]
id = "fleet/test"
type = "passthrough"
target = "t1"

[fleet_readiness.resource]
deepseek_gated = ["deepseek-flash"]
"#
    );
    let message = match switchyard_runner::Runner::from_toml(&base) {
        Ok(_) => panic!("a gated list with no observation source must be refused"),
        Err(error) => error.to_string(),
    };
    assert!(
        message.contains("deepseek_gated") && message.contains("deepseek_url"),
        "the refusal must name the gated list and the missing source, got: {message}"
    );

    // With the source configured, the same shape loads.
    let complete = base.replace(
        "deepseek_gated = [\"deepseek-flash\"]",
        "deepseek_url = \"https://api.deepseek.com/user/balance\"\ndeepseek_api_key_env = \"DS_KEY\"\ndeepseek_currency = \"CNY\"\ndeepseek_gated = [\"deepseek-flash\"]",
    );
    assert!(
        switchyard_runner::Runner::from_toml(&complete).is_ok(),
        "a fully sourced gated list must load"
    );
}

#[test]
fn a_typo_in_a_readiness_field_name_is_refused_not_ignored() {
    // `deny_unknown_fields` keeps guarding the schema: a mistyped
    // `expected_resident` would otherwise leave the gate in legacy mode while
    // the operator believes it is exact. The assertion is on the PARSED RESULT
    // (the field must not survive as an ignored key), not merely on the
    // presence of an "unknown field" phrase in some message.
    let field = "expected_resident_name";
    let doc = format!(
        r#"
schema_version = 1

[fleet_readiness]
ready = ["other/model"]

[llm_clients.local]
format = "openai_chat"
base_url = "https://example.invalid"

[targets.t1]
id = "m1"
llm_client = "local"

[routes."fleet/test"]
id = "fleet/test"
type = "passthrough"
target = "t1"

[fleet_readiness.comfy]
url = "https://comfy.invalid/v1/resource"
auth_token_env = "TOKEN"
model = "m1"
{field} = "typo"
"#
    );
    assert!(
        switchyard_runner::Runner::from_toml(&doc).is_err(),
        "a mistyped readiness field name must be REFUSED, not silently ignored"
    );

    // Positive control: the correctly spelled field loads and is carried.
    let good = doc.replace(field, "expected_resident");
    let runner = switchyard_runner::Runner::from_toml(&good)
        .expect("the correctly spelled field must load (positive control)");
    let readiness = runner.fleet_readiness().expect("readiness declaration");
    let comfy = readiness.comfy.as_ref().expect("comfy source");
    assert_eq!(
        comfy.expected_resident.as_deref(),
        Some("typo"),
        "the correctly spelled field must be parsed and carried"
    );
}

#[test]
fn a_typo_in_a_top_level_readiness_field_is_refused_not_ignored() {
    // The sub-table guards (comfy/htpc/resource) each deny unknown fields in
    // their own right, so a sub-table typo cannot prove the TOP-LEVEL guard
    // works. This exercises `[fleet_readiness]` directly: a misspelled
    // `observe_interval_seconds` would otherwise fall back to the 30s default
    // while the operator believes they configured their interval.
    let base = |field: &str| {
        format!(
            r#"
schema_version = 1

[llm_clients.local]
format = "openai_chat"
base_url = "https://example.invalid"

[targets.t1]
id = "m1"
llm_client = "local"

[routes."fleet/test"]
id = "fleet/test"
type = "passthrough"
target = "t1"

[fleet_readiness]
{field} = 5
"#
        )
    };
    assert!(
        switchyard_runner::Runner::from_toml(&base("observe_intervall_seconds")).is_err(),
        "a misspelled top-level readiness field must be REFUSED, not defaulted away"
    );
    let runner = switchyard_runner::Runner::from_toml(&base("observe_interval_seconds"))
        .expect("the correctly spelled top-level field must load (positive control)");
    let readiness = runner.fleet_readiness().expect("readiness declaration");
    assert_eq!(
        readiness.observe_interval_seconds, 5,
        "the correctly spelled field must be parsed and carried"
    );
}
