// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Row 40 acceptance: exact per-candidate context admission.
//!
//! Row 40 answers one question: does this request provably fit this candidate's
//! qualified usable context? It does not decide readiness (F6), reasoning or
//! dialect (F7), preference (F4), or execution (F8).
//!
//! The producer is a separate concern and lives in the serving host
//! (`prepare_candidate_context_facts` in production). What is proven here is the
//! CONSUMER contract and the producer SEAM, on deterministic fixtures — no
//! provider is contacted.

use std::collections::BTreeSet;

use libsy::{
    CandidateEligibility, CandidateInputTokens, ContextFacts, ContextIneligibleReason, ContextVerdict,
    FleetCandidate, candidate_context_verdict, context_admission_filter, record_input_token_fact,
    request_output_budget,
};
use switchyard_protocol::{ModelId, Request};

/// Location of the live deployment config. Overridable for a relocated tree.
const LIVE_CONFIG: &str = match option_env!("SWITCHYARD_LIVE_CONFIG") {
    Some(path) => path,
    None => "/home/vincent/.local/lib/localclaw-switchyard/routes.toml",
};

fn model(id: &str) -> ModelId {
    ModelId::from(id.to_string())
}

fn candidate(target: &str, capacity: Option<u64>) -> FleetCandidate {
    FleetCandidate {
        target: model(target),
        tool_calling: true,
        reasoning: true,
        supports_vision: true,
        preference_rank: 1,
        usable_context_tokens: capacity,
    }
}

/// Facts from an AUTHORITATIVE producer: the admission rule is active and
/// fail-closed. Every pre-existing assertion in this file is a producer-active
/// case, so this helper carries the activation bit explicitly.
fn facts(entries: &[(&str, u64)]) -> ContextFacts {
    let mut map = CandidateInputTokens::new();
    for (target, count) in entries {
        record_input_token_fact(&mut map, &model(target), *count);
    }
    ContextFacts::Enabled(map)
}

/// An active producer that published NO fact (it ran, but measured nothing).
/// Distinct from [`ContextFacts::Disabled`], which means no producer exists.
fn enabled_empty() -> ContextFacts {
    ContextFacts::Enabled(CandidateInputTokens::new())
}

fn eligible(candidates: &[FleetCandidate]) -> Vec<CandidateEligibility<'_>> {
    candidates
        .iter()
        .map(|c| CandidateEligibility::Eligible(c))
        .collect()
}

// ---------------------------------------------------------------------------
// The admission rule: input + max_output <= capacity, checked, fail-closed.
// ---------------------------------------------------------------------------

#[test]
fn an_undeclared_capacity_is_unmanaged_and_always_admitted() {
    let c = candidate("m/a", None);
    assert_eq!(
        candidate_context_verdict(&c, &enabled_empty(), None),
        ContextVerdict::Unmanaged,
        "a candidate with no declared capacity asserts nothing to enforce"
    );
    // Even a wildly oversized request is admitted: there is no capacity to
    // exceed, and inventing one is exactly what must not happen.
    assert_eq!(
        candidate_context_verdict(&c, &facts(&[("m/a", u64::MAX)]), Some(u64::MAX)),
        ContextVerdict::Unmanaged
    );
}

#[test]
fn a_declared_capacity_with_no_fact_is_excluded() {
    let c = candidate("m/a", Some(1000));
    assert_eq!(
        candidate_context_verdict(&c, &enabled_empty(), Some(100)),
        ContextVerdict::Ineligible(ContextIneligibleReason::NoInputTokenFact),
        "an unknown input count cannot be treated as zero or as fitting"
    );
    // A fact for a DIFFERENT target does not count.
    assert_eq!(
        candidate_context_verdict(&c, &facts(&[("m/other", 10)]), Some(100)),
        ContextVerdict::Ineligible(ContextIneligibleReason::NoInputTokenFact),
        "facts are per-candidate; another target's count is not this one's"
    );
}

#[test]
fn a_declared_capacity_with_a_fact_but_no_output_budget_is_excluded() {
    // A fit cannot be proven without knowing how much must be generated, so
    // production excludes rather than assuming zero output.
    let c = candidate("m/a", Some(1000));
    assert_eq!(
        candidate_context_verdict(&c, &facts(&[("m/a", 100)]), None),
        ContextVerdict::Ineligible(ContextIneligibleReason::NoOutputBudget)
    );
}

#[test]
fn the_boundary_is_inclusive_at_exactly_the_capacity() {
    let c = candidate("m/a", Some(1000));
    assert_eq!(
        candidate_context_verdict(&c, &facts(&[("m/a", 900)]), Some(100)),
        ContextVerdict::Fits,
        "input 900 + output 100 == capacity 1000 must be admitted"
    );
    assert_eq!(
        candidate_context_verdict(&c, &facts(&[("m/a", 901)]), Some(100)),
        ContextVerdict::Ineligible(ContextIneligibleReason::DoesNotFit),
        "one token over the capacity must be excluded"
    );
}

#[test]
fn overflow_is_excluded_rather_than_wrapping_into_a_fit() {
    // The classic unchecked-add bug: u64::MAX + 1 wraps to 0, which would
    // compare <= capacity and admit an enormous request.
    let c = candidate("m/a", Some(1000));
    assert_eq!(
        candidate_context_verdict(&c, &facts(&[("m/a", u64::MAX)]), Some(1)),
        ContextVerdict::Ineligible(ContextIneligibleReason::Overflow),
        "an overflowing sum must fail closed, not wrap into a fit"
    );
    // The same overflow must not be conflated with an ordinary non-fit: both
    // exclude, but the diagnosis differs.
    assert_ne!(
        candidate_context_verdict(&c, &facts(&[("m/a", u64::MAX)]), Some(1)),
        candidate_context_verdict(&c, &facts(&[("m/a", 1000)]), Some(1))
    );
}

#[test]
fn a_zero_fact_is_a_real_fact_not_an_absence() {
    let c = candidate("m/a", Some(1000));
    assert_eq!(
        candidate_context_verdict(&c, &facts(&[("m/a", 0)]), Some(1000)),
        ContextVerdict::Fits,
        "a counted zero is a fact and must be honoured exactly"
    );
}

// ---------------------------------------------------------------------------
// Filter behaviour.
// ---------------------------------------------------------------------------

#[test]
fn the_filter_marks_rather_than_deletes_and_preserves_input_order() {
    let candidates = vec![
        candidate("m/c", Some(1000)),
        candidate("m/a", Some(1000)),
        candidate("m/b", Some(1000)),
    ];
    let marked = context_admission_filter(&eligible(&candidates), &facts(&[("m/b", 10)]), Some(10));
    assert_eq!(marked.len(), 3, "every input is marked; nothing is dropped");
    assert_eq!(
        marked
            .iter()
            .map(|e| e.candidate.target.clone())
            .collect::<Vec<_>>(),
        vec![model("m/c"), model("m/a"), model("m/b")],
        "input order is preserved, never sorted"
    );
    assert!(!marked[0].is_admitted());
    assert!(!marked[1].is_admitted());
    assert!(marked[2].is_admitted());
}

#[test]
fn an_unmanaged_candidate_survives_alongside_an_excluded_bounded_one() {
    let candidates = vec![
        candidate("m/bounded", Some(10)),
        candidate("m/unmanaged", None),
    ];
    let marked = context_admission_filter(
        &eligible(&candidates),
        &facts(&[("m/bounded", 500)]),
        Some(500),
    );
    assert!(
        !marked[0].is_admitted(),
        "the bounded candidate does not fit"
    );
    assert!(
        marked[1].is_admitted(),
        "the unmanaged candidate is unaffected"
    );
}

#[test]
fn the_filter_never_reorders_by_preference_rank_even_when_ranks_differ() {
    // A rank sort is a stable no-op when every rank is equal, so the
    // "preserves input order" property needs a case where declared order and
    // rank order genuinely disagree. Here rank 9 is declared FIRST, so any
    // sort by rank moves it and the assertion below can observe it.
    let mut first_declared_high_rank = candidate("m/aaa", Some(10_000));
    first_declared_high_rank.preference_rank = 9;
    let mut second_declared_low_rank = candidate("m/zzz", Some(10_000));
    second_declared_low_rank.preference_rank = 1;
    let mut third_declared_mid_rank = candidate("m/mmm", Some(10_000));
    third_declared_mid_rank.preference_rank = 5;

    let declared = vec![
        first_declared_high_rank,
        second_declared_low_rank,
        third_declared_mid_rank,
    ];
    let marked = context_admission_filter(
        &eligible(&declared),
        &facts(&[("m/aaa", 10), ("m/zzz", 10), ("m/mmm", 10)]),
        Some(10),
    );
    assert_eq!(
        marked
            .iter()
            .map(|e| e.candidate.target.clone())
            .collect::<Vec<_>>(),
        vec![model("m/aaa"), model("m/zzz"), model("m/mmm")],
        "declared order is preserved even though sorting by rank would reorder it"
    );
}

#[test]
fn context_admission_ignores_preference_rank() {
    // The lowest-preference candidate fits while the highest does not; rank
    // must not rescue or suppress either.
    let mut high = candidate("m/high", Some(10));
    high.preference_rank = 1;
    let mut low = candidate("m/low", Some(10_000));
    low.preference_rank = 9;
    let pair = [high, low];
    let marked = context_admission_filter(
        &eligible(&pair),
        &facts(&[("m/high", 5000), ("m/low", 5000)]),
        Some(500),
    );
    assert!(
        !marked[0].is_admitted(),
        "rank 1 does not admit an oversized request"
    );
    assert!(
        marked[1].is_admitted(),
        "rank 9 does not exclude a fitting request"
    );
}

#[test]
fn a_changed_fact_changes_the_next_decision_without_touching_the_candidate() {
    let c = candidate("m/a", Some(1000));
    let before = c.clone();
    assert!(candidate_context_verdict(&c, &facts(&[("m/a", 100)]), Some(100)).is_admitted());
    assert!(!candidate_context_verdict(&c, &facts(&[("m/a", 5000)]), Some(100)).is_admitted());
    let after = &c;
    assert_eq!(before.target, after.target);
    assert_eq!(before.usable_context_tokens, after.usable_context_tokens);
    assert_eq!(before.preference_rank, after.preference_rank);
}

// ---------------------------------------------------------------------------
// The producer seam.
// ---------------------------------------------------------------------------

#[test]
fn a_recorded_fact_is_readable_and_absent_targets_stay_absent() {
    let mut map = CandidateInputTokens::new();
    assert!(map.is_empty());
    record_input_token_fact(&mut map, &model("m/a"), 4242);
    assert_eq!(map.get(&model("m/a")), Some(&4242));
    assert_eq!(
        map.get(&model("m/b")),
        None,
        "an unrecorded target has no fact, not a zero"
    );
    // Re-recording replaces, so a newer generation supersedes rather than
    // accumulating.
    record_input_token_fact(&mut map, &model("m/a"), 7);
    assert_eq!(map.get(&model("m/a")), Some(&7));
}

#[test]
fn facts_are_keyed_per_target_so_alias_targets_can_differ() {
    // Two distinct target names resolving to distinct model ids may legitimately
    // carry different counts; the map is keyed by resolved model id.
    let map = facts(&[("m/one", 10), ("m/two", 20)]);
    let published = map
        .facts()
        .expect("an active producer carries the published fact map");
    assert_eq!(published.len(), 2);
    assert_eq!(published.get(&model("m/one")), Some(&10));
    assert_eq!(published.get(&model("m/two")), Some(&20));
    assert!(map.is_enabled());
}

#[test]
fn the_output_budget_is_read_from_the_request_and_absent_means_none() {
    let mut request = Request::default();
    assert_eq!(
        request_output_budget(&request),
        None,
        "no declared budget reads as absent"
    );
    request.llm_request.output.max_output_tokens = Some(256);
    assert_eq!(request_output_budget(&request), Some(256));
    request.llm_request.output.max_output_tokens = Some(0);
    assert_eq!(
        request_output_budget(&request),
        Some(0),
        "an explicit zero budget is a real budget, not an absence"
    );
}

// ---------------------------------------------------------------------------
// Live production configuration.
// ---------------------------------------------------------------------------

fn live_document() -> toml::Table {
    let text = std::fs::read_to_string(LIVE_CONFIG)
        .unwrap_or_else(|error| panic!("live config {LIVE_CONFIG} must be readable: {error}"));
    toml::from_str(&text).unwrap_or_else(|error| panic!("live config must be valid TOML: {error}"))
}

#[test]
fn the_live_file_declares_no_candidate_context_policy() {
    // The `context_policy` surface is owner-supported but has ZERO live use, so
    // it is not carried: admission is driven by the flat
    // `usable_context_tokens` field the live file actually uses.
    let document = live_document();
    let text = serde_json::to_string(&document).expect("serialisable");
    assert_eq!(
        text.matches("context_policy").count(),
        0,
        "the live file carries no candidate context_policy block"
    );
    assert_eq!(
        text.matches("input_token_source").count(),
        0,
        "the live file declares no input_token_source"
    );
    assert_eq!(
        text.matches("min_context_tokens").count(),
        0,
        "the live file declares no route min_context_tokens"
    );
}

#[test]
fn every_live_capacity_is_positive_and_would_be_admissible_when_absent_from_policy() {
    let document = live_document();
    let routes = document
        .get("routes")
        .and_then(toml::Value::as_table)
        .cloned()
        .expect("live config must have a [routes] table");
    let mut capacities: BTreeSet<u64> = BTreeSet::new();
    let mut declared = 0usize;
    let mut absent = 0usize;
    for value in routes.values() {
        let table = value.as_table().expect("route table");
        if table.get("type").and_then(toml::Value::as_str) != Some("fleet_router") {
            continue;
        }
        for entry in table
            .get("candidates")
            .and_then(toml::Value::as_array)
            .unwrap_or(&Vec::new())
        {
            let Some(entry) = entry.as_table() else {
                continue;
            };
            match entry
                .get("usable_context_tokens")
                .and_then(|v| v.as_integer())
            {
                Some(raw) => {
                    assert!(raw > 0, "a declared capacity must be positive, got {raw}");
                    capacities.insert(raw as u64);
                    declared += 1;
                }
                None => absent += 1,
            }
        }
    }
    assert!(declared > 0, "the live file must declare capacities");
    assert_eq!(
        declared + absent,
        46,
        "the census must cover every live candidate"
    );
    assert!(
        absent > 0,
        "the live file leaves some capacities undeclared"
    );
    assert!(
        capacities.iter().all(|c| *c > 0),
        "no live capacity may be zero, which production rejects at load"
    );
}

#[test]
fn a_zero_capacity_is_rejected_at_load_time() {
    // Production refuses a zero configured capacity; a zero-capacity bounded
    // candidate is an invalid admission contract, not an unlimited one.
    let doc = r#"
schema_version = 1

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
  { target = "t1", tool_calling = false, reasoning = false, supports_vision = false, usable_context_tokens = 0 },
]
"#;
    match switchyard_runner::Runner::from_toml(doc) {
        Ok(_) => panic!("a zero usable_context_tokens must be refused at load"),
        Err(error) => {
            let message = error.to_string();
            assert!(
                message.contains("usable_context_tokens"),
                "the refusal must name the offending field, got: {message}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// ACTIVATION-STATE CONTRACT (the 0.3.0 cutover defect).
//
// The row-40 contract (candidates declare `usable_context_tokens`) and the
// row-40 PRODUCER (per-candidate input-token facts) are separate. With no
// producer wired, context admission must NOT participate; with one
// authoritative, the rule stays fail-closed. Activation is explicit and is
// NEVER inferred from the map's contents.
// ---------------------------------------------------------------------------

#[test]
fn a_disabled_producer_leaves_a_bounded_candidate_admitted() {
    // THE REGRESSION: this state zeroed every bounded production route and
    // returned "no immediately-eligible candidate".
    let c = candidate("m/a", Some(1000));
    let verdict = candidate_context_verdict(&c, &ContextFacts::Disabled, Some(100));
    assert_eq!(
        verdict,
        ContextVerdict::Unmanaged,
        "with no producer wired, admission does not participate and a bounded \
         candidate must not be rejected for a fact nobody could produce"
    );
    assert!(verdict.is_admitted());
}

#[test]
fn a_disabled_producer_admits_through_the_real_filter() {
    let c = candidate("m/a", Some(1000));
    let admitted = context_admission_filter(
        &eligible(std::slice::from_ref(&c)),
        &ContextFacts::Disabled,
        Some(100),
    );
    assert_eq!(admitted.len(), 1, "the filter must not drop the candidate");
    assert!(admitted[0].is_admitted());
}

#[test]
fn an_enabled_producer_still_fails_closed_on_a_missing_fact() {
    let c = candidate("m/a", Some(1000));
    let verdict = candidate_context_verdict(&c, &enabled_empty(), Some(100));
    assert_eq!(
        verdict,
        ContextVerdict::Ineligible(ContextIneligibleReason::NoInputTokenFact),
        "an enabled producer that published nothing must fail closed, not open"
    );
    assert!(!verdict.is_admitted());
}

#[test]
fn an_empty_map_is_not_an_activation_signal() {
    // The two states carry byte-identical maps and must still differ.
    let c = candidate("m/a", Some(1000));
    let disabled = candidate_context_verdict(&c, &ContextFacts::Disabled, Some(100));
    let enabled = candidate_context_verdict(&c, &enabled_empty(), Some(100));
    assert_eq!(disabled, ContextVerdict::Unmanaged);
    assert_eq!(
        enabled,
        ContextVerdict::Ineligible(ContextIneligibleReason::NoInputTokenFact)
    );
    assert_ne!(
        disabled, enabled,
        "an empty fact map must never be read as the activation bit"
    );
}

#[test]
fn enabled_admits_below_and_at_the_bound_and_excludes_one_over() {
    let c = candidate("m/a", Some(1000));
    assert_eq!(
        candidate_context_verdict(&c, &facts(&[("m/a", 500)]), Some(100)),
        ContextVerdict::Fits
    );
    assert_eq!(
        candidate_context_verdict(&c, &facts(&[("m/a", 900)]), Some(100)),
        ContextVerdict::Fits,
        "exact equality with the bound is a fit, not an overflow"
    );
    assert_eq!(
        candidate_context_verdict(&c, &facts(&[("m/a", 901)]), Some(100)),
        ContextVerdict::Ineligible(ContextIneligibleReason::DoesNotFit),
        "one token over the bound must be excluded"
    );
}

#[test]
fn a_collected_failure_is_excluded_not_treated_as_absent_evidence() {
    // The producer ran; it simply published no count for this candidate.
    let c = candidate("m/a", Some(1000));
    let verdict = candidate_context_verdict(
        &c,
        &facts(&[("m/other", 42)]),
        Some(100),
    );
    assert_eq!(
        verdict,
        ContextVerdict::Ineligible(ContextIneligibleReason::NoInputTokenFact),
        "another target's fact is not this candidate's fact"
    );
}

#[test]
fn an_unbounded_candidate_is_unaffected_in_both_states() {
    let c = candidate("m/a", None);
    assert_eq!(
        candidate_context_verdict(&c, &ContextFacts::Disabled, Some(u64::MAX)),
        ContextVerdict::Unmanaged
    );
    assert_eq!(
        candidate_context_verdict(&c, &enabled_empty(), Some(u64::MAX)),
        ContextVerdict::Unmanaged
    );
    assert_eq!(
        candidate_context_verdict(&c, &facts(&[("m/a", u64::MAX)]), Some(u64::MAX)),
        ContextVerdict::Unmanaged,
        "no declared capacity asserts nothing to enforce in any producer state"
    );
}

#[test]
fn the_default_activation_state_is_disabled() {
    assert_eq!(ContextFacts::default(), ContextFacts::Disabled);
    assert!(!ContextFacts::default().is_enabled());
    assert!(ContextFacts::default().facts().is_none());
}


/// R40-B DORMANCY, asserted positively over the real production source.
///
/// The row-40 CONTRACT may exist (candidates declare a capacity) while the
/// row-40 PRODUCER is absent. This asserts the production decision path
/// activates context production explicitly as Disabled, and that nothing at
/// runtime constructs an authoritative producer - so activating the producer
/// remains separate, deliberate work.
#[test]
fn the_production_route_passes_context_production_disabled() {
    let source = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../libsy/src/algorithms/fleet_router.rs"),
    )
    .expect("the routing source must be readable");
    assert!(
        source.contains("self.decide(&request, &ContextFacts::Disabled)"),
        "the production route must decide with context production explicitly Disabled"
    );
    // No runtime construction of an authoritative producer. The production
    // decision path is the `Algorithm::route` body; nothing between the file's
    // start and the `#[cfg(test)]` module may build an Enabled producer.
    // Everything before the test module is runtime code. The ONLY places a
    // runtime `ContextFacts::Enabled(` may appear are the type's own match arms
    // and its From impl, which both live between the `ContextFacts` enum
    // declaration and the context-verdict section.
    let (runtime, test_module) = source
        .split_once("#[cfg(test)]")
        .expect("the routing source has a test module");
    let type_start = runtime
        .find("pub enum ContextFacts")
        .expect("the ContextFacts type must exist");
    let type_end = runtime
        .find("/// The context standing of one candidate")
        .expect("the context-verdict section must follow the type");
    assert!(type_start < type_end, "the type must precede the verdict section");
    let (before_type, type_block, after_type) =
        (&runtime[..type_start], &runtime[type_start..type_end], &runtime[type_end..]);
    assert!(
        !before_type.contains("ContextFacts::Enabled(")
            && !after_type.contains("ContextFacts::Enabled("),
        "outside the ContextFacts type definition, no runtime code may construct \
         an authoritative producer"
    );
    // Every `ContextFacts::Enabled(` line inside the type block must be one of
    // the three sanctioned ones: the match arm that reads the payload, and the
    // From impl that adopts a map. This is an ALLOWLIST rather than an
    // exclusion, so runtime code smuggled into the block is caught by name
    // rather than merely escaping a whole-section check.
    let sanctioned = [
        "ContextFacts::Enabled(facts) => Some(facts),",
        "matches!(self, ContextFacts::Enabled(_))",
        "ContextFacts::Enabled(facts)",
    ];
    for line in type_block.lines() {
        if line.contains("ContextFacts::Enabled(") {
            let body = line.trim();
            assert!(
                sanctioned.contains(&body),
                "unexpected Enabled construction inside the ContextFacts block: {body:?}"
            );
        }
    }
    assert!(
        runtime.contains("impl From<CandidateInputTokens> for ContextFacts"),
        "the type must keep an explicit adoption path for callers holding a map"
    );
    // And the production decision really does say Disabled, somewhere in the
    // runtime section (it precedes the type definition in this file).
    assert!(
        runtime.contains("self.decide(&request, &ContextFacts::Disabled)?"),
        "the production route must decide with context production Disabled"
    );
    // The active-producer fail-closed case IS exercised, inside the test module.
    assert!(
        test_module.contains("ContextFacts::Enabled(CandidateInputTokens::new())"),
        "the producer-active fail-closed case must remain covered by tests"
    );
    // The producer seam still exists and is still uncalled outside tests.
    assert!(
        source.contains("pub fn record_input_token_fact"),
        "the row-40 producer seam must remain available for separate activation"
    );
}
