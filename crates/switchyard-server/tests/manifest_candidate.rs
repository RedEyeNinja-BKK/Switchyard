// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration proof for the typed-service candidate, loaded from the REAL candidate bytes.
//!
//! The fixture `fixtures/routes.candidate.p10.toml` is the actual proposed candidate manifest:
//! the deployed P10 manifest (sha256 e887392c...) with only the approved typed deltas applied
//! (four NVIDIA-free privacy flags, smart-rerank, smart-gemini-embedding, and the
//! smart-gemini-decision leaf used inside smart-decision). It is NOT an inline fixture and NOT
//! the stale redacted projection. These tests prove the complete candidate bytes load through
//! the real loader, the legacy tier1-decision ordered contract is preserved, and the typed
//! surfaces resolve with the required privacy declarations.
//!
//! No live provider is exercised. The gate this supports is "the candidate is loadable and the
//! typed surfaces resolve", which is what a host dry-run needs before activation.

use switchyard_runner::Runner;

/// The real candidate manifest, derived from deployed bytes + approved deltas.
const CANDIDATE: &str = include_str!("fixtures/routes.candidate.p10.toml");

fn load() -> Runner {
    // The loader reads each api_key_env at load time, exactly as the host does through its
    // EnvironmentFiles. A test process has none, so set throwaway values. Never dispatched.
    for name in [
        "SWITCHYARD_COMFYNINJA_API_KEY",
        "SWITCHYARD_DEEPSEEK_API_KEY",
        "SWITCHYARD_GEMINI_API_KEY",
        "SWITCHYARD_GEMINI_STEPHEN_API_KEY",
        "SWITCHYARD_HTPC_API_KEY",
        "SWITCHYARD_JEV_STYLE_API_KEY",
        "SWITCHYARD_LAYA_API_KEY",
        "SWITCHYARD_OPENROUTER_API_KEY",
        "SWITCHYARD_STEPHEN_OPENROUTER_API_KEY",
        "SWITCHYARD_THAILLM_API_KEY",
        "SY_OPENAI_OAUTH_TOKEN",
    ] {
        // SAFETY: test-only setup; the value is a throwaway and no request is sent.
        unsafe { std::env::set_var(name, "test-placeholder") };
    }
    Runner::from_toml(CANDIDATE).expect("the real candidate manifest must load")
}

#[test]
fn the_candidate_loads_and_resolves_the_typed_services() {
    let runner = load();
    for id in [
        "smart-rerank",
        "smart-gemini-embedding",
        "gemini-embedding-2",
    ] {
        assert!(
            runner.capability(id).is_some(),
            "capability {id} must resolve"
        );
    }
}

#[test]
fn the_legacy_tier1_decision_order_is_preserved() {
    // The deployed compatibility contract: tier1-decision composes free then local, and the
    // Google-first replacement is NOT authorized. The candidate must not touch it.
    let runner = load();
    let names: Vec<&str> = runner
        .decision_lane_children("tier1-decision")
        .expect("tier1-decision resolves")
        .iter()
        .map(|c| c.as_str())
        .collect();
    assert_eq!(
        names,
        vec!["smartfree-decision", "smartlocal-decision"],
        "tier1-decision must stay ordered free->local; the Google-first change was not authorized"
    );
}

#[test]
fn the_smart_decision_composition_gains_the_gemini_leaf() {
    // smart-decision is the capability_first selector; the new leaf joins it without removing
    // the existing local/free/google children.
    let runner = load();
    let names: Vec<&str> = runner
        .decision_lane_children("smart-decision")
        .expect("smart-decision resolves")
        .iter()
        .map(|c| c.as_str())
        .collect();
    assert!(
        names.contains(&"smart-gemini-decision"),
        "smart-decision must include the Gemini leaf: {names:?}"
    );
    assert!(names.contains(&"smartlocal-decision"), "local retained");
    assert!(names.contains(&"smartfree-decision"), "free retained");
    // google-decision stays a standalone compatibility alias; it is NOT composed into
    // smart-decision, because it shares the same physical Google target as the Gemini leaf and
    // would double the selection weight / attempt the same canonical target twice.
    assert!(
        !names.contains(&"google-decision"),
        "google-decision must not be composed: {names:?}"
    );
    assert_eq!(
        names.len(),
        3,
        "exactly the three intended families: {names:?}"
    );
}

#[test]
fn the_rerank_selector_is_local_first_with_a_free_fallback() {
    use switchyard_runner::capability::EndpointPrivacy;
    let runner = load();
    let selector = runner
        .capability("smart-rerank")
        .expect("selector resolves");
    assert_eq!(
        selector.targets.len(),
        2,
        "one declared target plus one candidate"
    );
    assert!(
        selector.targets.iter().any(|t| t.privacy.is_none()),
        "the local-first engine declares no restriction"
    );
    assert!(
        selector
            .targets
            .iter()
            .any(|t| t.privacy == Some(EndpointPrivacy::ThirdPartyFree)),
        "the free candidate is declared third-party-free"
    );
}

#[test]
fn the_gemini_embedding_alias_shares_one_space() {
    let runner = load();
    let canonical = runner.capability("gemini-embedding-2").expect("canonical");
    let alias = runner.capability("smart-gemini-embedding").expect("alias");
    assert_eq!(alias.contract, canonical.contract, "same value space");
    assert_eq!(alias.dimensions, canonical.dimensions);
    assert_eq!(
        alias.targets[0].client, canonical.targets[0].client,
        "one client: the alias is not a second endpoint"
    );
    assert_eq!(alias.targets.len(), 1, "no cross-space fallback");
}

#[test]
fn every_nvidia_free_capability_is_declared_third_party_free() {
    use switchyard_runner::capability::EndpointPrivacy;
    let runner = load();
    for (capability, free_client) in [
        ("smartfree-embedding", "embedding-free-nemotron"),
        ("smartfree-rerank", "reranker-free"),
    ] {
        let resolved = runner.capability(capability).expect("resolves");
        let target = resolved
            .targets
            .iter()
            .find(|t| t.client == free_client)
            .expect("free target present");
        assert_eq!(
            target.privacy,
            Some(EndpointPrivacy::ThirdPartyFree),
            "{free_client} must be declared third_party_free"
        );
    }
    for (capability, local_client) in [
        ("smartlocal-embedding", "embedding-local"),
        ("smartlocal-rerank", "reranker-local"),
    ] {
        let resolved = runner.capability(capability).expect("resolves");
        let target = resolved
            .targets
            .iter()
            .find(|t| t.client == local_client)
            .expect("local target present");
        assert_eq!(target.privacy, None, "{local_client} is internal");
    }
}
