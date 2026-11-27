// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Cost-containment proof: V4 Pro is unreachable through `smart-route`.
//!
//! Owner directive 2026-11-27 removed `direct-deepseek-deepseek-v4-pro` from the
//! smart-route ladder. This test loads the REAL corrected manifest bytes through the
//! real loader and inspects every surface a dispatcher draws from:
//!
//! * `models().all_models()` — every candidate the route may dispatch;
//! * `terminal_models()` — the exhaustion-only tiers;
//!
//! and then re-checks after the cheapest member of the DeepSeek group (Flash) is
//! removed, which is the state the directive names explicitly. If V4 Pro were still
//! reachable it would appear in one of those sets, and removing Flash would be the
//! moment it surfaced.
//!
//! No provider is contacted.

use switchyard_runner::Runner;

const CANDIDATE: &str = include_str!("fixtures/routes.cost-containment-20261127.toml");

fn load() -> Runner {
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
        // SAFETY: test-only setup; throwaway value, no request is sent.
        unsafe { std::env::set_var(name, "test-placeholder") };
    }
    Runner::from_toml(CANDIDATE).expect("the corrected manifest must load")
}

fn smart_route_models() -> Vec<String> {
    let runner = load();
    let route = runner.route("smart-route").expect("smart-route resolves");
    route
        .models()
        .all_models()
        .into_iter()
        .map(|m| m.as_str().to_string())
        .collect()
}

#[test]
fn smart_route_cannot_dispatch_v4_pro() {
    let models = smart_route_models();
    assert!(
        !models.iter().any(|m| m == "deepseek-v4-pro"),
        "V4 Pro is still dispatchable through smart-route: {models:?}"
    );
    assert_eq!(
        models.iter().filter(|m| *m == "deepseek-flash").count(),
        1,
        "the DeepSeek group must contribute exactly one wire model: {models:?}"
    );
    // The ladder is the four declared groups and nothing else. `models()` reports the
    // provider wire id each target would be called with, not the target name.
    let expected = [
        "gemini-4-hydrogen",
        "gemini-3.8-flash",
        "gemini-3.6-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
        "baidu/ernie-5.5-flash-preview:free",
        "inclusionai/ling-3.2-flash:free",
        "nvidia/nemotron-3.5-super-vl:free",
        "deepseek-flash",
        "tencent/hy4-flash",
        "z-ai/glm-5.3-flash",
    ];
    let mut got = models.clone();
    got.sort();
    let mut want: Vec<String> = expected.iter().map(|s| s.to_string()).collect();
    want.sort();
    assert_eq!(got, want, "smart-route dispatch set changed unexpectedly");
}

#[test]
fn v4_pro_is_not_a_terminal_tier_of_smart_route() {
    let runner = load();
    let route = runner.route("smart-route").expect("smart-route resolves");
    let terminal: Vec<String> = route
        .terminal_models()
        .into_iter()
        .map(|m| m.as_str().to_string())
        .collect();
    assert!(
        !terminal.iter().any(|m| m == "deepseek-v4-pro"),
        "V4 Pro is reachable as a terminal tier: {terminal:?}"
    );
    assert!(terminal.is_empty(), "smart-route declares terminal tiers: {terminal:?}");
}

#[test]
fn flash_unavailability_does_not_promote_a_more_expensive_sibling() {
    let runner = load();
    let route = runner.route("smart-route").expect("smart-route resolves");
    // The state the directive names: DeepSeek Flash is unusable. An ineligible target is
    // removed from the set, never replaced by a sibling the ladder does not declare.
    let remaining = route.models().filtered(&|m| m.as_str() != "deepseek-flash");
    let names: Vec<String> = remaining
        .all_models()
        .into_iter()
        .map(|m| m.as_str().to_string())
        .collect();
    assert!(
        !names.iter().any(|m| m == "deepseek-v4-pro"),
        "removing Flash promoted V4 Pro: {names:?}"
    );
    // Group 3 ends up empty rather than gaining a member: the DeepSeek flash target is
    // the group's only member, so nothing else in that price band can answer.
    assert!(
        !names.iter().any(|m| m.starts_with("deepseek-")),
        "a DeepSeek target survived Flash removal: {names:?}"
    );
}

#[test]
fn the_corrected_manifest_still_loads_its_preserved_surfaces() {
    let runner = load();
    for id in [
        "direct-deepseek-deepseek-v4-pro",
        "vc-paid-core",
        "VC_Paid",
        "smart-deepseek",
        "smart-deepseek-flash",
    ] {
        assert!(runner.has_route(id), "{id} must still resolve");
    }
    // The compatibility alias and the renamed group are the same singleton.
    for alias in ["smart-deepseek", "smart-deepseek-flash"] {
        let models: Vec<String> = runner
            .route(alias)
            .expect("route resolves")
            .models()
            .all_models()
            .into_iter()
            .map(|m| m.as_str().to_string())
            .collect();
        assert_eq!(models, vec!["deepseek-flash".to_string()], "{alias} membership");
    }
}
