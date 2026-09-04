// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! P8 live-config zero-translation battery.
//!
//! The live `routes.toml` must parse UNCHANGED on the candidate build and the
//! candidate's model surface must equal the production binary's `/v1/models`
//! advertisement (route IDs + capability route IDs, no additions, no drops).
//! This is verification, not migration: no translation layer, no rewrite step.

use switchyard_runner::Runner;

static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn seeded_runner_from_live_config(source: &str) -> Runner {
    // This battery validates SCHEMA parsing only: satisfy the fail-loud
    // credential guard with placeholder values for every env-var NAME the file
    // references (names are configuration, not secrets; no real credential
    // values are read or printed here).
    let guard = ENV_LOCK.lock().unwrap();
    for name in source.lines().filter_map(|line| {
        let (key, value) = line.split_once('=')?;
        let key = key.trim();
        let wants = key == "api_key_env" || key == "auth_token_env";
        wants.then(|| value.trim().trim_matches('"').to_string())
    }) {
        if std::env::var(&name).is_err() {
            // SAFETY: see above - placeholder-only seeding while holding the
            // process-global env lock; battery tests are the only writers.
            unsafe {
                std::env::set_var(&name, "parity-battery-placeholder");
            }
        }
    }
    let runner = Runner::from_toml(source).expect("live routes.toml parses on candidate build");
    drop(guard);
    runner
}

fn candidate_ids(runner: &Runner) -> Vec<String> {
    let mut ids: Vec<String> = runner
        .models()
        .map(|model| model.id.as_str().to_string())
        .collect();
    ids.extend(
        runner
            .capabilities()
            .values()
            .map(|config| config.id.clone()),
    );
    ids.sort();
    ids.dedup();
    ids
}

// The P8 acceptance criterion: candidate parse of the live config equals the
// production advertisement exactly. A diff names the divergent IDs (fail
// loud); production being unreachable fails loudly too - this battery is a
// qualification gate, not a best-effort smoke.
#[tokio::test]
async fn live_config_parses_and_advertises_exactly_like_production() {
    const PRODUCTION_BASE_URL: &str = "http://127.0.0.1:4000";

    let source =
        std::fs::read_to_string("/home/vincent/.local/lib/localclaw-switchyard/routes.toml")
            .expect("live routes.toml readable");
    let runner = seeded_runner_from_live_config(&source);
    let candidate = candidate_ids(&runner);
    assert!(
        candidate.len() >= 58,
        "expected the full live route set (58+), got {}",
        candidate.len()
    );

    let production = reqwest::Client::new()
        .get(format!("{PRODUCTION_BASE_URL}/v1/models"))
        .timeout(std::time::Duration::from_secs(5))
        .send()
        .await
        .expect("production /v1/models reachable (qualification battery)")
        .error_for_status()
        .expect("production /v1/models 200")
        .json::<serde_json::Value>()
        .await
        .expect("production /v1/models JSON");
    let mut production: Vec<String> = production["data"]
        .as_array()
        .expect("data list")
        .iter()
        .map(|entry| entry["id"].as_str().expect("string id").to_string())
        .collect();
    production.sort();
    production.dedup();

    let missing: Vec<&String> = production
        .iter()
        .filter(|id| !candidate.contains(id))
        .collect();
    let extra: Vec<&String> = candidate
        .iter()
        .filter(|id| !production.contains(id))
        .collect();
    assert!(
        missing.is_empty() && extra.is_empty(),
        "advertisement parity broken\nproduction-only: {missing:?}\ncandidate-only: {extra:?}"
    );
}
