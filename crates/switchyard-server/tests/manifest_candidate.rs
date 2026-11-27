// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration proof for the typed-service candidate additions.
//!
//! The production redacted manifest is a generated review projection that has drifted from
//! the loader (it still names the retired `smart-bounded` family and advertises a context no
//! child serves), so it is not loadable as published -- that drift is a separate C7
//! reconciliation item, not something this candidate silently edits. What these tests prove
//! is that the additions themselves are loadable and resolve through the real loader: the
//! rerank selector, the Gemini embedding alias over one space, the Gemini decision
//! composition, and the third-party-free privacy declarations on every NVIDIA free client.
//!
//! No live provider is exercised. The gate this supports is "the candidate is loadable and
//! the typed surfaces resolve", which is what a host dry-run needs before activation.

use switchyard_runner::Runner;

/// A candidate carrying exactly the typed-service additions, on the same clients the
/// production manifest declares. The internal engines need no credential; the free engines
/// and the Google lane name env vars the loader reads at load time.
const CANDIDATE: &str = r#"
schema_version = 1

[llm_clients.responses]
format = "openai_responses"
base_url = "https://completion.test/v1"

[targets.only]
id = "vendor/only"
llm_client = "responses"

[routes.only]
id = "probe/only"
type = "passthrough"
target = "only"

[llm_clients.google_gemini]
format = "openai_chat"
base_url = "https://generativelanguage.googleapis.com/v1beta/openai"
api_key_env = "SWITCHYARD_GEMINI_API_KEY"
max_retries = 0

[targets.direct-google-gemini-decision-1]
id = "gemini-decision-1"
llm_client = "google_gemini"
decision_transport = "google_decisions"
decision_path = "/v1beta/decisions"
decision_base_url = "https://generativelanguage.googleapis.com"
decision_api_key_env = "SWITCHYARD_GEMINI_API_KEY"
free_tier = true

[targets.direct-google-gemini-decision-1.supported_types]
noul = true
choice = true
score = true

[targets.direct-google-gemini-decision-1.state_forms]
plain_string = false
structured = true

[decision_lanes.smart-gemini-decision]
targets = ["direct-google-gemini-decision-1"]

[targets.direct-openrouter-span-01-lite]
id = "respan/span-01-lite"
llm_client = "responses"
decision_transport = "open_router_alpha"
decision_path = "/api/alpha/decisions"
decision_base_url = "https://openrouter.ai"
decision_api_key_env = "SWITCHYARD_OPENROUTER_API_KEY"
supported_types = { noul = true, choice = false, score = false }
state_forms = { plain_string = true, structured = false }

[decision_lanes.smartfree-decision]
targets = ["direct-openrouter-span-01-lite"]

# tier1-decision COMPOSES the two SMART policy lanes above, plus the native Gemini
# lane first so a caller that names tier1-decision reaches Google before the free pools.
[decision_lane_composition.tier1-decision]
children = ["smart-gemini-decision", "smartfree-decision"]

[capability_clients.embedding-local]
format = "openai_embeddings"
base_url = "https://reninja.tailnet-example.ts.net:8448"
model = "Qwen3-Embedding-0.6B"

[capability_clients.embedding-free-nemotron]
format = "openai_embeddings"
base_url = "https://openrouter.ai/api"
model = "nvidia/nemotron-3.5-embed-8b:free"
api_key_env = "SWITCHYARD_OPENROUTER_API_KEY"
timeout_seconds = 120
privacy = "third_party_free"

[capability_clients.reranker-local]
format = "cohere_jina_rerank"
base_url = "https://reninja.tailnet-example.ts.net:8449"
model = "qwen3-reranker-0.6b-q8_0"
timeout_seconds = 120

[capability_clients.reranker-free]
format = "cohere_jina_rerank"
base_url = "https://openrouter.ai/api"
model = "nvidia/nemotron-3.5-rerank-8b:free"
api_key_env = "SWITCHYARD_OPENROUTER_API_KEY"
timeout_seconds = 120
privacy = "third_party_free"

[capability_clients.embedding-google2]
format = "google_embed_content_v2"
base_url = "https://generativelanguage.googleapis.com"
model = "gemini-embedding-2"
api_key_env = "SWITCHYARD_GEMINI_API_KEY"
timeout_seconds = 120

[capabilities.smartlocal-embedding]
id = "smartlocal-embedding"
target = "embedding-local"
contract = "local-embedding-space:v1"
dimensions = 1024
normalization = "L2"
max_batch = 64

[capabilities.smartfree-embedding]
id = "smartfree-embedding"
target = "embedding-free-nemotron"
contract = "openrouter-embedding-space:nemotron3.5-embed8b-v1"
dimensions = 4096
normalization = "L2"
max_batch = 64

[capabilities.smartlocal-rerank]
id = "smartlocal-rerank"
target = "reranker-local"
max_candidates = 64
top_n = 64
max_doc_chars = 1024
max_query_chars = 512

[capabilities.smartfree-rerank]
id = "smartfree-rerank"
target = "reranker-free"
max_candidates = 64
top_n = 64
max_doc_chars = 1024
max_query_chars = 512

[capabilities.smart-rerank]
id = "smart-rerank"
target = "reranker-local"
candidates = ["reranker-free"]
max_candidates = 64
top_n = 64
max_doc_chars = 1024
max_query_chars = 512

[capabilities.gemini-embedding-2]
id = "gemini-embedding-2"
target = "embedding-google2"
contract = "google-embedding-space:gemini-embedding-2-v1"
dimensions = 3072
normalization = "L2"
max_batch = 8
max_input_tokens = 8192

[capabilities.smart-gemini-embedding]
id = "smart-gemini-embedding"
target = "embedding-google2"
contract = "google-embedding-space:gemini-embedding-2-v1"
dimensions = 3072
normalization = "L2"
max_batch = 8
max_input_tokens = 8192
"#;

fn load() -> Runner {
    // The loader reads each api_key_env at load time, exactly as the host does through its
    // EnvironmentFiles. A test process has none, so set a throwaway value. Never dispatched.
    for name in [
        "SWITCHYARD_GEMINI_API_KEY",
        "SWITCHYARD_OPENROUTER_API_KEY",
    ] {
        // SAFETY: test-only setup; the value is a throwaway and no request is sent.
        unsafe { std::env::set_var(name, "test-placeholder") };
    }
    Runner::from_toml(CANDIDATE).expect("the typed-service candidate must load")
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
fn the_rerank_selector_is_local_first_with_a_free_fallback() {
    use switchyard_runner::capability::EndpointPrivacy;
    let runner = load();
    let selector = runner.capability("smart-rerank").expect("selector resolves");
    assert_eq!(selector.targets.len(), 2, "one declared target plus one candidate");
    assert!(
        selector
            .targets
            .iter()
            .any(|target| target.privacy.is_none()),
        "the local-first engine declares no restriction"
    );
    assert!(
        selector.targets.iter().any(|target| {
            target.privacy == Some(EndpointPrivacy::ThirdPartyFree)
        }),
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
            .find(|target| target.client == free_client)
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
            .find(|target| target.client == local_client)
            .expect("local target present");
        assert_eq!(target.privacy, None, "{local_client} is internal");
    }
}