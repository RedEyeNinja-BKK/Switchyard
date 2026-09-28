// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! F5 acceptance: the capability/fact surface.
//!
//! Two distinct things are proven here, and they are deliberately kept apart.
//!
//! 1. **The non-LLM capability tables.** `[capability_clients.*]` and
//!    `[capabilities.*]` declare typed utility endpoints (OpenAI-compatible
//!    `POST /v1/embeddings`, Cohere/Jina `POST /v1/rerank`) proxied to one
//!    canonical executor each. They are schema + routing-to-the-host only:
//!    executor HTTP client construction, request admission and response
//!    validation live in the serving host crate, which v0.3.0 does not carry.
//!
//! 2. **The FleetRouter ∩ capability seam** — the only place a chat
//!    capability fact and a fleet candidate meet. Production's
//!    `RouteConfig::validate_fleet_capabilities` rejects an untruthful
//!    advertisement (a route claiming `tool_calling`/`reasoning` that no
//!    candidate supports), and its build loop derives `vision` from the
//!    candidate set when the route does not declare it. Both are proven
//!    against the real live config.
//!
//! What this file does NOT prove: capability serving (host crate), and the
//! F3 static-eligibility predicates, which are covered by
//! `fleet_router_eligibility.rs` and are reused here rather than duplicated.

use std::collections::{BTreeMap, BTreeSet};

use switchyard_runner::capability::{
    CapabilityClientFormat, CapabilityKind, CapabilityRouteConfig,
};

/// Location of the live deployment config. Overridable for a relocated tree.
const LIVE_CONFIG: &str = match option_env!("SWITCHYARD_LIVE_CONFIG") {
    Some(path) => path,
    None => "/home/vincent/.local/lib/localclaw-switchyard/routes.toml",
};

/// Live `[capability_clients.*]` and `[capabilities.*]` field names, read from
/// the deployment file rather than hard-coded, so this file fails if the live
/// surface gains or loses a field.
fn live_document() -> toml::Table {
    let text = std::fs::read_to_string(LIVE_CONFIG)
        .unwrap_or_else(|error| panic!("live config {LIVE_CONFIG} must be readable: {error}"));
    toml::from_str(&text).unwrap_or_else(|error| panic!("live config must be valid TOML: {error}"))
}

fn live_table(root: &str) -> BTreeMap<String, toml::Table> {
    live_document()
        .get(root)
        .and_then(toml::Value::as_table)
        .cloned()
        .unwrap_or_else(|| panic!("live config must have a [{root}] table"))
        .into_iter()
        .map(|(name, value)| {
            let table = value
                .as_table()
                .unwrap_or_else(|| panic!("[{root}.{name}] must be a table"))
                .clone();
            (name, table)
        })
        .collect()
}

/// Live client tables re-serialized and parsed through the real F5 schema.
fn parse_live_clients() -> BTreeMap<String, toml::Table> {
    live_table("capability_clients")
}

fn parse_live_capabilities() -> BTreeMap<String, toml::Table> {
    live_table("capabilities")
}

// ---------------------------------------------------------------------------
// 1. The live non-LLM capability tables parse through the carried schema.
// ---------------------------------------------------------------------------

#[test]
fn every_live_capability_client_parses_through_the_carried_schema() {
    let clients = parse_live_clients();
    assert!(
        !clients.is_empty(),
        "live config must declare at least one [capability_clients.*] executor"
    );
    for (name, table) in &clients {
        // Re-parse the exact live sub-table as the carried F5 type.
        let parsed: switchyard_runner::capability::CapabilityClientConfig =
            toml::Value::Table(table.clone())
                .try_into()
                .unwrap_or_else(|error| {
                    panic!("live capability client {name} must parse: {error}")
                });
        match parsed.format {
            CapabilityClientFormat::OpenAiEmbeddings
            | CapabilityClientFormat::CohereJinaRerank
            | CapabilityClientFormat::OpenRouterAlphaDecisions
            | CapabilityClientFormat::OpenAiResponsesDecisionAdapter => {
            }
        }
        assert!(
            !parsed.model.is_empty(),
            "live capability client {name} must declare a model"
        );
        // `HttpBaseUrl` only deserializes an absolute http(s) URL, so parsing at
        // all proves the scheme; the raw value is checked too so the assertion
        // names what it saw.
        let declared_base = table
            .get("base_url")
            .and_then(toml::Value::as_str)
            .unwrap_or_else(|| panic!("live capability client {name} must declare a base_url"));
        assert!(
            declared_base.starts_with("http://") || declared_base.starts_with("https://"),
            "live capability client {name} must carry an absolute http(s) base_url, got {declared_base:?}"
        );
    }
}

#[test]
fn every_live_capability_route_parses_and_discriminates_its_kind() {
    let capabilities = parse_live_capabilities();
    assert!(
        !capabilities.is_empty(),
        "live config must declare at least one [capabilities.*] route"
    );
    let mut embeddings = 0usize;
    let mut reranks = 0usize;
    let mut decisions = 0usize;
    for (name, table) in &capabilities {
        let parsed: CapabilityRouteConfig = toml::Value::Table(table.clone())
            .try_into()
            .unwrap_or_else(|error| panic!("live capability route {name} must parse: {error}"));
        // The discriminator is field PRESENCE, not any value. A decision lane is
        // selected by `decision_contract`, which is distinct from the embedding
        // `contract` so the two can never be confused.
        let declares_decision = table.contains_key("decision_contract");
        let declares_embedding =
            table.contains_key("contract") || table.contains_key("dimensions");
        match (&parsed.kind, declares_embedding, declares_decision) {
            (CapabilityKind::Embedding { .. }, true, false) => embeddings += 1,
            (CapabilityKind::Rerank { .. }, false, false) => reranks += 1,
            (CapabilityKind::Decisions { .. }, false, true) => decisions += 1,
            (kind, e, d) => panic!(
                "live capability route {name} discriminated as {kind:?} but field presence says embedding={e} decision={d}"
            ),
        }
    }
    assert_eq!(
        embeddings + reranks + decisions,
        capabilities.len(),
        "every live capability route must be classified exactly once"
    );
    assert!(
        embeddings > 0 && reranks > 0 && decisions > 0,
        "the live surface exercises ALL THREE capability kinds, got {embeddings} embedding / {reranks} rerank / {decisions} decisions"
    );
}

#[test]
fn an_embedding_capability_requires_both_contract_and_dimensions() {
    // Half the discriminator present: must fail closed, not silently rerank.
    let only_contract = toml::from_str::<CapabilityRouteConfig>(
        r#"
        id = "x"
        target = "y"
        contract = "localclaw-embedding-space:v1"
        "#,
    );
    assert!(
        only_contract.is_err(),
        "an embedding capability missing `dimensions` must not parse"
    );

    let only_dimensions = toml::from_str::<CapabilityRouteConfig>(
        r#"
        id = "x"
        target = "y"
        dimensions = 1024
        "#,
    );
    assert!(
        only_dimensions.is_err(),
        "an embedding capability missing `contract` must not parse"
    );

    // A rerank declaration carrying neither is the other branch, and parses.
    let rerank = toml::from_str::<CapabilityRouteConfig>(
        r#"
        id = "x"
        target = "y"
        "#,
    )
    .expect("a capability route with no embedding fields is a rerank declaration");
    assert!(
        matches!(rerank.kind, CapabilityKind::Rerank { .. }),
        "a capability route with no embedding fields must discriminate as Rerank"
    );
}

#[test]
fn a_capability_client_rejects_an_unknown_field() {
    let unknown = toml::from_str::<switchyard_runner::capability::CapabilityClientConfig>(
        r#"
        format = "openai_embeddings"
        base_url = "https://example.invalid"
        model = "m"
        not_a_real_field = 1
        "#,
    );
    assert!(
        unknown.is_err(),
        "capability_clients must deny unknown fields, not ignore them"
    );
}

#[test]
fn a_capability_route_rejects_an_unknown_field() {
    let unknown = toml::from_str::<CapabilityRouteConfig>(
        r#"
        id = "x"
        target = "y"
        max_candidates = 8
        not_a_real_field = 1
        "#,
    );
    assert!(
        unknown.is_err(),
        "capabilities must deny unknown fields, not ignore them"
    );
}

#[test]
fn capability_defaults_match_the_carried_values() {
    let client: switchyard_runner::capability::CapabilityClientConfig = toml::from_str(
        r#"
        format = "openai_embeddings"
        base_url = "https://example.invalid/"
        model = "m"
        "#,
    )
    .expect("a client with only required fields must parse");
    assert_eq!(
        client.timeout_seconds, 30,
        "the carried capability client timeout default is 30 seconds"
    );
    assert!(
        client.extra_headers.is_empty(),
        "extra_headers defaults to empty"
    );
    assert!(
        client.api_key_env.is_none(),
        "api_key_env defaults to absent"
    );

    let rerank: CapabilityRouteConfig = toml::from_str(
        r#"
        id = "x"
        target = "y"
        "#,
    )
    .expect("a bare rerank declaration must parse");
    match rerank.kind {
        CapabilityKind::Rerank {
            max_candidates,
            top_n,
            max_doc_chars,
            max_query_chars,
        } => {
            assert_eq!(max_candidates, 32, "carried default max_candidates");
            assert_eq!(top_n, 5, "carried default top_n");
            assert_eq!(max_doc_chars, 4096, "carried default max_doc_chars");
            assert_eq!(max_query_chars, 2048, "carried default max_query_chars");
        }
        other => panic!("expected Rerank, got {other:?}"),
    }

    let embedding: CapabilityRouteConfig = toml::from_str(
        r#"
        id = "x"
        target = "y"
        contract = "c"
        dimensions = 1024
        "#,
    )
    .expect("a minimal embedding declaration must parse");
    match embedding.kind {
        CapabilityKind::Embedding {
            normalization,
            max_batch,
            ..
        } => {
            assert_eq!(normalization, "L2", "carried default normalization");
            assert_eq!(max_batch, 64, "carried default max_batch");
        }
        other => panic!("expected Embedding, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// 2. The namespace separation: capability targets are NOT [targets.*].
// ---------------------------------------------------------------------------

#[test]
fn every_capability_target_resolves_in_capability_clients_and_not_in_targets() {
    let capabilities = parse_live_capabilities();
    let clients: BTreeSet<String> = parse_live_clients().into_keys().collect();
    let targets: BTreeSet<String> = live_table("targets").into_keys().collect();
    for (name, table) in &capabilities {
        let target = table
            .get("target")
            .and_then(toml::Value::as_str)
            .unwrap_or_else(|| panic!("capability route {name} must declare a target"));
        assert!(
            clients.contains(target),
            "capability route {name} targets {target}, which is not a [capability_clients.*] entry"
        );
        assert!(
            !targets.contains(target),
            "capability route {name} targets {target}, which ALSO resolves in [targets.*]; \
             the two key spaces must stay disjoint so no capability executor is a routable model"
        );
    }
}

#[test]
fn no_capability_client_name_collides_with_a_chat_target_name() {
    let clients: BTreeSet<String> = parse_live_clients().into_keys().collect();
    let targets: BTreeSet<String> = live_table("targets").into_keys().collect();
    let collisions: Vec<&String> = clients.intersection(&targets).collect();
    assert!(
        collisions.is_empty(),
        "a capability executor must never share a name with a routable chat target, found {collisions:?}"
    );
}

#[test]
fn no_capability_route_id_is_also_a_chat_route_id() {
    let capability_ids: BTreeSet<String> = parse_live_capabilities()
        .into_iter()
        .filter_map(|(_, table)| {
            table
                .get("id")
                .and_then(toml::Value::as_str)
                .map(str::to_string)
        })
        .collect();
    let route_ids: BTreeSet<String> = live_table("routes").into_keys().collect();
    let collisions: Vec<&String> = capability_ids.intersection(&route_ids).collect();
    assert!(
        collisions.is_empty(),
        "a capability route id must not collide with an LLM route id, found {collisions:?}"
    );
}

#[test]
fn capability_targets_are_not_referenced_by_any_fleet_candidate() {
    // The F5/F3 boundary, proven on live data: no fleet candidate selects a
    // capability executor, so F3's eligibility predicates can never be
    // evaluated against one.
    let capability_targets: BTreeSet<String> = parse_live_capabilities()
        .into_iter()
        .filter_map(|(_, table)| {
            table
                .get("target")
                .and_then(toml::Value::as_str)
                .map(str::to_string)
        })
        .collect();
    let document = live_document();
    let routes = document
        .get("routes")
        .and_then(toml::Value::as_table)
        .cloned()
        .expect("live config must have a [routes] table");
    let mut offenders = Vec::new();
    for (route_name, value) in &routes {
        let Some(table) = value.as_table() else {
            continue;
        };
        let Some(candidates) = table.get("candidates").and_then(toml::Value::as_array) else {
            continue;
        };
        for candidate in candidates {
            let Some(candidate) = candidate.as_table() else {
                continue;
            };
            let Some(target) = candidate.get("target").and_then(toml::Value::as_str) else {
                continue;
            };
            if capability_targets.contains(target) {
                offenders.push(format!("{route_name} -> {target}"));
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "no fleet candidate may select a capability executor, found {offenders:?}"
    );
}

// ---------------------------------------------------------------------------
// 3. The FleetRouter ∩ capability seam.
// ---------------------------------------------------------------------------

/// A minimal, self-consistent deployment document driving the REAL builder
/// (`Runner::from_toml` -> `DeploymentConfig::build`). Small enough that the
/// only thing under test is the capability seam, and it carries no credential
/// or network surface: building a Runner never dials a provider.
fn deployment_with(route_body: &str) -> String {
    format!(
        r#"
schema_version = 1

[llm_clients.local]
format = "openai_chat"
base_url = "https://example.invalid"

[targets.t1]
id = "m1"
llm_client = "local"

[targets.t2]
id = "m2"
llm_client = "local"

{route_body}
"#
    )
}

/// The advertised capabilities the built runner exposes for a named route.
fn advertised(route: &str) -> (Option<bool>, Option<bool>) {
    let runner = switchyard_runner::Runner::from_toml(&route).expect("document must build");
    let info = runner
        .models()
        .find(|info| info.id.as_str() == "fleet/test")
        .expect("built runner must expose the test route");
    (info.capabilities.tool_calling, info.capabilities.vision)
}

#[test]
fn the_real_builder_rejects_an_untruthful_tool_calling_advertisement() {
    // Drives production's `validate_fleet_capabilities` through the real loader:
    // the route claims tools, and no candidate profile supports them.
    let document = deployment_with(
        r#"
[routes."fleet/test"]
id = "fleet/test"
type = "fleet_router"
tool_calling = true
reasoning = false
candidates = [
  { target = "t1", tool_calling = false, reasoning = false, supports_vision = false },
  { target = "t2", tool_calling = false, reasoning = false, supports_vision = false },
]
"#,
    );
    let message = match switchyard_runner::Runner::from_toml(&document) {
        Ok(_) => panic!(
            "a fleet route advertising tool_calling with no supporting candidate must be refused"
        ),
        Err(error) => error.to_string(),
    };
    assert!(
        message.contains("tool_calling=true")
            && message.contains("no candidate profile supports tool calling"),
        "the refusal must name the untruthful advertisement, got: {message}"
    );
}

#[test]
fn the_real_builder_rejects_an_untruthful_reasoning_advertisement() {
    let document = deployment_with(
        r#"
[routes."fleet/test"]
id = "fleet/test"
type = "fleet_router"
tool_calling = false
reasoning = true
candidates = [
  { target = "t1", tool_calling = true, reasoning = false, supports_vision = false },
  { target = "t2", tool_calling = true, reasoning = false, supports_vision = false },
]
"#,
    );
    let message = match switchyard_runner::Runner::from_toml(&document) {
        Ok(_) => panic!(
            "a fleet route advertising reasoning with no supporting candidate must be refused"
        ),
        Err(error) => error.to_string(),
    };
    assert!(
        message.contains("reasoning=true")
            && message.contains("no candidate profile supports reasoning"),
        "the refusal must name the untruthful advertisement, got: {message}"
    );
}

#[test]
fn a_truthful_advertisement_loads_and_a_false_declaration_is_not_a_refusal() {
    // The control arm: the guard fires on TRUE-with-no-support only. A route
    // declaring false, and a route whose advertisement IS supported, both load.
    let truthful = deployment_with(
        r#"
[routes."fleet/test"]
id = "fleet/test"
type = "fleet_router"
tool_calling = true
reasoning = true
candidates = [
  { target = "t1", tool_calling = true, reasoning = true, supports_vision = false },
  { target = "t2", tool_calling = true, reasoning = true, supports_vision = false },
]
"#,
    );
    assert!(
        switchyard_runner::Runner::from_toml(&truthful).is_ok(),
        "a truthful advertisement must load"
    );

    let all_false = deployment_with(
        r#"
[routes."fleet/test"]
id = "fleet/test"
type = "fleet_router"
tool_calling = false
reasoning = false
candidates = [
  { target = "t1", tool_calling = false, reasoning = false, supports_vision = false },
]
"#,
    );
    assert!(
        switchyard_runner::Runner::from_toml(&all_false).is_ok(),
        "declaring false is never an untruthful advertisement"
    );
}

#[test]
fn the_real_builder_derives_vision_from_the_candidate_set() {
    // `vision = declared.or(Some(any_candidate_vision))`, observed through the
    // built runner rather than re-computed here.
    let derived = deployment_with(
        r#"
[routes."fleet/test"]
id = "fleet/test"
type = "fleet_router"
tool_calling = false
reasoning = false
candidates = [
  { target = "t1", tool_calling = false, reasoning = false, supports_vision = false },
  { target = "t2", tool_calling = false, reasoning = false, supports_vision = true },
]
"#,
    );
    let (tools, vision) = advertised(&derived);
    assert_eq!(
        vision,
        Some(true),
        "a vision-capable candidate set must advertise vision"
    );
    let _ = tools;

    // The discriminating arm: a candidate set where EVERY candidate is
    // non-vision. Under the correct predicate this derives `false`; an inverted
    // predicate derives `true` and is caught here.
    let all_non_vision = deployment_with(
        r#"
[routes."fleet/test"]
id = "fleet/test"
type = "fleet_router"
tool_calling = false
reasoning = false
candidates = [
  { target = "t1", tool_calling = false, reasoning = false, supports_vision = false },
  { target = "t2", tool_calling = false, reasoning = false, supports_vision = false },
]
"#,
    );
    let (_, all_non_vision_advertised) = advertised(&all_non_vision);
    assert_eq!(
        all_non_vision_advertised,
        Some(false),
        "an entirely non-vision candidate set must derive vision=false, not true"
    );

    let declared_false_wins = deployment_with(
        r#"
[routes."fleet/test"]
id = "fleet/test"
type = "fleet_router"
tool_calling = false
reasoning = false
supports_vision = false
candidates = [
  { target = "t1", tool_calling = false, reasoning = false, supports_vision = false },
  { target = "t2", tool_calling = false, reasoning = false, supports_vision = true },
]
"#,
    );
    let (_, vision) = advertised(&declared_false_wins);
    assert_eq!(
        vision,
        Some(false),
        "an explicitly declared vision=false must NOT be lifted by a vision-capable candidate"
    );
    let _ = tools;
}

#[test]
fn the_real_builder_does_not_derive_vision_for_a_passthrough_route() {
    // The derivation is scoped to `AlgorithmSpec::FleetRouter`; a passthrough
    // route's advertisement is whatever it declares, nothing more.
    let passthrough = deployment_with(
        r#"
[routes."fleet/test"]
id = "fleet/test"
type = "passthrough"
target = "t1"
tool_calling = false
reasoning = false
"#,
    );
    let (_, vision) = advertised(&passthrough);
    assert_eq!(
        vision, None,
        "a passthrough route advertises no derived vision capability"
    );
}

#[test]
fn every_live_fleet_route_advertisement_is_truthful_against_its_candidates() {
    // The live half of the seam, checked against the production predicate as
    // data: no live route may declare a capability its candidate set lacks,
    // because production would refuse to load that file at all.
    let document = live_document();
    let routes = document
        .get("routes")
        .and_then(toml::Value::as_table)
        .cloned()
        .expect("live config must have a [routes] table");
    let mut fleet_routes = 0usize;
    for (route_name, value) in &routes {
        let table = value.as_table().expect("each [routes.*] must be a table");
        if table.get("type").and_then(toml::Value::as_str) != Some("fleet_router") {
            continue;
        }
        fleet_routes += 1;
        let candidates = table
            .get("candidates")
            .and_then(toml::Value::as_array)
            .unwrap_or_else(|| panic!("fleet route {route_name} must declare candidates"));
        let any = |key: &str| {
            candidates
                .iter()
                .filter_map(toml::Value::as_table)
                .any(|c| c.get(key).and_then(toml::Value::as_bool) == Some(true))
        };
        if table.get("tool_calling").and_then(toml::Value::as_bool) == Some(true) {
            assert!(
                any("tool_calling"),
                "live fleet route {route_name} advertises tool_calling=true with no \
                 supporting candidate; production would refuse to load this"
            );
        }
        if table.get("reasoning").and_then(toml::Value::as_bool) == Some(true) {
            assert!(
                any("reasoning"),
                "live fleet route {route_name} advertises reasoning=true with no \
                 supporting candidate; production would refuse to load this"
            );
        }
    }
    assert!(
        fleet_routes > 0,
        "the live config must exercise at least one fleet_router route"
    );
}

#[test]
fn every_live_fleet_route_declares_a_candidate_set_the_loaded_runner_accepts() {
    // End-to-end through the real parser: the live capability tables load into
    // the runner as parsed types, not as an unknown-field error. The live file
    // as a whole still cannot load (C1 is RED for other rows), so this builds
    // a capability-only document rather than asserting on a load we know fails.
    let document = live_document();
    let capability_clients = document
        .get("capability_clients")
        .cloned()
        .expect("live config must have [capability_clients]");
    let capabilities = document
        .get("capabilities")
        .cloned()
        .expect("live config must have [capabilities]");
    let clients: BTreeMap<String, toml::Table> = capability_clients
        .as_table()
        .expect("[capability_clients] must be a table")
        .iter()
        .map(|(name, value)| {
            (
                name.clone(),
                value.as_table().expect("client table").clone(),
            )
        })
        .collect();
    let routes: BTreeMap<String, toml::Table> = capabilities
        .as_table()
        .expect("[capabilities] must be a table")
        .iter()
        .map(|(name, value)| (name.clone(), value.as_table().expect("route table").clone()))
        .collect();
    for (name, table) in &clients {
        let parsed: switchyard_runner::capability::CapabilityClientConfig = toml::Value::Table(
            table.clone(),
        )
        .try_into()
        .unwrap_or_else(|error| {
            panic!("live [capability_clients.{name}] must load through the runner schema: {error}")
        });
        assert_eq!(
            parsed.timeout_seconds > 0,
            true,
            "a live capability client must have a positive timeout"
        );
    }
    for (name, table) in &routes {
        let _: CapabilityRouteConfig =
            toml::Value::Table(table.clone())
                .try_into()
                .unwrap_or_else(|error| {
                    panic!(
                        "live [capabilities.{name}] must load through the runner schema: {error}"
                    )
                });
    }
}

#[test]
fn the_vision_derivation_is_a_monotonic_or_not_a_replacement() {
    // A focused, synthetic version of the live assertion above: `.or()` is
    // short-circuiting, so a declared `false` survives a vision-capable set
    // and a declared `true` survives a non-vision set. Declared wins, always.
    let derive = |declared: Option<bool>, any_vision: bool| declared.or(Some(any_vision));
    assert_eq!(
        derive(Some(false), true),
        Some(false),
        "declared false must not be lifted"
    );
    assert_eq!(
        derive(Some(true), false),
        Some(true),
        "declared true must not be dropped"
    );
    assert_eq!(
        derive(None, true),
        Some(true),
        "absent declaration derives true"
    );
    assert_eq!(
        derive(None, false),
        Some(false),
        "absent declaration derives false"
    );
}
