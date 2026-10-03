// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Ordered composition of complete child routes.
//!
//! A child route already owns everything that makes a route work: its algorithm, client
//! router, request preparation, reasoning policy, native retry, native candidate fallback,
//! continuation handling and telemetry. Composition therefore invokes whole children and
//! adds nothing to them. It never flattens children into their targets, never runs the
//! children over HTTP, and keeps no health or cooldown state of its own, because each
//! child already owns that.
//!
//! Three outcomes are kept distinct, and conflating them is the whole risk here.
//!
//! A child that cannot serve the request was never eligible. That is routing: the next
//! child is tried and the skipped one received no provider call.
//!
//! A child that was eligible and then hit a transient serving failure may yield to the
//! next child, but only after the child's own candidates have been exhausted. A child
//! recovers within itself first; only a child that genuinely cannot serve yields.
//!
//! A child that was eligible and then failed for a reason another route cannot fix stops
//! the parent. Retrying elsewhere would hide a contract, policy, credential or
//! configuration defect behind a different provider's answer.

use serde_json::Value;
use switchyard_llm_client::FailureClass;
use switchyard_protocol::Request;

use crate::{Route, RunOutput, Runner, RunnerError, request_fit};

/// An ordered list of child route names.
#[derive(Clone, Debug, Default)]
pub struct CompositeSpec {
    /// Child routes in preference order.
    pub children: Vec<String>,
}

/// What a composite did for one request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ChildOutcome {
    /// The child's declared capabilities cannot serve this request, so it was never
    /// called.
    Skipped { capability: &'static str },
    /// The child served the request.
    Served { target: String },
    /// The child was called and failed in a way another child may recover from.
    Failed { class: FailureClass, detail: String },
}

impl ChildOutcome {
    /// A short, stable reason for telemetry.
    pub fn reason(&self) -> &'static str {
        match self {
            ChildOutcome::Skipped { .. } => "ineligible",
            ChildOutcome::Served { .. } => "served",
            ChildOutcome::Failed { .. } => "serving_failure",
        }
    }
}

/// The result of one composed execution, with everything needed to reconstruct it.
///
/// The served answer is held rather than cloned, because it owns a live response.
pub struct CompositeOutput {
    /// The served answer.
    pub output: RunOutput,
    /// The parent route that was invoked.
    pub parent_route: String,
    /// The child that served it.
    pub child_route: String,
    /// The final target the child selected.
    pub final_target: String,
    /// What happened to each child, in order.
    pub children: Vec<(String, ChildOutcome)>,
    /// True when a serving failure moved the request to a later child.
    pub crossed_child: bool,
}

/// What one execution did, including composition context when a composite served it.
///
/// A leaf carries no composition metadata. A composite carries enough to reconstruct what
/// happened without asking the server to infer it: which route was invoked, which child
/// answered, which model actually served, and why each other child was not used.
/// The served answer is owned rather than cloned, because it holds a live response.
pub struct ExecutionOutput {
    /// The served answer.
    pub output: RunOutput,
    /// Composition context, when the route was a composite.
    pub composition: Option<CompositionMetadata>,
}

/// Composition context for one composite execution.
#[derive(Clone, Debug)]
pub struct CompositionMetadata {
    /// The composite route that was invoked.
    pub parent_route: String,
    /// The child that served the request.
    pub child_route: String,
    /// The model that actually served, which may differ from the child's first selection.
    pub final_target: String,
    /// What happened at each child, in order.
    pub children: Vec<(String, ChildOutcome)>,
    /// Whether a serving failure moved the request to a later child.
    pub crossed_child: bool,
    /// Why the request crossed children, when it did.
    pub cross_child_reason: Option<String>,
}

impl CompositionMetadata {
    /// The reason a child was skipped or failed, for a single stage.
    pub fn cross_child_reason(&self) -> Option<String> {
        self.cross_child_reason.clone().or_else(|| {
            self.children
                .iter()
                .find(|(_, outcome)| outcome.reason() != "served")
                .map(|(name, outcome)| format!("{name}:{}", outcome.reason()))
        })
    }
}

/// Which route a request resolved to, and whether it was composite.
pub enum Selection<'a> {
    /// A leaf route with its own native behaviour.
    Leaf(&'a Route),
    /// An ordered composite over complete child routes.
    Composite(&'a CompositeSpec),
}

impl Runner {
    /// Executes a named route, resolving a leaf or an ordered composite in-process.
    ///
    /// This is the single execution seam. HTTP ingress resolves a route and then calls
    /// this, so tier behaviour never reaches the serving crate.
    pub async fn execute_route(
        &self,
        route_id: &str,
        request: Request,
        observer: Option<switchyard_llm_client::RunObserver>,
    ) -> Result<ExecutionOutput, RunnerError> {
        match self.select(route_id)? {
            Selection::Leaf(route) => Ok(ExecutionOutput {
                output: route.execute(request, observer).await?,
                composition: None,
            }),
            Selection::Composite(_) => {
                let composed = self.execute_composite(route_id, request, observer).await?;
                Ok(ExecutionOutput {
                    output: composed.output,
                    composition: Some(CompositionMetadata {
                        parent_route: composed.parent_route,
                        child_route: composed.child_route,
                        final_target: composed.final_target,
                        children: composed.children,
                        crossed_child: composed.crossed_child,
                        cross_child_reason: None,
                    }),
                })
            }
        }
    }

    /// Executes a composite, recording what happened at each child.
    pub async fn execute_composite(
        &self,
        parent_route: &str,
        request: Request,
        observer: Option<switchyard_llm_client::RunObserver>,
    ) -> Result<CompositeOutput, RunnerError> {
        let Selection::Composite(spec) = self.select(parent_route)? else {
            return Err(RunnerError::UnknownRouteModel(parent_route.to_string()));
        };
        let mut children = Vec::new();
        let mut crossed = false;
        let mut last_error: Option<RunnerError> = None;

        // A provider-bound continuation must stay with the child that owns it. Restarting
        // at the first child would either lose the conversation or silently serve it from a
        // provider that cannot reconstruct it, so the owning child is pinned and its own
        // failure surfaces instead of being absorbed elsewhere.
        let claimants = self.continuation_claimants(parent_route, &request);
        if claimants.len() > 1 {
            // Two children claim the same provider state. Choosing by child order would make
            // the outcome depend on configuration order, so this is reported instead and no
            // provider is contacted.
            return Err(RunnerError::StateConflict(format!(
                "composite route {parent_route} has children claiming the same continuation: {}",
                claimants.join(", ")
            )));
        }
        let owner = claimants.into_iter().next();
        let order: Vec<&String> = match owner.as_ref() {
            Some(owner) => {
                let owned: Vec<&String> = spec
                    .children
                    .iter()
                    .filter(|child| *child == owner)
                    .collect();
                if owned.is_empty() {
                    // The recorded owner is not a child of this composite, so this tier
                    // cannot serve the continuation at all.
                    return Err(RunnerError::configuration(format!(
                        "composite route {parent_route} cannot serve a continuation owned by \
                         {owner}, which is not one of its children"
                    )));
                }
                owned
            }
            None => spec.children.iter().collect(),
        };

        for child_name in order.into_iter() {
            let Some(route) = self.route(child_name) else {
                return Err(RunnerError::UnknownRouteModel(child_name.clone()));
            };

            // A child that cannot serve this request was never eligible. Skipping it is
            // routing, not recovery, and it receives no provider call at all.
            if let Some(capability) = request_fit::unsupported_capability(
                route.capabilities(),
                &request.llm_request,
                request.raw_request.as_ref().unwrap_or(&serde_json::Value::Null),
            ) {
                children.push((
                    child_name.clone(),
                    ChildOutcome::Skipped { capability },
                ));
                continue;
            }

            // Each child receives the pristine request. Whatever the previous child
            // prepared for its provider is deliberately dropped, so a child cannot inherit
            // another child's model, body or reasoning fields.
            let child_request = request.clone();
            match route.execute(child_request, observer.clone()).await {
                Ok(output) => {
                    // The response names the candidate that actually served it, which is not
                    // always the one the child selected: after the child's own fallback a later
                    // target may have answered. The routing selection is the fallback only when
                    // the response carries no served model.
                    let final_target = output
                        .response
                        .served_model()
                        .cloned()
                        .unwrap_or_else(|| output.selected_model.clone())
                        .as_str()
                        .to_string();
                    children.push((
                        child_name.clone(),
                        ChildOutcome::Served {
                            target: final_target.clone(),
                        },
                    ));
                    return Ok(CompositeOutput {
                        output,
                        parent_route: parent_route.to_string(),
                        child_route: child_name.clone(),
                        final_target,
                        children,
                        crossed_child: crossed,
                    });
                }
                Err(error) => {
                    let class = self.classify(&error);
                    if !class.allows_cross_route() {
                        // Terminal for the parent. Surfacing the child's own error keeps a
                        // contract, policy or credential fault visible instead of letting a
                        // different route answer for it.
                        children.push((
                            child_name.clone(),
                            ChildOutcome::Failed {
                                class,
                                detail: error.to_string(),
                            },
                        ));
                        return Err(error);
                    }
                    crossed = true;
                    children.push((
                        child_name.clone(),
                        ChildOutcome::Failed {
                            class,
                            detail: error.to_string(),
                        },
                    ));
                    last_error = Some(error);
                }
            }
        }

        // Nothing answered. If no child was even eligible, the request itself asks for
        // something no child can serve, which is the caller's problem rather than a
        // deployment fault.
        if let Some(capability) = children.iter().find_map(|(_, outcome)| match outcome {
            ChildOutcome::Skipped { capability } => Some(*capability),
            _ => None,
        }) {
            return Err(RunnerError::unsupported_capability(parent_route, capability));
        }
        Err(last_error.unwrap_or_else(|| {
            RunnerError::unsupported_capability(parent_route, "an eligible route")
        }))
    }

    /// Resolves a route name to either a leaf or a composite.
    pub fn select(&self, route_id: &str) -> Result<Selection<'_>, RunnerError> {
        if let Some(spec) = self.composite_spec(route_id) {
            return Ok(Selection::Composite(spec));
        }
        self.route(route_id)
            .map(Selection::Leaf)
            .ok_or_else(|| RunnerError::UnknownRouteModel(route_id.to_string()))
    }

    /// Every child that holds provider state for this request.
    ///
    /// More than one claimant is a state conflict, not a choice: the parent must not pick by
    /// child order.
    pub fn continuation_claimants(
        &self,
        parent_route: &str,
        request: &Request,
    ) -> Vec<String> {
        let Ok(Selection::Composite(spec)) = self.select(parent_route) else {
            return Vec::new();
        };
        spec.children
            .iter()
            .filter(|child| {
                self.route(child)
                    .and_then(|route| route.continuation_owner(request))
                    .is_some()
            })
            .cloned()
            .collect()
    }

    /// The child route that holds provider state for this request, if any.
    ///
    /// Read-only: the owning child keeps its own state, and nothing is copied or
    /// reconstructed here.
    pub fn continuation_owner(
        &self,
        parent_route: &str,
        request: &Request,
    ) -> Option<String> {
        let Selection::Composite(spec) = self.select(parent_route).ok()? else {
            return None;
        };
        spec.children
            .iter()
            .find(|child| {
                self.route(child)
                    .and_then(|route| route.continuation_owner(request))
                    .is_some()
            })
            .cloned()
    }

    /// Classifies a child failure for cross-child purposes.
    ///
    /// The client's own classifier is reused, so status and body interpretation is not
    /// repeated here. Only the decision about whether a whole route may be crossed differs
    /// from candidate fallback inside a route, and that lives in
    /// [`FailureClass::allows_cross_route`].
    fn classify(&self, error: &RunnerError) -> FailureClass {
        match error {
            RunnerError::Algorithm(inner) => switchyard_llm_client::failure_class(inner),
            _ => FailureClass::Other,
        }
    }
}

/// Filters an ordered candidate list by what each target can actually serve.
///
/// This is declared order filtered by request eligibility, not scoring or selection: a
/// target the request cannot use is dropped, and the order of the rest is untouched, so
/// the native execution, retry, cooldown and fallback machinery still decides among them.
///
/// Eligibility is decided with the same predicate that governs route admission and
/// whole-route child eligibility, so "this request needs vision" means one thing at every
/// level. A target with no declared facts is left in place: absence of a fact is not a
/// refusal, and a newly added target is never silently unroutable.
pub fn eligible_candidates(
    ordered: &[&str],
    capabilities: &std::collections::BTreeMap<String, crate::ModelCapabilities>,
    request: &Request,
) -> Vec<String> {
    let raw = request.raw_request.clone().unwrap_or(Value::Null);
    ordered
        .iter()
        .filter(|name| match capabilities.get(**name) {
            Some(capabilities) => {
                crate::request_fit::unsupported_capability(
                    *capabilities,
                    &request.llm_request,
                    &raw,
                )
                .is_none()
            }
            None => true,
        })
        .map(|name| (*name).to_string())
        .collect()
}

#[cfg(test)]
mod target_eligibility_tests {
    use super::*;
    use serde_json::json;
    use std::collections::BTreeMap;
    use switchyard_protocol::{
        ContentBlock, ImageSource, LlmRequest, Message, Role,
    };
    use crate::ModelCapabilities;

    /// A target is a factual property of a backing deployment; a route is a policy promise.
    /// This is the input the filter uses, and it is deliberately separate from the route's
    /// advertised capability.
    #[test]
    fn a_target_filter_preserves_declared_order_among_eligible_targets() {
        let capabilities: BTreeMap<String, ModelCapabilities> = [
            (
                "text-only".to_string(),
                ModelCapabilities {
                    vision: Some(false),
                    ..ModelCapabilities::default()
                },
            ),
            (
                "vision-a".to_string(),
                ModelCapabilities {
                    vision: Some(true),
                    ..ModelCapabilities::default()
                },
            ),
            (
                "vision-b".to_string(),
                ModelCapabilities {
                    vision: Some(true),
                    ..ModelCapabilities::default()
                },
            ),
        ]
        .into_iter()
        .collect();

        let ordered = ["text-only", "vision-a", "vision-b"];

        // A text request keeps the declared order untouched.
        let text = plain_request();
        assert_eq!(
            eligible_candidates(&ordered, &capabilities, &text),
            vec!["text-only", "vision-a", "vision-b"],
            "a text request is not filtered, and declared order is preserved"
        );

        // A vision request drops the text-only primary and keeps the order of the rest.
        let vision = vision_request();
        assert_eq!(
            eligible_candidates(&ordered, &capabilities, &vision),
            vec!["vision-a", "vision-b"],
            "the ineligible primary is dropped and the declared order of the rest is kept"
        );
    }

    /// A target with no declared facts is treated as able to serve, so adding the filter
    /// cannot silently remove a target that was never characterised.
    #[test]
    fn an_uncharacterised_target_is_not_filtered_out() {
        let capabilities = BTreeMap::new();
        let ordered = ["unknown"];
        assert_eq!(
            eligible_candidates(&ordered, &capabilities, &vision_request()),
            vec!["unknown"],
            "an absent fact must not become a refusal"
        );
    }

    /// Filtering uses the same predicate as route admission and child eligibility, so
    /// "requires vision" means one thing at all three levels.
    #[test]
    fn the_filter_uses_the_shared_request_fit_predicate() {
        let capabilities: BTreeMap<String, ModelCapabilities> = [(
            "text-only".to_string(),
            ModelCapabilities {
                vision: Some(false),
                ..ModelCapabilities::default()
            },
        )]
        .into_iter()
        .collect();
        let request = vision_request();
        let raw = request.raw_request.clone().unwrap_or(Value::Null);
        // The shared predicate says this target cannot serve it, and the filter agrees.
        assert_eq!(
            crate::request_fit::unsupported_capability(
                capabilities["text-only"],
                &request.llm_request,
                &raw
            ),
            Some("vision")
        );
        assert!(
            eligible_candidates(&["text-only"], &capabilities, &request).is_empty(),
            "the filter must agree with the shared predicate"
        );
    }

    #[tokio::test]
    async fn the_runtime_filter_keeps_declared_order_for_a_vision_request() {
        use crate::Runner;
        let config = r#"
schema_version = 1

[llm_clients.c]
format = "openai_responses"
base_url = "https://example.test/v1"

[targets.text]
id = "vendor/text"
llm_client = "c"
vision = false

[targets.vl_a]
id = "vendor/vl-a"
llm_client = "c"
vision = true

[targets.vl_b]
id = "vendor/vl-b"
llm_client = "c"
vision = true

[routes.r]
id = "r"
type = "passthrough"
target = "text"
candidates = ["vl_a", "vl_b"]
vision = true
"#;
        let runner = Runner::from_toml(config).expect("builds");
        let error = match runner.execute_route("r", vision_request(), None).await {
            Ok(_) => panic!("no provider is reachable in this fixture"),
            Err(error) => error.to_string(),
        };
        // The error names the LAST candidate native execution tried. Both facts matter: the
        // text-only primary must never appear, and execution must have walked only the
        // vision-capable candidates in declared order.
        assert!(
            !error.contains("vendor/text"),
            "the ineligible primary must not be dispatched: {error}"
        );
        assert!(
            error.contains("vendor/vl-"),
            "execution must reach a vision-capable candidate: {error}"
        );
    }

    /// With exactly one eligible survivor, the candidate native execution reaches must be that
    /// one. This distinguishes "filtered" from "not filtered" and "first" from "reordered",
    /// which an error naming the last-tried candidate cannot.
    #[tokio::test]
    async fn exactly_one_eligible_candidate_is_the_one_dispatched() {
        use crate::Runner;
        let config = r#"
schema_version = 1

[llm_clients.c]
format = "openai_responses"
base_url = "https://example.test/v1"

[targets.text]
id = "vendor/text"
llm_client = "c"
vision = false

[targets.vl]
id = "vendor/vl"
llm_client = "c"
vision = true

[routes.r]
id = "r"
type = "passthrough"
target = "text"
candidates = ["vl"]
vision = true
"#;
        let runner = Runner::from_toml(config).expect("builds");
        let error = match runner.execute_route("r", vision_request(), None).await {
            Ok(_) => panic!("no provider is reachable in this fixture"),
            Err(error) => error.to_string(),
        };
        assert!(
            !error.contains("vendor/text"),
            "the ineligible primary must be filtered before dispatch: {error}"
        );
        assert!(
            error.contains("vendor/vl"),
            "the only eligible candidate must be dispatched: {error}"
        );
    }

    /// A route whose every target lacks a capability is a request error, not a server fault.
    #[tokio::test]
    async fn no_eligible_target_is_a_request_capability_error() {
        use crate::Runner;
        let config = r#"
schema_version = 1

[llm_clients.c]
format = "openai_responses"
base_url = "https://example.test/v1"

[targets.text]
id = "vendor/text"
llm_client = "c"
vision = false

[routes.r]
id = "r"
type = "passthrough"
target = "text"
candidates = []
"#;
        let runner = Runner::from_toml(config).expect("builds");
        let error = match runner.execute_route("r", vision_request(), None).await {
            Ok(_) => panic!("expected a capability refusal"),
            Err(error) => error,
        };
        assert!(
            matches!(error, crate::RunnerError::UnsupportedCapability { capability: "vision", .. }),
            "an image request against a text-only route is a request error: {error}"
        );
    }

    /// Two eligible vision candidates: the one native execution reaches FIRST must be the
    /// first one declared. Reordering the survivors would dispatch the second instead, which
    /// is the difference between filtering and selecting.
    #[tokio::test]
    async fn the_first_eligible_candidate_declared_is_the_first_one_tried() {
        use crate::Runner;
        let config = r#"
schema_version = 1

[llm_clients.c]
format = "openai_responses"
base_url = "https://example.test/v1"

[targets.text]
id = "vendor/text"
llm_client = "c"
vision = false

[targets.vl_a]
id = "vendor/vl-a"
llm_client = "c"
vision = true

[targets.vl_b]
id = "vendor/vl-b"
llm_client = "c"
vision = true

[routes.r]
id = "r"
type = "passthrough"
target = "text"
candidates = ["vl_a", "vl_b"]
vision = true
"#;
        let runner = Runner::from_toml(config).expect("builds");
        let error = match runner.execute_route("r", vision_request(), None).await {
            Ok(_) => panic!("no provider is reachable in this fixture"),
            Err(error) => error.to_string(),
        };
        assert!(
            !error.contains("vendor/text"),
            "the ineligible primary must be filtered: {error}"
        );
        // Both survivors are unreachable, so the reported failure is the last one tried. That
        // is only reachable when execution started at the FIRST declared survivor and walked
        // forward, which is the ordering guarantee.
        assert!(
            error.contains("vendor/vl-b"),
            "execution must walk the eligible candidates in declared order: {error}"
        );
    }

    pub(crate) fn plain_request() -> Request {
        Request {
            llm_request: LlmRequest {
                model: Some("route".into()),
                messages: vec![Message::text(Role::User, "hello")],
                ..LlmRequest::default()
            },
            raw_request: Some(json!({
                "model": "route",
                "input": [{"role": "user", "content": [{"type": "input_text", "text": "hello"}]}]
            })),
            metadata: None,
        }
    }

    pub(crate) fn vision_request() -> Request {
        Request {
            llm_request: LlmRequest {
                model: Some("route".into()),
                messages: vec![Message {
                    role: Role::User,
                    content: vec![
                        ContentBlock::Text { text: "what is this".into() },
                        ContentBlock::Image {
                            source: ImageSource::Url {
                                url: "https://example.test/a.png".into(),
                                detail: None,
                            },
                        },
                    ],
                }],
                ..LlmRequest::default()
            },
            raw_request: Some(json!({
                "model": "route",
                "input": [{"role": "user", "content": [
                    {"type": "input_text", "text": "what is this"},
                    {"type": "input_image", "image_url": "https://example.test/a.png"}
                ]}]
            })),
            metadata: None,
        }
    }
}
