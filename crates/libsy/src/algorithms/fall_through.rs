// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Fall-through classifier routing: a composable [`Algorithm`] that routes each turn
//! through a processor chain and a classifier cascade.
//!
//! Each turn: request-side [`Processor`]s fold facts into the composition's state; the
//! [`Classifier`] cascade is consulted in order and the first to score decides the target
//! (its `argmax`); the selected model is replayed to the processors so
//! stateful ones (latch, affinity) can bind it.
//!
//! The default `FallThrough<()>` carries no composition state. Stateful compositions share one
//! private state value across turns with the same session ID. Requests without a session ID use
//! unretained per-run state.
//!
//! The selected target is offered first, followed by every other configured target. The consumer
//! may fall through that ordered candidate list when a model call fails.

use std::{
    collections::HashMap,
    sync::{Arc, Once, Weak},
    time::{Duration, Instant},
};

use async_trait::async_trait;
use parking_lot::Mutex;
use tokio::sync::Mutex as AsyncMutex;

use crate::core::algorithm::{self, Algorithm, Driver};
use crate::core::classifier::{Classification, Classifier, Score};
use crate::core::processor::{Event, Processor};
use crate::{LibsyError, Result, RoutingOutcome};
use switchyard_protocol::{ModelId, Request, Response};

struct SessionState<S> {
    state: Arc<AsyncMutex<S>>,
    last_accessed: Instant,
}

type SessionStates<S> = Mutex<HashMap<String, SessionState<S>>>;

/// Delete sessions that have been inactive this long. Catches sessions that did not terminate
/// cleanly.
/// A user resuming a deleted session is not fatal. Algorithms will be missing some context
/// so may route less well for the first turn or two.
const SESSION_STATE_TTL: Duration = Duration::from_secs(60 * 60);

/// Run the expired session cleanup code this often.
const SESSION_CLEANUP_INTERVAL: Duration = Duration::from_secs(60 * 60);

/// Terminal classifier for a cascade whose classifiers may all abstain.
///
/// A classifier abstains when it cannot decide, which lets the next one try. The
/// last has no next, so a cascade that could abstain all the way through needs a
/// decider that never does. Which target that is belongs to whoever assembles the
/// cascade, not to the classifiers in it.
pub struct DefaultTarget {
    target: ModelId,
}

impl DefaultTarget {
    /// Close a cascade with `target`.
    pub fn new(target: impl Into<ModelId>) -> Self {
        Self {
            target: target.into(),
        }
    }
}

#[async_trait]
impl<S: Send> Classifier<S> for DefaultTarget {
    async fn score(
        &self,
        _state: &mut S,
        _request: &mut Request,
        _driver: Option<&Driver>,
    ) -> Result<(Classification, Option<Response>)> {
        // Zero confidence: this is a fallback, not a judgement.
        Ok((
            Classification::Scores(vec![Score {
                target: self.target.clone(),
                confidence: 0.0,
            }]),
            None,
        ))
    }
}

/// Processor chain → classifier cascade → routed model call. See the module docs.
///
/// The generic state type is shared by every processor and classifier in the composition.
pub struct FallThrough<S = ()> {
    name: String,
    processors: Vec<Arc<dyn Processor<S>>>,
    classifiers: Vec<Arc<dyn Classifier<S>>>,
    targets: Vec<ModelId>,
    session_states: Option<Arc<SessionStates<S>>>,
    cleanup_started: Once,
}

impl FallThrough<()> {
    /// Creates an empty stateless router.
    pub fn new(targets: Vec<ModelId>) -> Self {
        Self {
            name: "fall_through".to_string(),
            processors: Vec::new(),
            classifiers: Vec::new(),
            targets,
            session_states: None,
            cleanup_started: Once::new(),
        }
    }
}

impl<S> FallThrough<S>
where
    S: Default + Send + 'static,
{
    /// Creates a router that retains one private `S` per session.
    pub fn new_with_state(targets: Vec<ModelId>) -> Self {
        Self {
            name: "fall_through".to_string(),
            processors: Vec::new(),
            classifiers: Vec::new(),
            targets,
            session_states: Some(Arc::new(Mutex::new(HashMap::new()))),
            cleanup_started: Once::new(),
        }
    }

    /// Sets the stable, low-cardinality telemetry name for this composition.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Appends a processor to the head-of-request chain.
    pub fn with_processor(mut self, processor: Arc<dyn Processor<S>>) -> Self {
        self.processors.push(processor);
        self
    }

    /// Appends a classifier to the cascade.
    pub fn with_classifier(mut self, classifier: Arc<dyn Classifier<S>>) -> Self {
        self.classifiers.push(classifier);
        self
    }
    /// Executes the processor and classifier sequence for wrappers and the trait entrypoint.
    pub(crate) async fn execute(&self, driver: Driver, request: Request) -> Result<RoutingOutcome> {
        self.start_cleanup_task();
        let session = session_id(&request);
        let session_final = request
            .metadata
            .as_ref()
            .and_then(|metadata| metadata.session_final)
            == Some(true);
        let result = self.execute_session(driver, request).await;
        if session_final && let Some(session) = session.as_deref() {
            self.remove_session(session);
        }
        result
    }

    /// Starts one cleanup task on the first request handled by a stateful router.
    fn start_cleanup_task(&self) {
        let Some(states) = &self.session_states else {
            return;
        };
        let states = Arc::downgrade(states);
        self.cleanup_started.call_once(move || {
            drop(tokio::spawn(cleanup_inactive_sessions(states)));
        });
    }

    async fn execute_session(&self, driver: Driver, request: Request) -> Result<RoutingOutcome> {
        // The request is threaded mutably through the whole fold: any component may rewrite
        // it, later components see the rewrite, and the final value reaches the model.
        let mut request = request;
        let session_state = self.session_state(&request);
        let (target, served) = match session_state {
            Some(state) => {
                let mut state = state.lock().await;
                self.route(&mut state, &driver, &mut request).await?
            }
            None => {
                let mut state = S::default();
                self.route(&mut state, &driver, &mut request).await?
            }
        };

        // A classifier that already called a model — because deciding required one, and that
        // call also answers the turn — hands its response back here, so the turn is not paid
        // for twice.
        // Nothing reads it on the way out: streamed or buffered, it reaches the caller
        // untouched.
        match served {
            Some(response) => Ok(RoutingOutcome::answered(target, request, response)),
            None => {
                let fallback_models = self.fallbacks(&target);
                Ok(RoutingOutcome::route_to(target, fallback_models, request))
            }
        }
    }

    /// Drops retained routing state once the host marks a session complete.
    fn remove_session(&self, session: &str) {
        if let Some(states) = &self.session_states {
            states.lock().remove(session);
        }
    }

    /// Every configured target other than the selection, in fallback order.
    fn fallbacks(&self, target: &ModelId) -> Vec<ModelId> {
        self.targets
            .iter()
            .filter(|candidate| *candidate != target)
            .cloned()
            .collect()
    }

    /// Returns this request's retained state without holding the registry lock.
    fn session_state(&self, request: &Request) -> Option<Arc<AsyncMutex<S>>> {
        let states = self.session_states.as_ref()?;
        let session_id = session_id(request)?;
        let mut states = states.lock();
        let now = Instant::now();
        let session = states.entry(session_id).or_insert_with(|| SessionState {
            state: Arc::new(AsyncMutex::new(S::default())),
            last_accessed: now,
        });
        session.last_accessed = now;
        Some(Arc::clone(&session.state))
    }

    async fn route(
        &self,
        state: &mut S,
        driver: &Driver,
        request: &mut Request,
    ) -> Result<(ModelId, Option<Response>)> {
        // 1. Processor chain accumulates request-side facts into the composition's state.
        for processor in &self.processors {
            processor.process(state, Event::Request(request)).await?;
        }

        // 2. Fall through the cascade: the first classifier to score decides (argmax). The
        //    per-request driver is offered to each — driver-backed classifiers use it.
        let mut routed = None;
        for classifier in &self.classifiers {
            let (scores, response) = classifier.score(state, request, Some(driver)).await?;
            if let Some(score) = scores.argmax(false)? {
                // Only the deciding classifier's response answers the turn; an abstaining
                // classifier selected nothing for it to be the answer to.
                routed = Some((score, Arc::clone(classifier), response));
                break;
            }
        }
        let Some((score, deciding, served)) = routed else {
            return Err(LibsyError::AlgorithmError {
                message: "every classifier abstained".to_string(),
            });
        };

        // 3. Resolve the target and log the choice.
        algorithm::ensure_model_is_target(&self.targets, &score.target)?;
        let target = score.target.clone();
        // A fallback or affinity-reuse decider carries no tier of its own, so
        // resolve it across the cascade rather than logging it as None.
        let tier = deciding.routing_tier(&target).or_else(|| {
            self.classifiers
                .iter()
                .find_map(|c| c.routing_tier(&target))
        });
        tracing::info!(algorithm=self.name, target=%score.target, confidence=score.confidence, tier = ?tier, "Model selected");

        // 4. Post-decision replay: every processor sees the selection so stateful ones
        //    can bind it, and may rewrite the outbound request (e.g. add a target prompt).
        for processor in &self.processors {
            let event = Event::Decision {
                request,
                selected_model_id: &target,
            };
            processor.process(state, event).await?;
        }

        Ok((target, served))
    }
}

async fn cleanup_inactive_sessions<S>(states: Weak<SessionStates<S>>)
where
    S: Send + 'static,
{
    let start = tokio::time::Instant::now() + SESSION_CLEANUP_INTERVAL;
    let mut interval = tokio::time::interval_at(start, SESSION_CLEANUP_INTERVAL);
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    loop {
        interval.tick().await;
        let Some(states) = states.upgrade() else {
            break;
        };
        remove_inactive_sessions(&states, Instant::now(), SESSION_STATE_TTL);
    }
}

fn remove_inactive_sessions<S>(states: &SessionStates<S>, now: Instant, ttl: Duration) {
    states.lock().retain(|_, session| {
        // The registry owns one strong reference. Any others belong to active requests,
        // which must keep sharing this state until their routing work finishes.
        Arc::strong_count(&session.state) > 1
            || now.saturating_duration_since(session.last_accessed) < ttl
    });
}

fn session_id(request: &Request) -> Option<String> {
    request
        .metadata
        .as_ref()?
        .session_id
        .as_deref()
        .filter(|id| !id.is_empty())
        .map(str::to_string)
}

#[async_trait]
impl<S> Algorithm for FallThrough<S>
where
    S: Default + Send + 'static,
{
    fn name(&self) -> &str {
        &self.name
    }

    async fn route(self: Arc<Self>, driver: Driver, request: Request) -> Result<RoutingOutcome> {
        self.execute(driver, request).await
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::util::prompts;
    use crate::core::classifier::Classification;
    use crate::{SystemPromptProcessor, TargetPrompts};

    use crate::core::testing::{Serve, echo, reply, test_drive};
    use switchyard_protocol::{LlmRequest, Message, Metadata, Role, completion_text, text_request};

    #[derive(Debug, thiserror::Error)]
    #[error("{0}")]
    struct TestError(&'static str);

    fn test_error(message: &'static str) -> LibsyError {
        LibsyError::external("test", TestError(message))
    }

    // --- fixtures ----------------------------------------------------------------------

    /// Echoes the routed model name, capturing the request it was handed so a test can
    /// assert on what actually reached the model.
    fn capturing(into: Arc<Mutex<Option<Request>>>) -> impl Serve {
        move |target: ModelId, request: Request| {
            let into = Arc::clone(&into);
            async move {
                *into.lock() = Some(request);
                Ok(reply(target))
            }
        }
    }

    const CAPABLE_PROMPT: &str = "diagnose before you edit";
    const EFFICIENT_PROMPT: &str = "follow the settled plan";
    const NOTE: &str = "the previous model was stalling";

    /// One model call as the prompt and note tests observe it.
    #[derive(Clone, Debug, Default)]
    struct RecordedCall {
        target: String,
        messages: Vec<String>,
        instructions: Vec<String>,
    }

    /// Captures the prompt-bearing request that reached the selected target.
    #[derive(Default)]
    struct PromptRecorder(Mutex<Option<RecordedCall>>);

    impl PromptRecorder {
        fn serve(self: &Arc<Self>) -> impl Serve {
            let recorder = Arc::clone(self);
            move |target: ModelId, request: Request| {
                let recorder = Arc::clone(&recorder);
                async move {
                    *recorder.0.lock() = Some(RecordedCall {
                        target: target.to_string(),
                        messages: request
                            .llm_request
                            .messages
                            .iter()
                            .filter_map(|message| message.text_content("|"))
                            .collect(),
                        instructions: request
                            .llm_request
                            .instructions
                            .iter()
                            .filter_map(|block| block.content.iter().find_map(text_of))
                            .collect(),
                    });
                    Ok(reply(target))
                }
            }
        }
    }

    fn text_of(block: &switchyard_protocol::ContentBlock) -> Option<String> {
        match block {
            switchyard_protocol::ContentBlock::Text { text } => Some(text.clone()),
            _ => None,
        }
    }

    fn target_set(names: &[&str]) -> Vec<ModelId> {
        names.iter().map(|name| ModelId::from(*name)).collect()
    }

    fn target_prompts() -> TargetPrompts {
        TargetPrompts::default()
            .with("capable", CAPABLE_PROMPT)
            .with("efficient", EFFICIENT_PROMPT)
    }

    /// Routes one turn on a prompt test cascade and returns the recorded model call.
    async fn routed_prompt_call(
        recorder: &Arc<PromptRecorder>,
        router: FallThrough,
    ) -> Result<RecordedCall> {
        test_drive(
            Arc::new(router),
            Request {
                llm_request: text_request(Some("auto".to_string()), "fix the build"),
                raw_request: None,
                metadata: None,
            },
            recorder.serve(),
        )
        .await?;
        let call = recorder.0.lock().take();
        match call {
            Some(call) => Ok(call),
            None => panic!("the model was never called"),
        }
    }

    /// A prompt cascade that always routes to `target`.
    fn prompt_router(target: &str, prompts: TargetPrompts) -> FallThrough {
        FallThrough::new(target_set(&["capable", "efficient"]))
            .with_processor(Arc::new(SystemPromptProcessor::new(prompts)))
            .with_classifier(Arc::new(DefaultTarget::new(target)))
    }

    /// A classifier that emits fixed scores (empty = abstain).
    struct FixedClassifier(Vec<Score>);

    #[async_trait]
    impl Classifier for FixedClassifier {
        async fn score(
            &self,
            _state: &mut (),
            _request: &mut Request,
            _driver: Option<&Driver>,
        ) -> Result<(Classification, Option<Response>)> {
            Ok((
                Classification::Scores(
                    self.0
                        .iter()
                        .map(|s| Score {
                            confidence: s.confidence,
                            target: s.target.clone(),
                        })
                        .collect(),
                ),
                None,
            ))
        }
    }

    fn score(target: &str, confidence: f64) -> Score {
        Score {
            confidence,
            target: ModelId::from(target),
        }
    }

    fn fixed(scores: Vec<Score>) -> Arc<dyn Classifier> {
        Arc::new(FixedClassifier(scores))
    }

    fn request() -> Request {
        Request {
            llm_request: LlmRequest {
                model: Some("auto".to_string()),
                messages: vec![Message::text(Role::User, "hi")],
                ..LlmRequest::default()
            },
            raw_request: None,
            metadata: Some(Metadata {
                session_id: Some("session-1".to_string()),
                ..Metadata::default()
            }),
        }
    }

    /// Drives a shared router with one request, returning the completion text and selection.
    async fn run_request<S>(
        router: &Arc<FallThrough<S>>,
        request: Request,
        serve: impl Serve,
    ) -> Result<(String, ModelId)>
    where
        S: Default + Send + 'static,
    {
        let (selected_model, response) = test_drive(router.clone(), request, serve).await?;
        let text = response
            .llm_response
            .into_agg()
            .await
            .map(|agg| completion_text(&agg))
            .map_err(|error| LibsyError::external("aggregating fall-through response", error))?;
        Ok((text, selected_model))
    }

    /// Drives a shared router through one turn in the default test session.
    async fn run_turn<S>(
        router: &Arc<FallThrough<S>>,
        serve: impl Serve,
    ) -> Result<(String, ModelId)>
    where
        S: Default + Send + 'static,
    {
        run_request(router, request(), serve).await
    }

    /// Drives a fresh router through one turn with a specific `serve`.
    async fn run_with(router: FallThrough, serve: impl Serve) -> Result<(String, ModelId)> {
        run_turn(&Arc::new(router), serve).await
    }

    // --- tests -------------------------------------------------------------------------

    #[tokio::test]
    async fn each_target_gets_its_own_prompt() -> Result<()> {
        for (target, expected) in [("capable", CAPABLE_PROMPT), ("efficient", EFFICIENT_PROMPT)] {
            let recorder = Arc::new(PromptRecorder::default());
            let call =
                routed_prompt_call(&recorder, prompt_router(target, target_prompts())).await?;
            assert_eq!(call.target, target);
            assert_eq!(call.instructions, vec![expected.to_string()]);
        }
        Ok(())
    }

    #[tokio::test]
    async fn a_target_with_no_prompt_is_left_untouched() -> Result<()> {
        let recorder = Arc::new(PromptRecorder::default());
        let only_capable = TargetPrompts::default().with("capable", CAPABLE_PROMPT);

        let call = routed_prompt_call(&recorder, prompt_router("efficient", only_capable)).await?;

        assert!(
            call.instructions.is_empty(),
            "one target's prompt must not leak onto another: {:?}",
            call.instructions
        );
        Ok(())
    }

    #[tokio::test]
    async fn the_prompt_follows_the_target_whichever_classifier_picked_it() -> Result<()> {
        // The first classifier abstains, so the second decides; the prompt follows the
        // target the cascade settled on rather than the classifier that named it.
        struct Abstains;

        #[async_trait]
        impl Classifier for Abstains {
            async fn score(
                &self,
                _state: &mut (),
                _request: &mut Request,
                _driver: Option<&Driver>,
            ) -> Result<(Classification, Option<Response>)> {
                Ok((Classification::Ambiguous(Vec::new()), None))
            }
        }

        let recorder = Arc::new(PromptRecorder::default());
        let router = FallThrough::new(target_set(&["capable", "efficient"]))
            .with_processor(Arc::new(SystemPromptProcessor::new(target_prompts())))
            .with_classifier(Arc::new(Abstains))
            .with_classifier(Arc::new(DefaultTarget::new("capable")));

        let call = routed_prompt_call(&recorder, router).await?;

        assert_eq!(call.target, "capable");
        assert_eq!(call.instructions, vec![CAPABLE_PROMPT.to_string()]);
        Ok(())
    }

    #[tokio::test]
    async fn a_note_reaches_the_model_in_the_conversation() -> Result<()> {
        // Appends a note to every outbound request, the way a router would on a turn it
        // wants to explain.
        struct Noting;

        #[async_trait]
        impl Processor for Noting {
            async fn process(&self, _state: &mut (), event: Event<'_>) -> Result<()> {
                if let Event::Decision { request, .. } = event {
                    prompts::append_note(request, NOTE);
                }
                Ok(())
            }
        }

        let recorder = Arc::new(PromptRecorder::default());
        let router = FallThrough::new(target_set(&["capable", "efficient"]))
            .with_processor(Arc::new(Noting))
            .with_classifier(Arc::new(DefaultTarget::new("capable")));

        let call = routed_prompt_call(&recorder, router).await?;

        assert_eq!(call.messages, vec![format!("fix the build|{NOTE}")]);
        assert!(call.instructions.is_empty(), "a note is not an instruction");
        Ok(())
    }

    #[tokio::test]
    async fn selected_target_leads_the_ordered_candidate_list() -> Result<()> {
        use futures::StreamExt;

        let router = Arc::new(
            FallThrough::<()>::new(target_set(&["weak", "mid", "strong"]))
                .with_classifier(fixed(vec![score("mid", 0.9)])),
        );
        let stream = router.run_stream(request());
        tokio::pin!(stream);
        while let Some(step) = stream.next().await {
            if let crate::Step::Done(outcome) = step? {
                assert_eq!(outcome.selected_model_id, ModelId::from("mid"));
                assert_eq!(outcome.fallback_models, target_set(&["weak", "strong"]));
                assert_eq!(outcome.request.llm_request.model.as_deref(), Some("mid"));
                assert!(outcome.response.is_none());
                return Ok(());
            }
        }
        Err(test_error("expected a Done step"))
    }

    #[tokio::test]
    async fn argmax_picks_the_highest_confidence_target() -> Result<()> {
        let router = FallThrough::<()>::new(target_set(&["strong", "weak"]))
            .with_classifier(fixed(vec![score("weak", 0.2), score("strong", 0.9)]));
        let (model, selected_model) = run_with(router, echo()).await?;
        assert_eq!(model, "strong");
        assert_eq!(selected_model, "strong");
        Ok(())
    }

    #[tokio::test]
    async fn falls_through_the_first_abstaining_classifier() -> Result<()> {
        // First classifier abstains (empty); the second decides.
        let router = FallThrough::<()>::new(target_set(&["strong", "weak"]))
            .with_classifier(fixed(vec![]))
            .with_classifier(fixed(vec![score("weak", 1.0)]));
        let (model, _) = run_with(router, echo()).await?;
        assert_eq!(model, "weak");
        Ok(())
    }

    #[tokio::test]
    async fn first_deciding_classifier_wins_the_cascade() -> Result<()> {
        // The first classifier decides; the second is never consulted.
        let router = FallThrough::<()>::new(target_set(&["strong", "weak"]))
            .with_classifier(fixed(vec![score("strong", 0.6)]))
            .with_classifier(fixed(vec![score("weak", 1.0)]));
        let (model, _) = run_with(router, echo()).await?;
        assert_eq!(model, "strong");
        Ok(())
    }

    #[tokio::test]
    async fn all_abstaining_is_an_error() -> Result<()> {
        let router =
            FallThrough::<()>::new(target_set(&["strong", "weak"])).with_classifier(fixed(vec![]));
        let error = run_with(router, echo())
            .await
            .err()
            .ok_or_else(|| test_error("expected classifiers to abstain"))?;
        assert!(matches!(
            error,
            LibsyError::AlgorithmError { message } if message == "every classifier abstained"
        ));
        Ok(())
    }

    #[tokio::test]
    async fn classifiers_receive_the_per_request_driver() -> Result<()> {
        // A classifier that only decides when handed a driver — proving the cascade offers
        // the per-request driver to every classifier (driver-backed ones need it).
        struct NeedsDriver;

        #[async_trait]
        impl Classifier for NeedsDriver {
            async fn score(
                &self,
                _state: &mut (),
                _request: &mut Request,
                driver: Option<&Driver>,
            ) -> Result<(Classification, Option<Response>)> {
                match driver {
                    Some(_) => Ok((Classification::Scores(vec![score("strong", 1.0)]), None)),
                    None => Err(test_error("expected a driver")),
                }
            }
        }

        let router = FallThrough::<()>::new(target_set(&["strong", "weak"]))
            .with_classifier(Arc::new(NeedsDriver));
        let (model, _) = run_with(router, echo()).await?;
        assert_eq!(model, "strong");
        Ok(())
    }

    #[tokio::test]
    async fn processor_observes_request_then_decision() -> Result<()> {
        use parking_lot::Mutex;

        // Records which event kinds it saw, proving the replay order: the inbound
        // request, then the routing decision (which carries the request to the model).
        struct RecordingProcessor(Arc<Mutex<Vec<&'static str>>>);

        #[async_trait]
        impl Processor for RecordingProcessor {
            async fn process(&self, _state: &mut (), event: Event<'_>) -> Result<()> {
                let kind = match event {
                    Event::Request(_) => "request",
                    Event::Decision { .. } => "decision",
                    _ => "other",
                };
                self.0.lock().push(kind);
                Ok(())
            }
        }

        let seen = Arc::new(Mutex::new(Vec::new()));
        let router = FallThrough::<()>::new(target_set(&["strong", "weak"]))
            .with_processor(Arc::new(RecordingProcessor(seen.clone())))
            .with_classifier(fixed(vec![score("strong", 1.0)]));
        run_with(router, echo()).await?;

        assert_eq!(*seen.lock(), vec!["request", "decision"]);
        Ok(())
    }

    #[tokio::test]
    async fn a_rewrite_propagates_down_the_chain_and_into_the_model_call() -> Result<()> {
        use parking_lot::Mutex;

        /// Appends a marker message to the request it observes.
        struct Appender(&'static str);

        #[async_trait]
        impl Processor for Appender {
            async fn process(&self, _state: &mut (), event: Event<'_>) -> Result<()> {
                if let Event::Request(request) = event {
                    request
                        .llm_request
                        .messages
                        .push(Message::text(Role::User, self.0));
                }
                Ok(())
            }
        }

        /// Records the marker trail it was handed, then appends its own — proving the
        /// classifier scored the processors' rewrite rather than the original request.
        struct TrailClassifier(Arc<Mutex<Vec<String>>>);

        #[async_trait]
        impl Classifier for TrailClassifier {
            async fn score(
                &self,
                _state: &mut (),
                request: &mut Request,
                _driver: Option<&Driver>,
            ) -> Result<(Classification, Option<Response>)> {
                *self.0.lock() = request
                    .llm_request
                    .messages
                    .iter()
                    .filter_map(|message| message.text_content(""))
                    .collect();
                request
                    .llm_request
                    .messages
                    .push(Message::text(Role::User, "classifier"));
                Ok((Classification::Scores(vec![score("strong", 1.0)]), None))
            }
        }

        let seen_by_classifier = Arc::new(Mutex::new(Vec::new()));
        let seen_by_model = Arc::new(Mutex::new(None));
        let targets = target_set(&["strong"]);
        let router = FallThrough::new(targets)
            .with_processor(Arc::new(Appender("first")))
            .with_processor(Arc::new(Appender("second")))
            .with_classifier(Arc::new(TrailClassifier(seen_by_classifier.clone())));

        run_turn(&Arc::new(router), capturing(seen_by_model.clone())).await?;

        // The classifier saw both processors' edits, in chain order, on top of the original.
        assert_eq!(*seen_by_classifier.lock(), vec!["hi", "first", "second"]);

        // ...and the request that reached the model carries the classifier's edit too.
        let routed = seen_by_model
            .lock()
            .take()
            .ok_or_else(|| test_error("the model was never called"))?;
        let trail: Vec<String> = routed
            .llm_request
            .messages
            .iter()
            .filter_map(|message| message.text_content(""))
            .collect();
        assert_eq!(trail, vec!["hi", "first", "second", "classifier"]);
        Ok(())
    }

    #[tokio::test]
    async fn state_is_shared_within_a_session_and_isolated_between_sessions() -> Result<()> {
        #[derive(Default)]
        struct TurnState {
            count: u32,
        }

        // Increments the session turn count on every request.
        struct CountingProcessor;

        #[async_trait]
        impl Processor<TurnState> for CountingProcessor {
            async fn process(&self, state: &mut TurnState, event: Event<'_>) -> Result<()> {
                if let Event::Request(_) = event {
                    state.count += 1;
                }
                Ok(())
            }
        }

        // Routes weak on a session's first turn and strong on later turns.
        struct ThresholdClassifier;

        #[async_trait]
        impl Classifier<TurnState> for ThresholdClassifier {
            async fn score(
                &self,
                state: &mut TurnState,
                _request: &mut Request,
                _driver: Option<&Driver>,
            ) -> Result<(Classification, Option<Response>)> {
                let target = if state.count >= 2 { "strong" } else { "weak" };
                Ok((Classification::Scores(vec![score(target, 1.0)]), None))
            }
        }

        let router = Arc::new(
            FallThrough::<TurnState>::new_with_state(target_set(&["strong", "weak"]))
                .with_processor(Arc::new(CountingProcessor))
                .with_classifier(Arc::new(ThresholdClassifier)),
        );

        let (turn1, _) = run_turn(&router, echo()).await?;
        let (turn2, _) = run_turn(&router, echo()).await?;
        let final_request = Request {
            metadata: Some(Metadata {
                session_id: Some("session-1".to_string()),
                session_final: Some(true),
                ..Metadata::default()
            }),
            ..request()
        };
        let (final_turn, _) = run_request(&router, final_request, echo()).await?;
        let (restarted_session, _) = run_turn(&router, echo()).await?;
        let (second_session, _) = run_request(
            &router,
            Request {
                metadata: Some(Metadata {
                    session_id: Some("session-2".to_string()),
                    ..Metadata::default()
                }),
                ..request()
            },
            echo(),
        )
        .await?;
        let anonymous = Request {
            metadata: None,
            ..request()
        };
        let (anonymous1, _) = run_request(&router, anonymous.clone(), echo()).await?;
        let (anonymous2, _) = run_request(&router, anonymous, echo()).await?;

        assert_eq!(turn1, "weak");
        assert_eq!(turn2, "strong");
        assert_eq!(final_turn, "strong");
        assert_eq!(restarted_session, "weak");
        assert_eq!(second_session, "weak");
        assert_eq!(anonymous1, "weak");
        assert_eq!(anonymous2, "weak");
        Ok(())
    }

    #[tokio::test]
    async fn final_session_is_removed_when_routing_fails() {
        let router = Arc::new(FallThrough::<u32>::new_with_state(target_set(&["strong"])));
        let final_request = Request {
            metadata: Some(Metadata {
                session_id: Some("session-1".to_string()),
                session_final: Some(true),
                ..Metadata::default()
            }),
            ..request()
        };

        let result = test_drive(router.clone(), final_request, echo()).await;

        assert!(matches!(result, Err(LibsyError::AlgorithmError { .. })));
        let states = router
            .session_states
            .as_ref()
            .expect("stateful router has a session registry")
            .lock();
        assert!(!states.contains_key("session-1"));
    }

    #[test]
    fn cleanup_removes_only_inactive_idle_sessions() {
        let router = FallThrough::<u32>::new_with_state(target_set(&["strong"]));
        let _active_state = router
            .session_state(&request()) // session-1
            .expect("session state was inserted");
        let inactive_request = Request {
            metadata: Some(Metadata {
                session_id: Some("session-2".to_string()),
                ..Metadata::default()
            }),
            ..request()
        };
        drop(router.session_state(&inactive_request));

        let states = router
            .session_states
            .as_ref()
            .expect("stateful router has a session registry");
        let now = Instant::now() + SESSION_STATE_TTL + Duration::from_secs(1);

        remove_inactive_sessions(states, now, SESSION_STATE_TTL);

        let states = states.lock();
        assert!(states.contains_key("session-1"));
        assert!(!states.contains_key("session-2"));
    }

    // ------------------------------------------------------------------
    // REALIGN-1B disposable proof: authority-safe candidate set + fallback contract.
    // ------------------------------------------------------------------
    #[cfg(test)]
    mod authority_safe_proof {
    use super::*;
    use crate::core::classifier::{Classification, Score};
    use std::collections::BTreeSet;
    use std::sync::Arc;
    use switchyard_protocol::{LlmClientError, text_request};

    // ---------------- GENERIC FACT PLANE (provider-agnostic) --------------------

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Thinking {
        Thinking,
        NonThinking,
        Both,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum PressureTier {
        Normal,
        Conserve,
        StrongConserve,
    }

    /// Generic target fact: facts + capability + policy pressure. No routing decisions.
    #[derive(Clone, Debug)]
    pub struct TargetFact {
        pub target_id: &'static str,
        pub available: bool,
        pub qualified: bool,
        pub ready: bool,
        pub thinking: Thinking,
        pub pressure: PressureTier,
        pub service_class: &'static str,
    }

    #[derive(Clone, Debug, Default)]
    pub struct FactState {
        pub targets: Vec<TargetFact>,
    }

    /// Per-request capability contract supplied by the caller/route.
    #[derive(Clone, Copy, Debug)]
    pub struct RequestContract {
        pub service_class: &'static str,
        pub thinking: Thinking,
        pub pinned: Option<&'static str>,
    }

    // ---------------- GENERIC FACT INJECTOR (Processor<FactState>) --------------

    #[derive(Clone, Debug)]
    pub struct FactInjector {
        pub fixture: fn(&'static str) -> TargetFact,
        pub ids: Vec<&'static str>,
    }

    #[async_trait::async_trait]
    impl Processor<FactState> for FactInjector {
        async fn process(&self, state: &mut FactState, _e: Event<'_>) -> crate::Result<()> {
            state.targets.clear();
            state
                .targets
                .extend(self.ids.iter().copied().map(self.fixture));
            Ok(())
        }
    }

    // ---------------- GENERIC ADMISSIBILITY CLASSIFIER --------------------------

    pub struct AdmissibleClassifier {
        pub contract: RequestContract,
    }

    impl AdmissibleClassifier {
        /// Admissible = available + service-class + qualified + thinking-compatible +
        /// ready-now. Readiness is a SELECTION gate here, NOT a permanent ineligibility
        /// (RE-1 will allow a transition-cost override for qualified-not-ready targets).
        pub fn admissible_target(&self, t: &TargetFact) -> bool {
            if let Some(pinned) = self.contract.pinned {
                return t.target_id == pinned && t.available && t.qualified;
            }
            if !t.available || !t.qualified || !t.ready {
                return false;
            }
            if t.service_class != self.contract.service_class {
                return false;
            }
            matches!(
                (self.contract.thinking, t.thinking),
                (Thinking::Thinking, Thinking::Thinking | Thinking::Both)
                    | (Thinking::NonThinking, Thinking::NonThinking | Thinking::Both)
                    | (Thinking::Both, _)
            )
        }

        pub fn admissible(&self, state: &FactState) -> BTreeSet<String> {
            state
                .targets
                .iter()
                .filter(|t| self.admissible_target(t))
                .map(|t| t.target_id.to_string())
                .collect()
        }
    }

    #[async_trait::async_trait]
    impl Classifier<FactState> for AdmissibleClassifier {
        async fn score(
            &self,
            state: &mut FactState,
            _req: &mut Request,
            _driver: Option<&Driver>,
        ) -> crate::Result<(Classification, Option<Response>)> {
            let mut scores = Vec::new();
            for t in state.targets.iter() {
                if !self.admissible_target(t) {
                    continue;
                }
                let mut conf = 0.6f64;
                match t.pressure {
                    PressureTier::Normal => {}
                    PressureTier::Conserve => conf += 0.1,
                    PressureTier::StrongConserve => conf -= 0.25,
                }
                scores.push(Score {
                    confidence: conf,
                    target: ModelId::from(t.target_id),
                });
            }
            Ok((Classification::Scores(scores), None))
        }
    }

    // ---------------- STOCK-FALLBACK GAP PROOFS (deliverables 2, 3) -------------

    fn req() -> Request {
        Request {
            llm_request: text_request(Some("none".into()), "x"),
            raw_request: None,
            metadata: None,
        }
    }

    /// Drive a stock FallThrough and return (selected, fallback list).
    async fn drive_stock(
        targets: &'static [&'static str],
        contract: RequestContract,
        with_default: bool,
        fixture: fn(&'static str) -> TargetFact,
    ) -> crate::Result<(String, Vec<String>)> {
        let ids: &[&'static str] = targets;
        let target_ids: Vec<ModelId> = targets.iter().map(|s| ModelId::from(*s)).collect();
        let mut router = FallThrough::<FactState>::new_with_state(target_ids)
            .with_name("stock-gap")
            .with_processor(Arc::new(FactInjector { fixture, ids: ids.to_vec() }))
            .with_classifier(Arc::new(AdmissibleClassifier { contract }));
        if with_default {
            // stock terminal: never abstains
            router = router.with_classifier(Arc::new(DefaultTarget::new(ModelId::from(
                targets[targets.len() - 1],
            ))));
        }
        let outcome = crate::drive(Arc::new(router), req(), |_c| {
            Box::pin(async { Ok(()) })
        })
        .await?;
        let fb: Vec<String> = outcome
            .fallback_models
            .iter()
            .map(|m| m.to_string())
            .collect();
        Ok((outcome.selected_model_id.to_string(), fb))
    }

    // ---------------- SCENARIOS (synthetic fixtures, no real provider) ----------

    pub mod fx {
        use super::*;
        pub const LUNA: &str = "openai/gpt-5.6-luna";
        pub const SOL: &str = "openai/gpt-5.6-sol";
        pub const FLASH: &str = "deepseek/flash";
        pub const LOCAL_QR: &str = "local/qwen-fast";
        pub const LOCAL_NOTREADY: &str = "local/notready";
        pub const LOCAL_UNQUAL: &str = "local/unqualified";

        pub fn cloud_all_thinking(id: &'static str) -> TargetFact {
            TargetFact {
                target_id: id,
                available: true,
                qualified: true,
                ready: true,
                thinking: Thinking::Thinking,
                pressure: PressureTier::Normal,
                service_class: "generation",
            }
        }
        pub fn sol_unavailable(id: &'static str) -> TargetFact {
            TargetFact {
                target_id: id,
                available: id != SOL,
                qualified: true,
                ready: true,
                thinking: Thinking::Thinking,
                pressure: PressureTier::Normal,
                service_class: "generation",
            }
        }
        pub fn mixed_thinking(id: &'static str) -> TargetFact {
            TargetFact {
                target_id: id,
                available: true,
                qualified: true,
                ready: true,
                thinking: if id.contains("openai") {
                    Thinking::Thinking
                } else {
                    Thinking::NonThinking
                },
                pressure: if id.contains("luna") {
                    PressureTier::StrongConserve
                } else {
                    PressureTier::Normal
                },
                service_class: "generation",
            }
        }
        pub fn local(id: &'static str) -> TargetFact {
            TargetFact {
                target_id: id,
                available: true,
                qualified: id != LOCAL_UNQUAL,
                ready: id != LOCAL_NOTREADY,
                thinking: Thinking::NonThinking,
                pressure: PressureTier::Normal,
                service_class: "generation",
            }
        }
    }

    // ---- CONFIRM the stock fallback-escape (gap 1) ------------------------------

    #[tokio::test]
    async fn confirms_stock_fallback_escapes_unavailable_target() -> crate::Result<()> {
        // Targets = full static fleet. Sol is unavailable. Classifier excludes it at
        // primary; stock fallbacks() = every static target except the selected one, so
        // the unavailable Sol target WILL appear in the terminal fallback list. This is
        // the authority gap: a gate-rejected target escapes into fallback.
        let (sel, fb) = drive_stock(
            &[fx::LUNA, fx::SOL, fx::FLASH],
            RequestContract {
                service_class: "generation",
                thinking: Thinking::Both,
                pinned: None,
            },
            true, // with DefaultTarget
            fx::sol_unavailable,
        )
        .await?;
        // CONFIRMED FINDING: the unavailable Sol appears in the stock terminal fallback
        // list. This documents the gap positively (the assertion asserts the leak exists),
        // so the proof records the stock behavior rather than merely failing.
        assert_ne!(sel, fx::SOL, "sol unavailable so must not be primary");
        assert!(
            fb.contains(&fx::SOL.to_string()),
            "expected stock fallback to leak unavailable target; got {fb:?}"
        );
        Ok(())
    }

    #[tokio::test]
    async fn confirms_stock_defaulttarget_fails_open() -> crate::Result<()> {
        // All targets unavailable -> classifier abstains -> DefaultTarget picks flash
        // (fail open), NOT a no-eligible-target error.
        let out = drive_stock(
            &[fx::LUNA, fx::SOL, fx::FLASH],
            RequestContract {
                service_class: "generation",
                thinking: Thinking::Both,
                pinned: None,
            },
            true, // DefaultTarget present
            |_| TargetFact {
                target_id: "x",
                available: false,
                qualified: true,
                ready: true,
                thinking: Thinking::Thinking,
                pressure: PressureTier::Normal,
                service_class: "generation",
            },
        )
        .await;
        // default target rescued an empty admissible set => NOT a no-eligible result
        let (sel, _) = out.unwrap();
        assert_eq!(sel, fx::FLASH, "DefaultTarget failed OPEN instead of no-eligible");
        Ok(())
    }

    // ---- AUTHORITY-SAFE PATTERN (deliverables 4, 8, 9, 10) ---------------------

    /// Build an authority-safe composition:
    ///   admissible := AdmissibleClassifier{contract}.admissible(state)
    ///   targets    := admissible            (so stock fallbacks() = admissible-minus-selected)
    ///   DefaultTarget OMITTED for the no-eligible case.
    async fn drive_safe(
        ids: &'static [&'static str],
        contract: RequestContract,
        fixture: fn(&'static str) -> TargetFact,
    ) -> crate::Result<(String, Vec<String>)> {
        // 1. compute admissible
        let mut st = FactState::default();
        FactInjector { fixture, ids: ids.to_vec() }
            .process(&mut st, Event::Request(&mut req()))
            .await?;
        let admissible = AdmissibleClassifier { contract }.admissible(&st);
        if admissible.is_empty() {
            // explicit no-eligible-target (the authority-safe semantic)
            return Err(LibsyError::AlgorithmError {
                message: "no eligible target for this request".to_string(),
            });
        }
        let targets: Vec<ModelId> =
            admissible.iter().map(|s| ModelId::from(s.as_str())).collect();

        // 2. build FallThrough with targets = admissible, classifier only (NO DefaultTarget)
        let mut router = FallThrough::<FactState>::new_with_state(targets.clone())
            .with_name("authority-safe")
            .with_processor(Arc::new(FactInjector { fixture, ids: ids.to_vec() }))
            .with_classifier(Arc::new(AdmissibleClassifier { contract }));
        if let Some(pinned) = contract.pinned {
            // pinned + no caller fallback authorization => fallback empty; add a terminal
            // that routes to the pin only if it is admissible (it is, by construction).
            router = router.with_classifier(Arc::new(DefaultTarget::new(ModelId::from(pinned))));
        } else {
            // closed cascade to the top admissible target (still within the admissible set)
            let top = targets.first().cloned().unwrap();
            router = router.with_classifier(Arc::new(DefaultTarget::new(top)));
        }
        let outcome = crate::drive(Arc::new(router), req(), |_c| {
            Box::pin(async { Ok(()) })
        })
        .await?;
        let fb: Vec<String> = outcome
            .fallback_models
            .iter()
            .map(|m| m.to_string())
            .collect();
        Ok((outcome.selected_model_id.to_string(), fb))
    }

    use fx::*;

    #[tokio::test]
    async fn pin_sol_no_fallback() -> crate::Result<()> {
        // A: pin sol; fallback must be empty (no luna/deepseek/local substitution).
        let (sel, fb) = drive_safe(
            &[LUNA, SOL, FLASH],
            RequestContract {
                service_class: "generation",
                thinking: Thinking::Both,
                pinned: Some(SOL),
            },
            cloud_all_thinking,
        )
        .await?;
        assert_eq!(sel, SOL);
        assert!(fb.is_empty(), "pin must have empty fallback, got {fb:?}");
        Ok(())
    }

    #[tokio::test]
    async fn constrained_thinking_no_nonthinking_escape() -> crate::Result<()> {
        // B: thinking envelope -> non-thinking deepseek cannot appear primary or fallback.
        let (sel, fb) = drive_safe(
            &[LUNA, FLASH],
            RequestContract {
                service_class: "generation",
                thinking: Thinking::Thinking,
                pinned: None,
            },
            mixed_thinking,
        )
        .await?;
        assert!(!sel.contains("deepseek"), "non-thinking selected: {sel}");
        assert!(
            fb.iter().all(|x| !x.contains("deepseek")),
            "non-thinking escaped fallback: {fb:?}"
        );
        Ok(())
    }

    #[tokio::test]
    async fn constrained_nonthinking_no_thinking_escape() -> crate::Result<()> {
        // C: non-thinking envelope -> thinking OpenAI cannot appear primary or fallback.
        let (sel, fb) = drive_safe(
            &[LUNA, FLASH],
            RequestContract {
                service_class: "generation",
                thinking: Thinking::NonThinking,
                pinned: None,
            },
            mixed_thinking,
        )
        .await?;
        assert!(sel.contains("deepseek"), "wrong primary: {sel}");
        assert!(
            fb.iter().all(|x| !x.contains("openai")),
            "thinking escaped fallback: {fb:?}"
        );
        Ok(())
    }

    #[tokio::test]
    async fn unavailable_target_never_in_primary_or_fallback() -> crate::Result<()> {
        // D: unavailable sol must not appear primary or fallback.
        let (sel, fb) = drive_safe(
            &[LUNA, SOL, FLASH],
            RequestContract {
                service_class: "generation",
                thinking: Thinking::Both,
                pinned: None,
            },
            sol_unavailable,
        )
        .await?;
        assert_ne!(sel, SOL);
        assert!(
            !fb.contains(&SOL.to_string()),
            "unavailable leaked into fallback: {fb:?}"
        );
        Ok(())
    }

    #[tokio::test]
    async fn local_qualified_ready_participates_others_excluded() -> crate::Result<()> {
        // F: qualified+ready local participates; ready-but-unqualified and
        // unqualified-but-ready do not. (Readiness is a selection gate, not a permanent
        // ineligibility label.)
        let (sel, fb) = drive_safe(
            &[LOCAL_QR, LOCAL_NOTREADY, LOCAL_UNQUAL],
            RequestContract {
                service_class: "generation",
                thinking: Thinking::NonThinking,
                pinned: None,
            },
            local,
        )
        .await?;
        assert_eq!(sel, LOCAL_QR);
        assert!(fb.is_empty());
        Ok(())
    }

    #[tokio::test]
    async fn no_eligible_target_is_explicit_error_not_failopen() -> crate::Result<()> {
        // G: all unavailable -> explicit no-eligible-target error (no DefaultTarget escape).
        let out = drive_safe(
            &[LUNA, SOL, FLASH],
            RequestContract {
                service_class: "generation",
                thinking: Thinking::Both,
                pinned: None,
            },
            |_| TargetFact {
                target_id: "x",
                available: false,
                qualified: true,
                ready: true,
                thinking: Thinking::Thinking,
                pressure: PressureTier::Normal,
                service_class: "generation",
            },
        )
        .await;
        assert!(out.is_err(), "expected no-eligible-target error, got Ok");
        Ok(())
    }
    // ---- scenario H (mandatory): selected-target runtime failure -> every attempted
    // terminal fallback stays inside the admissible candidate set. -----------------

    #[tokio::test]
    async fn h_primary_failure_fallback_stays_in_admissible_set() -> crate::Result<()> {
        // Admissible set (thinking envelope): luna + flash (luna + flash available,
        // thinking-compatible). Sol is factually unavailable -> must NEVER appear in
        // primary OR any terminal fallback.
        let ids: &[&'static str] = &[LUNA, SOL, FLASH];
        let contract = RequestContract {
            service_class: "generation",
            thinking: Thinking::Thinking,
            pinned: None,
        };
        let fixture = sol_unavailable;

        // 1. compute admissible
        let mut st = FactState::default();
        FactInjector {
            fixture,
            ids: ids.to_vec(),
        }
        .process(&mut st, Event::Request(&mut req()))
        .await?;
        let admissible = AdmissibleClassifier { contract }.admissible(&st);
        assert!(!admissible.contains(&SOL.to_string()), "sol must be excluded");

        // 2. build authority-safe composition (targets = admissible set, DefaultTarget
        //    within the admissible set only)
        let targets: Vec<ModelId> =
            admissible.iter().map(|s| ModelId::from(s.as_str())).collect();
        let top = targets.first().cloned().unwrap();
        let router = FallThrough::<FactState>::new_with_state(targets.clone())
            .with_name("authority-safe-H")
            .with_processor(Arc::new(FactInjector { fixture, ids: ids.to_vec() }))
            .with_classifier(Arc::new(AdmissibleClassifier { contract }))
            .with_classifier(Arc::new(DefaultTarget::new(top)));

        // 3. drive with a serve that FAILS the primary with a fallback-eligible transport
        //    error, so the caller would advance to fallback_models.
        let outcome = crate::drive(Arc::new(router), req(), |_call| {
            Box::pin(async {
                Err(LibsyError::client_call(
                    ModelId::from("openai/gpt-5.6-luna"),
                    LlmClientError::Transport {
                        source: std::io::Error::other("simulated transport failure").into(),
                    },
                ))
            })
        })
        .await?;

        // 4. THE CONTRACT: every terminal fallback is inside the admissible set; sol and
        //    any non-thinking target can never be attempted.
        for m in &outcome.fallback_models {
            assert!(
                admissible.contains(&m.to_string()),
                "fallback {m} outside admissible set {admissible:?}"
            );
        }
        Ok(())
    }
    }
}
