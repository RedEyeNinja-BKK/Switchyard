// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Provider-aware conversation continuity.
//!
//! WHY THIS EXISTS
//!
//! A routed conversation is not portable by default. Reasoning a provider emitted
//! is either opaque ciphertext bound to that provider, or plaintext whose meaning
//! depends on the protocol it was produced under. Handing either to a different
//! provider is not a benign transformation: opaque reasoning is unverifiable
//! elsewhere, and an unfinished tool cycle is mid-protocol, so its pending call
//! belongs to the model that issued it.
//!
//! PROVENANCE IS PER ITEM, NOT PER TURN
//!
//! A single producer label for a conversation is not enough. One history can hold
//! turns produced by several providers -- a composite that crossed children, an
//! escalation to a stronger model, a session resumed after a credential change --
//! and the correct treatment differs item by item: the item produced by the target
//! itself may be replayed, while a foreign opaque item may not. Deciding per turn
//! would either discard a same-provider item that must be kept or forward a foreign
//! one that must not be.
//!
//! WHICH PROTOCOL A DECISION IS ABOUT
//!
//! A wire protocol is a pair: the caller's inbound format and the target's outbound
//! format. Stored reasoning is meaningful in terms of the OUTBOUND format that
//! produced it, because that is the shape it was generated in, so the comparison
//! here is always outbound-to-outbound. A Responses-native history replayed to a
//! Chat upstream is a protocol change even though the caller never changed
//! anything, which is what makes the second of the two known failure shapes
//! visible.
//!
//! THIS MODULE DECIDES, IT DOES NOT REWRITE
//!
//! It returns what must happen to each item. It does not edit history and it does
//! not choose how a compatible transformation is spelled -- that belongs to the
//! codec, which knows the target protocol. Enforcement is a separate change; this
//! exists so the contract can be tested and cited before anything acts on it, and
//! so every Switchyard client gets the same rules rather than each one inventing
//! its own.

use std::fmt;

/// Which provider, protocol and model produced one item of conversation history.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ItemProvenance {
    /// Configured client name that produced the item (`openrouter`, `deepseek`).
    ///
    /// The configured client, not the vendor: two targets can share a model id
    /// across two credentials, and reasoning is bound to the credential's provider.
    pub provider: String,
    /// Wire protocol the item was produced under, as its stable name.
    pub protocol: &'static str,
    /// Model id that produced the item.
    pub model: String,
}

impl ItemProvenance {
    /// Creates provenance for an item produced by `provider` over `protocol`.
    pub fn new(
        provider: impl Into<String>,
        protocol: &'static str,
        model: impl Into<String>,
    ) -> Self {
        Self {
            provider: provider.into(),
            protocol,
            model: model.into(),
        }
    }
}

/// How one item carries reasoning, which is what portability turns on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReasoningRepresentation {
    /// No reasoning on this item.
    None,
    /// Readable reasoning text. Portable in principle, but only where the target's
    /// codec can express it in its own shape; this layer never assumes it is.
    Plaintext,
    /// Ciphertext, a signature, or a provider-scoped item id. Verifiable only by
    /// the provider that issued it.
    Opaque,
}

/// Whether an item sits at a boundary a conversation may cross.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Boundary {
    /// A completed turn: nothing is in flight.
    Safe,
    /// A tool call whose result has not been returned, or a reasoning turn whose
    /// continuation the provider has not finished. Bound to its origin.
    MidCycle,
}

/// One item of history and everything needed to decide its portability.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HistoryItem {
    /// Position in the history, for reporting and for the caller to act on.
    pub index: usize,
    /// Who produced it.
    pub provenance: ItemProvenance,
    /// How it carries reasoning.
    pub reasoning: ReasoningRepresentation,
    /// Whether it is at a safe transfer boundary.
    pub boundary: Boundary,
    /// Tool call id or provider continuation id this item participates in, when
    /// it has one. Used to report which continuation a refusal concerns.
    pub continuation: Option<String>,
}

impl HistoryItem {
    /// Creates an item at a safe boundary carrying no reasoning.
    pub fn new(index: usize, provenance: ItemProvenance) -> Self {
        Self {
            index,
            provenance,
            reasoning: ReasoningRepresentation::None,
            boundary: Boundary::Safe,
            continuation: None,
        }
    }

    /// Marks how this item carries reasoning.
    pub fn with_reasoning(mut self, reasoning: ReasoningRepresentation) -> Self {
        self.reasoning = reasoning;
        self
    }

    /// Marks this item as mid-cycle, optionally naming the continuation.
    pub fn mid_cycle(mut self, continuation: impl Into<String>) -> Self {
        self.boundary = Boundary::MidCycle;
        self.continuation = Some(continuation.into());
        self
    }
}

/// The target a history is being considered for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Target {
    /// Configured client that would serve the request.
    pub provider: String,
    /// Outbound protocol the target would be driven with.
    pub protocol: &'static str,
}

impl Target {
    /// Creates a target for `provider` over `protocol`.
    pub fn new(provider: impl Into<String>, protocol: &'static str) -> Self {
        Self {
            provider: provider.into(),
            protocol,
        }
    }
}

/// What must happen to one item.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ItemAction {
    /// Send it as stored: the target is the one that produced it, on the same
    /// wire protocol, so opaque reasoning is verifiable there.
    Keep,
    /// Send it without the reasoning it carries. Ordinary content survives; opaque
    /// reasoning does not, because the target could not verify it.
    DropReasoning,
    /// Send it with its plaintext reasoning intact where the codec can express it.
    KeepPlaintext,
}

/// Per-item verdict for a whole history.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ContinuityPlan {
    /// One action per item, in the order given.
    pub actions: Vec<ItemAction>,
    /// Why the plan is unsafely servable, when it is not.
    pub refusal: Option<Refusal>,
}

impl ContinuityPlan {
    /// Whether the history may be sent to the target at all.
    pub fn is_servable(&self) -> bool {
        self.refusal.is_none()
    }

    /// A stable label for logs.
    pub fn as_str(&self) -> &'static str {
        match &self.refusal {
            Some(_) => "refuse",
            None if self
                .actions
                .iter()
                .all(|action| *action == ItemAction::Keep) =>
            {
                "reuse"
            }
            None => "transform",
        }
    }
}

/// Why a history cannot be sent to a target.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Refusal {
    /// Stable reason code for telemetry and client-facing errors.
    pub reason: &'static str,
    /// Index of the item that caused the refusal.
    pub item_index: usize,
    /// The continuation it belongs to, when it names one.
    pub continuation: Option<String>,
    /// The item's producer, for the operator to see where the cycle started.
    pub produced_by: String,
}

impl fmt::Display for Refusal {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "cannot continue this conversation on another provider: item {} is mid-cycle \
             (continuation {:?}) opened by {}",
            self.item_index,
            self.continuation.as_deref().unwrap_or("unnamed"),
            self.produced_by
        )
    }
}

impl std::error::Error for Refusal {}

/// Decides what to do with every item of a history for one target.
///
/// The rules, per item:
///
/// 1. Mid-cycle on a different provider or protocol is a refusal. The pending call
///    belongs to the model that issued it; another provider cannot continue it, and
///    dropping it would silently change what the conversation is doing rather than
///    report the incompatibility.
/// 2. Same provider and same protocol keeps the item as stored, opaque reasoning
///    included. This is the only case where opaque reasoning may be forwarded.
/// 3. A different provider, or the same provider on a different protocol, keeps
///    ordinary content. Opaque reasoning is dropped because the target cannot
///    verify it. Plaintext is preserved only where the target's protocol has a
///    representation for it, which the caller learns from the returned action
///    rather than being promised universally.
///
/// The first refusal wins: the callback reports the item that made the history
/// unservable, which is more useful than the first of several.
pub fn plan(history: &[HistoryItem], target: &Target) -> ContinuityPlan {
    let mut actions = Vec::with_capacity(history.len());
    for item in history {
        let same_lane = item.provenance.provider == target.provider
            && item.provenance.protocol == target.protocol;

        if item.boundary == Boundary::MidCycle && !same_lane {
            return ContinuityPlan {
                actions,
                refusal: Some(Refusal {
                    reason: "unfinished_tool_cycle_across_providers",
                    item_index: item.index,
                    continuation: item.continuation.clone(),
                    produced_by: item.provenance.provider.clone(),
                }),
            };
        }

        actions.push(if same_lane {
            ItemAction::Keep
        } else {
            match item.reasoning {
                ReasoningRepresentation::None | ReasoningRepresentation::Opaque => {
                    ItemAction::DropReasoning
                }
                // Plaintext survives a compatible transition, but the codec decides
                // how; this layer only says it may be carried.
                ReasoningRepresentation::Plaintext => ItemAction::KeepPlaintext,
            }
        });
    }

    ContinuityPlan {
        actions,
        refusal: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn deepseek() -> ItemProvenance {
        ItemProvenance::new("deepseek", "openai_chat", "deepseek-flash")
    }

    fn target_deepseek() -> Target {
        Target::new("deepseek", "openai_chat")
    }

    fn target_openrouter() -> Target {
        Target::new("openrouter", "openai_chat")
    }

    #[test]
    fn same_lane_reuses_opaque_reasoning() {
        let history =
            [HistoryItem::new(0, deepseek()).with_reasoning(ReasoningRepresentation::Opaque)];
        let plan = plan(&history, &target_deepseek());
        assert_eq!(plan.actions, vec![ItemAction::Keep]);
        assert!(plan.is_servable());
        assert_eq!(plan.as_str(), "reuse");
    }

    #[test]
    fn another_provider_drops_opaque_reasoning_and_serves() {
        let history =
            [HistoryItem::new(0, deepseek()).with_reasoning(ReasoningRepresentation::Opaque)];
        let plan = plan(&history, &target_openrouter());
        assert_eq!(plan.actions, vec![ItemAction::DropReasoning]);
        assert!(plan.is_servable());
        assert_eq!(plan.as_str(), "transform");
    }

    #[test]
    fn plaintext_is_carried_across_a_compatible_transition() {
        let history =
            [HistoryItem::new(0, deepseek()).with_reasoning(ReasoningRepresentation::Plaintext)];
        let plan = plan(&history, &target_openrouter());
        assert_eq!(plan.actions, vec![ItemAction::KeepPlaintext]);
        assert!(plan.is_servable());
    }

    #[test]
    fn an_unfinished_cycle_is_bound_to_the_provider_that_opened_it() {
        let history = [
            HistoryItem::new(0, deepseek()),
            HistoryItem::new(1, deepseek()).mid_cycle("call_1"),
        ];
        let plan = plan(&history, &target_openrouter());
        assert!(!plan.is_servable());
        let label = plan.as_str();
        let refusal = plan.refusal.expect("refusal");
        assert_eq!(refusal.reason, "unfinished_tool_cycle_across_providers");
        assert_eq!(refusal.item_index, 1);
        assert_eq!(refusal.continuation.as_deref(), Some("call_1"));
        assert!(refusal.to_string().contains("call_1"));
        assert_eq!(label, "refuse");
    }

    // The same cycle on its own lane is fine: nothing is crossing.
    #[test]
    fn an_unfinished_cycle_on_its_own_lane_is_reused() {
        let history = [HistoryItem::new(0, deepseek()).mid_cycle("call_1")];
        let plan = plan(&history, &target_deepseek());
        assert!(plan.is_servable());
        assert_eq!(plan.actions, vec![ItemAction::Keep]);
    }

    // Mixed history: the item produced by the target is kept, the foreign one loses
    // its reasoning. A per-turn decision could not express this.
    #[test]
    fn a_mixed_history_is_decided_item_by_item() {
        let other = ItemProvenance::new("openrouter", "openai_chat", "vendor/omni");
        let history = [
            HistoryItem::new(0, other.clone()).with_reasoning(ReasoningRepresentation::Opaque),
            HistoryItem::new(1, deepseek()).with_reasoning(ReasoningRepresentation::Opaque),
            HistoryItem::new(2, other).with_reasoning(ReasoningRepresentation::Plaintext),
        ];
        let plan = plan(&history, &target_deepseek());
        assert!(plan.is_servable());
        assert_eq!(
            plan.actions,
            vec![
                ItemAction::DropReasoning,
                ItemAction::Keep,
                ItemAction::KeepPlaintext,
            ]
        );
    }

    // A protocol change is a lane change: the same credential over a different
    // outbound format cannot verify stored native reasoning either.
    #[test]
    fn a_protocol_change_is_treated_as_a_boundary() {
        let history =
            [HistoryItem::new(0, deepseek()).with_reasoning(ReasoningRepresentation::Opaque)];
        let plan = plan(&history, &Target::new("deepseek", "openai_responses"));
        assert_eq!(plan.actions, vec![ItemAction::DropReasoning]);
    }

    // Model identity alone must not make two lanes interchangeable: one model id
    // really does sit on two credentials in this deployment.
    #[test]
    fn the_same_model_id_on_another_credential_is_still_a_lane_change() {
        let gpt = ItemProvenance::new("openai", "openai_responses", "gpt-6-luna");
        let history = [HistoryItem::new(0, gpt).with_reasoning(ReasoningRepresentation::Opaque)];
        let plan = plan(&history, &Target::new("openai_nt", "openai_responses"));
        assert_eq!(plan.actions, vec![ItemAction::DropReasoning]);
    }

    // The first refusal is reported, not the first of a run of them, so the caller
    // sees where the unservable run starts.
    #[test]
    fn the_first_mid_cycle_item_is_the_one_reported() {
        let history = [
            HistoryItem::new(0, deepseek()).mid_cycle("call_1"),
            HistoryItem::new(1, deepseek()).mid_cycle("call_2"),
        ];
        let refusal = plan(&history, &target_openrouter())
            .refusal
            .expect("refusal");
        assert_eq!(refusal.item_index, 0);
        assert_eq!(refusal.continuation.as_deref(), Some("call_1"));
    }
}
