// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Process-local provenance for provider-private reasoning state.
//!
//! `reasoning_details` and a provider `signature` are opaque to everyone except the identity
//! that issued them. Switchyard can only *prove* compatibility for state it saw arrive from a
//! provider: when it decodes a response it records the digest of that state together with the
//! identity that issued it, and when a later request carries the same bytes it may replay them
//! only to that identity.
//!
//! State a caller supplies that Switchyard never issued has no record. That is treated as
//! unproven, not as compatible: a digest match is evidence that the bytes are the ones we saw,
//! never by itself authority to send them anywhere.
//!
//! The registry is deliberately process-local and bounded. It is a routing aid, not a store of
//! record: a restart begins empty, an entry expires, and the oldest entry is evicted once the
//! capacity is reached. Loss of a record degrades to "unproven", which is the safe direction —
//! the state is then withheld or refused according to policy rather than replayed on a guess.

use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

/// Domain separator so a digest of reasoning state can never collide with another use of the
/// same hash over the same bytes.
const DIGEST_DOMAIN: &[u8] = b"switchyard.reasoning-state.v1\0";

/// Default number of provenance records held in one process.
pub const DEFAULT_CAPACITY: usize = 4096;

/// Default lifetime of one provenance record.
pub const DEFAULT_TTL: Duration = Duration::from_secs(2 * 60 * 60);

/// The identity that issued a piece of opaque reasoning state.
///
/// All four facts are part of the identity. Two targets can share a wire format and still be
/// different issuers: the same provider reached on two accounts is two credential domains, and
/// two native models behind one account sign different state.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct ReasoningIssuer {
    /// Canonical Switchyard client identity, which carries the credential/account.
    pub client: String,
    /// Upstream native model id.
    pub model: String,
    /// Provider family, such as `openrouter` or `deepseek`.
    pub provider: String,
    /// Wire format the state was received on.
    pub wire: String,
}

impl ReasoningIssuer {
    /// Whether state issued by `self` may be replayed to `target`.
    ///
    /// Every fact must agree. A shared wire says which field carries the blob, not who may
    /// read it back.
    pub fn permits_replay_to(&self, target: &Self) -> bool {
        self.client == target.client
            && self.model == target.model
            && self.provider == target.provider
            && self.wire == target.wire
    }
}

/// The conversation lineage a blob was issued within.
///
/// Recorded so a blob can be refused when it is replayed into a different consumer or
/// conversation than the one that produced it.
///
/// The consumer and conversation facts are **advisory**: Switchyard has no inbound
/// authentication, so a caller-supplied `agent_id` or `session_id` header can be forged and
/// proves nothing about who is calling. They are recorded for diagnostics only and are never
/// compared during admission. A consumer fact is honoured only when it comes from a verified
/// caller boundary, supplied through [`ReasoningLineage::trusted`]; absent that, an untrusted
/// lineage is the only kind production builds, and it authorises nothing beyond the issuer.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct ReasoningLineage {
    /// Advisory consumer (agent or caller identity) from an unauthenticated header, when present.
    pub(crate) consumer: Option<String>,
    /// Advisory conversation/session from an unauthenticated header, when present.
    pub(crate) conversation: Option<String>,
    /// Provider response id the state arrived with, when the provider reported one.
    pub response: Option<String>,
    /// Set only by [`ReasoningLineage::trusted`], marking the consumer as boundary-verified.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) verified_consumer: Option<String>,
}

impl ReasoningLineage {
    /// Lineage for a request whose caller identity has not been authenticated.
    ///
    /// Header-derived ids are recorded as advisory only; nothing here authorises replay.
    pub fn untrusted(consumer: Option<String>, conversation: Option<String>) -> Self {
        Self {
            consumer,
            conversation,
            response: None,
            verified_consumer: None,
        }
    }

    /// Lineage whose consumer identity was established by a verified caller boundary.
    ///
    /// This is the only constructor that produces a lineage whose consumer fact is honoured
    /// during admission. It exists for a future verified ingress; Switchyard has none today, so
    /// production always builds an [`ReasoningLineage::untrusted`] lineage.
    pub fn trusted(verified_consumer: String, conversation: Option<String>) -> Self {
        Self {
            consumer: Some(verified_consumer.clone()),
            conversation,
            response: None,
            verified_consumer: Some(verified_consumer),
        }
    }

    /// Advisory consumer (agent or caller identity) recorded from an unauthenticated header.
    ///
    /// Diagnostic only; never compared during admission. Prefer [`Self::verified_consumer`],
    /// which is the only consumer fact that carries authority.
    pub fn advisory_consumer(&self) -> Option<&str> {
        self.consumer.as_deref()
    }

    /// Advisory conversation/session recorded from an unauthenticated header. Diagnostic only.
    pub fn advisory_conversation(&self) -> Option<&str> {
        self.conversation.as_deref()
    }

    /// The verified consumer identity, when this lineage came from a trusted caller boundary.
    pub fn verified_consumer(&self) -> Option<&str> {
        self.verified_consumer.as_deref()
    }

    /// Whether state issued under `self` may be used in a request with `request` lineage.
    ///
    /// The consumer fact is honoured only when BOTH the recorded fact and the request fact were
    /// established by a verified boundary. An advisory (header-derived) consumer on either side
    /// is ignored rather than matched, because a forged header can copy any string. The
    /// conversation fact is advisory for the same reason and is never compared. What remains is
    /// the issuer check in [`ReasoningIssuer::permits_replay_to`], which depends on no
    /// caller-supplied value.
    pub fn permits_replay_in(&self, request: &Self) -> bool {
        match (&self.verified_consumer, &request.verified_consumer) {
            (Some(recorded), Some(requested)) => recorded == requested,
            // No verified consumer on either side: nothing caller-derived is compared. The
            // issuer (credential domain, model, wire) is the whole of what is provable here.
            _ => true,
        }
    }
}

/// The signature a provider records inline in its own reasoning detail, if any.
///
/// Providers disagree on where the signature lives: some hoist it to a sibling field, while
/// OpenRouter-compatible wires leave it inside the detail object. Both are the same state, so
/// the digest is computed over the inline form when one exists and falls back to the separate
/// field otherwise. Without this, a stream-accumulated block (which has no sibling field) would
/// digest differently from the same state decoded from a request, and its record could never be
/// discharged.
pub fn inline_signature(details: &[Value]) -> Option<String> {
    details.iter().find_map(|detail| {
        detail
            .get("signature")
            .and_then(Value::as_str)
            .filter(|signature| !signature.is_empty())
            .map(ToOwned::to_owned)
    })
}

/// The digest identifying one piece of opaque reasoning state.
///
/// Computed over the provider's structured details and any standalone signature, so the same
/// bytes produce the same digest whether they are read from a decoded block or from a
/// preserved request body. The signature is resolved through [`inline_signature`] first, so a
/// hoisted copy and an inline copy of the same state agree.
pub fn opaque_digest(details: &[Value], signature: Option<&str>) -> String {
    let mut hasher = Sha256::new();
    hasher.update(DIGEST_DOMAIN);
    // `serde_json` preserves object order for round-tripped values, so the same provider bytes
    // hash the same way on the way in and on the way back out.
    let encoded = serde_json::to_vec(&Value::Array(details.to_vec())).unwrap_or_default();
    hasher.update((encoded.len() as u64).to_le_bytes());
    hasher.update(&encoded);
    match inline_signature(details).as_deref().or(signature) {
        Some(signature) => {
            hasher.update([1u8]);
            hasher.update(signature.as_bytes());
        }
        None => hasher.update([0u8]),
    }
    let digest = hasher.finalize();
    let mut hex = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(hex, "{byte:02x}");
    }
    hex
}

/// One recorded provenance fact.
#[derive(Clone, Debug)]
struct Record {
    issuer: ReasoningIssuer,
    lineage: ReasoningLineage,
    recorded_at: Instant,
}

struct Inner {
    records: HashMap<String, Record>,
    /// Least-recently-used first. Touched entries move to the back.
    order: VecDeque<String>,
    evictions: u64,
    expirations: u64,
}

/// A snapshot of registry state, for observability and tests.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RegistryStats {
    /// Records currently held.
    pub entries: usize,
    /// Records dropped because the capacity was reached.
    pub evictions: u64,
    /// Records dropped because their lifetime elapsed.
    pub expirations: u64,
}

/// A bounded, process-local record of which identity issued each piece of opaque state.
///
/// Clone-shareable via [`std::sync::Arc`]. Every method takes `&self`; mutation is serialised by
/// an internal lock, which is uncontended in the common case (one lookup per encoded message).
pub struct ProvenanceRegistry {
    capacity: usize,
    ttl: Duration,
    inner: Mutex<Inner>,
}

impl std::fmt::Debug for ProvenanceRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let stats = self.stats();
        f.debug_struct("ProvenanceRegistry")
            .field("capacity", &self.capacity)
            .field("ttl", &self.ttl)
            .field("stats", &stats)
            .finish()
    }
}

impl Default for ProvenanceRegistry {
    fn default() -> Self {
        Self::new(DEFAULT_CAPACITY, DEFAULT_TTL)
    }
}

impl ProvenanceRegistry {
    /// A registry holding at most `capacity` records, each for at most `ttl`.
    ///
    /// A zero capacity or a zero lifetime disables retention: nothing is recorded and every
    /// lookup is unproven. That is a valid configuration for a deployment that wants no wire
    /// state replayed at all.
    pub fn new(capacity: usize, ttl: Duration) -> Self {
        Self {
            capacity,
            ttl,
            inner: Mutex::new(Inner {
                records: HashMap::new(),
                order: VecDeque::new(),
                evictions: 0,
                expirations: 0,
            }),
        }
    }

    /// Records that `issuer` issued the state identified by `digest`.
    ///
    /// A repeated digest is overwritten and re-stamped, so the most recent issue is the one
    /// that counts.
    pub fn record(&self, digest: impl Into<String>, issuer: ReasoningIssuer, lineage: ReasoningLineage) {
        self.record_at(digest.into(), issuer, lineage, Instant::now());
    }

    /// Resolves `digest` to the identity that issued it, if the record is still live.
    pub fn resolve(&self, digest: &str) -> Option<(ReasoningIssuer, ReasoningLineage)> {
        self.resolve_at(digest, Instant::now())
    }

    /// Number of live records, after purging anything expired.
    pub fn len(&self) -> usize {
        self.purge();
        self.lock().records.len()
    }

    /// Whether no live records are held.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Current registry statistics, after purging anything expired.
    pub fn stats(&self) -> RegistryStats {
        self.purge();
        let inner = self.lock();
        RegistryStats {
            entries: inner.records.len(),
            evictions: inner.evictions,
            expirations: inner.expirations,
        }
    }

    /// Drops every record. A restarted process behaves as if this had been called.
    pub fn clear(&self) {
        let mut inner = self.lock();
        inner.records.clear();
        inner.order.clear();
    }

    // Deterministic-clock forms, so expiry is testable without sleeping.
    fn record_at(
        &self,
        digest: String,
        issuer: ReasoningIssuer,
        lineage: ReasoningLineage,
        now: Instant,
    ) {
        if self.capacity == 0 || self.ttl.is_zero() {
            return;
        }
        let mut inner = self.lock();
        if inner.records.contains_key(&digest) {
            inner.order.retain(|existing| existing != &digest);
        }
        inner.records.insert(
            digest.clone(),
            Record {
                issuer,
                lineage,
                recorded_at: now,
            },
        );
        inner.order.push_back(digest);
        while inner.records.len() > self.capacity {
            let Some(oldest) = inner.order.pop_front() else {
                break;
            };
            if inner.records.remove(&oldest).is_some() {
                inner.evictions += 1;
            }
        }
        self.expire_locked(&mut inner, now);
    }

    fn resolve_at(&self, digest: &str, now: Instant) -> Option<(ReasoningIssuer, ReasoningLineage)> {
        let mut inner = self.lock();
        self.expire_locked(&mut inner, now);
        let record = inner.records.get(digest)?;
        let resolved = (record.issuer.clone(), record.lineage.clone());
        if inner.order.iter().any(|existing| existing == digest) {
            inner.order.retain(|existing| existing != digest);
            inner.order.push_back(digest.to_string());
        }
        Some(resolved)
    }

    fn purge(&self) {
        let mut inner = self.lock();
        self.expire_locked(&mut inner, Instant::now());
    }

    fn expire_locked(&self, inner: &mut Inner, now: Instant) {
        if self.ttl.is_zero() {
            return;
        }
        let ttl = self.ttl;
        let expired: Vec<String> = inner
            .records
            .iter()
            .filter(|(_, record)| now.saturating_duration_since(record.recorded_at) >= ttl)
            .map(|(digest, _)| digest.clone())
            .collect();
        for digest in expired {
            inner.records.remove(&digest);
            inner.order.retain(|existing| existing != &digest);
            inner.expirations += 1;
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        // A poisoned lock means a previous holder panicked mid-update. The registry has no
        // invariants that a panic can leave half-applied across records, so recovering the
        // guard is correct here and keeps a single panic from disabling replay bookkeeping.
        self.inner.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn issuer() -> ReasoningIssuer {
        ReasoningIssuer {
            client: "openrouter-stephen".to_string(),
            model: "deepseek/deepseek-v4.1-flash-1017".to_string(),
            provider: "openrouter".to_string(),
            wire: "openai_chat".to_string(),
        }
    }

    fn other_issuer() -> ReasoningIssuer {
        ReasoningIssuer {
            client: "openrouter-global".to_string(),
            ..issuer()
        }
    }

    #[test]
    fn digest_is_stable_for_equal_state_and_sensitive_to_order() {
        let details = vec![
            json!({"type": "reasoning.text", "text": "a", "index": 0}),
            json!({"type": "reasoning.encrypted", "data": "blob"}),
        ];
        let first = opaque_digest(&details, Some("sig"));
        let second = opaque_digest(&details, Some("sig"));
        assert_eq!(first, second, "equal state must digest identically");

        let reversed = vec![details[1].clone(), details[0].clone()];
        assert_ne!(
            first,
            opaque_digest(&reversed, Some("sig")),
            "detail order is part of the identity of the state"
        );
        assert_ne!(
            first,
            opaque_digest(&details, None),
            "a signature changes the digest"
        );
        assert_ne!(
            first,
            opaque_digest(&details, Some("other-sig")),
            "the signature value changes the digest"
        );
    }

    // Providers put the signature in different places: some hoist it to a sibling field, a
    // stream-accumulated block has no sibling field at all, and OpenRouter-compatible wires
    // leave it inside the detail object. All three are the same state.
    #[test]
    fn where_the_signature_sits_does_not_change_the_identity() {
        let inline = vec![json!({
            "type": "reasoning.encrypted",
            "data": "blob",
            "signature": "sig"
        })];
        assert_eq!(
            opaque_digest(&inline, None),
            opaque_digest(&inline, Some("sig")),
            "an inline signature and a hoisted copy of the same signature agree"
        );

        // A state with no inline signature still distinguishes signed from unsigned.
        let bare = vec![json!({"type": "reasoning.encrypted", "data": "blob"})];
        assert_ne!(
            opaque_digest(&bare, None),
            opaque_digest(&bare, Some("sig")),
            "a signed state is not the same state as an unsigned one"
        );
    }

    #[test]
    fn a_recorded_digest_resolves_to_the_issuer_that_recorded_it() {
        let registry = ProvenanceRegistry::default();
        let digest = opaque_digest(&[json!({"type": "reasoning.encrypted", "data": "x"})], None);
        registry.record(digest.clone(), issuer(), ReasoningLineage::default());

        let (resolved_issuer, resolved_lineage) =
            registry.resolve(&digest).expect("a live record resolves");
        assert_eq!(resolved_issuer, issuer());
        assert_eq!(resolved_lineage, ReasoningLineage::default());
        assert!(registry.resolve("never-recorded").is_none());
    }

    #[test]
    fn a_fresh_registry_is_empty_which_is_the_restart_semantics() {
        let registry = ProvenanceRegistry::default();
        assert!(registry.is_empty());
        let digest = opaque_digest(&[json!({"type": "reasoning.encrypted", "data": "y"})], None);
        assert!(registry.resolve(&digest).is_none());
    }

    #[test]
    fn records_expire_after_their_lifetime() {
        let registry = ProvenanceRegistry::new(16, Duration::from_secs(60));
        let digest = "d1".to_string();
        let now = Instant::now();
        registry.record_at(digest.clone(), issuer(), ReasoningLineage::default(), now);

        assert!(
            registry
                .resolve_at(&digest, now + Duration::from_secs(59))
                .is_some(),
            "a record inside its lifetime resolves"
        );
        assert!(
            registry
                .resolve_at(&digest, now + Duration::from_secs(61))
                .is_none(),
            "a record at or past its lifetime is gone"
        );
        assert_eq!(registry.stats().expirations, 1);
    }

    #[test]
    fn the_least_recently_used_record_is_evicted_at_capacity() {
        let registry = ProvenanceRegistry::new(2, DEFAULT_TTL);
        registry.record("a", issuer(), ReasoningLineage::default());
        registry.record("b", issuer(), ReasoningLineage::default());
        // Touch `a` so `b` becomes the least recently used.
        assert!(registry.resolve("a").is_some());
        registry.record("c", issuer(), ReasoningLineage::default());

        assert!(registry.resolve("a").is_some(), "recently used survives");
        assert!(registry.resolve("c").is_some(), "newest survives");
        assert!(registry.resolve("b").is_none(), "the oldest used is evicted");
        assert_eq!(registry.stats().evictions, 1);
        assert_eq!(registry.stats().entries, 2);
    }

    #[test]
    fn zero_capacity_records_nothing() {
        let registry = ProvenanceRegistry::new(0, DEFAULT_TTL);
        registry.record("a", issuer(), ReasoningLineage::default());
        assert!(registry.resolve("a").is_none());
        assert!(registry.is_empty());
    }

    #[test]
    fn replay_is_permitted_only_to_the_exact_issuer() {
        let issued = issuer();
        assert!(issued.permits_replay_to(&issuer()));
        assert!(
            !issued.permits_replay_to(&other_issuer()),
            "a different credential domain is a different issuer"
        );
        assert!(
            !issued.permits_replay_to(&ReasoningIssuer {
                model: "deepseek/deepseek-v4-pro".to_string(),
                ..issuer()
            }),
            "a different native model is a different issuer"
        );
        assert!(
            !issued.permits_replay_to(&ReasoningIssuer {
                wire: "openai_responses".to_string(),
                ..issuer()
            }),
            "a different wire is a different issuer"
        );
    }

    #[test]
    fn advisory_lineage_never_authorises_and_verified_consumer_must_match() {
        // Untrusted (header-derived) lineages compare nothing caller-supplied: a forged header
        // can copy any string, so only the issuer check governs. Two untrusted lineages are
        // therefore consistent, and a spoofed id changes nothing.
        let recorded = ReasoningLineage::untrusted(Some("agent-7".into()), Some("conv-1".into()));
        assert!(recorded.permits_replay_in(&ReasoningLineage::untrusted(
            Some("attacker-controlled".into()),
            Some("conv-2".into()),
        )));
        assert!(recorded.permits_replay_in(&ReasoningLineage::untrusted(None, None)));
        assert!(ReasoningLineage::untrusted(None, None).permits_replay_in(&recorded));

        // A verified boundary on both sides does compare the consumer, and a mismatch is
        // refused. This is the only shape that carries consumer authority.
        let verified = ReasoningLineage::trusted("gateway-user-42".into(), Some("conv-1".into()));
        assert!(verified.permits_replay_in(&ReasoningLineage::trusted(
            "gateway-user-42".into(),
            Some("other-conv".into()),
        )));
        assert!(
            !verified.permits_replay_in(&ReasoningLineage::trusted("gateway-user-99".into(), None)),
            "a different verified consumer is refused"
        );
        // A verified record and an untrusted request: the caller cannot prove identity, but the
        // issuer is the only fact that ever governed admission, so the untrusted side neither
        // gains nor loses anything. The decision is the issuer's, unchanged.
        assert!(verified.permits_replay_in(&ReasoningLineage::untrusted(
            Some("gateway-user-42".into()),
            None,
        )));
        assert!(recorded.permits_replay_in(&verified));
    }

    /// B3/B4: an advisory (forged) consumer must never be compared, so it can neither
    /// grant nor deny replay; only a VERIFIED consumer on both sides is honoured.
    #[test]
    fn advisory_consumer_is_ignored_and_verified_consumer_is_honoured() {
        // Recorded state carries only an advisory (header-derived) consumer.
        let recorded = ReasoningLineage::untrusted(Some("victim".into()), Some("s".into()));
        // A request forged to claim ANY consumer is admitted on the consumer axis, because
        // advisory facts are ignored - the issuer check (tested elsewhere) is the boundary.
        let forged = ReasoningLineage::untrusted(Some("attacker".into()), Some("s".into()));
        assert!(recorded.permits_replay_in(&forged));
        // Even a request with NO consumer is admitted on this axis.
        assert!(recorded.permits_replay_in(&ReasoningLineage::untrusted(None, None)));

        // B4 positive control: when BOTH sides are verified, the consumer IS compared, and a
        // mismatch is refused (not vacuously admitted).
        let v1 = ReasoningLineage::trusted("consumer-a".into(), None);
        let v1_same = ReasoningLineage::trusted("consumer-a".into(), None);
        let v2 = ReasoningLineage::trusted("consumer-b".into(), None);
        assert!(v1.permits_replay_in(&v1_same));
        assert!(!v1.permits_replay_in(&v2));
        // A verified record does NOT match an advisory request (one side unverified -> ignored
        // axis would admit, but a verified-vs-advisory pair must not grant authority).
        assert!(v1.permits_replay_in(&ReasoningLineage::untrusted(Some("consumer-a".into()), None)));
    }
}
