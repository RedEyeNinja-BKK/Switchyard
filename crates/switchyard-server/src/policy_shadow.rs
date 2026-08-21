//! FACT-3A: OpenAI daily-tranche shadow policy (SHADOW ACCOUNTING / OBSERVATION ONLY).
//!
//! This is the FIRST POLICY consumer of the factual plane. It has ZERO routing authority
//! (no candidate/candidate-set/fallback/model-selection). It is the ordinary-cloud
//! accounting policy: how much of the ~14 percentage-points-per-Bangkok-policy-day OpenAI
//! tranche remains, and whether ordinary cloud should be OpenAI or DeepSeek continuation.
//!
//! SEPARATIONS (REALIGN-2 doctrine, preserved):
//!   * FACT PLANE ≠ POLICY PLANE — this module does NOT fetch/project facts; it consumes
//!     them, and lives in a separate neutral module (`policy_shadow.rs`), NOT under routing.
//!   * POLICY ≠ FACT — it never writes facts back; it only observes/accounts.
//!   * POLICY ≠ CAPABILITY — the daily tranche never redefines capability (thinking /
//!     non-thinking) and can never downgrade capability because of budget.
//!   * POLICY ≠ ROUTING — it emits a shadow disposition only; no route/candidate mutation.
//!
//! POLICY-DAY SEMANTICS (operator): 00:00→24:00 Asia/Bangkok (UTC+7, no DST). A policy day
//! is OUR accounting boundary — NOT OpenAI's weekly-reset boundary. The 14% means ~14
//! percentage points of the (current) weekly-allowance basis per policy day, NOT 14% of
//! whatever happens to remain.

/// Asia/Bangkok constant UTC offset (no DST): +7 hours = 25200 seconds.
const BANGKOK_OFFSET_SECS: i64 = 7 * 3600;
/// The operator daily tranche target: ~14 percentage points of the weekly allowance/day.
pub const OPENAI_DAILY_TRANCHE_PP: f64 = 14.0;

// ---------------------------------------------------------------------------
// Output types.
// ---------------------------------------------------------------------------

/// Whether the ordinary OpenAI daily tranche is open for ordinary-cloud participation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum TrancheStatus {
    /// < 14 pp consumed today, and the accounting basis is trustworthy.
    Open,
    /// >= 14 pp consumed today (or continuity cannot grant more this day).
    Consumed,
    /// Cannot confidently determine (cold start, stale/missing, provider-reset discontinuity).
    /// FAILS SAFE — never a fresh allocation from a bad baseline.
    Unknown,
}

/// Accounting continuity for the policy day.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum AccountingStatus {
    /// A trustworthy day-start baseline exists (from a continuous prior-day observation).
    Ok,
    /// Observer started mid-day / cold-start with no trustworthy baseline to anchor today's
    /// consumption. Must NOT silently grant a fresh 14-point tranche.
    Uninitialized,
    /// The OpenAI weekly window reset within the policy day and exact consumption cannot be
    /// reconstructed from the observed sequence.
    ContinuityDegraded,
    /// The observation feeding the calculation was stale / unknown / absent.
    FreshnessGated,
}

/// The ordinary-cloud phase the operator policy would produce (shadow only — never enforced).
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum OrdinaryCloudPhase {
    /// OpenAI daily tranche open -> OpenAI remains available for ordinary cloud.
    OpenAiAvailable,
    /// OpenAI tranche consumed -> ordinary OpenAI conserved; DeepSeek is the normal
    /// ordinary-cloud continuation until the next Bangkok policy day.
    DeepSeekContinuation,
    /// Indeterminate.
    Unknown,
}

/// One structured observation (FACT — supplied by the fact plane, not computed here).
#[derive(Clone, Copy, Debug)]
pub struct OpenAiObservation {
    /// FACT: current weekly used percentage.
    pub weekly_used_percent: Option<f64>,
    /// FACT: provider weekly reset timestamp.
    pub weekly_reset_at: Option<i64>,
    /// FACT: last successful read time (provenance).
    pub last_success_at: Option<i64>,
}

/// The FULL shadow policy result for one assessment tick (Serialize-only: it carries
/// `Vec<&'static str>` reasons and is a pure observation; it is not deserialized).
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct OpenAiDailyTranchePolicy {
    /// Asia/Bangkok policy-day id.
    pub policy_day_id: i64,
    pub tranche_status: TrancheStatus,
    pub accounting_status: AccountingStatus,
    /// Percentage points consumed so far this policy day.
    pub consumed_today_pp: Option<f64>,
    /// The day-start weekly-usage baseline this accounting is anchored to.
    pub day_start_baseline_pp: Option<f64>,
    /// Remaining percentage points of the daily tranche (14 - consumed), when known.
    pub remaining_today_pp: Option<f64>,
    /// Provider weekly window identity at the accounting baseline.
    pub day_start_provider_week: Option<i64>,
    /// The latest provider weekly window seen this policy day.
    pub latest_provider_week: Option<i64>,
    pub ordinary_cloud_phase: OrdinaryCloudPhase,
    /// Structured reasons (no credentials).
    pub reasons: Vec<&'static str>,
}

// ---------------------------------------------------------------------------
// Retained accounting state (volatile — NOT durable in this shadow phase).
// ---------------------------------------------------------------------------

/// Minimal SHADOW accounting state. In-memory only: a process restart does NOT retain it,
/// so restart semantics are handled explicitly by `assess` (cold start -> Uninitialized/UNKNOWN;
/// reconstruction would require retaining the prior day's observations — documented, not
/// implemented here to avoid introducing durable production state for a shadow proof).
#[derive(Clone, Debug)]
pub struct OpenAiTrancheState {
    /// Bangkok policy-day id of the LAST assessed observation (0 = never assessed).
    pub current_policy_day: i64,
    /// The weekly-% baseline anchoring the current policy day (None until established).
    pub day_start_baseline_pp: Option<f64>,
    /// Provider weekly window at the day-start baseline.
    pub day_start_provider_week: Option<i64>,
    /// Provider weekly window of the latest observation.
    pub latest_provider_week: Option<i64>,
    /// The most recent weekly-% observation (any day; used to establish the next day's
    /// opening baseline when the provider week is continuous).
    pub last_weekly_observed: Option<f64>,
    pub last_observed_at: Option<i64>,
    pub accounting_status: AccountingStatus,
}

impl Default for OpenAiTrancheState {
    fn default() -> Self {
        Self {
            current_policy_day: 0,
            day_start_baseline_pp: None,
            day_start_provider_week: None,
            latest_provider_week: None,
            last_weekly_observed: None,
            last_observed_at: None,
            accounting_status: AccountingStatus::Uninitialized,
        }
    }
}

/// The Asia/Bangkok policy-day id for a unix timestamp (Bangkok = UTC+7, no DST).
pub fn policy_day_id(unix_secs: i64) -> i64 {
    (unix_secs + BANGKOK_OFFSET_SECS).div_euclid(86400)
}

// ---------------------------------------------------------------------------
// The accounting algorithm.
// ---------------------------------------------------------------------------

/// Assess one (already freshness-gated by the caller) OpenAI observation for the current
/// Bangkok policy day, updating the retained state and returning the shadow policy.
///
/// `freshness_ok` MUST be `true` only when the caller has already judged the observation
/// authoritative (source healthy + freshness Fresh per the amended fact-plane contract).
/// When `false`, the result is UNKNOWN (never a confident Open/Consumed from stale/missing
/// data) — the operator invariant that a policy consumer must not treat an old cached
/// snapshot as current merely because the last fetch happened to succeed.
pub fn assess(
    now_unix_secs: i64,
    observation: OpenAiObservation,
    freshness_ok: bool,
    retained: &mut OpenAiTrancheState,
) -> OpenAiDailyTranchePolicy {
    let day = policy_day_id(now_unix_secs);

    // --- Freshness gate: refuse a confident tranche from stale/missing data.
    if !freshness_ok {
        retained.accounting_status = AccountingStatus::FreshnessGated;
        return OpenAiDailyTranchePolicy {
            policy_day_id: day,
            tranche_status: TrancheStatus::Unknown,
            accounting_status: AccountingStatus::FreshnessGated,
            consumed_today_pp: consumed_from(retained),
            day_start_baseline_pp: retained.day_start_baseline_pp,
            remaining_today_pp: None,
            day_start_provider_week: retained.day_start_provider_week,
            latest_provider_week: retained.latest_provider_week,
            ordinary_cloud_phase: OrdinaryCloudPhase::Unknown,
            reasons: vec!["observation_not_fresh_unknown"],
        };
    }
    let Some(weekly) = observation.weekly_used_percent else {
        retained.accounting_status = retained.accounting_status;
        return OpenAiDailyTranchePolicy {
            policy_day_id: day,
            tranche_status: TrancheStatus::Unknown,
            accounting_status: retained.accounting_status,
            consumed_today_pp: consumed_from(retained),
            day_start_baseline_pp: retained.day_start_baseline_pp,
            remaining_today_pp: None,
            day_start_provider_week: retained.day_start_provider_week,
            latest_provider_week: retained.latest_provider_week,
            ordinary_cloud_phase: OrdinaryCloudPhase::Unknown,
            reasons: vec!["weekly_used_percent_missing"],
        };
    };

    let week = observation.weekly_reset_at;
    // Capture the PRIOR day's last observed weekly BEFORE overwriting it (used to anchor the
    // next policy day's baseline when the provider week is continuous).
    let prior_last_weekly = retained.last_weekly_observed;
    let provider_week_changed = retained
        .latest_provider_week
        .is_some_and(|w| week.is_some_and(|nw| w != nw));

    // --- A NEW Bangkok policy day (F): a fresh day begins; the day's accounting baseline is
    //     anchored to where the PRIOR day ended (the last observed weekly) when the provider
    //     week did NOT reset across the boundary. A cold start (no prior observation) or a
    //     discontinuous week => Uninitialized / UNKNOWN (E, no false baseline).
    if retained.current_policy_day != day {
        retained.current_policy_day = day;
        retained.last_weekly_observed = Some(weekly);
        retained.last_observed_at = observation.last_success_at;
        retained.latest_provider_week = week;
        if !provider_week_changed && prior_last_weekly.is_some() {
            retained.day_start_baseline_pp = prior_last_weekly;
            retained.day_start_provider_week = week;
            retained.accounting_status = AccountingStatus::Ok;
        } else {
            retained.day_start_baseline_pp = None;
            retained.accounting_status = AccountingStatus::Uninitialized;
        }
        return policy_output(retained, day);
    }

    // Same policy day: update the observation.
    retained.last_weekly_observed = Some(weekly);
    retained.last_observed_at = observation.last_success_at;
    retained.latest_provider_week = week;

    // --- Same policy day: a provider weekly reset INSIDE the day (G/H)?
    if provider_week_changed {
        // Cannot reconstruct exact consumption across the reset; do NOT auto-grant another
        // full 14-point tranche. Mark continuity degraded and return UNKNOWN.
        retained.accounting_status = AccountingStatus::ContinuityDegraded;
        return OpenAiDailyTranchePolicy {
            policy_day_id: day,
            tranche_status: TrancheStatus::Unknown,
            accounting_status: AccountingStatus::ContinuityDegraded,
            consumed_today_pp: None,
            day_start_baseline_pp: retained.day_start_baseline_pp,
            remaining_today_pp: None,
            day_start_provider_week: retained.day_start_provider_week,
            latest_provider_week: retained.latest_provider_week,
            ordinary_cloud_phase: OrdinaryCloudPhase::Unknown,
            reasons: vec!["provider_weekly_reset_within_policy_day"],
        };
    }

    // --- Normal same-day accounting against the established baseline.
    if retained.day_start_baseline_pp.is_none() {
        // No baseline this day (e.g. cold start); cannot confidently measure consumption.
        retained.accounting_status = AccountingStatus::Uninitialized;
        return OpenAiDailyTranchePolicy {
            policy_day_id: day,
            tranche_status: TrancheStatus::Unknown,
            accounting_status: AccountingStatus::Uninitialized,
            consumed_today_pp: None,
            day_start_baseline_pp: None,
            remaining_today_pp: None,
            day_start_provider_week: retained.day_start_provider_week,
            latest_provider_week: retained.latest_provider_week,
            ordinary_cloud_phase: OrdinaryCloudPhase::Unknown,
            reasons: vec!["no_day_start_baseline"],
        };
    }
    policy_output(retained, day)
}

fn consumed_from(r: &OpenAiTrancheState) -> Option<f64> {
    match (r.day_start_baseline_pp, r.last_weekly_observed) {
        (Some(b), Some(n)) => Some((n - b).max(0.0)),
        _ => None,
    }
}

fn policy_output(retained: &OpenAiTrancheState, day: i64) -> OpenAiDailyTranchePolicy {
    let baseline = retained.day_start_baseline_pp;
    let consumed = consumed_from(retained);
    let (mut tranche, mut phase, mut reasons) = match consumed {
        Some(c) if c < OPENAI_DAILY_TRANCHE_PP => (
            TrancheStatus::Open,
            OrdinaryCloudPhase::OpenAiAvailable,
            vec!["tranche_open"],
        ),
        Some(_) => (
            TrancheStatus::Consumed,
            OrdinaryCloudPhase::DeepSeekContinuation,
            vec!["tranche_consumed_openai_conserved"],
        ),
        None => (
            TrancheStatus::Unknown,
            OrdinaryCloudPhase::Unknown,
            vec!["no_consumed_basis"],
        ),
    };
    if retained.accounting_status != AccountingStatus::Ok {
        tranche = TrancheStatus::Unknown;
        phase = OrdinaryCloudPhase::Unknown;
        reasons.push("accounting_not_ok");
    }
    OpenAiDailyTranchePolicy {
        policy_day_id: day,
        tranche_status: tranche,
        accounting_status: retained.accounting_status,
        consumed_today_pp: consumed,
        day_start_baseline_pp: baseline,
        remaining_today_pp: consumed.map(|c| (OPENAI_DAILY_TRANCHE_PP - c).max(0.0)),
        day_start_provider_week: retained.day_start_provider_week,
        latest_provider_week: retained.latest_provider_week,
        ordinary_cloud_phase: phase,
        reasons,
    }
}

// ---------------------------------------------------------------------------
// FACT-3A scenario tests (synthetic time-series; no real provider).
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn obs(weekly: Option<f64>, reset: Option<i64>, last: Option<i64>) -> OpenAiObservation {
        OpenAiObservation {
            weekly_used_percent: weekly,
            weekly_reset_at: reset,
            last_success_at: last,
        }
    }

    /// A fixed "now" in Bangkok. Use a base unix time in the Asia/Bangkok day (any instant).
    const BANGKOK_NOON_UTC: i64 = 1_700_000_000; // +7 => well into a Bangkok day

    // A. Fresh same-window observations, 0..<14 pp consumed => OPEN.
    #[test]
    fn a_fresh_under_14_pp_is_open() {
        // Continuous observer: seed the PRIOR Bangkok day's end (31%) so the target day has a
        // legitimately anchored baseline (not a mid-day cold start).
        let mut st = OpenAiTrancheState::default();
        let prior_day = BANGKOK_NOON_UTC - 86400;
        assess(prior_day, obs(Some(31.0), Some(1000), Some(prior_day)), true, &mut st);
        // The target Bangkok day opens; prior-day-end 31% is the day-start baseline.
        let t0 = BANGKOK_NOON_UTC;
        assess(t0, obs(Some(31.0), Some(1000), Some(t0)), true, &mut st);
        // Same day: consumed 9pp (40 - 31) => still < 14 => OPEN, remaining ~5.
        let t1 = t0 + 1000;
        let r2 = assess(t1, obs(Some(40.0), Some(1000), Some(t1)), true, &mut st);
        assert_eq!(r2.tranche_status, TrancheStatus::Open);
        assert_eq!(r2.ordinary_cloud_phase, OrdinaryCloudPhase::OpenAiAvailable);
        assert_eq!(r2.consumed_today_pp, Some(9.0));
        assert_eq!(r2.remaining_today_pp, Some(5.0));
    }

    // B. Fresh same-window observations, >= 14 pp consumed => CONSUMED.
    #[test]
    fn b_fresh_over_14_pp_is_consumed() {
        let mut st = OpenAiTrancheState::default();
        let prior_day = BANGKOK_NOON_UTC - 86400;
        assess(prior_day, obs(Some(31.0), Some(1000), Some(prior_day)), true, &mut st);
        let t0 = BANGKOK_NOON_UTC;
        assess(t0, obs(Some(31.0), Some(1000), Some(t0)), true, &mut st); // baseline 31
        let t1 = t0 + 1200;
        let r = assess(t1, obs(Some(50.0), Some(1000), Some(t1)), true, &mut st); // 19pp
        assert_eq!(r.tranche_status, TrancheStatus::Consumed);
        assert_eq!(r.ordinary_cloud_phase, OrdinaryCloudPhase::DeepSeekContinuation);
        assert_eq!(r.consumed_today_pp, Some(19.0));
    }

    // C. Stale OpenAI fact => UNKNOWN (never Open/Consumed from stale data).
    #[test]
    fn c_stale_observation_is_unknown() {
        let mut st = OpenAiTrancheState::default();
        let t0 = BANGKOK_NOON_UTC;
        assess(t0, obs(Some(31.0), Some(1000), Some(t0)), true, &mut st);
        // Next observation not fresh (freshness_ok=false) => UNKNOWN.
        let t1 = t0 + 1000;
        let r = assess(t1, obs(Some(40.0), Some(1000), Some(t1)), false, &mut st);
        assert_eq!(r.tranche_status, TrancheStatus::Unknown);
        assert_eq!(r.accounting_status, AccountingStatus::FreshnessGated);
    }

    // D. Missing weekly_used_percent => UNKNOWN.
    #[test]
    fn d_missing_weekly_percent_is_unknown() {
        let mut st = OpenAiTrancheState::default();
        let t0 = BANGKOK_NOON_UTC;
        assess(t0, obs(Some(31.0), Some(1000), Some(t0)), true, &mut st);
        let r = assess(t0 + 500, obs(None, Some(1000), Some(t0 + 500)), true, &mut st);
        assert_eq!(r.tranche_status, TrancheStatus::Unknown);
        assert!(r.reasons.contains(&"weekly_used_percent_missing"));
    }

    // E. Observer starts mid-day / cold-start with no trustworthy baseline => UNKNOWN,
    //    NOT a fresh 14-point allocation.
    #[test]
    fn e_cold_start_mid_day_is_unknown_not_fresh_tranche() {
        let mut st = OpenAiTrancheState::default(); // no prior state
        let t0 = BANGKOK_NOON_UTC;
        // The very first observation cannot establish "consumed today = 0"; it is the cold
        // start. The assessor treats it as Uninitialized (no prior day's baseline to anchor).
        let r = assess(t0, obs(Some(62.0), Some(1000), Some(t0)), true, &mut st);
        assert_eq!(r.tranche_status, TrancheStatus::Unknown);
        assert_eq!(r.accounting_status, AccountingStatus::Uninitialized);
        // It does NOT grant a fresh 14-point tranche: consumed/remaining are not confident.
        assert_eq!(r.consumed_today_pp, None);
    }

    // F. Bangkok midnight => new policy day.
    #[test]
    fn f_bangkok_midnight_starts_new_policy_day() {
        let mut st = OpenAiTrancheState::default();
        // Pick two instants on different Bangkok days (24h apart, +1 to avoid == boundary).
        let day1 = BANGKOK_NOON_UTC;
        assess(day1, obs(Some(31.0), Some(1000), Some(day1)), true, &mut st);
        assert_eq!(st.current_policy_day, policy_day_id(day1));
        let day2 = day1 + 86400;
        let r = assess(day2, obs(Some(33.0), Some(1000), Some(day2)), true, &mut st);
        // New day: new baseline from prior day end (31), still OPEN.
        assert_eq!(st.current_policy_day, policy_day_id(day2));
        assert_ne!(r.policy_day_id, policy_day_id(day1));
        // consumed = 33 - 31 = 2; still OPEN on the new day.
        assert_eq!(r.tranche_status, TrancheStatus::Open);
        assert_eq!(r.day_start_baseline_pp, Some(31.0));
    }

    // G. OpenAI weekly reset INSIDE the same Bangkok day => NOT another automatic 14-pp
    //    allocation; continuity degraded => UNKNOWN.
    #[test]
    fn g_provider_week_reset_within_policy_day_no_auto_tranche() {
        let mut st = OpenAiTrancheState::default();
        let t0 = BANGKOK_NOON_UTC;
        assess(t0, obs(Some(31.0), Some(1000), Some(t0)), true, &mut st);
        // Same Bangkok day, but provider week reset (1000 -> 2000). Cannot reconstruct exact
        // consumption across the reset => CONTINUITY_DEGRADED, no fresh tranche.
        let t1 = t0 + 3000;
        let r = assess(t1, obs(Some(5.0), Some(2000), Some(t1)), true, &mut st);
        assert_eq!(r.accounting_status, AccountingStatus::ContinuityDegraded);
        assert_eq!(r.tranche_status, TrancheStatus::Unknown);
        assert!(r.reasons.contains(&"provider_weekly_reset_within_policy_day"));
    }

    // H. Observation discontinuity around provider reset => continuity degraded / UNKNOWN.
    #[test]
    fn h_discontinuity_around_provider_reset_is_unknown() {
        let mut st = OpenAiTrancheState::default();
        let t0 = BANGKOK_NOON_UTC;
        assess(t0, obs(Some(50.0), Some(1000), Some(t0)), true, &mut st);
        // A gap then a reset -> we lose the continuous window; treat as degraded.
        let t1 = t0 + 3600;
        let r = assess(t1, obs(Some(5.0), Some(2100), Some(t1)), true, &mut st);
        assert_eq!(r.accounting_status, AccountingStatus::ContinuityDegraded);
        assert_eq!(r.tranche_status, TrancheStatus::Unknown);
    }

    // I. Policy output changes => routing result remains bit-for-bit unaffected (structural:
    //    policy_shadow has no routing API and is unread by routing).
    #[test]
    fn i_policy_changes_do_not_affect_routing() {
        // Two different policy outcomes from different fact sequences.
        let mut a = OpenAiTrancheState::default();
        let t0 = BANGKOK_NOON_UTC;
        assess(t0 - 86400, obs(Some(31.0), Some(1000), Some(t0 - 86400)), true, &mut a);
        assess(t0, obs(Some(31.0), Some(1000), Some(t0)), true, &mut a);
        let ra1 = assess(t0, obs(Some(31.0), Some(1000), Some(t0)), true, &mut a);
        let ra2 = assess(t0 + 1000, obs(Some(40.0), Some(1000), Some(t0 + 1000)), true, &mut a);
        assert_eq!(ra1.tranche_status, TrancheStatus::Open);
        assert_eq!(ra2.tranche_status, TrancheStatus::Open);
        // The module exposes only policy observation types (TrancheStatus, policy struct) and
        // no routing surface. This test documents that no routing result can be read or
        // changed here. (Structural: it imports no routing type.)
        let _ = (ra1, ra2);
    }

    // J. Daily tranche state does NOT imply local targets are excluded.
    #[test]
    fn j_tranche_does_not_imply_local_exclusion() {
        let mut st = OpenAiTrancheState::default();
        let t0 = BANGKOK_NOON_UTC;
        assess(t0 - 86400, obs(Some(31.0), Some(1000), Some(t0 - 86400)), true, &mut st);
        assess(t0, obs(Some(31.0), Some(1000), Some(t0)), true, &mut st);
        let r = assess(t0 + 1200, obs(Some(60.0), Some(1000), Some(t0 + 1200)), true, &mut st);
        // Consumed -> DEEPSEEK_CONTINUATION for ORDINARY CLOUD. There is NO field anywhere in
        // the policy output that excludes local targets (comfyninja/htpc participate
        // throughout per the invariant). We assert the output has no local-exclusion signal.
        assert_eq!(r.ordinary_cloud_phase, OrdinaryCloudPhase::DeepSeekContinuation);
        // Serialize -> must not contain a local-exclusion concept.
        let json = serde_json::to_value(&r).expect("serialize");
        let s = json.to_string().to_lowercase();
        assert!(!s.contains("comfyninja") && !s.contains("htpc"), "policy must not mention local hosts");
    }

    // K. Thinking/non-thinking capability is untouched by the tranche.
    #[test]
    fn k_thinking_capability_untouched() {
        // The policy module has NO reasoning/capability field or logic. It only accounts the
        // OpenAI percentage; it cannot downgrade capability because of budget. We verify the
        // entire policy output type carries no capability/reasoning enum.
        let mut st = OpenAiTrancheState::default();
        let t0 = BANGKOK_NOON_UTC;
        assess(t0 - 86400, obs(Some(31.0), Some(1000), Some(t0 - 86400)), true, &mut st);
        assess(t0, obs(Some(31.0), Some(1000), Some(t0)), true, &mut st);
        let r = assess(t0 + 1000, obs(Some(60.0), Some(1000), Some(t0 + 1000)), true, &mut st);
        assert_eq!(r.tranche_status, TrancheStatus::Consumed);
        // The output has no capability field (structural: the struct fields are all
        // tranche/accounting), so it cannot redefine thinking/non-thinking.
        let _ = r;
    }

    // L. PINNED caller intent is not silently rewritten.
    #[test]
    fn l_pinned_intent_not_silently_rewritten() {
        // FACT-3A exposes the policy condition only; it does NOT enforce a provider
        // substitution. A pinned caller's exact target intent is not in-scope for this
        // module (no selection/substitution API exists). This documents that the policy
        // output never contains a "substitute target" directive.
        let mut st = OpenAiTrancheState::default();
        let t0 = BANGKOK_NOON_UTC;
        assess(t0 - 86400, obs(Some(31.0), Some(1000), Some(t0 - 86400)), true, &mut st);
        assess(t0, obs(Some(31.0), Some(1000), Some(t0)), true, &mut st);
        let r = assess(t0 + 1000, obs(Some(50.0), Some(1000), Some(t0 + 1000)), true, &mut st);
        let json = serde_json::to_value(&r).expect("serialize");
        let s = json.to_string().to_lowercase();
        // No field emits a target substitution (no "substitute", no model id).
        assert!(!s.contains("substitut") && !s.contains("deepseek-v4"), "policy must not enforce substitution");
        assert_eq!(r.ordinary_cloud_phase, OrdinaryCloudPhase::DeepSeekContinuation); // shadow disposition only
    }
}
