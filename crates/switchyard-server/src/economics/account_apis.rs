// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The concrete provider account APIs Switchyard reads.
//!
//! Each implementation uses the same credential Switchyard uses to serve inference
//! from that provider. Nothing here creates, rotates or broadens a credential.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde_json::Value;

use super::providers::{
    AccountAmount, AccountApi, AccountBalanceComponent, AccountObservation, AccountScope,
    FetchOutcome, ReportingPeriod, RequestAllowanceReading,
};

/// Parses an RFC 3339 UTC timestamp (`2026-11-26T00:00:00Z`) into a clock reading.
///
/// Only the `Z` form is accepted. A provider that states an offset instead would need a
/// timezone database to interpret, and a wrong reset time is worse than none: the reading
/// is still useful for its counts, so an unparsed reset is reported as absent rather than
/// approximated.
fn parse_rfc3339_utc(text: &str) -> Option<SystemTime> {
    let text = text.strip_suffix('Z')?;
    let (date, time) = text.split_once('T')?;

    // Exactly three date components and no more. A trailing component is not a shape any
    // provider sends; accepting it would store a timestamp this parser only half-read.
    let date: Vec<&str> = date.split('-').collect();
    let [year, month, day] = date.as_slice() else {
        return None;
    };
    if !is_digits(year, 4) {
        return None;
    }
    let year: i64 = year.parse().ok()?;
    let month: i64 = parse_two_digits(month)?;
    let day: i64 = parse_two_digits(day)?;

    // Exactly three time components. A fraction is digits after the seconds, and only
    // digits: `.5abc` is a malformed timestamp, not a timestamp with ignorable text.
    let time: Vec<&str> = time.split(':').collect();
    let [hour, minute, second] = time.as_slice() else {
        return None;
    };
    let hour: i64 = parse_two_digits(hour)?;
    let minute: i64 = parse_two_digits(minute)?;
    let (second_text, fraction) = match second.split_once('.') {
        Some((whole, fraction)) => (whole, Some(fraction)),
        None => (*second, None),
    };
    if let Some(fraction) = fraction
        && (fraction.is_empty() || !fraction.bytes().all(|byte| byte.is_ascii_digit()))
    {
        return None;
    }
    let second: i64 = parse_two_digits(second_text)?;

    // A bounded year. The arithmetic below is calendar arithmetic over a day count; an
    // unbounded year turns a malformed figure into an overflow rather than a refusal, and
    // no provider states a reset outside this range.
    if !(1970..=9999).contains(&year) {
        return None;
    }
    if !(1..=12).contains(&month) || !(0..24).contains(&hour) || !(0..60).contains(&minute) {
        return None;
    }
    // A leap second is written as :60 and is the only value above 59 that is real.
    if !(0..=60).contains(&second) {
        return None;
    }
    if !(1..=days_in_month(year, month)).contains(&day) {
        return None;
    }
    let days = days_from_civil(year, month, day);
    let seconds = days
        .checked_mul(86_400)?
        .checked_add(hour.checked_mul(3_600)?)?
        .checked_add(minute.checked_mul(60)?)?
        .checked_add(second)?;
    u64::try_from(seconds)
        .ok()
        .map(|seconds| UNIX_EPOCH + Duration::from_secs(seconds))
}

/// Whether `text` is exactly `width` ASCII digits.
///
/// Fixed-width fields are checked rather than parsed loosely: `+1` parses as a year and
/// `1` parses as a day, and neither is a date component a provider writes.
fn is_digits(text: &str, width: usize) -> bool {
    text.len() == width && text.bytes().all(|byte| byte.is_ascii_digit())
}

/// A two-digit component of an RFC 3339 timestamp.
fn parse_two_digits(text: &str) -> Option<i64> {
    is_digits(text, 2).then(|| text.parse().ok()).flatten()
}

/// The number of days in a month of the proleptic Gregorian calendar.
///
/// Checked rather than left to the day-number arithmetic: that arithmetic normalises an
/// out-of-range day into the following month, so 2026-02-31 would silently become
/// 2026-03-03. A reset time that is three days wrong is worse than no reset time.
fn days_in_month(year: i64, month: i64) -> i64 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if (year % 4 == 0 && year % 100 != 0) || year % 400 == 0 => 29,
        2 => 28,
        _ => 0,
    }
}

/// Days since 1970-01-01 for a proleptic Gregorian date.
fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let year = if month <= 2 { year - 1 } else { year };
    let era = if year >= 0 { year } else { year - 399 } / 400;
    let year_of_era = year - era * 400;
    let shifted_month = (month + 9) % 12;
    let day_of_year = (153 * shifted_month + 2) / 5 + day - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    era * 146_097 + day_of_era - 719_468
}

/// Reads OpenRouter's `free_model_daily_requests` block into a request allowance.
///
/// Kept as a pure function over the response body so the accepted shapes are stated and
/// tested in one place. A block that is absent, carries a missing field, or carries a
/// value of the wrong type yields nothing: the account did not report a usable figure, and
/// inventing one -- a zero, a limit of one, a remainder equal to the limit -- would admit
/// or refuse traffic on a number the provider never sent.
fn openrouter_free_model_allowance(
    data: &Value,
    observed_at: SystemTime,
) -> Option<RequestAllowanceReading> {
    let block = data.get("free_model_daily_requests")?;
    let number = |key: &str| block.get(key).and_then(Value::as_u64);
    let used = number("used")?;
    let limit = number("limit")?;
    let remaining = number("remaining")?;
    // `resets_at` is only present on some responses; an absent one is a missing fact,
    // not a zero.
    let reset_at = block
        .get("resets_at")
        .and_then(Value::as_str)
        .and_then(parse_rfc3339_utc);
    // A figure above the block's own limit is not a usable answer at all, so it is refused
    // rather than clamped: clamping would present a fabricated limit as if the provider had
    // stated it.
    if used > limit || remaining > limit {
        return None;
    }
    // The provider states the remainder twice: outright as `remaining`, and implicitly as
    // `limit - used`. When they agree there is nothing to decide. When they disagree the
    // provider's own accounting is lagging or reserved against, and the honest figure is
    // the smaller one -- admitting on the larger would spend requests that may not exist.
    let remaining = remaining.min(limit.saturating_sub(used));
    let reading = RequestAllowanceReading::new(
        "free_model_daily_requests",
        used,
        limit,
        remaining,
        observed_at,
        reset_at,
    );
    // A reading whose own numbers disagree is refused here rather than stored and then
    // discarded downstream, so the reason is visible where the response was parsed.
    reading.is_consistent().then_some(reading)
}

/// The HTTP client the account readers use.
///
/// A dedicated client with a short timeout, so a provider that stalls cannot hold the
/// collection task and starve the other providers' observations.
fn account_client() -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
        .unwrap_or_default()
}

/// Reads OpenRouter's key usage and account credits.
///
/// Both endpoints are documented read-only surfaces. The key endpoint reports usage
/// over several windows; the credits endpoint reports the account-wide purchased
/// credit and the usage drawn against it.
pub struct OpenRouterAccount {
    api_key: String,
    base_url: String,
}

impl OpenRouterAccount {
    /// Builds the reader from the credential Switchyard serves OpenRouter with.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: "https://openrouter.ai/api/v1".to_string(),
        }
    }

    /// Overrides the API base, for a test double.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    async fn get_json(&self, path: &str) -> Result<Value, FetchOutcome> {
        let response = account_client()
            .get(format!("{}{path}", self.base_url))
            .bearer_auth(&self.api_key)
            .send()
            .await
            .map_err(|error| FetchOutcome::Failed(error.to_string()))?;
        let status = response.status();
        if status == reqwest::StatusCode::UNAUTHORIZED || status == reqwest::StatusCode::FORBIDDEN {
            return Err(FetchOutcome::Unauthorized);
        }
        if !status.is_success() {
            return Err(FetchOutcome::Failed(format!("HTTP {status}")));
        }
        response
            .json::<Value>()
            .await
            .map_err(|error| FetchOutcome::Failed(error.to_string()))
    }
}

#[async_trait::async_trait]
impl AccountApi for OpenRouterAccount {
    fn provider(&self) -> &str {
        "openrouter"
    }

    async fn fetch(&self) -> FetchOutcome {
        self.fetch_async().await
    }
}

impl OpenRouterAccount {
    /// Reads the key usage and the account credits.
    async fn fetch_async(&self) -> FetchOutcome {
        let mut observation = AccountObservation::default();

        // The key endpoint reports the usage windows. An authorization failure is
        // reported as such rather than swallowed: a provider that rejected the
        // credential must not be recorded as one that reported no account state,
        // because those are different facts and only one of them is actionable.
        match self.get_json("/key").await {
            Ok(body) => {
                if let Some(data) = body.get("data") {
                    // The account this key belongs to, as the provider itself reports it.
                    // Two keys on one account share this value, which is what lets the
                    // collector recognise them as one account instead of counting the
                    // same money twice. It is an opaque identifier, not a secret: it
                    // grants nothing and is never published.
                    if let Some(owner) = data.get("creator_user_id").and_then(Value::as_str) {
                        observation.account_identity = Some(owner.to_string());
                    } else if let Some(workspace) = data.get("workspace_id").and_then(Value::as_str)
                    {
                        // A provisioning key may report no creator. The workspace is
                        // the next identity the provider states outright, and is a
                        // weaker but still provider-reported grouping.
                        observation.account_identity = Some(format!("workspace:{workspace}"));
                    }
                    // The free-model request count. This is per ACCOUNT, not per key:
                    // it is the provider's own tally of every free-model request the
                    // account made today, whichever key sent it. That is precisely why
                    // it is the figure a deployment must admit on -- a local counter
                    // sees only its own traffic -- and why it is published as a
                    // request count with its limit and reset rather than as a
                    // percentage, so it can be reconciled rather than merely displayed.
                    if let Some(reading) = openrouter_free_model_allowance(data, SystemTime::now())
                    {
                        observation.request_allowances.push(reading);
                    }
                    // The key endpoint reports the usage of this key over several
                    // windows. Each window is a different amount, so each is
                    // published under its own period rather than folded into one
                    // series.
                    for (field, period) in [
                        ("usage", ReportingPeriod::Total),
                        ("usage_daily", ReportingPeriod::Daily),
                        ("usage_weekly", ReportingPeriod::Weekly),
                        ("usage_monthly", ReportingPeriod::Monthly),
                    ] {
                        if let Some(usage) = data.get(field).and_then(Value::as_f64) {
                            observation.usage.push(AccountAmount::new(
                                "USD",
                                AccountScope::Key,
                                period,
                                usage,
                            ));
                        }
                    }
                }
            }
            Err(outcome) => return outcome,
        }

        // The credits endpoint reports the account-wide purchased credit and the usage
        // drawn against it. It is a separate request because a provider may authorize
        // one and refuse the other, so a failure here does not discard the reading
        // already taken.
        //
        // The purchased credit and the consumed credit are different facts. The
        // remaining balance is the difference between them, and is published as its
        // own figure so no reader has to subtract one series from another to know
        // what is left.
        if let Ok(body) = self.get_json("/credits").await
            && let Some(data) = body.get("data")
        {
            let credits = data.get("total_credits").and_then(Value::as_f64);
            let consumed = data.get("total_usage").and_then(Value::as_f64);
            if let Some(credits) = credits {
                observation.credits.push(AccountAmount::new(
                    "USD",
                    AccountScope::Account,
                    ReportingPeriod::Total,
                    credits,
                ));
            }
            if let Some(consumed) = consumed {
                observation.usage.push(AccountAmount::new(
                    "USD",
                    AccountScope::Account,
                    ReportingPeriod::Total,
                    consumed,
                ));
            }
            if let Some(remaining) = credits
                .zip(consumed)
                .map(|(credits, consumed)| credits - consumed)
            {
                observation.balances.push(AccountAmount::new(
                    "USD",
                    AccountScope::Account,
                    ReportingPeriod::Total,
                    remaining,
                ));
            }
        }

        if observation.is_empty() {
            return FetchOutcome::Failed("provider reported no account state".to_string());
        }
        // The request-count reading is what a deployment can act on: it is the only
        // statement that covers traffic this host never saw. A figure that contradicts
        // itself is not usable evidence, so it is dropped rather than published, and the
        // window fields the provider did state are kept.
        observation
            .request_allowances
            .retain(|reading| reading.is_consistent());
        FetchOutcome::Ok(Box::new(observation))
    }
}

/// Reads DeepSeek's prepaid balance.
///
/// DeepSeek's balance endpoint is the only account surface it exposes. It reports the
/// granted and topped-up balance per currency, which is enough to show whether the
/// lane is funded.
pub struct DeepSeekAccount {
    api_key: String,
    base_url: String,
}

impl DeepSeekAccount {
    /// Builds the reader from the credential Switchyard serves DeepSeek with.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: "https://api.deepseek.com".to_string(),
        }
    }

    /// Overrides the API base, for a test double.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }
}

#[async_trait::async_trait]
impl AccountApi for DeepSeekAccount {
    fn provider(&self) -> &str {
        "deepseek"
    }

    async fn fetch(&self) -> FetchOutcome {
        let response = account_client()
            .get(format!("{}/user/balance", self.base_url))
            .bearer_auth(&self.api_key)
            .send()
            .await;
        let body = match response {
            Ok(response) => {
                let status = response.status();
                if status == reqwest::StatusCode::UNAUTHORIZED
                    || status == reqwest::StatusCode::FORBIDDEN
                {
                    return FetchOutcome::Unauthorized;
                }
                if !status.is_success() {
                    return FetchOutcome::Failed(format!("HTTP {status}"));
                }
                match response.json::<Value>().await {
                    Ok(body) => body,
                    Err(error) => {
                        return FetchOutcome::Failed(error.to_string());
                    }
                }
            }
            Err(error) => return FetchOutcome::Failed(error.to_string()),
        };
        let mut observation = AccountObservation::default();
        if let Some(balances) = body.get("balance_infos").and_then(Value::as_array) {
            for balance in balances {
                let Some(currency) = balance.get("currency").and_then(Value::as_str) else {
                    continue;
                };
                // The balance endpoint reports the account's balance, not the key's.
                if let Some(total) = balance.get("total_balance").and_then(decode_amount) {
                    observation.balances.push(AccountAmount::new(
                        currency,
                        AccountScope::Account,
                        ReportingPeriod::Total,
                        total,
                    ));
                }
                // The same response breaks the balance into the two kinds of credit it
                // is made of. They are components of the total above, not additions
                // to it: granted credit was given by the provider, topped-up credit
                // was paid for, and `total_balance` is their sum.
                //
                // `topped_up_balance` is the topped-up money still remaining. It is
                // NOT the amount ever purchased, and must never be read as lifetime
                // top-ups: a heavily used account can hold a small remaining topped-up
                // balance after buying a great deal.
                for (field, component) in [
                    ("granted_balance", "granted"),
                    ("topped_up_balance", "topped_up"),
                ] {
                    if let Some(value) = balance.get(field).and_then(decode_amount) {
                        observation
                            .balance_components
                            .push(AccountBalanceComponent {
                                component: component.to_string(),
                                amount: AccountAmount::new(
                                    currency,
                                    AccountScope::Account,
                                    ReportingPeriod::Total,
                                    value,
                                ),
                            });
                    }
                }
            }
        }
        if observation.is_empty() {
            return FetchOutcome::Failed("provider reported no balance".to_string());
        }
        FetchOutcome::Ok(Box::new(observation))
    }
}

/// The provider account surface a client's base URL points at.
///
/// This is decided from the base URL rather than from the client's name, because the
/// name is the operator's label for a lane and carries no guarantee. A deployment
/// that adds a second OpenRouter client called anything at all still points at
/// `openrouter.ai`, and must still get an account reader.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum AccountSurface {
    /// OpenRouter's key and credits endpoints.
    OpenRouter,
    /// DeepSeek's balance endpoint.
    DeepSeek,
    /// The OAuth-backed Codex allowance windows.
    CodexAllowance,
}

impl AccountSurface {
    /// The label this surface's account carries when a deployment names a client
    /// after the provider itself.
    ///
    /// This is not a fallback name for the reader; it is the well-known label a
    /// dashboard already selects on, kept stable so an added client cannot rename an
    /// account that is already being read.
    pub const fn canonical_label(self) -> &'static str {
        match self {
            Self::OpenRouter => "openrouter",
            Self::DeepSeek => "deepseek",
            Self::CodexAllowance => "openai",
        }
    }
}

/// Returns the account surface a base URL points at, if Switchyard can read one.
/// A base URL that matches none of these is a provider whose account surface
/// Switchyard does not read. That is not an error: a local inference server has no
/// account, and a provider may expose an inference API without an account API.
pub fn account_surface_for_base_url(base_url: &str) -> Option<AccountSurface> {
    let url = base_url.to_ascii_lowercase();
    if url.contains("openrouter.ai") {
        Some(AccountSurface::OpenRouter)
    } else if url.contains("api.deepseek.com") {
        Some(AccountSurface::DeepSeek)
    } else if url.contains("chatgpt.com") || url.contains("api.openai.com") {
        Some(AccountSurface::CodexAllowance)
    } else {
        None
    }
}

/// Reads an amount a provider reported.
///
/// A provider may report an amount as a JSON number or as a decimal string, and the
/// two are the same amount. The live DeepSeek balance endpoint reports strings, so
/// a reader that accepted only numbers would find no balance on a funded account
/// and report the lane as unmeasured. A value that is neither is not an amount, and
/// is left absent rather than being read as zero.
fn decode_amount(value: &Value) -> Option<f64> {
    match value {
        Value::Number(number) => number.as_f64(),
        Value::String(text) => text.trim().parse::<f64>().ok(),
        _ => None,
    }
}

/// Reads the allowance windows of an OAuth-backed provider.
///
/// The Codex backend reports a primary window and a secondary window, each with a
/// used percentage and a reset time. The observation keeps them as pools, so the
/// dashboard can show the five-hour window and the weekly window separately.
pub struct CodexAllowanceAccount {
    oauth_token: String,
    account_id: Option<String>,
    base_url: String,
}

impl CodexAllowanceAccount {
    /// Builds the reader from the credential Switchyard serves the Codex lane with.
    pub fn new(oauth_token: impl Into<String>, account_id: Option<String>) -> Self {
        Self {
            oauth_token: oauth_token.into(),
            account_id,
            base_url: "https://chatgpt.com/backend-api".to_string(),
        }
    }

    /// Overrides the API base, for a test double.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }
}

#[async_trait::async_trait]
impl AccountApi for CodexAllowanceAccount {
    fn provider(&self) -> &str {
        "openai"
    }

    async fn fetch(&self) -> FetchOutcome {
        let mut request = account_client()
            .get(format!("{}/wham/usage", self.base_url))
            .bearer_auth(&self.oauth_token);
        if let Some(account_id) = &self.account_id {
            request = request.header("chatgpt-account-id", account_id);
        }
        let response = match request.send().await {
            Ok(response) => response,
            Err(error) => return FetchOutcome::Failed(error.to_string()),
        };
        let status = response.status();
        if status == reqwest::StatusCode::UNAUTHORIZED || status == reqwest::StatusCode::FORBIDDEN {
            return FetchOutcome::Unauthorized;
        }
        if !status.is_success() {
            return FetchOutcome::Failed(format!("HTTP {status}"));
        }
        let body = match response.json::<Value>().await {
            Ok(body) => body,
            Err(error) => return FetchOutcome::Failed(error.to_string()),
        };
        let mut observation = AccountObservation::default();
        let Some(limit) = body.get("rate_limit") else {
            return FetchOutcome::Failed("provider reported no allowance".to_string());
        };
        for (pool, key) in [
            ("primary", "primary_window"),
            ("secondary", "secondary_window"),
        ] {
            let Some(window) = limit.get(key) else {
                continue;
            };
            if let Some(used) = window.get("used_percent").and_then(Value::as_f64) {
                observation
                    .allowance_remaining_percent
                    .insert(pool.to_string(), 100.0 - used);
            }
            if let Some(reset_at) = window.get("reset_at").and_then(Value::as_f64) {
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|elapsed| elapsed.as_secs_f64())
                    .unwrap_or(0.0);
                observation
                    .allowance_reset_seconds
                    .insert(pool.to_string(), (reset_at - now).max(0.0));
            }
        }
        if observation.is_empty() {
            return FetchOutcome::Failed("provider reported no allowance".to_string());
        }
        FetchOutcome::Ok(Box::new(observation))
    }
}

#[cfg(test)]
mod allowance_parsing_tests {
    use super::*;
    use serde_json::json;

    fn utc(text: &str) -> Option<u64> {
        parse_rfc3339_utc(text).map(|at| {
            at.duration_since(UNIX_EPOCH)
                .map(|elapsed| elapsed.as_secs())
                .unwrap_or(0)
        })
    }

    /// A well-formed reset time is read to the second.
    #[test]
    fn a_reset_time_is_parsed_to_the_second() {
        // 2026-01-16T00:00:00Z
        assert_eq!(utc("2026-01-16T00:00:00Z"), Some(1_768_521_600));
        // 2026-01-15T23:00:00Z
        assert_eq!(utc("2026-01-15T23:00:00Z"), Some(1_768_518_000));
    }

    /// A fractional second is truncated, not rejected: the provider states a sub-second
    /// reset and the whole second is the useful precision.
    #[test]
    fn a_fractional_second_is_truncated() {
        assert_eq!(utc("2026-01-16T00:00:00.000Z"), utc("2026-01-16T00:00:00Z"));
    }

    /// A timestamp without the UTC marker is refused. A reset time carries a zone, and
    /// guessing one would place the reset hours away from where the provider meant it.
    #[test]
    fn a_timestamp_without_the_utc_marker_is_refused() {
        assert_eq!(utc("2026-01-16T00:00:00"), None);
        assert_eq!(utc("2026-01-16T00:00:00+07:00"), None);
        assert_eq!(utc("2026-01-16 00:00:00Z"), None);
    }

    /// Missing and malformed shapes are refused rather than defaulted.
    #[test]
    fn a_malformed_timestamp_is_refused() {
        for text in [
            "",
            "Z",
            "2026-01-16T00:00:00",
            "2026-01T00:00:00Z",
            "2026-01-16",
            "not-a-date",
            "2026-01-16T00:00Z",
            "2026-13-16T00:00:00Z",
            "2026-00-16T00:00:00Z",
        ] {
            assert_eq!(utc(text), None, "{text} must not parse");
        }
    }

    /// Out-of-range clock fields are refused. A minute of 99 is not a time, and accepting
    /// it would place the reset an hour and a half from where the provider said.
    #[test]
    fn an_out_of_range_clock_field_is_refused() {
        assert_eq!(utc("2026-01-16T24:00:00Z"), None);
        assert_eq!(utc("2026-01-16T00:60:00Z"), None);
        assert_eq!(utc("2026-01-16T00:00:61Z"), None);
    }

    /// A day that does not exist in its month is refused, not rolled into the next month.
    #[test]
    fn an_impossible_calendar_day_is_refused() {
        assert_eq!(utc("2026-02-31T00:00:00Z"), None);
        assert_eq!(utc("2026-04-31T00:00:00Z"), None);
        assert_eq!(utc("2026-02-29T00:00:00Z"), None);
        // 2024 is a leap year, so its February does have a 29th.
        assert!(utc("2024-02-29T00:00:00Z").is_some());
        // 2000 is divisible by 400; 1900 is not divisible by 400.
        assert!(utc("2000-02-29T00:00:00Z").is_some());
        assert_eq!(utc("1900-02-29T00:00:00Z"), None);
    }

    /// The observed time is what the caller supplies, so a reading is stamped with when it
    /// was taken rather than with whatever the provider's clock happened to say.
    #[test]
    fn the_reading_carries_the_observation_instant_it_was_given() {
        let at = UNIX_EPOCH + Duration::from_secs(1_768_518_000);
        let reading = openrouter_free_model_allowance(
            &json!({
                "free_model_daily_requests": {
                    "used": 12, "limit": 1000, "remaining": 988,
                    "resets_at": "2026-01-16T00:00:00Z"
                }
            }),
            at,
        )
        .expect("a complete block parses");
        assert_eq!(reading.used, 12);
        assert_eq!(reading.limit, 1000);
        assert_eq!(reading.remaining, 988);
        assert_eq!(reading.observed_at, at);
        assert_eq!(reading.pool, "free_model_daily_requests");
        assert_eq!(reading.reset_at, parse_rfc3339_utc("2026-01-16T00:00:00Z"));
    }

    /// An absent block reports nothing. It is not a spent pool and not an empty one.
    #[test]
    fn an_absent_block_reports_nothing() {
        assert!(openrouter_free_model_allowance(&json!({}), UNIX_EPOCH).is_none());
        assert!(
            openrouter_free_model_allowance(
                &json!({"free_model_daily_requests": null}),
                UNIX_EPOCH
            )
            .is_none()
        );
    }

    /// A partially reported block is refused: a limit with no remainder cannot say how much
    /// is left, and filling the gap with the limit would admit the whole budget again.
    #[test]
    fn a_block_missing_a_field_reports_nothing() {
        for block in [
            json!({"used": 1, "limit": 1000}),
            json!({"used": 1, "remaining": 999}),
            json!({"limit": 1000, "remaining": 999}),
        ] {
            assert!(
                openrouter_free_model_allowance(
                    &json!({"free_model_daily_requests": block}),
                    UNIX_EPOCH
                )
                .is_none()
            );
        }
    }

    /// A field of the wrong type is refused rather than coerced. A stringified number is
    /// not the provider's figure, and treating it as absent or as zero would both be wrong.
    #[test]
    fn a_field_of_the_wrong_type_reports_nothing() {
        for block in [
            json!({"used": "12", "limit": 1000, "remaining": 988}),
            json!({"used": 12, "limit": 1000.5, "remaining": 988}),
            json!({"used": -1, "limit": 1000, "remaining": 988}),
            json!({"used": 12, "limit": 1000, "remaining": null}),
        ] {
            assert!(
                openrouter_free_model_allowance(
                    &json!({"free_model_daily_requests": block}),
                    UNIX_EPOCH
                )
                .is_none()
            );
        }
    }

    /// When the two stated headrooms disagree, the smaller one is kept.
    ///
    /// The provider states the remainder outright and implicitly as `limit - used`. These
    /// normally agree; when they do not, its accounting is lagging or something is
    /// reserved, and admitting on the larger figure would spend requests that may not
    /// exist. This is a normalisation rather than a refusal because either way the reading
    /// is usable -- refusing it would silently fall back to the local budget, which is the
    /// less informed authority.
    #[test]
    fn a_contradicting_remainder_is_reduced_to_the_stricter_figure() {
        // `remaining` claims more than the arithmetic allows.
        let reading = openrouter_free_model_allowance(
            &json!({"free_model_daily_requests": {
                "used": 990, "limit": 1000, "remaining": 100
            }}),
            UNIX_EPOCH,
        )
        .expect("a usable reading");
        assert_eq!(reading.remaining, 10, "limit - used is the stricter figure");

        // The arithmetic claims more than `remaining`.
        let reading = openrouter_free_model_allowance(
            &json!({"free_model_daily_requests": {
                "used": 12, "limit": 1000, "remaining": 100
            }}),
            UNIX_EPOCH,
        )
        .expect("a usable reading");
        assert_eq!(reading.remaining, 100, "remaining is the stricter figure");

        // Agreement is preserved exactly.
        let reading = openrouter_free_model_allowance(
            &json!({"free_model_daily_requests": {
                "used": 12, "limit": 1000, "remaining": 988
            }}),
            UNIX_EPOCH,
        )
        .expect("a usable reading");
        assert_eq!(reading.remaining, 988);
    }

    /// A block whose figures exceed its own limit is still refused: `used` above `limit`
    /// has no reading at all that could be true.
    #[test]
    fn a_block_that_exceeds_its_limit_reports_nothing() {
        assert!(
            openrouter_free_model_allowance(
                &json!({"free_model_daily_requests": {
                    "used": 1001, "limit": 1000, "remaining": 0
                }}),
                UNIX_EPOCH
            )
            .is_none()
        );
        assert!(
            openrouter_free_model_allowance(
                &json!({"free_model_daily_requests": {
                    "used": 0, "limit": 1000, "remaining": 1001
                }}),
                UNIX_EPOCH
            )
            .is_none()
        );
    }

    /// An absent or unparseable reset time leaves the field empty instead of inventing one.
    #[test]
    fn an_absent_reset_time_is_left_absent() {
        let block = json!({"free_model_daily_requests": {
            "used": 0, "limit": 1000, "remaining": 1000
        }});
        let reading =
            openrouter_free_model_allowance(&block, UNIX_EPOCH).expect("the figures are complete");
        assert_eq!(reading.reset_at, None);

        let block = json!({"free_model_daily_requests": {
            "used": 0, "limit": 1000, "remaining": 1000, "resets_at": "tomorrow"
        }});
        let reading = openrouter_free_model_allowance(&block, UNIX_EPOCH)
            .expect("the figures are complete without a usable reset");
        assert_eq!(reading.reset_at, None);
    }

    /// Every component shape is validated, so a timestamp this parser only half-reads is
    /// refused instead of stored.
    #[test]
    fn a_timestamp_with_trailing_components_is_refused() {
        for text in [
            "2026-01-16T00:00:00:00Z",
            "2026-01-16-01T00:00:00Z",
            "2026-01-16T00:00Z",
            "2026-01-16T00:00:00+00:00",
        ] {
            assert_eq!(parse_rfc3339_utc(text), None, "{text} was accepted");
        }
    }

    /// A fraction is digits or it is nothing. Text after the seconds is not a timestamp
    /// with ignorable trailing characters.
    #[test]
    fn a_malformed_fraction_is_refused() {
        for text in [
            "2026-01-16T00:00:00.Z",
            "2026-01-16T00:00:00.5abcZ",
            "2026-01-16T00:00:00..5Z",
        ] {
            assert_eq!(parse_rfc3339_utc(text), None, "{text} was accepted");
        }
        assert!(parse_rfc3339_utc("2026-01-16T00:00:00.123456Z").is_some());
    }

    /// The year is bounded, so a malformed figure is refused rather than overflowing the
    /// day arithmetic into a timestamp nobody wrote.
    #[test]
    fn an_unbounded_year_is_refused_rather_than_overflowing() {
        for text in [
            "999999999-01-16T00:00:00Z",
            "0000-01-01T00:00:00Z",
            "1969-12-31T23:59:59Z",
        ] {
            assert_eq!(parse_rfc3339_utc(text), None, "{text} was accepted");
        }
    }

    /// A well-formed timestamp is read exactly, so the parser change above cannot have
    /// made every reset time look absent.
    #[test]
    fn a_well_formed_timestamp_is_read_exactly() {
        assert_eq!(
            parse_rfc3339_utc("2026-01-16T00:00:00Z"),
            Some(UNIX_EPOCH + Duration::from_secs(1_768_521_600))
        );
    }
}
