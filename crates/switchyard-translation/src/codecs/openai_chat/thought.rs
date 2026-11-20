// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Hosted reasoning markers, and the filter that keeps them out of the answer.
//!
//! Some free hosted models emit their chain of thought inline in the ordinary message text
//! instead of a separate reasoning channel. Google's hosted Gemma endpoints on the
//! OpenAI-compatible surface do exactly this: a reply arrives as
//!
//! ```text
//! <thought>The user wants a single word…</thought>ready
//! ```
//!
//! and `usage.completion_tokens` counts only `ready` while `total_tokens` counts both. A
//! transport that treats the whole string as the answer therefore returns the model's
//! private monologue to the caller, which is a correctness and a privacy defect at once:
//! downstream agents read it as output, it lands in transcripts, and it silently changes
//! token accounting because the visible answer no longer matches the billed completion.
//!
//! The fix is a marker, not a heuristic. A target states which marker it uses; the codec
//! splits the block into a reasoning block and the answer that follows. Text that merely
//! resembles the marker does not match, because only a marker at the start of the message
//! opens a block.
//!
//! Two decisions here are deliberate and both are refusals rather than repairs:
//!
//! * an unterminated block is an error, not "assume the rest is the answer" — the answer is
//!   unknown, and guessing would either leak thoughts or truncate a reply;
//! * a closing marker with no opening one is an error too, because the visible text is then
//!   known to be incomplete in a way this layer cannot reconstruct.
//!
//! Refusing is safe here because the caller sees a translation failure and the route's
//! ordinary failure handling applies; silently emitting half a thought is not recoverable.

use serde::{Deserialize, Serialize};

/// The opening token of an inline hosted reasoning block.
pub const THOUGHT_OPEN: &str = "<thought>";
/// The closing token of an inline hosted reasoning block.
pub const THOUGHT_CLOSE: &str = "</thought>";

/// Which inline reasoning marker an upstream uses.
///
/// One variant today. It is an enum rather than a boolean so that a second provider with a
/// different marker does not require a second field and a second flag on every call.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HostedReasoningMarker {
    /// `<thought>…</thought>` preceding the answer, as hosted Gemma emits it.
    ThoughtTag,
}

impl HostedReasoningMarker {
    /// The marker a manifest string names, or `None` for an unrecognised value.
    ///
    /// Returning `Option` rather than defaulting is what makes a typo in a target
    /// configuration a load error instead of a silently disabled filter.
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "thought_tag" => Some(Self::ThoughtTag),
            _ => None,
        }
    }

    /// The opening and closing tokens for this marker.
    pub const fn tokens(self) -> (&'static str, &'static str) {
        match self {
            Self::ThoughtTag => (THOUGHT_OPEN, THOUGHT_CLOSE),
        }
    }
}

/// Why a message could not be split into reasoning and answer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ThoughtLeak {
    /// An opening marker with no closing one.
    Unterminated,
    /// A closing marker that no opening marker introduced.
    UnexpectedClose,
    /// An opening marker after the answer had already begun.
    MisplacedOpen,
    /// The stream ended while still inside a reasoning block.
    UnterminatedStream,
    /// A streamed block would exceed the sanity bound, so the marker is not what it seemed.
    Oversized,
}

impl std::fmt::Display for ThoughtLeak {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let text = match self {
            Self::Unterminated => "an inline reasoning block was opened and never closed",
            Self::UnexpectedClose => "an inline reasoning block was closed without being opened",
            Self::MisplacedOpen => "an inline reasoning marker appeared after the answer began",
            Self::UnterminatedStream => "the stream ended inside an inline reasoning block",
            Self::Oversized => "an inline reasoning block exceeded its bound",
        };
        formatter.write_str(text)
    }
}

impl std::error::Error for ThoughtLeak {}

/// A message split into its private reasoning and its answer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Split<'a> {
    /// The reasoning block's content, when the message carried one.
    pub reasoning: Option<&'a str>,
    /// The text a caller may see.
    pub answer: &'a str,
}

/// Splits one complete message into reasoning and answer.
///
/// Only a marker at the start of the message opens a block. A `<thought>` anywhere later is
/// reported rather than stripped, because it means the upstream's shape is not the one this
/// filter was written for.
pub fn split_hosted_reasoning(text: &str) -> Result<Split<'_>, ThoughtLeak> {
    let Some(rest) = text.strip_prefix(THOUGHT_OPEN) else {
        // An opening marker after the answer began is reported before a stray closing one:
        // it is the stronger signal, and a message carrying both is confusing in a way
        // "closed without opening" would describe badly.
        if text.contains(THOUGHT_OPEN) {
            return Err(ThoughtLeak::MisplacedOpen);
        }
        if text.contains(THOUGHT_CLOSE) {
            return Err(ThoughtLeak::UnexpectedClose);
        }
        return Ok(Split {
            reasoning: None,
            answer: text,
        });
    };
    let Some(end) = rest.find(THOUGHT_CLOSE) else {
        return Err(ThoughtLeak::Unterminated);
    };
    let reasoning = &rest[..end];
    let answer = &rest[end + THOUGHT_CLOSE.len()..];
    // The block and the answer are glued together by a newline in practice; it belongs to
    // neither, and leaving it would put a stray blank line at the top of every answer.
    let answer = answer.trim_start_matches(['\n', '\r']);
    if answer.contains(THOUGHT_OPEN) || answer.contains(THOUGHT_CLOSE) {
        return Err(ThoughtLeak::MisplacedOpen);
    }
    Ok(Split {
        reasoning: Some(reasoning),
        answer,
    })
}

/// One piece of filtered output.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Filtered {
    /// Text belonging to the private reasoning block.
    Reasoning(String),
    /// Text a caller may see.
    Answer(String),
}

/// An upper bound on a buffered reasoning block.
///
/// A marker that opens a block and never closes it would otherwise accumulate the rest of
/// the response in memory. The bound is generous enough for a long deliberation and small
/// enough that it cannot be mistaken for a runaway buffer.
const MAX_REASONING: usize = 256 * 1024;

/// Streaming counterpart of [`split_hosted_reasoning`].
///
/// The filter holds back only as much text as it needs: enough to tell whether the message
/// begins with the marker, and then the block itself until its close arrives. Answer text is
/// released as soon as the block ends, so a streaming caller is not delayed by a preamble it
/// was never going to see.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ThoughtStreamFilter {
    buffer: String,
    phase: Phase,
    reasoning_len: usize,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
enum Phase {
    /// Nothing decided: the message may or may not start with a marker.
    #[default]
    Start,
    /// Inside a reasoning block.
    Reasoning,
    /// Past the block; everything from here is answer text.
    Answer,
}

impl ThoughtStreamFilter {
    /// A filter that has seen nothing.
    pub fn new() -> Self {
        Self::default()
    }

    /// Takes one delta and returns whatever is now safe to emit.
    pub fn push(&mut self, delta: &str) -> Result<Vec<Filtered>, ThoughtLeak> {
        if delta.is_empty() {
            return Ok(Vec::new());
        }
        match self.phase {
            Phase::Answer => Ok(vec![Filtered::Answer(delta.to_string())]),
            Phase::Start => {
                self.buffer.push_str(delta);
                // A short buffer that could still become the opening marker must wait: any
                // output now could turn out to be the marker's own prefix.
                if THOUGHT_OPEN.starts_with(self.buffer.as_str())
                    && self.buffer.len() < THOUGHT_OPEN.len()
                {
                    return Ok(Vec::new());
                }
                if let Some(rest) = self.buffer.strip_prefix(THOUGHT_OPEN).map(str::to_string) {
                    self.phase = Phase::Reasoning;
                    self.buffer.clear();
                    return self.scan_reasoning(&rest);
                }
                self.phase = Phase::Answer;
                let buffered = std::mem::take(&mut self.buffer);
                Ok(vec![Filtered::Answer(buffered)])
            }
            Phase::Reasoning => {
                self.buffer.push_str(delta);
                let buffered = std::mem::take(&mut self.buffer);
                self.scan_reasoning(&buffered)
            }
        }
    }

    /// Ends the message.
    ///
    /// A message that ended inside a reasoning block is refused: the answer was never
    /// emitted and inventing one from a truncated block is exactly the leak this filter
    /// exists to prevent.
    pub fn finish(&mut self) -> Result<Vec<Filtered>, ThoughtLeak> {
        let buffered = std::mem::take(&mut self.buffer);
        match self.phase {
            Phase::Reasoning => Err(ThoughtLeak::UnterminatedStream),
            Phase::Answer => {
                self.phase = Phase::Answer;
                Ok(if buffered.is_empty() {
                    Vec::new()
                } else {
                    vec![Filtered::Answer(buffered)]
                })
            }
            Phase::Start => {
                // A buffer that never reached the marker is the answer itself: a real
                // message does not begin with a partial marker and then stop.
                self.phase = Phase::Answer;
                Ok(if buffered.is_empty() {
                    Vec::new()
                } else {
                    vec![Filtered::Answer(buffered)]
                })
            }
        }
    }

    fn scan_reasoning(&mut self, text: &str) -> Result<Vec<Filtered>, ThoughtLeak> {
        let Some(end) = text.find(THOUGHT_CLOSE) else {
            self.reasoning_len = self.reasoning_len.saturating_add(text.len());
            if self.reasoning_len > MAX_REASONING {
                return Err(ThoughtLeak::Oversized);
            }
            self.buffer.push_str(text);
            return Ok(Vec::new());
        };
        let mut events = Vec::new();
        let head = &text[..end];
        let mut reasoning = std::mem::take(&mut self.buffer);
        reasoning.push_str(head);
        if !reasoning.is_empty() {
            events.push(Filtered::Reasoning(reasoning));
        }
        self.phase = Phase::Answer;
        let answer = &text[end + THOUGHT_CLOSE.len()..];
        let answer = answer.trim_start_matches(['\n', '\r']);
        if !answer.is_empty() {
            events.push(Filtered::Answer(answer.to_string()));
        }
        Ok(events)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_message_without_a_marker_is_all_answer() {
        let split = split_hosted_reasoning("ready").expect("no marker");
        assert_eq!(split.reasoning, None);
        assert_eq!(split.answer, "ready");
    }

    #[test]
    fn a_measured_gemma_reply_splits_into_reasoning_and_answer() {
        // The exact shape recorded from gemma-4-26b-a4b-it on 2026-11-21.
        let text = "<thought>The user is asking for a single-word reply: \"ready\".</thought>ready";
        let split = split_hosted_reasoning(text).expect("split");
        assert_eq!(
            split.reasoning,
            Some("The user is asking for a single-word reply: \"ready\".")
        );
        assert_eq!(split.answer, "ready");
    }

    #[test]
    fn the_newline_between_block_and_answer_does_not_reach_the_caller() {
        let split = split_hosted_reasoning("<thought>plan</thought>\n\nThe answer").expect("split");
        assert_eq!(split.answer, "The answer");
    }

    #[test]
    fn an_unterminated_block_is_refused_rather_than_guessed() {
        assert_eq!(
            split_hosted_reasoning("<thought>still thinking").unwrap_err(),
            ThoughtLeak::Unterminated
        );
    }

    #[test]
    fn a_close_without_an_open_is_refused() {
        assert_eq!(
            split_hosted_reasoning("answer</thought>").unwrap_err(),
            ThoughtLeak::UnexpectedClose
        );
    }

    #[test]
    fn a_marker_after_the_answer_is_refused_not_stripped() {
        assert_eq!(
            split_hosted_reasoning("answer <thought>late</thought>").unwrap_err(),
            ThoughtLeak::MisplacedOpen
        );
    }

    #[test]
    fn a_second_block_in_the_answer_is_refused() {
        assert_eq!(
            split_hosted_reasoning("<thought>a</thought>then <thought>b</thought>").unwrap_err(),
            ThoughtLeak::MisplacedOpen
        );
    }

    #[test]
    fn an_unclosed_marker_mid_text_is_refused_as_ambiguous() {
        // A target that states this marker is telling us that these tokens mean something.
        // A `<thought>` in the middle with no close is either a block this layer cannot
        // reconstruct or text that happens to look like one, and the two cannot be told
        // apart from the string alone. Refusing is the only answer that never leaks.
        assert_eq!(
            split_hosted_reasoning("Use <thought> only at the start of a message.").unwrap_err(),
            ThoughtLeak::MisplacedOpen
        );
    }

    #[test]
    fn text_with_neither_token_is_untouched() {
        let text = "Use angle brackets sparingly.";
        assert_eq!(split_hosted_reasoning(text).expect("split").answer, text);
    }

    #[test]
    fn a_partial_marker_is_ordinary_text() {
        // `"<thought"` never becomes a block, so it is answer text and is returned as-is.
        let split = split_hosted_reasoning("<thought").expect("split");
        assert_eq!(split.reasoning, None);
        assert_eq!(split.answer, "<thought");
    }

    #[test]
    fn streaming_holds_back_a_possible_marker_prefix() {
        let mut filter = ThoughtStreamFilter::new();
        assert!(filter.push("<tho").unwrap().is_empty(), "released a prefix");
        let events = filter.push("ught>plan</thought>hello").unwrap();
        assert_eq!(
            events,
            vec![
                Filtered::Reasoning("plan".to_string()),
                Filtered::Answer("hello".to_string()),
            ]
        );
    }

    #[test]
    fn streaming_releases_an_answer_that_can_no_longer_be_a_marker() {
        let mut filter = ThoughtStreamFilter::new();
        let events = filter.push("Hello").unwrap();
        assert_eq!(events, vec![Filtered::Answer("Hello".to_string())]);
        // Once past the marker, later text passes straight through.
        let events = filter.push(" world").unwrap();
        assert_eq!(events, vec![Filtered::Answer(" world".to_string())]);
    }

    #[test]
    fn streaming_reassembles_a_block_split_across_deltas() {
        let mut filter = ThoughtStreamFilter::new();
        assert!(filter.push("<thought>").unwrap().is_empty());
        assert!(filter.push("half ").unwrap().is_empty());
        let events = filter.push("done</thought>").unwrap();
        assert_eq!(events, vec![Filtered::Reasoning("half done".to_string())]);
    }

    #[test]
    fn a_stream_that_ends_inside_a_block_is_refused() {
        let mut filter = ThoughtStreamFilter::new();
        filter.push("<thought>never closed").unwrap();
        assert_eq!(
            filter.finish().unwrap_err(),
            ThoughtLeak::UnterminatedStream
        );
    }

    #[test]
    fn a_short_plain_message_flushes_at_end() {
        let mut filter = ThoughtStreamFilter::new();
        // Shorter than the marker, so it is still buffered when the stream ends.
        assert!(filter.push("<th").unwrap().is_empty());
        let events = filter.finish().unwrap();
        assert_eq!(events, vec![Filtered::Answer("<th".to_string())]);
    }

    #[test]
    fn a_thought_block_longer_than_the_bound_is_refused() {
        let mut filter = ThoughtStreamFilter::new();
        filter.push("<thought>").unwrap();
        let chunk = "x".repeat(64 * 1024);
        let mut error = None;
        for _ in 0..8 {
            match filter.push(&chunk) {
                Ok(_) => {}
                Err(leak) => {
                    error = Some(leak);
                    break;
                }
            }
        }
        assert_eq!(error, Some(ThoughtLeak::Oversized));
    }

    #[test]
    fn a_marker_name_is_parsed_strictly() {
        // A typo must be a configuration error, not a silently disabled filter: the whole
        // point of the marker is that text is only moved when a target asks for it.
        assert_eq!(
            HostedReasoningMarker::parse("thought_tag"),
            Some(HostedReasoningMarker::ThoughtTag)
        );
        for rejected in ["thought", "ThoughtTag", "thought-tag", "", " thought_tag"] {
            assert_eq!(HostedReasoningMarker::parse(rejected), None, "{rejected:?}");
        }
    }
}
