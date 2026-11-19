// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The input estimator decides context admission, so it must price media as media.
//!
//! A provider prices an inline image by decoded resolution (a few hundred to a few
//! thousand tokens) and never by the length of its base64 text. Charging `chars / 4` for a
//! `data:` URI therefore turns an ordinary screenshot into a quarter-million-token prompt
//! and makes every child whose window is smaller than that misestimate look ineligible.

use serde_json::json;
use switchyard_runner::request_fit::estimate_input_tokens;

/// Roughly ten thousand tokens of genuine text, plus one ~1 MiB inline PNG.
///
/// The text alone must still be charged in full; the image must not add three hundred
/// thousand tokens to the estimate.
#[test]
fn an_inline_image_is_not_priced_by_its_base64_length() {
    let text = "q".repeat(40_000); // about 10_000 tokens
    let base64 = "A".repeat(1_400_000); // about 350_000 tokens if priced as text
    let body = json!({
        "model": "m",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "input_text", "text": text},
                {
                    "type": "input_image",
                    "image_url": format!("data:image/png;base64,{base64}")
                }
            ]
        }]
    });

    let estimate = estimate_input_tokens(&body);
    assert!(
        estimate < 32_768,
        "an inline image must be charged as media, not as {} tokens of text: got {estimate}",
        base64.len() / 4
    );
    assert!(
        estimate >= 10_000,
        "the accompanying text must still be charged in full: got {estimate}"
    );
}

/// An inline video is media too: bounded, not length-priced. The bound is deliberately
/// larger than the image allowance, because a decoded video carries far more tokens.
#[test]
fn an_inline_video_is_bounded_rather_than_length_priced() {
    let base64 = "A".repeat(4_000_000); // about 1_000_000 tokens if priced as text
    let body = json!({
        "model": "m",
        "input": [{
            "role": "user",
            "content": [{
                "type": "input_video",
                "video_url": format!("data:video/mp4;base64,{base64}")
            }]
        }]
    });

    let estimate = estimate_input_tokens(&body);
    assert!(
        estimate < 262_144,
        "an inline video must stay bounded: got {estimate}"
    );
}

/// A long document is still a long document: the fix must not discount ordinary text.
#[test]
fn a_large_text_prompt_is_still_estimated_in_full() {
    let body = json!({
        "model": "m",
        "messages": [{"role": "user", "content": "q".repeat(400_000)}],
    });

    let estimate = estimate_input_tokens(&body);
    assert!(
        estimate >= 99_000,
        "four hundred thousand ASCII characters are about a hundred thousand tokens: got {estimate}"
    );
}

/// A `data:` URI whose media type is not media is still text, and is charged as such.
#[test]
fn a_non_media_data_uri_is_still_text() {
    let payload = "A".repeat(40_000); // about 10_000 tokens
    let body = json!({
        "model": "m",
        "messages": [{
            "role": "user",
            "content": format!("data:text/plain;base64,{payload}")
        }]
    });

    let estimate = estimate_input_tokens(&body);
    assert!(
        estimate >= 9_000,
        "a text data URI is text: got {estimate}"
    );
}
