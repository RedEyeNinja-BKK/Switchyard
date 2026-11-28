// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `POST /v1/robotics/spatial` — embodied spatial reasoning over a scene image.
//!
//! `gemini-robotics-er-2-preview` is a `generateContent` model, but it is not a chat lane
//! with a picture attached: it answers a *typed* spatial question — normalized coordinates
//! for objects, boxes, or a trajectory. This surface therefore owns the whole contract:
//!
//! * the instruction is built here from the requested task, so the JSON contract the model
//!   is asked to follow cannot drift from the one the response is validated against;
//! * the response is parsed and validated against that contract, and prose is refused
//!   rather than returned — a caller that asked for coordinates and received a sentence has
//!   been handed something it cannot use;
//! * the scene is measured before dispatch (container, dimensions, byte ceiling) so the
//!   request cannot spend a minute window on an image nobody sized.
//!
//! Nothing here actuates equipment. The output is coordinates and labels; a separate
//! controller decides what, if anything, to do with them. The streaming Robotics variant
//! (`gemini-robotics-er-2-streaming-preview`, `bidiGenerateContent`) is a different model
//! with a different wire and is deliberately not reachable from this surface.

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use base64::Engine as _;
use serde::Deserialize;
use serde_json::{Value, json};
use switchyard_runner::capability::ResolvedCapability;

use crate::ServerState;
use crate::capability_surfaces::{release_physical_call, reserve_physical_call_for};

/// A spatial question the model can be asked.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpatialTask {
    /// Point at objects, returning one normalized `[y, x]` per label.
    Point,
    /// Detect objects, returning normalized bounding boxes.
    Boxes,
    /// Return ordered waypoints with an action per step.
    Trajectory,
}

impl SpatialTask {
    fn name(self) -> &'static str {
        match self {
            Self::Point => "point",
            Self::Boxes => "boxes",
            Self::Trajectory => "trajectory",
        }
    }

    /// The instruction the model is given, including the exact JSON shape expected back.
    ///
    /// The validation in [`parse_spatial`] mirrors this text field for field. Changing one
    /// without the other would make the refusals lie about what was asked for.
    fn instruction(self, max_objects: u32) -> String {
        match self {
            Self::Point => format!(
                "Point to no more than {max_objects} items in the image. The label returned \
                 should be an identifying name for the object detected. The answer should \
                 follow the json format: [{{\"point\": [y, x], \"label\": <label>}}, ...]. \
                 The points are in [y, x] format normalized to 0-1000. Return only the JSON \
                 array, with no prose and no code fence."
            ),
            Self::Boxes => format!(
                "Detect no more than {max_objects} objects in this image and return bounding \
                 boxes. The answer should follow the JSON format: [{{\"label\": <label>, \
                 \"y\": <y_min>, \"x\": <x_min>, \"y2\": <y_max>, \"x2\": <x_max>}}, ...] \
                 where coordinates are normalized to 0-1000 and y <= y2, x <= x2. Return \
                 only the JSON array, with no prose and no code fence."
            ),
            Self::Trajectory => format!(
                "Generate a trajectory with no more than {max_objects} waypoints. Return a \
                 list of waypoints as JSON: [{{\"step\": <int>, \"point\": [y, x], \
                 \"action\": <description>}}, ...] where coordinates are normalized to \
                 0-1000 and step increases from 0. Return only the JSON array, with no prose \
                 and no code fence."
            ),
        }
    }
}

/// Request body.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpatialRequest {
    /// Capability id to serve this request.
    pub model: String,
    /// The scene to reason about.
    pub scene: Scene,
    /// Which spatial question to ask.
    pub task: SpatialTask,
    /// Additional instruction appended to the declared contract.
    #[serde(default)]
    pub instruction: Option<String>,
    /// Largest number of items the answer may carry.
    #[serde(default)]
    pub max_objects: Option<u32>,
}

/// The scene image, inline.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Scene {
    /// Base64 image bytes.
    pub data: String,
    /// Container of those bytes.
    pub mime_type: String,
}

/// Most items one answer may carry when the caller names no bound.
const DEFAULT_MAX_OBJECTS: u32 = 10;
/// Hard ceiling on items, so a caller cannot ask for an unbounded answer.
const MAX_OBJECTS_CEILING: u32 = 64;
/// Longest additional instruction, in characters.
const MAX_INSTRUCTION_CHARS: usize = 2000;

/// Why a spatial response is unusable.
///
/// Every variant names what was wrong and where, because the caller's next move depends on
/// it: a range violation means the provider answered badly, an empty array means it found
/// nothing and the caller may want a different prompt.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SpatialInvalid {
    /// The provider's text was not JSON at all.
    NotJson { detail: String },
    /// The JSON was not an array of the expected shape.
    NotAnArray,
    /// An entry is not an object.
    EntryNotObject { index: usize },
    /// An entry carries a field the task does not define.
    UnknownField { index: usize, field: String },
    /// A required field is missing.
    MissingField { index: usize, field: &'static str },
    /// A numeric coordinate is outside 0..=1000 or is not an integer.
    BadCoordinate {
        index: usize,
        field: String,
        value: String,
    },
    /// A label or action is empty, or too long to be a label.
    BadText { index: usize, field: &'static str },
    /// A box has y2 < y or x2 < x.
    InvertedBox { index: usize },
    /// Steps are not strictly increasing from zero.
    UnorderedSteps { index: usize },
    /// More items than the request allowed.
    TooMany { count: usize, allowed: u32 },
}

impl SpatialInvalid {
    /// A caller-facing explanation, naming the item and field.
    pub fn message(&self) -> String {
        match self {
            Self::NotJson { detail } => {
                format!("the provider answered with text that is not JSON: {detail}")
            }
            Self::NotAnArray => {
                "the provider answered with JSON that is not an array of spatial items".to_string()
            }
            Self::EntryNotObject { index } => {
                format!("item {index} is not an object")
            }
            Self::UnknownField { index, field } => {
                format!("item {index} carries field {field:?}, which the task does not define")
            }
            Self::MissingField { index, field } => {
                format!("item {index} is missing required field {field:?}")
            }
            Self::BadCoordinate {
                index,
                field,
                value,
            } => format!(
                "item {index} field {field:?} is {value}, which is not an integer in 0..=1000"
            ),
            Self::BadText { index, field } => {
                format!("item {index} field {field:?} must be non-empty text")
            }
            Self::InvertedBox { index } => format!(
                "item {index} has an inverted box (y2 < y or x2 < x), which describes no area"
            ),
            Self::UnorderedSteps { index } => format!(
                "item {index} breaks the step ordering; steps must increase from 0 by one \
                 position each"
            ),
            Self::TooMany { count, allowed } => {
                format!("the answer carries {count} items but the request allowed {allowed}")
            }
        }
    }

    /// A stable code for machine matching.
    pub fn code(&self) -> &'static str {
        match self {
            Self::NotJson { .. } => "spatial_not_json",
            Self::NotAnArray => "spatial_not_an_array",
            Self::EntryNotObject { .. } => "spatial_entry_not_object",
            Self::UnknownField { .. } => "spatial_unknown_field",
            Self::MissingField { .. } => "spatial_missing_field",
            Self::BadCoordinate { .. } => "spatial_bad_coordinate",
            Self::BadText { .. } => "spatial_bad_text",
            Self::InvertedBox { .. } => "spatial_inverted_box",
            Self::UnorderedSteps { .. } => "spatial_unordered_steps",
            Self::TooMany { .. } => "spatial_too_many",
        }
    }
}

/// Dimensions of a scene image, in pixels, read from its container header.
///
/// The container is inspected rather than trusted: the token estimate that admits a request
/// depends on the pixel count, so an image whose size cannot be read is refused instead of
/// being charged at a guessed cost.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageSize {
    pub width: u32,
    pub height: u32,
}

/// Reads the pixel dimensions of a PNG, JPEG or WebP image.
pub fn image_size(bytes: &[u8], mime_type: &str) -> Option<ImageSize> {
    match mime_type {
        "image/png" => png_size(bytes),
        "image/jpeg" | "image/jpg" => jpeg_size(bytes),
        "image/webp" => webp_size(bytes),
        _ => None,
    }
}

fn png_size(bytes: &[u8]) -> Option<ImageSize> {
    if bytes.len() < 24 || &bytes[..8] != b"\x89PNG\r\n\x1a\n" {
        return None;
    }
    // The first chunk of a PNG must be IHDR, whose payload starts at byte 16.
    if &bytes[12..16] != b"IHDR" {
        return None;
    }
    let width = u32::from_be_bytes(bytes[16..20].try_into().ok()?);
    let height = u32::from_be_bytes(bytes[20..24].try_into().ok()?);
    (width > 0 && height > 0).then_some(ImageSize { width, height })
}

fn jpeg_size(bytes: &[u8]) -> Option<ImageSize> {
    if bytes.len() < 4 || bytes[0] != 0xFF || bytes[1] != 0xD8 {
        return None;
    }
    let mut index = 2usize;
    while index + 9 < bytes.len() {
        if bytes[index] != 0xFF {
            index += 1;
            continue;
        }
        let marker = bytes[index + 1];
        // SOF0..SOF3, SOF5..SOF7, SOF9..SOF11, SOF13..SOF15 carry the frame dimensions.
        // SOF4 (0xC4) and SOF8/SOF12 (0xC8/0xCC) are Huffman/arithmetic tables, not frames.
        if matches!(marker,
            0xC0..=0xC3 | 0xC5..=0xC7 | 0xC9..=0xCB | 0xCD..=0xCF)
        {
            let height = u16::from_be_bytes([bytes[index + 5], bytes[index + 6]]) as u32;
            let width = u16::from_be_bytes([bytes[index + 7], bytes[index + 8]]) as u32;
            return (width > 0 && height > 0).then_some(ImageSize { width, height });
        }
        let length = u16::from_be_bytes([bytes[index + 2], bytes[index + 3]]) as usize;
        if length < 2 {
            return None;
        }
        index += 2 + length;
    }
    None
}

fn webp_size(bytes: &[u8]) -> Option<ImageSize> {
    if bytes.len() < 30 || &bytes[..4] != b"RIFF" || &bytes[8..12] != b"WEBP" {
        return None;
    }
    match &bytes[12..16] {
        b"VP8X" => {
            // Canvas size is stored as two 24-bit little-endian values minus one.
            let width = 1 + u32::from_le_bytes([bytes[24], bytes[25], bytes[26], 0]);
            let height = 1 + u32::from_le_bytes([bytes[27], bytes[28], bytes[29], 0]);
            Some(ImageSize { width, height })
        }
        b"VP8 " => {
            // A key frame's start code is followed by 14-bit dimensions.
            if bytes.len() < 30 || &bytes[23..26] != b"\x9d\x01\x2a" {
                return None;
            }
            let width = u16::from_le_bytes([bytes[26], bytes[27]]) as u32 & 0x3FFF;
            let height = u16::from_le_bytes([bytes[28], bytes[29]]) as u32 & 0x3FFF;
            (width > 0 && height > 0).then_some(ImageSize { width, height })
        }
        b"VP8L" => {
            if bytes.len() < 25 || bytes[20] != 0x2F {
                return None;
            }
            let bits = u32::from_le_bytes([bytes[21], bytes[22], bytes[23], bytes[24]]);
            let width = 1 + (bits & 0x3FFF);
            let height = 1 + ((bits >> 14) & 0x3FFF);
            Some(ImageSize { width, height })
        }
        _ => None,
    }
}

/// Tokens one image is estimated to occupy, from the documented tiling.
///
/// Google's documented image accounting fits the picture into 768x768 tiles of 258 tokens
/// each (a single tile for anything at or below 384x384 in both dimensions). The estimate is
/// used only to admit or refuse a request against the minute window; the provider's own
/// `usageMetadata.promptTokenCount` is what the receipt compares it against.
pub fn estimated_image_tokens(size: ImageSize) -> u64 {
    if size.width <= 384 && size.height <= 384 {
        return 258;
    }
    let tiles_x = size.width.div_ceil(768) as u64;
    let tiles_y = size.height.div_ceil(768) as u64;
    tiles_x * tiles_y * 258
}

/// Rationale-free estimate of the text part, in tokens.
fn estimated_text_tokens(text: &str) -> u64 {
    (text.len() as u64).div_ceil(4).max(1)
}

/// Parses and validates a provider answer against the task's contract.
pub fn parse_spatial(
    task: SpatialTask,
    text: &str,
    max_objects: u32,
) -> Result<Vec<Value>, SpatialInvalid> {
    let cleaned = strip_code_fence(text);
    let value: Value = serde_json::from_str(cleaned).map_err(|error| SpatialInvalid::NotJson {
        detail: error.to_string(),
    })?;
    let Some(items) = value.as_array() else {
        return Err(SpatialInvalid::NotAnArray);
    };
    if items.len() > max_objects as usize {
        return Err(SpatialInvalid::TooMany {
            count: items.len(),
            allowed: max_objects,
        });
    }
    let mut validated = Vec::with_capacity(items.len());
    let mut previous_step: Option<i64> = None;
    for (index, item) in items.iter().enumerate() {
        let Some(object) = item.as_object() else {
            return Err(SpatialInvalid::EntryNotObject { index });
        };
        let allowed: &[&str] = match task {
            SpatialTask::Point => &["point", "label"],
            SpatialTask::Boxes => &["label", "y", "x", "y2", "x2"],
            SpatialTask::Trajectory => &["step", "point", "action"],
        };
        for field in object.keys() {
            if !allowed.contains(&field.as_str()) {
                return Err(SpatialInvalid::UnknownField {
                    index,
                    field: field.clone(),
                });
            }
        }

        match task {
            SpatialTask::Point => {
                let point = read_point(object, index, "point")?;
                read_text(object, index, "label")?;
                validated.push(json!({ "point": point, "label": object["label"] }));
            }
            SpatialTask::Boxes => {
                let y = read_coordinate(object, index, "y")?;
                let x = read_coordinate(object, index, "x")?;
                let y2 = read_coordinate(object, index, "y2")?;
                let x2 = read_coordinate(object, index, "x2")?;
                if y2 < y || x2 < x {
                    return Err(SpatialInvalid::InvertedBox { index });
                }
                read_text(object, index, "label")?;
                validated.push(json!({
                    "label": object["label"], "y": y, "x": x, "y2": y2, "x2": x2
                }));
            }
            SpatialTask::Trajectory => {
                let Some(step) = object.get("step").and_then(Value::as_i64) else {
                    return Err(SpatialInvalid::MissingField {
                        index,
                        field: "step",
                    });
                };
                let expected = previous_step.map_or(0, |previous| previous + 1);
                if step != expected {
                    return Err(SpatialInvalid::UnorderedSteps { index });
                }
                previous_step = Some(step);
                let point = read_point(object, index, "point")?;
                read_text(object, index, "action")?;
                validated.push(json!({
                    "step": step, "point": point, "action": object["action"]
                }));
            }
        }
    }
    Ok(validated)
}

/// Removes a surrounding Markdown code fence, which models add despite being asked not to.
///
/// Applied before parsing only; the validated output is rebuilt from the parsed values, so a
/// fence can never survive into the response.
fn strip_code_fence(text: &str) -> &str {
    let trimmed = text.trim();
    let Some(rest) = trimmed.strip_prefix("```") else {
        return trimmed;
    };
    // Drop the language tag on the fence line.
    let rest = match rest.find('\n') {
        Some(newline) => &rest[newline + 1..],
        None => rest,
    };
    rest.strip_suffix("```").unwrap_or(rest).trim()
}

fn read_coordinate(
    object: &serde_json::Map<String, Value>,
    index: usize,
    field: &str,
) -> Result<i64, SpatialInvalid> {
    let Some(value) = object.get(field) else {
        return Err(SpatialInvalid::MissingField {
            index,
            field: leak(field),
        });
    };
    match value.as_i64() {
        Some(number) if (0..=1000).contains(&number) => Ok(number),
        _ => Err(SpatialInvalid::BadCoordinate {
            index,
            field: field.to_string(),
            value: value.to_string(),
        }),
    }
}

fn read_point(
    object: &serde_json::Map<String, Value>,
    index: usize,
    field: &str,
) -> Result<Value, SpatialInvalid> {
    let Some(value) = object.get(field) else {
        return Err(SpatialInvalid::MissingField {
            index,
            field: leak(field),
        });
    };
    let Some(pair) = value.as_array().filter(|pair| pair.len() == 2) else {
        return Err(SpatialInvalid::BadCoordinate {
            index,
            field: field.to_string(),
            value: value.to_string(),
        });
    };
    let mut coordinates = [0i64; 2];
    for (position, coordinate) in pair.iter().enumerate() {
        match coordinate.as_i64() {
            Some(number) if (0..=1000).contains(&number) => coordinates[position] = number,
            _ => {
                return Err(SpatialInvalid::BadCoordinate {
                    index,
                    field: field.to_string(),
                    value: value.to_string(),
                });
            }
        }
    }
    Ok(json!([coordinates[0], coordinates[1]]))
}

fn read_text(
    object: &serde_json::Map<String, Value>,
    index: usize,
    field: &str,
) -> Result<(), SpatialInvalid> {
    let Some(value) = object.get(field) else {
        return Err(SpatialInvalid::MissingField {
            index,
            field: leak(field),
        });
    };
    let Some(text) = value.as_str() else {
        return Err(SpatialInvalid::BadText {
            index,
            field: leak(field),
        });
    };
    let trimmed = text.trim();
    if trimmed.is_empty() || trimmed.chars().count() > 200 {
        return Err(SpatialInvalid::BadText {
            index,
            field: leak(field),
        });
    }
    Ok(())
}

/// A `&'static str` from a compile-time field name.
///
/// The field names passed to these readers are literals in this module, so the leak is
/// bounded and happens once per name rather than per request.
fn leak(field: &str) -> &'static str {
    match field {
        "point" => "point",
        "label" => "label",
        "action" => "action",
        "y" => "y",
        "x" => "x",
        "y2" => "y2",
        "x2" => "x2",
        "step" => "step",
        other => Box::leak(other.to_string().into_boxed_str()),
    }
}

/// `POST /v1/robotics/spatial`.
pub(crate) async fn spatial(
    State(state): State<ServerState>,
    body: std::result::Result<Json<SpatialRequest>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let Json(request) = match body {
        Ok(parsed) => parsed,
        Err(error) => return contract_error(StatusCode::BAD_REQUEST, error.body_text()),
    };
    let Some(capability) = state.runner.capability(&request.model).cloned() else {
        return unknown_capability(&request.model, &state.capability_ids);
    };
    if !capability.format.serves_spatial() {
        return contract_error(
            StatusCode::BAD_REQUEST,
            format!(
                "capability {} does not answer spatial questions",
                capability.id
            ),
        );
    }

    let max_objects = request
        .max_objects
        .unwrap_or(DEFAULT_MAX_OBJECTS)
        .clamp(1, MAX_OBJECTS_CEILING);
    let Ok(image) = base64::engine::general_purpose::STANDARD.decode(&request.scene.data) else {
        return contract_error(
            StatusCode::BAD_REQUEST,
            "scene.data is not valid base64".to_string(),
        );
    };
    if image.is_empty() {
        return contract_error(
            StatusCode::BAD_REQUEST,
            "scene.data decodes to zero bytes".to_string(),
        );
    }
    if let Some(limit) = capability.max_image_bytes
        && image.len() as u64 > limit
    {
        return contract_error(
            StatusCode::BAD_REQUEST,
            format!(
                "scene is {} bytes but capability {} accepts at most {limit}",
                image.len(),
                capability.id
            ),
        );
    }
    let Some(size) = image_size(&image, &request.scene.mime_type) else {
        return contract_error(
            StatusCode::BAD_REQUEST,
            format!(
                "scene.mime_type {:?} is not a measurand container: this surface reads PNG, \
                 JPEG and WebP headers so the request's token cost is known before dispatch",
                request.scene.mime_type
            ),
        );
    };

    let mut instruction = request.task.instruction(max_objects);
    if let Some(extra) = &request.instruction {
        let trimmed = extra.trim();
        if trimmed.is_empty() {
            return contract_error(
                StatusCode::BAD_REQUEST,
                "instruction, when present, must not be empty".to_string(),
            );
        }
        if trimmed.chars().count() > MAX_INSTRUCTION_CHARS {
            return contract_error(
                StatusCode::BAD_REQUEST,
                format!(
                    "instruction is {} characters but at most {MAX_INSTRUCTION_CHARS} are accepted",
                    trimmed.chars().count()
                ),
            );
        }
        instruction.push_str("\n\nAdditional instruction: ");
        instruction.push_str(trimmed);
    }

    let estimated_tokens = estimated_image_tokens(size) + estimated_text_tokens(&instruction);
    if let Some(limit) = capability.max_input_tokens
        && estimated_tokens > limit as u64
    {
        return contract_error(
            StatusCode::BAD_REQUEST,
            format!(
                "a {}x{} scene plus its instruction estimates {estimated_tokens} tokens, above \
                 capability {}'s {limit}-token input limit",
                size.width, size.height, capability.id
            ),
        );
    }

    let client = match state.capability_http(capability.timeout_seconds) {
        Ok(client) => client,
        Err(message) => return contract_error(StatusCode::BAD_GATEWAY, message),
    };
    let credential = match &capability.api_key_env {
        None => None,
        Some(variable) => match std::env::var(variable) {
            Ok(value) if !value.trim().is_empty() => Some(value),
            _ => return upstream_contract_error(&capability, "credential unavailable"),
        },
    };

    let payload = json!({
        "contents": [{"parts": [
            {"inlineData": {"mimeType": request.scene.mime_type, "data": request.scene.data}},
            {"text": instruction}
        ]}],
        "generationConfig": {"responseMimeType": "application/json"}
    });

    // One caller request is one physical call: the model has no fan-out and no batch form
    // on this surface.
    let ticket = match reserve_physical_call_for(&capability, estimated_tokens) {
        Ok(ticket) => ticket,
        Err(refusal) => return *refusal,
    };
    let mut builder = client.post(&capability.url).json(&payload);
    if let Some(credential) = &credential {
        builder = builder.header("x-goog-api-key", credential);
    }
    let response = match builder.send().await {
        Ok(response) => response,
        Err(error) => {
            release_physical_call(&capability, &ticket);
            return contract_error(
                StatusCode::BAD_GATEWAY,
                format!("capability {} is unreachable: {}", capability.id, error),
            );
        }
    };
    let status =
        StatusCode::from_u16(response.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let body = response.bytes().await.unwrap_or_default();
    if !status.is_success() {
        release_physical_call(&capability, &ticket);
        return error_body(
            status,
            format!(
                "capability {} failed: {}",
                capability.id,
                String::from_utf8_lossy(&body)
            ),
            "upstream_error",
        );
    }
    let Ok(value) = serde_json::from_slice::<Value>(&body) else {
        return upstream_contract_error(&capability, "upstream returned non-JSON success");
    };
    let Some(parts) = value["candidates"][0]["content"]["parts"].as_array() else {
        return upstream_contract_error(
            &capability,
            "no content part in the response (candidates[0].content.parts)",
        );
    };
    let text: String = parts
        .iter()
        .filter_map(|part| part["text"].as_str())
        .collect::<Vec<_>>()
        .join("");
    if text.trim().is_empty() {
        let reason = value["candidates"][0]["finishReason"]
            .as_str()
            .unwrap_or("no finishReason");
        return upstream_contract_error(
            &capability,
            &format!("the model returned no spatial answer (finishReason {reason})"),
        );
    }
    let items = match parse_spatial(request.task, &text, max_objects) {
        Ok(items) => items,
        Err(invalid) => {
            return error_body(
                StatusCode::BAD_GATEWAY,
                format!(
                    "capability {} answered outside its spatial contract ({}): {}",
                    capability.id,
                    invalid.code(),
                    invalid.message()
                ),
                "spatial_contract_error",
            );
        }
    };

    let reported = value["usageMetadata"]["promptTokenCount"].as_u64();
    let mut headers = axum::http::HeaderMap::new();
    if let Ok(value) = request.task.name().parse() {
        headers.insert("x-switchyard-spatial-task", value);
    }
    if let Ok(value) = items.len().to_string().parse() {
        headers.insert("x-switchyard-spatial-count", value);
    }
    if let Ok(value) = format!("{estimated_tokens}").parse() {
        headers.insert("x-switchyard-estimated-tokens", value);
    }
    if let Some(reported) = reported
        && let Ok(value) = reported.to_string().parse()
    {
        headers.insert("x-switchyard-provider-prompt-tokens", value);
    }
    (
        StatusCode::OK,
        headers,
        Json(json!({
            "model": capability.id,
            "task": request.task.name(),
            "items": items,
            "count": items.len(),
            "usage": {
                "estimated_prompt_tokens": estimated_tokens,
                "provider_prompt_tokens": reported,
            }
        })),
    )
        .into_response()
}

fn error_body(status: StatusCode, message: String, kind: &str) -> Response {
    (
        status,
        axum::Json(serde_json::json!({"error": {"message": message, "type": kind}})),
    )
        .into_response()
}

fn unknown_capability(model: &str, available: &[String]) -> Response {
    (
        StatusCode::NOT_FOUND,
        axum::Json(serde_json::json!({
            "error": {
                "message": format!(
                    "No capability registered for model {model}. Available capabilities: {}",
                    if available.is_empty() {
                        "(none configured)".to_string()
                    } else {
                        available.join(", ")
                    }
                ),
                "type": "model_not_found"
            }
        })),
    )
        .into_response()
}

fn upstream_contract_error(capability: &ResolvedCapability, reason: &str) -> Response {
    (
        StatusCode::BAD_GATEWAY,
        axum::Json(serde_json::json!({
            "error": {
                "message": format!(
                    "capability {} upstream contract mismatch: {reason}",
                    capability.id
                ),
                "type": "upstream_contract_error"
            }
        })),
    )
        .into_response()
}

fn contract_error(status: StatusCode, message: String) -> Response {
    (
        status,
        axum::Json(
            serde_json::json!({"error": {"message": message, "type": "invalid_request_error"}}),
        ),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_point_answer_is_validated_and_rebuilt() {
        let text = r#"[{"point":[376,508],"label":" small banana "},{"point":[287,609],"label":"teacup"}]"#;
        let items = parse_spatial(SpatialTask::Point, text, 10).expect("valid");
        assert_eq!(items.len(), 2);
        assert_eq!(
            items[0],
            json!({"point": [376, 508], "label": " small banana "})
        );
    }

    #[test]
    fn prose_is_refused_rather_than_returned() {
        let invalid = parse_spatial(
            SpatialTask::Point,
            "I found a banana in the upper left of the image.",
            10,
        )
        .expect_err("prose is not a spatial answer");
        assert_eq!(invalid.code(), "spatial_not_json");
        assert!(invalid.message().contains("not JSON"));
    }

    #[test]
    fn a_coordinate_outside_the_normalized_range_is_refused() {
        let invalid = parse_spatial(
            SpatialTask::Point,
            r#"[{"point":[1200,10],"label":"wall"}]"#,
            10,
        )
        .expect_err("out of range");
        assert!(matches!(invalid, SpatialInvalid::BadCoordinate { .. }));
        assert!(invalid.message().contains("0..=1000"));
    }

    #[test]
    fn an_unknown_field_is_refused_by_name() {
        let invalid = parse_spatial(
            SpatialTask::Point,
            r#"[{"point":[1,2],"label":"x","confidence":0.9}]"#,
            10,
        )
        .expect_err("confidence is not in the contract");
        assert_eq!(
            invalid,
            SpatialInvalid::UnknownField {
                index: 0,
                field: "confidence".to_string()
            }
        );
    }

    #[test]
    fn an_inverted_box_is_refused() {
        let invalid = parse_spatial(
            SpatialTask::Boxes,
            r#"[{"label":"plate","y":300,"x":100,"y2":200,"x2":400}]"#,
            10,
        )
        .expect_err("inverted");
        assert!(matches!(invalid, SpatialInvalid::InvertedBox { index: 0 }));
    }

    #[test]
    fn a_box_answer_is_validated() {
        let items = parse_spatial(
            SpatialTask::Boxes,
            r#"[{"label":"cup","y":100,"x":200,"y2":300,"x2":400}]"#,
            10,
        )
        .expect("valid");
        assert_eq!(
            items[0],
            json!({"label":"cup","y":100,"x":200,"y2":300,"x2":400})
        );
    }

    #[test]
    fn trajectory_steps_must_start_at_zero_and_increase() {
        let good = r#"[{"step":0,"point":[1,2],"action":"approach"},{"step":1,"point":[3,4],"action":"grasp"}]"#;
        assert_eq!(
            parse_spatial(SpatialTask::Trajectory, good, 10)
                .expect("valid")
                .len(),
            2
        );

        let skipped =
            r#"[{"step":0,"point":[1,2],"action":"a"},{"step":2,"point":[3,4],"action":"b"}]"#;
        assert!(matches!(
            parse_spatial(SpatialTask::Trajectory, skipped, 10),
            Err(SpatialInvalid::UnorderedSteps { index: 1 })
        ));
    }

    #[test]
    fn an_answer_over_the_requested_count_is_refused() {
        let text = r#"[{"point":[1,2],"label":"a"},{"point":[3,4],"label":"b"}]"#;
        assert!(matches!(
            parse_spatial(SpatialTask::Point, text, 1),
            Err(SpatialInvalid::TooMany {
                count: 2,
                allowed: 1
            })
        ));
    }

    #[test]
    fn a_code_fence_is_tolerated_but_never_survives_into_the_answer() {
        let text = "```json\n[{\"point\":[10,20],\"label\":\"ball\"}]\n```";
        let items = parse_spatial(SpatialTask::Point, text, 10).expect("fenced JSON");
        assert_eq!(items[0]["label"], json!("ball"));
        assert!(!items[0].to_string().contains("```"));
    }

    #[test]
    fn an_empty_array_is_a_valid_find_nothing_answer() {
        // Zero items is not a violation: a scene with no requested object is a real result
        // and the caller decides what it means. Only malformed or out-of-contract answers
        // are refused.
        assert!(
            parse_spatial(SpatialTask::Point, "[]", 10)
                .expect("valid")
                .is_empty()
        );
    }

    /// A minimal PNG header with the given dimensions.
    fn png(width: u32, height: u32) -> Vec<u8> {
        let mut bytes = b"\x89PNG\r\n\x1a\n".to_vec();
        bytes.extend_from_slice(&13u32.to_be_bytes());
        bytes.extend_from_slice(b"IHDR");
        bytes.extend_from_slice(&width.to_be_bytes());
        bytes.extend_from_slice(&height.to_be_bytes());
        bytes.extend_from_slice(&[8, 2, 0, 0, 0]);
        bytes
    }

    #[test]
    fn png_dimensions_are_read_from_the_header() {
        assert_eq!(
            image_size(&png(1024, 768), "image/png"),
            Some(ImageSize {
                width: 1024,
                height: 768
            })
        );
        // A truncated header cannot be measured, and an unmeasurable scene is refused.
        assert_eq!(image_size(&png(1024, 768)[..12], "image/png"), None);
        assert_eq!(
            image_size(b"not a png at all, but long enough", "image/png"),
            None
        );
        assert_eq!(image_size(&png(4, 4), "image/gif"), None);
    }

    #[test]
    fn jpeg_dimensions_are_read_from_the_frame_header() {
        let mut bytes = vec![0xFF, 0xD8];
        // APP0 segment of length 4 (2 marker bytes + 2 length bytes), then SOF0.
        bytes.extend_from_slice(&[0xFF, 0xE0, 0x00, 0x04, 0x00, 0x00]);
        bytes.extend_from_slice(&[0xFF, 0xC0, 0x00, 0x11, 0x08]);
        bytes.extend_from_slice(&480u16.to_be_bytes());
        bytes.extend_from_slice(&640u16.to_be_bytes());
        bytes.extend_from_slice(&[0x03, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(
            image_size(&bytes, "image/jpeg"),
            Some(ImageSize {
                width: 640,
                height: 480
            })
        );
    }

    #[test]
    fn webp_dimensions_are_read_from_the_vp8x_header() {
        let mut bytes = b"RIFF".to_vec();
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(b"WEBP");
        bytes.extend_from_slice(b"VP8X");
        bytes.extend_from_slice(&[0u8; 8]);
        // width-1 and height-1 as 24-bit little-endian.
        bytes.extend_from_slice(&[0xFF, 0x03, 0x00]);
        bytes.extend_from_slice(&[0x57, 0x02, 0x00]);
        assert_eq!(
            image_size(&bytes, "image/webp"),
            Some(ImageSize {
                width: 1024,
                height: 600
            })
        );
    }

    #[test]
    fn the_image_token_estimate_follows_the_documented_tiling() {
        // At or below 384x384 in both dimensions, one tile.
        assert_eq!(
            estimated_image_tokens(ImageSize {
                width: 384,
                height: 384
            }),
            258
        );
        // Over 384 in either dimension, the picture is cropped into 768x768 tiles: 800 wide
        // is two tiles, 400 tall is one, so 2 x 258.
        assert_eq!(
            estimated_image_tokens(ImageSize {
                width: 800,
                height: 400
            }),
            2 * 258
        );
        // Exactly one tile is one tile; one pixel past it is two.
        assert_eq!(
            estimated_image_tokens(ImageSize {
                width: 768,
                height: 768
            }),
            258
        );
        assert_eq!(
            estimated_image_tokens(ImageSize {
                width: 769,
                height: 768
            }),
            2 * 258
        );
        // A 1536x1536 image is 2x2 tiles.
        assert_eq!(
            estimated_image_tokens(ImageSize {
                width: 1536,
                height: 1536
            }),
            4 * 258
        );
    }

    #[test]
    fn each_task_instruction_states_the_shape_the_validator_accepts() {
        assert!(SpatialTask::Point.instruction(10).contains("[y, x]"));
        assert!(SpatialTask::Boxes.instruction(10).contains("\"y2\""));
        assert!(SpatialTask::Trajectory.instruction(10).contains("\"step\""));
        assert!(SpatialTask::Point.instruction(7).contains("no more than 7"));
    }
}
