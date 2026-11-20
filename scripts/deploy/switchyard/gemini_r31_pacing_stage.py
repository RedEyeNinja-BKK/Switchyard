#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Add per-minute admission to the four Google targets of a deployed manifest.

This is the R3.1 change and it is deliberately narrow: the R3 adaptive lane is already
deployed, so the manifest this runs against *contains* the Gemma and Lite targets. Four
additive fields declare the provider's measured request and input-token windows; two
per-request ceilings move from the documentation figure (16,000) to the measured one
(14,400), because a ceiling above the minute's budget can never be served.

The script edits the deployed file in place block by block, refuses if any anchor is
missing or ambiguous, and re-parses the result before writing it. Hand-editing a
1,700-line production manifest is how a deployment gains a regression nobody diffed.
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import sys
import tomllib

# (target header, lines appended to its block, replacements inside the block)
EDITS: dict[str, tuple[list[str], dict[str, str]]] = {
    '[targets."direct-google-gemini-3.5-flash-lite"]': (
        [
            "# The owner's measured free-tier row for this model: 10 requests a minute,",
            "# 500 a day. A burst is what actually spends a day, so the minute is declared.",
            "requests_per_minute = 10",
        ],
        {},
    ),
    '[targets."direct-google-gemini-3.1-flash-lite"]': (
        ["requests_per_minute = 12"],
        {},
    ),
    '[targets."direct-google-gemma-4-31b-it"]': (
        [
            "# 14,400 input tokens a minute is this project's measured free-tier window for",
            "# Gemma. The per-request ceiling is lowered to match: a request larger than a",
            "# whole minute can never be admitted, and stating the window lets admission skip",
            "# the leg instead of spending a call on a 429.",
            "requests_per_minute = 15",
            "input_tokens_per_minute = 14400",
        ],
        {"max_input_tokens = 16000": "max_input_tokens = 14400"},
    ),
    '[targets."direct-google-gemma-4-26b-a4b-it"]': (
        ["requests_per_minute = 7", "input_tokens_per_minute = 14400"],
        {"max_input_tokens = 16000": "max_input_tokens = 14400"},
    ),
}

# Every field this script may introduce, so the verification can prove nothing else moved.
INTRODUCED = ("requests_per_minute", "input_tokens_per_minute")


def blocks(lines: list[str]) -> list[tuple[str, int, int]]:
    """(header, start, end) for every table header in the document."""
    starts = [
        (line.strip(), index)
        for index, line in enumerate(lines)
        if line.startswith("[") and line.strip().endswith("]")
    ]
    out = []
    for position, (header, start) in enumerate(starts):
        end = starts[position + 1][1] if position + 1 < len(starts) else len(lines)
        out.append((header, start, end))
    return out


def splice(text: str) -> tuple[str, list[str]]:
    lines = text.splitlines(keepends=True)
    notes: list[str] = []
    spans = {header: (start, end) for header, start, end in blocks(lines)}
    # Bottom up: each splice changes the line count, so earlier spans stay valid only if
    # the edits below them are applied first.
    ordered = sorted(
        EDITS.items(), key=lambda item: spans.get(item[0], (0, 0))[0], reverse=True
    )
    for header, (additions, replacements) in ordered:
        if header not in spans:
            raise SystemExit(f"target {header} is not in the deployed manifest")
        start, end = spans[header]
        body = "".join(lines[start:end])
        if any(addition.split(" = ")[0] in body for addition in additions):
            raise SystemExit(f"target {header} already declares a per-minute limit")
        for anchor, replacement in replacements.items():
            if body.count(anchor) != 1:
                raise SystemExit(
                    f"{header}: anchor {anchor!r} matched {body.count(anchor)} times"
                )
            body = body.replace(anchor, replacement)
            notes.append(f"{header}: {anchor} -> {replacement}")
        # Append after the block's last non-blank line so the insert stays inside the table.
        trailing = len(body) - len(body.rstrip("\n"))
        content = body[: len(body) - trailing] if trailing else body
        content = content.rstrip("\n") + "\n" + "\n".join(additions) + "\n"
        content += "\n" * trailing
        lines[start:end] = [content]
        notes.append(f"{header}: +{len(additions)} line(s)")
    return "".join(lines), notes


def verify(candidate: str, live: str) -> dict:
    doc = tomllib.loads(candidate)
    live_doc = tomllib.loads(live)
    targets = doc.get("targets", {})
    failures: list[str] = []

    expected = {
        "direct-google-gemini-3.5-flash-lite": (10, None),
        "direct-google-gemini-3.1-flash-lite": (12, None),
        "direct-google-gemma-4-31b-it": (15, 14400),
        "direct-google-gemma-4-26b-a4b-it": (7, 14400),
    }
    for name, (rpm, tpm) in expected.items():
        target = targets.get(name, {})
        if target.get("requests_per_minute") != rpm:
            failures.append(f"{name}: requests_per_minute is {target.get('requests_per_minute')!r}")
        if target.get("input_tokens_per_minute") != tpm:
            failures.append(f"{name}: input_tokens_per_minute is {target.get('input_tokens_per_minute')!r}")
        ceiling = target.get("max_input_tokens")
        if tpm is not None and (ceiling is None or ceiling > tpm):
            failures.append(f"{name}: max_input_tokens {ceiling!r} exceeds the minute budget")
        if target.get("daily_request_limit") is None:
            failures.append(f"{name}: the daily budget disappeared")

    # Nothing else may have gained a window, and no other target may have changed.
    for name, target in targets.items():
        if name not in expected and any(field in target for field in INTRODUCED):
            failures.append(f"{name}: gained a per-minute limit it did not have")

    for name, target in live_doc.get("targets", {}).items():
        after = targets.get(name)
        if after is None:
            failures.append(f"{name}: target disappeared")
            continue
        for field, value in target.items():
            if field in INTRODUCED:
                continue
            if after.get(field) != value and not (
                name.startswith("direct-google-gemma") and field == "max_input_tokens"
            ):
                failures.append(f"{name}.{field}: {value!r} -> {after.get(field)!r}")

    if doc.get("routes") != live_doc.get("routes"):
        failures.append("routes changed")
    if doc.get("composites") != live_doc.get("composites"):
        failures.append("composites changed")
    if doc.get("llm_clients") != live_doc.get("llm_clients"):
        failures.append("llm_clients changed")

    if failures:
        for failure in failures:
            print(f"FAIL {failure}", file=sys.stderr)
        raise SystemExit(1)
    return {"targets": len(targets), "routes": len(doc.get("routes", {}))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    live = pathlib.Path(args.live).read_text()
    candidate, notes = splice(live)
    summary = verify(candidate, live)
    pathlib.Path(args.out).write_text(candidate)
    print("\n".join(notes))
    print(
        f"live sha256 {hashlib.sha256(live.encode()).hexdigest()}\n"
        f"candidate sha256 {hashlib.sha256(candidate.encode()).hexdigest()}\n"
        f"targets {summary['targets']} routes {summary['routes']}"
    )


if __name__ == "__main__":
    main()
