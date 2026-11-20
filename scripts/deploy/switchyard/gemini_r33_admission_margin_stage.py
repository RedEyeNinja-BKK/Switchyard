#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pad the two free Gemma lanes' per-minute admission so a near-limit request is refused.

The owner's AI Studio reading records Gemma 26B at 18.05K input tokens against a 16K
per-minute limit. The request that caused it was estimated at 15,202 tokens by the local
estimator and counted as 18,054 by the provider: the estimator under-counted by 18.8%, and
admission reserved against the smaller number, so a request that could not fit was sent.

A pad is the narrow fix. Each reservation is multiplied by 1,300 per mille, so a request is
admitted only when its padded estimate fits the whole minute. The largest estimate the
16,000-token window can then admit is 12,307, and the per-request ceiling is lowered to match
-- a ceiling the governor would refuse on every large request is a promise, not a limit.

The pad is not a claim of accuracy. The two measurements this project has disagree in
direction (one over-count near 24%, one under-count near 16-19%), so no fixed error bound is
asserted here; the pad is a margin against the measured overshoot and the send path learns the
reported count back into the ledger for the life of the window.

The Lite lanes are untouched. Their window is 250,000 tokens, where the same estimator error
is absorbed by the window rather than crossing it, and narrowing their ceiling would withdraw
capacity the provider actually grants.

This is not a config-only release. `admission_margin_permille` is a field the running binary
does not know, and load refuses an unknown field, so the candidate manifest is paired with the
binary that implements the pad. The live R3.2 binary cannot read this document.
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import sys
import tomllib

MARGIN = 1_300
WINDOW = 16_000
# floor(16_000 * 1000 / 1_300) -- the send path's own arithmetic, repeated here so the manifest
# states the ceiling the governor will enforce instead of one it would refuse.
SAFE = WINDOW * 1_000 // MARGIN

TARGETS = (
    "direct-google-gemma-4-31b-it",
    "direct-google-gemma-4-26b-a4b-it",
)

# The 31B block states why the ceiling sits where it does; that sentence becomes false once the
# pad exists, and a comment that contradicts its own field is drift the next reader trusts.
COMMENTS: list[tuple[str, str]] = [
    (
        "# The free-tier row for this project: 30 requests and 16,000 input tokens a\n"
        "# minute, 14,400 requests a day. The per-request ceiling sits at the minute's\n"
        "# token budget, because a larger request can never be admitted however long it\n"
        "# waits, and stating the window lets admission skip the leg.\n",
        "# The free-tier row for this project: 30 requests and 16,000 input tokens a\n"
        "# minute, 14,400 requests a day. Admission pads every reservation by 1,300 per\n"
        "# mille, because the estimator under-counted this lane once already -- 15,202\n"
        "# estimated against 18,054 provider tokens -- and a request that cannot fit the\n"
        "# minute must be refused before egress rather than after it. The ceiling is the\n"
        "# largest estimate the padded window admits, not the window itself: a ceiling the\n"
        "# governor refuses on every large request states capacity that does not exist.\n",
    ),
]


def blocks(lines: list[str]) -> dict[str, tuple[int, int]]:
    starts = [
        (line.strip(), index)
        for index, line in enumerate(lines)
        if line.startswith("[") and line.strip().endswith("]")
    ]
    spans = {}
    for position, (header, start) in enumerate(starts):
        end = starts[position + 1][1] if position + 1 < len(starts) else len(lines)
        spans[header] = (start, end)
    return spans


def splice(text: str) -> tuple[str, list[str]]:
    lines = text.splitlines(keepends=True)
    spans = blocks(lines)
    notes: list[str] = []
    ordered = sorted(TARGETS, key=lambda name: spans.get(f'[targets."{name}"]', (0, 0))[0], reverse=True)
    for name in ordered:
        header = f'[targets."{name}"]'
        if header not in spans:
            raise SystemExit(f"target {header} is not in the deployed manifest")
        start, end = spans[header]
        body = "".join(lines[start:end])

        for before, after in COMMENTS:
            if body.count(before) != 1:
                continue
            body = body.replace(before, after, 1)
            if before != after:
                notes.append(f"{name}: comment corrected")

        ceiling = "max_input_tokens = 16000"
        if body.count(ceiling) != 1:
            raise SystemExit(f"{name}: ceiling anchor matched {body.count(ceiling)} times")
        body = body.replace(ceiling, f"max_input_tokens = {SAFE}", 1)
        notes.append(f"{name}: max_input_tokens = 16000 -> {SAFE}")

        if "admission_margin_permille" in body:
            raise SystemExit(f"{name}: already declares a margin; refusing to stack pads")
        note = (
            "# The pad the estimator's measured under-count on this lane requires.\n"
            f"admission_margin_permille = {MARGIN}\n"
        )
        anchor = "input_tokens_per_minute = 16000"
        if body.count(anchor) != 1:
            raise SystemExit(f"{name}: token-window anchor matched {body.count(anchor)} times")
        if body.count(anchor + "\n") == 1:
            body = body.replace(anchor + "\n", anchor + "\n" + note, 1)
        else:
            raise SystemExit(f"{name}: token-window line is not newline-terminated")
        notes.append(f"{name}: + admission_margin_permille = {MARGIN}")
        lines[start:end] = [body]
    return "".join(lines), notes


def verify(candidate: str, live: str) -> dict:
    doc = tomllib.loads(candidate)
    live_doc = tomllib.loads(live)
    targets = doc.get("targets", {})
    failures: list[str] = []

    for name in TARGETS:
        target = targets.get(name, {})
        if target.get("admission_margin_permille") != MARGIN:
            failures.append(f"{name}: margin is {target.get('admission_margin_permille')!r}")
        if target.get("max_input_tokens") != SAFE:
            failures.append(f"{name}: ceiling is {target.get('max_input_tokens')!r}, want {SAFE}")
        window = target.get("input_tokens_per_minute")
        if window != WINDOW:
            failures.append(f"{name}: window moved to {window!r}")
        if target.get("max_input_tokens", 0) > window * 1_000 // MARGIN:
            failures.append(f"{name}: ceiling above the padded window")
        for field in ("requests_per_minute", "daily_request_limit", "free_tier", "hosted_reasoning"):
            if target.get(field) != live_doc["targets"][name].get(field):
                failures.append(f"{name}.{field} moved")

    allowed = {f'[targets."{n}"]' for n in TARGETS}
    for name, target in live_doc.get("targets", {}).items():
        after = targets.get(name)
        if after is None:
            failures.append(f"{name}: target disappeared")
            continue
        if f'[targets."{name}"]' in allowed:
            continue
        if after != target:
            failures.append(f"{name}: an unpadded target changed")

    for section in ("routes", "composites", "llm_clients"):
        if doc.get(section) != live_doc.get(section):
            failures.append(f"{section} changed")
    if doc.get("schema_version") != live_doc.get("schema_version"):
        failures.append("schema_version changed")

    if failures:
        for failure in failures:
            print(f"FAIL {failure}", file=sys.stderr)
        raise SystemExit(1)
    return {"targets": len(targets), "safe": SAFE}


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
        f"targets {summary['targets']} safe_ceiling {summary['safe']}"
    )


if __name__ == "__main__":
    main()
