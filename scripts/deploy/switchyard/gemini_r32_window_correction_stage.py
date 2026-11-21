#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Correct the four Google targets' per-minute windows to the researched table.

The R3.1 rollout transposed two columns of the free-tier table: the Gemma targets were
given 7 and 15 requests a minute and a 14,400-token window, where the table states
30 requests, 16,000 input tokens and a 14,400 *daily* request budget; the Lite targets were
given 10 and 12 requests a minute from the owner's used-of-500 UI reading rather than the
published 15 requests and 250,000 tokens a minute.

A too-small window is not harmless: it makes the router treat a 26B leg as scarce and cross
to a more expensive candidate, and it refuses requests the provider would have served. This
script rewrites exactly those fields on the deployed manifest and proves nothing else moved.
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import sys
import tomllib

# The comment that justified the transposed value must move with it: a line that states
# 14,400 or "10 requests" beside a 30/16,000 field is drift the next reader would trust.
COMMENTS: dict[str, list[tuple[str, str]]] = {
    '[targets."direct-google-gemma-4-31b-it"]': [
        (
            "# 14,400 input tokens a minute is this project's measured free-tier window for\n"
            "# Gemma. The per-request ceiling is lowered to match: a request larger than a\n"
            "# whole minute can never be admitted, and stating the window lets admission skip\n"
            "# the leg instead of spending a call on a 429.\n",
            "# The free-tier row for this project: 30 requests and 16,000 input tokens a\n"
            "# minute, 14,400 requests a day. The per-request ceiling sits at the minute's\n"
            "# token budget, because a larger request can never be admitted however long it\n"
            "# waits, and stating the window lets admission skip the leg.\n",
        )
    ],
    '[targets."direct-google-gemini-3.5-flash-lite"]': [
        (
            "# The owner's measured free-tier row for this model: 10 requests a minute,\n"
            "# 500 a day. A burst is what actually spends a day, so the minute is declared.\n",
            "# The free-tier row: 15 requests and 250,000 input tokens a minute, 500 a day.\n"
            "# A burst is what actually spends a day, so the minute is declared.\n",
        )
    ],
}

# target header -> {field: new value}
EDITS: dict[str, dict[str, int]] = {
    '[targets."direct-google-gemini-3.5-flash-lite"]': {
        "requests_per_minute": 15,
        "input_tokens_per_minute": 250_000,
    },
    '[targets."direct-google-gemini-3.1-flash-lite"]': {
        "requests_per_minute": 15,
        "input_tokens_per_minute": 250_000,
    },
    '[targets."direct-google-gemma-4-31b-it"]': {
        "requests_per_minute": 30,
        "input_tokens_per_minute": 16_000,
        "max_input_tokens": 16_000,
    },
    '[targets."direct-google-gemma-4-26b-a4b-it"]': {
        "requests_per_minute": 30,
        "input_tokens_per_minute": 16_000,
        "max_input_tokens": 16_000,
    },
}


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
    ordered = sorted(EDITS.items(), key=lambda item: spans.get(item[0], (0, 0))[0], reverse=True)
    for header, fields in ordered:
        if header not in spans:
            raise SystemExit(f"target {header} is not in the deployed manifest")
        start, end = spans[header]
        body = "".join(lines[start:end])
        for before, after in COMMENTS.get(header, []):
            if body.count(before) != 1:
                raise SystemExit(f"{header}: comment anchor matched {body.count(before)} times")
            body = body.replace(before, after, 1)
            notes.append(f"{header}: comment corrected")
        for field, value in fields.items():
            prefix = f"{field} = "
            matches = [line for line in body.splitlines() if line.startswith(prefix)]
            if len(matches) > 1:
                raise SystemExit(f"{header}: {field} appears {len(matches)} times")
            if matches:
                before = matches[0]
                body = body.replace(before, f"{prefix}{value}", 1)
                notes.append(f"{header}: {before.strip()} -> {field} = {value}")
                continue
            # Absent: R3.1 declared only the request axis on the Lite targets. Insert the
            # token axis beside it so the window is stated in one place.
            anchor = "requests_per_minute = "
            anchors = [line for line in body.splitlines() if line.startswith(anchor)]
            if len(anchors) != 1:
                raise SystemExit(
                    f"{header}: cannot place {field}; {len(anchors)} request-axis anchors"
                )
            body = body.replace(anchors[0], f"{anchors[0]}\n{prefix}{value}", 1)
            notes.append(f"{header}: + {field} = {value}")
        lines[start:end] = [body]
    return "".join(lines), notes


def verify(candidate: str, live: str) -> dict:
    doc = tomllib.loads(candidate)
    live_doc = tomllib.loads(live)
    targets = doc.get("targets", {})
    failures: list[str] = []

    for header, fields in EDITS.items():
        name = header[len("[targets.") : -1].strip('"')
        target = targets.get(name, {})
        for field, value in fields.items():
            if target.get(field) != value:
                failures.append(f"{name}.{field} is {target.get(field)!r}, expected {value!r}")
        tokens = target.get("input_tokens_per_minute")
        ceiling = target.get("max_input_tokens")
        if tokens and (ceiling is None or ceiling > tokens):
            failures.append(f"{name}: ceiling {ceiling!r} above the minute budget {tokens}")
        if target.get("daily_request_limit") is None:
            failures.append(f"{name}: the daily budget disappeared")
        if target.get("free_tier") is not True:
            failures.append(f"{name}: free_tier was lost")

    for name, target in live_doc.get("targets", {}).items():
        after = targets.get(name)
        if after is None:
            failures.append(f"{name}: target disappeared")
            continue
        allowed = EDITS.get(f'[targets."{name}"]', {})
        for field, value in target.items():
            if field in allowed:
                continue
            if after.get(field) != value:
                failures.append(f"{name}.{field}: {value!r} -> {after.get(field)!r}")

    for section in ("routes", "composites", "llm_clients"):
        if doc.get(section) != live_doc.get(section):
            failures.append(f"{section} changed")

    if failures:
        for failure in failures:
            print(f"FAIL {failure}", file=sys.stderr)
        raise SystemExit(1)
    return {"targets": len(targets)}


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
        f"targets {summary['targets']}"
    )


if __name__ == "__main__":
    main()
