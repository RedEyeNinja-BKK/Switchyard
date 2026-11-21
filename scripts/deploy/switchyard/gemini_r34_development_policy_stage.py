#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Record the owner's standing approval of their own development material.

One line in `[privacy]`. With it, a request reaching a privacy-gated provider without a
clearance header is admitted under the `owner-approved-development` class instead of being
refused, and the payload is scanned under the development policy -- ordinary repository
paths, documentation addresses and build identifiers pass, while private keys, API keys,
bearer tokens, credential-store paths, non-reserved addresses and card-shaped digit runs
still refuse.

Without it nothing changes: a clearance must still be presented and signed.

`development_policy` is a field the R3.3 binary does not know, so this manifest is paired
with the binary that implements it.
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import sys
import tomllib

POLICY = "owner-approved-development"

# The existing comment above `clearance_key_env` explains the key; the new line needs its own
# reason beside it, because "why is this deployment not refusing everything" is the question
# the next reader will have.
COMMENT = """# The owner's standing decision about their own non-secret material. Ordinary engineering
# work -- source, paths, issue discussion, harmless fixtures -- is approved for the free
# provider whose terms the owner accepted; a credential still refuses before egress.
development_policy = "owner-approved-development"
"""


def splice(text: str) -> tuple[str, list[str]]:
    lines = text.splitlines(keepends=True)
    # Locate the [privacy] table and its extent.
    start = None
    for index, line in enumerate(lines):
        if line.strip() == "[privacy]":
            start = index
            break
    if start is None:
        raise SystemExit("the deployed manifest has no [privacy] table")
    end = len(lines)
    for index in range(start + 1, len(lines)):
        stripped = lines[index].strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            end = index
            break

    body = "".join(lines[start:end])
    if "development_policy" in body:
        raise SystemExit("[privacy] already declares development_policy; refusing to rewrite")

    anchor = "clearance_key_env = "
    matches = [line for line in body.splitlines() if line.startswith(anchor)]
    if len(matches) != 1:
        raise SystemExit(f"[privacy] has {len(matches)} clearance_key_env lines, expected 1")
    line = matches[0]
    if not body.count(line + "\n"):
        raise SystemExit("[privacy] clearance_key_env line is not newline-terminated")

    body = body.replace(line + "\n", line + "\n" + COMMENT, 1)
    lines[start:end] = [body]
    return "".join(lines), ["[privacy]: + development_policy = owner-approved-development"]


def verify(candidate: str, live: str) -> dict:
    doc = tomllib.loads(candidate)
    live_doc = tomllib.loads(live)
    failures: list[str] = []

    privacy = doc.get("privacy", {})
    if privacy.get("development_policy") != POLICY:
        failures.append(f"development_policy is {privacy.get('development_policy')!r}")
    if privacy.get("clearance_key_env") != live_doc.get("privacy", {}).get("clearance_key_env"):
        failures.append("clearance_key_env moved")

    for section in ("routes", "composites", "llm_clients", "targets", "quota"):
        if doc.get(section) != live_doc.get(section):
            failures.append(f"{section} changed")
    if doc.get("schema_version") != live_doc.get("schema_version"):
        failures.append("schema_version changed")

    if failures:
        for failure in failures:
            print(f"FAIL {failure}", file=sys.stderr)
        raise SystemExit(1)
    return {"targets": len(doc.get("targets", {})), "routes": len(doc.get("routes", {}))}


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
