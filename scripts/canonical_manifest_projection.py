#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Project a deployed routes manifest into a reviewable, secret-free canonical form.

The deployed manifest is the authority for what this service serves, but it is not itself
a reviewable artifact: it names the environment variable a credential is read from and the
endpoint each client talks to, and it carries operator notes. This script derives a
projection that keeps every routing, capability and budget fact and drops only what does
not belong in a repository:

  * ``api_key_env`` becomes ``credential_present = true``: which variable a deployment
    reads is deployment state, not contract.
  * ``base_url`` becomes ``endpoint_class``: a coarse class of the host, so a reviewer can
    see that a lane is a public Google endpoint or a private engine without the
    deployment's exact address.
  * comments are dropped, because a comment is a message from whoever edited that copy.

Everything else -- ids, formats, formulas, budgets, holds, capability ceilings -- is
preserved, and the projection records the SHA-256 of the manifest it came from, so the
projection can be tied to the bytes that were actually deployed.

The projection is deterministic: the same input produces the same bytes, which is what
makes it usable as a checked-in artifact.

Usage:
    canonical_manifest_projection.py <manifest.toml> --out <path>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import sys
import tomllib
from urllib.parse import urlsplit

# Ordered so a reviewer reads the projection in the same order as the manifest's own
# sections. Anything this list does not name is emitted after these, sorted by name, so a
# new section type can never be silently dropped.
SECTION_ORDER = [
    "schema_version",
    "server",
    "llm_clients",
    "targets",
    "routes",
    "decision_lanes",
    "capability_clients",
    "capabilities",
    "composites",
]

# Public hosts get their own class so a reviewer can see the provider at a glance.
# Anything unrecognised is reported as "configured" rather than guessing a provider.
ENDPOINT_CLASSES = {
    "generativelanguage.googleapis.com": "google-ai-studio",
    "openrouter.ai": "openrouter",
    "api.openai.com": "openai",
    "api.deepseek.com": "deepseek",
    "api.thaillm.or.th": "thaillm",
}

# Deployment state travels under prefixed spellings too (`decision_base_url`,
# `decision_api_key_env`), so the rule is on the suffix, not a fixed key set.
ENDPOINT_KEY = re.compile(r"(?:^|_)(base_url)$")
CREDENTIAL_KEY = re.compile(r"(?:^|_)(api_key_env)$")
BARE_KEY = re.compile(r"^[A-Za-z0-9_-]+$")


def classify(url: str) -> str:
    host = urlsplit(url).hostname or ""
    if host in ENDPOINT_CLASSES:
        return ENDPOINT_CLASSES[host]
    if host.endswith(".ts.net"):
        return "tailnet"
    if host in ("", "localhost", "127.0.0.1", "::1"):
        return "loopback"
    if host.startswith(("10.", "192.168.", "172.16.", "172.17.", "172.18.", "172.19.")):
        return "private-network"
    if host.startswith("127."):
        return "loopback"
    return "configured"


def project_table(table: dict) -> dict:
    """Replace the two deployment-state keys, recursively, leaving everything else alone."""
    out: dict = {}
    for key, value in table.items():
        if match := CREDENTIAL_KEY.search(key):
            out[key[: match.start(1)] + "credential_present"] = True
            continue
        if match := ENDPOINT_KEY.search(key):
            out[key[: match.start(1)] + "endpoint_class"] = classify(value)
            continue
        if isinstance(value, dict):
            out[key] = project_table(value)
            continue
        if isinstance(value, list) and any(isinstance(item, dict) for item in value):
            out[key] = [project_table(item) if isinstance(item, dict) else item for item in value]
            continue
        out[key] = value
    return out


def key(name: str) -> str:
    return name if BARE_KEY.match(name) else json.dumps(name)


def render(scalar: object) -> str:
    if isinstance(scalar, bool):
        return "true" if scalar else "false"
    if isinstance(scalar, (int, float)):
        return repr(scalar)
    if isinstance(scalar, str):
        return json.dumps(scalar)
    if isinstance(scalar, list):
        return "[" + ", ".join(render(item) for item in scalar) + "]"
    raise TypeError(f"cannot render {type(scalar).__name__}")


def is_table_array(value: object) -> bool:
    return isinstance(value, list) and any(isinstance(item, dict) for item in value)


def emit_table(lines: list[str], path: list[str], table: dict) -> None:
    for name, value in table.items():
        if isinstance(value, dict) or is_table_array(value):
            continue
        lines.append(f"{key(name)} = {render(value)}")
    for name, value in table.items():
        if isinstance(value, dict):
            lines.append("")
            lines.append(f"[{'.'.join(key(part) for part in [*path, name])}]")
            emit_table(lines, [*path, name], value)
        elif is_table_array(value):
            for item in value:
                lines.append("")
                lines.append(f"[[{'.'.join(key(part) for part in [*path, name])}]]")
                if isinstance(item, dict):
                    emit_table(lines, [*path, name], item)
                else:
                    lines.append(f"value = {render(item)}")


def emit(document: dict, source_sha: str) -> str:
    projected = project_table(document)
    lines: list[str] = [
        "# Generated by scripts/canonical_manifest_projection.py -- do not edit by hand.",
        "#",
        "# Every routing, capability and budget fact of the deployed manifest is preserved.",
        "# api_key_env -> credential_present, base_url -> endpoint_class, comments dropped.",
        "# Regenerate from the deployed manifest and compare these bytes; a difference is a",
        "# deployed change that has not been reviewed.",
        "",
        "[projection]",
        f"source_manifest_sha256 = {render(source_sha)}",
        "projection_version = 1",
    ]

    scalars = {name: value for name, value in projected.items() if not isinstance(value, dict)}
    for name in SECTION_ORDER:
        if name in scalars:
            lines.append(f"{key(name)} = {render(scalars[name])}")
    for name in sorted(set(scalars) - set(SECTION_ORDER)):
        lines.append(f"{key(name)} = {render(scalars[name])}")

    sections = {name: value for name, value in projected.items() if isinstance(value, dict)}
    ordered = [name for name in SECTION_ORDER if name in sections]
    ordered += sorted(set(sections) - set(SECTION_ORDER))
    for section in ordered:
        for name, table in sections[section].items():
            if not isinstance(table, dict):
                lines.append(f"{key(section)}.{key(name)} = {render(table)}")
                continue
            lines.append("")
            lines.append(f"[{key(section)}.{key(name)}]")
            emit_table(lines, [section, name], table)
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=pathlib.Path)
    parser.add_argument("--out", type=pathlib.Path)
    arguments = parser.parse_args()

    raw = arguments.manifest.read_bytes()
    document = tomllib.loads(raw.decode("utf-8"))
    text = emit(document, hashlib.sha256(raw).hexdigest())

    if arguments.out is None:
        sys.stdout.write(text)
        return 0
    arguments.out.write_text(text)
    print(f"{arguments.out} ({len(text)} bytes) from sha256 {hashlib.sha256(raw).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
