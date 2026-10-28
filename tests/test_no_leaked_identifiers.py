# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guards against committing credentials or real infrastructure identifiers.

The repository is public. A test fixture that carries the real name of a private
network, or a real key, publishes it to everyone who can read the repository — and
a forward fix does not remove it from history. This test reads the files Git
actually tracks, so it catches an identifier the moment it is added rather than
when someone notices it later.

Test fixtures need host names that behave like the real ones. The rule here is not
"no host names"; it is "no *real* ones". A placeholder that ends in
``tailnet-example.ts.net`` exercises the same code path and identifies nothing.
"""

import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Hostname suffixes that identify a private network, minus the placeholder the
# tests are allowed to use.
TAILNET_PLACEHOLDER = "tailnet-example.ts.net"

FORBIDDEN = [
    # Provider API keys, in the shapes the providers actually mint.
    (re.compile(r"\bsk-or-v1-[A-Za-z0-9]{16,}"), "OpenRouter API key"),
    (re.compile(r"\bsk-(?!or-v1-)[A-Za-z0-9]{32,}"), "provider API key"),
    (re.compile(r"\b(ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}"), "GitHub token"),
    (re.compile(r"\bgithub_pat_[A-Za-z0-9_]{40,}"), "GitHub fine-grained token"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "AWS access key id"),
    (re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}"), "Slack token"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), "private key"),
    # A JSON Web Token: three base64url segments with the characteristic prefix.
    (re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"), "JWT"),
]

TAILNET = re.compile(r"\b[a-z0-9][a-z0-9-]*\.ts\.net\b", re.IGNORECASE)


def tracked_files():
    """Every file Git tracks, which is exactly what a clone would receive."""
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files", "-z"],
        capture_output=True,
        check=True,
    )
    for raw in result.stdout.split(b"\0"):
        if not raw:
            continue
        path = REPO_ROOT / raw.decode("utf-8", "surrogateescape")
        if path.is_file():
            yield path


def read_text(path):
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        # A binary file is not where a hostname or a key would be pasted.
        return ""


def findings():
    problems = []
    for path in tracked_files():
        text = read_text(path)
        if not text:
            continue
        relative = path.relative_to(REPO_ROOT)
        for pattern, description in FORBIDDEN:
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                problems.append(f"{relative}:{line}: {description}")
        for match in TAILNET.finditer(text):
            host = match.group(0)
            if not host.lower().endswith(TAILNET_PLACEHOLDER):
                line = text.count("\n", 0, match.start()) + 1
                # The name itself is not repeated in the failure message: a test
                # that prints the identifier it is protecting has published it
                # again, in the CI log.
                problems.append(f"{relative}:{line}: private network hostname")
    return problems


def test_no_credentials_or_private_hostnames_are_tracked() -> None:
    problems = findings()
    assert not problems, "committed material that must not be public:\n" + "\n".join(problems)


def test_the_scanner_detects_a_key_it_is_shown() -> None:
    # A scanner that silently matches nothing looks identical to a clean tree, so
    # its own detection is asserted rather than assumed.
    sample = 'key = "sk-or-v1-' + "a" * 32 + '"'
    assert any(pattern.search(sample) for pattern, _ in FORBIDDEN)


def test_the_scanner_detects_a_private_hostname_it_is_shown() -> None:
    # The sample hostname is assembled at runtime rather than written out. A
    # literal here would be a private-network-shaped hostname in a tracked file,
    # which is the very thing this scanner reports - it would fail on itself.
    fake_tailnet = "realtail" + "net1234"
    sample = 'url = "http://some-host.' + fake_tailnet + '.ts.net:8446/v1"'
    matches = [m.group(0) for m in TAILNET.finditer(sample)]
    assert matches, "the tailnet pattern must match a real-shaped hostname"
    assert not matches[0].lower().endswith(TAILNET_PLACEHOLDER)


def test_the_placeholder_hostname_is_not_itself_flagged() -> None:
    # The fix must not trip the guard, or the guard gets disabled.
    sample = f'url = "http://gpu-box.{TAILNET_PLACEHOLDER}:8446/v1"'
    matches = [m.group(0) for m in TAILNET.finditer(sample)]
    assert matches
    assert all(m.lower().endswith(TAILNET_PLACEHOLDER) for m in matches)
