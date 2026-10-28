#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a release's own dry-run under the service's environment, without leaking it.

WHY THIS REPLACES `set -a; . file`

    systemd and bash do not agree on what an EnvironmentFile is. systemd takes the
    right-hand side literally: `$` is not expanded, a backtick is not a command,
    and quotes are a delimiter rather than part of the value. Sourcing the same
    file with `.` executes the expansion and, for a value with the wrong quoting,
    returns non-zero -- which under `set -e` aborts an activation that was fine.

    It also does something worse than fail. Sourcing puts every provider key into
    the installer's own environment, where it survives into the restart, into the
    rollback, and into anything an operator runs in the same session.

    Both are avoided by parsing the files here and handing the merged environment
    to one child process. Nothing is exported into a shell, and nothing is copied.

WHAT IS PRINTED

    One line on success. On failure, the child's own diagnosis with every value
    from the environment files replaced by a placeholder, because a config error
    that quotes a key is common and a build log is not a place for it. The full,
    unscrubbed output goes to a 0600 file whose path is printed, so an operator can
    read the real thing without it entering a transcript.

EXIT CODES
    0  the candidate validated
    1  the candidate refused to load
    2  usage or environment-file problem
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys

# Long enough to describe a load failure, short enough not to become a dump.
DETAIL_LINES = 12

# Patterns that are secret-shaped even when the value did not come from an
# environment file we parsed. A config error may print a key it read from
# somewhere else, and the log should not carry it further.
SECRET_SHAPED = [
    re.compile(r"\bsk-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"\bsk_(?:live|test)_[A-Za-z0-9]{8,}"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{8,}"),
]


def parse_environment_file(path: str) -> dict[str, str]:
    """systemd's EnvironmentFile grammar, as far as it can be read.

    Documented behaviour implemented here: comment lines, `KEY=value`, single or
    double quotes around the whole value, literal values otherwise, and a trailing
    backslash joining the next line. Anything that is not an assignment is
    rejected rather than skipped, because a file systemd would refuse is a file
    whose environment this script cannot honestly claim to reproduce.
    """
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        raw = fh.read()

    # Join continuations before tokenising: a trailing backslash continues the
    # assignment onto the next line.
    lines: list[str] = []
    buffer = ""
    for line in raw.splitlines():
        if line.endswith("\\"):
            buffer += line[:-1]
            continue
        lines.append(buffer + line)
        buffer = ""
    if buffer:
        lines.append(buffer)

    env: dict[str, str] = {}
    for number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or stripped[0] in "#;":
            continue
        if "=" not in stripped:
            raise ValueError(f"{path}:{number}: not a KEY=value assignment")
        key, _, value = stripped.partition("=")
        key = key.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(f"{path}:{number}: {key!r} is not a valid variable name")
        value = value.lstrip()
        if value[:1] in ("'", '"'):
            quote = value[0]
            end = value.find(quote, 1)
            if end < 0:
                raise ValueError(f"{path}:{number}: unterminated {quote} quote")
            value = value[1:end]
        else:
            value = value.rstrip()
        env[key] = value
    return env


def scrub(text: str, secrets: list[str]) -> str:
    for value in secrets:
        if len(value) >= 6:
            text = text.replace(value, "<redacted>")
    for pattern in SECRET_SHAPED:
        text = pattern.sub("<redacted>", text)
    return text


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--env-file", action="append", default=[])
    parser.add_argument("--command", default=None,
                        help="testing seam: run this command instead of the binary")
    parser.add_argument("--log", default=None, help="where to write full detail")
    args = parser.parse_args(argv)

    if not os.path.isfile(args.binary):
        print(f"FAILED: no binary at {args.binary}", file=sys.stderr)
        return 2

    env: dict[str, str] = {}
    try:
        for path in args.env_file:
            env.update(parse_environment_file(path))
    except ValueError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 2

    # The child gets the service environment and a minimal PATH. It does not get
    # this process's environment: inheriting it would reintroduce exactly the leak
    # this helper exists to prevent.
    child_env = {"PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"}
    child_env.update(env)

    if args.command:
        command = ["/bin/bash", "-c", args.command]
    else:
        command = [args.binary, "--config", args.manifest, "--dry-run"]

    try:
        completed = subprocess.run(command, env=child_env, capture_output=True,
                                   text=True, timeout=180)
    except subprocess.TimeoutExpired:
        print("FAILED: validation did not finish within 180s", file=sys.stderr)
        return 1

    output = (completed.stdout or "") + (completed.stderr or "")
    secrets = [value for value in env.values() if value]

    if args.log:
        # Full detail, unscrubbed, root-only: the operator's copy. Written before
        # the scrubbed version is printed so a crash cannot lose it.
        try:
            descriptor = os.open(args.log, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(descriptor, "w") as fh:
                fh.write(f"$ {' '.join(shlex.quote(part) for part in command)}\n")
                fh.write(f"exit {completed.returncode}\n\n")
                fh.write(output)
        except OSError as exc:
            print(f"note: could not write {args.log}: {exc}", file=sys.stderr)

    if completed.returncode != 0:
        print(f"FAILED: the candidate refused to load (exit {completed.returncode})",
              file=sys.stderr)
        detail = scrub(output, secrets).strip().splitlines()
        for line in detail[:DETAIL_LINES]:
            print(f"  {line}", file=sys.stderr)
        if len(detail) > DETAIL_LINES:
            print(f"  ... {len(detail) - DETAIL_LINES} more lines in {args.log}",
                  file=sys.stderr)
        return 1

    summary = scrub(output, secrets).strip().splitlines()
    first = summary[0] if summary else "loaded"
    print(f"candidate validates: {first[:120]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
