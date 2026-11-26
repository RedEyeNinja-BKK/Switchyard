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

P9 RELEASE CONTRACT
    All normal candidate validations refuse unless a pinned, unprivileged-write
    checker reports PASS for the exact candidate manifest. Unavailable, untrusted,
    failing, malformed and identity-mismatched verdicts all refuse before the
    binary is executed and before any pointer or service changes. The explicit
    --command substitution is ONLY a test seam; it does not establish admission.
    A pass prints a receipt naming the manifest digest, the counts, and the
    checker revision, because a gate that passes silently leaves no evidence of
    what it passed against.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys

try:  # 3.11+. Without it the manifest cannot be read, so P9 refuses rather than passing.
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised by the refusal test
    tomllib = None

# Long enough to describe a load failure, short enough not to become a dump.
DETAIL_LINES = 12

# Digests of checker revisions the release path accepts. Identity is pinned here, in
# source control, so replacing the checker is a reviewed change to this list rather than
# an implicit trust of whatever file sits at the path.
P9_CHECKER_SHA256 = frozenset({
    "72838c340dc55f8ca6958ad0c15a1e7afaac623649d78335b3b7b61865835877",
})

# Where the versioned checker may live, written out rather than derived from one
# another. The installed tools sit in /opt/switchyard/bin while the checker is deployed
# under /opt/deploy, so a single parent-arithmetic walk silently encodes both
# directories' current depth - and when that breaks, the refusal must name the paths
# rather than leave an operator to reverse-engineer them.
def _p9_checker_paths() -> list[str]:
    here = os.path.dirname(os.path.realpath(__file__))
    relative = os.path.join("deploy", "stephen-st-routing-v1", "direct_capability_gate.py")
    installed = os.path.join("/opt", relative)
    checkout = os.path.realpath(os.path.join(here, "..", "..", relative))
    bundle = os.path.join(here, "direct_capability_gate.py")
    # The layout this validator runs from decides which form is authoritative, so a
    # developer checkout never grades against an unrelated /opt tree and an installed
    # validator never grades against a stale working copy.
    if os.path.realpath(__file__).startswith("/opt/switchyard/"):
        return [installed, checkout, bundle]
    return [checkout, installed, bundle]


P9_CHECKER_PATHS = _p9_checker_paths()

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
    # A file that is not there is not an empty file. Returning {} would let a
    # candidate be validated under an environment with a credential silently
    # missing, and the caller would report "the candidate loads" about an
    # environment the service never has. The caller decides which files are
    # optional; this parser only reads what it was given.
    if not os.path.isfile(path):
        raise ValueError(f"{path}: no such environment file")
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            raw = fh.read()
    except OSError as exc:
        raise ValueError(f"{path}: cannot be read ({exc.strerror})") from exc

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


def validate_p9(manifest: str) -> tuple[bool, str]:
    """Mandatory OFFLINE P9 admission preflight for a real release candidate.

    Keep the implementation in the versioned Switchyard checkout; never treat a
    missing checker as success. This checks the candidate's bytes, not the
    currently serving /v1/models (which is verified AFTER deployment).
    """
    script, reason = resolve_p9_checker()
    if script is None:
        return False, reason
    try:
        result = subprocess.run(
            [sys.executable, script, "--json", "--manifest", manifest],
            env={"PATH": "/usr/local/bin:/usr/bin:/bin"},
            capture_output=True, text=True, timeout=90,
        )
    except subprocess.TimeoutExpired:
        return False, "P9_GATE_TIMEOUT: capability preflight exceeded 90 seconds"
    except OSError as exc:
        return False, "P9_GATE_EXECUTION_FAILED: " + str(exc)
    if result.returncode:
        return False, (
            f"P9_GATE_REFUSED exit={result.returncode}\n"
            + (result.stdout or "") + (result.stderr or "")
        )
    return judge_p9_report(result.stdout, manifest, script, reason)


def resolve_p9_checker() -> tuple[str | None, str]:
    """Return the checker path, or a refusal naming every path tried and why.

    A path is authoritative only when its bytes are pinned here and nobody but
    root can write it. That is the difference between "the file we meant" and
    "whatever currently sits where we looked": a checker that exits 0 without
    reading the manifest is not detectable from the exit status, so the identity
    of the tool has to be established before its verdict is trusted.
    """
    attempts: list[str] = []
    for candidate in P9_CHECKER_PATHS:
        path = os.path.realpath(candidate)
        if not os.path.exists(path):
            attempts.append(f"{path}: absent")
            continue
        if not os.path.isfile(path):
            attempts.append(f"{path}: not a regular file")
            continue
        try:
            with open(path, "rb") as handle:
                digest = hashlib.sha256(handle.read()).hexdigest()
        except OSError as exc:
            attempts.append(f"{path}: unreadable ({exc.strerror or exc})")
            continue
        if digest not in P9_CHECKER_SHA256:
            attempts.append(
                f"{path}: sha256 {digest} is not a pinned checker revision"
            )
            continue
        try:
            info = os.stat(path)
        except OSError as exc:
            attempts.append(f"{path}: unstattable ({exc.strerror or exc})")
            continue
        if info.st_mode & 0o022:
            attempts.append(f"{path}: writable by group or other (mode {info.st_mode & 0o777:o})")
            continue
        attempts.append(f"{path}: OK sha256={digest}")
        return path, digest
    return None, "P9_GATE_UNAVAILABLE: no pinned P9 checker\n" + "\n".join(attempts)


def manifest_direct_routes(manifest: str) -> dict[str, tuple[str, str, str]]:
    """The 1:1 direct aliases this manifest declares, read independently of the gate.

    A direct alias is a passthrough route with no `candidates` whose name carries
    the `direct-` prefix; that is the shape the checker grades. Deriving the set
    here - from the exact bytes about to be installed - is what binds the gate's
    verdict to this candidate: a report describing a different manifest cannot
    match, and a manifest whose direct routes were silently emptied cannot pass
    with an empty report.
    """
    if tomllib is None:
        raise ValueError("tomllib is unavailable, so the manifest cannot be read")
    with open(manifest, "rb") as handle:
        doc = tomllib.load(handle)
    routes = doc.get("routes")
    targets = doc.get("targets")
    if not isinstance(routes, dict) or not isinstance(targets, dict):
        raise ValueError("manifest has no [routes] or no [targets] table")
    derived: dict[str, tuple[str, str, str]] = {}
    for name, route in routes.items():
        if not isinstance(route, dict):
            raise ValueError(f"route {name} is not a table")
        if route.get("type") != "passthrough" or route.get("candidates"):
            continue
        if not name.startswith("direct-"):
            continue
        target_name = route.get("target")
        target = targets.get(target_name) if isinstance(target_name, str) else None
        if not isinstance(target, dict):
            raise ValueError(f"direct route {name} names unknown target {target_name!r}")
        model = target.get("id")
        client = target.get("llm_client")
        if not isinstance(model, str) or not isinstance(client, str):
            raise ValueError(f"target {target_name} lacks a string id/llm_client")
        derived[name] = (target_name, model, client)
    return derived


def judge_p9_report(stdout: str, manifest: str, checker: str,
                    checker_sha: str) -> tuple[bool, str]:
    """Accept the gate's verdict only when it describes this exact candidate.

    Four questions, all answered from one JSON report: did the gate pass, did it
    examine every direct alias this manifest declares, does each reported row
    still name the model and client this manifest's target declares, and can the
    report be reconciled with the manifest's own digest. A malformed report is a
    refusal rather than a parse error escaping to the caller - the caller is an
    activation path, and an exception there is not a decision.
    """
    try:
        with open(manifest, "rb") as handle:
            manifest_sha = hashlib.sha256(handle.read()).hexdigest()
    except OSError as exc:
        return False, f"P9_MANIFEST_UNREADABLE: {exc.strerror or exc}"
    try:
        report = json.loads(stdout)
    except ValueError:
        return False, "P9_GATE_MALFORMED: report is not JSON\n" + (stdout or "")[:2000]
    if not isinstance(report, dict):
        return False, "P9_GATE_MALFORMED: report is not an object"
    if report.get("status") != "PASS":
        return False, f"P9_GATE_MALFORMED: status {report.get('status')!r} is not PASS"
    failures = report.get("failures")
    if not isinstance(failures, list):
        return False, "P9_GATE_MALFORMED: failures is not a list"
    if failures:
        shown = "\n".join(str(item) for item in failures[:DETAIL_LINES])
        return False, f"P9_GATE_REFUSED {len(failures)} route failure(s)\n{shown}"
    rows = report.get("routes")
    if not isinstance(rows, list) or not rows:
        return False, "P9_GATE_MALFORMED: routes is not a non-empty list"
    if report.get("direct_count") != len(rows):
        return False, (
            "P9_GATE_MALFORMED: direct_count "
            f"{report.get('direct_count')!r} disagrees with {len(rows)} reported rows"
        )

    try:
        derived = manifest_direct_routes(manifest)
    except (ValueError, OSError) as exc:
        return False, f"P9_MANIFEST_IDENTITY_MISMATCH: {exc}"

    reported: dict[str, dict] = {}
    for row in rows:
        if not isinstance(row, dict):
            return False, "P9_GATE_MALFORMED: a route row is not an object"
        name = row.get("route")
        if not isinstance(name, str) or not name:
            return False, "P9_GATE_MALFORMED: a route row names no route"
        reported[name] = row

    missing = sorted(set(derived) - set(reported))
    extra = sorted(set(reported) - set(derived))
    if missing or extra:
        detail = [f"declared but not graded: {', '.join(missing[:8])}" if missing else "",
                  f"graded but not declared: {', '.join(extra[:8])}" if extra else ""]
        return False, (
            "P9_MANIFEST_IDENTITY_MISMATCH: the report does not describe "
            f"{manifest_sha}\n" + "\n".join(line for line in detail if line)
        )

    drift: list[str] = []
    for name, (target_name, model, client) in sorted(derived.items()):
        row = reported[name]
        if row.get("target") != target_name:
            drift.append(f"{name}: report target {row.get('target')!r} != {target_name!r}")
        elif row.get("model") != model:
            drift.append(f"{name}: report model {row.get('model')!r} != {model!r}")
        elif row.get("client") != client:
            drift.append(f"{name}: report client {row.get('client')!r} != {client!r}")
    if drift:
        return False, (
            "P9_MANIFEST_IDENTITY_MISMATCH: reported rows disagree with "
            f"{manifest_sha}\n" + "\n".join(drift[:DETAIL_LINES])
        )

    return True, (
        f"P9 PASS direct={len(reported)} qualified={len(reported)} "
        f"failures=0 manifest_sha256={manifest_sha} "
        f"checker_sha256={checker_sha} checker={checker}"
    )


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

    # A real deployment must fail closed on the candidate's P9 direct-route
    # contract BEFORE its binary dry-run. --command is an existing explicit test
    # seam, not a deployable proof; the actual activate-release.sh production
    # path supplies no --command.
    p9_receipt = ""
    if not args.command:
        p9_ok, p9_detail = validate_p9(args.manifest)
        if p9_ok:
            # The receipt is printed on success: a gate that passed silently leaves no
            # evidence of which manifest and which checker revision it passed against.
            p9_receipt = p9_detail
        if not p9_ok:
            print("FAILED: P9 direct-route capability release gate", file=sys.stderr)
            detail = scrub(p9_detail, [value for value in env.values() if value])
            for line in detail.strip().splitlines()[:DETAIL_LINES]:
                print(f"  {line}", file=sys.stderr)
            return 1

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
                if p9_receipt:
                    fh.write(p9_receipt + "\n\n")
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
    if p9_receipt:
        print(p9_receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
