#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build-time capability contract for a Switchyard release.

WHY THIS EXISTS
    On 2026-10-27 a release was activated whose binary had been built from a
    branch that predated the economics instrumentation. The service started, was
    healthy, served all 46 routes correctly, and passed its dry-run against the
    live configuration -- and exported no economics metric at all. Every existing
    gate was satisfied and all of them were blind to it, because they ask
    "does it work" and never "does it still contain what we promised".

    The gate answers the second question. It is deliberately built so that
    completeness is established independently of the candidate's own claims.

WHAT IS DERIVED AND WHAT IS DECLARED
    Derived, mechanically, at build time -- never hand-written:
      * the instrumentation namespace found in the source tree
      * the metric-name strings present in the compiled binary
      * the exported family name for each instrument, by the Prometheus naming
        rule for its instrument kind

    Declared, in capabilities.toml, because no scanner can infer it:
      * whether each family is always_present or event_driven
      * which subsystem it belongs to
      * the deterministic trigger for an event-driven family

    An instrument that appears in either derived set and is not classified is a
    hard failure. That is what stops a subsystem being added without review.

COMMANDS
    generate --binary B --source S --out M
        Build the contract for one artifact.

    validate --candidate M --baseline B
        Cross-release check. A candidate differing from the deployed baseline is
        rejected unless every difference is a reviewed change.

    verify-live --manifest M --metrics-url U [--wait N]
        Post-deployment check of the always_present set only.

Exit codes
    0 ok
    2 usage / input error
    3 unclassified instrument (contract incomplete)
    4 classification names something that does not exist (stale contract)
    5 contract file malformed
    6 derived inventory disagrees with the binary
    7 capability removal without approval
    8 a required capability changed in a way that needs review
    9 live verification failed
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

EXIT_USAGE = 2
EXIT_UNCLASSIFIED = 3
EXIT_STALE = 4
EXIT_CONTRACT = 5
EXIT_DRIFT = 6
EXIT_REMOVAL = 7
EXIT_CAPABILITY_CHANGE = 8
EXIT_LIVE = 9

EXIT_CODES = {
    EXIT_USAGE: "usage or input error",
    EXIT_UNCLASSIFIED: "an instrument is not classified in capabilities.toml",
    EXIT_STALE: "capabilities.toml classifies an instrument that no longer exists",
    EXIT_CONTRACT: "contract file malformed",
    EXIT_DRIFT: "derived inventory disagrees with the compiled binary",
    EXIT_REMOVAL: "capability removed without an approved contract change",
    EXIT_CAPABILITY_CHANGE: "capability changed in a way that requires review",
    EXIT_LIVE: "live verification failed",
}

# Instrument kinds that can appear on a meter, and the Prometheus family name each
# produces. The mapping is pinned by test_capability_contract.py against a live
# exposition, so a change in the exporter's behaviour fails a test rather than
# silently making the contract describe the wrong family.
INSTRUMENT_KINDS = {
    "u64_counter": "counter",
    "f64_counter": "counter",
    "i64_counter": "counter",
    "u64_gauge": "gauge",
    "f64_gauge": "gauge",
    "i64_gauge": "gauge",
    "u64_histogram": "histogram",
    "f64_histogram": "histogram",
    # Up-down counters export as gauges in Prometheus: the value can decrease, so
    # a `_total` suffix would be a lie.
    "u64_up_down_counter": "updowncounter",
    "f64_up_down_counter": "updowncounter",
    "i64_up_down_counter": "updowncounter",
    # Observable (callback-sampled) instruments. These read a value the process
    # maintains itself rather than accumulating calls, and they export exactly
    # like their non-observable counterparts. `switchyard.total_requests` and
    # `switchyard.total_errors` are observable gauges; omitting these from the
    # table left both untyped, which is a gap in the table rather than ambiguity
    # in the source.
    "u64_observable_gauge": "gauge",
    "f64_observable_gauge": "gauge",
    "i64_observable_gauge": "gauge",
    "u64_observable_counter": "counter",
    "f64_observable_counter": "counter",
    "i64_observable_counter": "counter",
    "u64_observable_up_down_counter": "updowncounter",
    "f64_observable_up_down_counter": "updowncounter",
}

# Direct construction with a literal name: `meter.f64_gauge("switchyard.x")`.
DIRECT_RE = re.compile(
    r"\.(u64_counter|f64_counter|i64_counter|u64_gauge|f64_gauge|i64_gauge"
    r"|u64_histogram|f64_histogram|u64_up_down_counter|f64_up_down_counter"
    r"|i64_up_down_counter|u64_observable_gauge|f64_observable_gauge"
    r"|i64_observable_gauge|u64_observable_counter|f64_observable_counter"
    r"|i64_observable_counter|u64_observable_up_down_counter"
    r"|f64_observable_up_down_counter)"
    r"\s*\(\s*\"(switchyard\.[A-Za-z0-9_.]+)\""
)

# Any dotted instrument-namespace literal in the tree. This catches the names
# passed as an argument to a publishing helper (`publish_amounts(&meter,
# "switchyard.economics_provider_balance", ...)`), which the direct pattern
# cannot see. Four economics families are published that way, so a scanner
# limited to direct construction would silently under-report the surface -- the
# exact failure mode this tool exists to prevent.
NAMESPACE_RE = re.compile(r"\"(switchyard\.[a-z0-9_]+(?:\.[a-z0-9_]+)*)\"")

# The namespace is shared with distributed tracing: `span.record("switchyard.route",
# ...)` names a span field, not an instrument. Both are `switchyard.*` string
# literals, so the literal alone cannot distinguish them.
#
# The discriminating rule is the STATEMENT the literal lives in. A metric name is
# constructed on, published to, or filtered over a meter/instrument; a span field
# is set on a span. Splitting on `;` handles the multi-line helper calls that the
# direct pattern cannot see.
METRIC_CONTEXT = re.compile(r"\b(meter|Meter|instrument|Instrument|publish_|counter|Counter|gauge|Gauge|histogram|Histogram)\b")


class ContractError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code


def fail(code: int, message: str) -> None:
    raise ContractError(code, message)


# ---------------------------------------------------------------------------
# derivation
# ---------------------------------------------------------------------------


def families_for(name: str, kind: str) -> list[str]:
    """Exported Prometheus family names for one OTel instrument name.

    OTel converts a dotted name to underscores and appends `_total` to counters.
    Histograms expand into three series rather than one family, which is why the
    contract lists families and not instruments.
    """
    base = name.replace(".", "_")
    if kind == "counter":
        return [base + "_total"]
    if kind in ("gauge", "updowncounter"):
        return [base]
    if kind == "histogram":
        return [base + "_bucket", base + "_sum", base + "_count"]
    fail(EXIT_CONTRACT, f"unknown instrument kind {kind!r} for {name}")


# The crates that compose the switchyard-server artifact. The contract describes
# THAT binary, so instrumentation in a separately-shipped deliverable (the soak
# harness, the relay plugin, the Python bindings) is out of scope: including it
# would let a name from a crate that never reaches this binary inflate or
# misdescribe the surface.
SERVER_CRATES = ("crates/switchyard-server", "crates/libsy", "crates/libsy-llm-client")


def scan_source(root: str) -> tuple[dict[str, str], set[str]]:
    """Instrument name -> kind, from the source tree, plus non-metric references.

    Returns (instruments, non_metric). The second set holds `switchyard.*`
    literals that live outside instrumentation context -- span field names --
    and is reported so a reviewer can see the scanner did not silently drop
    something that mattered.
    """
    found: dict[str, str] = {}
    non_metric: set[str] = set()
    test_only: set[str] = set()
    rust_files = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames
                             if d not in ("target", ".git", "node_modules"))
        # Test trees assert on metric names as fixtures. They describe what the
        # suite expects, not what ships, so they must not define the universe --
        # a stale expectation in a test would otherwise look like instrumentation.
        rel_dir = os.path.relpath(dirpath, root)
        if not any(rel_dir == c or rel_dir.startswith(c + os.sep) for c in SERVER_CRATES):
            continue
        in_tests = "/tests/" in (dirpath + "/") or dirpath.endswith("/tests")
        for name in sorted(filenames):
            if not name.endswith(".rs"):
                continue
            is_test = in_tests or name.startswith("test_") or name.endswith("_test.rs")
            if is_test:
                text = open(os.path.join(dirpath, name), encoding="utf-8",
                            errors="replace").read()
                test_only.update(NAMESPACE_RE.findall(text))
                continue
            rust_files += 1
            path = os.path.join(dirpath, name)
            try:
                text = open(path, encoding="utf-8", errors="replace").read()
            except OSError as exc:
                fail(EXIT_USAGE, f"cannot read {path}: {exc}")
            for kind, instrument in DIRECT_RE.findall(text):
                family_kind = INSTRUMENT_KINDS[kind]
                previous = found.get(instrument)
                if previous and previous != family_kind:
                    fail(EXIT_CONTRACT,
                         f"{instrument} is declared as both {previous} and {family_kind}")
                found[instrument] = family_kind
            # Statement-level classification for the name-passed-to-a-helper form.
            for statement in text.split(";"):
                for candidate in NAMESPACE_RE.findall(statement):
                    if candidate in found:
                        continue
                    if METRIC_CONTEXT.search(statement):
                        found.setdefault(candidate, None)
                    else:
                        non_metric.add(candidate)
    if not rust_files:
        fail(EXIT_USAGE, f"no non-test Rust sources under {root}; wrong --source?")
    # A name asserted only by tests is not instrumentation; report it so the
    # distinction is visible rather than assumed.
    non_metric |= (test_only - set(found))
    return found, non_metric


def load_binary(path: str) -> bytes:
    """The compiled artifact as bytes, for presence checks.

    A Rust string literal sits in the binary contiguously with whatever preceded
    it, so scanning the whole blob for `switchyard.*` names yields mangled
    symbols, LLVM hashes and runs of adjacent literals rather than an inventory.
    The blob is therefore only ever used as a PRESENCE test per known name, which
    is what it is reliable for; the inventory comes from the source scan.
    """
    if not os.path.isfile(path):
        fail(EXIT_USAGE, f"no binary at {path}")
    return open(path, "rb").read()


def reconcile(source: dict[str, str]) -> tuple[dict[str, str], list[str]]:
    """Resolve the kinds the source scan could type, and name the ones it could not.

    The universe is the typed source scan. The binary is then used as a PRESENCE
    check per instrument, because only what is compiled in can ship. A derived
    instrument absent from the artifact means it was compiled out or dead, which
    is reported rather than smoothed over.

    A name whose construction site passes the instrument name as a value --
    `meter.u64_counter(name)` in a loop over a table of literals, or
    `.f64_gauge(name)` inside a publishing helper -- cannot be typed by a scanner
    that reads call sites. Those names are returned separately so the CALLER can
    resolve them from the declared `kind` in capabilities.toml. They are not
    silently guessed: a name with neither a source kind nor a declared kind is a
    build failure, and a declared kind that contradicts a typed one is too.

    """
    resolved: dict[str, str] = {}
    untyped: list[str] = []
    for name, kind in source.items():
        if kind is None:
            untyped.append(name)
            continue
        resolved[name] = kind
    return resolved, untyped


def load_contract(path: str) -> dict:
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover
        fail(EXIT_USAGE, "python 3.11+ is required (tomllib)")
    try:
        with open(path, "rb") as fh:
            data = tomllib.load(fh)
    except FileNotFoundError:
        fail(EXIT_USAGE, f"no contract at {path}")
    except Exception as exc:
        fail(EXIT_CONTRACT, f"cannot parse {path}: {exc}")
    if not isinstance(data.get("families"), dict):
        fail(EXIT_CONTRACT, f"{path} has no [families] table")
    for name, entry in data["families"].items():
        if entry.get("class") not in ("always_present", "event_driven"):
            fail(EXIT_CONTRACT,
                 f"{name}: class must be always_present or event_driven, "
                 f"got {entry.get('class')!r}")
        if not entry.get("subsystem"):
            fail(EXIT_CONTRACT, f"{name}: subsystem is required")
        if entry.get("class") == "event_driven" and not entry.get("trigger"):
            fail(EXIT_CONTRACT,
                 f"{name}: an event_driven family needs a trigger, otherwise the "
                 "only way to prove it works is to hope it appears")
        if "kind" in entry and entry["kind"] not in INSTRUMENT_KINDS:
            fail(EXIT_CONTRACT,
                 f"{name}: kind {entry['kind']!r} is not an OpenTelemetry instrument "
                 f"constructor; known kinds are {', '.join(sorted(INSTRUMENT_KINDS))}")
        # `sparse` marks a family that the deployed configuration cannot produce at
        # all -- no traffic shape reaches it. Its absence is permanent rather than
        # rare, so it needs a deterministic proof that it is still wired, and the
        # name of that proof travels with the release manifest.
        if entry.get("sparse"):
            if entry.get("class") != "event_driven":
                fail(EXIT_CONTRACT,
                     f"{name}: sparse is only meaningful for an event_driven family")
            if not entry.get("reachability_test"):
                fail(EXIT_CONTRACT,
                     f"{name}: a sparse family needs reachability_test naming the test "
                     "that drives its recorder, because no deployment traffic will "
                     "ever prove it")
    return data


def git_identity(source_root: str) -> dict:
    def git(*args: str) -> str:
        try:
            out = subprocess.run(["git", "-C", source_root, *args],
                                 capture_output=True, text=True, timeout=30)
            return out.stdout.strip() if out.returncode == 0 else ""
        except (OSError, subprocess.SubprocessError):
            return ""
    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
    }


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------


def cmd_generate(args: argparse.Namespace) -> int:
    contract = load_contract(args.contract)
    declared = contract["families"]

    source, non_metric = scan_source(args.source)
    blob = load_binary(args.binary)
    typed, untyped = reconcile(source)

    # Resolve the names the scanner could not type from the contract's declared
    # `kind`. This is the only place a human decision enters the inventory, and it
    # is checked in both directions: a declared kind with no construction site is
    # still listed in the source scan (so it cannot invent a capability), and a
    # declared kind that contradicts a typed construction site is a hard failure
    # rather than a quiet correction.
    resolved = dict(typed)
    kind_basis: dict[str, str] = {name: "source" for name in typed}
    for name in sorted(untyped):
        entry = declared.get(name) or {}
        declared_kind = entry.get("kind")
        if declared_kind is None:
            fail(EXIT_CONTRACT,
                 f"{name} is referenced by name but no instrument construction was "
                 "found for it, and capabilities.toml declares no kind. Add "
                 f"`kind = \"<constructor>\"` to its entry; the scanner will not "
                 "guess an instrument kind.")
        resolved[name] = INSTRUMENT_KINDS[declared_kind]
        kind_basis[name] = f"declared:{declared_kind}"
    for name, kind in sorted(typed.items()):
        declared_kind = (declared.get(name) or {}).get("kind")
        if declared_kind is not None and INSTRUMENT_KINDS[declared_kind] != kind:
            fail(EXIT_CONTRACT,
                 f"{name}: capabilities.toml declares kind {declared_kind!r} "
                 f"({INSTRUMENT_KINDS[declared_kind]}) but the source constructs it as "
                 f"{kind}. Fix the contract; a contradiction here would describe the "
                 "wrong Prometheus family.")

    def in_binary(instrument: str) -> bool:
        spellings = (instrument.encode(), instrument.replace(".", "_").encode())
        return any(sp in blob for sp in spellings)

    # Instrumentation that exists in the source but is compiled out -- a recorder no
    # call site reaches, which the optimiser drops along with its name literal. The
    # contract may state that explicitly, and then the claim is checked in the
    # opposite direction: a name declared not-compiled that IS in the binary is a
    # contradiction, because the manifest would be describing yesterday's build.
    not_compiled: list[dict] = []
    for name, entry in sorted(declared.items()):
        reason = entry.get("not_compiled")
        if not reason:
            continue
        if name not in resolved:
            fail(EXIT_STALE,
                 f"{name} is declared not_compiled, but no construction site exists in "
                 "the source at all; remove the entry instead of marking it dead")
        if in_binary(name):
            fail(EXIT_DRIFT,
                 f"{name} is declared not_compiled in capabilities.toml but the string "
                 "is present in the artifact. The contract is out of date with the "
                 "build: remove the not_compiled marker or rebuild.")
        not_compiled.append({"instrument": name, "reason": reason})
        del resolved[name]
        kind_basis.pop(name, None)

    absent_from_binary: list[str] = []
    for instrument in sorted(resolved):
        if not in_binary(instrument):
            absent_from_binary.append(instrument)
    if absent_from_binary:
        for name in absent_from_binary:
            print(f"NOT-IN-BINARY  {name}", file=sys.stderr)
        fail(EXIT_DRIFT,
             f"{len(absent_from_binary)} instrument(s) are declared in source but do "
             "not appear in the compiled artifact. Either the code is dead or this "
             "is the wrong binary; a manifest generated here would claim a "
             "capability the artifact does not contain.")

    families: dict[str, dict] = {}
    unclassified: list[str] = []
    for instrument, kind in sorted(resolved.items()):
        entry = declared.get(instrument)
        if entry is None:
            unclassified.append(instrument)
            continue
        if kind == "unknown":
            fail(EXIT_DRIFT,
                 f"{instrument} appears in the binary but no instrument construction "
                 "was found in the source; the scanner cannot state its kind")
        for family in families_for(instrument, kind):
            families[family] = {
                "instrument": instrument,
                "kind": kind,
                "kind_basis": kind_basis[instrument],
                "subsystem": entry["subsystem"],
                "class": entry["class"],
                **({"max_wait_seconds": entry["max_wait_seconds"]}
                   if "max_wait_seconds" in entry else {}),
                **({"trigger": entry["trigger"]} if "trigger" in entry else {}),
                **({"reachability_test": entry["reachability_test"]}
                   if "reachability_test" in entry else {}),
                **({"sparse": True} if entry.get("sparse") else {}),
                **({"attribution": True} if entry.get("attribution") else {}),
            }

    if unclassified:
        for name in unclassified:
            print(f"UNCLASSIFIED  {name}", file=sys.stderr)
        fail(EXIT_UNCLASSIFIED,
             f"{len(unclassified)} instrument(s) are compiled into this build but have "
             "no entry in capabilities.toml. Classify each as always_present or "
             "event_driven before this build can be released; a default would hide "
             "exactly the kind of silent capability change this gate exists to catch.")

    # A name marked not_compiled was removed from `resolved` above, so it must not
    # read as stale: the entry is deliberate and its absence is the claim.
    exempt = {entry["instrument"] for entry in not_compiled}
    stale = sorted(set(declared) - set(resolved) - exempt)
    if stale:
        for name in stale:
            print(f"STALE  {name}", file=sys.stderr)
        fail(EXIT_STALE,
             f"{len(stale)} classified instrument(s) do not exist in this build. A "
             "contract entry with no instrument behind it makes the gate lie: it "
             "would report a capability as present when nothing exports it.")

    subsystems: dict[str, list[str]] = {}
    for family, meta in families.items():
        subsystems.setdefault(meta["subsystem"], []).append(family)

    manifest = {
        "contract_version": contract.get("meta", {}).get("contract_version", 1),
        "build": {
            "binary_sha256": hashlib.sha256(blob).hexdigest(),
            "binary_bytes": len(blob),
            **git_identity(args.source),
        },
        "subsystems": {name: sorted(fams) for name, fams in sorted(subsystems.items())},
        "families": dict(sorted(families.items())),
        # Not part of the contract: names the binary contains that the source
        # scan could not type. Recorded so a reviewer can see the scanner's
        # blind spot rather than inferring it does not have one.
        "non_metric_namespace_literals": sorted(non_metric),
        "not_compiled": not_compiled,
        # Families no deployment traffic can produce. Preflight requires their
        # STRINGS in the artifact even though it must never require their series.
        "sparse_families": sorted(f for f, m in families.items() if m.get("sparse")),
        # Families that must be compiled for the release to be said to have request
        # accounting or token attribution at all. Their series appear only after
        # traffic, so their presence is a property of the ARTIFACT, checked against
        # the bytes, never of the exposition.
        "attribution_families": sorted(f for f, m in families.items() if m.get("attribution")),
        "counts": {
            "instruments": len(resolved),
            "families": len(families),
            "always_present": sum(1 for f in families.values() if f["class"] == "always_present"),
            "event_driven": sum(1 for f in families.values() if f["class"] == "event_driven"),
            "declared_kind": sum(1 for f in families.values()
                                 if f["kind_basis"].startswith("declared:")),
        },
    }
    if args.out == "-":
        json.dump(manifest, sys.stdout, indent=2, sort_keys=True)
        print()
    else:
        with open(args.out, "w") as fh:
            json.dump(manifest, fh, indent=2, sort_keys=True)
        print(f"wrote {args.out}: {manifest['counts']['instruments']} instruments, "
              f"{manifest['counts']['families']} families "
              f"({manifest['counts']['always_present']} always_present, "
              f"{manifest['counts']['event_driven']} event_driven), "
              f"{len(manifest['subsystems'])} subsystems")
    return 0


def load_manifest(path: str) -> dict:
    try:
        data = json.load(open(path))
    except FileNotFoundError:
        fail(EXIT_USAGE, f"no manifest at {path}")
    except (ValueError, OSError) as exc:
        fail(EXIT_USAGE, f"cannot read {path}: {exc}")
    if not isinstance(data.get("families"), dict):
        fail(EXIT_USAGE, f"{path} is not a capability manifest")
    return data


def load_approvals(path: str | None) -> dict[str, str]:
    if not path:
        return {}
    try:
        return json.load(open(path)).get("removed", {})
    except FileNotFoundError:
        fail(EXIT_USAGE, f"no approvals file at {path}")
    except (ValueError, OSError) as exc:
        fail(EXIT_USAGE, f"cannot read {path}: {exc}")


def compare(candidate: dict, baseline: dict, approved: dict[str, str]) -> int:
    """The cross-release comparison, shared by validate and preflight.

    Deliberately compares against the DEPLOYED manifest, not against the
    candidate's own. A candidate whose internally generated manifest is
    perfectly self-consistent can still have lost a subsystem -- that is exactly
    what happened on 2026-10-27.
    """
    cand_fams = set(candidate["families"])
    base_fams = set(baseline["families"])

    removed = sorted(base_fams - cand_fams)
    added = sorted(cand_fams - base_fams)
    unapproved = [f for f in removed if f not in approved]

    changed: list[str] = []
    for family in sorted(cand_fams & base_fams):
        c, b = candidate["families"][family], baseline["families"][family]
        if c.get("class") != b.get("class"):
            changed.append(f"{family}: {b.get('class')} -> {c.get('class')}")
        elif c.get("subsystem") != b.get("subsystem"):
            changed.append(f"{family}: subsystem {b.get('subsystem')} -> {c.get('subsystem')}")

    base_subs = {s: set(f) for s, f in baseline.get("subsystems", {}).items()}
    cand_subs = {s: set(f) for s, f in candidate.get("subsystems", {}).items()}
    lost_subsystems = sorted(
        s for s, fams in base_subs.items()
        if not cand_subs.get(s) and fams and s not in approved
    )

    print(f"baseline: {baseline['build'].get('commit', '?')[:12]} "
          f"{baseline['build'].get('binary_sha256', '?')[:16]} "
          f"({len(base_fams)} families, {len(base_subs)} subsystems)")
    print(f"candidate: {candidate['build'].get('commit', '?')[:12]} "
          f"{candidate['build'].get('binary_sha256', '?')[:16]} "
          f"({len(cand_fams)} families, {len(cand_subs)} subsystems)")

    rc = 0
    if added:
        print(f"\nadded ({len(added)}): a capability gain is reported, not blocked")
        for f in added:
            print(f"  + {f}")
    if removed:
        print(f"\nremoved ({len(removed)}):")
        for f in removed:
            mark = "APPROVED" if f in approved else "UNAPPROVED"
            print(f"  - {f}  [{mark}]")
        if unapproved:
            rc = EXIT_REMOVAL
    if changed:
        print(f"\nchanged ({len(changed)}):")
        for c in changed:
            print(f"  ~ {c}")
        rc = rc or EXIT_CAPABILITY_CHANGE
    if lost_subsystems:
        print(f"\nsubsystems that disappeared entirely ({len(lost_subsystems)}):")
        for s in lost_subsystems:
            print(f"  ! {s}  [UNAPPROVED]")
        rc = EXIT_REMOVAL

    if rc == 0:
        print("\nPASS: every capability the deployed release provides is either "
              "preserved or explicitly approved as removed")
    else:
        print(f"\nFAIL ({EXIT_CODES[rc]}): this candidate changes the deployed "
              "capability contract")
        if unapproved:
            print("  A capability may only be removed through a reviewed contract "
                  "change. If the removal is intended, record it in an approvals "
                  "file with a reason -- do not edit the baseline.")
    return rc


def cmd_validate(args: argparse.Namespace) -> int:
    """Cross-release check against the deployed baseline."""
    return compare(load_manifest(args.candidate), load_manifest(args.baseline),
                   load_approvals(args.approvals))


def instrument_literal(family: str, meta: dict) -> str:
    """The source literal a family is built from.

    A family name is derived from the instrument name by the Prometheus naming
    rule, so the check has to look for the instrument, not the family.
    """
    return meta["instrument"]


def cmd_preflight(args: argparse.Namespace) -> int:
    """Deployment gate: identity, capability preservation, and compiled presence.

    Runs where there is no source tree -- on the deployment host -- so every
    judgement it makes is from bytes plus two manifests: the candidate's (built
    where the source lives) and the DEPLOYED baseline's.

    Three questions, in the order that makes a failure unambiguous:
      1. Is this manifest about this binary? A manifest that describes another
         artifact is worthless evidence, so the hash binding is checked first.
      2. Does the binary still contain what the deployed release contained? The
         literal is looked for in the bytes, so a manifest that merely CLAIMS a
         capability while the artifact lacks it also fails here.
      3. Does the candidate's own manifest preserve the baseline's families and
         subsystems? This is the cross-release comparison, and it is what makes a
         self-consistent but incomplete manifest fail.
    """
    candidate = load_manifest(args.manifest)
    baseline = load_manifest(args.baseline)
    blob = load_binary(args.binary)
    digest = hashlib.sha256(blob).hexdigest()

    print(f"candidate binary: {digest[:16]} ({len(blob)} bytes)")
    claimed = candidate.get("build", {}).get("binary_sha256")
    if claimed != digest:
        print(f"  manifest claims {str(claimed)[:16]}", file=sys.stderr)
        fail(EXIT_DRIFT,
             "the capability manifest does not describe this binary. Rebuild the "
             "manifest beside the artifact; a manifest generated from another "
             "build proves nothing about this one.")
    print("  identity: manifest and binary agree")

    # Families the deployed release treats as needing proof of presence: the
    # always_present set, plus every family flagged sparse, whose whole claim is
    # that it is compiled even though deployment traffic never reaches it.
    required_for_artifact = (set(baseline.get("sparse_families", []))
                             | set(baseline.get("attribution_families", [])))
    must_contain = sorted(
        family for family, meta in baseline["families"].items()
        if meta["class"] == "always_present" or family in required_for_artifact
    )
    missing_literal = []
    for family in must_contain:
        meta = baseline["families"][family]
        literal = instrument_literal(family, meta)
        if literal.encode() not in blob and literal.replace(".", "_").encode() not in blob:
            missing_literal.append((family, literal))
    if missing_literal:
        for family, literal in missing_literal:
            print(f"  ABSENT  {family}  (instrument {literal} not in the artifact)",
                  file=sys.stderr)
        fail(EXIT_DRIFT,
             f"{len(missing_literal)} capabilit{'y' if len(missing_literal) == 1 else 'ies'} "
             "the deployed release provides are not compiled into this artifact. "
             "This is the 2026-10-27 failure: a healthy build that had silently "
             "lost a subsystem.")
    print(f"  compiled: all {len(must_contain)} required families are present in the "
          f"artifact ({len(baseline.get('attribution_families', []))} attribution, "
          f"{len(baseline.get('sparse_families', []))} sparse)")

    rc = compare(candidate, baseline, load_approvals(args.approvals))
    if rc != 0:
        return rc
    print("PREFLIGHT PASS")
    return 0


def fetch_exposition(url: str, timeout: int = 20) -> str:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as fh:
            return fh.read().decode("utf-8", "replace")
    except (urllib.error.URLError, OSError) as exc:
        fail(EXIT_LIVE, f"cannot scrape {url}: {exc}")


def families_in(text: str) -> set[str]:
    return set(re.findall(r"^([a-z_][a-z0-9_]*)\{", text, re.MULTILINE)) | \
           set(re.findall(r"^([a-z_][a-z0-9_]*) ", text, re.MULTILINE))


def cmd_verify_live(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    required = {
        f: m for f, m in manifest["families"].items()
        if m["class"] == "always_present"
    }
    # Wait only as long as the slowest required family says it needs. Anything
    # event_driven is excluded by construction: its absence is not a failure.
    budget = args.wait if args.wait is not None else max(
        [m.get("max_wait_seconds", 60) for m in required.values()] or [60]
    )

    started = time.time()
    seen: set[str] = set()
    while True:
        text = fetch_exposition(args.metrics_url)
        seen = families_in(text)
        missing = {f for f in required if f not in seen}
        if not missing:
            break
        if time.time() - started >= budget:
            break
        time.sleep(args.interval)

    elapsed = int(time.time() - started)
    missing = {f for f in required if f not in seen}

    print(f"scraped {args.metrics_url}: {len(seen)} families visible after {elapsed}s")
    print(f"required always_present: {len(required)}")

    ignored = sorted(f for f in manifest["families"] if f not in required and f in seen)
    if ignored:
        print(f"event_driven families that happen to be present: {len(ignored)} "
              "(informational, never required)")

    if missing:
        print()
        for f in sorted(missing):
            sub = required[f]["subsystem"]
            print(f"  MISSING  {f}  [{sub}]")
        fail(EXIT_LIVE,
             f"{len(missing)} always_present famil{'y' if len(missing) == 1 else 'ies'} "
             f"did not appear within {budget}s. This release does not provide the "
             "capability its own manifest claims.")

    print("PASS: every always_present family is exposed")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="build the capability manifest for an artifact")
    g.add_argument("--binary", required=True)
    g.add_argument("--source", required=True)
    g.add_argument("--contract", default=os.path.join(os.path.dirname(__file__),
                                                      "capabilities.toml"))
    g.add_argument("--out", required=True)
    g.set_defaults(func=cmd_generate)

    v = sub.add_parser("validate", help="compare a candidate against the deployed baseline")
    v.add_argument("--candidate", required=True)
    v.add_argument("--baseline", required=True)
    v.add_argument("--approvals", default=None,
                   help="JSON with {\"removed\": {\"<family>\": \"<reason>\"}}")
    v.set_defaults(func=cmd_validate)

    p = sub.add_parser("preflight",
                       help="deployment gate: identity, preservation, compiled presence")
    p.add_argument("--binary", required=True)
    p.add_argument("--manifest", required=True,
                   help="the capability manifest built beside the candidate binary")
    p.add_argument("--baseline", required=True,
                   help="the deployed release's capability manifest")
    p.add_argument("--approvals", default=None,
                   help="JSON with {\"removed\": {\"<family>\": \"<reason>\"}}")
    p.set_defaults(func=cmd_preflight)

    L = sub.add_parser("verify-live", help="post-deployment always_present check")
    L.add_argument("--manifest", required=True)
    L.add_argument("--metrics-url", required=True)
    L.add_argument("--wait", type=int, default=None)
    L.add_argument("--interval", type=float, default=5.0)
    L.set_defaults(func=cmd_verify_live)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ContractError as exc:
        print(f"FAILED [{EXIT_CODES.get(exc.code, exc.code)}]: {exc}", file=sys.stderr)
        return exc.code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
