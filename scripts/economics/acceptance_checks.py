#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Post-deployment acceptance checks for the economics surface.

These assert the things that must be true before a deployment is called
successful, and that cannot be checked by looking at one panel: that every
configured financial account is actually being read, that a provider's balance
equals the parts it reports, and that the rolling windows exist with a stated
coverage.

Read-only. It queries Prometheus and prints a verdict; it changes nothing.

    python3 acceptance_checks.py \
        --launcher /path/to/launch-grafana-readonly.sh \
        --datasource-uid <uid> \
        --job switchyard-v7

Exit code is 0 when every check passes, 1 when any fails. A check that cannot
be evaluated — because the series is absent — is a failure, not a skip: after a
deployment the absence of a series that should exist is exactly the thing worth
knowing.
"""

import argparse
import json
import subprocess
import sys

# Accounts this deployment must be reading. Each is a separate financial account.
REQUIRED_ACCOUNTS = ("openai", "deepseek", "openrouter", "openrouter_stephen")

# The rolling windows the deployment publishes.
REQUIRED_WINDOWS = ("1h", "24h")


SCRAPE = None


def parse_scrape(text):
    """Parses a /metrics body into {metric: [labels]} so the checks can run offline."""
    import re

    out = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = re.match(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+(\S+)", line)
        if not match:
            continue
        name, raw, value = match.group(1), match.group(2) or "", match.group(3)
        labels = {}
        for part in raw.strip("{}").split(","):
            if "=" not in part:
                continue
            key, _, val = part.partition("=")
            labels[key.strip()] = val.strip().strip('"')
        try:
            out.setdefault(name, []).append((labels, float(value)))
        except ValueError:
            continue
    return out


def query(launcher, datasource_uid, expr):
    if SCRAPE is not None:
        # Offline mode: answer the selector from a scrape taken elsewhere. Only the
        # plain `metric{job="x",label="y"}` shape these checks use is supported,
        # which is enough to exercise them without a live deployment.
        import re

        match = re.match(r"^([a-z_][a-z0-9_]*)\s*\{(.*)\}$", expr.strip())
        if not match:
            return []
        name, body = match.group(1), match.group(2)
        wanted = []
        for part in body.split(","):
            if "=" not in part:
                continue
            key, _, val = part.partition("=")
            wanted.append((key.strip(), val.strip().strip('"')))
        wanted = [(k, v) for k, v in wanted if k not in ("job", "instance")]
        out = []
        for labels, value in SCRAPE.get(name, []):
            if all(labels.get(k) == v for k, v in wanted):
                out.append((labels, value))
        return out
    """Runs one instant query, returning [(labels, value)]."""
    payload = json.dumps({"datasource_uid": datasource_uid, "expr": expr, "kind": "instant"})
    proc = subprocess.run(
        ["sudo", "-n", launcher, "query"],
        input=payload.encode(),
        capture_output=True,
        timeout=120,
    )
    if proc.returncode != 0:
        raise SystemExit(f"query failed: {(proc.stderr or proc.stdout).decode()[:300]}")
    body = json.loads(proc.stdout.decode())
    if body.get("status") != 200:
        raise SystemExit(f"query returned HTTP {body.get('status')}: {expr}")
    out = []
    for frame in (body.get("data") or {}).values():
        for series in frame.get("series", []):
            # The read tool returns a placeholder series with no points when a
            # metric family does not exist. Counting series would therefore report
            # an absent family as present, which is the opposite of the truth.
            if series.get("last") is None or (series.get("points") or 0) == 0:
                continue
            out.append((series.get("labels") or {}, series.get("last")))
    return out


class Report:
    def __init__(self):
        self.results = []

    def check(self, name, ok, detail=""):
        self.results.append((name, bool(ok), detail))
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {name}" + (f" — {detail}" if detail else ""))
        return ok

    def exit_code(self):
        return 0 if all(ok for _, ok, _ in self.results) else 1

    def summary(self):
        passed = sum(1 for _, ok, _ in self.results if ok)
        print(f"\n{passed}/{len(self.results)} checks passed")
        return self.exit_code()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launcher", required=True, help="The read-only Grafana launcher.")
    parser.add_argument("--datasource-uid", required=True)
    parser.add_argument("--job", required=True, help="The Prometheus job label.")
    parser.add_argument(
        "--scrape-file",
        default=None,
        help=(
            "Answer the checks from a saved /metrics scrape instead of Prometheus. "
            "Used to prove the checks themselves pass against a known-good scrape."
        ),
    )
    parser.add_argument(
        "--balance-tolerance",
        type=float,
        default=0.02,
        help="Absolute tolerance when checking that a balance equals its components.",
    )
    args = parser.parse_args(argv)
    global SCRAPE
    if args.scrape_file:
        SCRAPE = parse_scrape(open(args.scrape_file, encoding="utf-8").read())
    job = args.job
    report = Report()

    def q(expr):
        return query(args.launcher, args.datasource_uid, expr)

    print("Account coverage")
    freshness = q(f'switchyard_economics_account_freshness_seconds{{job="{job}"}}')
    seen = {labels.get("provider") for labels, _ in freshness}
    for account in REQUIRED_ACCOUNTS:
        report.check(f"account {account} is read", account in seen, f"freshness present for {account}")
    unexpected = seen - set(REQUIRED_ACCOUNTS)
    if unexpected:
        print(f"  [note] additional accounts read: {sorted(unexpected)}")

    print("\nAccount reads are healthy")
    # Zero failures is reported as an absent series, not as a zero. That is only a
    # clean bill of health if the collector is running at all, so the parent family
    # is checked first: a deployment that reads nothing has no failed reads either.
    any_reads = q(f'switchyard_economics_account_fetch_total{{job="{job}"}}')
    failures = q(
        f'sum(switchyard_economics_account_fetch_total{{job="{job}",outcome="failed"}})'
    )
    failed = failures[0][1] if failures else 0.0
    report.check(
        "the account collector is running",
        bool(any_reads),
        f"{len(any_reads)} fetch outcomes observed",
    )
    report.check("no account read has failed", failed == 0, f"failed reads = {failed:g}")

    print("\nBalance integrity")
    # DeepSeek reports a total and the parts it is made of. The parts must add up
    # to the total; if they do not, one of the two readings is being misread.
    totals = {
        labels.get("currency"): value
        for labels, value in q(
            f'switchyard_economics_provider_balance{{job="{job}",provider="deepseek"}}'
        )
    }
    parts = {}
    for labels, value in q(
        f'switchyard_economics_provider_balance_component{{job="{job}",provider="deepseek"}}'
    ):
        parts.setdefault(labels.get("currency"), {})[labels.get("component")] = value
    if not totals:
        report.check("DeepSeek reports a balance", False, "no balance series")
    for currency, total in totals.items():
        components = parts.get(currency, {})
        if not components:
            report.check(
                f"DeepSeek {currency} balance has components", False, "no component series"
            )
            continue
        summed = sum(components.values())
        report.check(
            f"DeepSeek {currency} total equals its components",
            abs(summed - total) <= args.balance_tolerance,
            f"total={total} components={components} sum={summed:.6f}",
        )

    print("\nRolling windows")
    coverage = {
        labels.get("window"): value
        for labels, value in q(f'switchyard_economics_rolling_coverage_seconds{{job="{job}"}}')
    }
    for window in REQUIRED_WINDOWS:
        report.check(
            f"rolling window {window} reports coverage",
            window in coverage,
            f"coverage={coverage[window]:g}s" if window in coverage else "series absent",
        )
    for family in ("requests", "tokens"):
        present = q(f'switchyard_economics_rolling_{family}{{job="{job}"}}')
        report.check(f"rolling {family} series exist", bool(present), f"{len(present)} series")

    print("\nWindows are distinguishable")
    # A rolling figure must never be labelled as the calendar day: they are
    # published under different metric names, and this asserts that both exist so
    # neither can be silently substituted for the other.
    calendar = q(f'switchyard_economics_calendar_tokens{{job="{job}"}}')
    report.check("Bangkok calendar-day series exist", bool(calendar), f"{len(calendar)} series")
    report.check(
        "rolling and calendar families are separate",
        bool(calendar) and bool(q(f'switchyard_economics_rolling_tokens{{job="{job}"}}')),
        "both published under their own names",
    )

    return report.summary()


if __name__ == "__main__":
    sys.exit(main())
