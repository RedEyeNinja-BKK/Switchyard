#!/usr/bin/env python3
"""Fail-closed SOURCE-ONLY composer for the ACTUAL #51+#52 release manifest.

The P9 builder alone makes a six-stage metadata candidate from its input;
merging PR #51 into PR #52 does not magically add the judge route's third tier.
Order matters: apply the exact anchored judge edit to pristine CURRENT first,
then build the P9 declaration chain against THAT modified manifest. This script
checks that the gate passed the FINAL composed candidate, not a P9-only file.

No API calls, credentials, active manifest writes, restarts or deployments.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tomllib

HERE = Path(__file__).resolve().parent
NEW_ALIAS = "direct-htpc-qwen3_5-9b-mtp-nt"
NEW_TARGET = "htpc-qwen3_5-9b-mtp-nt"
RETIRED_ALIAS = "direct-openrouter-liquid-lfm-2_5-2_6b-free"
RETIRED_TARGET = "openrouter-liquid-lfm-2_5-2_6b-free"
TIERS = [["openai-gpt-6-luna-nt"], [NEW_TARGET]]
BLOCKED_WRITE_ROOTS = (Path("/etc/switchyard"), Path("/var/lib/switchyard/releases"),
                       Path("/opt/switchyard"), Path("/opt/deploy"))


class Refused(RuntimeError):
    pass


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def below(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def require_safe_input_output(baseline: Path, out_dir: Path, expected_sha: str) -> None:
    if len(expected_sha) != 64 or any(x not in "0123456789abcdef" for x in expected_sha):
        raise Refused("expected baseline SHA256 must be 64 lowercase hex characters")
    actual = sha256(baseline)
    if actual != expected_sha:
        raise Refused(f"STALE_BASELINE: expected {expected_sha}, actual {actual}")
    b, o = baseline.resolve(), out_dir.resolve()
    if below(b, o) or below(o, b):
        raise Refused("input and output paths overlap")
    if any(below(o, root) for root in BLOCKED_WRITE_ROOTS):
        raise Refused("scratch output under a protected production root")
    if o.exists() and any(o.iterdir()):
        raise Refused("work directory must be empty to avoid stale stage artifacts")
    src = load(baseline)
    if (src.get("routes") or {}).get(NEW_ALIAS):
        raise Refused("already-tiered CURRENT: refuse double application")


def verify_composition(before: dict, after: dict, gate: dict) -> None:
    b_routes, a_routes = before.get("routes") or {}, after.get("routes") or {}
    b_targets, a_targets = before.get("targets") or {}, after.get("targets") or {}
    b_clients, a_clients = before.get("llm_clients") or {}, after.get("llm_clients") or {}

    for kind, old, new, added, retired in (
        ("route", b_routes, a_routes, NEW_ALIAS, RETIRED_ALIAS),
        ("target", b_targets, a_targets, NEW_TARGET, RETIRED_TARGET),
        ("client", b_clients, a_clients, "htpc-nt", None),
    ):
        new_names = set(new) - set(old)
        removed = set(old) - set(new)
        if new_names != {added} or removed != ({retired} if retired else set()):
            raise Refused(f"COMPOSITION_{kind.upper()}_SET: added {sorted(new_names)}, "
                          f"removed {sorted(removed)}")

    judge = a_routes.get("smart-judge") or {}
    if (judge.get("terminal_fallback") != TIERS
            or judge.get("terminal_policy") != "availability"):
        raise Refused("COMPOSITION_JUDGE_TIERS: missing ordered Luna NT then HTPC NT")
    old_judge = b_routes.get("smart-judge") or {}
    if old_judge.get("terminal_policy") != "quota_only":
        raise Refused("COMPOSITION_BASE_POLICY: expected original judge quota-only")
    if judge.get("candidates") != old_judge.get("candidates"):
        raise Refused("COMPOSITION_PRIMARY_POOL: altered Gemma candidates")
    for key, value in old_judge.items():
        if key not in ("terminal_fallback", "terminal_policy") and judge.get(key) != value:
            raise Refused(f"COMPOSITION_JUDGE_DRIFT: {key}")

    leaf = a_routes[NEW_ALIAS]
    target = a_targets[NEW_TARGET]
    if (leaf.get("target") != NEW_TARGET or leaf.get("tool_calling") is not True
            or leaf.get("vision") is not False or leaf.get("video") is not False
            or leaf.get("reasoning") is not False
            or target.get("reasoning_policy") != "disabled"
            or target.get("reasoning_dialect") != "llama_cpp_enable_thinking"
            or target.get("llm_client") != "htpc-nt"
            or target.get("tool_calling") is not True
            or target.get("vision") is not False or target.get("video") is not False):
        raise Refused("COMPOSITION_HTPC_LEAF: no-thinking tool-only leaf mismatch")
    if b_targets.get("htpc-qwen3_5-9b-mtp", {}).get("id") != target.get("id"):
        raise Refused("COMPOSITION_WEIGHTS: existing sibling model changed")
    if a_clients["htpc-nt"].get("quota_group") != "htpc":
        raise Refused("COMPOSITION_QUOTA: htpc-nt must share HTPC quota")

    expected_direct = sum(name.startswith("direct-") for name in b_routes)
    actual_direct = sum(name.startswith("direct-") for name in a_routes)
    if actual_direct != expected_direct:  # +1 HTPC and -1 retired Liquid
        raise Refused(f"COMPOSITION_COUNT: expected {expected_direct}, got {actual_direct}")
    if (gate.get("status") != "PASS" or gate.get("failures")
            or gate.get("direct_count") != actual_direct):
        raise Refused("COMPOSITION_P9: strict P9 did not pass the final combined manifest")


def run_stage(label: str, argv: list[str]) -> None:
    result = subprocess.run(argv, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise Refused(f"{label} failed rc={result.returncode}: "
                      f"{(result.stderr or result.stdout).strip()[:1500]}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True,
                        help="exact CURRENT routes.toml read-only snapshot")
    parser.add_argument("--expected-baseline-sha", required=True,
                        help="sha256 of confirmed CURRENT routes.toml")
    parser.add_argument("--work-dir", type=Path, required=True,
                        help="new empty non-production scratch directory")
    args = parser.parse_args(argv)
    try:
        require_safe_input_output(args.baseline, args.work_dir, args.expected_baseline_sha)
        args.work_dir.mkdir(parents=True, exist_ok=True)
        judge = args.work_dir / "1-smart-judge.toml"
        chain = args.work_dir / "p9"
        final = chain / "6-candidate.toml"

        run_stage("three-tier manifest", [
            sys.executable, str(HERE / "apply_smart_judge_three_tier.py"),
            "--baseline", str(args.baseline), "--out", str(judge)])
        run_stage("P9 qualification on three-tier manifest", [
            sys.executable, str(HERE / "build_p9_candidate.py"),
            "--baseline", str(judge), "--out-dir", str(chain)])
        if not final.is_file():
            raise Refused("COMPOSITION_MISSING_FINAL: no p9/6-candidate.toml")
        result = subprocess.run([
            sys.executable, str(HERE / "direct_capability_gate.py"),
            "--manifest", str(final), "--json"],
            capture_output=True, text=True, check=False)
        if result.returncode:
            raise Refused(f"COMPOSITION_P9_GATE: rc={result.returncode} "
                          f"{(result.stderr or result.stdout)[:1000]}")
        try:
            report = json.loads(result.stdout)
        except (ValueError, TypeError) as error:
            raise Refused("COMPOSITION_P9_REPORT: malformed gate output") from error

        verify_composition(load(args.baseline), load(final), report)
        receipt = {
            "status": "COMPOSED_SOURCE_CANDIDATE_P9_PASS_NOT_DEPLOYED",
            "baseline_sha256": sha256(args.baseline),
            "judge_manifest_sha256": sha256(judge),
            "final_manifest_sha256": sha256(final),
            "gate_script_sha256": sha256(HERE / "direct_capability_gate.py"),
            "direct_routes": report["direct_count"],
            "P9_findings": len(report["failures"]),
            "new_alias": NEW_ALIAS, "terminal_tiers": TIERS,
            "final_path": str(final),
            "warning": "Does not build/test the Switchyard binary, publish releases, or prove live error-rate recovery",
        }
        (args.work_dir / "composed-receipt.json").write_text(
            json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(receipt, indent=2))
        return 0
    except (OSError, Refused, ValueError, tomllib.TOMLDecodeError) as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
