#!/usr/bin/env python3
"""Build a P9-valid candidate from the LIVE manifest, in one auditable chain.

Each step is its own applier with its own refusals; this script only decides
the ORDER, keeps every intermediate file for review, and runs the gate last.
Nothing here writes to the release store: the output is a candidate file that
the guarded release path validates again before anything is installed.

Order matters and is deliberate:

  1. remove a retired leaf whose provider no longer serves the model, so a dead
     target is never offered qualification evidence it cannot have;
  2. widen a route whose explicit ``tool_calling = false`` was a placeholder
     that the target's own evidence now contradicts;
  3. write target ``tool_calling = true`` where evidence exists;
  4. declare target ``tool_calling = false`` where the provider itself refuses
     tool requests, which is only legal behind a route that already refuses;
  5. normalise the remaining absent declarations (video, carriers, inherited
     flags) with the existing safe reconciler;
  6. run the P9 contract over the result.

Usage:
  python3 build_p9_candidate.py --baseline CURRENT/routes.toml \
      --out-dir /var/tmp/p9-candidate [--models live-models.json]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable


def run(step: str, argv: list[str], accept: tuple[int, ...] = (0,)) -> tuple[bool, str]:
    completed = subprocess.run(argv, capture_output=True, text=True, cwd=str(HERE))
    output = (completed.stdout + completed.stderr).strip()
    print(f"[{step}] rc={completed.returncode}")
    for line in output.splitlines():
        print(f"    {line}")
    return completed.returncode in accept, output


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--models", type=Path, default=None,
                        help="saved live GET /v1/models JSON for the parity check")
    parser.add_argument("--evidence-dir", type=Path,
                        default=HERE / "evidence" / "tool-qualification-20261123")
    args = parser.parse_args(argv)

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    evidence = args.evidence_dir
    current = args.baseline

    # (name, argv, accepted exit codes). The reconciler writes its output and
    # THEN reports how many declarations it could not back with evidence, so
    # rc=2 there means "file written, gaps remain" — the expected mid-pipeline
    # state. rc=3 (refusal, nothing written) stays fatal everywhere.
    steps = [
        ("retired-leaf", [
            PY, str(HERE / "remove_retired_direct_leaf.py"),
            "--baseline", str(current),
            "--evidence", str(evidence / "retired_direct_leaf_evidence.json"),
            "--out", str(out / "1-retired-removed.toml")], (0,)),
        ("route-admission", [
            PY, str(HERE / "apply_route_tool_admission.py"),
            "--baseline", str(out / "1-retired-removed.toml"),
            "--evidence", str(evidence / "route_admission_evidence.json"),
            "--out", str(out / "2-routes-widened.toml")], (0,)),
        ("target-tools", [
            PY, str(HERE / "apply_target_tool_qualification.py"),
            "--baseline", str(out / "2-routes-widened.toml"),
            "--evidence", str(evidence / "target_tool_evidence.json"),
            "--out", str(out / "3-targets-qualified.toml")], (0,)),
        ("target-refusal", [
            PY, str(HERE / "declare_target_tool_refusal.py"),
            "--baseline", str(out / "3-targets-qualified.toml"),
            "--evidence", str(evidence / "target_tool_refusal_evidence.json"),
            "--out", str(out / "4-targets-refused.toml")], (0,)),
        ("normalise", [
            PY, str(HERE / "reconcile_direct_capability_declarations.py"),
            "--baseline", str(out / "4-targets-refused.toml"),
            "--out", str(out / "5-candidate.toml")], (0, 2)),
    ]
    for name, command, accept in steps:
        ok, _ = run(name, command, accept)
        if not ok:
            print(f"STOPPED at {name}: the candidate was not built")
            return 1

    gate = [PY, str(HERE / "direct_capability_gate.py"),
            "--manifest", str(out / "5-candidate.toml")]
    if args.models:
        gate += ["--models", str(args.models)]
    ok, output = run("p9-gate", gate)
    print("P9 PASS" if ok else "P9 REFUSED")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
