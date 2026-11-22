#!/usr/bin/env python3
"""Mutation control for verify_batch_a.py.

A green verifier proves nothing unless it goes red on the defect it claims to catch.
Every mutant below is applied to the candidate manifest (or the truth table), the
verifier is run as a separate process, and the run must exit 1 while naming the
expected check. A mutant that does not change the bytes is a harness failure, not a
survivor.

Usage: mutants_batch_a.py --baseline B --candidate C --truth T --facts F
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import apply_batch_a as edits  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
VERIFIER = os.path.join(HERE, "verify_batch_a.py")


def mutate_lines(text: str, mutate) -> str:
    lines = text.splitlines(keepends=True)
    mutate(lines)
    return "".join(lines)


def m_st_free_plain_candidate(text: str) -> str:
    return mutate_lines(
        text,
        lambda lines: edits.set_list(
            lines,
            "routes.ST_Free",
            "candidates",
            [
                "stephen-ling-3_2-flash-free",
                "stephen-nemotron-3-5-super-vl-free",
                "stephen-flint-v1-0-free",
                "stephen-dots-3-note-preview-free",
                "stephen-gemma-4-26b-a4b-it-free",
                "stephen-gemma-4-31b-it-free",
                "stephen-openrouter-free",
            ],
        ),
    )


def m_st_free_paid_leg(text: str) -> str:
    return mutate_lines(
        text,
        lambda lines: edits.set_list(
            lines,
            "routes.ST_Free",
            "candidates",
            [
                "stephen-ling-3_2-flash-free",
                "stephen-nemotron-3-5-super-vl-free",
                "stephen-flint-v1-0-free",
                "stephen-dots-3-note-preview-free",
                "stephen-gemma-4-26b-a4b-it-free",
                "stephen-gemma-4-31b-it-free",
                "stephen-paid-glm-5-3-flash",
            ],
        ),
    )


def m_context_inherits_cap(text: str) -> str:
    return mutate_lines(
        text,
        lambda lines: edits.set_scalar(
            lines,
            "routes.direct-openrouter-stephen-ernie-5_5-flash-preview-free",
            "context_window",
            "262144",
        ),
    )


def m_paid_route_free_leg(text: str) -> str:
    return mutate_lines(
        text,
        lambda lines: edits.set_list(
            lines,
            "routes.ST_Paid",
            "candidates",
            ["stephen-paid-glm-5-3-flash", "stephen-flint-v1-0-free"],
        ),
    )


def m_cross_policy_widened(text: str) -> str:
    return mutate_lines(
        text,
        lambda lines: edits.set_scalar(
            lines, "composites.ST_Route", "cross_on", '"failure_class"'
        ),
    )


def m_embed_dimension(text: str) -> str:
    return mutate_lines(
        text,
        lambda lines: edits.set_scalar(lines, "capabilities.ST_Embed", "dimensions", "3072"),
    )


def m_nested_composite(text: str) -> str:
    return mutate_lines(
        text,
        lambda lines: edits.set_list(
            lines,
            "composites.smart-nt-bounded-trial",
            "children",
            [
                "direct-google-gemini-3.5-flash-lite",
                "direct-google-gemini-3.1-flash-lite",
                "direct-openai-gpt-6-luna-nt",
                "direct-deepseek-deepseek-flash-nt",
                "VC_Paid",
            ],
        ),
    )


def m_paid_declared_free(text: str) -> str:
    return mutate_lines(
        text,
        lambda lines: edits.insert_key(
            lines,
            "targets.stephen-paid-glm-5-3-flash",
            "free_tier",
            "true",
            after="llm_client",
        ),
    )


def m_owner_route_touched(text: str) -> str:
    return mutate_lines(
        text,
        lambda lines: edits.set_scalar(
            lines, "routes.direct-openrouter-glm-5_3-flash", "context_window", "524288"
        ),
    )


def m_target_added(text: str) -> str:
    marker = "\n# ---- Stephen primitive routes ----\n"
    assert text.count(marker) == 1
    block = (
        '\n[targets.stephen-synthetic-extra]\n'
        'id = "vendor/extra:free"\n'
        'llm_client = "openrouter_stephen"\n'
    )
    return text.replace(marker, block + marker, 1)


def m_wrong_client(text: str) -> str:
    return mutate_lines(
        text,
        lambda lines: edits.set_scalar(
            lines, "targets.stephen-flint-v1-0-free", "llm_client", '"openrouter"'
        ),
    )


def m_dangling_candidate(text: str) -> str:
    return mutate_lines(
        text,
        lambda lines: edits.set_list(
            lines,
            "routes.ST_Free",
            "candidates",
            [
                "stephen-ling-3_2-flash-free",
                "stephen-nemotron-3-5-super-vl-free",
                "stephen-flint-v1-0-free",
                "stephen-dots-3-note-preview-free",
                "stephen-gemma-4-26b-a4b-it-free",
                "stephen-gemma-4-31b-it-free",
                "stephen-retired-leaf",
            ],
        ),
    )


MUTANTS = [
    ("M1 ST_Free keeps the dynamic router", m_st_free_plain_candidate, "A8"),
    ("M2 ST_Free gains a metered leg", m_st_free_paid_leg, "A8"),
    ("M3 direct leaf inherits the parent cap", m_context_inherits_cap, "A15"),
    ("M4 ST_Paid gains a free leg", m_paid_route_free_leg, "A10"),
    ("M5 ST_Route crosses on every failure class", m_cross_policy_widened, "A21"),
    ("M6 ST_Embed dims drift to 3072", m_embed_dimension, "A22"),
    ("M7 a composite names a composite", m_nested_composite, "A20"),
    ("M8 a metered target declares free_tier", m_paid_declared_free, "A17"),
    ("M9 an owner route is edited", m_owner_route_touched, "A7"),
    ("M10 a new target appears", m_target_added, "A2"),
    ("M11 a Stephen target uses another client", m_wrong_client, "A11"),
    ("M12 a candidate names a missing target", m_dangling_candidate, "A18"),
]


def run_verifier(baseline: str, candidate: str, truth: str, facts: str) -> tuple[int, str]:
    proc = subprocess.run(
        [
            sys.executable,
            VERIFIER,
            "--baseline",
            baseline,
            "--candidate",
            candidate,
            "--truth",
            truth,
            "--facts",
            facts,
        ],
        capture_output=True,
        text=True,
    )
    return proc.returncode, proc.stdout + proc.stderr


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--facts", required=True)
    args = ap.parse_args(argv)

    base_text = open(args.candidate).read()
    truth_text = open(args.truth).read()
    failures = 0
    work = tempfile.mkdtemp(prefix="batch-a-mutants-")
    try:
        # Control: the unmutated candidate passes.
        rc, out = run_verifier(args.baseline, args.candidate, args.truth, args.facts)
        ok = rc == 0
        print(f"{'PASS' if ok else 'FAIL'}  CONTROL unmutated candidate passes (rc={rc})")
        failures += 0 if ok else 1

        for name, mutate, expected in MUTANTS:
            mutated = mutate(base_text)
            if mutated == base_text:
                print(f"FAIL  {name}: mutation did not change the file")
                failures += 1
                continue
            path = os.path.join(work, "candidate.toml")
            with open(path, "w") as handle:
                handle.write(mutated)
            rc, out = run_verifier(args.baseline, path, args.truth, args.facts)
            hit = any(
                line.startswith("FAIL") and f" {expected} " in line
                for line in out.splitlines()
            )
            ok = rc == 1 and hit
            print(
                f"{'PASS' if ok else 'FAIL'}  {name}: rc={rc} caught_by={expected} "
                f"{'ok' if hit else 'NOT CAUGHT'}"
            )
            failures += 0 if ok else 1

        # Truth drift: the config is right and the document lies.
        truth = json.loads(truth_text)
        for leaf in truth["leaves"]:
            if leaf["route"] == "direct-openrouter-stephen-ernie-5_5-flash-preview-free":
                leaf["declared_route_context"] = 262144
        path = os.path.join(work, "truth.json")
        with open(path, "w") as handle:
            json.dump(truth, handle)
        rc, out = run_verifier(args.baseline, args.candidate, path, args.facts)
        hit = any(line.startswith("FAIL") and " A24 " in line for line in out.splitlines())
        print(f"{'PASS' if rc == 1 and hit else 'FAIL'}  M13 truth table drifts from config: rc={rc} caught_by=A24")
        failures += 0 if rc == 1 and hit else 1

        # The empty-verdict guard must refuse rather than report success.
        guard = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; sys.path.insert(0, %r); import verify_batch_a as v; "
                "raise SystemExit(v.Verdict().report())" % HERE,
            ],
            capture_output=True,
            text=True,
        )
        ok = guard.returncode == 3
        print(f"{'PASS' if ok else 'FAIL'}  CONTROL empty verdict refuses (rc={guard.returncode})")
        failures += 0 if ok else 1
    finally:
        shutil.rmtree(work, ignore_errors=True)

    total = len(MUTANTS) + 3
    print(f"\n{total} mutants/controls, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
