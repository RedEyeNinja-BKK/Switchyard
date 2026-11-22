#!/usr/bin/env python3
"""Prove verify_smart_decision_manifest.py actually bites.

Each mutant is a change a plausible mistake would make. The harness applies it to
the candidate manifest, asserts the mutation really changed the bytes (a mutation
that silently failed to apply would otherwise be reported as "caught"), and
requires the verifier to fail with the check that owns that mistake.

The unmutated candidate must pass, so a verifier that refuses everything is not
mistaken for a sensitive one.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

VERIFIER = Path(__file__).with_name("verify_smart_decision_manifest.py")

MUTANTS = [
    (
        "M1 selection downgraded to ordered",
        'selection = "capability_first"\n',
        'selection = "ordered"\n',
        "R1",
    ),
    (
        "M2 a constituent lane dropped from the pool",
        'children = ["smartlocal-decision", "smartfree-decision", "google-decision"]',
        'children = ["smartlocal-decision", "smartfree-decision"]',
        "R2",
    ),
    (
        "M3 a second OpenRouter leaf smuggled into smartfree-decision",
        'targets = ["direct-openrouter-span-01-lite"]\n\n[decision_lanes.smartlocal-decision]',
        'targets = ["direct-openrouter-span-01-lite", "direct-openrouter-stephen-span-01-lite"]'
        "\n\n[decision_lanes.smartlocal-decision]",
        "R4",
    ),
    (
        "M4 the allowlisted OpenRouter leaf replaced by Stephen's",
        '[decision_lanes.smartfree-decision]\ntargets = ["direct-openrouter-span-01-lite"]',
        '[decision_lanes.smartfree-decision]\ntargets = ["direct-openrouter-stephen-span-01-lite"]',
        "R4",
    ),
    (
        "M5 Google target pointed at Stephen's credential",
        'decision_api_key_env = "SWITCHYARD_GEMINI_API_KEY"',
        'decision_api_key_env = "SWITCHYARD_STEPHEN_OPENROUTER_API_KEY"',
        "R5",
    ),
    (
        "M6 legacy ordered tier reordered",
        'children = ["smartfree-decision", "smartlocal-decision"]',
        'children = ["smartlocal-decision", "smartfree-decision"]',
        "R6",
    ),
    (
        "M7 selection declaration deleted, so the lane silently orders",
        'selection = "capability_first"\n',
        "",
        "R1",
    ),
    (
        "M8 Google state forms reverted to the measured-wrong declaration",
        'state_forms = { plain_string = true, structured = false }\nfree_tier = true\n',
        'state_forms = { plain_string = false, structured = true }\nfree_tier = true\n',
        "R9",
    ),
    (
        "M9 a pool member declares no state form it can serve",
        'state_forms = { plain_string = true, structured = false }\nfree_tier = true\n',
        'state_forms = { plain_string = false, structured = false }\nfree_tier = true\n',
        "R9",
    ),
]


def run(manifest: Path, deployed: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(VERIFIER),
            "--manifest",
            str(manifest),
            "--deployed",
            str(deployed),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: mutants_smart_decision_manifest.py CANDIDATE DEPLOYED", file=sys.stderr)
        return 2
    candidate = Path(sys.argv[1])
    deployed = Path(sys.argv[2])
    original = candidate.read_text()

    failures: list[str] = []

    with tempfile.TemporaryDirectory() as directory:
        control = Path(directory) / "control.toml"
        shutil.copy(candidate, control)
        result = run(control, deployed)
        if result.returncode != 0:
            failures.append(f"CONTROL: the unmutated candidate was refused\n{result.stderr}")
        else:
            print("CONTROL unmutated candidate passes")

        for label, old, new, expected in MUTANTS:
            occurrences = original.count(old)
            if occurrences != 1:
                failures.append(
                    f"{label}: anchor matched {occurrences} times, expected exactly 1; "
                    "the mutation would not have applied"
                )
                continue
            mutated = Path(directory) / "mutant.toml"
            mutated.write_text(original.replace(old, new))
            if mutated.read_text() == original:
                failures.append(f"{label}: mutation did not change the bytes")
                continue
            result = run(mutated, deployed)
            if result.returncode == 0:
                failures.append(f"{label}: SURVIVED (verifier passed a mutated manifest)")
                print(f"SURVIVED {label}")
                continue
            combined = result.stdout + result.stderr
            if expected not in combined:
                failures.append(
                    f"{label}: refused, but not by {expected}:\n{combined}"
                )
                print(f"WRONG-CHECK {label}")
                continue
            print(f"CAUGHT {label} -> {expected}")

    if failures:
        print("\nFAILURES:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1
    print(f"\n{len(MUTANTS)} mutants, 0 survived")
    return 0


if __name__ == "__main__":
    sys.exit(main())
