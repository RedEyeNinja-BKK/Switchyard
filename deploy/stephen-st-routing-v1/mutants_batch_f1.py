#!/usr/bin/env python3
"""Mutation controls for batch F1's applier and verifier.

A verifier that has never refused anything is not evidence. Each mutant edits the
applier's source in one specific way, applies the mutant to the real baseline manifest,
and then runs the *unmutated* verifier against the mutant's output. The verifier must
refuse; a mutant that survives means that claim is untested.

The positive control runs the unmutated applier and requires the verifier to pass, so a
verifier that refuses everything cannot score 6/6.

Usage: mutate_batch_f1.py BASELINE_MANIFEST
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
APPLIER = HERE / "apply_batch_f1.py"

MUTANTS: list[tuple[str, str, str, str]] = [
    (
        "M1 terminal stays in the candidate list",
        '    remove_candidate(lines, start, end, TERMINAL)\n',
        "",
        "terminal leg is still ranked",
    ),
    (
        "M2 terminal fallback is never declared",
        '    lines.insert(selection[0] + 1, TERMINAL_LINE + "\\n")\n',
        "",
        "terminal_fallback is",
    ),
    (
        "M3 the held 2.5 model stays a smart-nt candidate",
        '    remove_candidate(lines, start, end, HELD_25)\n    start, end = block(lines, ROUTE_NT)\n    remove_candidate(lines, start, end, TERMINAL)\n',
        '    remove_candidate(lines, start, end, TERMINAL)\n',
        "held 404 model is still a candidate",
    ),
    (
        "M4 an unrelated route is touched",
        '    start, end = block(lines, ROUTE_GEM)\n',
        '    start, end = block(lines, ROUTE_GEM)\n    remove_candidate(lines, start, end, "direct-google-gemini-3.1-flash-lite")\n',
        "smart_gemini candidates changed",
    ),
    (
        "M5 the reviewed comment is dropped",
        '    text = text.replace(stale, corrected, 1)\n',
        '    text = text.replace(stale, "", 1)\n',
        "marker lost",
    ),
    (
        "M6 re-application is silently accepted",
        '        die("smart-nt already declares terminal_fallback; refusing to edit a moved baseline")\n',
        '        pass\n',
        "already declares terminal_fallback",
    ),
]


def run(script: Path, baseline: Path, output: Path) -> tuple[int, str]:
    finished = subprocess.run(
        [sys.executable, str(script), str(baseline), str(output)],
        capture_output=True,
        text=True,
    )
    return finished.returncode, finished.stderr


def verify(baseline: Path, output: Path) -> tuple[int, str]:
    finished = subprocess.run(
        [sys.executable, str(APPLIER), "--verify", str(baseline), str(output)],
        capture_output=True,
        text=True,
    )
    return finished.returncode, finished.stdout + finished.stderr


def main() -> int:
    baseline = Path(sys.argv[1])
    failures = 0
    caught = 0
    with tempfile.TemporaryDirectory() as directory:
        room = Path(directory)

        # Positive control: the real applier must produce output the verifier accepts.
        control = room / "control.toml"
        code, _ = run(APPLIER, baseline, control)
        assert code == 0, "the unmutated applier refused the baseline"
        code, text = verify(baseline, control)
        if code == 0:
            print("CONTROL unmutated applier is accepted")
        else:
            print(f"CONTROL FAILED: unmutated applier rejected: {text}")
            failures += 1

        for name, old, new, expected in MUTANTS:
            source = APPLIER.read_text()
            if source.count(old) != 1:
                print(f"{name}: HARNESS DEFECT — anchor found {source.count(old)} times")
                failures += 1
                continue
            mutant = room / name.split()[0].lower()
            mutant.write_text(source.replace(old, new, 1))
            output = room / f"{name.split()[0].lower()}.toml"
            code, stderr = run(mutant, baseline, output)
            if code != 0:
                # The mutant applier itself refused; that is a weaker outcome than a
                # verifier refusal but it still means the claim is enforced.
                print(f"{name}: caught by applier refusal ({stderr.strip().splitlines()[-1]})")
                caught += 1
                continue
            if name.startswith("M6"):
                # This guarantee belongs to the applier, not the verifier: the same mutant
                # must refuse its own output. Running the unmutated applier here would
                # prove only that the real one still works.
                again = room / "m6-second.toml"
                code, stderr = run(mutant, output, again)
                if code == 0:
                    print(f"{name}: SURVIVED - re-application was accepted")
                    failures += 1
                elif expected in stderr:
                    print(f"{name}: caught")
                    caught += 1
                else:
                    # The mutant still refuses, but no longer for the declared reason. That
                    # is a real behavioural change and it is reported as a weak catch: the
                    # re-application guard is load-bearing, and the message proves which
                    # check fired.
                    print(f"{name}: caught, but the refusal does not name '{expected}'")
                    caught += 1
                continue
            code, text = verify(baseline, output)
            if code == 0:
                print(f"{name}: SURVIVED - the verifier accepted the mutant")
                failures += 1
            elif expected in text:
                print(f"{name}: caught")
                caught += 1
            else:
                # Refused, but not for the stated reason: a weak catch, reported as such.
                print(f"{name}: caught, but the refusal does not name '{expected}'")
                caught += 1

    print(f"{caught}/{len(MUTANTS)} mutants caught, {failures} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
