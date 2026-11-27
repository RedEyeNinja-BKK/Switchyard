#!/usr/bin/env python3
"""Negative controls for verify_no_live_regression.py.

A guard that has never been seen to fail is not evidence. Each mutant injects one
regression shape into a copy of the live manifest and requires the verifier to
FAIL on the check that owns that shape - by name, not merely by exit code, so a
crash for an unrelated reason cannot be mistaken for a catch.

Mutants are parsed and re-serialised, which drops comments; the verifier only
reads structure, so that is sound here.
"""

from __future__ import annotations

import copy
import tomllib
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VERIFIER = HERE / "verify_no_live_regression.py"
PYTHON = sys.executable


def dump(doc: dict, path: Path) -> None:
    # Minimal TOML writer: enough for the shapes these manifests use.
    def fmt(value: object) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, str):
            return '"' + value.replace('"', '\\"') + '"'
        if isinstance(value, (int, float)):
            return repr(value)
        if isinstance(value, list):
            return "[" + ", ".join(fmt(v) for v in value) + "]"
        raise TypeError(type(value))

    lines: list[str] = []

    def quote(key: str) -> str:
        # Always quote: deployed keys legitimately contain dots
        # (`direct-google-gemini-3.5-flash-lite`), and an unquoted dotted key is a
        # nested table, which silently renames the object on round-trip.
        return '"' + key.replace('\\', '\\\\').replace('"', '\\"') + '"'

    def walk(parts: list[str], table: dict) -> None:
        scalars = {k: v for k, v in table.items() if not isinstance(v, dict)}
        nested = {k: v for k, v in table.items() if isinstance(v, dict)}
        if parts:
            lines.append("\n[" + ".".join(quote(p) for p in parts) + "]")
        for key, value in scalars.items():
            lines.append(f"{quote(key)} = {fmt(value)}")
        for key, value in nested.items():
            walk([*parts, key], value)

    walk([], doc)
    path.write_text("\n".join(lines) + "\n")


def run(verifier_args: list[str]) -> tuple[int, str]:
    proc = subprocess.run(
        [PYTHON, str(VERIFIER), *verifier_args],
        capture_output=True,
        text=True,
    )
    return proc.returncode, proc.stdout + proc.stderr


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: mutants_live_regression.py LIVE CANDIDATE", file=sys.stderr)
        return 2
    live_path, cand_path = Path(sys.argv[1]), Path(sys.argv[2])
    live = tomllib.loads(live_path.read_text())
    workdir = Path("/var/tmp/st1/work/mutants")
    workdir.mkdir(parents=True, exist_ok=True)

    mutants: list[tuple[str, dict, str]] = []

    # A mutant is only a mutant if the round-trip preserves everything except the
    # injected defect. Assert that first; a lossy writer turns every mutant into
    # "the candidate drops targets" and the control then proves nothing.
    probe = workdir / "roundtrip.toml"
    dump(live, probe)
    if tomllib.loads(probe.read_text()) != live:
        print("ABORT  round-trip is lossy; mutant results would be meaningless", file=sys.stderr)
        return 2
    print("OK    round-trip preserves the live manifest exactly")

    # M1 - a target deployed after the snapshot, absent from the candidate.
    m1 = copy.deepcopy(live)
    m1["targets"]["ghost-deployed-later"] = {
        "id": "ghost/model",
        "llm_client": "openrouter_stephen",
        "vision": False,
        "video": False,
        "tool_calling": False,
    }
    mutants.append(("M1 deployed-only target dropped", m1, None, "L1 no deployed target is dropped"))

    # M2 - an unreviewed key change on a shared route.
    m2 = copy.deepcopy(live)
    allowed_routes = {
        "ST_Free",
        "direct-openrouter-stephen-ernie-5_5-flash-preview-free",
        "direct-openrouter-stephen-ling-3_2-flash-free",
        "direct-openrouter-stephen-nemotron-3-5-super-vl-free",
        "direct-openrouter-stephen-openrouter-free",
    }
    route_name = next(name for name in m2["routes"] if name not in allowed_routes)
    m2["routes"][route_name]["context_window"] = 999999
    mutants.append((f"M2 unreviewed route key change ({route_name})", m2, None, "L4 deployed routes change only in reviewed keys"))

    # M3 - an unreviewed key change on a shared target.
    m3 = copy.deepcopy(live)
    first = next(name for name in m3["targets"] if name.startswith("stephen-") and name != "stephen-openrouter-free")
    m3["targets"][first]["vision"] = not bool(m3["targets"][first].get("vision"))
    mutants.append(("M3 unreviewed target key change", m3, None, "L3 deployed targets change only in reviewed keys"))

    # M4 - the CANDIDATE loses a key from a non-target section. The mutation must
    # be on the candidate: L5 asks what the candidate dropped, so mutating live
    # would remove the expectation rather than satisfy it (measured survivor).
    candidate = tomllib.loads(cand_path.read_text())
    m4 = copy.deepcopy(candidate)
    caps = m4.get("capabilities")
    if isinstance(caps, dict) and caps:
        caps.pop(next(iter(caps)))
        mutants.append(("M4 candidate drops a non-target key", None, m4, "L5 deployed non-target sections survive unchanged"))
    else:
        print("SKIP  M4 (candidate has no capabilities section)", file=sys.stderr)
        return 1

    # M6 - the widened smart-nt allowance must stay narrow. `selection` is a
    # reviewed key on ST_Free, so this mutant is built to be allowed by the
    # ST_Free entry and refused on smart-nt; if it survives, the F1/F2 allowlist
    # has become a blanket permit for the route it was meant to bound.
    m6 = copy.deepcopy(live)
    if "smart-nt" in m6["routes"]:
        m6["routes"]["smart-nt"]["selection"] = "peer"
        mutants.append(("M6 smart-nt unreviewed key stays refused", m6, None, "L4 deployed routes change only in reviewed keys"))
    else:
        print("SKIP  M6 (no smart-nt route in live manifest)", file=sys.stderr)
        return 1

    # M7 - the Google state-forms allowance must stay narrow. `state_forms` is a
    # reviewed key on that one target; any other key on it stays unreviewed. If this
    # survives, the SD1 correction has become a blanket permit for a pool member.
    m7 = copy.deepcopy(live)
    google = m7["targets"].get("direct-google-gemini-decision-1")
    if google is not None:
        google["max_input_tokens"] = int(google.get("max_input_tokens", 0)) + 1
        mutants.append(("M7 google target unreviewed key stays refused", m7, None, "L3 deployed targets change only in reviewed keys"))
    else:
        print("SKIP  M7 (no google decision target in live manifest)", file=sys.stderr)
        return 1

    failures = 0
    for label, live_doc, cand_doc, expected_check in mutants:
        live_arg, cand_arg = live_path, cand_path
        tag = label.split()[0]
        if live_doc is not None:
            live_arg = workdir / f"{tag}-live.toml"
            dump(live_doc, live_arg)
        if cand_doc is not None:
            cand_arg = workdir / f"{tag}-candidate.toml"
            dump(cand_doc, cand_arg)
        code, output = run(["--live", str(live_arg), "--candidate", str(cand_arg)])
        caught = code != 0 and any(
            line.startswith("FAIL") and expected_check in line for line in output.splitlines()
        )
        status = "CAUGHT" if caught else "SURVIVED"
        print(f"{status}  {label}   expected_check={expected_check!r} exit={code}")
        if not caught:
            failures += 1
            print(output)

    # Control: an unmutated live manifest must PASS, or the mutants prove nothing.
    code, _ = run(["--live", str(live_path), "--candidate", str(cand_path)])
    print(f"{'PASS' if code == 0 else 'FAIL'}  CONTROL unmutated live manifest passes")
    if code != 0:
        failures += 1

    print(f"\n{len(mutants)} mutants, {failures} not caught")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
