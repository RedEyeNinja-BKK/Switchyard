#!/usr/bin/env python3
"""Mutation controls for the parity verifier.

A verifier that passes is only evidence if it fails when the property it names is broken.
Each mutation below breaks exactly one of P1–P8 and must be reported by that check, by
name. The final control run proves the restored tree is green again.

Restoration is by byte copy with a sha256 equality check, so a crashed run cannot leave a
mutated catalogue behind.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# The baseline this suite verifies against. It moved from 53fd9a84 to e008aa96 when the
# catalogue was re-anchored to the deployed bytes, and a mutation harness that names the
# wrong baseline would report every mutation as CAUGHT for the wrong reason (P7 failing).
MANIFEST = Path(os.environ.get("PARITY_MANIFEST", "/var/tmp/c5/live-e008aa96.toml"))
EXPECTED_HASH = os.environ.get("PARITY_HASH", "e008aa96")
OUT_DIR = Path(os.environ.get("PARITY_OUT", "/var/tmp/c5/parity-out"))
BACKUP = Path("/var/tmp/st1/work/mut-backup-parity")
FILES = ["catalogue.json", "profile-owner.json", "profile-stephen.json", "project.py"]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def backup() -> dict[str, str]:
    BACKUP.mkdir(parents=True, exist_ok=True)
    digests = {}
    for name in FILES:
        source = HERE / name
        (BACKUP / name).write_bytes(source.read_bytes())
        digests[name] = sha(BACKUP / name)
    return digests


def restore(baseline: dict[str, str]) -> None:
    for name in FILES:
        (HERE / name).write_bytes((BACKUP / name).read_bytes())
        if sha(HERE / name) != baseline[name]:
            raise SystemExit(f"restore mismatch for {name}")


def edit_json(name: str, mutate) -> None:
    path = HERE / name
    document = json.loads(path.read_text())
    mutate(document)
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n")


def edit_text(name: str, old: str, new: str) -> None:
    path = HERE / name
    text = path.read_text()
    if text.count(old) != 1:
        raise SystemExit(f"{name}: anchor occurs {text.count(old)} times")
    path.write_text(text.replace(old, new, 1))


def verify() -> tuple[int, str]:
    completed = subprocess.run(
        [
            sys.executable,
            "verify_parity.py",
            "--manifest",
            str(MANIFEST),
            "--expected-capture-hash",
            EXPECTED_HASH,
        ],
        cwd=HERE,
        capture_output=True,
        text=True,
    )
    return completed.returncode, completed.stdout + completed.stderr


def catalogue_fact(mutate) -> None:
    def apply(document: dict) -> None:
        mutate({model["key"]: model for model in document["models"]})

    edit_json("catalogue.json", apply)


MUTATIONS = [
    (
        "PM1-catalogue-quietly-changes-a-model-fact",
        lambda: catalogue_fact(
            lambda models: models["gemma-4-31b-it"]["behaviour"].__setitem__(
                "hosted_reasoning", "none"
            )
        ),
        "P6-owner-projection-matches-deployed",
    ),
    (
        "PM2-a-served-model-is-dropped-from-the-catalogue",
        lambda: edit_json(
            "catalogue.json",
            lambda document: document.__setitem__(
                "models",
                [m for m in document["models"] if m["key"] != "gemma-4-31b-it"],
            ),
        ),
        "P1-deployed-targets-are-catalogued",
    ),
    (
        "PM3-a-profile-starts-holding-model-facts",
        lambda: edit_json(
            "profile-owner.json",
            lambda document: document["state"]["gemini-3.5-flash-lite"].__setitem__(
                "context_window", 1048576
            ),
        ),
        "P8-no-model-facts-in-profiles",
    ),
    (
        "PM4-the-two-accounts-share-a-credential",
        lambda: edit_json(
            "profile-stephen.json",
            lambda document: (
                document["clients"][0].__setitem__("api_key_env", "SWITCHYARD_GEMINI_API_KEY"),
                document["clients"][1].__setitem__("api_key_env", "SWITCHYARD_GEMINI_API_KEY"),
            ),
        ),
        "P3-accounts-are-isolated",
    ),
    (
        "PM5-the-owner-policy-is-replaced-by-peer-selection",
        lambda: edit_json(
            "profile-owner.json",
            lambda document: document["chat_routes"][0].__setitem__("selection", "peer"),
        ),
        "P5-route-policies-preserved",
    ),
    (
        "PM6-the-pool-ships-before-qualification",
        lambda: edit_text(
            "project.py",
            'serving = [key for key in route["members"] if key in servable_keys(prof)]',
            'serving = list(route["members"])',
        ),
        "P4-qualification-gates-the-pool",
    ),
]


def main() -> int:
    baseline = backup()
    report = []
    failures = 0
    for label, apply, expected in MUTATIONS:
        apply()
        code, output = verify()
        caught = expected in output and "FAIL" in output
        # The verifier must fail for the named reason, not merely fail.
        named = any(
            line.startswith(f"FAIL {expected}") for line in output.splitlines()
        )
        if not (code != 0 and caught and named):
            failures += 1
        report.append(
            {
                "mutation": label,
                "expected_check": expected,
                "exit_code": code,
                "caught": bool(code != 0 and caught and named),
                "output_tail": output[-1500:],
            }
        )
        print(f"{label}: {'CAUGHT' if code != 0 and caught and named else 'SURVIVED'}")
        restore(baseline)

    code, output = verify()
    report.append(
        {
            "mutation": "CONTROL-after-restore",
            "expected_check": "none",
            "exit_code": code,
            "caught": code == 0 and "0 failure(s)" in output,
        }
    )
    print(f"CONTROL-after-restore: exit={code}")
    if code != 0 or "0 failure(s)" not in output:
        failures += 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "mutants-parity.json").write_text(json.dumps(report, indent=2))
    print(f"mutations={len(MUTATIONS)} failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
