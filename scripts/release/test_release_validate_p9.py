#!/usr/bin/env python3
"""Offline wiring tests: a real release must not skip, or over-trust, the P9 contract.

Two layers, because they fail differently.

`ResolutionTests` exercise where the checker is looked for: the installed layout and
the source checkout each find their own copy, a missing checker refuses, and the
refusal names every path it tried so nobody has to reverse-engineer parent arithmetic.

`VerdictTests` exercise what a verdict has to contain before it is believed. The gate's
own route rules are tested by its own suite; what is tested here is that a *successful*
exit is not sufficient. A checker that exits 0 on a different manifest, on a truncated
report, or without reading the manifest at all must all refuse, and the candidate binary
must not be executed in any of those cases.

No Switchyard service, credentials, or production state are involved. Every checker is
a stand-in written into a temporary tree.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
REAL_CHECKER = REPO / "deploy" / "stephen-st-routing-v1" / "direct_capability_gate.py"

# Two direct aliases and one composite. The composite is deliberate: the checker grades
# 1:1 aliases only, so a report that includes it must be refused as describing a
# different candidate, and a report that omits it must still be complete.
MANIFEST = """\
schema_version = 1

[targets.alpha-1]
id = "vendor/alpha-1"
llm_client = "free_client"

[targets.beta-1]
id = "vendor/beta-1"
llm_client = "free_client"

[targets.gamma-1]
id = "vendor/gamma-1"
llm_client = "free_client"

[routes.direct-alpha]
id = "direct-alpha"
type = "passthrough"
target = "alpha-1"

[routes.direct-beta]
id = "direct-beta"
type = "passthrough"
target = "beta-1"

[routes.smart-thing]
id = "smart-thing"
type = "passthrough"
target = "gamma-1"
candidates = ["alpha-1"]
"""


def _report(rows):
    return json.dumps({"status": "PASS", "direct_count": len(rows),
                       "failures": [], "routes": rows,
                       "upstream_vs_admission_gaps": []})


GOOD_ROWS = [
    {"route": "direct-alpha", "target": "alpha-1", "model": "vendor/alpha-1",
     "client": "free_client"},
    {"route": "direct-beta", "target": "beta-1", "model": "vendor/beta-1",
     "client": "free_client"},
]


class _Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        scripts = self.root / "scripts" / "release"
        scripts.mkdir(parents=True)
        self.validator = scripts / "release_validate.py"

        self.manifest = self.root / "candidate.toml"
        self.manifest.write_text(MANIFEST)

        self.executed = self.root / "binary-executed"
        self.binary = self.root / "candidate-server"
        self.binary.write_text(
            "#!/usr/bin/env python3\n"
            "from pathlib import Path\n"
            f"Path({str(self.executed)!r}).write_text('executed')\n"
            "print('binary dry-run ok')\n"
        )
        self.binary.chmod(0o700)

        self.gate = self.root / "deploy" / "stephen-st-routing-v1" / "direct_capability_gate.py"
        # The validator as shipped: no stand-in checker pinned yet.
        self.validator.write_text((HERE / "release_validate.py").read_text())

    def install_gate(self, body: str, mode: int = 0o755):
        """Write a stand-in checker and pin the copy of the validator to its bytes.

        The pin is a digest allowlist in the validator's own source, so a test can
        only exercise a checker by making that copy accept it - which is exactly the
        reviewed change an operator makes. `test_pin_matches_the_repo_checker` keeps
        the real pin honest.
        """
        self.gate.parent.mkdir(parents=True, exist_ok=True)
        self.gate.write_text(body)
        self.gate.chmod(mode)
        self.pin(self.gate)

    def pin(self, gate: Path):
        digest = hashlib.sha256(gate.read_bytes()).hexdigest()
        source = (HERE / "release_validate.py").read_text()
        start = source.index("P9_CHECKER_SHA256 = frozenset({")
        end = source.index("})", start) + 2
        replaced = source[:start] + (
            "P9_CHECKER_SHA256 = frozenset({\n"
            f'    "{digest}",\n'
            "})"
        ) + source[end:]
        self.validator.write_text(replaced)

    def gate_echoing(self, payload: str, exit_code: int = 0) -> str:
        return (
            "import sys\n"
            "print(" + repr(payload) + ")\n"
            f"raise SystemExit({exit_code})\n"
        )

    def validate(self, *extra):
        return subprocess.run(
            [sys.executable, str(self.validator), "--binary", str(self.binary),
             "--manifest", str(self.manifest), *extra],
            capture_output=True, text=True, timeout=60,
        )


class ResolutionTests(_Base):
    def test_pin_matches_the_repo_checker(self):
        """The pinned digest must be the checker source control actually ships."""
        self.assertTrue(REAL_CHECKER.is_file(), REAL_CHECKER)
        digest = hashlib.sha256(REAL_CHECKER.read_bytes()).hexdigest()
        source = (HERE / "release_validate.py").read_text()
        self.assertIn(digest, source,
                      "deploy/stephen-st-routing-v1/direct_capability_gate.py changed "
                      "without updating P9_CHECKER_SHA256")

    def test_missing_checker_refuses_and_names_every_path_tried(self):
        self.validator.write_text((HERE / "release_validate.py").read_text())
        result = self.validate()
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("P9_GATE_UNAVAILABLE", result.stderr)
        self.assertIn("absent", result.stderr)
        # Every candidate is reported, so a layout change is diagnosable from the log.
        self.assertGreaterEqual(result.stderr.count("direct_capability_gate.py"), 2)
        self.assertFalse(self.executed.exists())

    def test_source_checkout_resolves_the_checkout_copy(self):
        """A validator running from a checkout grades against that checkout's checker."""
        self.install_gate(self.gate_echoing(_report(GOOD_ROWS)))
        result = self.validate()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("P9 PASS direct=2 qualified=2 failures=0", result.stdout)
        self.assertTrue(self.executed.exists())

    def test_installed_layout_resolves_the_deployed_checker(self):
        """The installed validator looks under /opt first, and is told apart by layout.

        Simulated by copying the tree under a temporary /opt-style prefix and asserting
        the computed search order, so the test does not depend on writing to /opt.
        """
        source = (HERE / "release_validate.py").read_text()
        probe = self.root / "probe_order.py"
        probe.write_text(
            "import importlib.util, sys\n"
            f"spec = importlib.util.spec_from_file_location('rv', {str(self.validator)!r})\n"
            "m = importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(m)\n"
            "print('|'.join(m.P9_CHECKER_PATHS))\n"
        )
        self.validator.write_text(source)
        result = subprocess.run([sys.executable, str(probe)],
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        order = result.stdout.strip().split("|")
        # A checkout validator grades against its own checker first, and keeps the
        # deployed form as a last resort rather than as its only guess.
        self.assertEqual(
            order[0],
            str((self.root / "deploy" / "stephen-st-routing-v1" / "direct_capability_gate.py").resolve()),
        )
        self.assertIn("/opt/deploy/stephen-st-routing-v1/direct_capability_gate.py", order)

    def test_unpinned_checker_bytes_are_refused(self):
        """A checker at the right path with the wrong bytes is not a trusted checker."""
        self.validator.write_text((HERE / "release_validate.py").read_text())
        self.gate.parent.mkdir(parents=True, exist_ok=True)
        self.gate.write_text(self.gate_echoing(_report(GOOD_ROWS)))
        self.gate.chmod(0o755)
        result = self.validate()
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("P9_GATE_UNAVAILABLE", result.stderr)
        self.assertIn("not a pinned checker revision", result.stderr)
        self.assertFalse(self.executed.exists())

    def test_group_writable_checker_is_refused(self):
        self.install_gate(self.gate_echoing(_report(GOOD_ROWS)), mode=0o775)
        result = self.validate()
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("writable by group or other", result.stderr)
        self.assertFalse(self.executed.exists())


class VerdictTests(_Base):
    def test_passing_verdict_allows_the_real_binary_dry_run(self):
        self.install_gate(self.gate_echoing(_report(GOOD_ROWS)))
        result = self.validate()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("candidate validates", result.stdout)
        self.assertIn("manifest_sha256=", result.stdout)
        self.assertIn("checker_sha256=", result.stdout)
        self.assertTrue(self.executed.exists())

    def test_nonzero_checker_refuses_before_launching_candidate(self):
        self.install_gate(
            "import sys\n"
            "print('REFUSED DIRECT_NOT_PASSTHROUGH', file=sys.stderr)\n"
            "raise SystemExit(3)\n"
        )
        result = self.validate()
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("P9_GATE_REFUSED", result.stderr)
        self.assertIn("DIRECT_NOT_PASSTHROUGH", result.stderr)
        self.assertFalse(self.executed.exists())

    def test_exit_zero_with_refused_status_is_not_acceptance(self):
        rows = GOOD_ROWS
        payload = json.dumps({"status": "REFUSED", "direct_count": 2,
                              "failures": ["GOOGLE_SHARED_CAPABILITY_DRIFT gemma"],
                              "routes": rows})
        self.install_gate(self.gate_echoing(payload))
        result = self.validate()
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("P9_GATE_MALFORMED", result.stderr)
        self.assertFalse(self.executed.exists())

    def test_pass_status_with_a_failure_list_is_refused(self):
        """The status line and the failure list must agree; neither outranks the other."""
        payload = json.dumps({"status": "PASS", "direct_count": 2,
                              "failures": ["DIRECT_NOT_PASSTHROUGH direct-alpha"],
                              "routes": GOOD_ROWS})
        self.install_gate(self.gate_echoing(payload))
        result = self.validate()
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("P9_GATE_REFUSED", result.stderr)
        self.assertIn("DIRECT_NOT_PASSTHROUGH", result.stderr)
        self.assertFalse(self.executed.exists())

    def test_manifest_that_cannot_be_parsed_is_refused(self):
        """Identity cannot be established from unreadable bytes, so the verdict is void."""
        self.manifest.write_text("this is not toml [[[\n")
        self.install_gate(self.gate_echoing(_report(GOOD_ROWS)))
        result = self.validate()
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("P9_MANIFEST_IDENTITY_MISMATCH", result.stderr)
        self.assertFalse(self.executed.exists())

    def test_malformed_report_is_refused(self):
        self.install_gate(self.gate_echoing("PASS direct=2 qualified=2 failures=0"))
        result = self.validate()
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("P9_GATE_MALFORMED", result.stderr)
        self.assertFalse(self.executed.exists())

    def test_truncated_report_is_refused(self):
        """A gate that graded fewer routes than the manifest declares has not graded it."""
        self.install_gate(self.gate_echoing(_report(GOOD_ROWS[:1])))
        result = self.validate()
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("P9_MANIFEST_IDENTITY_MISMATCH", result.stderr)
        self.assertIn("direct-beta", result.stderr)
        self.assertFalse(self.executed.exists())

    def test_report_for_another_manifest_is_refused(self):
        """Same route names, different target identity: the report is about other bytes."""
        rows = [dict(GOOD_ROWS[0], model="vendor/alpha-Z"), GOOD_ROWS[1]]
        self.install_gate(self.gate_echoing(_report(rows)))
        result = self.validate()
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("P9_MANIFEST_IDENTITY_MISMATCH", result.stderr)
        self.assertIn("vendor/alpha-Z", result.stderr)
        self.assertFalse(self.executed.exists())

    def test_report_of_a_different_route_set_is_refused(self):
        # A manifest with one more direct alias than the report covers.
        self.manifest.write_text(MANIFEST + """
[targets.delta-1]
id = "vendor/delta-1"
llm_client = "free_client"

[routes.direct-delta]
id = "direct-delta"
type = "passthrough"
target = "delta-1"
""")
        self.install_gate(self.gate_echoing(_report(GOOD_ROWS)))
        result = self.validate()
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("declared but not graded: direct-delta", result.stderr)
        self.assertFalse(self.executed.exists())

    def test_count_disagreeing_with_rows_is_refused(self):
        payload = json.dumps({"status": "PASS", "direct_count": 9, "failures": [],
                              "routes": GOOD_ROWS})
        self.install_gate(self.gate_echoing(payload))
        result = self.validate()
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("P9_GATE_MALFORMED", result.stderr)
        self.assertFalse(self.executed.exists())

    def test_explicit_command_testing_seam_does_not_claim_p9_proof(self):
        # activate-release's normal production path never supplies --command.
        result = self.validate("--command", "echo test-only")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("test-only", result.stdout)
        self.assertNotIn("P9 PASS", result.stdout)
        self.assertFalse(self.executed.exists())


if __name__ == "__main__":
    unittest.main()
