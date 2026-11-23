#!/usr/bin/env python3
"""Offline wiring tests: a real release must not skip the P9 contract.

Uses a copied validator and a stand-in gate to exercise invocation/refusal.
The direct_capability_gate's *content* is tested by its own negative suite.
No Switchyard service, credentials, or production state are involved.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent


class P9ReleaseValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        scripts = self.root / "scripts" / "release"
        scripts.mkdir(parents=True)
        self.validator = scripts / "release_validate.py"
        shutil.copy2(HERE / "release_validate.py", self.validator)

        self.manifest = self.root / "candidate.toml"
        self.manifest.write_text("candidate = true\n")
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

    def install_fake_p9(self, exit_code: int):
        self.gate.parent.mkdir(parents=True, exist_ok=True)
        self.gate.write_text(
            "import sys\n"
            "assert sys.argv[1] == '--manifest'\n"
            f"assert sys.argv[2] == {str(self.manifest)!r}\n"
            "print('named P9 route deficit' if "
            f"{exit_code} else 'PASS direct=53')\n"
            f"raise SystemExit({exit_code})\n"
        )

    def validate(self, *extra):
        return subprocess.run(
            [sys.executable, str(self.validator), "--binary", str(self.binary),
             "--manifest", str(self.manifest), *extra],
            capture_output=True, text=True, timeout=10,
        )

    def test_missing_p9_refuses_before_launching_candidate(self):
        result = self.validate()
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("P9_GATE_UNAVAILABLE", result.stderr)
        self.assertFalse(self.executed.exists())

    def test_failing_p9_refuses_before_launching_candidate(self):
        self.install_fake_p9(3)
        result = self.validate()
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("P9_GATE_REFUSED", result.stderr)
        self.assertIn("named P9 route deficit", result.stderr)
        self.assertFalse(self.executed.exists())

    def test_passing_p9_allows_real_binary_dry_run(self):
        self.install_fake_p9(0)
        result = self.validate()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("candidate validates", result.stdout)
        self.assertTrue(self.executed.exists())

    def test_explicit_command_testing_seam_does_not_claim_p9_proof(self):
        # activate-release's normal production path never supplies --command.
        result = self.validate("--command", "echo test-only")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("test-only", result.stdout)
        self.assertFalse(self.executed.exists())


if __name__ == "__main__":
    unittest.main()
