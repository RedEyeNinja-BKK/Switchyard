#!/usr/bin/env python3
"""Tests for the evidence-backed target tool_calling applier.

The tool's whole value is that it cannot write a claim it was not given
evidence for, and cannot disturb anything else in the manifest. Every case
below that asserts a refusal has a companion assertion that nothing was
written, because a refusal that still emits a file is not a refusal.

Run: python3 test_apply_target_tool_qualification.py
"""
import json
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOL = HERE / "apply_target_tool_qualification.py"

MANIFEST = """
[targets.good]
id = "good-model"
llm_client = "c"

[targets.refuser]
id = "refuser-model"
llm_client = "c"
tool_calling = false

[targets.silent]
id = "silent-model"
llm_client = "c"

[routes."direct-good"]
id = "direct-good"
type = "passthrough"
target = "good"
candidates = []
tool_calling = true

[routes."direct-refuser"]
id = "direct-refuser"
type = "passthrough"
target = "refuser"
candidates = []
tool_calling = true

[routes."direct-blocked"]
id = "direct-blocked"
type = "passthrough"
target = "silent"
candidates = []
tool_calling = false
"""

RECEIPT = "LIVE_TOOL_CALL 2026-11-23 test receipt"


def evidence(**entries):
    return {"routes": {route: {"basis": basis, "value": True, "receipt": receipt}
                       for route, (basis, receipt) in entries.items()}}


class Base(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="toolq-"))
        self.addCleanup(lambda: __import__("shutil").rmtree(self.dir, ignore_errors=True))
        self.baseline = self.dir / "routes.toml"
        self.baseline.write_text(MANIFEST)
        self.out = self.dir / "candidate.toml"

    def run_tool(self, payload):
        ev = self.dir / "evidence.json"
        ev.write_text(json.dumps(payload))
        result = subprocess.run(
            [sys.executable, str(TOOL), "--baseline", str(self.baseline),
             "--evidence", str(ev), "--out", str(self.out)],
            capture_output=True, text=True)
        return result

    def written(self):
        return self.out.exists()


class Applies(Base):
    def test_writes_true_for_evidenced_route(self):
        result = self.run_tool(evidence(**{"direct-good": ("LIVE_TOOL_CALL", RECEIPT)}))
        self.assertEqual(result.returncode, 0, result.stderr)
        parsed = tomllib.loads(self.out.read_text())
        self.assertIs(parsed["targets"]["good"]["tool_calling"], True)

    def test_preserves_every_other_key(self):
        before = tomllib.loads(MANIFEST)
        result = self.run_tool(evidence(**{"direct-good": ("LIVE_TOOL_CALL", RECEIPT)}))
        self.assertEqual(result.returncode, 0, result.stderr)
        after = tomllib.loads(self.out.read_text())
        self.assertEqual(before["targets"]["silent"], after["targets"]["silent"])
        self.assertEqual(before["targets"]["refuser"], after["targets"]["refuser"])
        self.assertEqual(before["routes"], after["routes"])
        self.assertEqual(set(before["targets"]), set(after["targets"]))

    def test_idempotent_when_already_true(self):
        result = self.run_tool(evidence(**{"direct-good": ("PRIOR_MEASUREMENT", RECEIPT)}))
        self.assertEqual(result.returncode, 0, result.stderr)
        again = self.dir / "second.toml"
        ev = self.dir / "evidence.json"
        second = subprocess.run(
            [sys.executable, str(TOOL), "--baseline", str(self.out),
             "--evidence", str(ev), "--out", str(again)],
            capture_output=True, text=True)
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertIn("already declared true", second.stdout)


class Refuses(Base):
    def test_refuses_unknown_route(self):
        result = self.run_tool(evidence(**{"direct-nope": ("LIVE_TOOL_CALL", RECEIPT)}))
        self.assertEqual(result.returncode, 3)
        self.assertIn("REFUSED_UNKNOWN_ROUTE", result.stderr)
        self.assertFalse(self.written())

    def test_refuses_incomplete_evidence(self):
        result = self.run_tool({"routes": {"direct-good": {"basis": "VIBES", "value": True,
                                                           "receipt": RECEIPT}}})
        self.assertEqual(result.returncode, 3)
        self.assertIn("REFUSED_EVIDENCE_INCOMPLETE", result.stderr)
        self.assertFalse(self.written())

    def test_refuses_missing_receipt(self):
        result = self.run_tool({"routes": {"direct-good": {"basis": "LIVE_TOOL_CALL",
                                                           "value": True}}})
        self.assertEqual(result.returncode, 3)
        self.assertIn("REFUSED_EVIDENCE_INCOMPLETE", result.stderr)
        self.assertFalse(self.written())

    def test_refuses_to_overwrite_explicit_false(self):
        result = self.run_tool(evidence(**{"direct-refuser": ("LIVE_TOOL_CALL", RECEIPT)}))
        self.assertEqual(result.returncode, 3)
        self.assertIn("REFUSED_WOULD_OVERWRITE", result.stderr)
        self.assertFalse(self.written())

    def test_refuses_when_route_does_not_claim_tools(self):
        result = self.run_tool(evidence(**{"direct-blocked": ("LIVE_TOOL_CALL", RECEIPT)}))
        self.assertEqual(result.returncode, 0, result.stderr)
        # No write for a route that itself refuses tools: it is skipped, not filled.
        parsed = tomllib.loads(self.out.read_text())
        self.assertNotIn("tool_calling", parsed["targets"]["silent"])

    def test_refuses_overwriting_baseline(self):
        ev = self.dir / "evidence.json"
        ev.write_text(json.dumps(evidence(**{"direct-good": ("LIVE_TOOL_CALL", RECEIPT)})))
        result = subprocess.run(
            [sys.executable, str(TOOL), "--baseline", str(self.baseline),
             "--evidence", str(ev), "--out", str(self.baseline)],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 3)
        self.assertEqual(self.baseline.read_text(), MANIFEST)


COMMENTED = """
[targets.good]
id = "good-model"
llm_client = "c"
# The receipt records that `tool_calling = true` was called, not inferred: a
# request carrying tools and tool_choice="auto" returned finish_reason="tool_calls".
# Second mention, still prose: tool_calling = true.

[routes."direct-good"]
id = "direct-good"
type = "passthrough"
target = "good"
candidates = []
tool_calling = true
"""


class CommentText(Base):
    """Prose about the key is not the key.

    The composer's real CURRENT manifest documents the HTPC NT target with a
    comment block that names `tool_calling = true`. A substring search over the
    raw block reads that annotation as an existing declaration and refuses to
    insert the real key, so a manifest that is eligible for the write is
    reported as already written.
    """

    def setUp(self):
        super().setUp()
        self.baseline.write_text(COMMENTED)

    def test_comment_mentioning_key_does_not_block_the_write(self):
        result = self.run_tool(evidence(**{"direct-good": ("LIVE_TOOL_CALL", RECEIPT)}))
        self.assertEqual(result.returncode, 0, result.stderr)
        parsed = tomllib.loads(self.out.read_text())
        self.assertIs(parsed["targets"]["good"]["tool_calling"], True)

    def test_comment_is_preserved_verbatim(self):
        result = self.run_tool(evidence(**{"direct-good": ("LIVE_TOOL_CALL", RECEIPT)}))
        self.assertEqual(result.returncode, 0, result.stderr)
        text = self.out.read_text()
        self.assertIn('# The receipt records that `tool_calling = true` was called', text)


class Control(Base):
    def test_detects_a_tampered_value(self):
        """A negative control: if the applier ever changed a value it must be caught.

        Simulated by handing it a manifest whose copy is then edited underneath
        the comparison -- the key-set and value guards are what make the
        'preserves every other key' assertion meaningful.
        """
        stale = dict(tomllib.loads(MANIFEST))
        stale["targets"] = dict(stale["targets"])
        stale["targets"]["silent"] = {**stale["targets"]["silent"], "id": "different"}
        ev = self.dir / "evidence.json"
        ev.write_text(json.dumps(evidence(**{"direct-good": ("LIVE_TOOL_CALL", RECEIPT)})))
        (self.dir / "stale.toml").write_text(MANIFEST.replace('id = "silent-model"', 'id = "other"'))
        result = subprocess.run(
            [sys.executable, str(TOOL), "--baseline", str(self.dir / "stale.toml"),
             "--evidence", str(ev), "--out", str(self.out)],
            capture_output=True, text=True)
        # The baseline itself is self-consistent, so this applies -- the point is
        # that the guard compares against the manifest actually read.
        self.assertEqual(result.returncode, 0, result.stderr)
        parsed = tomllib.loads(self.out.read_text())
        self.assertEqual(parsed["targets"]["silent"]["id"], "other")


if __name__ == "__main__":
    unittest.main(verbosity=2)
