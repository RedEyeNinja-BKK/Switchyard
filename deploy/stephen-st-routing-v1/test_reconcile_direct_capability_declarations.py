#!/usr/bin/env python3
"""Proof the candidate normalizer never promotes unqualified support or rewrites routes."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import tomllib
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("reconcile", HERE / "reconcile_direct_capability_declarations.py")
repair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repair)


BASE = """
[llm_clients.openrouter]
format = "openai_responses"
api_key_env = "ROUTER"
[targets.vl]
id = "nvidia/nemotron-3.5-super-vl:free"
llm_client = "openrouter"
tool_calling = true
vision = true
video = true
image_carrier = "inline"
video_carrier = "inline"
[targets.text]
id = "text"
llm_client = "openrouter"
vision = false
tool_calling = false
[targets.unknown_tools]
id = "unknown"
llm_client = "openrouter"
vision = true
[targets.untouched]
id = "other"
llm_client = "openrouter"
vision = true
[targets.untouched.extra_body]
temperature = 0.5
[routes."direct-openrouter-vl"]
id = "direct-openrouter-vl"
type = "passthrough"
target = "vl"
candidates = []
tool_calling = true
vision = true
video = true
[routes."direct-openrouter-text"]
id = "direct-openrouter-text"
type = "passthrough"
target = "text"
candidates = []
[routes."direct-openrouter-unknown"]
id = "direct-openrouter-unknown"
type = "passthrough"
target = "unknown_tools"
candidates = []
vision = true
tool_calling = true
[routes.other]
id = "other"
type = "passthrough"
target = "untouched"
candidates = []
"""


class CandidateTests(unittest.TestCase):
    def test_inserts_only_safe_scope_and_does_not_affect_unrelated(self):
        after, edits = repair.apply(BASE)
        b, a = tomllib.loads(BASE), tomllib.loads(after)
        self.assertEqual(b["llm_clients"], a["llm_clients"])
        self.assertEqual(b["routes"]["other"], a["routes"]["other"])
        self.assertEqual(b["targets"]["untouched"], a["targets"]["untouched"])
        self.assertEqual(a["routes"]["direct-openrouter-vl"]["image_carrier"], "inline")
        self.assertEqual(a["routes"]["direct-openrouter-vl"]["video_carrier"], "inline")
        self.assertIs(a["routes"]["direct-openrouter-text"]["vision"], False)
        self.assertIs(a["routes"]["direct-openrouter-text"]["tool_calling"], False)
        self.assertIs(a["routes"]["direct-openrouter-text"]["video"], False)
        self.assertIs(a["targets"]["text"]["video"], False)
        self.assertIs(a["targets"]["unknown_tools"]["video"], False)
        self.assertNotIn("tool_calling", a["targets"]["unknown_tools"])
        self.assertGreater(len(edits), 2)

    def test_existing_disabled_is_not_turned_on(self):
        modified = BASE.replace('vision = true\nvideo = true\n[routes."direct-openrouter-text"]',
                                'vision = false\nvideo = true\n[routes."direct-openrouter-text"]')
        after, _ = repair.apply(modified)
        self.assertIs(tomllib.loads(after)["routes"]["direct-openrouter-vl"]["vision"], False)

    def test_source_capability_unknown_remains_a_named_blocker(self):
        after, _ = repair.apply(BASE)
        result = repair.gate.analyze(tomllib.loads(after))
        self.assertIn("TARGET_CAPABILITY_UNQUALIFIED", str(result["failures"]))

    def test_candidate_is_idempotent(self):
        after, _ = repair.apply(BASE)
        again, edits = repair.apply(after)
        self.assertEqual(again, after)
        self.assertEqual(edits, {})

    def test_hidden_fallback_refused(self):
        mutant = BASE.replace('target = "text"\ncandidates = []',
                              'target = "text"\ncandidates = ["vl"]')
        with self.assertRaises(repair.gate.ContractError):
            repair.apply(mutant)

    def test_reject_baseline_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "source.toml"
            file.write_text(BASE)
            exitcode = repair.main(["--baseline", str(file), "--out", str(file)])
            self.assertEqual(exitcode, 3)
            self.assertEqual(file.read_text(), BASE)


if __name__ == "__main__":
    unittest.main()
