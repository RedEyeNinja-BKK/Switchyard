"""Exact incident cooldown mutation tests. No live service, credentials or model calls."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import tomllib
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "apply_gemma31_cooldown_reset", HERE / "apply_gemma31_cooldown_reset.py")
fix = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fix)

BASE = """[llm_clients.google_gemma_31b]
format = "openai_chat"
failure_cooldown_ms = 300000
max_retries = 1

[llm_clients.google_gemma_26b]
format = "openai_chat"
failure_cooldown_ms = 5000

[routes.smart-judge]
terminal_policy = "availability"
"""

class Gemma31CooldownTests(unittest.TestCase):
    def test_exact_known_override_removes_only_one_field(self):
        result = tomllib.loads(fix.apply(BASE))
        self.assertNotIn("failure_cooldown_ms", result["llm_clients"]["google_gemma_31b"])
        self.assertEqual(result["llm_clients"]["google_gemma_26b"]["failure_cooldown_ms"], 5000)
        self.assertEqual(result["routes"]["smart-judge"]["terminal_policy"], "availability")
        self.assertEqual(result["llm_clients"]["google_gemma_31b"]["max_retries"], 1)

    def test_refuses_30000_instead_of_300000(self):
        with self.assertRaisesRegex(fix.Refused, "exact 300000ms"):
            fix.apply(BASE.replace("300000", "30000"))

    def test_refuses_wrong_client(self):
        with self.assertRaisesRegex(fix.Refused, "exact 300000ms"):
            fix.apply(BASE.replace("[llm_clients.google_gemma_31b]",
                                   "[llm_clients.other_model]"))

    def test_refuses_missing_override_or_repeat(self):
        with self.assertRaisesRegex(fix.Refused, "exact 300000ms"):
            fix.apply(BASE.replace("failure_cooldown_ms = 300000\n", ""))
        with self.assertRaises((fix.Refused, tomllib.TOMLDecodeError)):
            fix.apply(BASE.replace("failure_cooldown_ms = 300000",
                                   "failure_cooldown_ms = 300000\nfailure_cooldown_ms = 300000"))

if __name__ == "__main__":
    unittest.main()
