"""Mutation-style tests for the smart-judge fit_escalation applier.

No live service, credentials or model calls: every case runs against a literal manifest.
Each refusal is asserted by the reason it gives, because "it raised something" is also
satisfied by a crash.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import tomllib
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "apply_fit_escalation", HERE / "apply_fit_escalation.py"
)
applier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(applier)

BASE = """schema_version = 1

[llm_clients.luna]
format = "openai_responses"
base_url = "https://luna.test/v1"

[targets.luna]
id = "openai/gpt-6-luna-nt"
llm_client = "luna"

[targets.local]
id = "Qwen3.5-9B-MTP"
llm_client = "luna"

[routes.other]
id = "other"
type = "passthrough"
target = "luna"
terminal_fallback = [["local"]]
terminal_policy = "availability"

[routes.smart-judge]
id = "smart-judge"
type = "passthrough"
target = "luna"
candidates = []
selection = "measured"
terminal_fallback = [["openai-gpt-6-luna-nt"], ["htpc-qwen3_5-9b-mtp-nt"]]
terminal_policy = "availability"
context_window = 12307
tool_calling = true
reasoning = false
"""


class FitEscalationApplierTests(unittest.TestCase):
    def test_the_candidate_declares_the_option_and_nothing_else_moves(self):
        candidate = applier.apply(BASE)
        after = tomllib.loads(candidate)
        self.assertIs(after["routes"]["smart-judge"]["fit_escalation"], True)
        before = tomllib.loads(BASE)
        self.assertEqual(
            after["routes"]["other"], before["routes"]["other"], "another route changed"
        )
        self.assertEqual(after["targets"], before["targets"])
        self.assertEqual(after["llm_clients"], before["llm_clients"])

    def test_the_edit_is_exactly_one_added_line(self):
        candidate = applier.apply(BASE)
        self.assertEqual(
            len(candidate.splitlines()) - len(BASE.splitlines()), 1, "not a single-line add"
        )
        self.assertIn('terminal_policy = "availability"\nfit_escalation = true\n', candidate)

    def test_an_already_declared_manifest_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            applier.apply(BASE.replace("terminal_policy = \"availability\"\ncontext_window",
                                       "terminal_policy = \"availability\"\nfit_escalation = true\ncontext_window"))
        self.assertIn("already declares", str(caught.exception))

    def test_a_missing_anchor_is_refused_by_count(self):
        # The flat, one-tier declaration that preceded the three-tier release must not match:
        # silently escalating on a manifest whose tiers were shaped differently is exactly the
        # kind of edit this applier exists to prevent.
        flat = BASE.replace(
            'terminal_fallback = [["openai-gpt-6-luna-nt"], ["htpc-qwen3_5-9b-mtp-nt"]]',
            'terminal_fallback = ["openai-gpt-6-luna-nt"]',
        )
        with self.assertRaises(ValueError) as caught:
            applier.apply(flat)
        self.assertIn("matched 0 times", str(caught.exception))

    def test_a_doubled_anchor_is_refused(self):
        doubled = BASE + BASE[BASE.index("[routes.smart-judge]"):].replace(
            "[routes.smart-judge]", "[routes.smart-judge-copy]"
        )
        with self.assertRaises(ValueError) as caught:
            applier.apply(doubled)
        self.assertIn("matched 2 times", str(caught.exception))

    def test_the_option_lands_inside_the_route_table(self):
        candidate = applier.apply(BASE)
        smart = candidate[candidate.index("[routes.smart-judge]"):]
        declaration = smart.index("fit_escalation = true")
        self.assertLess(declaration, smart.index("[routes.", 1) if "[routes." in smart[1:] else len(smart))


if __name__ == "__main__":
    unittest.main()
