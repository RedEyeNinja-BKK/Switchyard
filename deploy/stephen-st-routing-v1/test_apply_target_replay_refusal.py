#!/usr/bin/env python3
"""A target may only be declared replay-incapable on a measured provider receipt.

The declaration removes a leg from every conversation that carries tool history, so
each test that removes one precondition must end in a refusal and write nothing.
"""
import importlib.util
import json
import tomllib
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "apply_target_replay_refusal", HERE / "apply_target_replay_refusal.py")
applier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(applier)

BASELINE = """\
[llm_clients.google]
format = "openai_chat"
base_url = "https://generativelanguage.googleapis.com/v1beta/openai"

[targets.google-lite]
tool_calling = true
id = "gemini-3.5-flash-lite"
llm_client = "google"
vision = false
groups = ["ST_Gemini_Free"]

[targets.other]
tool_calling = true
id = "other-model"
llm_client = "google"
groups = ["ST_OR_Free"]

[routes.ST_Free]
id = "ST_Free"
type = "passthrough"
target = "other"
candidates = []
pool_groups = ["ST_OR_Free", "ST_Gemini_Free"]
tool_calling = true
"""

PROVIDER_MESSAGE = (
    "Function call is missing a thought_signature in functionCall parts. "
    "This is required for tools to work correctly."
)


def evidence(**overrides):
    record = {
        "basis": "PROVIDER_REFUSES_REPLAY",
        "target": "google-lite",
        "route": "ST_Free",
        "replayed_tool_call": True,
        "receipt": {"status": 400, "message": PROVIDER_MESSAGE},
    }
    record.update(overrides)
    return record


class TheDeclarationLands(unittest.TestCase):
    def test_it_writes_one_line_beside_tool_calling_and_changes_nothing_else(self):
        candidate, target, route = applier.apply(BASELINE, evidence())
        self.assertEqual(target, "google-lite")
        self.assertEqual(route, "ST_Free")
        # The line sits immediately after the target's own tool_calling.
        lines = candidate.splitlines()
        index = lines.index('tool_calling = true')
        self.assertEqual(lines[index + 1], "tool_call_replay = false")

        before = tomllib.loads(BASELINE)
        after = tomllib.loads(candidate)
        self.assertIs(after["targets"]["google-lite"]["tool_call_replay"], False)
        after["targets"]["google-lite"].pop("tool_call_replay")
        self.assertEqual(before, after)

    def test_only_the_named_target_is_touched(self):
        candidate, _target, _route = applier.apply(BASELINE, evidence())
        after = tomllib.loads(candidate)
        self.assertNotIn("tool_call_replay", after["targets"]["other"])

    def test_a_direct_candidate_is_reachable_too(self):
        source = BASELINE.replace(
            'candidates = []', 'candidates = ["google-lite"]')
        candidate, _target, _route = applier.apply(source, evidence())
        self.assertIs(
            tomllib.loads(candidate)["targets"]["google-lite"]["tool_call_replay"], False)


class EveryMissingPreconditionRefuses(unittest.TestCase):
    def assertRefused(self, text, record, needle):
        with self.assertRaises(applier.Refused) as caught:
            applier.apply(text, record)
        self.assertIn(needle, str(caught.exception))

    def test_wrong_basis(self):
        self.assertRefused(BASELINE, evidence(basis="SOMETHING_ELSE"), "REFUSED_BASIS")

    def test_receipt_that_is_not_the_provider_400(self):
        self.assertRefused(
            BASELINE,
            evidence(receipt={"status": 429, "message": PROVIDER_MESSAGE}),
            "REFUSED_RECEIPT",
        )

    def test_message_without_the_provider_phrase(self):
        self.assertRefused(
            BASELINE,
            evidence(receipt={"status": 400, "message": "Provider returned error"}),
            "REFUSED_RECEIPT",
        )

    def test_receipt_of_a_definitional_call_is_not_a_replay(self):
        self.assertRefused(
            BASELINE, evidence(replayed_tool_call=False), "REFUSED_EVIDENCE")

    def test_unknown_target(self):
        self.assertRefused(
            BASELINE, evidence(target="nope"), "REFUSED_UNKNOWN_TARGET")

    def test_target_not_reachable_from_the_named_route(self):
        source = BASELINE.replace(
            'pool_groups = ["ST_OR_Free", "ST_Gemini_Free"]', 'pool_groups = ["ST_OR_Free"]'
        ).replace('candidates = []', 'candidates = ["other"]')
        self.assertRefused(source, evidence(), "REFUSED_UNREACHABLE")

    def test_an_unknown_route_is_a_refusal(self):
        self.assertRefused(BASELINE, evidence(route="nope"), "REFUSED_UNREACHABLE")

    def test_a_second_run_is_refused_rather_than_rewritten(self):
        once, _target, _route = applier.apply(BASELINE, evidence())
        self.assertRefused(once, evidence(), "REFUSED_ALREADY_DECLARED")

    def test_a_target_that_already_refuses_tools_is_a_no_effect(self):
        source = BASELINE.replace(
            "[targets.google-lite]\ntool_calling = true",
            "[targets.google-lite]\ntool_calling = false",
        )
        self.assertRefused(source, evidence(), "REFUSED_NO_EFFECT")

    def test_no_anchor_is_a_refusal_not_a_guess(self):
        source = BASELINE.replace(
            "[targets.google-lite]\ntool_calling = true\n",
            "[targets.google-lite]\n",
        )
        self.assertRefused(source, evidence(), "REFUSED_NO_ANCHOR")

    def test_a_collateral_edit_is_caught(self):
        original = applier.render
        try:
            def vandalise(text, target_name):
                return original(text, target_name).replace('id = "other-model"', 'id = "x"')

            applier.render = vandalise
            with self.assertRaises(applier.Refused) as caught:
                applier.apply(BASELINE, evidence())
            self.assertIn("REFUSED_COLLATERAL", str(caught.exception))
        finally:
            applier.render = original


class CliPaths(unittest.TestCase):
    def test_dry_run_writes_nothing(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "routes.toml").write_text(BASELINE, encoding="utf-8")
            (root / "evidence.json").write_text(json.dumps(evidence()), encoding="utf-8")
            code = applier.main([
                "--baseline", str(root / "routes.toml"),
                "--evidence", str(root / "evidence.json"),
                "--out", str(root / "out.toml"),
                "--dry-run",
            ])
            self.assertEqual(code, 0)
            self.assertFalse((root / "out.toml").exists())

    def test_a_refusal_exits_nonzero_and_writes_nothing(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "routes.toml").write_text(BASELINE, encoding="utf-8")
            (root / "evidence.json").write_text(
                json.dumps(evidence(basis="WRONG")), encoding="utf-8")
            code = applier.main([
                "--baseline", str(root / "routes.toml"),
                "--evidence", str(root / "evidence.json"),
                "--out", str(root / "out.toml"),
            ])
            self.assertEqual(code, 2)
            self.assertFalse((root / "out.toml").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
