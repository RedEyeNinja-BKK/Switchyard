"""Offline tests: official model docs qualify exactly five pre-admitted tool targets."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import unittest
import apply_official_model_tool_caps as app

HERE = Path(__file__).resolve().parent

def fixture():
    lines = []
    for route, (target, model, wire) in app.ALLOW.items():
        client = "test_" + target
        lines += [f'[llm_clients."{client}"]', f'format = "{wire}"', "",
                  f'[targets."{target}"]', f'id = "{model}"',
                  f'llm_client = "{client}"', "",
                  f'[routes."{route}"]', f'target = "{target}"',
                  "tool_calling = true", ""]
    return "\n".join(lines)

def evidence():
    return json.loads((HERE / "official-model-tool-capabilities.json").read_text())

class OfficialModelDocTests(unittest.TestCase):
    def test_five_exact_targets_are_qualified_without_route_mutation(self):
        text, added = app.apply(fixture(), evidence())
        self.assertEqual(len(added), 5)
        self.assertEqual(text.count("tool_calling = true"), 10)
        again, same = app.apply(text, evidence())
        self.assertEqual(again, text)
        self.assertEqual(same, [])

    def test_an_extra_route_is_refused(self):
        data = evidence()
        data["capabilities"].append(dict(data["capabilities"][0], route="direct-extra"))
        with self.assertRaisesRegex(app.Refused, "exactly five"):
            app.apply(fixture(), data)

    def test_wrong_model_card_is_refused(self):
        data = evidence()
        data["capabilities"][0]["source"] = "https://example.invalid"
        with self.assertRaisesRegex(app.Refused, "evidence/model-card mismatch"):
            app.apply(fixture(), data)

    def test_owner_route_cannot_claim_another_model(self):
        before = fixture().replace('id = "gemini-3.8-flash"', 'id = "not-the-same-model"', 1)
        with self.assertRaisesRegex(app.Refused, "model or readiness mismatch"):
            app.apply(before, evidence())

    def test_refused_target_is_not_overwritten(self):
        before = fixture().replace('id = "gemini-3.8-flash"', 'id = "gemini-3.8-flash"\ntool_calling = false', 1)
        with self.assertRaisesRegex(app.Refused, "refuses to overwrite"):
            app.apply(before, evidence())

    def test_wrong_transport_is_refused(self):
        before = fixture().replace('format = "openai_chat"', 'format = "openai_responses"', 1)
        with self.assertRaisesRegex(app.Refused, "wire mismatch"):
            app.apply(before, evidence())

    def test_route_must_already_admit_tools(self):
        before = fixture().replace('tool_calling = true', 'tool_calling = false', 1)
        with self.assertRaisesRegex(app.Refused, "not a tool-enabled"):
            app.apply(before, evidence())

if __name__ == "__main__":
    unittest.main()
