#!/usr/bin/env python3
"""Offline regression gates for native Google 2.5 removal and Stephen direct projection.
Run: python -m unittest discover -s deploy/stephen-st-routing-v1/gemini-suite -p 'test_*.py'
No Google credential, HTTP request, service state or deployment is touched.
"""
import json
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import project
import reconcile_remove_native_gemini_25 as prune


class NativeGoogleContractTests(unittest.TestCase):
    def setUp(self):
        self.cat = project.catalogue()
        self.owner = project.profile("owner")
        self.stephen = project.profile("stephen")
        self.keys = {m["key"] for m in self.cat["models"] if m["lane"] == "chat"}
        self.retired = {"gemini-2.5-flash", "gemini-2.5-flash-lite"}

    def test_exact_eight_canonical_chat_models_and_no_retired(self):
        self.assertEqual(len(self.keys), 8)
        self.assertFalse(self.keys & self.retired)
        project.assert_removed_native_models_absent(self.cat, self.owner, self.stephen)

    def test_eight_owner_and_eight_stephen_direct_leaves_are_rendered(self):
        for which, profile in (("owner", self.owner), ("stephen", self.stephen)):
            leaf = tomllib.loads(project.render_owner_leaf_routes(which, self.cat, profile))
            routes = leaf["routes"]
            self.assertEqual(len(routes), 8)
            for model in self.cat["models"]:
                if model["lane"] != "chat":
                    continue
                name = project.route_name(which, model)
                self.assertIn(name, routes)
                self.assertEqual(routes[name]["target"], project.target_name(which, model))
            self.assertFalse(any("gemini-2.5-" in name for name in routes))

    def test_stephen_alias_parity_catches_missing_direct_route(self):
        manifest = "\n".join([
            project.render_targets("owner", self.cat, self.owner),
            project.render_targets("stephen", self.cat, self.stephen),
            project.render_owner_leaf_routes("owner", self.cat, self.owner),
            project.render_owner_leaf_routes("stephen", self.cat, self.stephen),
        ])
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "synthetic.toml"
            path.write_text(manifest)
            self.assertEqual(
                project.check_native_google_direct_routes(self.cat, self.stephen, path),
                [],
            )
            missing = project.route_name("stephen", next(
                m for m in self.cat["models"] if m["key"]=="gemini-3.5-flash-lite"
            ))
            fragment = tomllib.loads(manifest)
            # Build a missing-route mutation by deleting the exact raw block
            needle = f'[routes."{missing}"]'
            lines = manifest.splitlines(keepends=True)
            start = next(i for i, line in enumerate(lines) if line.strip() == needle)
            end = next((i for i in range(start+1,len(lines))
                        if lines[i].startswith("[routes.")),len(lines))
            path.write_text("".join(lines[:start]+lines[end:]))
            self.assertTrue(any("Stephen direct native aliases" in x
                                for x in project.check_native_google_direct_routes(
                                    self.cat, self.stephen, path)))

    def test_existing_model_capability_specs_have_all_eight(self):
        facts = json.loads((HERE/"canonical-google-model-capabilities-20261123.json").read_text())
        self.assertEqual(set(facts["models"]),self.keys)
        self.assertFalse(self.retired & set(facts["models"]))
        for model in facts["models"].values():
            self.assertIn("model_input",model)
            self.assertIn("model_output",model)
            self.assertIn("transport",model)
            self.assertIn("modality_source",model)

    def test_prune_only_explicit_four_routes_and_four_targets(self):
        routes = []
        targets = []
        for suffix in sorted(self.retired):
            for profile in ("owner","stephen"):
                route_prefix = "direct-google-" if profile == "owner" else "direct-google-stephen-"
                name = route_prefix+suffix
                target = ("google-" if profile == "owner" else "stephen-google-")+suffix
                routes.append(f'[routes."{name}"]\nid="{name}"\ntype="passthrough"\ntarget="{target}"\n')
                targets.append(f'[targets."{target}"]\nid="{suffix}"\nllm_client="google_gemini_{profile}"\n')
        routes.append('[routes.ST_Free]\nid="ST_Free"\n')
        targets.append('[targets.other]\nid="gemini-3.5-flash-lite"\nllm_client="google_gemini_owner"\n')
        text = "\n".join(routes+targets)
        result, removed = prune.remove_only(text)
        after = tomllib.loads(result)
        self.assertEqual(len(removed),8)
        self.assertEqual(set(after["routes"]),{"ST_Free"})
        self.assertEqual(set(after["targets"]),{"other"})
        self.assertNotIn("gemini-2.5-flash",result)

    def test_prune_refuses_any_live_non_direct_reference(self):
        old = []
        for suffix in sorted(self.retired):
            for profile in ("owner","stephen"):
                p = "direct-google-" if profile == "owner" else "direct-google-stephen-"
                t = ("google-" if profile == "owner" else "stephen-google-")+suffix
                old.append(f'[routes."{p+suffix}"]\nid="{p+suffix}"\ntarget="{t}"\n')
                old.append(f'[targets."{t}"]\nid="{suffix}"\nllm_client="google_gemini_{profile}"\n')
        old.append('[routes.some_other_route]\nid="some_other_route"\ntarget="stephen-google-gemini-2.5-flash"\n')
        with self.assertRaisesRegex(ValueError,"still referenced"):
            prune.remove_only("\n".join(old))


if __name__=="__main__":
    unittest.main()
