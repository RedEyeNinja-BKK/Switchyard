#!/usr/bin/env python3
"""Apply owner-accepted official MODEL documentation to five P9 tool declarations.

This is a source-only, fail-closed TOML transformation. Vendor function-calling
documentation proves a MODEL-level feature; it does not prove API-key access,
remaining quota, an actual completed tool call, or successful deployment.
No network, credentials or release store are used.

Usage: python3 apply_official_model_tool_caps.py --baseline 5-candidate.toml
       --evidence official-model-tool-capabilities.json --out 6-model-cards.toml
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import tomllib

# Exact owner-authorized five, including transport and documented model.
ALLOW = {
    "direct-google-gemini-3.8-flash":
        ("direct-google-gemini-3.8-flash", "gemini-3.8-flash", "openai_chat"),
    "direct-google-stephen-gemini-3.8-flash":
        ("stephen-google-gemini-3.8-flash", "gemini-3.8-flash", "openai_chat"),
    "direct-openai-gpt-6-luna":
        ("openai-gpt-6-luna", "gpt-6-luna", "openai_responses"),
    "direct-openai-gpt-6-luna-nt":
        ("openai-gpt-6-luna-nt", "gpt-6-luna", "openai_responses"),
    "direct-openai-gpt-6-sol":
        ("openai-gpt-6-sol", "gpt-6-sol", "openai_responses"),
}
URLS = {
    "gemini-3.8-flash": "https://aistudio.google.com/docs/models/gemini-3.8-flash",
    "gpt-6-luna": "https://developers.openai.com/api/docs/models/gpt-6-luna",
    "gpt-6-sol": "https://developers.openai.com/api/docs/models/gpt-6-sol",
}

class Refused(ValueError):
    pass

def apply(text: str, evidence: dict) -> tuple[str, list[str]]:
    if evidence.get("policy") != "owner_authorized_official_model_docs_2026-11-24":
        raise Refused("missing exact owner model-documentation decision")
    records = evidence.get("capabilities")
    if not isinstance(records, list) or len(records) != len(ALLOW):
        raise Refused("expected exactly five documented capabilities")
    by_route = {record.get("route"): record for record in records}
    if len(by_route) != len(records) or set(by_route) != set(ALLOW):
        raise Refused("unapproved, duplicate or missing route")
    doc = tomllib.loads(text)
    routes, targets, clients = (doc.get("routes") or {}), (doc.get("targets") or {}), (doc.get("llm_clients") or {})
    lines = text.splitlines(keepends=True)
    headings = [(i, line.strip()) for i, line in enumerate(lines)
                if line.startswith("[") and line.rstrip().endswith("]")]
    spans = {h: (i, headings[n+1][0] if n+1 < len(headings) else len(lines))
             for n, (i, h) in enumerate(headings)}
    additions = []
    for route, (target_name, model, wire) in ALLOW.items():
        rec = by_route[route]
        if (rec.get("target"), rec.get("model"), rec.get("capability"),
            rec.get("value"), rec.get("evidence_class"), rec.get("source")) != (
            target_name, model, "tool_calling", True, "OFFICIAL_MODEL_DOC", URLS[model]):
            raise Refused(f"evidence/model-card mismatch for {route}")
        spec, target = routes.get(route), targets.get(target_name)
        if not isinstance(spec, dict) or not isinstance(target, dict):
            raise Refused(f"missing route or target: {route}")
        if spec.get("target") != target_name or spec.get("tool_calling") is not True:
            raise Refused(f"not a tool-enabled 1:1 direct route: {route}")
        if spec.get("candidates") or spec.get("terminal_fallback"):
            raise Refused(f"not 1:1: {route}")
        if target.get("id") != model or target.get("held") or not target.get("llm_client"):
            raise Refused(f"model or readiness mismatch: {route}")
        if (clients.get(target["llm_client"]) or {}).get("format") != wire:
            raise Refused(f"wire mismatch: {route}")
        if target.get("tool_calling") is True:
            continue
        if "tool_calling" in target:
            raise Refused(f"refuses to overwrite existing target flag: {target_name}")
        aliases = [f"[targets.{target_name}]", f'[targets."{target_name}"]']
        matched = [h for h in aliases if h in spans]
        if len(matched) != 1:
            raise Refused(f"ambiguous target heading: {target_name}")
        start, end = spans[matched[0]]
        if any(line.partition("=")[0].strip() == "tool_calling" for line in lines[start+1:end]):
            raise Refused(f"unexpected tool_calling line in {target_name}")
        additions.append((start+1, target_name))
    for pos, name in sorted(additions, reverse=True):
        lines.insert(pos, "tool_calling = true\n")
    output = "".join(lines)
    after = tomllib.loads(output)
    for section in set(doc) | set(after):
        if section != "targets" and after.get(section) != doc.get(section):
            raise Refused(f"unexpected change outside targets: {section}")
    if set(after["targets"]) != set(doc["targets"]):
        raise Refused("changed target names")
    added = {name for _, name in additions}
    for name, before in doc["targets"].items():
        expected = dict(before)
        if name in added:
            expected["tool_calling"] = True
        if after["targets"][name] != expected:
            raise Refused(f"unexpected target change: {name}")
    return output, sorted(added)

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--evidence", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    if a.baseline.resolve() == a.out.resolve():
        print("REFUSED: input and output are the same path", file=sys.stderr)
        return 3
    try:
        output, changed = apply(a.baseline.read_text(), json.loads(a.evidence.read_text()))
        a.out.write_text(output)
    except (OSError, ValueError, KeyError, tomllib.TOMLDecodeError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 3
    print(f"OFFICIAL_MODEL_DOC: {len(changed)} target flags qualified; {', '.join(changed)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
