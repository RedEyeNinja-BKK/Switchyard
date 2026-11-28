#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Offline V7.98 corrector/generator (P11 final corrections).

Pure offline transform: reviewed V7.98 blob in, corrected candidate out.
"""
import argparse
import json
import re

FAMILIES = [
    ("A", "ComfyNinja", "comfyninja/qwen3[.]8-27b-q4-mtp|comfyninja/qwen3[.]8-27b-q4-mtp-vl"),
    ("B", "DeepSeek", "deepseek-flash|deepseek-v4-pro"),
    ("C", "HTPC / local", "Qwen3[.]5-9B-MTP|Qwen3[.]5-9B-Q4_0|jev-style-0[.]8b-decision-v3|laya-rl-agent"),
    ("D", "OpenRouter free pool", "baidu/ernie-5[.]5-flash-preview:free|inclusionai/ling-3[.]2-flash:free|deepseek/deepseek-v4[.]1-flash|deepseek/deepseek-v4[.]1-flash-1017|dots-studio/dots-3-note-preview:free|google/gemma-4-26b-a4b-it:free|google/gemma-4-31b-it:free|liquid/lfm-2[.]5-2[.]6b:free|nvidia/nemotron-3[.]5-super-vl:free|qwen/qwen3[.]7-flash|qwen/qwen3[.]8-flash|recursive/flint-v1[.]0:free|respan/span-01-lite|z-ai/glm-5[.]3-flash|openrouter/free|tencent/hy4-flash|nvidia/nemotron-3-ultra-550b-a55b:free"),
    ("E", "ThaiLLM", "openthaigpt-thaillm-8b-instruct-v7[.]2|qwen3[.]5-9b|qwen3[.]6-35b-a3b|thalle-0[.]2-thaillm-8b-fa|typhoon-s-thaillm-8b-instruct"),
    ("F", "OpenAI", "gpt-6-luna|gpt-6-sol"),
    ("G", "Google", "gemini-.+|gemma-[0-9]+-.+"),
]
_ALL_FAMILY_REGEX = "|".join(f[2] for f in FAMILIES)
NEUTRAL = "#64748B"

def _flatten(panels):
    flat = {}
    def walk(ps):
        for p in ps:
            flat[p['id']] = p
            if 'panels' in p: walk(p['panels'])
    walk(panels)
    return flat

def _strip_retired_lines(content):
    out = []
    for line in content.splitlines():
        if re.match(r'^\| `?(vc-paid-core|VC_Paid|smart-nt-bounded-trial)`? \|', line):
            continue
        out.append(line)
    return '\n'.join(out)

def _renumber(content):
    out = []
    n = 0
    for line in content.splitlines():
        m = re.match(r'^\| (\d+) \| (.*)$', line)
        if m:
            n += 1
            out.append(f'| {n} | {m.group(2)}')
        else:
            out.append(line)
    return '\n'.join(out)

def correct(doc):
    doc['title'] = 'SWITCHYARD DASHBOARD V7.98'
    doc['version'] = 98
    flat = _flatten(doc['panels'])

    # 1. 9006
    p = flat[9006]
    p['title'] = 'MODEL-FAMILY TRAFFIC (native model labels, heuristic)'
    p['description'] = 'Request rate grouped into model families by the native `model` label. Families are heuristic and mutually exclusive; an `unclassified` series carries traffic matching no named family so the sum reconciles to all-model traffic. Legends name the family only: a shared native model id cannot be attributed to an owner vs Stephen credential from this metric. Unit requests/s.'
    p['fieldConfig']['defaults']['unit'] = 'reqps'
    targets = []
    for rid, name, rx in FAMILIES:
        targets.append({"datasource": {"type": "prometheus", "uid": "df7wrp11a9hq8c"}, "expr": f'sum(rate(switchyard_requests_total{{job="switchyard-v7",model=~"{rx}"}}[5m]))', "legendFormat": name, "refId": rid})
    targets.append({"datasource": {"type": "prometheus", "uid": "df7wrp11a9hq8c"}, "expr": f'sum(rate(switchyard_requests_total{{job="switchyard-v7",model!~"{_ALL_FAMILY_REGEX}"}}[5m]))', "legendFormat": "unclassified", "refId": "H"})
    p['targets'] = targets
    p['fieldConfig']['overrides'] = [{"matcher": {"id": "byName", "options": "unclassified"}, "properties": [{"id": "color", "value": {"mode": "fixed", "fixedColor": NEUTRAL}}]}]

    # 2. 13003
    p = flat[13003]
    p['title'] = 'UPSTREAM HTTP CODES (code / outcome, 24h)'
    p['description'] = 'Estimated upstream attempts in 24h by HTTP code and outcome. Instant table: columns HTTP code / Outcome / Estimated attempts (24h), count descending. The Outcome cell is coloured per value; a 429 may be a daily allowance or a throttle.'
    p['type'] = 'table'
    p['targets'][0]['instant'] = True
    p['fieldConfig']['overrides'] = [{"matcher": {"id": "byName", "options": "Outcome"}, "properties": [{"id": "custom.cellOptions", "value": {"mode": "color-background"}}, {"id": "mappings", "value": [{"type": "value", "options": {"ok": {"color": "#299C46", "text": "ok", "index": 0}, "error": {"color": "#D44A4A", "text": "error", "index": 1}, "other_error": {"color": "#D44A4A", "text": "other_error", "index": 2}, "retryable_error": {"color": "#E08B26", "text": "retryable_error", "index": 3}, "client_disconnected": {"color": "#8B929A", "text": "client_disconnected", "index": 4}}}]}]}]
    p['options'] = {"showHeader": True}

    # 3. 9110
    p = flat[9110]
    p['title'] = 'STEPHEN CASCADE CHILD OUTCOMES (rolling 24h)'
    p['description'] = 'Stephen cascade child outcomes in 24h (parent_route=ST_Route). Legend is the compound `child_route · outcome`; colour is applied by anchored byRegexp on that compound legend.'
    for t in p['targets']:
        t['expr'] = t['expr'].replace('parent_route=~"ST_Route|ST_Route"', 'parent_route="ST_Route"')
        t['legendFormat'] = '{{child_route}} · {{outcome}}'
    p['fieldConfig']['defaults']['color'] = {'mode': 'fixed', 'fixedColor': NEUTRAL}
    p['fieldConfig']['overrides'] = [
        {"matcher": {"id": "byRegexp", "options": "^.+ · attempt_failed$"}, "properties": [{"id": "color", "value": {"mode": "fixed", "fixedColor": "#D44A4A"}}]},
        {"matcher": {"id": "byRegexp", "options": "^.+ · skipped_ineligible$"}, "properties": [{"id": "color", "value": {"mode": "fixed", "fixedColor": "#8B929A"}}]},
        {"matcher": {"id": "byRegexp", "options": "^.+ · selected$"}, "properties": [{"id": "color", "value": {"mode": "fixed", "fixedColor": NEUTRAL}}]},
    ]

    # 4. Reference panels
    c = _strip_retired_lines(flat[12001]['options']['content'])
    c = c.replace('and legacy paths are untouched: the direct route remains, `vc-paid-core` keeps it as the capable target, and the `VC_Paid` composite keeps `vc-paid-core` as its first child. The 54 rows are still 54, because the ',
                  'and legacy paths are untouched. ')
    c = _renumber(c)
    flat[12001]['options']['content'] = c

    c = _strip_retired_lines(flat[12002]['options']['content'])
    c = c.replace('20 advertised non-direct routes, 3 composites', '19 advertised non-direct routes, 1 composite')
    c = c.replace('`ead40bb6c05aed26…`, 136,493 B', '`18b377cd465862f3…`, 133,736 B')
    c = c.replace('**75 of 75 advertised**', '**74 of 74 advertised**')
    flat[12002]['options']['content'] = c

    flat[12003]['options']['content'] = (
        "**TYPED OPERATIONS - decisions / embed / rerank (separate from the 74 chat routes).**\n\n"
        "| Capability | Target | Format | Declared limits |\n|---|---|---|---|\n"
        "| `smart-decision` (composition) | children: `smart-gemini-decision`, `smartlocal-decision`, `smartfree-decision` | capability_first | declared order |\n"
        "| `smart-gemini-decision` | direct-google-gemini-decision-1 | google decisions | /v1beta/decisions |\n"
        "| `smartlocal-decision` | jev-style / laya (HTPC) | typed | /v1/systemone |\n"
        "| `smartfree-decision` | span-01-lite (Owner OR) | typed | /api/alpha/decisions |\n"
        "| `ST_Decisions` | span-01-lite (Stephen OR) | typed | /v1/decisions |\n\n"
        "**5 physical decision targets:** Gemini Decision 1, JEV Style, Laya, owner SPAN-01 Lite, Stephen SPAN-01 Lite. "
        "Embedding (`smartlocal/free/ST_Embed`) and reranking (`smartlocal/free/ST_Rerank`) lanes are unchanged; owner and Stephen credentials are never pooled."
    )

    c = _strip_retired_lines(flat[12004]['options']['content'])
    c = c.replace('`9069fe66…`', '`18b377cd…`').replace('Of the 54 direct routes', 'Of the 55 direct routes')
    c = c.replace('(`ST_Route`, `VC_Paid`, `ST_Rerank`, `smartfree-rerank`, `smartlocal-rerank`)',
                  '(`ST_Route`, `ST_Rerank`, `smartfree-rerank`, `smartlocal-rerank`)')
    flat[12004]['options']['content'] = c

    c = flat[12006]['options']['content']
    c = c.replace('`VC_Paid` is not a cost/quality judge.', 'No composite is a cost/quality judge.')
    flat[12006]['options']['content'] = c

    flat[12007]['options']['content'] = (
        "**DEPLOYMENT SNAPSHOT - live P11 record (read 2026-11-28, sha256sum-verified).**\n\n"
        "| Item | Recorded evidence |\n|---|---|\n"
        "| CURRENT (deployed) | `20261128T002506Z-6a8209f97101563b-18b377cd` |\n"
        "| Release action / label | `deploy` / `p11-functional-cleanup` |\n"
        "| Binary sha256 | `6a8209f97101563b58944f8614001fd0b057247da21a3daf57624ff798a22e82` (31,281,480 B) |\n"
        "| Manifest sha256 | `18b377cd465862f3183fcb10cf01300659ec57ee2925735d7408d5ffd7e8c15e` (133,736 B installed routes.toml) |\n"
        "| Capabilities sha256 | `987dd039ce618d5a0d124bd5607498b3c8d6845488af0d593fb3b3f7839f77b9` (25,499 B) |\n"
        "| Rollback target | `20261127T223854Z-6a8209f97101563b-ead40bb6` (P10, protected) |\n"
        "| LAST-KNOWN-GOOD | `20261126T061331Z-65757ec7aaa51166-737a6887` (not promoted) |\n"
        "| Running service | PID 873638, NRestarts=0, /v1/models 74 of 74 |\n"
        "| Basis | config-only on qualified binary 6a8209f9; removed vc-paid-core, VC_Paid, smart-nt-bounded-trial; google-decision reconciled into smart-gemini-decision |\n\n"
        "P11 is a functional-cleanup release: binary unchanged from P10; only manifest bytes differ. No rebuild, no LKG promotion, no provider traffic."
    )
    return doc

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    d = json.load(open(a.input))
    doc = correct(d['document'])
    with open(a.out, 'w') as f:
        json.dump(d, f, indent=2)
    print(f"wrote {a.out}")

if __name__ == '__main__':
    main()
