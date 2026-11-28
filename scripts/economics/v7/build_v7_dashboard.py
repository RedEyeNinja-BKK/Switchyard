#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build SWITCHYARD DASHBOARD V7.98 from explicit versioned inputs.

Reads a base dashboard export JSON and a directory of reference panel text files,
applies the P11 functional cleanup panel modifications and semantic colour
conventions, and emits the V7.98 dashboard document.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from semantic_colors import SEMANTIC_COLORS, NEUTRAL_DEFAULT

def load_dashboard(export_path):
    with open(export_path) as f:
        return json.load(f)

def flatten_panels(panels):
    flat = {}
    def walk(ps):
        for p in ps:
            flat[p['id']] = p
            if 'panels' in p:
                walk(p['panels'])
    walk(panels)
    return flat

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--export', required=True, help='Path to base server export JSON')
    ap.add_argument('--ref-dir', required=True, help='Directory containing gen92-panel-*.txt')
    ap.add_argument('--out', required=True, help='Output JSON path')
    args = ap.parse_args(argv)

    d = load_dashboard(args.export)
    doc = d['document']
    doc['title'] = 'SWITCHYARD DASHBOARD V7.98'
    doc['version'] = 98
    flat = flatten_panels(doc['panels'])

    # Apply semantic colors to outcome panels
    for pid in [9, 14, 26, 34, 35, 9009, 91001, 91002]:
        if pid in flat:
            p = flat[pid]
            p['fieldConfig']['defaults']['color'] = {'mode': 'fixed', 'fixedColor': NEUTRAL_DEFAULT}
            p['fieldConfig']['overrides'] = []
            for name, col in SEMANTIC_COLORS.items():
                p['fieldConfig']['overrides'].append({
                    "matcher": {"id": "byName", "options": name},
                    "properties": [{"id": "color", "value": {"mode": "fixed", "fixedColor": col}}]
                })

    # Load reference panels from explicit ref-dir
    for pid in [12001, 12002, 12003, 12004, 12005, 12006, 12007]:
        ref_file = os.path.join(args.ref_dir, f'gen92-panel-{pid}.txt')
        if not os.path.exists(ref_file):
            raise FileNotFoundError(f"Missing required reference input: {ref_file}")
        with open(ref_file) as rf:
            flat[pid]['options']['content'] = rf.read()

    with open(args.out, 'w') as out:
        json.dump(d, out, indent=2)
    print(f"Wrote {args.out}")

if __name__ == '__main__':
    main()
