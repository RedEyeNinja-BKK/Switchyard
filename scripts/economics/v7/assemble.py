#!/usr/bin/env python3
"""Assemble gen98 dashboard JSON from the modified gen97 export + gen92 reference text panels."""
import json

d = json.load(open('/var/tmp/v7-gen98-candidate.json'))
doc = d['document']
doc['title'] = 'SWITCHYARD DASHBOARD V7.98'
doc['version'] = 98

flat = {}
def walk(ps):
    for p in ps:
        flat[p['id']] = p
        if 'panels' in p: walk(p['panels'])
walk(doc['panels'])

# Insert text panels from gen92 files
for pid in [12001, 12002, 12003, 12004, 12005, 12006, 12007]:
    if pid in flat:
        try:
            txt = open(f'/var/tmp/gen92-panel-{pid}.txt').read()
            flat[pid]['options']['content'] = txt
            print(f"Updated panel {pid} content from gen92 file ({len(txt)} chars)")
        except Exception as e:
            print(f"Could not read gen92 panel {pid}: {e}")

json.dump(d, open('/var/tmp/v7-gen98-export.json', 'w'), indent=2)
print("Wrote /var/tmp/v7-gen98-export.json successfully")
