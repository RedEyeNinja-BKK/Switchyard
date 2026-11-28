import json
d=json.load(open('/var/tmp/v7-gen98-export.json'))
flat={}
def walk(ps):
    for p in ps:
        flat[p['id']]=p
        if 'panels' in p: walk(p['panels'])
walk(d['document']['panels'])

# Fix 12001 row 33 Ultra
p=flat[12001]
txt=p['options']['content']
old33='| 50 | `direct-openrouter-nemotron-3-ultra-free` | Stephen OR | `nvidia/nemotron-3-ultra-550b-a55b:free` | 1,048,576 | Y | t✓ i? a? v? p? | text | ? | Y/N/N | chat:cfg | [card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4) · hosted :free never served |'
new33='| 50 | `direct-openrouter-nemotron-3-ultra-free` | Stephen OR | `nvidia/nemotron-3-ultra-550b-a55b:free` | 1,048,576 | Y | t✓ iNO aNO vNO pNO | text | ✓ | Y/N/N | chat:cfg | [HC card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4) · [OR hosted](https://openrouter.ai/nvidia/nemotron-3-ultra-550b-a55b:free) (text in/out, tools ✓, 1M ctx) · never served |'
if old33 in txt:
    txt=txt.replace(old33, new33)
    print("Updated row 33 in 12001")
else:
    print("WARN: old33 not found")
p['options']['content']=txt

# Fix 12007 with exact live release.json
p=flat[12007]
p['options']['content']="""**DEPLOYMENT SNAPSHOT - read from the running service's release record on 2026-11-28 (`20261128T002506Z-6a8209f97101563b-18b377cd`, label `p11-functional-cleanup`).** Every hash below was recomputed with `sha256sum` against the installed bytes in `/var/lib/switchyard/releases/CURRENT/` and matches `release.json` exactly.

| Item | Recorded evidence |
|---|---|
| CURRENT (deployed) | `20261128T002506Z-6a8209f97101563b-18b377cd` |
| Release action / label | `deploy` / `p11-functional-cleanup`, created `2026-11-28T00:25:07Z` |
| Binary sha256 | `6a8209f97101563b58944f8614001fd0b057247da21a3daf57624ff798a22e82` (31,281,480 B) - unchanged from P10 qualified binary |
| Manifest sha256 | `18b377cd465862f3183fcb10cf01300659ec57ee2925735d7408d5ffd7e8c15e` (133,736 B) - P11 config-only cleanup (75→74 routes, composites 3→1, decision_lanes 5→4) |
| Capabilities sha256 | `987dd039ce618d5a0d124bd5607498b3c8d6845488af0d593fb3b3f7839f77b9` (25,499 B) |
| Rollback target | `20261127T223854Z-6a8209f97101563b-ead40bb6` (P10, manifest `ead40bb6…`) - protected in store |
| LAST-KNOWN-GOOD | `20261126T061331Z-65757ec7aaa51166-737a6887` - unchanged, **not** promoted |
| Store-protected releases | 12 ids (23 entries incl. LKG/CURRENT symlinks) |
| Source | `3d478f5b0e124b14db70b2d8f2dbcb4d69456579` on `feat/nemotron-ultra-peer-recovery-20261127`, dirty=false; attested at deploy (`declared-at-deploy`), **not** claimed reproducible |
| Running service | PID `873638`, `NRestarts=0`, `/health` `{"status":"ok"}`, `/v1/models` 74 of 74 advertised |
| Declared basis | config-only on qualified binary `6a8209f9`: removed `vc-paid-core`, `VC_Paid`, `tier1-decision`, `smart-nt-bounded-trial`, `google-decision` (reconciled into `smart-gemini-decision`); `smart-decision` composition retained |

**What this release changes, and why it is not a routing change.** P11 is a functional cleanup release: it removes a stale direct route (`vc-paid-core`), retires two unused composites (`VC_Paid`, `smart-nt-bounded-trial`), and reconciles the `google-decision` lane into the existing `smart-gemini-decision` lane. The binary is unchanged from the P10 qualified build; only the manifest bytes differ. No provider probes, no rebuild, no LKG promotion.
"""
print("Updated 12007")
json.dump(d, open('/var/tmp/v7-gen98-export.json','w'), indent=2)
