#!/usr/bin/env python3
"""OFFLINE, fail-closed removal of four owner-requested Switchyard identities.

Remove owner openrouter/free direct alias+physical target, vc-paid-core route,
VC_Paid composite, and Qwen 3.7 direct alias+physical target. Also remove the
one dynamic openrouter/free target from smartfree.candidates. Any other live
reference to a removed identifier REFUSES, not silently rewrites. Does not add
Hy4, invoke providers, write CURRENT, or apply a new smart topology.

Usage: python3 apply_owner_v2_retirements.py --baseline candidate.toml --out staged.toml
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import sys
import tomllib
from pathlib import Path

RETIRE = {
    'routes': frozenset({'direct-openrouter-openrouter-free',
                         'direct-openrouter-qwen3_7-flash', 'vc-paid-core'}),
    'targets': frozenset({'openrouter-openrouter-free', 'openrouter-qwen3_7-flash'}),
    'composites': frozenset({'VC_Paid'}),
}
TARGET_PAIRS = {
    'direct-openrouter-openrouter-free': 'openrouter-openrouter-free',
    'direct-openrouter-qwen3_7-flash': 'openrouter-qwen3_7-flash',
}
ALL_RETIRED = frozenset().union(*RETIRE.values())

class Refused(ValueError):
    pass

def sections(text: str) -> tuple[list[str], dict[str, tuple[int,int]]]:
    lines = text.splitlines(keepends=True)
    found = [(i, line.strip()) for i, line in enumerate(lines)
             if line.startswith('[') and line.rstrip().endswith(']')]
    mapping = {}
    for j, (start, head) in enumerate(found):
        if head in mapping:
            raise Refused(f'REFUSED_DUPLICATE_TOML_HEADING {head}')
        mapping[head] = (start, found[j+1][0] if j+1 < len(found) else len(lines))
    return lines, mapping

def heading(mapping: dict, kind: str, key: str):
    variants = [f'[{kind}.{key}]', f'[{kind}."{key}"]']
    hit = [v for v in variants if v in mapping]
    if len(hit) != 1:
        raise Refused(f'REFUSED_HEADING_NOT_UNIQUE {kind}.{key} {hit}')
    return hit[0]

def refs(obj, path='manifest'):
    """Visit exact TOML string VALUES, not substrings. Never count map keys."""
    if isinstance(obj, str):
        if obj in ALL_RETIRED:
            yield path, obj
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            yield from refs(value, f'{path}[{i}]')
    elif isinstance(obj, dict):
        for key, value in obj.items():
            yield from refs(value, f'{path}.{key}')

def apply(text: str) -> tuple[str, dict]:
    doc = tomllib.loads(text)
    for kind, names in RETIRE.items():
        if not isinstance(doc.get(kind), dict):
            raise Refused(f'REFUSED_MISSING_SECTION {kind}')
        for name in names:
            if name not in doc[kind]:
                raise Refused(f'REFUSED_MISSING_EXACT_ID {kind}.{name}')
    for alias, target in TARGET_PAIRS.items():
        route = doc['routes'][alias]
        if route.get('type') != 'passthrough' or route.get('target') != target or route.get('candidates'):
            raise Refused(f'REFUSED_DIRECT_NOT_1TO1 {alias} => {target}')
    smartfree = doc['routes'].get('smartfree')
    if not isinstance(smartfree, dict):
        raise Refused('REFUSED_SMARTFREE_NOT_PRESENT')
    candidates = smartfree.get('candidates')
    if not isinstance(candidates, list) or candidates.count('openrouter-openrouter-free') != 1:
        raise Refused('REFUSED_SMARTFREE_DYNAMIC_REFERENCE_NOT_UNIQUE')
    kept = [x for x in candidates if x != 'openrouter-openrouter-free']
    if not kept:
        raise Refused('REFUSED_SMARTFREE_NO_CANDIDATES_LEFT')

    # All data that will survive. No unreviewed consumer may reference a retired
    # route, target or composite, even from an unrelated nested TOML section.
    survivor = {}
    for kind, value in doc.items():
        if kind in RETIRE:
            survivor[kind] = {k:v for k,v in value.items() if k not in RETIRE[kind]}
        else:
            survivor[kind] = value
    survivor['routes']['smartfree'] = {**smartfree, 'candidates': kept}
    remaining = list(refs(survivor))
    if remaining:
        raise Refused('REFUSED_RETIRED_CONSUMER ' + repr(remaining[:10]))

    lines, blocks = sections(text)
    smart_header = heading(blocks, 'routes', 'smartfree')
    begin,end = blocks[smart_header]
    found = [i for i in range(begin+1,end) if re.match(r'^\s*candidates\s*=', lines[i])]
    if len(found) != 1:
        raise Refused('REFUSED_SMARTFREE_CANDIDATES_LINE_NOT_UNIQUE')
    i = found[0]
    # Exactly one physical line: never corrupt a multiline TOML array.
    if not re.fullmatch(r'\s*candidates\s*=\s*\[[^\]\n]*\]\s*(?:#.*)?\n?', lines[i]):
        raise Refused('REFUSED_SMARTFREE_CANDIDATES_NOT_SINGLE_LINE')
    newline = '\n' if lines[i].endswith('\n') else ''
    lines[i] = 'candidates = ' + json.dumps(kept, ensure_ascii=False) + newline
    for kind,names in RETIRE.items():
        for name in names:
            b,e = blocks[heading(blocks,kind,name)]
            for j in range(b,e):
                lines[j] = ''
    new_text = ''.join(lines)
    new_doc = tomllib.loads(new_text)
    if new_doc != survivor:
        raise Refused('REFUSED_SEMANTIC_DELTA_OUTSIDE_SCOPED_RETIREMENTS')
    report = {'removed': {k:sorted(v) for k,v in RETIRE.items()},
              'smartfree_candidates_before': candidates, 'smartfree_candidates_after': kept,
              'before': {k:len(doc[k]) for k in RETIRE},
              'after': {k:len(new_doc[k]) for k in RETIRE},
              'input_sha256':hashlib.sha256(text.encode()).hexdigest(),
              'output_sha256':hashlib.sha256(new_text.encode()).hexdigest()}
    return new_text, report

def main(argv=None) -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args(argv)
    if args.baseline.resolve() == args.out.resolve():
        print('REFUSED: CANNOT_OVERWRITE_BASELINE',file=sys.stderr)
        return 3
    try:
        result,report=apply(args.baseline.read_text())
        args.out.write_text(result)
    except (OSError,ValueError,tomllib.TOMLDecodeError) as e:
        print('REFUSED: '+str(e),file=sys.stderr)
        return 3
    print(json.dumps(report,indent=2,sort_keys=True))
    print('CANDIDATE_ONLY; HY4_NOT_ADDED; P9_AND_GUARDED_DEPLOY_STILL_REQUIRED')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
