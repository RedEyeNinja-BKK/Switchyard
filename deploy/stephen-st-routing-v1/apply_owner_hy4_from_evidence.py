#!/usr/bin/env python3
"""Add owner Hy4 Flash CHAT direct from scoped REAL OpenRouter wire evidence.

OFFLINE candidate only. No service/credential write, no paid smart fallback.
The existing owner llm_clients.openrouter transport MUST be the tested wire.
No provider documentation can substitute for a real tools+JSON response.
"""
from __future__ import annotations
import argparse, json, sys, tomllib
from pathlib import Path

MODEL='tencent/hy4-flash'
TARGET='openrouter-tencent-hy4-flash'
ROUTE='direct-openrouter-tencent-hy4-flash'
VENDOR_CTX=1048576
VENDOR_OUTPUT=65536
class Refused(ValueError): pass

def _positive(value):
    return type(value) is int and value > 0

def apply(raw: str, proof: dict) -> tuple[str, dict]:
    doc=tomllib.loads(raw)
    client=(doc.get('llm_clients') or {}).get('openrouter')
    if not isinstance(client,dict) or not client.get('api_key_env'):
        raise Refused('HY4_OWNER_CLIENT_NOT_CONFIGURED')
    if 'stephen' in client['api_key_env'].lower() or client.get('format') not in ('openai_responses','openai_chat'):
        raise Refused('HY4_OWNER_CLIENT_NOT_ISOLATED')
    if proof.get('account_client')!='openrouter' or proof.get('upstream_id')!=MODEL:
        raise Refused('HY4_PROVIDER_OR_OWNER_MISMATCH')
    if proof.get('wire_format')!=client.get('format'):
        raise Refused('HY4_WIRE_FORMAT_UNTESTED')
    for key in ('text_http_status','tool_http_status','json_http_status'):
        if proof.get(key)!=200:
            raise Refused(f'HY4_REAL_WIRE_NOT_200 {key}={proof.get(key)!r}')
    if not _positive(proof.get('tool_calls')) or proof.get('structured_output_valid') is not True:
        raise Refused('HY4_TOOL_OR_JSON_NOT_QUALIFIED')
    if not proof.get('receipt') or not proof.get('measured_at'):
        raise Refused('HY4_REAL_PROOF_RECEIPT_MISSING')
    if proof.get('cost_mode') != 'DIRECT_OPT_IN_ONLY':
        raise Refused('HY4_PAID_AUTO_FALLBACK_FORBIDDEN')
    ctx=proof.get('route_context_window')
    inp=proof.get('max_input_tokens')
    if not _positive(ctx) or ctx>VENDOR_CTX or not _positive(inp) or inp>=ctx:
        raise Refused('HY4_CONTEXT_WINDOW_NOT_CONSERVATIVE')
    if ctx>262144:
        raise Refused('HY4_OWNER_ROUTE_WINDOW_CAP_262144')
    image_ok=proof.get('image_http_status')==200
    carrier=proof.get('image_carrier')
    if image_ok and carrier not in ('inline','remote'):
        raise Refused('HY4_IMAGE_CARRIER_UNQUALIFIED')
    if not image_ok and carrier is not None:
        raise Refused('HY4_IMAGE_CARRIER_WITHOUT_WIRE_PROOF')
    if proof.get('image_http_status') not in (None,200,400,415,422):
        raise Refused('HY4_IMAGE_STATUS_NOT_A_QUALIFICATION')
    for kind,name in (('targets',TARGET),('routes',ROUTE)):
        if name in (doc.get(kind) or {}):
            raise Refused(f'HY4_ALREADY_DEFINED {kind}.{name}')
    # Fail closed if this new target already has references from preexisting routes.
    for route,name in (doc.get('routes') or {}).items():
        if name.get('target') == TARGET or TARGET in (name.get('candidates') or []):
            raise Refused(f'HY4_TARGET_ALREADY_REFERENCED {route}')
    lines=['', f'[targets.{TARGET}]',f'id = {json.dumps(MODEL)}',
           'llm_client = "openrouter"','free_tier = false','tool_calling = true',
           f'vision = {str(image_ok).lower()}','video = false',
           f'context_window = {VENDOR_CTX}',f'max_input_tokens = {inp}']
    if image_ok: lines.append(f'image_carrier = {json.dumps(carrier)}')
    lines += ['',f'[routes.{ROUTE}]',f'id = {json.dumps(ROUTE)}',
              'type = "passthrough"',f'target = {json.dumps(TARGET)}',
              'candidates = []',f'context_window = {ctx}',
              'tool_calling = true',f'vision = {str(image_ok).lower()}', 'video = false']
    if image_ok: lines.append(f'image_carrier = {json.dumps(carrier)}')
    result=raw.rstrip()+'\n'+'\n'.join(lines)+'\n'
    after=tomllib.loads(result)
    expected={**doc,
              'targets':{**doc.get('targets',{}),TARGET:after['targets'][TARGET]},
              'routes':{**doc.get('routes',{}),ROUTE:after['routes'][ROUTE]}}
    if after!=expected:
        raise Refused('HY4_UNSCOPED_MANIFEST_DELTA')
    return result, {'direct_route':ROUTE,'model':MODEL,'client':'openrouter',
                    'wire':client['format'],'tools_qualified':True,
                    'json_qualified':True,'image_qualified':image_ok,
                    'ctx':ctx,'max_input_tokens':inp,'vendor_documented_max_output':VENDOR_OUTPUT,
                    'receipt':proof['receipt'],'measured_at':proof['measured_at'],
                    'not_added_to_smart_paid_pool':True}

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--evidence',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args(argv)
    if args.out.resolve() in (args.baseline.resolve(),args.evidence.resolve()):
        print('REFUSED: HY4_CANNOT_OVERWRITE_INPUT',file=sys.stderr);return 3
    try:
        out,report=apply(args.baseline.read_text(),json.loads(args.evidence.read_text()))
        args.out.write_text(out)
    except (OSError,ValueError,tomllib.TOMLDecodeError,json.JSONDecodeError) as e:
        print('REFUSED: '+str(e),file=sys.stderr);return 3
    print(json.dumps(report,indent=2,sort_keys=True))
    print('CANDIDATE_ONLY: run P9, release_validate, guarded deployment, live proof')
    return 0
if __name__=='__main__': raise SystemExit(main())
