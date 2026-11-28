#!/usr/bin/env python3
"""Rows 02-07."""
import json
SRC='/var/tmp/v7-gen98-candidate.json'
d=json.load(open(SRC))
doc=d['document']
flat={}
def walk(ps):
    for p in ps:
        flat[p['id']]=p
        if 'panels' in p: walk(p['panels'])
walk(doc['panels'])

SEM={'ok':'#299C46','error':'#D44A4A','other_error':'#D44A4A','retryable_error':'#E08B26','client_disconnected':'#8B929A'}
NEUTRAL='#64748B'
def set_fixed_default(p,c=NEUTRAL): p['fieldConfig']['defaults']['color']={'mode':'fixed','fixedColor':c}
def add_byname(p,m):
    for n,c in m:
        p['fieldConfig']['overrides'].append({"matcher":{"id":"byName","options":n},"properties":[{"id":"color","value":{"mode":"fixed","fixedColor":c}}]})
def set_legend(p,f):
    for t in p.get('targets',[]): t['legendFormat']=f
def set_title(p,t): p['title']=t
def set_desc(p,d): p['description']=d
def set_unit(p,u): p['fieldConfig']['defaults']['unit']=u

# Row 02
# 9006: complete expression replacement
p=flat[9006]
set_title(p,'MODEL-FAMILY TRAFFIC · HEURISTIC ATTRIBUTION')
set_desc(p,"Request rate by model-family heuristic (native model labels). Pairwise disjoint; sum compared to all-model traffic; unmatched=unclassified. Caller label, not selected model or billing account. Unit requests/s.")
set_unit(p,'reqps')
p['targets']=[
 {"datasource":{"type":"prometheus","uid":"df7wrp11a9hq8c"},"expr":"sum(rate(switchyard_requests_total{job=\"switchyard-v7\",model=~\"comfyninja/qwen3\\.8-27b-q4-mtp|comfyninja/qwen3\\.8-27b-q4-mtp-vl\"}[5m]))","legendFormat":"2 ComfyNinja","refId":"A"},
 {"datasource":{"type":"prometheus","uid":"df7wrp11a9hq8c"},"expr":"sum(rate(switchyard_requests_total{job=\"switchyard-v7\",model=~\"deepseek-flash|deepseek-v4-pro\"}[5m]))","legendFormat":"3 owner DeepSeek","refId":"B"},
 {"datasource":{"type":"prometheus","uid":"df7wrp11a9hq8c"},"expr":"sum(rate(switchyard_requests_total{job=\"switchyard-v7\",model=~\"Qwen3\\.5-9B-MTP|Qwen3\\.5-9B-Q4_0|jev-style-0\\.8b-decision-v3|laya-rl-agent\"}[5m]))","legendFormat":"3 HTPC / local","refId":"C"},
 {"datasource":{"type":"prometheus","uid":"df7wrp11a9hq8c"},"expr":"sum(rate(switchyard_requests_total{job=\"switchyard-v7\",model=~\"baidu/ernie-5\\.5-flash-preview:free|inclusionai/ling-3\\.2-flash:free|deepseek/deepseek-v4\\.1-flash|deepseek/deepseek-v4\\.1-flash-1017|dots-studio/dots-3-note-preview:free|google/gemma-4-26b-a4b-it:free|google/gemma-4-31b-it:free|liquid/lfm-2\\.5-2\\.6b:free|nvidia/nemotron-3\\.5-super-vl:free|qwen/qwen3\\.7-flash|qwen/qwen3\\.8-flash|recursive/flint-v1\\.0:free|respan/span-01-lite|z-ai/glm-5\\.3-flash|openrouter/free|tencent/hy4-flash|nvidia/nemotron-3-ultra-550b-a55b:free\"}[5m]))","legendFormat":"14 owner OpenRouter","refId":"D"},
 {"datasource":{"type":"prometheus","uid":"df7wrp11a9hq8c"},"expr":"sum(rate(switchyard_requests_total{job=\"switchyard-v7\",model=~\"openthaigpt-thaillm-8b-instruct-v7\\.2|qwen3\\.5-9b|qwen3\\.6-35b-a3b|thalle-0\\.2-thaillm-8b-fa|typhoon-s-thaillm-8b-instruct\"}[5m]))","legendFormat":"5 ThaiLLM","refId":"E"},
 {"datasource":{"type":"prometheus","uid":"df7wrp11a9hq8c"},"expr":"sum(rate(switchyard_requests_total{job=\"switchyard-v7\",model=~\"gpt-6-luna|gpt-6-sol\"}[5m]))","legendFormat":"3 owner OpenAI","refId":"F"},
 {"datasource":{"type":"prometheus","uid":"df7wrp11a9hq8c"},"expr":"sum(rate(switchyard_requests_total{job=\"switchyard-v7\",model=~\"gemini-.+|gemma-[0-9]+-.+\"}[5m]))","legendFormat":"8 owner Google","refId":"G"},
]
# 8: legend {{model}}, title INCOMING MODEL REQUEST RATE
p=flat[8]; set_title(p,'INCOMING MODEL REQUEST RATE'); set_desc(p,"Incoming request rate by the raw `model` metric label (caller label, not selected model or account)."); set_legend(p,'{{model}}'); set_unit(p,'reqps')
# 19
p=flat[19]; set_title(p,'LLM CALLS BY SELECTED MODEL'); set_desc(p,"LLM call rate by `selected_model` physical label. No successful-completion or account claim."); set_legend(p,'{{selected_model}}'); set_unit(p,'reqps')
# 20
p=flat[20]; set_title(p,'CALLS BY ALGORITHM'); set_desc(p,"Call rate by routing algorithm (identity palette). Scope matches panels 16/17 algorithm filter."); set_legend(p,'{{algorithm}}'); set_unit(p,'reqps')
# 16
p=flat[16]; set_title(p,'DECISIONS BY ALGORITHM / SELECTED MODEL'); set_desc(p,"Decision rate by algorithm and selected_model. Existing algorithm filter."); set_legend(p,'{{algorithm}} · {{selected_model}}'); set_unit(p,'reqps')
# 17: table 24h, columns Algorithm / Selected model / Estimated decisions (24h)
p=flat[17]; set_title(p,'DECISION TOTALS (24h)'); set_desc(p,"Estimated decisions in 24h by algorithm and selected_model. Columns Algorithm / Selected model / Estimated decisions (24h), numeric descending sort.")
# 9106: instant table, columns Incoming model / Estimated requests (24h)
p=flat[9106]; set_title(p,'REQUESTS BY MODEL (24h)'); set_desc(p,"Estimated requests in 24h by incoming model. One row per model, numeric descending sort. No model-to-cost inference.")
p['targets'][0]['instant']=True
p['options']={'showHeader':True}

print("Row 02 done")
json.dump(d,open(SRC,'w'),indent=2)
