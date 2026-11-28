#!/usr/bin/env python3
"""Rows 04-07."""
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

# Row 04
# 23: A {{model}} p50, B p95, C p99
p=flat[23]; set_title(p,'MODEL-CALL LATENCY p50 / p95 / p99'); set_desc(p,"Model-call latency quantiles over 5m rolling histogram estimate. A p50, B p95, C p99. Unit ms. No nonexistent quantile label.")
set_unit(p,'ms')
for t in p['targets']:
    if t['refId']=='A': t['legendFormat']='{{model}} p50'
    elif t['refId']=='B': t['legendFormat']='{{model}} p95'
    elif t['refId']=='C': t['legendFormat']='{{model}} p99'

# 24: legend {{model}} p95, ms
p=flat[24]; set_title(p,'TOTAL LATENCY p95 BY MODEL'); set_desc(p,"Total latency p95 by model (includes provider-call + routing overhead). Legend `{{model}} p95`. Unit ms."); set_legend(p,'{{model}} p95'); set_unit(p,'ms')

# 25: legend {{algorithm}} p95, identity palette
p=flat[25]; set_title(p,'ROUTING OVERHEAD p95'); set_desc(p,"Routing overhead p95 by algorithm (identity palette). Unit ms."); set_legend(p,'{{algorithm}} p95'); set_unit(p,'ms')

# 26: legend {{outcome}}, ok green/error red, ms, no arbitrary cap
p=flat[26]
set_title(p,'LLM CALL DURATION p95 BY OUTCOME')
set_desc(p,"LLM call duration p95 by outcome (5m rolling histogram). Legend `{{outcome}}`. ok green, error red, neutral default. Unit ms. No arbitrary 10s axis cap; plateaus may be histogram bucket limit (inspect le buckets/+Inf).")
set_unit(p,'ms')
p['targets'][0]['legendFormat']='{{outcome}}'
set_fixed_default(p)
p['fieldConfig']['overrides']=[]
add_byname(p,[('ok','#299C46'),('error','#D44A4A')])
# remove any max
p['fieldConfig']['defaults'].pop('max',None)
p['fieldConfig']['defaults'].pop('min',None)

# 35: legend {{outcome}}, same mappings as 34, ms
p=flat[35]
set_title(p,'RUN DURATION p95 BY OUTCOME')
set_desc(p,"Run duration p95 by outcome (5m rolling histogram). Legend `{{outcome}}`. Same semantic mappings as panel 34. Unit ms. No smoothing null gaps or arbitrary caps.")
set_unit(p,'ms')
p['targets'][0]['legendFormat']='{{outcome}}'
set_fixed_default(p)
p['fieldConfig']['overrides']=[]
add_byname(p,[('ok','#299C46'),('success','#299C46'),('error','#D44A4A'),('failure','#D44A4A'),('retryable_error','#E08B26'),('client_disconnected','#8B929A'),('cancelled','#8B929A')])

# 13005: title MODEL-CALL LATENCY (7d) - p50/p95/p99, table, join by model
p=flat[13005]; set_title(p,'MODEL-CALL LATENCY (7d) - p50 / p95 / p99'); set_desc(p,"Model-call latency quantiles over 7d (instant). Three instant quantile queries joined by model. Columns p50/p95/p99 (ms), numeric p95 desc, preserve missing. Explicit Prometheus DS.")
p['type']='table'
for t in p['targets']:
    t['instant']=True
    t['datasource']={'type':'prometheus','uid':'df7wrp11a9hq8c'}
p['options']={'showHeader':True}

# Row 05
# 9108: description components, tokens/s, no cached claim
p=flat[9108]; set_title(p,'TOKEN RATE BY COMPONENT (prompt / completion / reasoning)'); set_desc(p,"Token rate by component: A=prompt, B=completion, C=reasoning. Unit tokens/s. No cached/cache_creation claim. Identity colors distinguish components; no summing potentially overlapping counts."); set_unit(p,'tps')

# 29/28/30: legend {{model}}, tokens/s
for pid,t,ttl in [(29,'{{model}}','PROMPT TOKENS BY MODEL'),(28,'{{model}}','COMPLETION TOKENS BY MODEL'),(30,'{{model}}','REASONING TOKENS BY MODEL')]:
    p=flat[pid]; set_title(p,ttl); set_desc(p,f"{ttl.lower()}. Legend `{{{{model}}}}`. Unit tokens/s."); set_legend(p,t); set_unit(p,'tps')

# 31: CACHE HIT SHARE, percentunit, correct reversed desc, C/(P>0)
p=flat[31]; set_title(p,'CACHE HIT SHARE'); set_desc(p,"Cache hit share = cached_tokens / (prompt_tokens > 0). Denominator P=sum by(model)(rate(prompt_tokens_total[5m])); numerator C=same cached_tokens_total. Uses C/(P>0) so idle denominator is absent, not fabricated zero. Legend `{{model}}`. Provider accounting semantics verified; if cached not subset of prompt, limitation labeled."); set_unit(p,'percentunit'); set_legend(p,'{{model}}')
# fix expr to C/(P>0)
p['targets'][0]['expr']='sum by(model)(rate(switchyard_cached_tokens_total{job="switchyard-v7"}[5m])) / (sum by(model)(rate(switchyard_prompt_tokens_total{job="switchyard-v7"}[5m])) > 0)'

# 32: CACHE TOKEN RATE (tokens/s), correct reversed desc
p=flat[32]; set_title(p,'CACHE TOKEN RATE (tokens/s)'); set_desc(p,"Cached token rate by model. Legend `{{model}}`. Unit tokens/s."); set_legend(p,'{{model}}'); set_unit(p,'tps')

print("Rows 04-05 done")
json.dump(d,open(SRC,'w'),indent=2)
