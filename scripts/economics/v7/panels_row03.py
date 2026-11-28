#!/usr/bin/env python3
"""Rows 03-07."""
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

# Row 03
# 9009: by(outcome,code), legend, anchored byRegexp
p=flat[9009]
set_title(p,'UPSTREAM ATTEMPTS BY OUTCOME / CODE')
set_desc(p,"Upstream attempt rate by outcome and HTTP code. Legend `{{outcome}} · {{code}}`. Neutral fixed default; anchored byRegexp: ^ok green, ^error/^other_error red, ^retryable_error orange, ^client_disconnected grey. Unit attempts/s. Status not inferred from HTTP code alone.")
set_unit(p,'reqps')
p['targets'][0]['legendFormat']='{{outcome}} · {{code}}'
set_fixed_default(p)
p['fieldConfig']['overrides']=[
 {"matcher":{"id":"byRegexp","options":"^ok · .+$"},"properties":[{"id":"color","value":{"mode":"fixed","fixedColor":"#299C46"}}]},
 {"matcher":{"id":"byRegexp","options":"^error · .+$"},"properties":[{"id":"color","value":{"mode":"fixed","fixedColor":"#D44A4A"}}]},
 {"matcher":{"id":"byRegexp","options":"^other_error · .+$"},"properties":[{"id":"color","value":{"mode":"fixed","fixedColor":"#D44A4A"}}]},
 {"matcher":{"id":"byRegexp","options":"^retryable_error · .+$"},"properties":[{"id":"color","value":{"mode":"fixed","fixedColor":"#E08B26"}}]},
 {"matcher":{"id":"byRegexp","options":"^client_disconnected · .+$"},"properties":[{"id":"color","value":{"mode":"fixed","fixedColor":"#8B929A"}}]},
]

# 11: legend {{model}}, identity palette allowed, remove 24h ref
p=flat[11]; set_title(p,'ERRORS BY MODEL'); set_desc(p,"Error rate by model (errors/s). Identity palette (all are error counts, no status semantics)."); set_legend(p,'{{model}}'); set_unit(p,'reqps')

# 9: legend {{outcome}}, fixed neutral, exact overrides, responses/s
p=flat[9]
set_title(p,'CLIENT RESPONSES BY OUTCOME')
set_desc(p,"Client response rate by outcome. Legend `{{outcome}}`. Neutral fixed default; exact byName: ok green, retryable_error orange, other_error red, client_disconnected grey. Unit responses/s.")
set_unit(p,'reqps')
p['targets'][0]['legendFormat']='{{outcome}}'
set_fixed_default(p)
p['fieldConfig']['overrides']=[]
add_byname(p,[('ok','#299C46'),('retryable_error','#E08B26'),('other_error','#D44A4A'),('client_disconnected','#8B929A')])

# 14: legend {{outcome}}, ok green/error red, calls/s
p=flat[14]
set_title(p,'LLM CALL RATE BY OUTCOME')
set_desc(p,"LLM call rate by outcome. Legend `{{outcome}}`. ok green, error red, neutral unknown. Unit calls/s.")
set_unit(p,'reqps')
p['targets'][0]['legendFormat']='{{outcome}}'
set_fixed_default(p)
p['fieldConfig']['overrides']=[]
add_byname(p,[('ok','#299C46'),('error','#D44A4A')])

# 34: legend {{outcome}}, success/ok green, error/failure red, retryable orange, disconnected/cancelled grey, runs/s
p=flat[34]
set_title(p,'RUN TERMINATIONS BY OUTCOME')
set_desc(p,"Run termination rate by outcome. Legend `{{outcome}}`. ok/success green, error/failure red, retryable orange, disconnected/cancelled grey, unknown neutral. Unit runs/s (not request count).")
set_unit(p,'reqps')
p['targets'][0]['legendFormat']='{{outcome}}'
set_fixed_default(p)
p['fieldConfig']['overrides']=[]
add_byname(p,[('ok','#299C46'),('success','#299C46'),('error','#D44A4A'),('failure','#D44A4A'),('retryable_error','#E08B26'),('client_disconnected','#8B929A'),('cancelled','#8B929A')])

# 21: RECOVERED RETRIES (rolling 24h), fixed green, count not rate
p=flat[21]; set_title(p,'RECOVERED RETRIES (rolling 24h)'); set_desc(p,"Recovered retries in rolling 24h (count, not rate). Fixed green for actual recovery. Multiple scrape labels preserved."); set_unit(p,'short'); set_fixed_default(p,'#299C46'); p['fieldConfig']['overrides']=[]

# 13003: instant TABLE, columns HTTP code / Outcome / Estimated attempts (24h)
p=flat[13003]
set_title(p,'UPSTREAM HTTP CODES (code/outcome breakdown)')
set_desc(p,"Estimated upstream attempts in 24h by HTTP code and outcome. Instant table: columns HTTP code / Outcome / Estimated attempts (24h), count descending, hide Time/__name__. Outcome field text colored by exact value. 429 may be daily quota or throttle.")
p['type']='table'
p['targets'][0]['instant']=True
p['targets'][0]['legendFormat']='{{code}} · {{outcome}}'
set_fixed_default(p)
# color Outcome field text by value
p['fieldConfig']['overrides']=[
 {"matcher":{"id":"byName","options":"Outcome"},"properties":[{"id":"color","value":{"mode":"fixed","fixedColor":"#299C46"}},{"id":"custom.cellOptions","value":{"mode":"color-background"}}]},
]
p['options']={'showHeader':True}

print("Row 03 done")
json.dump(d,open(SRC,'w'),indent=2)
