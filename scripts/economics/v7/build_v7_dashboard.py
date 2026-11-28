#!/usr/bin/env python3
"""Apply the owner's exhaustive panel specification to the V7 dashboard.
Reads the gen97 export, applies all Row 01-07 fixes, writes gen98 candidate.
Row 08 (text panels 12001-12007) handled separately after research.
"""
import json, copy, sys

SRC = '/var/tmp/v7-gen97-export.json'
DST = '/var/tmp/v7-gen98-candidate.json'

d = json.load(open(SRC))
doc = d['document']
flat = {}
def walk(ps):
    for p in ps:
        flat[p['id']] = p
        if 'panels' in p: walk(p['panels'])
walk(doc['panels'])

SEM = {  # semantic outcome -> color
    'ok': '#299C46',
    'error': '#D44A4A',
    'other_error': '#D44A4A',
    'retryable_error': '#E08B26',
    'client_disconnected': '#8B929A',
}
NEUTRAL = '#64748B'

def set_fixed_default(panel, default_color=NEUTRAL):
    panel['fieldConfig']['defaults']['color'] = {'mode':'fixed','fixedColor':default_color}

def add_byname_overrides(panel, mapping):
    """mapping: list of (name, color)"""
    ovs = panel['fieldConfig']['overrides']
    for name,color in mapping:
        ovs.append({
            "matcher":{"id":"byName","options":name},
            "properties":[{"id":"color","value":{"mode":"fixed","fixedColor":color}}]
        })

def add_byrefid_overrides(panel, mapping):
    for rid,color in mapping:
        panel['fieldConfig']['overrides'].append({
            "matcher":{"id":"byFrameRefID","options":rid},
            "properties":[{"id":"color","value":{"mode":"fixed","fixedColor":color}}]
        })

def set_legend(panel, fmt):
    for t in panel.get('targets',[]):
        t['legendFormat'] = fmt

def set_title(panel, title):
    panel['title'] = title

def set_desc(panel, desc):
    panel['description'] = desc

def set_unit(panel, unit):
    panel['fieldConfig']['defaults']['unit'] = unit

# ---- Row 01 ----
# Panel 2: remove GrYlRd, neutral, title BUILD VERSION
p=flat[2]
p['fieldConfig']['defaults'].pop('color',None)
p['fieldConfig']['defaults']['color']={'mode':'fixed','fixedColor':NEUTRAL}
set_title(p,'BUILD VERSION')
set_desc(p,"Reads the `version` label of `switchyard_build_info`. The sample value is the constant 1 - a presence marker, never a version string - so the panel displays the label instead of the value. Scrape freshness is the SCRAPE AGE panel beside it. Current release is shown in panel 12007.")

# Panel 3: title SCRAPE TARGET UP
set_title(flat[3],'SCRAPE TARGET UP')
set_desc(flat[3],"`up` for the Switchyard scrape target. 0=red (down), 1=green (up), no-series=neutral. This is target reachability, not provider availability.")

# Panel 4: title SCRAPE AGE, label latest build-info sample age
set_title(flat[4],'SCRAPE AGE')
set_desc(flat[4],"Seconds since the latest `switchyard_build_info` sample was scraped. Green <300s, orange 300-900s, red >900s, absent=neutral. This is the age of the latest build-info sample, not proof every provider is healthy.")

# Panel 6: thresholds green/orange/red not yellow OK
p=flat[6]
set_title(p,'ALGORITHMS IN FLIGHT')
set_desc(p,"In-flight algorithm count (gauge). Green at normal concurrency, orange warning (20), red high (50). Thresholds are not proven service capacity.")
# ensure thresholds mode
p['fieldConfig']['defaults']['color']={'mode':'thresholds'}
p['fieldConfig']['defaults']['thresholds']={'mode':'absolute','steps':[
    {'color':'green','value':None},
    {'color':'orange','value':20},
    {'color':'red','value':50},
]}

# Panel 12: title CLIENT NON-OK RATE (5m)
set_title(flat[12],'CLIENT NON-OK RATE (5m)')
set_desc(flat[12],"Client non-ok rate over 5m (includes disconnects). Green <5%, orange 5-20%, red >20%, no-series=neutral. Numerator/denominator unchanged. 91007 supplies idle distinction. The clamp caveat: rate is clamped to [0,1] when denominator is zero.")

# Panel 9003: neutral traffic count, title ACTIVE REQUEST MODEL LABELS (5m)
p=flat[9003]
set_title(p,'ACTIVE REQUEST MODEL LABELS (5m)')
set_desc(p,"Count of distinct model labels with >0 requests in 5m. Neutral color including zero; no red for zero or green for mere activity. Uses sum of per-model boolean > bool 0 if series exist; no artificial zero if no series.")
set_fixed_default(p, NEUTRAL)

# Panel 13001: fix DS, D timestamp seconds, rename recency label
p=flat[13001]
set_title(p,'GOOGLE MODEL ACTIVITY (model-family attribution, not account)')
set_desc(p,"Google-family model activity by heuristic (selected_model=~gemini-.+|gemma-…), not owner account. A=calls(1h) neutral, B=ok(24h) green, C=error(24h) red, D=latest counter scrape age in seconds (dateTimeFromNow). Model-family heuristic, not owner account.")
# fix D: timestamp()*1000 for dateTimeFromNow
for t in p['targets']:
    if t['refId']=='D':
        t['expr']='max(timestamp(switchyard_llm_calls_total{job="switchyard-v7", selected_model=~"gemini-.+|gemma-[0-9]+-.+"}))*1000'
        t['legendFormat']='latest counter scrape (not a call)'
    if t['refId']=='A':
        t['legendFormat']='calls (1h)'
# ensure all targets have prom DS
for t in p['targets']:
    t['datasource']={'type':'prometheus','uid':'df7wrp11a9hq8c'}

# Panel 5: title PROCESS REQUESTS
set_title(flat[5],'PROCESS REQUESTS')
set_desc(flat[5],"Process request counter (instant). Neutral blue/text. No lifetime/spend claim.")

# Panel 9101: neutral numeric including zero
p=flat[9101]
set_title(p,'REQUEST ATTEMPTS - ALL MODELS (24h)')
set_desc(p,"24h all-model request count (instant). Neutral numeric including zero; no red for idle.")
set_fixed_default(p, NEUTRAL)

# Panel 9004: label UPSTREAM NON-OK (24h)
p=flat[9004]
set_title(p,'UPSTREAM NON-OK (24h)')
set_desc(p,"Non-ok upstream attempts (24h). 0=green, 1=orange, 20=red, missing=neutral. These are upstream attempts, not client failures.")
p['fieldConfig']['defaults']['color']={'mode':'thresholds'}
p['fieldConfig']['defaults']['thresholds']={'mode':'absolute','steps':[
    {'color':'green','value':None},
    {'color':'orange','value':1},
    {'color':'red','value':20},
]}

# Panel 9104: 0 green/1 orange/20 red
p=flat[9104]
set_title(p,'UPSTREAM 429s (24h)')
set_desc(p,"HTTP 429 upstream count (24h). 0=green, 1=orange, 20=red, missing=neutral. 429 may be throttling or daily allowance; not HTTP status alone.")
p['fieldConfig']['defaults']['color']={'mode':'thresholds'}
p['fieldConfig']['defaults']['thresholds']={'mode':'absolute','steps':[
    {'color':'green','value':None},
    {'color':'orange','value':1},
    {'color':'red','value':20},
]}

# Panel 13: neutral historical count
p=flat[13]
set_title(p,'TOTAL ERRORS (SINCE PROCESS START)')
set_desc(p,"Process error counter (historical, since process start). Neutral; not current health. Resets on restart.")
set_fixed_default(p, NEUTRAL)

# Panel 9103: neutral including zero
p=flat[9103]
set_title(p,'MODEL SERIES THAT EXIST (not advertised census)')
set_desc(p,"Count of model series that exist (instant). Neutral including zero; no service health or route census claim.")
set_fixed_default(p, NEUTRAL)

# Panel 91001: bargauge, fixed neutral default, A/B/C/D colors, min0, no legend, values visible
p=flat[91001]
set_title(p,'CLIENT OUTCOMES (5m) - COUNTS BY OUTCOME')
set_desc(p,"Client response counts in 5m by outcome (extrapolated increase()). A=ok green, B=retryable_error orange, C=other_error red, D=client_disconnected grey. Neutral fixed default. Shared min0 scale. No fabricated zero. Counts are extrapolated increase().")
set_fixed_default(p, NEUTRAL)
p['fieldConfig']['defaults']['min']=0
p['fieldConfig']['defaults']['unit']='short'
# byFrameRefID for A/B/C/D
p['fieldConfig']['overrides']=[]  # clear old byName
add_byrefid_overrides(p, [('A','#299C46'),('B','#E08B26'),('C','#D44A4A'),('D','#8B929A')])
p['options']['legend']={'displayMode':'hidden','placement':'bottom'}
p['options']['reduceOptions']={'calcs':['lastNotNull'],'fields':'','values':True}

# Panel 91002: neutral fixed default, byName, title, desc
p=flat[91002]
set_title(p,'CLIENT OUTCOMES (15m rolling counts)')
set_desc(p,"15-minute rolling response counts by outcome (stacked bars, one count axis, no total/percentages). Neutral fixed default, byName semantic colors. Panel 12 is a KPI, not a percentage trend; the 5m ratio is in 91006.")
set_fixed_default(p, NEUTRAL)
# keep existing byName overrides (they're correct)

# Panel 91005: title RESPONSES (5m)
p=flat[91005]
set_title(p,'RESPONSES (5m)')
set_desc(p,"Responses in 5m (extrapolated). Neutral; explicit missing vs zero; no rounding claim exact.")
set_fixed_default(p, NEUTRAL)
p['noValue']='NO SERIES'

# Panel 91006: title CLIENT ERROR SHARE (5m, excludes disconnects from numerator)
p=flat[91006]
set_title(p,'CLIENT ERROR SHARE (5m, excludes disconnects from numerator)')
set_desc(p,"Genuine error share (5m): numerator excludes client_disconnected, denominator includes disconnects. Green/orange/red percentage; NaN/idle/missing neutral. If error series absent while denominator positive, distinguishes proven no errors from absent telemetry; does not silently change denominator or inject unconditional zeros.")

# Panel 91007: title TRAFFIC STATE (5m)
p=flat[91007]
set_title(p,'TRAFFIC STATE (5m)')
set_desc(p,"Traffic state (5m): zero→IDLE grey, positive→traffic neutral (not green implying success), absent→NO SERIES grey, query error distinct. Companion to panel 12.")

print("Row 01 done")
json.dump(d, open(DST,'w'), indent=2)
print("wrote", DST)
