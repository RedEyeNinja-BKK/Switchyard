#!/usr/bin/env python3
"""Row 06-07."""
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
def set_title(p,t): p['title']=t
def set_desc(p,d): p['description']=d

# Row 06
# 9105: instant table, columns Parent/Child/Outcome/Estimated events (24h)
p=flat[9105]; set_title(p,'CASCADE CHILD OUTCOMES (24h)'); set_desc(p,"Estimated cascade events in 24h. Instant table: Parent / Child / Outcome / Estimated events (24h), numeric desc, no duplicate Time. Outcome value colors: attempt_failed red, skipped_ineligible neutral/orange, selected neutral. No retired ST_RouteFree text; historical data retained. Query does not group capability.")
p['type']='table'
p['targets'][0]['instant']=True
p['options']={'showHeader':True}

# 9107: crossings by parent/reason, title CASCADE CROSSINGS (rolling 5m)
p=flat[9107]; set_title(p,'CASCADE CROSSINGS (rolling 5m)'); set_desc(p,"Cascade crossings by parent/reason (rolling 5m, count). Legend `{{parent_route}}: {{reason}}`. Crossings not errors; neutral default, orange known temporary capacity, red actual failure. No stale invented reason taxonomy.")
for t in p['targets']: t['legendFormat']='{{parent_route}}: {{reason}}'
set_fixed_default(p)
p['fieldConfig']['overrides']=[]

# 9109: A skipped_ineligible, B attempt_failed, by refId
p=flat[9109]; set_title(p,'CASCADE SKIPS & FAILURES (rolling 24h, capability label)'); set_desc(p,"Cascade skips (A) and failures (B) in 24h by capability. Legend `skipped · {{parent_route}} · {{capability}}` / `failed · ...`. Colors by refId: grey/orange skips, red failure. Capability is not complete reason taxonomy.")
p['fieldConfig']['overrides']=[
 {"matcher":{"id":"byFrameRefID","options":"A"},"properties":[{"id":"color","value":{"mode":"fixed","fixedColor":"#8B929A"}}]},
 {"matcher":{"id":"byFrameRefID","options":"B"},"properties":[{"id":"color","value":{"mode":"fixed","fixedColor":"#D44A4A"}}]},
]
set_fixed_default(p)

# 9110: parent_route="ST_Route", title STEPHEN CASCADE CHILD OUTCOMES (rolling 24h)
p=flat[9110]; set_title(p,'STEPHEN CASCADE CHILD OUTCOMES (rolling 24h)'); set_desc(p,"Stephen cascade child outcomes in 24h (parent_route=ST_Route). Legend `{{child_route}} · {{outcome}}`. selected neutral, attempt_failed red, skipped_ineligible grey/orange. No borrowed owner routing.")
for t in p['targets']:
    t['expr']=t['expr'].replace('parent_route=~"ST_Route|ST_Route"','parent_route="ST_Route"')
    t['legendFormat']='{{child_route}} · {{outcome}}'
set_fixed_default(p)
p['fieldConfig']['overrides']=[]
add_byname(p,[('attempt_failed','#D44A4A'),('skipped_ineligible','#8B929A')])

# 9111: same parent matcher, title STEPHEN CASCADE SELECTIONS (rolling 24h)
p=flat[9111]; set_title(p,'STEPHEN CASCADE SELECTIONS (rolling 24h)'); set_desc(p,"Stephen cascade selections in 24h by child_route,target. Legend `{{child_route}} → {{target}}`. Selection is not serving unless proven by emitter; neutral identity colors. No invented served counter.")
for t in p['targets']:
    t['expr']=t['expr'].replace('parent_route=~"ST_Route|ST_Route"','parent_route="ST_Route"')
    t['legendFormat']='{{child_route}} → {{target}}'
set_fixed_default(p)
p['fieldConfig']['overrides']=[]

# Row 07
# 10506: provider credit & balance, account attribution unavailable
p=flat[10506]; set_title(p,'PROVIDER CREDIT & BALANCE (account attribution unavailable)'); set_desc(p,"Purchased credit and remaining balance by provider/currency (gauges, not spend). Current queries aggregate provider,currency; authoritative account label absent, so account-level split unavailable. Columns Provider / Currency / Purchased credit / Remaining balance. Preserve missing and currency; no unsafe sums.")

print("Rows 06-07 done")
json.dump(d,open(SRC,'w'),indent=2)
