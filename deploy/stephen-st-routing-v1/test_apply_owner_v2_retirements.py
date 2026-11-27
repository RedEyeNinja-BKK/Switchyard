import importlib.util
from pathlib import Path
import tomllib
import unittest
M=Path(__file__).with_name('apply_owner_v2_retirements.py')
s=importlib.util.spec_from_file_location('owner_v2',M)
v2=importlib.util.module_from_spec(s); s.loader.exec_module(v2)
BASE='''
[targets.openrouter-openrouter-free]
id="openrouter/free"
[targets.openrouter-qwen3_7-flash]
id="qwen/qwen3.7-flash"
[targets.deepseek]
id="deepseek"
[routes.direct-openrouter-openrouter-free]
id="direct-openrouter-openrouter-free"
type="passthrough"
target="openrouter-openrouter-free"
candidates=[]
[routes.direct-openrouter-qwen3_7-flash]
id="direct-openrouter-qwen3_7-flash"
type="passthrough"
target="openrouter-qwen3_7-flash"
candidates=[]
[routes.vc-paid-core]
id="vc-paid-core"
type="auto"
capable_target="deepseek"
[routes.smartfree]
id="smartfree"
type="passthrough"
target="deepseek"
candidates=["openrouter-openrouter-free", "local-safe"]
[routes.direct-openrouter-qwen3_7-flash-nt]
id="direct-openrouter-qwen3_7-flash-nt"
type="passthrough"
target="deepseek"
candidates=[]
[composites.VC_Paid]
children=["vc-paid-core", "direct-openrouter-openrouter-free"]
cross_on="failure_class"
[composites.ST_Route]
children=["smartfree"]
'''
class Tests(unittest.TestCase):
 def test_clean_exact_deltas_only(self):
  out,receipt=v2.apply(BASE)
  a=tomllib.loads(out)
  self.assertNotIn('VC_Paid',a['composites'])
  self.assertNotIn('openrouter-openrouter-free',a['targets'])
  self.assertNotIn('direct-openrouter-qwen3_7-flash',a['routes'])
  self.assertIn('direct-openrouter-qwen3_7-flash-nt',a['routes'])
  self.assertEqual(a['routes']['smartfree']['candidates'],['local-safe'])
  self.assertEqual(receipt['before']['routes']-receipt['after']['routes'],3)
 def test_refuses_other_consumer(self):
  s=BASE.replace('[composites.ST_Route]\nchildren=["smartfree"]','[composites.ST_Route]\nchildren=["direct-openrouter-openrouter-free"]')
  with self.assertRaisesRegex(v2.Refused,'REFUSED_RETIRED_CONSUMER'):v2.apply(s)
 def test_refuses_target_used_elsewhere(self):
  s=BASE.replace('target="deepseek"\ncandidates=[]\n[composites.VC_Paid]', 'target="openrouter-qwen3_7-flash"\ncandidates=[]\n[composites.VC_Paid]')
  with self.assertRaisesRegex(v2.Refused,'REFUSED_RETIRED_CONSUMER'):v2.apply(s)
 def test_refuses_missing_exact(self):
  with self.assertRaisesRegex(v2.Refused,'REFUSED_MISSING_EXACT_ID'):v2.apply(BASE.replace('vc-paid-core','vc-paid-gone'))
 def test_no_downgrade_to_free_pool_empty(self):
  s=BASE.replace('"openrouter-openrouter-free", "local-safe"','"openrouter-openrouter-free"')
  with self.assertRaisesRegex(v2.Refused,'REFUSED_SMARTFREE_NO_CANDIDATES_LEFT'):v2.apply(s)
 def test_refuses_candidate_rewritten_to_other_target(self):
  s=BASE.replace('target="openrouter-qwen3_7-flash"','target="deepseek"')
  with self.assertRaisesRegex(v2.Refused,'REFUSED_DIRECT_NOT_1TO1'):v2.apply(s)
 def test_refuses_ambiguous_multiline_candidates(self):
  s=BASE.replace('candidates=["openrouter-openrouter-free", "local-safe"]','candidates=[\n "openrouter-openrouter-free",\n "local-safe"\n]')
  with self.assertRaisesRegex(v2.Refused,'REFUSED_SMARTFREE_CANDIDATES_NOT_SINGLE_LINE'):v2.apply(s)
 def test_refuses_extra_use_of_vc_paid(self):
  s=BASE.replace('children=["smartfree"]','children=["VC_Paid"]')
  with self.assertRaisesRegex(v2.Refused,'REFUSED_RETIRED_CONSUMER'):v2.apply(s)
if __name__=='__main__':unittest.main()
