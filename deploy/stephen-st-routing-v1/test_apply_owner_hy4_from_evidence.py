import importlib.util
from pathlib import Path
import unittest, tomllib
spec=importlib.util.spec_from_file_location('hy4',Path(__file__).with_name('apply_owner_hy4_from_evidence.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
BASE='''[llm_clients.openrouter]\nformat="openai_responses"\napi_key_env="OWNER_OR_KEY"\n[routes.smart]\nid="smart"\ntype="passthrough"\ntarget="existing"\n[targets.existing]\nid="x"\nllm_client="openrouter"\n'''
PROOF={'account_client':'openrouter','upstream_id':'tencent/hy4-flash','wire_format':'openai_responses',
       'text_http_status':200,'tool_http_status':200,'json_http_status':200,'tool_calls':1,
       'structured_output_valid':True,'receipt':'2026-11-23 host path to safe transcript',
       'measured_at':'2026-11-23T07:00+07:00','cost_mode':'DIRECT_OPT_IN_ONLY',
       'route_context_window':262144,'max_input_tokens':245760}
class Tests(unittest.TestCase):
 def test_writes_one_target_one_route_only(self):
  out,meta=m.apply(BASE,PROOF)
  doc=tomllib.loads(out)
  self.assertEqual(doc['routes']['smart'],tomllib.loads(BASE)['routes']['smart'])
  self.assertTrue(doc['routes'][m.ROUTE]['tool_calling'])
  self.assertFalse(doc['routes'][m.ROUTE]['vision'])
  self.assertFalse(doc['routes'][m.ROUTE]['video'])
  self.assertFalse(doc['targets'][m.TARGET]['free_tier'])
  self.assertEqual(doc['targets'][m.TARGET]['id'],m.MODEL)
  self.assertTrue(meta['not_added_to_smart_paid_pool'])
 def test_reject_wrong_account(self):
  with self.assertRaisesRegex(m.Refused,'HY4_PROVIDER_OR_OWNER_MISMATCH'):
   m.apply(BASE,{**PROOF,'account_client':'openrouter_stephen'})
 def test_reject_wire_mismatch(self):
  with self.assertRaisesRegex(m.Refused,'HY4_WIRE_FORMAT_UNTESTED'):
   m.apply(BASE,{**PROOF,'wire_format':'openai_chat'})
 def test_reject_fake_tools(self):
  with self.assertRaisesRegex(m.Refused,'HY4_TOOL_OR_JSON_NOT_QUALIFIED'):
   m.apply(BASE,{**PROOF,'tool_calls':0})
 def test_reject_paid_fallback(self):
  with self.assertRaisesRegex(m.Refused,'HY4_PAID_AUTO_FALLBACK_FORBIDDEN'):
   m.apply(BASE,{**PROOF,'cost_mode':'AUTO'})
 def test_reject_context_overcap(self):
  with self.assertRaisesRegex(m.Refused,'HY4_CONTEXT_WINDOW_NOT_CONSERVATIVE'):
   m.apply(BASE,{**PROOF,'max_input_tokens':262144})
 def test_reject_false_image_claim(self):
  with self.assertRaisesRegex(m.Refused,'HY4_IMAGE_CARRIER_UNQUALIFIED'):
   m.apply(BASE,{**PROOF,'image_http_status':200})
 def test_positive_image_wire(self):
  doc=tomllib.loads(m.apply(BASE,{**PROOF,'image_http_status':200,'image_carrier':'inline'})[0])
  self.assertTrue(doc['routes'][m.ROUTE]['vision'])
  self.assertEqual(doc['targets'][m.TARGET]['image_carrier'],'inline')
 def test_reject_duplicate(self):
  first,_=m.apply(BASE,PROOF)
  with self.assertRaisesRegex(m.Refused,'HY4_ALREADY_DEFINED'):
   m.apply(first,PROOF)
if __name__=='__main__':unittest.main()
