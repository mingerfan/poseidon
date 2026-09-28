"""Regression gates for r177; no credentials, network or FHE claims."""
import copy,json,unittest
from unittest.mock import patch
import numpy as np
from benchmark_suite import Builder,generate
from benchmark_graph import digest,samples
from benchmark_math import evaluate
from benchmark_torch import evaluate as reference
from compiler_configuration import PROFILE_SHA256
from unified_graph_contract import prepare,validate_request,validate_candidate,static_repair_hint
from unified_graph_lowering import lower,candidate_source
from unified_precise_guidance import FEATURE_RULES
from test_packed_prefix_lowering import public_execute
import test_provider_retries
from test_deepseek_provider import completion
from deepseek_provider import retryable_failure,ProviderError

class PreciseGuidanceTests(unittest.TestCase):
 def request(self,exercise=None,version="explicit-v7"):
  b=Builder([(2,)]);g=b.finish(b.node("square",["input0"]))
  return prepare(g,PROFILE_SHA256,construction_profile="hecate-unified-public-v1",
   construction=exercise,generation_guidance=version)
 def test_v6_metadata_preserved(self):
  old=self.request(version="explicit-v6");new=self.request()
  a=copy.deepcopy(new);a["generation_guidance"].pop("precise_construction_rules")
  a["generation_guidance"]["version"]="explicit-v6"
  a["request_id"]=digest({k:v for k,v in a.items() if k!="request_id"})
  self.assertEqual(a,old)
 def test_sorted_roundtrip_and_public_export(self):
  from deepseek_provider import public_request
  r=self.request();validate_request(json.loads(json.dumps(r,sort_keys=True)))
  self.assertEqual(public_request(r),r)
  self.assertNotIn(candidate_source(r)[0],str(r["generation_guidance"]))
 def test_rehashed_changed_rules_rejected(self):
  r=self.request();r["generation_guidance"]["precise_construction_rules"]["identifier_rule"]="allow any"
  r["request_id"]=digest({k:v for k,v in r.items() if k!="request_id"})
  with self.assertRaises(ValueError):validate_request(r)
 def test_every_new_feature_registered_and_scoped(self):
  import unified_public_exercises as public
  for feature,text in FEATURE_RULES.items():
   matches=[public.spec(k) for k in public.SPECS if public.spec(k)["required_features"]==[feature]]
   self.assertTrue(matches,feature);r=self.request(matches[0]["id"])
   self.assertEqual(r["generation_guidance"]["precise_construction_rules"]["directed_acceptance"],
                    [dict(feature=feature,acceptance=text)])
 def test_binding_feedback_does_not_echo_diagnostic(self):
  r=self.request();s=candidate_source(r)[0]
  h=static_repair_hint(s,r,"Read-only or invalid binding PRIVATE")
  self.assertIn("Leading underscore",h);self.assertNotIn("PRIVATE",h)
  old=self.request(version="explicit-v6")
  self.assertEqual(static_repair_hint(s,old,"Read-only or invalid binding"),"")
 def test_invalid_names_and_scalarized_zero_dim_still_rejected(self):
  r=self.request()
  for body in ["_ = x; return [x]", "_i = 1; return [x]", "zero_ct = x; return [x]"]:
   with self.assertRaises(ValueError):
    validate_candidate(dict(schema=1,request_id=r["request_id"],
     hecate_source='@hc.func("c,c")\ndef golden(x, zero_ct):\n    '+body+'\n'),r)
  r=self.request("unified-public-zero_dim")
  source='@hc.func("c,c")\ndef golden(x, zero_ct):\n    a=np.array(x,dtype=object)\n    b=np.array(0.5,dtype=object)\n    y=a[()]+b[()]\n    return [y]\n'
  with self.assertRaisesRegex(ValueError,"Missing contributing"):
   validate_candidate(dict(schema=1,request_id=r["request_id"],hecate_source=source),r)
 def test_for_else_noop_still_rejected(self):
  r=self.request("unified-public-event-for_else")
  s='@hc.func("c,c")\ndef golden(x,zero_ct):\n    y=x\n    for i in range(1):\n        y=x\n    else:\n        y=x\n    return [y]\n'
  with self.assertRaisesRegex(ValueError,"Missing contributing"):
   validate_candidate(dict(schema=1,request_id=r["request_id"],hecate_source=s),r)
 def test_cli_sandbox_and_no_unknown_feedback(self):
  from run_candidate import parse_args
  from candidate_sandbox import MODULES
  self.assertEqual(parse_args(["--prepare","--case","x","--unified-guidance","explicit-v7"]).unified_guidance,"explicit-v7")
  self.assertIn("unified_precise_guidance.py",MODULES)
  r=self.request();self.assertEqual(static_repair_hint(candidate_source(r)[0],r,"SECRET"),"")

class PackedBalancedTests(unittest.TestCase):
 def test_original_silu_models_two_independent_references(self):
  count=0
  for row in generate():
   g=row["model"]
   if g["id"] not in {"bench_helper_%04d"%i for i in range(112,120)}:continue
   count+=1;r=prepare(g,PROFILE_SHA256);before=copy.deepcopy(r)
   s=lower(r,packed_prefix=True,balanced_chebyshev=True)
   validate_candidate(dict(schema=1,request_id=r["request_id"],hecate_source=s),r)
   for x in samples(g,16):
    a=public_execute(s,r,x)[0][:np.prod(g["inputs"][0]["shape"])]
    b=evaluate(g,x)["output0"].reshape(-1)
    np.testing.assert_allclose(a,b,atol=1e-12,rtol=1e-12)
    np.testing.assert_allclose(b,reference(g,x)["output0"].reshape(-1),atol=1e-12,rtol=1e-12)
   self.assertEqual(before,r)
  self.assertEqual(count,8)
 def test_polynomial_padding_does_not_leak_into_reduction(self):
  for size in (1,3,5,17):
   b=Builder([(size,)]);p=b.node("polynomial",["input0",b.const([.25,.125,-.0625,1e-38])],basis="chebyshev")
   g=b.finish(b.node("mean",[p],axes=[0],keepdims=False))
   r=prepare(g,PROFILE_SHA256);s=lower(r,packed_prefix=True,balanced_chebyshev=True)
   validate_candidate(dict(schema=1,request_id=r["request_id"],hecate_source=s),r)
   for x in samples(g,4):
    np.testing.assert_allclose(public_execute(s,r,x)[0][:1],evaluate(g,x)["output0"],atol=1e-12,rtol=1e-12)

class ConnectEOFTests(unittest.TestCase):
 def test_exact_retry_classification(self):
  good=dict(stage="connect",tls_error="unexpected_eof",tls_reason="UNEXPECTED_EOF_WHILE_READING")
  self.assertTrue(retryable_failure("transport_tls_failed",good))
  for change in (dict(stage="body"),dict(tls_error="certificate_verification"),
   dict(tls_error="protocol_error"),dict(tls_reason="DECRYPTION_FAILED_OR_BAD_RECORD_MAC"),
   dict(tls_reason="unknown")):
   self.assertFalse(retryable_failure("transport_tls_failed",{**good,**change}))
  self.assertFalse(retryable_failure("transport_certificate_failed",good))
  self.assertFalse(retryable_failure("transport_tls_failed",{}))
 @patch("deepseek_provider.time.sleep")
 def test_same_payload_success_and_four_attempt_ceiling(self,sleep):
  fixture=test_provider_retries.RetryTests();fixture.setUp()
  d=dict(stage="connect",tls_error="unexpected_eof",tls_reason="UNEXPECTED_EOF_WHILE_READING")
  p=fixture.provider([ProviderError("transport_tls_failed"),completion("{}")],d)
  self.assertEqual(p.generate(fixture.request,[]),"{}")
  self.assertEqual(fixture.transport.sent[0],fixture.transport.sent[1])
  p=fixture.provider([ProviderError("transport_tls_failed")]*4,d)
  with self.assertRaises(ProviderError):p.generate(fixture.request,[])
  self.assertEqual(len(fixture.transport.sent),4)
  with self.assertRaises(ProviderError):p.generate(fixture.request,[])
  self.assertEqual(len(fixture.transport.sent),4)

if __name__=="__main__":unittest.main()
