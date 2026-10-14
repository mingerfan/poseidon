"""v9 compatibility, privacy and unchanged language limits; no paid calls."""
import copy
import unittest
from benchmark_suite import Builder
from benchmark_graph import digest
from compiler_configuration import PROFILE_SHA256
from unified_graph_contract import prepare,validate_request,validate_candidate,static_repair_hint
from unified_expression_guidance import repair_hint
from unified_graph_lowering import lower

class ExpressionGuidanceTests(unittest.TestCase):
    def request(self,version="explicit-v9",public=False):
        b=Builder([(3,)])
        g=b.finish(b.node("polynomial",["input0",b.const([.1,.2,.3])],basis="chebyshev"))
        return prepare(g,PROFILE_SHA256,generation_guidance=version,
                       construction_profile="hecate-unified-public-v1" if public else None)
    def test_v8_bytes_and_validation_remain_compatible(self):
        old=self.request("explicit-v8");new=self.request()
        restored=copy.deepcopy(new)
        restored["generation_guidance"].pop("expression_construction_rules")
        restored["generation_guidance"]["version"]="explicit-v8"
        restored["request_id"]=digest({k:v for k,v in restored.items() if k!="request_id"})
        self.assertEqual(restored,old)
        from component_contract import reconstruct_request
        from deepseek_provider import public_request
        self.assertEqual(reconstruct_request(new),new)
        self.assertEqual(public_request(new),new)
    def test_rehashed_relaxed_contract_rejected(self):
        q=self.request()
        q["generation_guidance"]["expression_construction_rules"]["native_helpers"]="Unlimited"
        q["request_id"]=digest({k:v for k,v in q.items() if k!="request_id"})
        with self.assertRaisesRegex(ValueError,"Changed unified generation guidance"):validate_request(q)
    def test_response_contains_no_model_answer_or_private_data(self):
        q=self.request();meta=q["generation_guidance"]["expression_construction_rules"]
        self.assertNotIn(lower(q),str(meta))
        for field in ("hecate_source","reference_outputs","test_inputs"):self.assertNotIn(field,meta)
    def test_parameter_limit_hint_and_actual_rejection(self):
        q=self.request();args=[f"a{i}" for i in range(17)]
        src='@hc.func("'+",".join(["c"]*17)+'")\ndef helper('+",".join(args)+'):\n    return a0\n'+lower(q)
        with self.assertRaisesRegex(ValueError,"parameter limit"):
            validate_candidate(dict(schema=1,request_id=q["request_id"],hecate_source=src),q)
        hint=static_repair_hint(src,q,"Native function parameter limit SECRET")
        self.assertIn("16 positional",hint);self.assertIn("lexical scope",hint);self.assertNotIn("SECRET",hint)
    def test_plain_and_slot_errors_are_specific(self):
        q=self.request();src=lower(q)
        self.assertIn("negate",static_repair_hint(src,q,"Native negation requires ciphertext SECRET"))
        self.assertIn("rotation",static_repair_hint(src,q,"Indexing requires a result container SECRET"))
        self.assertNotIn("SECRET",static_repair_hint(src,q,"Indexing requires a result container SECRET"))
        bad='@hc.func("c,c")\ndef golden(x,zero_ct):\n    return x[0]\n'
        with self.assertRaises(ValueError):
            validate_candidate(dict(schema=1,request_id=q["request_id"],hecate_source=bad),q)
    def test_compiler_feedback_public_fixed_messages_only(self):
        q=self.request();src='@hc.func("c,c")\ndef golden(x,zero_ct):\n    y=x*x\n    return y*2\n'
        hint=repair_hint(q,src,"/secret/path PRIVATE",stage="compiler")
        self.assertIn("doubling",hint);self.assertIn("SAME",hint)
        self.assertNotIn("PRIVATE",hint);self.assertNotIn("/secret",hint);self.assertNotIn(src,hint)
        self.assertEqual(repair_hint(self.request("explicit-v8"),src,stage="compiler"),"")
    def test_public_profile_does_not_gain_native_constraints(self):
        q=self.request(public=True);meta=q["generation_guidance"]["expression_construction_rules"]
        self.assertIn("undecorated",meta["native_helpers"])
        self.assertNotIn("16 positional",meta["native_helpers"])
        self.assertEqual(repair_hint(q,lower(q),"Native function parameter limit"),"")
    def test_malformed_oversize_and_unknown_diagnostics(self):
        q=self.request()
        for s in (None,"a"*65537,"def broken","\n".join(["x=1"]*1600)):
            self.assertEqual(repair_hint(q,s,"PRIVATE"),"")
        self.assertEqual(repair_hint(q,lower(q),"PRIVATE"),"")
    def test_cli_and_sandbox(self):
        from run_candidate import parse_args
        from candidate_sandbox import MODULES
        self.assertEqual(parse_args(["--prepare","--case","x","--unified-guidance","explicit-v9"]).unified_guidance,"explicit-v9")
        self.assertIn("unified_expression_guidance.py",MODULES)

if __name__=="__main__":unittest.main()
