"""Adapter and type-preserving contribution regression tests; no model API."""
import ast
import json
from pathlib import Path
import tempfile
import unittest
from benchmark_suite import Builder
from compiler_configuration import PROFILE_SHA256, configuration
from unified_graph_contract import prepare, validate_candidate
from component_contract import reconstruct_request, validate_program

class RemainingWitnessTests(unittest.TestCase):
    def request(self, feature):
        b=Builder([(2,)]);model=b.finish(b.node("square",["input0"]))
        return prepare(model,PROFILE_SHA256,construction_profile="hecate-unified-public-v1",
                       construction="unified-public-"+feature)
    def source(self,body):
        return '@hc.func("c,c")\ndef golden(x, zero_ct):\n'+''.join('    '+line+'\n' for line in body.splitlines())
    def check(self,r,body):
        return validate_candidate(dict(schema=1,request_id=r["request_id"],hecate_source=self.source(body)),r)
    def test_values_keep_length_and_symbolic_cells(self):
        r=self.request("call-values")
        self.check(r,"d={'a':x*x,'b':zero_ct,'c':zero_ct,'d':zero_ct,'e':zero_ct,'f':zero_ct}\nv=list(d.values())\nreturn [v[0]+v[5]]")
        with self.assertRaisesRegex(ValueError,"Missing contributing"):
            self.check(r,"d={'a':x*x}\nunused=d.values()\nreturn [x*x]")
    def test_mapping_write_keeps_inserted_key(self):
        r=self.request("counter-mapping_writes")
        self.check(r,"d={}\nd[0]=x*x\nreturn [d[0]]")
        with self.assertRaisesRegex(ValueError,"Missing contributing"):
            self.check(r,"d={}\nd[0]=x*x\nreturn [x*x]")
    def test_slice_write_keeps_cipher_element_type(self):
        for feature in ("counter-slice_writes","slice-write"):
            r=self.request(feature)
            self.check(r,"a=[None]\na[0:1]=[x*x]\nreturn [a[0]]")
            with self.assertRaisesRegex(ValueError,"Missing contributing"):
                self.check(r,"a=[None]\na[0:1]=[x*x]\nreturn [x*x]")
    def test_sort_key_order_and_commutative_negative(self):
        r=self.request("counter-sort_key_calls")
        self.check(r,"a=[x*x,zero_ct]\nidx=sorted([0,1],key=lambda i:i)\nreturn [a[idx[0]]]")
        with self.assertRaisesRegex(ValueError,"Missing contributing"):
            self.check(r,"a=[x*x,zero_ct]\nidx=sorted([0,1],key=lambda i:i)\nreturn [a[idx[0]]+a[idx[1]]]")
    def test_starred_sequence_arguments(self):
        r=self.request("counter-starred_expansions")
        self.check(r,"a=[[1.0,0.0],[0.0,1.0]]\nv=list(zip(*a))\nreturn [x*x*v[0][0]]")
        with self.assertRaisesRegex(ValueError,"Missing contributing"):
            self.check(r,"a=[[1.0,0.0],[0.0,1.0]]\nunused=zip(*a)\nreturn [x*x]")
    def test_list_comprehension_preserves_cipher_cells(self):
        r=self.request("node-ListComp")
        self.check(r,"a,b=[v for v in [x*x,zero_ct]]\nreturn [a+b]")
        with self.assertRaisesRegex(ValueError,"Missing contributing"):
            self.check(r,"unused=[v for v in [x*x,zero_ct]]\nreturn [x*x]")
    def test_mapping_update_keeps_keys_and_write_contribution(self):
        r=self.request("counter-mapping_writes")
        self.check(r,"d={}\nd.update({'a':x*x})\nreturn [d['a']]")
        with self.assertRaisesRegex(ValueError,"Missing contributing"):
            self.check(r,"d={}\nd.update({'a':x*x})\nreturn [x*x]")
    def test_omitted_construct_has_explicit_reason(self):
        r=self.request("counter-membership_tests")
        out=validate_program(dict(schema=1,request_id=r["request_id"],hecate_source=self.source("return [x*x]")),r)
        self.assertEqual(out["failure"]["code"],"required_construct_not_executed")
        self.assertEqual(out["failure"]["construction_evidence"]["reason"],"not_executed")

class AdapterTests(unittest.TestCase):
    def context(self,folder,verify):
        from validation_adapter import ValidationContext
        return ValidationContext(Path(folder),{},dict(attempts=[]),{},dict(atol=1e-5,rtol=1e-4),45,
                Path(folder)/"keys",True,lambda *a,**k:self.fail("Native process reached for malformed input"),verify)
    def test_parse_failure_and_integrity_are_independent(self):
        from validation_adapter import CandidateValidationAdapter
        with tempfile.TemporaryDirectory() as folder:
            checks=[];ctx=self.context(folder,lambda:checks.append(True))
            feedback=CandidateValidationAdapter(ctx).evaluate("not-json",0)
            self.assertEqual(feedback["layer"],"response_parse")
            self.assertEqual(len(checks),2)
            self.assertFalse(ctx.report["attempts"][0]["executed"])
            self.assertTrue((Path(folder)/"attempt-00/report.json").is_file())
        with tempfile.TemporaryDirectory() as folder:
            def broken():raise ValueError("Frozen context changed")
            feedback=CandidateValidationAdapter(self.context(folder,broken)).evaluate("{}",0)
            self.assertEqual(feedback["layer"],"integrity")
            self.assertEqual(feedback["category"],"integrity")

class CompositionTests(unittest.TestCase):
    def fixture(self,construction="unified-native-call-scalar_cipher",variant=0):
        from capability_combination_cases import cases
        return next(v for v in cases() if v["construction"]==construction and v["variant"]==variant)
    def test_nine_compositions_and_both_witnesses(self):
        from capability_combination_cases import cases
        for row in cases():
            with self.subTest(name=row["name"]):
                r=row["request"];self.assertEqual(reconstruct_request(r),r)
                checked=validate_candidate(row["candidate"],r)
                self.assertIn("construction_exercise",checked);self.assertIn("upstream_exercise",checked)
    def test_unopted_and_public_chunk_compositions_rejected(self):
        row=self.fixture();r=row["request"]
        from capability_combinations import NAME,HELPER_PROFILE
        base=dict(construction=row["construction"],helper_profile=HELPER_PROFILE,helper_exercise=["HE_BN0"])
        with self.assertRaises(ValueError):prepare(r["model"],PROFILE_SHA256,**base)
        for extra in (dict(construction_profile="hecate-unified-public-v1"),dict(chunk_period=4),dict(helper_exercise=[])):
            with self.assertRaises(ValueError):prepare(r["model"],PROFILE_SHA256,capability_composition=NAME,**(base|extra))
    def test_unused_native_and_unused_helper_do_not_cross_credit(self):
        row=self.fixture();r=row["request"]
        s=row["candidate"]["hecate_source"]
        with self.assertRaisesRegex(ValueError,"Missing contributing"):
            validate_candidate(dict(row["candidate"],hecate_source=s.replace("h=invoke(x)","unused=invoke(x)\n    h=HE_BN0(x)")),r)
        # Keep a contributing native helper while removing the actual BN call.
        with self.assertRaisesRegex(ValueError,"reachable upstream"):
            validate_candidate(dict(row["candidate"],hecate_source=s.replace("return HE_BN0(v)","return v")),r)
    def test_other_upstream_callee_is_not_silently_combined(self):
        row=self.fixture()
        with self.assertRaisesRegex(ValueError,"outside BN composition"):
            validate_candidate(dict(row["candidate"],hecate_source=row["candidate"]["hecate_source"].replace("HE_BN0(v)","HE_SiLU(v)")),row["request"])

class QualificationStageTests(unittest.TestCase):
    def test_numeric_failure_retains_completed_execution(self):
        from component_backend import observed_stages
        out = observed_stages({"status":"failed","attempts":[dict(
            trace={"frontend":"real_Hecate"},compiled=True,executed=True,
            comparison={"passed":False},failure_layer="numerical_comparison")]})
        self.assertTrue(out["encrypted_execution"])
        self.assertTrue(out["numerically_compared"])
        self.assertTrue(out["evidence_integrity_not_rejected"])

    def test_artifact_failure_and_integrity_are_distinct(self):
        from component_backend import observed_stages
        out = observed_stages({"attempts":[dict(compiled=True,executed=False,failure_layer="artifact_gate")]})
        self.assertTrue(out["compiled"])
        self.assertFalse(out["encrypted_execution"])
        self.assertFalse(out["numerically_compared"])
        rejected = observed_stages({"failure_layer":"integrity","attempts":[dict(executed=True)]})
        self.assertFalse(rejected["evidence_integrity_not_rejected"])
        self.assertFalse(observed_stages({})["encrypted_execution"])

if __name__=="__main__":unittest.main()
