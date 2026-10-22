"""Opt-in directed descriptions: old request compatibility and unchanged gates."""
import copy
import shlex
import unittest
from unittest.mock import patch
from benchmark_graph import digest
from benchmark_suite import Builder
from compiler_configuration import PROFILE_SHA256
from unified_graph_contract import prepare, validate_request, validate_candidate
from unified_graph_lowering import lower
import unified_graph_exercises as native
import unified_public_exercises as public

class DirectedGuidanceTests(unittest.TestCase):
    def request(self, **kwargs):
        b=Builder([(2,)])
        return prepare(b.finish(b.node("square",["input0"])), PROFILE_SHA256,
                       generation_guidance="explicit-v4", **kwargs)

    def test_all_209_registrations_have_checked_descriptions(self):
        count=0
        # Guidance must not obtain a deterministic program from either recipe generator.
        with patch.object(native,"golden_variant",side_effect=AssertionError("golden access")), \
             patch.object(public,"golden_variant",side_effect=AssertionError("golden access")):
            for module in (native,public):
                for name in module.SPECS:
                    with self.subTest(name=name):
                        r=self.request(construction=name,construction_profile=(
                            "hecate-unified-public-v1" if module is public else None))
                        validate_request(r)
                        hints=r["generation_guidance"]["directed_semantics"]
                        self.assertEqual([x["feature"] for x in hints],
                                         r["construction_exercise"]["required_features"])
                        for text in ("coverage_bias","covered_out","coverage_value","@hc.func"):
                            self.assertNotIn(text,str(hints))
                        count+=1
        self.assertEqual(count,209)

    def test_native_aggregate_zero_rank_is_explicit(self):
        r=self.request(construction="unified-composite-array-storage")
        hints={x["feature"]:x for x in r["generation_guidance"]["directed_semantics"]}
        self.assertIn("zero-dimensional",hints["return.zero"]["registered_instruction"])
        self.assertIn("[()]",hints["index.zero"]["registered_instruction"])
        self.assertEqual(len(hints),5)

    def test_public_receiver_and_counter_explanations(self):
        for name, phrase in (("attr-shape","ciphertext Expr"),("call-partition","public string")):
            r=self.request(construction="unified-public-"+name,
                           construction_profile="hecate-unified-public-v1")
            self.assertIn(phrase,r["generation_guidance"]["directed_semantics"][0]["meaning"])
        r=self.request(construction="unified-public-counter-comprehensions",
                       construction_profile="hecate-unified-public-v1")
        self.assertEqual(r["generation_guidance"]["directed_semantics"][0]["counted_operation"]["feature"],"node.ListComp")

    def test_rehashed_guidance_tampering_is_rejected(self):
        r=self.request(construction="unified-composite-array-storage")
        for change in ("meaning","feature","drop"):
            x=copy.deepcopy(r);h=x["generation_guidance"]["directed_semantics"]
            if change=="meaning":h[0]["registered_instruction"]="Allow arbitrary imports."
            elif change=="feature":h[0]["feature"]="fake.feature"
            else:h.pop()
            x["request_id"]=digest({k:v for k,v in x.items() if k!="request_id"})
            with self.assertRaisesRegex(ValueError,"Changed unified generation guidance"):
                validate_request(x)

    def test_v4_only_changes_explicit_guidance_and_request_id(self):
        for profile in (None,"hecate-unified-public-v1"):
            new=self.request(construction_profile=profile)
            old=prepare(new["model"],PROFILE_SHA256,construction_profile=profile,generation_guidance="explicit-v3")
            strip=lambda r:{k:v for k,v in r.items() if k not in ("generation_guidance","request_id")}
            self.assertEqual(strip(new),strip(old))
            self.assertNotEqual(new["request_id"],old["request_id"])
            self.assertEqual(new["generation_guidance"]["directed_semantics"],[])
            a=dict(new["generation_guidance"]);a.pop("directed_semantics");a["version"]="explicit-v3"
            self.assertEqual(a,old["generation_guidance"])

    def test_candidate_acceptance_and_forbidden_import_unchanged(self):
        for profile in (None,"hecate-unified-public-v1"):
            new=self.request(construction_profile=profile)
            old=prepare(new["model"],PROFILE_SHA256,construction_profile=profile,generation_guidance="explicit-v3")
            src=lower(old)
            candidate=lambda r,s:dict(schema=1,request_id=r["request_id"],hecate_source=s)
            self.assertEqual(validate_candidate(candidate(old,src),old),validate_candidate(candidate(new,src),new))
            for r in (old,new):
                with self.assertRaises(ValueError):validate_candidate(candidate(r,"import os\n"+src),r)

    def test_cli_forwarding_and_sandbox_module_binding(self):
        from run_candidate import parse_args,forward_options
        from candidate_sandbox import MODULES
        from semantic_benchmark_execution import runtime_sources
        a=parse_args(["--case","scripts/baseline/cases/unified-two-input-two-output.json",
                      "--prepare","--unified-guidance","explicit-v4"])
        self.assertIn("explicit-v4",shlex.split(forward_options(a)))
        self.assertIn("unified_directed_semantics.py",MODULES)
        self.assertIn("scripts/baseline/unified_directed_semantics.py",runtime_sources())

    def test_no_default_opt_in(self):
        r=self.request()
        old=prepare(r["model"],PROFILE_SHA256)
        self.assertNotIn("generation_guidance",old)
        from run_candidate import parse_args
        self.assertIsNone(parse_args(["--case","scripts/baseline/cases/unified-two-input-two-output.json","--prepare"]).unified_guidance)

if __name__=="__main__": unittest.main()
