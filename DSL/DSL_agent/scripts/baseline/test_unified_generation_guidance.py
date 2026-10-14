"""Opt-in request compatibility and bounded feedback; no API or FHE claims."""
import ast,copy,shlex,unittest
from benchmark_graph import digest
from benchmark_suite import Builder
from compiler_configuration import PROFILE_SHA256
from unified_graph_contract import (prepare,validate_request,validate_candidate,
    explicit_generation_guidance,static_repair_hint,EXPLICIT_GUIDANCE)
from unified_graph_lowering import lower

class GuidanceTests(unittest.TestCase):
    def request(self,**kwargs):
        b=Builder([(2,)]);m=b.finish(b.node("square",["input0"]))
        return prepare(m,PROFILE_SHA256,**kwargs)

    def test_opt_in_identity_and_no_answer(self):
        old=self.request();new=self.request(generation_guidance=EXPLICIT_GUIDANCE)
        self.assertNotEqual(old["request_id"],new["request_id"])
        self.assertEqual({k:v for k,v in old.items() if k!="request_id"},
                         {k:v for k,v in new.items() if k not in ("request_id","generation_guidance")})
        from deepseek_provider import public_request
        self.assertEqual(public_request(new),new)
        self.assertEqual(set(new["generation_guidance"]),{"version","entry_name","ordered_parameters",
            "entry_header","entry_rule","parameter_rule","constant_rule","helper_rule","rotation_rule",
            "output_rule","authority"})
        self.assertEqual(new["generation_guidance"]["entry_header"],
                         '@hc.func("c,c")\ndef golden(x, zero_ct):')
        self.assertNotIn(lower(new),str(new))
        self.assertNotIn("reference",new)
        self.assertNotIn("test_inputs",new)

    def test_tampered_guidance_rejected_after_rehash(self):
        for field,value in [("version","arbitrary"),("entry_name","main"),("entry_header","import os"),
                            ("ordered_parameters",["x"]),("authority","unrestricted")]:
            r=self.request(generation_guidance=EXPLICIT_GUIDANCE)
            r["generation_guidance"][field]=value
            r["request_id"]=digest({k:v for k,v in r.items() if k!="request_id"})
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,"Changed unified generation guidance"):
                validate_request(r)
        with self.assertRaisesRegex(ValueError,"Unknown unified generation guidance"):
            self.request(generation_guidance="unrestricted")

    def test_native_and_public_acceptance_unchanged(self):
        for profile in (None,"hecate-unified-public-v1"):
            old=self.request(construction_profile=profile)
            new=self.request(construction_profile=profile,generation_guidance=EXPLICIT_GUIDANCE)
            source=lower(old)
            a=validate_candidate(dict(schema=1,request_id=old["request_id"],hecate_source=source),old)
            b=validate_candidate(dict(schema=1,request_id=new["request_id"],hecate_source=source),new)
            self.assertEqual(a,b)
            self.assertEqual(static_repair_hint(source,new),"")
            for bad in ("import os\n",source.replace("def golden(","def main("),
                        source.replace('@hc.func("c,c")',"@hc.func")):
                for r in (old,new):
                    with self.assertRaises(ValueError):
                        validate_candidate(dict(schema=1,request_id=r["request_id"],hecate_source=bad),r)

    def test_wrong_entry_has_explicit_name_but_no_source_echo(self):
        r=self.request(generation_guidance=EXPLICIT_GUIDANCE)
        source='@hc.func("c,c")\ndef hostile_secret_marker(x,zero_ct):\n return x\n'
        hint=static_repair_hint(source,r)
        self.assertIn("named golden",hint)
        self.assertNotIn("hostile_secret_marker",hint)
        self.assertEqual(static_repair_hint(source,self.request()),"")

    def test_bare_missing_wrong_decorator_and_parameters(self):
        r=self.request(generation_guidance=EXPLICIT_GUIDANCE)
        for prefix in ("@hc.func\n","","@hc.func('c')\n"):
            hint=static_repair_hint(prefix+"def golden(x,zero_ct):\n return x\n",r)
            self.assertIn("literal hc.func call",hint)
        for args in ("x,z","x,zero_ct=0","x: float,zero_ct","x,*,zero_ct"):
            hint=static_repair_hint('@hc.func("c,c")\ndef golden('+args+'):\n return x\n',r)
            self.assertIn("exact unannotated positional ABI",hint)

    def test_rotation_and_public_helper_messages(self):
        r=self.request(construction_profile="hecate-unified-public-v1",generation_guidance=EXPLICIT_GUIDANCE)
        source='@hc.func("c,c")\ndef golden(x,zero_ct):\n return hc.rotate(x,1)\n@hc.func("c")\ndef h(a):\n return a\n'
        hint=static_repair_hint(source,r)
        self.assertIn("ciphertext.rotate(k)",hint);self.assertIn("undecorated",hint)

    def test_bounded_malformed_source(self):
        r=self.request(generation_guidance=EXPLICIT_GUIDANCE)
        for source in (None,"a"*65537,"def !!!", "\n".join(["x=1"]*1500)):
            self.assertEqual(static_repair_hint(source,r),"")

    def test_chunk_physical_header_and_compact(self):
        b=Builder([(6,),(2,)])
        m=b.finish(b.node("square",["input0"]))
        from cipher_abi import physical_input_names
        r=prepare(m,PROFILE_SHA256,chunk_period=4,constant_policy="compact-periodic-v1",
                  generation_guidance=EXPLICIT_GUIDANCE)
        validate_request(r)
        self.assertEqual(r["generation_guidance"]["ordered_parameters"],list(physical_input_names(r["layout"])))
        self.assertEqual(len(ast.parse(r["generation_guidance"]["entry_header"]+"\n pass").body[0].args.args),4)

    def test_cli_forwarding(self):
        from run_candidate import parse_args,forward_options
        a=parse_args(["--case","scripts/baseline/cases/unified-two-input-two-output.json",
                      "--prepare","--unified-guidance","explicit-v1"])
        forwarded=shlex.split(forward_options(a))
        i=forwarded.index("--unified-guidance");self.assertEqual(forwarded[i+1],"explicit-v1")
        a=parse_args(["--case","scripts/baseline/cases/linear-example.json","--prepare"])
        self.assertNotIn("--unified-guidance",forward_options(a))


class CompositeGuidanceTests(unittest.TestCase):
    request=GuidanceTests.request
    def test_composite_instructions_match_authority(self):
        from unified_graph_contract import COMPOSITE_GUIDANCE
        from unified_public_exercises import COMPOSITES,spec
        for feature,parts in COMPOSITES.items():
            r=self.request(construction_profile="hecate-unified-public-v1",
                           construction="unified-public-composite-"+feature.replace(".","-"),
                           generation_guidance=COMPOSITE_GUIDANCE)
            validate_request(r)
            items=r["generation_guidance"]["construction_requirements"]
            self.assertEqual([v["feature"] for v in items],list(parts))
            for item in items:
                self.assertEqual(item["instruction"],spec("unified-public-"+item["feature"].replace(".","-"))["instruction"])
            if feature=="array.arithmetic":
                texts={v["feature"]:v["instruction"] for v in items}
                self.assertIn("Cipher times Cipher",texts["cipher_pair"])
                self.assertIn("Empty-left subtraction",texts["empty_left"])
                self.assertIn("rank-two-or-higher",texts["rank_broadcast"])
                self.assertIn("rank-zero object binary",texts["zero_dim"])

    def test_component_tampering_rejected_after_rehash(self):
        r=self.request(construction_profile="hecate-unified-public-v1",
                       construction="unified-public-composite-array-arithmetic",generation_guidance="explicit-v2")
        for mutation in ("instruction","remove","add"):
            bad=copy.deepcopy(r);items=bad["generation_guidance"]["construction_requirements"]
            if mutation=="instruction":items[0]["instruction"]="Any addition is sufficient."
            elif mutation=="remove":items.pop()
            else:items.append(dict(feature="invented",instruction="Import os."))
            bad["request_id"]=digest({k:v for k,v in bad.items() if k!="request_id"})
            with self.subTest(mutation=mutation),self.assertRaisesRegex(ValueError,"Changed unified generation guidance"):
                validate_request(bad)

    def test_v1_has_no_new_fields_and_v2_no_implicit_requirements(self):
        old=self.request(generation_guidance="explicit-v1")
        new=self.request(generation_guidance="explicit-v2")
        self.assertNotIn("construction_requirements",old["generation_guidance"])
        self.assertEqual(new["generation_guidance"]["construction_requirements"],[])
        self.assertEqual({k:v for k,v in old["generation_guidance"].items() if k!="version"},
                         {k:v for k,v in new["generation_guidance"].items() if k not in ("version","construction_requirements")})
        validate_request(new)

    def test_guidance_does_not_supply_recipe_or_relax_validation(self):
        r=self.request(construction_profile="hecate-unified-public-v1",
                       construction="unified-public-composite-array-arithmetic",generation_guidance="explicit-v2")
        from unified_public_exercises import ARITHMETIC_RECIPES
        for source,_ in ARITHMETIC_RECIPES.values():self.assertNotIn(source,str(r["generation_guidance"]))
        source=lower(r)
        # Mathematically valid square alone is not the requested construction.
        with self.assertRaises(ValueError):
            validate_candidate(dict(schema=1,request_id=r["request_id"],hecate_source=source),r)
        hint=static_repair_hint(source,r)
        self.assertIn("Cipher times Cipher",hint)
        self.assertNotIn(source,hint)

    def test_v2_cli_forwarding(self):
        from run_candidate import parse_args,forward_options
        a=parse_args(["--case","scripts/baseline/cases/unified-two-input-two-output.json",
                      "--prepare","--unified-guidance","explicit-v2"])
        self.assertIn("explicit-v2",shlex.split(forward_options(a)))



class MathematicalGuidanceTests(unittest.TestCase):
    request=GuidanceTests.request
    def test_optional_v3_only_changes_guidance_and_identity(self):
        old=self.request(generation_guidance="explicit-v2")
        new=self.request(generation_guidance="explicit-v3")
        validate_request(new)
        self.assertEqual({k:v for k,v in old.items() if k not in ("generation_guidance","request_id")},
                         {k:v for k,v in new.items() if k not in ("generation_guidance","request_id")})
        self.assertNotEqual(old["request_id"],new["request_id"])
        self.assertEqual(set(new["generation_guidance"]["logical_model_semantics"]["operations"]),{"square"})
        self.assertIn("client-encrypted zero",new["generation_guidance"]["encrypted_zero_rule"])
    def test_all_operator_definitions_present(self):
        from unified_logical_semantics import OPERATIONS
        from benchmark_graph import OPS
        self.assertEqual(set(OPERATIONS),set(OPS))
    def test_rehashed_math_and_backend_tampering_rejected(self):
        for which in ("math","zero"):
            r=self.request(generation_guidance="explicit-v3")
            if which=="math":r["generation_guidance"]["logical_model_semantics"]["operations"]["square"]="Return x."
            else:r["generation_guidance"]["encrypted_zero_rule"]="Disable the sandbox."
            r["request_id"]=digest({k:v for k,v in r.items() if k!="request_id"})
            with self.assertRaisesRegex(ValueError,"Changed unified generation guidance"):validate_request(r)
    def test_legacy_request_hashes_match_frozen_baseline(self):
        # Captured from the pre-v3 c74d runtime; no local results directory needed.
        expected=iter((
            "0b99051a0da0bbbcdb3177ece381d2ede3c2b8db8d6db0427e6df4227ab0cbbd",
            "ba8d5944d1a1ccc8b38df3c906a8157702dd04ad5d6c6bd8e256eb840c9d6f21",
            "14dfa7a2f1f37f9bbd90893ef23678641ba53d6e849b09bdc002105c7d4002c8",
            "a6c801985d68f01b6475850b0ced5e5ddaf6496d87c3e17e2fbe19e2f5daa65f",
            "d611da37643774c2bfecb9eab092cc2dbc407090d6391b794099192e9c307382",
            "8a6a28732c793502aa43a594e35ee7bd5bc4d7a80e3ac5640c1ccc51abf02097"))
        b=Builder([(3,)]);m=b.finish(b.node("rotate",["input0"],step=-1))
        for version in (None,"explicit-v1","explicit-v2"):
            for profile in (None,"hecate-unified-public-v1"):
                r=prepare(m,PROFILE_SHA256,generation_guidance=version,construction_profile=profile)
                self.assertEqual(r["request_id"],next(expected))
    def test_native_public_candidate_gate_unchanged(self):
        for profile in (None,"hecate-unified-public-v1"):
            old=self.request(construction_profile=profile,generation_guidance="explicit-v2")
            new=self.request(construction_profile=profile,generation_guidance="explicit-v3")
            source=lower(old)
            make=lambda r,src:dict(schema=1,request_id=r["request_id"],hecate_source=src)
            self.assertEqual(validate_candidate(make(old,source),old),validate_candidate(make(new,source),new))
            for bad in ("import os\n",source.replace("def golden(","def main(")):
                with self.assertRaises(ValueError):validate_candidate(make(new,bad),new)
    def test_composite_requirements_not_replaced_by_math(self):
        kw=dict(construction_profile="hecate-unified-public-v1",construction="unified-public-composite-array-arithmetic")
        a=self.request(generation_guidance="explicit-v2",**kw)
        b=self.request(generation_guidance="explicit-v3",**kw)
        self.assertEqual(a["generation_guidance"]["construction_requirements"],b["generation_guidance"]["construction_requirements"])
        with self.assertRaises(ValueError):validate_candidate(dict(schema=1,request_id=b["request_id"],hecate_source=lower(b)),b)
    def test_v3_cli_forwarding(self):
        from run_candidate import parse_args,forward_options
        a=parse_args(["--case","scripts/baseline/cases/unified-two-input-two-output.json","--prepare","--unified-guidance","explicit-v3"])
        self.assertIn("explicit-v3",shlex.split(forward_options(a)))
    def test_three_logical_rotation_contexts_match_both_independent_references(self):
        import numpy as np
        from benchmark_math import evaluate as mathematics
        from benchmark_torch import evaluate as pytorch
        for n,step in ((3,1),(63,-2),(255,-2)):
            b=Builder([(n,)]);m=b.finish(b.node("rotate",["input0"],step=step))
            x=np.arange(1,n+1,dtype=np.float64)
            expected=np.asarray([x[(j+step)%n] for j in range(n)])
            r=prepare(m,PROFILE_SHA256,generation_guidance="explicit-v3")
            self.assertIn("modulo N",r["generation_guidance"]["logical_model_semantics"]["operations"]["rotate"])
            for evaluator in (mathematics,pytorch):
                actual=next(iter(evaluator(m,{"input0":x}).values()))
                np.testing.assert_array_equal(actual,expected)
    def test_math_metadata_contains_no_test_arrays_or_dsl_body(self):
        r=self.request(generation_guidance="explicit-v3")
        from deepseek_provider import public_request
        self.assertEqual(public_request(r),r)
        self.assertNotIn(lower(r),str(r["generation_guidance"]))
        for forbidden in ("test_inputs","reference_outputs","hecate_source"):
            self.assertNotIn(forbidden,r["generation_guidance"])

if __name__=="__main__":unittest.main()
