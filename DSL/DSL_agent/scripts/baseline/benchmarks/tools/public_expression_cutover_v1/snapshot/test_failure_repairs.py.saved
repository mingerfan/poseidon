"""Regression for typed guidance, construction identity and early SEAL rejection."""
import ast, copy, struct, unittest
from benchmark_graph import digest
from benchmark_suite import Builder
from compiler_configuration import PROFILE_SHA256
from unified_graph_contract import prepare,validate_request,validate_candidate,static_repair_hint
from unified_graph_lowering import lower
from seal_artifact_gate import inspect_artifacts
from test_seal_cpu_golden import artifact

class FailureRepairTests(unittest.TestCase):
    def request(self, version="explicit-v5", construction=None):
        b=Builder([(2,)])
        return prepare(b.finish(b.node("square",["input0"])),PROFILE_SHA256,
                       construction_profile="hecate-unified-public-v1",
                       construction=construction,generation_guidance=version)
    def test_v5_only_adds_guidance(self):
        old=self.request("explicit-v4");new=self.request()
        a=dict(new["generation_guidance"]);a.pop("typed_construction_semantics");a["version"]="explicit-v4"
        self.assertEqual(a,old["generation_guidance"])
        self.assertEqual({k:v for k,v in old.items() if k not in ("request_id","generation_guidance")},
                         {k:v for k,v in new.items() if k not in ("request_id","generation_guidance")})
        validate_request(new)
    def test_tamper_rehashed_rejected(self):
        r=self.request();r["generation_guidance"]["typed_construction_semantics"]["contribution"]=[]
        r["request_id"]=digest({k:v for k,v in r.items() if k!="request_id"})
        with self.assertRaisesRegex(ValueError,"Changed unified generation guidance"):validate_request(r)
    def test_hints_do_not_echo_input_or_change_old_versions(self):
        s='@hc.func("c,c")\ndef golden(x, zero_ct):\n    return [x*x]\n'
        self.assertEqual(static_repair_hint(s,self.request("explicit-v4"),"Missing contributing SECRET"),"")
        hint=static_repair_hint(s,self.request(),"Missing contributing SECRET")
        self.assertIn("correct computation",hint);self.assertNotIn("SECRET",hint)
    def test_forbidden_cast_and_import_stay_rejected(self):
        r=self.request()
        for body in ("return [float(x)]","return [np.asarray([x])[0]]","return [+x]"):
            with self.subTest(body=body),self.assertRaises(ValueError):
                validate_candidate(dict(schema=1,request_id=r["request_id"],hecate_source='@hc.func("c,c")\ndef golden(x, zero_ct):\n    '+body),r)
        with self.assertRaises(ValueError):validate_candidate(dict(schema=1,request_id=r["request_id"],hecate_source="import os\n"+lower(r)),r)
    def test_helper_name_cannot_impersonate_mapping_update(self):
        r=self.request(construction="unified-public-call-update")
        src='def update(a):\n    return a*a\n@hc.func("c,c")\ndef golden(x, zero_ct):\n    return [update(x)]\n'
        with self.assertRaisesRegex(ValueError,"Missing contributing"):
            validate_candidate(dict(schema=1,request_id=r["request_id"],hecate_source=src),r)
        src='@hc.func("c,c")\ndef golden(x, zero_ct):\n    d={"v":zero_ct}\n    d.update({"v":x*x})\n    return [d["v"]]\n'
        validate_candidate(dict(schema=1,request_id=r["request_id"],hecate_source=src),r)
    def test_dead_list_rejected_but_consumed_list_accepted(self):
        r=self.request(construction="unified-public-call-list")
        head='@hc.func("c,c")\ndef golden(x, zero_ct):\n'
        for body,ok in [('    q=list([0.5])\n    return [x*x]\n',False),
                        ('    q=list([1.0])\n    return [x*x*q[0]]\n',True)]:
            c=dict(schema=1,request_id=r["request_id"],hecate_source=head+body)
            if ok:validate_candidate(c,r)
            else:
                with self.assertRaisesRegex(ValueError,"Missing contributing"):validate_candidate(c,r)
    def test_cipher_container_intervention_and_unused_side_effect_rejection(self):
        for kind in ("list","tuple"):
            r=self.request(construction="unified-public-call-"+kind)
            head='@hc.func("c,c")\ndef golden(x, zero_ct):\n'
            good=head+'    return '+kind+'(iter([x*x]))\n'
            validate_candidate(dict(schema=1,request_id=r["request_id"],hecate_source=good),r)
            bad='def produce(a,x):\n    a[0]=x*x\n    return [x]\n'+head+'    a=[zero_ct]\n    unused='+kind+'(produce(a,x))\n    return a\n'
            with self.assertRaisesRegex(ValueError,"Missing contributing"):
                validate_candidate(dict(schema=1,request_id=r["request_id"],hecate_source=bad),r)
    def test_zero_plain_mul_rejected_but_mask_add_and_encrypted_zero_allowed(self):
        enc=(0,0,0,(13<<10)|40)
        for v in ((0.,),(-0.,),(0.,)*4):
            cst=struct.pack("<qq",1,len(v))+struct.pack("<"+"d"*len(v),*v)
            with self.assertRaisesRegex(ValueError,"Transparent ciphertext risk"):
                inspect_artifacts(artifact((enc,(9,0,0,0)),result_scale=80),cst)
            inspect_artifacts(artifact((enc,(7,0,0,0))),cst)
        cst=struct.pack("<qq4d",1,4,1,0,0,0)
        inspect_artifacts(artifact((enc,(9,0,0,0)),result_scale=80),cst)
        inspect_artifacts(artifact(),struct.pack("<q",0))
    def test_scale_diagnostic_preserves_boundary(self):
        data=bytearray(artifact(result_level=1));struct.pack_into("<Q",data,80,75)
        with self.assertRaisesRegex(ValueError,r"output\[0\].*remaining_level=1, log2_scale=75"):
            inspect_artifacts(data,struct.pack("<q",0))
    def test_cli_and_sandbox(self):
        from run_candidate import parse_args
        from candidate_sandbox import MODULES
        self.assertEqual(parse_args(["--prepare","--case","x","--unified-guidance","explicit-v5"]).unified_guidance,"explicit-v5")
        self.assertIn("unified_failure_semantics.py",MODULES)
if __name__=="__main__":unittest.main()
