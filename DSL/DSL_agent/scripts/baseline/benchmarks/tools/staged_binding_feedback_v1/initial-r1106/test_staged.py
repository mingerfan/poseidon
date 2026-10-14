"""Pure component/context tests for staged files; no provider/compiler/runtime."""
import copy
import hashlib
import json
import sys
import unittest
from pathlib import Path

STAGE=Path(__file__).resolve().parent
ROOT=STAGE.parents[4]
sys.path.insert(0,str(ROOT/"scripts/baseline"))
sys.path.insert(0,str(STAGE/"targets"))

from application_diagnostics import RepairDecision,attach,checked,repeat_key
from benchmark_graph import FORMAT,CAPACITY_FORMAT,digest
from component_contract import prepare_task,validate_program,reconstruct_request
from candidate_contract import strict_json
from unified_binding_guidance import specification,repair_hint
from unified_poseidon_guidance import specification as old_specification

RESULTS=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux/results")
LIVE=RESULTS/"agent-approved-pending-continuation-r1099"

def graph(shape=(2,2),axes=(1,),profile=None):
    model=dict(format=FORMAT,id="unseen",inputs=[dict(name="x",shape=list(shape))],
        constants={},nodes=[dict(id="reduce",op="mean",inputs=["x"],
            attrs=dict(axes=list(axes),keepdims=True),outputs=["m"]),
            dict(id="center",op="subtract",inputs=["x","m"],attrs={},outputs=["y"])],
        outputs=[dict(name="out",value="y")])
    if profile:
        model["format"]=CAPACITY_FORMAT
        model["resource_profile"]=profile
    return model

def request(model=None):
    options=dict(generation_guidance="explicit-v10")
    if model and model["format"]==CAPACITY_FORMAT:
        options["chunk_period"]=4
    return prepare_task(model or graph(),options)

def candidate(req,source):
    return dict(schema=1,request_id=req["request_id"],hecate_source=source)

class ContextTests(unittest.TestCase):
    def test_capacity_reduction_is_selected(self):
        req=request(graph(profile="general-tensors-v1"))
        self.assertNotIn("packed_reduction",old_specification(req)["selection"])
        new=specification(req)
        self.assertIn("packed_reduction",new["selection"])
        self.assertEqual(new["version"],"explicit-v11")
        self.assertEqual(new["shape_examples"]["reductions"][0]["axes"],[1])

    def test_non_scalar_row_broadcast(self):
        req=request()
        self.assertNotIn("scalar_broadcast",old_specification(req)["selection"])
        new=specification(req)
        self.assertIn("tensor_broadcast",new["selection"])
        item=new["shape_examples"]["broadcasts"][0]
        self.assertEqual(item["output_shape"],[2,2])
        self.assertEqual(item["expanded_operands"][0]["shape"],[2,1])

    def test_old_periodic_capacity_v2_v3_and_v4(self):
        from unified_capacity_layout import ABIS
        req=request()
        for abi in ("unified-periodic-inputs-v1",*ABIS):
            variant=copy.deepcopy(req);variant["layout"]["execution_abi"]=abi
            self.assertIn("packed_reduction",specification(variant)["selection"])
        req["layout"]["execution_abi"]="unknown-arbitrary-layout"
        with self.assertRaises(ValueError):
            specification(req)

    def test_constant_roles_distinguish_mask_and_scalar(self):
        req=request()
        req["public_constants"]={"scalar":.25,"one_lane":[.25,0,0,0],
                                 "weights":[.1,.2,.3,.4],"uniform":[1,1,1,1]}
        bindings=specification(req)["constant_role_examples"]["bindings"]
        kinds={b["name"]:b["kind"] for b in bindings}
        self.assertEqual(kinds,{"scalar":"scalar","one_lane":"single_lane_vector",
                               "weights":"coordinate_vector","uniform":"uniform_vector"})

    def test_node_and_model_names_do_not_select_rules(self):
        req=request();old=specification(req)
        changed=copy.deepcopy(req);changed["model"]["id"]="brand_new_model"
        self.assertEqual(specification(changed),old)

    def test_rank32_uses_logical_not_container_rank(self):
        req=request(graph(shape=(1,)*31+(4,),axes=(31,),profile="general-tensors-v1"))
        context=specification(req)
        self.assertEqual(len(context["shape_examples"]["reductions"][0]["input_shape"]),32)
        self.assertIn("tensor_broadcast",context["selection"])

    def test_unrelated_context_is_not_exfiltrated(self):
        req=request();req["private_inputs"]=[123.987];req["reference_outputs"]=[998.765]
        req["credentials"]="PRIVATE_SENTINEL"
        payload=json.dumps(specification(req))
        for forbidden in ("PRIVATE_SENTINEL","123.987","998.765","private_inputs","reference_outputs","credentials","hecate_source"):
            self.assertNotIn(forbidden,payload)

    def test_repair_has_fixed_numeric_shape_hints(self):
        req=request();req["generation_guidance"]["version"]="explicit-v11"
        hint=repair_hint(req,'def golden(x):\n return x\n',"/private/SECRET.npy","numerical_comparison")
        self.assertIn("single-lane",hint)
        self.assertIn("Broadcast every declared dimension",hint)
        self.assertNotIn("SECRET",hint)
        self.assertLessEqual(len(hint.encode()),5000)
        self.assertEqual(repair_hint(req,"bad syntax!",stage="numerical_comparison"),"")

    def test_wrong_backend_cannot_use_context(self):
        req=request();req["compiler_configuration"]["backend"]="seal"
        with self.assertRaises(ValueError):
            specification(req)

class StaticFailureTests(unittest.TestCase):
    def test_undefined_symbol_has_exact_source_location(self):
        req=request()
        source='@hc.func("c,c")\ndef golden(x0, zero_ct):\n    return x0 * missing\n'
        result=validate_program(candidate(req,source),req)
        failure=result["failure"]
        self.assertEqual(failure["reason_code"],"undefined_value")
        self.assertEqual(failure["location"],dict(kind="source",line=3,column=16))
        public=attach(dict(status="failed",failure=failure))["public_feedback"]
        self.assertIn("public_constants",public["diagnostic"])
        self.assertIn('"line": 3',public["diagnostic"])
        self.assertIsNone(repeat_key(failure))

    def test_scalar_global_is_valid_but_logical_constant_name_is_not(self):
        req=request()
        known=next(n for n,v in req["public_constants"].items() if type(v) in (int,float))
        source='@hc.func("c,c")\ndef golden(x0, zero_ct):\n    return x0 * '+known+'\n'
        self.assertEqual(validate_program(candidate(req,source),req)["semantic_validation"],"accepted")

    def test_helper_lexical_scope_and_local_order_unchanged(self):
        req=request()
        good='@hc.func("c")\ndef helper(a):\n    t=a+a\n    return t\n@hc.func("c,c")\ndef golden(x0, zero_ct):\n    return helper(x0)\n'
        self.assertEqual(validate_program(candidate(req,good),req)["semantic_validation"],"accepted")
        bad='@hc.func("c,c")\ndef golden(x0, zero_ct):\n    a=late\n    late=x0\n    return a\n'
        self.assertEqual(validate_program(candidate(req,bad),req)["failure"]["reason_code"],"undefined_value")

    def test_numeric_array_and_object_array_remain_distinct(self):
        req=request()
        prefix='@hc.func("c,c")\ndef golden(x0, zero_ct):\n    a=np.array([x0],dtype='
        invalid=validate_program(candidate(req,prefix+'np.float64)\n    return a[0]\n'),req)
        self.assertEqual(invalid["failure"]["reason_code"],"native_array_constructor")
        self.assertEqual(invalid["failure"]["location"]["line"],3)
        valid=validate_program(candidate(req,prefix+'object)\n    return a[0]\n'),req)
        self.assertEqual(valid["semantic_validation"],"accepted")

    def test_public_feedback_does_not_include_arbitrary_error_text(self):
        req=request()
        source='@hc.func("c,c")\ndef golden(x0, zero_ct):\n return x0*undefined\n'
        result=validate_program(candidate(req,source),req)
        result["failure"]["diagnostic"]="PRIVATE /secret/model-reference.npy"
        output=attach(dict(status="failed",failure=result["failure"]))
        self.assertEqual(output["failure"]["reason_code"],"environment_failure")
        self.assertNotIn("PRIVATE",json.dumps(output["public_feedback"]))
        self.assertNotIn("/secret",json.dumps(output["public_feedback"]))

    def test_existing_closed_static_failures_get_specific_reasons(self):
        group=json.loads((LIVE/"functional15/report.json").read_text())
        found=[]
        for row in group["rows"]:
            t=LIVE/"functional15"/row["id"]/"tasks"/row["task_id"]
            req=json.loads((t/"task.json").read_text())["request"]
            for index in range(row["result"]["generations"]):
                validation=json.loads((t/("validation-%d.json"%index)).read_text())
                if validation.get("failure",{}).get("layer")=="static_check":
                    raw=json.loads((t/("response-%d.json"%index)).read_text())["raw"]
                    failure=validate_program(strict_json(raw),req)["failure"]
                    self.assertEqual(failure["location"]["kind"],"source")
                    checked(failure)
                    found.append(failure["reason_code"])
        self.assertEqual(found.count("undefined_value"),4)
        self.assertEqual(found.count("native_array_constructor"),1)

    def test_all_returned_pass_candidates_still_static_accept(self):
        group=json.loads((LIVE/"functional15/report.json").read_text())
        count=0
        for row in group["rows"]:
            if row["status"]!="passed":continue
            folder=Path(row["package"])
            req=json.loads((LIVE/"functional15"/row["id"]/"tasks"/row["task_id"]/"task.json").read_text())["request"]
            result=validate_program(candidate(req,(folder/"candidate.py").read_text()),req)
            self.assertEqual(result["semantic_validation"],"accepted",row["id"])
            count+=1
        self.assertEqual(count,10)

    def test_old_request_reconstruction_stays_identical(self):
        group=json.loads((LIVE/"functional15/report.json").read_text())
        for row in group["rows"]:
            req=json.loads((LIVE/"functional15"/row["id"]/"tasks"/row["task_id"]/"task.json").read_text())["request"]
            self.assertEqual(reconstruct_request(req),req)

class DuplicateTests(unittest.TestCase):
    def raw(self,source,request_id="a"*64):
        return json.dumps(dict(schema=1,request_id=request_id,hecate_source=source))
    def test_identical_ast_with_comments_is_repeated(self):
        decision=RepairDecision()
        self.assertIsNone(decision.before_candidate(self.raw('def golden(x):\n return x+x\n')))
        self.assertEqual(decision.before_candidate(self.raw('def golden(x):\n    # retained math\n    return x + x\n')),"validation_failed")
        self.assertEqual(decision.last_failure["code"],"repeated_candidate")
    def test_changed_math_or_bound_name_remains_a_new_candidate(self):
        decision=RepairDecision()
        for source in ('def golden(x):\n return x*p0\n',
                       'def golden(x):\n return x*p1\n',
                       'def golden(x):\n return x+p0\n'):
            self.assertIsNone(decision.before_candidate(self.raw(source)))
    def test_invalid_syntax_retains_raw_fallback(self):
        decision=RepairDecision()
        self.assertIsNone(decision.before_candidate(self.raw("invalid syntax!")))
        self.assertEqual(decision.before_candidate(self.raw("invalid syntax!")),"validation_failed")
    def test_correcting_response_identity_is_not_an_identical_candidate(self):
        decision=RepairDecision()
        source='def golden(x):\n return x\n'
        self.assertIsNone(decision.before_candidate(self.raw(source,"b"*64)))
        self.assertIsNone(decision.before_candidate(self.raw(source,"a"*64)))

if __name__=="__main__":
    unittest.main(verbosity=2)
