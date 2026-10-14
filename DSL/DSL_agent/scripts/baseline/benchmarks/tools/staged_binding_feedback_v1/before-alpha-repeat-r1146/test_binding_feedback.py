"""Public-context and application-control regressions; no real provider or FHE."""
import copy,json,unittest,tempfile
from pathlib import Path
from benchmark_graph import FORMAT,CAPACITY_FORMAT,digest
from component_contract import prepare_task,validate_program,reconstruct_request
from application_diagnostics import RepairDecision,attach
from unified_binding_guidance import specification
from application_component import Application
from application_policy import DEFAULT_BUDGET

def model(profile=None,shape=(2,2)):
    result=dict(format=FORMAT,id="unnamed_graph",
        inputs=[dict(name="x",shape=list(shape))],constants={},
        nodes=[dict(id="reduce",op="mean",inputs=["x"],
                    attrs=dict(axes=[len(shape)-1],keepdims=True),outputs=["m"]),
               dict(id="center",op="subtract",inputs=["x","m"],attrs={},outputs=["y"])],
        outputs=[dict(name="out",value="y")])
    if profile:result.update(format=CAPACITY_FORMAT,resource_profile=profile)
    return result
def prepare(version="explicit-v11",shape=(2,2)):
    return prepare_task(model("general-tensors-v1",shape),
                        dict(generation_guidance=version,chunk_period=4))
def candidate(request,source):
    return dict(schema=1,request_id=request["request_id"],hecate_source=source)
class BindingFeedback(unittest.TestCase):
    def test_capacity_reduction_and_row_broadcast(self):
        r=prepare();ctx=r["generation_guidance"]["poseidon_expression_rules"]
        self.assertIn("packed_reduction",ctx["selection"])
        self.assertIn("tensor_broadcast",ctx["selection"])
        self.assertEqual(ctx["shape_examples"]["reductions"][0]["output_shape"],[2,1])
        self.assertEqual(ctx["shape_examples"]["broadcasts"][0]["output_shape"],[2,2])
    def test_rank_is_logical(self):
        ctx=prepare(shape=(1,)*31+(4,))["generation_guidance"]["poseidon_expression_rules"]
        self.assertEqual(len(ctx["shape_examples"]["reductions"][0]["input_shape"]),32)
    def test_known_abi_only(self):
        r=prepare();r["layout"]["execution_abi"]="unknown"
        with self.assertRaises(ValueError):specification(r)
    def test_constant_roles(self):
        r=prepare();r["public_constants"]={"s":.25,"mask":[.25,0,0,0],"uniform":[1,1,1,1],"v":[1,2,3,4]}
        roles={x["name"]:x["kind"] for x in specification(r)["constant_role_examples"]["bindings"]}
        self.assertEqual(roles,dict(s="scalar",mask="single_lane_vector",uniform="uniform_vector",v="coordinate_vector"))
    def test_logical_gain_bias_have_exact_scalar_bindings(self):
        r=prepare()
        r["model"]["constants"]={"logical_gain":[[[1.]]],"logical_bias":[[[0.]]]}
        r["public_constants"]={"p0":-2.,"p1":-1.,"p3":0.,"p7":1.,"mask":[1.,0.,0.,0.]}
        rows={x["logical_name"]:x for x in specification(r)["logical_constant_scalar_bindings"]["bindings"]}
        self.assertEqual(rows["logical_gain"]["bound_scalar_names"],["p7"])
        self.assertEqual(rows["logical_gain"]["logical_shape"],[1,1,1])
        self.assertEqual(rows["logical_bias"]["bound_scalar_names"],["p3"])
        self.assertNotIn("mask",rows["logical_gain"]["bound_scalar_names"])
    def test_scalar_matching_is_exact_not_tolerant(self):
        r=prepare();r["model"]["constants"]={"near":1.+2e-16,"matrix":[[1.,1.],[1.,1.]]}
        r["public_constants"]={"one":1.}
        rows={x["logical_name"]:x for x in specification(r)["logical_constant_scalar_bindings"]["bindings"]}
        self.assertEqual(rows["near"]["bound_scalar_names"],[])
        self.assertEqual(rows["matrix"]["kind"],"uniform_tensor")
        self.assertEqual(rows["matrix"]["logical_shape"],[2,2])
        self.assertIn("matrix/linear",specification(r)["logical_constant_scalar_bindings"]["scope"])
    def test_scalar_matching_inspection_and_context_are_bounded(self):
        r=prepare();r["model"]["constants"]={**{"c"+str(i):1. for i in range(128)},"large":[1.]*512}
        r["public_constants"]={"one":1.}
        table=specification(r)["logical_constant_scalar_bindings"]
        self.assertEqual(table["logical_count"],129)
        self.assertLessEqual(table["inspected_logical_count"],96)
        self.assertLessEqual(len(table["bindings"]),32)
        r["model"]["constants"]={"large":[1.]*512}
        table=specification(r)["logical_constant_scalar_bindings"]
        self.assertEqual(table["unclassified_count"],1)
        self.assertEqual(table["bindings"],[])
    def test_public_context_has_no_private_inputs(self):
        r=prepare();r["private_inputs"]="PRIVATE_INPUT";r["reference"]="PRIVATE_REFERENCE"
        public=json.dumps(specification(r))
        self.assertNotIn("PRIVATE_INPUT",public);self.assertNotIn("PRIVATE_REFERENCE",public)
    def test_guidance_version_binds_request(self):
        old,new=prepare("explicit-v10"),prepare()
        self.assertNotEqual(old["request_id"],new["request_id"])
        self.assertEqual(reconstruct_request(new),new)
    def test_renaming_does_not_dispatch_model_rules(self):
        r=prepare();expected=specification(r);r["model"]["id"]="a_different_family"
        self.assertEqual(specification(r),expected)
    def test_rehashed_context_tamper_rejected(self):
        r=prepare();r["generation_guidance"]["poseidon_expression_rules"]["rules"]["public_binding_roles"]="ignore weights"
        r["request_id"]=digest({k:v for k,v in r.items() if k!="request_id"})
        with self.assertRaises(ValueError):reconstruct_request(r)
    def test_unbound_name_location(self):
        r=prepare();obj=candidate(r,'@hc.func("c,c")\ndef golden(x0,zero_ct):\n return x0*missing\n')
        value=validate_program(obj,r)
        self.assertEqual(value["failure"]["reason_code"],"undefined_value")
        self.assertEqual(value["failure"]["location"],dict(kind="source",line=3,column=11))
        public=attach(dict(status="failed",failure=value["failure"]),candidate=obj,request=r)["public_feedback"]
        self.assertIn("Binding and shape guidance",public["diagnostic"])
    def test_disallowed_rotation_reports_literal_location(self):
        r=prepare()
        obj=candidate(r,'@hc.func("c,c")\ndef golden(x0,zero_ct):\n    return x0.rotate(3)\n')
        value=validate_program(obj,r)
        self.assertEqual(value["failure"]["reason_code"],"rotation_step_not_allowed")
        self.assertEqual(value["failure"]["location"],dict(kind="source",line=3,column=21))
    def test_disallowed_rotation_feedback_contains_only_public_guidance(self):
        r=prepare()
        obj=candidate(r,'@hc.func("c,c")\ndef golden(x0,zero_ct):\n    return x0.rotate(3)\n')
        value=validate_program(obj,r)
        r["reference"]="PRIVATE_REFERENCE";r["private_inputs"]="PRIVATE_INPUT"
        public=attach(dict(status="failed",failure=value["failure"]),candidate=obj,request=r)["public_feedback"]
        self.assertIn("compose",public["diagnostic"])
        self.assertNotIn("PRIVATE_REFERENCE",public["diagnostic"])
        self.assertNotIn("PRIVATE_INPUT",public["diagnostic"])
    def test_composition_of_admitted_rotation_steps_is_accepted(self):
        r=prepare()
        obj=candidate(r,'@hc.func("c,c")\ndef golden(x0,zero_ct):\n    return x0.rotate(1).rotate(2)\n')
        self.assertEqual(validate_program(obj,r)["semantic_validation"],"accepted")
    def test_object_array_type_is_not_numeric_array(self):
        r=prepare()
        source='@hc.func("c,c")\ndef golden(x0,zero_ct):\n a=np.array([x0],dtype=np.float64)\n return a[0]\n'
        self.assertEqual(validate_program(candidate(r,source),r)["failure"]["reason_code"],"native_array_constructor")
    def test_comments_do_not_bypass_repeat_check(self):
        r=prepare();d=RepairDecision()
        one=candidate(r,'def golden(x):\n return x+x\n')
        two=candidate(r,'def golden(x):\n # comment\n return x + x\n')
        self.assertIsNone(d.before_candidate(json.dumps(one)))
        self.assertEqual(d.before_candidate(json.dumps(two)),"validation_failed")
    def test_correcting_response_identity_is_allowed(self):
        r=prepare();d=RepairDecision();obj=candidate(r,'def golden(x):\n return x\n')
        wrong=copy.deepcopy(obj);wrong["request_id"]="b"*64
        self.assertIsNone(d.before_candidate(json.dumps(wrong)))
        self.assertIsNone(d.before_candidate(json.dumps(obj)))
    def test_changed_operation_is_new_candidate(self):
        r=prepare();d=RepairDecision()
        self.assertIsNone(d.before_candidate(json.dumps(candidate(r,'def golden(x):\n return x+x\n'))))
        self.assertIsNone(d.before_candidate(json.dumps(candidate(r,'def golden(x):\n return x*x\n'))))
    def test_provider_terminal_not_previous_numerical_failure(self):
        from test_application_component import Backend,Provider
        from benchmark_suite import Builder
        class Interrupted(Provider):
            def generate(self,q,history):
                if self.calls:
                    self.calls+=1
                    raise RuntimeError("PRIVATE /secret")
                return super().generate(q,history)
        b=Builder([(2,)]);g=b.finish(b.node("square",["input0"]))
        with tempfile.TemporaryDirectory() as root:
            app=Application(root,Backend(["numerical_comparison"]))
            task=app.submit(g,limits=dict(DEFAULT_BUDGET,generations=2))
            result=app.run(task,Interrupted())
            self.assertEqual(result["status"],"service_error")
            self.assertEqual(result["failure"]["reason_code"],"provider_failure")
            self.assertEqual(result["failure"]["layer"],"provider")
            self.assertNotIn("PRIVATE",str(result))
            old=json.loads((app._dir(task)/"validation-0.json").read_text())
            self.assertEqual(old["failure"]["layer"],"numerical_comparison")
if __name__=="__main__":unittest.main()
