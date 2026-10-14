"""Spatial-domain routing and application controls; runtime interfaces are injected."""
import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from benchmark_graph import digest
from operator_arithmetic_graph import decompose,evaluate_reference,FORMATS
from benchmark_math import evaluate as core
from execution_input_domain import ApproximationDomainRejected,check_approximation_domains
from component_contract import prepare_task
FIXTURES=Path(__file__).parent/"benchmarks/tools/spatial_domain_preflight_v1/model-cases.json"
def models():return json.loads(FIXTURES.read_text())
def input_range(name):
    if name=="transpose2_nonlinear":return .5,2.5
    if name=="transpose1_nonlinear":return -.5,.5
    if name=="public_left_spatial":return -.5,1.5
    return -1.,1.
def sample(name,model,value=None):
    low,high=input_range(name);shape=model["inputs"][0]["shape"]
    x=np.linspace(low,high,int(np.prod(shape)),dtype=np.float64).reshape(shape)if value is None else np.full(shape,value,dtype=np.float64)
    return dict(x=x)
def invalid_sample(name,model):
    value={"transpose2_nonlinear":3.,"two_domains":1.25,"public_left_spatial":-1.}.get(name,2.)
    return sample(name,model,value)
def expected_index(name):return 2 if name in("two_domains","public_left_spatial")else 1
def request(model):
    low,_=decompose(model,resource_profile="general-tensors-v1")
    return prepare_task(low,dict(chunk_period=4,capacity_packing_version="capacity-public-tensors-v9",
     generation_guidance="explicit-v13",public_weight_policy="descriptors-math-roles-v2"))
class SpatialDomainTests(unittest.TestCase):
    def rejected(self,name,value=None,index=None,sample_index=0):
        model=models()[name];inputs=invalid_sample(name,model)if value is None else value
        with self.assertRaises(ApproximationDomainRejected)as found:
            check_approximation_domains(model,[inputs]if type(inputs)is dict else inputs)
        failure=found.exception.public_failure
        self.assertEqual(failure["location"]["index"],expected_index(name)if index is None else index)
        self.assertEqual(failure["context"]["sample_index"],sample_index)
        return failure
    def test_six_registered_combinations_accept_sixteen_bounded_samples(self):
        rng=np.random.default_rng(714)
        for name,model in models().items():
            with self.subTest(case=name):
                decompose(model,resource_profile="general-tensors-v1")
                low,high=input_range(name);shape=model["inputs"][0]["shape"]
                inputs=[sample(name,model,low),sample(name,model,high),sample(name,model)]
                inputs += [dict(x=rng.uniform(low,high,size=shape).astype(np.float64))for _ in range(13)]
                result=check_approximation_domains(model,inputs)
                self.assertEqual((result["status"],result["samples"]),("passed",16))
                self.assertEqual(result["profile_count"],2 if name=="two_domains"else 1)
                self.assertEqual(result["source_sha256"],digest(model))
    def test_unchanged_reference_and_decomposition_agree(self):
        rng=np.random.default_rng(715)
        for name,model in models().items():
            with self.subTest(case=name):
                lowered,_=decompose(model,resource_profile="general-tensors-v1")
                low,high=input_range(name);shape=model["inputs"][0]["shape"]
                for _ in range(16):
                    inputs=dict(x=rng.uniform(low,high,size=shape).astype(np.float64))
                    np.testing.assert_allclose(core(lowered,inputs)["out"],
                      evaluate_reference(model,inputs,ideal=False)["out"],rtol=1e-12,atol=1e-12)
    def test_each_combination_rejects_its_first_actual_violation(self):
        for name in models():
            with self.subTest(case=name):self.rejected(name)
    def test_nearest_resize_keeps_closed_boundaries(self):
        model=models()["resize1_gelu"]
        self.assertEqual(check_approximation_domains(model,[sample("resize1_gelu",model)])["status"],"passed")
    def test_bilinear_resize_checks_interpolated_values(self):
        model=models()["resize2_gelu"]
        value=dict(x=np.array([[[[-1.,1.],[1.,-1.]]]],dtype=np.float64))
        self.assertEqual(check_approximation_domains(model,[value])["status"],"passed")
        value["x"][0,0,0,0]=1.25
        self.rejected("resize2_gelu",value)
    def test_transpose_checks_output_domain_rather_than_raw_input(self):
        model=models()["transpose1_nonlinear"]
        self.assertEqual(check_approximation_domains(model,[sample("transpose1_nonlinear",model,.5)])["status"],"passed")
        self.rejected("transpose1_nonlinear",sample("transpose1_nonlinear",model,.625))
    def test_sqrt_transpose_checks_both_closed_endpoints(self):
        model=models()["transpose2_nonlinear"]
        for value in(.5,2.5):
            self.assertEqual(check_approximation_domains(model,[sample("transpose2_nonlinear",model,value)])["status"],"passed")
    def test_one_ulp_outside_sqrt_transpose_is_rejected(self):
        model=models()["transpose2_nonlinear"]
        for value in(np.nextafter(.5,-np.inf),np.nextafter(2.5,np.inf)):
            self.rejected("transpose2_nonlinear",sample("transpose2_nonlinear",model,float(value)))
    def test_two_domains_locate_earlier_and_later_failures(self):
        model=models()["two_domains"]
        self.rejected("two_domains",sample("two_domains",model,2.25),index=0)
        self.rejected("two_domains",sample("two_domains",model,1.25),index=2)
    def test_public_left_subtraction_is_evaluated_before_domain(self):
        model=models()["public_left_spatial"]
        for value in(-.5,1.5):
            self.assertEqual(check_approximation_domains(model,[sample("public_left_spatial",model,value)])["status"],"passed")
        self.rejected("public_left_spatial",sample("public_left_spatial",model,-.625))
    def test_sixteenth_sample_retains_stable_failure_location(self):
        model=models()["two_domains"]
        inputs=[sample("two_domains",model,.25)for _ in range(15)]+[invalid_sample("two_domains",model)]
        self.rejected("two_domains",inputs,sample_index=15)
    def test_model_and_caller_arrays_are_not_mutated(self):
        for name,model in models().items():
            before=copy.deepcopy(model);batch=sample(name,model);raw=batch["x"].tobytes()
            check_approximation_domains(model,[batch]*16)
            self.assertEqual(model,before);self.assertEqual(batch["x"].tobytes(),raw)
    def test_spatial_graph_without_domains_is_not_certified(self):
        model=models()["resize1_gelu"];model["nodes"]=model["nodes"][:1]
        model["outputs"][0]["value"]=model["nodes"][0]["outputs"][0]
        with patch("operator_spatial_reference.evaluate",side_effect=AssertionError("unneeded reference")):
            result=check_approximation_domains(model,[sample("resize1_gelu",model)])
        self.assertEqual(result["status"],"not_declared")
        self.assertEqual(result["profile_count"],0)
    def test_legacy_formats_still_use_their_registered_reference(self):
        base=dict(id="legacy_domain",inputs=[dict(name="x",shape=[2])],constants={},
          nodes=[dict(id="e",op="exp",inputs=["x"],attrs=dict(approximation=dict(method="taylor",domain=[-1.,1.],degree=1)),outputs=["y"])],
          outputs=[dict(name="out",value="y")])
        for format in FORMATS:
            with self.subTest(format=format):
                m=dict(base,format=format)
                self.assertEqual(check_approximation_domains(m,[dict(x=np.zeros(2,dtype=np.float64))])["status"],"passed")
    def test_unrelated_reference_failure_is_redacted(self):
        model=models()["resize1_gelu"]
        with patch("operator_spatial_reference.original_resize",side_effect=ValueError("PRIVATE /env 9.123456789")):
            with self.assertRaises(ValueError)as found:check_approximation_domains(model,[sample("resize1_gelu",model)])
        self.assertNotIsInstance(found.exception,ApproximationDomainRejected)
        self.assertEqual(str(found.exception),"Execution domain preflight could not evaluate inputs")
    def test_public_failure_omits_observed_inputs_and_weights(self):
        model=models()["transpose1_nonlinear"]
        error=self.rejected("transpose1_nonlinear",sample("transpose1_nonlinear",model,9.123456789))
        raw=json.dumps(error)
        for text in("9.123456789","coefficients","weights","reference","actual",'"inputs":',"/home/"):
            self.assertNotIn(text,raw)
    def test_post_domain_suffix_is_not_evaluated(self):
        import execution_input_domain
        model=models()["resize1_gelu"]
        model["nodes"].append(dict(id="tail",op="square",inputs=["y"],attrs={},outputs=["z"]))
        model["outputs"][0]["value"]="z"
        with patch("execution_input_domain._evaluate_prefix",wraps=execution_input_domain._evaluate_prefix)as calls:
            result=check_approximation_domains(model,[sample("resize1_gelu",model)])
        self.assertEqual(calls.call_args.args[2],2)
        self.assertNotIn("outputs",result);self.assertNotIn("numerically_validated",result)
class PackingReached(Exception):pass
class SpatialExecutionControls(unittest.TestCase):
    def boundary(self,name,invalid):
        import program_execution
        model=models()[name];q=request(model)
        inputs=invalid_sample(name,model)if invalid else sample(name,model)
        with patch("candidate_bundle.check_execution_dependencies",return_value=({},q)),\
             patch("program_execution.package_execution_metadata",return_value=dict(original_model=model,output_structure=None)),\
             patch("program_execution._pack_logical_samples",side_effect=PackingReached)as packing,\
             patch("poseidon_backend_identity.identity")as identity,\
             patch("tempfile.mkdtemp")as directory,\
             patch("poseidon_validation.execute")as runtime,\
             patch("native_execution_slots.native_slot")as slot:
            expected=ApproximationDomainRejected if invalid else PackingReached
            with self.assertRaises(expected):
                program_execution.execute_program(".",[inputs])
            self.assertEqual(packing.call_count,0 if invalid else 1)
            for function in(identity,directory,runtime,slot):function.assert_not_called()
    def test_six_valid_application_inputs_reach_packing(self):
        for name in models():
            with self.subTest(case=name):self.boundary(name,False)
    def test_six_invalid_application_inputs_stop_before_packing_and_runtime(self):
        for name in models():
            with self.subTest(case=name):self.boundary(name,True)
    def test_application_exposes_typed_failure_without_generation(self):
        from application_component import Application
        from program_execution import normalize_samples
        model=models()["two_domains"];q=request(model)
        class Backend:
            def execute(self,folder,samples,**options):
                check_approximation_domains(model,normalize_samples(q,samples))
                raise AssertionError("invalid input unexpectedly accepted")
        with tempfile.TemporaryDirectory()as folder:
            app=Application(folder,Backend())
            with patch.object(app,"result",return_value=dict(status="succeeded",package=dict(task_id="a"*32))):
                result=app.execute("a"*32,[invalid_sample("two_domains",model)])
        self.assertEqual(result["status"],"validation_failed")
        self.assertEqual(result["failure"]["code"],"approximation_domain_rejected")
        self.assertEqual(result["failure"]["location"]["index"],2)
        self.assertFalse(result["encrypted_execution"])
        self.assertEqual(result["agent_calls"],0);self.assertEqual(result["compiler_calls"],0)
if __name__=="__main__":unittest.main()
