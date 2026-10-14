"""Request resources and budget rejection; unit checks do not execute FHE."""
import ast,copy,json,resource,subprocess,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from application_planning.resources import (analyze_request,check_analysis,assess_plan,submit_planned,
    native_address_limit,canonical,sha,_integer)
from application_planning.layout import plan_model,check_plan
from application_policy import prepare
from benchmark_graph import validate
from hecate_python_env import WORK,VENV
from application_component import Application
import test_application_component as fixtures

def model(n=8,outputs=1):
    return dict(format="poseidon-model-graph-v2",resource_profile="general-tensors-v1",id="resource_square",
        inputs=[dict(name="x",shape=[n])],constants={},
        nodes=[dict(id="s",op="square",inputs=["x"],attrs={},outputs=["v"])],
        outputs=[dict(name="out"+str(i),value="v") for i in range(outputs)])

class ResourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=model()
        cls.request=prepare(cls.model,"poseidon_cpu","numerical",dict(chunk_period=4))[0]
        cls.result=analyze_request(cls.request)
    def test_actual_parameter_and_input_storage_counts(self):
        r=self.result
        self.assertEqual(r["parameters"]["poly_degree"],32768)
        self.assertEqual(r["parameters"]["data_modulus_count"],13)
        self.assertEqual(r["parameters"]["input_components"],2)
        self.assertEqual(r["resources"]["input_ciphertexts"]["value"],2)
        self.assertEqual(r["resources"]["auxiliary_ciphertexts"]["value"],1)
        self.assertEqual(r["resources"]["input_ciphertext_coefficient_payload_bytes"]["value"],
                         3*32768*13*2*8)
        self.assertEqual(r["resources"]["input_ciphertext_coefficient_payload_bytes"]["bound"],"lower")
    def test_native_source_basis_is_explicitly_bound(self):
        from application_planning.resources import ROOT
        basis=self.result["parameters"]["source_basis_sha256"]
        self.assertEqual(set(basis),{"src/poseidon/ciphertext.cpp","src/poseidon/ciphertext.h",
            "src/poseidon/tools/agent_runtime/cpu_runner.cpp"})
        for name,value in basis.items():self.assertEqual(value,sha((ROOT/name).read_bytes()))
        changed=copy.deepcopy(self.result)
        changed["parameters"]["source_basis_sha256"]["src/poseidon/ciphertext.cpp"]="0"*64
        changed["binding"]=sha(canonical({k:v for k,v in changed.items() if k!="binding"}))
        with self.assertRaises(ValueError):check_analysis(changed,self.request)
    def test_native_limit_matches_isolated_existing_policy(self):
        code="import resource;from sandbox_policy import limits;limits();print(resource.getrlimit(resource.RLIMIT_AS)[0])"
        before=resource.getrlimit(resource.RLIMIT_AS)
        actual=int(subprocess.check_output([str(VENV/"bin/python"),"-B","-c",code],text=True,timeout=10))
        self.assertEqual(native_address_limit(),actual)
        self.assertEqual(resource.getrlimit(resource.RLIMIT_AS),before)
    def test_exact_budget_boundary_does_not_certify_executability(self):
        floor=self.result["resources"]["input_ciphertext_coefficient_payload_bytes"]["value"]
        self.assertEqual(analyze_request(self.request,memory_budget_bytes=floor-1)["status"],"blocked")
        accepted=analyze_request(self.request,memory_budget_bytes=floor)
        self.assertEqual(accepted["status"],"assessed")
        self.assertEqual(accepted["execution_readiness"],"unknown")
        self.assertFalse(accepted["encrypted_execution"])
    def test_packed_samples_scale_dataset_only(self):
        a=analyze_request(self.request,sample_count=1);b=analyze_request(self.request,sample_count=16)
        key="input_ciphertext_coefficient_payload_bytes"
        self.assertEqual(a["resources"][key],b["resources"][key])
        key="packed_dataset_float64_payload_bytes"
        self.assertEqual(b["resources"][key]["value"],a["resources"][key]["value"]*16)
    def test_period_changes_physical_storage_and_tail_padding(self):
        request=prepare(model(17),"poseidon_cpu","numerical",dict(chunk_period=32))[0]
        r=analyze_request(request)
        self.assertEqual(r["resources"]["input_ciphertexts"]["value"],1)
        self.assertEqual(r["resources"]["packed_sample_float64_payload_bytes"]["value"],2*32*8)
        self.assertEqual(r["logical"]["largest_cipher_tensor_elements"],17)
    def test_output_aliases_do_not_inflate_storage_lower_bound(self):
        request=prepare(model(outputs=2),"poseidon_cpu","numerical",dict(chunk_period=4))[0]
        r=analyze_request(request)
        self.assertEqual(r["resources"]["input_ciphertext_coefficient_payload_bytes"],
                         self.result["resources"]["input_ciphertext_coefficient_payload_bytes"])
    def test_register_values_are_not_encoded_plaintext_sizes(self):
        m=model();m["constants"]={"bias":[.125,-.25,.375,-.5,.5,-.375,.25,-.125]}
        m["nodes"]=[dict(id="a",op="add",inputs=["x","bias"],attrs={},outputs=["v"])]
        q=prepare(m,"poseidon_cpu","numerical",dict(chunk_period=4,public_weight_policy="descriptors-v1"))[0]
        r=analyze_request(q)
        self.assertGreater(r["resources"]["public_registry_float64_payload_bytes"]["value"],0)
        self.assertEqual(r["resources"]["peak_memory_bytes"]["certainty"],"unknown")
    def test_logical_path_is_distinct_from_ckks_schedule(self):
        m=model();m["nodes"]+=[
            dict(id="p",op="power",inputs=["v"],attrs={"exponent":4},outputs=["pout"]),
            dict(id="prod",op="multiply",inputs=["pout","x"],attrs={},outputs=["z"])]
        m["outputs"]=[dict(name="out",value="z")]
        r=analyze_request(prepare(m,"poseidon_cpu","numerical",dict(chunk_period=4))[0])
        self.assertEqual(r["logical"]["cipher_multiply_path"]["value"],4)
        self.assertEqual(r["resources"]["ckks_level_consumption"]["certainty"],"unknown")
    def test_public_multiply_does_not_count_as_cipher_product(self):
        m=model();m["constants"]={"gain":.5}
        m["nodes"]=[dict(id="mul",op="multiply",inputs=["x","gain"],attrs={},outputs=["v"])]
        r=analyze_request(prepare(m,"poseidon_cpu","numerical",dict(chunk_period=4))[0])
        self.assertEqual(r["logical"]["cipher_multiply_path"]["value"],0)
    def test_polynomial_path_is_unknown(self):
        m=model();m["constants"]={"coef":[.125,.5,.25]}
        m["nodes"]=[dict(id="poly",op="polynomial",inputs=["x","coef"],attrs={"basis":"power"},outputs=["v"])]
        r=analyze_request(prepare(m,"poseidon_cpu","numerical",dict(chunk_period=4))[0])
        self.assertIsNone(r["logical"]["cipher_multiply_path"]["value"])
    def test_rank32_is_preserved(self):
        m=model();m["inputs"][0]["shape"]=[1]*31+[8]
        q=prepare(m,"poseidon_cpu","numerical",dict(chunk_period=4))[0]
        self.assertEqual(analyze_request(q)["logical"]["largest_cipher_tensor_elements"],8)
        self.assertEqual(q["model"],m)
    def test_invalid_memory_and_sample_limits_rejected(self):
        for budget in (True,0,-1,1.5,2**40):
            with self.assertRaises(ValueError):analyze_request(self.request,memory_budget_bytes=budget)
        for count in (True,0,17,None,1.0):
            with self.assertRaises(ValueError):analyze_request(self.request,sample_count=count)
    def test_stale_and_resigned_analysis_rejected(self):
        for mutate in (
            lambda v:v["resources"]["input_ciphertext_coefficient_payload_bytes"].update(value=0),
            lambda v:v["parameters"].update(data_modulus_count=1),
            lambda v:v.update(request_sha256="0"*64),
            lambda v:v["source_authority"].update(sha256="0"*64)):
            v=copy.deepcopy(self.result);mutate(v)
            v["binding"]=sha(canonical({k:x for k,x in v.items() if k!="binding"}))
            with self.assertRaises(ValueError):check_analysis(v,self.request)
    def test_changed_request_and_bad_hash_rejected(self):
        q=copy.deepcopy(self.request);q["layout"]["input_slot_period"]=32
        with self.assertRaises(ValueError):check_analysis(self.result,q)
        bad=copy.deepcopy(self.result);bad["binding"]="0"*64
        with self.assertRaises(ValueError):check_analysis(bad,self.request)
    def test_spec_bytes_require_approved_hash(self):
        with patch("model_weight_package.regular_bytes",return_value=b"{}"):
            with self.assertRaises(ValueError):analyze_request(self.request)
    def test_static_policy_parser_rejects_execution_and_overflow(self):
        for expr in ("__import__('os').system('false')","True","-1","2**17","2**16*2**40"):
            with self.assertRaises(ValueError):_integer(ast.parse(expr,mode="eval").body)
    def test_same_request_unchanged_and_repeatable(self):
        before=copy.deepcopy(self.request)
        self.assertEqual(check_analysis(self.result,self.request),self.result)
        self.assertEqual(self.request,before)
    def test_resource_gate_uses_original_application_and_cancel(self):
        planned=plan_model(self.model,options={"chunk_period":4})
        checked=check_plan(planned,self.model)
        analysis=assess_plan(planned,self.model)
        with tempfile.TemporaryDirectory(dir=WORK/"results") as name:
            app=Application(name,fixtures.Backend())
            result=submit_planned(app,self.model,planned,analysis)
            task=result["task_id"];payload=json.loads((Path(name)/task/"task.json").read_text())
            self.assertEqual(payload["request"],checked)
            self.assertEqual(payload["budget"]["generations"],4)
            self.assertEqual(app.cancel(task)["status"],"cancelled")
            blocked=assess_plan(planned,self.model,memory_budget_bytes=1)
            with patch.object(app,"submit") as submit:
                with self.assertRaises(ValueError):submit_planned(app,self.model,planned,blocked)
                submit.assert_not_called()

if __name__=="__main__":unittest.main()
