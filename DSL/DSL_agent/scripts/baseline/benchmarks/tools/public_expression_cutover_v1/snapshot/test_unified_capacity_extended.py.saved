"""Physical capacity admission, immutable ABI authorization and old-request compatibility."""
import ast,copy,json,math,unittest
from pathlib import Path
import numpy as np
from benchmark_graph import validate,samples,canonical
from component_contract import prepare_task,reconstruct_request
from unified_graph_contract import prepare,validate_request,validate_candidate
from unified_capacity_layout import ABI,EXTENDED_ABI,layout,validate_layout,entry_parameter_limit
from unified_graph_lowering import candidate_source
from cipher_abi import execution_options,physical_input_names
from test_unified_capacity_linear import interpreted_output

DIRECTORY=Path(__file__).parent/"cases/capacity-extended-v1"
def models():
    return [json.loads((DIRECTORY/e["file"]).read_text())
            for e in json.loads((DIRECTORY/"manifest.json").read_text())["cases"]]

class ExtendedCapacityTests(unittest.TestCase):
    def test_complete_capacity_and_cifar_shapes(self):
        plans=[layout(g) for g in models()]
        self.assertEqual(len(plans),9)
        self.assertTrue(all(p["execution_abi"]==EXTENDED_ABI for p in plans))
        self.assertEqual(len(plans[0]["inputs"]),16)
        self.assertEqual(plans[0]["output_ciphertexts"],16)
        self.assertEqual(len(plans[1]["inputs"]),12)
        self.assertEqual(plans[1]["logical_inputs"][0]["shape"],[1,3,32,32])
        self.assertEqual(plans[8]["inputs"][-1]["elements"],1)
        self.assertEqual(plans[8]["output_ciphertexts"],16)
    def test_generation_contract_has_no_four_chunk_conflict(self):
        r=prepare_task(models()[0],{"chunk_period":256})
        self.assertNotIn("At most four model chunks",r["rules"])
        self.assertNotIn("Chunk ABI unified-chunked-inputs-v1:",r["rules"])
        self.assertIn("At most sixteen model chunks",r["rules"])
        from unified_public_contract import CONTRACT
        public=prepare_task(models()[0],{"chunk_period":256,"construction_profile":CONTRACT})
        self.assertNotIn("At most four model chunks",public["rules"])
        g=copy.deepcopy(models()[0]);g["resource_profile"]="medium-v1";g["inputs"][0]["shape"]=[1024]
        previous=prepare_task(g,{"chunk_period":256})
        self.assertIn("At most four model chunks",previous["rules"])
    def test_fixed_auxiliary_and_runtime_count(self):
        r=prepare_task(models()[0],{"chunk_period":256})
        p=r["layout"];names=physical_input_names(p)
        self.assertEqual(names,tuple("x"+str(i) for i in range(16))+("zero_ct",))
        options=execution_options(p)
        self.assertEqual((options["expected_inputs"],options["logical_inputs"],options["logical_tensor_count"]),(17,16,1))
        self.assertTrue(options["encrypted_zero_input"])
    def test_logical_budget_does_not_override_padding_budget(self):
        g=copy.deepcopy(models()[8])
        for entry,size in zip(g["inputs"],[257,513,769,1793]):entry["shape"]=[1]*(len(entry["shape"])-1)+[size]
        # 3332 logical elements fit, but need 17 separate physical blocks.
        validate(g)
        with self.assertRaisesRegex(ValueError,"maximum 16"):layout(g,256)
    def test_original_v2_retains_four_chunks(self):
        with self.assertRaisesRegex(ValueError,"bounded physical"):layout(models()[0],execution_abi=ABI)
    def test_medium_cannot_select_extended(self):
        g=copy.deepcopy(models()[0]);g["resource_profile"]="medium-v1";g["inputs"][0]["shape"]=[1024]
        with self.assertRaisesRegex(ValueError,"configured sixteen-chunk profile"):layout(g,256,execution_abi=EXTENDED_ABI)
    def test_unknown_abi_rejected(self):
        with self.assertRaisesRegex(ValueError,"Unknown capacity"):layout(models()[0],256,execution_abi="unrestricted")
    def test_period_count_is_not_bypassed(self):
        with self.assertRaisesRegex(ValueError,"input ciphertext"):layout(models()[0],128)
    def test_logical_output_capacity_remains_bound(self):
        g=copy.deepcopy(models()[0]);g["outputs"]=[dict(name="first",value="squared"),dict(name="second",value="squared")]
        with self.assertRaisesRegex(ValueError,"Total output"):validate(g)
    def test_every_layout_field_is_reconstructed(self):
        for mutate in (
                lambda p:p["inputs"][15].update(dsl_name="x0"),
                lambda p:p["inputs"][15].update(offset=0),
                lambda p:p["output_selectors"].__setitem__(4095,[0,0]),
                lambda p:p["outputs"][0]["chunks"][-1].update(elements=1),
                lambda p:p.update(execution_abi=ABI),
                lambda p:p["resource_profile"].update(input_elements=8192)):
            p=layout(models()[0],256);mutate(p)
            with self.assertRaises(ValueError):validate_layout(p)
    def test_entry_binding_cannot_be_forged(self):
        p=layout(models()[0],256);names=tuple(x["dsl_name"] for x in p["inputs"])+("zero_ct",)
        self.assertEqual(entry_parameter_limit(p,names,16,256),17)
        for bad_names,out,period in ((names[:-1],16,256),(names,1,256),(names,16,128)):
            with self.assertRaisesRegex(ValueError,"binding mismatch"):entry_parameter_limit(p,bad_names,out,period)
    def test_seventeen_entry_parameters_do_not_expand_helper_limits(self):
        r=prepare_task(models()[0],{"chunk_period":256});source,_=candidate_source(r)
        validate_candidate(dict(schema=1,request_id=r["request_id"],hecate_source=source),r)
        helper='@hc.func("'+','.join('c' for _ in range(17))+'")\ndef helper('+','.join('a'+str(i) for i in range(17))+'):\n    return a0\n'
        with self.assertRaisesRegex(ValueError,"helper declares 17.*maximum 16"):
            validate_candidate(dict(schema=1,request_id=r["request_id"],hecate_source=helper+source),r)
        from decorated_functions import validate as native
        with self.assertRaisesRegex(ValueError,"Invalid input names"):
            native(source,r["public_constants"],16,input_names=physical_input_names(r["layout"]),slot_period=256,array_mutation=True)
    def test_public_construction_uses_same_layout_authority(self):
        r=prepare_task(models()[0],{"chunk_period":256})
        source,_=candidate_source(r)
        from unified_public_contract import CONTRACT
        public=prepare_task(models()[0],{"chunk_period":256,"construction_profile":CONTRACT})
        validate_candidate(dict(schema=1,request_id=public["request_id"],hecate_source=source),public)
        from function_construction import normalize
        from hecate_contract import UNIFIED_FLAT_CONTRACT
        with self.assertRaises(ValueError):
            normalize(source,public["public_constants"],16,input_names=physical_input_names(public["layout"]),
                      flat_contract=UNIFIED_FLAT_CONTRACT,slot_period=256)
    def test_old_resnet_v2_request_can_be_reconstructed(self):
        g=copy.deepcopy(models()[0]);g["inputs"][0]["shape"]=[1024]
        r=prepare(g,"f"*64,chunk_period=256,capacity_abi=ABI)
        self.assertEqual(r["layout"]["inputs"][0]["dsl_name"],"x")
        self.assertEqual(canonical(reconstruct_request(r)),canonical(r))
    def test_model_request_and_rule_reference_agree(self):
        from benchmark_math import evaluate as mathematical
        from benchmark_torch import evaluate as torch_reference
        for g in models():
            r=prepare_task(g,{"chunk_period":256})
            self.assertEqual(canonical(reconstruct_request(r)),canonical(r))
            self.assertEqual(validate_request(r),256)
            src,_=candidate_source(r)
            for inputs in samples(g,16):
                first=mathematical(g,inputs);second=torch_reference(g,inputs)
                expected=np.concatenate([first[o["name"]].reshape(-1) for o in g["outputs"]])
                for name in first:np.testing.assert_allclose(first[name],second[name],atol=1e-12,rtol=1e-12)
                np.testing.assert_allclose(interpreted_output(src,r,inputs),expected,atol=1e-10,rtol=1e-12)
    def test_previous_bound_packages_reconstruct_exactly(self):
        from hecate_python_env import WORK
        for name in ("poseidon-current-agent-replay-r506","poseidon-pytorch-matrix-r503",
                     "poseidon-cipher-broadcast-r503","poseidon-pytorch-medium-r504"):
            for row in json.loads((WORK/"results"/name/"report.json").read_text())["rows"]:
                p=Path(row["program"])/"request.json"
                r=json.loads(p.read_text())
                self.assertEqual(canonical(reconstruct_request(r)),canonical(r))
if __name__=="__main__":unittest.main()
