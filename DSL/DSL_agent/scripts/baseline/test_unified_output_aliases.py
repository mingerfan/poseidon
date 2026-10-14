"""Exact-reference output bindings and application/old-request compatibility.

These controls do not count as encrypted execution or Agent generation.
"""
import copy
import json
from pathlib import Path
import unittest
from benchmark_graph import CAPACITY_FORMAT, FORMAT, canonical
from component_contract import prepare_task, reconstruct_request, runner_options
from model_resources import identity
from unified_capacity_layout import ABI, EXTENDED_ABI, WIDE_ABI, layout, validate_layout
from unified_graph_contract import validate_request
from unified_output_aliases import POLICY, declarations, physical_output_items
from public_weight_context import request_digest

def graph(size=5, outputs=("z","z"), profile="general-tensors-v1"):
    return dict(format=CAPACITY_FORMAT, id="output_alias", resource_profile=profile,
        inputs=[dict(name="a",shape=[size])], constants={},
        nodes=[dict(id="n",op="square",inputs=["a"],attrs={},outputs=["z"])],
        outputs=[dict(name="out"+str(i),value=v) for i,v in enumerate(outputs)])

def options(period=4):
    return dict(chunk_period=period,capacity_packing_version="capacity-public-tensors-v9",
                output_binding_policy=POLICY,generation_guidance="explicit-v15")

class ExactBindings(unittest.TestCase):
    def test_aliases_share_complete_tail_chunks(self):
        value=layout(graph(7),4,output_binding_policy=POLICY)
        self.assertEqual(value["output_ciphertexts"],2)
        self.assertEqual(value["output_shape"],[14])
        self.assertEqual(value["outputs"][1]["alias_of"],"out0")
        self.assertEqual(value["outputs"][0]["chunks"],value["outputs"][1]["chunks"])
        self.assertEqual(value["outputs"][0]["chunks"][-1]["elements"],3)
        self.assertEqual(validate_layout(value),4)

    def test_interleaved_aliases_keep_logical_order(self):
        model=graph(5,("z","a","z","a"))
        value=layout(model,4,output_binding_policy=POLICY)
        self.assertEqual([v["name"] for v,b in physical_output_items(model,value)],["out0","out1"])
        self.assertEqual(value["output_ciphertexts"],4)
        selectors=value["output_selectors"]
        self.assertEqual(selectors[:5],selectors[10:15])
        self.assertEqual(selectors[5:10],selectors[15:20])
        self.assertNotEqual(selectors[:5],selectors[5:10])

    def test_numerically_equal_different_refs_are_not_aliases(self):
        model=graph(5,("z","b"))
        model["nodes"].append(dict(id="n2",op="square",inputs=["a"],attrs={},outputs=["b"]))
        value=layout(model,4,output_binding_policy=POLICY)
        self.assertEqual(value["output_ciphertexts"],4)
        self.assertTrue(all("alias_of" not in row for row in value["outputs"]))

    def test_medium_capacity_counts_unique_physical_results(self):
        model=graph(9,profile="medium-v1")
        with self.assertRaisesRegex(ValueError,"output ciphertext budget"):layout(model,4)
        value=layout(model,4,output_binding_policy=POLICY)
        self.assertEqual(value["execution_abi"],ABI)
        self.assertEqual(value["output_ciphertexts"],3)
        self.assertEqual(value["output_shape"],[18])

    def test_extended_capacity_does_not_remove_physical_limit(self):
        model=graph(20,("z",)*4)
        with self.assertRaisesRegex(ValueError,"output ciphertext budget"):layout(model,4)
        self.assertEqual(layout(model,4,output_binding_policy=POLICY)["output_ciphertexts"],5)
        oversize=graph(1)
        oversize["nodes"]=[dict(id="n",op="concat",inputs=["a"]*8,attrs={"axis":0},outputs=["v"]),
                          dict(id="n2",op="concat",inputs=["v"]*8,attrs={"axis":0},outputs=["w"]),
                          dict(id="n3",op="concat",inputs=["w","a"],attrs={"axis":0},outputs=["z"])]
        with self.assertRaisesRegex(ValueError,"output ciphertext budget"):
            layout(oversize,4,output_binding_policy=POLICY)

    def test_logical_budget_still_counts_aliases(self):
        with self.assertRaises(ValueError):layout(graph(1500,("z",)*3),256,output_binding_policy=POLICY)
        outputs=[dict(name="o"+str(i),shape=[1500],value="z")for i in range(3)]
        with self.assertRaisesRegex(ValueError,"logical element budget"):
            declarations([dict(name="a",shape=[1500])],outputs,256,"general-tensors-v1",EXTENDED_ABI,"0"*64)

    def test_configured_periods_and_transport_are_unchanged(self):
        for period in (4,8,16,32,64,128,256,512,1024,2048,4096):
            value=layout(graph(),period,output_binding_policy=POLICY)
            self.assertEqual(value["execution_abi"],WIDE_ABI if period>=512 else EXTENDED_ABI)
            self.assertEqual(validate_layout(value),period)
        for period in (1,2,3,5,8192,True):
            with self.assertRaises(ValueError):layout(graph(),period,output_binding_policy=POLICY)

    def test_auto_period_depends_on_physical_unique_values(self):
        value=layout(graph(20,("z",)*4),output_binding_policy=POLICY)
        self.assertEqual(value["input_slot_period"],4)
        self.assertEqual(value["output_ciphertexts"],5)

    def test_wrong_alias_shape_is_rejected(self):
        with self.assertRaisesRegex(ValueError,"alias shape mismatch"):
            declarations([dict(name="a",shape=[5])],
                [dict(name="o",value="z",shape=[5]),dict(name="p",value="z",shape=[1,5])],
                4,"general-tensors-v1",EXTENDED_ABI,"0"*64)

    def test_scalar_carrier_and_ranked_shape_preserved(self):
        for shape in ([1],[1,1],[1,1,1,1],[1]*32):
            m=graph(1);m["inputs"][0]["shape"]=shape
            value=layout(m,4,output_binding_policy=POLICY)
            self.assertEqual(value["outputs"][0]["shape"],shape)
            self.assertEqual(value["output_selectors"],[[0,0],[0,0]])

    def test_layout_tampering_is_rejected(self):
        original=layout(graph(),4,output_binding_policy=POLICY)
        changes=(lambda v:v["output_selectors"][5].__setitem__(0,1),
                 lambda v:v["outputs"][1].__setitem__("alias_of","wrong"),
                 lambda v:v.__setitem__("output_ciphertexts",3),
                 lambda v:v["outputs"][0]["chunks"][1].__setitem__("elements",4),
                 lambda v:v["output_binding"].__setitem__("extra",0))
        for change in changes:
            with self.subTest(change=change):
                value=copy.deepcopy(original);change(value)
                with self.assertRaises(ValueError):validate_layout(value)

    def test_detached_alias_chunks(self):
        value=layout(graph(),4,output_binding_policy=POLICY)
        value["outputs"][1]["chunks"][0]["elements"]=1
        self.assertEqual(value["outputs"][0]["chunks"][0]["elements"],4)

    def test_selector_probe_preserves_every_logical_value(self):
        for size in range(1,33):
            model=graph(size,("z","a","z","a"))
            value=layout(model,4,output_binding_policy=POLICY)
            physical=[[10000*c+i for i in range(4)]for c in range(value["output_ciphertexts"])]
            actual=[physical[c][i]for c,i in value["output_selectors"]]
            self.assertEqual(actual[:size],actual[2*size:3*size])
            self.assertEqual(actual[size:2*size],actual[3*size:])
            self.assertEqual(len(actual),size*4)

class ApplicationBindings(unittest.TestCase):
    def test_component_reconstruction_and_runner_forwarding(self):
        q=prepare_task(graph(),options())
        self.assertEqual(reconstruct_request(q),q)
        self.assertIn("--output-binding",runner_options(q))
        self.assertEqual(q["generation_guidance"]["version"],"explicit-v15")
        self.assertEqual(q["generation_guidance"]['poseidon_expression_rules']["final_output_binding"]["physical_outputs"],2)

    def test_old_request_bytes_and_hashes_unchanged(self):
        rows=json.loads((Path(__file__).parent/"benchmarks/tools/output_alias_integration_v1/legacy-requests.json").read_text())
        for row in rows:
            with self.subTest(options=row["options"]):
                self.assertEqual(canonical(prepare_task(row["model"],row["options"])),canonical(row["request"]))
                self.assertEqual(reconstruct_request(row["request"]),row["request"])
                self.assertNotIn("--output-binding",runner_options(row["request"]))

    def test_policy_and_guidance_require_registered_capacity_graph(self):
        old=graph();old.pop("resource_profile");old["format"]=FORMAT
        with self.assertRaises(ValueError):prepare_task(old,options())
        with self.assertRaises(ValueError):prepare_task(graph(),dict(output_binding_policy="unknown"))
        with self.assertRaises(ValueError):prepare_task(graph(),dict(generation_guidance="explicit-v15"))

    def test_rehashed_value_forgery_is_rejected(self):
        q=prepare_task(graph(),options())
        logical=copy.deepcopy(q["layout"]["output_binding"]["logical_outputs"])
        logical[1]["value"]="a"
        q["layout"]=declarations(q["model"]["inputs"],logical,4,"general-tensors-v1",EXTENDED_ABI,q["layout"]["output_binding"]["model_sha256"])
        q["request_id"]=request_digest(q)
        with self.assertRaisesRegex(ValueError,"Model/layout mismatch"):validate_request(q)

    def test_rehashed_model_identity_forgery_is_rejected(self):
        q=prepare_task(graph(),options())
        q["layout"]["output_binding"]["model_sha256"]="0"*64
        q["request_id"]=request_digest(q)
        with self.assertRaises(ValueError):validate_request(q)

    def test_capability_and_resources_do_not_claim_execution(self):
        from application_capabilities import describe,assess_model
        self.assertEqual(describe()["output_binding_policies"][0]["policy"],POLICY)
        row=assess_model(graph(),options=options())
        self.assertEqual(row["resources"]["output_ciphertexts"]["value"],2)
        self.assertEqual(row["resources"]["output_elements"]["value"],10)
        self.assertFalse(row["encrypted_execution"])

    def test_repair_feedback_contains_only_frozen_public_binding(self):
        from unified_output_alias_guidance import repair_hint
        q=prepare_task(graph(),options())
        source='@hc.func("c,c")\ndef golden(x0, zero_ct):\n    return [x0,x0]\n'
        hint=repair_hint(q,source,stage="artifact_gate")
        self.assertIn("exact-values-v1",hint)
        for private in ("arrays.npz","/home/","API_KEY","private_reference_marker"):self.assertNotIn(private,hint)


class LoweringDifferential(unittest.TestCase):
    def test_capacity_and_mixed_final_output_order(self):
        import numpy as np
        from benchmark_graph import samples
        from benchmark_math import evaluate
        from test_unified_capacity_linear import interpreted_output
        from unified_graph_lowering import candidate_source
        from unified_mixed_layout import lower as mixed_lower
        for size, refs in ((5,("z","z")),(7,("z","a","z","a")),(20,("z",)*4)):
            model=graph(size,refs);q=prepare_task(model,options())
            for lower in (candidate_source,mixed_lower):
                source,record=lower(q)
                for probe in samples(model,4):
                    expected=evaluate(model,probe)
                    np.testing.assert_allclose(interpreted_output(source,q,probe),
                        np.concatenate([expected[o["name"]].reshape(-1)for o in model["outputs"]]),
                        atol=1e-12,rtol=1e-12)
                self.assertEqual(record["physical_output_chunks"],q["layout"]["output_ciphertexts"])
    def test_distinct_refs_retain_existing_runtime_identity_handling(self):
        from unified_graph_lowering import candidate_source
        model=graph(3,("z","b"))
        model["nodes"].append(dict(id="n2",op="square",inputs=["a"],attrs={},outputs=["b"]))
        q=prepare_task(model,options());source,record=candidate_source(q)
        self.assertEqual(q["layout"]["output_ciphertexts"],2)
        self.assertIsNotNone(record["output_identity"])


class BoundReferenceAdapters(unittest.TestCase):
    def test_complete_request_preparation_keeps_alias_arrays(self):
        import tempfile
        import numpy as np
        from unified_graph_prepare import prepare_request
        for selected in (True,False):
            model=graph(7,("z","a","z","a"))
            opts=options()if selected else dict(chunk_period=4,capacity_packing_version="capacity-public-tensors-v9")
            q=prepare_task(model,opts)
            with tempfile.TemporaryDirectory()as tmp:
                folder=Path(tmp);prepare_request(q,folder)
                self.assertEqual(json.loads((folder/"logical-bindings.json").read_text()),q["layout"])
                with np.load(folder/"arrays.npz",allow_pickle=False)as data:
                    refs=data["reference"]
                    np.testing.assert_array_equal(refs[:,:7],refs[:,14:21])
                    np.testing.assert_array_equal(refs[:,7:14],refs[:,21:])
                    self.assertEqual(data["inputs"].shape,(4,2,4))
    def test_changed_complete_request_is_rejected_before_arrays(self):
        import tempfile
        from unified_graph_prepare import prepare_request
        q=prepare_task(graph(),options());q["layout"]["outputs"][1]["alias_of"]="bad"
        q["request_id"]=request_digest(q)
        with tempfile.TemporaryDirectory()as tmp:
            with self.assertRaises(ValueError):prepare_request(q,Path(tmp))
            self.assertFalse((Path(tmp)/"arrays.npz").exists())
