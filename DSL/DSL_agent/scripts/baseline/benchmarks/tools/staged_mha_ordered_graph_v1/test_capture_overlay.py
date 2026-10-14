"""Standard-library controls; real Torch/numerical acceptance is separate."""
import ast,copy,hashlib,importlib.util,json,sys,unittest
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
BASE=HERE.parents[2]
OVERLAY=HERE/"overlay"
sys.path.insert(0,str(BASE))
spec=importlib.util.spec_from_file_location("ordered_mha_helper_control",OVERLAY/"pytorch_mha_source.py")
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)

class CaptureOverlay(unittest.TestCase):
    def test_original_production_hashes(self):
        manifest=json.loads((HERE/"capture-overlay.json").read_text())
        for name,expected in manifest["original"].items():
            self.assertEqual(hashlib.sha256((BASE/name).read_bytes()).hexdigest(),expected)
    def test_candidates_compile_and_match_manifest(self):
        manifest=json.loads((HERE/"capture-overlay.json").read_text())
        for name,expected in manifest["candidates"].items():
            raw=(OVERLAY/name).read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(),expected)
            compile(raw.decode(),name,"exec")
        self.assertFalse(manifest["production_applied"])
        self.assertEqual([manifest[n] for n in ("nix_entries","api","compiler","encrypted_execution")],[0]*4)
    def test_manifest_self_binding(self):
        value=json.loads((HERE/"capture-overlay.json").read_text());claimed=value.pop("binding")
        self.assertEqual(claimed,helper.digest(value))
    def test_frozen_math_functions(self):
        original=ast.parse((BASE/"benchmarks/tools/static_mha_source_draft_v1/mha_source_plan.py").read_text())
        candidate=ast.parse((OVERLAY/"pytorch_mha_math.py").read_text())
        def functions(tree):
            return {n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        a,b=functions(original),functions(candidate);a.pop("implementation");b.pop("implementation")
        self.assertEqual(a,b)
    def test_explicit_policy_defaults(self):
        for name,entry in (("pytorch_static_trace.py","trace"),("pytorch_model_capture.py","capture")):
            function=next(n for n in ast.parse((OVERLAY/name).read_text()).body if isinstance(n,ast.FunctionDef) and n.name==entry)
            defaults=dict(zip([x.arg for x in function.args.kwonlyargs],function.args.kw_defaults))
            self.assertIsNone(ast.literal_eval(defaults["mha_source_policy"]))
        self.assertEqual(helper.IMPLEMENTATION,("pytorch_mha_source.py","pytorch_mha_math.py"))
    def test_no_model_case_dispatch(self):
        tree=ast.parse((OVERLAY/"pytorch_mha_source.py").read_text())
        calls={n.func.id for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)}
        self.assertFalse({"eval","exec","__import__"}&calls)
        names={n.id for n in ast.walk(tree) if isinstance(n,ast.Name)}
        self.assertFalse({"case_name","model_name","family","factory"}&names)
    def test_complete_three_input_signature(self):
        values=helper.arguments(("q","k","v"),{})
        self.assertEqual([values[n] for n in helper.NAMES[:3]],["q","k","v"])
        self.assertEqual([values[n] for n in helper.NAMES[3:]],[None,True,None,True,False])
        self.assertEqual(helper.arguments((),dict(query="q",key="k",value="v",need_weights=False))["need_weights"],False)
    def test_missing_operand_rejected(self):
        with self.assertRaises(ValueError):helper.arguments(("q","k"),{})
    def test_duplicate_operand_rejected(self):
        with self.assertRaises(ValueError):helper.arguments(("q","k","v"),{"query":"other"})
    def test_unknown_call_option_rejected(self):
        with self.assertRaises(ValueError):helper.arguments(("q","k","v"),{"training":False})
    def test_non_boolean_options_rejected(self):
        for field in ("need_weights","average_attn_weights","is_causal"):
            with self.subTest(field=field),self.assertRaises(ValueError):
                helper.arguments(("q","k","v"),{field:1})
    def test_module_paths_exact_ancestors(self):
        helper._module_paths(["outer","outer.attention"],"outer.attention")
        for paths in (["different","outer.attention"],["outer"],["outer.attention","outer.attention"]):
            with self.subTest(paths=paths),self.assertRaises(ValueError):
                helper._module_paths(paths,"outer.attention")
    def test_projection_sources_packed_and_separate(self):
        nodes=[dict(op="linear",inputs=["x","w"+str(i),"b"+str(i)]) for i in range(4)]
        snapshot=dict(config=dict(embed_dim=4,same_projection=True),parameters={"in_proj_bias":[],"out_proj.bias":[]})
        rows=helper._projection_specs(snapshot,dict(nodes=nodes))
        self.assertEqual([x["slice"]["start"] for x in rows if x["parameter"]=="in_proj_weight"],[0,4,8])
        self.assertTrue(all(x["slice"] is None for x in rows if x["role"]=="output"))
        snapshot["config"]["same_projection"]=False
        rows=helper._projection_specs(snapshot,dict(nodes=nodes))
        self.assertEqual([x["parameter"] for x in rows[::2]],["q_proj_weight","k_proj_weight","v_proj_weight","out_proj.weight"])
        self.assertTrue(all(x["slice"] is None for x in rows[::2]))
    def test_projection_inventory_rejects_omission(self):
        with self.assertRaises(ValueError):helper._projection_specs({"config":{"embed_dim":2,"same_projection":True}},{"nodes":[]})

class SharedMaskControls(unittest.TestCase):
    def fixture(self):
        mask=dict(dtype="bool",values=[[False,True],[False,False]],negative_infinity=None)
        return {"attn_mask":copy.deepcopy(mask),"key_padding_mask":copy.deepcopy(mask)}, {
                "attn_mask":"mask","key_padding_mask":"mask"}
    def test_shared_original_mask_same_typed_values(self):
        options,bindings=self.fixture()
        result=helper._mask_states(options,bindings)
        self.assertEqual(set(result),{"mask"})
        self.assertEqual(result["mask"],options["attn_mask"])
    def test_shared_original_mask_first_value_tamper_rejected(self):
        options,bindings=self.fixture();options["attn_mask"]["values"][0][0]=True
        with self.assertRaisesRegex(ValueError,"Shared original MHA mask"):
            helper._mask_states(options,bindings)
    def test_shared_original_mask_dtype_tamper_rejected(self):
        options,bindings=self.fixture();options["attn_mask"]["dtype"]="float64"
        options["attn_mask"]["values"]=[[0.,1.],[0.,0.]]
        with self.assertRaisesRegex(ValueError,"Shared original MHA mask"):
            helper._mask_states(options,bindings)
    def test_missing_mask_source_or_untyped_path_rejected(self):
        options,bindings=self.fixture()
        for path in (None,[],""):
            with self.subTest(path=path),self.assertRaises(ValueError):
                helper._mask_states(options,dict(bindings,attn_mask=path))

class ProvenanceControls(unittest.TestCase):
    def fixture(self):
        raw=dict(nodes=[dict(id="fx_mha_exp",op="exp",attrs={"approximation":{"degree":4}}),
                        dict(id="fx_mha_recip",op="reciprocal",attrs={"approximation":{"iterations":8}})])
        high=copy.deepcopy(raw);high["nodes"][0]["id"]="node_1";high["nodes"][1]["id"]="node_2"
        origins={"node_1":{"source_id":"fx_mha_exp"},"node_2":{"source_id":"fx_mha_recip"}}
        locations={n["id"]:{"source_node":"attention","module_paths":["attention"]} for n in raw["nodes"]}
        return raw,high,origins,locations
    def test_normalized_map_checks_exact_non_linearity(self):
        raw,high,origins,locations=self.fixture()
        result=helper._locations(high,raw,locations,origins)
        self.assertEqual(set(result),{"node_1","node_2"})
        self.assertEqual(result["node_1"],locations["fx_mha_exp"])
    def test_normalized_map_missing_component_rejected(self):
        raw,high,origins,locations=self.fixture()
        high["nodes"].pop();origins.pop("node_2")
        with self.assertRaises(ValueError):helper._locations(high,raw,locations,origins)
    def test_normalized_map_double_component_rejected(self):
        raw,high,origins,locations=self.fixture()
        high["nodes"].append(dict(id="node_3",op="exp",attrs=raw["nodes"][0]["attrs"]))
        origins["node_3"]={"source_id":"fx_mha_exp"}
        with self.assertRaises(ValueError):helper._locations(high,raw,locations,origins)
    def test_normalized_map_profile_change_rejected(self):
        raw,high,origins,locations=self.fixture();high["nodes"][0]["attrs"]["approximation"]["degree"]=7
        with self.assertRaises(ValueError):helper._locations(high,raw,locations,origins)
    def test_normalized_origins_required_for_distinct_graph(self):
        raw,high,origins,locations=self.fixture()
        with self.assertRaises(ValueError):helper._locations(high,raw,locations,None)
    def test_record_cannot_omit_marked_source_segment(self):
        emissions=dict(nodes=[dict(id="fx_attention",op="linear")],constants={})
        with patch.object(helper,"_check_entry",return_value=({},["fx_attention"])),patch.object(helper,"_locations"):
            with self.assertRaises(ValueError):
                helper.record(emissions,emissions,[],source_origins={"fx_attention":{"mha":{}}},
                              public_constant_views={},resource_profile="general-tensors-v1")
    def test_record_rejects_reusing_source_segment(self):
        emissions=dict(nodes=[dict(id="fx_attention",op="linear")],constants={})
        entries=[{"source_node":"attention"},{"source_node":"attention_1"}]
        with patch.object(helper,"_check_entry",return_value=({},["fx_attention"])),patch.object(helper,"_locations"):
            with self.assertRaises(ValueError):
                helper.record(emissions,emissions,entries,source_origins={"fx_attention":{"mha":{}}},
                              public_constant_views={},resource_profile="general-tensors-v1")
    def test_partial_loaded_binding_rejected_before_math(self):
        with self.assertRaises(ValueError):
            helper.check_record({}, {},emissions={},source_origins={},public_constant_views={},
                                state_constant_sources={},resource_profile="general-tensors-v1")

if __name__=="__main__":unittest.main()
