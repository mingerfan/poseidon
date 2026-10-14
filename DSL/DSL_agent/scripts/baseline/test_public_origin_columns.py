"""Lossless metadata, binding, prompt budget and forged expansion rejection."""
import copy
import unittest
from benchmark_graph import canonical,digest
from public_origin_columns import compact,restore,project,registries,FORMAT,ROWS_FORMAT,MAX_ROWS
from public_weight_context import COMPACT_POLICY,project as public_project,request_digest,verify_binding,verify_context
from component_contract import prepare_task,reconstruct_request
from deepseek_provider import public_request
from public_weight_cases import models,period

def origins(rows=None):
    if rows is None:
        rows=[dict(node_id="n0",role="tensor_route",source="x",key="pv0",
            operand=0,input_chunk=0,output_chunk=0,step=2,baby_step=2,giant_step=0)for _ in range(128)]
    return dict(kind="capacity",policy="frozen",packing_masks="public",
        capacity_tensor_packing=dict(version="capacity-public-tensors-v9",bindings=rows,
            linear_strategies={},tensor_strategies={"n0":"route"}))
class OriginColumnsTests(unittest.TestCase):
    def test_exact_roundtrip(self):
        raw=origins();self.assertEqual(canonical(restore(compact(raw))),canonical(raw))
    def test_empty_bindings(self):
        raw=origins([]);self.assertEqual(restore(compact(raw)),raw)
    def test_no_packing(self):
        raw={"kind":"plain","scalar":-0.0};self.assertEqual(canonical(restore(compact(raw))),canonical(raw))
    def test_long_repeated_rows_shrink(self):
        raw=origins();self.assertLess(len(canonical(compact(raw))),len(canonical(raw))//3)
    def test_ragged_keys_and_order_preserved(self):
        rows=[{"x":1,"y":"a"},{"x":2},{"x":3,"y":"b"},{"x":4,"y":"b"}]
        raw=origins(rows);self.assertEqual(restore(compact(raw)),raw)
    def test_numeric_types_and_signed_zero(self):
        raw=origins([{"v":v}for v in (1,1.0,True,0.0,-0.0,None)])
        self.assertEqual(canonical(restore(compact(raw))),canonical(raw))
    def test_literals_with_unshared_values(self):
        raw=origins([{"v":str(i)}for i in range(16)])
        self.assertEqual(restore(compact(raw)),raw)
    def test_original_not_mutated(self):
        raw=origins();before=canonical(raw);out=compact(raw)
        out["value"]["capacity_tensor_packing"]["tensor_strategies"]["n0"]="changed"
        self.assertEqual(canonical(raw),before)
    def test_idempotent_public_projection(self):
        out=compact(origins());self.assertEqual(project(out),out)
    def test_hash_tamper(self):
        out=compact(origins());out["content_sha256"]="f"*64
        with self.assertRaisesRegex(ValueError,"content hash"):restore(out)
    def test_extra_fields_rejected(self):
        out=compact(origins());out["extra"]=0
        with self.assertRaisesRegex(ValueError,"format"):restore(out)
    def test_boolean_row_count_rejected(self):
        out=compact(origins());out["value"]["capacity_tensor_packing"]["bindings"]["row_count"]=True
        with self.assertRaisesRegex(ValueError,"row budget"):restore(out)
    def test_group_count_tamper_rejected(self):
        out=compact(origins());out["value"]["capacity_tensor_packing"]["bindings"]["groups"][0]["count"]+=1
        with self.assertRaisesRegex(ValueError,"total rows"):restore(out)
    def test_duplicate_column_rejected(self):
        out=compact(origins());group=out["value"]["capacity_tensor_packing"]["bindings"]["groups"][0]
        group["columns"].append(group["columns"][0])
        with self.assertRaisesRegex(ValueError,"column fields"):restore(out)
    def test_boolean_run_count_rejected(self):
        out=compact(origins());group=out["value"]["capacity_tensor_packing"]["bindings"]["groups"][0]
        group["data"]["node_id"]={"encoding":"rle","runs":[["n0",True]]}
        with self.assertRaisesRegex(ValueError,"run count"):restore(out)
    def test_noncanonical_alternate_encoding_rejected(self):
        out=compact(origins());group=out["value"]["capacity_tensor_packing"]["bindings"]["groups"][0]
        group["data"]["node_id"]={"encoding":"literal","values":["n0"]*128}
        with self.assertRaisesRegex(ValueError,"Canonical"):restore(out)
    def test_over_rows_rejected(self):
        out=compact(origins());out["value"]["capacity_tensor_packing"]["bindings"]["row_count"]=MAX_ROWS+1
        with self.assertRaisesRegex(ValueError,"row budget"):restore(out)
    def test_expansion_bomb_rejected_before_materialization(self):
        out=compact(origins());rows=out["value"]["capacity_tensor_packing"]["bindings"]
        rows["row_count"]=MAX_ROWS;group=rows["groups"][0];group["count"]=MAX_ROWS
        group["data"]["node_id"]={"encoding":"constant","value":"a"*256}
        with self.assertRaisesRegex(ValueError,"expanded byte budget"):restore(out)
    def test_nested_tree_rejected(self):
        raw={};cursor=raw
        for _ in range(70):cursor["next"]={};cursor=cursor["next"]
        with self.assertRaisesRegex(ValueError,"tree budget"):compact(raw)
    def test_nonfinite_rejected(self):
        with self.assertRaisesRegex(ValueError,"Finite"):compact({"nan":float("nan")})
class CompactPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=models()[0]
        cls.options=dict(chunk_period=period(cls.model),generation_guidance="explicit-v14",
            public_weight_policy=COMPACT_POLICY)
        cls.request=prepare_task(cls.model,cls.options)
    def test_host_and_public_are_separate(self):
        q=self.request;public=public_request(q)
        self.assertNotEqual(q["constant_origins"],public["constant_origins"])
        self.assertEqual(restore(public["constant_origins"]),q["constant_origins"])
        self.assertEqual(verify_binding(q),public)
        self.assertEqual(verify_context(public),public)
    def test_budget_retained(self):
        self.assertLessEqual(len(canonical(public_request(self.request))),131072)
    def test_no_dsl_answer_or_private_arrays_in_public_metadata(self):
        public=public_request(self.request)
        self.assertEqual(public["model"]["constants"],registries(public)["model_constants"])
        self.assertEqual(public["public_constants"],registries(public)["public_constants"])
        self.assertNotIn("hecate_source",public)
        self.assertNotIn("reference",public)
        self.assertNotIn("test_inputs",public)
    def test_roundtrip_request(self):
        self.assertEqual(reconstruct_request(self.request),self.request)
    def test_mutated_public_columns_rejected(self):
        public=public_request(self.request)
        public["constant_origins"]["content_sha256"]="f"*64
        with self.assertRaisesRegex(ValueError,"content hash"):verify_context(public)
    def test_origin_hash_changes_with_contents(self):
        a=origins();b=copy.deepcopy(a);b["capacity_tensor_packing"]["bindings"][0]["step"]=1
        self.assertNotEqual(compact(a)["content_sha256"],compact(b)["content_sha256"])
    def test_forged_host_origins_rejected(self):
        q=copy.deepcopy(self.request)
        q["constant_origins"]["packing_masks"]="forged"
        q["request_id"]=request_digest(q)
        with self.assertRaisesRegex(ValueError,"binding changed"):verify_binding(q)

    def test_registry_reference_hash_rejected(self):
        public=public_request(self.request)
        public["public_weight_binding"]["public_constants"]["content_sha256"]="f"*64
        with self.assertRaisesRegex(ValueError,"registry reference"):verify_context(public)
    def test_registry_path_is_closed(self):
        public=public_request(self.request)
        public["public_weight_binding"]["model_constants"]["source"]="outside/file"
        with self.assertRaisesRegex(ValueError,"registry reference"):verify_context(public)
    def test_projected_request_provider_roundtrip(self):
        public=public_request(self.request)
        self.assertEqual(public_request(public),public)

    def test_exact_typed_registry_reference(self):
        from public_origin_columns import typed_bindings,typed_reference
        public=public_request(self.request)
        registry=registries(public)["public_constants"]
        self.assertEqual(public["generation_guidance"]["typed_request_bindings"]["constant_bindings"],typed_reference(registry))
        self.assertEqual(self.request["generation_guidance"]["typed_request_bindings"]["constant_bindings"],typed_bindings(registry))
    def test_typed_reference_count_tamper(self):
        public=public_request(self.request)
        public["generation_guidance"]["typed_request_bindings"]["constant_bindings"]["count"]+=1
        with self.assertRaisesRegex(ValueError,"typed binding reference"):verify_context(public)
    def test_legacy_typed_list_not_accepted_as_compact_context(self):
        public=public_request(self.request)
        public["generation_guidance"]["typed_request_bindings"]["constant_bindings"]=self.request["generation_guidance"]["typed_request_bindings"]["constant_bindings"]
        with self.assertRaisesRegex(ValueError,"Canonical compact typed"):verify_context(public)

    def test_rank32_descriptor_walk_is_bounded(self):
        import subprocess,sys,os
        from pathlib import Path
        code="""from public_weight_context import descriptor
from public_math_context import shape
v=1.0
for _ in range(6):v=[v,v]
for _ in range(26):v=[v]
expected=[1]*26+[2]*6
assert descriptor(v)['shape']==expected
assert shape(v)==expected
"""
        env={k:v for k,v in os.environ.items()if k in ("PATH","LD_LIBRARY_PATH","LANG","IN_NIX_SHELL")}
        env.update(PYTHONPATH=str(Path(__file__).resolve().parent),PYTHONDONTWRITEBYTECODE="1",PYTHONNOUSERSITE="1")
        done=subprocess.run([sys.executable,"-B","-c",code],env=env,capture_output=True,text=True,timeout=5)
        self.assertEqual(done.returncode,0,done.stderr)
    def test_deep_rectangular_data_still_checks_each_branch(self):
        from public_weight_context import descriptor
        from public_math_context import shape
        for check in (descriptor,shape):
            v=[[1.0],[2.0,3.0]]
            for _ in range(25):v=[v]
            with self.assertRaises(ValueError):check(v)
            v=[[1.0],[float("nan")]]
            for _ in range(25):v=[v]
            with self.assertRaises(ValueError):check(v)
