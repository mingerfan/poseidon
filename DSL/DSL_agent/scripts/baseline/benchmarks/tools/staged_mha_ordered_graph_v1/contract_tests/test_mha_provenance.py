"""Loaded-state MHA host checks and re-signed closure rejection controls.

The read-only captures retain their old source identities and reports. These
controls call the current Agent-owned MHA checker with the original manifest and
actual no-pickle NPY files; they do not reinterpret old reference identity as a
new source capture. The added shared-mask fixture is direct pinned-Torch capture,
not a bubblewrap/FHE/Agent-generation acceptance result.
"""
import copy
import hashlib
import io
import json
import sys
import unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
STAGE=HERE.parent
BASE=STAGE.parents[2]
OVERLAY=STAGE/"overlay"
sys.path.insert(0,str(BASE));sys.path.insert(0,str(OVERLAY));sys.path.insert(0,str(HERE))
import test_ordered_mha_dataflow as dataflow
from benchmark_graph import digest

CAPTURE=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-mha-ordered-capture-v1/boolean_padding_residual")


def resign(value):
    value["binding"]=digest({k:v for k,v in value.items() if k!="binding"})
    return value


class ProvenanceControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dataflow.OrderedMhaDataflow.setUpClass()
        import pytorch_mha_source as source
        import pytorch_mha_math as math
        import pytorch_weight_package as weights
        cls.source,cls.math,cls.weights=source,math,weights
        cls.harness=dataflow.OrderedMhaDataflow()
        cls.report=json.loads((CAPTURE/"capture-reference.json").read_text())
        cls.original=json.loads((CAPTURE/"original.json").read_text())
        cls.manifest=json.loads((CAPTURE/"trusted-python/manifest.json").read_text())
        arrays,_=weights.read(CAPTURE/"trusted-python",cls.manifest["weights"],resource_profile="general-tensors-v1")
        cls.expected=weights.binding(arrays,cls.manifest["weights"])
        cls.records=cls.manifest["weights"]
        cls.frozen={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [CAPTURE/"capture-reference.json",CAPTURE/"original.json",CAPTURE/"trusted-python/manifest.json"]+
            list((CAPTURE/"trusted-python").glob("*.npy"))}

    @classmethod
    def tearDownClass(cls):
        for name,expected in cls.frozen.items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=expected:
                raise AssertionError("A historical capture, manifest or public weight was changed")

    def bundle(self):
        metadata=copy.deepcopy(self.report["capture_metadata"])
        alias=metadata["ordered_alias_capture"]
        return dict(high=copy.deepcopy(self.original),metadata=metadata,alias=alias,
            emissions=alias["capture_emissions"],record=metadata["mha_source_capture"],
            origins=alias["capture_source_origins"],views=alias["capture_public_constant_views"],
            normalized=alias["normalization"]["mathematical_node_origins"],
            state_sources=alias["capture_emission_state_constant_sources"],
            records=copy.deepcopy(self.records),expected=copy.deepcopy(self.expected))

    def bind_record(self,bundle):
        bundle["record"]["graph_sha256"]=digest(bundle["high"])
        bundle["record"]["emission_graph_sha256"]=digest(bundle["emissions"])
        resign(bundle["record"])

    def check(self,bundle,*,loaded=True):
        extra=dict(state_constant_sources=bundle["state_sources"],declared_records=bundle["records"],
                   expected_weight_binding=bundle["expected"]) if loaded else {}
        return self.source.check_record(bundle["record"],bundle["high"],emissions=bundle["emissions"],
            normalized_origins=bundle["normalized"],source_origins=bundle["origins"],
            public_constant_views=bundle["views"],resource_profile="general-tensors-v1",**extra)

    def test_original_manifest_and_actual_npys_match_loaded_state(self):
        bundle=self.bundle();locations=self.check(bundle)
        self.assertTrue(locations)
        self.assertTrue(set(locations)<={n["id"] for n in bundle["emissions"]["nodes"]})
        self.assertEqual(bundle["expected"],bundle["metadata"]["weight_binding"])

    def test_resigned_mask_direction_rejected(self):
        b=self.bundle();b["record"]["entries"][0]["mask_direction"]="bool True allows"
        self.bind_record(b)
        with self.assertRaisesRegex(ValueError,"call/mask inventory"):self.check(b)

    def test_original_mask_state_change_rejected(self):
        b=self.bundle();entry=b["record"]["entries"][0]
        path=entry["mask_bindings"]["attn_mask"]
        constant=b["emissions"]["constants"][entry["state_bindings"][path]]
        constant[0][0]=1-float(constant[0][0]);self.bind_record(b)
        with self.assertRaisesRegex(ValueError,"original state differs"):self.check(b)

    def test_actual_loaded_mask_dtype_cannot_be_relabeled(self):
        b=self.bundle();entry=b["record"]["entries"][0]
        path=entry["mask_bindings"]["attn_mask"]
        self.assertEqual(b["records"][path]["dtype"],"|b1")
        b["records"][path]["dtype"]="<f8"
        # Re-sign the declared-record hash. The exact semantic dtype comparison
        # must still reject even when normalized float64 contents have same hash.
        b["expected"]["declarations_sha256"]=digest(b["records"])
        with self.assertRaisesRegex(ValueError,"original mask dtype"):self.check(b)

    def test_packed_projection_indices_tamper_rejected(self):
        b=self.bundle();entry=b["record"]["entries"][0]
        projection=next(p for p in entry["projections"] if p["slice"] is not None)
        actual=entry["constant_bindings"][projection["constant"]]
        b["views"][actual]["indices"][0][0]+=1
        with self.assertRaisesRegex(ValueError,"packed projection slice provenance"):self.check(b)

    def test_packed_projection_source_tamper_rejected(self):
        b=self.bundle();entry=b["record"]["entries"][0]
        projection=next(p for p in entry["projections"] if p["slice"] is not None)
        actual=entry["constant_bindings"][projection["constant"]]
        b["views"][actual]["source"]=entry["state_bindings"][entry["module_path"]+".out_proj.weight"]
        with self.assertRaisesRegex(ValueError,"packed projection slice provenance"):self.check(b)

    def test_direct_parameter_state_source_is_exact(self):
        b=self.bundle();entry=b["record"]["entries"][0]
        path=entry["module_path"]+".out_proj.weight"
        b["state_sources"][entry["state_bindings"][path]]=entry["module_path"]+".in_proj_weight"
        with self.assertRaisesRegex(ValueError,"original emission state source"):self.check(b)

    def test_missing_entry_and_missing_origin_marker_rejected(self):
        b=self.bundle();b["record"]["entries"]=[];self.bind_record(b)
        with self.assertRaisesRegex(ValueError,"Complete native MHA source segments"):self.check(b)
        b=self.bundle();entry=b["record"]["entries"][0]
        identifier=next(iter(entry["node_bindings"].values()))
        del b["origins"][identifier]["mha"]
        with self.assertRaisesRegex(ValueError,"call/emission origin binding"):self.check(b)

    def test_original_call_inputs_cannot_be_swapped(self):
        b=self.bundle();entry=b["record"]["entries"][0]
        entry["call_inputs"]["query"]["value"]=entry["call_inputs"]["key"]["value"]
        self.bind_record(b)
        with self.assertRaisesRegex(ValueError,"input call binding|call/emission origin binding"):self.check(b)

    def test_original_fx_node_and_module_selector_path_rejected(self):
        for field,value in (("source_node","invented_source"),("module_paths",["invented_module"])):
            b=self.bundle();b["record"]["entries"][0][field]=value;self.bind_record(b)
            with self.assertRaisesRegex(ValueError,"identity|module paths|call/emission origin binding"):self.check(b)

    def test_output_port_and_noncomponent_mapping_tamper_rejected(self):
        b=self.bundle();entry=b["record"]["entries"][0]
        entry["outputs"]["attn_output"]=entry["outputs"]["attn_weights"];self.bind_record(b)
        with self.assertRaisesRegex(ValueError,"tuple output binding"):self.check(b)

    def shared_bundle(self):
        torch=self.harness.torch;np=self.harness.np;before=torch.get_rng_state().clone()
        class Shared(torch.nn.Module):
            def __init__(self):
                super().__init__();self.attention=torch.nn.MultiheadAttention(2,1,batch_first=True,dtype=torch.float64).eval()
                self.register_buffer("shared_mask",torch.tensor([[False,True],[False,True]],dtype=torch.bool))
                with torch.no_grad():
                    offset=0
                    for value in self.parameters():
                        value.copy_((torch.arange(offset+1,offset+1+value.numel(),dtype=torch.float64).sin()*.04).reshape(value.shape))
                        offset+=value.numel()
            def forward(self,q):
                return self.attention(q,q,q,attn_mask=self.shared_mask,key_padding_mask=self.shared_mask)
        try:module=Shared().eval()
        finally:torch.set_rng_state(before)
        from pytorch_matrix_plan import POLICY as MATRIX
        from pytorch_fused_matrix_normalization import POLICY as FUSED
        from pytorch_approximation import POLICY as APPROX
        from pytorch_model_capture import ADAPTIVE_FUNCTION_POLICY
        prepared=self.harness.ordered.prepare_module(module,[{"name":"q","shape":[2,2,2]}],["result","weights"],
            approximations=dataflow.configuration(),resource_profile="general-tensors-v1",dtype_policy="strict-float64-v1",
            matrix_policy=MATRIX,approximation_policy=APPROX,functional_policy=ADAPTIVE_FUNCTION_POLICY,
            common_calls=True,fused_matrix_policy=FUSED,gelu_source_policy="poseidon-gelu-source-variants-v1",
            mha_source_policy=dataflow.POLICY)
        high,_,binding,_=prepared;metadata=binding["pytorch_capture"];alias=metadata["ordered_alias_capture"]
        records={};arrays={}
        for i,(name,value) in enumerate(module.state_dict().items()):
            array=np.asarray(value.detach().numpy());stream=io.BytesIO();np.save(stream,array,allow_pickle=False);raw=stream.getvalue()
            item={"file":f"shared_{i}.npy","sha256":hashlib.sha256(raw).hexdigest(),"bytes":len(raw),"shape":list(array.shape),"dtype":array.dtype.str}
            arrays[name]=self.weights.decode(raw,item);records[name]=item
        expected=self.weights.binding(arrays,records)
        return dict(high=high,metadata=metadata,alias=alias,emissions=alias["capture_emissions"],record=metadata["mha_source_capture"],
            origins=alias["capture_source_origins"],views=alias["capture_public_constant_views"],
            normalized=alias["normalization"]["mathematical_node_origins"],
            state_sources=alias["capture_emission_state_constant_sources"],records=records,expected=expected)

    def test_shared_original_mask_accepts_both_uses_and_single_bound_state(self):
        b=self.shared_bundle();entry=b["record"]["entries"][0]
        self.assertEqual(entry["mask_bindings"]["attn_mask"],entry["mask_bindings"]["key_padding_mask"])
        self.assertTrue(self.check(b))

    def test_shared_mask_first_option_changed_with_full_mini_rebuild_rejected(self):
        b=self.shared_bundle();entry=b["record"]["entries"][0]
        request=entry["source_request"]
        request["options"]["attn_mask"]["values"]=[[False,False],[False,False]]
        entry["source_request_sha256"]=digest(request)
        mini=self.math.plan(**request)["mathematical_graph"]
        # Both masks retain the same native source path. We rebuild every mini
        # constant and operator so rejection cannot be credited to stale hashes.
        self.assertEqual(set(entry["constant_bindings"]),set(mini["constants"]))
        for local,value in mini["constants"].items():b["emissions"]["constants"][entry["constant_bindings"][local]]=copy.deepcopy(value)
        remap={x["name"]:x["name"] for x in mini["inputs"]};remap.update(entry["constant_bindings"])
        nodes={n["id"]:n for n in b["emissions"]["nodes"]}
        for node in mini["nodes"]:
            actual=nodes[entry["node_bindings"][node["id"]]]
            actual.update(op=node["op"],inputs=[remap[v] for v in node["inputs"]],attrs=copy.deepcopy(node["attrs"]))
            remap[node["outputs"][0]]=entry["value_bindings"][node["outputs"][0]]
        self.bind_record(b)
        with self.assertRaisesRegex(ValueError,"Shared original MHA mask options differ"):self.check(b)


if __name__=="__main__":unittest.main(verbosity=2)
