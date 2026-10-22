"""Lossless capacity metadata transport and bounded malformed-input rejection."""
import copy,json,unittest
from unified_request_storage import compact_request,restore_public_constants,compact_bindings,restore_bindings,BINDINGS_VERSION
from component_contract import prepare_task,reconstruct_request
from candidate_contract import request_file_bytes,trace_payload_json,MAX_BYTES
from unified_graph_lowering import candidate_source
from unified_graph_contract import validate_request

def graph():
    def node(name,op,refs,attrs):return dict(id=name,op=op,inputs=refs,attrs=attrs,outputs=[name+"_out"])
    return dict(format="poseidon-model-graph-v2",id="wire_reduction",resource_profile="medium-v1",
        inputs=[dict(name="x",shape=[3,129]),dict(name="y",shape=[1,129])],constants={"factor":0.5},
        nodes=[node("join","concat",["x","y"],dict(axis=0)),
               node("total","sum",["join_out"],dict(axes=[-1],keepdims=True)),
               node("average","mean",["join_out"],dict(axes=[0],keepdims=False)),
               node("half","multiply",["total_out","factor"],{}),
               node("stacked","stack",["total_out","half_out"],dict(axis=-1))],
        outputs=[dict(name="stacked_totals",value="stacked_out"),dict(name="average",value="average_out")])

class BindingStorageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.request=prepare_task(graph(),dict(chunk_period=256,generation_guidance="explicit-v10"))
    def descriptor(self):
        return copy.deepcopy(compact_request(self.request)["constant_origins"]["capacity_tensor_packing"])
    def test_complete_request_roundtrip_and_hash_preserved(self):
        before=copy.deepcopy(self.request)
        restored=json.loads(request_file_bytes(self.request))
        self.assertEqual(restored["constant_origins"]["capacity_tensor_packing"]["bindings"]["format"],BINDINGS_VERSION)
        validate_request(restored)
        self.assertEqual(restored,self.request)
        self.assertEqual(reconstruct_request(restored),self.request)
        self.assertEqual(self.request,before)
    def test_original_combined_trace_byte_gate_preserved(self):
        source,_=candidate_source(self.request)
        candidate=dict(schema=1,request_id=self.request["request_id"],hecate_source=source)
        encoded=trace_payload_json(self.request,candidate)
        self.assertLessEqual(len(encoded.encode()),MAX_BYTES)
        with self.assertRaisesRegex(ValueError,"Trace payload size"):
            trace_payload_json(self.request,dict(candidate,hecate_source="x"*MAX_BYTES))
    def test_older_capacity_wire_versions_unchanged(self):
        for version in ("capacity-public-tensors-v1","capacity-public-tensors-v2","capacity-public-tensors-v3","capacity-public-tensors-v4"):
            r=copy.deepcopy(self.request);r["constant_origins"]["capacity_tensor_packing"]["version"]=version
            self.assertIs(compact_request(r),r)
    def test_optional_fields_and_order_preserved(self):
        rows=[dict(node_id="node"+str(i),role="route",source="x",key="pv0",**({"step":i} if i%2 else {}))
              for i in range(20)]
        encoded=compact_bindings(rows)
        self.assertIsInstance(encoded,dict)
        d=dict(bindings=encoded);restore_bindings(d)
        self.assertEqual(d["bindings"],rows)
    def test_binding_rows_limit(self):
        d=self.descriptor();d["bindings"]["rows"]=[d["bindings"]["rows"][0]]*1025
        with self.assertRaisesRegex(ValueError,"storage rows"):restore_bindings(d)
    def test_duplicate_columns_rejected(self):
        d=self.descriptor();d["bindings"]["columns"][1]=d["bindings"]["columns"][0]
        with self.assertRaisesRegex(ValueError,"storage columns"):restore_bindings(d)
    def test_width_and_unknown_format_rejected(self):
        for mutate in (lambda t:t["rows"][0].append(1),lambda t:t.update(format="unknown")):
            d=self.descriptor();mutate(d["bindings"])
            with self.assertRaises(ValueError):restore_bindings(d)
    def test_nested_bool_nonfinite_and_large_cells_rejected(self):
        for value in ({},True,float("nan"),2**40,"x"*129):
            d=self.descriptor();d["bindings"]["rows"][0][0]=value
            with self.assertRaisesRegex(ValueError,"storage cell"):restore_bindings(d)
    def test_empty_row_rejected(self):
        d=self.descriptor();d["bindings"]["rows"][0]=[None]*len(d["bindings"]["columns"])
        with self.assertRaisesRegex(ValueError,"empty row"):restore_bindings(d)
    def test_tampered_routes_still_rejected_by_semantic_identity(self):
        r=copy.deepcopy(compact_request(self.request))
        d=r["constant_origins"]["capacity_tensor_packing"]["bindings"]
        index=d["columns"].index("key");d["rows"][0][index]="pv_unknown"
        with self.assertRaises(ValueError):validate_request(r)
if __name__=="__main__":unittest.main()
