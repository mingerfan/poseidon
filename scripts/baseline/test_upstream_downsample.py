
"""Actual upstream public vectors and immutable downsampling contracts."""
import copy,unittest
import numpy as np
from benchmark_graph import samples,digest
from benchmark_math import evaluate
from benchmark_torch import evaluate as reference
from unified_graph_contract import layout,prepare,validate_candidate,validate_request
from compiler_configuration import PROFILE_SHA256,configuration
from upstream_adapters import downsample_node as adapter
from upstream_downsample_cases import cases,corpus,graph
from upstream_candidate_helpers import manifest,DS_PROFILE,FUSED_PROFILE
from test_upstream_spatial import Vector

class DownsampleVectors(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from upstream_adapters.test_batch_norm import BatchNormAdapterTests
        BatchNormAdapterTests.setUpClass.__func__(cls)
    def test_actual_original_and_boundary_calls(self):
        for row in cases():
            g=row["model"];p=layout(g)["input_slot_period"];cap=manifest(g,DS_PROFILE)
            bindings={cap["helpers"][n]["binding"]["node_id"]:cap["helpers"][n]["binding"] for n in row["required_helpers"]}
            with self.subTest(case=row["name"]):
                for inputs in samples(g,16):
                    expected=evaluate(g,inputs);ref=reference(g,inputs)
                    env={k:Vector(np.tile(np.pad(v.reshape(-1),(0,p-v.size)),16384//p)) for k,v in inputs.items()}
                    for node in g["nodes"]:
                        if node["id"] in bindings:
                            b=bindings[node["id"]];x=env[b["input_value"]]
                            y,record=adapter.apply(b,x,self.helpers,self.mpcb);adapter.verify_record(b,record)
                            np.testing.assert_allclose(y.data[:p],adapter.probe(b,tuple(x.data[:p])),atol=1e-12,rtol=1e-12)
                        elif node["op"]=="negate":y=Vector(-env[node["inputs"][0]].data)
                        elif node["op"]=="square":y=Vector(env[node["inputs"][0]].data**2)
                        else:continue
                        env[node["outputs"][0]]=y
                    for output in g["outputs"]:
                        a=env[output["value"]].data;e=expected[output["name"]]
                        np.testing.assert_array_equal(a[:e.size],e.reshape(-1))
                        np.testing.assert_allclose(e,ref[output["name"]],atol=1e-12,rtol=1e-12)
                        np.testing.assert_array_equal(a,np.tile(a[:p],16384//p))
    def test_basis_and_unselected_values(self):
        g=graph((1,2,4,4));p=32;b=adapter.bind_node(g,g["nodes"][-1]["id"],p)
        for i in range(p):
            x=np.zeros(p);x[i]=1
            y,record=adapter.apply(b,Vector(np.tile(x,16384//p)),self.helpers,self.mpcb)
            np.testing.assert_array_equal(y.data[:p],adapter.probe(b,tuple(x)))
            adapter.verify_record(b,record)
    def test_actual_record_forgery_rejected(self):
        g=corpus()[0];p=layout(g)["input_slot_period"];b=adapter.bind_node(g,g["nodes"][-1]["id"],p)
        _,record=adapter.apply(b,Vector(np.tile(np.arange(p),16384//p)),self.helpers,self.mpcb)
        for kind in ("helper","geometry","position","calls","work"):
            bad=copy.deepcopy(record)
            if kind=="helper":bad["inner_calls"][0]["helper"]="HE_Avg"
            if kind=="geometry":bad["inner_calls"][0]["geometry"]["s"]=1
            if kind=="position":bad["inner_calls"][0]["output_positions"][0]+=1
            if kind=="calls":bad["inner_calls"]=[]
            if kind=="work":bad["mapping_operations"]+=1
            with self.subTest(kind=kind),self.assertRaises(ValueError):adapter.verify_record(b,bad)

class DownsampleContracts(unittest.TestCase):
    def request(self,row):
        return prepare(row["model"],PROFILE_SHA256,configuration(row["configuration"]),
                       helper_profile=DS_PROFILE,helper_exercise=row["required_helpers"])
    def test_all_sources_and_contribution(self):
        for row in cases():
            with self.subTest(case=row["name"]):
                r=self.request(row);validate_candidate(dict(schema=1,request_id=r["request_id"],hecate_source=row["source"]),r)
    def test_rehashed_parameter_and_layout_forgery(self):
        r=self.request(cases()[0])
        for kind in ("slice","index","geometry","position","map","work"):
            bad=copy.deepcopy(r);b=bad["upstream_helpers"]["helpers"]["HE_DS0"]["binding"]
            if kind=="slice":b["original_slices"][0]["attrs"]["step"]=1
            if kind=="index":b["row_indices"][0]+=1
            if kind=="geometry":b["calls"][0]["inner"]["geometry"]["s"]=1
            if kind=="position":b["calls"][0]["inner"]["output_positions"][0]+=1
            if kind=="map":b["calls"][0]["after"][0]["mask"][0]+=.1
            if kind=="work":b["work"]=1
            bad["request_id"]=digest({k:v for k,v in bad.items() if k!="request_id"})
            with self.subTest(kind=kind),self.assertRaises(ValueError):validate_request(bad)
    def test_old_profile_and_chunk_boundaries(self):
        g=corpus()[0]
        self.assertFalse(any(n.startswith("HE_DS") for n in manifest(g,FUSED_PROFILE)["helpers"]))
        self.assertNotIn("upstream_helpers",prepare(g,PROFILE_SHA256))
        with self.assertRaises(ValueError):prepare(g,PROFILE_SHA256,helper_profile=DS_PROFILE,chunk_period=4)
    def test_reject_wrong_axis_step_and_nonproducer(self):
        for kind in ("axis","step","producer"):
            g=copy.deepcopy(corpus()[1]);n=g["nodes"][-1]
            if kind=="axis":n["attrs"].update(axis=-1,stop=2)
            if kind=="step":n["attrs"]["step"]=1
            if kind=="producer":n["inputs"]=["input0"]
            with self.subTest(kind=kind),self.assertRaises(ValueError):adapter.bind_node(g,n["id"],16)
    def test_cancelling_real_call_not_contribution(self):
        r=self.request(cases()[0])
        src='@hc.func("c,c")'+chr(10)+'def golden(x,zero_ct):'+chr(10)+'    h = HE_DS0(x)'+chr(10)+'    return h-h+x'+chr(10)
        with self.assertRaisesRegex(ValueError,"contribution"):
            validate_candidate(dict(schema=1,request_id=r["request_id"],hecate_source=src),r)

if __name__=="__main__":unittest.main()
