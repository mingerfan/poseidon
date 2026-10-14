"""Real 65536-slot helpers on public vectors, independent of ciphertext reference."""
import copy,unittest,math
import numpy as np
from benchmark_suite import Builder,generate
from benchmark_graph import samples
from benchmark_math import evaluate
from benchmark_torch import evaluate as reference
from unified_graph_contract import layout
from upstream_adapters import virtual_node as adapter
from test_upstream_spatial import Vector

from upstream_virtual_cases import models

class VirtualNodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from upstream_adapters.test_batch_norm import BatchNormAdapterTests
        BatchNormAdapterTests.setUpClass.__func__(cls)
    def test_original_and_extended_against_dual_reference(self):
        for index,(family,g) in enumerate(models()):
            p=layout(g)["input_slot_period"]
            node=next(n for n in g["nodes"] if n["op"]==("batch_norm" if family=="HE_MPBN" else "linear"))
            binding=adapter.bind_node(g,node["id"],p,family);before=copy.deepcopy(g)
            for inputs in samples(g,4):
                a=inputs["input0"].reshape(-1)
                v=Vector(np.tile(np.pad(a,(0,p-a.size)),16384//p));zero=Vector(np.zeros(16384))
                with self.subTest(index=index,family=family,shape=a.shape):
                    out,record=adapter.apply(binding,v,zero,self.helpers,self.mpcb)
                    y=out.data[:math.prod(binding["output_shape"])]
                    if g["nodes"][-1]["op"]=="negate":y=-y
                    expected=evaluate(g,inputs)["output0"].reshape(-1)
                    np.testing.assert_allclose(y,expected,atol=1e-12,rtol=1e-12)
                    np.testing.assert_allclose(expected,reference(g,inputs)["output0"].reshape(-1),atol=1e-12,rtol=1e-12)
                    np.testing.assert_array_equal(out.data,np.tile(out.data[:p],16384//p))
                    np.testing.assert_allclose(adapter.probe(binding,tuple(v.data[:p]),(0.,)*p),out.data[:p],atol=1e-12,rtol=1e-12)
                    adapter.verify_record(binding,record)
            self.assertEqual(before,g)
    def test_transcript_and_binding_forgery_rejected(self):
        family,g=models()[8];p=layout(g)["input_slot_period"];n=next(n for n in g["nodes"] if n["op"]=="linear")
        b=adapter.bind_node(g,n["id"],p,family)
        _,r=adapter.apply(b,Vector(np.tile(np.arange(p)/16,16384//p)),Vector(np.zeros(16384)),self.helpers,self.mpcb)
        for kind in ("slots","constant","rotation","work","helper","binding"):
            bad=copy.deepcopy(r)
            v=bad["inner_calls"][0]["virtual"]
            if kind=="slots":v["virtual_slots"]=16384
            if kind=="constant":v["full_constants"][0]["sha256"]="0"*64
            if kind=="rotation":v["events"][1]["operation"]="add"
            if kind=="work":bad["operation_count"]+=1
            if kind=="helper":bad["helper"]="HE_BN"
            if kind=="binding":bad["binding_sha256"]="0"*64
            with self.subTest(kind=kind),self.assertRaises(ValueError):adapter.verify_record(b,bad)
    def test_reshape_mapping_is_nonidentity(self):
        rows=[(f,g) for f,g in models() if f=="HE_ReshapeLinear" and g["inputs"][0]["shape"]==[1,4,2,2]]
        f,g=rows[0];n=g["nodes"][-1];b=adapter.bind_node(g,n["id"],16,f)
        perm=b["calls"][0]["packed_to_original"]
        self.assertNotEqual(perm,list(range(16)));self.assertEqual(sorted(perm),list(range(16)))


class VirtualContracts(unittest.TestCase):
    def request(self,row):
        from compiler_configuration import PROFILE_SHA256,configuration
        from unified_graph_contract import prepare
        from upstream_candidate_helpers import VR_PROFILE
        return prepare(row["model"],PROFILE_SHA256,configuration(row["configuration"]),
                       helper_profile=VR_PROFILE,helper_exercise=row["required_helpers"])
    def test_all_fixed_sources_and_output_contribution(self):
        from upstream_virtual_cases import cases
        from unified_graph_contract import validate_candidate
        for row in cases():
            with self.subTest(case=row["name"]):
                r=self.request(row);validate_candidate(dict(schema=1,request_id=r["request_id"],hecate_source=row["source"]),r)
    def test_parameter_rehash_does_not_authorize_forgery(self):
        from upstream_virtual_cases import cases
        from unified_graph_contract import validate_request
        from benchmark_graph import digest
        r=self.request(cases()[8]);bad=copy.deepcopy(r)
        b=bad["upstream_helpers"]["helpers"]["HE_Linear0"]["binding"]
        b["calls"][0]["parameters"][0][0][0]+=.1
        bad["request_id"]=digest({k:v for k,v in bad.items() if k!="request_id"})
        with self.assertRaises(ValueError):validate_request(bad)
    def test_old_profile_and_chunk_gate(self):
        from upstream_virtual_cases import cases
        from upstream_candidate_helpers import manifest,DS_PROFILE,VR_PROFILE
        from unified_graph_contract import prepare
        from compiler_configuration import PROFILE_SHA256
        g=cases()[0]["model"]
        self.assertNotIn("HE_MPBN0",manifest(g,DS_PROFILE)["helpers"])
        self.assertNotIn("upstream_helpers",prepare(g,PROFILE_SHA256))
        with self.assertRaises(ValueError):prepare(g,PROFILE_SHA256,helper_profile=VR_PROFILE,chunk_period=4)
    def test_zero_argument_type_and_cancellation(self):
        from upstream_virtual_cases import cases
        from unified_graph_contract import validate_candidate
        r=self.request(cases()[0])
        for expr in ("HE_MPBN0(x)","HE_MPBN0(x,1.)","HE_MPBN0(x,zero_ct)-HE_MPBN0(x,zero_ct)+x"):
            src='@hc.func("c,c")'+chr(10)+'def golden(x,zero_ct):'+chr(10)+'    return '+expr+chr(10)
            with self.assertRaises(ValueError):validate_candidate(dict(schema=1,request_id=r["request_id"],hecate_source=src),r)

if __name__=="__main__":unittest.main()
