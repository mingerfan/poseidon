"""Actual upstream helpers on public full-slot vectors; not FHE evidence."""
import unittest,math
import numpy as np
from benchmark_suite import Builder
from benchmark_graph import samples
from benchmark_math import evaluate
from benchmark_torch import evaluate as torch_reference
from upstream_adapters.spatial_node import bind_node,apply
from unified_graph_contract import layout

class Vector:
    def __init__(self,data):self.data=np.asarray(data,dtype=np.float64)
    def other(self,value):
        if isinstance(value,Vector):return value.data
        array=np.asarray(value,dtype=np.float64).reshape(-1)
        return np.tile(array,len(self.data)//len(array))
    def __add__(self,x):return Vector(self.data+self.other(x))
    __radd__=__add__
    def __sub__(self,x):return Vector(self.data-self.other(x))
    def __rsub__(self,x):return Vector(self.other(x)-self.data)
    def __mul__(self,x):return Vector(self.data*self.other(x))
    __rmul__=__mul__
    def rotate(self,k):return Vector(np.roll(self.data,-k))

from upstream_spatial_cases import models

class SpatialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from upstream_adapters.test_batch_norm import BatchNormAdapterTests
        BatchNormAdapterTests.setUpClass.__func__(cls)
    def test_actual_helpers_against_two_independent_references(self):
        for name,g in models():
            p=layout(g)["input_slot_period"]
            blocked=(name in ("conv_1_2_2_3_2","conv_1_4_4_3_2",
                             "avg_2_4_4_3_1","avg_3_2_4_3_1"))
            if blocked:
                with self.assertRaisesRegex(ValueError,"Upstream HE_"):bind_node(g,g["nodes"][0]["id"],p)
                continue
            binding=bind_node(g,g["nodes"][0]["id"],p)
            with self.subTest(name=name):
                for inputs in samples(g,4):
                    expected=evaluate(g,inputs)["output0"].reshape(-1)
                    np.testing.assert_allclose(expected,torch_reference(g,inputs)["output0"].reshape(-1),atol=1e-12,rtol=1e-12)
                    flat=inputs["input0"].reshape(-1)
                    vec=Vector(np.tile(np.pad(flat,(0,p-len(flat))),16384//p))
                    out,record=apply(binding,vec,self.helpers,self.mpcb)
                    np.testing.assert_allclose(out.data[:len(expected)],expected,atol=1e-12,rtol=1e-12)
                    np.testing.assert_array_equal(out.data,np.tile(out.data[:p],16384//p))
                    self.assertLessEqual(record["operation_count"],binding["work"])
                    self.assertTrue(record["unchanged_upstream_functions"])
                    self.assertFalse(record["bootstrap_removed"])
                    from upstream_adapters.spatial_node import verify_record
                    verify_record(binding,record)
                    from upstream_helper_coverage import helper_evaluator
                    from unified_graph_contract import prepare
                    from compiler_configuration import PROFILE_SHA256
                    from packed_native_exercises import Cell
                    request=prepare(g,PROFILE_SHA256,helper_profile="upstream-poly-spatial-v4")
                    actual=helper_evaluator(request)(binding["helper"]+"0",[Cell("c",tuple(vec.data[:p]))]).values
                    np.testing.assert_allclose(actual,out.data[:p],atol=1e-12,rtol=1e-12)


class SpatialContractTests(unittest.TestCase):
    def request(self,row):
        from unified_graph_contract import prepare
        from compiler_configuration import PROFILE_SHA256,configuration
        return prepare(row['model'],PROFILE_SHA256,configuration('seal-cpu-eva-w45-v1'),
            helper_profile='upstream-poly-spatial-v4',helper_exercise=row['required_helpers'])
    def test_all_directed_contexts_and_return_interventions(self):
        from upstream_spatial_cases import directed_cases
        from unified_graph_contract import validate_candidate
        for row in directed_cases():
            with self.subTest(row=row['name']):
                request=self.request(row)
                checked=validate_candidate(dict(schema=1,request_id=request['request_id'],hecate_source=row['source']),request)
                self.assertEqual(set(checked['upstream_exercise']['witnesses']),set(row['required_helpers']))
                self.assertFalse(checked['upstream_exercise']['all_input_proof'])
    def test_cancelled_or_discarded_spatial_helpers_rejected(self):
        from upstream_spatial_cases import directed_cases
        from unified_graph_contract import validate_candidate
        for row in directed_cases()[::3]:
            request=self.request(row);call=row['required_helpers'][0]+'(x)'
            for body in ('    h = '+call+'\n    return x\n','    h = '+call+'\n    return h-h+x\n'):
                with self.subTest(row=row['name'],body=body),self.assertRaisesRegex(ValueError,'contribution'):
                    validate_candidate(dict(schema=1,request_id=request['request_id'],hecate_source='@hc.func("c,c")\ndef golden(x,zero_ct):\n'+body),request)
    def test_rehashed_binding_tampering_rejected(self):
        import copy
        from benchmark_graph import digest
        from upstream_spatial_cases import directed_cases
        from unified_graph_contract import validate_request
        row=directed_cases()[0];r=self.request(row)
        for field,value in (('period',256),('work',100000),('output_positions',[0]),('kind','global_pool')):
            bad=copy.deepcopy(r);bad['upstream_helpers']['helpers']['HE_Conv0']['binding'][field]=value
            bad['request_id']=digest({k:v for k,v in bad.items() if k!='request_id'})
            with self.subTest(field=field),self.assertRaises(ValueError):validate_request(bad)
    def test_original_profiles_and_chunk_boundaries_remain(self):
        from upstream_spatial_cases import directed_cases
        from unified_graph_contract import prepare
        from compiler_configuration import PROFILE_SHA256
        row=directed_cases()[0]
        for profile in ('upstream-poly-silu-v1','upstream-poly-bn-silu-v2','upstream-poly-concat-bn-silu-v3'):
            request=prepare(row['model'],PROFILE_SHA256,helper_profile=profile)
            self.assertNotIn('HE_Conv0',request['upstream_helpers']['helpers'])
        with self.assertRaises(ValueError):
            prepare(row['model'],PROFILE_SHA256,helper_profile='upstream-poly-spatial-v4',chunk_period=4)
    def test_public_ring_rotation_and_constant_gates(self):
        from upstream_adapters.periodic_ring import PeriodicExpr
        for p in (4,8,16,32,64,128,256):
            record=dict(operations=[],rotations=[])
            x=PeriodicExpr(Vector(np.tile(np.arange(p),16384//p)),p,record,1024)
            for k in (-p-1,-1,0,1,p+1):
                np.testing.assert_array_equal(x.rotate(k).value.data,np.roll(x.value.data,-k))
            for bad in ([0.]*(p+1),float('nan'),[1025.]*p):
                with self.assertRaises(ValueError):x+bad
        x=PeriodicExpr(Vector([1.]*16384),4,dict(operations=[],rotations=[]),1)
        x+1
        with self.assertRaisesRegex(ValueError,'budget'):x+1

if __name__=="__main__":unittest.main()
