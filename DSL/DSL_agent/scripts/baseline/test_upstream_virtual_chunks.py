"""Chunked virtual representation: public-vector checks, never FHE evidence."""
import unittest
import numpy as np
from upstream_adapters.virtual_ring import Context,VirtualExpr,NT,proxy_type
from test_upstream_virtual_ring import Vector,expand

def bind(values,p,cls=VirtualExpr):
    values=np.asarray(values,dtype=float);chunks=[]
    for start in range(0,len(values),p):
        tail=np.pad(values[start:start+p],(0,p-len(values[start:start+p])),constant_values=91.)
        # Deliberately dirty unused lanes: the boundary must mask them.
        chunks.append(Vector(np.tile(tail,16384//p)))
    return cls.input_chunks(Context(p),chunks,len(values))

def decode(value,total):
    p=value.context.period
    chunks=[value.export_chunk(start,min(p,total-start),total,Vector(np.zeros(16384)))
            for start in range(0,total,p)]
    for chunk in chunks:
        np.testing.assert_array_equal(chunk.data,np.tile(chunk.data[:p],16384//p))
    return np.concatenate([chunk.data[:min(p,total-start)]
                           for chunk,start in zip(chunks,range(0,total,p))])

class VirtualChunkTests(unittest.TestCase):
    def test_all_periods_cross_chunk_rotations_and_roundtrip(self):
        for p in (4,8,16,32,64,128,256):
            n=min(256,3*p+1);values=np.arange(n,dtype=float)/256
            x=bind(values,p);dense=np.pad(values,(0,NT-n))
            np.testing.assert_array_equal(expand(x),dense)
            for step in (1,-1,p-1,p,p+1,-p-1,NT-1):
                y=x.rotate(step)
                np.testing.assert_array_equal(expand(y),np.roll(dense,-step))
                np.testing.assert_array_equal(expand(y.rotate(-step)),dense)
            np.testing.assert_array_equal(decode(x,n),values)
            np.testing.assert_array_equal(expand(x),dense)
    def test_tail_and_public_only_projection(self):
        x=bind(np.arange(13)/16,4);constant=np.zeros(NT);constant[[4,8,12,NT-1]]=[.1,.2,.3,.4]
        public=x*np.zeros(NT)+constant
        np.testing.assert_array_equal(decode(public,13),constant[:13])
        self.assertIn(NT//4-1,public.values)
        self.assertTrue(public.context.events[-1]["encrypted_zero_argument_used"])
        self.assertEqual(public.context.events[-1]["selected_start"],12)
        self.assertIn(NT//4-1,public.context.events[-1]["discarded_nonzero_blocks"])
    def test_bounds_refuse_partial_or_oversized_io(self):
        v=Vector(np.ones(16384))
        for chunks,count in (([v],5),([v]*5,17),([v]*4,17),([],0),([v],True),([v],257)):
            with self.subTest(count=count,len=len(chunks)),self.assertRaises(ValueError):
                VirtualExpr.input_chunks(Context(4),chunks,count)
        x=bind(np.arange(5)/16,4);zero=Vector(np.zeros(16384))
        for args in ((1,4,5),(4,4,5),(0,3,5),(8,1,5),(0,4,17),(False,4,5)):
            with self.subTest(args=args),self.assertRaises(ValueError):x.export_chunk(*args,zero)
        with self.assertRaisesRegex(ValueError,"operation budget"):
            VirtualExpr.input_chunks(Context(4,limit=1),[v,v],5).rotate(1)
    def test_old_single_prefix_api_is_unchanged(self):
        values=np.arange(4)/16;v=Vector(np.tile(values,4096))
        x=VirtualExpr.input(Context(4),v,3)
        self.assertEqual(x.context.events[0]["operation"],"input")
        out=x.export_prefix(3,Vector(np.zeros(16384)))
        self.assertEqual(x.context.events[-1]["operation"],"output_projection")
        np.testing.assert_array_equal(out.data[:4],np.r_[values[:3],0.])

class ActualChunkHelpers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from upstream_adapters.test_batch_norm import BatchNormAdapterTests
        BatchNormAdapterTests.setUpClass.__func__(cls)
        cls.proxy=proxy_type(cls.helpers.hc.Expr)
    def test_actual_linear_crosses_cipher_boundaries(self):
        import torch
        from types import SimpleNamespace
        from upstream_adapters.periodic_ring import proxy_type as dense_type
        dense=dense_type(self.helpers.hc.Expr)
        for p,n,m in ((4,5,7),(4,13,9),(8,17,13),(16,33,17),(4,3,13)):
            values=np.arange(n)/128-.1;x=bind(values,p,self.proxy)
            w=np.arange(m*n).reshape(m,n)/max(256,m*n)-.25;b=np.arange(m)/64
            width=max(n,m);weight=np.pad(w,((0,0),(0,width-n)))
            params=SimpleNamespace(weight=torch.tensor(weight,dtype=torch.float64),bias=torch.tensor(b,dtype=torch.float64))
            out=self.helpers.HE_Linear(None,np.array([x],dtype=object),params)[0]
            with self.subTest(p=p,n=n,m=m):
                np.testing.assert_allclose(decode(out,m),w@values+b,atol=1e-12,rtol=1e-12)
                # Entire 65536 ring, including non-output tails, must match the
                # original helper on one dense public vector.
                d=dense(Vector(np.pad(values,(0,NT-n))),NT,{"operations":[],"rotations":[]})
                expected=self.helpers.HE_Linear(None,np.array([d],dtype=object),params)[0]
                np.testing.assert_allclose(expand(out),expected.value.data,atol=1e-12,rtol=1e-12)
    def test_actual_mpbn_all_output_chunks(self):
        import torch
        from types import SimpleNamespace
        for p,n in ((4,13),(8,29),(16,61),(64,256)):
            values=np.arange(n)/max(16,n)-.5;x=bind(values,p,self.proxy)
            mean=np.arange(n)/max(16,n);var=np.ones(n)*.7;gamma=np.linspace(-.4,.6,n);beta=np.linspace(.1,-.1,n)
            params=SimpleNamespace(running_mean=torch.tensor(mean),running_var=torch.tensor(var),
                weight=torch.tensor(gamma),bias=torch.tensor(beta),eps=.001)
            out=self.helpers.HE_MPBN(np.array([x],dtype=object),params)[0]
            np.testing.assert_allclose(decode(out,n),(values-mean)*gamma/np.sqrt(var+.001)+beta,atol=1e-12,rtol=1e-12)
    def test_actual_reshape_linear_chunked_nonidentity_layout(self):
        import torch
        from types import SimpleNamespace
        p=4;n=16;m=7;h=w=2;ko=2;to=1
        values=np.arange(n)/32-.25
        perm=[((((t*ko+a)*ko+b)*h+y)*w+z)
              for t in range(to) for y in range(h) for a in range(ko) for z in range(w) for b in range(ko)]
        self.assertNotEqual(perm,list(range(n)))
        x=bind(values[perm],p,self.proxy);weight=np.arange(m*n).reshape(m,n)/256-.1;bias=np.arange(m)/64
        params=SimpleNamespace(weight=torch.tensor(weight),bias=torch.tensor(bias))
        out=self.helpers.HE_ReshapeLinear(None,np.array([x],dtype=object),params,
                                       reshape=dict(ko=ko,ho=h,wo=w,to=to))[0]
        np.testing.assert_allclose(decode(out,m),weight@values+bias,atol=1e-12,rtol=1e-12)

class ChunkNodeBindings(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from upstream_adapters.test_batch_norm import BatchNormAdapterTests
        BatchNormAdapterTests.setUpClass.__func__(cls)
    def test_actual_three_families_and_leading_rows(self):
        import math
        from upstream_virtual_cases import models
        from upstream_adapters import chunk_virtual_node as adapter
        from benchmark_graph import samples
        from benchmark_math import evaluate
        from benchmark_torch import evaluate as reference
        selected=[]
        for family,g in models():
            shape=g["inputs"][0]["shape"]
            if (family,shape) in (("HE_MPBN",[2,3,2]),("HE_Linear",[2,3]),
                                  ("HE_Linear",[15]),("HE_ReshapeLinear",[1,4,2,2])):
                selected.append((family,g))
        self.assertEqual(len(selected),4)
        for family,g in selected:
            node=next(n for n in g["nodes"] if n["op"]==("batch_norm" if family=="HE_MPBN" else "linear"))
            p=4;total=math.prod(__import__("benchmark_graph").validate(g)["shapes"][node["outputs"][0]])
            bindings=[adapter.bind_node(g,node["id"],p,family,i) for i in range((total+p-1)//p)]
            for inputs in samples(g,4):
                values=inputs["input0"].reshape(-1);chunks=[]
                for start in range(0,len(values),p):
                    v=np.pad(values[start:start+p],(0,p-len(values[start:start+p])),constant_values=17.)
                    chunks.append(Vector(np.tile(v,16384//p)))
                outputs=[]
                for b in bindings:
                    out,record=adapter.apply(b,chunks,Vector(np.zeros(16384)),self.helpers)
                    adapter.verify_record(b,record)
                    outputs.extend(out.data[:b["output_chunk"]["elements"]])
                expected=evaluate(g,inputs)["output0"].reshape(-1)
                np.testing.assert_allclose(outputs,expected,atol=1e-12,rtol=1e-12)
                np.testing.assert_allclose(expected,reference(g,inputs)["output0"].reshape(-1),atol=1e-12,rtol=1e-12)
    def test_transcript_and_projection_forgery_rejected(self):
        import copy
        from upstream_virtual_cases import models
        from upstream_adapters import chunk_virtual_node as adapter
        family,g=next((f,g) for f,g in models() if f=="HE_Linear" and g["inputs"][0]["shape"]==[15])
        b=adapter.bind_node(g,g["nodes"][0]["id"],4,family,0);record=adapter.expected_record(b)
        for kind in ("constant","tail","projection","work"):
            bad=copy.deepcopy(record)
            if kind=="constant":bad["virtual"]["full_constants"][0]["sha256"]="0"*64
            if kind=="tail":bad["virtual"]["nonperiodic_constant_truncation"]=True
            if kind=="projection":bad["virtual"]["events"][-1]["selected_start"]=4
            if kind=="work":bad["virtual"]["physical_operations"].append("multiply")
            with self.subTest(kind=kind),self.assertRaises(ValueError):adapter.verify_record(b,bad)
        with self.assertRaises(ValueError):adapter.bind_node(g,g["nodes"][0]["id"],4,family,1)

class ChunkHelperContracts(unittest.TestCase):
    def test_all_sources_and_bound_return_contribution(self):
        from upstream_chunk_cases import cases
        from compiler_configuration import PROFILE_SHA256,configuration
        from unified_graph_contract import prepare,validate_candidate
        for row in cases():
            with self.subTest(name=row["name"]):
                r=prepare(row["model"],PROFILE_SHA256,configuration(row["configuration"]),
                    helper_profile=row["profile"],helper_exercise=row["required_helpers"],chunk_period=row["chunk_period"])
                checked=validate_candidate(dict(schema=1,request_id=r["request_id"],hecate_source=row["source"]),r)
                for name,witness in checked["upstream_exercise"]["witnesses"].items():
                    binding=r["upstream_helpers"]["helpers"][name]["binding"]
                    expected=[o["name"] for o in r["model"]["outputs"] if o["value"]==binding["output_value"]]
                    if expected:self.assertEqual(witness["outputs"],expected)
    def test_explicit_profile_and_immutable_binding(self):
        import copy
        from upstream_chunk_cases import cases
        from upstream_candidate_helpers import CHUNK_PROFILE,VR_PROFILE
        from compiler_configuration import PROFILE_SHA256
        from unified_graph_contract import prepare,validate_request
        from benchmark_graph import digest
        row=cases()[0];g=row["model"]
        for options in (dict(helper_profile=CHUNK_PROFILE),dict(helper_profile=VR_PROFILE,chunk_period=4)):
            with self.assertRaises(ValueError):prepare(g,PROFILE_SHA256,**options)
        r=prepare(g,PROFILE_SHA256,helper_profile=CHUNK_PROFILE,chunk_period=4)
        validate_request(r)
        for field in ("output","period","parameters","work"):
            bad=copy.deepcopy(r);spec=bad["upstream_helpers"]["helpers"]["HE_Linear0_chunk0"];b=spec["binding"]
            if field=="output":b["output_chunk"]["offset"]=4
            if field=="period":b["period"]=8
            if field=="parameters":b["inner"]["calls"][0]["parameters"][0][0][0]+=.1
            if field=="work":spec["work"]-=1
            bad["request_id"]=digest({k:v for k,v in bad.items() if k!="request_id"})
            with self.subTest(field=field),self.assertRaises(ValueError):validate_request(bad)

if __name__=="__main__":unittest.main()
