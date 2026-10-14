"""Static broadcast admission, alias semantics and contract rejection tests."""
import copy, unittest
from benchmark_graph import digest
from pytorch_static_broadcast import plan, arguments, normalize, check, POLICY, SOURCE_FORMAT
PROFILE = "general-tensors-v1"

class StaticBroadcastPlanTests(unittest.TestCase):
    def test_dimensions(self):
        for source,target,result in (([3],[2,3],[2,3]),([1,3],[4,-1],[4,3]),([],[2,3],[2,3]),([],[],[]),([2,1,3],[2,4,3],[2,4,3])):
            with self.subTest(source=source,target=target):
                self.assertEqual(plan(source,target,resource_profile=PROFILE),result)
    def test_invalid_dimensions(self):
        for source,target in (([3],[2]),([3],[-1,3]),([3],[True,3]),([3],[0,3]),([3],[2.0,3]),([3],[3,-2]),([2,3],[3]),([1],[65537]),([1],[2]*33)):
            with self.subTest(source=source,target=target):
                with self.assertRaises(ValueError):plan(source,target,resource_profile=PROFILE)
    def test_signature(self):
        self.assertEqual(arguments("broadcast_to",("tensor",),dict(size=[2,3])),("tensor",[2,3]))
        self.assertEqual(arguments("expand_as",("tensor","other"),{}),("tensor","other"))
    def test_bad_signature(self):
        for method,args,kwargs in (("broadcast_to",("x",),{}),("broadcast_to",("x",[3]),dict(size=[3])),("expand_as",("x","y"),dict(out="z")),("broadcast_to",("x",2,3),{})):
            with self.subTest(method=method):
                with self.assertRaises(ValueError):arguments(method,args,kwargs)
    def test_versioned_admission(self):
        import pytorch_model_intake as facade, pytorch_ordered_contract as ordered
        from pytorch_spatial_policy import supports_input_call,supports_spatial
        self.assertIn(SOURCE_FORMAT,facade.SOURCE_FORMATS)
        self.assertIn(SOURCE_FORMAT,ordered.SOURCE_FORMATS)
        self.assertTrue(supports_input_call(dict(format=SOURCE_FORMAT)))
        self.assertTrue(supports_spatial(dict(format=SOURCE_FORMAT)))
        self.assertEqual(ordered.package_version(dict(format=SOURCE_FORMAT)),"poseidon-trusted-pytorch-provenance-v28")
        self.assertEqual(ordered.reference_version(dict(format=SOURCE_FORMAT)),"poseidon-trusted-pytorch-reference-v28")
        self.assertFalse(supports_spatial(dict(format="poseidon-trusted-pytorch-v26")))

class StaticBroadcastFXTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)
    def normalized(self,module,shapes):
        import torch
        from pytorch_static_trace import trace,ALIAS_POLICY
        inputs=[dict(name=f"x{i}",shape=shape)for i,shape in enumerate(shapes)]
        captured=trace(module,inputs,resource_profile=PROFILE,policy=ALIAS_POLICY,broadcast_source_policy=POLICY)
        result,record=normalize(captured,inputs,resource_profile=PROFILE)
        check(record,inputs,resource_profile=PROFILE)
        return result,record,inputs
    def test_exact_math_and_views(self):
        import torch, numpy as np
        for method in ("expand_as","broadcast_to"):
            for context in ("arithmetic","transpose","scalar"):
                with self.subTest(method=method,context=context):
                    class Model(torch.nn.Module):
                        def forward(self,x,y):
                            source=x.mean() if context=="scalar" else x.transpose(0,1) if context=="transpose" else x
                            z=source.expand_as(other=y) if method=="expand_as" else source.broadcast_to(size=y.shape)
                            return z + y if context=="arithmetic" else z
                    shapes=([3,1],[2,3])if context=="transpose"else([1,3],[2,3])
                    module=Model().double().eval()
                    result,record,_=self.normalized(module,shapes)
                    a=np.array([-.31,.19,.73],dtype=np.float64).reshape(shapes[0])
                    b=np.array([.2,-.4,.7,-.8,.1,.3],dtype=np.float64).reshape(shapes[1])
                    x,y=torch.from_numpy(a),torch.from_numpy(b)
                    expected=np.broadcast_to(a.mean()if context=="scalar"else a.T if context=="transpose"else a,shapes[1])
                    if context=="arithmetic":expected=expected+b
                    actual=result(x,y)
                    np.testing.assert_array_equal(actual.detach().numpy(),expected)
                    np.testing.assert_array_equal(actual.detach().numpy(),module(x,y).detach().numpy())
                    self.assertEqual(record["entries"][0]["source_method"],method)
                    if context!="arithmetic":
                        self.assertEqual(actual.stride(),module(x,y).stride())
                        if context!="scalar":self.assertEqual(actual.untyped_storage().data_ptr(),x.untyped_storage().data_ptr())
    def test_source_dtype_ignores_target_dtype(self):
        import torch
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.register_buffer("mask",torch.tensor([[True,False,True],[False,True,False]]))
            def forward(self,x):return x.expand_as(self.mask)
        module=Model();result,record,_=self.normalized(module,[[1,3]])
        self.assertEqual(result(torch.tensor([[1.,2.,3.]],dtype=torch.float64)).dtype,torch.float64)
        self.assertEqual(record["entries"][0]["dtype"],"float64")
    def test_alias_following_source_write(self):
        import torch
        for method in ("expand_as","broadcast_to"):
            with self.subTest(method=method):
                class Model(torch.nn.Module):
                    def forward(self,x,y):
                        z=x.clone()
                        view=z.expand_as(y)if method=="expand_as"else z.broadcast_to(y.shape)
                        z.add_(.25)
                        return view
                result,_,_=self.normalized(Model(),[[1,3],[2,3]])
                x=torch.tensor([[.1,-.3,.7]],dtype=torch.float64);y=torch.zeros((2,3),dtype=torch.float64)
                torch.testing.assert_close(result(x,y),Model()(x,y),rtol=0,atol=0)
    def test_minus_one_and_list(self):
        import torch
        class Model(torch.nn.Module):
            def forward(self,x):return x.broadcast_to(size=[2,-1])
        result,record,_=self.normalized(Model(),[[1,3]])
        self.assertEqual(record["entries"][0]["result_shape"],[2,3])
        torch.testing.assert_close(result(torch.ones((1,3),dtype=torch.float64)),torch.ones((2,3),dtype=torch.float64))
    def test_old_policy_does_not_admit_methods(self):
        import torch
        from pytorch_static_trace import trace,ALIAS_POLICY
        class Model(torch.nn.Module):
            def forward(self,x):
                z=x.broadcast_to([2,3])
                return z.reshape(z.shape)
        with self.assertRaisesRegex(ValueError,"Static shape unavailable"):
            trace(Model(),[dict(name="x",shape=[1,3])],resource_profile=PROFILE,policy=ALIAS_POLICY)
    def test_rebound_tampering_rejected(self):
        import torch
        class Model(torch.nn.Module):
            def forward(self,x):return x.broadcast_to((2,3))
        _,record,inputs=self.normalized(Model(),[[1,3]])
        for change in ("result","entry","capacity"):
            with self.subTest(change=change):
                altered=copy.deepcopy(record)
                if change=="result":altered["normalized_inventory"][1]["target"]="clone"
                elif change=="entry":altered["entries"][0]["result_shape"]=[3,2]
                else:altered["resource_profile"]="legacy-v1"
                altered["binding"]=digest({k:v for k,v in altered.items()if k!="binding"})
                with self.assertRaises(ValueError):check(altered,inputs,resource_profile=PROFILE)

if __name__=="__main__":unittest.main()
