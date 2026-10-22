"""Registered native MHA production capture, no Provider or encrypted runner."""
import copy, hashlib, unittest
from pathlib import Path
POLICY="poseidon-mha-ordered-source-v1"
BASE=Path(__file__).resolve().parent
def imports():
    import torch, numpy as np
    import pytorch_model_capture, pytorch_static_trace, pytorch_ordered_capture, pytorch_source_contract
    for module in (pytorch_model_capture,pytorch_static_trace,pytorch_ordered_capture,pytorch_source_contract):
        if Path(module.__file__).resolve().parent!=BASE:
            raise RuntimeError("Production module required: "+module.__name__)
    return torch,np,pytorch_ordered_capture,pytorch_source_contract

def configuration():
    return {"format":"poseidon-pytorch-approximations-v1","defaults":{
        "exp":{"method":"taylor","domain":[-.5,.5],"degree":6},
        "reciprocal":{"method":"goldschmidt","domain":[.5,8.],"iterations":8}},
        "modules":[],"nodes":[]}

class OrderedMhaComponent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.torch,cls.np,cls.ordered,cls.outputs=imports()
        from platform_config import require_python_packages
        require_python_packages(cls.torch,cls.np)
        cls.torch.set_num_threads(1)

    def module(self,kind="self",need=True):
        torch=self.torch
        class Source(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attention=torch.nn.MultiheadAttention(2,1,batch_first=True,dtype=torch.float64).eval()
                self.post=torch.nn.Linear(2,2,dtype=torch.float64)
                offset=0
                with torch.no_grad():
                    for value in self.parameters():
                        raw=torch.arange(offset+1,offset+1+value.numel(),dtype=torch.float64).sin()*.04
                        value.copy_(raw.reshape(value.shape));offset+=value.numel()
                self.register_buffer("mask",torch.tensor([[False,True],[False,False]],dtype=torch.bool))
            def forward(self,q):
                value,weights=self.attention(q,q,q,need_weights=need,
                    attn_mask=self.mask if kind=="masked" else None)
                value=self.post(value)+q
                if kind=="scalar":return {"value":value,"mean":value.mean(),"none":weights}
                if kind=="repeat":
                    second,w2=self.attention(value,value,value,need_weights=need)
                    return {"first":value,"second":second,"w1":weights,"w2":w2}
                return {"value":value,"weights":weights}
        return Source().eval()

    def prepare(self,module,outputs,config=None,policy=POLICY):
        from pytorch_matrix_plan import POLICY as MATRIX
        from pytorch_fused_matrix_normalization import POLICY as FUSED
        from pytorch_approximation import POLICY as APPROX
        from pytorch_model_capture import ADAPTIVE_FUNCTION_POLICY
        return self.ordered.prepare_module(module,[{"name":"q","shape":[1,2,2]}],outputs,
            approximations=configuration() if config is None else config,
            resource_profile="general-tensors-v1",dtype_policy="strict-float64-v1",
            matrix_policy=MATRIX,approximation_policy=APPROX,
            functional_policy=ADAPTIVE_FUNCTION_POLICY,common_calls=True,
            fused_matrix_policy=FUSED,gelu_source_policy="poseidon-gelu-source-variants-v1",
            mha_source_policy=policy)

    def assert_closed(self,prepared):
        high,lowered,binding,context=prepared
        known={x["name"] for x in high["inputs"]}|set(high["constants"])
        ids=set()
        for node in high["nodes"]:
            self.assertNotIn(node["id"],ids);ids.add(node["id"])
            self.assertTrue(set(node["inputs"])<=known)
            self.assertFalse(set(node["outputs"])&known)
            known.update(node["outputs"])
        self.assertTrue(all(x["value"] in known for x in high["outputs"]))
        self.assertEqual(context["original_module"].training,False)
        self.assertIsNot(context["original_module"],context["normalized_module"])
        self.assertEqual(binding["pytorch_capture"]["graph_sha256"],__import__("benchmark_graph").digest(high))
        return high,lowered,binding,context

    def test_composed_mha_preserves_linear_and_residual_dataflow(self):
        high,_,binding,_=self.assert_closed(self.prepare(self.module(),["value","weights"]))
        self.assertGreaterEqual(sum(n["op"]=="linear" for n in high["nodes"]),5)
        self.assertTrue(any(n["op"]=="add" for n in high["nodes"]))
        self.assertIn("ordered_alias_capture",binding["pytorch_capture"])
        self.assertIn("mha_source_capture",binding["pytorch_capture"])

    def test_two_calls_same_module_have_disjoint_emitted_values(self):
        high,_,binding,_=self.assert_closed(self.prepare(self.module("repeat"),["first","second","w1","w2"]))
        self.assertGreaterEqual(sum(n["op"]=="linear" for n in high["nodes"]),9)
        self.assertEqual(len(high["outputs"]),4)
        selections=binding["pytorch_capture"]["approximation_selection"]["selections"]
        originals={s["node_id"] for s in selections}
        self.assertEqual(len(originals),2)
        self.assertEqual({s["function"] for s in selections},{"exp","reciprocal"})

    def test_need_weights_false_keeps_none_without_extra_named_output(self):
        high,_,binding,_=self.assert_closed(self.prepare(self.module(need=False),["value"]))
        self.assertEqual([o["name"] for o in high["outputs"]],["value"])
        tree=binding["pytorch_capture"]["output_structure"]
        restored=self.outputs.restore_outputs({"value":17},tree,allow_none=True)
        self.assertEqual(restored,{"value":17,"weights":None})

    def test_scalar_and_none_retain_rank_zero_output_abi(self):
        high,_,binding,_=self.assert_closed(self.prepare(self.module("scalar",need=False),["value","mean"]))
        tree=binding["pytorch_capture"]["output_structure"]
        restored=self.outputs.restore_outputs({"value":17,"mean":self.np.asarray([.125])},tree,allow_none=True)
        self.assertEqual(restored,{"value":17,"mean":.125,"none":None})
        self.assertEqual(len(high["outputs"]),2)

    def test_bool_mask_is_state_bound_without_global_boolean_arithmetic(self):
        module=self.module("masked")
        before={n:t.detach().clone() for n,t in module.state_dict().items()}
        _,_,binding,_=self.assert_closed(self.prepare(module,["value","weights"]))
        self.assertEqual(binding["pytorch_capture"]["source_state_dtypes"]["mask"],"bool")
        self.assertEqual(set(before),set(module.state_dict()))
        self.assertTrue(all(self.torch.equal(t,module.state_dict()[n]) for n,t in before.items()))

    def test_all_masked_public_row_is_rejected(self):
        module=self.module("masked")
        module.mask[0].fill_(True)
        with self.assertRaises(ValueError) as caught:
            self.prepare(module,["value","weights"])
        self.assertEqual(getattr(caught.exception,"code",None),"mha_all_masked_unsupported")
        feedback=caught.exception.feedback()
        self.assertEqual(feedback["layer"],"input")
        self.assertEqual(feedback["location"]["parameter"],"mask")

    def test_explicit_source_policy_required(self):
        with self.assertRaises((ValueError,RuntimeError)):
            self.prepare(self.module(),["value","weights"],policy=None)

    def test_module_and_node_selection_precedence_uses_original_mha_locations(self):
        config=configuration()
        override=copy.deepcopy(config["defaults"]["exp"]);override["degree"]=4
        config["modules"]=[{"path":"attention","function":"exp","profile":override}]
        config["nodes"]=[{"name":"attention","function":"exp","profile":config["defaults"]["exp"]}]
        _,_,binding,_=self.assert_closed(self.prepare(self.module(),["value","weights"],config))
        selections=binding["pytorch_capture"]["approximation_selection"]["selections"]
        selected=[s for s in selections if s["function"]=="exp"]
        self.assertTrue(selected)
        self.assertTrue(all(s["node_id"]=="fx_attention" and s["module_paths"]==["attention"] and
                            s["source"]=={"kind":"nodes","selector":0} for s in selected))

    def test_unmatched_original_node_selector_rejected(self):
        config=configuration()
        config["nodes"]=[{"name":"not_a_source_node","function":"exp","profile":config["defaults"]["exp"]}]
        with self.assertRaisesRegex(ValueError,"Unmatched approximation"):
            self.prepare(self.module(),["value","weights"],config)

    def test_missing_approximation_is_not_implicitly_filled(self):
        config=configuration();del config["defaults"]["reciprocal"]
        with self.assertRaisesRegex(ValueError,"Explicit approximation|required|approximation"):
            self.prepare(self.module(),["value","weights"],config)

    def test_source_caller_and_rng_are_unchanged(self):
        module=self.module();before={n:t.detach().clone() for n,t in module.state_dict().items()}
        rng=self.torch.get_rng_state().clone()
        self.assert_closed(self.prepare(module,["value","weights"]))
        self.assertTrue(self.torch.equal(rng,self.torch.get_rng_state()))
        self.assertTrue(all(self.torch.equal(t,module.state_dict()[n]) for n,t in before.items()))


if __name__=="__main__":unittest.main(verbosity=2)
