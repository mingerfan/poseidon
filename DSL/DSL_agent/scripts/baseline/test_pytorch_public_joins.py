"""Public and mixed tensor assembly contracts and immutable provenance."""
import copy,json,tempfile,unittest,hashlib
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from benchmark_graph import digest
from pytorch_public_join_source import SOURCE_FORMAT,VALUE_POLICY,POLICY
from pytorch_public_join_values import arguments,result_shape,public_plan
from pytorch_public_join_normalization import normalized_inventory,check,encode,implementation
from pytorch_public_expression_flow import Ref,state_identity,expected_flow
PROFILE="general-tensors-v1"
def typed(name,op,target,args=(),kwargs=None,shape=None,dtype="float64"):
    return dict(name=name,op=op,target=target,args=encode(args),kwargs=encode(kwargs or {}),
                tensor=None if shape is None else dict(policy="signature-static-ordered-alias-staging-v1",shape=shape,dtype=dtype))
def inventory(dtype="bool",first_plain=False):
    x=Ref("x");w=Ref("w");values=(w,x)if first_plain else(x,w)
    return [typed("x","placeholder","x",shape=[2,2]),
      typed("w","get_attr","w",shape=[2,1],dtype=dtype),
      typed("joined","call_function","torch.cat",(values,),dict(dim=1),[2,3]),
      typed("out","output","output",(Ref("joined"),))]
class PublicValues(unittest.TestCase):
    def join(self,arrays,op="cat",axis=0,remaining=100):
        names={str(i):a for i,a in enumerate(arrays)}
        logical,physical,proof=public_plan("call_function","torch."+op,(tuple(names),),dict(dim=axis),
           resolve_value=lambda n:(n,names[n])if n in names else None,resource_profile=PROFILE,remaining=remaining)
        return logical,physical,proof
    def test_torch_promotion_and_float32_rounding(self):
        for left,right in [("bool","int32"),("int32","int64"),("int64","float32"),("float32","float64"),("bool","float64")]:
            a=np.array([True,False])if left=="bool"else np.array([16777217,-3],dtype=left)
            b=np.array([False,True])if right=="bool"else np.array([5,-7],dtype=right)
            for op in("cat","stack"):
                with self.subTest(left=left,right=right,op=op):
                    got,physical,proof=self.join([a,b],op)
                    expected=getattr(torch,op)((torch.from_numpy(a),torch.from_numpy(b)),dim=0).numpy()
                    self.assertEqual(got.dtype,expected.dtype);np.testing.assert_array_equal(got,expected)
                    np.testing.assert_array_equal(physical,expected.astype(np.float64)if expected.dtype==np.float32 else expected)
    def test_aliases_and_negative_axis(self):
        a=np.arange(4,dtype=np.float64).reshape(2,2)
        for op in("cat","concat","concatenate","stack"):
            logical,_,proof=self.join([a,a*.5],op,-1)
            self.assertEqual(proof["operation"],"stack"if op=="stack"else"concat")
            self.assertTrue(np.isfinite(logical).all())
    def test_signature_rejections(self):
        for args,kwargs in [((("x",),1),dict(dim=0)),((("x",),),dict(dim=True)),
          (((),),{}),((tuple(str(i)for i in range(9)),),{}),((("x",),),dict(out="x")),
          ((("x",),),dict(dtype="float64"))]:
            with self.subTest(args=args,kwargs=kwargs),self.assertRaises(ValueError):
                arguments("call_function","torch.cat",args,kwargs)
    def test_shape_rejections(self):
        for op,ss,axis in [("cat",[[],[]],0),("cat",[[2,2],[3,1]],1),
          ("stack",[[2],[3]],0),("stack",[[2],[2]],2),("cat",[[2],[2]],True)]:
            with self.subTest(op=op,ss=ss),self.assertRaises(ValueError):
                result_shape(op,ss,axis,resource_profile=PROFILE)
    def test_scalar_stack(self):
        result,_,proof=self.join([np.asarray(.25),np.asarray(-.125)],"stack",-1)
        np.testing.assert_array_equal(result,[.25,-.125]);self.assertEqual(proof["axis"],0)
    def test_budget_checked_before_allocation(self):
        with patch("pytorch_public_join_values.np.concatenate",side_effect=AssertionError("allocated"))as allocate:
            with self.assertRaisesRegex(ValueError,"public_join_total_constant_capacity"):
                self.join([np.ones(4),np.ones(4)],remaining=7)
            allocate.assert_not_called()
    def test_unbound_and_nonfinite_rejected(self):
        for a in (None,np.array([np.inf])):
            with self.subTest(a=a),self.assertRaises(ValueError):
                public_plan("call_function","torch.cat",(("x",),),{},resolve_value=lambda _:
                    None if a is None else("x",a),resource_profile=PROFILE,remaining=100)
    def test_unknown_policy_and_future_source_rejected(self):
        from pytorch_public_value_ops import checked_policy
        from pytorch_ordered_contract import package_version
        with self.assertRaises(ValueError):checked_policy("poseidon-public-value-expressions-v999")
        with self.assertRaises(ValueError):package_version(dict(format="poseidon-trusted-pytorch-v999"))
class MixedProvenance(unittest.TestCase):
    def normalize(self,source=None,original=None):
        return normalized_inventory(source or inventory(),original or dict(w=np.array([[True],[False]])),resource_profile=PROFILE)
    def record(self):
        source=inventory();original=dict(w=np.array([[True],[False]]))
        normalized,entries,full,effective,arrays,dropped=self.normalize(source,original)
        r=dict(format=POLICY,input_signature=[dict(name="x",shape=[2,2])],resource_profile=PROFILE,
          source_inventory=source,normalized_inventory=normalized,entries=entries,
          original_state=state_identity(original,resource_profile=PROFILE),
          full_normalized_state=state_identity(full,resource_profile=PROFILE),
          effective_state=state_identity(effective,resource_profile=PROFILE),
          dropped_get_attr_nodes=dropped,implementation=implementation(),constant_materialization="zero-from-encrypted-anchor-v1")
        r["binding"]=digest(r);return r,original
    def test_public_first_and_immutable_source(self):
        source=inventory(first_plain=True);before=copy.deepcopy(source)
        normalized,entries,full,effective,arrays,dropped=self.normalize(source)
        self.assertEqual(source,before);self.assertEqual(entries[0]["encrypted_anchor"],"x")
        self.assertEqual(entries[0]["source_operands"],["w","x"]);self.assertEqual(entries[0]["extra_cipher_operations"],3)
        self.assertEqual(dropped,["w"]);self.assertEqual(next(iter(arrays.values())).dtype,np.float64)
    def test_zero_before_sum_avoids_finite_input_overflow(self):
        rows,entries,*_=self.normalize()
        ops=[(r["op"],r["target"])for r in rows]
        self.assertLess(ops.index(("call_function","torch.mul")),ops.index(("call_function","torch.sum")))
        x=np.array([[1e308,1e308],[-1e308,-1e308]])
        self.assertEqual(np.sum(x*0.),0.)
    def test_generated_name_collision(self):
        source=inventory();source.insert(0,typed("joined_join_zero","placeholder","z",shape=[1]))
        with self.assertRaisesRegex(ValueError,"generated_name_collision"):self.normalize(source)
    def test_integer_precision(self):
        source=inventory(dtype="int64")
        with self.assertRaisesRegex(ValueError,"integer_precision"):
            self.normalize(source,dict(w=np.array([[2**53+1],[0]],dtype=np.int64)))
    def test_encrypted_dtype_rejected(self):
        source=inventory();source[0]["tensor"]["dtype"]="bool"
        with self.assertRaisesRegex(ValueError,"encrypted_dtype"):self.normalize(source)
    def test_dynamic_axis_and_out_rejected(self):
        for kwargs in (dict(dim=Ref("x")),dict(dim=1,out=Ref("x"))):
            source=inventory();source[2]["kwargs"]=encode(kwargs)
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):self.normalize(source)
    def test_forged_output_shape_rejected(self):
        source=inventory();source[2]["tensor"]["shape"]=[2,4]
        with self.assertRaisesRegex(ValueError,"result_metadata"):self.normalize(source)
    def test_round_trip_and_changed_public_bytes_rejected(self):
        record,original=self.record()
        check(record,record["input_signature"],original,resource_profile=PROFILE)
        altered=dict(w=np.logical_not(original["w"]))
        with self.assertRaisesRegex(ValueError,"original_state_binding"):
            check(record,record["input_signature"],altered,resource_profile=PROFILE)
    def test_rehashed_forged_plan_rejected(self):
        record,original=self.record()
        for mutation in("axis","operand","zero"):
            changed=copy.deepcopy(record)
            if mutation=="axis":changed["entries"][0]["axis"]=0
            elif mutation=="operand":changed["entries"][0]["source_operands"].reverse()
            else:next(row for row in changed["normalized_inventory"]if row["target"]=="torch.mul")["args"]["items"][1]["value"]=1.
            changed.pop("binding");changed["binding"]=digest(changed)
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                check(changed,changed["input_signature"],original,resource_profile=PROFILE)
    def test_legacy_public_flow_does_not_enable_new_operations(self):
        source=[typed("a","get_attr","a",shape=[2]),typed("b","get_attr","b",shape=[2]),
          typed("j","call_function","torch.cat",((Ref("a"),Ref("b")),),{},[4]),
          typed("out","output","output",(Ref("j"),))]
        original=dict(a=np.array([.25,-.125]),b=np.array([-.5,.125]))
        with self.assertRaisesRegex(ValueError,"not_registered"):
            expected_flow(source,original,resource_profile=PROFILE,policy="poseidon-public-value-expressions-v6")
        full,effective,record=expected_flow(source,original,resource_profile=PROFILE,policy=VALUE_POLICY)
        self.assertEqual(record["format"],"poseidon-public-expression-flow-v7")
        self.assertEqual(len(record["entries"]),1)
class SourceContracts(unittest.TestCase):
    def test_versions_inherit_registered_capabilities(self):
        from pytorch_model_intake import ordered,PACKAGES,REFERENCES
        from pytorch_source_package_policy import selected
        from pytorch_spatial_policy import supports_input_call,supports_spatial
        from pytorch_static_tensor_transforms import selected as transforms
        from pytorch_static_broadcast import supports_source
        spec=dict(format=SOURCE_FORMAT)
        self.assertTrue(all(f(spec)for f in(ordered,selected,supports_input_call,supports_spatial,transforms,supports_source)))
        self.assertEqual(PACKAGES[SOURCE_FORMAT],"poseidon-trusted-pytorch-provenance-v34")
        self.assertEqual(REFERENCES[SOURCE_FORMAT],"poseidon-trusted-pytorch-reference-v34")
    def test_source_closure_mounts_and_binds_new_modules(self):
        from pytorch_ordered_contract import implementation_paths,source_identity
        paths=implementation_paths();identity=source_identity()
        for name in("pytorch_public_join_source.py","pytorch_public_join_values.py","pytorch_public_join_normalization.py"):
            self.assertIn(name,paths);self.assertEqual(hashlib.sha256(paths[name].read_bytes()).hexdigest(),identity[name])
if __name__=="__main__":unittest.main()
