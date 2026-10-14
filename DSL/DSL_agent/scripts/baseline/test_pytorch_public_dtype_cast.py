"""Explicit public casts, immutable provenance, and legacy contract compatibility."""
import copy,hashlib,importlib.util,json,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from benchmark_graph import digest
from pytorch_public_value_ops import POLICY,CAST_POLICY,DATA_TYPES,plan,check_plan,evaluate
from pytorch_public_expression_flow import normalize,check as check_flow,_arrays,state_identity
from pytorch_public_expression_capture import prepare,wrapped_metadata,derived_arrays,check_record
from pytorch_public_arithmetic_capture import TRACE_OPTIONS
from pytorch_static_trace import trace
from pytorch_weight_package import export_state,binding
from pytorch_ordered_contract import package_version,reference_version,allows_none_output
from pytorch_model_intake import ordered

ROOT=Path(__file__).resolve().parent
V24="poseidon-trusted-pytorch-v24"
DTYPES=(("<f4","f32"),("<f8","f64"),("<i4","i32"),("<i8","i64"),("|b1","bool"))

def payload(dtype):
    data=np.array([[.125000017,-.375000019,.625000023],[.875000029,-1.125000031,1.375000037]],dtype="<f8")
    if dtype in("<i4","<i8"):data=np.array([[1,-2,3],[4,-5,6]])
    if dtype=="|b1":data=np.array([[True,False,True],[False,True,False]])
    return np.asarray(data,dtype=dtype)

def converted(value,target):
    tensor=torch.from_numpy(np.array(value,copy=True))
    return tensor.to(dtype=torch.float32 if target=="<f4"else torch.float64).numpy().copy()

class PublicModule(torch.nn.Module):
    def __init__(self,dtype,target,context=0,roundtrip=False):
        super().__init__();self.register_buffer("public",torch.from_numpy(payload(dtype).copy()))
        self.target=target;self.context=context;self.roundtrip=roundtrip
    def forward(self,x):
        w=self.public
        if self.target=="<f4":w=w.float()
        else:w=w.to(dtype=torch.float64)
        if self.roundtrip:w=w.double()
        if self.context==1:return x+w.transpose(0,1)
        y=x*w
        if self.context==2:return {"primary":y+x,"saved":[x,None]}
        return y

def specification(module,format=V24,folder=None):
    if folder is None:raise ValueError("public fixture folder")
    weights=export_state(module,folder,resource_profile="general-tensors-v1")
    shape=[3,2]if module.context==1 else[2,3]
    source="def build():\n    return None\n"
    return dict(format=format,files={"model.py":hashlib.sha256(source.encode()).hexdigest()},
        entry="model.build",inputs=[dict(name="x",shape=shape)],
        outputs=["y","saved"]if module.context==2 else["y"],
        factory=dict(args=[],kwargs={}),weights=weights,
        dtype_policy="convert-float64-v1"if module.public.dtype is torch.float32 else"strict-float64-v1",
        approximations=dict(format="poseidon-pytorch-approximations-v1",nodes=[],modules=[],defaults={}))

def captured(dtype="<f8",target="<f4",context=0,roundtrip=False):
    module=PublicModule(dtype,target,context,roundtrip)
    with tempfile.TemporaryDirectory()as path:
        folder=Path(path);weights=folder/"weights";weights.mkdir();out=folder/"capture";out.mkdir()
        spec=specification(module,folder=weights)
        arrays=_arrays(module)
        record=prepare(module,spec,out,expected_weight_binding=binding(arrays,spec["weights"]))
        checked=check_record(record,spec,arrays,expected_record_binding=record["binding"])
        assert checked==record
        return module,spec,arrays,record

class PublicCastMathTests(unittest.TestCase):
    def test_all_registered_dtype_pairs_match_torch_in_three_shapes(self):
        for dtype,_ in DTYPES:
            for target in("<f4","<f8"):
                for shape in((),(2,3),(1,2,3)):
                    with self.subTest(source=dtype,target=target,shape=shape):
                        value=payload(dtype).reshape(shape)if shape else np.array(payload(dtype).reshape(-1)[0],dtype=dtype)
                        original=value.copy();kind="float"if target=="<f4"else"double"
                        logical,physical,record=plan(value,"call_method",kind,("public",),{},policy=CAST_POLICY)
                        expected=converted(value,target)
                        np.testing.assert_array_equal(logical,expected)
                        self.assertEqual(logical.dtype.str,target)
                        np.testing.assert_array_equal(physical,expected.astype("<f8"))
                        np.testing.assert_array_equal(value,original)
                        self.assertFalse(logical.flags.writeable);self.assertFalse(physical.flags.writeable)
                        self.assertEqual(record["input"]["dtype"],dtype)
    def test_to_positional_and_keyword_dtype(self):
        value=payload("<i8")
        for args,kwargs in[(("p","torch.float64"),{}),(("p",),dict(dtype="torch.float64"))]:
            logical,_,record=plan(value,"call_method","to",args,kwargs,policy=CAST_POLICY)
            np.testing.assert_array_equal(logical,converted(value,"<f8"))
            check_plan(record,value,"call_method","to",args,kwargs,expected_binding=record["binding"],policy=CAST_POLICY)
    def test_explicit_rounding_is_bound(self):
        value=np.array([16777217.,-16777217.,.1],dtype="<f8")
        logical,physical,record=plan(value,"call_method","float",("p",),{},policy=CAST_POLICY)
        self.assertNotEqual(float(physical[0]),float(value[0]))
        np.testing.assert_array_equal(logical,converted(value,"<f4"))
        self.assertNotEqual(record["input"]["sha256"],record["materialized_output"]["sha256"])
    def test_integer_rounding_matches_torch(self):
        for value in[(2**53+1),-(2**53+1),2**63-1,-(2**63)]:
            source=np.array(value,dtype="<i8")
            for target in("<f4","<f8"):
                result,_,_=plan(source,"call_method","float"if target=="<f4"else"double",("p",),{},policy=CAST_POLICY)
                np.testing.assert_array_equal(result,converted(source,target))
    def test_overflow_rejects_before_materialization(self):
        value=np.array([1e300],dtype="<f8");before=value.copy()
        with self.assertRaisesRegex(ValueError,"public_expression_nonfinite"):
            plan(value,"call_method","float",("p",),{},policy=CAST_POLICY)
        np.testing.assert_array_equal(value,before)
    def test_device_copy_integer_and_dynamic_cast_arguments_reject(self):
        value=payload("<f8")
        for args,kwargs in[(("p","cuda"),{}),(("p","torch.int64"),{}),
                            (("p",),dict(dtype="torch.float32",device="cpu")),
                            (("p",),dict(dtype="torch.float32",copy=True)),(("p",object()),{})]:
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):
                plan(value,"call_method","to",args,kwargs,policy=CAST_POLICY)
    def test_unknown_policy_and_dtype_operation_reject(self):
        for policy in(None,True,"poseidon-public-value-expressions-v999"):
            with self.assertRaises(ValueError):
                plan(payload("<f8"),"call_method","float",("p",),{},policy=policy)
        with self.assertRaises(ValueError):
            evaluate(payload("<f8"),"dtype_cast",dict(dtype="<f4"),policy=POLICY)
    def test_plan_and_weight_tampering_reject(self):
        value=payload("<f8")
        _,_,record=plan(value,"call_method","float",("p",),{},policy=CAST_POLICY)
        mutated=value.copy();mutated[0,0]+=1e-12
        for original,proof in[(mutated,record),(value,{**record,"format":POLICY})]:
            with self.assertRaises(ValueError):
                check_plan(proof,original,"call_method","float",("p",),{},expected_binding=record["binding"],policy=CAST_POLICY)
    def test_equal_rounded_values_keep_different_source_bindings(self):
        value=payload("<f8");other=value.copy();other[0,0]+=1e-12
        a,_,first=plan(value,"call_method","float",("p",),{},policy=CAST_POLICY)
        b,_,second=plan(other,"call_method","float",("p",),{},policy=CAST_POLICY)
        np.testing.assert_array_equal(a,b);self.assertNotEqual(first["binding"],second["binding"])
    def test_old_bool_plan_is_byte_equivalent_to_baseline(self):
        path=ROOT/"benchmarks/tools/public_dtype_cast_v1/before/pytorch_public_value_ops.py"
        loader=importlib.util.spec_from_file_location("_trusted_public_cast_baseline",path)
        module=importlib.util.module_from_spec(loader);loader.loader.exec_module(module)
        for target in("float","double"):
            a,b,expected=module.plan(payload("|b1"),"call_method",target,("p",),{})
            x,y,actual=plan(payload("|b1"),"call_method",target,("p",),{})
            np.testing.assert_array_equal(a,x);np.testing.assert_array_equal(b,y)
            self.assertEqual(actual,expected)
    def test_old_contract_keeps_float_and_integer_cast_rejection(self):
        for dtype in("<f4","<f8","<i4","<i8"):
            with self.assertRaisesRegex(ValueError,"public_expression_bool_cast"):
                plan(payload(dtype),"call_method","double",("p",),{})
    def test_unsupported_array_and_nonfinite_input_reject(self):
        for value in[np.ones(2,dtype=np.float16),np.array([complex(1,2)]),np.array([np.nan]),np.array([np.inf])]:
            with self.assertRaises(ValueError):
                plan(value,"call_method","double",("p",),{},policy=CAST_POLICY)

class PublicCastCaptureTests(unittest.TestCase):
    def test_roundtrip_capture_and_data_only_reload(self):
        module,spec,arrays,record=captured(roundtrip=True)
        self.assertEqual(record["source_intake_version"],V24)
        entries=record["public_expression_flow"]["entries"]
        self.assertEqual([e["plan"]["operation"]for e in entries],["dtype_cast","dtype_cast"])
        self.assertEqual(entries[0]["plan"]["logical_output"]["dtype"],"<f4")
        self.assertEqual(entries[1]["plan"]["logical_output"]["dtype"],"<f8")
        first=derived_arrays(record,spec,arrays,resource_profile="general-tensors-v1")
        second=derived_arrays(record,spec,arrays,resource_profile="general-tensors-v1")
        self.assertEqual(set(first),set(second))
        for n in first:np.testing.assert_array_equal(first[n],second[n])
        np.testing.assert_array_equal(_arrays(module)["public"],arrays["public"])
    def test_integer_and_bool_capture_match_independent_reference(self):
        for dtype,target in[("<i4","<f4"),("<i8","<f8"),("|b1","<f8")]:
            with self.subTest(dtype=dtype):
                _,_,_,record=captured(dtype,target,1)
                self.assertEqual(record["original_reference"]["status"],"passed")
                self.assertEqual(record["original_reference"]["groups"],16)
    def test_named_none_output_is_preserved(self):
        _,spec,_,record=captured(context=2)
        self.assertTrue(allows_none_output(spec))
        metadata=wrapped_metadata(record)
        self.assertEqual(metadata["output_structure"]["kind"],"dict")
        self.assertEqual(metadata["output_structure"]["items"][1]["value"]["items"][1],{"kind":"none"})
    def test_source_policy_and_original_state_tampering_reject(self):
        _,spec,arrays,record=captured()
        for changed_spec,changed_arrays in[({**spec,"format":"poseidon-trusted-pytorch-v23"},arrays),
                                          (spec,{"public":arrays["public"]+1})]:
            with self.assertRaises(ValueError):
                check_record(record,changed_spec,changed_arrays,expected_record_binding=record["binding"])

class PublicCastVersionTests(unittest.TestCase):
    def test_v24_has_separate_source_package_and_reference(self):
        spec=dict(format=V24);self.assertTrue(ordered(spec))
        self.assertEqual(package_version(spec),"poseidon-trusted-pytorch-provenance-v24")
        self.assertEqual(reference_version(spec),"poseidon-trusted-pytorch-reference-v24")
    def test_legacy_package_versions_stay_separate(self):
        for version in range(12,24):
            spec=dict(format=f"poseidon-trusted-pytorch-v{version}")
            self.assertEqual(package_version(spec),f"poseidon-trusted-pytorch-provenance-v{version}")
            self.assertEqual(reference_version(spec),f"poseidon-trusted-pytorch-reference-v{version}")
    def test_future_contract_is_not_implicitly_enabled(self):
        with self.assertRaises(ValueError):package_version(dict(format="poseidon-trusted-pytorch-v1000"))
        self.assertFalse(allows_none_output(dict(format="poseidon-trusted-pytorch-v1000")))
if __name__=="__main__":unittest.main()
