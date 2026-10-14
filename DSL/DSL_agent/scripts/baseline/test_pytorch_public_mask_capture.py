"""Public mask intake replay, old contracts and registered alias effects."""
import copy,json,tempfile,hashlib
from pathlib import Path
import unittest
import numpy as np
import torch
from benchmark_graph import digest
from pytorch_public_expression_flow import _arrays
from pytorch_public_expression_capture import prepare,check_record,derived_arrays
from pytorch_weight_package import binding,export_state
from pytorch_public_mask_selection import check as check_masks
from pytorch_public_mask_source import SOURCE_FORMAT

TOOLS=Path(__file__).with_name("benchmarks")/"tools/public_mask_intake_v1"
class MaskCapture(unittest.TestCase):
    def model(self,row):
        manifest=Path(row["manifest"]);spec=json.loads(manifest.read_text());folder=manifest.parent
        env={}
        code=(folder/"model.py").read_text()
        if(folder/"helpers.py").exists():
            exec(compile((folder/"helpers.py").read_text(),"owned_mask_fixture","exec"),env)
            code=code.replace("from helpers import masked_selection\n","")
        exec(compile(code,"owned_mask_fixture","exec"),env)
        return env["make"](),spec
    def capture(self,module,spec):
        with tempfile.TemporaryDirectory()as tmp:
            record=prepare(module,spec,Path(tmp),expected_weight_binding=binding(_arrays(module),spec["weights"]))
        check_record(record,spec,_arrays(module),expected_record_binding=record["binding"])
        return record
    def test_all_eighteen_declared_source_contexts(self):
        rows=json.loads((TOOLS/"cases.json").read_text())[:18]
        for row in rows:
            with self.subTest(case=row["name"]):
                module,spec=self.model(row);before=_arrays(module)
                record=self.capture(module,spec)
                self.assertEqual(record["provider_calls"],0);self.assertEqual(record["compiler_calls"],0);self.assertEqual(record["encrypted_calls"],0)
                for key,value in before.items():np.testing.assert_array_equal(_arrays(module)[key],value)
                flow=record["public_mask_selection_flow"]
                self.assertFalse(flow["encrypted_conditions"])
                if not row["operator"].startswith("public_"):
                    self.assertEqual(len(flow["entries"]),1)
                    self.assertEqual(flow["entries"][0]["inplace"],row["operator"]=="masked_fill_")
                else:
                    self.assertEqual(len(flow["entries"]),0)
                    plans=[v["plan"]for v in record["public_expression_flow"]["entries"]]
                    self.assertTrue(any(p["format"]=="poseidon-public-mask-value-v1"for p in plans))
                arrays=derived_arrays(record,spec,before,resource_profile="general-tensors-v1")
                self.assertTrue(any(v.dtype.str=="<f8"for v in arrays.values()))
    def test_v32_stays_closed(self):
        for row in json.loads((TOOLS/"cases.json").read_text())[:18]:
            with self.subTest(case=row["name"]):
                module,spec=self.model(row);spec["format"]="poseidon-trusted-pytorch-v32"
                with self.assertRaises(ValueError):self.capture(module,spec)
    def test_replay_rejects_changed_mask(self):
        row=json.loads((TOOLS/"cases.json").read_text())[0];module,spec=self.model(row);record=self.capture(module,spec)
        arrays=_arrays(module);arrays["mask"][0]=not arrays["mask"][0]
        with self.assertRaises(ValueError):check_record(record,spec,arrays,expected_record_binding=record["binding"])
    def test_generated_plan_tampering_and_extra_fields(self):
        row=json.loads((TOOLS/"cases.json").read_text())[0];module,spec=self.model(row);record=self.capture(module,spec)
        for mutate in("condition","output","extra"):
            bad=copy.deepcopy(record)
            masks=bad["public_mask_selection_flow"]
            if mutate=="condition":masks["entries"][0]["condition"]="x"
            elif mutate=="output":masks["normalized_inventory"][-2]["target"]="torch.sub"
            else:masks["extra"]=True
            masks["binding"]=digest({k:v for k,v in masks.items()if k!="binding"})
            bad["binding"]=digest({k:v for k,v in bad.items()if k!="binding"})
            with self.subTest(mutate=mutate),self.assertRaises(ValueError):
                check_record(bad,spec,_arrays(module),expected_record_binding=bad["binding"])
    def simple(self,kind):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.register_buffer("mask",torch.tensor([True,False]))
            def forward(self,x):
                if kind=="private_mask":return torch.where(x>0,x,-x)
                if kind=="float_mask":return x.masked_fill(self.mask.double(),0.)
                if kind=="private_fill":return x.masked_fill(self.mask,x.sum())
                if kind=="out":return torch.where(self.mask,x,-x,out=x)
                if kind=="inplace_shape":return x.reshape(1,2).masked_fill_(self.mask.reshape(2,1),.125)
                return x.masked_fill(self.mask,float("inf"))
        return Model().eval()
    def test_dynamic_private_and_invalid_calls(self):
        for kind in("private_mask","float_mask","private_fill","out","inplace_shape","nonfinite"):
            with self.subTest(kind=kind):
                module=self.simple(kind)
                with tempfile.TemporaryDirectory()as tmp:
                    p=Path(tmp);weights=export_state(module,p,resource_profile="general-tensors-v1")
                    spec=dict(format=SOURCE_FORMAT,files={"model.py":hashlib.sha256(b"def make(): return None").hexdigest()},entry="model.make",
                        inputs=[dict(name="x",shape=[2])],outputs=["output"],factory=dict(args=[],kwargs={}),
                        call=dict(format="poseidon-pytorch-call-v1",args=[dict(kind="tensor",name="x")],kwargs={}),weights=weights,
                        dtype_policy="strict-float64-v1",approximations=dict(format="poseidon-pytorch-approximations-v1",nodes=[],modules=[],defaults={}))
                    with self.assertRaises((ValueError,RuntimeError,TypeError)):
                        self.capture(module,spec)
    def test_working_copy_preserves_caller_and_original_math(self):
        from pytorch_input_binding import plan,signature_record,reference_module
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.register_buffer("mask",torch.tensor([True,False]))
            def forward(self,x):
                before=x.clone();x.view(2).masked_fill_(self.mask,.125);return before+x
        source=Model().eval()
        spec=dict(inputs=[dict(name="x",shape=[2])],call=dict(format="poseidon-pytorch-call-v1",
            args=[dict(kind="tensor",name="x")],kwargs={}))
        call=plan(spec,signature_record(source))
        module=reference_module(source,call,input_effect_policy="registered-working-copy-inputs-v1")
        for values in([0.,0.],[-.5,.25],[2.,3.]):
            x=torch.tensor(values,dtype=torch.float64);before=x.clone();result=module(x)
            expected=np.array(values)+np.where(np.array([True,False]),.125,np.array(values))
            np.testing.assert_array_equal(result.numpy(),expected);self.assertTrue(torch.equal(x,before))
        with self.assertRaisesRegex(ValueError,"reference_tensor_mutated"):
            reference_module(source,call)(torch.tensor([2.,3.],dtype=torch.float64))
        with self.assertRaisesRegex(ValueError,"effect_policy"):
            reference_module(source,call,input_effect_policy="anything")
    def test_working_copy_does_not_permit_container_mutation(self):
        from pytorch_input_binding import plan,signature_record,reference_module
        class Model(torch.nn.Module):
            def forward(self,data):
                data["x"]=data["x"]+1;return data["x"]
        source=Model().eval()
        spec=dict(inputs=[dict(name="x",shape=[2])],call=dict(format="poseidon-pytorch-call-v1",
            args=[dict(kind="dict",items=[["x",dict(kind="tensor",name="x")]])],kwargs={}))
        call=plan(spec,signature_record(source))
        x=torch.tensor([2.,3.],dtype=torch.float64);before=x.clone()
        with self.assertRaises((ValueError,TypeError)):
            reference_module(source,call,input_effect_policy="registered-working-copy-inputs-v1")(x)
        self.assertTrue(torch.equal(x,before))
    def test_input_effect_policy_bound_to_v33(self):
        row=json.loads((TOOLS/"cases.json").read_text())[9];module,spec=self.model(row);record=self.capture(module,spec)
        record["input_effect_policy"]="immutable-inputs-v1"
        record["binding"]=digest({k:v for k,v in record.items()if k!="binding"})
        with self.assertRaisesRegex(ValueError,"input_effect_policy"):
            check_record(record,spec,_arrays(module),expected_record_binding=record["binding"])
    def test_explicit_routes_and_output_none(self):
        from pytorch_ordered_contract import SOURCE_FORMATS,package_version,reference_version,allows_none_output
        from pytorch_model_intake import ordered
        from pytorch_static_tensor_transforms import selected
        from pytorch_source_package_policy import selected as packaged
        from pytorch_spatial_policy import supports_input_call,supports_spatial
        self.assertIn(SOURCE_FORMAT,SOURCE_FORMATS)
        spec={"format":SOURCE_FORMAT}
        self.assertEqual(package_version(spec),"poseidon-trusted-pytorch-provenance-v33")
        self.assertEqual(reference_version(spec),"poseidon-trusted-pytorch-reference-v33")
        self.assertTrue(all(fn(spec)for fn in(ordered,selected,packaged,supports_input_call,supports_spatial,allows_none_output)))
if __name__=="__main__":unittest.main()
