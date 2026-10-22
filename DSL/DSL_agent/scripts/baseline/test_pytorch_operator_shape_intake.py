"""Explicit operator result metadata, unchanged older policies and provenance."""
import copy,json,unittest
from pathlib import Path
import torch
from pytorch_static_trace import trace,policy_record,POLICY,INDEX_POLICY,ARITHMETIC_POLICY,OPERATOR_POLICY
from pytorch_model_capture import capture,REGISTERED_FUNCTION_POLICY,ADAPTIVE_FUNCTION_POLICY
from pytorch_arithmetic_normalization import normalize,rename_record,bind_record
from pytorch_weight_package import binding as weight_binding
import pytorch_model_intake as intake
PROFILE="general-tensors-v1"
ROOT=(Path(__file__).resolve().parents[4] if Path(__file__).resolve().parents[2].name == "DSL_agent" and Path(__file__).resolve().parents[3].name == "DSL" else Path(__file__).resolve().parents[2])
class Adaptive(torch.nn.Module):
    def __init__(self,functional=True,query=True):
        super().__init__();self.functional=functional;self.query=query
        self.pool=torch.nn.AdaptiveAvgPool1d(2)
    def forward(self,x):
        y=torch.nn.functional.adaptive_avg_pool1d(x,2) if self.functional else self.pool(x)
        return y.reshape(y.numel()) if self.query else y.flatten()
def bound():
    spec=json.loads((ROOT/"scripts/baseline/cases/trusted-pytorch-operator-shape-v1/adaptive_pool_0_functional_query/manifest.json").read_text())
    spec["format"]=intake.OPERATOR_SHAPE_FORMAT
    source=trace(Adaptive().eval(),spec["inputs"],resource_profile=PROFILE,policy=OPERATOR_POLICY)
    normal,record=normalize(source,resource_profile=PROFILE)
    traced=trace(normal,spec["inputs"],resource_profile=PROFILE,policy=INDEX_POLICY)
    record=rename_record(record,normal,traced)
    original,_,captured=capture(traced,spec["inputs"],spec["outputs"],approximations=spec["approximations"],
        resource_profile=PROFILE,dtype_policy="strict-float64-v1",preserve_output_structure=True,
        record_state_sources=True,capture_policy=INDEX_POLICY,approximation_policy="node-module-default-v1",
        matrix_policy="static-tensor-matmul-v1",functional_policy=ADAPTIVE_FUNCTION_POLICY)
    metadata=captured["pytorch_capture"];metadata["source_state_dtypes"]={}
    metadata["state_constant_sources"]={};metadata["weight_binding"]=weight_binding({},spec["weights"])
    metadata["arithmetic_capture"]=bind_record(record,original,captured_fx=traced)
    metadata["source_shape_capture"]=policy_record(spec["inputs"],PROFILE,policy=OPERATOR_POLICY)
    return original,spec,metadata
class OperatorShapePolicies(unittest.TestCase):
    def test_module_and_functional_result_queries_are_explicit(self):
        signature=[dict(name="x",shape=[1,1,5])]
        for functional in (True,False):
            model=Adaptive(functional).eval()
            traced=trace(model,signature,resource_profile=PROFILE,policy=OPERATOR_POLICY)
            value=torch.tensor([[[-.5,.125,.25,-.75,1.]]],dtype=torch.float64)
            torch.testing.assert_close(traced(value),model(value),atol=0,rtol=0)
            for old in (POLICY,INDEX_POLICY,ARITHMETIC_POLICY):
                with self.subTest(old=old,functional=functional),self.assertRaisesRegex(ValueError,"Static shape unavailable"):
                    trace(model,signature,resource_profile=PROFILE,policy=old)
    def test_old_functional_capture_default_does_not_expand(self):
        signature=[dict(name="x",shape=[1,1,5])]
        source=trace(Adaptive(query=False).eval(),signature,resource_profile=PROFILE,policy=INDEX_POLICY)
        with self.assertRaisesRegex(ValueError,"Unsupported PyTorch call"):
            capture(source,signature,["result"],resource_profile=PROFILE,capture_policy=INDEX_POLICY,preserve_output_structure=True)
    def test_invalid_functional_policy_is_rejected(self):
        with self.assertRaisesRegex(ValueError,"Functional capture policy"):
            capture(Adaptive(),[],[],functional_policy="all-functions")
    def test_unknown_result_shape_is_not_guessed(self):
        class Unknown(torch.nn.Module):
            def forward(self,x):
                y=torch.flip(x,(2,))
                return y.reshape(y.numel())
        with self.assertRaisesRegex(ValueError,"Static shape unavailable"):
            trace(Unknown(),[dict(name="x",shape=[1,1,5])],resource_profile=PROFILE,policy=OPERATOR_POLICY)
    def test_data_control_after_registered_operator_is_rejected(self):
        class DataBranch(torch.nn.Module):
            def forward(self,x):
                y=torch.nn.functional.adaptive_avg_pool1d(x,2)
                if y.sum()>0:return y
                return -y
        with self.assertRaisesRegex(Exception,"control flow"):
            trace(DataBranch(),[dict(name="x",shape=[1,1,5])],resource_profile=PROFILE,policy=OPERATOR_POLICY)
    def test_public_split_and_arithmetic_metadata_are_inherited(self):
        class Split(torch.nn.Module):
            def forward(self,x):
                y=x.multiply(.25).add(.125)
                a,b=y.split([1,1],dim=1)
                return a.reshape(a.numel()),b.reshape(b.numel())
        model=Split();signature=[dict(name="x",shape=[1,2,3])]
        traced=trace(model,signature,resource_profile=PROFILE,policy=OPERATOR_POLICY)
        value=torch.arange(6,dtype=torch.float64).reshape(1,2,3)
        actual,expected=traced(value),model(value)
        for a,e in zip(actual,expected):torch.testing.assert_close(a,e,atol=0,rtol=0)
class OperatorShapeProvenance(unittest.TestCase):
    def test_new_source_formats_and_policy_are_bound(self):
        original,spec,metadata=bound()
        intake.check_capture_metadata(metadata,original,spec,resource_profile=PROFILE)
        _,binding=intake.capture_binding(original,spec,resource_profile=PROFILE,capture_metadata=metadata)
        self.assertEqual(binding["capture_policy_version"],intake.OPERATOR_SHAPE_FORMAT)
        self.assertEqual(intake.package_format(spec),intake.OPERATOR_SHAPE_PACKAGE_FORMAT)
        self.assertNotEqual(intake.OPERATOR_SHAPE_REFERENCE_FORMAT,intake.SHAPE_REFERENCE_FORMAT)
        self.assertEqual(intake.source_shape_policy({"format":intake.FORMAT}),INDEX_POLICY)
        self.assertEqual(intake.source_shape_policy({"format":intake.SHAPE_FORMAT}),ARITHMETIC_POLICY)
        self.assertEqual(intake.source_shape_policy(spec),OPERATOR_POLICY)
    def test_v9_cannot_claim_v10_metadata(self):
        original,spec,metadata=bound();spec["format"]=intake.SHAPE_FORMAT
        with self.assertRaisesRegex(ValueError,"source shape policy binding"):
            intake.check_capture_metadata(metadata,original,spec,resource_profile=PROFILE)
    def test_shape_policy_rejects_cross_version_and_malformed_fields(self):
        original,spec,metadata=bound()
        for field,value in [("version",ARITHMETIC_POLICY),("max_fx_compute_nodes",True),
                            ("input_signature_sha256","0"*64),("shape_sources","guessed")]:
            bad=copy.deepcopy(metadata);bad["source_shape_capture"][field]=value
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,"source shape policy binding"):
                intake.check_capture_metadata(bad,original,spec,resource_profile=PROFILE)
        for remove in (True,False):
            bad=copy.deepcopy(metadata)
            if remove:bad.pop("source_shape_capture")
            else:bad["source_shape_capture"]["extra"]=True
            with self.assertRaisesRegex(ValueError,"source shape policy binding"):
                intake.check_capture_metadata(bad,original,spec,resource_profile=PROFILE)
    def test_capabilities_preserve_v9_and_do_not_claim_backend_evidence(self):
        from application_capabilities import input_sources
        entry=input_sources()["trusted_pytorch"]
        self.assertIn(intake.OPERATOR_SHAPE_FORMAT,entry["formats"])
        self.assertEqual(entry["arithmetic_result_shape"]["version"],intake.SHAPE_FORMAT)
        self.assertEqual(entry["operator_result_shape"]["version"],intake.OPERATOR_SHAPE_FORMAT)
        self.assertEqual(entry["operator_result_shape"]["policy"],OPERATOR_POLICY)
        self.assertEqual(entry["functional_adaptive_average"]["policy"],ADAPTIVE_FUNCTION_POLICY)
        for key in ("operator_result_shape","functional_adaptive_average"):
            for layer in ("compiler_status","encrypted_status","agent_status"):
                self.assertEqual(entry[key][layer],"unverified")
if __name__=="__main__":unittest.main()
