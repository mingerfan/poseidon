"""Source variants, original math and production V15 routing without FHE/API."""
import copy,hashlib,json,math,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from benchmark_graph import digest
from model_decomposition import polynomial_profile
from pytorch_model_capture import capture
from pytorch_ordered_capture import prepare_module
from pytorch_gelu_source import POLICY,check_record
from pytorch_gelu_reference import original_scalar,evaluate_source,validate
from pytorch_approximation import FORMAT as APPROXIMATION_FORMAT
from pytorch_ordered_contract import (GELU_SOURCE_FORMAT,check_capture_metadata,capture_binding,
                                    check_reference,source_identity)
from pytorch_weight_package import binding as weight_binding
import pytorch_model_intake as facade

A=1/math.sqrt(2*math.pi)
P=dict(method="polynomial",basis="power",domain=[-2.,2.],
       coefficients=[0.,.5,A,0.,-A/6])
def config():
    return dict(format=APPROXIMATION_FORMAT,defaults=dict(gelu=copy.deepcopy(P)),modules=[],nodes=[])

class Model(torch.nn.Module):
    def __init__(self,variant,style):
        super().__init__();self.variant=variant;self.style=style
        self.activation=torch.nn.GELU(approximate=variant)
    def forward(self,x):
        y=x*.125
        if self.style=="module":return self.activation(y)+y
        if self.style=="functional":return F.gelu(y,approximate=self.variant).square()
        return F.gelu(input=y,approximate=self.variant),y

def prepared(variant,style):
    outputs=["result","prior"] if style=="input_keyword" else ["result"]
    inputs=[dict(name="x",shape=[2,3] if style=="module" else [4])]
    result=prepare_module(Model(variant,style),inputs,outputs,approximations=config(),
        resource_profile="general-tensors-v1",dtype_policy="strict-float64-v1",
        matrix_policy="static-tensor-matmul-v1",approximation_policy="node-module-default-v1",
        common_calls=True,fused_matrix_policy="poseidon-fx-fused-matrix-draft-v1",
        gelu_source_policy=POLICY)
    return result,inputs,outputs

def source_spec(inputs,outputs):
    source="def make(): pass\n"
    return dict(format=GELU_SOURCE_FORMAT,files={"models.py":hashlib.sha256(source.encode()).hexdigest()},
                entry="models.make",factory=dict(args=[],kwargs={}),dtype_policy="strict-float64-v1",
                inputs=inputs,outputs=outputs,approximations=config(),weights={})

class GeluSourceCaptureTests(unittest.TestCase):
    def test_six_actual_capture_contexts_match_source_and_two_polynomial_references(self):
        for variant in ("none","tanh"):
            for style in ("module","functional","input_keyword"):
                with self.subTest(variant=variant,style=style):
                    value,inputs,outputs=prepared(variant,style)
                    high,low,binding,context=value;metadata=binding["pytorch_capture"]
                    rows=metadata["gelu_source_capture"]["entries"]
                    self.assertEqual([r["variant"] for r in rows],[variant])
                    finite=validate(value,inputs,outputs,count=16)
                    self.assertEqual(finite["status"],"passed")
                    self.assertLessEqual(finite["max_original_torch_vs_independent_ideal"],1e-12)
                    self.assertLessEqual(finite["max_decomposition_error"],1e-12)
                    self.assertFalse(finite["approximation_accuracy_certified"])

    def test_tanh_is_distinct_from_erf_source_function(self):
        self.assertGreater(abs(original_scalar(-.75,"tanh")-original_scalar(-.75,"none")),
                           1e-5+1e-4*abs(original_scalar(-.75,"tanh")))
        for variant in ("none","tanh"):
            x=np.array([-2.,-.75,0.,.3,1.2,2.])
            expected=np.array([original_scalar(float(v),variant) for v in x])
            actual=F.gelu(torch.from_numpy(x),approximate=variant).numpy()
            np.testing.assert_allclose(expected,actual,atol=1e-15,rtol=1e-15)

    def test_old_capture_none_hash_and_default_policy_unchanged(self):
        model=torch.nn.Sequential(torch.nn.GELU()).double().eval()
        args=([dict(name="x",shape=[4])],["out"])
        options=dict(approximations=dict(gelu=P),resource_profile="general-tensors-v1")
        default=capture(model,*args,**options)
        explicit=capture(model,*args,**options,gelu_source_policy="exact-gelu-source-v1")
        selected=capture(model,*args,**options,gelu_source_policy=POLICY)
        self.assertEqual(default,explicit)
        self.assertEqual(default[:2],selected[:2])
        self.assertNotIn("gelu_source_capture",default[2]["pytorch_capture"])
        self.assertIn("gelu_source_capture",selected[2]["pytorch_capture"])

    def test_old_capture_tanh_still_rejected(self):
        model=torch.nn.Sequential(torch.nn.GELU(approximate="tanh"))
        with self.assertRaisesRegex(ValueError,"Exact GELU"):
            capture(model,[dict(name="x",shape=[4])],["out"],approximations=dict(gelu=P))

    def test_v15_manifest_and_current_capture_provenance_reconstruct(self):
        value,inputs,outputs=prepared("tanh","module")
        high,low,captured,context=value
        spec=source_spec(inputs,outputs);weights=weight_binding({}, {})
        metadata=captured["pytorch_capture"];metadata["weight_binding"]=weights
        actual, bound=capture_binding(high,spec,resource_profile="general-tensors-v1",
                                     capture_metadata=metadata,expected_weight_binding=weights)
        self.assertEqual(actual,low)
        finite=validate((high,actual,bound,context),inputs,outputs,count=16)
        from model_resources import identity
        import trusted_pytorch_import as legacy
        report=dict(format="poseidon-trusted-pytorch-reference-v15",mode="isolated_trusted_pytorch_reference",
            original_sha256=digest(high),model_sha256=digest(low),manifest_sha256=digest(spec),
            source_hashes=source_identity(),runtime=legacy.runtime_identity(),probes=16,
            resource_profile=identity("general-tensors-v1"),reference_status="passed",reference_dtype="float64",
            capture_metadata=metadata,finite_reference=finite,source_precision_execution_compared=False,
            approximation_accuracy_certified=False,runtime_layout_validated=False,
            encrypted_execution=False,agent_generated=False,paid_calls=0)
        check_reference(report,high,low,spec,resource_profile="general-tensors-v1",expected_weight_binding=weights)
        self.assertTrue(facade.ordered(spec))
        self.assertEqual(facade.package_format(spec),"poseidon-trusted-pytorch-provenance-v15")

    def test_v14_rejects_v15_policy_metadata(self):
        value,inputs,outputs=prepared("tanh","module")
        high,low,bound,_=value;metadata=bound["pytorch_capture"]
        weights=weight_binding({}, {});metadata["weight_binding"]=weights
        spec=source_spec(inputs,outputs);spec["format"]="poseidon-trusted-pytorch-v14"
        with self.assertRaisesRegex(ValueError,"metadata fields"):
            check_capture_metadata(metadata,high,spec,resource_profile="general-tensors-v1",
                                   expected_weight_binding=weights)

    def test_polynomial_and_source_identity_tampering_rejected(self):
        value,_,_=prepared("tanh","module");high,_,bound,_=value
        metadata=bound["pytorch_capture"];alias=metadata["ordered_alias_capture"]
        record=metadata["gelu_source_capture"]
        options=dict(emissions=alias["capture_emissions"],
                     normalized_origins=alias["normalization"]["mathematical_node_origins"],
                     resource_profile="general-tensors-v1")
        check_record(record,high,**options)
        for edit in (lambda r:r["entries"][0].update(source_identity="torch-2.0.1-gelu-erf-v1"),
                     lambda r:r["entries"][0].update(approximation_sha256="0"*64),
                     lambda r:r["entries"].clear()):
            bad=copy.deepcopy(record);edit(bad)
            bad["binding"]=digest({k:v for k,v in bad.items() if k!="binding"})
            with self.assertRaises(ValueError):check_record(bad,high,**options)

    def test_missing_approximation_and_unknown_policy_rejected(self):
        with self.assertRaisesRegex(ValueError,"Explicit approximation"):
            capture(torch.nn.Sequential(torch.nn.GELU(approximate="tanh")),[dict(name="x",shape=[4])],["out"],
                    gelu_source_policy=POLICY)
        with self.assertRaisesRegex(ValueError,"GELU source policy"):
            capture(torch.nn.GELU(),[dict(name="x",shape=[4])],["out"],
                    approximations=dict(gelu=P),gelu_source_policy="automatic")

    def test_submit_source_v15_uses_existing_application_boundary(self):
        from application_model_source import _snapshot,FORMAT
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);source="def make(): pass\n";(folder/"models.py").write_text(source)
            inputs=[dict(name="x",shape=[4])];spec=source_spec(inputs,["result"])
            (folder/"manifest.json").write_text(json.dumps(spec))
            kind,files=_snapshot(dict(format=FORMAT,kind="trusted_pytorch",
                manifest=str(folder/"manifest.json"),trust_local_code=True),"general-tensors-v1")
            self.assertEqual(kind,"trusted_pytorch")
            self.assertEqual(set(files),{"manifest.json","models.py"})
            self.assertEqual(json.loads(files["manifest.json"])["format"],GELU_SOURCE_FORMAT)

if __name__=="__main__":unittest.main()
