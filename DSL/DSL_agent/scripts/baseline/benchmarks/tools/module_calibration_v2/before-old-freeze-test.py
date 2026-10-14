"""Frozen approximation controls; injected observations, no source/FHE/API."""
import copy,tempfile,unittest,hashlib
from pathlib import Path
from unittest.mock import patch
from benchmark_graph import digest,canonical
from hecate_python_env import WORK
from source_approximation_preparation import freeze,PreparationError
from source_calibration_signature import FORMAT

class FreezeTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=WORK/"results");self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.source=self.root/"original";self.source.mkdir()
        code="import torch\nclass Model(torch.nn.Module):\n    def forward(self,x): return torch.nn.functional.gelu(x)\ndef make(): return Model().eval()\n"
        (self.source/"model.py").write_text(code)
        self.spec=dict(format="poseidon-trusted-pytorch-v38",files={"model.py":hashlib.sha256(code.encode()).hexdigest()},
            entry="model.make",inputs=[dict(name="x",shape=[2])],outputs=["out"],
            factory=dict(args=[],kwargs={}),dtype_policy="convert-float64-v1",weights={},
            approximations=dict(format="poseidon-pytorch-approximations-v1",defaults={},modules=[],nodes=[]),
            call=dict(format="poseidon-pytorch-call-v1",args=[dict(kind="tensor",name="x")],kwargs={}))
        (self.source/"manifest.json").write_bytes(canonical(self.spec))
        self.signature=dict(format=FORMAT,outputs=[dict(name="out",shape=[2])])
        self.config=dict(format="poseidon-pytorch-approximations-v1",
            defaults={"gelu":dict(method="polynomial",basis="power",domain=[-1.,1.],coefficients=[0.,.5,.25])},
            modules=[],nodes=[])
        self.graph=dict(nodes=[dict(name="gelu",op="call_function",target="torch.nn.functional.gelu")],
                        graph_sha256="a"*64)
        self.request=dict(resource_profile="general-tensors-v1",output_structure={"kind":"tensor","name":"out"},
            selection=dict(selectors=[dict(id="value",node="gelu",op="call_function",
                target="torch.nn.functional.gelu",point="input",argument=0,projection={"kind":"tensor"})]))
        self.report=dict(binding="b"*64,fx_graph=self.graph,observations={"value":dict(
            op="call_function",target="torch.nn.functional.gelu",minimum=-.4,maximum=.4,samples=3,values=6)})
    def call(self,**changes):
        args=dict(expected_manifest_sha256=digest(self.spec),source_signature=self.signature,
            calibration_evidence=[dict(directory=str(self.root/"observed"),report_binding="b"*64)],
            configuration=self.config,trust_local_code=True)
        args.update(changes)
        with patch("source_approximation_preparation.evidence",return_value=(self.request,self.report)):
            return freeze(self.source/"manifest.json",self.root/"frozen",**args)
    def test_explicit_profile_freeze_preserves_source_and_weights(self):
        before={p.name:p.read_bytes()for p in self.source.iterdir()}
        result=self.call()
        self.assertEqual(result["status"],"frozen");self.assertFalse(result["profile_fitted"])
        self.assertFalse(result["ckks_validation_performed"]);self.assertFalse(result["original_application_accuracy_certified"])
        self.assertEqual(result["configuration"],self.config)
        frozen=__import__("json").loads(Path(result["configured_manifest"]).read_bytes())
        self.assertEqual({k:v for k,v in frozen.items()if k!="approximations"},
                         {k:v for k,v in self.spec.items()if k!="approximations"})
        self.assertEqual(before,{p.name:p.read_bytes()for p in self.source.iterdir()})
    def test_outside_domain_never_changes_configuration(self):
        original=copy.deepcopy(self.config);self.report["observations"]["value"]["maximum"]=2.
        with self.assertRaises(PreparationError)as e:self.call()
        self.assertEqual(e.exception.code,"approximation_calibration_domain_exceeded")
        self.assertEqual(self.config,original);self.assertFalse((self.root/"frozen").exists())
    def test_missing_observation_rejected(self):
        self.report["observations"].clear();self.request["selection"]["selectors"]=[]
        with self.assertRaises(PreparationError)as e:self.call()
        self.assertEqual(e.exception.code,"approximation_calibration_missing")
    def test_wrong_node_target_rejected(self):
        self.report["observations"]["value"]["target"]="torch.rsqrt"
        with self.assertRaises(PreparationError):self.call()
    def test_output_or_non_operand_observation_rejected(self):
        for changed in (dict(point="output",argument=None),dict(point="input",argument=1)):
            self.request["selection"]["selectors"][0].update(changed)
            with self.assertRaises(PreparationError):self.call()
    def test_module_configuration_is_explicitly_unsupported(self):
        value=copy.deepcopy(self.config);value["modules"]=[dict(path="activation",function="gelu",
            profile=value["defaults"]["gelu"])]
        with self.assertRaises(PreparationError)as e:self.call(configuration=value)
        self.assertEqual(e.exception.code,"approximation_calibration_location_unsupported")
    def test_stale_manifest_and_missing_trust_rejected(self):
        with self.assertRaises(PreparationError):self.call(expected_manifest_sha256="0"*64)
        with self.assertRaises(PreparationError):self.call(trust_local_code=False)
    def test_duplicate_evidence_rejected(self):
        item=dict(directory=str(self.root/"observed"),report_binding="b"*64)
        with self.assertRaises(PreparationError):self.call(calibration_evidence=[item,item])
    def test_node_override_has_precedence_and_profile_binding(self):
        value=copy.deepcopy(self.config);profile=copy.deepcopy(value["defaults"]["gelu"]);profile["domain"]=[-.5,.5]
        value["nodes"]=[dict(name="gelu",function="gelu",profile=profile)]
        result=self.call(configuration=value)
        self.assertEqual(result["source_locations"][0]["domain"],[-.5,.5])
        self.assertEqual(result["approximation_selection"]["selections"][0]["source"],dict(kind="nodes",selector=0))
    def test_unmatched_selector_and_invalid_profile_rejected(self):
        value=copy.deepcopy(self.config);value["nodes"]=[dict(name="missing",function="gelu",profile=value["defaults"]["gelu"])]
        with self.assertRaises(ValueError):self.call(configuration=value)
        value=copy.deepcopy(self.config);value["defaults"]["gelu"]["coefficients"]=[float("nan")]
        with self.assertRaises(ValueError):self.call(configuration=value)

if __name__=="__main__":unittest.main()
