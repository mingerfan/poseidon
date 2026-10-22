"""Actual isolated v4 capture and inert provenance validation."""
import copy,hashlib,json,tempfile,unittest
from pathlib import Path
import torch
from benchmark_graph import digest
from hecate_python_env import ROOT,WORK
from trusted_pytorch_import import run
from candidate_bundle import load_model_provenance,check_provenance
from pytorch_weight_package import export_state

class StaticImportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(dir=WORK/"results")
        cls.root=Path(cls.temp.name)
        cls.imports={}
        fixture=ROOT/"scripts/baseline/cases/trusted-pytorch-static-v4"
        for case in ("loop_shape","scalar_dot"):
            folder=cls.root/case
            run(fixture/(case+".json"),folder,resource_profile="general-tensors-v1",trusted=True)
            model=json.loads((folder/"model.json").read_text())
            cls.imports[case]=(folder,model,load_model_provenance(folder,model))
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()
    def test_real_isolated_capture_binding_and_sixteen_references(self):
        for _,model,value in self.imports.values():
            check_provenance(value,model)
            report=value["reference_report"]
            self.assertEqual(report["format"],"poseidon-trusted-pytorch-reference-v4")
            self.assertEqual(report["probes"],16)
            self.assertTrue(report["pure_forward_checked"])
            self.assertFalse(report["encrypted_execution"])
            self.assertEqual(report["paid_calls"],0)
            self.assertEqual(report["capture_metadata"]["weight_binding"]["normalized_float64_sha256"],{})
            self.assertIn("pytorch_static_trace.py",report["source_hashes"])
    def test_static_policy_and_scalar_tree_tamper_rejected(self):
        _,model,value=self.imports["scalar_dot"]
        for modify in (
            lambda v:v["reference_report"]["capture_metadata"]["static_capture"].update(input_signature_sha256="0"*64),
            lambda v:v["reference_report"]["capture_metadata"]["static_capture"].update(max_fx_compute_nodes=1),
            lambda v:v["reference_report"]["capture_metadata"]["output_structure"]["items"][0]["value"].update(shape=[1])):
            bad=copy.deepcopy(value);modify(bad)
            with self.assertRaises(ValueError):check_provenance(bad,model)
    def test_v3_cannot_claim_v4_scalar_metadata(self):
        _,model,value=self.imports["scalar_dot"]
        bad=copy.deepcopy(value);bad["trusted_pytorch"]["manifest"]["format"]="poseidon-trusted-pytorch-v3"
        with self.assertRaises(ValueError):check_provenance(bad,model)
    def test_output_structure_is_bound_and_restored_without_source_execution(self):
        from unittest.mock import patch
        from program_execution import package_output_structure
        from pytorch_source_contract import restore_outputs
        from benchmark_graph import canonical
        _,model,value=self.imports["scalar_dot"]
        raw=canonical(value)
        with tempfile.TemporaryDirectory(dir=WORK/"results") as folder:
            folder=Path(folder);(folder/"model-provenance.json").write_bytes(raw)
            manifest=dict(model_provenance="included",files={"model-provenance.json":hashlib.sha256(raw).hexdigest()})
            with patch("importlib.import_module",side_effect=AssertionError("Provenance must be inert")):
                tree=package_output_structure(folder,manifest,model)
            restored=restore_outputs(dict(score=[.5],features=[1.,2.,3.],matrix=[[1.,2.,3.]]),tree)
            self.assertEqual(restored,dict(score=.5,features=([1.,2.,3.],[[[1.,2.,3.]]])))
    def test_nonempty_state_package_v4_retains_exact_binding(self):
        with tempfile.TemporaryDirectory(dir=WORK/"results") as folder:
            folder=Path(folder)
            text="import torch\ndef make(): return torch.nn.Linear(2,2).double()\n"
            (folder/"model.py").write_text(text)
            module=torch.nn.Linear(2,2).double()
            with torch.no_grad():
                module.weight.copy_(torch.tensor([[.25,-.5],[.125,.375]],dtype=torch.float64))
                module.bias.copy_(torch.tensor([.25,-.125],dtype=torch.float64))
            records=export_state(module,folder/"state",resource_profile="general-tensors-v1")
            for record in records.values():
                (folder/record["file"]).write_bytes((folder/"state"/record["file"]).read_bytes())
            spec=dict(format="poseidon-trusted-pytorch-v4",
                files={"model.py":hashlib.sha256(text.encode()).hexdigest()},entry="model.make",
                inputs=[dict(name="x",shape=[1,2])],outputs=["out"],approximations={},
                factory=dict(args=[],kwargs={}),dtype_policy="strict-float64-v1",weights=records)
            (folder/"manifest.json").write_text(json.dumps(spec))
            imported=folder/"captured";run(folder/"manifest.json",imported,resource_profile="general-tensors-v1",trusted=True)
            model=json.loads((imported/"model.json").read_text())
            value=load_model_provenance(imported,model)
            self.assertEqual(set(value["reference_report"]["capture_metadata"]["state_constant_sources"].values()),
                             {"weight","bias"})
            changed=copy.deepcopy(value);changed["reference_report"]["capture_metadata"]["weight_binding"]["normalized_float64_sha256"]["weight"]="0"*64
            with self.assertRaises(ValueError):check_provenance(changed,model)

if __name__=="__main__":unittest.main()
