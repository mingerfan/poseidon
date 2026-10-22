"""Isolated v3 state-weight imports and inert provenance; no FHE/API claims."""
import copy, hashlib, json, tempfile, unittest
from pathlib import Path
import numpy as np
from benchmark_graph import digest
from hecate_python_env import ROOT,WORK
from trusted_pytorch_import import run,check_capture_metadata
from candidate_bundle import load_model_provenance,check_provenance
from pytorch_weight_package import read,binding
from component_contract import prepare_task
from deepseek_provider import public_request
from program_execution import package_output_structure
SOURCE=ROOT/"scripts/baseline/cases/trusted-pytorch-state-v3"

class StateImportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(dir=WORK/"results");cls.imports={}
        for case in ("mlp","conv_pool","residual","batchnorm","matrix_fusion","shape_linear"):
            folder=Path(cls.temp.name)/case
            result=run(SOURCE/case/"manifest.json",folder,resource_profile="general-tensors-v1",trusted=True)
            model=json.loads((folder/"model.json").read_text())
            value=load_model_provenance(folder,model)
            cls.imports[case]=(folder,model,value,result)
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_six_isolated_captures_reference_and_exact_state_binding(self):
        for case,(_,model,value,result) in self.imports.items():
            check_provenance(value,model)
            spec=value["trusted_pytorch"]["manifest"]
            arrays,_=read(SOURCE/case,spec["weights"],resource_profile="general-tensors-v1")
            meta=value["reference_report"]["capture_metadata"]
            self.assertEqual(meta["weight_binding"],binding(arrays,spec["weights"]))
            self.assertEqual(set(meta["state_constant_sources"].values()),
                             set(spec["weights"])-{n for n in spec["weights"] if n.endswith("num_batches_tracked")})
            self.assertEqual(value["reference_report"]["probes"],16)
            self.assertTrue(result["reference_verified"])
            self.assertFalse(result["encrypted_execution"])
            self.assertEqual(value["reference_report"]["paid_calls"],0)

    def test_integer_scalar_buffer_keeps_source_dtype(self):
        _,_,value,_=self.imports["batchnorm"]
        meta=value["reference_report"]["capture_metadata"]
        self.assertEqual(meta["source_state_dtypes"]["norm.num_batches_tracked"],"int64")
        self.assertEqual(value["trusted_pytorch"]["manifest"]["weights"]["norm.num_batches_tracked"]["shape"],[])

    def test_source_snapshot_includes_verified_npy_files(self):
        for case,(folder,_,value,_) in self.imports.items():
            for item in value["trusted_pytorch"]["manifest"]["weights"].values():
                self.assertEqual((folder/"trusted-python"/item["file"]).read_bytes(),
                                 (SOURCE/case/item["file"]).read_bytes())

    def test_weight_dtype_source_and_declared_binding_tampering_rejected(self):
        _,model,value,_=self.imports["mlp"]
        for mutate in (
            lambda v:v["reference_report"]["capture_metadata"]["source_state_dtypes"].update({"layers.0.weight":"float64"}),
            lambda v:v["trusted_pytorch"]["manifest"]["weights"]["layers.0.weight"].update(sha256="0"*64),
            lambda v:v["reference_report"]["capture_metadata"]["weight_binding"]["normalized_float64_sha256"].update({"layers.0.weight":"0"*64}),
            lambda v:v["reference_report"]["capture_metadata"]["state_constant_sources"].update({"unknown":"layers.0.weight"})):
            bad=copy.deepcopy(value);mutate(bad)
            with self.assertRaises(ValueError):check_provenance(bad,model)

    def test_retained_weight_file_hash_is_checked_before_application_admission(self):
        folder,model,value,_=self.imports["residual"]
        item=next(iter(value["trusted_pytorch"]["manifest"]["weights"].values()))
        file=folder/"trusted-python"/item["file"];raw=file.read_bytes()
        try:
            file.write_bytes(raw[:-1]+bytes([raw[-1]^1]))
            with self.assertRaises(ValueError):load_model_provenance(folder,model)
        finally:file.write_bytes(raw)

    def test_state_sources_stay_out_of_provider_request(self):
        _,model,value,_=self.imports["mlp"]
        request=prepare_task(model,dict(chunk_period=4,
                             generation_guidance="explicit-v10",public_weight_policy="descriptors-v1"))
        context=public_request(request)
        for field in ("trusted_pytorch","model_provenance","weight_binding","source_state_dtypes","state_constant_sources"):
            self.assertNotIn(field,context)
        self.assertNotIn("models.py",json.dumps(context))
        changed=copy.deepcopy(model)
        key=next(k for k,v in changed["constants"].items() if type(v) is list and type(v[0]) is list)
        changed["constants"][key][0][0]+=.03125
        other=prepare_task(changed,dict(chunk_period=4,
                           generation_guidance="explicit-v10",public_weight_policy="descriptors-v1"))
        self.assertNotEqual(request["request_id"],other["request_id"])

    def test_static_output_structure_survives_v3_provenance(self):
        _,model,value,_=self.imports["matrix_fusion"]
        from benchmark_graph import canonical
        with tempfile.TemporaryDirectory(dir=WORK/"results") as root:
            root=Path(root);raw=canonical(value);(root/"model-provenance.json").write_bytes(raw)
            manifest=dict(model_provenance="included",files={"model-provenance.json":hashlib.sha256(raw).hexdigest()})
            tree=package_output_structure(root,manifest,model)
            self.assertEqual(tree["kind"],"dict")
            self.assertEqual(tree["items"][1]["value"]["kind"],"list")

if __name__=="__main__":unittest.main()
