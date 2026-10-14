"""Real isolated v7 captures and inert provenance/old-format boundary checks."""
import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from hecate_python_env import ROOT,WORK
from trusted_pytorch_import import run,prepare,preparation_options,PACKAGE_V7
from candidate_bundle import load_model_provenance,check_provenance
CASES=ROOT/"scripts/baseline/cases/trusted-pytorch-matrix-v7"
class MatrixTensorImportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(dir=WORK/"results");cls.imports={}
        for name in ("cipher_vv_0","cipher_mv_1","cipher_vm_2","cipher_batch_1","public_vector_l_0","public_batch_r_2"):
            folder=Path(cls.temp.name)/name
            run(CASES/name/"manifest.json",folder,resource_profile="general-tensors-v1",trusted=True)
            model=json.loads((folder/"model.json").read_text());value=load_model_provenance(folder,model)
            cls.imports[name]=(folder,model,value)
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()
    def test_six_real_sandbox_captures_bind_matrix_rules_and_original_reference(self):
        for _,model,value in self.imports.values():
            check_provenance(value,model)
            report=value["reference_report"]
            self.assertEqual(report["format"],"poseidon-trusted-pytorch-reference-v7")
            self.assertEqual(value["trusted_pytorch"]["format"],PACKAGE_V7)
            self.assertTrue(report["original_module_checked"]);self.assertTrue(report["pure_forward_checked"])
            self.assertEqual(report["probes"],16);self.assertFalse(report["encrypted_execution"])
            self.assertEqual(report["paid_calls"],0)
            self.assertIn("pytorch_matrix_plan.py",report["source_hashes"])
    def test_v7_keeps_index_scalar_approximation_and_weight_contracts(self):
        _,_,value=self.imports["cipher_vv_0"]
        metadata=value["reference_report"]["capture_metadata"]
        self.assertEqual(metadata["output_structure"],{"kind":"tensor","name":"out","shape":[]})
        self.assertEqual(metadata["static_capture"]["version"],"signature-static-index-v1")
        self.assertEqual(metadata["index_capture"]["entries"],[])
        self.assertIn("approximation_selection",metadata);self.assertIn("weight_binding",metadata)
    def test_matrix_source_and_plan_tampering_rejected(self):
        _,model,value=self.imports["public_batch_r_2"]
        mutations=[
            lambda v:v["reference_report"]["source_hashes"].pop("pytorch_matrix_plan.py"),
            lambda v:v["reference_report"]["capture_metadata"].pop("matrix_capture"),
            lambda v:v["reference_report"]["capture_metadata"]["matrix_capture"].update(version="unknown"),
            lambda v:v["reference_report"]["capture_metadata"]["matrix_capture"]["entries"][0]["plan"].update(core_shape=[1]),
            lambda v:v["trusted_pytorch"]["manifest"].update(format="poseidon-trusted-pytorch-v6")]
        for mutate in mutations:
            bad=copy.deepcopy(value);mutate(bad)
            with self.subTest(mutate=mutate),self.assertRaises(ValueError):check_provenance(bad,model)
    def test_matrix_provenance_check_stays_data_only(self):
        _,model,value=self.imports["cipher_batch_1"]
        with patch("importlib.import_module",side_effect=AssertionError("No uploaded source import")):
            check_provenance(value,model)
    def test_v7_routes_new_imports_and_preserves_explicit_packing(self):
        from unified_capacity_constants import INDEX_VERSION,CHANNEL_VERSION
        _,model,value=self.imports["cipher_batch_1"]
        self.assertEqual(preparation_options(value,model,{"chunk_period":8}),
                         {"chunk_period":8,"capacity_packing_version":INDEX_VERSION})
        given={"capacity_packing_version":CHANNEL_VERSION}
        self.assertEqual(preparation_options(value,model,given),given)
    def test_explicit_trust_is_required_before_source_execution(self):
        with self.assertRaisesRegex(ValueError,"Explicit"):
            run(CASES/"cipher_vv_0/manifest.json",Path(self.temp.name)/"untrusted",
                resource_profile="general-tensors-v1",trusted=False)
if __name__=="__main__":unittest.main()
