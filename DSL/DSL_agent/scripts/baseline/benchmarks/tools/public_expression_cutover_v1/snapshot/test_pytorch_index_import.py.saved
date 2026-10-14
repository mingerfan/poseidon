"""Isolated v6 capture, private provenance and old-format boundary checks."""
import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from hecate_python_env import ROOT,WORK
from trusted_pytorch_import import run,prepare
from candidate_bundle import load_model_provenance,check_provenance
from pytorch_source_contract import validate_manifest
from application_capabilities import input_sources
CASES=ROOT/"scripts/baseline/cases/trusted-pytorch-index-v6"
class IndexImportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(dir=WORK/"results");cls.imports={}
        for name in ("slicing_0","axes_0","narrow_0","vector_1","index_select_0","index_select_1"):
            folder=Path(cls.temp.name)/name
            run(CASES/(name+".json"),folder,resource_profile="general-tensors-v1",trusted=True)
            model=json.loads((folder/"model.json").read_text());value=load_model_provenance(folder,model)
            cls.imports[name]=(folder,model,value)
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()
    def test_actual_sandbox_capture_and_sixteen_reference_probes(self):
        self.assertEqual(len(self.imports),6)
        for _,model,value in self.imports.values():
            check_provenance(value,model);report=value["reference_report"]
            self.assertEqual(value["trusted_pytorch"]["format"],"poseidon-trusted-pytorch-provenance-v6")
            self.assertEqual(report["format"],"poseidon-trusted-pytorch-reference-v6")
            self.assertEqual(report["probes"],16)
            self.assertFalse(report["encrypted_execution"]);self.assertEqual(report["paid_calls"],0)
            self.assertIn("pytorch_static_index.py",report["source_hashes"])
            self.assertEqual(report["capture_metadata"]["static_capture"]["version"],"signature-static-index-v1")
    def test_index_provenance_checks_do_not_execute_model_code(self):
        _,model,value=self.imports["vector_1"]
        with patch("importlib.import_module",side_effect=AssertionError("Provenance must remain inert")):
            check_provenance(value,model)
    def test_public_index_tensor_state_bound_without_float_conversion(self):
        _,_,value=self.imports["vector_1"];metadata=value["reference_report"]["capture_metadata"]
        self.assertEqual(metadata["source_state_dtypes"],{"idx":"int64"})
        self.assertEqual(value["trusted_pytorch"]["manifest"]["weights"]["idx"]["dtype"],"<i8")
        self.assertFalse(metadata["source_precision_execution_compared"])
    def test_plan_and_version_tampering_rejected(self):
        _,model,value=self.imports["slicing_0"]
        for mutate in (
            lambda v:v["reference_report"]["capture_metadata"]["index_capture"].update(graph_sha256="0"*64),
            lambda v:v["reference_report"]["capture_metadata"]["static_capture"].update(version="signature-static-v1"),
            lambda v:v["trusted_pytorch"]["manifest"].update(format="poseidon-trusted-pytorch-v5"),
            lambda v:v["reference_report"]["capture_metadata"].pop("index_capture")):
            bad=copy.deepcopy(value);mutate(bad)
            with self.subTest(mutate=mutate),self.assertRaises(ValueError):check_provenance(bad,model)
    def test_shared_capability_manifest_registers_v6_without_execution_claim(self):
        entry=input_sources()["trusted_pytorch"]
        self.assertIn("poseidon-trusted-pytorch-v6",entry["formats"])
        self.assertEqual(entry["static_indexing"]["version"],"poseidon-trusted-pytorch-v6")
        self.assertNotIn("encrypted_execution",entry["static_indexing"])
    def test_fifteen_source_manifests_use_existing_resource_profile(self):
        for p in CASES.glob("*.json"):
            spec,_=prepare(p);validate_manifest(spec)
            self.assertEqual(spec["format"],"poseidon-trusted-pytorch-v6")
    def test_v6_application_default_and_explicit_legacy_packing(self):
        from application_component import Application
        from trusted_pytorch_import import preparation_options,PACKAGE_V5
        from unified_capacity_constants import INDEX_VERSION,CHANNEL_VERSION
        _,model,provenance=self.imports["slicing_0"]
        original={"chunk_period":4}
        self.assertEqual(preparation_options(provenance,model,original),
            {"chunk_period":4,"capacity_packing_version":INDEX_VERSION})
        self.assertEqual(original,{"chunk_period":4})
        options={"capacity_packing_version":CHANNEL_VERSION}
        self.assertEqual(preparation_options(provenance,model,options),options)
        older=copy.deepcopy(provenance);older["trusted_pytorch"]["format"]=PACKAGE_V5
        self.assertEqual(preparation_options(older,model,original),original)
        app=object.__new__(Application)
        with patch.object(app,"submit",return_value="checked") as submit:
            self.assertEqual(app.submit_import(self.imports["slicing_0"][0],options=original),"checked")
            self.assertEqual(submit.call_args.kwargs["options"]["capacity_packing_version"],INDEX_VERSION)
        with patch.object(app,"submit",return_value="checked") as submit:
            app.submit_import(self.imports["slicing_0"][0],options=options)
            self.assertEqual(submit.call_args.kwargs["options"]["capacity_packing_version"],CHANNEL_VERSION)

    def test_explicit_trust_is_required(self):
        with self.assertRaisesRegex(ValueError,"Explicit"):
            run(CASES/"slicing_0.json",Path(self.temp.name)/"not_trusted",resource_profile="general-tensors-v1",trusted=False)

if __name__=="__main__":unittest.main()
