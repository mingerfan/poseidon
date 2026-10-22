"""Isolated v5 references and package metadata retain approximation provenance."""
import copy,hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from hecate_python_env import ROOT,WORK
from trusted_pytorch_import import run,prepare,validate_package
from candidate_bundle import load_model_provenance,check_provenance
from pytorch_source_contract import validate_manifest

class ApproximationImportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(dir=WORK/"results");cls.root=Path(cls.temp.name)
        cls.imports={};cls.fixture=ROOT/"scripts/baseline/cases/trusted-pytorch-approximation-v5"
        for name in ("nodes","norms"):
            folder=cls.root/name
            run(cls.fixture/(name+".json"),folder,resource_profile="general-tensors-v1",trusted=True)
            model=json.loads((folder/"model.json").read_text())
            cls.imports[name]=(folder,model,load_model_provenance(folder,model))
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()
    def test_seven_manifest_packages_registered(self):
        for name in ("modules","nodes","nested","functions","norms","compound","functions_nodes"):
            spec,paths=prepare(self.fixture/(name+".json"))
            validate_manifest(spec);validate_package(spec,{k:p.read_text() for k,p in paths.items()})
            self.assertEqual(spec["format"],"poseidon-trusted-pytorch-v5")
    def test_real_isolated_capture_and_independent_references(self):
        for _,model,value in self.imports.values():
            check_provenance(value,model)
            report=value["reference_report"]
            self.assertEqual(value["trusted_pytorch"]["format"],"poseidon-trusted-pytorch-provenance-v5")
            self.assertEqual(report["format"],"poseidon-trusted-pytorch-reference-v5")
            self.assertEqual(report["probes"],16)
            self.assertTrue(report["original_module_checked"])
            self.assertFalse(report["approximation_accuracy_certified"])
            self.assertFalse(report["encrypted_execution"])
            self.assertEqual(report["paid_calls"],0)
            self.assertIn("pytorch_approximation.py",report["source_hashes"])
    def test_tampered_selection_and_graph_profile_rejected(self):
        _,model,value=self.imports["nodes"]
        for edit in (
            lambda v:v["reference_report"]["capture_metadata"]["approximation_selection"].update(configuration_sha256="0"*64),
            lambda v:v["reference_report"]["capture_metadata"]["approximation_selection"]["selections"][0].update(profile_sha256="0"*64),
            lambda v:v["reference_report"]["capture_metadata"]["approximation_selection"]["selections"][0].update(source=dict(kind="defaults",selector="relu")),
            lambda v:v["reference_report"]["capture_metadata"]["approximation_selection"]["selections"][1].update(node_id="fx_act"),
            lambda v:v["reference_report"]["capture_metadata"].pop("approximation_selection")):
            bad=copy.deepcopy(value);edit(bad)
            with self.subTest(edit=edit),self.assertRaises(ValueError):check_provenance(bad,model)
    def test_package_metadata_never_reexecutes_trusted_sources(self):
        _,model,value=self.imports["nodes"]
        with patch("importlib.import_module",side_effect=AssertionError("Inert provenance required")):
            check_provenance(value,model)
    def test_old_format_cannot_claim_scoped_metadata(self):
        _,model,value=self.imports["nodes"]
        bad=copy.deepcopy(value);bad["trusted_pytorch"]["manifest"]["format"]="poseidon-trusted-pytorch-v4"
        with self.assertRaises(ValueError):check_provenance(bad,model)
    def test_v5_requires_versioned_config_exact_fields(self):
        spec,_=prepare(self.fixture/"nodes.json")
        for edit in (lambda v:v.update(approximations={"relu":v["approximations"]["defaults"]["relu"]}),
                     lambda v:v["approximations"].update(fallback=True),
                     lambda v:v["approximations"]["nodes"][0].update(callback="choose")):
            bad=copy.deepcopy(spec);edit(bad)
            with self.subTest(edit=edit),self.assertRaises(ValueError):validate_manifest(bad)
    def test_unmatched_selector_worker_failure_retained(self):
        spec,paths=prepare(self.fixture/"nodes.json")
        spec["approximations"]["nodes"].append(copy.deepcopy(spec["approximations"]["nodes"][0]))
        spec["approximations"]["nodes"][-1]["name"]="missing"
        folder=self.root/"unmatched_source";folder.mkdir()
        for name,path in paths.items():(folder/name).write_bytes(path.read_bytes())
        (folder/"manifest.json").write_text(json.dumps(spec))
        output=self.root/"unmatched_capture"
        with self.assertRaisesRegex(ValueError,"Trusted capture failed"):
            run(folder/"manifest.json",output,resource_profile="general-tensors-v1",trusted=True)
        self.assertIn("Unmatched approximation nodes selector",(output/"capture.log").read_text())
        self.assertFalse((output/"model.json").exists())

if __name__=="__main__":unittest.main()
