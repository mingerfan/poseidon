"""Capacity provenance retains its declared profile through export and audit."""
import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import candidate_bundle as bundle
from trusted_pytorch_import import run,PACKAGE_FORMAT
from hecate_python_env import ROOT,WORK

class MediumProvenanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(dir=WORK/"results")
        cls.folder=Path(cls.temp.name)/"imported"
        run(ROOT/"scripts/baseline/cases/trusted-pytorch-medium-v1/reduction_fusion.json",
            cls.folder,resource_profile="medium-v1",trusted=True)
        read=lambda name:json.loads((cls.folder/name).read_text())
        cls.model=read("model.json")
        manifest=read("trusted-python/manifest.json")
        cls.value=dict(original=read("original.json"),decomposition=read("decomposition.json"),
            reference_report=read("reference-report.json"),trusted_pytorch=dict(format=PACKAGE_FORMAT,
            manifest=manifest,files={name:(cls.folder/"trusted-python"/name).read_text() for name in manifest["files"]}))
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()
    def test_medium_binding_and_finite_reference(self):
        binding=bundle.check_provenance(self.value,self.model)
        self.assertEqual(binding["resource_profile"]["name"],"medium-v1")
        self.assertEqual(self.value["reference_report"]["probes"],16)
        self.assertTrue(self.value["reference_report"]["original_module_checked"])
    def test_provenance_check_does_not_import_user_source(self):
        with patch("importlib.import_module",side_effect=AssertionError("User source execution")) as importing:
            bundle.check_provenance(self.value,self.model)
            importing.assert_not_called()
    def test_independent_audit_retains_capacity(self):
        bundle.audit_provenance(self.value,self.model)
    def test_declared_profile_downgrade_rejected(self):
        model=copy.deepcopy(self.model);model["resource_profile"]="legacy-v1"
        with self.assertRaises(ValueError):bundle.check_provenance(self.value,model)
    def test_reference_profile_mismatch_rejected(self):
        value=copy.deepcopy(self.value);value["reference_report"]["resource_profile"]["name"]="legacy-v1"
        with self.assertRaisesRegex(ValueError,"runtime/probes"):bundle.check_provenance(value,self.model)
    def test_decomposition_profile_mismatch_rejected(self):
        value=copy.deepcopy(self.value);value["decomposition"]["resource_profile"]["name"]="resnet20-v1"
        with self.assertRaisesRegex(ValueError,"provenance mismatch"):bundle.check_provenance(value,self.model)
    def test_weight_and_source_identity_still_checked(self):
        value=copy.deepcopy(self.value);value["trusted_pytorch"]["files"]["model.py"]+="\n# changed\n"
        with self.assertRaisesRegex(ValueError,"hash"):bundle.check_provenance(value,self.model)
if __name__=="__main__":unittest.main()
