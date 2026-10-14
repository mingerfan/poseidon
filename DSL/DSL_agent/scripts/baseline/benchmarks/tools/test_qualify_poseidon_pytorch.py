"""Admission and frozen-source regressions for the developer import runner."""
import copy,json,tempfile,unittest
from pathlib import Path
from qualify_poseidon_pytorch import load_cases,ROOT

class ImportManifestTests(unittest.TestCase):
    def setUp(self):
        self.folder=ROOT/"scripts/baseline/cases/trusted-pytorch-extended-v1"
        self.manifest=json.loads((self.folder/"acceptance.json").read_text())
    def altered(self,change):
        with tempfile.TemporaryDirectory() as name:
            target=Path(name)
            for path in self.folder.iterdir():
                if path.is_file():(target/path.name).write_bytes(path.read_bytes())
            value=copy.deepcopy(self.manifest);change(value)
            (target/"acceptance.json").write_text(json.dumps(value))
            return load_cases(target/"acceptance.json")
    def test_extended_sources_and_four_logical_inputs(self):
        cases,files=load_cases(self.folder/"acceptance.json")
        self.assertEqual(len(cases),5)
        self.assertEqual(len(cases[2][2]["inputs"]),4)
        self.assertEqual({x[0]["resource_profile"] for x in cases},{"resnet20-v1","medium-v1"})
        self.assertEqual(len(files),8)
    def test_original_medium_manifest(self):
        cases,_=load_cases(ROOT/"scripts/baseline/cases/trusted-pytorch-medium-v1/acceptance.json")
        self.assertEqual(len(cases),3)
        self.assertTrue(all(x[0]["resource_profile"]=="medium-v1" for x in cases))
    def test_v1_cannot_admit_resnet_profile(self):
        with self.assertRaisesRegex(ValueError,"Frozen medium layout"):
            self.altered(lambda d:d.update(format="poseidon-pytorch-acceptance-list-v1"))
    def test_unknown_profile_rejected(self):
        with self.assertRaisesRegex(ValueError,"Configured capacity profile"):
            self.altered(lambda d:d["cases"][0].update(resource_profile="larger"))
    def test_unconfigured_period_rejected(self):
        for period in (True,257,1024):
            with self.subTest(period=period),self.assertRaisesRegex(ValueError,"Configured packing period"):
                self.altered(lambda d:d["cases"][0].update(chunk_period=period))
    def test_source_drift_rejected(self):
        with self.assertRaisesRegex(ValueError,"Case hash"):
            self.altered(lambda d:d["cases"][0].update(sha256="0"*64))
    def test_repeated_identity_rejected(self):
        with self.assertRaisesRegex(ValueError,"Case identity"):
            self.altered(lambda d:d["cases"].append(copy.deepcopy(d["cases"][0])))
    def test_shard_limit_preserved(self):
        with self.assertRaisesRegex(ValueError,"Case manifest"):
            self.altered(lambda d:d.update(cases=d["cases"]*17))

if __name__=="__main__":unittest.main()
