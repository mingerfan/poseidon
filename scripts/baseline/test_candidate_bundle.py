import copy,json,tempfile,unittest
from pathlib import Path
from candidate_bundle import check_bundle,sha
from benchmark_graph import digest
from compiler_configuration import configuration
class BundleTests(unittest.TestCase):
    def bundle(self,p):
        (p/"model.json").write_text("{}")
        (p/"candidate.py").write_text("def golden(x): return x\n")
        m=dict(format="poseidon-dsl-bundle-v1",files={n:sha(p/n) for n in ("model.json","candidate.py")},
               compiler_configuration=configuration("seal-cpu-eva-w45-v1"),origin={},
               agent_generated=False)
        m["binding"]=digest(m);(p/"manifest.json").write_text(json.dumps(m));return m
    def test_valid_manifest_is_integrity_only(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);m=self.bundle(p)
            self.assertFalse(check_bundle(p)["agent_generated"])
    def test_tampered_source_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);self.bundle(p);(p/"candidate.py").write_text("changed")
            with self.assertRaisesRegex(ValueError,"integrity"):check_bundle(p)
    def test_extra_executable_rejected_even_if_rehashed(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);m=self.bundle(p);m["files"]["../outside.py"]="x"
            m["binding"]=digest({k:v for k,v in m.items() if k!="binding"});(p/"manifest.json").write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError,"Unexpected"):check_bundle(p)
    def test_changed_profile_rejected_even_if_rehashed(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);m=self.bundle(p);m["compiler_configuration"]["waterline"]=5
            m["binding"]=digest({k:v for k,v in m.items() if k!="binding"});(p/"manifest.json").write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError,"profile"):check_bundle(p)
if __name__=="__main__":unittest.main()
