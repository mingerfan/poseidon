"""Synthetic reference-binding gates; no numerical or FHE execution claim."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_model_semantic_contexts as subject

class SupplementaryReferenceGateTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix="poseidon-supp-reference-gate-")
        self.addCleanup(self.temp.cleanup);self.folder=Path(self.temp.name)
        self.sources={"example.py":"0"*64}
        self.bundle=dict(bundle_sha256="b"*64,supplemental_models=[
            dict(model={"id":"one"},model_sha256="1"*64,provenance={"recipe":"input_negate"}),
            dict(model={"id":"two"},model_sha256="2"*64,provenance={"recipe":"output_negate"})])
        runner=subject.HERE/"verify_model_semantic_contexts.py"
        h=hashlib.sha256(runner.read_bytes()).hexdigest()
        binding=subject.digest(dict(bundle_sha256=self.bundle["bundle_sha256"],
            runtime_sources=self.sources,validator_sha256=h,probes=16,atol=1e-12,rtol=1e-12))
        self.run=dict(binding=binding,source_hashes=self.sources,bundle_sha256=self.bundle["bundle_sha256"],
                      runner_sha256=h,reference_atol=1e-12,reference_rtol=1e-12,probes_per_model=16)
        self.records={r["model"]["id"]:dict(model_sha256=r["model_sha256"],provenance=r["provenance"],
                                          state="passed",probes=16) for r in self.bundle["supplemental_models"]}
        self.report=dict(binding=binding,result_hashes={})
        self.write()
    def write(self):
        for name,record in self.records.items():
            path=self.folder/(name+".json");path.write_text(json.dumps(record))
            self.report["result_hashes"][name]=hashlib.sha256(path.read_bytes()).hexdigest()
        (self.folder/"run.json").write_text(json.dumps(self.run))
        (self.folder/"report.json").write_text(json.dumps(self.report))
    def check(self):return subject.reference_gate(self.folder,self.bundle,self.sources)
    def test_exact_records_and_failed_status_remain_distinct(self):
        self.assertEqual(self.check(),{"one":"passed","two":"passed"})
        self.records["two"]["state"]="failed";self.records["two"]["probes"]=3;self.write()
        self.assertEqual(self.check(),{"one":"passed","two":"failed"})
    def test_partial_reference_cannot_be_pass(self):
        self.records["one"]["probes"]=4;self.write()
        with self.assertRaisesRegex(ValueError,"Partial reference pass"):self.check()
    def test_changed_report_denominator_rejected(self):
        self.report["result_hashes"].pop("two")
        (self.folder/"report.json").write_text(json.dumps(self.report))
        with self.assertRaisesRegex(ValueError,"denominator"):self.check()
    def test_tolerance_change_rejected(self):
        self.run["reference_atol"]=1e-5;self.write()
        with self.assertRaisesRegex(ValueError,"comparison contract"):self.check()
    def test_changed_record_hash_rejected(self):
        p=self.folder/"one.json";p.write_text(p.read_text()+" ")
        with self.assertRaisesRegex(ValueError,"result hash"):self.check()
    def test_wrong_runtime_or_model_rejected(self):
        original=copy.deepcopy(self.run)
        self.run["source_hashes"]={};self.write()
        with self.assertRaisesRegex(ValueError,"source/bundle binding"):self.check()
        self.run=original;self.records["one"]["model_sha256"]="f"*64;self.write()
        with self.assertRaisesRegex(ValueError,"model identity"):self.check()
if __name__=="__main__":unittest.main()
