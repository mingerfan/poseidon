"""Approval, quota, request and immutable-source scope controls only."""
import copy,json,tempfile,unittest,sys
from pathlib import Path
from unittest import mock
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parents[2]))
import scope
from benchmark_graph import digest

class ScopeControls(unittest.TestCase):
    def setUp(self):self.plan=scope.read(scope.PROPOSAL)
    def forged(self):
        result=copy.deepcopy(self.plan)
        return result
    def rehash(self,value):
        value["binding"]=digest({k:v for k,v in value.items()if k!="binding"});return value
    def failure(self,value,code,**kwargs):
        with self.assertRaisesRegex(ValueError,code):scope.verify(self.rehash(value),**kwargs)
    def test_exact_prepared_scope(self):
        scope.verify(self.plan,sdk=False)
        self.assertEqual(len(self.plan["cases"]),18)
        self.assertEqual(self.plan["scope"]["provider_calls"],0)
        self.assertEqual(self.plan["scope"]["http_attempts"],0)
    def test_budget_expansion_rejected(self):
        bad=self.forged();bad["scope"]["wall_seconds"]+=1
        self.failure(bad,"offline_exact_scope",sdk=False)
    def test_case_substitution_rejected(self):
        bad=self.forged();bad["cases"][0]["case"]="replacement"
        self.failure(bad,"offline_exact_scope",sdk=False)
    def test_changed_candidate_bytes_rejected(self):
        bad=self.forged();row=bad["cases"][0];row["files"][row["candidate"]]="0"*64
        self.failure(bad,"offline_case_integrity",sdk=False)
    def test_changed_request_binding_rejected(self):
        bad=self.forged();bad["cases"][0]["request_sha256"]="0"*64
        self.failure(bad,"offline_request_binding",sdk=False)
    def test_changed_sdk_rejected(self):
        with mock.patch.object(scope,"identity",return_value=dict(manifest_sha256="0"*64)):
            with self.assertRaisesRegex(ValueError,"offline_sdk_identity"):scope.verify(self.plan)
    def test_absent_human_receipt_rejected(self):
        with tempfile.TemporaryDirectory(dir=HERE)as temp:
            with mock.patch.object(scope,"AUTHORIZATION",Path(temp)/"missing.json"):
                with self.assertRaises(FileNotFoundError):scope.authorization(self.plan)

if __name__=="__main__":
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ScopeControls))
    output=dict(tests=result.testsRun,failed=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
                provider_calls=0,compiler_calls=0,encrypted_calls=0,proposal_binding=scope.read(scope.PROPOSAL)["binding"])
    with(HERE/"scope-controls.json").open("x")as f:json.dump(output,f,sort_keys=True,indent=2);f.write("\n")
    raise SystemExit(0 if result.wasSuccessful()else 1)
