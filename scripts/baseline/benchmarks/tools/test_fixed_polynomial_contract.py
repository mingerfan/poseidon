"""Contract, forgery and legacy-request checks; no FHE claims."""
import copy,json,sys,unittest
from pathlib import Path
BASE=Path(__file__).resolve().parents[2];sys.path.insert(0,str(BASE))
from fixed_polynomial_cases import cases
from unified_graph_contract import prepare,validate_request,validate_candidate
from upstream_candidate_helpers import POLYNOMIAL_PROFILE,PROFILE
from upstream_adapters.fixed_polynomial import coefficients
from benchmark_graph import digest
from compiler_configuration import PROFILE_SHA256,configuration
class ContractTests(unittest.TestCase):
    def request(self,row,profile=POLYNOMIAL_PROFILE):
        return prepare(row["model"],PROFILE_SHA256,configuration("seal-cpu-eva-w40-v1"),
                       helper_profile=profile,helper_exercise=[row["helper"]] if profile==POLYNOMIAL_PROFILE else None)
    def check(self,r,s):
        return validate_candidate(dict(schema=1,request_id=r["request_id"],hecate_source=s),r)
    def test_independent_coefficients_and_contexts(self):
        for row in cases():
            with self.subTest(case=row["id"]):
                r=self.request(row);validate_request(r)
                self.assertEqual(coefficients(row["helper"]),row["model"]["constants"]["c0"])
                self.assertEqual(len(self.check(r,row["source"])["upstream_calls"]),1)
    def test_no_implicit_capability_or_raw_factory(self):
        row=cases()[0]
        for profile in (None,PROFILE):
            with self.subTest(profile=profile),self.assertRaises(ValueError):
                self.check(self.request(row,profile),row["source"])
        for expression in ('__import__("poly")','Poly_Default(x,2)','Poly_Default(1)',
                           'Poly_Default([x])','Poly_Default(v=x)','GenPoly()(x)',
                           'Poly_sign(x)','Poly_genRelu6(x)','hc.bootstrap(x)'):
            with self.subTest(expression=expression),self.assertRaises(ValueError):
                self.check(self.request(row),'@hc.func("c,c")\ndef golden(x,zero_ct):\n    return '+expression+'\n')
    def test_rehashed_capability_mutations(self):
        row=cases()[0]
        for field,value in (("coefficients",[0.,1.]),("work",0),("bootstrap",True),("source","custom")):
            r=self.request(row);r["upstream_helpers"]["helpers"]["Poly_Default"][field]=value
            r["request_id"]=digest({k:v for k,v in r.items() if k!="request_id"})
            with self.subTest(field=field),self.assertRaises(ValueError):validate_request(r)
    def test_discarded_cancelled_dead_and_wrong_helper_not_coverage(self):
        from upstream_helper_coverage import check_exercise
        row=cases()[0];r=self.request(row)
        for body in ("unused=Poly_Default(x)\n    return x",
                     "v=Poly_Default(x)\n    return v-v+x","return Poly_Leaf1(x)"):
            source='@hc.func("c,c")\ndef golden(x,zero_ct):\n    '+body+'\n'
            with self.subTest(body=body),self.assertRaises(ValueError):check_exercise(source,r)
    def test_legacy_requests_still_validate(self):
        from workspace_paths import RESULTS
        report=json.loads((RESULTS/"stage2-polynomial-w40-r73/report.json").read_text())
        for row in report["rows"]:
            validate_request(json.loads((Path(row["folder"])/"request.json").read_text()))
if __name__=="__main__":unittest.main()
