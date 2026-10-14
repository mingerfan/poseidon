"""No-HTTP tests for original-scope, ledger and no-restart guards."""
import copy,tempfile,unittest
from pathlib import Path
from run_poseidon_whole_live import document,fixed_scope,provider_counts,seal,write,IDS
class WholeLiveGuards(unittest.TestCase):
 def prior(self):
  x={k:k for k in ("provider","limits","numeric_tolerance","outbound","excluded")}
  x.update(maximum_generations=12,maximum_http_attempts=48,bootstrapping=False,seal_fallback=False,money_cap=None)
  x["cases"]=[dict(id=i,model_path=i,model_sha256=i,model_file_sha256=i,request_id=i,request_sha256=i,live_command=[i]) for i in IDS]
  return x
 def test_identical_scope(self):
  p=self.prior();fixed_scope(copy.deepcopy(p),p)
 def test_no_extra_model(self):
  p=self.prior();q=copy.deepcopy(p);q["cases"].append(q["cases"][0])
  with self.assertRaises(ValueError):fixed_scope(q,p)
 def test_no_model_or_request_change(self):
  p=self.prior()
  for k in ("model_sha256","model_file_sha256","request_id","request_sha256","live_command"):
   q=copy.deepcopy(p);q["cases"][0][k]="changed"
   with self.subTest(k=k),self.assertRaises(ValueError):fixed_scope(q,p)
 def test_no_budget_or_provider_drift(self):
  p=self.prior()
  for k in ("provider","maximum_generations","maximum_http_attempts","limits","numeric_tolerance","bootstrapping","seal_fallback"):
   q=copy.deepcopy(p);q[k]="changed"
   with self.subTest(k=k),self.assertRaises(ValueError):fixed_scope(q,p)
 def test_zero_and_maximum_calls(self):
  self.assertEqual(provider_counts({}), (0,0))
  self.assertEqual(provider_counts(dict(generation_attempts=4,request_attempts=16,calls=[{}]*16)),(4,16))
 def test_excess_call_budget(self):
  for g,h in [(5,16),(4,17),(1,5),(3,2),(-1,0)]:
   with self.subTest(g=g,h=h),self.assertRaises(ValueError):
    provider_counts(dict(generation_attempts=g,request_attempts=h,calls=[{}]*max(h,0)))
 def test_reject_missing_or_false_ledger(self):
  for m in [dict(generation_attempts=True,request_attempts=1,calls=[{}]),dict(generation_attempts=1,request_attempts=1,calls=[])]:
   with self.assertRaises(ValueError):provider_counts(m)
 def test_inflight_attempt_consumes_budget(self):
  self.assertEqual(provider_counts(dict(generation_attempts=1,request_attempts=1,calls=[dict(status="in_flight")])),(1,1))
 def test_exclusive_dispatch_and_case_start(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/"dispatch.json";write(p,seal(dict(binding_test="initial")),True)
   with self.assertRaises(FileExistsError):write(p,seal(dict(binding_test="retry")),True)
   self.assertEqual(document(p)["binding_test"],"initial")
 def test_binding_rejects_mutation(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/"plan.json";v=seal(dict(value=1));write(p,v,True)
   self.assertEqual(document(p)["value"],1);v["value"]=2;write(p,v)
   with self.assertRaises(ValueError):document(p)
if __name__=="__main__":unittest.main()
