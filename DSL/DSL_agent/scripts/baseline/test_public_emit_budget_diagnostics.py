"""Boundary rejection remains authoritative and reveals only a fixed public budget."""
import ast,unittest
from application_diagnostics import attach,checked,make
from candidate_check_diagnostic import CandidateCheckError
from component_contract import validate_program
from test_public_construction_diagnostics import request,candidate
from unified_graph_contract import validate_candidate
class PublicEmitBudgetDiagnostics(unittest.TestCase):
 def setUp(self):self.q=request()
 def make_candidate(self,n):
  body='out = x0\nfor i in range('+str(n//2)+'):\n    out = (out + zero_ct) + zero_ct\n'
  if n%2:body+='out = out + zero_ct\n'
  return candidate(self.q,body+'return [out]')
 def test_255_and_256_operations_still_admitted(self):
  for n in [255,256]:
   with self.subTest(n=n):
    out=validate_program(self.make_candidate(n),self.q);self.assertEqual(out['status'],'static_accepted_not_executed');self.assertFalse(out['encrypted_execution'])
 def test_257th_operation_has_original_source_witness(self):
  c=self.make_candidate(257)
  with self.assertRaises(CandidateCheckError)as caught:validate_candidate(c,self.q)
  self.assertEqual(caught.exception.reason,'construction_emitted_operation_budget');self.assertEqual(caught.exception.location,dict(kind='source',line=6,column=10));self.assertEqual(caught.exception.context,dict(observed=257,limit=256))
  out=attach(validate_program(c,self.q),candidate=c,request=self.q);self.assertEqual(out['failure']['reason_code'],caught.exception.reason);self.assertEqual(out['failure']['context'],caught.exception.context);self.assertFalse(out['failure']['repeat_key_available']);self.assertIn('256',out['public_feedback']['diagnostic'])
 def test_feedback_rejects_forged_or_private_context(self):
  loc=dict(kind='source',line=6,column=10)
  for context in [{},{'observed':256,'limit':256},{'observed':258,'limit':256},{'observed':257,'limit':257},{'observed':True,'limit':256},{'observed':257,'limit':256,'private':0}]:
   with self.subTest(context=context):
    with self.assertRaises(ValueError):make('construction_emitted_operation_budget',location=loc,context=context)
  good=make('construction_emitted_operation_budget',location=loc,context=dict(observed=257,limit=256));self.assertEqual(checked(good),good)
  with self.assertRaises(ValueError):checked(dict(good,repeat_key_available=True))
 def test_integrity_failure_still_overrides_budget_failure(self):
  d=make('construction_emitted_operation_budget',location=dict(kind='source',line=6,column=10),context=dict(observed=257,limit=256));out=attach(dict(status='failed',failure=dict(layer='integrity'),structured_diagnostic=d));self.assertEqual(out['failure']['reason_code'],'integrity_rejected')
