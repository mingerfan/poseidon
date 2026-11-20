"""Public AST failures flow through real static admission and bounded feedback."""
import ast,unittest
from application_diagnostics import attach,checked,make,render
from candidate_check_diagnostic import CandidateCheckError
from component_contract import prepare_task,validate_program
from unified_graph_contract import validate_candidate
def request():
 model=dict(format='poseidon-model-graph-v1',id='diagnostic_square',inputs=[dict(name='x',shape=[4])],constants={},nodes=[dict(id='sq',op='square',inputs=['x'],attrs={},outputs=['s'])],outputs=[dict(name='output',value='s')])
 return prepare_task(model,dict(construction_profile='hecate-unified-public-v1',generation_guidance='explicit-v1',compiler_configuration='poseidon-cpu-eva-w45-current-v1'))
def candidate(q,body):
 names=[item['dsl_name']for item in q['layout']['inputs']]+['zero_ct']
 body=body.replace('x0',names[0])
 header='@hc.func("'+','.join('c'for _ in names)+'")\ndef golden('+', '.join(names)+'):\n'
 return dict(schema=1,request_id=q['request_id'],hecate_source=header+''.join('    '+s+'\n'for s in body.splitlines()))
class PublicConstructionDiagnostics(unittest.TestCase):
 def setUp(self):self.q=request()
 def rejected(self,body,reason,line):
  c=candidate(self.q,body)
  with self.assertRaises(CandidateCheckError)as caught:validate_candidate(c,self.q)
  self.assertEqual(caught.exception.reason,reason);self.assertEqual(caught.exception.location['line'],line)
  outcome=validate_program(c,self.q);self.assertEqual(outcome['status'],'rejected');self.assertFalse(outcome['encrypted_execution']);self.assertFalse(outcome['numerically_validated'])
  delivered=attach(dict(outcome,public_feedback=dict(status='failed',layer='static_check',category='candidate',diagnostic='PRIVATE /secret/reference.npy')),candidate=c,request=self.q)
  self.assertEqual(delivered['failure']['reason_code'],reason);self.assertEqual(delivered['failure']['location']['line'],line)
  feedback=delivered['public_feedback'];self.assertEqual(set(feedback),{'status','layer','category','diagnostic'});self.assertIn('Location:',feedback['diagnostic']);self.assertNotIn('PRIVATE',str(feedback));self.assertNotIn('/secret',str(feedback));return delivered
 def test_unsupported_numpy_call_has_authoritative_location(self):
  self.rejected('m = np.zeros(4)\nreturn [x0 * m]','construction_call_not_supported',3)
 def test_unsupported_attribute_has_authoritative_location(self):
  self.rejected('m = np.zeros\nreturn [x0 * x0]','construction_attribute_not_supported',3)
 def test_identity_operator_maps_to_enclosing_source(self):
  self.rejected('v = None\nif v is None:\n    return [x0 * x0]\nreturn [x0 * x0]','construction_syntax_not_supported',4)
 def test_shared_ast_operator_reports_first_occurrence(self):
  self.rejected('v = None\nif v is None:\n    v = 1\nif v is None:\n    v = 2\nreturn [x0 * x0]','construction_syntax_not_supported',4)
 def test_unknown_global_uses_existing_undefined_value_diagnostic(self):
  self.rejected('return [x0 * missing_global]', 'undefined_value', 3)
 def test_admitted_public_loop_still_normalizes(self):
  c=candidate(self.q,'first = True\nout = zero_ct\nfor i in range(2):\n    if first:\n        out = x0 * x0\n        first = False\n    else:\n        out = out + zero_ct\nreturn [out]')
  outcome=validate_program(c,self.q);self.assertEqual(outcome['status'],'static_accepted_not_executed');self.assertFalse(outcome['encrypted_execution']);self.assertFalse(outcome['numerically_validated'])
 def test_new_reasons_require_empty_context_and_real_source_witness(self):
  for reason in ['construction_call_not_supported','construction_attribute_not_supported','construction_syntax_not_supported']:
   with self.assertRaises(ValueError):make(reason)
   with self.assertRaises(ValueError):make(reason,location=dict(kind='source',line=3,column=8),context={'actual':1})
   good=make(reason,location=dict(kind='source',line=3,column=8));self.assertEqual(checked(good),good)
   forged=dict(good,repeat_key_available=True)
   with self.assertRaises(ValueError):checked(forged)
 def test_integrity_still_overrides_typed_candidate_failure(self):
  value=make('construction_call_not_supported',location=dict(kind='source',line=3,column=8));out=attach(dict(status='failed',failure=dict(layer='integrity'),structured_diagnostic=value));self.assertEqual(out['failure']['reason_code'],'integrity_rejected')
