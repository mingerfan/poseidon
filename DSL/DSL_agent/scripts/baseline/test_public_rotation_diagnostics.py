"""Public rotation rejection diagnostics; no candidate execution or FHE claims."""
import ast,json,tempfile,unittest
from function_construction import normalize
from hecate_contract import UNIFIED_FLAT_CONTRACT
from candidate_check_diagnostic import CandidateCheckError
from application_diagnostics import attach
from component_contract import validate_program
from unified_graph_contract import validate_candidate
from test_public_construction_diagnostics import request,candidate

def source(expr):
 return '@hc.func("c,c")\ndef golden(x,zero_ct):\n    return '+expr.replace('x0','x')+'\n'
def expand(raw,constants=None,*,unified=True):
 return normalize(raw,{}if constants is None else constants,input_names=('x','zero_ct'),
  flat_contract=UNIFIED_FLAT_CONTRACT if unified else 'hecate-function-v5',slot_period=32 if unified else 4)

class PublicRotationDiagnostics(unittest.TestCase):
 def test_invalid_unified_steps_have_original_argument_location(self):
  for step in [0,3,-1,-2,31,32]:
   with self.subTest(step=step):
    raw=source('x0.rotate('+str(step)+')')
    node=next(n for n in ast.walk(ast.parse(raw))if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=='rotate').args[0]
    with self.assertRaises(CandidateCheckError)as caught:expand(raw)
    self.assertEqual(caught.exception.reason,'rotation_step_not_allowed')
    self.assertEqual(caught.exception.location,dict(kind='source',line=node.lineno,column=node.col_offset))
    self.assertEqual(caught.exception.context,{})
 def test_all_provisioned_steps_remain_admitted(self):
  for step in [1,2,4,8,16]:
   with self.subTest(step=step):
    out=expand(source('x0.rotate('+str(step)+')'))
    calls=[n for n in ast.walk(ast.parse(out['source']))if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=='rotate']
    self.assertEqual([n.args[0].value for n in calls],[step])
 def test_legacy_negative_and_composite_steps_remain_admitted(self):
  for step in [-3,-2,-1,1,2,3]:
   with self.subTest(step=step):expand(source('x0.rotate('+str(step)+')'),unified=False)
 def test_plain_receiver_does_not_claim_a_bad_rotation_step(self):
  for step in [1,3]:
   with self.subTest(step=step):
    with self.assertRaises(ValueError)as caught:expand(source('w.rotate('+str(step)+')'),{'w':[.5]})
    self.assertNotIsInstance(caught.exception,CandidateCheckError)
    self.assertIn('Rotation requires a ciphertext',str(caught.exception))
 def test_noninteger_steps_keep_original_type_rejection(self):
  for step in ['True','.5','x0','w']:
   with self.subTest(step=step):
    with self.assertRaises(ValueError)as caught:expand(source('x0.rotate('+step+')'),{'w':[.5]})
    self.assertNotIsInstance(caught.exception,CandidateCheckError)
 def test_static_feedback_forwards_only_existing_public_reason(self):
  q=request();c=candidate(q,'return [x0.rotate(0)]')
  with self.assertRaises(CandidateCheckError):validate_candidate(c,q)
  out=attach(dict(validate_program(c,q),public_feedback=dict(status='failed',layer='static_check',category='candidate',diagnostic='PRIVATE /secret/reference.npy')),candidate=c,request=q)
  self.assertEqual(out['failure']['reason_code'],'rotation_step_not_allowed')
  self.assertEqual(out['failure']['location']['line'],3)
  self.assertEqual(out['failure']['context'],{})
  self.assertFalse(out.get('encrypted_execution',False))
  self.assertNotIn('PRIVATE',json.dumps(out['public_feedback']))
  self.assertNotIn('/secret',json.dumps(out['public_feedback']))
 def test_Application_stops_before_backend_qualification(self):
  from application_component import Application
  from application_policy import DEFAULT_BUDGET
  from test_application_component import Backend
  class InvalidRotation:
   kind='scripted_replay';agent_calls=0;calls=0
   def generate(self,q,history):
    self.calls+=1;return json.dumps(candidate(q,'return [x0.rotate(0)]'))
  with tempfile.TemporaryDirectory()as folder:
   backend=Backend();app=Application(folder,backend);provider=InvalidRotation()
   task=app.submit(request()['model'],options=dict(construction_profile='hecate-unified-public-v1',generation_guidance='explicit-v1',compiler_configuration='poseidon-cpu-eva-w45-current-v1'),limits=dict(DEFAULT_BUDGET,generations=1,http_attempts=1))
   result=app.run(task,provider)
   self.assertEqual(provider.calls,1);self.assertEqual(backend.calls,0)
   self.assertEqual(result['failure']['reason_code'],'rotation_step_not_allowed')
   self.assertEqual(result['failure']['location']['line'],3)

if __name__=='__main__':unittest.main()
