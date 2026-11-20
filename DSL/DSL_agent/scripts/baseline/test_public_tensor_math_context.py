"""Bounded public mathematical tensors; real preparation/transport, API/compiler/FHE0."""
import copy,json,math,unittest
from pathlib import Path
from benchmark_graph import canonical,digest
from component_contract import prepare_task,reconstruct_request
from general_tensor_cases import Builder
from public_tensor_math_context import POLICY,FORMAT,ROLE_FORMAT
from public_weight_context import binding,project,request_digest,verify_binding,verify_context
from public_math_context import binary64,descriptor,verify_projected
from deepseek_provider import public_request,Config,HTTPSTransport,SYSTEM
B=Path('/home/lhohy/Code/Poseidon');PRE=B/'DSL/DSL_agent/docs/whole_models/workplans/b3-attention-free-r1851'
def model(values,shape,op='multiply'):
 b=Builder('unseen_tensor_values',[shape]);return b.finish(b.node(op,['input0',b.constant(values)]))
def request(m,period=8):
 return prepare_task(m,dict(chunk_period=period,generation_guidance='explicit-v1',public_weight_policy=POLICY,construction_profile='hecate-unified-public-v1',compiler_configuration='poseidon-cpu-eva-w45-current-v1'))
def tensor_row(q):return next(r for r in q['public_weight_binding']['math_roles']['constants']if r['kind']=='tensor_operand')
def rebind(c):
 r=c['public_weight_binding']['math_roles'];r['binding']=digest({k:v for k,v in r.items()if k!='binding'});c['request_id']=request_digest(c)
class TensorMathContext(unittest.TestCase):
 def setUp(self):self.q=request(model([[.5,-1.],[1.,.5]],[2,2]))
 def test_nonuniform_c_order_shape_and_exact_plain_bindings(self):
  r=tensor_row(self.q);self.assertEqual(r['descriptor']['shape'],[2,2]);self.assertEqual([e['value']for e in r['elements']],[.5,-1.,1.,.5]);self.assertEqual([e['index']for e in r['elements']],list(range(4)))
  for e in r['elements']:
   self.assertTrue(e['plain_names'])
   for name in e['plain_names']:self.assertEqual(binary64(self.q['public_constants'][name]),binary64(e['value']))
 def test_uniform_and_broadcast_vectors_preserve_logical_tensor(self):
  for shape,values in [([2,2],[.5,.5]),([2,3],[1.,-1.,0.])]:
   with self.subTest(shape=shape):
    q=request(model(values,shape));r=tensor_row(q);self.assertEqual(r['descriptor'],descriptor(values));self.assertEqual([e['value']for e in r['elements']],values)
 def test_unbound_packed_broadcast_leaf_refuses_before_provider(self):
  with self.assertRaisesRegex(ValueError,'missing_required_public_plain_binding'):request(model([1.,-.5,0.],[2,3]))
 def test_new_format_is_explicit_and_roundtrips_original_model(self):
  self.assertEqual(self.q['public_weight_binding']['policy'],POLICY);self.assertEqual(self.q['public_weight_binding']['format'],FORMAT);self.assertEqual(self.q['public_weight_binding']['math_roles']['format'],ROLE_FORMAT)
  self.assertEqual(reconstruct_request(self.q),self.q);self.assertEqual(verify_binding(self.q),project(self.q));self.assertEqual(verify_context(public_request(self.q)),public_request(self.q));self.assertEqual(public_request(public_request(self.q)),public_request(self.q))
 def test_missing_nonzero_binding_refuses_before_provider(self):
  q=copy.deepcopy(self.q);q['public_constants']={k:v for k,v in q['public_constants'].items()if not(type(v)in(int,float)and v==.5)}
  with self.assertRaisesRegex(ValueError,'missing_required_public_plain_binding'):binding(q,policy=POLICY)
 def test_matching_never_uses_numeric_tolerance(self):
  q=copy.deepcopy(self.q);q['public_constants']={k:v for k,v in q['public_constants'].items()if not(type(v)in(int,float)and v==.5)};q['public_constants']['near']=.5000000000000001
  with self.assertRaisesRegex(ValueError,'missing_required_public_plain_binding'):binding(q,policy=POLICY)
 def test_signed_zero_keeps_original_tensor_hash(self):
  q=request(model([-0.,0.],[2]));r=tensor_row(q);self.assertEqual(r['descriptor']['content_sha256'],digest([-0.,0.]));self.assertNotEqual(r['descriptor']['content_sha256'],digest([0.,0.]));self.assertTrue(math.copysign(1.,r['elements'][0]['value'])<0);self.assertTrue(all(e['exact_zero']and not e['binding_required']for e in r['elements']));verify_context(public_request(q))
 def test_single_tensor_capacity_not_raised(self):
  with self.assertRaisesRegex(ValueError,'tensor_math_capacity'):request(model([.5]*257,[257]),period=256)
 def test_aggregate_scalar_and_tensor_capacity_not_raised(self):
  b=Builder('aggregate_math_limit',[[128]]);x=b.node('multiply',['input0',b.constant([.5]*128)]);x=b.node('add',[x,b.constant([1.]*128)]);m=b.finish(b.node('add',[x,b.constant(.5)]))
  with self.assertRaisesRegex(ValueError,'public_math_metadata_capacity'):request(m,period=128)
 def test_changed_value_and_plain_binding_rejected_after_rehash(self):
  for field,value in [('value',.25),('plain_names',['nonexistent'])]:
   with self.subTest(field=field):
    c=public_request(self.q);tensor_row(c)['elements'][0][field]=value;rebind(c)
    with self.assertRaises(ValueError):verify_context(c)
 def test_changed_shape_or_element_order_rejected_after_rehash(self):
  c=public_request(self.q);tensor_row(c)['elements'].reverse();rebind(c)
  with self.assertRaises(ValueError):verify_context(c)
  c=public_request(self.q);tensor_row(c)['descriptor']['shape']=[4];rebind(c)
  with self.assertRaises(ValueError):verify_context(c)
 def test_missing_required_role_cannot_be_reclassified_opaque(self):
  c=public_request(self.q);r=c['public_weight_binding']['math_roles'];row=tensor_row(c);r['constants'].remove(row);r['opaque_constants'].append(dict(logical_name=row['logical_name'],descriptor=row['descriptor'],usages=row['usages'],exact_whole_plain_names=[]));r['math_elements']-=len(row['elements']);r['prebound_math_scalars']={};rebind(c)
  with self.assertRaises(ValueError):verify_context(c)
 def test_private_unknown_fields_and_false_capability_claims_rejected(self):
  for change in [lambda c:c.update(private_reference=[1.]),lambda c:c['public_weight_binding']['math_roles'].update(private_answers=[1.]),lambda c:tensor_row(c)['elements'][0].update(private_answer=1.),lambda c:c['public_weight_binding']['math_roles'].update(required_tensor_operand_roles_complete=False),lambda c:c['public_weight_binding']['math_roles'].update(backend_executable_not_established=False)]:
   c=public_request(self.q);change(c);rebind(c)
   with self.assertRaises(ValueError):verify_context(c)
 def test_old_role_reader_mode_rejects_optin_new_format(self):
  c=public_request(self.q)
  with self.assertRaises(ValueError):verify_projected(c)
  verify_projected(c,tensor_operands=True)
 def test_default_and_existing_v1_v2_v3_stay_distinct(self):
  for policy in ['descriptors-v1','descriptors-math-roles-v2','descriptors-compact-origins-v3']:
   q=prepare_task(self.q['model'],dict(chunk_period=8,generation_guidance='explicit-v1',public_weight_policy=policy,compiler_configuration='poseidon-cpu-eva-w45-current-v1',construction_profile='hecate-unified-public-v1'))
   self.assertEqual(q['public_weight_binding']['policy'],policy);self.assertNotEqual(q['request_id'],self.q['request_id']);self.assertNotIn('required_tensor_operand_roles_complete',q['public_weight_binding'].get('math_roles',{}));verify_context(public_request(q))
 def test_scalar_polynomial_roles_and_large_linear_weights_remain(self):
  b=Builder('mixed_math_roles',[[2]]);x=b.node('multiply',['input0',b.constant([.5,1.])]);x=b.node('add',[x,b.constant(.25)]);m=b.finish(b.node('polynomial',[x,b.constant([1.,.5,.25])],basis='power'))
  q=request(m);self.assertEqual({r['kind']for r in q['public_weight_binding']['math_roles']['constants']},{'tensor_operand','scalar_operand','polynomial_coefficients'})
  from public_weight_cases import models
  m=models()[-1];q=request(m,period=128);roles=q['public_weight_binding']['math_roles'];self.assertTrue(any(r['descriptor']['elements']==10000 for r in roles['opaque_constants']));self.assertLessEqual(len(canonical(public_request(q))),131072)
 def test_transport_dry_run_has_no_exercise_or_dsl_answer(self):
  c=public_request(self.q);self.assertNotIn('construction_exercise',c);self.assertNotIn('upstream_exercise',c);self.assertNotIn('hecate_source',canonical(c['public_weight_binding']['math_roles']).decode());self.assertNotIn('def golden',canonical(c['public_weight_binding']['math_roles']).decode());cfg=Config(model='deepseek-flash',reasoning_effort='low',max_tokens=65536,max_calls=1,timeout_seconds=1200,stream=True)
  body=canonical(dict(cfg.settings(),messages=[dict(role='system',content=SYSTEM),dict(role='user',content=canonical(c).decode())]));transport=HTTPSTransport(api_key='synthetic-only',approved_request_id=c['request_id'],enabled=True,approved_config=cfg);transport.validate(body,cfg.timeout_seconds)
 def test_complete_attention_four_original_rope_vectors_become_public(self):
  old=json.loads((PRE/'request.json').read_text());opts=json.loads((PRE/'options.json').read_text());opts['public_weight_policy']=POLICY;q=prepare_task(old['model'],opts);c=public_request(q);self.assertEqual(q['model'],old['model']);self.assertEqual(q['layout'],old['layout']);self.assertEqual(q['public_constants'],old['public_constants']);self.assertEqual(q['constant_origins'],old['constant_origins']);self.assertNotEqual(q['request_id'],old['request_id']);self.assertLessEqual(len(canonical(c)),131072)
  roles=c['public_weight_binding']['math_roles'];self.assertLessEqual(roles['math_elements'],256)
  for name in ['generated_34','generated_37','generated_53','generated_56']:
   row=next(r for r in roles['constants']if r['logical_name']==name);self.assertEqual(row['kind'],'tensor_operand');self.assertEqual([e['value']for e in row['elements']],old['model']['constants'][name]);self.assertEqual(row['descriptor']['shape'],[2]);self.assertTrue(all(e['plain_names']for e in row['elements']))
  self.assertEqual(old,json.loads((PRE/'request.json').read_text()));verify_context(c)
if __name__=='__main__':unittest.main()
