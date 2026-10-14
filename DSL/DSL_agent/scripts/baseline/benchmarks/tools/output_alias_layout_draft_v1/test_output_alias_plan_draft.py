import copy,random,unittest
from output_alias_plan_draft import build,check,digest
BIND="0"*64
class OutputAliasPlan(unittest.TestCase):
 def make(self,outputs=None,shapes=None,**kw):
  defaults=dict(period=4,logical_element_limit=64,physical_ciphertext_limit=16,rank_limit=32,model_binding=BIND);defaults.update(kw)
  return build(outputs or[dict(name="a",value="x"),dict(name="b",value="x")],shapes or{"x":[4]},**defaults)
 def test_shared_ref_has_one_physical_output(self):
  p=self.make();self.assertEqual(p["physical_ciphertexts"],1);self.assertEqual(p["logical_outputs"][1]["alias_of"],"a")
  self.assertEqual(p["selectors"],[[0,i]for i in range(4)]*2)
 def test_different_refs_do_not_alias_equal_shapes(self):
  p=self.make([dict(name="a",value="x"),dict(name="b",value="y")],{"x":[4],"y":[4]})
  self.assertEqual(p["physical_ciphertexts"],2);self.assertIsNone(p["logical_outputs"][1]["alias_of"])
 def test_scalar_alias(self):
  p=self.make(shapes={"x":[]});self.assertEqual(p["physical_ciphertexts"],1);self.assertEqual(p["selectors"],[[0,0],[0,0]])
 def test_tail_alias_and_offsets(self):
  p=self.make(shapes={"x":[5]});self.assertEqual(p["physical_ciphertexts"],2)
  self.assertEqual(p["selectors"],[[0,0],[0,1],[0,2],[0,3],[1,0]]*2)
 def test_interleaved_sources_preserve_order(self):
  p=self.make([dict(name="a",value="x"),dict(name="b",value="y"),dict(name="c",value="x")],{"x":[3],"y":[2]})
  self.assertEqual(p["logical_outputs"][2]["alias_of"],"a");self.assertEqual(p["selectors"],[[0,0],[0,1],[0,2],[1,0],[1,1],[0,0],[0,1],[0,2]])
 def test_duplicate_names_rejected(self):
  with self.assertRaises(ValueError):self.make([dict(name="a",value="x"),dict(name="a",value="x")])
 def test_unknown_source_rejected(self):
  with self.assertRaises(ValueError):self.make([dict(name="a",value="unknown")])
 def test_logical_limit_does_not_disappear_with_alias(self):
  with self.assertRaisesRegex(ValueError,"logical_elements"):self.make(logical_element_limit=7)
 def test_physical_limit_applies_to_distinct_sources(self):
  with self.assertRaisesRegex(ValueError,"physical_ciphertexts"):self.make([dict(name="a",value="x"),dict(name="b",value="y")],{"x":[5],"y":[5]},physical_ciphertext_limit=3)
  self.assertEqual(self.make(shapes={"x":[5]},physical_ciphertext_limit=2)["physical_ciphertexts"],2)
 def test_bad_shape_and_enormous_shape_rejected_before_selectors(self):
  for shape in([0],[-1],[True],[],[2**40]):
   if not shape:continue
   with self.subTest(shape=shape),self.assertRaises(ValueError):self.make(shapes={"x":shape})
 def test_rank32_boundary(self):
  self.assertEqual(self.make(shapes={"x":[1]*32})["physical_ciphertexts"],1)
  with self.assertRaises(ValueError):self.make(shapes={"x":[1]*33})
 def test_period_and_limit_types(self):
  for kw in (dict(period=True),dict(period=0),dict(period=4097),dict(rank_limit=True),dict(logical_element_limit=0),dict(physical_ciphertext_limit=-1)):
   with self.subTest(kw=kw),self.assertRaises(ValueError):self.make(**kw)
 def test_model_binding_required(self):
  for value in ("","A"*64,"g"*64,"0"*63):
   with self.subTest(value=value),self.assertRaises(ValueError):self.make(model_binding=value)
 def test_rehashed_forged_selector_rejected(self):
  p=self.make();p["selectors"][0]=[0,3];p["binding"]=digest({k:v for k,v in p.items()if k!="binding"})
  with self.assertRaisesRegex(ValueError,"alias_plan_changed"):check(p,[dict(name="a",value="x"),dict(name="b",value="x")],{"x":[4]},period=4,logical_element_limit=64,physical_ciphertext_limit=16,rank_limit=32,model_binding=BIND)
 def test_changed_source_rejected(self):
  p=self.make()
  with self.assertRaises(ValueError):check(p,[dict(name="a",value="x"),dict(name="b",value="y")],{"x":[4],"y":[4]},period=4,logical_element_limit=64,physical_ciphertext_limit=16,rank_limit=32,model_binding=BIND)
 def test_changed_shape_rejected(self):
  p=self.make()
  with self.assertRaises(ValueError):check(p,[dict(name="a",value="x"),dict(name="b",value="x")],{"x":[3]},period=4,logical_element_limit=64,physical_ciphertext_limit=16,rank_limit=32,model_binding=BIND)
 def test_source_input_not_mutated(self):
  outputs=[dict(name="a",value="x"),dict(name="b",value="x")];shapes={"x":[2,2]};before=copy.deepcopy((outputs,shapes))
  self.make(outputs,shapes);self.assertEqual((outputs,shapes),before)
 def test_independent_selectors_match_256_contexts(self):
  rng=random.Random(20261012)
  for index in range(256):
   shapes={f"v{i}":[rng.randint(1,7)] for i in range(rng.randint(1,4))}
   refs=list(shapes);outputs=[dict(name=f"o{i}",value=rng.choice(refs))for i in range(rng.randint(1,4))]
   values={ref:[rng.uniform(-1.,1.)for _ in shape and range(shape[0])]for ref,shape in shapes.items()}
   p=self.make(outputs,shapes,period=rng.choice((4,8,16)))
   physical=[]
   for item in p["physical_outputs"]:
    data=values[item["source_value"]][item["offset"]:item["offset"]+item["elements"]]
    physical.append(data+[0.]*(p["period"]-len(data)))
   decoded=[physical[c][slot]for c,slot in p["selectors"]]
   expected=[value for out in outputs for value in values[out["value"]]]
   self.assertEqual(decoded,expected,(index,outputs,shapes))
if __name__=="__main__":unittest.main(verbosity=2)
