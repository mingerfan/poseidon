import copy,json,math,unittest
from codec import encode,decode,raw,sha,FORMAT,WIRE_BYTES,LOGICAL_BYTES
def rebound(p):
 p=copy.deepcopy(p);p.pop("binding",None);p["binding"]=sha(p,sorted_keys=True);return p
class CodecTests(unittest.TestCase):
 def test_exact_types_and_signed_zero(self):
  v={"integer":1,"real":1.0,"yes":True,"nothing":None,"negative_zero":-0.0,"text":"中文\\n"}
  restored=decode(encode(v));self.assertEqual(raw(v),raw(restored))
  self.assertIs(type(restored["integer"]),int);self.assertIs(type(restored["real"]),float)
  self.assertIs(type(restored["yes"]),bool);self.assertEqual(math.copysign(1,restored["negative_zero"]),-1)
 def test_empty_containers(self):
  v={"empty_dict":{},"empty_list":[]};self.assertEqual(decode(encode(v)),v)
 def test_identical_payloads_share_storage_not_runtime_aliases(self):
  rows=[[.125*i+.0625*j for j in range(16)]for i in range(16)]
  v={"a":rows,"b":copy.deepcopy(rows),"c":{"again":copy.deepcopy(rows)}}
  p=encode(v);new=decode(p);self.assertLess(len(raw(p)),len(raw(v)))
  self.assertIsNot(new["a"],new["b"]);self.assertIsNot(new["a"][0],new["b"][0])
  old=new["b"][0][0];new["a"][0][0]=123.;self.assertEqual(new["b"][0][0],old)
 def test_key_order_preserved(self):
  v={"weights":{"z":[2.,3.],"a":[5.,7.]},"source":"module"}
  self.assertEqual(list(decode(encode(v))["weights"]),["z","a"])
 def test_equal_maps_with_distinct_order_not_coalesced(self):
  v={"a":{"x":1,"y":2},"b":{"y":2,"x":1}}
  p=encode(v);r=decode(p);self.assertEqual(raw(v),raw(r))
  self.assertEqual(list(r["a"]),["x","y"]);self.assertEqual(list(r["b"]),["y","x"])
 def test_payload_hash_tamper(self):
  p=encode({"x":[1.,2.]});p["nodes"][0][1][0]=3.
  with self.assertRaisesRegex(ValueError,"packet_binding"):decode(p)
 def test_rebound_payload_still_needs_original_hash(self):
  p=encode({"x":[1.,2.]});p["nodes"][0][1][0]=3.;p=rebound(p)
  with self.assertRaisesRegex(ValueError,"logical_content_binding"):decode(p)
 def test_size_claim_tamper(self):
  p=encode({"x":[1.]});p["logical_bytes"]+=1;p=rebound(p)
  with self.assertRaisesRegex(ValueError,"logical_size_claim"):decode(p)
 def test_self_reference_rejected(self):
  p=encode({"x":[1.]});p["nodes"][0][1]=[["ref",0]];p=rebound(p)
  with self.assertRaisesRegex(ValueError,"provenance_reference"):decode(p)
 def test_reference_boolean_rejected(self):
  p=encode({"x":[1.]});p["root"]=["ref",True];p=rebound(p)
  with self.assertRaisesRegex(ValueError,"provenance_root"):decode(p)
 def test_unreachable_node_rejected(self):
  p=encode({"x":[1.]});p["nodes"].append(["list",[2.]]);p=rebound(p)
  with self.assertRaisesRegex(ValueError,"unreachable"):decode(p)
 def test_noncanonical_duplicate_rejected(self):
  p=encode({"x":[1.]});p["nodes"].append(copy.deepcopy(p["nodes"][0]));p=rebound(p)
  with self.assertRaisesRegex(ValueError,"noncanonical_duplicate"):decode(p)
 def test_duplicate_keys_rejected(self):
  p=encode({"x":[1.]});p["nodes"][-1][1].append(copy.deepcopy(p["nodes"][-1][1][0]));p=rebound(p)
  with self.assertRaisesRegex(ValueError,"provenance_key"):decode(p)
 def test_expansion_bomb_rejected_before_reconstruction(self):
  nodes=[["list",[0.0]]]
  for i in range(1,24):nodes.append(["list",[["ref",i-1],["ref",i-1]]])
  nodes.append(["dict",[["payload",["ref",23]]]])
  p=rebound(dict(format=FORMAT,root=["ref",24],nodes=nodes,logical_bytes=1000000,
   canonical_sha256="0"*64,ordered_sha256="0"*64,production_active=False))
  with self.assertRaisesRegex(ValueError,"expanded_provenance_budget"):decode(p,logical_limit=512)
 def test_explicit_budget_rejected(self):
  for limit in(0,-1,True,LOGICAL_BYTES+1,"16MB"):
   with self.subTest(limit=limit),self.assertRaises(ValueError):encode({},logical_limit=limit)
 def test_logical_budget_respected_even_when_wire_tiny(self):
  v={"data":[[1.,2.]]*500}
  p=encode(v)
  with self.assertRaisesRegex(ValueError,"expanded_provenance_budget"):decode(p,logical_limit=128)
 def test_source_cycles_rejected(self):
  v={};v["again"]=v
  with self.assertRaises(ValueError):encode(v)
 def test_source_depth_bound(self):
  v=[]
  for _ in range(65):v=[v]
  with self.assertRaisesRegex(ValueError,"provenance_depth"):encode({"x":v})
 def test_nonfinite_and_non_json_rejected(self):
  for v in(float("nan"),float("inf"),set((1,)),(1,2),1<<5000):
   with self.subTest(kind=type(v).__name__),self.assertRaises(ValueError):encode({"x":v})
 def test_oversized_atom(self):
  with self.assertRaisesRegex(ValueError,"atom_budget"):encode({"s":"x"*(WIRE_BYTES+1)})
 def test_unknown_packet_field(self):
  p=encode({"x":[1.]});p["external_path"]="/unexpected";p=rebound(p)
  with self.assertRaisesRegex(ValueError,"provenance_packet"):decode(p)
 def test_deterministic_packet(self):
  v={"x":[[1.,2.],[1.,2.]],"z":{"flag":False}}
  self.assertEqual(encode(v),encode(copy.deepcopy(v)))
if __name__=="__main__":unittest.main()
