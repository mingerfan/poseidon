"""A packed Plain value cannot create public scalar-literal authority."""
import copy,importlib.util,json,sys,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parents[1];BASE=HERE.parents[2]
sys.path.insert(0,str(BASE))
from unified_request_storage import restore_public_constants
spec=importlib.util.spec_from_file_location("rule_authority",HERE/"replicated_scalar_lowering.py")
rule=importlib.util.module_from_spec(spec);spec.loader.exec_module(rule)
class LiteralVectorAuthority(unittest.TestCase):
 def test_same_value_inside_plain_vector_is_not_scalar_authority(self):
  p=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-mha-ordered-backend-v1/self_post_linear_residual__declared_shallow_v1/request.json")
  original=json.loads(p.read_bytes());q=copy.deepcopy(original);restore_public_constants(q)
  value=.12345678912345678
  self.assertNotIn(value.hex(),{float(x).hex() for x in q["public_constants"].values() if type(x) in (int,float)})
  q["public_constants"]["pv_control_only"]=[value,value,value,value]
  before=copy.deepcopy(q)
  emitter=rule.ReplicatedEmitter(q,{})
  with self.assertRaisesRegex(ValueError,"original public provenance"):emitter.literal(value)
  self.assertEqual(q,before)
  self.assertNotIn("pv_control_only",original["public_constants"])
