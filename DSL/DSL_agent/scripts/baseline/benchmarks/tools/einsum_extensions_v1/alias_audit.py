"""Check Torch storage sharing and both mutation directions, independently."""
import copy,hashlib,io,json,math,pathlib,sys,time,unittest
ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);sys.path.insert(0,str(ROOT/"scripts/baseline"))
from planner import plan
from alias_projection import projection
HERE=pathlib.Path(__file__).resolve().parent
CASES=[("diagonal","ii->i",[[3,3]],True),("batched_diagonal","...ii->...i",[[2,2,2]],True),
 ("permuted_diagonal","i...i->...i",[[2,3,2]],True),("two_repeated_axes","ijij->ji",[[2,3,2,3]],True),
 ("triple_diagonal","iii->i",[[2,2,2]],True),("plain_transpose","ij->ji",[[2,3]],True),
 ("scalar_identity","->",[[]],True),("trace","ii->",[[2,2]],False),
 ("implicit_trace","ii",[[2,2]],False),("elementwise_two_inputs","i,i->i",[[2],[2]],False),
 ("scalar_tensor_factor","i,->i",[[2],[]],False),("two_scalars",",->",[[],[]],False),
 ("contract","ij,jk->ik",[[2,2],[2,2]],False)]
class AliasTests(unittest.TestCase):
 def check_case(self,equation,shapes,expected_view):
  import torch
  values=[torch.arange(math.prod(s),dtype=torch.float64).reshape(s)+i/7 for i,s in enumerate(shapes)]
  record=plan(equation,shapes);meta=projection(record)
  actual=torch.einsum(equation,*values)
  self.assertEqual(meta["role"]=="view",expected_view)
  aliases=[actual.untyped_storage().data_ptr()==v.untyped_storage().data_ptr()for v in values]
  self.assertEqual(any(aliases),expected_view)
  self.assertFalse(meta["production_source_policy"])
  if expected_view:
   self.assertEqual(meta["source_operand"],0)
   flat=actual.flatten().tolist()
   self.assertEqual(flat,[float(values[0].reshape(-1)[i])for i in meta["source_flat_indices"]])
   first,last=meta["source_flat_indices"][0],meta["source_flat_indices"][-1]
   old=float(values[0].reshape(-1)[first]);actual[(0,)*actual.ndim]+=0.375
   self.assertEqual(float(values[0].reshape(-1)[first]),old+0.375)
   old=float(actual[tuple(n-1 for n in actual.shape)])
   values[0].reshape(-1)[last]+=0.125
   self.assertEqual(float(actual[tuple(n-1 for n in actual.shape)]),old+0.125)
  else:self.assertIsNone(meta["source_flat_indices"])
for name,eq,shapes,view in CASES:
 def test(self,eq=eq,s=shapes,v=view):self.check_case(eq,s,v)
 setattr(AliasTests,"test_"+name,test)
if __name__=="__main__":
 path=HERE/"alias-controls.json";log=HERE/"alias-controls.log"
 if path.exists()or log.exists():raise SystemExit("preserve alias evidence")
 buffer=io.StringIO();started=time.monotonic()
 result=unittest.TextTestRunner(stream=buffer,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(AliasTests))
 log.write_text(buffer.getvalue())
 record=dict(format="poseidon-einsum-alias-controls-v1",status="passed"if result.wasSuccessful()and not result.skipped else"failed",
  tests_run=result.testsRun,failed=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
  seconds=time.monotonic()-started,sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in[HERE/"planner.py",HERE/"alias_projection.py",pathlib.Path(__file__)]},
  provider_calls=0,compiler_calls=0,encrypted_calls=0,production_source_policy=False)
 path.write_text(json.dumps(record,sort_keys=True,indent=2)+"\n")
 print(json.dumps(record,sort_keys=True))
 if not result.wasSuccessful():print(buffer.getvalue()[-7000:])
 raise SystemExit(0 if record["status"]=="passed"else 1)
