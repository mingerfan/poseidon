"""Read-only diagnostic of this candidate's admitted period-4 slot arithmetic."""
import ast,hashlib,json,math
from pathlib import Path
BASE=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux/results/candidate-replay-5z8pvlvi")
OUT=Path(__file__).with_name("slot-probe.json")
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def vector(value):
 if type(value) in (int,float):
  assert math.isfinite(value);return [float(value)]*4
 assert type(value) is list and len(value)==4 and all(type(x) in (int,float) and math.isfinite(x) for x in value)
 return [float(x) for x in value]
def expression(node,env):
 if isinstance(node,ast.Name):return vector(env[node.id])
 if isinstance(node,ast.Constant):return vector(node.value)
 if isinstance(node,ast.BinOp):
  left=expression(node.left,env);right=expression(node.right,env)
  op=type(node.op)
  if op is ast.Add:return [x+y for x,y in zip(left,right)]
  if op is ast.Sub:return [x-y for x,y in zip(left,right)]
  if op is ast.Mult:return [x*y for x,y in zip(left,right)]
  raise ValueError("unsupported_binary")
 if isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.UAdd,ast.USub)):
  value=expression(node.operand,env)
  return value if isinstance(node.op,ast.UAdd) else [-x for x in value]
 if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=="rotate":
  assert len(node.args)==1 and not node.keywords
  arg=node.args[0];assert isinstance(arg,ast.Constant) and type(arg.value) is int
  value=expression(node.func.value,env);return [value[(i+arg.value)%4] for i in range(4)]
 raise ValueError("unsupported_expression")
def interpret(function,constants,inputs):
 assert len(function.args.args)==len(inputs) and not function.args.defaults
 env={n:vector(v) for n,v in constants.items()}
 env.update({n.arg:vector(v) for n,v in zip(function.args.args,inputs)})
 for statement in function.body:
  if isinstance(statement,ast.Assign):
   assert len(statement.targets)==1 and isinstance(statement.targets[0],ast.Name)
   env[statement.targets[0].id]=expression(statement.value,env)
  elif isinstance(statement,ast.Return):
   assert isinstance(statement.value,ast.Tuple)
   return [expression(n,env) for n in statement.value.elts],env
  else:raise ValueError("unsupported_statement")
 raise ValueError("no_return")
def metrics(actual,expected):
 assert len(actual)==len(expected)
 errors=[abs(x-y) for x,y in zip(actual,expected)]
 return dict(values=len(errors),max_absolute_error=max(errors),mae=sum(errors)/len(errors),
 violating_values=sum(e>1e-5+1e-4*abs(y) for e,y in zip(errors,expected)))
def controls():
 assert expression(ast.parse("x.rotate(1)",mode="eval").body,{"x":[1.,2.,3.,4.]})==[2.,3.,4.,1.]
 assert expression(ast.parse("x*2+1",mode="eval").body,{"x":[1.,2.,3.,4.]})==[3.,5.,7.,9.]
 for source in ("x.decrypt()","open('x')","x.__class__","x[0]"):
  try:expression(ast.parse(source,mode="eval").body,{"x":[1.,2.,3.,4.]})
  except (ValueError,AssertionError):continue
  raise AssertionError("unsupported expression accepted")
 return dict(passed=6,failed=0,skipped=0)
def main():
 p=BASE/"attempt-00/trace-payload.json";payload=json.loads(p.read_text())
 source=payload["candidate"]["hecate_source"]
 tree=ast.parse(source);assert len(tree.body)==1 and isinstance(tree.body[0],ast.FunctionDef)
 function=tree.body[0]
 q=BASE/"attempt-00/output/inputs.json";samples=json.loads(q.read_text())
 r=BASE/"attempt-00/report.json";comparison=json.loads(r.read_text())["comparison"]
 constants=payload["request"]["public_constants"];layout=payload["request"]["layout"]
 values=[];scores=[]
 for sample in samples:
  result,env=interpret(function,constants,sample)
  values.append([result[ct][slot] for ct,slot in layout["output_selectors"]])
  expected0=[env["P00"][0]+env["P00"][1],env["P01"][0]+env["P01"][1],
             env["P10"][0]+env["P10"][1],env["P11"][0]+env["P11"][1]]
  expected1=[env["P00"][2]+env["P00"][3],env["P01"][2]+env["P01"][3],
             env["P10"][2]+env["P10"][3],env["P11"][2]+env["P11"][3]]
  scores.extend([abs(x-y) for x,y in zip(env["G0"]+env["G1"],expected0+expected1)])
 flatten=lambda x:[v for row in x for v in row]
 value=dict(format="poseidon-candidate-slot-semantic-probe-v1",controls=controls(),input_groups=len(samples),
  candidate_source_sha256=hashlib.sha256(source.encode()).hexdigest(),request_id=payload["request"]["request_id"],
  evidence_sha256={str(x):sha(x) for x in (p,q,r)},public_period=4,
  candidate_plaintext_vs_model_reference=metrics(flatten(values),flatten(comparison["reference"])),
  fhe_vs_candidate_plaintext=metrics(flatten(comparison["actual"]),flatten(values)),
  paired_score_lane_deviation=dict(max_absolute_error=max(scores),nonzero_differences=sum(x>1e-12 for x in scores)),
  scope="diagnostic of frozen candidate only, not an Agent pass or model correctness acceptance",
  external_arrays_written=False,new_provider_calls=0,new_compiler_calls=0,new_fhe_calls=0)
 value["binding"]=hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
 with OUT.open("x") as stream:stream.write(json.dumps(value,sort_keys=True,indent=2)+"\n")
 print(json.dumps(value,ensure_ascii=False))
if __name__=="__main__":main()
