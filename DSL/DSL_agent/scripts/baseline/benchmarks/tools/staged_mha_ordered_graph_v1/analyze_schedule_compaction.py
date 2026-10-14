"""Analyse existing trusted scheduling/compaction; no backend or Provider calls."""
import ast,hashlib,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[2]
sys.path.insert(0,str(BASE));sys.path.insert(0,str(HERE/"overlay"))
import numpy as np
from benchmark_graph import canonical,digest,require
from capacity_algebraic_schedule import schedule
from unified_capacity_lowering import compact_scalar_source
from unified_graph_contract import validate_candidate
from hecate_python_env import WORK
from pytorch_ordered_contract import source_identity
NAMES=["self_post_linear_residual__declared_shallow_v1","linear_into_separate_cross_none__declared_shallow_v1","boolean_padding_residual__declared_shallow_v1"]
def interpret(source,values):
 tree=ast.parse(source);functions={f.name:f for f in tree.body}
 def call(fn,args):
  env=dict(values);env.update(zip((a.arg for a in fn.args.args),args))
  def expr(n):
   if type(n) is ast.Name:return env[n.id]
   if type(n) is ast.Constant and type(n.value) in (int,float):return n.value
   if type(n) is ast.UnaryOp and type(n.op) is ast.USub:return -expr(n.operand)
   if type(n) is ast.BinOp:
    a,b=expr(n.left),expr(n.right)
    if type(n.op) is ast.Add:return a+b
    if type(n.op) is ast.Sub:return a-b
    if type(n.op) is ast.Mult:return a*b
   if type(n) in (ast.List,ast.Tuple):return [expr(a) for a in n.elts]
   if type(n) is ast.Call:
    if type(n.func) is ast.Name and n.func.id in functions:return call(functions[n.func.id],[expr(a) for a in n.args])
    if type(n.func) is ast.Attribute and n.func.attr=="rotate":
     require(len(n.args)==1 and not n.keywords,"Control rotate");return np.roll(expr(n.func.value),-int(expr(n.args[0])))
   raise ValueError("Unregistered control AST:"+type(n).__name__)
  for s in fn.body:
   if type(s) is ast.Assign and len(s.targets)==1 and type(s.targets[0]) is ast.Name:env[s.targets[0].id]=expr(s.value)
   elif type(s) is ast.Return:return expr(s.value)
   else:raise ValueError("Unregistered control statement:"+type(s).__name__)
 return call(functions["golden"],[values[a.arg] for a in functions["golden"].args.args])
def main():
 out=Path(sys.argv[1]);require(not out.exists(),"Preserve analysis evidence");out.mkdir()
 original=source_identity()
 report=dict(format="poseidon-mha-trusted-schedule-compaction-analysis-v1",cases=[],
 source_implementation=original,tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 compiler_calls=0,encrypted_execution=False,api_calls=0,production_applied=False,
 existing_compaction_only=True,provider_candidate_rewritten=False,reference_changed=False)
 for name in NAMES:
  old=WORK/"results/agent-mha-ordered-backend-v1"/name
  source=(old/"candidate.py").read_text();request=json.loads((old/"request.json").read_bytes())
  row=dict(id=name,request_id=request["request_id"],original_candidate_sha256=hashlib.sha256(source.encode()).hexdigest())
  try:
   planned,record=schedule(source,request);compacted=compact_scalar_source(planned)
   row.update(schedule=record,original_ast_nodes=sum(1 for _ in ast.walk(ast.parse(source))),
    scheduled_ast_nodes=sum(1 for _ in ast.walk(ast.parse(planned))),
    compacted_ast_nodes=sum(1 for _ in ast.walk(ast.parse(compacted))))
   case=out/name;case.mkdir();(case/"candidate.py").write_text(compacted)
   (case/"request.json").write_bytes((old/"request.json").read_bytes())
   row["compacted_candidate_sha256"]=hashlib.sha256(compacted.encode()).hexdigest()
   try:
    validate_candidate(dict(schema=1,request_id=request["request_id"],hecate_source=compacted),request)
    row["static_status"]="passed"
   except ValueError as error:row.update(static_status="failed",diagnostic=str(error))
   rng=np.random.default_rng(713);maximum=0.;count=0
   for group in range(16):
    values={k:np.asarray(v,dtype=np.float64) for k,v in request["public_constants"].items()}
    values.update({i["dsl_name"]:rng.uniform(-.25,.25,size=request["layout"]["input_slot_period"]) for i in request["layout"]["inputs"]})
    values["zero_ct"]=np.zeros(request["layout"]["input_slot_period"],dtype=np.float64)
    expected=interpret(source,values);actual=interpret(compacted,values)
    require(len(actual)==len(expected),"Control output ports")
    for a,b in zip(actual,expected):
     err=np.abs(np.asarray(a)-np.asarray(b));maximum=max(maximum,float(np.max(err)))
     require(np.all(err<=1e-12+1e-12*np.abs(b)),"Compaction control values");count+=np.asarray(a).size
   row.update(program_semantics_control="passed",compared_values=count,maximum_absolute_error=maximum)
  except Exception as error:row.update(static_status="failed",diagnostic=type(error).__name__+":"+str(error))
  report["cases"].append(row);print(name,json.dumps(row),flush=True)
 require(source_identity()==original,"Source changed during analysis")
 report["binding"]=digest(report);(out/"report.json").write_bytes(canonical(report))
 return 0
if __name__=="__main__":raise SystemExit(main())
