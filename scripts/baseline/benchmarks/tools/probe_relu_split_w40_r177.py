"""One exact-model balanced-tree probe, without compiler or parameter changes."""
import json,sys,shlex,hashlib
from pathlib import Path
BASE=Path(__file__).resolve().parents[2];ROOT=BASE.parents[1];sys.path.insert(0,str(BASE))
def main():
 from hecate_python_env import enter_nix,VENV
 if "--inside" not in sys.argv:
  return enter_nix('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"]),seconds=240)
 from workspace_paths import RESULTS
 from semantic_benchmark_execution import runtime_sources
 from unified_graph_lowering import lower
 from unified_graph_contract import validate_candidate
 from component_backend import qualify
 from benchmark_graph import digest
 out=RESULTS/"stage2-noncompiler-r177/relu-split-w40";out.mkdir(parents=True)
 index=json.loads((ROOT/"docs/baseline/compiler-blocked-models-r159/index.json").read_text())
 model=next(r for r in index["rows"] if r["model_id"]=="bench_helper_0105")
 shard=json.loads(Path(model["origin_shard"]).read_text())
 task=next(r for r in shard["cases"] if r["id"]==model["task_id"])
 q=task["request"];row={"id":task["id"]}
 old_request_id=q["request_id"]
 from compiler_configuration import configuration
 from copy import deepcopy
 q=deepcopy(q)
 q["compiler_configuration"]=configuration("seal-cpu-eva-w40-v1")
 q["request_id"]=digest({k:v for k,v in q.items() if k!="request_id"})
 before=json.loads((RESULTS/"stage2-remaining-r174/before.json").read_text())
 sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
 assert all(sha(p)==h for p,h in before["compiler"].items())
 sources=runtime_sources();value=dict(id=row["id"],paid_calls=0,new_agent_success=False,model=q["model"],
  original_request_id=old_request_id,new_request_id=q["request_id"],configuration_change="existing registered w45 to w40, unchanged security profile",source_hashes=sources,compiler_changed=False)
 try:
  # Diagnostic only: lane-local original graph, no extraction/repacking overhead.
  # Every Chebyshev coefficient is retained; no shape/layout/config changes.
  from unified_graph_lowering import Emitter,Scalar
  def balanced(x,coeff,zero):
   # Chebyshev divide-and-conquer identity:
   # T_(m+k)=2*T_m*T_k-T_(m-k), with m a power of two.
   ts={1:x}
   def term(m):
    if m not in ts:
     half=term(m//2);v=half*half;ts[m]=v+v-1.
    return ts[m]
   def evaluate(c):
    while len(c)>1 and c[-1]==0.:c=c[:-1]
    if len(c)==1:return zero+c[0]
    if len(c)==2:return x*c[1]+c[0]
    m=1<<((len(c)-1).bit_length()-1)
    low=list(c[:m]);high=[c[m]]+[2.*v for v in c[m+1:]]
    for k in range(1,len(c)-m):low[m-k]-=c[m+k]
    return evaluate(low)+term(m)*evaluate(high)
   return evaluate([float(v) for v in coeff])
  original_literal=Emitter.literal
  def derived_literal(self,x):
   # Public coefficient algebra only, within the existing literal bound.
   import math
   if float(x) in self.constants:return original_literal(self,x)
   assert math.isfinite(x) and abs(x)<=1024
   return repr(float(x))
  Emitter.literal=derived_literal
  assert q["layout"]["execution_abi"]=="unified-periodic-inputs-v1"
  assert len(q["layout"]["inputs"])==1 and len(q["model"]["outputs"])==1
  em=Emitter(q);values=dict(q["model"]["constants"])
  binding=q["layout"]["inputs"][0]
  values[binding["name"]]=Scalar(em,binding["dsl_name"])
  for node in q["model"]["nodes"]:
   args=[values[k] for k in node["inputs"]]
   assert len(node["outputs"])==1
   if node["op"]=="polynomial":
    assert node["attrs"]=={"basis":"chebyshev"}
    answer=balanced(args[0],args[1],em.zero)
   elif node["op"]=="add":answer=args[0]+args[1]
   elif node["op"]=="multiply":answer=args[0]*args[1]
   else:raise ValueError("Unsupported diagnostic lane-local graph")
   values[node["outputs"][0]]=answer
  answer=values[q["model"]["outputs"][0]["value"]]
  source=chr(10).join(['@hc.func("c,c")', "def golden("+binding["dsl_name"]+", zero_ct):", *em.lines, "    return ["+answer.name+"]", ""])
  value["strategy"]="divide-and-conquer Chebyshev per packed lane; exact original coefficients"

  candidate=dict(schema=1,request_id=q["request_id"],hecate_source=source)
  (out/"job.json").write_text(json.dumps(dict(request=q,candidate=candidate,manual_fixture=True),indent=2))
  validate_candidate(candidate,q)
  value["result"]=qualify(q,candidate)
 except (ValueError,TypeError) as exc:value.update(status="preflight_failed",diagnostic=str(exc))
 assert sources==runtime_sources()
 assert all(sha(p)==h for p,h in before["compiler"].items())
 value["binding"]=digest(value)
 (out/"report.json").write_text(json.dumps(value,indent=2))
 print(json.dumps({k:v for k,v in value.items() if k not in ("model","source_hashes")}))
 return 0
if __name__=="__main__":raise SystemExit(main())
