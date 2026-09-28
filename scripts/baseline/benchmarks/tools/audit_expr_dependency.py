"""Actual pinned Expr protocols with independent FHE numerical witnesses; no candidate privilege expansion."""
import argparse,hashlib,json,os,shlex,sys,time,tempfile
from pathlib import Path
BASE=Path(__file__).resolve().parents[2];ROOT=BASE.parents[1]
sys.path.insert(0,str(BASE))
from benchmark_graph import digest
from benchmark_runner import strict_file,dump
WORKER=r'''
import hashlib,json,sys
from pathlib import Path
import numpy as np,torch
from candidate_trace import load_frontend
hc=load_frontend()
spec=json.loads(Path("/payload.json").read_text())["manual_spec"]
if spec["context"] not in (0,1,2) or spec["regime"] not in ("normal_public","reverse_public","augmented_public","cipher_pair","negate_rotate","public_conversions","empty_protocol","empty_operators"):raise ValueError("Trusted Expr case")
events=[];checks=[]
def observe(frame,event,arg):
 if frame.f_code.co_filename==hc.__file__ and event=="return":
  name=frame.f_code.co_name
  events.append(dict(symbol=name,receiver=type(frame.f_locals.get("self")).__name__,opcode=frame.f_locals.get("opcode"),return_handle=int(arg.obj) if isinstance(arg,hc.Expr) else None))
@hc.func("c,c")
def golden(x,zero_ct):
 if spec["context"]==0:x=x*.125
 elif spec["context"]==1:x=x*.125+.0625
 else:x=(x*x)*.125
 old=int(x.obj);regime=spec["regime"]
 sys.setprofile(observe)
 try:
  if regime=="normal_public":result=((x+.125)-.25)*.5
  elif regime=="reverse_public":result=.5*(.25-(.125+x))
  elif regime=="augmented_public":
   result=x;result+=.125;result-=.25;result*=.5
   if int(x.obj)!=old or result is x:raise ValueError("Augmented alias changed")
   checks.append("augmented_rebind_preserves_alias")
  elif regime=="cipher_pair":
   y=x.rotate(1);result=(x+y)*(x-y)
  elif regime=="negate_rotate":result=(-x).rotate(2)
  elif regime=="public_conversions":
   vectors=[[((i+k)%7-3)/32 for i in range(16)] for k in range(4)]
   values=[3,.25,vectors[0],np.array(vectors[1]),torch.tensor(vectors[2],dtype=torch.float64)]
   owned=np.array(vectors[3]);plain=hc.Plain(owned);owned[:]=99.
   result=x
   for value in values:result=result+hc.resolveType(value)
   result=result+plain
   if hc.resolveType(x) is not x:raise ValueError("Expr identity")
   checks.append("constant_ownership_and_Expr_identity")
  elif regime=="empty_protocol":
   values=[]
   for name in ("__add__","__radd__","__iadd__","__sub__","__rsub__","__isub__"):
    value=getattr(hc.Empty(),name)(x)
    if value is not x:raise ValueError("Empty protocol identity")
    values.append(value)
   result=values[0]
   for value in values[1:]:result=result+value
   result=result*.125;checks.append("six_direct_protocol_returns_contribute")
  else:
   a=hc.Empty()+x;b=hc.Empty()-x;c=hc.Empty();c+=x;d=hc.Empty();d-=x
   result=(a+b+c+d)*.125
   checks.append("public_empty_left_and_augmented")
   # Expr consumes unsupported Empty in resolveType; reverse Python dispatch
   # is NOT inferred from the successful direct protocol checks.
   for name,fn in (("Expr+Empty",lambda:x+hc.Empty()),("Expr-Empty",lambda:x-hc.Empty())):
    try:fn()
    except Exception as error:checks.append(dict(rejected=name,type=type(error).__name__))
    else:raise ValueError("Unexpected right Empty dispatch")
 finally:sys.setprofile(None)
 checks.append(dict(golden_return_handle=int(result.obj)))
 return [result]
hc.save("/out","/out")
Path("/out/dependency-calls.json").write_text(json.dumps(dict(events=events,checks=checks,
 actual_upstream=True,helper_replaced=False,
 source_sha256=hashlib.sha256(Path(hc.__file__).read_bytes()).hexdigest())))
'''
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def cases():
 from benchmark_suite import Builder
 result=[]
 for regime in ("normal_public","reverse_public","augmented_public","cipher_pair","negate_rotate","public_conversions","empty_protocol","empty_operators"):
  for context in range(3):
   b=Builder([(16,)]);x="input0"
   if context==0:x=b.node("multiply",[x,b.const(.125)])
   elif context==1:x=b.node("add",[b.node("multiply",[x,b.const(.125)]),b.const(.0625)])
   else:x=b.node("multiply",[b.node("square",[x]),b.const(.125)])
   if regime in ("normal_public","augmented_public"):
    y=b.node("multiply",[b.node("subtract",[b.node("add",[x,b.const(.125)]),b.const(.25)]),b.const(.5)])
   elif regime=="reverse_public":
    # JSON keeps encrypted operand first; .25-(.125+x) = -x+.125.
    y=b.node("multiply",[b.node("add",[b.node("negate",[x]),b.const(.125)]),b.const(.5)])
   elif regime=="cipher_pair":
    z=b.node("rotate",[x],step=1)
    y=b.node("multiply",[b.node("add",[x,z]),b.node("subtract",[x,z])])
   elif regime=="negate_rotate":y=b.node("rotate",[b.node("negate",[x])],step=2)
   elif regime=="public_conversions":
    y=x
    for value in [3,.25]+[[((i+k)%7-3)/32 for i in range(16)] for k in range(4)]:
     y=b.node("add",[y,b.const(value)])
   else:
    y=x
    for i in range((6 if regime=="empty_protocol" else 4)-1):y=b.node("add",[y,x])
    y=b.node("multiply",[y,b.const(.125)])
   model=b.finish(y);model["id"]="expr_dependency_"+regime+"_"+str(context)
   result.append(dict(id=model["id"],regime=regime,context=context,model=model,model_sha256=digest(model)))
 return result

def independent_values(spec):
 from benchmark_graph import samples
 refs=[]
 for probe in samples(spec["model"],4):
  x=probe["input0"].reshape(-1).tolist()
  x=[v*.125+.0625 if spec["context"]==1 else v*v*.125 if spec["context"]==2 else v*.125 for v in x]
  regime=spec["regime"]
  if regime in ("normal_public","augmented_public"):y=[(v-.125)*.5 for v in x]
  elif regime=="reverse_public":y=[(.125-v)*.5 for v in x]
  elif regime=="cipher_pair":y=[x[i]**2-x[(i+1)%16]**2 for i in range(16)]
  elif regime=="negate_rotate":y=[-x[(i+2)%16] for i in range(16)]
  elif regime=="public_conversions":y=[v+3.25+sum(((i+k)%7-3)/32 for k in range(4)) for i,v in enumerate(x)]
  else:y=[v*(.75 if regime=="empty_protocol" else .5) for v in x]
  refs.append(y)
 # Removing any of the identical Empty operands changes nonzero probes.
 return refs,dict(helper_identity_intervention_delta=max(abs(v-x) for row,probe in zip(refs,samples(spec["model"],4))
       for v,x in zip(row,probe["input0"].reshape(-1))),
   scope="Bounded actual Expr protocols and Python dispatch; no arbitrary direct candidate method access")

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True)
 p.add_argument("--execute",action="store_true");p.add_argument("--inside",action="store_true",help=argparse.SUPPRESS);a=p.parse_args()
 compile(WORKER,"<expr_protocol-worker>","exec")
 from workspace_paths import RESULTS
 from hecate_python_env import enter_nix,VENV,WORK
 chosen=cases()
 if not a.execute:print(json.dumps(dict(cases=[r["id"] for r in chosen],max_wall_seconds=900,paid_calls=0)));return 0
 if a.output.exists() or not a.output.resolve().is_relative_to(RESULTS.resolve()):p.error("New platform result directory")
 if not a.inside:
  return enter_nix('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),*sys.argv[1:],"--inside"]),seconds=960)
 if os.environ.get("IN_NIX_SHELL")!="pure" or Path(sys.prefix)!=VENV:p.error("Pinned pure environment")
 import numpy as np,torch
 torch.set_num_threads(1)
 from platform_config import require_python_packages,identity
 require_python_packages(torch,np)
 from semantic_benchmark_execution import runtime_sources
 from upstream_candidate_helpers import verify_sources
 from poly_dependencies import verify
 from unified_graph_prepare import prepare_case
 from unified_graph_contract import prepare
 from compiler_configuration import configuration,PROFILE_SHA256,verify_artifact_configuration
 from candidate_contract import request_input_names,request_rotations
 from cipher_abi import artifact_options
 from seal_artifact_gate import inspect_artifacts
 from seal_cpu_golden import KEY_BUILD,PROFILE,compare
 from python_compiler_smoke import BUILD,logged
 from native_execution_slots import native_slot
 from result_retention import cleanup_run
 import candidate_sandbox as sandbox
 sources=runtime_sources();helpers=verify_sources();dependency=verify()
 binaries={str(p):sha(p) for p in (BUILD/"bin/hecate-opt",BUILD/"lib/libSEAL_HEVM.so",KEY_BUILD/"seal_packed_keys",KEY_BUILD/"libseal_packed_metadata.so",BUILD/"lib/libHecateFrontend.so")}
 a.output.mkdir(parents=True);start=time.monotonic();rows=[];experiments=[]
 plan=dict(models=chosen,runtime_sources=sources,helper_sources=helpers,dependency=dependency,
  binaries=binaries,runner_sha256=sha(Path(__file__)),worker_sha256=digest(WORKER),
  protocol_scope="Bounded scalar/cipher arithmetic, dispatch, conversion and Empty identities",
  max_wall_seconds=900,max_retained_mib=256,platform=identity(),compiler_configuration="seal-cpu-eva-w45-v1",
  concurrency=1,paid_calls=0,diagnostic_models_not_added_to_frozen_corpus=True)
 plan["binding"]=digest(plan);dump(a.output/"plan.json",plan)
 def boundary():
  if runtime_sources()!=sources or verify_sources()!=helpers:raise ValueError("Source integrity")
  if any(sha(Path(n))!=h for n,h in binaries.items()):raise ValueError("Binary integrity")
  if time.monotonic()-start>=900:raise TimeoutError("Batch wall boundary")
  retained=sum(f.stat().st_size for directory in [a.output,*experiments] for f in directory.rglob("*") if f.is_file() and "private-keys" not in f.parts)
  if retained>256*1024**2:raise ValueError("Retained evidence boundary")
 def run(payload,out,argv,log,seconds=60,keys=None,helpers=False):
  boundary()
  with native_slot(WORK/"cache/agent-native-slots"):
   return sandbox.run(payload,out,argv,log,min(seconds,max(1,int(900-(time.monotonic()-start)))),keys,helpers=helpers)
 for spec in chosen:
  boundary()
  folder=Path(tempfile.mkdtemp(prefix="seal-cpu-golden-expr_protocol-",dir=RESULTS));experiments.append(folder)
  dump(a.output/"experiments.json",[str(p) for p in experiments])
  out=folder/"output";out.mkdir()
  report=dict(id=spec["id"],regime=spec["regime"],model_sha256=spec["model_sha256"],status="failed",
   compiled=False,encrypted_execution=False,agent_calls=0)
  stage="reference"
  try:
   prepare_case(spec["model"],folder)
   request=prepare(spec["model"],PROFILE_SHA256,configuration("seal-cpu-eva-w45-v1"))
   dump(folder/"model.json",spec["model"]);dump(folder/"request.json",request)
   payload=folder/"payload.json";dump(payload,dict(request=request,manual_spec={k:spec[k] for k in ("regime","context")}))
   references,diagnostics=independent_values(spec)
   with np.load(folder/"arrays.npz",allow_pickle=False) as arrays:
    np.testing.assert_allclose(arrays["reference"],references,atol=1e-12,rtol=1e-12)
   report.update(diagnostics)
   report["helper_zero_intervention_delta"]=float(np.max(np.abs(references)))
   if report["helper_zero_intervention_delta"]<=1e-8 or report["helper_identity_intervention_delta"]<=1e-8:
    raise ValueError("Nondiscriminating probes")
   stage="trace"
   if run(payload,out,[str(VENV/"bin/python"),"-B","-c",WORKER],folder/"trace.log",helpers=True)!=0:raise ValueError("Actual dependency trace failed")
   call=strict_file(out/"dependency-calls.json",1024**2)
   if call["source_sha256"]!=sha(ROOT/"third_party/dacapo/python/hecate/hecate/expr.py"):raise ValueError("Helper source mismatch")
   report["calls"]=call
   mlir=list(out.glob("*.mlir"))
   if len(mlir)!=1:raise ValueError("Ambiguous traced IR")
   stage="compiler"
   argv=["/hecate-opt","/out/"+mlir[0].name,"--eva","--ckks-config=/profile.json","--waterline=45",
         "--enable-debug-printer","--mlir-disable-threading","--verify-each","-o","/out/lowered.mlir"]
   if run(payload,out,argv,folder/"compile.log")!=0:raise ValueError("Dependency compiler failed")
   report["compiled"]=True;stage="artifact_gate"
   gate=inspect_artifacts((out/"lowered._hecate_golden.hevm").read_bytes(),(out/"_hecate_golden.cst").read_bytes(),
     rotation_steps=request_rotations(request),expected_inputs=len(request_input_names(request)),**artifact_options(request["layout"]))
   verify_artifact_configuration(request,gate,sha(PROFILE));report["artifact_gate"]=gate
   stage="key_setup";keys=folder/"private-keys";keys.mkdir(mode=0o700)
   with native_slot(WORK/"cache/agent-native-slots"):
    if logged([str(KEY_BUILD/"seal_packed_keys"),str(keys),"16"],folder/"parameters.json",120)!=0:raise ValueError("Key setup failed")
   with np.load(folder/"arrays.npz",allow_pickle=False) as arrays:np.savez(out/"arrays.npz",inputs=arrays["inputs"])
   protected={f.name:sha(f) for f in out.iterdir() if f.is_file()}
   stage="seal_runtime"
   if run(payload,out,[str(VENV/"bin/python"),"/app/candidate_worker.py","execute"],folder/"execute.log",150,keys)!=0:raise ValueError("SEAL execution failed")
   report["encrypted_execution"]=True
   if any(sha(out/name)!=h for name,h in protected.items()):raise ValueError("Execution artifact mutation")
   stage="comparison"
   with np.load(folder/"arrays.npz",allow_pickle=False) as arrays:
    comparison=compare(np.load(out/"decrypted.npy",allow_pickle=False),arrays["reference"],1e-5,1e-4)
   report["comparison"]=comparison
   if not comparison["passed"]:raise ValueError("Frozen error threshold")
   report["status"]="passed"
  except Exception as error:report.update(failure_layer=stage,diagnostic=str(error))
  finally:
   # Existing retention policy needs a terminal report and a direct results child.
   dump(folder/"report.json",report)
   if (folder/"private-keys").exists():cleanup_run(folder,RESULTS)
   boundary()
  rows.append(dict(id=spec["id"],regime=spec["regime"],status=report["status"],
   failure_layer=report.get("failure_layer"),diagnostic=report.get("diagnostic"),folder=str(folder),
   max_absolute_error=report.get("comparison",{}).get("max_absolute_error"),
   files={str(f.relative_to(folder)):sha(f) for f in folder.rglob("*") if f.is_file()}))
  dump(a.output/"progress.json",rows)
  if report.get("failure_layer") in ("key_setup","seal_runtime"):raise ValueError("Environment/runtime failure; related batch stopped")
 summary=dict(format="poseidon-expr_protocol-dependency-audit-v1",plan_binding=plan["binding"],rows=rows,
  status="passed" if all(r["status"]=="passed" for r in rows) else "failed",
  seconds=time.monotonic()-start,paid_calls=0,candidate_access_added=False,
  upstream_helper_source_unchanged=True,original_corpus_unchanged=True)
 summary["binding"]=digest(summary);dump(a.output/"report.json",summary)
 print(json.dumps(dict(status=summary["status"],passed=sum(r["status"]=="passed" for r in rows),
   failures=[dict(id=r["id"],layer=r["failure_layer"],reason=r["diagnostic"]) for r in rows if r["status"]!="passed"],paid_calls=0)))
 return int(summary["status"]!="passed")
if __name__=="__main__":raise SystemExit(main())
