"""Bounded actual upstream SumSlots/roll acceptance; no candidate privilege expansion."""
import argparse,hashlib,json,os,shlex,sys,time,tempfile
from pathlib import Path
BASE=Path(__file__).resolve().parents[2];ROOT=BASE.parents[1]
sys.path.insert(0,str(BASE))
from benchmark_graph import digest
from benchmark_runner import strict_file,dump
WORKER=r'''
import hashlib,inspect,json,sys
from pathlib import Path
from candidate_trace import load_frontend
hc=load_frontend();sys.modules["hecate"]=hc
sys.path[:0]=["/poly-deps","/upstream-poly"]
import poly.MPCB as mpcb
from upstream_adapters.periodic_ring import proxy_type
payload=json.loads(Path("/payload.json").read_text());spec=payload["manual_spec"]
if spec["m"] not in (1,3,4,5,7) or spec["p"] not in (1,2) or spec["context"] not in (0,1,2):raise ValueError("Trusted dependency case")
geometry=mpcb.InferShapes(dict(nt=16,bb=1.,fh=1,fw=1,s=1,hi=1,wi=16,ki=1,ci=1,co=1))
closure=mpcb.shapeClosure(**geometry);found={};visited=set()
def visit(fn):
 if not inspect.isfunction(fn) or id(fn) in visited:return
 visited.add(id(fn));found[fn.__name__]=fn
 for cell in fn.__closure__ or ():
  if inspect.isfunction(cell.cell_contents):visit(cell.cell_contents)
for fn in closure.values():visit(fn)
sum_slots=found["SumSlots"]
if sum_slots.__code__.co_filename!="/upstream-poly/poly/MPCB.py":raise ValueError("Unpinned closure")
record=dict(operations=[],rotations=[]);events=[];proxy=proxy_type(hc.Expr)
def observe(frame,event,arg):
 if event=="call" and frame.f_code is mpcb.roll.__code__:
  events.append(dict(callee="MPCB.roll",offset=frame.f_locals["i"]))
 if event=="return" and frame.f_code is sum_slots.__code__:
  if not isinstance(arg,proxy):raise ValueError("Unexpected helper return")
  events.append(dict(callee="MPCB.shapeClosure.SumSlots",m=spec["m"],p=spec["p"],return_handle=int(arg.value.obj)))
@hc.func("c,c")
def golden(x,zero_ct):
 if spec["context"]==0:x=x*.5
 elif spec["context"]==1:x=x*.25+.125
 elif spec["context"]==2:x=x*x
 a=proxy(x,16,record)
 sys.setprofile(observe)
 try:result=sum_slots(a,spec["m"],spec["p"])
 finally:sys.setprofile(None)
 returned=[e for e in events if e["callee"]=="MPCB.shapeClosure.SumSlots"]
 if len(returned)!=1 or returned[0]["return_handle"]!=int(result.value.obj):raise ValueError("Return identity")
 record["golden_return_handle"]=int(result.value.obj)
 return [result.value]
hc.save("/out","/out")
Path("/out/dependency-calls.json").write_text(json.dumps(dict(events=events,adapter=record,
 geometry=geometry,closure_keys=sorted(closure),actual_upstream=True,helper_replaced=False,
 source_sha256=hashlib.sha256(Path("/upstream-poly/poly/MPCB.py").read_bytes()).hexdigest())))
'''
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def cases():
 from benchmark_suite import Builder
 result=[]
 for regime,ms in (("identity",(1,1,1)),("power_of_two",(4,4,4)),("remainder",(3,5,7))):
  for context,m in enumerate(ms):
   step=2 if regime=="remainder" and context==1 else 1
   b=Builder([(16,)]);x="input0"
   if context==0:x=b.node("multiply",[x,b.const(.5)])
   elif context==1:x=b.node("add",[b.node("multiply",[x,b.const(.25)]),b.const(.125)])
   elif context==2:x=b.node("square",[x])
   output=x
   for j in range(1,m):output=b.node("add",[output,b.node("rotate",[x],step=j*step)])
   model=b.finish(output);model["id"]="sumslots_"+regime+"_"+str(context)
   result.append(dict(id=model["id"],regime=regime,m=m,p=step,context=context,model=model,model_sha256=digest(model)))
 return result
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True)
 p.add_argument("--execute",action="store_true");p.add_argument("--inside",action="store_true",help=argparse.SUPPRESS);a=p.parse_args()
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
  folder=Path(tempfile.mkdtemp(prefix="seal-cpu-golden-sumslots-",dir=RESULTS));experiments.append(folder)
  dump(a.output/"experiments.json",[str(p) for p in experiments])
  out=folder/"output";out.mkdir()
  report=dict(id=spec["id"],regime=spec["regime"],model_sha256=spec["model_sha256"],status="failed",
   compiled=False,encrypted_execution=False,agent_calls=0)
  stage="reference"
  try:
   prepare_case(spec["model"],folder)
   request=prepare(spec["model"],PROFILE_SHA256,configuration("seal-cpu-eva-w45-v1"))
   dump(folder/"model.json",spec["model"]);dump(folder/"request.json",request)
   payload=folder/"payload.json";dump(payload,dict(request=request,manual_spec={k:spec[k] for k in ("m","p","context")}))
   # Independent coordinate recurrence, separate from both graph interpreters.
   from benchmark_graph import samples
   references=[];identity_deltas=[]
   for probe in samples(spec["model"],4):
    values=probe["input0"].reshape(-1).tolist()
    values=[v*.25+.125 if spec["context"]==1 else v*v if spec["context"]==2 else v*.5 for v in values]
    expected=[sum(values[(i+j*spec["p"])%16] for j in range(spec["m"])) for i in range(16)]
    references.append(expected);identity_deltas.append(max(abs(expected[i]-spec["m"]*values[i]) for i in range(16)))
   with np.load(folder/"arrays.npz",allow_pickle=False) as arrays:
    np.testing.assert_allclose(arrays["reference"],references,atol=1e-12,rtol=1e-12)
   report["rotation_intervention_delta"]=max(identity_deltas)
   report["helper_zero_intervention_delta"]=float(np.max(np.abs(references)))
   if report["helper_zero_intervention_delta"]<=1e-8 or (spec["m"]>1 and max(identity_deltas)<=1e-8):raise ValueError("Nondiscriminating probes")
   stage="trace"
   if run(payload,out,[str(VENV/"bin/python"),"-B","-c",WORKER],folder/"trace.log",helpers=True)!=0:raise ValueError("Actual dependency trace failed")
   call=strict_file(out/"dependency-calls.json",1024**2)
   if call["source_sha256"]!=sha(ROOT/"third_party/dacapo/python/poly/poly/MPCB.py"):raise ValueError("Helper source mismatch")
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
 summary=dict(format="poseidon-sumslots-dependency-audit-v1",plan_binding=plan["binding"],rows=rows,
  status="passed" if all(r["status"]=="passed" for r in rows) else "failed",
  seconds=time.monotonic()-start,paid_calls=0,candidate_access_added=False,
  upstream_helper_source_unchanged=True,original_corpus_unchanged=True)
 summary["binding"]=digest(summary);dump(a.output/"report.json",summary)
 print(json.dumps(dict(status=summary["status"],passed=sum(r["status"]=="passed" for r in rows),
   failures=[dict(id=r["id"],layer=r["failure_layer"],reason=r["diagnostic"]) for r in rows if r["status"]!="passed"],paid_calls=0)))
 return int(summary["status"]!="passed")
if __name__=="__main__":raise SystemExit(main())
