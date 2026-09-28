"""Bounded actual upstream polynomial dependency acceptance; no candidate privilege expansion."""
import argparse,hashlib,json,os,shlex,sys,time,tempfile
from pathlib import Path
BASE=Path(__file__).resolve().parents[2];ROOT=BASE.parents[1]
sys.path.insert(0,str(BASE))
from benchmark_graph import digest
from benchmark_runner import strict_file,dump
WORKER=r'''
import hashlib,json,sys
from pathlib import Path
import numpy as np
from candidate_trace import load_frontend
hc=load_frontend();sys.modules["hecate"]=hc
sys.path[:0]=["/poly-deps","/upstream-poly"]
import poly.MPCB as mpcb
import poly.Poly as poly
spec=json.loads(Path("/payload.json").read_text())["manual_spec"]
if spec["context"] not in (0,1,2) or spec["regime"] not in ("leaf","tree15_first","tree15_second","tree27","default","sign","relua","genRelu6"):raise ValueError("Trusted polynomial context")
events=[];returned=[]
def observe(frame,event,arg):
 if event=="return" and frame.f_code.co_filename in (mpcb.__file__,poly.__file__):
  value=arg
  if isinstance(value,np.ndarray) and value.dtype==object and value.size==1:value=value.flat[0]
  if isinstance(value,hc.Expr):
   events.append(dict(symbol=Path(frame.f_code.co_filename).stem+"."+frame.f_code.co_name,return_handle=int(value.obj)))
@hc.func("c,c")
def golden(x,zero_ct):
 if spec["contexts"]=="unscaled":
  if spec["context"]==1:x=-x
  elif spec["context"]==2 and spec.get("rotation_placement","before")=="before":x=x.rotate(1)
 elif spec["contexts"]=="scaled":
  if spec["context"]==0:x=x*.125
  elif spec["context"]==1:x=x*.125+.0625
  else:x=(x*x)*.125
 else:raise ValueError("Unknown context family")
 x=np.array([x],dtype=object)
 sys.setprofile(observe)
 try:
  if spec["regime"]=="leaf":fn=mpcb.GenPoly(["0"],["0",".25","0",".0625"],4,scale=spec["scale"])
  elif spec["regime"]=="tree15_first":fn=poly.poly1
  elif spec["regime"]=="tree15_second":fn=poly.poly2
  elif spec["regime"]=="tree27":fn=poly.poly3
  elif spec["regime"]=="default":fn=poly.GenPoly()
  elif spec["regime"]=="sign":fn=poly.sign
  elif spec["regime"]=="relua":fn=lambda v:poly.relua(v,.125)
  else:fn=poly.genRelu6(48)
  result=fn(x)
 finally:sys.setprofile(None)
 value=result.flat[0]
 if not events or events[-1]["return_handle"]!=int(value.obj):raise ValueError("Actual helper return mismatch")
 helper_return=int(value.obj)
 if spec["post_silu"]:value=x.flat[0]*(value+.5)
 if spec.get("rotation_placement","before")=="after":value=value.rotate(1)
 returned.append(dict(helper_return_handle=helper_return,golden_return_handle=int(value.obj),post_silu=spec["post_silu"],rotation_placement=spec.get("rotation_placement","before")))
 return [value]
hc.save("/out","/out")
Path("/out/dependency-calls.json").write_text(json.dumps(dict(events=events,golden_return_handles=returned,
 actual_upstream=True,helper_replaced=False,
 source_sha256=hashlib.sha256(Path(mpcb.__file__).read_bytes()).hexdigest(),
 poly_source_sha256=hashlib.sha256(Path(poly.__file__).read_bytes()).hexdigest())))
'''
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def cases(contexts="scaled",post_silu=False):
 from benchmark_suite import Builder,coefficient_tables
 tables,default=coefficient_tables()
 result=[]
 for regime in ("leaf","tree15_first","tree15_second","tree27","default","sign","relua","genRelu6"):
  for context in range(3):
   scale=(1.,1.7,2.)[context]
   b=Builder([(16,)]);x="input0"
   if contexts=="unscaled":
    if context==1:x=b.node("negate",[x])
    elif context==2:x=b.node("rotate",[x],step=1)
   else:
    if context==0:x=b.node("multiply",[x,b.const(.125)])
    elif context==1:x=b.node("add",[b.node("multiply",[x,b.const(.125)]),b.const(.0625)])
    else:x=b.node("multiply",[b.node("square",[x]),b.const(.125)])
   def polynomial(x,coeff):
    # Pinned GenPoly evaluates odd leaf coefficients. These even-degree divisors
    # preserve parity; the ignored even terms are explicitly recorded below.
    return b.node("polynomial",[x,b.const([v if i%2 else 0. for i,v in enumerate(coeff)])],basis="chebyshev")
   def sign(x):
    for coeff,divisor in tables:x=polynomial(x,[v/divisor for v in coeff])
    return x
   if regime=="leaf":y=polynomial(x,[0.,.25/scale,0.,.0625/scale])
   elif regime.startswith("tree"):
    i={"tree15_first":0,"tree15_second":1,"tree27":2}[regime]
    coeff,divisor=tables[i];y=polynomial(x,[v/divisor for v in coeff])
   elif regime=="default":y=polynomial(x,default)
   elif regime=="sign":y=sign(x)
   else:
    shifted=b.node("subtract",[x,b.const(.125)])
    y=b.node("add",[b.node("multiply",[sign(x),x]),
       b.node("multiply",[sign(shifted),b.node("negate",[shifted])])])
    y=b.node("add",[y,b.const(.0625)])
   if post_silu:y=b.node("multiply",[x,b.node("add",[y,b.const(.5)])])
   model=b.finish(y);model["id"]="poly_dependency_"+regime+"_"+contexts+("_post_silu" if post_silu else "")+"_"+str(context)
   result.append(dict(id=model["id"],regime=regime,context=context,contexts=contexts,post_silu=post_silu,scale=scale,
      model=model,model_sha256=digest(model)))
 return result

def independent_values(spec):
 """Scalar Chebyshev recurrence, separate from NumPy Clenshaw and Torch graph."""
 from benchmark_graph import samples
 from benchmark_suite import coefficient_tables
 tables,default=coefficient_tables()
 def cheb(x,coeff,project=True):
  ts=[1.,x]
  for i in range(2,len(coeff)):ts.append(2*x*ts[-1]-ts[-2])
  return sum(c*ts[i] for i,c in enumerate(coeff) if i%2 or not project)
 def sign(x):
  for coeff,divisor in tables:x=cheb(x,[v/divisor for v in coeff])
  return x
 refs=[];ideal=[];intervention=[];full_coefficient_delta=[]
 for probe in samples(spec["model"],4):
  values=probe["input0"].reshape(-1).tolist()
  if spec["contexts"]=="unscaled":
   if spec["context"]==1:values=[-v for v in values]
   elif spec["context"]==2:values=values[1:]+values[:1]
  else:values=[v*.125+.0625 if spec["context"]==1 else v*v*.125 if spec["context"]==2 else v*.125 for v in values]
  row=[];ideal_row=[]
  for x in values:
   regime=spec["regime"];coeff=None
   if regime=="leaf":coeff=[0.,.25/spec["scale"],0.,.0625/spec["scale"]]
   elif regime.startswith("tree"):
    coeff,divisor=tables[{"tree15_first":0,"tree15_second":1,"tree27":2}[regime]]
    coeff=[v/divisor for v in coeff]
   elif regime=="default":coeff=default
   if coeff is not None:
    y=cheb(x,coeff);target=None
    full_coefficient_delta.append(abs(y-cheb(x,coeff,False)))
   elif regime=="sign":
    y=sign(x);target=.5 if x>0 else -.5 if x<0 else 0.
   else:
    y=sign(x)*x+sign(x-.125)*(.125-x)+.0625;target=min(max(x,0.),.125)
   if spec.get("post_silu"):y=x*(y+.5);target=None
   row.append(y);ideal_row.append(target)
   intervention.append(abs(y-x))
  refs.append(row);ideal.append(ideal_row)
 return refs,dict(helper_identity_intervention_delta=max(intervention),
   ignored_even_coefficient_max_effect=max(full_coefficient_delta,default=0.),
   approximation_max_absolute_error=max((abs(y-t) for row,targets in zip(refs,ideal) for y,t in zip(row,targets) if t is not None),default=None),
   approximation_target="half_sign" if spec["regime"]=="sign" else "clamp_0_0.125" if spec["regime"] in ("relua","genRelu6") else None,
   approximation_accuracy_certified=False,
   upstream_even_leaf_coefficients_not_evaluated=True)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True)
 p.add_argument("--compiler-configuration",choices=("seal-cpu-eva-w45-v1","seal-cpu-eva-w40-v1"),default="seal-cpu-eva-w45-v1");p.add_argument("--post-silu",action="store_true",help="Separate parent composition partition; never replaces direct-return failures");p.add_argument("--contexts",choices=("scaled","unscaled"),default="scaled");p.add_argument("--regime",action="append",choices=("leaf","tree15_first","tree15_second","tree27","default","sign","relua","genRelu6"));p.add_argument("--rotation-placement",choices=("before","after"),default="before",help="After: same pointwise polynomial model, commute rotation after actual helper; original failure retained");p.add_argument("--execute",action="store_true");p.add_argument("--inside",action="store_true",help=argparse.SUPPRESS);a=p.parse_args()
 compile(WORKER,"<polynomial-worker>","exec")
 from workspace_paths import RESULTS
 from hecate_python_env import enter_nix,VENV,WORK
 if a.post_silu and (a.contexts!="unscaled" or a.regime!=["default"]):p.error("Post composition restricted to default unscaled polynomial")
 chosen=[r for r in cases(a.contexts,a.post_silu) if not a.regime or r["regime"] in a.regime]
 if a.rotation_placement=="after":
  if a.contexts!="unscaled" or a.regime!=["default"] or a.post_silu:p.error("Post rotation restricted to the original direct default polynomial case")
  chosen=[r for r in chosen if r["context"]==2]
  for r in chosen:r["rotation_placement"]="after"
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
 (a.output/"runner.py").write_bytes(Path(__file__).read_bytes())
 (a.output/"worker.py").write_text(WORKER)
 plan=dict(models=chosen,runtime_sources=sources,helper_sources=helpers,dependency=dependency,
  binaries=binaries,runner_sha256=sha(Path(__file__)),worker_sha256=digest(WORKER),
  polynomial_scope="Explicit odd-coefficient behavior of fixed upstream trees; full even coefficient discrepancy and approximation error separate",
  max_wall_seconds=900,max_retained_mib=256,platform=identity(),compiler_configuration=a.compiler_configuration,
  concurrency=1,paid_calls=0,diagnostic_models_not_added_to_frozen_corpus=True)
 plan["binding"]=digest(plan);dump(a.output/"plan.json",plan)
 def boundary():
  if sha(Path(__file__))!=plan["runner_sha256"]:raise ValueError("Runner integrity")
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
  folder=Path(tempfile.mkdtemp(prefix="seal-cpu-golden-polynomial-",dir=RESULTS));experiments.append(folder)
  dump(a.output/"experiments.json",[str(p) for p in experiments])
  out=folder/"output";out.mkdir()
  report=dict(id=spec["id"],regime=spec["regime"],model_sha256=spec["model_sha256"],status="failed",
   compiled=False,encrypted_execution=False,agent_calls=0)
  stage="reference"
  try:
   prepare_case(spec["model"],folder)
   request=prepare(spec["model"],PROFILE_SHA256,configuration(a.compiler_configuration))
   dump(folder/"model.json",spec["model"]);dump(folder/"request.json",request)
   payload=folder/"payload.json";dump(payload,dict(request=request,manual_spec={k:spec[k] for k in ("regime","scale","context","contexts","post_silu","rotation_placement") if k in spec}))
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
   if call["source_sha256"]!=sha(ROOT/"third_party/dacapo/python/poly/poly/MPCB.py") or call["poly_source_sha256"]!=sha(ROOT/"third_party/dacapo/python/poly/poly/Poly.py"):raise ValueError("Helper source mismatch")
   report["calls"]=call
   mlir=list(out.glob("*.mlir"))
   if len(mlir)!=1:raise ValueError("Ambiguous traced IR")
   stage="compiler"
   argv=["/hecate-opt","/out/"+mlir[0].name,"--eva","--ckks-config=/profile.json","--waterline="+str(request["compiler_configuration"]["waterline"]),
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
 summary=dict(format="poseidon-polynomial-dependency-audit-v1",plan_binding=plan["binding"],rows=rows,
  status="passed" if all(r["status"]=="passed" for r in rows) else "failed",
  seconds=time.monotonic()-start,paid_calls=0,candidate_access_added=False,
  upstream_helper_source_unchanged=True,original_corpus_unchanged=True)
 summary["binding"]=digest(summary);dump(a.output/"report.json",summary)
 print(json.dumps(dict(status=summary["status"],passed=sum(r["status"]=="passed" for r in rows),
   failures=[dict(id=r["id"],layer=r["failure_layer"],reason=r["diagnostic"]) for r in rows if r["status"]!="passed"],paid_calls=0)))
 return int(summary["status"]!="passed")
if __name__=="__main__":raise SystemExit(main())
