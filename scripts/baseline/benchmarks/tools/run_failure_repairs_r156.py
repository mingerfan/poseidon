"""Bounded offline diagnosis and manual repair regressions; NEVER calls a provider."""
import argparse,json,os,shlex,subprocess,sys,time,unittest,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from stage2_agent_campaign import ROOT,BASE
from benchmark_graph import digest
from benchmark_runner import dump
from campaign_live_state import sealed
from stage2_agent_pilot_plan import sha
from run_stage2_guidance_retest import evidence_path

PAIRS={"014_2":"uuoau9b0","049_2":"ho8ulryh","054_1":"7dlrkik6","082_1":"tcqynh8i",
       "088_0":"xpl3ywtu","103_0":"huevv43b","112_2":"6svyal2c","124_0":"s62d5xax","133_0":"_wp2t_6w"}

def diagnose(results):
 import numpy as np
 rows=[]
 for ident,folder in PAIRS.items():
  if ident in ("049_2","054_1"):continue
  root=results/("agent-deepseek-"+folder);a=np.load(root/"arrays.npz",allow_pickle=False)
  inputs=a["inputs"];inputs=inputs[:,None,:] if inputs.ndim==2 else inputs
  x=inputs[:,0,:];reference=a["reference"]
  y=inputs[:,1,:] if inputs.shape[1]>1 else None
  z=inputs[:,2,:] if inputs.shape[1]>2 else None
  t=inputs[:,3,:] if inputs.shape[1]>3 else None
  W=np.array([[-1/12,-1/48,1/24],[1/16,-1/16,0.]])
  roll=lambda v,k:np.roll(v,-k,axis=-1)
  if ident=="014_2":
   bad=(y[:,0]+y[:,1])/32+1/16
   pred=np.column_stack((x[:,0]**2,x[:,1]**2,-bad/4+1/32,3*bad/16+1/32))
   pred+=(z[:,:2].mean(1)+t[:,:2].mean(1))[:,None]
   reason="First Linear weights are +1/32,+1/32 instead of -1/8,-1/32."
  elif ident=="082_1":
   pred=(x[:,:3]@W.T+1/16)*(y[:,:3]**2@W.T+1/32)
   pred+=(z[:,:3].mean(1)+t[:,:3].mean(1))[:,None]+1
   reason="Always-true helper update adds 1 to every output; helper name also impersonated dictionary update coverage."
  elif ident=="088_0":
   p={2:-1/12,3:-1/16,4:-1/48,5:0.,6:1/32,7:1/24,8:1/16};q=1/8
   c0=x*np.array([p[8],p[4],0,0])+roll(x,1)*[0,p[2],0,0]+roll(x,3)*[p[3],p[7],0,0]
   c0+=roll(y,1)*[0,0,-p[8]*q,p[7]*q]+roll(y,2)*[0,0,p[8]*q,p[4]*q]+roll(y,3)*[0,0,0,p[2]*q]
   c0+=z.sum(1)[:,None]/3+np.array([p[8],p[8],p[6],p[6]])
   c1=x*[p[2],p[3],0,0]+roll(x,1)*[0,p[8],0,0]+roll(x,2)*[p[7],0,0,0]+roll(x,3)*[p[4],0,0,0]+np.array([p[8],p[8],0,0])
   pred=np.concatenate((c0,c1[:,:2]),axis=1)
   reason="Linear diagonal masks and rotation offsets use wrong source columns."
  elif ident=="103_0":
   out=x/8+roll(y*y,3)+z.sum(1)[:,None]/3
   pred=np.concatenate((out[:,:6],x[:,:3]/8),axis=1)
   reason="Append at slot 3 in period 8 needs rotate(+5); generated helper uses +3."
  elif ident=="112_2":
   poly=((x/64+1/32)*x-1/8)*x+1/16
   pred=(poly+roll(y*y,5))[:,:6]
   reason="Polynomial bias fills padding slots; concatenated second tensor gets +1/16."
  elif ident=="124_0":
   poly=((x/64+1/32)*x-1/8)*x+1/16
   pred=poly.sum(1,keepdims=True)/17
   reason="Mean includes 15 padded slots containing 1/16; bias=15/272."
  else:
   pred=(-x[:,0]/8-x[:,1]/32+1/16)[:,None]
   reason="Pinned Empty - Expr returns Expr; output misses required negation."
  actual=np.load(root/"attempt-03/output/decrypted.npy",allow_pickle=False)
  if actual.shape!=pred.shape:raise ValueError((ident,actual.shape,pred.shape))
  err=np.abs(actual-pred);match=bool(np.all(err<=1e-5+1e-4*np.abs(pred)))
  if not match:raise ValueError("Independent diagnosis mismatch: "+ident)
  files=["request.json","model.json","arrays.npz","attempt-03/candidate.py","attempt-03/output/decrypted.npy","report.json"]
  rows.append(dict(id="construct_"+ident,reason=reason,independent_formula_matches_decryption=True,
       max_formula_decryption_error=float(err.max()),wrong_formula_reference_max_error=float(np.abs(pred-reference).max()),
       compared_values=pred.size,private_arrays_sent_to_provider=False,parents={str(root/n):sha(root/n) for n in files}))
 return rows

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True);p.add_argument("--inside",action="store_true")
 a=p.parse_args()
 from hecate_python_env import VENV,enter_nix
 if not a.inside:
  return enter_nix('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+
   shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),*sys.argv[1:],"--inside"]),seconds=3600)
 from workspace_paths import RESULTS
 from semantic_benchmark_execution import runtime_sources
 from unified_graph_contract import prepare,validate_candidate
 from audit_unified_candidate import verify_candidate
 from campaign_request_files import read_request
 if os.environ.get("IN_NIX_SHELL")!="pure" or Path(sys.prefix)!=VENV:raise ValueError("Pinned pure environment required")
 if a.output.exists() or a.output.resolve().parent!=RESULTS.resolve():raise ValueError("Fresh direct result directory required")
 if os.statvfs(RESULTS).f_bavail*os.statvfs(RESULTS).f_frsize<5*1024**3:raise ValueError("Disk reserve")
 a.output.mkdir();before=runtime_sources();fixtures=BASE/"benchmarks/fixtures/failure-repairs-r156"
 fixture_hashes={str(f):sha(f) for f in fixtures.glob("*.py")}
 plan=sealed(dict(source_hashes=before,fixtures=fixture_hashes,runner_sha256=sha(Path(__file__)),
  cases=["construct_"+i for i in PAIRS],max_seconds=3600,case_timeout_seconds=300,native_concurrency=1,
  new_paid_calls=0,agent_generation=False,threshold=dict(atol=1e-5,rtol=1e-4)))
 dump(a.output/"plan.json",plan)
 tests=unittest.defaultTestLoader.loadTestsFromNames(["test_failure_repairs","test_unified_generation_guidance","test_unified_directed_guidance","test_seal_cpu_golden.SealGateTests"])
 with (a.output/"tests.log").open("x") as f:res=unittest.TextTestRunner(stream=f,verbosity=2).run(tests)
 testinfo=dict(run=res.testsRun,failed=len(res.failures),errors=len(res.errors),skipped=len(res.skipped))
 dump(a.output/"tests.json",testinfo);print("tests",testinfo,flush=True)
 if not res.wasSuccessful():return 1
 diagnoses=diagnose(RESULTS);dump(a.output/"diagnoses.json",sealed(dict(rows=diagnoses,source_hashes=before)))
 # Original producer/runtime sources remain versioned history; do not claim recompilation.
 oldpipe=RESULTS/"stage2-targeted-pipeline-diagnosis-r153.json"
 dump(a.output/"scale-evidence.json",dict(parent=str(oldpipe),sha256=sha(oldpipe),
   conclusion="Capacity failure in fixed compiler configuration; remaining level 1 / scale 75 also violates output gate. No safety/scale edits or successful recompile claimed."))
 command=[str(VENV/"bin/python"),"-B",str(BASE/"benchmarks/tools/verify_directed_guidance.py"),
  "--inside","--index",str(RESULTS/"stage2-agent-campaign-v7-r130/index.json"),"--output",str(a.output/"compatibility.json")]
 with (a.output/"compatibility.log").open("x") as f:subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=180)
 prepared=[];rows=[]
 for ident,folder in PAIRS.items():
  case_id="construct_"+ident;oldroot=RESULTS/("agent-deepseek-"+folder);old=json.loads((oldroot/"request.json").read_text())
  kw=dict(construction_profile=old.get("construction_profile"),constant_policy=old["constant_origins"].get("policy"))
  exercise=old["construction_exercise"]["id"]
  request=prepare(old["model"],old["compiler_profile_sha256"],old["compiler_configuration"],exercise,generation_guidance="explicit-v5",**kw)
  source=(fixtures/(case_id+".py")).read_text()
  try:
   validate_candidate(dict(schema=1,request_id=request["request_id"],hecate_source=source),request)
  except Exception as e:
   rows.append(dict(id=case_id,status="static_failed",diagnostic=str(e),scripted=True));continue
  work=a.output/case_id;work.mkdir();dump(work/"model.json",old["model"])
  dump(work/"responses.json",[json.dumps(dict(schema=1,request_id=request["request_id"],hecate_source=source))])
  prepared.append((case_id,work,request,exercise))
 dump(a.output/"preflight.json",dict(prepared=[x[0] for x in prepared],failed=rows))
 for case_id,work,request,exercise in prepared:
  if runtime_sources()!=before or any(sha(Path(f))!=h for f,h in fixture_hashes.items()):raise ValueError("Source/fixture drift")
  cmd=[str(VENV/"bin/python"),"-B",str(BASE/"run_candidate.py"),"--inside","--case",str(work/"model.json"),
       "--unified-guidance","explicit-v5","--compiler-configuration","seal-cpu-eva-w45-v1","--unified-exercise",exercise,
       "--unified-profile","public-v1","--replay",str(work/"responses.json"),"--max-repairs","0"]
  started=time.monotonic()
  with (work/"run.log").open("x") as f:code=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=300).returncode
  folder=evidence_path(work/"run.log",RESULTS)
  if folder is None:raise ValueError("Missing replay evidence")
  report=json.loads((folder/"report.json").read_text());read_request(folder/"request.json",request)
  if report["agent_calls"]!=0:raise ValueError("Unexpected API call")
  row=dict(id=case_id,status=report["status"],evidence=str(folder),report_sha256=sha(folder/"report.json"),
     manual_source_repair=case_id!="construct_054_1",unchanged_generated_source_replay=case_id=="construct_054_1",
     new_agent_generation=False,seconds=time.monotonic()-started,exit_code=code,
     failure_layer=report["attempts"][-1].get("failure_layer"),diagnostic=report["attempts"][-1].get("diagnostic"))
  if not code and report["status"]=="passed":
   row["audit"]=verify_candidate(folder)
   row["max_absolute_error"]=report["attempts"][-1]["comparison"]["max_absolute_error"]
  rows.append(row);dump(a.output/"progress.json",rows);print(case_id,row["status"],row.get("diagnostic"),flush=True)
  if row["failure_layer"] in ("environment","integrity","key_setup"):raise ValueError("Stop environment/integrity failure")
 if runtime_sources()!=before:raise ValueError("Source drift")
 result=sealed(dict(plan_binding=plan["binding"],source_hashes=before,rows=rows,tests=testinfo,
  passed=sum(x["status"]=="passed" for x in rows),failed=sum(x["status"]!="passed" for x in rows),
  skipped=0,new_paid_calls=0,new_agent_generation=False,diagnosed_numerical_cases=len(diagnoses),
  max_formula_decryption_error=max(x["max_formula_decryption_error"] for x in diagnoses),
  parents={str(a.output/n):sha(a.output/n) for n in ["plan.json","tests.json","diagnoses.json","compatibility.json","scale-evidence.json"]}))
 dump(a.output/"report.json",result);print(json.dumps({k:result[k] for k in ("passed","failed","binding")}),flush=True)
 return 0 if not result["failed"] else 1
if __name__=="__main__":raise SystemExit(main())
