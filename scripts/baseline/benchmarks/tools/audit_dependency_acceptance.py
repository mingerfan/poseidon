"""Independently recheck bounded API evidence, retaining every failed model."""
import argparse,ast,hashlib,importlib.util,json,os,re,shlex,sys
from pathlib import Path
BASE=Path(__file__).resolve().parents[2];ROOT=BASE.parents[1]
sys.path.insert(0,str(BASE))
from benchmark_graph import digest,signature
from benchmark_runner import strict_file,dump

BATCHES={"stage2-polynomial-r68":24,"stage2-expr-dependency-r69":24,
 "stage2-polynomial-unscaled-r70":6,"stage2-polynomial-parent-r71":3,"stage2-polynomial-w40-r73":6,"stage2-polynomial-parent-w40-r75":3,"stage2-polynomial-scaled-w40-r77":12,"stage2-polynomial-postrotation-r85":1}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def need(ok,why):
 if not ok:raise ValueError(why)
def bound(p):
 d=strict_file(p,8*1024**2)
 need(digest({k:v for k,v in d.items() if k!="binding"})==d["binding"],"Binding "+str(p))
 return d
def trace_check(result,regime,is_expr):
 call=result["calls"];need(call["actual_upstream"] and not call["helper_replaced"],"Actual upstream required")
 events=call["events"];need(events,"Missing trace events")
 if is_expr:
  functions={e["symbol"] for e in events};checks=call["checks"]
  if regime in ("normal_public","augmented_public","cipher_pair"):
   need({6,7,8}<={e["opcode"] for e in events if e["symbol"]=="binaryMethod"},"Missing ordinary operation")
  if regime=="reverse_public":
   need({6,7,8}<={e["opcode"] for e in events if e["symbol"]=="binaryReverseMethod"},"Missing reverse operation")
  if regime=="augmented_public":need("augmented_rebind_preserves_alias" in checks,"Alias mutation")
  if regime=="negate_rotate":need({"innerMethod","rotate"}<=functions,"Missing unary/rotation")
  if regime=="public_conversions":
   need("constant_ownership_and_Expr_identity" in checks and "resolveType" in functions,"Conversion ownership")
  if regime=="empty_protocol":
   methods=[e for e in events if e["receiver"]=="Empty" and e["symbol"]!="__init__"]
   need({e["symbol"] for e in methods}=={"__add__","__radd__","__iadd__","__sub__","__rsub__","__isub__"},"Empty protocol set")
   need(len({e["return_handle"] for e in methods})==1 and methods[0]["return_handle"] is not None,"Empty identity")
  if regime=="empty_operators":
   need({c["rejected"] for c in checks if isinstance(c,dict) and "rejected" in c}=={"Expr+Empty","Expr-Empty"},"Wrong reverse-dispatch claim")
  final=next(c["golden_return_handle"] for c in checks if isinstance(c,dict) and "golden_return_handle" in c)
  need(final in [e["return_handle"] for e in events],"Untraced final output")
 else:
  last=events[-1];returned=call["golden_return_handles"][0]
  need(last["return_handle"]==(returned["helper_return_handle"] if isinstance(returned,dict) else returned),"Polynomial return dependence")
  need("MPCB.polynomial" in {e["symbol"] for e in events},"Missing actual polynomial closure")

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True)
 p.add_argument("--inside",action="store_true",help=argparse.SUPPRESS);a=p.parse_args()
 from hecate_python_env import enter_nix,VENV
 if a.output.exists():p.error("Preserve old evidence")
 if not a.inside:return enter_nix('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),*sys.argv[1:],"--inside"]),seconds=240)
 need(os.environ.get("IN_NIX_SHELL")=="pure" and Path(sys.prefix)==VENV,"Pinned environment")
 import numpy as np,torch
 from platform_config import require_python_packages
 require_python_packages(torch,np);torch.set_num_threads(1)
 from workspace_paths import RESULTS
 from benchmark_math import evaluate as math_reference
 from benchmark_torch import evaluate as torch_reference
 from benchmark_graph import samples
 from seal_cpu_golden import compare
 from seal_artifact_gate import inspect_artifacts
 from candidate_contract import request_input_names,request_rotations
 from cipher_abi import artifact_options
 from compiler_configuration import verify_artifact_configuration
 from seal_cpu_golden import PROFILE
 rows=[];sources={};negative=0
 for batch,count in BATCHES.items():
  folder=RESULTS/batch;plan=bound(folder/"plan.json");report=bound(folder/"report.json")
  need(report["plan_binding"]==plan["binding"] and len(report["rows"])==count,"Batch denominator")
  need(sha(folder/"runner.py")==plan["runner_sha256"],"Historical runner")
  worker=next(ast.literal_eval(n.value) for n in ast.parse((folder/"runner.py").read_text()).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="WORKER" for t in n.targets))
  need(digest(worker)==plan["worker_sha256"] and worker==(folder/"worker.py").read_text(),"Historical worker")
  for name,hsh in plan["runtime_sources"].items():need(sha(ROOT/name)==hsh,"Current runtime drift")
  sources[batch]=dict(report=str(folder/"report.json"),report_sha256=sha(folder/"report.json"),plan_sha256=sha(folder/"plan.json"),
   runner_sha256=sha(folder/"runner.py"),worker_sha256=sha(folder/"worker.py"),binding=report["binding"])
  module_spec=importlib.util.spec_from_file_location("witness_"+batch.replace("-","_"),folder/"runner.py")
  module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
  specs={s["id"]:s for s in plan["models"]}
  need(len(specs)==count and {r["id"] for r in report["rows"]}==set(specs),"Model set mismatch")
  for row in report["rows"]:
   spec=specs[row["id"]];evidence=Path(row["folder"])
   need(evidence.parent==RESULTS and not evidence.is_symlink(),"Evidence location")
   for name,hsh in row["files"].items():
    rel=Path(name);path=evidence/rel
    need(not rel.is_absolute() and ".." not in rel.parts and not path.is_symlink() and sha(path)==hsh,"Evidence file identity")
   result=strict_file(evidence/"report.json",8*1024**2)
   need(result["status"]==row["status"] and digest(strict_file(evidence/"model.json",131072))==spec["model_sha256"],"Model/result identity")
   need(result["calls"]==strict_file(evidence/"output/dependency-calls.json",1024**2),"Trace record/report mismatch")
   trace_check(result,spec["regime"],"expr" in batch)
   forged=json.loads(json.dumps(result));forged["calls"]["events"]=[]
   try:trace_check(forged,spec["regime"],"expr" in batch)
   except ValueError:negative+=1
   else:raise ValueError("Missing-event control accepted")
   # Recheck BOTH graph references and the third scalar recurrence against the
   # retained expected values, without using upstream helpers or DSL lowering.
   refs=[]
   for probe in samples(spec["model"],4):
    m=math_reference(spec["model"],probe);t=torch_reference(spec["model"],probe)
    for key in m:np.testing.assert_allclose(t[key],m[key],atol=1e-12,rtol=1e-12)
    refs.append(np.concatenate([m[o["name"]].reshape(-1) for o in spec["model"]["outputs"]]))
   with np.load(evidence/"arrays.npz",allow_pickle=False) as data:np.testing.assert_allclose(data["reference"],refs,atol=1e-12,rtol=1e-12)
   independent,diagnostics=module.independent_values(spec)
   np.testing.assert_allclose(refs,independent,atol=1e-12,rtol=1e-12)
   item=dict(id=row["id"],batch=batch,regime=spec["regime"],context=spec["context"],
      topology=signature(spec["model"],True),model_without_id_sha256=digest({k:v for k,v in spec["model"].items() if k!="id"}),model_sha256=spec["model_sha256"],folder=str(evidence),
      status=result["status"],failure_layer=result.get("failure_layer"),
      reference_rechecked=True,actual_upstream_trace=True,diagnostics=diagnostics,
      report_sha256=sha(evidence/"report.json"),files=row["files"])
   if result["encrypted_execution"]:
    required={"output/lowered.earth.mlir","output/lowered.ckks.mlir","output/lowered._hecate_golden.hevm",
     "output/_hecate_golden.cst","output/execution.json","output/decrypted.npy","parameters.json","key-cleanup-outcome.json"}
    need(required<=set(row["files"]),"Missing FHE chain")
    execution=strict_file(evidence/"output/execution.json",1024**2)
    parameters=strict_file(evidence/"parameters.json",65536)
    need(all(execution["platform_identity"][k]==plan["platform"][k] for k in ("system","machine","pointer_bits","byteorder")),"Executed platform mismatch")
    need(result["compiled"] and result["encrypted_execution"] and execution["encrypted_execution"] and
      execution["backend"]=="upstream_SEAL_HEVM_CPU" and not execution["bootstrap_executed"] and execution["input_batches"]==4,"Runtime identity")
    need(parameters["seal_version"]=="4.0.0" and parameters["polynomial_degree"]==32768 and
      parameters["security_check"]=="tc128" and parameters["parameters_set"],"Security parameters")
    request=strict_file(evidence/"request.json",8*1024**2);out=evidence/"output"
    gate=inspect_artifacts((out/"lowered._hecate_golden.hevm").read_bytes(),(out/"_hecate_golden.cst").read_bytes(),
      rotation_steps=request_rotations(request),expected_inputs=len(request_input_names(request)),**artifact_options(request["layout"]))
    verify_artifact_configuration(request,gate,sha(PROFILE))
    comparison=compare(np.load(out/"decrypted.npy",allow_pickle=False),np.asarray(refs),1e-5,1e-4)
    need(comparison["passed"]==result["comparison"]["passed"]==(result["status"]=="passed"),"Frozen numerical comparison outcome")
    if not comparison["passed"]:need(result["failure_layer"]=="comparison","Numerical failure layer")
    need((result["comparison"]["atol"],result["comparison"]["rtol"])==(1e-5,1e-4),"Changed threshold")
    need(strict_file(evidence/"key-cleanup-outcome.json",65536)["complete"] and not (evidence/"private-keys").exists(),"Private key cleanup")
    item.update(comparison=comparison,actual_compile=True,actual_ciphertext_execution=True)
   else:
    need(result["failure_layer"] in ("compiler","artifact_gate") and not result["encrypted_execution"],"Unexpected failure scope")
    item.update(actual_compile=result["compiled"],actual_ciphertext_execution=False,diagnostic=result["diagnostic"])
   if spec.get("rotation_placement")=="after":
    need(batch=="stage2-polynomial-postrotation-r85" and spec["context"]==2 and spec["contexts"]=="unscaled" and spec["regime"]=="default" and not spec["post_silu"],"Postrotation diagnostic scope")
    old=next(r for r in rows if r["batch"]=="stage2-polynomial-w40-r73" and r["id"]==item["id"])
    need(old["status"]=="failed" and old["failure_layer"]=="comparison" and old["model_sha256"]==item["model_sha256"],"Original failure model must be preserved")
    need((Path(old["folder"])/"request.json").read_bytes()==(evidence/"request.json").read_bytes(),"Changed model/layout/configuration")
    with np.load(Path(old["folder"])/"arrays.npz",allow_pickle=False) as before, np.load(evidence/"arrays.npz",allow_pickle=False) as after:
     need(set(before.files)==set(after.files),"Changed input/reference arrays")
     for key in before.files:np.testing.assert_array_equal(before[key],after[key])
    ir=(evidence/"output/<string>.mlir").read_text()
    rotations=re.findall(r'(%\d+) = "earth.rotate"\((%\d+)\) <\{offset = array<i64: ([-\d]+)>\}',ir)
    need(len(rotations)==1 and rotations[0][2]=="1","Expected one final rotation")
    dest,operand,_=rotations[0]
    need('"func.return"('+dest+')' in ir and re.search(re.escape(operand)+r' = "earth.add".*loc\("/upstream-poly/poly/MPCB.py":77:0\)',ir),"Actual helper return must feed final rotation")
    need('loc("/upstream-poly/poly/MPCB.py"' in ir and result["calls"]["golden_return_handles"][0]["rotation_placement"]=="after","Observed upstream/postrotation relation")
    unrotated=dict(spec,context=0)
    raw,_=module.independent_values(unrotated)
    np.testing.assert_allclose(np.roll(np.asarray(raw),-1,axis=1),refs,atol=1e-12,rtol=1e-12)
    need(float(np.max(np.abs(np.asarray(raw)-np.asarray(refs))))>1e-8,"Rotation must influence output")
    item["same_model_repair"]=dict(original_folder=old["folder"],original_report_sha256=old["report_sha256"],original_status="failed",
      mathematical_identity="p(rotate(x,1)) == rotate(p(x),1) for this pointwise polynomial",
      identical_inputs_reference_request=True,upstream_helper_unchanged=True,final_rotation_dependency_verified=True,
      current_passing_spelling=True,agent_generated=False,production_lowering_changed=False,
      scope="One original model and four frozen probes; no proof of all CKKS noise draws or all programs")
   rows.append(item)
 coverage={}
 for r in rows:
  key=r["batch"]+":"+r["regime"];coverage.setdefault(key,dict(topologies=set(),passed=0,failed=0))
  coverage[key]["topologies"].add(r["topology"]);coverage[key]["passed" if r["status"]=="passed" else "failed"]+=1
 for key,v in coverage.items():
  required=1 if key=="stage2-polynomial-postrotation-r85:default" else 3
  need(len(v["topologies"])==required,"Distinct contexts required")
  v["topologies"]=sorted(v["topologies"]);v["diagnostic_repair_only"]=required==1
 result=dict(format="poseidon-dependency-acceptance-v1",sources=sources,rows=rows,coverage=coverage,
   same_model_repairs=[r["same_model_repair"] for r in rows if "same_model_repair" in r],execution_attempts=len(rows),unique_model_ids=len({r["id"] for r in rows}),
   distinct_model_definitions_ignoring_ids=len({r["model_without_id_sha256"] for r in rows}),
   actual_ciphertext_executions=sum(r["actual_ciphertext_execution"] for r in rows),encrypted_passed=sum(r["status"]=="passed" for r in rows),
   failed=sum(r["status"]!="passed" for r in rows),skipped=0,negative_trace_controls=negative,
   compared_values=sum(r.get("comparison",{}).get("compared_values",0) for r in rows),
   max_absolute_error_including_failures=max(r.get("comparison",{}).get("max_absolute_error",0.) for r in rows),
   passed_max_absolute_error=max(r.get("comparison",{}).get("max_absolute_error",0.) for r in rows if r["status"]=="passed"),
   all_recorded_outcomes_verified=True,all_cases_passed=all(r["status"]=="passed" for r in rows),
   paid_calls=0,new_fhe_during_audit=0,runner_sha256=sha(Path(__file__)))
 result["binding"]=digest(result);dump(a.output,result)
 print(json.dumps({k:v for k,v in result.items() if k not in ("rows","coverage","sources")}))
 return 0
if __name__=="__main__":raise SystemExit(main())
