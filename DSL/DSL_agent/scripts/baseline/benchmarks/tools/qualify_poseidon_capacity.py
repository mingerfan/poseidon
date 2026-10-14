"""Bounded Poseidon CPU numerical capacity and legacy-interface regressions."""
import argparse,contextlib,json,shlex,sys,time,shutil
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[6] if Path(__file__).resolve().parents[4].name == "DSL_agent" and Path(__file__).resolve().parents[5].name == "DSL" else Path(__file__).resolve().parents[4])
sys.path[:0]=[str(ROOT/"scripts/baseline"),str(Path(__file__).resolve().parent)]
def main():
 from hecate_python_env import WORK,VENV,enter_nix
 p=argparse.ArgumentParser();p.add_argument("--inside",action="store_true");p.add_argument("--output",type=Path,required=True);p.add_argument("--ids",nargs="+");p.add_argument("--suite",choices=("capacity","linear","diagonal","tiles","tensor","tensor-composition","tensor-collision","spatial","stream","broadcast","extended"),default="capacity")
 a=p.parse_args()
 if not a.inside:
  cmd='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join(
    [str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside","--output",str(a.output),"--suite",a.suite]+(["--ids",*a.ids] if a.ids else []))
  return enter_nix(cmd,seconds=1900,environment="poseidon-cpu")
 from unified_capacity_cases import models
 from benchmark_graph import require,canonical,digest
 from component_contract import prepare_task
 from component_backend import qualify
 from unified_graph_lowering import candidate_source
 from candidate_bundle import export_bundle,replay_bundle
 from semantic_benchmark_execution import runtime_sources
 from poseidon_backend_identity import identity
 from check_poseidon_helpers import snapshot
 require(not a.output.exists(),"Fresh acceptance")
 a.output.mkdir();frozen=runtime_sources();backend=identity();begin=time.monotonic()
 case_files={}
 if a.suite in ("stream","broadcast","extended"):
  import hashlib
  from benchmark_runner import strict_file
  directory=ROOT/"scripts/baseline/cases"/({"stream":"capacity-stream-v1","broadcast":"capacity-cipher-broadcast-v1","extended":"capacity-extended-v1"}[a.suite])
  index=directory/"manifest.json";manifest=strict_file(index,131072)
  require(set(manifest)=={"format","cases"} and manifest["format"]==("poseidon-capacity-regression-list-v1" if a.suite=="stream" else "poseidon-capacity-regression-list-v2")
          and type(manifest["cases"]) is list and 1<=len(manifest["cases"])<=48,"Capacity manifest")
  case_files[str(index)]=hashlib.sha256(index.read_bytes()).hexdigest();graph_cases=[]
  for entry in manifest["cases"]:
   require(set(entry)==({"id","file","sha256"} if a.suite=="stream" else {"id","file","sha256","chunk_period"}) and Path(entry["file"]).name==entry["file"]
           and entry["file"].endswith(".json"),"Capacity case fields")
   path=directory/entry["file"];require(not path.is_symlink(),"Capacity case symlink")
   require(hashlib.sha256(path.read_bytes()).hexdigest()==entry["sha256"],"Capacity case hash")
   model=strict_file(path,131072);require(model["id"]==entry["id"],"Capacity model id")
   require(entry["id"] not in {g["id"] for g,_,_ in graph_cases},"Duplicate capacity case")
   case_files[str(path)]=entry["sha256"]
   period=entry.get("chunk_period",256)
   require(type(period) is int and period in (4,8,16,32,64,128,256),"Capacity case period")
   graph_cases.append((model,{"chunk_period":period},"capacity_composition"))
 elif a.suite=="spatial":
  from unified_capacity_spatial_cases import models as spatial_models
  from unified_capacity_tensor_cases import models as tensor_models
  from unified_capacity_tiled_cases import models as tiled_models
  graph_cases=[(g,{"chunk_period":256},"capacity_composition" if "composition" in g["id"] else "capacity") for g in spatial_models()]
  graph_cases.extend((g,{"chunk_period":256},"capacity_regression") for g in tensor_models()
                    if g["id"] in {"capacity_stack_axes_3","capacity_mean_axes_1"})
  graph_cases.extend((g,{"chunk_period":256},"capacity_regression") for g in tiled_models()
                    if g["id"]=="capacity_batched_linear_2")
 elif a.suite=="tensor-collision":
  g=json.loads((ROOT/"scripts/baseline/cases/capacity-tensor-collision-v1/capacity_tensor_name_collision.json").read_text())
  graph_cases=[(g,{"chunk_period":256},"capacity_composition")]
  from unified_capacity_tensor_cases import models as tensor_models
  graph_cases.extend((g,{"chunk_period":256},"capacity_regression") for g in tensor_models() if g["id"]=="capacity_stack_axes_3")
 elif a.suite=="tensor-composition":
  g=json.loads((ROOT/"scripts/baseline/cases/capacity-tensor-composition-v1/capacity_tensor_multi_output.json").read_text())
  graph_cases=[(g,{"chunk_period":256},"capacity_composition")]
 elif a.suite=="tensor":
  from unified_capacity_tensor_cases import models as tensor_models
  from unified_capacity_tiled_cases import models as tiled_models
  graph_cases=[(g,{"chunk_period":256},"capacity") for g in tensor_models()]
  graph_cases.extend((g,{"chunk_period":256},"capacity_regression") for g in tiled_models() if g["id"]=="capacity_batched_linear_2")
  graph_cases.extend((g,{"chunk_period":256},"capacity_regression") for g in models() if g["id"]=="capacity_mean_tail_2")
 elif a.suite=="tiles":
  from unified_capacity_tiled_cases import models as tiled_models
  from unified_capacity_diagonal_cases import models as diagonal_models
  from unified_capacity_linear_cases import models as linear_models
  graph_cases=[(g,{"chunk_period":256},"capacity") for g in tiled_models()]
  graph_cases.extend((g,{"chunk_period":256},"capacity_regression") for g in diagonal_models() if g["id"].endswith("_4"))
  graph_cases.extend((g,{"chunk_period":256},"capacity_regression") for g in linear_models()
                    if g["id"] in {"capacity_public_tensor_2","capacity_linear_0"})
 elif a.suite=="diagonal":
  from unified_capacity_diagonal_cases import models as diagonal_models
  from unified_capacity_linear_cases import models as linear_models
  graph_cases=[(g,{"chunk_period":256},"capacity") for g in diagonal_models()]
  graph_cases.extend((g,{"chunk_period":256},"capacity_regression") for g in linear_models()
                    if g["id"] in {"capacity_public_tensor_2","capacity_linear_0","capacity_linear_square_linear_1"})
 elif a.suite=="linear":
  from unified_capacity_linear_cases import models as linear_models
  graph_cases=[(g,{"chunk_period":256},"capacity") for g in linear_models()]
  selected={"capacity_mean_tail_2","capacity_scalar_broadcast_0","capacity_elementwise_1","capacity_join_aligned_2"}
  graph_cases.extend((g,{"chunk_period":256},"capacity_regression") for g in models() if g["id"] in selected)
 else:
  graph_cases=[(g,{"chunk_period":256},"capacity") for g in models()]
  auto=__import__("copy").deepcopy(models()[0]);auto["id"]="capacity_auto_p128"
  graph_cases.append((auto,{},"capacity_auto"))
 if a.suite=="extended":
  import copy
  from unified_public_contract import CONTRACT
  g=copy.deepcopy(graph_cases[0][0]);g["id"]="capacity_extended_public_0"
  graph_cases.append((g,{"chunk_period":256,"construction_profile":CONTRACT},"capacity_composition"))
 for name in ["linear","linear_square_linear"]:
  old=WORK/"results/poseidon-historical-agent-replay-r371"/name/"program/model.json"
  graph_cases.append((json.loads(old.read_text()),{},"legacy"))
 if a.ids:
  require(len(set(a.ids))==len(a.ids) and set(a.ids)<={g["id"] for g,_,_ in graph_cases},"Frozen capacity selection")
  graph_cases=[row for row in graph_cases if row[0]["id"] in a.ids]
 rows=[dict(id=g["id"],status="not_run",category=category) for g,_,category in graph_cases]
 plan=dict(format="poseidon-capacity-numerical-plan-v1",models=[g for g,_,_ in graph_cases],case_files=case_files,
  developer_sha256=__import__("hashlib").sha256(Path(__file__).read_bytes()).hexdigest(),
  source_hashes=frozen,application_source_hashes=snapshot(),backend_identity=backend,
  limits=dict(seconds=1800,native_workers=1,compile_jobs=2,link_jobs=1,artifact_bytes=1024**3,
              global_bytes=32*1024**3,free_bytes=4*1024**3),origin_category="rule_baseline",paid_calls=0)
 (a.output/"plan.json").write_bytes(canonical(plan))
 def save():
  summary=dict(format="poseidon-capacity-numerical-v1",binding=digest(plan),planned=len(rows),rows=rows,
   passed=sum(r["status"]=="passed" for r in rows),failed=sum(r["status"]=="failed" for r in rows),
   not_run=sum(r["status"]=="not_run" for r in rows),blocked=sum(r["status"]=="blocked" for r in rows),skipped=0,paid_calls=0,agent_generated=False,
   seconds=time.monotonic()-begin)
  (a.output/"report.json").write_bytes(canonical(summary))
 def boundary():
  require(runtime_sources()==frozen and identity()==backend,"Frozen capacity binding changed")
  import hashlib
  require(hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==plan["developer_sha256"]
          and all(hashlib.sha256(Path(path).read_bytes()).hexdigest()==h for path,h in case_files.items()),"Capacity developer/case drift")
  require(time.monotonic()-begin<1800 and shutil.disk_usage(WORK).free>=4*1024**3,"Capacity time/free-space boundary")
  import subprocess
  total=int(subprocess.check_output(["du","-sb",str(WORK/"results")],timeout=30,text=True).split()[0])
  require(total<=32*1024**3,"Global artifact budget")
 save()
 for i,(g,opts,category) in enumerate(graph_cases):
  boundary();row=rows[i];folder=a.output/g["id"];folder.mkdir();start=time.monotonic()
  try:
   req=prepare_task(g,opts);src,strategy=candidate_source(req)
   if category=="capacity_composition":
    import numpy as np
    from benchmark_graph import samples
    from benchmark_math import evaluate as math_ref
    from benchmark_torch import evaluate as torch_ref
    from test_unified_capacity_linear import interpreted_output
    for probe in samples(g,16):
     first,second=math_ref(g,probe),torch_ref(g,probe)
     for name in first:np.testing.assert_allclose(first[name],second[name],atol=1e-12,rtol=1e-12)
     expected=np.concatenate([first[o["name"]].reshape(-1) for o in g["outputs"]])
     np.testing.assert_allclose(interpreted_output(src,req,probe),expected,atol=1e-12,rtol=1e-12)
    row["independent_reference_probes"]=16
   (folder/"model.json").write_bytes(canonical(g));(folder/"request.json").write_bytes(__import__("candidate_contract").request_file_bytes(req))
   (folder/"candidate.py").write_text(src)
   result=qualify(req,dict(schema=1,request_id=req["request_id"],hecate_source=src))
   row.update(result);row["backend_terminal_status"]=result["status"]
   row["status"]="passed" if result["status"]=="passed" else "failed"
   row.update(strategy=strategy,request_id=req["request_id"],physical_input_chunks=len(req["layout"]["inputs"]),
       physical_output_chunks=req["layout"]["output_ciphertexts"],input_elements=sum(__import__("math").prod(x["shape"]) for x in g["inputs"]),
       output_elements=len(req["layout"]["output_selectors"]))
   # One reload in each added semantic partition, at its largest context.
   if row["status"]=="passed" and category in ("capacity","capacity_composition") and (category=="capacity_composition" or g["id"].endswith("_2") or (a.suite=="diagonal" and g["id"].endswith("_4"))):
    package=folder/"program";manifest=export_bundle(Path(result["evidence"]),package)
    before=set((WORK/"results").glob("poseidon-package-replay-*"))
    with (folder/"reload.log").open("w") as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
     require(replay_bundle(package)==0,"Capacity program numerical reload")
    added=set((WORK/"results").glob("poseidon-package-replay-*"))-before;require(len(added)==1,"Unique capacity reload")
    reload=next(iter(added));rr=json.loads((reload/"report.json").read_text())
    require(rr["status"]=="passed" and rr["encrypted_execution"] and rr["agent_calls"]==rr["compiler_calls"]==0,"Capacity repeated FHE")
    row.update(program=str(package),package_binding=manifest["binding"],reload_evidence=str(reload),
       reload_comparison=rr["comparison"],package_generation_calls=0,package_compiler_calls=0)
  except (ValueError,AssertionError,KeyError,TypeError,OSError) as e:
   if str(e) in ("Capacity public tensor registry budget","Unified constant registry budget","Capacity spatial map work budget","Capacity spatial route budget"):
    row.update(status="blocked",failure=dict(layer="request_preparation",code="public_binding_budget",
                 diagnostic=str(e),owner="capacity_layout",retry_policy="never_automatic"))
   else:row.update(status="failed",diagnostic=str(e)[:2000])
  row["seconds"]=time.monotonic()-start
  print("CAPACITY",g["id"],row["status"],row.get("failure",row.get("diagnostic")),flush=True)
  save();boundary()
  trace=Path(row["evidence"])/"attempt-00/trace.log" if row.get("evidence") else None
  startup_failure=bool(trace and trace.is_file() and "ModuleNotFoundError:" in trace.read_text())
  if startup_failure:
   row["shared_environment_fault"]=True;save()
  if row.get("failure",{}).get("layer") in ("integrity","environment","sandbox") or startup_failure:
   raise ValueError("Shared capacity infrastructure failure; original records retained")
 paths={a.output}
 for row in rows:
  for key in ["evidence","reload_evidence"]:
   if row.get(key):paths.add(Path(row[key]))
 used=sum(p.stat().st_size for d in paths for p in d.rglob("*") if p.is_file())
 require(used<=1024**3,"Capacity artifact boundary")
 summary=json.loads((a.output/"report.json").read_text())
 summary.update(artifact_bytes=used,compared_values=sum(r.get("compared_values",0) for r in rows if r["status"]=="passed"),
  max_absolute_error=max((r.get("max_absolute_error",0) for r in rows if r["status"]=="passed"),default=0),
  reload_values=sum(r.get("reload_comparison",{}).get("compared_values",0) for r in rows),
  reload_max_absolute_error=max((r.get("reload_comparison",{}).get("max_absolute_error",0) for r in rows),default=0))
 (a.output/"report.json").write_bytes(canonical(summary))
 return int(any(r["status"]!="passed" for r in rows))
if __name__=="__main__":raise SystemExit(main())
