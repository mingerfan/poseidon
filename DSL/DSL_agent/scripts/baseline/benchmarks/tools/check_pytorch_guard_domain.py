"""Bounded offline checks of developer guards and program-input approximation domain."""
import argparse, hashlib, importlib.util, json, os, statistics, subprocess, sys, time
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[6] if Path(__file__).resolve().parents[4].name == "DSL_agent" and Path(__file__).resolve().parents[5].name == "DSL" else Path(__file__).resolve().parents[4])
sys.path.insert(0,str(ROOT/"scripts/baseline"))
def run(args):
 from hecate_python_env import WORK, VENV
 from benchmark_graph import canonical, require
 from semantic_benchmark_execution import runtime_sources
 from application_execution.context import WorkerContext
 from application_execution.backend import application
 from component_contract import prepare_task
 from program_execution import pack_samples
 from operator_reference import evaluate
 import numpy as np
 out=args.output.resolve()
 require(not out.exists() and out.parent == (WORK/"results").resolve(),"Fresh guard evidence")
 out.mkdir(mode=0o700)
 started=time.monotonic()
 completed=subprocess.run([str(VENV/"bin/python"),"-B",
  str(ROOT/"scripts/baseline/benchmarks/tools/test_check_pytorch_source_guard.py")],
  stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=90)
 (out/"unit.log").write_text(completed.stdout)
 print(completed.stdout,flush=True)
 require(completed.returncode==0,"Guard unit tests failed")
 spec=importlib.util.spec_from_file_location("pytorch_source_guard",ROOT/"scripts/baseline/benchmarks/tools/check_pytorch_sources.py")
 tool=importlib.util.module_from_spec(spec);spec.loader.exec_module(tool)
 context=WorkerContext.capture(out/"tasks")
 app=application(context)
 sources=runtime_sources();identity=app.backend.identity()
 def old_boundary():
  context.check_current()
  require(runtime_sources()==sources and app.backend.identity()==identity,"Old identity drift")
 def new_boundary():
  tool.check_source_boundary(context,app.backend,sources,identity)
 timings={"previous":[],"current":[]}
 for i in range(2):
  for label,func in (("previous",old_boundary),("current",new_boundary)):
   t=time.perf_counter();func();timings[label].append(time.perf_counter()-t)
   print(json.dumps({"boundary":label,"repeat":i,"seconds":timings[label][-1]}),flush=True)
 timing_report={k:dict(seconds=v,median=statistics.median(v)) for k,v in timings.items()}
 fixture=args.model_import.resolve()
 require(fixture.is_relative_to(WORK/"results") and not fixture.is_symlink(),"Explicit domain fixture")
 original=json.loads((fixture/"original.json").read_text())
 model=json.loads((fixture/"model.json").read_text())
 request=prepare_task(model,dict(chunk_period=4,generation_guidance="explicit-v10",public_weight_policy="descriptors-v1"))
 require(len(original["inputs"])==1 and any(n["op"]=="layer_norm" for n in original["nodes"]),
         "Domain probe requires a captured single-input LayerNorm fixture")
 input_spec=original["inputs"][0]
 inputs={input_spec["name"]:np.resize(np.array([-10.,10.],dtype=np.float64),input_spec["shape"])}
 packed=pack_samples(request,[inputs])
 try:
  evaluate(original,inputs)
  raise AssertionError("Expected approximation-domain rejection")
 except ValueError as error:
  require(str(error)=="Reference input outside explicit approximation domain","Unexpected domain diagnostic")
  domain=dict(status="gap_confirmed",input_packing="accepted",independent_domain_check="rejected",
   diagnostic=str(error),actual_execution=False,numerical_verification=False,
   input_values_excluded_from_public_report=True)
 report=dict(format="poseidon-pytorch-guard-domain-audit-v1",status="passed",unit_tests=6,
  failed=0,skipped=0,boundary_timing=timing_report,approximation_domain=domain,
  paid_calls=0,compiler_calls=0,encrypted_execution=False,seconds=time.monotonic()-started,
  source_hashes=sources,tool_sha256=hashlib.sha256(Path(tool.__file__).read_bytes()).hexdigest())
 (out/"report.json").write_bytes(canonical(report))
 print(json.dumps({k:report[k] for k in ["status","unit_tests","boundary_timing","approximation_domain","paid_calls","compiler_calls","encrypted_execution","seconds"]}),flush=True)
 if args.pending_plan:
  sys.path.insert(0,str(ROOT/"scripts/baseline/benchmarks/tools"))
  import run_application_matrix as live
  path=args.pending_plan.resolve()
  require(path.is_relative_to(ROOT/"docs/baseline"),"Explicit repository proposal")
  plan=json.loads(path.read_text())
  fresh=live.freeze(live.items_from_plan(plan))
  require(fresh==plan,"Pending live proposal binding drift")
  (out/"pending-binding.json").write_bytes(canonical(dict(binding=plan["binding"],cases=len(plan["cases"]),current=True,
   paid_calls=0,source_hashes_changed=0)))
  print("PENDING_FULL_BINDING_CURRENT "+plan["binding"],flush=True)
 require(runtime_sources()==sources and time.monotonic()-started<600,"Guard source drift or wall budget")
 return 0

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument("--output",type=Path,required=True)
 parser.add_argument("--model-import",type=Path,required=True,
                     help="Explicit captured LayerNorm domain fixture")
 parser.add_argument("--pending-plan",type=Path,
                     help="Verify a frozen proposal offline; no provider call")
 return run(parser.parse_args())

if __name__=="__main__":raise SystemExit(main())
