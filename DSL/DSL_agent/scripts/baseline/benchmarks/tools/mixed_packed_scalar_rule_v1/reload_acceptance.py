"""Declared output-tree followup and actual package repeats; no recompilation."""
import hashlib,json,math,shutil,sys,time,traceback,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[2];MHA=HERE.parent/"staged_mha_ordered_graph_v1"
sys.path[:]=[str(HERE),str(MHA/"overlay"),str(BASE)]+sys.path
import numpy as np
from benchmark_graph import canonical,digest,require
from benchmark_math import evaluate
from application_backend import PoseidonBackend
from candidate_bundle import check_bundle
from hecate_python_env import WORK,ROOT
from numerical_results import compare
from pytorch_ordered_contract import source_identity
from backend_acceptance import declared_structure
from program_execution import package_output_structure
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def total(root):return sum(p.stat().st_size for p in root.rglob("*") if p.is_file() and not p.is_symlink())
def inputs_for(model):
 samples=[]
 for group in range(3):
  sample={}
  for item in model["inputs"]:
   n=math.prod(item["shape"]);v=(np.arange(n,dtype=np.float64)+1)/(n+1)*.35
   if group==0:v=-v
   if group==2:v=np.cos(np.arange(n,dtype=np.float64))*.27
   sample[item["name"]]=v.reshape(item["shape"]).tolist()
  samples.append(sample)
 return samples
def compare_execution(execution,model,inputs,tree):
 require(execution["status"]=="executed" and execution["encrypted_execution"] and execution["executor_secret_key"] is False,"Trusted execution layers")
 require(execution["samples"]==len(inputs) and execution["output_structure"]==tree,"Input/output scope")
 require(canonical([declared_structure(tree,row) for row in execution["outputs"]])==canonical(execution["structured_outputs"]),"Declared output restoration")
 observed=np.array([[v for o in model["outputs"] for v in np.asarray(row[o["name"]]).reshape(-1)] for row in execution["outputs"]])
 expected=np.array([[v for o in model["outputs"] for v in evaluate(model,{k:np.asarray(v,dtype=np.float64) for k,v in row.items()})[o["name"]].reshape(-1)] for row in inputs])
 r=compare(observed,expected,1e-5,1e-4);require(r["passed"],"Frozen numerical gate")
 return {k:r[k] for k in ("passed","mae","max_absolute_error","compared_values")}
def main():
 out=WORK/"results/agent-mha-mixed-package-reload-v1";require(not out.exists(),"Preserve old followup")
 previous=WORK/"results/agent-mha-mixed-packed-scalar-backend-v2"
 prior=json.loads((previous/"report.json").read_bytes());require(prior["status"]=="complete","Terminal source boundary")
 prior_control=json.loads((WORK/"results/agent-mha-mixed-packed-scalar-backend-v1/report.json").read_bytes())
 limits=dict(wall_seconds=min(600,int(1200-prior["elapsed_seconds"]-prior_control["elapsed_seconds"])),new_bytes=4*1024**3,total_bytes=32*1024**3,free_bytes=4*1024**3,concurrency=1)
 require(limits["wall_seconds"]>0,"Original cumulative window exhausted")
 backend=PoseidonBackend();identity=backend.identity();stages=backend.identity_manifest();capture=source_identity()
 files={str(p):sha(p) for p in HERE.rglob("*") if p.is_file() and not p.is_symlink() and "__pycache__" not in p.parts}
 interpreter=MHA/"analyze_schedule_compaction.py";files[str(interpreter)]=sha(interpreter)
 historic=WORK/"results/agent-mha-ordered-transport-reload-v2/report.json"
 recorded=json.loads(historic.read_bytes())["tool_sha256"]
 require(recorded[str(interpreter)]==sha(interpreter),"Original independent AST interpreter changed")
 report=dict(format="poseidon-mixed-package-reload-followup-v1",status="running",limits=limits,
  stage_identity=stages,backend_identity=identity,source_capture_implementation=capture,tool_files=files,
  prior_report=dict(path=str(previous/"report.json"),sha256=sha(previous/"report.json"),binding=prior["binding"],passed=prior["passed"],failed=prior["failed"]),
  previous_record_corrections=dict(repeated_execution_flag="Prior single package call does not prove repeated execution",cross_failure="KeyError weights: declared output is result and aux [None, projected]"),
  cases=[],api_calls=0,http_attempts=0,compiler_calls=0,backend_modified=False,reference_changed=False,
  new_agent_generation=False,production_applied=False,numerical_tolerance=dict(atol=1e-5,rtol=1e-4))
 started=time.monotonic();known={p.name for p in (WORK/"results").iterdir()};out.mkdir()
 def save():
  report["binding"]=digest({k:v for k,v in report.items() if k!="binding"});(out/"report.json").write_bytes(canonical(report))
 def boundary():
  require(time.monotonic()-started<limits["wall_seconds"],"Followup cumulative time boundary")
  require(sha(previous/"report.json")==report["prior_report"]["sha256"],"Historical result drift")
  require(all(sha(p)==h for p,h in files.items()),"Followup frozen source drift")
  require(backend.identity()==identity and backend.identity_manifest()==stages and source_identity()==capture,"Frozen dependency drift")
  require(total(WORK/"results")<=limits["total_bytes"] and shutil.disk_usage(WORK).free>=limits["free_bytes"],"Global disk boundary")
  require(sum(total(p) if p.is_dir() else p.stat().st_size for p in (WORK/"results").iterdir() if p.name not in known)<=limits["new_bytes"],"New artifact boundary")
 save()
 try:
  boundary()
  # These representation controls are neither FHE nor numerical evidence.
  class OutputTreeControls(unittest.TestCase):
   def test_none_dict_list_and_tuple(self):
    tree=dict(kind="dict",items=[dict(key="result",value=dict(kind="tensor",name="result")),dict(key="aux",value=dict(kind="list",items=[dict(kind="none"),dict(kind="tensor",name="projected")]))])
    self.assertEqual(declared_structure(tree,dict(result=[1.],projected=[2.])),dict(result=[1.],aux=[None,[2.]]))
   def test_scalar(self):self.assertEqual(declared_structure(dict(kind="tensor",name="x",shape=[]),dict(x=[2.])),2.)
   def test_unsupported_tree(self):
    with self.assertRaises(ValueError):declared_structure(dict(kind="callable"),{})
  with (out/"controls.log").open("w") as log:tests=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(OutputTreeControls))
  report["controls"]=dict(ran=tests.testsRun,passed=tests.testsRun-len(tests.failures)-len(tests.errors),failed=len(tests.failures),errors=len(tests.errors),skipped=len(tests.skipped));save()
  require(tests.wasSuccessful(),"Tree controls failed")
  threshold=(previous/"controls.log").stat().st_mtime-2
  newreports=[]
  for path in (WORK/"results").iterdir():
   f=path/"report.json"
   if path.name.startswith("poseidon-program-execution-") and f.exists() and f.stat().st_mtime>=threshold:
    newreports.append((path,json.loads(f.read_bytes())))
  for case in prior["cases"]:
   boundary();require(case["qualification_status"]=="passed","Numerical qualification required")
   package=previous/case["id"]/"program";manifest=check_bundle(package)
   request=json.loads((previous/case["id"]/"request.json").read_bytes());model=request["model"]
   tree=package_output_structure(package,manifest,model);samples=inputs_for(model)
   matched=[(p,v) for p,v in newreports if v["bundle_binding"]==manifest["binding"]]
   require(len(matched)==1,"Unique retained new-input execution")
   ep,execution=matched[0]
   old_comparison=compare_execution(execution,model,samples,tree)
   row=dict(id=case["id"],status="running",package_binding=manifest["binding"],qualified_evidence=case["evidence"],
     retained_new_input_comparison=old_comparison,retained_new_input_evidence=str(ep),declared_structure=tree)
   report["cases"].append(row);save()
   require(check_bundle(package)==manifest,"Independent reload")
   repeated=backend.execute(package,[samples[0]])
   repeat_comparison=compare_execution(repeated,model,[samples[0]],tree)
   (out/(case["id"]+".json")).write_bytes(canonical(dict(retained_new_input_comparison=old_comparison,repeat_execution=repeated,repeat_comparison=repeat_comparison)))
   row.update(status="passed",repeat_comparison=repeat_comparison,repeat_evidence=repeated["evidence"],reloaded=True,repeated_execution=True)
   save();print(case["id"],"reload/repeat passed",repeat_comparison["max_absolute_error"],flush=True)
  boundary();report.update(status="passed",passed=len(report["cases"]),failed=0,elapsed_seconds=time.monotonic()-started);save()
  print("followup_passed",report["binding"],flush=True);return 0
 except Exception as e:
  report.update(status="failed",diagnostic=str(e),traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-started);save();print(report["traceback"],flush=True);return 1
if __name__=="__main__":raise SystemExit(main())
