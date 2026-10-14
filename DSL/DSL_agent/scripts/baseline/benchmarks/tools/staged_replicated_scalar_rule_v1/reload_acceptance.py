"""Explicit transport repair and original-checker package acceptance."""
import copy,hashlib,importlib.util,json,math,shutil,sys,time,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[2]
MHA=HERE.parent/"staged_mha_ordered_graph_v1"
sys.path.insert(0,str(BASE));sys.path.insert(0,str(MHA/"overlay"));sys.path.insert(0,str(HERE))
import numpy as np
from benchmark_graph import canonical,digest,require
from benchmark_math import evaluate
from candidate_bundle import check_bundle,load_model_provenance
from numerical_results import compare
from application_backend import PoseidonBackend
from hecate_python_env import WORK
from pytorch_ordered_contract import source_identity
from pytorch_model_intake import capture_binding
from ordered_provenance_transport import repack
NAME="boolean_padding_residual__declared_shallow_v1"
out=WORK/"results/agent-mha-ordered-transport-reload-v2"
require(not out.exists(),"Preserve prior evidence")
out.mkdir()
started=time.monotonic()
backend=PoseidonBackend();identity=backend.identity();stages=backend.identity_manifest();capture_identity=source_identity()
oldroot=WORK/"results/agent-mha-replicated-scalar-backend-v1"
original_report=json.loads((oldroot/"report.json").read_bytes())
require(original_report["status"]=="complete","Safe terminal boundary")
paths=[Path(__file__),HERE/"ordered_provenance_transport.py",HERE/"tests/test_ordered_provenance_transport.py",MHA/"analyze_schedule_compaction.py",HERE/"tests/test_literal_vector_authority.py",HERE/"replicated_scalar_lowering.py"]
tools={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
capture_folder=WORK/"results/agent-mha-ordered-capture-v2"/NAME
capture_files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in capture_folder.rglob("*") if p.is_file() and not p.is_symlink()}
limits=dict(wall_seconds=600,new_bytes=4*1024**3,total_bytes=32*1024**3,free_bytes=4*1024**3,execution_concurrency=1)
known={p.name for p in (WORK/"results").iterdir()}
def total(p):return sum(f.stat().st_size for f in p.rglob("*") if f.is_file() and not f.is_symlink())
def boundary():
 require(time.monotonic()-started<limits["wall_seconds"],"Transport window time boundary")
 require(total(WORK/"results")<=limits["total_bytes"] and shutil.disk_usage(WORK).free>=limits["free_bytes"],"Artifact global boundary")
 require(sum(total(p) for p in (WORK/"results").iterdir() if p.is_dir() and p.name not in known)+total(out)<=limits["new_bytes"],"New artifact boundary")
 require(backend.identity()==identity and backend.identity_manifest()==stages and source_identity()==capture_identity,"Dependency/source drift")
 require(all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==s for p,s in tools.items()),"Transport tool drift")
 require(all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==s for p,s in capture_files.items()),"Frozen capture file drift")
spec=importlib.util.spec_from_file_location("test_transport",paths[2])
tests=importlib.util.module_from_spec(spec);spec.loader.exec_module(tests)
suite=unittest.defaultTestLoader.loadTestsFromModule(tests)
extra=importlib.util.spec_from_file_location("test_literal_authority",paths[4])
extra_tests=importlib.util.module_from_spec(extra);extra.loader.exec_module(extra_tests)
suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(extra_tests))
# Test modules can declare the base import path; restore the explicit staging
# selection before their class fixtures perform complete source reconstruction.
selected_paths=[str(HERE),str(MHA/"overlay"),str(BASE)]
sys.path[:]=selected_paths+[p for p in sys.path if p not in selected_paths]
with (out/"controls.log").open("w") as log:
 control=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
report=dict(format="poseidon-ordered-provenance-transport-acceptance-v1",status="running",source_kind="deterministic_rule_baseline",
 controls=dict(ran=control.testsRun,passed=control.testsRun-len(control.failures)-len(control.errors)-len(control.skipped),failed=len(control.failures),errors=len(control.errors),skipped=len(control.skipped)),
 tool_sha256=tools,capture_file_sha256=capture_files,limits=limits,backend_identity=identity,stage_identity=stages,source_identity=capture_identity,
 prior_report=dict(path=str(oldroot/"report.json"),sha256=hashlib.sha256((oldroot/"report.json").read_bytes()).hexdigest(),binding=original_report["binding"],status=original_report["status"],passed=original_report["passed"],failed=original_report["failed"]),
 api_calls=0,http_attempts=0,production_applied=False,backend_modified=False,models_changed=False,reference_changed=False,
 numerical_tolerance=dict(atol=1e-5,rtol=1e-4),compiler_reexecuted=False,new_agent_generation=False,original_accuracy_certified=False)
def save():
 report["binding"]=digest({k:v for k,v in report.items() if k!="binding"})
 (out/"report.json").write_bytes(canonical(report))
save()
if not control.wasSuccessful():
 print((out/"controls.log").read_text(),flush=True);report["status"]="control_failed";save();raise SystemExit(1)
boundary()
try:
 previous=oldroot/NAME
 qualification=json.loads((previous/"qualification.json").read_bytes())
 require(qualification["status"]=="passed" and qualification["numerically_validated"] is True,"Unchanged qualification is numerical")
 model=json.loads((previous/"request.json").read_bytes())["model"]
 capture=WORK/"results/agent-mha-ordered-capture-v2"/NAME
 provenance=load_model_provenance(capture,model,max_bytes=4*1024**2)
 package=out/"program"
 manifest,transport=repack(previous/"program",package,provenance)
 require(check_bundle(package)==manifest,"Original package checker")
 report.update(transport=transport,package_binding=manifest["binding"],package_check_passed=True,source_qualification=qualification)
 save()
 samples=[]
 for group in range(3):
  sample={}
  for index,item in enumerate(model["inputs"]):
   n=math.prod(item["shape"])
   value=(np.arange(n,dtype=np.float64)+1)/(n+1)*.35
   if group==0:value=-value
   if group==2:value=np.cos(np.arange(n,dtype=np.float64)+index*.19)*.27
   sample[item["name"]]=value.reshape(item["shape"]).tolist()
  samples.append(sample)
 executions=[]
 for title,inputs in [("new_input",samples),("repeated_input",[samples[0]])]:
  boundary()
  require(check_bundle(package)==manifest,"Independent package reload")
  execution=backend.execute(package,inputs)
  observed=np.array([[v for output in model["outputs"] for v in np.asarray(row[output["name"]]).reshape(-1)] for row in execution["outputs"]])
  expected=np.array([[v for output in model["outputs"] for v in evaluate(model,{k:np.asarray(v,dtype=np.float64) for k,v in row.items()})[output["name"]].reshape(-1)] for row in inputs])
  comparison=compare(observed,expected,1e-5,1e-4)
  require(comparison["passed"],"Frozen numerical gate")
  (out/(title+".json")).write_bytes(canonical(dict(execution=execution,comparison=comparison)))
  executions.append(dict(kind=title,groups=len(inputs),comparison={k:comparison[k] for k in ("passed","mae","max_absolute_error","compared_values")},evidence=execution["evidence"],resource_precheck_binding=execution["resource_precheck"]["binding"]))
  report["executions"]=executions;save()
  print(title,"passed",comparison["max_absolute_error"],flush=True)
 boundary()
 report.update(status="passed",reloaded=True,new_input_passed=True,repeated_execution=True,elapsed_seconds=time.monotonic()-started)
 save()
 print("transport_acceptance_passed",report["binding"],flush=True)
except Exception as e:
 import traceback
 report.update(status="failed",diagnostic=str(e),traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-started);save()
 print(report["traceback"],flush=True);raise SystemExit(1)
