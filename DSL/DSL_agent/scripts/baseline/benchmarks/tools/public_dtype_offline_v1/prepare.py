"""Prepare one bounded offline batch. No provider, compiler or FHE execution."""
import json,os,sys,hashlib,time,shutil,shlex
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[2];ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])
sys.path.insert(0,str(BASE))
from benchmark_graph import digest,require
from hecate_python_env import WORK,VENV,enter_nix
PROPOSAL=ROOT/"docs/baseline/agent-public-dtype-offline-proposal-r1375.json"
OUT=WORK/"results/agent-public-dtype-cast-cpu-v24-v1"
OPTIONS=dict(capacity_packing_version="capacity-public-tensors-v9",generation_guidance="explicit-v13",
 public_weight_policy="descriptors-math-roles-v2",chunk_period=4)
SCOPE=dict(wall_seconds=1200,case_seconds=180,new_bytes=512*1024**2,global_bytes=32*1024**3,
 reserve_bytes=4*1024**3,compiler_jobs=2,link_jobs=1,native_concurrency=1,native_address_bytes=4*1024**3,
 native_cpu_seconds=150,maximum_compiler_calls=36,maximum_encrypted_calls=42,provider_calls=0,http_attempts=0)
REUSE=["f64_to_f32_c0","f64_roundtrip_c2","compat_v23_named"]
def write(path,value):
 with path.open("x")as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write("\n")
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def inside():
 from application_identity import current_identity_manifest
 from application_execution.context import WorkerContext
 from application_execution.backend import application
 from unified_graph_lowering import candidate_source
 from candidate_bundle import check_provenance
 from pytorch_model_intake import run,package_format
 from pytorch_ordered_contract import encode_weight_files
 require(os.environ.get("IN_NIX_SHELL")=="pure"and Path(sys.prefix)==VENV,"Pinned request preparation")
 require(not PROPOSAL.exists()and not OUT.exists(),"Fresh public cast request preparation")
 source=HERE.parent/"public_dtype_cast_v1";controlled=json.loads((source/"isolated-controls.json").read_text())
 checks=json.loads((source/"controls.json").read_text())
 require(checks["tests"]==161 and checks["failed"]==checks["errors"]==checks["skipped"]==0,"Qualified controls")
 sdk=current_identity_manifest()
 require(controlled["passed"]==33 and not any(controlled[k]for k in["failed","skipped","not_run"])and
  sdk["manifest_sha256"]==controlled["sdk_identity"],"Qualified isolated sources")
 source_rows=list(controlled["cases"])
 compat_root=WORK/"results/agent-public-dtype-compat-source-v1";compat_root.mkdir(mode=0o700)
 configs=[("compat_v23_bool","poseidon-trusted-pytorch-v23","bool_to_f32_c0"),
          ("compat_v23_named","poseidon-trusted-pytorch-v23","bool_to_f64_c2"),
          ("compat_v22_mask","poseidon-trusted-pytorch-v22","bool_to_f64_c0")]
 for name,version,parent in configs:
  original=next(r for r in source_rows if r["case"]==parent);spec=json.loads(Path(original["manifest"]).read_text())
  directory=source/"compat-fixtures"/name;directory.mkdir(parents=True)
  parent_dir=Path(original["manifest"]).parent
  for filename in spec["files"]:(directory/filename).write_bytes((parent_dir/filename).read_bytes())
  for w in spec["weights"].values():(directory/w["file"]).write_bytes((parent_dir/w["file"]).read_bytes())
  spec["format"]=version
  if version.endswith("v22"):
   body=(directory/"model.py").read_text()
   first=body.index("    def forward(self,x):")
   last=body.index("\ndef build(",first)
   body=body[:first]+"    def forward(self,x):\n        return x*self.public\n"+body[last:]
   (directory/"model.py").write_text(body);spec["files"]["model.py"]=sha(directory/"model.py")
   spec["outputs"]=["y"]
  manifest=directory/"manifest.json";write(manifest,spec)
  output=compat_root/name;run(manifest,output,resource_profile="general-tensors-v1",trusted=True)
  model=json.loads((output/"model.json").read_text())
  provenance=dict(original=json.loads((output/"original.json").read_text()),
   decomposition=json.loads((output/"decomposition.json").read_text()),
   reference_report=json.loads((output/"reference-report.json").read_text()),
   trusted_pytorch=dict(format=package_format(spec),manifest=spec,
    files={n:(directory/n).read_text()for n in spec["files"]},
    weight_files=encode_weight_files(spec,{w["file"]:(directory/w["file"]).read_bytes()for w in spec["weights"].values()},
                                    resource_profile="general-tensors-v1")))
  check_provenance(provenance,model);write(output/"provenance.json",provenance)
  source_rows.append(dict(case=name,model=str(output/"model.json"),provenance=str(output/"provenance.json"),manifest=str(manifest)))
 require(len(source_rows)==36,"Exact new and compatibility scope")
 import subprocess
 total=int(subprocess.check_output(["du","--count-links","-s","-B1",str(WORK/"results")],text=True,timeout=20).split()[0])
 require(total+SCOPE["new_bytes"]<=SCOPE["global_bytes"]and
  shutil.disk_usage(WORK).free>=SCOPE["reserve_bytes"]+SCOPE["new_bytes"],"Bounded batch disk preflight")
 OUT.mkdir(mode=0o700);rows=[]
 for source_row in source_rows:
  name=source_row["case"];directory=OUT/name;directory.mkdir()
  model=json.loads(Path(source_row["model"]).read_text());provenance=json.loads(Path(source_row["provenance"]).read_text())
  check_provenance(provenance,model)
  store=directory/"tasks";app=application(WorkerContext.capture(store))
  task=app.submit(model,model_provenance=provenance,options=OPTIONS,
    limits=dict(generations=1,http_attempts=1,wall_seconds=180,api_timeout_seconds=1200,max_tokens=384000))
  require(app.status(task)["status"]=="queued","Prepared queued offline task")
  request=app._checked_task_metadata(task)["request"]
  request_path=directory/"request.json";write(request_path,request)
  candidate=directory/"candidate.py";candidate.write_text(candidate_source(request)[0])
  manifest=Path(source_row["manifest"]);spec=json.loads(manifest.read_text())
  files={str(p):sha(p)for p in[request_path,candidate,Path(source_row["model"]),Path(source_row["provenance"]),manifest]}
  for n in spec["files"]:files[str(manifest.parent/n)]=sha(manifest.parent/n)
  for w in spec["weights"].values():files[str(manifest.parent/w["file"])]=sha(manifest.parent/w["file"])
  rows.append(dict(case=name,source_contract=spec["format"],task_id=task,task_store=str(store),
   request=str(request_path),request_id=request["request_id"],candidate=str(candidate),model_sha256=digest(model),files=files))
 require(current_identity_manifest()["manifest_sha256"]==sdk["manifest_sha256"],"Stable source and backend")
 plan=dict(format="poseidon-public-dtype-offline-proposal-v1",backend="poseidon_cpu",scope=SCOPE,options=OPTIONS,
  cases=rows,reuse_cases=REUSE,source_controls_binding=controlled["binding"],controls_binding=checks["binding"],
  sdk_identity=sdk["manifest_sha256"],stage_identities={k:sdk[k]["sha256"]for k in["generation","compilation","execution","validation"]},
  unique_requests=len({r["request_id"]for r in rows}),candidate_origin="deterministic_rule",approved=False,
  compiler_calls=0,encrypted_calls=0,provider_calls=0,http_attempts=0,numeric_tolerance=dict(atol=1e-5,rtol=1e-4))
 plan["binding"]=digest(plan);write(PROPOSAL,plan);write(OUT/"proposal.json",plan)
 print(json.dumps(dict(status="prepared",cases=len(rows),unique_requests=plan["unique_requests"],binding=plan["binding"],
  sdk_identity=sdk["manifest_sha256"],provider_calls=0,compiler_calls=0,encrypted_calls=0)))
if __name__=="__main__":
 if "--inside"in sys.argv:raise SystemExit(inside())
 command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join(
  [str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 raise SystemExit(enter_nix(command,seconds=240,keep_env=(),environment="poseidon-cpu"))
