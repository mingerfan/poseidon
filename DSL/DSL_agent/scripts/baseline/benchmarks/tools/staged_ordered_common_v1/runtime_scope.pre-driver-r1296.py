"""Finite developer scope for staged common-call Poseidon acceptance."""
import hashlib,json,os,shutil,subprocess,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline"
WORK=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux");RESULTS=WORK/"results"
CASES=("alpha","dropout","boolean");PHASES=("qualification","new_inputs","cache")
SCOPE=dict(wall_seconds=1800,phase_seconds=180,artifact_bytes=4*1024**3,
 global_bytes=32*1024**3,reserve_bytes=4*1024**3,provider_calls=0,http_attempts=0,
 native_workers=1,compile_jobs=2,link_jobs=1)
def require(value,message):
 if not value:raise ValueError(message)
def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(value):return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def write(path,value):
 with Path(path).open("xb") as stream:stream.write(canonical(value))
def measure(folder):
 return max(int(subprocess.check_output(["du",arg,"-s","-B1",str(folder)],text=True,timeout=30).split()[0])
            for arg in ("--apparent-size","--count-links"))
def source_binding():
 original=read(ROOT/"docs/baseline/agent-v12-live-proposal-r1232.json")["source_hashes"]
 require(all(sha(ROOT/p)==h for p,h in original.items()),"Frozen production changed")
 draft=read(HERE/"plan.json")
 require(draft["stage"]=="staged_source_and_component_verified" and draft["production_applied"] is False,
         "Source/component acceptance required")
 expected=dict(original)
 for name,item in draft["targets"].items():
  actual=sha(BASE/name) if (BASE/name).exists() else None
  require(actual==item["original_sha256"] and sha(HERE/"targets"/name)==item["target_sha256"],
          "Staged target binding")
  expected["scripts/baseline/"+name]=item["target_sha256"]
 files={str(p):sha(p) for p in (HERE/"runtime_scope.py",HERE/"runtime_acceptance.py",HERE/"plan.json")}
 for name in CASES:
  spec=read(HERE/"fixtures"/name/"manifest.json")
  require(spec["format"]=="poseidon-trusted-pytorch-v13","Reviewed V13 source")
  selected={"manifest.json",*spec["files"],*(item["file"] for item in spec["weights"].values())}
  require(set(p.name for p in (HERE/"fixtures"/name).iterdir())==selected,"Fixed source inventory")
  for filename in selected:
   p=HERE/"fixtures"/name/filename
   require(p.is_file() and not p.is_symlink(),"Regular frozen source fixture")
   files[str(p)]=sha(p)
  require(all(sha(HERE/"fixtures"/name/p)==h for p,h in spec["files"].items()),"Fixture source hash")
  require(all((HERE/"fixtures"/name/i["file"]).stat().st_size==i["bytes"] and
              sha(HERE/"fixtures"/name/i["file"])==i["sha256"] for i in spec["weights"].values()),
          "Fixture public weight binding")
 protected={}
 for name in ("protected-backend.json","protected-references.json"):
  path=RESULTS/"agent-common-pytorch-offline-prepared-r1040"/name
  record=read(path);require(all(sha(ROOT/p)==h for p,h in record.items()),"Protected backend/reference")
  protected.update(record);files[str(path)]=sha(path)
 reference=RESULTS/"agent-ordered-common-tests-r1248/report.json"
 verified=read(reference)
 require(verified["status"]=="passed" and verified["tests"]==9 and
         verified["failed"]==verified["errors"]==verified["skipped"]==0,"Verified source fixtures")
 rows={r["case"]:r for r in verified["source_capture_rows"]}
 require(set(CASES)<=set(rows),"Finite source model inventory")
 for name in CASES:require(rows[name]["source_package_sha256"]==
                          hashlib.sha256(canonical(read(HERE/"fixtures"/name/"manifest.json"))).hexdigest(),
                          "Same mathematical source package")
 files[str(reference)]=sha(reference)
 return dict(original_production=original,inside_sources=expected,protected_sources=protected,
             files=files,source_reference={n:rows[n] for n in CASES},source_reference_sha256=sha(reference))
def plan():
 data=dict(format="poseidon-ordered-common-offline-plan-v1",cases=list(CASES),phases=list(PHASES),
   scope=SCOPE,bindings=source_binding(),candidate_origin="rule_baseline",
   new_agent_generation=False,production_applied=False,bootstrap=False,cuda=False,
   resource_profile="general-tensors-v1",generation_guidance="explicit-v11",
   capacity_packing_version="capacity-public-tensors-v9",public_weight_policy="descriptors-v1")
 data["binding"]=hashlib.sha256(canonical(data)).hexdigest()
 return data
def check_plan(value):
 require(value["binding"]==hashlib.sha256(canonical({k:v for k,v in value.items() if k!="binding"})).hexdigest(),
         "Offline plan integrity")
 require(value["cases"]==list(CASES) and value["phases"]==list(PHASES) and value["scope"]==SCOPE,
         "Finite offline scope")
 for path,want in value["bindings"]["files"].items():require(sha(path)==want,"Frozen tool/fixture changed")
def expected_namespace(value):
 check_plan(value)
 expected=value["bindings"]["inside_sources"]
 require(all((ROOT/n).is_file() and sha(ROOT/n)==h for n,h in expected.items()),"Read-only staged source changed")
 for name,want in value["bindings"]["protected_sources"].items():require(sha(ROOT/name)==want,"Backend/reference changed")
 return expected
def resource_boundary(output,value):
 require(time.time()<value["deadline_wall"],"Offline time boundary")
 require(shutil.disk_usage(WORK).free>=SCOPE["reserve_bytes"],"Offline disk reserve")
 dirs=[p for p in RESULTS.iterdir() if p.name not in value["initial_result_directories"] or p==Path(output)]
 require(sum(measure(p) for p in dirs)<=SCOPE["artifact_bytes"],"Offline artifact boundary")
 require(measure(RESULTS)<=SCOPE["global_bytes"],"Global artifact boundary")
