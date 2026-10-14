"""Batch six original-source captures and draft integrity controls; no Provider or compiler."""
import io,json,os,shlex,sys,time,traceback,unittest,shutil,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline"
sys.path.insert(0,str(BASE))
os.environ.update(POSEIDON_PLATFORM="aarch64-linux",POSEIDON_WORK_ROOT="/home/lhohy/poseidon-work")
OUT=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-public-tensor-source-draft-v1")
def measure(path):
 return max(int(subprocess.check_output(["du",arg,"-s","-B1",str(path)],text=True,timeout=30).split()[0])for arg in("--apparent-size","--count-links"))
def inside():
 from benchmark_graph import digest
 from sandbox import run,draft_identity
 from hecate_python_env import VENV
 from pytorch_ordered_contract import source_identity
 from public_tensor_division import sha
 require=lambda ok,code:None if ok else (_ for _ in()).throw(ValueError(code))
 require(os.environ.get("IN_NIX_SHELL")=="pure"and Path(sys.prefix)==VENV,"Pinned staging controls")
 require(not OUT.exists(),"Fresh draft source run")
 OUT.mkdir(mode=0o700);started=time.monotonic();frozen=source_identity();draft=draft_identity()
 plan=dict(format="poseidon-public-tensor-source-draft-plan-v1",scope="isolated source capture and host revalidation",
  source_hashes=frozen,draft_hashes=draft,created_wall=time.time(),wall_seconds=480,new_bytes=128*1024**2,
  global_result_bytes=32*1024**3,reserve_bytes=4*1024**3,provider_calls=0,compiler_calls=0,encrypted_calls=0)
 plan["binding"]=digest(plan);(OUT/"plan.json").write_text(json.dumps(plan,allow_nan=False))
 rows=[]
 for c in json.loads((HERE/"cases.json").read_text()):
  require(time.monotonic()-started<plan["wall_seconds"],"Staging wall budget")
  require(measure(OUT)<plan["new_bytes"]and measure(OUT.parent)<plan["global_result_bytes"]and shutil.disk_usage(OUT).free>=plan["reserve_bytes"],"Staging storage budget")
  name=c["case"];manifest=ROOT/c["manifest"]
  require(sha(manifest.read_bytes())==c["manifest_sha256"],"Original source fixture changed")
  row=dict(case=name,status="running")
  try:
   receipt=run(manifest,OUT/name,trusted=True);row.update(status="captured",receipt_binding=receipt["binding"],seconds=receipt["seconds"])
  except Exception as e:
   row.update(status="failed",error_type=type(e).__name__,message=str(e)[:140])
   (OUT/(name+"-host-failure.log")).write_text(traceback.format_exc())
  rows.append(row);print(json.dumps(row),flush=True)
  require(source_identity()==frozen and draft_identity()==draft,"Frozen source implementation changed")
 report=dict(format="poseidon-public-tensor-source-draft-report-v1",status="captured"if all(r["status"]=="captured"for r in rows)else"failed",
  rows=rows,plan_binding=plan["binding"],source_hashes=frozen,draft_hashes=draft,seconds=time.monotonic()-started,
  source_capture_calls=len(rows),source_capture_passed=sum(r["status"]=="captured"for r in rows),
  provider_calls=0,compiler_calls=0,encrypted_calls=0,production_activation=False,network_configuration_changes=0)
 report["binding"]=digest(report);(OUT/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False))
 print(json.dumps({k:report[k]for k in("status","seconds","source_capture_calls","source_capture_passed","provider_calls","production_activation")}),flush=True)
 return 0 if report["status"]=="captured"else 1
def outer():
 from hecate_python_env import VENV,enter_nix
 cmd='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 return enter_nix(cmd,seconds=480,keep_env=(),environment="poseidon-cpu")
if __name__=="__main__":raise SystemExit(inside()if sys.argv[1:]==["--inside"]else outer())
