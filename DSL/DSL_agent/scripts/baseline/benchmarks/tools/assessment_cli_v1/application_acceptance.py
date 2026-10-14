"""Public Ubuntu CLI, separate processes, no new encrypted execution or API."""
import hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);sys.path.insert(0,str(ROOT/"scripts/baseline"))
from benchmark_graph import canonical,digest
WORK=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux/results")
PREVIOUS=WORK/"agent-application-assessment-acceptance-v1"
DATA=WORK/"agent-dataset-io-acceptance-v2/input-dataset/dataset.json"
OUT=WORK/"agent-assessment-cli-acceptance-v1"
def hashes(folder):
 files={}
 for p in folder.rglob("*"):
  assert not p.is_symlink() and p.name!=".env" and p.suffix not in (".sk",".pem")
  if p.is_file():files[str(p.relative_to(folder))]=hashlib.sha256(p.read_bytes()).hexdigest()
 return files
def total(folder):return sum(p.stat().st_size for p in folder.rglob("*")if p.is_file())
def pinned(path):
 value=json.loads(path.read_text());assert value["binding"]==digest({k:v for k,v in value.items()if k!="binding"});return value
def write(path,value):
 with path.open("xb")as f:f.write(canonical(value))
def main():
 start=time.monotonic();old=hashes(PREVIOUS);prior=pinned(PREVIOUS/"report.json")
 assert prior["status"]=="passed"
 old_handle=json.loads((PREVIOUS/"handle.json").read_text())
 job=PREVIOUS/"assessment-store"/old_handle["evaluation_id"]
 plan=pinned(job/"plan.json");state=pinned(job/"state.json")
 assert state["status"]=="validation_failed"and len(state["receipts"])==2
 assert total(WORK)<32*1024**3 and shutil.disk_usage(WORK).free>=4*1024**3
 executions={p.name for p in WORK.glob("poseidon-program-execution-*")}
 OUT.mkdir(mode=0o700)
 cache={}
 for name,index in state["receipts"].items():
  receipt=pinned(job/"receipts"/(name+".json"));assert receipt["binding"]==index["binding"]
  cache[name]=dict(receipt=receipt,binding=receipt["binding"])
 write(OUT/"cache.json",cache);write(OUT/"criterion.json",dict(kind="max_absolute_error",limit=.01))
 write(OUT/"input-dtypes.json",dict(q="float64",k="float64",v="float64"))
 limits=dict(wall_seconds=600,new_bytes=128*1024**2,global_bytes=32*1024**3,reserve_bytes=4*1024**3)
 scope=dict(format="poseidon-assessment-cli-local-scope-v1",task=plan["program_task"],
  package_binding=plan["package_binding"],old_dataset_binding=plan["dataset_file_binding"],
  max_new_assessments=0,max_encrypted_calls=0,max_source_reference_workers=0,
  compiler_calls=0,api_calls=0,commands=["submit","advance","result"],limits=limits)
 scope["binding"]=digest(scope);write(OUT/"scope.json",scope)
 base=[sys.executable,"-B",str(ROOT/"scripts/agent_assessment.py"),
  "--tasks",str(PREVIOUS/"tasks"),"--store",str(OUT/"assessment-store"),
  "--platform","aarch64-linux","--work-root","/home/lhohy/poseidon-work","--timeout","180"]
 calls=[]
 def call(name,args,expected):
  begin=time.monotonic()
  process=subprocess.run(base+args,cwd=ROOT,stdin=subprocess.DEVNULL,capture_output=True,timeout=190)
  (OUT/(name+".stdout.json")).write_bytes(process.stdout)
  (OUT/(name+".stderr.log")).write_bytes(process.stderr)
  assert process.returncode==expected,(name,process.returncode)
  result=json.loads(process.stdout);assert result["format"]=="poseidon-application-assessment-command-v1"
  assert result["binding"]==digest({k:v for k,v in result.items()if k!="binding"})
  assert process.stderr==b""
  calls.append(dict(action=name,exit_code=process.returncode,response_binding=result["binding"],
    entry_identity=result["entry_identity"],elapsed_seconds=time.monotonic()-begin))
  return result
 submitted=call("submit",["submit","--task",plan["program_task"],"--dataset",str(DATA),
  "--dataset-binding",plan["dataset_file_binding"],"--reference-policy",plan["reference_policy"],
  "--criterion-file",str(OUT/"criterion.json"),"--input-dtypes-file",str(OUT/"input-dtypes.json"),
  "--trust-local-code","--max-new-assessments","0","--part-samples","2",
  "--cache-receipts-file",str(OUT/"cache.json")],0)
 assert submitted["status"]=="submitted";h=submitted["handle"]
 selection=["--evaluation-id",h["evaluation_id"],"--submission-binding",h["binding"]]
 advanced=call("advance",["advance",*selection,"--max-new-parts","0"],3)
 result=call("result",["result",*selection],3)
 assert advanced["result"]["binding"]==result["result"]["binding"]
 value=result["result"];assert value["status"]=="validation_failed"and value["budget"]["used"]==0
 assert value["completed_parts"]==2 and value["summary"]["evaluated_samples"]==3
 assert value["summary"]["ckks_comparison"]["compared_values"]==30
 assert value["acceptance"]["numerical_on_declared_dataset"]is True
 assert value["acceptance"]["application_criterion_validated"]is False
 assert value["failure"]["code"]=="assessment_application_criterion_failed"
 assert value["ledger"]==[] and value["evidence"]["receipts"]=={n:v["binding"]for n,v in cache.items()}
 assert len({c["entry_identity"]for c in calls})==1
 assert hashes(PREVIOUS)==old
 assert {p.name for p in WORK.glob("poseidon-program-execution-*")}==executions
 assert not any(p.name.startswith("native-")for p in(OUT/"assessment-store"/h["evaluation_id"]).iterdir())
 assert total(OUT)<=limits["new_bytes"]and time.monotonic()-start<=limits["wall_seconds"]
 assert total(WORK)<limits["global_bytes"]and shutil.disk_usage(WORK).free>=limits["reserve_bytes"]
 report=dict(format="poseidon-assessment-cli-local-acceptance-v1",status="passed",
  scope_binding=scope["binding"],commands=calls,controller_result_binding=value["binding"],
  assessment_status=value["status"],failure_code=value["failure"]["code"],
  receipts_reused=2,samples_reused=3,values_reused=30,
  cached_ckks_max_absolute_error=value["summary"]["ckks_comparison"]["max_absolute_error"],
  cached_original_source_max_absolute_error=value["summary"]["application_criterion"]["max_absolute_error"],
  original_source_accuracy_passed=False,new_encrypted_calls=0,new_source_reference_workers=0,
  new_compiler_calls=0,new_api_calls=0,used_quota=0,
  existing_task_package_worker_and_history_unchanged=True,
  public_cli_cross_process_reload=True,outer_python=list(sys.version_info[:3]),
  elapsed_seconds=time.monotonic()-start,bytes=total(OUT))
 report["binding"]=digest(report);write(OUT/"report.json",report)
 print(json.dumps(report))
if __name__=="__main__":main()
