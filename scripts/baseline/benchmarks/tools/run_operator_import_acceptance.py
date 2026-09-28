"""Bounded serial acceptance for the new operator/Python import path; no provider."""
import argparse,hashlib,json,os,re,shlex,signal,subprocess,sys,time
from pathlib import Path
BASE=Path(__file__).resolve().parents[2];ROOT=BASE.parents[1]
sys.path.insert(0,str(BASE))
from benchmark_runner import dump,strict_file
from benchmark_graph import digest
from semantic_benchmark_execution import runtime_sources
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--execute",action="store_true")
 p.add_argument("--output",type=Path,required=True);p.add_argument("--inside",action="store_true",help=argparse.SUPPRESS)
 p.add_argument("--case",action="append",choices=("mlp","rmsnorm","attention_decode"))
 p.add_argument("--compiler-configuration",choices=("seal-cpu-eva-w45-v1","seal-cpu-eva-w40-v1"),default="seal-cpu-eva-w45-v1")
 p.add_argument("--case-manifest",type=Path)
 a=p.parse_args()
 from hecate_python_env import enter_nix,VENV
 from workspace_paths import RESULTS
 cases=[("mlp","--python-manifest","mlp-python.json"),("rmsnorm","--graph","rmsnorm.json"),("attention_decode","--graph","attention-decode.json")]
 case_root=BASE/"cases/operator-decomposition-v1"
 if a.case_manifest:
  if a.case:p.error("Choose built-in filter or manifest")
  manifest=strict_file(a.case_manifest,131072)
  if manifest.get("format")!="poseidon-offline-case-list-v1" or not 1<=len(manifest["cases"])<=48:p.error("Case manifest")
  case_root=a.case_manifest.resolve().parent;cases=[];seen=set()
  for item in manifest["cases"]:
   name=item["file"];label=item["id"]
   if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}",label) or label in seen or Path(name).name!=name or not name.endswith(".json"):p.error("Unsafe/duplicate case")
   seen.add(label);path=case_root/name
   strict_file(path,131072)
   if hashlib.sha256(path.read_bytes()).hexdigest()!=item["sha256"]:p.error("Case hash")
   cases.append((label,"--graph",name))
 if a.case:
  if len(a.case)!=len(set(a.case)):p.error("Duplicate case")
  cases=[c for c in cases if c[0] in a.case]
 if not a.execute:print(json.dumps(dict(cases=cases,compiler_configuration=a.compiler_configuration,max_wall_seconds=1800,max_result_mib=1024,concurrency=1,paid_calls=0)));return 0
 if not a.output.resolve().is_relative_to(RESULTS.resolve()) or a.output.exists():p.error("New platform output directory required")
 if not a.inside:
  command=[str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),*sys.argv[1:],"--inside"]
  return enter_nix('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join(command),seconds=1860)
 if os.environ.get("IN_NIX_SHELL")!="pure" or Path(sys.prefix)!=VENV:p.error("Pinned pure environment")
 from audit_unified_candidate import verify_candidate
 a.output.mkdir(parents=True);sources=runtime_sources()
 plan=dict(runtime_sources=sources,runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           cases=cases,compiler_configuration=a.compiler_configuration,max_wall_seconds=1800,max_result_mib=1024,concurrency=1,paid_calls=0)
 plan["case_files"]={name:hashlib.sha256((case_root/name).read_bytes()).hexdigest() for _,_,name in cases}
 if a.case_manifest:plan["case_manifest_sha256"]=hashlib.sha256(a.case_manifest.read_bytes()).hexdigest()
 plan["binding"]=digest(plan);dump(a.output/"plan.json",plan)
 start=time.monotonic();rows=[]
 def run(command,log):
  if runtime_sources()!=sources:raise ValueError("Source drift")
  remaining=1800-(time.monotonic()-start)
  if remaining<=0:raise TimeoutError("Batch wall budget")
  with log.open("w") as stream:
   child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
   deadline=time.monotonic()+min(600,remaining)
   try:
    while child.poll() is None:
     if time.monotonic()>=deadline:raise subprocess.TimeoutExpired(command,min(600,remaining))
     evidence_paths=set()
     for prior_log in a.output.glob("*-fhe.log"):
      for raw in re.findall(r"^Candidate evidence: (.+)$",prior_log.read_text(),re.M):
       path=Path(raw).resolve()
       if not path.is_relative_to(RESULTS.resolve()):raise ValueError("Unexpected evidence path")
       evidence_paths.add(path)
     used=0
     for folder in [a.output,*evidence_paths]:
      for f in folder.rglob("*"):
       try:
        if f.is_file():used+=f.stat().st_size
       except FileNotFoundError:pass  # Existing key-retention cleanup can remove transient files.
     if used>1024*1024**2:raise subprocess.TimeoutExpired(command,"disk budget")
     time.sleep(.25)
    return child.wait()
   except BaseException as error:
    # Integrity/environment failures stop the child too; never leave a batch
    # running after the supervising audit has failed.
    if child.poll() is None:
     try:os.killpg(child.pid,signal.SIGINT)
     except ProcessLookupError:pass
     try:child.wait(timeout=15)
     except subprocess.TimeoutExpired:
      try:os.killpg(child.pid,signal.SIGKILL)
      except ProcessLookupError:pass
      child.wait(timeout=10)
    if isinstance(error,subprocess.TimeoutExpired):
     raise TimeoutError("Candidate budget; evidence preserved") from error
    raise
 for label,kind,name in cases:
  folder=a.output/label
  command=[str(VENV/"bin/python"),"-B",str(ROOT/"scripts/prepare_model.py"),kind,str(case_root/name),"--write","--inside","--output",str(folder)]
  code=run(command,a.output/(label+"-reference.log"))
  row=dict(id=label,reference_exit=code,status="reference_failed",agent_calls=0)
  if code==0:
   command=[str(VENV/"bin/python"),"-B",str(BASE/"run_candidate.py"),"--inside","--case",str(folder/"model.json"),
            "--self-test","--max-repairs","0","--compiler-configuration",a.compiler_configuration]
   log=a.output/(label+"-fhe.log");code=run(command,log);row["execution_exit"]=code;row["status"]="failed"
   matches=re.findall(r"^Candidate evidence: (.+)$",log.read_text(),re.M)
   if matches:
    evidence=Path(matches[-1]);require_path=evidence.resolve().is_relative_to(RESULTS.resolve())
    if not require_path:raise ValueError("Evidence path")
    report=strict_file(evidence/"report.json",8*1024**2)
    if report.get("agent_calls")!=0:raise ValueError("Unexpected provider")
    row.update(evidence=str(evidence),reported_status=report["status"],
               report_sha256=hashlib.sha256((evidence/"report.json").read_bytes()).hexdigest())
    if code==0 and report["status"]=="passed":
     audit=verify_candidate(evidence)
     if audit["model_sha256"]!=digest(strict_file(folder/"model.json",131072)):raise ValueError("Model drift")
     dump(folder/"fhe-audit.json",audit);row.update(status="passed",comparison=audit["comparison"])
    else:row["diagnostic"]=report.get("diagnostic") or [x.get("diagnostic") for x in report.get("attempts",[])]
  rows.append(row);dump(a.output/"progress.json",rows)
 if runtime_sources()!=sources:raise ValueError("Source drift")
 report=dict(plan_binding=plan["binding"],rows=rows,seconds=time.monotonic()-start,paid_calls=0,
             original_benchmark_models_unchanged=True,agent_generated=False)
 dump(a.output/"report.json",report)
 print(json.dumps(dict(statuses={r["id"]:r["status"] for r in rows},seconds=report["seconds"],paid_calls=0)))
 return int(any(r["status"]!="passed" for r in rows))
if __name__=="__main__":raise SystemExit(main())
