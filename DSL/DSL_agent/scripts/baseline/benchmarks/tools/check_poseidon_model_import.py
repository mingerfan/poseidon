"""Bounded developer acceptance of existing static Python and trusted FX MLPs."""
import hashlib,json,os,re,signal,subprocess,sys,time,contextlib
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[6] if Path(__file__).resolve().parents[4].name == "DSL_agent" and Path(__file__).resolve().parents[5].name == "DSL" else Path(__file__).resolve().parents[4])
sys.path.insert(0,str(ROOT/"scripts/baseline"))
from hecate_python_env import VENV,WORK
from semantic_benchmark_execution import runtime_sources,verify_numerical_report
from poseidon_backend_identity import identity
from benchmark_graph import require,digest
from benchmark_runner import dump,strict_file
from audit_unified_candidate import verify_candidate
from candidate_bundle import export_bundle,check_bundle,replay_bundle
from poseidon_benchmark_entries import storage

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run(cmd,log):
    with log.open("x") as f:
        proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=proc.wait(timeout=300)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGINT)
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=10)
            raise ValueError("Import/acceptance step budget exhausted")
    require(code==0,"Step failed: "+str(log))
    return log.read_text()

def main():
    require(os.environ.get("IN_NIX_SHELL")=="pure" and Path(sys.prefix)==VENV,"Locked Poseidon environment")
    out=Path(sys.argv[1]);require(not out.exists() and out.parent.resolve()==(WORK/"results").resolve(),"Fresh platform result directory")
    out.mkdir()
    frozen=runtime_sources();backend=identity();begin=time.monotonic();rows=[]
    plan=dict(format="poseidon-model-import-acceptance-v1",source_hashes=frozen,backend_identity=backend,
        driver_sha256=sha(Path(__file__)),configuration="poseidon-cpu-eva-w45-v1",paid_calls=0,
        limits=dict(batch_seconds=900,step_seconds=300,artifact_bytes=512*1024**2,native_jobs=1,compile_jobs=2,link_jobs=1),
        cases=["static_multifile_mlp","trusted_module_mlp"])
    plan["binding"]=digest(plan);dump(out/"plan.json",plan)
    def save():
        dump(out/"report.json",dict(format=plan["format"],binding=plan["binding"],rows=rows,
            planned=2,passed=sum(r["status"]=="passed" for r in rows),failed=sum(r["status"]=="failed" for r in rows),
            not_run=2-len(rows),skipped=0,seconds=time.monotonic()-begin,paid_calls=0,agent_calls=0))
    save()
    examples=[
        ("static_multifile_mlp",["--python-manifest",str(ROOT/"scripts/baseline/cases/operator-decomposition-v1/mlp-python.json")]),
        ("trusted_module_mlp",["--trusted-pytorch-manifest",str(ROOT/"scripts/baseline/cases/trusted-pytorch-v1/manifest.json"),"--trust-local-code"])]
    for name,arguments in examples:
        require(runtime_sources()==frozen and identity()==backend,"Source/backend changed at boundary")
        require(time.monotonic()-begin<900,"Batch time budget")
        folder=out/name;folder.mkdir();imported=folder/"imported"
        row=dict(id=name,status="failed",manual_candidate=True,agent_generated=False)
        try:
            run([str(VENV/"bin/python"),"-B",str(ROOT/"scripts/prepare_model.py"),*arguments,"--inside","--write","--output",str(imported)],folder/"import.log")
            reference=strict_file(imported/"reference-report.json",131072)
            require(reference["reference_status"]=="passed" and reference["probes"]==16 and not reference["encrypted_execution"],"Import reference scope")
            text=run([str(VENV/"bin/python"),"-B",str(ROOT/"scripts/baseline/run_candidate.py"),"--inside","--case",str(imported/"model.json"),"--self-test","--max-repairs","0","--compiler-configuration","poseidon-cpu-eva-w45-v1"],folder/"candidate.log")
            matches=re.findall(r"^Candidate evidence: (.+)$",text,re.M);require(len(matches)==1,"Single candidate evidence")
            evidence=Path(matches[0]);report=strict_file(evidence/"report.json",8*1024**2)
            verify_numerical_report(report,plan["configuration"]);checked=verify_candidate(evidence)
            require(report["status"]=="passed" and report["agent_calls"]==0 and not report["llm_generation_validated"],"Manual numerical provenance")
            package=folder/"program";manifest=export_bundle(evidence,package,model_import=imported)
            loaded=check_bundle(package)
            require(loaded["format"]=="poseidon-dsl-bundle-v4" and loaded["model_provenance"]=="included" and
                loaded["validation"]["level"]=="numerical","Poseidon package with source provenance")
            before=set((WORK/"results").glob("poseidon-package-replay-*"))
            with (folder/"replay.log").open("x") as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
                require(replay_bundle(package)==0,"Package replay")
            added=set((WORK/"results").glob("poseidon-package-replay-*"))-before;require(len(added)==1,"Single package reload evidence")
            replay=next(iter(added));reloaded=strict_file(replay/"report.json",8*1024**2)
            require(reloaded["status"]=="passed" and reloaded["agent_calls"]==0 and reloaded["compiler_calls"]==0 and
                reloaded["backend"]=="poseidon_cpu","Direct reload without generation or compilation")
            row.update(status="passed",import_path=str(imported),import_report_sha256=sha(imported/"reference-report.json"),
                reference=reference,evidence=str(evidence),report_sha256=sha(evidence/"report.json"),
                comparison=checked["comparison"],package=str(package),package_binding=loaded["binding"],
                provenance_sha256=sha(package/"model-provenance.json"),reload_evidence=str(replay),
                reload=dict(status=reloaded["status"],comparison=reloaded["comparison"],
                    agent_calls=0,compiler_calls=0,numerically_validated=reloaded["numerically_validated"]))
        except Exception as error:row.update(error_type=type(error).__name__,diagnostic=str(error))
        rows.append(row);save()
        storage(out,[])
        print(name,row["status"],row.get("diagnostic",""),flush=True)
        if row["status"]!="passed":break
    require(runtime_sources()==frozen and identity()==backend,"Final source/backend binding")
    save();return int(len(rows)!=2 or any(r["status"]!="passed" for r in rows))
if __name__=="__main__":raise SystemExit(main())
