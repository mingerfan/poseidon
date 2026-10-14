"""Bounded offline operator regression using the Poseidon CPU candidate runner."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from benchmark_graph import digest, samples, require
from hecate_python_env import ROOT, WORK, VENV, enter_nix
from semantic_benchmark_execution import runtime_sources,CONFIG,CPU_CONFIGURATIONS,cpu_configuration,verify_numerical_report

def dump(path,value):
    path.write_text(json.dumps(value,indent=2)+"\n")

def disk_bytes(path):
    return int(subprocess.check_output(["du","-sb",str(path)],timeout=30,text=True).split()[0])

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--inside",action="store_true")
    p.add_argument("--run-local",action="store_true")
    p.add_argument("--compiler-configuration",choices=CPU_CONFIGURATIONS,default=CONFIG)
    p.add_argument("--case-id",action="append",default=[])
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--manifest",type=Path,default=ROOT/"scripts/baseline/cases/operator-extension-v3/manifest.json")
    a=p.parse_args()
    manifest=a.manifest.resolve()
    require(manifest.is_relative_to((ROOT/"scripts/baseline/cases").resolve()),"Manifest must be in cases")
    spec=json.loads(manifest.read_text())
    require(0<len(spec["cases"])<=48,"Shard size")
    require(len({x["id"] for x in spec["cases"]})==len(spec["cases"]),"Duplicate case")
    for item in spec["cases"]:
        require(re.fullmatch(r"[a-z0-9_]+",item["id"]) is not None,"Case id")
        for key in ("model","source","binding"):
            path=manifest.parent/item[key]
            require(path.parent==manifest.parent and not path.is_symlink() and path.is_file(),"Case file")
    if a.case_id:
        require(len(set(a.case_id))==len(a.case_id) and set(a.case_id)<={c["id"] for c in spec["cases"]},"Unknown/duplicate case")
        spec=dict(spec,cases=[c for c in spec["cases"] if c["id"] in a.case_id])
    cfg=cpu_configuration(a.compiler_configuration)
    frozen=runtime_sources()
    plan=dict(format="poseidon-operator-regression-plan-v2",backend="poseidon_cpu",manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
              source_hashes=frozen,cases=spec["cases"],paid_calls=0,agent_generated=False,
              compiler_configuration=cfg["name"],max_seconds=3600,max_case_seconds=300,
              batch_bytes=4*1024**3,global_bytes=32*1024**3,free_bytes=4*1024**3)
    if not a.run_local:
        print(json.dumps(plan,indent=2));return 0
    if not a.inside:
        return enter_nix('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+
            shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside","--run-local","--output",str(a.output),"--manifest",str(manifest),"--compiler-configuration",cfg["name"],*[v for name in a.case_id for v in ("--case-id",name)]]),seconds=3650,environment="poseidon-cpu")
    require(a.output.resolve().is_relative_to((WORK/"results").resolve()),"Output must be in results")
    require(not a.output.exists(),"Preserve previous evidence")
    a.output.mkdir();dump(a.output/"plan.json",plan)
    binding=digest(plan);start=time.monotonic();used=0
    rows=[dict(id=r["id"],status="not_run") for r in spec["cases"]]
    def save():
        dump(a.output/"report.json",dict(binding=binding,rows=rows,seconds=time.monotonic()-start,
              evidence_bytes=used,paid_calls=0,agent_generated=False))
    save()
    import numpy as np
    from benchmark_math import evaluate as math_ref
    from benchmark_torch import evaluate as torch_ref
    from operator_reference import evaluate as high_ref
    from model_decomposition import decompose
    for i,item in enumerate(spec["cases"]):
        require(runtime_sources()==frozen,"Source changed at case boundary")
        require(hashlib.sha256(manifest.read_bytes()).hexdigest()==plan["manifest_sha256"],"Manifest changed")
        if time.monotonic()-start>=3600 or used>=4*1024**3 or disk_bytes(WORK/"results")>=32*1024**3 or shutil.disk_usage(WORK).free<4*1024**3:
            rows[i]["status"]="budget_stopped";save();break
        model_path=manifest.parent/item["model"];source_path=manifest.parent/item["source"]
        low=json.loads(model_path.read_text());high=json.loads(source_path.read_text())
        require(digest(low)==item["model_sha256"] and digest(high)==item["source_sha256"],"Model changed")
        generated,db=decompose(high);require(generated==low,"Decomposition changed")
        bound=json.loads((manifest.parent/item["binding"]).read_text())
        require(digest({k:v for k,v in bound.items() if k!="binding_sha256"})==bound["binding_sha256"],"Original decomposition binding hash")
        require(bound["original_sha256"]==item["source_sha256"] and bound["lowered_sha256"]==item["model_sha256"],"Original decomposition identity")
        version_fields={"registry_version","implementation_sha256","binding_sha256"}
        require({k:v for k,v in db.items() if k not in version_fields}==
                {k:v for k,v in bound.items() if k not in version_fields},"Decomposition semantics changed")
        # Exact model equality above and dual references below validate current
        # code; preserve both bindings instead of relabelling historical evidence.
        dump(a.output/(item["id"]+"-decomposition.json"),dict(original=bound,current=db))
        approximation_error=0.;reference_error=0.
        for inputs in samples(low,16):
            expected=high_ref(high,inputs)["result"]
            ideal=high_ref(high,inputs,ideal=True)["result"]
            approximation_error=max(approximation_error,float(np.max(np.abs(expected-ideal))))
            for fn in (math_ref,torch_ref):
                actual=fn(low,inputs)["result"]
                np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=1e-12)
                reference_error=max(reference_error,float(np.max(np.abs(actual-expected))))
        row=dict(id=item["id"],binding=binding,status="failed",reference_probes=16,
                 reference_max_error=reference_error,approximation_max_error=approximation_error,
                 model_sha256=item["model_sha256"],source_sha256=item["source_sha256"],
                 backend="poseidon_cpu",original_decomposition_binding=bound["binding_sha256"],
                 current_decomposition_binding=db["binding_sha256"])
        log=a.output/(item["id"]+".log")
        command=[str(VENV/"bin/python"),"-B",str(ROOT/"scripts/baseline/run_candidate.py"),
                 "--inside","--case",str(model_path),"--self-test","--max-repairs","0",
                 "--compiler-configuration",plan["compiler_configuration"]]
        timed_out=False;begin=time.monotonic()
        with log.open("x") as f:
            child=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=child.wait(timeout=max(1,min(300,3600-(time.monotonic()-start))))
            except subprocess.TimeoutExpired:
                timed_out=True;os.killpg(child.pid,signal.SIGINT)
                try:code=child.wait(timeout=20)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait(timeout=10)
        row.update(exit_code=code,seconds=time.monotonic()-begin,timed_out=timed_out,
                   log_sha256=hashlib.sha256(log.read_bytes()).hexdigest())
        paths=re.findall(r"^Candidate evidence: (.+)$",log.read_text(),re.M)
        if paths:
            evidence=Path(paths[-1]).resolve()
            require(evidence.is_relative_to((WORK/"results").resolve()),"Evidence location")
            rp=evidence/"report.json"
            if rp.exists():
                report=json.loads(rp.read_text());attempts=report.get("attempts",[])
                verify_numerical_report(report,plan["compiler_configuration"])
                require(not report.get("agent_calls") and not report.get("llm_generation_validated"),"Offline identity")
                passed=code==0 and report.get("status")=="passed"
                row.update(status="passed" if passed else "failed",evidence=str(evidence),
                           report_sha256=hashlib.sha256(rp.read_bytes()).hexdigest(),
                           validation_level=report.get("validation_level"),
                           executed=any(x.get("executed") for x in attempts),
                           failure_layer=report.get("failure_layer") or next((x.get("failure_layer") for x in attempts if x.get("failure_layer")),None))
                used+=disk_bytes(evidence)
        rows[i]=row;save()
        print(item["id"],row["status"],row.get("failure_layer"),flush=True)
        if timed_out or not paths or row.get("failure_layer") in ("environment","integrity","sandbox","harness_launch"):
            break
    require(runtime_sources()==frozen,"Source changed before final report")
    save()
    return 0 if all(x["status"]=="passed" for x in rows) else 1
if __name__=="__main__":raise SystemExit(main())
