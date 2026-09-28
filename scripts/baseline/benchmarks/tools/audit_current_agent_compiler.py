"""Current-version compiler evidence from one independently audited Agent batch; no execution."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
BASE=Path(__file__).resolve().parents[2];ROOT=BASE.parents[1]
sys.path.insert(0,str(BASE))
from benchmark_graph import digest,signature,require
from benchmark_runner import strict_file,dump
from compiler_artifact_evidence import lineage
from compiler_configuration import PROFILE_SHA256,verify_artifact_configuration
RESULTS=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux/results")
BUILD=RESULTS.parent/"build-dacapo/hecate-18.1.2-nix"
KEY_BUILD=RESULTS.parent/"build-dacapo/seal-golden-keys"
PROFILE=ROOT/"third_party/dacapo/profiled_SEAL_CPU.json"

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return strict_file(Path(p),16*1024**2)
def bound(value):
    require(value["binding"]==digest({k:v for k,v in value.items() if k!="binding"}),"Report binding")
def verify_files(folder,hashes):
    for name,hsh in hashes.items():
        p=folder/name
        require(not Path(name).is_absolute() and ".." not in Path(name).parts and
                p.resolve().is_relative_to(folder.resolve()) and not p.is_symlink(),"Unsafe evidence path")
        require(sha(p)==hsh,"Retained evidence changed: "+name)

def verify_observations(trace,execution,params):
    require(execution["encrypted_execution"] is True and execution["bootstrap_executed"] is False,"Real CPU execution required")
    require(params["parameters_set"] is True and params["security_check"]=="tc128" and
            params["polynomial_degree"]==32768 and params["slots"]==16384 and
            params["modulus_bits"]==[60]*14 and params["data_modulus_count"]==13,"Frozen security parameters")
    require(execution["rotation_key_check"]==dict(actual_key_file_verified=True,
             required_steps=trace["gate"]["rotation_steps"]),"Rotation key evidence")
    observations=execution["ciphertext_metadata"]
    require(len(observations)==4,"Four input batches")
    for batch in observations:
        require(len(batch["outputs"])==len(trace["outputs"]),"Metadata arity")
        for actual,expected in zip(batch["outputs"],trace["outputs"]):
            require(actual["data_modulus_count"]==expected["level"] and
                    abs(actual["log2_scale"]-expected["scale"])<=1e-6 and
                    actual["polynomials"]==2,"Observed level/scale/ciphertext size")
    return True

def build(aggregate_path):
    from semantic_benchmark_execution import runtime_sources
    aggregate_path=Path(aggregate_path)
    require(aggregate_path.resolve().parent.parent==RESULTS and not aggregate_path.is_symlink(),"Direct audit result required")
    aggregate=read(aggregate_path);bound(aggregate)
    require(aggregate["source_sha256"]==digest(runtime_sources()),"Historical audit cannot become current compiler evidence")
    plans=[]
    for name,hsh in aggregate["parents"].items():
        path=Path(name)
        require(path.resolve().is_relative_to(RESULTS) and not path.is_symlink() and path.name!=".env","Unsafe audit parent")
        require(sha(path)==hsh,"Changed independent audit parent")
        if path.name=="plan.json":
            candidate=read(path)
            if candidate.get("binding")==aggregate["plan_binding"]:bound(candidate);plans.append(candidate)
    require(len(plans)==1,"Unique independently bound execution plan")
    pilot=plans[0]
    require(pilot["source_hashes"]==runtime_sources(),"Execution plan source drift")
    # The lineage implementation itself has not changed since the evaluated batch.
    for name in ("compiler_artifact_evidence.py","seal_artifact_gate.py","compiler_configuration.py"):
        require(sha(BASE/name)==pilot["source_hashes"]["scripts/baseline/"+name],"Artifact checker drift")
    require(sha(PROFILE)==PROFILE_SHA256,"Profile integrity")
    cpp=ROOT/"third_party/dacapo/lib/Runtime/SEAL_HEVM.cpp"
    pinned=subprocess.check_output(["git","-C",str(ROOT/"third_party/dacapo"),"show",
         "4616402710f39df3e5f5bd7930a6c036025aaac3:lib/Runtime/SEAL_HEVM.cpp"],timeout=10)
    require(cpp.read_bytes()==pinned,"Pinned runtime source identity")
    require("evaluator->relinearize_inplace(ciphers[dst], *relin_key);" in pinned.decode(),"Fused runtime relinearization")
    specs={c["id"]:c for c in pilot["cases"]}
    records=[];parents={str(aggregate_path):sha(aggregate_path),str(PROFILE):sha(PROFILE),str(cpp):sha(cpp)}
    for row in aggregate["rows"]:
        if row["status"]!="passed":continue
        audit_path=aggregate_path.parent/(row["id"]+".audit.json")
        require(sha(audit_path)==row["audit_sha256"],"Independent Agent audit changed")
        parents[str(audit_path)]=sha(audit_path)
        audit=read(audit_path);bound(audit)
        folder=Path(row["evidence"])
        require(folder.resolve().parent==RESULTS and not folder.is_symlink(),"Evidence location")
        verify_files(folder,audit["files"])
        report=read(folder/"report.json")
        require(sha(folder/"report.json")==row["report_sha256"] and report["agent_calls"]>0 and
                report["provider"]=="deepseek_api" and report["llm_generation_validated"],"Actual Agent provenance")
        request=read(folder/"request.json")
        require(request==specs[row["id"]]["request"] and audit["request_id"]==request["request_id"],"Frozen request")
        require(audit["plan_binding"]==pilot["binding"] and audit["model_sha256"]==row["model_sha256"],"Audit scope")
        passed=[a for a in report["attempts"] if a["status"]=="passed"]
        require(len(passed)==1,"One successful attempt")
        a=passed[0];attempt=folder/("attempt-%02d"%a["index"]);out=attempt/"output"
        require(a["compiled"] and a["executed"] and a["numerically_correct"] and
                a["comparison"]==audit["comparison"] and a["comparison"]["atol"]==1e-5 and
                a["comparison"]["rtol"]==1e-4,"Audited encrypted numerical evidence")
        raw=json.loads((attempt/"response.txt").read_bytes())
        require(raw["request_id"]==request["request_id"] and
                raw["hecate_source"].encode()==(attempt/"candidate.py").read_bytes(),"Exact retained Agent answer")
        verify_files(out,a["artifact_hashes"])
        trace=lineage((out/"lowered._hecate_golden.hevm").read_bytes(),(out/"_hecate_golden.cst").read_bytes(),request)
        require(trace["gate"]==a["artifact_gate"],"Artifact metadata mismatch")
        verify_artifact_configuration(request,trace["gate"],sha(PROFILE))
        execution=read(out/"execution.json")
        require(execution==a["execution"],"Execution metadata mismatch")
        require(str(BUILD/"lib/libSEAL_HEVM.so") in execution["mapped_libraries"],"Actual loaded runtime")
        for name,key in [("lib/libSEAL_HEVM.so","runtime_sha256"),("lib/libHecateFrontend.so","frontend_sha256")]:
            require(sha(BUILD/name)==report[key],"Runtime/frontend binary changed")
        for name,key in [("libseal_golden_metadata.so","metadata_observer_sha256"),
                         ("libseal_packed_metadata.so","packed_observer_sha256"),
                         ("seal_packed_keys","packed_key_helper_sha256")]:
            require(sha(KEY_BUILD/name)==report[key],"Observer/key helper changed")
        verify_observations(trace,execution,report["parameters"])
        require(read(folder/"key-cleanup-outcome.json")["complete"],"Key cleanup")
        counts=trace["gate"]["opcode_counts"]
        evidence=dict(rescale=int(counts.get("3",0)),modswitch=int(counts.get("4",0)),
            relinearization=int(counts.get("8",0)),security_parameters=True,
            rotation_keys=bool(execution["rotation_key_check"]["required_steps"]))
        records.append(dict(id=row["id"],evidence=str(folder),model_sha256=row["model_sha256"],
            topology=signature(request["model"],True),request_id=request["request_id"],
            evaluated_runtime_digest=aggregate["source_sha256"],successful_attempt=a["index"],
            independent_agent_audit_sha256=sha(audit_path),compiler_evidence=evidence,lineage=trace,
            actual_parameters=report["parameters"],output_metadata=execution["ciphertext_metadata"],
            rotation_key_check=execution["rotation_key_check"],runtime_sha256=report["runtime_sha256"]))
    partitions={}
    for name in ("rescale","modswitch","relinearization","security_parameters","rotation_keys"):
        rows=[r for r in records if r["compiler_evidence"][name]]
        topologies=sorted({r["topology"] for r in rows})
        partitions["compiler."+name]=dict(case_ids=[r["id"] for r in rows],
            distinct_topologies=len(topologies),topologies=topologies,
            three_contexts_verified=len(topologies)>=3,
            source_scope="current_bound_batch",
            evidence_scope="artifact operation, pinned runtime implementation and observed final metadata; not per-instruction runtime tracing")
    result=dict(format="poseidon-current-agent-compiler-artifact-audit-v1",parents=parents,
        independent_agent_audit_binding=aggregate["binding"],evaluated_runtime_digest=aggregate["source_sha256"],records=records,partitions=partitions,
        runtime_source_sha256=sha(cpp),checker_sources={n:sha(BASE/n) for n in
          ("compiler_artifact_evidence.py","seal_artifact_gate.py","compiler_configuration.py")},
        actual_agent_evidence=True,new_agent_generation=False,new_paid_calls=0,new_encrypted_executions=0,
        per_instruction_runtime_tracing=False,formal_error_bound=False,
        compiler_contexts_complete=all(p["three_contexts_verified"] for p in partitions.values()),
        full_Agent_acceptance=False,
        stage2_complete=False,runner_sha256=sha(Path(__file__)))
    result["binding"]=digest(result);return result

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True);p.add_argument("--agent-audit",type=Path,required=True);a=p.parse_args()
    require(not a.output.exists(),"Preserve previous audit")
    report=build(a.agent_audit);dump(a.output,report)
    print(json.dumps(dict(binding=report["binding"],partitions=report["partitions"],new_paid_calls=0),indent=2))
