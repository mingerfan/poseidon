"""Terminal source-route audit; independent reference arithmetic only, no provider/compile/FHE calls."""
import ast, hashlib, importlib.util, json, math, os, shlex, struct, sys, zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from benchmark_graph import digest, require
from unified_request_storage import compact_request
from hecate_python_env import VENV, enter_nix
if "--inside"not in sys.argv:
    command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join(
        [str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
    raise SystemExit(enter_nix(command,seconds=180,environment="poseidon-cpu"))
require(os.environ.get("IN_NIX_SHELL")=="pure"and Path(sys.prefix)==VENV,"audit_pinned_reference_environment")
import numpy as np
from benchmark_graph import samples
from operator_arithmetic_graph import evaluate_reference
ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])
PROPOSAL=ROOT/"docs/baseline/agent-spatial-source-offline-proposal-r1405.json"
AUTH=ROOT/"docs/baseline/agent-spatial-source-offline-authorization-r1405.json"
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def npy(raw):
    require(type(raw)is bytes and len(raw)<=64*1024**2 and raw[:6]==b"\x93NUMPY","audit_npy_magic")
    version=tuple(raw[6:8])
    require(version in ((1,0),(2,0)),"audit_npy_version")
    start=10 if version==(1,0) else 12
    size=struct.unpack("<H" if start==10 else "<I",raw[8:start])[0]
    require(size<=65536 and len(raw)>=start+size,"audit_npy_header_size")
    header=ast.literal_eval(raw[start:start+size].decode("latin1"))
    require(type(header)is dict and set(header)=={"descr","fortran_order","shape"}
            and header["descr"]=="<f8" and header["fortran_order"]is False,"audit_npy_type")
    shape=header["shape"]
    require(type(shape)is tuple and all(type(n)is int and n>=0 for n in shape),"audit_npy_shape")
    count=math.prod(shape)
    payload=raw[start+size:]
    require(count<=1000000 and len(payload)==count*8,"audit_npy_payload")
    values=[item[0]for item in struct.iter_unpack("<d",payload)]
    require(all(math.isfinite(x)for x in values),"audit_npy_finite")
    return shape,values
def compare_values(actual,reference):
    require(len(actual)==len(reference)and bool(actual),"audit_numeric_length")
    differences=[abs(a-r)for a,r in zip(actual,reference)]
    require(all(error<=1e-5+1e-4*abs(r)for error,r in zip(differences,reference)),"audit_frozen_tolerance")
    relative=[error/abs(r)for error,r in zip(differences,reference)if r!=0.]
    return dict(compared_values=len(actual),maximum_absolute_error=max(differences),
                mean_absolute_error=sum(differences)/len(differences),
                nonzero_reference_max_relative_error=max(relative)if relative else None)
def main():
    plan=read(PROPOSAL)
    driver=Path(__file__).with_name("offline_source_r1405.py")
    spec=importlib.util.spec_from_file_location("audit_parent_driver",driver)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.verify(plan)
    out=Path(plan["output"]);report=read(out/"report.json")
    require(report["binding"]==digest({k:v for k,v in report.items()if k!="binding"}),"audit_report_binding")
    require(report["status"]in("complete","failed")and report["proposal_binding"]==plan["binding"]
            and report["sdk_identity"]==plan["sdk_identity"],"audit_terminal_identity")
    require(report["provider_calls"]==report["http_attempts"]==0
            and report["agent_generated"]is False,"audit_offline_origin")
    authorization=read(AUTH)
    require(authorization.get("approved")is True and authorization["proposal_binding"]==plan["binding"]
            and authorization["binding"]==digest({k:v for k,v in authorization.items()if k!="binding"}),"audit_explicit_approval")
    work=Path(plan["output"]).parent
    import shutil
    current=module.measure(work);free=shutil.disk_usage(work).free
    new_bytes=current-authorization["preflight"]["global_artifact_bytes"]
    require(report["compiler_reserved"]<=plan["scope"]["maximum_compiler_calls"]
            and report["encrypted_reserved"]<=plan["scope"]["maximum_encrypted_calls"]
            and report["seconds"]<=plan["scope"]["wall_seconds"]
            and new_bytes<=plan["scope"]["new_bytes"]
            and current<=plan["scope"]["global_bytes"]and free>=plan["scope"]["reserve_bytes"],"audit_approved_budget")
    originals={case["case"]:case for case in plan["cases"]}
    rows=[];file_hashes={};groups=values=compiled=encrypted=reuse_passed=0;peak_error=0.
    for row in report["rows"]:
        case=originals[row["case"]]
        item=dict(case=row["case"],status=row["status"],compiled=row["compiled"],
                  encrypted_execution=row["encrypted_execution"],package_reloaded=row["package_reloaded"])
        rows.append(item)
        if row["status"]!="passed":
            item["failure"]=row.get("failure") or row.get("error_type")
            continue
        folder=out/case["case"];qualification=read(folder/"qualification.json")
        accept=qualification["acceptance"]
        require(qualification["status"]=="succeeded"and qualification["validation_level"]=="numerical"
                and accept["compiled"]is True and accept["encrypted_execution"]is True
                and accept["numerically_validated"]is True and accept["input_groups"]==4
                and qualification["generation"]["agent_generated"]is False
                and qualification["generation"]["http_attempts"]==0,"audit_application_grade")
        ev=Path(row["validation_evidence"]);raw=read(ev/"report.json")
        attempt=raw["attempts"]
        require(raw["backend"]=="poseidon_cpu"and raw["provider"]=="scripted_replay"
                and raw["agent_calls"]==0 and raw["status"]=="passed" and len(attempt)==1
                and attempt[0]["trace"]["frontend"]=="real_Hecate" and
                attempt[0]["compiled"]is True and attempt[0]["executed"]is True
                and attempt[0]["numerically_correct"]is True,"audit_backend_stage")
        require(raw["tolerance"]=={'atol': 1e-05, 'rtol': 0.0001},"audit_tolerance_identity")
        require(raw["request_id"]==case["request_id"]and read(ev/"request.json")==compact_request(read(case["request"])),
                "audit_backend_request")
        output=ev/"attempt-00/output"
        encrypted_record=read(output/"execution.json")
        require(encrypted_record["backend"]=="poseidon_cpu"and encrypted_record["encrypted_execution"]is True
                and encrypted_record["executor_secret_key"]is False and encrypted_record["bootstrap"]is False
                and encrypted_record["security"]=="tc128"and encrypted_record["degree"]==32768
                and encrypted_record["data_moduli"]==13 and encrypted_record["special_moduli"]==1,"audit_native_execution")
        program=Path(row["package"]);manifest=read(program/"manifest.json")
        require(manifest["binding"]==row["package_binding"]==digest({k:v for k,v in manifest.items()if k!="binding"})
                and manifest["agent_generated"]is False,"audit_bundle_binding")
        require(manifest["compiler_configuration"]["backend"]=="poseidon_cpu"
                and manifest["validation"]==dict(level="numerical",encrypted_execution=True,numerically_validated=True),
                "audit_bundle_backend_grade")
        require(manifest["origin"]["model_source"]["format"]=="poseidon-model-source-origin-v1"
                and manifest["origin"]["request_id"]==case["request_id"],"audit_bundle_source")
        require(sha(program/"candidate.py")==sha(case["candidate"])
                and read(program/"request.json")==compact_request(read(case["request"])),"audit_frozen_candidate_request")
        expected=read(case["provenance"]);actual_provenance=read(program/"model-provenance.json")
        require(actual_provenance["original"]==expected["original"],"audit_original_model")
        if "source_manifest"in case:
            source_spec=read(case["source_manifest"])
            actual_spec=actual_provenance["trusted_pytorch"]["manifest"]
            require(actual_spec==source_spec and source_spec["format"]==case["source_contract"],"audit_source_manifest")
            require(actual_provenance["trusted_pytorch"]["files"]==expected["trusted_pytorch"]["files"],
                    "audit_unchanged_source_files")
            item["source_contract"]=source_spec["format"]
        require(actual_provenance["decomposition"]==expected["decomposition"]
                and actual_provenance["reference_report"]==expected["reference_report"],"audit_source_reference_binding")
        for name,expected in manifest["files"].items():
            path=program/name
            require(path.is_file()and not path.is_symlink()and path.resolve().is_relative_to(program.resolve())
                    and sha(path)==expected,"audit_bundle_file")
            file_hashes[str(path)]=expected
        operator=read(program/"artifacts/operator-spec.json")
        require(sha(program/"artifacts/operator-spec.json")==plan["backend_identity"]["operator_spec_sha256"],
                "audit_operator_spec")
        require((program/"artifacts/lowered.earth.mlir").stat().st_size>0
                and (program/"artifacts/lowered.ckks.mlir").stat().st_size>0
                and (program/"artifacts/bound/plan.json").stat().st_size>0,"audit_compiler_artifacts")
        require(encrypted_record["q"]==attempt[0]["artifact_gate"]["bound"]["q"]
                and encrypted_record["p"]==attempt[0]["artifact_gate"]["bound"]["p"],"audit_modulus_identity")
        decrypted=output/"decrypted.npy"
        shape,actual=npy(decrypted.read_bytes())
        arrays=ev/"arrays.npz"
        with zipfile.ZipFile(arrays)as archive:
            entry=archive.getinfo("reference.npy")
            require(entry.file_size<=16*1024**2,"audit_reference_size")
            expected_shape,reference=npy(archive.read(entry))
        require(shape==expected_shape and shape[0]==4 and len(actual)==accept["compared_values"],"audit_reference_shape")
        request=read(case["request"])
        inputs=samples(request["model"],4)
        independent=[np.concatenate([evaluate_reference(expected["original"],batch)[port["name"]].reshape(-1)
                     for port in request["model"]["outputs"]])for batch in inputs]
        independent=np.asarray(independent,dtype=np.float64)
        require(independent.shape==expected_shape,"audit_independent_reference_shape")
        # The source capture's independent reference and stored evaluation reference
        # must agree at their own tight plaintext threshold; FHE tolerance stays fixed.
        np.testing.assert_allclose(np.asarray(reference).reshape(expected_shape),independent,atol=2e-12,rtol=2e-12)
        compare_values(actual,independent.reshape(-1).tolist())
        stats=compare_values(actual,reference)
        require(stats["maximum_absolute_error"]==accept["maximum_absolute_error"],"audit_error_aggregate")
        item["qualification"]=dict(input_groups=shape[0],**stats)
        item["immutable_candidate_sha256"]=sha(program/"candidate.py")
        item["request_id"]=case["request_id"]
        groups+=shape[0];values+=len(actual);peak_error=max(peak_error,stats["maximum_absolute_error"])
        compiled+=1;encrypted+=1
        for path in (decrypted,arrays,ev/"report.json",output/"execution.json"):
            file_hashes[str(path)]=sha(path)
        if case.get("repeat_package"):
            require(row.get("cached_without_generation")is True and set(row["reuse"])=={"new-inputs","repeat","named-equivalence"},
                    "audit_required_package_reuse")
            cached=read(folder/"cached.json")
            require(cached["status"]=="succeeded"and cached["package"]["reused"]is True
                    and cached["generations"]==cached["http_attempts"]==0,"audit_cache_generation")
            item["reuse"]={}
            for name,count in (("new-inputs",3),("repeat",1),("named-equivalence",3)):
                result=read(folder/(name+".json"));comparison=read(folder/(name+"-comparison.json"))
                require(result["status"]=="executed"and result["encrypted_execution"]is True
                        and result["executor_secret_key"]is False and result["agent_calls"]==result["compiler_calls"]==0
                        and len(result["outputs"])==count and comparison["passed"]is True
                        and row["reuse"][name]["groups"]==count,"audit_reuse_execution")
                metric=row["reuse"][name]
                require(metric["values"]==comparison["compared_values"]
                        and metric["maximum_absolute_error"]==comparison["max_absolute_error"],"audit_reuse_comparison")
                item["reuse"][name]=metric
                probes=[{k:np.asarray(v,dtype=np.float64)*.5+.015 for k,v in batch.items()}for batch in inputs[:3]]
                batches=probes[:1]if name=="repeat"else probes
                actual_reuse=np.asarray([np.concatenate([np.asarray(batch[port["name"]],dtype=np.float64).reshape(-1)
                           for port in request["model"]["outputs"]])for batch in result["outputs"]])
                reference_reuse=np.asarray([np.concatenate([evaluate_reference(expected["original"],batch)[port["name"]].reshape(-1)
                           for port in request["model"]["outputs"]])for batch in batches])
                require(actual_reuse.shape==reference_reuse.shape and actual_reuse.shape[0]==count,
                        "audit_reuse_independent_shape")
                independent_reuse=compare_values(actual_reuse.reshape(-1).tolist(),reference_reuse.reshape(-1).tolist())
                require(independent_reuse["maximum_absolute_error"]==metric["maximum_absolute_error"],
                        "audit_reuse_independent_error")
                groups+=count;values+=metric["values"];peak_error=max(peak_error,metric["maximum_absolute_error"]);encrypted+=1
                for path in (folder/(name+".json"),folder/(name+"-comparison.json")):
                    file_hashes[str(path)]=sha(path)
            reuse_passed+=1
    require(compiled==report["compiler_confirmed"]and encrypted==report["encrypted_confirmed"],"audit_actual_call_counts")
    require(len(rows)==len({r["case"]for r in rows})and all(r["case"]in originals for r in rows),"audit_unique_cases")
    expected_reuse=sum(c.get("repeat_package")is True for c in plan["cases"])
    if report["status"]=="complete"and report["failed"]==0:
        require(len(rows)==len(plan["cases"])and report["passed"]==len(plan["cases"])and report["not_run"]==0
                and compiled==len(plan["cases"])and encrypted==len(plan["cases"])+3*expected_reuse
                and reuse_passed==expected_reuse and groups==4*len(plan["cases"])+7*expected_reuse,"audit_complete_scope")
    summary=dict(format="poseidon-spatial-source-offline-independent-audit-v1",
        proposal_binding=plan["binding"],authorization_binding=authorization["binding"],
        report_binding=report["binding"],sdk_identity=plan["sdk_identity"],status="verified",
        completed=report["status"]=="complete",planned=len(plan["cases"]),passed=report["passed"],failed=report["failed"],
        blocked=report.get("blocked",0),not_run=report["not_run"],compiler_confirmed=compiled,
        encrypted_confirmed=encrypted,package_reuse_passed=reuse_passed,input_groups=groups,compared_values=values,
        maximum_absolute_error=peak_error,provider_calls=0,http_attempts=0,agent_generated=False,
        measured_new_bytes=new_bytes,current_bytes=current,free_bytes=free,
        file_hashes=file_hashes,rows=rows)
    summary["binding"]=digest(summary)
    dest=out/"independent-audit.json"
    with dest.open("x")as f:json.dump(summary,f,indent=2);f.write("\n")
    print(json.dumps({k:v for k,v in summary.items()if k not in("rows","file_hashes")},indent=2))
if __name__=="__main__":main()
