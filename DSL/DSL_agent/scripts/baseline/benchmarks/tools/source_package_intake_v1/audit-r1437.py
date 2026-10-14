"""Independent read-only evidence audit; no capture, compilation or generation."""
import ast,copy,hashlib,json,re,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline"
sys.path[:0]=[str(BASE),str(BASE/"benchmarks/tools/static_broadcast_live_v1")]
import run as live
from benchmark_graph import digest
def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def check_binding(value):assert value["binding"]==digest({k:v for k,v in value.items()if k!="binding"})
live.no_active()
assert subprocess.check_output(["git","branch","--show-current"],cwd=ROOT,text=True).strip()=="lhy-agent-dsl"
assert subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()=="7588e48093631eaf8385506e3db058e7ca3cfe7e"
before=read(HERE/"identity-before.json");after=read(HERE/"identity-after-controls.json")
assert before["backend"]==after["backend"]
changed=[];stage_diffs={}
expected={"application_capabilities.py","application_model_source.py","candidate_bundle.py","pytorch_model_intake.py","pytorch_ordered_contract.py","pytorch_ordered_intake.py","pytorch_public_expression_capture.py","pytorch_spatial_policy.py","pytorch_static_broadcast.py","pytorch_source_package.py","pytorch_source_package_policy.py"}
for stage in("generation","compilation","execution","validation"):
    old,new=before[stage]["sources"],after[stage]["sources"]
    diff=sorted(k for k in set(old)|set(new)if old.get(k)!=new.get(k))
    assert set(diff)==expected and not set(old)-set(new),stage
    assert all(sha(BASE/name)==value for name,value in new.items()),stage
    stage_diffs[stage]=dict(before=before[stage]["sha256"],after=after[stage]["sha256"],changed=diff)
controls=read(HERE/"controls-corrected.json")
assert controls["tests"]==controls["passed"]==301 and controls["failed"]==controls["errors"]==controls["skipped"]==0
assert len(set(controls["test_ids"]))==301
assert all(sha(BASE/(name+".py"))==value for name,value in controls["source_hashes"].items())
log=(HERE/"controls-corrected.log").read_text()
assert re.search(r"^Ran 301 tests in ",log,re.M) and re.search(r"^OK$",log,re.M)
assert controls["real_capture_calls"]==controls["compiler_calls"]==controls["encrypted_calls"]==controls["provider_calls"]==0
versions={}
for node in ast.parse((ROOT/"scripts/prepare_pytorch_source.py").read_text()).body:
    if isinstance(node,ast.Assign)and isinstance(node.value,ast.Constant):
        for target in node.targets:
            if isinstance(target,ast.Name)and target.id in("DEFAULT_FORMAT","LATEST_FORMAT"):versions[target.id]=node.value.value
assert versions==dict(DEFAULT_FORMAT="poseidon-trusted-pytorch-v18",LATEST_FORMAT="poseidon-trusted-pytorch-v28")
caps_before=read(HERE/"capabilities-before.json");caps_after=read(HERE/"capabilities-after.json")
old=copy.deepcopy(caps_after);trusted=old["trusted_pytorch"];trusted.pop("source_package")
v29="poseidon-trusted-pytorch-v29"
trusted["formats"].remove(v29)
for key in("input_call","spatial_calls","static_broadcast"):trusted[key]["versions"].remove(v29)
assert old==caps_before
package=caps_after["trusted_pytorch"]["source_package"]
assert package["input_status"]=="registered"and all(package[k]=="unverified"for k in("compiler_status","encrypted_status","agent_status"))
plan=read(HERE/"capture-plan-v4.json");check_binding(plan)
assert sha(HERE/"capture-v4.py")==plan["capture_tool_sha256"]
assert plan["sdk"]==after["manifest_sha256"]
for case in plan["cases"]:
    assert sha(case["manifest"])==case["manifest_sha256"]
    assert all(sha(ROOT/name)==value for name,value in case["files"].items())
    assert all(sha(ROOT/name)==value for name,value in plan["sdk_sources"].items())
run=Path(plan["campaign"]);capture=read(run/"report.json");check_binding(capture)
assert capture["plan_binding"]==plan["binding"] and capture["sdk"]==after["manifest_sha256"]
assert capture["passed"]==capture["planned"]==3 and capture["failed"]==capture["not_run"]==capture["skipped"]==0
cases=[]
for row in capture["cases"]:
    assert row["status"]=="passed"and row["budget_stop"]is None and row["background_survived"]is False
    ref=read(run/row["case"]/"reference-report.json");finite=ref["finite_reference"]
    assert ref["reference_status"]=="passed"and ref["probes"]==finite["groups"]==row["groups"]==16
    assert finite["compared_values"]==row["values"] and finite["max_reference_tolerance_ratio"]<=1 and finite["max_normalization_tolerance_ratio"]<=1
    assert not finite["compiler_validated"] and not finite["encrypted_execution_validated"]and finite["paid_calls"]==0
    assert finite["comparison_tolerance"]==dict(atol=1e-12,rtol=1e-12)
    case=next(c for c in plan["cases"]if c["name"]==row["case"])
    source=read(case["manifest"]);retained=run/row["case"]/"trusted-python"
    assert read(retained/"manifest.json")==source
    assert all(sha(retained/name)==value for name,value in source["files"].items())
    assert all(sha(retained/item["file"])==item["sha256"]for item in source["weights"].values())
    artifacts={name:sha(run/row["case"]/name)for name in("model.json","original.json","decomposition.json","reference-report.json","capture-input.json","capture.log","fx-input-source.txt","fx-public-normalized.txt")}
    cases.append(dict(case=row["case"],groups=row["groups"],values=row["values"],source_files=len(source["files"]),weights=len(source["weights"]),
     reference_max_absolute_error=finite["max_original_torch_vs_independent_ideal"],decomposition_max_absolute_error=finite["max_decomposition_error"],
     source_format=row["source_format"],reference_binding=finite["binding"],artifacts=artifacts,bytes=live.measure(run/row["case"])))
assert sum(c["groups"]for c in cases)==48 and sum(c["values"]for c in cases)==224
for name in("cli-controls.json","provenance-controls.json"):
    value=read(HERE/name);check_binding(value)
    assert value["passed"]==3 and value["failed"]==value["skipped"]==0
    assert value["compiler_calls"]==value["encrypted_calls"]==value["provider_calls"]==0
requests=read(HERE/"prepared-requests.json");check_binding(requests)
assert requests["sdk"]==after["manifest_sha256"]and requests["prepared"]==requests["distinct_requests"]==3
for row in requests["cases"]:
    assert sha(row["request_file"])==row["request_file_sha256"]and sha(row["public_request_file"])==row["public_request_file_sha256"]
oldscope=read(BASE/"benchmarks/tools/static_broadcast_intake_v1/prepared-source-scope.json")
for row in oldscope["cases"]:
    assert sha(row["request_file"])==row["request_file_sha256"]and sha(row["public_request_file"])==row["public_request_sha256"]
    assert read(row["request_file"])["request_id"]==row["request_id"]
first=read(HERE/"controls-first.json")
assert first["tests"]==300 and first["passed"]==287 and first["failed"]==6 and first["errors"]==7
failed_capture=[]
for folder in("agent-source-package-capture-v1","agent-source-package-capture-v3"):
    path=run.parent/folder/"report.json";value=read(path);check_binding(value)
    assert value["passed"]==0 and value["failed"]==1 and value["not_run"]==2
    row=value["cases"][0]
    assert row["error_type"]=="SubprocessError"and "preexec_fn" in row["reason"]
    assert (path.parent/row["case"]/"capture.log").stat().st_size==0
    failed_capture.append(dict(report=str(path),sha256=sha(path),failure_layer="development_launcher",worker_started=False))
actual_bytes=live.measure(Path(plan["results_root"]))
assert actual_bytes<=plan["budget"]["global_result_bytes"]
assert live.measure(run)<=plan["budget"]["new_artifact_bytes"]
target_paths=list(read(HERE/"core-patch-plan.json"))+["scripts/prepare_pytorch_source.py",
 "scripts/baseline/test_application_source_versions.py","scripts/baseline/test_pytorch_source_package.py",
 "scripts/baseline/test_pytorch_source_package_intake.py"]
check=subprocess.run(["git","diff","--check","--",*target_paths],cwd=ROOT,capture_output=True,text=True)
assert check.returncode==0,check.stdout
report=dict(format="poseidon-source-package-intake-audit-v1",status="passed",head="7588e48093631eaf8385506e3db058e7ca3cfe7e",
 branch="lhy-agent-dsl",sdk_before=before["manifest_sha256"],sdk_after=after["manifest_sha256"],stage_differences=stage_diffs,
 backend_parameters_dependencies_unchanged=True,versions=versions,source_package=package,
 controls=dict(passed=301,failed=0,errors=0,skipped=0,unique_test_ids=301),cli_controls=dict(passed=3,failed=0,skipped=0),
 provenance_controls=dict(passed=3,failed=0,skipped=0),capture=dict(passed=3,failed=0,skipped=0,not_run=0,groups=48,values=224,
 max_reference_absolute_error=max(c["reference_max_absolute_error"]for c in cases),cases=cases),
 history=dict(initial_controls_preserved=True,initial_controls=dict(passed=287,failed=6,errors=7),failed_capture_launches=failed_capture,
 old_request_files_unchanged=6,old_request_ids_unchanged=3,old_encrypted_qualifications_not_relabelled=True),
 prepared_requests=dict(binding=requests["binding"],count=3),resources=dict(results_bytes=actual_bytes,global_limit_bytes=plan["budget"]["global_result_bytes"],
 remaining_bytes=plan["budget"]["global_result_bytes"]-actual_bytes,capture_bytes=live.measure(run)),
 calls=dict(successful_source_captures=3,worker_launch_failures=2,compiler=0,encrypted=0,provider=0),
 git_diff_check="passed",source_changes={name:sha(ROOT/name)for name in target_paths},no_live_owner=True)
report["evidence"]={name:sha(HERE/name)for name in("controls-corrected.json","controls-corrected.log","cli-controls.json",
 "provenance-controls.json","prepared-requests.json","capture-plan-v4.json","identity-before.json","identity-after-controls.json")}
report["binding"]=digest(report)
with(HERE/"audit-r1437.json").open("x")as f:json.dump(report,f,sort_keys=True,indent=2)
print(json.dumps({k:report[k]for k in("status","binding","sdk_after","controls","calls","resources")}),flush=True)
