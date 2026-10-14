"""Exercise actual application submission, v9 compatibility and bounded input refusals."""
import argparse,copy,hashlib,json,os,re,subprocess,sys,time
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[6] if Path(__file__).resolve().parents[4].name == "DSL_agent" and Path(__file__).resolve().parents[5].name == "DSL" else Path(__file__).resolve().parents[4])
sys.path.insert(0,str(ROOT/"scripts/baseline"))
from benchmark_graph import canonical,digest,require
from benchmarks.tools.audit_pytorch_operator_shape_contexts import checked,frozen,classify,sha
PROFILE="general-tensors-v1"
def run(public,bindings,output):
    from hecate_python_env import VENV,WORK
    from pytorch_model_intake import check_reference,source_identity,SHAPE_REFERENCE_FORMAT,SHAPE_FORMAT,OPERATOR_SHAPE_FORMAT
    from pytorch_static_trace import ARITHMETIC_POLICY
    from candidate_bundle import load_model_provenance,check_provenance
    from application_component import Application
    from application_backend import PoseidonBackend
    require(os.environ.get("IN_NIX_SHELL")=="pure" and Path(sys.prefix)==VENV,"Pinned pure Python")
    require(not output.exists() and output.parent.resolve().is_relative_to(WORK/"results"),"Fresh application compatibility evidence")
    bound=checked(json.loads(bindings.read_text()));snapshot=frozen(bound);sources=source_identity()
    previous=checked(json.loads((public/"report.json").read_text()))
    require(previous["passed"]==39 and previous["input_rejected"]==0 and previous["capture_identity"]==sources,"Complete current public source intake")
    retained={str(p):sha(p) for p in public.rglob("*") if p.is_file()}
    output.mkdir(mode=0o700);(output/"compatibility").mkdir();started=time.monotonic()
    app=Application(output/"tasks",PoseidonBackend());application=[];tamper_rejections=0
    def reject(bad,original,model,spec):
        try:check_reference(bad,original,model,spec,resource_profile=PROFILE)
        except ValueError:return
        raise ValueError("Mutated v10 source evidence accepted")
    for row in previous["rows"]:
        path=public/"captures"/row["id"];model=json.loads((path/"model.json").read_text())
        provenance=load_model_provenance(path,model);check_provenance(provenance,model)
        spec=provenance["trusted_pytorch"]["manifest"];original=provenance["original"];report=provenance["reference_report"]
        require(spec["format"]==OPERATOR_SHAPE_FORMAT,"Explicit v10 source")
        for mutation in range(5):
            bad=copy.deepcopy(report);selected=copy.deepcopy(spec)
            if mutation==0:bad["capture_metadata"]["source_shape_capture"]["version"]=ARITHMETIC_POLICY
            elif mutation==1:selected["format"]=SHAPE_FORMAT
            elif mutation==2:bad["format"]=SHAPE_REFERENCE_FORMAT
            elif mutation==3:bad["source_hashes"]["pytorch_model_capture.py"]="0"*64
            else:bad["model_sha256"]="0"*64
            reject(bad,original,model,selected);tamper_rejections+=1
        task=app.submit_import(path,validation_level="numerical")
        state=app.status(task);require(state["status"]=="queued","Actual application input failed: "+row["id"])
        require(state["validation_level"]=="numerical" and state["generations"]==state["http_attempts"]==0
            and state["package"] is None and state["acceptance"] is None,"No generation or acceptance occurred")
        cancelled=app.cancel(task);require(cancelled["status"]=="cancelled" and app.result(task)["package"] is None,"Queued task cancellation")
        application.append(dict(id=row["id"],task=task,status="queued_then_cancelled",generation_count=0,http_attempts=0))
    fixture=ROOT/"scripts/baseline/cases/trusted-pytorch-operator-shape-v1"
    old=checked(json.loads((WORK/"results/agent-operator-shape-intake-r980/report.json").read_text()))
    catalog=checked(json.loads((fixture/"catalog.json").read_text()))
    failures={r["id"] for r in old["rows"] if r["status"]=="rejected"}
    controls={"adaptive_pool_0_module_literal","adaptive_pool_1_module_literal","adaptive_pool_2_module_literal",
        "average_pool_control_0_functional_query","instance_norm_0_module_literal","padding_1_functional_query"}
    selected=[c for c in catalog["cases"] if c["id"] in failures|controls];require(len(selected)==24,"Exact v9 compatibility scope")
    compatibility=[];guards=[]
    def invoke(manifest,destination):
        seconds=140-(time.monotonic()-started);require(seconds>0,"Input compatibility time bound")
        args=[str(VENV/"bin/python"),"-B",str(ROOT/"scripts/prepare_model.py"),"--trusted-pytorch-manifest",str(manifest),
            "--trust-local-code","--resource-profile",PROFILE,"--write","--output",str(destination),"--inside"]
        child=subprocess.run(args,capture_output=True,text=True,timeout=min(60,seconds))
        (output/"compatibility"/(destination.name+"-cli.log")).write_text(child.stdout+child.stderr)
        require((destination/"capture.log").exists(),"Isolated capture not started")
        return child.returncode,(destination/"capture.log").read_text()
    for case in selected:
        destination=output/"compatibility"/("v9_"+case["id"])
        code,log=invoke(fixture/case["manifest"],destination)
        if case["id"] in failures:
            cause=classify(case,log);require(code!=0 and cause is not None,"v9 retained rejection changed")
            compatibility.append(dict(id=case["id"],status="rejected_as_before",failure=cause))
        else:
            require(code==0,"v9 retained acceptance failed")
            model=json.loads((destination/"model.json").read_text())
            provenance=load_model_provenance(destination,model);check_provenance(provenance,model)
            require(provenance["trusted_pytorch"]["manifest"]["format"]==SHAPE_FORMAT,"Actual v9 path")
            compatibility.append(dict(id=case["id"],status="accepted"))
    original_guards=ROOT/"scripts/baseline/benchmarks/tools/fixtures/operator-shape-rejections-v1"
    case_inventory=json.loads((original_guards/"inventory.json").read_text())
    new_guard_root=output/"owned-v10-guards";new_guard_root.mkdir()
    for case in case_inventory["cases"]:
        manifest=Path(case["manifest"]);spec=json.loads(manifest.read_text());d=new_guard_root/case["id"];d.mkdir()
        for name,h in spec["files"].items():
            raw=(manifest.parent/name).read_bytes();require(hashlib.sha256(raw).hexdigest()==h,"Owned guard source")
            (d/name).write_bytes(raw)
        if not case["id"].startswith("old_v8"):spec["format"]=OPERATOR_SHAPE_FORMAT
        (d/"manifest.json").write_bytes(canonical(spec))
        code,log=invoke(d/"manifest.json",output/"compatibility"/("guard_"+case["id"]))
        terminal=log.splitlines()[-1];expected=case["expected_terminal"]
        if case["id"]=="adaptive_upsample":
            expected="ValueError: node fx_adaptive_avg_pool1d (adaptive_avg_pool): Adaptive pool N,C,spatial downsampling shape"
        match=code!=0 and (terminal.startswith(expected) if case["id"] in ("missing_norm_approximation","old_v8_functional") else terminal==expected)
        if case["id"]=="old_v8_functional":match=match and re.fullmatch(
            r"ValueError: Unsupported PyTorch call: <built-in method adaptive_avg_pool1d of type object at 0x[0-9a-f]+>",terminal) is not None
        require(match,"Actual public guard mismatch: "+case["id"]+" "+terminal)
        guards.append(dict(id=case["id"],status="rejected_as_expected",terminal=terminal))
    require(frozen(bound)==snapshot and source_identity()==sources and all(sha(p)==h for p,h in retained.items()),"Source and completed evidence unchanged")
    result=dict(format="poseidon-pytorch-operator-shape-application-compatibility-v1",
        public_report_binding=previous["binding"],source_identity=sources,application_planned=39,application_accepted=39,
        application_rows=application,reference_tamper_rejections=tamper_rejections,
        v9_positive_controls=6,v9_original_rejections_preserved=18,compatibility=compatibility,
        public_rejection_checks=len(guards),guards=guards,seconds=time.monotonic()-started,
        compiled=0,hecate_tracing=0,encrypted_execution=0,decrypted=0,generated=0,paid_calls=0,
        backend_changes=0,reference_changes=0,parameter_changes=0,scope="Submit/cancel and input provenance; no worker execution or program package")
    result["binding"]=digest(result);(output/"report.json").write_bytes(canonical(result))
    print(json.dumps({k:result[k] for k in ("binding","application_accepted","reference_tamper_rejections",
        "v9_positive_controls","v9_original_rejections_preserved","public_rejection_checks","seconds")}))
    return 0
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for n in ("public","bindings","output"):p.add_argument("--"+n,type=Path,required=True)
    a=p.parse_args();raise SystemExit(run(a.public,a.bindings,a.output))
