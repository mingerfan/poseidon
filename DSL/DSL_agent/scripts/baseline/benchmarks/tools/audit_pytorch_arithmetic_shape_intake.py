"""Public V9 intake/reference audit; never compiler, Provider or FHE acceptance."""
import argparse,copy,hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[6] if Path(__file__).resolve().parents[4].name == "DSL_agent" and Path(__file__).resolve().parents[5].name == "DSL" else Path(__file__).resolve().parents[4])
sys.path.insert(0,str(ROOT/"scripts/baseline"))
from benchmark_graph import canonical,digest,require,samples
from strict_json_files import strict_file
PROFILE="general-tensors-v1"
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def checked(value):
    require(type(value) is dict and value.get("binding")==digest({k:v for k,v in value.items() if k!="binding"}),
            "Audit data binding")
    return value
def run(fixtures,output):
    import numpy as np
    from hecate_python_env import VENV,WORK
    from pytorch_model_intake import source_identity,check_reference,SHAPE_FORMAT,SHAPE_PACKAGE_FORMAT
    from trusted_pytorch_import import runtime_identity
    from candidate_bundle import load_model_provenance,check_provenance
    from operator_arithmetic_graph import decompose,evaluate_reference
    from benchmark_math import evaluate as mathematics
    from benchmark_torch import evaluate as independent
    from component_contract import prepare_task
    require(os.environ.get("IN_NIX_SHELL")=="pure" and Path(sys.prefix)==VENV,"Pinned pure Python")
    output=Path(output);fixtures=Path(fixtures)
    require(not output.exists() and output.parent.resolve().is_relative_to(WORK/"results"),"Fresh audit output")
    catalog=checked(strict_file(fixtures/"catalog.json",131072))
    require(catalog["format"]=="poseidon-pytorch-arithmetic-shape-fixtures-v2" and
            len(catalog["cases"])==36,"Exact owned V9 fixtures")
    old_fixtures=ROOT/"scripts/baseline/cases/trusted-pytorch-shape-composition-v1"
    old_capture=WORK/"results/agent-pytorch-shape-capture-r955/captures"
    review=checked(strict_file(WORK/"results/agent-pytorch-shape-review-r956/report.json",4*1024**2))
    old_rows={r["id"]:r for r in review["rows"]}
    fixture_hashes={str(p):sha(p) for root in (fixtures,old_fixtures) for p in root.rglob("*") if p.is_file()}
    old_evidence={str(p):sha(p) for p in [WORK/"results/agent-pytorch-shape-review-r956/report.json",
        ROOT/"docs/baseline/agent-cache-delivery-acceptance-r968.json",
        ROOT/"docs/baseline/agent-pytorch-shape-composition-acceptance-r960.json",
        ROOT/"docs/baseline/agent-arithmetic-matrix-offline-proposal-r950.json"]}
    current=source_identity();runtime=runtime_identity();tool=sha(__file__)
    output.mkdir(mode=0o700);(output/"captures").mkdir();started=time.monotonic()
    rows=[];legacy_rows=[];tamper=0;halt=None
    def preserved():
        require(source_identity()==current and runtime_identity()==runtime and sha(__file__)==tool,"Source/runtime/tool changed")
        require(all(sha(p)==h for p,h in fixture_hashes.items()),"Fixture bytes changed")
        require(all(sha(p)==h for p,h in old_evidence.items()),"Historical evidence changed")
    def capture_case(manifest,destination,case):
        remaining=240-(time.monotonic()-started)
        require(remaining>10,"Intake audit 240-second bound")
        command=[str(VENV/"bin/python"),"-B",str(ROOT/"scripts/prepare_model.py"),
            "--trusted-pytorch-manifest",str(manifest),"--trust-local-code",
            "--resource-profile",PROFILE,"--write","--output",str(destination),"--inside"]
        child=subprocess.run(command,capture_output=True,text=True,timeout=min(185,remaining))
        log=destination.parent/(destination.name+"-cli.log");log.write_text(child.stdout+child.stderr)
        row=dict(id=case["id"],cli_exit=child.returncode,cli_log_sha256=sha(log),
                 kind=case["kind"],capture_status="rejected" if child.returncode else "captured")
        if child.returncode:
            require((destination/"capture.log").is_file(),"No isolated capture diagnostic")
            text=(destination/"capture.log").read_text()
            terminal=text.splitlines()[-1] if text.splitlines() else ""
            reason=case.get("rejection")
            valid=(reason=="encrypted_branch" and terminal.startswith("torch.fx.proxy.TraceError: symbolically traced variables cannot be used as inputs to control flow")
                or reason=="inplace" and terminal=="ValueError: In-place arithmetic is unsupported"
                or reason=="out_mutation" and terminal=="ValueError: Arithmetic out mutation is unsupported"
                or reason=="old_metadata" and terminal=="ValueError: Static shape unavailable for this operation")
            require(valid,"Unexpected public capture failure; preserve log")
            row.update(correct_rejection=True,failure_layer="input",terminal=terminal,
                       capture_log_sha256=sha(destination/"capture.log"))
            return row,None,None,None
        require(case["kind"]=="positive","Forbidden source unexpectedly accepted")
        core=strict_file(destination/"model.json",131072)
        provenance=load_model_provenance(destination,core);check_provenance(provenance,core)
        spec=strict_file(manifest,131072);report=provenance["reference_report"]
        check_reference(report,provenance["original"],core,spec,resource_profile=PROFILE)
        require(report["probes"]==16 and report["normalized_module_checked"] is True,"Original/normalized references")
        row.update(probes=16,reference_sha256=sha(destination/"reference-report.json"),model_sha256=digest(core),
                   provenance_sha256=digest(provenance),format=spec["format"])
        return row,core,provenance,report
    try:
        for case in catalog["cases"]:
            preserved()
            destination=output/"captures"/case["id"]
            manifest=fixtures/case["manifest"]
            row,core,provenance,report=capture_case(manifest,destination,case)
            row.update(operator=case["operator"],style=case["style"],query=case["query"],
                       mathematical_group=case["mathematical_group"])
            if core is not None:
                require(provenance["trusted_pytorch"]["format"]==SHAPE_PACKAGE_FORMAT and
                        report["capture_metadata"]["source_shape_capture"]["version"]=="signature-static-arithmetic-shape-v1",
                        "V9 source policy binding")
                graph=strict_file(fixtures/case["graph"],131072)
                reference_core,_=decompose(graph,resource_profile=PROFILE);maximum=0.
                for inputs in samples(core,16):
                    expected=evaluate_reference(graph,inputs)["result"]
                    for model in (core,reference_core):
                        for fn in (mathematics,independent):
                            actual=fn(model,inputs,resource_profile=PROFILE)["result"]
                            np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=1e-12)
                            maximum=max(maximum,float(np.max(np.abs(actual-expected))))
                row["json_reference_groups"]=16;row["maximum_reference_error"]=maximum
                q=prepare_task(core,dict(generation_guidance="explicit-v10"))
                row["request_id"]=q["request_id"]
                # Retained historical successes must keep their mathematical graph.
                if old_rows[case["id"]]["capture_status"]=="captured":
                    old=strict_file(old_capture/case["id"]/"model.json",131072)
                    row["historical_core_unchanged"]=digest(old)==digest(core)
                    require(row["historical_core_unchanged"],"Old accepted mathematical graph changed")
                for mutation in ("metadata_version","signature","manifest_version","source_content","weight_hash","resource_count"):
                    bad=copy.deepcopy(provenance)
                    if mutation=="metadata_version":bad["reference_report"]["capture_metadata"]["source_shape_capture"]["version"]="signature-static-index-v1"
                    elif mutation=="signature":bad["reference_report"]["capture_metadata"]["source_shape_capture"]["input_signature_sha256"]="0"*64
                    elif mutation=="manifest_version":bad["trusted_pytorch"]["manifest"]["format"]="poseidon-trusted-pytorch-v8"
                    elif mutation=="source_content":
                        name=next(iter(bad["trusted_pytorch"]["files"]));bad["trusted_pytorch"]["files"][name]+="\n"
                    elif mutation=="weight_hash":bad["trusted_pytorch"]["manifest"]["weights"]["public"]["sha256"]="0"*64
                    else:bad["reference_report"]["capture_metadata"]["source_shape_capture"]["max_fx_compute_nodes"]=True
                    try:check_provenance(bad,core)
                    except (ValueError,TypeError,KeyError):tamper+=1
                    else:raise ValueError("Altered provenance was accepted")
            rows.append(row)
        # Compatibility: all nine retained old failures and six old accepted sources.
        selected=[c for c in catalog["cases"] if c["kind"]=="positive" and
            ((c["style"]=="method" and c["query"] in ("result_shape","result_size","result_numel"))
             or (c["style"] in ("function","operator") and c["query"]=="result_shape"))]
        require(len(selected)==15,"Legacy compatibility scope")
        for case in selected:
            previous=old_rows[case["id"]]
            source_case=copy.deepcopy(case)
            if previous["capture_status"]=="rejected":
                source_case.update(kind="rejection",rejection="old_metadata")
            folder=output/"captures"/("old_"+case["id"])
            row,core,provenance,report=capture_case(old_fixtures/case["manifest"],folder,source_case)
            if core is not None:
                old=strict_file(old_capture/case["id"]/"model.json",131072)
                require(digest(old)==digest(core),"V8 mathematical graph compatibility")
                row["old_core_unchanged"]=True
                require("source_shape_capture" not in report["capture_metadata"],"V8 policy changed")
            legacy_rows.append(row)
    except Exception as error:
        halt=type(error).__name__+": "+str(error)
    preserved()
    positive=[r for r in rows if r["kind"]=="positive"]
    negative=[r for r in rows if r["kind"]=="rejection"]
    report=dict(format="poseidon-pytorch-arithmetic-shape-intake-audit-v1",planned_positive=33,
        planned_rejections=3,passed=sum(r["capture_status"]=="captured" for r in positive),
        failed=sum(r["capture_status"]!="captured" for r in positive),rejections_passed=len(negative),
        not_run=36-len(rows),legacy_compatibility_planned=15,legacy_compatibility_completed=len(legacy_rows),
        legacy_successes=sum(r["capture_status"]=="captured" for r in legacy_rows),
        legacy_rejections=sum(r["capture_status"]=="rejected" for r in legacy_rows),
        source_variants=33,mathematical_groups=3,reference_groups=sum(r.get("probes",0) for r in rows),
        json_reference_groups=sum(r.get("json_reference_groups",0) for r in rows),
        maximum_reference_error=max([r.get("maximum_reference_error",0) for r in rows] or [0]),
        provenance_rejections=tamper,rows=rows,legacy_rows=legacy_rows,halt=halt,
        runtime=runtime,capture_identity=current,fixture_hashes=fixture_hashes,old_evidence=old_evidence,
        tool_sha256=tool,seconds=time.monotonic()-started,
        compiled=0,encrypted_execution=0,decrypted=0,agent_generated=0,paid_calls=0,backend_changes=0,
        scope="Public CLI, isolated trusted PyTorch capture and finite independent references; no DSL compilation/FHE success")
    report["binding"]=digest(report);(output/"report.json").write_bytes(canonical(report))
    print(json.dumps({k:v for k,v in report.items() if k not in ("rows","legacy_rows","capture_identity","fixture_hashes","old_evidence")},ensure_ascii=False))
    return 0 if not halt and len(rows)==36 and len(legacy_rows)==15 else 1
if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();raise SystemExit(run(args.fixtures,args.output))
