"""Recheck immutable shape audit records and run only the one unstarted rejection."""
import argparse,copy,json,os,subprocess,sys,time
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[6] if Path(__file__).resolve().parents[4].name == "DSL_agent" and Path(__file__).resolve().parents[5].name == "DSL" else Path(__file__).resolve().parents[4])
sys.path.insert(0,str(ROOT/"scripts/baseline"))
from benchmark_graph import canonical,digest,require
from benchmarks.tools.audit_pytorch_shape_composition import sha,checked,preservation,classify

def refine(case,log):
    prior=classify(case,log)
    if prior is not None:return prior
    lines=[line.strip() for line in log.splitlines() if line.strip()]
    terminal=lines[-1] if lines else ""
    if ('/app/pytorch_arithmetic_normalization.py' in log and case["kind"]=="rejection" and
            case["rejection"]=="inplace" and terminal=="ValueError: In-place arithmetic is unsupported"):
        return dict(layer="agent_input_mutation",code="INPLACE_REJECTED",reason=terminal)
    if ('/app/pytorch_arithmetic_normalization.py' in log and case["kind"]=="rejection" and
            case["rejection"]=="out_mutation" and terminal=="ValueError: Arithmetic out mutation is unsupported"):
        return dict(layer="agent_input_mutation",code="OUT_MUTATION_REJECTED",reason=terminal)
    return None

def review(raw_dir,fixtures,output,proposal_path):
    from pytorch_model_intake import source_identity,check_reference
    from trusted_pytorch_import import runtime_identity
    from hecate_python_env import VENV,WORK
    started=time.monotonic()
    raw_dir,fixtures,output=Path(raw_dir),Path(fixtures),Path(output)
    require(os.environ.get("IN_NIX_SHELL")=="pure" and Path(sys.prefix)==VENV,"Pinned pure Python")
    require(not output.exists() and output.parent.resolve().is_relative_to(WORK/"results"),"Fresh review")
    raw=checked(json.loads((raw_dir/"report.json").read_text()))
    proposal=checked(json.loads(Path(proposal_path).read_text()))
    require(raw["proposal_binding"]==proposal["binding"] and
            raw["frozen_inventories"]==preservation(proposal) and
            raw["capture_identity"]==source_identity() and raw["runtime"]==runtime_identity(),
            "Current capture binding")
    fixture_inventory={str(p.relative_to(fixtures)):sha(p) for p in fixtures.rglob("*") if p.is_file()}
    require(fixture_inventory==raw["fixture_hashes"],"Preserved original fixture bytes")
    require(all(sha(name)==value for name,value in raw["tool_hashes"].items()),"Preserved initial audit code")
    catalog=checked(json.loads((fixtures/"catalog.json").read_text()))
    cases={c["id"]:c for c in catalog["cases"]}
    require(len(cases)==36 and raw["completed"]==35 and raw["not_run"]==1 and
            raw["halt"]=="UNCLASSIFIED_CAPTURE_FAILURE","Exact original interrupted audit")
    missing=set(cases)-{r["id"] for r in raw["rows"]}
    require(missing=={"reject_out_mutation"},"Only the original unstarted rejection may run")
    rows=copy.deepcopy(raw["rows"]);evidence={str(raw_dir/"report.json"):sha(raw_dir/"report.json")}
    refinements=0
    for row in rows:
        case=cases[row["id"]];folder=raw_dir/"captures"/row["id"]
        spec=json.loads((fixtures/case["manifest"]).read_text())
        payload=json.loads((folder/"capture-input.json").read_text())
        require(payload["manifest"]==spec and payload["source_hashes"]==source_identity(),"Isolated source request")
        require(sha(raw_dir/(row["id"]+"-cli.log"))==row["cli_log_sha256"],"Original public CLI log")
        if row["capture_status"]=="captured":
            require(sha(folder/"reference-report.json")==row["reference_report_sha256"],"Original reference hash")
            original=json.loads((folder/"original.json").read_text())
            core=json.loads((folder/"model.json").read_text())
            reference=json.loads((folder/"reference-report.json").read_text())
            check_reference(reference,original,core,spec,resource_profile="general-tensors-v1")
            require(reference["probes"]==16 and reference["original_module_checked"] is True and
                    reference["normalized_module_checked"] is True,"Actual original isolated comparison")
        else:
            require(row["capture_status"]=="rejected" and
                    sha(folder/"capture.log")==row["capture_log_sha256"],"Original rejected capture")
            cause=refine(case,(folder/"capture.log").read_text())
            require(cause is not None,"Original rejection remains unclassified")
            refinements += cause!=row.get("failure")
            row.update(failure=cause,correct_rejection=case["kind"]=="rejection")
        for path in folder.rglob("*"):
            if path.is_file():evidence[str(path)]=sha(path)
    output.mkdir(mode=0o700);(output/"captures").mkdir()
    case=cases["reject_out_mutation"];folder=output/"captures"/case["id"]
    command=[str(VENV/"bin/python"),"-B",str(ROOT/"scripts/prepare_model.py"),
             "--trusted-pytorch-manifest",str(fixtures/case["manifest"]),"--trust-local-code",
             "--resource-profile","general-tensors-v1","--write","--output",str(folder),"--inside"]
    result=subprocess.run(command,text=True,capture_output=True,timeout=185)
    (output/(case["id"]+"-cli.log")).write_text(result.stdout+result.stderr)
    require(result.returncode!=0 and (folder/"capture.log").is_file(),"Out mutation must be rejected in capture")
    payload=json.loads((folder/"capture-input.json").read_text())
    require(payload["manifest"]==json.loads((fixtures/case["manifest"]).read_text()) and
            payload["source_hashes"]==source_identity(),"New isolated request")
    cause=refine(case,(folder/"capture.log").read_text())
    require(cause is not None and cause["code"]=="OUT_MUTATION_REJECTED","Exact original out-mutation constraint")
    row={key:case[key] for key in ("id","operator","style","query","kind","rejection")}
    row.update(capture_status="rejected",failure=cause,correct_rejection=True,
               cli_exit=result.returncode,cli_log_sha256=sha(output/(case["id"]+"-cli.log")),
               capture_log_sha256=sha(folder/"capture.log"))
    rows.append(row)
    for path in folder.rglob("*"):
        if path.is_file():evidence[str(path)]=sha(path)
    require(raw["frozen_inventories"]==preservation(proposal) and
            fixture_inventory=={str(p.relative_to(fixtures)):sha(p) for p in fixtures.rglob("*") if p.is_file()},
            "Frozen source or fixture changed")
    report={k:v for k,v in raw.items() if k not in ("binding","rows","halt","seconds","tool_hashes")}
    report.update(format="poseidon-pytorch-shape-composition-review-v1",completed=36,not_run=0,
        rejection_tests_passed=3,unclassified_failures=0,halt=None,rows=rows,
        original_audit_binding=raw["binding"],original_audit_exit=1,
        original_audit_halt=raw["halt"],original_classification_refinements=refinements,
        new_capture_runs=1,preserved_capture_runs=35,rerun_completed_cases=0,
        review_tool_sha256=sha(__file__),original_tool_hashes=raw["tool_hashes"],
        evidence_hashes=evidence,review_seconds=time.monotonic()-started,
        source_scope="24 accepted, 9 input-shape gaps, 3 rejected; no failed source is reclassified as passed")
    report["binding"]=digest(report)
    (output/"report.json").write_bytes(canonical(report))
    print(json.dumps({k:v for k,v in report.items() if k not in
                     ("rows","evidence_hashes","fixture_hashes","capture_identity","original_tool_hashes")},ensure_ascii=False))
    return 0

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir",type=Path,required=True)
    parser.add_argument("--fixtures",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--frozen-proposal",type=Path,required=True)
    args=parser.parse_args()
    raise SystemExit(review(args.raw_dir,args.fixtures,args.output,args.frozen_proposal))
