"""Verify existing capture records and refine rejection causes without rerunning models."""
import hashlib,json,sys,time
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[6] if Path(__file__).resolve().parents[4].name == "DSL_agent" and Path(__file__).resolve().parents[5].name == "DSL" else Path(__file__).resolve().parents[4])
sys.path.insert(0,str(ROOT/"scripts/baseline"))
from benchmark_graph import canonical,digest,require
from benchmarks.tools.audit_operator_graph_versions import preservation,runtime_identity
from benchmarks.tools.audit_public_operand_capture import FIXTURES,sha

def checked_binding(value):
    require(type(value) is dict and digest({k:v for k,v in value.items() if k!="binding"})==value.get("binding"),
            "Audit report binding")
    return value

def classify_rejection(case,log):
    """Only exact terminal errors corroborated by the known fixture are reclassified."""
    lines=[s.strip() for s in log.splitlines() if s.strip()]
    terminal=lines[-1] if lines else ""
    if case["public_side"]!="left":
        return None
    if terminal in ("ValueError: First operand must be encrypted",\
                    "ValueError: node fx_sub (subtract): First operand must be encrypted") and case["public_kind"]=="scalar" and case["operator"]=="subtract":
        return dict(layer="agent_input_decomposition",code="PUBLIC_FIRST_OPERAND",
                    reason="First operand must be encrypted")
    targets={"add":"add","subtract":"sub","multiply":"mul"}
    target=targets.get(case["operator"])
    if case["public_kind"]=="tensor" and target is not None and terminal=="ValueError: Unsupported PyTorch call: "+target:
        return dict(layer="trusted_pytorch_normalization",code="UNMAPPED_TENSOR_ARITHMETIC_METHOD",
                    reason="Unsupported PyTorch call: "+target)
    return None

def review(capture_dir,output,proposal):
    from trusted_pytorch_import import source_identity,check_reference
    from model_resources import resolve
    from semantic_benchmark_execution import runtime_sources
    capture_dir=Path(capture_dir);output=Path(output)
    require(not output.exists(),"Fresh capture-review output")
    started=time.monotonic();raw=checked_binding(json.loads((capture_dir/"report.json").read_text()))
    require(raw["planned"]==raw["completed"]==12 and raw["not_run"]==0,"Completed original capture inventory")
    frozen=preservation(proposal);runtime=runtime_identity();sources=runtime_sources()
    require(frozen==raw["frozen_proposal"] and runtime==raw["runtime"] and
            source_identity()==raw["capture_source_identity"],"Actual bound capture implementation")
    require(raw["fixture_hashes"]=={str(p.relative_to(FIXTURES)):sha(p) for p in FIXTURES.rglob("*") if p.is_file()},
            "Bound capture fixtures")
    catalog=checked_binding(json.loads((FIXTURES/"catalog.json").read_text()))
    expected={c["id"]:c for c in catalog["cases"]}
    require(len(expected)==12 and len({r["id"] for r in raw["rows"]})==12 and
            set(expected)=={r["id"] for r in raw["rows"]},"Capture cases preserved")
    rows=[];evidence={str(capture_dir/"report.json"):sha(capture_dir/"report.json")}
    accepted=groups=0;maximum=0.
    for row in raw["rows"]:
        case=expected[row["id"]];folder=capture_dir/"captures"/row["id"]
        spec=json.loads((FIXTURES/case["manifest"]).read_text())
        require([row[k] for k in ("public_kind","public_side")]==[case[k] for k in ("public_kind","public_side")] and
                row["operator"]==case["op"],"Original capture role identity")
        payload=json.loads((folder/"capture-input.json").read_text())
        require(payload["manifest"]==spec and payload["resource_profile"]=="general-tensors-v1",
                "Actual isolated capture request")
        reviewed={k:row[k] for k in ("id","operator","public_kind","public_side","capture_admission")}
        if row["capture_admission"]=="accepted":
            require(case["expected_capture"]=="accepted","Unexpected capture acceptance")
            reference=json.loads((folder/"reference-report.json").read_text())
            original=json.loads((folder/"original.json").read_text())
            core=json.loads((folder/"model.json").read_text())
            require(sha(folder/"reference-report.json")==row["capture_report_sha256"],
                    "Original captured reference file")
            check_reference(reference,original,core,spec,resource_profile="general-tensors-v1")
            require(reference["probes"]==16 and reference["original_module_checked"] is True,
                    "Actual isolated module comparison")
            reviewed.update(reference_status="passed",reference_groups=16)
            accepted+=1;groups+=16
            maximum=max(maximum,reference["max_decomposition_error"],reference["max_original_module_error"])
        else:
            require(row["capture_admission"]=="rejected" and case["expected_capture"]=="rejected",
                    "Unexpected capture rejection")
            log=(folder/"capture.log").read_text()
            require(sha(folder/"capture.log")==row["log_sha256"],"Original rejected capture log")
            cause=classify_rejection(row,log);require(cause is not None,"Unclassified capture rejection")
            reviewed.update(failure=cause,reference_status="not_run")
        rows.append(reviewed)
        for path in folder.rglob("*"):
            if path.is_file():evidence[str(path)]=sha(path)
    require(runtime_sources()==sources and preservation(proposal)==frozen and runtime_identity()==runtime,
            "Frozen identities changed")
    report=dict(format="poseidon-public-operand-capture-review-v1",planned=12,records_checked=12,
        original_isolated_capture_accepted=accepted,original_isolated_capture_rejected=12-accepted,
        unexpected_rejections=0,module_reference_groups=groups,max_reference_absolute_error=maximum,
        rows=rows,evidence_hashes=evidence,frozen_proposal=frozen,runtime=runtime,
        original_report_binding=raw["binding"],original_classification_mismatches=raw["unexpected_outcomes"],
        review_tool_sha256=sha(__file__),seconds=time.monotonic()-started,
        new_capture_runs=0,production_adapter_changed=False,
        compiled=False,hecate_tracing=False,encrypted_execution=False,agent_generated=False,paid_calls=0,
        boundary="Verified original 8 accepted / 4 rejected records; refined causes do not turn rejection into pass")
    report["binding"]=digest(report);output.mkdir(mode=0o700);(output/"report.json").write_bytes(canonical(report))
    print(json.dumps({k:v for k,v in report.items() if k not in ("rows","runtime","evidence_hashes")},ensure_ascii=False))
    return 0

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--capture-dir",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--frozen-proposal",type=Path,required=True)
    args=p.parse_args()
    raise SystemExit(review(args.capture_dir,args.output,args.frozen_proposal))
