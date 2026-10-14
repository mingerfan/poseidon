"""Developer-only audit of the unmodified public V8 intake, not FHE acceptance."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[6] if Path(__file__).resolve().parents[4].name == "DSL_agent" and Path(__file__).resolve().parents[5].name == "DSL" else Path(__file__).resolve().parents[4])
sys.path.insert(0,str(ROOT/"scripts/baseline"))
from benchmark_graph import canonical,digest,require
from benchmarks.tools.generate_pytorch_shape_composition_sources import VERSION

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def checked(value):
    require(type(value) is dict and digest({k:v for k,v in value.items() if k!="binding"})==value.get("binding"),
            "Audit/proposal binding")
    return value

def preservation(proposal):
    require(type(proposal.get("source_hashes")) is dict and
            type(proposal.get("case_hashes")) is dict and
            type(proposal.get("tool_hashes")) is dict,"Frozen inventory")
    for category in ("source_hashes","case_hashes","tool_hashes"):
        for name,expected in proposal[category].items():
            path=ROOT/name
            require(path.is_file() and sha(path)==expected,"Frozen identity changed: "+name)
    return {key:digest(proposal[key]) for key in ("source_hashes","case_hashes","tool_hashes")}

def classify(case,log):
    lines=[line.strip() for line in log.splitlines() if line.strip()]
    terminal=lines[-1] if lines else ""
    # Only a precise fixture + failing initial trace yields this causal attribution.
    if (case["kind"]=="positive" and case["style"]=="method" and
            case["query"] in ("result_shape","result_size","result_numel") and
            terminal=="ValueError: Static shape unavailable for this operation" and
            'traced=trace(module,spec["inputs"],resource_profile=profile,policy=INDEX_POLICY)' in log):
        return dict(layer="agent_input_static_shape",
                    code="TENSOR_ARITHMETIC_RESULT_METADATA_MISSING",
                    reason="Initial static trace lacks metadata for arithmetic Tensor methods")
    if case["kind"]=="rejection" and case["rejection"]=="encrypted_branch" and terminal.startswith(
            "torch.fx.proxy.TraceError: symbolically traced variables cannot be used as inputs to control flow"):
        return dict(layer="agent_input_control_flow",code="ENCRYPTED_BRANCH_REJECTED",reason=terminal)
    if case["kind"]=="rejection" and case["rejection"]=="inplace" and terminal in (
            "ValueError: Unsupported PyTorch call: add_","ValueError: Unsupported FX method: add_",
            "ValueError: In-place arithmetic is not supported"):
        return dict(layer="agent_input_mutation",code="INPLACE_REJECTED",reason=terminal)
    if case["kind"]=="rejection" and case["rejection"]=="out_mutation" and terminal in (
            "ValueError: Arithmetic call keywords", "ValueError: Arithmetic keyword fields",
            "ValueError: Arithmetic normalization call shape"):
        return dict(layer="agent_input_mutation",code="OUT_MUTATION_REJECTED",reason=terminal)
    return None

def run(fixtures,output,proposal_path,seconds=300):
    import numpy as np
    from benchmark_graph import samples
    from benchmark_math import evaluate as mathematical
    from benchmark_torch import evaluate as independent
    from operator_arithmetic_graph import decompose,evaluate_reference
    from pytorch_model_intake import source_identity,check_reference
    from hecate_python_env import VENV,WORK
    from trusted_pytorch_import import runtime_identity
    require(os.environ.get("IN_NIX_SHELL")=="pure" and Path(sys.prefix)==VENV,"Pinned pure Python required")
    require(type(seconds) is int and 180<=seconds<=360,"Development capture time bound")
    output=Path(output);fixtures=Path(fixtures)
    require(not output.exists() and output.parent.resolve().is_relative_to(WORK/"results"),
            "Fresh audit results directory")
    proposal=checked(json.loads(Path(proposal_path).read_text()))
    frozen=preservation(proposal)
    catalog=checked(json.loads((fixtures/"catalog.json").read_text()))
    require(catalog["format"]==VERSION and len(catalog["cases"])==36 and
            len({r["id"] for r in catalog["cases"]})==36,"Fixture inventory")
    inventory={str(p.relative_to(fixtures)):sha(p) for p in fixtures.rglob("*") if p.is_file()}
    sources=source_identity();runtime=runtime_identity()
    tool_paths=[Path(__file__),Path(__file__).with_name("generate_pytorch_shape_composition_sources.py")]
    own={str(p):sha(p) for p in tool_paths}
    output.mkdir(mode=0o700);(output/"captures").mkdir()
    started=time.monotonic();rows=[];halt=None

    def write():
        positive=[r for r in rows if r["kind"]=="positive"]
        negative=[r for r in rows if r["kind"]=="rejection"]
        result=dict(format="poseidon-pytorch-shape-composition-audit-v1",
            planned_positive=33,planned_rejections=3,completed=len(rows),not_run=36-len(rows),
            positive_capture_passed=sum(r["capture_status"]=="captured" for r in positive),
            positive_capture_failed=sum(r["capture_status"]=="rejected" for r in positive),
            rejection_tests_passed=sum(r.get("correct_rejection",False) for r in negative),
            unexpected_acceptances=sum(r["capture_status"]=="captured" for r in negative),
            unclassified_failures=sum(r.get("failure") is not None and
                                      r["failure"].get("code")=="UNCLASSIFIED_CAPTURE_FAILURE" for r in rows),
            mathematical_groups=3,independent_json_reference_groups=sum(r.get("json_reference_groups",0) for r in rows),
            original_module_reference_groups=sum(r.get("module_reference_groups",0) for r in rows),
            paired_json_capture_groups=sum(r.get("paired_reference_groups",0) for r in rows),
            maximum_json_reference_error=max([r.get("max_json_error",0) for r in rows] or [0]),
            maximum_paired_reference_error=max([r.get("max_paired_error",0) for r in rows] or [0]),
            rows=rows,halt=halt,seconds=time.monotonic()-started,
            runtime=runtime,capture_identity=sources,fixture_hashes=inventory,tool_hashes=own,
            proposal_binding=proposal["binding"],frozen_inventories=frozen,
            production_source_changes=0,backend_changes=0,reference_changes=0,
            hecate_tracing=0,compiled=0,encrypted_execution=0,decrypted=0,
            new_agent_generation=0,paid_calls=0,installs=0,downloads=0,
            scope="Owned-source isolated PyTorch capture and independent plaintext references; not DSL/FHE acceptance")
        result["binding"]=digest(result)
        (output/"report.json").write_bytes(canonical(result))
        return result

    write()
    for case in catalog["cases"]:
        if time.monotonic()-started>seconds-30:
            halt="DEVELOPMENT_AUDIT_TIME_BOUND";break
        require(preservation(proposal)==frozen and source_identity()==sources,"Frozen capture changed")
        manifest=fixtures/case["manifest"];spec=json.loads(manifest.read_text())
        require(digest(spec)==case["manifest_sha256"],"Fixture manifest changed")
        row={key:case[key] for key in ("id","operator","style","query","kind","rejection")}
        json_core=None
        if case["kind"]=="positive":
            graph=json.loads((fixtures/case["graph"]).read_text())
            require(digest(graph)==case["graph_sha256"],"Fixture graph changed")
            json_core,binding=decompose(graph,resource_profile="general-tensors-v1")
            maximum=0.
            for inputs in samples(json_core,16):
                high=evaluate_reference(graph,inputs)
                math=mathematical(json_core,inputs,resource_profile="general-tensors-v1")
                torch=independent(json_core,inputs,resource_profile="general-tensors-v1")
                for name in high:
                    for actual in (math[name],torch[name]):
                        np.testing.assert_allclose(actual,high[name],atol=1e-12,rtol=1e-12)
                        maximum=max(maximum,float(np.max(np.abs(actual-high[name]))))
            row.update(json_reference_groups=16,max_json_error=maximum,
                       json_decomposition_binding=binding["binding_sha256"])
        destination=output/"captures"/case["id"]
        command=[str(VENV/"bin/python"),"-B",str(ROOT/"scripts/prepare_model.py"),
            "--trusted-pytorch-manifest",str(manifest),"--trust-local-code",
            "--resource-profile","general-tensors-v1","--write","--output",str(destination),"--inside"]
        remaining=seconds-(time.monotonic()-started)
        result=subprocess.run(command,text=True,capture_output=True,timeout=min(185,max(1,remaining)))
        (output/(case["id"]+"-cli.log")).write_text(result.stdout+result.stderr)
        row.update(cli_exit=result.returncode,cli_log_sha256=sha(output/(case["id"]+"-cli.log")))
        if result.returncode:
            if not (destination/"capture.log").is_file():
                halt="PRE_CAPTURE_ENVIRONMENT_OR_INTEGRITY_FAILURE"
                row.update(capture_status="not_started",failure=dict(
                    layer="capture_service",code=halt,reason="Public capture did not create isolated execution evidence"))
                rows.append(row);write();break
            log=(destination/"capture.log").read_text()
            cause=classify(case,log)
            row.update(capture_status="rejected",capture_log_sha256=sha(destination/"capture.log"),
                failure=cause or dict(layer="capture",code="UNCLASSIFIED_CAPTURE_FAILURE",
                                     reason=log.splitlines()[-1] if log.splitlines() else "No terminal diagnostic"),
                correct_rejection=case["kind"]=="rejection" and cause is not None)
            if cause is None:
                halt="UNCLASSIFIED_CAPTURE_FAILURE";rows.append(row);write();break
        else:
            require(case["kind"]=="positive","Unsafe source unexpectedly accepted")
            ref=json.loads((destination/"reference-report.json").read_text())
            high=json.loads((destination/"original.json").read_text())
            core=json.loads((destination/"model.json").read_text())
            check_reference(ref,high,core,spec,resource_profile="general-tensors-v1")
            require(ref["probes"]==16 and ref["reference_status"]=="passed" and
                    ref["original_module_checked"] is True and ref["normalized_module_checked"] is True,
                    "Actual isolated module reference")
            maximum=0.
            for inputs in samples(json_core,16):
                expected=mathematical(json_core,inputs,resource_profile="general-tensors-v1")["result"]
                actual=mathematical(core,inputs,resource_profile="general-tensors-v1")["result"]
                np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=1e-12)
                maximum=max(maximum,float(np.max(np.abs(actual-expected))))
            row.update(capture_status="captured",module_reference_groups=16,
                paired_reference_groups=16,max_paired_error=maximum,
                reference_report_sha256=sha(destination/"reference-report.json"))
        rows.append(row);write()
        print(json.dumps(dict(id=case["id"],kind=case["kind"],capture_status=row["capture_status"],
                              failure_code=row.get("failure",{}).get("code")),ensure_ascii=False),flush=True)
    require(preservation(proposal)==frozen and source_identity()==sources and runtime_identity()==runtime,
            "Identity changed during audit")
    require(inventory=={str(p.relative_to(fixtures)):sha(p) for p in fixtures.rglob("*") if p.is_file()} and
            own=={str(p):sha(p) for p in tool_paths},"Audit fixtures or tools changed")
    report=write()
    print(json.dumps({k:v for k,v in report.items() if k not in
                      ("rows","capture_identity","fixture_hashes","tool_hashes")},ensure_ascii=False))
    return 0 if not report["halt"] and report["not_run"]==0 else 1

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--frozen-proposal",type=Path,required=True)
    parser.add_argument("--seconds",type=int,default=300)
    args=parser.parse_args()
    raise SystemExit(run(args.fixtures,args.output,args.frozen_proposal,args.seconds))
