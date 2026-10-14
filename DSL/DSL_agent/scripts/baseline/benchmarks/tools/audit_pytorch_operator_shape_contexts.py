"""Audit current public operator/shape compositions without modifying runtime sources."""
import argparse,copy,hashlib,json,os,re,subprocess,sys,time
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[6] if Path(__file__).resolve().parents[4].name == "DSL_agent" and Path(__file__).resolve().parents[5].name == "DSL" else Path(__file__).resolve().parents[4])
sys.path.insert(0,str(ROOT/"scripts/baseline"))
from benchmark_graph import canonical,digest,require,samples
PROFILE="general-tensors-v1"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def checked(value):
    require(type(value) is dict and value.get("binding")==digest({k:v for k,v in value.items() if k!="binding"}),"Bound audit data")
    return value
def frozen(proposal):
    for key in ("source_hashes","case_hashes","tool_hashes"):
        require(all(sha(ROOT/p)==h for p,h in proposal[key].items()),"Frozen "+key)
    return {key:digest(proposal[key]) for key in ("source_hashes","case_hashes","tool_hashes")}
def classify(case,log):
    terminal=log.splitlines()[-1] if log.splitlines() else ""
    if (case["variant"].endswith("query") and '/model/models.py' in log and 'pytorch_static_trace.py' in log and
            terminal=="ValueError: Static shape unavailable for this operation"):
        return dict(layer="agent_input_static_shape",code="REGISTERED_OPERATOR_RESULT_METADATA_MISSING",
            group=case["group"],reason="Initial source trace lacks registered operator result metadata")
    function="adaptive_avg_pool"+str(case["rank"])+"d"
    if (case["group"]=="adaptive_pool" and case["variant"]=="functional_literal" and
            'pytorch_model_capture.py' in log and re.fullmatch(
                r"ValueError: Unsupported PyTorch call: <(?:function|built-in method) "+function+
                r"(?: of type object)? at 0x[0-9a-f]+>",terminal)):
        return dict(layer="agent_input_function_mapping",code="FUNCTIONAL_ADAPTIVE_POOL_MAPPING_MISSING",
            group=case["group"],reason="Functional spelling lacks the registered adaptive pool mapping")
    return None
def run(fixtures,output,proposal_path):
    import numpy as np
    from hecate_python_env import VENV,WORK
    from pytorch_model_intake import source_identity,check_reference
    from trusted_pytorch_import import runtime_identity
    from candidate_bundle import load_model_provenance,check_provenance
    from operator_arithmetic_graph import decompose,evaluate_reference
    from benchmark_math import evaluate as mathematics
    from benchmark_torch import evaluate as tensor_reference
    require(os.environ.get("IN_NIX_SHELL")=="pure" and Path(sys.prefix)==VENV,"Pinned pure Python")
    fixtures,output=Path(fixtures),Path(output)
    require(not output.exists() and output.parent.resolve().is_relative_to(WORK/"results"),"Fresh audit output")
    proposal=checked(json.loads(Path(proposal_path).read_text()))
    bound=frozen(proposal);own=sha(__file__);sources=source_identity();runtime=runtime_identity()
    catalog=checked(json.loads((fixtures/"catalog.json").read_text()))
    require(catalog["source_variants"]==39 and catalog["mathematical_groups"]==12,"Exact operator shape contexts")
    inventory={str(p):sha(p) for p in fixtures.rglob("*") if p.is_file()}
    output.mkdir(mode=0o700);(output/"captures").mkdir()
    started=time.monotonic();rows=[];halt=None
    def save():
        accepted=[r for r in rows if r.get("status")=="captured"]
        result=dict(format="poseidon-pytorch-operator-shape-audit-v1",planned=39,completed=len(rows),not_run=39-len(rows),
            passed=len(accepted),input_rejected=sum(r.get("status")=="rejected" for r in rows),
            unclassified=sum(r.get("failure",{}).get("code")=="UNCLASSIFIED_INPUT_FAILURE" for r in rows),
            mathematical_groups=12,independent_json_groups=sum(r.get("json_probes",0) for r in rows),
            isolated_module_reference_groups=sum(r.get("module_probes",0) for r in rows),
            paired_reference_groups=sum(r.get("paired_probes",0) for r in rows),
            maximum_reference_error=max([r.get("reference_error",0) for r in rows] or [0]),rows=rows,halt=halt,
            capture_identity=sources,runtime=runtime,fixture_hashes=inventory,tool_sha256=own,
            proposal_binding=proposal["binding"],frozen_inventories=bound,seconds=time.monotonic()-started,
            production_changes=0,backend_changes=0,reference_changes=0,hecate_tracing=0,compiled=0,
            encrypted_execution=0,decrypted=0,agent_generated=0,paid_calls=0,downloads=0,installs=0,
            scope="Owned source public FX intake and independent finite plaintext references; not FHE acceptance")
        result["binding"]=digest(result);(output/"report.json").write_bytes(canonical(result));return result
    save()
    for case in catalog["cases"]:
        require(frozen(proposal)==bound and source_identity()==sources and sha(__file__)==own,"Frozen inputs changed")
        if time.monotonic()-started>270:
            halt="AUDIT_TIME_BOUND";break
        row={k:case[k] for k in ("id","group","variant","query","shape","rank","mathematical_group")}
        graph=json.loads((fixtures/case["graph"]).read_text());require(digest(graph)==case["graph_sha256"],"Graph data binding")
        model,_=decompose(graph,resource_profile=PROFILE);maximum=0.
        for inputs in samples(model,16):
            expected=evaluate_reference(graph,inputs)["result"]
            for fn in (mathematics,tensor_reference):
                actual=fn(model,inputs,resource_profile=PROFILE)["result"]
                np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=1e-12)
                maximum=max(maximum,float(np.max(np.abs(actual-expected))))
        row.update(json_probes=16,reference_error=maximum)
        destination=output/"captures"/case["id"]
        spec=json.loads((fixtures/case["manifest"]).read_text())
        require(digest(spec)==case["manifest_sha256"],"Manifest data binding")
        command=[str(VENV/"bin/python"),"-B",str(ROOT/"scripts/prepare_model.py"),
            "--trusted-pytorch-manifest",str(fixtures/case["manifest"]),"--trust-local-code",
            "--resource-profile",PROFILE,"--write","--output",str(destination),"--inside"]
        child=subprocess.run(command,text=True,capture_output=True,timeout=min(185,280-(time.monotonic()-started)))
        cli=output/(case["id"]+"-cli.log");cli.write_text(child.stdout+child.stderr)
        row.update(cli_exit=child.returncode,cli_sha256=sha(cli))
        if child.returncode:
            if not (destination/"capture.log").exists():
                row.update(status="not_started",failure=dict(layer="capture_environment",code="PRE_CAPTURE_FAILURE"))
                rows.append(row);halt="PRE_CAPTURE_FAILURE";save();break
            log=(destination/"capture.log").read_text();cause=classify(case,log)
            row.update(status="rejected",capture_sha256=sha(destination/"capture.log"),
                failure=cause or dict(layer="input_capture",code="UNCLASSIFIED_INPUT_FAILURE",
                    reason=log.splitlines()[-1] if log.splitlines() else "Missing terminal diagnostic"))
        else:
            captured=json.loads((destination/"model.json").read_text())
            provenance=load_model_provenance(destination,captured);check_provenance(provenance,captured)
            report=provenance["reference_report"]
            check_reference(report,provenance["original"],captured,spec,resource_profile=PROFILE)
            require(report["probes"]==16 and report["original_module_checked"] and report["normalized_module_checked"],
                    "Original/normalized isolated module reference")
            for inputs in samples(model,16):
                expected=mathematics(model,inputs,resource_profile=PROFILE)["result"]
                actual=mathematics(captured,inputs,resource_profile=PROFILE)["result"]
                np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=1e-12)
                maximum=max(maximum,float(np.max(np.abs(actual-expected))))
            row.update(status="captured",module_probes=16,paired_probes=16,reference_error=maximum,
                reference_sha256=sha(destination/"reference-report.json"),model_sha256=digest(captured))
        rows.append(row);save()
    require(frozen(proposal)==bound and source_identity()==sources and runtime_identity()==runtime,
            "Final frozen identities")
    require(all(sha(p)==h for p,h in inventory.items()) and sha(__file__)==own,"Owned fixtures/tools changed")
    report=save()
    print(json.dumps({k:v for k,v in report.items() if k not in ("rows","capture_identity","fixture_hashes")},ensure_ascii=False))
    return 0 if len(rows)==39 and not halt and not report["unclassified"] else 1
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("fixtures","output","proposal"):p.add_argument("--"+name,type=Path,required=True)
    args=p.parse_args();raise SystemExit(run(args.fixtures,args.output,args.proposal))
