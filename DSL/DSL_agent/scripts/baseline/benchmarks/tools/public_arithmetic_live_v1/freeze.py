"""Freeze public requests from completed qualification; no source generation or paid calls."""
import json,sys,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline"
sys.path.insert(0,str(BASE))
from hecate_python_env import VENV,enter_nix
def main():
    if "--inside" not in sys.argv:
        import shlex
        command='PYTHONPATH='+shlex.quote(str(BASE))+' LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
        return enter_nix(command,seconds=90,keep_env=(),environment="poseidon-cpu")
    from scope import CASES,PROVIDER,BOUNDS,OPTIONS,PROFILE,PROPOSAL,digest,read,sha
    from deepseek_provider import public_request
    from pytorch_ordered_contract import source_identity
    from application_backend import PoseidonBackend
    from candidate_bundle import check_bundle
    from unified_request_storage import restore_public_constants
    import copy
    qualification=ROOT/"docs/baseline/agent-public-arithmetic-offline-results-r1386.json"
    accepted=read(qualification)
    assert accepted["passed"]==18 and accepted["failed"]==accepted["skipped"]==accepted["not_run"]==0
    rows={r["case"]:r for r in accepted["completed"]}
    assert not PROPOSAL.exists()
    prepared=HERE/"requests";assert not prepared.exists();prepared.mkdir()
    first=rows[CASES[0]];stage=read(Path(first["package"]).parent/"task.json")["identity"]
    backend=PoseidonBackend()
    assert backend.identity_manifest()==stage and backend.identity()["tools"]==stage["backend"]
    assert accepted["sdk_identity"]==stage["manifest_sha256"]
    selected=read(ROOT/"scripts/baseline/benchmarks/tools/public_arithmetic_expression_v1/selected-offline-cases-r1385.json")
    specs={c["case"]:c for c in selected}
    cases=[]
    for name in CASES:
        row=rows[name];package=Path(row["package"]);meta=read(package.parent/"task.json");q=meta["request"]
        checked=check_bundle(package)
        assert checked["binding"]==row["package_binding"] and meta["identity"]==stage
        assert all(row[k] for k in ("source_bound","real_tracing","compiled","artifact_checked",
                                    "encrypted_execution","independent_reference_verified","package_reloaded"))
        stored=copy.deepcopy(read(package/"request.json"));restore_public_constants(stored)
        assert stored==q
        manifest=Path(specs[name]["source_manifest"]);m=read(manifest)
        assert m["format"]=="poseidon-trusted-pytorch-v25"
        full=prepared/(name+".request.json");public=prepared/(name+".public.json")
        full.write_text(json.dumps(q,sort_keys=True,indent=2,allow_nan=False)+"\n")
        public.write_text(json.dumps(public_request(q),sort_keys=True,indent=2,allow_nan=False)+"\n")
        case=dict(id=name,source_manifest=str(manifest.relative_to(ROOT)),source_manifest_sha256=sha(manifest),
            source_format=m["format"],input_signature=m["inputs"],output_names=m["outputs"],dtype_policy=m["dtype_policy"],
            resource_profile=PROFILE,generation_guidance="explicit-v13",slot_period=4,validation_level="numerical",
            track="free_generation",max_repairs=3,max_generations=4,max_http_attempts=16,
            approximation_accuracy_certified=False,
            source_files={str(manifest.parent/f):h for f,h in m["files"].items()},
            declared_weight_files={str(manifest.parent/v["file"]):v["sha256"] for v in m["weights"].values()},
            request_id=q["request_id"],model_sha256=digest(q["model"]),request_file=str(full),request_file_sha256=sha(full),
            public_request_file=str(public),public_request_sha256=sha(public),
            qualification=dict(candidate_origin="deterministic_rule",package=str(package),
                package_binding=row["package_binding"],producer_stage_manifest=stage["manifest_sha256"],
                not_current_agent_generation=True))
        cases.append(case)
    proposal=dict(format="poseidon-public-arithmetic-live-proposal-v1",status="awaiting_user_approval",
        source_root=str(ROOT),head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True,timeout=10).strip(),
        branch="lhy-agent-dsl",provider=PROVIDER,bounds=BOUNDS,numeric_tolerance=dict(atol=1e-5,rtol=1e-4),
        monetary_cap=None,retries_may_duplicate_billing=True,source_stage_identity=stage,
        source_stage_counts={s:len(stage[s]["sources"]) for s in ("generation","compilation","execution","validation")},
        source_capture_implementation=source_identity(),cases=cases,qualification_records={str(qualification):sha(qualification)},
        paid_calls=0,http_attempts=0,old_paid_campaigns_closed=True,no_retry_of_historical_unknown_calls=True,
        external_content=["public model/decomposed graph","public weights or descriptors","layout","DSL rules","bounded diagnostics"],
        excluded_content=["rule candidate/answer","private test inputs","reference outputs","credentials","private keys"],
        protocol_evidence=dict(local_config="generation_protocol.py",
            checked_date="2026-10-11",
            model_and_effort="https://api-docs.deepseek.com/api/list-models/",
            chat_completion="https://api-docs.deepseek.com/api/create-chat-completion/",
            documented_max_tokens=393216,requested_max_tokens=384000,
            verification="official_docs_and_local_config; no credentialed_probe_or_generation"),
        network_changes=False,dependency_installation=False,backend_changes=False,parameter_changes=False)
    proposal["binding"]=digest(proposal)
    PROPOSAL.write_text(json.dumps(proposal,sort_keys=True,indent=2,allow_nan=False)+"\n")
    text=(HERE/"scope.py").read_text().replace("FROZEN_BINDING",proposal["binding"]).replace("FROZEN_SHA",sha(PROPOSAL))
    (HERE/"scope.py").write_text(text)
    from scope import MANIFEST,tools
    manifest=dict(format="poseidon-public-arithmetic-live-runner-v1",proposal_binding=proposal["binding"],
        proposal_sha256=sha(PROPOSAL),tool_hashes=tools())
    manifest["binding"]=digest(manifest)
    MANIFEST.write_text(json.dumps(manifest,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(json.dumps(dict(proposal=str(PROPOSAL),binding=proposal["binding"],cases=len(cases),paid_calls=0,compiler_calls=0)))
    return 0
if __name__=="__main__":sys.exit(main())
