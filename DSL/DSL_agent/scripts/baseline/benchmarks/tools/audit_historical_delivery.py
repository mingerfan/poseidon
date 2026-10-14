"""Audit a delivery envelope without confusing historical authorship with a new call."""
import hashlib
import json
from pathlib import Path

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def verify_delivery(folder, *, verify_program=True):
    from benchmark_graph import require,digest
    from candidate_contract import strict_json
    def read(path):
        require(path.is_file() and not path.is_symlink() and path.stat().st_size<=8*1024**2,"Delivery file")
        return strict_json(path.read_text())
    envelope=read(folder/"delivery.json")
    bound=dict(envelope);binding=bound.pop("binding")
    require(digest(bound)==binding,"Delivery binding")
    require(envelope["format"]=="poseidon-historical-agent-delivery-v1" and
            envelope["origin_category"]=="historical_agent_source" and
            envelope["new_agent_generation"] is False and envelope["manual_source_repair"] is False,
            "Delivery origin category")
    names={"historical/"+n for n in ("report.json","request.json","model.json","response.txt","candidate.py")}
    require(set(envelope["historical_files"])==names,"Historical retained files")
    for name,h in envelope["historical_files"].items():
        file=folder/name
        require(file.is_file() and not file.is_symlink() and sha(file)==h,"Historical delivery file changed")
    report=read(folder/"historical/report.json");old=read(folder/"historical/request.json")
    answer=read(folder/"historical/response.txt")
    require(report["status"]=="passed" and report["provider"]=="deepseek_api" and
            report["llm_generation_validated"] is True and report["agent_calls"]>0,"Historical live origin")
    from agent_response_provenance import provider_accounting
    m=report["provider_metrics"]
    provider_accounting(report,dict(provider=m["service_provider"],model=m["model"],
        reasoning_effort=m["reasoning_effort"],max_tokens=m["max_tokens"],
        max_repairs=m["max_calls"]-1,provider_retries=m["provider_retries"],
        api_timeout_seconds=m["timeout_seconds"]))
    require(answer["request_id"]==old["request_id"]==report["request_id"]==envelope["historical_request_id"],
            "Historical request identity")
    source=(folder/"historical/candidate.py").read_bytes()
    require(answer["hecate_source"].encode()==source and sha(folder/"historical/candidate.py")==envelope["source_sha256"],
            "Historical answer/source mismatch")
    program=folder/"program";manifest=read(program/"manifest.json");current=read(program/"request.json")
    require(sha(program/"manifest.json")==envelope["program_manifest_sha256"] and
            manifest["binding"]==envelope["package_binding"],"Program binding")
    require((program/"candidate.py").read_bytes()==source,"Program source was changed")
    require(current["request_id"]==envelope["poseidon_request_id"] and current["model"]==old["model"] and
            current["public_constants"]==old["public_constants"] and current["layout"]==old["layout"],
            "Historical semantic binding drift")
    require(manifest["compiler_configuration"]["backend"]=="poseidon_cpu" and
            manifest["dependencies"]["backend_identity"]==envelope["backend_identity"] and
            envelope["validation_level"]=="numerical","Poseidon numerical binding")
    if verify_program:
        from candidate_bundle import check_bundle
        checked=check_bundle(program)
        require(checked["validation"]["numerically_validated"] is True,"Program validation level")
    return dict(origin_category="historical_agent_source",new_agent_generation=False,
                manual_source_repair=False,source_sha256=envelope["source_sha256"],
                historical_request_id=old["request_id"],poseidon_request_id=current["request_id"],
                program=str(program),binding=binding,validation_level="numerical",
                origin_proof_scope="retained response and transport ledger; not provider-signed proof")
