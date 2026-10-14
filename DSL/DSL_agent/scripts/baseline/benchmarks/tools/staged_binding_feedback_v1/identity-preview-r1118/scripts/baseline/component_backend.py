"""Qualification adapter, called only in an isolated pinned Python worker."""
import copy
from pathlib import Path
from component_contract import failure

def observed_stages(report):
    """Host observations; a completed execution can still fail numerical acceptance."""
    last = report.get("attempts", [])[-1] if report.get("attempts") else {}
    intact = last.get("failure_layer") != "integrity" and report.get("failure_layer") != "integrity"
    return dict(traced=isinstance(last.get("trace"), dict),
                compiled=last.get("compiled") is True,
                encrypted_execution=last.get("executed") is True,
                numerically_compared=isinstance(last.get("comparison"), dict),
                evidence_integrity_not_rejected=intact)

def qualify(request, candidate, *, validation_level="numerical"):
    from benchmark_graph import canonical, require
    from hecate_python_env import WORK
    from qualification_session import QualificationSession
    require(validation_level in ("compiled", "numerical"), "Validation level")
    require(request.get("compiler_configuration",{}).get("backend")=="poseidon_cpu",
            "Historical backend execution is retired; prepare a new Poseidon request")
    code, report, evidence = QualificationSession.qualify(
        request, candidate, validation_level=validation_level)
    terminal = dict(report=report, evidence=str(evidence))
    output = dict(format="poseidon-agent-component-v1", status=report["status"], exit_code=code,
        request_id=request["request_id"], evidence=terminal["evidence"], paid_calls=0,
        new_agent_generation=False, encrypted_execution=False, numerically_validated=False)
    stages = observed_stages(report)
    output["observed_stages"] = stages
    # Observation and numerical qualification are separate; integrity rejection
    # keeps the unqualified observations without claiming trusted execution.
    output["encrypted_execution"] = stages["encrypted_execution"] and stages["evidence_integrity_not_rejected"]
    if report["status"] == "passed":
        from audit_unified_candidate import verify_candidate
        audit = verify_candidate(Path(terminal["evidence"]), validation_level=validation_level)
        require(audit["request_id"] == request["request_id"], "Component audit request mismatch")
        output.update(validation_level=validation_level, compiled_validated=True)
        if validation_level == "numerical":
            output.update(encrypted_execution=True, numerically_validated=True,
                          max_absolute_error=audit["comparison"]["max_absolute_error"],
                          compared_values=audit["comparison"]["compared_values"])
    else:
        last = next((v for v in reversed(report["attempts"]) if v.get("failure_layer")), report)
        history = report.get("loop", {}).get("feedback_history", [])
        if history and history[-1].get("status") == "failed":
            from public_feedback_contract import public_feedback
            output["public_feedback"] = public_feedback(history[-1])
        output["failure"] = failure(last.get("failure_layer", "pipeline"), last.get("diagnostic", report["status"]))
        if isinstance(last.get("compiler_diagnosis"),dict):
            output["compiler_diagnosis"]=copy.deepcopy(last["compiler_diagnosis"])
    return output
