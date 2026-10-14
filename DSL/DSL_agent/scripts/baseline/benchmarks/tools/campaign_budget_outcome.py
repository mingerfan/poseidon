"""Terminal outcome with explicit approved budget, no legacy global configuration."""
from stage2_agent_provenance import provider_accounting

def outcome(report, code, paid):
    if report.get("provider") != "deepseek_api":
        raise ValueError("Scripted result cannot count as Agent evaluation")
    if "provider_metrics" in report:
        provider_accounting(report, paid)
    elif report.get("agent_calls") != 0 or report.get("status") == "passed":
        raise ValueError("Missing HTTP/generation ledger")
    successful = [a for a in report.get("attempts", []) if a.get("status") == "passed"]
    claimed = code == 0 and report.get("status") == "passed"
    if claimed:
        if not report.get("llm_generation_validated") or report["agent_calls"] == 0 or len(successful) != 1:
            raise ValueError("Success without actual Agent provenance")
        a = successful[0]
        if not all(a.get(k) for k in ("compiled", "executed", "numerically_correct")):
            raise ValueError("Success without full execution")
        if a["trace"]["frontend"] != "real_Hecate" or not a["execution"]["encrypted_execution"]:
            raise ValueError("Not real Hecate/SEAL execution")
        if a["comparison"]["atol"] != 1e-5 or a["comparison"]["rtol"] != 1e-4:
            raise ValueError("Frozen numerical tolerance changed")
        # Retained raw response, independent references/artifacts and witnesses
        # still require a post-run audit. Never count this preliminary state as pass.
        return "runner_reported_pass_pending_audit"
    return "failed"
