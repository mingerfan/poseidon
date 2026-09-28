"""Audit retained Agent answer provenance and distinguish generations from HTTP attempts.

This checks internal retained evidence, not a provider-signed proof of inference
or a billing receipt. It never reads credentials or makes requests.
"""
import hashlib
from pathlib import Path

def ensure(condition, message):
    if not condition:
        raise ValueError(message)

def provider_accounting(report, paid):
    ensure(report.get("provider") == "deepseek_api", "Expected live provider identity")
    m = report.get("provider_metrics", {})
    calls = m.get("calls")
    ensure(type(calls) is list, "Missing HTTP attempt ledger")
    generations = m.get("generation_attempts")
    attempts = m.get("request_attempts")
    max_generations = paid["max_repairs"]+1
    max_http = max_generations*(paid["provider_retries"]+1)
    ensure(type(generations) is int and 0 <= generations <= max_generations,
           "Generation budget/accounting")
    ensure(type(attempts) is int and 0 <= attempts <= max_http
           and attempts == len(calls) == m.get("agent_calls") == report.get("agent_calls"),
           "HTTP attempt budget/accounting")
    ensure(m.get("transport_retries") == attempts-generations, "Transport retry accounting")
    for key, expected in {
        "provider":"deepseek_api", "service_provider":paid["provider"], "model":paid["model"],
        "wire_model":paid["model"], "reasoning_effort":paid["reasoning_effort"],
        "max_tokens":paid["max_tokens"], "max_calls":max_generations,
        "timeout_seconds":paid["api_timeout_seconds"], "provider_retries":paid["provider_retries"],
        "loopback_proxy_port":0,
    }.items():
        ensure(m.get(key) == expected, "Approved provider configuration mismatch: "+key)
    groups = {}
    ensure([c.get("generation_index") for c in calls] == sorted(c.get("generation_index") for c in calls),
           "Interleaved generation ledger")
    for i, call in enumerate(calls):
        ensure(call.get("index") == i, "Nonsequential HTTP ledger")
        g = call.get("generation_index")
        ensure(type(g) is int and 0 <= g < generations, "Generation index")
        group = groups.setdefault(g, [])
        ensure(call.get("retry_index") == len(group) <= paid["provider_retries"],
               "Retry index/budget")
        ensure(call.get("status") in ("failed","response_received","in_flight"), "Unknown transport state")
        if group:
            ensure(group[-1]["status"] == "failed" and group[-1].get("retry_scheduled") is True,
                   "Retry without recorded retryable failure")
        if not group:
            ensure(g == len(groups)-1, "Generation order")
            if g:
                ensure(groups[g-1][-1]["status"] == "response_received",
                       "New generation after terminal transport failure")
        group.append(call)
    ensure(len(groups) == generations, "Missing generation ledger")
    return dict(generations=generations, http_attempts=attempts,
                received=[group[-1] for group in groups.values() if group[-1]["status"] == "response_received"])

def verify_answers(folder, report, request, paid):
    from candidate_contract import strict_json
    from unified_graph_contract import validate_candidate
    accounting = provider_accounting(report, paid)
    ensure(report.get("status") == "passed" and report.get("llm_generation_validated") is True
           and report.get("artifact_replay_only") is False, "Not a successful live candidate")
    attempts = report.get("attempts", [])
    received = accounting["received"]
    ensure(len(attempts) == len(received) == accounting["generations"] > 0,
           "Responses do not match evaluated attempts")
    loop = report.get("loop", {})
    ensure(loop.get("status") == "passed" and loop.get("provider") == "deepseek_api"
           and loop.get("agent_calls") == accounting["http_attempts"]
           and loop.get("attempts") == len(attempts)
           and loop.get("provider_metrics") == report["provider_metrics"],
           "Feedback loop/provider ledger mismatch")
    ensure(len(loop.get("feedback_history", [])) == len(attempts), "Missing feedback history")
    records = []
    for i, (attempt, call) in enumerate(zip(attempts, received)):
        ensure(attempt.get("index") == i and call["generation_index"] == i, "Attempt identity")
        directory = folder / ("attempt-%02d" % i)
        raw_path = directory/"response.txt"
        ensure(not raw_path.is_symlink() and raw_path.is_file()
               and raw_path.stat().st_size <= 131072, "Missing or unsafe retained answer")
        raw = raw_path.read_bytes().decode("utf-8")
        ensure(call.get("finish_reason") == "stop" and call.get("response_model_matches") is True
               and call.get("model") == paid["model"]
               and call.get("content_characters") == len(raw), "Response diagnostics mismatch")
        feedback_path = directory/"feedback.json"
        feedback = strict_json(feedback_path.read_text())
        ensure(feedback == loop["feedback_history"][i], "Feedback record mismatch")
        if attempt.get("status") == "passed":
            candidate = strict_json(raw)
            validate_candidate(candidate, request)
            ensure(candidate["request_id"] == request["request_id"]
                   and (directory/"candidate.py").read_bytes().decode("utf-8") == candidate["hecate_source"],
                   "Executed candidate is not the retained model answer")
        records.append(dict(index=i, response_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                            feedback_sha256=hashlib.sha256(feedback_path.read_bytes()).hexdigest(),
                            http_attempt_index=call["index"], status=attempt.get("status")))
    successful = [r for r in records if r["status"] == "passed"]
    ensure(len(successful) == 1 and successful[0]["index"] == len(records)-1,
           "Invalid successful attempt sequence")
    ensure(loop.get("first_attempt_passed") is (records[0]["status"] == "passed")
           and loop.get("repairs_used") == len(records)-1, "First-pass/repair accounting")
    return dict(generations=accounting["generations"], http_attempts=accounting["http_attempts"],
                first_attempt_passed=records[0]["status"] == "passed", repairs_used=len(records)-1,
                answer_records=records, evidence_scope="retained live transport ledger and exact returned candidate; not provider-signed or billing proof")
