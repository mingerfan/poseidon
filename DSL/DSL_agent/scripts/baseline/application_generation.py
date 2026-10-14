"""Capture provenance through the trusted generation controller.

Data-only verification is shared with generation_receipt; legacy exports remain.
"""
from pathlib import Path
from benchmark_graph import digest, require
from candidate_contract import strict_json
from deepseek_provider import DeepSeekProvider, public_request
from generation_receipt import (FORMAT, FILE, MAX_BYTES, METRIC_KEYS, CALL_KEYS, CONFIG_KEYS,
                                _sha, checked_usage, safe_metrics, usage_summary, verify)

def capture(folder, provider, loop, request, identities, state):
    """Called by the trusted application controller after successful qualification."""
    require(isinstance(provider, DeepSeekProvider) and provider.kind == "deepseek_api",
            "Live provenance requires the configured provider")
    require(loop["status"] == "passed" and loop["provider"] == provider.kind and
            loop["provider_metrics"] == provider.metrics(), "Generation loop/provider mismatch")
    require(provider._request == public_request(request), "Generated public request changed")
    c = provider.config
    config = dict(provider=c.service_provider, model=c.model, reasoning_effort=c.reasoning_effort,
                  max_tokens=c.max_tokens, max_repairs=c.max_calls-1, api_timeout_seconds=c.timeout_seconds,
                  provider_retries=c.provider_retries, loopback_proxy_port=c.loopback_proxy_port, stream=c.stream)
    metrics = safe_metrics(provider.metrics())
    require(all("usage" in call for call in metrics["calls"] if call["status"]=="response_received"),
            "Missing generated response usage")
    require(state["http_attempts"] == metrics["request_attempts"] == loop["agent_calls"] and
            state["generations"] == metrics["generation_attempts"] == loop["attempts"],
            "Generation controller accounting mismatch")
    reserved = [event["index"] for event in state["ledger"] if event["event"] == "http_reserved"]
    require(reserved == list(range(metrics["request_attempts"])), "Generation HTTP reservations")
    answers = []
    for i in range(loop["attempts"]):
        p = Path(folder)/("response-%d.json" % i)
        g = Path(folder)/("generation-%d.json" % i)
        require(all(path.is_file() and not path.is_symlink() and path.stat().st_size <= 512*1024
                    for path in (p, g)), "Generation retained answer files")
        from strict_json_files import strict_file
        answer = strict_file(p,512*1024)
        raw = strict_file(g,512*1024)
        require(answer == dict(raw=raw["raw"], feedback=loop["feedback_history"][i]) and
                raw == dict(raw=provider._responses[i]), "Generation retained/provider answer mismatch")
        answers.append(dict(index=i, **answer))
    require(provider._feedback == loop["feedback_history"][:-1], "Generation repair feedback mismatch")
    source = strict_json(answers[-1]["raw"])["hecate_source"]
    value = dict(format=FORMAT, request_id=request["request_id"], candidate_sha256=_sha(source),
                 configuration=config, metrics=metrics, answers=answers,
                 generation_identity_sha256=identities["generation"]["sha256"])
    value["binding"] = digest(value)
    summary = verify(value, request, source, generation_identity_sha256=identities["generation"]["sha256"])
    return value, summary
