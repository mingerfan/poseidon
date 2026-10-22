"""Verify application generation evidence against the exact qualified source.

Retained transport records establish internal provenance, not a provider-signed
inference proof or billing receipt. The qualification replay keeps its own identity.
"""
import copy
import hashlib
import re
from benchmark_graph import canonical, digest, require
from candidate_contract import strict_json
from generation_protocol import Config, usage_counts
from public_feedback_contract import public_feedback
from agent_response_provenance import provider_accounting

FORMAT = "poseidon-application-generation-v1"
FILE = "artifacts/generation-provenance.json"
MAX_BYTES = 1024 * 1024
METRIC_KEYS = {"provider", "service_provider", "model", "wire_model", "reasoning_effort",
               "max_tokens", "max_calls", "timeout_seconds", "provider_retries",
               "loopback_proxy_port", "generation_attempts", "transport_retries",
               "request_attempts", "agent_calls", "calls"}
CALL_KEYS = {"index", "status", "generation_index", "retry_index", "error",
             "retry_scheduled", "model", "finish_reason", "response_model_matches",
             "content_characters", "usage"}
CONFIG_KEYS = {"provider", "model", "reasoning_effort", "max_tokens", "max_repairs",
               "api_timeout_seconds", "provider_retries", "loopback_proxy_port", "stream"}

def _sha(raw):
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def checked_usage(value, config):
    require(type(value) is dict and set(value) <= {
        "prompt_tokens", "completion_tokens", "total_tokens", "prompt_cache_hit_tokens",
        "prompt_cache_miss_tokens", "reasoning_tokens"}, "Generation usage fields")
    raw = copy.deepcopy(value)
    if "reasoning_tokens" in raw:
        raw["completion_tokens_details"] = dict(reasoning_tokens=raw.pop("reasoning_tokens"))
    checked = usage_counts(dict(usage=raw), config)
    require(checked == value, "Generation usage changed")
    return checked

def safe_metrics(metrics):
    """Persist only fixed counters and provider-reported token counts."""
    require(type(metrics) is dict and METRIC_KEYS <= set(metrics), "Generation metrics fields")
    result = {k:copy.deepcopy(metrics[k]) for k in METRIC_KEYS}
    require(type(result["calls"]) is list and len(result["calls"]) <= 16, "Generation HTTP ledger budget")
    result["calls"] = [{k:copy.deepcopy(c[k]) for k in CALL_KEYS if k in c} for c in result["calls"]]
    config = Config(max_tokens=result["max_tokens"])
    for call in result["calls"]:
        if "usage" in call:
            checked_usage(call["usage"], config)
    return result

def usage_summary(metrics):
    received = [c for c in metrics["calls"] if c["status"] == "response_received"]
    measured = [c["usage"] for c in received if "usage" in c]
    result = dict(responses_with_usage=len(measured), responses_without_usage=len(received)-len(measured),
                  unresolved_http_attempts=len(metrics["calls"])-len(received), cost_usd=None)
    if measured:
        for key in ("prompt_tokens", "completion_tokens", "total_tokens", "prompt_cache_hit_tokens",
                    "prompt_cache_miss_tokens", "reasoning_tokens"):
            if all(key in counts for counts in measured):
                result[key] = sum(counts[key] for counts in measured)
    return result

def verify(value, request, source, *, generation_identity_sha256=None):
    """Check a bounded, data-only receipt without calling a provider or backend."""
    require(type(value) is dict and set(value) == {
        "format", "request_id", "candidate_sha256", "configuration",
        "metrics", "answers", "generation_identity_sha256", "binding"}, "Generation receipt fields")
    require(len(canonical(value)) <= MAX_BYTES and value["format"] == FORMAT, "Generation receipt format/budget")
    require(value["binding"] == digest({k:v for k,v in value.items() if k != "binding"}),
            "Generation receipt binding")
    require(value["request_id"] == request["request_id"], "Generation request identity")
    require(type(source) is str and value["candidate_sha256"] == _sha(source), "Generation source identity")
    stage = value["generation_identity_sha256"]
    require(type(stage) is str and re.fullmatch("[0-9a-f]{64}", stage), "Generation stage identity")
    if generation_identity_sha256 is not None:
        require(stage == generation_identity_sha256, "Generation/package stage mismatch")
    config = value["configuration"]
    require(type(config) is dict and set(config) == CONFIG_KEYS and
            type(config["max_repairs"]) is int and 0 <= config["max_repairs"] <= 3,
            "Generation configuration fields")
    provider_config = Config(model=config["model"], service_provider=config["provider"],
           reasoning_effort=config["reasoning_effort"], max_tokens=config["max_tokens"],
           max_calls=config["max_repairs"]+1, timeout_seconds=config["api_timeout_seconds"],
           provider_retries=config["provider_retries"],
           loopback_proxy_port=config["loopback_proxy_port"], stream=config["stream"])
    metrics = value["metrics"]
    require(type(metrics) is dict and set(metrics) == METRIC_KEYS, "Generation metrics fields")
    calls = metrics["calls"]
    require(type(calls) is list and 1 <= len(calls) <= 16, "Generation HTTP ledger budget")
    for call in calls:
        require(type(call) is dict and set(call) <= CALL_KEYS, "Generation HTTP fields")
        for key in ("index", "generation_index", "retry_index"):
            require(type(call.get(key)) is int, "Generation HTTP indices")
        require(type(call.get("retry_scheduled", False)) is bool, "Generation retry status")
        if "usage" in call:
            checked_usage(call["usage"], provider_config)
    accounting = provider_accounting(
        dict(provider="deepseek_api", provider_metrics=metrics, agent_calls=metrics["agent_calls"]),
        config)
    answers = value["answers"]
    received = accounting["received"]
    require(type(answers) is list and 1 <= len(answers) == accounting["generations"] == len(received) <= 4,
            "Generation answers/accounting")
    for i, (answer, call) in enumerate(zip(answers, received)):
        require(type(answer) is dict and set(answer) == {"index", "raw", "feedback"} and
                type(answer["index"]) is int and answer["index"] == i and
                type(answer["raw"]) is str and len(answer["raw"].encode("utf-8")) <= 131072,
                "Generation answer fields")
        require(call.get("generation_index") == i and call.get("finish_reason") == "stop" and
                call.get("response_model_matches") is True and call.get("model") == config["model"] and
                type(call.get("content_characters")) is int and call["content_characters"] == len(answer["raw"]),
                "Generation answer diagnostics")
        if i < len(answers)-1:
            public_feedback(answer["feedback"])
        else:
            require(answer["feedback"] == dict(status="passed", layer="complete"), "Generation final acceptance")
    candidate = strict_json(answers[-1]["raw"])
    require(type(candidate) is dict and set(candidate) == {"schema", "request_id", "hecate_source"} and
            type(candidate["schema"]) is int and candidate["schema"] == 1 and
            candidate["request_id"] == request["request_id"] and candidate["hecate_source"] == source,
            "Qualified source is not the returned Agent answer")
    return dict(provider="deepseek_api", agent_generated=True,
                generations=accounting["generations"], http_attempts=accounting["http_attempts"],
                first_attempt_passed=len(answers) == 1, repairs_used=len(answers)-1,
                response_sha256=_sha(answers[-1]["raw"]), receipt_binding=value["binding"],
                usage=usage_summary(metrics), evidence_scope="retained transport ledger and exact returned source; not provider-signed or billing proof")

