"""Data-only generation configuration and usage contract; no transport or backend."""
from dataclasses import dataclass
import math

MAX_OUTPUT_TOKENS = 384000

MODEL_OUTPUT_LIMITS = {'deepseek-flash': 384000, 'deepseek-v4-pro': 384000, 'deepseek-v4-flash': 384000}

MAX_TIMEOUT_SECONDS = 1200

class ProviderError(ValueError):
    """Fixed diagnostic codes only, never credential-bearing native error text."""

def check(condition, code):
    if not condition:
        raise ProviderError(code)

@dataclass(frozen=True)
class Config:
    model: str = "deepseek-flash"
    reasoning_effort: str = "high"
    max_tokens: int = 8192
    max_calls: int = 4
    timeout_seconds: float = 120
    service_provider: str = "deepseek"
    stream: bool = False
    provider_retries: int = 0
    loopback_proxy_port: int = 0

    @property
    def wire_model(self):
        return self.model

    @property
    def response_models(self):
        return (self.model, self.wire_model)

    def settings(self):
        values = dict(model=self.wire_model, reasoning_effort=self.reasoning_effort,
                      max_tokens=self.max_tokens, response_format={"type": "json_object"}, stream=self.stream)
        if self.stream:
            values['stream_options'] = {'include_usage': True}
        values["thinking"] = {"type": "disabled" if self.reasoning_effort == "none" else "enabled"}
        return values

    def __post_init__(self):
        check(type(self.loopback_proxy_port) is int and 0 <= self.loopback_proxy_port <= 65535,
              'invalid_loopback_proxy_port')
        check(not self.loopback_proxy_port or self.service_provider == 'deepseek',
              'unsupported_proxy_provider')
        check(type(self.provider_retries) is int and 0 <= self.provider_retries <= 3, 'retry_limit')
        check(type(self.stream) is bool, 'invalid_stream_setting')
        check(self.service_provider == "deepseek", "unsupported_provider")
        check(self.model in MODEL_OUTPUT_LIMITS, "unsupported_model")
        check(type(self.reasoning_effort) is str and self.reasoning_effort in ("none", "low", "high", "max"), "unsupported_effort")
        check(type(self.max_tokens) is int and 1 <= self.max_tokens <= MODEL_OUTPUT_LIMITS[self.model], "output_token_limit")
        check(type(self.max_calls) is int and 1 <= self.max_calls <= 4, "call_limit")
        check(type(self.timeout_seconds) in (int, float) and math.isfinite(self.timeout_seconds)
              and 0 < self.timeout_seconds <= MAX_TIMEOUT_SECONDS, "timeout_limit")

def usage_counts(data, config):
    usage = data.get("usage")
    check(type(usage) is dict, "missing_usage")
    counts = {}
    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(field)
        check(type(value) is int and 0 <= value <= 10**7, "invalid_usage")
        counts[field] = value
    check(counts["total_tokens"] == counts["prompt_tokens"] + counts["completion_tokens"], "inconsistent_usage")
    check(counts["completion_tokens"] <= config.max_tokens, "output_token_limit_exceeded")
    for field in ("prompt_cache_hit_tokens", "prompt_cache_miss_tokens"):
        if field in usage:
            check(type(usage[field]) is int and 0 <= usage[field] <= counts["prompt_tokens"], "invalid_cache_usage")
            counts[field] = usage[field]
    if all(k in counts for k in ("prompt_cache_hit_tokens", "prompt_cache_miss_tokens")):
        check(counts["prompt_cache_hit_tokens"] + counts["prompt_cache_miss_tokens"] == counts["prompt_tokens"],
              "inconsistent_cache_usage")
    details = usage.get("completion_tokens_details", {})
    check(type(details) is dict, "invalid_reasoning_usage")
    if "reasoning_tokens" in details:
        value = details["reasoning_tokens"]
        check(type(value) is int and 0 <= value <= counts["completion_tokens"], "invalid_reasoning_usage")
        counts["reasoning_tokens"] = value
    return counts
