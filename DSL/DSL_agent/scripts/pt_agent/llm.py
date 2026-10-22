"""Bounded DeepSeek worker calls with content-free accounting and safe errors."""

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


REPO = Path(__file__).resolve().parents[4]
BASELINE = REPO / "DSL/DSL_agent/scripts/baseline"
if str(BASELINE) not in sys.path:
    sys.path.insert(0, str(BASELINE))
from agent_credentials import load_api_key
from deepseek_http_worker import safe_diagnostics
from generation_protocol import Config


_WORKER_ERRORS = frozenset(("transport_tls_failed", "transport_socket_timeout", "transport_dns_failed",
                            "transport_connection_failed", "transport_remote_disconnected", "transport_incomplete_read"))
_ERRORS = _WORKER_ERRORS | frozenset(("budget_exhausted", "settings_error", "credentials_error", "transport_timeout",
                                     "transport_unavailable", "transport_worker_failed", "invalid_worker_response",
                                     "http_error", "invalid_response", "llm_error"))


class LLMError(Exception):
    def __init__(self, code, diagnostics=None):
        self.code = code if code in _ERRORS else "llm_error"
        self.diagnostics = safe_diagnostics(diagnostics)
        super().__init__(self.code)


@dataclass
class LLMClient:
    budget_path: Path
    model: str = "deepseek-flash"
    reasoning_effort: str = "none"
    max_tokens: int = 16384
    timeout: float = 600
    max_total_calls: int = 300

    def chat(self, messages: list[dict], *, purpose: str) -> dict:
        path = Path(self.budget_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a+", encoding="utf-8") as budget:
            fcntl.flock(budget.fileno(), fcntl.LOCK_EX)
            budget.seek(0)
            if sum(1 for _ in budget) >= self.max_total_calls:
                raise LLMError("budget_exhausted")
            started = time.monotonic()
            record = dict(ts=datetime.now(timezone.utc).isoformat(), purpose=purpose, model=self.model,
                          status=None, seconds=None, prompt_tokens=None, completion_tokens=None,
                          finish_reason=None, error=None)
            error = None
            diagnostics = {}
            result = None
            try:
                try:
                    body = Config(model=self.model, reasoning_effort=self.reasoning_effort,
                                  max_tokens=self.max_tokens, timeout_seconds=self.timeout).settings()
                    body["messages"] = messages
                except Exception:
                    raise LLMError("settings_error") from None
                try:
                    key = load_api_key(REPO)
                except Exception:
                    raise LLMError("credentials_error") from None
                wire = json.dumps(dict(api_key=key, body=body, timeout=self.timeout,
                                       provider="deepseek", loopback_proxy_port=0), separators=(",", ":")).encode()
                worker = BASELINE / "deepseek_http_worker.py"
                env = {name: os.environ[name] for name in ("SYSTEMROOT", "WINDIR", "TEMP", "TMP", "LD_LIBRARY_PATH")
                       if name in os.environ}
                try:
                    completed = subprocess.run([sys.executable, "-I", "-B", str(worker)], input=wire,
                                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
                                               timeout=self.timeout, check=False)
                except subprocess.TimeoutExpired:
                    raise LLMError("transport_timeout") from None
                except OSError:
                    raise LLMError("transport_unavailable") from None
                if len(completed.stderr) <= 8192:
                    for line in completed.stderr.splitlines():
                        if line.startswith(b"DIAG "):
                            try:
                                diagnostics = safe_diagnostics(json.loads(line[5:]))
                            except (ValueError, TypeError):
                                pass
                if completed.returncode != 0:
                    marker = completed.stderr.splitlines()[0] if completed.stderr.splitlines() else b""
                    code = next((item for item in _WORKER_ERRORS if marker == item.encode()), "transport_worker_failed")
                    raise LLMError(code, diagnostics)
                if len(completed.stdout) > 32*1024**2+16:
                    raise LLMError("invalid_worker_response", diagnostics)
                try:
                    status, raw = completed.stdout.split(b"\n", 1)
                    if not re.fullmatch(rb"[1-5][0-9]{2}", status):
                        raise ValueError
                    record["status"] = int(status)
                except (ValueError, TypeError):
                    raise LLMError("invalid_worker_response", diagnostics) from None
                if record["status"] != 200:
                    raise LLMError("http_error", {**diagnostics, "status": record["status"]})
                try:
                    response = json.loads(raw)
                    choice = response["choices"][0]
                    content = choice["message"]["content"]
                    if not isinstance(content, str):
                        raise ValueError
                    usage = {name: value for name, value in response.get("usage", {}).items()
                             if type(value) is int and value >= 0}
                    finish = choice.get("finish_reason")
                    finish = finish if finish in ("stop", "length", "content_filter", "tool_calls", "insufficient_system_resource") else None
                    record.update(prompt_tokens=usage.get("prompt_tokens"), completion_tokens=usage.get("completion_tokens"),
                                  finish_reason=finish)
                    result = dict(content=content, finish_reason=finish, usage=usage, id=response.get("id"), seconds=None)
                except (ValueError, KeyError, IndexError, TypeError, AttributeError):
                    raise LLMError("invalid_response", diagnostics) from None
            except LLMError as exc:
                error = exc
                record["error"] = exc.code
            finally:
                record["seconds"] = time.monotonic()-started
                budget.seek(0, 2)
                budget.write(json.dumps(record, allow_nan=False)+"\n")
                budget.flush()
            if error is not None:
                raise error
            result["seconds"] = record["seconds"]
            return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", required=True)
    args = parser.parse_args()
    root = Path(os.environ["PT_RESULTS"])
    if not root.is_dir():
        raise RuntimeError("PT_RESULTS must exist")
    out = root / ("r7-llm-smoke-"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S"))
    out.mkdir()
    report = dict(round="R7", complete=False, status=None, error=None)
    try:
        reply = LLMClient(root / "llm_calls.jsonl").chat([
            dict(role="system", content='Reply with JSON {"ok": true}'), dict(role="user", content="ping")], purpose="llm-smoke")
        parsed = json.loads(reply["content"])
        if not isinstance(parsed, dict) or parsed.get("ok") is not True:
            raise ValueError
        report.update(complete=True, status=200, ok=True, usage=reply["usage"], seconds=reply["seconds"],
                      finish_reason=reply["finish_reason"])
    except LLMError as exc:
        report.update(error=exc.code, diagnostics=exc.diagnostics)
    except (ValueError, TypeError):
        report["error"] = "smoke_not_json"
    (out / "REPORT.json").write_text(json.dumps(report, indent=2)+"\n")
    from .registry import append, git_rev, new_run_id

    append(dict(run_id=new_run_id("R7", "none", "llm-smoke"), round="R7", model="none", mode="llm-smoke",
                git_rev=git_rev(), metrics=report, artifacts_dir=str(out), ts=datetime.now(timezone.utc).isoformat()))
    print(json.dumps(dict(report_path=str(out / "REPORT.json"), **report)), flush=True)
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
