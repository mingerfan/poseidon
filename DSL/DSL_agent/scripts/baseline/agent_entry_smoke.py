"""Offline provider fixtures through the REAL Agent compiler/execution entry.

No model service: saved baseline programs test wiring, not LLM generation quality.
Run from the source root; all new execution evidence goes to native results.
"""
import json
import os
from pathlib import Path
import shlex
import sys
from unittest.mock import patch

from hecate_python_env import ROOT, WORK, VENV, digest, enter_nix

SENTINEL = "OFFLINE-ENV-TEST-NOT-A-REAL-API-KEY"
BASELINE = WORK / "results/candidate-replay-7ommkr7q"


def main():
    if "--inside" not in sys.argv:
        # Never discover/read a real credential in this smoke test.
        os.environ["DEEPSEEK_API_KEY"] = SENTINEL
        command = (f'LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" PYTHONDONTWRITEBYTECODE=1 '
                   f'{shlex.quote(str(VENV / "bin/python"))} '
                   f'{shlex.quote(str(Path(__file__).resolve()))} --inside')
        return enter_nix(command, seconds=900, keep_env=("DEEPSEEK_API_KEY",))

    import run_candidate as runner
    from deepseek_provider import DeepSeekProvider
    from deepseek_http_worker import tls_context
    from test_deepseek_provider import FixtureTransport, completion
    if os.environ.get("DEEPSEEK_API_KEY") != SENTINEL:
        raise RuntimeError("Explicit key passthrough failed")
    # Offline TLS trust-store readiness, not an actual service connection.
    roots = len(tls_context().get_ca_certs())
    if roots == 0:
        raise RuntimeError("No trusted CA roots in the installed Python environment")
    prior = json.loads((BASELINE / "report.json").read_text())
    source_path = BASELINE / "rule-answer.py"
    if digest(source_path) != prior["frozen_hashes"]["rule-answer.py"]:
        raise RuntimeError("Saved fixture changed")
    source = source_path.read_text()
    providers = []

    def offline_provider(args, request, responses=None, api_key=""):
        if api_key != SENTINEL or "DEEPSEEK_API_KEY" in os.environ:
            raise RuntimeError("Key must be removed from native-child environment")
        if request["request_id"] != prior["request_id"]:
            raise RuntimeError("Fixture belongs to another immutable model request")
        correct = dict(schema=1, request_id=request["request_id"], hecate_source=source)
        wrong = dict(correct, hecate_source=source.replace("rotate(1)", "rotate(2)"))
        transport = FixtureTransport([completion("{invalid JSON"), completion(json.dumps(wrong)),
                                      completion(json.dumps(correct))])
        provider = DeepSeekProvider(transport=transport)
        providers.append(provider)
        return provider

    saved_reports = []
    real_dump = runner.dump
    def record(path, data):
        real_dump(path, data)
        if path.name == "report.json" and path.parent.name.startswith("agent-deepseek-"):
            saved_reports.append(path)

    args = runner.parse_args(["--case", str(ROOT / "scripts/baseline/cases/linear-example.json"), "--deepseek"])
    with patch.object(runner, "select_provider", offline_provider), patch.object(runner, "dump", record):
        code = runner.inside(args)
    if code != 0 or not saved_reports:
        return 1
    report_path = saved_reports[-1]
    report = json.loads(report_path.read_text())
    if report["agent_calls"] != 0 or report["llm_generation_validated"] or report["provider"] != "deepseek_offline":
        raise RuntimeError("Offline fixtures must not claim live generation")
    if [x.get("failure_layer", "complete") for x in report["attempts"]] != ["response_parse", "numerical_comparison", "complete"]:
        raise RuntimeError("Expected real parse/numerical failure and recovery")
    if not all(x["execution"]["encrypted_execution"] for x in report["attempts"][1:]):
        raise RuntimeError("Real encrypted execution was not reached")
    print(json.dumps(dict(status="passed", report=str(report_path), trusted_ca_count=roots,
                          actual_api_calls=0, offline_fixture_transport_requests=providers[0].request_attempts,
                          encrypted_runs=sum(x.get("executed", False) for x in report["attempts"])), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
