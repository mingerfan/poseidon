"""Shared admission for the existing CLI and its original feedback loop."""
from contextlib import nullcontext
import copy
import hashlib
import math
from pathlib import Path

from benchmark_graph import digest, require
from component_control import QualificationCancelled
from model_weight_package import regular_bytes
from shared_execution import coordinated_provider
from shared_worker import scope


def parameters(args):
    return {name: str(value.absolute()) if isinstance(value, Path) else copy.deepcopy(value)
            for name, value in vars(args).items()}


def authority(args, expected_request):
    from application_identity import source_closure
    from candidate_contract import MAX_BYTES
    from public_weight_context import TRUSTED_BYTES
    files = {}
    for name in ("case", "replay", "golden_file"):
        path = getattr(args, name, None)
        if path is not None:
            limit = (65536 if name == "golden_file" else
                     TRUSTED_BYTES if name == "case" and getattr(args, "public_weights", None) else MAX_BYTES)
            raw = regular_bytes(path, limit)
            files[name] = dict(path=str(path.absolute()), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    return dict(options=parameters(args), inputs=files, expected_request_sha256=digest(expected_request),
                sources=source_closure(Path(__file__).resolve().parent, ("candidate_shared.py", "run_candidate.py")))


class Execution:
    def __init__(self, session):
        self.session = session

    def observe(self, path):
        self.session.observe(path)

    def evaluator(self, original):
        def evaluate(raw, index):
            self.session.verify()
            with self.session.batch.stage("qualify"):
                value = original(raw, index)
            self.session.verify()
            return value
        return evaluate

    def provider(self, provider):
        return (coordinated_provider(self.session.batch, provider, check=self.session.verify)
                if getattr(provider, "kind", None) == "deepseek_api" else nullcontext())


def run(args, worker, *, expected_request=None, report_sink=None, validation_level="numerical", api_key=""):
    """Keep transport, retries, feedback and native validation with the caller."""
    from numerical_results import dump
    from run_candidate import print_outcome
    binding = dict(options=parameters(args), expected_request=copy.deepcopy(expected_request),
                   validation_level=validation_level)
    calls = 0 if args.prepare else args.max_repairs + 1
    stages = dict(qualify=calls, api=calls if args.deepseek else 0)
    seconds = 900
    if args.deepseek:
        from deepseek_provider import generation_deadline
        seconds = math.ceil(1800 + generation_deadline(args.api_timeout, calls, args.provider_retries))
    terminal = {}
    shared = None
    code = 1
    failed = False

    def capture(report, result):
        terminal.update(report=report, result=Path(result))

    try:
        with scope("candidate-cli", binding, stages=stages, wall_seconds=seconds,
                   verify_extra=lambda: authority(args, expected_request)) as shared:
            code = worker(args, expected_request=expected_request, report_sink=capture,
                          validation_level=validation_level, shared=Execution(shared), api_key=api_key)
            api_key = ""
            shared.status = ("succeeded" if code == 0 else "cancelled" if code == 130 else "failed")
            require(digest(binding) == digest(dict(options=parameters(args), expected_request=expected_request,
                    validation_level=validation_level)), "CLI input binding changed")
    except (KeyboardInterrupt, QualificationCancelled):
        failed, code = True, 130
    except Exception:
        # Provider and worker exceptions must never print credential-bearing text.
        failed, code = True, 1
    finally:
        api_key = ""
    if terminal:
        report, result = terminal["report"], terminal["result"]
        if failed:
            report.update(status="cancelled" if code == 130 else "infrastructure_failed",
                          failure_layer="shared_execution", llm_generation_validated=False,
                          diagnostic="Shared execution did not complete; inspect the private control receipt")
        if shared is not None:
            report["shared_control"] = dict(format="poseidon-shared-worker-v1", scope=str(shared.output),
                batch_name=shared.batch.name, plan_binding=shared.batch.binding,
                globally_verified=not failed)
        dump(result / "report.json", report)
        print_outcome(report)
        if report_sink is not None:
            report_sink(report, result)
    elif failed:
        print("Candidate shared admission failed before execution", flush=True)
    return code
