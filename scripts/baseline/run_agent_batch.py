"""Full catalog live Agent evaluation, with independent case processes and evidence.

Bounded opt-in transport retries; no fixture reuse or rule-answer fallback.
Run --deepseek explicitly; creates a fresh batch, never silently resumes one.
"""
import argparse
import hashlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import threading
from compiler_configuration import (CONFIGURATIONS, configuration, options as compiler_options,
                                    validate_continuation as validate_compiler_continuation)


def case_metrics(report):
    attempts = report.get("attempts", [])
    first = attempts[0] if attempts else {}
    calls = report.get("provider_metrics", {}).get("calls", [])
    passed = (report.get("status") == "passed" and report.get("llm_generation_validated") is True
              and report.get("provider") == "deepseek_api"
              and any(a.get("numerically_correct") and a.get("execution", {}).get("encrypted_execution")
                      for a in attempts))
    failed = [a for a in attempts if a.get("status") != "passed"]
    code = next((c.get("error") for c in reversed(calls) if c.get("error")), None)
    layer = report.get("failure_layer")
    category = None
    if not passed:
        if report.get("status") == "provider_failed":
            layer = "provider"
            category = ("provider_output" if code in ("incomplete_or_refused_response", "empty_or_oversized_content",
                        "invalid_http_json", "response_model_mismatch") else "provider_infrastructure")
        elif failed:
            layer = failed[-1].get("failure_layer", "unknown")
            category = failed[-1].get("category", "pipeline_unclassified")
        else:
            layer = layer or "batch_worker"
            category = "infrastructure"
    metrics = report.get('provider_metrics', {})
    # Preserve legacy report comparison byte-for-byte at the field level.
    extras = {} if 'generation_attempts' not in metrics else dict(
        generation_attempts=metrics['generation_attempts'], transport_retries=metrics.get('transport_retries', 0))
    return dict(**extras, passed=bool(passed), first_passed=bool(passed and first.get("numerically_correct")),
                first={k: first.get(k) is True for k in ("parsed", "source_parsed", "checked", "compiled", "executed", "numerically_correct")},
                ever={k: any(a.get(k) is True for a in attempts) for k in ("parsed", "compiled", "executed", "numerically_correct")},
                received_candidates=len(attempts), api_calls=report.get("agent_calls", 0),
                repairs_attempted=max(0, metrics.get('generation_attempts', report.get("agent_calls", 0))-1),
                failure_layer=layer, failure_category=category, provider_error=code,
                attempt_failures=[dict(layer=a.get("failure_layer"), category=a.get("category")) for a in failed],
                usage={k: sum(c.get("usage", {}).get(k, 0) for c in calls)
                       for k in ("prompt_tokens", "completion_tokens", "total_tokens")},
                calls_without_usage=sum("usage" not in c for c in calls),
                comparison=next((a["comparison"] for a in reversed(attempts) if passed and a.get("numerically_correct")), None),
                encrypted_input_executions=sum(a.get("execution", {}).get("input_batches", 0) for a in attempts),
                backend=report.get("backend"), poseidon_gpu_validated=report.get("poseidon_gpu_validated", False))


def row_family(row):
    """Benchmark labels never enter user graph descriptors or choose lowering."""
    return row.get("benchmark_family", row["descriptor"].get("family", "custom_graph"))


def batch_plan(extended=False):
    if extended:
        from expanded_model_suite import manifest
        benchmark = manifest()
        return [dict(descriptor=e["descriptor"], benchmark_family=e["family"], status="pending")
                for e in benchmark["entries"]], benchmark
    from expanded_model_suite import entries
    return [dict(descriptor=e["descriptor"], status="pending") for e in entries()[:48]], None


def summarize(rows):
    total = len(rows)
    done = [r for r in rows if "metrics" in r]
    def rate(n, d=total):
        return dict(numerator=n, denominator=d, rate=n/d if d else None)
    families = sorted({row_family(r) for r in rows})
    correct = [r for r in done if r["metrics"]["passed"]]
    generated = sum(r["metrics"]["received_candidates"] > 0 for r in done)
    return dict(planned=total, completed=len(done), not_completed=total-len(done),
        first={k: rate(sum(r["metrics"]["first"][k] for r in done))
               for k in ("parsed", "source_parsed", "checked", "compiled", "executed", "numerically_correct")},
        first_success=rate(sum(r["metrics"]["first_passed"] for r in done)),
        final_success=rate(len(correct)),
        final_success_given_candidate=rate(len(correct), generated),
        mean_repairs_all_completed=(sum(r["metrics"]["repairs_attempted"] for r in done)/len(done) if done else None),
        mean_repairs_successful=(sum(r["metrics"]["repairs_attempted"] for r in correct)/len(correct) if correct else None),
        api_calls=sum(r["metrics"]["api_calls"] for r in done),
        transport_retries=sum(r['metrics'].get('transport_retries', 0) for r in done),
        usage={k: sum(r["metrics"]["usage"][k] for r in done) for k in ("prompt_tokens", "completion_tokens", "total_tokens")},
        calls_without_usage=sum(r["metrics"]["calls_without_usage"] for r in done),
        encrypted_input_executions=sum(r["metrics"]["encrypted_input_executions"] for r in done),
        compared_values_successful=sum(r["metrics"]["comparison"]["compared_values"] for r in correct),
        max_absolute_error_successful=max((r["metrics"]["comparison"]["max_absolute_error"] for r in correct), default=None),
        final_failure_layers=dict(Counter(r["metrics"]["failure_layer"] for r in done if not r["metrics"]["passed"])),
        final_failure_categories=dict(Counter(r["metrics"]["failure_category"] for r in done if not r["metrics"]["passed"])),
        provider_errors=dict(Counter(r["metrics"]["provider_error"] for r in done if r["metrics"]["provider_error"])),
        attempt_failure_layers=dict(Counter(a["layer"] for r in done for a in r["metrics"]["attempt_failures"])),
        families={f: dict(planned=sum(row_family(r) == f for r in rows),
            completed=sum(row_family(r) == f for r in done),
            first_success=sum(row_family(r) == f and r["metrics"]["first_passed"] for r in done),
            final_success=sum(row_family(r) == f for r in correct),
            holdout=f in ("fanout", "residual")) for f in families})


def save(path, data):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def select_failed(catalog, prior):
    if prior.get("status") not in ("passed", "completed_with_failures"):
        raise ValueError("Source batch must be complete")
    rows = prior.get("cases", [])
    if [r.get("descriptor") for r in rows] != catalog:
        raise ValueError("Source batch must contain the exact full catalog in order")
    if any(r.get("status") not in ("passed", "failed") or type(r.get("metrics", {}).get("passed")) is not bool
           or (r["status"] == "passed") != r["metrics"]["passed"] for r in rows):
        raise ValueError("Inconsistent source case outcomes")
    chosen = [r["descriptor"] for r in rows if not r["metrics"]["passed"]]
    if not chosen:
        raise ValueError("No failed cases to retest")
    return chosen


def load_failed_source(catalog, source, results, seen=None):
    """Validate a bounded, hash-linked lineage before selecting a failure subset."""
    source, results = Path(source).resolve(), Path(results).resolve()
    seen = set() if seen is None else set(seen)
    if (source in seen or len(seen) >= 16 or not source.is_relative_to(results)
            or source.stat().st_size > 4 * 1024**2):
        raise ValueError("Invalid source report path, size or lineage")
    seen.add(source)
    prior = json.loads(source.read_text())
    selection = prior.get("selection")
    if selection:
        parent = Path(selection["source_report"]).resolve()
        if not parent.is_relative_to(results):
            raise ValueError("Source lineage outside results root")
        if selection.get('kind') == 'previous_failures_and_unfinished':
            from interrupted_batch import load_remaining_source
            ancestor, catalog = load_remaining_source(catalog, parent, results)
        else:
            ancestor, catalog = load_failed_source(catalog, parent, results, seen)
        if hashlib.sha256(parent.read_bytes()).hexdigest() != selection["source_sha256"]:
            raise ValueError("Source lineage hash mismatch")
    return prior, select_failed(catalog, prior)


def generation_settings(report):
    return dict(model=report["model"], service_provider=report.get("service_provider", "deepseek"), reasoning_effort=report.get("reasoning_effort", "high"),
                max_tokens=report["max_tokens"], max_calls=report["max_repairs"] + 1,
                timeout_seconds=report["api_timeout"], stream=report.get('stream', False),
                provider_retries=report.get('provider_retries', 0),
                loopback_proxy_port=report.get('loopback_proxy_port', 0))


def configuration_delta(before, after):
    return {k: dict(before=before[k], after=after[k]) for k in after if before[k] != after[k]}


def should_pause_batch(provider, metrics):
    """A model identity change is not repaired by sending more queued models."""
    return metrics.get('failure_layer') == 'provider' and (
        metrics.get('provider_error') == 'response_model_mismatch')


def construction_options(args):
    if getattr(args,'native_array_mutation',False):
        return ['--native-array-mutation']
    if getattr(args,'native_scalar_augmented',False):
        return ['--native-scalar-augmented']
    if getattr(args,'native_public_loops',False):
        return ['--native-public-loops']
    if getattr(args,'native_array_arithmetic',False):
        return ['--native-array-arithmetic']
    if getattr(args,'native_starred',False):
        return ['--native-starred']
    if getattr(args,'native_arrays',False):
        return ['--native-arrays']
    if getattr(args,'native_functions',False) or getattr(args,'native_function_exercises',False):
        return ['--native-functions']
    if getattr(args,"object_unary",False) or getattr(args,"object_unary_exercises",False):
        return ["--object-unary"]
    if getattr(args,"scalar_conversion",False) or getattr(args,"scalar_conversion_exercises",False):
        return ["--scalar-conversion"]
    if getattr(args,"object_arithmetic",False) or getattr(args,"object_arithmetic_exercises",False):
        return ["--object-arithmetic"]
    if getattr(args, "construction_exercises", False):
        return ["--public-mappings"]
    if getattr(args,'object_arrays',False):
        return ['--object-arrays']
    if getattr(args,'public_polynomial',False):
        return ['--public-polynomial']
    if getattr(args,'public_strings',False):
        return ['--public-strings']
    if getattr(args,'public_control',False):
        return ['--public-control']
    return ['--public-numbers'] if getattr(args,'public_numbers',False) else []


def validate_construction_continuation(prior, args):
    validate_compiler_continuation(prior, args)
    if bool(prior.get('native_array_mutation',False)) != bool(getattr(args,'native_array_mutation',False)):
        raise ValueError('Changed native array-mutation contract requires a fresh batch')
    if bool(prior.get('native_scalar_augmented',False)) != bool(getattr(args,'native_scalar_augmented',False)):
        raise ValueError('Changed native scalar-augmented contract requires a fresh batch')
    if prior.get('native_star_exercises'):
        raise ValueError('Frozen native starred requirements cannot be dropped by a generic restart')
    if bool(prior.get('native_public_loops',False)) != bool(getattr(args,'native_public_loops',False)):
        raise ValueError('Changed native public-loop contract requires a fresh batch')
    if bool(prior.get('native_array_arithmetic',False)) != bool(getattr(args,'native_array_arithmetic',False)):
        raise ValueError('Changed native array-arithmetic contract requires a fresh batch')
    if bool(prior.get('native_starred',False)) != bool(getattr(args,'native_starred',False)):
        raise ValueError('Changed native starred-call contract requires a fresh batch')
    if prior.get('native_array_exercises'):
        raise ValueError('Frozen native array requirements cannot be dropped by a generic restart')
    if bool(prior.get('native_arrays',False)) != bool(getattr(args,'native_arrays',False)):
        raise ValueError('Changed native array contract requires a fresh batch')
    if prior.get('native_function_exercises'):
        raise ValueError('Frozen native call requirements cannot be dropped by a generic restart')
    if bool(prior.get('native_functions',False)) != bool(getattr(args,'native_functions',False)):
        raise ValueError('Changed native function contract requires a fresh batch')
    if prior.get('construction_exercises') or prior.get('object_arithmetic_exercises') or prior.get('scalar_conversion_exercises') or prior.get('object_unary_exercises'):
        raise ValueError('Frozen targeted requirements cannot be dropped by a generic batch restart')
    if bool(prior.get("object_unary",False)) != bool(getattr(args,"object_unary",False)):
        raise ValueError("Changed object unary contract requires a fresh batch")
    if bool(prior.get("scalar_conversion",False)) != bool(getattr(args,"scalar_conversion",False)):
        raise ValueError("Changed scalar conversion contract requires a fresh batch")
    if bool(prior.get("object_arithmetic",False)) != bool(getattr(args,"object_arithmetic",False)):
        raise ValueError("Changed DSL contract requires a fresh batch, not a same-prompt retry")
    if bool(prior.get('object_arrays',False)) != bool(getattr(args,'object_arrays',False)):
        raise ValueError('Changed DSL contract requires a fresh batch, not a same-prompt retry')
    if bool(prior.get('public_polynomial',False)) != bool(getattr(args,'public_polynomial',False)):
        raise ValueError('Changed DSL contract requires a fresh batch, not a same-prompt retry')
    if bool(prior.get('public_strings',False)) != bool(getattr(args,'public_strings',False)):
        raise ValueError('Changed DSL contract requires a fresh batch, not a same-prompt retry')
    if bool(prior.get('public_control',False)) != bool(getattr(args,'public_control',False)):
        raise ValueError('Changed DSL contract requires a fresh batch, not a same-prompt retry')
    if bool(prior.get('public_numbers',False)) != bool(getattr(args,'public_numbers',False)):
        raise ValueError('Changed DSL contract requires a fresh batch, not a same-prompt retry')


def inside(args):
    from hecate_python_env import ROOT, WORK, VENV, digest
    from deepseek_provider import generation_deadline
    from native_execution_slots import NATIVE_CONCURRENCY
    retries = getattr(args, 'provider_retries', 0)
    proxy_port = getattr(args, 'loopback_proxy_port', 0)
    key = os.environ.pop("DEEPSEEK_API_KEY", "")
    if not key or not os.environ.get("IN_NIX_SHELL") or Path(sys.prefix) != VENV:
        raise RuntimeError("Pinned environment and explicit credential required")
    if shutil.disk_usage(WORK).free < 20 * 1024**3:
        raise RuntimeError("Need 20 GiB free for independent encrypted case evidence")
    os.umask(0o077)
    root = Path(tempfile.mkdtemp(prefix="agent-batch-", dir=WORK / "results"))
    print(f"Agent batch evidence: {root}", flush=True)
    custom_path = getattr(args, 'case_manifest', None)
    custom = None
    if custom_path:
        from custom_batch_manifest import load_manifest
        rows, custom, custom_hash = load_manifest(custom_path, args.manifest_sha256)
        benchmark = None
    else:
        rows, benchmark = batch_plan(getattr(args, "extended", False))
    catalog = [r["descriptor"] for r in rows]
    selection = None
    remaining = getattr(args, 'remaining_from', None)
    if args.failed_from or remaining:
        source = (args.failed_from or remaining).resolve()
        if remaining:
            from interrupted_batch import load_remaining_source
            prior, catalog = load_remaining_source(catalog, source, WORK / "results")
        else:
            prior, catalog = load_failed_source(catalog, source, WORK / "results")
        validate_construction_continuation(prior,args)
        after = dict(model=args.model, service_provider=args.provider, reasoning_effort=args.reasoning_effort,
                     max_tokens=args.max_tokens, max_calls=4, timeout_seconds=args.api_timeout, stream=args.stream,
                     provider_retries=retries, loopback_proxy_port=proxy_port)
        changes = configuration_delta(generation_settings(prior), after)
        if changes and not args.allow_config_change:
            raise ValueError("Retest must preserve source generation configuration")
        selection = dict(kind="previous_failures_and_unfinished" if remaining else "previous_failures_only", source_report=str(source),
                         source_sha256=digest(source), original_planned=len(prior["cases"]),
                         configuration_changes=changes)
        selection['execution_configuration_changes'] = dict(
            api_concurrency=dict(before=prior.get('api_concurrency', prior.get('concurrency')), after=args.jobs),
            native_concurrency=dict(before=prior.get('native_execution_concurrency'), after=NATIVE_CONCURRENCY))
    selected_ids = {d["id"] for d in catalog}
    rows = [r for r in rows if r["descriptor"]["id"] in selected_ids]
    report = dict(status="running", model=args.model, service_provider=args.provider, max_repairs=3, api_timeout=args.api_timeout,
                  stream=args.stream, provider_retries=retries, loopback_proxy_port=proxy_port,
                  connection_gate=None,  # Retain the historical report field, not the removed provider gate.
                  reasoning_effort=args.reasoning_effort, max_tokens=args.max_tokens, concurrency=args.jobs, cases=rows,
                  api_concurrency=args.jobs, native_execution_concurrency=NATIVE_CONCURRENCY,
                  backend="upstream_SEAL_HEVM_CPU", poseidon_gpu_validated=False,
                  runner_sha256=digest(Path(__file__)), started_unix=time.time())
    if getattr(args, 'compiler_configuration', None) is not None:
        from seal_cpu_golden import PROFILE
        report['compiler_configuration'] = configuration(args.compiler_configuration, digest(PROFILE))
    report['public_numbers'] = bool(getattr(args,'public_numbers',False))
    report['public_control'] = bool(getattr(args,'public_control',False))
    report['public_strings'] = bool(getattr(args,'public_strings',False))
    report['object_arrays'] = bool(getattr(args,'object_arrays',False))
    report['native_functions'] = bool(getattr(args,'native_functions',False))
    report['native_arrays'] = bool(getattr(args,'native_arrays',False))
    report['native_starred'] = bool(getattr(args,'native_starred',False))
    report['native_array_arithmetic'] = bool(getattr(args,'native_array_arithmetic',False))
    report['native_public_loops'] = bool(getattr(args,'native_public_loops',False))
    report['native_scalar_augmented'] = bool(getattr(args,'native_scalar_augmented',False))
    report['native_array_mutation'] = bool(getattr(args,'native_array_mutation',False))
    if getattr(args,'native_star_exercises',False):
        from native_star_exercises import CATALOG
        report['native_star_exercises'] = dict(schema=1,catalog_sha256=digest(CATALOG))
    if getattr(args,'native_array_exercises',False):
        from native_array_exercises import CATALOG
        report['native_array_exercises'] = dict(schema=1,catalog_sha256=digest(CATALOG))
    if getattr(args,'native_function_exercises',False):
        from native_function_exercises import CATALOG
        report['native_function_exercises'] = dict(schema=1,catalog_sha256=digest(CATALOG))
    if getattr(args,"object_unary_exercises",False):
        from object_unary_exercises import CATALOG
        report["object_unary_exercises"]=dict(schema=1,catalog_sha256=digest(CATALOG))
    report["object_unary"] = bool(getattr(args,"object_unary",False))
    report["scalar_conversion"] = bool(getattr(args,"scalar_conversion",False))
    if getattr(args,"scalar_conversion_exercises",False):
        from scalar_conversion_exercises import CATALOG
        report["scalar_conversion_exercises"] = dict(schema=1,catalog_sha256=digest(CATALOG))
    report["object_arithmetic"] = bool(getattr(args,"object_arithmetic",False))
    report['public_polynomial'] = bool(getattr(args,'public_polynomial',False))
    if selection:
        report["selection"] = selection
    if custom is not None:
        raw = custom_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != custom_hash:
            raise ValueError('Custom manifest changed while snapshotting')
        (root / 'custom-manifest.json').write_bytes(raw)
        report['custom_manifest'] = dict(file='custom-manifest.json', sha256=custom_hash,
                                        case_count=len(custom['cases']))
    if benchmark:
        save(root / "benchmark-manifest.json", benchmark)
        report["benchmark"] = {k:v for k, v in benchmark.items() if k != "entries"}
    if getattr(args, "construction_exercises", False):
        from construction_exercises import CATALOG
        report["construction_exercises"] = dict(schema=1, catalog_sha256=digest(CATALOG))
    if getattr(args,"object_arithmetic_exercises",False):
        from object_arithmetic_exercises import CATALOG
        report["object_arithmetic_exercises"] = dict(schema=1,catalog_sha256=digest(CATALOG))
    def checkpoint():
        report["summary"] = summarize(rows)
        report["elapsed_seconds"] = time.time()-report["started_unix"]
        save(root / "report.json", report)
    checkpoint()
    provider_stop = threading.Event()
    def execute(index):
        if provider_stop.is_set():
            return index, None
        row = rows[index]
        folder = root / row["descriptor"]["id"]
        folder.mkdir()
        save(folder / "model.json", row["descriptor"])
        case_timeout = 1800 + generation_deadline(args.api_timeout, 4, retries)
        command = ["/usr/bin/timeout", "-k", "5s", str(case_timeout) + "s", sys.executable,
                   str(ROOT / "scripts/baseline/run_candidate.py"), "--inside", "--deepseek",
                   "--case", str(folder / "model.json"), "--model", args.model, "--provider", args.provider,
                   "--reasoning-effort", args.reasoning_effort, "--max-tokens", str(args.max_tokens),
                   "--api-timeout", str(args.api_timeout), '--provider-retries', str(retries),
                   '--loopback-proxy-port', str(proxy_port)]
        command += construction_options(args)
        command += compiler_options(args)
        if getattr(args,'native_star_exercises',False) or getattr(args,'native_array_exercises',False) or getattr(args,'native_function_exercises',False) or getattr(args, "construction_exercises", False) or getattr(args,"object_arithmetic_exercises",False) or getattr(args,"scalar_conversion_exercises",False) or getattr(args,"object_unary_exercises",False):
            command += ["--construction-exercise", row["descriptor"]["id"].removeprefix("exercise-")]
        if args.stream:
            command += ['--stream']
        env = dict(os.environ, DEEPSEEK_API_KEY=key, PYTHONDONTWRITEBYTECODE="1")
        started = time.monotonic()
        output = ""
        try:
            process = subprocess.run(command, env=env, cwd=ROOT, capture_output=True, timeout=case_timeout + 10)
            output = process.stdout.decode("utf-8", errors="replace")
            # Only trusted runner stdout, never credentials/arguments/environment.
            (folder / "runner.log").write_text(output)
            paths = [line.removeprefix("Candidate evidence: ") for line in output.splitlines()
                     if line.startswith("Candidate evidence: ")]
            if len(paths) != 1:
                raise ValueError("missing_report")
            evidence = Path(paths[0])
            if evidence.parent != WORK / "results" or not evidence.name.startswith("agent-deepseek-"):
                raise ValueError("invalid_report_path")
            result = json.loads((evidence / "report.json").read_text())
            if json.loads((evidence / "model.json").read_text()) != row["descriptor"]:
                raise ValueError("case_identity_mismatch")
            if result.get('compiler_configuration') != report.get('compiler_configuration'):
                raise ValueError('compiler_configuration_mismatch')
            metrics = case_metrics(result)
            if should_pause_batch(args.provider, metrics):
                provider_stop.set()
            return index, dict(status="passed" if metrics["passed"] else "failed", metrics=metrics,
                               evidence=str(evidence), exit_code=process.returncode,
                               seconds=time.monotonic()-started)
        except Exception as error:
            # Unknown subprocess exceptions might contain the child environment.
            result = dict(status="infrastructure_failed", failure_layer="batch_worker", attempts=[])
            return index, dict(status="failed", metrics=case_metrics(result),
                               diagnostic_type=type(error).__name__, seconds=time.monotonic()-started)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(execute, i) for i in range(len(rows))]
        for future in as_completed(futures):
            index, outcome = future.result()
            if outcome is None:
                continue
            rows[index].update(outcome)
            checkpoint()
            m = outcome["metrics"]
            print(f"{report['summary']['completed']}/{len(rows)} {rows[index]['descriptor']['id']}: "
                  f"{outcome['status']} calls={m['api_calls']} "
                  f"layer={m['failure_layer']} error={m['provider_error']}", flush=True)
    if provider_stop.is_set():
        report['status'] = 'paused_provider_failure'
        report['stop_reason'] = 'Provider safety gate failed; no further queued cases started (in-flight calls may finish)'
    else:
        report["status"] = "passed" if all(r["metrics"]["passed"] for r in rows) else "completed_with_failures"
    checkpoint()
    print(json.dumps(report["summary"], indent=2), flush=True)
    return 0 if report["status"] == "passed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler-configuration', choices=tuple(CONFIGURATIONS),
                        help='Explicit immutable compiler profile; changing it requires a fresh batch')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--deepseek", "--live", dest="deepseek", action="store_true")
    mode.add_argument("--plan", action="store_true", help="Print cohort without loading credentials or calling any API")
    parser.add_argument("--construction-exercises", action="store_true",
                        help="Frozen per-construct implementation-form tests, not free synthesis")
    parser.add_argument("--object-unary",action="store_true")
    parser.add_argument('--native-functions',action='store_true',help='Independent typed native-call contract')
    parser.add_argument('--native-arrays',action='store_true',help='Independent native object-array return contract')
    parser.add_argument('--native-function-exercises',action='store_true',help='Frozen eleven-case native-call cohort')
    parser.add_argument('--native-array-exercises',action='store_true',help='Frozen ten-case native array cohort')
    parser.add_argument('--native-star-exercises',action='store_true',help='Frozen eight-case starred argument cohort')
    parser.add_argument('--native-starred',action='store_true',help='Native positional argument unpacking contract')
    parser.add_argument('--native-array-arithmetic',action='store_true',help='Native object-array arithmetic contract')
    parser.add_argument('--native-public-loops',action='store_true',help='Native construction-time public range contract')
    parser.add_argument('--native-scalar-augmented',action='store_true',help='Scalar native augmented rebinding contract')
    parser.add_argument('--native-array-mutation',action='store_true',help='Alias-aware native ndarray inplace contract')
    parser.add_argument("--object-unary-exercises",action="store_true",help="Frozen v22 eight-case cohort")
    parser.add_argument("--scalar-conversion",action="store_true")
    parser.add_argument("--scalar-conversion-exercises",action="store_true",help="Frozen v21 extraction cohort")
    parser.add_argument("--object-arithmetic",action="store_true")
    parser.add_argument("--object-arithmetic-exercises",action="store_true",help="Frozen v20 ten-case cohort")
    parser.add_argument("--extended", action="store_true", help="16-family x 6-configuration user-graph cohort")
    parser.add_argument('--public-numbers',action='store_true',help='Generate under request-v14 public numeric/array contract')
    parser.add_argument('--public-control',action='store_true',help='Generate under request-v15 public control contract')
    parser.add_argument('--public-strings',action='store_true',help='Generate under request-v16 public string contract')
    parser.add_argument('--public-polynomial',action='store_true',help='Generate under request-v17 public Chebyshev contract')
    parser.add_argument('--object-arrays',action='store_true',help='Generate under request-v18 object storage contract')
    parser.add_argument('--case-manifest', type=Path, help='Data-only JSON of user graphs; replaces the fixed catalog')
    parser.add_argument('--manifest-sha256', help=argparse.SUPPRESS)
    parser.add_argument("--provider", choices=("deepseek",), default="deepseek")
    parser.add_argument("--inside", action="store_true")
    selection_mode = parser.add_mutually_exclusive_group()
    selection_mode.add_argument("--failed-from", type=Path, help="Retest only failures from a completed, hash-linked catalog report")
    selection_mode.add_argument("--remaining-from", type=Path, help="Failures and unfinished cases from a sealed interrupted report")
    parser.add_argument("--allow-config-change", action="store_true", help="Explicit new generation experiment; preserve source report")
    parser.add_argument("--reasoning-effort", choices=("low", "high", "max"), default="high")
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--api-timeout", type=float, default=1200)
    parser.add_argument('--provider-retries', type=int, choices=range(4), default=0)
    parser.add_argument('--stream', action='store_true')
    parser.add_argument('--loopback-proxy-port', type=int, default=0,
                        help='Explicit DeepSeek CONNECT via 127.0.0.1; zero keeps the default route')
    parser.add_argument("--jobs", type=int, choices=range(1, 11), default=10,
                        help='Candidate/API workers (default 10); native execution independently limited to 2')
    parser.add_argument("--model", choices=("deepseek-flash", "deepseek-v4-pro", "deepseek-v4-flash"), default="deepseek-flash")
    args = parser.parse_args()
    if args.native_array_mutation:
        args.native_scalar_augmented = True
    if args.native_scalar_augmented:
        args.native_public_loops = True
    if args.native_star_exercises:
        if (args.native_array_exercises or args.native_function_exercises or args.case_manifest or args.extended or
                args.failed_from or args.remaining_from or args.native_array_arithmetic or args.native_public_loops):
            parser.error('Native starred exercises require a fresh fixed starred-only cohort')
        args.case_manifest = Path(__file__).with_name('cases')/'native-star-exercises-8-manifest.json'
        args.native_starred = True
    if args.native_public_loops:
        args.native_array_arithmetic = True
    if args.native_array_arithmetic:
        args.native_starred = True
    if args.native_starred:
        if args.native_array_exercises or args.native_function_exercises:
            parser.error('Frozen native exercises cannot change starred-call contract')
        args.native_arrays = True
    if args.native_array_exercises:
        if args.native_function_exercises or args.case_manifest or args.extended or args.failed_from or args.remaining_from:
            parser.error('Native array exercises require a fresh fixed cohort')
        args.case_manifest = Path(__file__).with_name('cases')/'native-array-exercises-10-manifest.json'
        args.native_arrays = True
    if args.native_arrays:
        args.native_functions = True
        if args.native_function_exercises:
            parser.error('Frozen native function exercises cannot change array contract')
    if args.native_function_exercises:
        if args.case_manifest or args.extended or args.failed_from or args.remaining_from:
            parser.error('Native function exercises require a fresh fixed cohort')
        args.case_manifest = Path(__file__).with_name('cases')/'native-function-exercises-11-manifest.json'
        args.native_functions = True
    if args.native_functions and any((args.construction_exercises,args.object_unary_exercises,
            args.scalar_conversion_exercises,args.object_arithmetic_exercises,args.object_unary,
            args.scalar_conversion,args.object_arithmetic,args.object_arrays,args.public_polynomial,
            args.public_strings,args.public_control,args.public_numbers)):
        parser.error('Native function contract is separate; do not mix grammar flags')
    if args.object_unary_exercises:
        if args.construction_exercises or args.scalar_conversion_exercises or args.object_arithmetic_exercises or args.case_manifest or args.extended or args.failed_from or args.remaining_from:
            parser.error("Object unary requires a fresh fixed cohort")
        args.case_manifest=Path(__file__).with_name("cases")/"object-unary-exercises-8-manifest.json"
        args.object_unary=True
    if args.object_unary and (args.construction_exercises or args.object_arithmetic_exercises or args.scalar_conversion_exercises):
        parser.error("Frozen earlier exercises cannot use object unary contract")
    if args.scalar_conversion_exercises:
        if args.construction_exercises or args.object_arithmetic_exercises or args.case_manifest or args.extended or args.failed_from or args.remaining_from:
            parser.error("Scalar extraction requires a fresh fixed cohort")
        args.case_manifest=Path(__file__).with_name("cases")/"scalar-conversion-exercises-15-manifest.json"
        args.scalar_conversion=True
    if args.scalar_conversion and (args.construction_exercises or args.object_arithmetic_exercises):
        parser.error("Frozen older exercises cannot use v21")
    if args.object_arithmetic_exercises:
        if args.construction_exercises or args.case_manifest or args.extended or args.failed_from or args.remaining_from:
            parser.error("Object arithmetic exercises require a fresh fixed cohort")
        args.case_manifest=Path(__file__).with_name("cases")/"object-arithmetic-exercises-10-manifest.json"
        args.object_arithmetic=True
    if args.construction_exercises and args.object_arithmetic:
        parser.error("Frozen v19 exercises cannot use v20")
    if args.construction_exercises:
        if args.case_manifest or args.extended or args.failed_from or args.remaining_from:
            parser.error("Construction exercises require a fresh fixed cohort")
        args.case_manifest = Path(__file__).with_name("cases")/"construction-exercises-30-manifest.json"
    if args.case_manifest and args.extended:
        parser.error('--case-manifest and --extended are mutually exclusive')
    if args.manifest_sha256 and (not args.inside or not args.case_manifest):
        parser.error('--manifest-sha256 is reserved for the isolated custom-batch entry')
    custom_rows = None
    if args.case_manifest:
        from custom_batch_manifest import load_manifest
        custom_rows, _, args.manifest_sha256 = load_manifest(args.case_manifest, args.manifest_sha256)
        if args.construction_exercises:
            from construction_exercises import EXERCISES, descriptor
            if [r['descriptor'] for r in custom_rows] != [descriptor(name) for name in EXERCISES]:
                parser.error('Construction exercise model cohort changed')
    if args.object_arithmetic_exercises:
        from object_arithmetic_exercises import EXERCISES,descriptor
        if [r["descriptor"] for r in custom_rows] != [descriptor(name) for name in EXERCISES]:
            parser.error("Object arithmetic cohort changed")
    if args.scalar_conversion_exercises:
        from scalar_conversion_exercises import EXERCISES,descriptor
        if [r["descriptor"] for r in custom_rows] != [descriptor(name) for name in EXERCISES]:
            parser.error("Scalar extraction cohort changed")
    if args.object_unary_exercises:
        from object_unary_exercises import EXERCISES,descriptor
        if [r["descriptor"] for r in custom_rows] != [descriptor(name) for name in EXERCISES]:
            parser.error("Object unary cohort changed")
    if args.native_function_exercises:
        from native_function_exercises import EXERCISES,descriptor
        if [r['descriptor'] for r in custom_rows] != [descriptor(name) for name in EXERCISES]:
            parser.error('Native function cohort changed')
    if args.native_array_exercises:
        from native_array_exercises import EXERCISES,descriptor
        if [r['descriptor'] for r in custom_rows] != [descriptor(name) for name in EXERCISES]:
            parser.error('Native array cohort changed')
    if args.native_star_exercises:
        from native_star_exercises import EXERCISES,descriptor
        if [r['descriptor'] for r in custom_rows] != [descriptor(name) for name in EXERCISES]:
            parser.error('Native starred cohort changed')
    from deepseek_provider import MODEL_OUTPUT_LIMITS
    if args.max_tokens is None:
        args.max_tokens = MODEL_OUTPUT_LIMITS[args.model]
    if args.plan:
        if args.inside or args.failed_from or args.remaining_from or args.allow_config_change:
            parser.error("--plan does not accept execution/retest flags")
        rows, benchmark = (custom_rows, None) if custom_rows is not None else batch_plan(args.extended)
        print(json.dumps(dict(mode="plan_only", agent_calls=0, cases=rows,
                              compiler_configuration=configuration(args.compiler_configuration) if args.compiler_configuration else None,
                              public_numbers=args.public_numbers,
                              public_control=args.public_control,
                              public_strings=args.public_strings,
                              public_polynomial=args.public_polynomial,
                              object_arrays=args.object_arrays,
                              native_functions=args.native_functions,
                              native_arrays=args.native_arrays,
                              native_starred=args.native_starred,
                              native_array_arithmetic=args.native_array_arithmetic,
                              native_public_loops=args.native_public_loops,
                              native_scalar_augmented=args.native_scalar_augmented,
                              native_array_mutation=args.native_array_mutation,
                              native_star_exercises=args.native_star_exercises,
                              native_array_exercises=args.native_array_exercises,
                              native_function_exercises=args.native_function_exercises,
                              object_unary=args.object_unary,
                              object_unary_exercises=args.object_unary_exercises,
                              scalar_conversion=args.scalar_conversion,
                              scalar_conversion_exercises=args.scalar_conversion_exercises,
                              object_arithmetic=args.object_arithmetic,
                              object_arithmetic_exercises=args.object_arithmetic_exercises,
                              service_provider=args.provider, model=args.model, api_concurrency=args.jobs,
                              native_execution_concurrency=2,
                              benchmark=benchmark, summary=summarize(rows)), indent=2))
        return 0
    from deepseek_provider import Config
    config = Config(model=args.model, service_provider=args.provider, reasoning_effort=args.reasoning_effort, max_tokens=args.max_tokens,
           timeout_seconds=args.api_timeout, stream=args.stream, provider_retries=args.provider_retries,
           loopback_proxy_port=args.loopback_proxy_port)
    if args.allow_config_change and not (args.failed_from or args.remaining_from):
        parser.error("--allow-config-change requires a source report")
    from hecate_python_env import ROOT, VENV, enter_nix
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError("Run from the configured source root")
    if args.inside:
        return inside(args)
    from agent_credentials import load_api_key
    previous = os.environ.get("DEEPSEEK_API_KEY")
    os.environ["DEEPSEEK_API_KEY"] = load_api_key(ROOT, provider=args.provider)
    try:
        extra = ["--failed-from", str(args.failed_from.resolve())] if args.failed_from else []
        extra += compiler_options(args)
        if args.remaining_from:
            extra += ["--remaining-from", str(args.remaining_from.resolve())]
        if args.stream:
            extra += ['--stream']
        extra += ["--provider", args.provider, "--reasoning-effort", args.reasoning_effort, "--max-tokens", str(args.max_tokens),
                  "--api-timeout", str(args.api_timeout), '--provider-retries', str(args.provider_retries),
                  '--loopback-proxy-port', str(args.loopback_proxy_port)]
        if args.allow_config_change:
            extra.append("--allow-config-change")
        if args.extended:
            extra.append("--extended")
        if args.construction_exercises:
            extra += ["--construction-exercises"]
        if args.object_arithmetic_exercises:
            extra += ["--object-arithmetic-exercises"]
        if args.scalar_conversion_exercises:
            extra += ["--scalar-conversion-exercises"]
        if args.object_unary_exercises:
            extra += ["--object-unary-exercises"]
        if args.native_function_exercises:
            extra += ['--native-function-exercises']
        if args.native_array_exercises:
            extra += ['--native-array-exercises']
        if args.native_star_exercises:
            extra += ['--native-star-exercises']
        if args.case_manifest and not args.native_star_exercises and not args.native_array_exercises and not args.native_function_exercises and not args.construction_exercises and not args.object_arithmetic_exercises and not args.scalar_conversion_exercises and not args.object_unary_exercises:
            extra += ['--case-manifest', str(args.case_manifest.resolve()), '--manifest-sha256', args.manifest_sha256]
        if not args.construction_exercises:
            extra += construction_options(args)
        command = ('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" PYTHONDONTWRITEBYTECODE=1 ' +
                   shlex.join([str(VENV / "bin/python"), str(Path(__file__).resolve()), "--inside", "--deepseek",
                               "--jobs", str(args.jobs), "--model", args.model, *extra]))
        # Covers the full catalog even with every allowed repair at its deadline.
        planned = len(custom_rows) if custom_rows is not None else 96 if args.extended else 48
        from deepseek_provider import generation_deadline
        batch_timeout = 600 + ((planned + args.jobs - 1) // args.jobs) * (1810 + generation_deadline(args.api_timeout, 4, args.provider_retries))
        return enter_nix(command, seconds=batch_timeout, keep_env=("DEEPSEEK_API_KEY",))
    finally:
        if previous is None:
            os.environ.pop("DEEPSEEK_API_KEY", None)
        else:
            os.environ["DEEPSEEK_API_KEY"] = previous


if __name__ == "__main__":
    raise SystemExit(main())
