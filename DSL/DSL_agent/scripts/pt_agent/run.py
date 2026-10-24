"""Run the rule PT-DSL planner and its separate development evaluation stages."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time


STAGES = ("validate", "plan", "check", "mock", "count", "emit", "export", "compile", "test")


def select(grid_rows):
    """Select only from completed validation rows with the prescribed margin."""
    eligible = [row for row in grid_rows if not row["diverged"] and row["max_u"] <= 0.95]
    if not eligible:
        raise ValueError("validation has no nondivergent candidate with max_u <= 0.95")
    row = min(eligible, key=lambda row: (-row["agree"], row["deg"], row["margin"]))
    return dict(deg=row["deg"], margin=row["margin"])


def _test_progress(path, program_text):
    state = dict(correct=0, diverged_indices=[], images_done=0, program_text=program_text)
    if path is not None and path.exists():
        saved = json.loads(path.read_text())
        if saved.get("program_text") == program_text:
            return saved
        stale = path.with_name("test_progress.stale.json")
        if stale.exists():
            stale = path.with_name("test_progress.stale."+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")+".json")
        path.rename(stale)
    return state


def evaluate_batches(run_batch, images, labels, batch, *, on_progress=None,
                     progress_path=None, program_text=None):
    """Count every image, isolating BackendError samples without changing the program."""
    import numpy as np
    from .backends.base import BackendError

    state = _test_progress(progress_path, program_text)
    saved_at = state["images_done"]

    def result():
        return dict(top1_test=100.0*state["correct"]/len(images), correct=state["correct"],
                    n_diverged=len(state["diverged_indices"]), diverged_indices=list(state["diverged_indices"]),
                    test_images_done=state["images_done"])

    def commit(correct, diverged, done):
        nonlocal state, saved_at
        state = dict(correct=correct, diverged_indices=diverged, images_done=done, program_text=program_text)
        if progress_path is not None and done-saved_at >= 500:
            _write(progress_path, state)
            saved_at = done

    try:
        for start in range(state["images_done"], len(images), batch):
            stop = min(start+batch, len(images))
            failed = False
            try:
                logits = run_batch(images[start:stop])
            except BackendError:
                failed = True
            if failed:
                gc.collect()
                for index in range(start, stop):
                    correct = state["correct"]
                    diverged = list(state["diverged_indices"])
                    try:
                        logits = run_batch(images[index:index+1])
                    except BackendError:
                        diverged.append(index)
                    else:
                        correct += int(logits.argmax(axis=1)[0] == labels[index])
                    commit(correct, diverged, index+1)
            else:
                correct = state["correct"]+int(np.sum(logits.argmax(axis=1) == labels[start:stop]))
                commit(correct, list(state["diverged_indices"]), stop)
            if on_progress is not None:
                on_progress(result())
        return result()
    finally:
        if progress_path is not None:
            _write(progress_path, state)


def report_test(run_batch, model, program_text, out, report, *, seconds=7200):
    """Evaluate a frozen program on report-only data, with a resumable deadline."""
    if getattr(model, 'samples', None) is not None:
        return report_model_test(run_batch, model.info, model.samples, out, report)
    import signal
    import numpy as np
    import torch
    from .data.cifar import report_test_set
    from .planners.agent_tools import VAL_BATCH

    progress_path = out/"test_progress.json"
    _write(progress_path, _test_progress(progress_path, program_text))
    started = time.monotonic()
    old_handler = signal.getsignal(signal.SIGALRM)
    old_timer = signal.getitimer(signal.ITIMER_REAL)

    def timeout(signum, frame):
        raise TimeoutError("test stage exceeded 7200 seconds")

    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        if getattr(model, 'samples', None) is not None:
            arrays, labels = model.samples.get('report')
            images = arrays[model.info.graph.inputs[0]]
        else:
            images, labels = report_test_set(normalization=model.normalization)
            images = model.adapt(images)
        report["test_split"] = "test[0:8000]"
        if report.get("torch_top1_report") is None:
            correct = 0
            with torch.no_grad():
                for start in range(0, len(images), VAL_BATCH):
                    logits = model.module(torch.from_numpy(images[start:start+VAL_BATCH]).float())
                    correct += int(np.sum(logits.argmax(dim=1).cpu().numpy() == labels[start:start+VAL_BATCH]))
            report["torch_top1_report"] = 100.0*correct/len(images)
            _write(out/"REPORT.json", report)

        def progress(result):
            report.update(result, test_seconds=time.monotonic()-started)
            _write(out/"REPORT.json", report)
            print(f"TEST images={result['test_images_done']} top1={result['top1_test']} "
                  f"n_diverged={result['n_diverged']}", flush=True)

        report.update(evaluate_batches(run_batch, images, labels, VAL_BATCH, on_progress=progress,
                                       progress_path=progress_path, program_text=program_text))
        report["m1_item3_pass"] = report["top1_test"] >= report["torch_top1_report"]-1.0
    except TimeoutError:
        state = json.loads(progress_path.read_text())
        report.update(correct=state["correct"], test_images_done=state["images_done"],
                      diverged_indices=state["diverged_indices"], n_diverged=len(state["diverged_indices"]),
                      top1_test=100.0*state["correct"]/8000, test_seconds=time.monotonic()-started)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
        if old_timer[0]:
            signal.setitimer(signal.ITIMER_REAL, *old_timer)


def report_model_test(run_batch, info, samples, out, report):
    """Generic external report stage; its results never enter the planner."""
    import numpy as np
    from .analyzer import reference_forward
    from .metrics import compare, passes
    from .planners.agent_tools import VAL_BATCH
    arrays, labels = samples.get('report')
    total = len(next(iter(arrays.values())))
    exact, approximate, diverged = [], [], []
    for start in range(0, total, VAL_BATCH):
        batch = {name: value[start:start+VAL_BATCH] for name, value in arrays.items()}
        reference = reference_forward(info.graph, batch)[info.graph.outputs[0]]
        try:
            values = run_batch(batch)
        except Exception as exc:
            from .backends.base import BackendError
            if not isinstance(exc, BackendError):
                raise
            values = np.full_like(reference, np.nan)
            for offset in range(len(reference)):
                try:
                    values[offset:offset+1] = run_batch({name: value[offset:offset+1] for name, value in batch.items()})
                except BackendError:
                    diverged.append(start+offset)
        exact.append(reference); approximate.append(values)
    metric_input = dict(values=np.concatenate(exact), task=info.task)
    if info.task == 'causal_lm':
        metric_input['lengths'] = arrays['attention_mask'].sum(axis=1).astype(int)
    result = compare(metric_input, np.concatenate(approximate), labels)
    report.update(report_metrics=result, report_pass=passes(result, info.threshold), test_split='report',
                  n_diverged=len(diverged), diverged_indices=diverged, test_images_done=total)
    _write(out/'REPORT.json', report)


def _write(path, record):
    temporary = path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def _defaults():
    from .registry import load

    for row in reversed(load()):
        metrics = row["metrics"]
        emit = metrics.get("emit") or {}
        if "deg" in emit and any(not case.get("diverged", True) and case["deg"] == emit["deg"]
                                 and case["margin"] == emit["margin"] for case in metrics.get("grid", [])):
            return emit["deg"], emit["margin"]
    return 63, 1.25


def _server_job(out, package, weights):
    from . import toolchain

    import shutil

    package, weights = Path(package), Path(weights)
    path = out / "server_jobs" / package.name / "run.sh"
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(package, path.parent/package.name)
    shutil.copy2(weights, path.parent/weights.name)
    path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n"
                    "# Run in the prepared Nix environment; transfer this entire job directory.\n"
                    + toolchain.server_env_exports() +
                    'export HECATE_BUILD="$PT_HECATE_BUILD"\n'
                    'export HECATE="$(dirname "$HECATE_BUILD")"\n'
                    'export PYTHONPATH="$PT_DACAPO_ROOT/python/hecate"\n'
                    'export LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH"\n'
                    'cd -- "$(dirname -- "$0")"\n'
                    'test "$(df -Pk . | awk \'NR==2 {print $4}\')" -ge 10485760\n'
                    'ulimit -v 6291456\nulimit -f 8388608\n'
                    'timeout 3600 python3 -B '+shlex.quote(package.name+"/trace_model.py")+
                    ' --weights '+shlex.quote(weights.name)+' --output-dir trace\n'
                    'timeout 3600 python3 -B '+shlex.quote(package.name+"/compile_model.py")+
                    ' --hecate-opt "$HECATE_BUILD/bin/hecate-opt" --input trace/trace_model.mlir --output-dir compiled\n')
    path.chmod(0o755)
    return path


def _compile_export(record):
    from . import toolchain
    from .cost import plan_latency
    toolchain.setup_env()
    from hecate_python_env import VENV

    package, weights = Path(record["package"]), Path(record["weights"])
    directory = package.parent
    trace_dir, compile_dir = directory/"trace", directory/"compiled"
    traced = toolchain.trusted_run(directory, [str(VENV/"bin/python"), "-B", str(package/"trace_model.py"),
        "--weights", str(weights), "--output-dir", str(trace_dir)], directory/"trace.log",
        env_extra=toolchain.package_trace_env())
    mlir = trace_dir/"trace_model.mlir"
    traced.update(mlir=mlir.is_file(), mlir_bytes=mlir.stat().st_size if mlir.is_file() else 0)
    result = dict(trace=traced, compile=None, plan=None, plan_latency=None)
    if traced["exit"] != 0 or not traced["mlir"]:
        return result
    compiled = toolchain.trusted_run(directory, [str(VENV/"bin/python"), "-B", str(package/"compile_model.py"),
        "--hecate-opt", str(toolchain.host_paths()["hecate_opt"]), "--input", str(mlir),
        "--output-dir", str(compile_dir)], directory/"compile.log",
        env_extra=dict(LD_LIBRARY_PATH=os.environ["HECATE_PYTHON_LIBRARY_PATH"]))
    name = json.loads((package/"config/model.json").read_text())["name"]
    plan_path = compile_dir/f"lowered._hecate_{name}.runtime-plan.json"
    compiled["plan"] = plan_path.is_file()
    result["compile"] = compiled
    if compiled["exit"] == 0 and compiled["plan"]:
        result.update(plan=toolchain.plan_stats(compile_dir, name), plan_latency=plan_latency(plan_path))
    return result


def _inside(args):
    import numpy as np
    from . import toolchain
    from .analyzer import calibrate, reference_forward
    from .backends import CountBackend, EmitBackend, MockBackend, RefreshBackend
    from .cost import PROFILES, estimate, plan_latency
    from .backends.base import BackendError
    from .data.cifar import calibration_set
    from .dsl.parser import parse
    from .dsl.typecheck import check
    from .lower import decode_outputs, lower, pack_inputs
    from .models.zoo import get
    from .planners.rule import plan
    from .kernels import MuxLayout
    from .planners.agent_tools import VAL_BATCH
    from .models.info import ModelInfo
    from .data.samples import legacy_cifar, StoredSplits

    toolchain.setup_env()
    out = args.out
    path = out / "REPORT.json"
    report = dict(round="R7" if args.planner == "agent" else "R6", complete=False, stage=None, frontend="aten",
                  deg=args.deg, margin=args.margin, refresh=args.refresh,
                  program_path=str(out / "program.ptdsl"), source_lines=None, check=None,
                  exact_max_abs_diff=None, agree=None, top1=None, logit_rmse=None, mock_seconds=None,
                  mock_images_done=0, count=None, counts=None, max_depth=None, emit=None, export=None,
                  constants_bytes=None, trace=None, compile=None, plan=None, server_job=None,
                  agent=None, llm_calls=0, prompt_tokens=0, completion_tokens=0, best_iteration=None,
                  latency_pred_us=None, activations=None,
                  plan_latency=None, pruned=None, validation=None, top1_test=None, n_diverged=None, diverged_indices=None,
                  m1_item3_pass=None, test_images_done=0)
    if (args.stages == ["test"] or ("check" in args.stages and "plan" not in args.stages)) and path.exists():
        report.update(json.loads(path.read_text()))
    report.update(complete=False, val_split="test[9000:10000]", test_split="test[0:8000]")
    report.pop("error", None)
    _write(path, report)
    graph = program = images = labels = calibration = zm = info = samples = None

    def prepare_model():
        nonlocal zm, graph, info, samples
        if getattr(args, 'model_info', None) is not None:
            from types import SimpleNamespace
            info = ModelInfo.load(args.model_info)
            graph = info.graph
            samples = StoredSplits(Path(args.model_info).parent / 'samples', info,
                                   getattr(args, 'prepare_split', None))
            zm = SimpleNamespace(info=info, samples=samples)
            report.update(model_info=str(args.model_info), val_split='heldout', test_split='report')
        else:
            zm = get(args.model, args.models_dir)
            graph = zm.import_graph()
            info, samples = legacy_cifar(graph, normalization=zm.normalization)
        info.save(out)

    def prepare_calibration():
        nonlocal images, labels, calibration
        arrays, labels = samples.get('calibration')
        images = arrays[graph.inputs[0]]
        calibration = calibrate(graph, arrays)
        _write(out / 'calibration.json', calibration)

    def evaluate(batch, *, exact=False, candidate=None, observer=None):
        active = program if candidate is None else candidate
        arrays = [batch[name] for name in graph.inputs] if isinstance(batch, dict) else [batch]
        backend = MockBackend(active.header.slots, batch=len(arrays[0]))
        backend.bind(pack_inputs(active, arrays))
        values, layouts = lower(active, graph.weights, backend, act_mode="exact" if exact else "program",
                               observer=observer)
        logits = decode_outputs([value.value for value in values], layouts)[0].copy()
        del backend, values
        gc.collect()
        return logits

    try:
        if "compile" in args.stages and "export" not in args.stages:
            raise RuntimeError("compile requires the export stage")
        if args.stages == ["test"]:
            prepare_model()
            program = parse((out/"program.ptdsl").read_text())
        elif "check" in args.stages and "plan" not in args.stages:
            prepare_model()
            arrays, labels = samples.get('calibration')
            images = arrays[graph.inputs[0]]
            calibration_path = out/"calibration.json"
            calibration = (json.loads(calibration_path.read_text()) if calibration_path.exists()
                           else calibrate(graph, arrays))
            _write(calibration_path, calibration)
            report["source_lines"] = sum(bool(line.strip()) for line in (out/"program.ptdsl").read_text().splitlines())
        for stage in STAGES:
            if stage not in args.stages:
                continue
            report["stage"] = stage
            _write(path, report)
            if stage in ("validate", "plan") and graph is None:
                prepare_model()
                prepare_calibration()
            if stage == "validate":
                from .data.cifar import validation_set

                if getattr(args, 'model_info', None) is not None:
                    validation_arrays, _ = samples.get('heldout')
                    validation_images = validation_arrays[graph.inputs[0]]
                else:
                    validation_images, _ = validation_set(2000, normalization=zm.normalization)
                    validation_images = zm.adapt(validation_images)
                act_names = [node.inputs[0] for node in graph.nodes if node.op == "act"]
                needed = dict.fromkeys(act_names, 0.0)
                references = []
                for start in range(0, len(validation_images), VAL_BATCH):
                    batch = validation_images[start:start+VAL_BATCH]
                    values = reference_forward(graph, {graph.inputs[0]: batch}, keep=True)
                    references.append(values[graph.outputs[0]].copy())
                    for name in act_names:
                        value = values[name].reshape(len(batch), -1)
                        limits = calibration[name]
                        ratios = np.maximum(value.max(axis=1)/limits["max"], value.min(axis=1)/limits["min"])
                        needed[name] = max(needed[name], float(ratios.max()))
                    del values
                references = np.concatenate(references)
                validation = dict(exact_needed_margin_overall=max(needed.values()),
                                  exact_needed_margin_by_act=dict(sorted(needed.items(), key=lambda item: -item[1])[:5]),
                                  grid=[], selected=None)
                report["validation"] = validation
                _write(path, report)
                for deg in (63, 127):
                    for margin in (1.4, 1.5, 1.75):
                        candidate = parse(plan(graph, calibration, deg=deg, margin=margin, refresh=args.refresh))
                        names = dict(zip((op.inputs[0] for op in candidate.ops if op.kind == "act"), act_names))
                        row = dict(deg=deg, margin=margin, diverged=False, images_done=0, agree=0,
                                   logit_rmse=None, max_u=0.0, max_u_act=None, seconds=0.0)
                        validation["grid"].append(row)

                        def observer(name, value, layout, cfg):
                            data = (value.value[..., layout.index_grid()] if isinstance(layout, MuxLayout)
                                    else value.value[..., :layout.size])
                            _, _, lo, hi = cfg
                            max_u = float(np.max(np.abs(data*(2/(hi-lo))-(hi+lo)/(hi-lo))))
                            if max_u > row["max_u"]:
                                row.update(max_u=max_u, max_u_act=names[name])

                        started = time.monotonic()
                        squared = 0.0
                        for start in range(0, len(validation_images), VAL_BATCH):
                            batch = validation_images[start:start+VAL_BATCH]
                            try:
                                logits = evaluate(batch, candidate=candidate, observer=observer)
                            except BackendError as exc:
                                row.update(diverged=True, error=f"{type(exc).__name__}: {exc}")
                            else:
                                reference = references[start:start+len(batch)]
                                row["agree"] += int(np.sum(logits.argmax(axis=1) == reference.argmax(axis=1)))
                                squared += float(np.sum((logits-reference)**2))
                                row["images_done"] = start+len(batch)
                                row["logit_rmse"] = float(np.sqrt(squared/(row["images_done"]*logits.shape[1])))
                            row["seconds"] = time.monotonic()-started
                            _write(path, report)
                            print(f"VALIDATE deg={deg} margin={margin} images={row['images_done']} "
                                  f"agree={row['agree']} diverged={row['diverged']} max_u={row['max_u']}", flush=True)
                            if row["diverged"]:
                                gc.collect()
                                break
                validation["selected"] = select(validation["grid"])
                args.deg, args.margin = validation["selected"]["deg"], validation["selected"]["margin"]
                report.update(deg=args.deg, margin=args.margin)
                del validation_images, references, candidate
                gc.collect()
            elif stage == "plan":
                from .intake.coverage import check as check_graph
                supported = check_graph(graph)
                if not supported['ok']:
                    report['graph_check'] = supported
                    raise RuntimeError('PT-Graph contains operations without lowering; see graph_check')
                if args.planner == "agent":
                    from .llm import LLMClient
                    from .planners.agent import run_agent
                    from .planners.agent_tools import build_context

                    report['tune_policy'] = getattr(args, 'tune_policy', 'new_best')
                    ctx = build_context(info, calibration, samples=samples,
                                        normalization=getattr(zm, 'normalization', None))
                    client = LLMClient(args.budget_path, model=args.llm_model, reasoning_effort=args.llm_reasoning,
                                       max_tokens=args.llm_max_tokens,
                                       max_total_calls=getattr(args, 'llm_call_limit', 300))
                    from .planners.prompts import (PROMPT_VERSION, build_system_prompt, initial_message,
                                                   select_examples, target_features)
                    from .intake.patterns import annotate
                    report['prompt_version'] = PROMPT_VERSION
                    report['selected_examples'] = select_examples(target_features(graph, annotate(graph)), ctx.model_name)
                    (out/'system_prompt.txt').write_text(build_system_prompt(ctx))
                    (out/'initial_message.json').write_text(initial_message(ctx))
                    result = run_agent(ctx, client, max_iters=args.agent_iters, trace_path=out / "agent_trace.jsonl",
                                       tuner=getattr(args, 'tuner', 'on') == 'on',
                                       tuner_budget_s=getattr(args, 'tuner_budget_s', 2400),
                                       tune_policy=report['tune_policy'])
                    selected = result['confirm']['selected']
                    report["agent"] = dict(result, confirm=dict(result['confirm'],
                        selected=asdict(selected) if selected is not None else None))
                    report.update({key: result[key] for key in ("llm_calls", "prompt_tokens", "completion_tokens", "best_iteration")})
                    if result["best_iteration"] is not None:
                        report["latency_pred_us"] = result['best']['latency_pred_us']
                    report.update(best_source=result['best_source'], llm_source_lines=result['llm_source_lines'],
                                  frozen_source_lines=result['frozen_source_lines'])
                    _write(path, report)
                    if result["status"] != "feasible":
                        raise RuntimeError("Agent status: "+result['status'])
                    text = result["best_program"]
                    del ctx, client
                    gc.collect()
                else:
                    text = plan(graph, calibration, deg=args.deg, margin=args.margin, refresh=args.refresh,
                                act_method=args.act_method)
                (out / "program.ptdsl").write_text(text)
                report["source_lines"] = sum(bool(line.strip()) for line in text.splitlines())
            elif stage == "check":
                if graph is None:
                    raise RuntimeError("check requires the plan stage")
                program = parse((out / "program.ptdsl").read_text())
                act_tensors = [node.inputs[0] for node in graph.nodes if node.op == "act"]
                report["activations"] = [dict(graph_tensor=name, program_line=op.lineno,
                                               deg=op.attrs["deg"], domain=op.attrs["domain"])
                                          for name, op in zip(act_tensors, (op for op in program.ops if op.kind == "act"))]
                checked = check(program, {name: weight.shape for name, weight in graph.weights.items()})
                report["check"] = dict(ok=checked.ok, diags=[asdict(diag) for diag in checked.diags],
                                       n_statements=len(program.ops), source_lines=report["source_lines"])
                _write(out / "check.json", report["check"])
                if not checked.ok:
                    raise RuntimeError("PT-DSL typecheck failed")
            elif program is None:
                raise RuntimeError(f"{stage} requires plan and check stages")
            elif stage == "mock":
                started = time.monotonic()
                logits = evaluate(images[:8], exact=True)
                reference = reference_forward(graph, {graph.inputs[0]: images[:8]})[graph.outputs[0]]
                report["exact_max_abs_diff"] = float(np.max(np.abs(logits-reference)))
                if report["exact_max_abs_diff"] > 1e-9:
                    raise RuntimeError("exact lowering logits difference exceeded 1e-9")
                agree = correct = 0
                squared = 0.0
                all_approx, all_exact = [], []
                for start in range(0, len(images), VAL_BATCH):
                    batch = images[start:start+VAL_BATCH]
                    logits = evaluate(batch)
                    reference = reference_forward(graph, {graph.inputs[0]: batch})[graph.outputs[0]]
                    from .metrics import compare
                    all_approx.append(logits)
                    all_exact.append(reference)
                    measured = compare(dict(values=np.concatenate(all_exact), task=info.task),
                                       np.concatenate(all_approx), labels[:start+len(batch)] if labels is not None else None)
                    agree = measured.get('agree')
                    squared += float(np.sum((logits-reference)**2))
                    done = start+len(batch)
                    report.update(mock_images_done=done, agree=agree, top1=measured.get('top1'), mock_metrics=measured,
                                  logit_rmse=float(np.sqrt(squared/(done*logits.shape[1]))),
                                  mock_seconds=time.monotonic()-started)
                    _write(path, report)
                    print(f"MOCK images={done} agree={agree} top1={report['top1']}", flush=True)
                del logits, reference
                gc.collect()
            elif stage == "count":
                if args.refresh == "agent":
                    report["count"] = estimate(program, graph.weights)
                else:
                    backend = CountBackend(program.header.slots)
                    lower(program, graph.weights, backend)
                    report["count"] = backend.stats()
                    del backend
                report.update(counts=report["count"]["counts"], max_depth=report["count"]["max_depth"])
                gc.collect()
            elif stage == "emit":
                directory = toolchain.new_run_dir("r6-resnet20-emit")
                backend = EmitBackend(program.header.slots)
                config = PROFILES["qwen48"]
                active = (RefreshBackend(backend, **{key: config[key] for key in ("input_level", "boot_in_level", "boot_out_level")})
                          if args.refresh == "agent" else backend)
                lower(program, graph.weights, active)
                backend.write(directory)
                size = (directory / "constants.npz").stat().st_size
                report.update(constants_bytes=size, emit=dict(run_dir=str(directory), counts=backend.stats()["counts"],
                                                             constants_bytes=size))
                del backend, active
                gc.collect()
            elif stage == "export":
                from .export import export_package

                directory = toolchain.new_run_dir("structured-export")
                package = export_package((out/"program.ptdsl").read_text(), program, graph.weights, directory,
                    run_info=dict(program_path=out/"program.ptdsl", run_dir=out))
                report["export"] = dict(package=str(package), weights=str(directory/"weights"/(program.name+".npz")),
                    model_lines=len((package/program.name/"model.py").read_text().splitlines()),
                    files=sum(path.is_file() for path in package.rglob("*")))
                report["server_job"] = str(_server_job(out, package, Path(report["export"]["weights"])))
            elif stage == "compile":
                if report["export"] is None:
                    raise RuntimeError("compile requires the export stage")
                report.update(_compile_export(report["export"]))
                _write(path, report)
                if report["trace"]["exit"] != 0 or not report["trace"]["mlir"]:
                    raise RuntimeError("trusted trace failed")
                if report["compile"]["exit"] != 0 or not report["compile"]["plan"]:
                    raise RuntimeError("trusted compile failed")
                directory = Path(report["export"]["package"]).parent
                report["pruned"] = dict(trace=toolchain.prune(directory/"trace"),
                                        compiled=toolchain.prune(directory/"compiled"))
            elif stage == "test":
                if not args.final:
                    raise RuntimeError("test stage requires --final")
                report_test(evaluate, zm, (out/"program.ptdsl").read_text(), out, report)
            _write(path, report)
        report.update(complete=True, stage="complete")
    except TimeoutError as exc:
        report.update(complete=False, error=f"{type(exc).__name__}: {exc}")
        _write(path, report)
        return 124
    except Exception as exc:
        report.update(complete=False, error=f"{type(exc).__name__}: {exc}")
    _write(path, report)
    print("REPORT="+str(path), flush=True)
    return 0 if report["complete"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("resnet20", "resnet20_silu", "cnn_plain", "mlp"), required=True)
    parser.add_argument("--models-dir", type=Path)
    parser.add_argument("--planner", choices=("rule", "agent"), required=True)
    parser.add_argument("--agent-iters", type=int, default=6)
    parser.add_argument("--llm-model", default="deepseek-flash")
    parser.add_argument("--llm-reasoning", default="high")
    parser.add_argument("--llm-max-tokens", type=int, default=32768)
    parser.add_argument("--budget-path", type=Path)
    parser.add_argument("--tuner", choices=('on', 'off'), default='on')
    parser.add_argument("--tuner-budget-s", type=float, default=2400)
    parser.add_argument("--tune-policy", choices=('new_best', 'all'), default='new_best')
    parser.add_argument("--deg", type=int)
    parser.add_argument("--margin", type=float)
    parser.add_argument("--act-method", choices=("cheb", "minimax_comp"), default="cheb")
    parser.add_argument("--refresh", choices=("compiler", "agent"), default="compiler")
    parser.add_argument("--stages", required=True)
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--inside", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    args.stages = args.stages.split(",")
    if any(stage not in STAGES for stage in args.stages):
        parser.error("unknown stage")
    if "test" in args.stages and not args.final:
        parser.error("test stage requires --final")
    if "validate" in args.stages and (args.deg is not None or args.margin is not None):
        parser.error("validate requires --deg and --margin to be absent")
    if not 1 <= args.agent_iters <= 6:
        parser.error("--agent-iters must be 1..6")
    if args.tuner_budget_s < 0:
        parser.error('--tuner-budget-s must be nonnegative')
    if args.inside:
        if args.out is None:
            parser.error("--inside requires --out")
        if args.models_dir is None:
            parser.error("--inside requires --models-dir")
        if args.planner == "agent" and args.budget_path is None:
            parser.error("agent --inside requires --budget-path")
        return _inside(args)
    if not os.environ.get("PT_RESULTS"):
        raise RuntimeError("PT_RESULTS must be set in the outer environment")
    if args.planner == "agent" and not Path(os.environ["PT_RESULTS"]).is_dir():
        raise RuntimeError("PT_RESULTS must exist")
    args.models_dir = (args.models_dir or Path(os.environ["PT_RESULTS"])/"models").resolve()
    if "validate" not in args.stages and args.planner == "rule":
        deg, margin = _defaults()
        args.deg = deg if args.deg is None else args.deg
        args.margin = margin if args.margin is None else args.margin
        if args.margin < 1.1:
            raise ValueError("rule planner requires margin >= 1.1")
    out = Path(os.environ["PT_RESULTS"]) / ("run-"+args.model+"-"+args.planner+"-"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S"))
    out.mkdir(parents=True)
    scripts = Path(__file__).resolve().parent.parent
    command = [sys.executable, "-B", "-m", "pt_agent.tools.nixrun", "--seconds", "10800", "--", "-m",
               "pt_agent.run", "--inside", "--out", str(out), "--model", args.model, "--planner", args.planner,
               "--models-dir", str(args.models_dir),
               "--refresh", args.refresh, "--act-method", args.act_method, "--stages", ",".join(args.stages)]
    if args.planner == "agent":
        command.extend(("--budget-path", str(Path(os.environ["PT_RESULTS"]) / "llm_calls.jsonl"),
                        "--agent-iters", str(args.agent_iters), "--llm-model", args.llm_model,
                        "--tuner", args.tuner, '--tuner-budget-s', str(args.tuner_budget_s),
                        '--tune-policy', args.tune_policy,
                        "--llm-reasoning", args.llm_reasoning, "--llm-max-tokens", str(args.llm_max_tokens)))
    if args.deg is not None:
        command.extend(("--deg", str(args.deg)))
    if args.margin is not None:
        command.extend(("--margin", str(args.margin)))
    if args.final:
        command.append("--final")
    completed = subprocess.run(command, cwd=scripts, capture_output=True, text=True)
    sys.stdout.write(completed.stdout)
    sys.stdout.flush()
    sys.stderr.write(completed.stderr)
    sys.stderr.flush()
    from .registry import append, git_rev, new_run_id

    report = json.loads((out / "REPORT.json").read_text())
    round_name = "R7" if args.planner == "agent" else "R6"
    append(dict(run_id=new_run_id(round_name, args.model, args.planner), round=round_name,
                model=args.model, mode=args.planner,
                git_rev=git_rev(), metrics=report, artifacts_dir=str(out), ts=datetime.now(timezone.utc).isoformat()))
    return completed.returncode


if __name__ == "__main__":
    sys.exit(main())
