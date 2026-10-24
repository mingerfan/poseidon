"""Finite U2 acceptance; the outer process owns result paths and registry I/O."""

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import gc
import importlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import tarfile
import time
import types

from ..paths import (RESULTS_DIR, REPO, PLATFORM_DIR, DACAPO_ROOT, WORK_ROOT,
                     HECATE_BUILD)


BASE = '125676bd'
FROZEN_NAME = 'r8e7-reconfirm-20261022T122558'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def baseline_package(directory):
    name = 'pt_agent_u2_baseline'
    root = directory/'DSL/DSL_agent/scripts/pt_agent'
    package = types.ModuleType(name)
    package.__path__ = [str(root)]
    package.__package__ = name
    sys.modules[name] = package
    return lambda module: importlib.import_module(name+'.'+module)


def equal(left, right, message):
    if left != right:
        raise AssertionError(message)


def single_proof(args):
    import numpy as np
    from ..backends import EmitBackend
    from ..cost import estimate
    from ..dsl.parser import parse
    from ..lower import lower
    from ..models.zoo import get
    from ..planners import agent_tools, confirm
    from ..tests.test_p2_export import cases

    old = baseline_package(args.snapshot)
    old_lower, old_emit = old('lower'), old('backends.emit')
    old_parse = old('dsl.parser').parse
    old_cost, old_tools, old_confirm = old('cost'), old('planners.agent_tools'), old('planners.confirm')
    programs = cases()+[(name+'-frozen', name, args.reference/'models'/name/'program.ptdsl')
                        for name in ('resnet20', 'resnet20_silu', 'cnn_plain', 'mlp')]
    rows = []
    for label, name, path in programs:
        weights = get(name, args.models_dir).import_graph().weights
        source = path.read_text()
        before, after = old_emit.EmitBackend(16384), EmitBackend(16384)
        old_lower.lower(old_parse(source), weights, before)
        lower(parse(source), weights, after)
        before_text, before_consts = before.render()
        after_text, after_consts = after.render()
        equal(before_text, after_text, label+': Emit text changed')
        equal(list(before_consts), list(after_consts), label+': constant order changed')
        if any(not np.array_equal(before_consts[k], after_consts[k]) for k in before_consts):
            raise AssertionError(label+': Emit constants changed')
        row = dict(program=label, baseline_rev=BASE, emit_text_equal=True, constants_equal=True,
                   constants=len(after_consts), source=str(path))
        rows.append(row); write(args.out/'SINGLE_EMIT.json', rows)
        print('SINGLE Emit '+label+' identical', flush=True)
        del before, after, before_text, after_text, before_consts, after_consts, weights
        gc.collect()
    validations = []
    for name in ('resnet20', 'resnet20_silu', 'cnn_plain', 'mlp'):
        model = get(name, args.models_dir); graph = model.import_graph()
        source = (args.reference/'models'/name/'program.ptdsl').read_text()
        calibration = json.loads((args.reference/'models'/name/'calibration.json').read_text())
        previous = old_tools.build_context(graph, calibration, n_val=20, normalization=model.normalization)
        current = agent_tools.build_context(graph, calibration, n_val=20, normalization=model.normalization)
        a, b = old_parse(source), parse(source)
        cost_a, cost_b = old_cost.estimate(a, graph.weights), estimate(b, graph.weights)
        equal(cost_a, cost_b, name+': estimate changed')
        measured = {}
        for engine in ('mock', 'tensor'):
            first = old_tools.validate(a, previous, engine=engine, cache=False)
            second = agent_tools.validate(b, current, engine=engine, cache=False)
            equal(first, second, name+': '+engine+' validation changed')
            measured[engine] = second
            print('SINGLE '+name+' '+engine+' validation identical', flush=True)
        # Confirm the same one frozen candidate on the first 20 confirmation
        # images. This checks confirmation behavior without a new selection run.
        first = old_confirm.build_confirm_context(previous)
        second = confirm.build_confirm_context(current)
        for ctx in (first, second):
            ctx.val_images = ctx.val_images[:20]
            ctx.val_labels = ctx.val_labels[:20]
            ctx.val_exact_logits = ctx.val_exact_logits[:20]
            if ctx.val_inputs is not None:
                ctx.val_inputs = {k: v[:20] for k, v in ctx.val_inputs.items()}
        def confirmed(module, ctx):
            candidate = module.Candidate(source, 'frozen', 0, cost_b['latency_pred_us'], cost_b['boots_pred'])
            result = module.confirm_select([candidate], ctx, max_evals=1)
            result['selected'] = asdict(result['selected']) if result['selected'] is not None else None
            return result
        equal(confirmed(old_confirm, first), confirmed(confirm, second), name+': confirmation changed')
        row = dict(model=name, heldout_images=20, confirmation_images=20,
                   validate_equal=dict(mock=True, tensor=True), estimate_equal=True, confirm_equal=True,
                   validations=measured, estimate=cost_b)
        validations.append(row); write(args.out/'SINGLE_MODELS.json', validations)
        print('SINGLE '+name+' cost/confirm identical', flush=True)
        del previous, current, first, second, graph, model
        gc.collect(); old('tensor_sim').clear_caches(); importlib.import_module('pt_agent.tensor_sim').clear_caches()
    return dict(programs=rows, models=validations)


def compile_case(case, record, package, weights_path, name):
    """Record the count comparison before the user-approved intermediate cleanup."""
    from .. import toolchain
    from ..cost import plan_latency

    toolchain.setup_env()
    trace_dir, compile_dir = case/'trace', case/'compiled'
    py = PLATFORM_DIR/'venvs/hecate-2.0.1-cpu/bin/python'
    trace = toolchain.trusted_run(case, [str(py), '-B', str(package/'trace_model.py'), '--weights',
        str(weights_path), '--output-dir', str(trace_dir), '--bytecode'],
        case/'trace.log', env_extra=toolchain.package_trace_env())
    record['trace'] = trace; write(case/'REPORT.json', record)
    if trace['exit'] != 0:
        raise AssertionError(f'{name}/{record["slots"]}: package trace failed')
    compiled = toolchain.trusted_run(case, [str(py), '-B', str(package/'compile_model.py'),
        '--hecate-opt', str(toolchain.host_paths()['hecate_opt']), '--input', str(trace_dir/'trace_model.mlirbc'),
        '--output-dir', str(compile_dir)], case/'compile.log',
        env_extra=dict(LD_LIBRARY_PATH=os.environ['HECATE_PYTHON_LIBRARY_PATH']))
    record['compile'] = compiled; write(case/'REPORT.json', record)
    if compiled['exit'] != 0:
        raise AssertionError(f'{name}/{record["slots"]}: package compilation failed')
    plan_path = compile_dir/f'lowered._hecate_{name}.runtime-plan.json'
    plan = plan_latency(plan_path)
    record.update(plan_latency=plan, boots_compiled=plan['boots'], runtime_plan=str(plan_path),
                  latency_error_pct=100*abs(record['latency_pred_us']-plan['latency_us'])/plan['latency_us'])
    write(case/'REPORT.json', record)
    equal(record['boots_pred'], plan['boots'], f'{name}/{record["slots"]}: bootstrap count differs')
    record['complete'] = True; write(case/'REPORT.json', record)
    record['pruned'] = toolchain.prune(compile_dir)
    write(case/'REPORT.json', record)


def model_acceptance(args):
    import numpy as np
    from ..backends import CountBackend, EmitBackend
    from ..cost import estimate, plan_latency
    from ..data.cifar import heldout_set
    from ..dsl.parser import parse
    from ..export import export_package
    from ..kernels.layout import layout_blocks, layout_record
    from ..lower import lower
    from ..models.zoo import get
    from ..tensor_sim import run_tensor, _metadata, clear_caches
    from ..tests.test_p2_export import emit_pair, package_modules, mock_reference
    from .. import toolchain

    rows = []
    names = args.only.split(',') if args.only else ('resnet20', 'cnn_plain', 'mlp')
    if args.resume_from is not None:
        for name in names:
            for slots in (8192, 4096):
                source_case = args.resume_from/(name+'-'+str(slots))
                path = source_case/'REPORT.json'
                if not path.exists():
                    continue
                record = json.loads(path.read_text())
                record.update(source_report=str(path), reused=True)
                if not record.get('complete'):
                    if (not all(record.get(key) for key in ('numeric_equal', 'classes_equal',
                            'package_numpy_exact', 'package_emit_exact')) or
                            any(record.get(key) is not None for key in ('trace', 'compile'))):
                        raise ValueError('Resume requires completed numeric/package checks and an unstarted trace')
                    case = args.out/(name+'-'+str(slots)); case.mkdir()
                    program = parse((source_case/'program.ptdsl').read_text())
                    packages = list((source_case/'export').glob('*/config/model.json'))
                    equal(len(packages), 1, 'Resume package must be unique')
                    package = packages[0].parents[1]
                    compile_case(case, record, package,
                                 source_case/'export/weights'/(program.name+'.npz'), program.name)
                    record['artifacts_dir'] = str(case)
                    write(case/'REPORT.json', record)
                else:
                    record['artifacts_dir'] = str(source_case)
                rows.append(record); write(args.out/'MODELS.json', rows)
                print('RESUME '+name+'/'+str(slots)+' complete', flush=True)
    completed = {(row['model'], row['slots']) for row in rows}
    for name in names:
        if all((name, slots) in completed for slots in (8192, 4096)):
            continue
        model = get(name, args.models_dir); graph = model.import_graph(); weights = graph.weights
        images = model.adapt(heldout_set(20, normalization=model.normalization)[0])
        original = (args.reference/'models'/name/'program.ptdsl').read_text()
        baseline_program = parse(original)
        equal(baseline_program.header.slots, 16384, 'Expected a slots=16384 frozen baseline')
        original_logits = mock_reference(baseline_program, weights, images)
        original_cost = estimate(baseline_program, weights)
        baseline_report = json.loads((args.reference/'models'/name/'REPORT.json').read_text())
        original_plan = baseline_report.get('plan_latency')
        baseline = dict(slots=16384, input_ciphertexts=1, output_ciphertexts=1,
                        boots_pred=original_cost['boots_pred'], rotations=original_cost['counts']['rot'],
                        latency_pred_us=original_cost['latency_pred_us'], plan_latency=original_plan)
        for slots in (8192, 4096):
            if (name, slots) in completed:
                continue
            case = args.out/(name+'-'+str(slots)); case.mkdir()
            text = original.replace('slots=16384', 'slots='+str(slots), 1)
            equal(text.replace('slots='+str(slots), 'slots=16384', 1), original, 'Changed more than slots')
            (case/'program.ptdsl').write_text(text)
            p = parse(text)
            actual = mock_reference(p, weights, images)
            tensor = run_tensor(p, weights, images, cache=False)
            diff = np.abs(actual-tensor)
            numeric = bool(np.all(diff <= 1e-9*(1+np.abs(tensor))))
            categories = bool(np.array_equal(actual.argmax(axis=1), original_logits.argmax(axis=1)))
            np.savez_compressed(case/'numeric.npz', mock=actual, tensor=tensor, baseline=original_logits)
            record = dict(model=name, slots=slots, images=20, numeric_equal=numeric,
                          max_abs_diff=float(diff.max()), max_normalized_diff=float((diff/(1+np.abs(tensor))).max()),
                          classes_equal=categories, baseline=baseline)
            write(case/'REPORT.json', record)
            if not numeric or not categories:
                raise AssertionError(f'{name}/{slots}: forced multi numeric or class mismatch')
            counter = CountBackend(slots, boot_out_level=5)
            values, output_layouts = lower(p, weights, counter)
            nodes = _metadata(p, weights)[0]
            layouts = [dict(op_id=op.id, kind=op.kind, ciphertexts=len(layout_blocks(lay)), layout=layout_record(lay))
                       for op, _, lay, _ in nodes if hasattr(lay, 'n')]
            maximum = max((item['ciphertexts'] for item in layouts), default=1)
            cost = estimate(p, weights)
            record.update(input_ciphertexts=len(counter._inputs), output_ciphertexts=len(values),
                          max_tensor_ciphertexts=maximum, layouts=layouts, estimate=cost,
                          boots_pred=cost['boots_pred'], rotations=cost['counts']['rot'],
                          latency_pred_us=cost['latency_pred_us'])
            package = export_package(text, p, weights, case/'export')
            with package_modules(package) as runtime:
                numpy_result = runtime.run_numpy(weights, [images])[0]
            record['package_numpy_exact'] = bool(np.array_equal(actual, numpy_result))
            if not record['package_numpy_exact']:
                raise AssertionError(f'{name}/{slots}: P2 NumPy differs from Mock')
            (a, ac), (b, bc) = emit_pair(text, weights, package)
            record['package_emit_exact'] = a == b and list(ac) == list(bc) and all(np.array_equal(ac[k], bc[k]) for k in ac)
            del a, b, ac, bc, values, counter
            gc.collect()
            if not record['package_emit_exact']:
                raise AssertionError(f'{name}/{slots}: P2 Emit differs from lowering')
            write(case/'REPORT.json', record)
            compile_case(case, record, package, case/'export/weights'/(p.name+'.npz'), p.name)
            record['artifacts_dir'] = str(case)
            write(case/'REPORT.json', record)
            rows.append(record); write(args.out/'MODELS.json', rows)
            print('MULTI '+json.dumps({k: record[k] for k in ('model','slots','input_ciphertexts','max_tensor_ciphertexts',
                  'boots_pred','boots_compiled','rotations','latency_pred_us','latency_error_pct')}), flush=True)
            clear_caches(); gc.collect()
    return dict(models=rows, latency_error_median_pct=float(np.median([r['latency_error_pct'] for r in rows])),
                latency_error_max_pct=max(r['latency_error_pct'] for r in rows))


def tests(args):
    import unittest
    if args.part == 'intake-tests':
        intake = PLATFORM_DIR/'venvs/intake-2.0.1-cpu/bin/python'
        if Path(sys.executable).absolute() != intake.absolute():
            os.execv(str(intake), [str(intake), '-B', '-m', 'pt_agent.experiments.u2_multi_layout', *sys.argv[1:]])
        suite = unittest.defaultTestLoader.loadTestsFromName('pt_agent.tests.test_u1_onnx')
    elif args.part == 'unit':
        suite = unittest.defaultTestLoader.discover('pt_agent/tests', pattern='test_u2_*.py')
    else:
        packaging = {'test_p2_export.py', 'test_toolchain_decoupled.py'}
        tensor = {'test_r8f_tensor_validate.py'}
        files = [p for p in sorted(Path('pt_agent/tests').glob('test_*.py')) if not p.name.startswith('test_u2_')]
        files = [p for p in files if (p.name in packaging if args.only == 'packaging' else
                 p.name in tensor if args.only == 'tensor' else p.name not in packaging | tensor)]
        suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName('pt_agent.tests.'+p.stem) for p in files)
        equal(suite.countTestCases(), {'main':385, 'packaging':17, 'tensor':17}[args.only], '419-test inventory changed')
    result = unittest.TextTestRunner(verbosity=2, failfast=True).run(suite)
    record = dict(tests=result.testsRun, failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped))
    write(args.out/'TESTS.json', record)
    if not result.wasSuccessful():
        raise AssertionError('U2 acceptance tests failed; see run.log')
    return record


def inside(args):
    result = dict(complete=False, part=args.part, report_images_read=0, llm_calls=0)
    try:
        import torch
        torch.set_num_threads(1)
        if args.part == 'single':
            result.update(single_proof(args))
        elif args.part == 'models':
            result.update(model_acceptance(args))
        else:
            result.update(tests(args))
        result['complete'] = True
    except Exception as exc:
        import traceback
        traceback.print_exc()
        result['error'] = f'{type(exc).__name__}: {exc}'
    write(args.out/'REPORT.json', result)
    return 0 if result['complete'] else 1


def limits():
    resource.setrlimit(resource.RLIMIT_AS, (4*1024**3, 6*1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (8*1024**3, 8*1024**3))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--part', choices=('unit','single','models','regression','intake-tests'), required=True)
    parser.add_argument('--only')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--models-dir', type=Path)
    parser.add_argument('--snapshot', type=Path)
    parser.add_argument('--resume-from', type=Path)
    parser.add_argument('--inside', action='store_true')
    args = parser.parse_args()
    if args.inside:
        return inside(args)
    args.out = (args.out or RESULTS_DIR/('u2-'+args.part+'-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f'))).resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    args.reference = (args.reference or RESULTS_DIR/FROZEN_NAME).resolve()
    args.models_dir = (args.models_dir or RESULTS_DIR/'models-v2').resolve()
    env = dict(os.environ, POSEIDON_WORK_ROOT=str(WORK_ROOT), PT_DACAPO_ROOT=str(DACAPO_ROOT),
               PT_HECATE_BUILD=str(HECATE_BUILD), PT_AGENT_RESULTS_ROOT=str(RESULTS_DIR))
    if args.part == 'single':
        args.snapshot = args.out/'baseline-tree'
        archive = args.out/'baseline.tar'
        with archive.open('wb') as stream:
            subprocess.run(['git', 'archive', BASE, 'DSL/DSL_agent/scripts/pt_agent'], cwd=REPO, stdout=stream, check=True)
        args.snapshot.mkdir()
        with tarfile.open(archive) as stream:
            stream.extractall(args.snapshot)
    command = [sys.executable, '-B', '-m', 'pt_agent.tools.nixrun', '--seconds', '3600', '--',
        '-m', 'pt_agent.experiments.u2_multi_layout', '--inside', '--part', args.part,
        '--out', str(args.out), '--reference', str(args.reference), '--models-dir', str(args.models_dir)]
    if args.snapshot is not None:
        command += ['--snapshot', str(args.snapshot)]
    if args.only:
        command += ['--only', args.only]
    if args.resume_from is not None:
        command += ['--resume-from', str(args.resume_from.resolve())]
    write(args.out/'argv.json', command)
    print('RUN='+str(args.out), flush=True)
    begin = time.monotonic()
    with (args.out/'run.log').open('w') as log:
        status = subprocess.run(command, cwd=Path(__file__).resolve().parents[2], env=env, stdout=log,
                                stderr=subprocess.STDOUT, preexec_fn=limits, timeout=3720).returncode
    path = args.out/'REPORT.json'
    record = json.loads(path.read_text()) if path.exists() else dict(complete=False)
    record.update(exit=status, seconds=time.monotonic()-begin)
    write(path, record)
    from ..registry import append, new_run_id, git_rev
    os.environ.setdefault('PT_RESULTS', str(RESULTS_DIR))
    append(dict(run_id=new_run_id('U2', args.only or args.part, 'acceptance'), round='U2',
                model=args.only or args.part, mode='acceptance', git_rev=git_rev(), metrics=record,
                artifacts_dir=str(args.out), ts=datetime.now(timezone.utc).isoformat()))
    print('REPORT='+str(path), flush=True)
    return status


if __name__ == '__main__':
    raise SystemExit(main())
