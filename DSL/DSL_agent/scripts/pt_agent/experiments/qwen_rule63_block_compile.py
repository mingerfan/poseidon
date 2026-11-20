"""Independent original Qwen layer0 Rule63/R0 compiler baseline.

Only SiLU activation degree changes from the trusted R0 generator's127 to63.
The original R0 norm/softmax selectors and the compiler's placement stay intact.
No Agent session, C2, generation head or explicit layer-boundary boot is used.
"""
import argparse
import ast
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
import gc
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sys
import threading
import time

from .r10_paper_metrics import StageRecorder, read_events, run_process, summarize_events, write_json
from .completion_metrics import artifact_bytes, persist_case_completion
from .cold_rule63_whole_model import append_registry, classify, clean
from .a_group_profile_sum_policy import profile_sum

ROUND = 'QWEN-RULE63-R0-BLOCK'
MODULE = 'pt_agent.experiments.qwen_rule63_block_compile'
PHASES = ('prepare', 'emit', 'trace', 'compile')
GIB = 1024**3
EXPECTED_OPS = dict(rmsnorm=2, linear=7, mul=5, shuffle=2, add=4, matmul=2, softmax=1, act=1)
SOURCE_OPTIONS = ('model_info', 'calibration', 'calibration_result', 'reference', 'inputs',
                  'source_rows', 'profile', 'frontend', 'metadata_template')


def read(path):
    return json.loads(Path(path).read_text())


def require(value, message):
    if not value:
        raise ValueError(message)


def slice_layer0(graph):
    """Select original graph objects and original arrays; change no arithmetic."""
    nodes = graph.nodes[:24]
    require(len(graph.nodes) == 579 and len(nodes) == 24
            and dict(Counter(node.op for node in nodes)) == EXPECTED_OPS,
            'Retain the original normalized579-node Qwen graph and complete24-node first block')
    require(nodes[0].attrs.get('weight') == 'original.model.layers.0.input_layernorm.weight'
            and graph.nodes[24].attrs.get('weight') == 'original.model.layers.1.input_layernorm.weight'
            and nodes[-1].op == 'add' and nodes[-1].outputs == ['add_7'],
            'Original layer0 boundaries differ')
    produced = {name for node in nodes for name in node.outputs}
    external = {name for node in nodes for name in node.inputs} - produced
    require(external == {'inputs_embeds', 'attention_mask'}, 'Original block external inputs differ')
    require(tuple(graph.tensors['inputs_embeds'].shape) == (1, 32, 896)
            and tuple(graph.tensors['attention_mask'].shape) == (1, 32)
            and tuple(graph.tensors['add_7'].shape) == (1, 32, 896), 'Original32-token block shape differs')
    refs = {value for node in nodes for key, value in node.attrs.items()
            if key in ('weight', 'bias', 'const') and value is not None}
    require(len(refs) == 16 and all(name in graph.weights for name in refs)
            and all(name.startswith(('original.model.layers.0.', 'u4.public.')) for name in refs),
            'Original first-layer weights or rotary constants differ')
    used = external | produced
    block = replace(graph, name='qwen2_original_layer0', nodes=list(nodes),
        tensors={name: graph.tensors[name] for name in used},
        weights={name: graph.weights[name] for name in sorted(refs)},
        inputs=['inputs_embeds', 'attention_mask'], outputs=['add_7'])
    block.intake_stats = dict(getattr(graph, 'intake_stats', {}), transformer_normalized=True)
    return block


def rule63_source(text):
    """Mirror the BERT act-only precedent, requiring the original single SiLU."""
    tree = ast.parse(text)
    changes = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id != 'act':
            continue
        fields = {item.arg: item for item in node.keywords}
        require(set(('fn', 'method', 'deg', 'domain')) <= set(fields)
                and ast.literal_eval(fields['fn'].value) == 'silu'
                and ast.literal_eval(fields['method'].value) == 'cheb'
                and ast.literal_eval(fields['deg'].value) == 127, 'Expected original R0 SiLU127')
        changes.append(dict(line=node.lineno, fn='silu', original_degree=127, degree=63,
                            domain=ast.literal_eval(fields['domain'].value)))
        fields['deg'].value = ast.Constant(value=63)
    require(len(changes) == 1, 'One original layer0 SiLU is required')
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + '\n', changes


def source_gate(args):
    info = read(args.model_info / 'model_info.json')
    config = info['source']['config']
    require(config.get('model_type') == 'qwen2' and config.get('num_hidden_layers') == 24
            and config.get('hidden_size') == 896 and config.get('vocab_size') == 151936,
            'Original trained Qwen2-0.5B source required')
    require([(row['name'], row['shape']) for row in info['inputs']] ==
            [('inputs_embeds', [1, 32, 896]), ('attention_mask', [1, 32]), ('sel', [1, 32])],
            'Original32-token ModelInfo signature required')
    proof = read(args.calibration_result)
    original = proof['source']
    require(proof.get('status') == 'ok' and proof.get('phase') == 'calibration'
            and proof.get('model') == 'Qwen2-0.5B' and proof.get('calibration_forward_completed') == 23
            and proof.get('calibration_samples') == 23 and proof.get('report_samples') == 0,
            'Completed original23-training-prefix calibration required')
    require(Path(original['model_info']).resolve() == args.model_info / 'model_info.json'
            and Path(original['samples']).resolve() == args.inputs
            and Path(original['source_rows']).resolve() == args.source_rows
            and proof.get('sample_accesses') == [dict(partition='calibration',
                indices=info['split']['calibration'], count=23)], 'Original training calibration source differs')
    require(args.calibration == args.calibration_result.parents[2] / 'calibration.json',
            'Calibration file must belong to its original successful phase')
    rows = read(args.source_rows)
    require(Path(rows['source']).name == 'train-00000-of-00001.parquet'
            and rows.get('column') == 'sentence' and rows.get('source_rows') == list(range(223)),
            'Original unlabelled SST2 training sentences required')
    reference = read(args.reference)
    identity = reference.get('original_model_identity') or {}
    require(reference.get('status') == 'ok' and reference.get('hf_forward_completed') == 6
            and Path(reference.get('original_hf_source', '')).resolve() == Path(info['source']['local_path']).resolve()
            and Path(reference.get('training_source', '')).resolve() == Path(rows['source']).resolve()
            and all(identity.get(key) == value for key, value in
                (('layers', 24), ('hidden_size', 896), ('vocabulary', 151936), ('sequence_length', 32))),
            'Saved real-training prefix metadata must retain the original trained model')
    profile = read(args.profile)
    expected = dict(polynomialDegree=65536, rescalingFactor=43, levelLowerBound=1,
                    levelUpperBound=8, bootstrapLevelLowerBound=1, bootstrapLevelUpperBound=5,
                    runtime='HEAAN-HEVM')
    require(all(profile.get(key) == value for key, value in expected.items()), 'Original qwen48 contract differs')
    for path in (args.inputs, args.frontend, args.metadata_template,
                 args.model_info / info['graph'] / 'weights.npz'):
        require(path.is_file(), 'Original source artifact missing: ' + str(path))
    return dict(original_model=info['source'], model_info=str(args.model_info),
        calibration=str(args.calibration), calibration_phase=str(args.calibration_result),
        calibration_indices=info['split']['calibration'], real_sequence_capacity=32,
        training_text=str(args.inputs), profile=str(args.profile), parameters=expected,
        C2_required=False, Agent_source_used=False, bridge_policy_used=False,
        recipe='Default transformer_rule R0; singleSiLU127to63 only',
        original_calibration_reused=True, new_quality_or_model_reference_claimed=False)


def load_block(args):
    from ..models.mapped_info import load_mapped_info
    info = load_mapped_info(args.model_info)
    require(len(info.graph.weights) == 559, 'Retain the original559 mapped members')
    return info, slice_layer0(info.graph)


def program_inputs(args):
    from ..dsl.parser import parse
    from ..dsl.typecheck import check
    info, graph = load_block(args)
    program = parse((args.out / 'work/program.ptdsl').read_text())
    checked = check(program, {key: value.shape for key, value in graph.weights.items()},
                    graph=graph, prompt_version='r8-v11')
    require(checked.ok and program.header.refresh == 'compiler'
            and program.header.placement == 'depth-dp' and program.header.slots == 16384,
            'Independent Rule63 block type/header check failed')
    require([p.name for p in program.params] == graph.inputs
            and len(program.outputs) == 1 and checked.shapes[program.outputs[0]] == (1, 32, 896),
            'Original complete block interface differs')
    return info, graph, program


def strict_training_math(args, info, graph, program):
    import numpy as np
    from ..backends.streaming_mock import StreamingMockBackend
    from ..lower import lower, pack_inputs, decode_outputs
    from ..tensor_sim import run_tensor
    from ..kernels.compact_linear import compact_transformer_projections
    reference = read(args.reference)
    require(reference.get('status') == 'ok' and reference.get('hf_forward_completed') == 6,
            'Original real-training HF prefix metadata required')
    trajectory = reference['trajectories'][0]
    require(trajectory['sample_id'] == 'train:0' and 0 in info.split['calibration'],
            'Use original train0 from the retained calibration split')
    row = trajectory['steps'][0]
    ids = np.asarray(row['prefix_token_ids'], dtype=np.int64)
    mask = np.asarray(row['attention_mask'], dtype=np.float64)
    require(ids.shape == (32,) and mask.shape == (1, 32), 'Original padded32-token input differs')
    embedding = info.graph.weights['original.model.embed_tokens.weight']
    arrays = dict(inputs_embeds=np.asarray(embedding[ids], dtype=np.float64)[None], attention_mask=mask)
    tensor = run_tensor(program, graph.weights, arrays, cache=False, act_mode='program')
    backend = StreamingMockBackend(16384, batch=1)
    try:
        backend.bind(pack_inputs(program, [arrays[p.name] for p in program.params]))
        with compact_transformer_projections():
            values, layouts = lower(program, graph.weights, backend, act_mode='program')
        packed = decode_outputs([v.value for v in values], layouts)[0]
    finally:
        backend.release_batch()
    require(tensor.shape == packed.shape == (1, 32, 896)
            and np.isfinite(tensor).all() and np.isfinite(packed).all(), 'Nonfinite or incorrect block math output')
    error = np.abs(tensor - packed)
    ratio = error / (1e-12 + 1e-12 * np.abs(tensor))
    observation = dict(sample_id='train:0', sequence_capacity=32, active_tokens=int(mask.sum()),
        shape=list(tensor.shape), max_abs_error=float(error.max()), max_tolerance_ratio=float(ratio.max()),
        criterion='1e-12+1e-12*abs(reference)', passed=bool((ratio <= 1).all()),
        scope='Same fixed approximate block Tensor versus packed arithmetic; task accuracy unmeasured')
    write_json(args.out / 'work/training-math.json', observation)
    require(observation['passed'], 'Strict original training Tensor/Mock mathematics failed')
    return observation


def worker(args):
    result = dict(status='running', error=None, phase=args.worker_phase,
                  models_freshly_imported=0, API_calls=0, confirm_samples=0, report_samples=0, CKKS_runs=0)
    recorder = StageRecorder(args.work_dir / 'events.jsonl')
    try:
        if args.worker_phase == 'prepare':
            from ..planners.transformer_rule import plan
            from ..kernels.compact_linear import compact_transformer_projections
            from ..backends.streaming_count import StreamingCountBackend
            from ..lower import lower
            result['source'] = source_gate(args)
            info, graph = load_block(args)
            with recorder.span('default_R0_block_rule_generation'):
                text = plan(graph, read(args.calibration), slots=16384, placement='depth-dp')
                text, changes = rule63_source(text)
                (args.out / 'work/program.ptdsl').write_text(text)
                result['activation_degree_changes'] = changes
                result['program_ast'] = ast.dump(ast.parse(text), include_attributes=False)
            del info, graph
            gc.collect()
            info, graph, program = program_inputs(args)
            with recorder.span('one_real_training_strict_mathematics'):
                result['training_math'] = strict_training_math(args, info, graph, program)
            with recorder.span('raw_unplaced_Count'):
                backend = StreamingCountBackend(16384)
                with compact_transformer_projections():
                    lower(program, graph.weights, backend, act_mode='program')
                result['raw_count'] = backend.stats()
                require(result['raw_count']['counts']['boot'] == 0, 'Compiler owns every Rule63 bootstrap')
            from .a1_bert_fixed_current import recipe_metadata
            result['recipe'] = recipe_metadata(program)
        elif args.worker_phase == 'emit':
            from ..backends.streaming_emit import StreamingEmitBackend
            from ..kernels.compact_linear import compact_transformer_projections
            from ..lower import lower
            info, graph, program = program_inputs(args)
            prepared = read(args.out / 'work/prepare/phase_result.json')
            require(prepared.get('status') == 'ok' and prepared.get('training_math', {}).get('passed') is True,
                    'Successful R0 Rule63 preparation/math required')
            require(prepared.get('program_ast') == ast.dump(ast.parse(
                (args.out / 'work/program.ptdsl').read_text()), include_attributes=False),
                'Prepared R0 Rule63 program semantics changed')
            backend = StreamingEmitBackend(16384, directory=args.work_dir, frontend=args.frontend,
                max_constant_bytes=(args.case_gib - 2) * GIB)
            try:
                with compact_transformer_projections():
                    lower(program, graph.weights, backend, act_mode='program')
                stats = backend.stats()
                require(all(stats[key] == prepared['raw_count'][key]
                            for key in ('counts', 'ops_total', 'distinct_consts'))
                        and stats['counts']['boot'] == 0, 'Unplaced block Count/Emit statistics differ')
                backend.write()
                result['emit_stats'] = stats
            finally:
                if not backend.finished:
                    backend.close()
        else:
            from .. import toolchain
            from .r29_local_compile_matrix import compiler_argv, plan_observation
            from .r30_common_model_e2e import diagnostic_compile_argv, compile_artifact_metadata
            host = toolchain.host_paths()
            emit = args.out / 'work/emit'
            env = dict(os.environ, HECATE=str(host['hecate']), HECATE_BUILD=str(host['hecate_build']),
                       LD_LIBRARY_PATH=os.environ['HECATE_PYTHON_LIBRARY_PATH'])
            if args.worker_phase == 'trace':
                argv = [sys.executable, '-B', str(emit / 'candidate_trace.py')]
            else:
                argv = compiler_argv(read(args.metadata_template), 'dacapo-depth-dp', binary=host['hecate_opt'],
                    source=emit / 'candidate_trace.mlir', profile=args.profile,
                    output=emit / 'lowered.mlir', debug_snapshots=False)
                argv = diagnostic_compile_argv(argv, elide_final_ir=True)
            write_json(args.work_dir / 'native-argv.json', argv)
            observation = run_process(argv, cwd=emit, log=args.work_dir / 'native.log',
                resources=args.work_dir / 'native-resources.jsonl', timeout=args.phase_seconds,
                mem_gib=24, fsize_gib=16, env=env)
            result['native_process'] = observation
            require(clean(observation), 'Original native ' + args.worker_phase + ' failed')
            if args.worker_phase == 'trace':
                require((emit / 'candidate_trace.mlir').is_file(), 'Native trace did not produce raw MLIR')
            else:
                plans = list(emit.glob('lowered.*.runtime-plan.json'))
                require(len(plans) == 1, 'One actual compiled RuntimePlan required')
                plan = read(plans[0])
                result.update(plan_observation(plan, read(args.profile)))
                result['runtime_plan'] = str(plans[0])
                result['compiler_artifacts'] = compile_artifact_metadata(emit, plans[0], plan, elided=True)
        result['status'] = 'ok'
    except BaseException as exc:
        result.update(status='error', error=type(exc).__name__ + ': ' + str(exc)[:3000])
        if hasattr(exc, 'report') and hasattr(exc, 'key'):
            result['approximation_failure'] = dict(key=exc.key, report=exc.report,
                issues=getattr(exc, 'issues', []), unchanged_R0_recipe=True)
    result['timing'] = summarize_events(read_events(recorder.path))
    write_json(args.work_dir / 'phase_result.json', result)
    return result


def phase_command(args, phase, remaining):
    from .. import paths
    seconds = min(7200 if phase == 'compile' else 1800, int(remaining) - 180)
    require(seconds > 0, 'Finite block allowance exhausted before ' + phase)
    inner = ['-B', '-m', MODULE, '--worker-phase', phase, '--out', str(args.out),
             '--work-dir', str(args.out / 'work' / phase), '--phase-seconds', str(seconds),
             '--case-gib', str(args.case_gib)]
    for name in SOURCE_OPTIONS:
        inner += ['--' + name.replace('_', '-'), str(getattr(args, name))]
    return [sys.executable, '-B', '-m', 'pt_agent.tools.nixrun', '--seconds', str(seconds + 60),
            '--python', str(paths.INTAKE_PYTHON), '--', *inner], seconds + 120


def paper_record(args, result, completion):
    from .a_group_runs import empty_record
    compiled = result['phases'].get('compile') or {}
    native = compiled.get('native_process') or {}
    policy = profile_sum(compiled.get('prediction'))
    spans = {row['stage_id']: row for row in result['timing']['spans']}
    stages = {phase: spans[phase]['inclusive_ns'] / 1e9 if phase in spans else None for phase in PHASES}
    used = sum(value for value in stages.values() if value is not None)
    require(used <= completion['complete_case_elapsed_seconds'], 'Outer block stages exceed complete case clock')
    stages['other'] = completion['complete_case_elapsed_seconds'] - used
    return empty_record(run_id=args.out.name, task='Qwen-Rule63-block', model='qwen2_0_5b_block',
        system='fixed_rule_depth', param_group='ours-N16-boot5', status=classify(result),
        status_detail=result.get('error'), budget_s=7200,
        compile_s=(native['ended_ns'] - native['started_ns']) / 1e9 if native.get('ended_ns') is not None else None,
        peak_rss_gib=result.get('peak_rss_gib'), boots_compiled=compiled.get('compiled_bootstraps'),
        exec_ops=compiled.get('execution_steps'), pred_latency_s=policy['pred_latency_s'],
        pred_table=str(args.profile), generation_e2e_s=None, ckks_inference_s=None,
        source=str(args.out / 'result.json'), runtime_plan=compiled.get('runtime_plan'),
        prediction_policy=policy, original_calibration_reused=True, candidate_frozen=False,
        stages_s=stages, finished_utc=datetime.now(timezone.utc).isoformat(),
        cold=False, full_model=False, layer=0, sequence_capacity=32, quality_retested=False,
        confirm_samples=0, report_samples=0, api_calls=0,
        isolated_pipeline_wall_s=completion['complete_case_elapsed_seconds'],
        pipeline_scope='Existing original model/calibration to one complete block artifact; independent of Agent/C2')


def execute(args):
    from .. import paths
    from ..registry import git_rev
    from .dacapo_compile_continuation import latest_resource, available_ram
    require(args.out.parent == paths.RESULTS_DIR.resolve() and not os.path.lexists(args.out),
            'Fresh direct results child required')
    args.out.mkdir(); (args.out / 'work').mkdir()
    started = time.perf_counter_ns(); recorder = StageRecorder(args.out / 'events.jsonl')
    result = dict(round=ROUND, run_id=args.out.name, model='Qwen2-0.5B-layer0', status='running',
        error=None, phases={}, processes=[], guard_error=None, monitor_error=None, monitor_stopped=False,
        API_calls=0, confirm_samples=0, report_samples=0, CKKS_runs=0, peak_rss_gib=None)
    stop = threading.Event(); state = dict(path=None, active=False); lock = threading.Lock()
    old = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM)}
    def interrupted(sig, frame):
        result['incoming_signal'] = sig
        raise InterruptedError('Rule63 block interrupted by signal ' + str(sig))
    def monitor():
        while not stop.wait(1):
            try:
                with lock: path = state['path']
                sample = latest_resource(path)
                rss = (sample or {}).get('subtree_plus_supervisor_rss_bytes', 0)
                result['peak_rss_gib'] = max(result['peak_rss_gib'] or 0, rss / GIB)
                if artifact_bytes(args.out) > args.case_gib * GIB or shutil.disk_usage(args.out).free < 10 * GIB \
                        or available_ram() < 1.25 * GIB or rss > 14 * GIB:
                    raise RuntimeError('resource_limit: RSS14GiB/available1.25GiB/case/free10GiB guard')
            except BaseException as exc:
                key = 'guard_error' if str(exc).startswith('resource_limit:') else 'monitor_error'
                result[key] = type(exc).__name__ + ': ' + str(exc)
                with lock:
                    if state['active']: os.kill(os.getpid(), signal.SIGTERM)
                return
    watcher = threading.Thread(target=monitor, daemon=True)
    try:
        for s in old: signal.signal(s, interrupted)
        signal.setitimer(signal.ITIMER_REAL, args.seconds); watcher.start()
        result['source'] = source_gate(args)
        require(available_ram() >= 1.25 * GIB and shutil.disk_usage(args.out).free >= 10 * GIB,
                'resource_limit: available1.25GiB/free10GiB required before first worker')
        source = args.out / 'source'; source.mkdir()
        for file in (Path(__file__), args.profile, args.metadata_template, args.calibration):
            shutil.copyfile(file, source / file.name)
        root = Path(__file__).parents[1]
        for relative in ('planners/transformer_rule.py', 'kernels/compact_linear.py',
                         'models/mapped_info.py', 'transformer_tensor.py', 'lower.py',
                         'backends/streaming_count.py', 'backends/streaming_emit.py',
                         'backends/streaming_mock.py'):
            shutil.copyfile(root / relative, source / relative.replace('/', '_'))
        env = {k: v for k, v in os.environ.items() if not k.startswith(('LD_', 'NIX_', '__NIX_', 'HECATE_'))
               and k not in ('HECATE', 'IN_NIX_SHELL', 'PYTHONHOME', 'PYTHONPATH', 'VIRTUAL_ENV')}
        env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1')
        for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            env[name] = '1'
        for phase in PHASES:
            require(not result['guard_error'] and not result['monitor_error'], 'Physical monitor rejected further work')
            directory = args.out / 'work' / phase; directory.mkdir(); (directory / 'tmp').mkdir()
            argv, seconds = phase_command(args, phase, args.seconds - (time.perf_counter_ns() - started) / 1e9)
            with recorder.span(phase, command=argv):
                with lock: state.update(path=directory / 'resources.jsonl', active=True)
                try:
                    observation = run_process(argv, cwd=paths.REPO / 'DSL/DSL_agent/scripts',
                        log=directory / 'worker.log', resources=directory / 'resources.jsonl',
                        timeout=seconds, mem_gib=24, fsize_gib=16, env=dict(env, TMPDIR=str(directory / 'tmp')))
                finally:
                    with lock: state['active'] = False
                result['processes'].append(dict(phase=phase, argv=argv, observation=observation))
                value = read(directory / 'phase_result.json') if (directory / 'phase_result.json').is_file() else None
                result['phases'][phase] = value
                require(clean(observation) and value and value.get('status') == 'ok',
                        phase + ': ' + str((value or {}).get('error', 'Incomplete worker')))
        result['status'] = 'ok'
    except BaseException as exc:
        result.update(status='error', error=type(exc).__name__ + ': ' + str(exc)[:3000])
    finally:
        for s in old: signal.signal(s, signal.SIG_IGN)
        signal.setitimer(signal.ITIMER_REAL, 0); stop.set()
        if watcher.is_alive(): watcher.join(timeout=3)
        result['monitor_stopped'] = not watcher.is_alive()
        if result['guard_error'] or result['monitor_error'] or not result['monitor_stopped']:
            result.update(status='error', error=result['guard_error'] or result['monitor_error'] or 'Monitor remains active')
        timing = summarize_events(read_events(recorder.path), termination_observed_ns=time.perf_counter_ns())
        result['timing'] = timing
        write_json(args.out / 'timing.json', timing); write_json(args.out / 'result.json', result)
        append_registry(paths.RESULTS_DIR / 'registry.jsonl', dict(run_id=args.out.name, round=ROUND,
            model=result['model'], mode='independent-original-block-R0-rule63-depth-compile', git_rev=git_rev(),
            metrics=dict(status=result['status'], API_calls=0, completion_path=str(args.out / 'completion.json')),
            artifacts_dir=str(args.out), ts=datetime.now(timezone.utc).isoformat()))
        completion = persist_case_completion(args.out, round_label=ROUND, started_ns=started,
            status=result['status'], timing=timing, required_artifact_bytes=artifact_bytes(args.out))
        row = paper_record(args, result, completion); write_json(args.out / 'paper-record.json', row)
        if args.paper_runs is not None:
            from .a_group_runs import append_record
            append_record(args.paper_runs, row)
        if result['monitor_stopped']:
            for s, handler in old.items(): signal.signal(s, handler)
    return result


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--worker-phase', choices=PHASES)
    parser.add_argument('--work-dir', type=Path)
    parser.add_argument('--paper-runs', type=Path)
    parser.add_argument('--seconds', type=int, default=12000)
    parser.add_argument('--phase-seconds', type=int, default=7200)
    parser.add_argument('--case-gib', type=int, default=64)
    for name in SOURCE_OPTIONS: parser.add_argument('--' + name.replace('_', '-'), type=Path)
    args = parser.parse_args(argv)
    require(180 < args.seconds <= 12000 and 0 < args.phase_seconds <= 7200
            and 16 <= args.case_gib <= 64, 'Finite original block budgets required')
    if args.worker_phase:
        require(args.work_dir is not None and all(getattr(args, name) is not None for name in SOURCE_OPTIONS),
                'All worker source paths must be explicit')
    else:
        from .. import paths
        root = paths.RESULTS_DIR
        defaults = dict(model_info=root / 'qwen-normalization-rebased-20261115/work/import/model_info',
            calibration=root / 'qwen-unit-width-newton7-agent-20261116/calibration.json',
            calibration_result=root / 'qwen-unit-width-newton7-agent-20261116/phases/001-calibration/worker-result.json',
            reference=root / 'qwen-full-reference-20261115/work/hf-reference/worker-result.json',
            inputs=root / 'r10-paper-qwen-import-20261115/worker/text-input/samples.jsonl',
            source_rows=root / 'r10-paper-qwen-import-20261115/worker/text-input/source_rows.json',
            profile=paths.PROFILES_DIR / 'qwen48/compiler-config.json',
            frontend=paths.DACAPO_ROOT / 'python/hecate/hecate/expr.py',
            metadata_template=root / 'E5d-agent-sessions-20261106/bert/acceptance/pars/export/Bert_PTAgent_DSL/config/compile_argv.json')
        for name, value in defaults.items(): setattr(args, name, getattr(args, name) or value)
    for name in ('out', 'work_dir', 'paper_runs', *SOURCE_OPTIONS):
        value = getattr(args, name)
        if value is not None: setattr(args, name, value.expanduser().resolve())
    return args


def main(argv=None):
    args = arguments(argv)
    result = worker(args) if args.worker_phase else execute(args)
    print(json.dumps(dict(status=result['status'], error=result['error'])))
    return int(result['status'] != 'ok')


if __name__ == '__main__':
    raise SystemExit(main())
