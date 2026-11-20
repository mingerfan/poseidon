"""One optimized native compile of a closed A2b Rule63 raw trace.

The actual clock covers this compile-only case. The separate Fig6 value reuses
the source entry-to-native-start prefix and adds this native compiler duration.
The scheduler owns the heavy lease. No model, Emit, trace, API or CKKS runs here.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
from types import FunctionType

from .. import paths
from . import cold_rule63_whole_model as original
from . import a_group_future_closed_archive as closed_source
from .r10_paper_metrics import StageRecorder, write_json

ROUND = 'A56-SHARED-RULE63-COMPILE'
MODE = 'optimized-native-compile-only-shared-A2b-rule63'
MODELS = dict(vgg16='vgg16_silu_avgpool', alexnet='alexnet_silu_avgpool')
OPTIONS = dict(deg=63, margin=1.5, slots=16384, placement='depth-dp',
               refresh='compiler', act_method='cheb')


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def raw_stat(path):
    stat = Path(path).lstat()
    return dict(inode=stat.st_ino, device=stat.st_dev, links=stat.st_nlink,
                mode=stat.st_mode, size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def metadata_cleanup(value):
    if isinstance(value, dict):
        require(not value.get('remaining_pids') and not value.get('cleanup_unknown'),
                'Original source descendant cleanup is unresolved')
        for child in value.values():
            metadata_cleanup(child)
    elif isinstance(value, list):
        for child in value:
            metadata_cleanup(child)


def source_gate(args, *, created=False):
    """Read closed source metadata and raw stat only; never inspect raw payload."""
    current = original.admission(args, created=created)
    source = args.source_case
    expected = paths.RESULTS_DIR.resolve() / ('a2b-' + args.short_model + '-rule63-fresh-cold-depth-20261117')
    require(source == expected and source.parent == paths.RESULTS_DIR.resolve(),
            'Only the matching original A2b Rule63 depth case is eligible')
    result, completion = read(source/'result.json'), read(source/'completion.json')
    require(result.get('status') == 'ok' and result.get('round') == original.ROUND
            and result.get('mode') == original.MODE and result.get('model') == args.model
            and result.get('run_id') == source.name and result.get('placement') == 'depth'
            and result.get('monitor_stopped') is True and not result.get('guard_error')
            and not result.get('monitor_error') and result.get('expected_bootstraps_match') is True,
            'A complete successful original A2b source is required')
    require(completion.get('round') == original.ROUND and completion.get('status') == 'ok'
            and completion.get('registry_persisted') is True
            and completion.get('required_artifacts_persisted') is True,
            'Original source completion is not durable')
    expected_phases = dict.fromkeys(original.PHASES, 1)
    require(result.get('attempted') == expected_phases and result.get('completed') == expected_phases,
            'All six original fresh phases must have completed once')
    require(all(result.get('phases', {}).get(p, {}).get('status') == 'ok' for p in original.PHASES),
            'Original source phase result is incomplete')
    rows = []
    with (paths.RESULTS_DIR/'registry.jsonl').open() as stream:
        fcntl.flock(stream, fcntl.LOCK_SH)
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if row.get('run_id') == source.name:
                    rows.append(row)
    require(len(rows) == 1 and set(rows[0]) == closed_source.REGISTRY_FIELDS,
            'One exact eight-field original source registry row is required')
    ledger = rows[0]
    require(all(ledger.get(k) == result[k] for k in ('round', 'model', 'mode', 'git_rev'))
            and ledger.get('metrics', {}).get('status') == 'ok'
            and Path(ledger.get('artifacts_dir', '')).resolve() == source,
            'Original source registry identity differs')
    metadata_cleanup(result)
    process_rows = result.get('processes', [])
    require([row.get('phase') for row in process_rows] == list(original.PHASES)
            and all(original.clean(row['observation']) for row in process_rows),
            'Every original outer phase process must be cleanly closed')
    generation = result['phases']['generation']
    math = generation.get('training_math') or {}
    require(generation.get('rule_options') == OPTIONS
            and generation.get('raw_Count', {}).get('stats', {}).get('counts', {}).get('boot') == 0
            and math.get('samples') == 3 and len(math.get('checks', [])) == 3
            and all(row.get('passed') is True for row in math.get('checks', [])),
            'Original fixed Rule63 Count and three training mathematics proof differ')
    source_profile = read(source/'source/compiler-profile.json')
    require(source_profile == read(args.profile), 'Preserve the complete original qwen48 cost table and parameters')
    require(read(source/'source/compile-metadata-template.json') == read(args.metadata_template),
            'Preserve the original compiler metadata recipe')
    native = result['phases']['compile'].get('native_process') or {}
    require(original.clean(native), 'Original A2b native compiler must be cleanly closed')
    started, compiler_started, compiler_ended, ended = (
        completion.get('started_ns'), native.get('started_ns'), native.get('ended_ns'), completion.get('ended_ns'))
    require(all(type(value) is int for value in (started, compiler_started, compiler_ended, ended))
            and 0 <= started < compiler_started < compiler_ended <= ended,
            'Same-host monotonic source prefix clocks must be ordered')
    outer = process_rows[-1]['observation']
    require(outer['started_ns'] <= compiler_started < compiler_ended <= outer['ended_ns'],
            'Source native compiler clock must be contained in its original compile phase')
    raw = source/'work/emit/candidate_trace.mlir'
    require(raw.is_file() and not raw.is_symlink() and raw.resolve() == raw,
            'Original source raw MLIR must remain a regular unmodified file')
    stat = raw_stat(raw)
    require(stat['size_bytes'] > 0, 'Original raw trace is empty')
    births = closed_source.producer_births(source)
    require(births > 0, 'Original producer birth evidence is required')
    return dict(current_metadata=current, source_case=str(source), source_result=str(source/'result.json'),
        source_completion=str(source/'completion.json'), source_raw=str(raw),
        source_raw_stat=stat,
        source_parameter_group='ours-N16-boot5', source_rule_options=OPTIONS,
        closed_producer_births=births, source_registry_run_id=source.name,
        shared_generation_prefix_ns=compiler_started-started,
        clock_evidence=dict(clock='Same-host time.perf_counter_ns in original R10/native and completion',
            source_entry_ns=started, source_native_compile_started_ns=compiler_started,
            source_native_compile_ended_ns=compiler_ended, source_case_ended_ns=ended),
        source_clock_scope='Original A2b entry through original native compiler start; old compile duration excluded',
        source_payload_reads=0, archived_input_recovery_attempts=0)


def prepare_source(args, proof):
    target = args.out/'source'
    target.mkdir()
    for file in (Path(__file__), Path(original.__file__),
                 *(Path(__file__).with_name(name+'.py') for name in
                   ('cold_rule63_worker', 'a_group_profile_sum_policy', 'a_group_runs'))):
        shutil.copyfile(file, target/file.name)
    shutil.copyfile(args.profile, target/'compiler-profile.json')
    shutil.copyfile(args.metadata_template, target/'compile-metadata-template.json')
    emit = args.out/'work/emit'
    emit.mkdir()
    raw = Path(proof['source_raw'])
    require(raw_stat(raw) == proof['source_raw_stat'],
            'Original raw stat changed after closed-source admission')
    (emit/'candidate_trace.mlir').symlink_to(raw)
    write_json(target/'shared-generation-source.json', proof)
    write_json(target/'expected-native-argv.json', expected_native_argv(args, proof))


def expected_native_argv(args, proof):
    from .r29_local_compile_matrix import compiler_argv
    from .r30_common_model_e2e import diagnostic_compile_argv
    command = compiler_argv(read(args.metadata_template), 'dacapo-optimized',
        binary=proof['current_metadata']['compiler'], source=args.out/'work/emit/candidate_trace.mlir',
        profile=args.profile, output=args.out/'work/emit/lowered.mlir', debug_snapshots=False)
    return diagnostic_compile_argv(command, elide_final_ir=args.diagnostic_elide_final_ir)


def terminal_source_observation(args, result, proof):
    """Stat and command metadata checks only; do not reopen MLIR or bundles."""
    if not proof:
        return
    result['source_raw_stat_preserved'] = False
    result['native_argv_verified'] = None
    try:
        require(raw_stat(proof['source_raw']) == proof['source_raw_stat'],
                'Original shared raw stat/identity changed during compile')
        result['source_raw_stat_preserved'] = True
        argv_file = args.out/'work/compile/native-argv.json'
        if argv_file.is_file():
            result['native_argv_verified'] = False
            command = read(argv_file)
            require(command == expected_native_argv(args, proof),
                    'Observed native command differs from the original optimized builder')
            require(command[1] == str(args.out/'work/emit/candidate_trace.mlir')
                    and command[command.index('-o')+1] == str(args.out/'work/emit/lowered.mlir'),
                    'Native raw input and new output paths differ')
            result['native_argv_verified'] = True
            result['native_argv_source'] = str(argv_file)
        elif result['status'] == 'ok':
            raise ValueError('Accepted native compilation has no saved actual command')
    except Exception as exc:
        result.update(status='error', source_preservation_error=type(exc).__name__+': '+str(exc))
        result['error'] = result.get('error') or result['source_preservation_error']


def phase_command(args, phase, remaining):
    require(phase == 'compile', 'Only one native compiler phase is supported')
    seconds = min(7200, int(remaining)-180)
    require(seconds > 0, 'Compile-only case allowance is exhausted')
    inner = ['-B', '-m', __spec__.name, '--worker', '--model', args.short_model,
        '--work-dir', str(args.out/'work/compile'), '--emit-dir', str(args.out/'work/emit'),
        '--profile', str(args.profile), '--metadata-template', str(args.metadata_template),
        '--native-seconds', str(seconds)]
    if args.diagnostic_elide_final_ir:
        inner.append('--diagnostic-elide-final-ir')
    return ([sys.executable, '-B', '-m', 'pt_agent.tools.nixrun', '--seconds', str(seconds+60),
             '--python', str(paths.INTAKE_PYTHON), '--', *inner], seconds+120)


def profile_policy(prediction):
    from .a_group_profile_sum_policy import profile_sum
    return profile_sum(prediction)


def paper_record(args, result, completion, timing):
    from .a_group_runs import empty_record
    compiled = result['phases'].get('compile') or {}
    process = compiled.get('native_process') or {}
    compile_s = ((process['ended_ns']-process['started_ns'])/1e9
                 if process.get('ended_ns') is not None else None)
    source = result.get('source') or {}
    prefix = source.get('shared_generation_prefix_ns')
    prefix_s = prefix/1e9 if prefix is not None else None
    derived = prefix_s+compile_s if prefix_s is not None and compile_s is not None else None
    policy = profile_policy(compiled.get('prediction'))
    row = empty_record(run_id=args.out.name, task='A5' if args.short_model == 'vgg16' else 'A6',
        model=args.short_model, system='dacapo_search_ours', param_group='ours-N16-boot5',
        status=original.classify(result), status_detail=result.get('error'), budget_s=7200,
        compile_s=compile_s, peak_rss_gib=original.peak_rss(args),
        boots_compiled=compiled.get('compiled_bootstraps'), exec_ops=compiled.get('execution_steps'),
        pred_latency_s=policy['pred_latency_s'], pred_table=str(args.profile),
        generation_e2e_s=derived, stages_s=dict(shared_generation_prefix=prefix_s, placement_compile=compile_s),
        api_calls=0, source=str(args.out/'result.json'), finished_utc=datetime.now(timezone.utc).isoformat())
    row.update(cold=False, estimated_from_shared_generation=True, derived_e2e_s=derived,
        shared_generation_prefix_s=prefix_s, source_case=str(args.source_case),
        source_generation_clock=source.get('clock_evidence'),
        generation_e2e_scope='Derived A2b entry-to-native-start prefix + this native compiler child elapsed; current launcher/persistence overhead excluded',
        actual_compile_only_case_seconds=completion['complete_case_elapsed_seconds'],
        actual_compile_only_completion=str(args.out/'completion.json'),
        actual_compile_only_scope='This new compile-only case entry through required result/registry persistence',
        placement_s=None, placement_in_compile=True, compiler_algorithm='dacapo-optimized',
        runtime_plan=compiled.get('runtime_plan'), execution_op_counts=compiled.get('execution_ops'),
        prediction_policy=policy, known_cost_partial_s=policy['known_cost_partial_s'],
        excluded_relinearize_counts=policy['excluded_counts'], missing_price_ops=policy['missing_remaining'],
        original_prediction=compiled.get('prediction'), compile_accepted=result['status'] == 'ok',
        model_forwards=0, Emit_runs=0, trace_runs=0, CKKS_runs=0, quality_retested=False)
    return row


class CompileRecorder(StageRecorder):
    def span(self, stage_id, **metadata):
        return super().span('shared_rule63_compile_only' if stage_id == 'fresh_cold_rule63' else stage_id, **metadata)


def execute(args):
    """Reuse the original supervised ordinary function in a private namespace."""
    state = {}
    def admit(current, *, created=False):
        state['source'] = source_gate(current, created=created)
        return state['source']
    def snapshot(current):
        prepare_source(current, state['source'])
    def durable(path, value):
        if Path(path).parent == args.out and Path(path).name in ('result.json', 'progress.json'):
            value.update(cold=False, estimated_from_shared_generation=True,
                shared_generation_reused=True, original_intake_reused=True, training_calibration_reused=True,
                new_model_forwards=0, new_Emit_runs=0, new_trace_runs=0, source_case=str(args.source_case),
                actual_clock_scope='This new compile-only case; derived shared-generation E2E reported separately')
            if Path(path).name == 'result.json':
                terminal_source_observation(args, value, state.get('source'))
        write_json(path, value)
    namespace = dict(original.__dict__)
    namespace.update(ROUND=ROUND, MODE=MODE, PHASES=('compile',), StageRecorder=CompileRecorder,
        admission=admit, snapshot=snapshot, phase_command=phase_command,
        paper_record=paper_record, write_json=durable)
    run = FunctionType(original.execute.__code__, namespace, original.execute.__name__,
                       original.execute.__defaults__, original.execute.__closure__)
    return run(args)


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--source-case', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--paper-runs', type=Path)
    parser.add_argument('--seconds', type=int, default=7500)
    parser.add_argument('--case-gib', type=int, default=32)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--native-seconds', type=int, default=7200)
    parser.add_argument('--diagnostic-elide-final-ir', action='store_true')
    for name in ('work-dir', 'emit-dir', 'profile', 'metadata-template'):
        parser.add_argument('--'+name, type=Path)
    args = parser.parse_args(argv)
    args.short_model = args.model
    args.model = MODELS[args.short_model]
    args.placement = 'optimized'
    require(180 < args.seconds <= 7500 and 0 < args.native_seconds <= 7200 and 16 <= args.case_gib <= 32,
            'Finite whole<=7500s/native<=7200s/case16..32GiB required')
    if args.worker:
        require(all(getattr(args, k) is not None for k in ('work_dir', 'emit_dir', 'profile', 'metadata_template')),
                'Every pure-worker path must be explicit')
        args.worker_phase = 'compile'
    else:
        require(all(getattr(args, k) is not None for k in ('source_case', 'out', 'paper_runs')),
                'Source, fresh out and paper-runs are required')
        args.profile = args.profile or paths.PROFILES_DIR/'qwen48/compiler-config.json'
        args.metadata_template = args.metadata_template or paths.RESULTS_DIR/'E5d-agent-sessions-20261106/bert/acceptance/pars/export/Bert_PTAgent_DSL/config/compile_argv.json'
        args.inputs = paths.WORK_RESULTS/'agent-cifar-offline-records-r1804/cifar10-records'
        args.selections = paths.WORK_RESULTS/'agent-resnet20-original-accuracy-r1805/frozen-selections.json'
        args.frontend = paths.DACAPO_ROOT/'python/hecate/hecate/expr.py'
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.expanduser().resolve())
    return args


def main(argv=None):
    args = arguments(argv)
    if args.worker:
        from .cold_rule63_worker import run
        return run(args)
    return int(execute(args)['status'] != 'ok')


if __name__ == '__main__':
    raise SystemExit(main())
