"""Compile the same closed Rule63/R0 Qwen first-block raw trace once.

The source is the independent depth baseline; its original model, arithmetic,
training mathematics and raw trace are reused. This new case measures only the
optimized compilation. It reports no cold Agent or full-model generation time.
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

from .. import paths, toolchain
from . import a5_a6_shared_rule63_compile as shared
from . import cold_rule63_whole_model as supervisor
from . import qwen_rule63_block_compile as depth
from .a_group_profile_sum_policy import profile_sum
from .r10_paper_metrics import StageRecorder, write_json

ROUND = 'QWEN-RULE63-SHARED-OPTIMIZED'
MODE = 'optimized-native-compile-only-shared-Qwen-rule63-block'
MODULE = 'pt_agent.experiments.qwen_rule63_shared_optimized_compile'
SOURCE_ID = 'qwen-rule63-r0-layer0-depth-qwen48-20261117'
MODEL = 'Qwen2-0.5B-layer0'
SOURCE_MODE = 'independent-original-block-R0-rule63-depth-compile'


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def source_gate(args, *, created=False):
    """Admit only the successful original block, metadata and raw stat only."""
    require(args.out.parent == paths.RESULTS_DIR.resolve()
            and not args.out.is_symlink() and (created or not os.path.lexists(args.out)),
            'Use a fresh direct results child for the compile-only case')
    require(args.host_pid_namespace == os.readlink('/proc/self/ns/pid'),
            'Saved producer births require the explicitly selected host namespace')
    source = args.source_case
    require(source == paths.RESULTS_DIR.resolve() / SOURCE_ID and source != args.out,
            'Only the independent original Rule63/R0 first-block depth case is eligible')
    result, completion = read(source / 'result.json'), read(source / 'completion.json')
    require(result.get('status') == 'ok' and result.get('round') == depth.ROUND
            and result.get('model') == MODEL and result.get('run_id') == source.name
            and result.get('monitor_stopped') is True and not result.get('guard_error')
            and not result.get('monitor_error'), 'Original block source did not close successfully')
    require(completion.get('round') == depth.ROUND and completion.get('status') == 'ok'
            and completion.get('registry_persisted') is True
            and completion.get('required_artifacts_persisted') is True,
            'Original block completion is incomplete')
    phases = result.get('phases') or {}
    processes = result.get('processes') or []
    require(set(phases) == set(depth.PHASES)
            and all(phases[p] and phases[p].get('status') == 'ok' for p in depth.PHASES)
            and [p.get('phase') for p in processes] == list(depth.PHASES)
            and all(supervisor.clean(p.get('observation') or {}) for p in processes),
            'All four independent depth phases must have completed cleanly once')
    shared.metadata_cleanup(result)
    prepared, emitted, traced, compiled = (phases[p] for p in depth.PHASES)
    require(prepared.get('training_math', {}).get('passed') is True
            and prepared.get('training_math', {}).get('sample_id') == 'train:0'
            and prepared.get('training_math', {}).get('sequence_capacity') == 32
            and prepared.get('training_math', {}).get('criterion') == '1e-12+1e-12*abs(reference)',
            'The original one-training-prefix strict block mathematics is required')
    changes = prepared.get('activation_degree_changes') or []
    require(len(changes) == 1 and changes[0].get('fn') == 'silu'
            and changes[0].get('original_degree') == 127 and changes[0].get('degree') == 63
            and prepared.get('program_ast'), 'Retain the original R0 SiLU-only degree63 rule')
    raw_count, emit_count = prepared.get('raw_count') or {}, emitted.get('emit_stats') or {}
    require(raw_count.get('counts', {}).get('boot') == 0
            and all(raw_count.get(key) == emit_count.get(key)
                    for key in ('counts', 'ops_total', 'distinct_consts')),
            'The source raw unplaced Count/Emit inventory differs')
    require(supervisor.clean(traced.get('native_process') or {})
            and supervisor.clean(compiled.get('native_process') or {})
            and type(compiled.get('compiled_bootstraps')) is int
            and compiled['compiled_bootstraps'] >= 0
            and Path(compiled.get('runtime_plan', '')).is_file(),
            'Original native trace and depth compiler must have clean successful artifacts')
    source_proof = result.get('source') or {}
    require(source_proof.get('C2_required') is False
            and source_proof.get('Agent_source_used') is False
            and source_proof.get('bridge_policy_used') is False,
            'Retain independent R0 block provenance')
    ledger = []
    with (paths.RESULTS_DIR / 'registry.jsonl').open() as stream:
        fcntl.flock(stream, fcntl.LOCK_SH)
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if row.get('run_id') == source.name:
                    ledger.append(row)
    require(len(ledger) == 1 and set(ledger[0]) == shared.closed_source.REGISTRY_FIELDS,
            'One matching eight-field original block registry record is required')
    record = ledger[0]
    require(record.get('round') == depth.ROUND and record.get('model') == MODEL
            and record.get('mode') == SOURCE_MODE and record.get('metrics', {}).get('status') == 'ok'
            and Path(record.get('artifacts_dir', '')).resolve() == source,
            'Original block registry identity differs')
    original_profile = read(source / 'source' / args.profile.name)
    expected = dict(polynomialDegree=65536, rescalingFactor=43, levelLowerBound=1,
                    levelUpperBound=8, bootstrapLevelLowerBound=1, bootstrapLevelUpperBound=5,
                    runtime='HEAAN-HEVM')
    require(original_profile == read(args.profile)
            and all(original_profile.get(k) == v for k, v in expected.items()),
            'Preserve the entire original qwen48 parameters and operation cost table')
    require(read(source / 'source' / args.metadata_template.name) == read(args.metadata_template),
            'Preserve the original compiler metadata template')
    binary = Path(toolchain.host_paths()['hecate_opt'])
    require(binary.is_file(), 'Existing native compiler is required')
    raw = source / 'work/emit/candidate_trace.mlir'
    require(raw.is_file() and not raw.is_symlink() and raw.resolve() == raw,
            'The complete original unarchived raw trace must be present')
    stat = shared.raw_stat(raw)
    require(stat['size_bytes'] > 0, 'Original block raw trace is empty')
    births = shared.closed_source.producer_births(source)
    require(births > 0, 'Original block producer birth evidence is required')
    require(shutil.disk_usage(args.out.parent).free >= 10 * supervisor.GIB,
            'resource_limit: free10GiB required before the compiler')
    return dict(source_case=str(source), source_result=str(source / 'result.json'),
        source_completion=str(source / 'completion.json'), source_raw=str(raw),
        source_raw_stat=stat, closed_producer_births=births,
        host_pid_namespace=args.host_pid_namespace, source_registry_run_id=source.name,
        current_metadata=dict(compiler=str(binary)), source_parameter_group='ours-N16-boot5',
        original_rule='Default transformer_rule R0; singleSiLU127to63 only',
        full_model=False, layer=0, sequence_capacity=32, source_payload_reads=0,
        C2_required=False, Agent_source_used=False, archived_input_recovery_attempts=0)


def prepare_source(args, proof):
    directory = args.out / 'source'; directory.mkdir()
    for file in (Path(__file__), Path(depth.__file__), Path(supervisor.__file__), Path(shared.__file__),
                 Path(__file__).with_name('cold_rule63_worker.py'),
                 Path(__file__).with_name('a_group_profile_sum_policy.py')):
        shutil.copyfile(file, directory / file.name)
    shutil.copyfile(args.profile, directory / 'compiler-profile.json')
    shutil.copyfile(args.metadata_template, directory / 'compile-metadata-template.json')
    emit = args.out / 'work/emit'; emit.mkdir()
    require(shared.raw_stat(proof['source_raw']) == proof['source_raw_stat'],
            'Original block raw stat changed after source admission')
    (emit / 'candidate_trace.mlir').symlink_to(proof['source_raw'])
    write_json(directory / 'shared-generation-source.json', proof)
    write_json(directory / 'expected-native-argv.json', shared.expected_native_argv(args, proof))


def phase_command(args, phase, remaining):
    require(phase == 'compile', 'Only the independent optimized native compile is supported')
    seconds = min(7200, int(remaining) - 180)
    require(seconds > 0, 'Compile-only case allowance is exhausted')
    inner = ['-B', '-m', MODULE, '--worker', '--work-dir', str(args.out / 'work/compile'),
             '--emit-dir', str(args.out / 'work/emit'), '--profile', str(args.profile),
             '--metadata-template', str(args.metadata_template), '--native-seconds', str(seconds)]
    if args.diagnostic_elide_final_ir:
        inner.append('--diagnostic-elide-final-ir')
    return [sys.executable, '-B', '-m', 'pt_agent.tools.nixrun', '--seconds', str(seconds + 60),
            '--python', str(paths.INTAKE_PYTHON), '--', *inner], seconds + 120


def paper_record(args, result, completion, timing):
    from .a_group_runs import empty_record
    compiled = result['phases'].get('compile') or {}
    process = compiled.get('native_process') or {}
    compile_s = ((process['ended_ns'] - process['started_ns']) / 1e9
                 if process.get('ended_ns') is not None else None)
    policy = profile_sum(compiled.get('prediction'))
    spans = {r['stage_id']: r for r in timing['spans']}
    stages = {name: spans[name]['inclusive_ns'] / 1e9 if name in spans else None
              for name in ('source_preflight', 'compile')}
    used = sum(value for value in stages.values() if value is not None)
    require(used <= completion['complete_case_elapsed_seconds'], 'Compile-only windows exceed its case clock')
    stages['other'] = completion['complete_case_elapsed_seconds'] - used
    return empty_record(run_id=args.out.name, task='Qwen-Rule63-block-search', model='qwen2_0_5b_block',
        system='dacapo_search_ours', param_group='ours-N16-boot5', status=supervisor.classify(result),
        status_detail=result.get('error'), budget_s=7200, compile_s=compile_s,
        peak_rss_gib=supervisor.peak_rss(args), boots_compiled=compiled.get('compiled_bootstraps'),
        exec_ops=compiled.get('execution_steps'), pred_latency_s=policy['pred_latency_s'],
        pred_table=str(args.profile), generation_e2e_s=None, stages_s=stages, api_calls=0,
        source=str(args.out / 'result.json'), finished_utc=datetime.now(timezone.utc).isoformat(),
        source_case=str(args.source_case), source_generation_reused=True, cold=False,
        generation_e2e_scope='Unavailable for the independent Qwen block; compile-only actual clock reported separately',
        full_model=False, layer=0, sequence_capacity=32, estimated_from_shared_generation=False,
        actual_compile_only_case_seconds=completion['complete_case_elapsed_seconds'],
        actual_compile_only_completion=str(args.out / 'completion.json'), placement_s=None,
        placement_in_compile=True, compiler_algorithm='dacapo-optimized',
        runtime_plan=compiled.get('runtime_plan'), execution_op_counts=compiled.get('execution_ops'),
        prediction_policy=policy, known_cost_partial_s=policy['known_cost_partial_s'],
        excluded_relinearize_counts=policy['excluded_counts'], missing_price_ops=policy['missing_remaining'],
        full_inference_prediction_s=None, full_inference_calibrated=False,
        compile_accepted=result['status'] == 'ok', model_forwards=0, Emit_runs=0, trace_runs=0,
        CKKS_runs=0, quality_retested=False, confirm_samples=0, report_samples=0)


class CompileRecorder(StageRecorder):
    def span(self, stage_id, **metadata):
        return super().span('shared_qwen_rule63_compile_only' if stage_id == 'fresh_cold_rule63'
                            else stage_id, **metadata)


def execute(args):
    state = {}
    def admit(current, *, created=False):
        state['source'] = source_gate(current, created=created)
        return state['source']
    def snapshot(current):
        prepare_source(current, state['source'])
    def durable(path, value):
        if Path(path).parent == args.out and Path(path).name in ('result.json', 'progress.json'):
            value.update(cold=False, historical_elapsed_reused=False, original_intake_reused=True,
                training_calibration_reused=True, source_generation_reused=True,
                new_model_forwards=0, new_Emit_runs=0, new_trace_runs=0, source_case=str(args.source_case),
                actual_clock_scope='Only this native compile case; no Qwen cold generation clock is claimed')
            if Path(path).name == 'result.json':
                shared.terminal_source_observation(args, value, state.get('source'))
        write_json(path, value)
    namespace = dict(supervisor.__dict__)
    namespace.update(ROUND=ROUND, MODE=MODE, PHASES=('compile',), StageRecorder=CompileRecorder,
        admission=admit, snapshot=snapshot, phase_command=phase_command,
        paper_record=paper_record, write_json=durable)
    run = FunctionType(supervisor.execute.__code__, namespace, supervisor.execute.__name__,
                       supervisor.execute.__defaults__, supervisor.execute.__closure__)
    return run(args)


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-case', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--paper-runs', type=Path)
    parser.add_argument('--host-pid-namespace')
    parser.add_argument('--seconds', type=int, default=7500)
    parser.add_argument('--case-gib', type=int, default=32)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--native-seconds', type=int, default=7200)
    parser.add_argument('--diagnostic-elide-final-ir', action='store_true')
    for name in ('work-dir', 'emit-dir', 'profile', 'metadata-template'):
        parser.add_argument('--' + name, type=Path)
    args = parser.parse_args(argv)
    args.model = MODEL; args.placement = 'optimized'
    require(180 < args.seconds <= 7500 and 0 < args.native_seconds <= 7200
            and 16 <= args.case_gib <= 32, 'Finite compile-only case/native/disk allowances are required')
    if args.worker:
        require(all(getattr(args, name) is not None
                    for name in ('work_dir', 'emit_dir', 'profile', 'metadata_template')),
                'Pure compile worker requires all explicit paths')
        args.worker_phase = 'compile'
    else:
        require(args.out is not None and args.paper_runs is not None and args.host_pid_namespace,
                'Fresh output, paper ledger and host process namespace must be explicit')
        args.source_case = args.source_case or paths.RESULTS_DIR / SOURCE_ID
        args.profile = args.profile or paths.PROFILES_DIR / 'qwen48/compiler-config.json'
        args.metadata_template = args.metadata_template or paths.RESULTS_DIR / 'E5d-agent-sessions-20261106/bert/acceptance/pars/export/Bert_PTAgent_DSL/config/compile_argv.json'
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
