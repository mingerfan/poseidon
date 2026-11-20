"""One optimized compile of the current A1 Chebyshev63 BERT raw program.

Reuse the closed source's Count/Emit, interface and native artifact evidence.
Original A1 mathematical/quality evaluation remains unmeasured. An archived raw
input is prepared only in this new case, with raw DEFLATE and no footer checks.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
from types import FunctionType

from .. import paths, toolchain
from . import a1_bert_fixed_current as source_driver
from . import a5_a6_shared_rule63_compile as shared
from . import cold_rule63_whole_model as supervisor
from .a_group_profile_sum_policy import profile_sum
from .bounded_gzip_input import restore_raw_member
from .completion_metrics import artifact_bytes
from .r10_paper_metrics import StageRecorder, read_events, summarize_events, write_json

ROUND = 'BERT-RULE63-SHARED-OPTIMIZED'
MODE = 'optimized-native-compile-only-shared-current-BERT-rule63'
MODULE = 'pt_agent.experiments.bert_rule63_shared_optimized_compile'
SOURCE_ID = 'bert-rule63-current-build-depth-recipe-json-fixed-a1-20261117'
ARCHIVE_ID = 'a1-bert-rule63-closed-raw-trace-archive-20261117'
MODEL = 'bert-tiny-sst2'
PHASES = ('restore', 'compile')


def read(path):
    return json.loads(Path(path).read_text())


def require(value, message):
    if not value:
        raise ValueError(message)


def unique_registry(case, result, records):
    found = [row for row in records if row.get('run_id') == case.name]
    require(len(found) == 1 and set(found[0]) == shared.closed_source.REGISTRY_FIELDS,
            'One matching eight-field closed source registry is required')
    row = found[0]
    require(row.get('round') == result['round'] and row.get('metrics', {}).get('status') == 'ok'
            and Path(row.get('artifacts_dir', '')).resolve() == case,
            'Closed source registry round, status or artifact path differs')
    return row


def durable_closed(case, result, completion):
    require(result.get('status') == completion.get('status') == 'ok'
            and result.get('round') == completion.get('round')
            and completion.get('registry_persisted') is True
            and completion.get('required_artifacts_persisted') is True,
            'Original result/completion must have closed successfully')
    shared.metadata_cleanup(result)


def source_gate(args, *, created=False):
    """Original A1 and archive metadata only; raw/gzip payloads stay unopened."""
    root = paths.RESULTS_DIR.resolve(); source = args.source_case
    require(args.out.parent == root and not args.out.is_symlink()
            and (created or not os.path.lexists(args.out)), 'Fresh direct results child required')
    require(args.host_pid_namespace == os.readlink('/proc/self/ns/pid'),
            'Use the explicitly selected host namespace for saved producer births')
    require(source == root / SOURCE_ID and source != args.out, 'Only the exact current A1 Rule63 source is eligible')
    result, completion = read(source / 'result.json'), read(source / 'completion.json')
    durable_closed(source, result, completion)
    require(result.get('round') == source_driver.ROUND and result.get('model') == source_driver.MODEL
            and result.get('mode') == source_driver.MODE and result.get('compile_accepted') is True
            and result.get('monitor_stopped') is True and result.get('compiled_bootstraps') == 12713,
            'Successful current Chebyshev63/12713-depth A1 source required')
    expected_phases = dict.fromkeys(source_driver.PHASES, 1)
    require(result.get('attempted') == expected_phases and result.get('completed') == expected_phases,
            'All four original A1 phases must have completed once')
    phases = result.get('phases') or {}; processes = result.get('processes') or []
    require(set(phases) == set(source_driver.PHASES)
            and all(phases[p].get('status') == 'ok' for p in source_driver.PHASES)
            and [p.get('phase') for p in processes] == list(source_driver.PHASES)
            and all(supervisor.clean(p.get('observation') or {}) for p in processes)
            and all(result.get(key) == 0 for key in source_driver.ZERO),
            'Original A1 stages and zero-model/API execution scope differ')
    prepared, emitted, traced, compiled = (phases[p] for p in source_driver.PHASES)
    changes = prepared.get('activation_degree_changes') or []
    require(Counter(row.get('fn') for row in changes) == dict(gelu=2, tanh=1)
            and all(row.get('original_degree') == 127 and row.get('degree') == 63 for row in changes),
            'Original R0 GELU/tanh activation degree63 choices differ')
    count, emit = prepared.get('raw_count') or {}, emitted.get('emit_stats') or {}
    require(count.get('counts', {}).get('boot') == 0
            and all(count.get(k) == emit.get(k) for k in ('counts', 'ops_total', 'distinct_consts'))
            and prepared.get('signature', {}).get('ok') is True
            and prepared.get('signature') == emitted.get('signature'),
            'Original A1 raw Count/Emit or interface evidence differs')
    require(supervisor.clean(traced.get('process') or {}) and supervisor.clean(compiled.get('process') or {})
            and compiled.get('compiled_bootstraps') == 12713
            and compiled.get('runtime_plan') == result.get('runtime_plan')
            and Path(compiled.get('runtime_plan', '')).is_file(),
            'Original A1 trace/depth compiler and backend artifact must be cleanly closed')
    with (root / 'registry.jsonl').open() as stream:
        fcntl.flock(stream, fcntl.LOCK_SH)
        records = [json.loads(line) for line in stream if line.strip()]
    ledger = unique_registry(source, result, records)
    require(ledger.get('model') == source_driver.MODEL and ledger.get('mode') == source_driver.MODE,
            'Original A1 model/mode registry identity differs')
    profile = read(source / 'compiler-config.json')
    expected = dict(polynomialDegree=65536, rescalingFactor=43, levelLowerBound=1, levelUpperBound=8,
                    bootstrapLevelLowerBound=1, bootstrapLevelUpperBound=5, runtime='HEAAN-HEVM')
    require(profile == read(args.profile) and all(profile.get(k) == v for k, v in expected.items()),
            'Retain the full original qwen48 profile and latency table')
    require(read(source / 'recorded-compile-template.json') == read(args.metadata_template),
            'Retain the original native compiler metadata template')
    binary = Path(toolchain.host_paths()['hecate_opt'])
    require(binary.is_file() and Path(result['source']['hecate_opt']).resolve() == binary.resolve(),
            'Retain the original current compiler binary')
    births = shared.closed_source.producer_births(source)
    require(births > 0, 'Original A1 producer birth evidence is required')
    raw = source / 'work/emit/candidate_trace.mlir'; archive = raw.with_name(raw.name + '.gz')
    projection = traced.get('mlir', {}).get('bytes')
    require(type(projection) is int and 0 < projection <= 16 * supervisor.GIB,
            'Original trace-size projection must fit the16GiB input-file allowance')
    require(shutil.disk_usage(args.out.parent).free >= 10 * supervisor.GIB,
            'resource_limit: free10GiB required before input preparation')
    proof = dict(source_case=str(source), source_result=str(source / 'result.json'),
        source_completion=str(source / 'completion.json'), current_metadata=dict(compiler=str(binary)),
        source_registry_run_id=source.name, source_parameter_group='ours-N16-boot5',
        closed_source_producer_births=births, host_pid_namespace=args.host_pid_namespace,
        source_math_status='Unmeasured in original A1; no new mathematical or quality acceptance is claimed',
        independent_math_passed=None, quality_retested=False, source_payload_reads=0,
        original_trace_size_projection_bytes=projection, decoded_size_identity_comparison=False)
    if raw.is_file() and not raw.is_symlink() and raw.resolve() == raw:
        proof.update(input_kind='raw-symlink', input_source=str(raw), input_source_stat=shared.raw_stat(raw),
                     native_raw=str(args.out / 'input/candidate_trace.mlir'), archive_case=None, projected_new_raw_bytes=0)
    else:
        require(not os.path.lexists(raw) and archive.is_file() and not archive.is_symlink()
                and archive.resolve() == archive, 'Original input must be a regular raw trace or its closed gzip archive')
        archive_case = root / ARCHIVE_ID
        ar, ac = read(archive_case / 'result.json'), read(archive_case / 'completion.json')
        durable_closed(archive_case, ar, ac); archived = ar.get('archive') or {}
        archive_ledger = unique_registry(archive_case, ar, records)
        require(ar.get('round') == 'A1-CLOSED-RAW-ARCHIVE' and ar.get('source_case') == str(source)
                and archive_ledger.get('model') == 'bert_tiny'
                and archive_ledger.get('mode') == 'closed-raw-MLIR-reversible-compression'
                and supervisor.clean(ar.get('process') or {})
                and archived.get('source') == str(raw) and archived.get('archive') == str(archive)
                and archived.get('status') == 'compressed' and archived.get('raw_bytes') == projection
                and archived.get('compressed_bytes') == archive.stat().st_size,
                'Original archive producer provenance and recorded size projection differ')
        archive_births = shared.closed_source.producer_births(archive_case)
        require(archive_births > 0, 'Closed archive producer birth evidence is required')
        require(artifact_bytes(args.out) + projection < args.case_gib * supervisor.GIB
                and shutil.disk_usage(args.out.parent).free >= projection + 10 * supervisor.GIB,
                'resource_limit: projected restored raw plus case allowance and10GiB reserve required')
        proof.update(input_kind='archived-input-preparation', input_source=str(archive),
            input_source_stat=shared.raw_stat(archive), native_raw=str(args.out / 'input/candidate_trace.mlir'),
            archive_case=str(archive_case), archive_result=str(archive_case / 'result.json'),
            archive_completion=str(archive_case / 'completion.json'), closed_archive_producer_births=archive_births,
            projected_new_raw_bytes=projection, archive_footer_read=False, CRC_checks=0)
    return proof


def expected_native_argv(args, proof):
    from .r29_local_compile_matrix import compiler_argv
    from .r30_common_model_e2e import diagnostic_compile_argv
    # Keep the single restored file in input/; the native worker also writes its
    # compiled artifacts there, without a second counted alias of the large raw.
    argv = compiler_argv(read(args.metadata_template), 'dacapo-optimized',
        binary=proof['current_metadata']['compiler'], source=args.out / 'input/candidate_trace.mlir',
        profile=args.profile, output=args.out / 'input/lowered.mlir', debug_snapshots=False)
    return diagnostic_compile_argv(argv, elide_final_ir=args.diagnostic_elide_final_ir)


def prepare_source(args, proof):
    directory = args.out / 'source'; directory.mkdir()
    for file in (Path(__file__), Path(__file__).with_name('bounded_gzip_input.py'),
                 Path(source_driver.__file__), Path(supervisor.__file__), Path(shared.__file__),
                 Path(__file__).with_name('cold_rule63_worker.py'), Path(__file__).with_name('a_group_profile_sum_policy.py')):
        shutil.copyfile(file, directory / file.name)
    shutil.copyfile(args.profile, directory / 'compiler-profile.json')
    shutil.copyfile(args.metadata_template, directory / 'compile-metadata-template.json')
    (args.out / 'input').mkdir()
    require(shared.raw_stat(proof['input_source']) == proof['input_source_stat'], 'Original source input stat changed')
    if proof['input_kind'] == 'raw-symlink':
        Path(proof['native_raw']).symlink_to(proof['input_source'])
    write_json(directory / 'shared-generation-source.json', proof)
    write_json(directory / 'expected-native-argv.json', expected_native_argv(args, proof))


def restore_worker(args):
    """Prepare the explicit input only; pure worker reads no registry/config loader."""
    result = dict(status='running', error=None, phase='restore', model=MODEL, API_calls=0,
                  model_forwards=0, Count_runs=0, Emit_runs=0, trace_runs=0, CKKS_runs=0)
    recorder = StageRecorder(args.work_dir / 'events.jsonl')
    try:
        with recorder.span('raw_input_preparation'):
            if args.archive is None:
                result['restore'] = dict(status='ok', raw_source_used=True, decoded_bytes_written=0,
                    scope='Existing original raw symlink; no restoration or payload read')
            else:
                def guard(additional):
                    require(artifact_bytes(args.case_root) + additional <= args.case_gib * supervisor.GIB,
                            'resource_limit: restored input exceeds case allowance')
                    require(shutil.disk_usage(args.case_root).free >= 10 * supervisor.GIB + additional,
                            'resource_limit: free10GiB required throughout restored input preparation')
                result['restore'] = restore_raw_member(args.archive, args.restored_input,
                    maximum_output_bytes=16 * supervisor.GIB, guard=guard)
        result['status'] = 'ok'
    except BaseException as exc:
        result.update(status='error', error=type(exc).__name__ + ': ' + str(exc)[:3000])
        result['partial_restored_input_bytes'] = args.restored_input.stat().st_size if args.restored_input.exists() else None
    result['timing'] = summarize_events(read_events(recorder.path))
    write_json(args.work_dir / 'phase_result.json', result)
    return int(result['status'] != 'ok')


def phase_command(args, phase, remaining):
    require(phase in PHASES, 'Only input preparation and one native compile are supported')
    seconds = min(1800 if phase == 'restore' else 7200, int(remaining) - 180)
    require(seconds > 0, 'BERT compile-only allowance exhausted')
    inner = ['-B', '-m', MODULE, '--worker-phase', phase, '--work-dir', str(args.out / 'work' / phase),
        '--emit-dir', str(args.out / 'input'), '--profile', str(args.profile),
        '--metadata-template', str(args.metadata_template), '--native-seconds', str(seconds),
        '--case-root', str(args.out), '--restored-input', str(args.out / 'input/candidate_trace.mlir'),
        '--case-gib', str(args.case_gib)]
    proof = read(args.out / 'source/shared-generation-source.json')
    if proof['input_kind'] == 'archived-input-preparation': inner += ['--archive', proof['input_source']]
    if args.diagnostic_elide_final_ir: inner.append('--diagnostic-elide-final-ir')
    return [sys.executable, '-B', '-m', 'pt_agent.tools.nixrun', '--seconds', str(seconds + 60),
            '--python', str(paths.INTAKE_PYTHON), '--', *inner], seconds + 120


def terminal_source_observation(args, result, proof):
    if not proof: return
    result['source_input_stat_preserved'] = False
    try:
        require(shared.raw_stat(proof['input_source']) == proof['input_source_stat'], 'Original source input stat changed')
        result['source_input_stat_preserved'] = True
        if result['status'] == 'ok':
            require((result.get('phases', {}).get('restore') or {}).get('status') == 'ok', 'Input preparation did not succeed')
            require(Path(proof['native_raw']).is_file(), 'Prepared compiler raw input is missing')
            require(supervisor.clean((result['phases']['compile'] or {}).get('native_process') or {}),
                    'Native compilation descendant cleanup is incomplete')
        argv = args.out / 'work/compile/native-argv.json'
        if argv.is_file():
            require(read(argv) == expected_native_argv(args, proof), 'Actual optimized compiler argv differs')
            result['native_argv_verified'] = True
        elif result['status'] == 'ok':
            raise ValueError('Successful compile requires its actual saved native command')
    except BaseException as exc:
        result.update(status='error', source_preservation_error=type(exc).__name__ + ': ' + str(exc))
        result['error'] = result.get('error') or result['source_preservation_error']


def paper_record(args, result, completion, timing):
    from .a_group_runs import empty_record
    compiled = result['phases'].get('compile') or {}; native = compiled.get('native_process') or {}
    policy = profile_sum(compiled.get('prediction')); spans = {r['stage_id']: r for r in timing['spans']}
    stages = {name: spans[name]['inclusive_ns'] / 1e9 if name in spans else None
              for name in ('source_preflight', 'restore', 'compile')}
    used = sum(v for v in stages.values() if v is not None)
    require(used <= completion['complete_case_elapsed_seconds'], 'Compile-only stages exceed the actual clock')
    stages['other'] = completion['complete_case_elapsed_seconds'] - used
    return empty_record(run_id=args.out.name, task='BERT-Rule63-same-program-search', model='bert_tiny',
        system='dacapo_search_ours', param_group='ours-N16-boot5', status=supervisor.classify(result),
        status_detail=result.get('error'), budget_s=7200,
        compile_s=(native['ended_ns'] - native['started_ns']) / 1e9 if native.get('ended_ns') is not None else None,
        peak_rss_gib=supervisor.peak_rss(args), boots_compiled=compiled.get('compiled_bootstraps'),
        exec_ops=compiled.get('execution_steps'), pred_latency_s=policy['pred_latency_s'], pred_table=str(args.profile),
        generation_e2e_s=None, stages_s=stages, api_calls=0, source=str(args.out / 'result.json'),
        finished_utc=datetime.now(timezone.utc).isoformat(), source_case=str(args.source_case),
        source_generation_reused=True, cold=False, estimated_from_shared_generation=False,
        actual_compile_only_case_seconds=completion['complete_case_elapsed_seconds'],
        generation_e2e_scope='Unmeasured cold BERT generation; input preparation and new native compile clocks are separate',
        input_preparation_scope='New-case bounded raw DEFLATE recovery when archived; no CRC/ISIZE/FHCRC/footer or decoded-size identity checks',
        placement_s=None, placement_in_compile=True, compiler_algorithm='dacapo-optimized',
        runtime_plan=compiled.get('runtime_plan'), execution_op_counts=compiled.get('execution_ops'),
        prediction_policy=policy, known_cost_partial_s=policy['known_cost_partial_s'],
        excluded_relinearize_counts=policy['excluded_counts'], missing_price_ops=policy['missing_remaining'],
        full_inference_prediction_s=None, full_inference_calibrated=False, quality_retested=False,
        independent_math_passed=None, source_math_scope='Original A1 math/quality unmeasured; no new claim',
        model_forwards=0, Count_runs=0, Emit_runs=0, trace_runs=0, CKKS_runs=0, confirm_samples=0, report_samples=0)


class CompileRecorder(StageRecorder):
    def span(self, stage_id, **metadata):
        return super().span('shared_BERT_rule63_input_and_compile' if stage_id == 'fresh_cold_rule63' else stage_id, **metadata)


def execute(args):
    state = {}
    def admit(current, *, created=False):
        state['source'] = source_gate(current, created=created); return state['source']
    def snapshot(current): prepare_source(current, state['source'])
    def durable(path, value):
        if Path(path).parent == args.out and Path(path).name in ('result.json', 'progress.json'):
            value.update(cold=False, source_generation_reused=True, original_intake_reused=True,
                training_calibration_reused=True, source_case=str(args.source_case),
                actual_clock_scope='New input preparation and native compilation only; no BERT cold generation claim')
            if Path(path).name == 'result.json': terminal_source_observation(args, value, state.get('source'))
        write_json(path, value)
    namespace = dict(supervisor.__dict__)
    namespace.update(ROUND=ROUND, MODE=MODE, PHASES=PHASES, StageRecorder=CompileRecorder,
        admission=admit, snapshot=snapshot, phase_command=phase_command, paper_record=paper_record, write_json=durable)
    run = FunctionType(supervisor.execute.__code__, namespace, supervisor.execute.__name__,
                       supervisor.execute.__defaults__, supervisor.execute.__closure__)
    return run(args)


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-case', type=Path)
    p.add_argument('--out', type=Path); p.add_argument('--paper-runs', type=Path)
    p.add_argument('--host-pid-namespace'); p.add_argument('--worker-phase', choices=PHASES)
    p.add_argument('--seconds', type=int, default=9600)
    p.add_argument('--case-gib', type=int, choices=(32,), default=32)
    p.add_argument('--native-seconds', type=int, default=7200)
    p.add_argument('--diagnostic-elide-final-ir', action='store_true')
    for name in ('work-dir','emit-dir','profile','metadata-template','case-root','restored-input','archive'):
        p.add_argument('--' + name, type=Path)
    args = p.parse_args(argv); args.model = MODEL; args.placement = 'optimized'
    require(180 < args.seconds <= 9600 and 0 < args.native_seconds <= 7200, 'Finite input-preparation/native compile budgets required')
    if args.worker_phase:
        require(all(getattr(args, k) is not None for k in ('work_dir','emit_dir','profile','metadata_template','case_root','restored_input')),
                'Every pure worker source and output path must be explicit')
    else:
        require(args.out is not None and args.paper_runs is not None and args.host_pid_namespace,
                'Fresh output, paper ledger and selected host namespace are required')
        args.source_case = args.source_case or paths.RESULTS_DIR / SOURCE_ID
        args.profile = args.profile or paths.PROFILES_DIR / 'qwen48/compiler-config.json'
        args.metadata_template = args.metadata_template or paths.RESULTS_DIR / 'E5d-agent-sessions-20261106/bert/acceptance/pars/export/Bert_PTAgent_DSL/config/compile_argv.json'
    for name, value in vars(args).items():
        if isinstance(value, Path): setattr(args, name, value.expanduser().resolve())
    return args


def main(argv=None):
    args = arguments(argv)
    if args.worker_phase == 'restore': return restore_worker(args)
    if args.worker_phase == 'compile':
        from .cold_rule63_worker import run
        return run(args)
    return int(execute(args)['status'] != 'ok')


if __name__ == '__main__':
    raise SystemExit(main())
