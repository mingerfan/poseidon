"""Repeat ResNet Rule63 search at the current 14GiB/7200s policy.

Reuse the exact closed A2b program. If its raw input was archived, recover one
new-case copy with the existing bounded raw-DEFLATE reader. The original A2b
and historical 6GiB crash records remain unchanged. No new model execution.
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
from . import bert_rule63_shared_optimized_compile as bridge
from . import cold_rule63_whole_model as supervisor
from .completion_metrics import artifact_bytes
from .r10_paper_metrics import StageRecorder, write_json

ROUND = 'A5R-RESNET-SHARED-RULE63-COMPILE'
MODE = 'optimized-native-compile-only-shared-A2b-resnet-rule63-rss14'
MODULE = 'pt_agent.experiments.a5r_resnet_rule63_shared_optimized_compile'
MODEL = 'resnet20_silu'
SOURCE_ID = 'a2b-resnet20-rule63-fresh-cold-depth-20261117'
ARCHIVE_ID = SOURCE_ID + '-raw-npz-resource-archive'
HISTORICAL_RUN_ID = 'local-compile-optimized-r29-20261116-dacapo-optimized'
PHASES = ('restore', 'compile')
read, require = shared.read, shared.require


def private(function, **replacements):
    namespace = dict(function.__globals__); namespace.update(replacements)
    selected = FunctionType(function.__code__, namespace, function.__name__,
                            function.__defaults__, function.__closure__)
    selected.__kwdefaults__ = function.__kwdefaults__
    return selected


def source_gate(args, *, created=False):
    """Read original scientific/archive metadata and filesystem stat only."""
    current = supervisor.admission(args, created=created)
    root = paths.RESULTS_DIR.resolve(); source = args.source_case
    require(args.out.parent == root and source == root / SOURCE_ID and source != args.out,
            'Only the exact ResNet A2b source and a fresh direct destination are eligible')
    require(args.host_pid_namespace == os.readlink('/proc/self/ns/pid'), 'Use the selected host namespace')
    result, completion = read(source / 'result.json'), read(source / 'completion.json')
    bridge.durable_closed(source, result, completion)
    require(result.get('round') == supervisor.ROUND and result.get('mode') == supervisor.MODE
            and result.get('run_id') == source.name and result.get('model') == MODEL
            and result.get('placement') == 'depth' and result.get('monitor_stopped') is True
            and not result.get('guard_error') and not result.get('monitor_error')
            and result.get('expected_bootstraps_match') is True,
            'A complete successful ResNet A2b depth case is required')
    expected = dict.fromkeys(supervisor.PHASES, 1)
    phases = result.get('phases') or {}; processes = result.get('processes') or []
    require(result.get('attempted') == expected and result.get('completed') == expected
            and set(phases) == set(supervisor.PHASES)
            and all(phases[p].get('status') == 'ok' for p in supervisor.PHASES)
            and [p.get('phase') for p in processes] == list(supervisor.PHASES)
            and all(supervisor.clean(p.get('observation') or {}) for p in processes),
            'All six original scientific phases must be cleanly complete')
    generation = phases['generation']; math = generation.get('training_math') or {}
    count = generation.get('raw_Count', {}).get('stats', {}); emit = phases['emit'].get('emit_stats') or {}
    require(generation.get('rule_options') == shared.OPTIONS
            and count.get('counts', {}).get('boot') == 0
            and all(count.get(k) == emit.get(k) for k in ('counts', 'ops_total', 'distinct_consts'))
            and generation.get('signature', {}).get('passed') is True
            and math.get('samples') == 3 and len(math.get('checks', [])) == 3
            and all(row.get('passed') is True for row in math['checks']),
            'Fixed Rule63, Count/Emit, signature or original training mathematics evidence differs')
    require(read(source / 'source/compiler-profile.json') == read(args.profile)
            and read(source / 'source/compile-metadata-template.json') == read(args.metadata_template)
            and Path(result['source']['compiler']).resolve() == Path(current['compiler']).resolve(),
            'Retain the same full qwen48 profile, native recipe and compiler build path')
    native = phases['compile'].get('native_process') or {}
    require(supervisor.clean(native) and phases['compile'].get('compiled_bootstraps') == 230
            and phases['compile'].get('runtime_plan') == result.get('runtime_plan')
            and Path(result.get('runtime_plan', '')).is_file(),
            'Original accepted ResNet depth artifact must remain the230-bootstrap control')
    starts = (completion.get('started_ns'), native.get('started_ns'), native.get('ended_ns'), completion.get('ended_ns'))
    require(all(type(v) is int for v in starts) and 0 <= starts[0] < starts[1] < starts[2] <= starts[3],
            'Original same-host generation-prefix clocks must be ordered')
    require(processes[-1]['observation']['started_ns'] <= starts[1] < starts[2]
            <= processes[-1]['observation']['ended_ns'], 'Native clock must lie inside its original worker')
    with (root / 'registry.jsonl').open() as stream:
        fcntl.flock(stream, fcntl.LOCK_SH)
        records = [json.loads(line) for line in stream if line.strip()]
    ledger = bridge.unique_registry(source, result, records)
    require(all(ledger.get(k) == result[k] for k in ('round', 'model', 'mode', 'git_rev')),
            'Original scientific registry identity differs')
    births = shared.closed_source.producer_births(source)
    require(births > 0, 'Original closed producer births are required')
    raw = source / 'work/emit/candidate_trace.mlir'; archive = raw.with_name(raw.name + '.gz')
    projection = phases['trace'].get('trace_bytes')
    require(type(projection) is int and 0 < projection <= 16 * supervisor.GIB,
            'Original trace projection must fit the16GiB file allowance')
    proof = dict(current_metadata=current, source_case=str(source), source_result=str(source/'result.json'),
        source_completion=str(source/'completion.json'), source_parameter_group='ours-N16-boot5',
        source_rule_options=shared.OPTIONS, source_registry_run_id=source.name,
        closed_source_producer_births=births, host_pid_namespace=args.host_pid_namespace,
        source_math_status='Original A2b three training Tensor/Mock checks passed; no new quality evaluation',
        source_independent_math_passed=True, quality_retested=False, source_payload_reads=0,
        shared_generation_prefix_ns=starts[1]-starts[0],
        clock_evidence=dict(source_entry_ns=starts[0], source_native_compile_started_ns=starts[1],
                            source_native_compile_ended_ns=starts[2], source_case_ended_ns=starts[3]),
        source_clock_scope='Original A2b entry to native compiler start; prior native compile excluded',
        original_trace_size_projection_bytes=projection, decoded_size_identity_comparison=False,
        native_raw=str(args.out/'input/candidate_trace.mlir'))
    if raw.is_file() and not raw.is_symlink() and raw.resolve() == raw:
        proof.update(input_kind='raw-symlink', input_source=str(raw), input_source_stat=shared.raw_stat(raw),
                     archive_case=None, projected_new_raw_bytes=0)
    else:
        require(not os.path.lexists(raw) and archive.is_file() and not archive.is_symlink()
                and archive.resolve() == archive, 'Closed raw or its exact regular archive is required')
        archive_case = root / ARCHIVE_ID
        ar, ac = read(archive_case/'result.json'), read(archive_case/'completion.json')
        bridge.durable_closed(archive_case, ar, ac)
        archive_ledger = bridge.unique_registry(archive_case, ar, records)
        selected = [r for r in ar.get('files', []) if r.get('source') == str(raw)]
        require(ar.get('round') == 'A-GROUP-COLD-CLOSED-RESOURCE-ARCHIVE'
                and ar.get('source_case') == str(source) and ar.get('scientific_status') == 'ok'
                and ar.get('resource_status') == 'archived_or_kept'
                and archive_ledger.get('model') == source.name
                and archive_ledger.get('mode') == 'resource-only-reversible-gzip'
                and len(selected) == 1 and selected[0].get('status') == 'compressed'
                and selected[0].get('archive') == str(archive)
                and selected[0].get('raw_bytes') == projection
                and selected[0].get('compressed_bytes') == archive.stat().st_size,
                'Exact original raw archive producer provenance differs')
        archive_producer_proof = None
        if getattr(args, 'archive_supervisor_case', None) is not None:
            from .archive_producer_closure import verify_archive_producer
            archive_producer_proof = verify_archive_producer(archive_case, args.archive_supervisor_case,
                host_pid_namespace=args.host_pid_namespace)
            archive_births = archive_producer_proof['closed_producer_births']
        else:
            archive_births = shared.closed_source.producer_births(archive_case)
        require(archive_births > 0, 'Closed archive producer births are required')
        require(artifact_bytes(args.out) + projection < args.case_gib * supervisor.GIB
                and shutil.disk_usage(root).free >= projection + 10 * supervisor.GIB,
                'resource_limit: restored input projection plus free10GiB required')
        proof.update(input_kind='archived-input-preparation', input_source=str(archive),
            input_source_stat=shared.raw_stat(archive), archive_case=str(archive_case),
            archive_result=str(archive_case/'result.json'), archive_completion=str(archive_case/'completion.json'),
            closed_archive_producer_births=archive_births, archive_producer_proof=archive_producer_proof,
            projected_new_raw_bytes=projection)
    return proof


expected_native_argv = bridge.expected_native_argv
terminal_source_observation = bridge.terminal_source_observation
restore_worker = private(bridge.restore_worker, MODEL=MODEL)
phase_command = private(bridge.phase_command, MODULE=MODULE)


def prepare_source(args, proof):
    target = args.out/'source'; target.mkdir()
    for file in (Path(__file__), Path(bridge.__file__), Path(shared.__file__), Path(supervisor.__file__),
                 Path(__file__).with_name('cold_rule63_worker.py'), Path(__file__).with_name('bounded_gzip_input.py'),
                 Path(__file__).with_name('a_group_profile_sum_policy.py'),
                 Path(__file__).with_name('archive_producer_closure.py')):
        shutil.copyfile(file, target/file.name)
    shutil.copyfile(args.profile, target/'compiler-profile.json')
    shutil.copyfile(args.metadata_template, target/'compile-metadata-template.json')
    (args.out/'input').mkdir()
    require(shared.raw_stat(proof['input_source']) == proof['input_source_stat'], 'Original source input stat changed')
    if proof['input_kind'] == 'raw-symlink': Path(proof['native_raw']).symlink_to(proof['input_source'])
    write_json(target/'shared-generation-source.json', proof)
    write_json(target/'expected-native-argv.json', expected_native_argv(args, proof))


def paper_record(args, result, completion, timing):
    row = bridge.paper_record(args, result, completion, timing)
    source = result.get('source') or {}; prefix = source.get('shared_generation_prefix_ns')
    prefix_s = prefix/1e9 if prefix is not None else None
    derived = prefix_s + row['compile_s'] if prefix_s is not None and row['compile_s'] is not None else None
    row.update(task='A5r', model='resnet20', generation_e2e_s=derived, derived_e2e_s=derived,
        estimated_from_shared_generation=True, shared_generation_prefix_s=prefix_s,
        source_generation_clock=source.get('clock_evidence'),
        generation_e2e_scope='Derived original A2b generation prefix plus this native compiler child; recovery and new orchestration excluded',
        actual_compile_only_scope='New case input preparation plus native compilation and required persistence',
        derived_stages_s=dict(shared_generation_prefix=prefix_s, placement_compile=row['compile_s']),
        source_independent_math_passed=source.get('source_independent_math_passed'),
        source_math_scope='Original A2b three training Tensor/Mock checks; no new mathematical or quality evaluation',
        compile_attempts=(1 if (result.get('phases', {}).get('compile') or {}).get('native_process')
                          else None if result.get('attempted', {}).get('compile') else 0),
        peak_rss_gib=private(supervisor.peak_rss, PHASES=PHASES)(args),
        supersedes_after_real_terminal=HISTORICAL_RUN_ID)
    return row


class CompileRecorder(StageRecorder):
    def span(self, stage_id, **metadata):
        return super().span('shared_ResNet_rule63_input_and_compile' if stage_id == 'fresh_cold_rule63' else stage_id, **metadata)


def execute(args):
    state = {}
    def admit(current, *, created=False):
        state['source'] = source_gate(current, created=created)
        return state['source']
    def snapshot(current): prepare_source(current, state['source'])
    def durable(path, value):
        if Path(path).parent == args.out and Path(path).name in ('result.json', 'progress.json'):
            value.update(cold=False, estimated_from_shared_generation=True,
                source_generation_reused=True, original_intake_reused=True,
                training_calibration_reused=True, source_case=str(args.source_case),
                actual_clock_scope='New ResNet input preparation and native compile only; derived A2b-generation time reported separately')
            if Path(path).name == 'result.json': terminal_source_observation(args, value, state.get('source'))
        write_json(path, value)
    return private(supervisor.execute, ROUND=ROUND, MODE=MODE, PHASES=PHASES,
        StageRecorder=CompileRecorder, admission=admit, snapshot=snapshot,
        phase_command=phase_command, paper_record=paper_record, write_json=durable)(args)


def arguments(argv=None):
    metadata = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    metadata.add_argument('--archive-supervisor-case', type=Path)
    extra, remaining = metadata.parse_known_args(sys.argv[1:] if argv is None else argv)
    args = private(bridge.arguments, MODEL=MODEL, SOURCE_ID=SOURCE_ID, __doc__=__doc__)(remaining)
    args.archive_supervisor_case = (extra.archive_supervisor_case.expanduser().resolve()
        if extra.archive_supervisor_case is not None else None)
    args.model = MODEL; args.short_model = 'resnet20'
    if not args.worker_phase:
        args.inputs = paths.WORK_RESULTS/'agent-cifar-offline-records-r1804/cifar10-records'
        args.selections = paths.WORK_RESULTS/'agent-resnet20-original-accuracy-r1805/frozen-selections.json'
        args.frontend = paths.DACAPO_ROOT/'python/hecate/hecate/expr.py'
    return args


def main(argv=None):
    args = arguments(argv)
    if args.worker_phase == 'restore': return restore_worker(args)
    if args.worker_phase == 'compile':
        from .cold_rule63_worker import run
        return run(args)
    return int(execute(args)['status'] != 'ok')


if __name__ == '__main__': raise SystemExit(main())
