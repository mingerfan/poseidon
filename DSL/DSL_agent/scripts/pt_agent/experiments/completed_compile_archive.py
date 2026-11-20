"""Explicit reversible archives of derived files in one closed compile case.

The original result/plan JSON, model, weights and datasets are never selected.
Single files reuse R34's close/fsync/rename ordering. A bundle directory streams
to one tar.gz and is removed only after the archive and its directory are durable.
No archive is reopened for an integrity comparison.
"""

import argparse
from datetime import datetime, timezone
import fcntl
import gzip
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import stat
import tarfile
import threading
import time

from .r10_paper_metrics import write_json
from .r34_archive_text import archive_one, archive_state

GIB = 1024 ** 3
FIELDS = {'run_id', 'round', 'model', 'mode', 'git_rev', 'metrics', 'artifacts_dir', 'ts'}


def _read(path):
    return json.loads(Path(path).read_text())


def _clean_process(observation):
    return ({'exit_code', 'timed_out', 'signal', 'signals_sent', 'remaining_pids'} <= set(observation)
            and observation.get('exit_code') == 0 and observation.get('timed_out') is False
            and observation.get('signal') is None and not observation.get('signals_sent')
            and observation.get('remaining_pids') == [])


def _validate_fresh_or_official(case, result, completion, results_root):
    """Two explicit closed-case layouts; only derived IR/constants are eligible."""
    if (result.get('status') != 'ok' or result.get('error') is not None
            or completion.get('status') != 'ok'
            or completion.get('required_artifacts_persisted') is not True
            or completion.get('registry_persisted') is not True
            or completion.get('timing_status') != 'ok'
            or completion.get('overlap_and_timing_diagnostics') != []):
        raise ValueError('A complete clean terminal case is required')
    mode = result['mode']
    if mode == 'fresh-common-model-agent-e2e':
        if result.get('model') not in ('resnet20_silu', 'vgg16_silu_avgpool', 'alexnet_silu_avgpool'):
            raise ValueError('Expected an original common-model Agent run')
        stages = ('intake', 'calibration', 'agent', 'placement', 'emit', 'trace', 'compile')
        expected = dict.fromkeys(stages, 1)
        if result.get('archive_emit_constants_after_trace') is True:
            expected['archive_emit_constants'] = 1
        phases = result.get('phases') or {}
        if (result.get('attempted') != expected or result.get('completed') != expected
                or result.get('monitor_error') is not None
                or any((phases.get(name) or {}).get('status') != 'ok' for name in expected)
                or len(result.get('processes', [])) != 7
                or not all(_clean_process(row.get('observation') or {}) for row in result['processes'])
                or not all(_clean_process(phases[n].get('process') or {}) for n in ('trace', 'compile'))):
            raise ValueError('Fresh Agent run must complete every original phase cleanly')
        audit_path = Path(result.get('quality_audit') or '')
        if audit_path != case / 'post-run-quality-audit.json' or not audit_path.is_file():
            raise ValueError('The original independent Agent audit is required')
        audit = _read(audit_path)
        if (audit.get('formal_frozen') is not True or audit.get('quality_acceptance_passed') is not True
                or audit.get('strict_math_passed') is not True):
            raise ValueError('The original Agent program has not passed its freeze audit')
        emit = case / 'post-freeze' / 'emit'
        plan = Path(phases['compile']['runtime_plan'])
        trace = Path(phases['trace']['mlir']['path'])
        if trace != emit / 'candidate_trace.mlir':
            raise ValueError('Fresh traced IR must originate in this case')
        eligible = {'compiler_ir_input': trace, 'compiler_ir_final': emit / 'lowered.mlir',
                    'emit_constants': emit / 'constants.npz'}
        names = ('status', 'attempted', 'completed', 'variant', 'api_delta', 'compiled_bootstraps',
                 'predicted_latency_us', 'CKKS_latency', 'accuracy_acceptance',
                 'recorded_elapsed_seconds', 'completion_path')
    else:
        official_models = {
            'dacapo-original-resnet20-silu': 'ResNet',
            'dacapo-original-vgg16-silu-avgpool': 'VGG16',
            'dacapo-original-alexnet-silu-avgpool': 'AlexNet',
        }
        if (mode != 'original-source-native-profile-compile'
                or result.get('model') not in official_models
                or result.get('original_source_variant') is not True
                or result.get('export_contract_compatible') is not True
                or any(result.get(n) != 1 for n in ('trace_attempts', 'trace_completed',
                    'compile_attempts', 'compile_completed', 'native_compile_attempts', 'native_compile_completed'))
                or not all(_clean_process(result.get(n) or {}) for n in ('trace_process', 'compile_process'))
                or not _clean_process((result.get('compile_worker') or {}).get('process') or {})):
            raise ValueError('Official DaCapo source must complete trace, compilation and export checks')
        emit = case / 'work' / 'compile'
        plan = Path(result['plan']['path'])
        eligible = {'compiler_ir_input': case / 'work' / 'trace' / 'traced' / (official_models[result['model']] + '.mlir'),
                    'compiler_ir_final': emit / 'lowered.mlir'}
        names = ('status', 'trace_attempts', 'trace_completed', 'compile_attempts', 'compile_completed',
                 'native_compile_attempts', 'native_compile_completed', 'bootstrap_count',
                 'export_contract_compatible', 'native_execution_ready', 'predicted_latency_seconds',
                 'actual_ckks_latency_seconds', 'accuracy_acceptance', 'security_acceptance')
    if plan.parent != emit or not plan.name.endswith('.runtime-plan.json') or not plan.is_file():
        raise ValueError('Keep the original RuntimePlan in its completed compiler directory')
    rows = []
    with (results_root / 'registry.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get('artifacts_dir') == str(case) and row.get('mode') == mode:
                rows.append(row)
    if (len(rows) != 1 or set(rows[0]) != FIELDS or rows[0]['run_id'] != case.name
            or rows[0]['model'] != result['model']
            or rows[0]['metrics'] != {name: result.get(name) for name in names}):
        raise ValueError('Require the unique original terminal registry row and metrics')
    return dict(case=case, result=result, completion=completion, eligible_files=eligible,
                eligible_bundle=plan.with_name(plan.name.removesuffix('.runtime-plan.json') + '.bundle'))


def validate_completed_case(result_path, *, results_root):
    """Read metadata only and require a complete original common-model case."""
    result_path, results_root = Path(result_path).absolute(), Path(results_root).resolve()
    result_path.relative_to(results_root)
    if result_path.is_symlink() or result_path.resolve() != result_path or result_path.name != 'result.json':
        raise ValueError('Use the original regular completed-case result path')
    case = result_path.parent
    result, completion = _read(result_path), _read(case / 'completion.json')
    if result.get('mode') == 'accepted-alex-continuation-trusted-postfreeze-compile':
        return _validate_alex_postfreeze(case,result,completion,results_root)
    if result.get('mode') == 'saved-original-trace-single-compile':
        return _validate_compile_continuation(case,result,completion,results_root)
    if result.get('mode') in ('fresh-common-model-agent-e2e', 'original-source-native-profile-compile'):
        return _validate_fresh_or_official(case, result, completion, results_root)
    expected = {'emit': 1, 'trace': 1, 'compile': 1}
    if result.get('archive_emit_constants_after_trace') is True:
        expected['archive_emit_constants'] = 1
    allowed_models = {
        'pt_agent.experiments.r13_common_models:' + name
        for name in ('resnet20_silu', 'vgg16_silu_avgpool', 'alexnet_silu_avgpool')
    }
    if (result.get('status') != 'ok' or result.get('error') is not None
            or result.get('model') not in allowed_models
            or result.get('system') != 'dacapo-depth-dp'
            or result.get('attempted') != expected or result.get('completed') != expected
            or result.get('resource_monitor_error') is not None
            or completion.get('status') != 'ok'
            or completion.get('required_artifacts_persisted') is not True
            or completion.get('registry_persisted') is not True
            or completion.get('timing_status') != 'ok'
            or completion.get('overlap_and_timing_diagnostics') != []):
        raise ValueError('The original common-model source case has not completed cleanly')
    phases = result.get('phases') or {}
    if any((phases.get(name) or {}).get('status') != 'ok' for name in expected):
        raise ValueError('Every original compile phase must have a successful terminal record')
    processes = [row.get('observation') or {} for row in result.get('processes', [])]
    processes += [(phases[name].get('process') or {}) for name in ('trace', 'compile')]
    if len(result.get('processes', [])) != 3 or not all(_clean_process(row) for row in processes):
        raise ValueError('The original process tree must have exited without signals or leftovers')
    registry = []
    with (results_root / 'registry.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get('artifacts_dir') == str(case):
                registry.append(row)
    if (len(registry) != 1 or set(registry[0]) != FIELDS
            or registry[0]['metrics'] != result.get('metrics')
            or registry[0]['run_id'] != result.get('run_id')):
        raise ValueError('The closed case needs its unique original eight-field registry row')
    emit = case / 'work' / 'emit'
    plan = Path(phases['compile']['runtime_plan'])
    if plan.parent != emit or not plan.name.endswith('.runtime-plan.json') or not plan.is_file():
        raise ValueError('RuntimePlan provenance must remain in the original emit directory')
    trace = Path(phases['trace']['mlir']['path'])
    constants = Path(phases['emit']['constants'])
    if trace != emit / 'candidate_trace.mlir' or constants != emit / 'constants.npz':
        raise ValueError('Derived source paths differ from the closed original case')
    return dict(case=case, result=result, completion=completion,
                eligible_files={'compiler_ir_input': trace, 'compiler_ir_final': emit / 'lowered.mlir',
                                'emit_constants': constants},
                eligible_bundle=plan.with_name(plan.name.removesuffix('.runtime-plan.json') + '.bundle'))


def _validate_compile_continuation(case,result,completion,results_root):
    """Keep both source cases closed when archiving a reused original trace."""
    from .dacapo_compile_continuation import source_gate
    if (result.get('status')!='ok' or result.get('error') is not None
            or result.get('original_trace_reused') is not True or result.get('trace_attempts')!=0
            or any(result.get(k)!=1 for k in ('compile_attempts','compile_completed','native_compile_attempts','native_compile_completed'))
            or result.get('export_contract_compatible') is not True or result.get('guard_trigger') is not None
            or completion.get('status')!='ok' or completion.get('registry_persisted') is not True
            or completion.get('required_artifacts_persisted') is not True
            or completion.get('timing_status')!='ok' or completion.get('overlap_and_timing_diagnostics')!=[]
            or len(result.get('processes',[]))!=1
            or not _clean_process(result['processes'][0].get('observation')or{})
            or not _clean_process((result.get('compile_worker')or{}).get('process')or{})):
        raise ValueError('Require one clean completed original compile continuation')
    emit=case/'emit';saved=result.get('source')or{}
    source=source_gate(saved['original_case'],emit/'lowered.mlir')
    if source!=saved:raise ValueError('Closed original trace provenance differs')
    plan=Path(result['plan']['path'])
    if plan.parent!=emit or not plan.name.endswith('.runtime-plan.json') or not plan.is_file():
        raise ValueError('Keep the original completed RuntimePlan in its compiler case')
    rows=[]
    with (results_root/'registry.jsonl').open()as stream:
        for line in stream:
            row=json.loads(line)
            if row.get('artifacts_dir')==str(case):rows.append(row)
    names=('status','compile_attempts','compile_completed','native_compile_attempts','native_compile_completed',
        'bootstrap_count','export_contract_compatible','actual_ckks_latency_seconds','security_acceptance')
    if (len(rows)!=1 or set(rows[0])!=FIELDS or rows[0]['run_id']!=case.name
            or rows[0]['model']!=result['model'] or rows[0]['mode']!=result['mode']
            or rows[0]['metrics']!={k:result.get(k)for k in names}):
        raise ValueError('Require the unique original continuation registry row')
    return dict(case=case,result=result,completion=completion,
        eligible_files={'compiler_ir_input':Path(source['trace_metadata']['path']),'compiler_ir_final':emit/'lowered.mlir'},
        eligible_bundle=plan.with_name(plan.name.removesuffix('.runtime-plan.json')+'.bundle'))


def _validate_alex_postfreeze(case,result,completion,results_root):
    """Reuse the exact continued freeze/compile gate; keep model and reports intact."""
    from .r30_alex_postfreeze import compiled_source
    if (result.get('status')!='ok' or result.get('error') is not None
            or completion.get('status')!='ok'
            or completion.get('required_artifacts_persisted') is not True
            or completion.get('registry_persisted') is not True
            or completion.get('timing_status')!='ok'
            or completion.get('overlap_and_timing_diagnostics')!=[]):
        raise ValueError('Require one clean closed continued Alex compilation')
    gate=compiled_source(case)
    if Path(gate['compile_case'])!=case or gate.get('clean_native_compile') is not True:
        raise ValueError('The frozen continuation compiler source is unresolved')
    phases=result.get('phases') or {};emit=case/'post-freeze/emit'
    plan=Path((phases.get('compile') or {}).get('runtime_plan',''))
    trace=Path((phases.get('trace') or {}).get('mlir',{}).get('path',''))
    if (plan.parent!=emit or not plan.name.endswith('.runtime-plan.json') or not plan.is_file()
            or trace!=emit/'candidate_trace.mlir'
            or Path(gate.get('compiled_input_logical_path',''))!=trace):
        raise ValueError('Keep the current frozen program trace and RuntimePlan in their original case')
    names=('status','attempted','completed','compiled_bootstraps','predicted_latency_us',
           'incoming_signal','guard_trigger','source_case')
    rows=[]
    with (results_root/'registry.jsonl').open() as stream:
        for line in stream:
            row=json.loads(line)
            if row.get('artifacts_dir')==str(case):rows.append(row)
    if (len(rows)!=1 or set(rows[0])!=FIELDS or rows[0]['run_id']!=case.name
            or rows[0]['model']!=result.get('model') or rows[0]['mode']!=result.get('mode')
            or rows[0]['metrics']!={k:result.get(k) for k in names}):
        raise ValueError('Require the unique original continued Alex compiler registry row')
    return dict(case=case,result=result,completion=completion,
        eligible_files={'compiler_ir_input':trace,'compiler_ir_final':emit/'lowered.mlir'},
        eligible_bundle=plan.with_name(plan.name.removesuffix('.runtime-plan.json')+'.bundle'))


def file_metadata(path, relative):
    """Reject links and special files without reading their contents."""
    path = Path(path)
    st = path.lstat()
    if path.is_symlink() or path.resolve() != path.absolute():
        raise ValueError('Symbolic paths are excluded from derived archives')
    if stat.S_ISREG(st.st_mode):
        if st.st_nlink != 1:
            raise ValueError('Hard-linked files are excluded from derived archives')
        kind = 'file'
    elif stat.S_ISDIR(st.st_mode):
        kind = 'directory'
    else:
        raise ValueError('Only regular files and directories are eligible')
    return dict(path=str(relative), kind=kind, bytes=st.st_size if kind == 'file' else 0,
                mtime_ns=st.st_mtime_ns, mode=stat.S_IMODE(st.st_mode),
                inode=st.st_ino, device=st.st_dev)


def tree_inventory(source):
    source = Path(source).absolute()
    entries = [file_metadata(source, source.name)]
    if entries[0]['kind'] != 'directory':
        raise ValueError('Runtime bundle must be a directory')
    for parent, directories, files in os.walk(source, followlinks=False):
        for name in sorted(directories + files):
            path = Path(parent) / name
            entries.append(file_metadata(path, path.relative_to(source.parent)))
    if not any(x['kind'] == 'file' for x in entries):
        raise ValueError('An empty bundle is not a derived runtime payload')
    return entries


def validate_manifest(closed, rows):
    """Restrict an explicit inventory to the derived outputs of the closed case."""
    if not rows or len({row['kind'] for row in rows}) != len(rows):
        raise ValueError('Choose a nonempty unique set of derived output roles')
    selected = []
    for row in rows:
        kind = row['kind']
        if kind == 'runtime_bundle':
            source = closed['eligible_bundle']
            target = source.with_name(source.name + '.tar.gz')
            inventory = tree_inventory(source)
        elif kind in closed['eligible_files']:
            source = closed['eligible_files'][kind]
            target = source.with_name(source.name + '.gz')
            inventory = [file_metadata(source, source.name)]
            if inventory[0]['kind'] != 'file':
                raise ValueError('A derived single-file role must remain a regular file')
        else:
            raise ValueError('Original model, weights, data and result JSON are excluded')
        if Path(row['path']).absolute() != source or row['inventory'] != inventory:
            raise ValueError('Explicit source path or saved size/mtime inventory changed')
        temporary = target.with_name(target.name + '.partial')
        if target.exists() or target.is_symlink() or temporary.exists() or temporary.is_symlink():
            raise ValueError('Preserve any earlier archive or partial output')
        restore = (['tar', '--extract', '--gzip', '--file', str(target), '--directory', str(source.parent),
                    '--no-same-owner', '--no-same-permissions'] if kind == 'runtime_bundle'
                   else ['gzip', '-dk', '--', str(target)])
        selected.append(dict(kind=kind, source=source, target=target, temporary=temporary,
                             inventory=inventory, restore_argv=restore))
    return selected


def archive_tree(source, target, temporary, inventory):
    """One streamed tar.gz; commit and fsync it before deleting any source entry."""
    source, target, temporary = Path(source), Path(target), Path(temporary)
    began = time.perf_counter_ns()
    raw_bytes = sum(row['bytes'] for row in inventory)
    with temporary.open('xb') as sink:
        with gzip.GzipFile(filename=source.name + '.tar', mode='wb', compresslevel=1, fileobj=sink) as encoded:
            with tarfile.open(fileobj=encoded, mode='w|', bufsize=1024 * 1024) as archive:
                for row in inventory:
                    relative = Path(row['path'])
                    if relative.is_absolute() or '..' in relative.parts or relative.parts[0] != source.name:
                        raise ValueError('Archive members must stay under the original bundle basename')
                    path = source.parent / relative
                    if file_metadata(path, relative) != row:
                        raise ValueError('Bundle size/mtime metadata changed while streaming')
                    header = archive.gettarinfo(str(path), arcname=relative.as_posix())
                    if row['kind'] == 'file':
                        with path.open('rb') as stream:
                            archive.addfile(header, stream)
                    else:
                        archive.addfile(header)
        sink.flush()
        os.fsync(sink.fileno())
    compressed_bytes = temporary.stat().st_size
    if compressed_bytes >= raw_bytes:
        temporary.unlink()
        return dict(source=str(source), status='kept_uncompressed', raw_bytes=raw_bytes,
                    compressed_bytes=None, released_bytes=0, elapsed_ns=time.perf_counter_ns()-began)
    temporary.rename(target)
    directory = os.open(str(target.parent), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
        if tree_inventory(source) != inventory:
            raise ValueError('Bundle inventory changed before source-directory removal')
        shutil.rmtree(source)
        os.fsync(directory)
    finally:
        os.close(directory)
    return dict(source=str(source), archive=str(target), status='compressed', raw_bytes=raw_bytes,
                compressed_bytes=compressed_bytes, released_bytes=raw_bytes-compressed_bytes,
                restore_argv=['tar', '--extract', '--gzip', '--file', str(target),
                              '--directory', str(source.parent), '--no-same-owner', '--no-same-permissions'],
                elapsed_ns=time.perf_counter_ns()-began)


def _metadata_size(path):
    total = 0
    for parent, _, files in os.walk(path):
        for name in files:
            try:
                total += (Path(parent) / name).stat().st_size
            except FileNotFoundError:
                # The recorder atomically renames its temporary JSON files.
                pass
    return total


def _durable_json(path, record):
    write_json(path, record)
    with Path(path).open('rb') as stream:
        os.fsync(stream.fileno())
    descriptor = os.open(str(Path(path).parent), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main(argv=None):
    from .. import paths
    from ..registry import git_rev
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--artifact-gib', type=int, default=12, choices=range(1, 13))
    parser.add_argument('--seconds', type=int, default=900)
    parser.add_argument('--minimum-free-gib',type=int,default=10,choices=range(4,11))
    args = parser.parse_args(argv)
    if not 1 <= args.seconds <= 1800:
        parser.error('Use a finite archive allowance of1..1800seconds')
    source_git_rev = git_rev()
    memory = 512 * 1024 ** 2
    _, hard = resource.getrlimit(resource.RLIMIT_AS)
    memory = memory if hard == resource.RLIM_INFINITY else min(memory, hard)
    resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
    began = time.perf_counter_ns()
    manifest = _read(args.manifest)
    closed = validate_completed_case(manifest['case_result'], results_root=paths.RESULTS_DIR)
    selected = validate_manifest(closed, manifest['files'])
    args.out = args.out.absolute()
    args.out.relative_to(paths.RESULTS_DIR.resolve())
    if args.out == closed['case'] or closed['case'] in args.out.parents:
        raise ValueError('Archive records must live outside the immutable original case result directory')
    args.out.mkdir(exist_ok=False)
    _durable_json(args.out / 'manifest.json', manifest)
    shutil.copyfile(Path(__file__), args.out / 'driver.py')
    result = dict(round='R34-completed', status='running', error=None, source_case=str(closed['case']),
                  files=[], selected_files=len(selected), compressed_artifact_allowance_bytes=args.artifact_gib*GIB,
                  address_space_limit_bytes=memory, disk_reserve_bytes=args.minimum_free_gib*GIB, seconds=args.seconds,
                  terminal_metadata_reserve_bytes=1024**2,
                  source_result_unchanged=True, integrity_checks=0, model_runs=0, compiler_runs=0,
                  keygen_runs=0, ckks_runs=0, llm_calls=0, downloads=0, installations=0)
    stop = threading.Event()
    guard_errors = []

    def abort(signum, frame):
        raise RuntimeError('Archive interrupted by signal' + str(signum))

    previous = {s:signal.signal(s, abort) for s in (signal.SIGTERM, signal.SIGINT)}

    def observe():
        amount = _metadata_size(args.out)
        for item in selected:
            for key in ('target', 'temporary'):
                try:
                    amount += item[key].stat().st_size
                except FileNotFoundError:
                    # Archive commit renames .partial to the final target.
                    pass
        process = {}
        for line in Path('/proc/self/status').read_text().splitlines():
            key, _, value = line.partition(':')
            if key in ('VmRSS', 'VmHWM', 'VmSize'):
                process[key + '_kib'] = int(value.split()[0])
        return dict(clock_ns=time.perf_counter_ns(), compressed_and_record_bytes=amount,
                    free_bytes=shutil.disk_usage(paths.RESULTS_DIR).free, process=process)

    def guard():
        while not stop.wait(.1):
            try:
                row = observe()
                with (args.out / 'resources.jsonl').open('a') as stream:
                    stream.write(json.dumps(row)+'\n')
                if (row['compressed_and_record_bytes'] > args.artifact_gib*GIB - 1024**2
                        or row['free_bytes'] < args.minimum_free_gib*GIB
                        or (time.perf_counter_ns()-began)/1e9 > args.seconds):
                    raise RuntimeError('Archive disk, reserve or wall allowance reached')
            except BaseException as exc:
                guard_errors.append(f'{type(exc).__name__}: {exc}')
                os.kill(os.getpid(), signal.SIGTERM)
                return

    watcher = threading.Thread(target=guard, daemon=True)
    active = None
    try:
        initial = observe()
        if initial['free_bytes'] < args.minimum_free_gib*GIB or initial['compressed_and_record_bytes'] > args.artifact_gib*GIB - 1024**2:
            raise RuntimeError('Archive resource allowance unavailable before streaming')
        watcher.start()
        for item in selected:
            if guard_errors:
                raise RuntimeError(guard_errors[0])
            # Persist the recovery command before the operation can remove any source.
            active = item
            result['in_progress'] = {**archive_state(item['source'], item['target'], item['temporary']),
                                     'kind':item['kind'], 'restore_argv':item['restore_argv']}
            _durable_json(args.out / 'progress.json', result)
            if item['kind'] == 'runtime_bundle':
                row = archive_tree(item['source'], item['target'], item['temporary'], item['inventory'])
            else:
                if file_metadata(item['source'], item['source'].name) != item['inventory'][0]:
                    raise ValueError('Single-file metadata changed before compression')
                row = archive_one(item['source'], item['target'], item['temporary'])
            row['kind'] = item['kind']
            result['files'].append(row)
            result['in_progress'] = None
            write_json(args.out / 'progress.json', result)
            active = None
        if guard_errors:
            raise RuntimeError(guard_errors[0])
        result['status'] = 'ok'
    except BaseException as exc:
        result.update(status='error', error=f'{type(exc).__name__}: {exc}')
        if active is not None:
            result['interrupted_file'] = {**archive_state(active['source'], active['target'], active['temporary']),
                                          'kind':active['kind'], 'restore_argv':active['restore_argv']}
    finally:
        for signum in previous:
            signal.signal(signum, signal.SIG_IGN)
        stop.set()
        if watcher.is_alive():
            watcher.join(timeout=2)
        result.update(elapsed_ns=time.perf_counter_ns()-began,
                      max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      completed_files=len(result['files']),
                      released_bytes=sum(x['released_bytes'] for x in result['files']),
                      guard_errors=guard_errors, ended_utc=datetime.now(timezone.utc).isoformat())
        for _ in range(12):
            write_json(args.out / 'result.json', result)
            final = observe()
            # The clock is not part of the byte fixed point.
            final.pop('clock_ns')
            previous_bytes = (result.get('final_resource_observation') or {}).get('compressed_and_record_bytes')
            result['final_resource_observation'] = final
            if previous_bytes == final['compressed_and_record_bytes']:
                break
        if (final['compressed_and_record_bytes'] > args.artifact_gib*GIB
                or final['free_bytes'] < args.minimum_free_gib*GIB):
            result['final_resource_error'] = 'Final archive byte accounting or disk reserve exceeded'
            if result['status'] == 'ok':
                result.update(status='error', error=result['final_resource_error'])
        _durable_json(args.out / 'result.json', result)
        row = dict(run_id=args.out.name, round='R34-completed', model=closed['result']['model'],
                   mode='reversible-completed-case-archive', git_rev=source_git_rev,
                   metrics={k:result[k] for k in ('status', 'completed_files', 'released_bytes', 'elapsed_ns', 'max_rss_kib')},
                   artifacts_dir=str(args.out), ts=result['ended_utc'])
        with (paths.RESULTS_DIR / 'registry.jsonl').open('a') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            stream.write(json.dumps(row, allow_nan=False)+'\n')
            stream.flush()
            fcntl.flock(stream, fcntl.LOCK_UN)
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    print(json.dumps(dict(status=result['status'],out=str(args.out),released_bytes=result['released_bytes'])))
    return 0 if result['status'] == 'ok' else 1


if __name__ == '__main__':
    raise SystemExit(main())
