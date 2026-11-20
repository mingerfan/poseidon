"""Bounded official Orbit metadata parsing alongside one identified BERT tune.

This explicit exception retains the original initializer lock and validates the
existing heavy-lease owner. It never acquires or releases that owner's lease.
Only this new case's descendants enter the existing R10 process supervisor.
"""
import argparse
import os
from pathlib import Path
import shutil
from types import FunctionType, SimpleNamespace

from .. import paths
from . import orbit_backend_parse_only as original
from . import orbit_common_backend_parse as common
from .r10_paper_metrics import _scan

ROUND = 'ORBIT-STATIC-PARSE-CPU-OVERLAP'
MODE = 'official-static-parser-only-during-identified-bert-tune3'
GIB = 1024**3
TOTAL_SECONDS = original.TOTAL_SECONDS
PHASE_SECONDS = original.PHASE_SECONDS
CLIENT = original.CLIENT
ZERO_FIELDS = original.ZERO_FIELDS
SCOPE = ('Restricted metadata-only parallel exception during the identified BERT '
         'tune3; original official parser/scheduler, no model arithmetic, external '
         'constant/sample payload, FHE context, keys or API. Timing overlaps CPU tune.')
COMPILERS = frozenset(('hecate-opt', 'heir-opt', 'cc1', 'cc1plus', 'collect2',
    'ld', 'ld.lld', 'lld', 'lld-link', 'gcc', 'g++', 'clang', 'clang++',
    'compile', 'asm', 'link', 'go', 'cgo'))


def read(path):
    return original.read(path)


def _same(left, right):
    return isinstance(left, (str, Path)) and Path(left).resolve() == Path(right).resolve()


def _family(table, pid):
    found = {pid}
    while True:
        expanded = found | {key for key, row in table.items() if row.get('ppid') in found}
        if expanded == found:
            return found
        found = expanded


def _process_name(pid):
    """Read only the public executable name, never arguments or environment."""
    try:
        return (Path('/proc')/str(pid)/'comm').read_text().strip()
    except (FileNotFoundError, ProcessLookupError):
        return None


def _lease_owners(path, text=None):
    """Match Linux flock's public PID/device/inode record to this exact lease."""
    stat = path.stat()
    device = (os.major(stat.st_dev), os.minor(stat.st_dev), stat.st_ino)
    owners = []
    for line in (Path('/proc/locks').read_text() if text is None else text).splitlines():
        fields = line.split()
        if len(fields) < 8 or fields[1:4] != ['FLOCK', 'ADVISORY', 'WRITE']:
            continue
        major, minor, inode = fields[5].split(':')
        if (int(major, 16), int(minor, 16), int(inode)) == device:
            owners.append(int(fields[4]))
    return owners


def _metadata_admission(args):
    """Only session/queue summaries are retained; request bodies are untouched."""
    state = read(args.bert_case/'session/SESSION.json')
    if (state.get('phase') != 'tune' or type(state.get('iteration')) is not int
            or state['iteration'] != 3 or state.get('failed') is not None
            or type(state.get('calls')) is not int or state['calls'] != 3):
        raise ValueError('The original BERT session must still be tune3/calls3 with no failure')
    run = read(args.bert_run_case/'result.json')
    if (run.get('round') != 'BERT-SERIAL-CONTINUATION' or run.get('model') != 'bert-tiny-sst2'
            or run.get('status') != 'running' or run.get('termination') is not None
            or not _same((run.get('source') or {}).get('source_case'), args.bert_case)
            or (run.get('attempted') or {}).get('step') != 1
            or (run.get('completed') or {}).get('step') != 0):
        raise ValueError('The identified BERT continuation must retain its active original step')
    supervisor = read(args.batch_case/'supervisor.json')
    process = supervisor.get('process') or {}
    if (supervisor.get('pid') != args.batch_pid or process.get('pid') != args.batch_pid
            or process.get('starttime') != args.batch_starttime):
        raise ValueError('Saved original batch supervisor identity differs')
    active = read(args.batch_case/'active.json')
    argv = active.get('argv') or []
    if (active.get('status') != 'running' or not _same(active.get('out'), args.bert_run_case)
            or argv.count('pt_agent.experiments.bert_serial_continuation') != 1
            or argv.count('--source-case') != 1
            or not _same(argv[argv.index('--source-case')+1], args.bert_case)):
        raise ValueError('Original batch must still own the identified BERT continuation')
    if (args.batch_case/'result.json').exists():
        raise ValueError('Original heavy batch has already written its terminal result')
    return dict(phase='tune', iteration=3, calls=3, failed=None,
        bert_case=str(args.bert_case), bert_run_case=str(args.bert_run_case),
        batch_case=str(args.batch_case), batch_job=active.get('id'))


def parallel_observation(args, sample, available, free):
    """Fail closed on identity/namespace uncertainty; never signal BERT."""
    reasons = []
    rss = None if sample is None else sample['subtree_rss_bytes']
    if rss is not None and rss > 4*GIB:
        reasons.append('Own child subtree RSS exceeds4GiB')
    if available < 2*GIB:
        reasons.append('Host available RAM below2GiB')
    if free < 10*GIB:
        reasons.append('Free disk below10GiB')
    summary = None
    foreign = []
    owners = []
    supervisor_rss = None
    try:
        summary = _metadata_admission(args)
        table = _scan()
        for label, pid, birth in (('BERT worker', args.bert_pid, args.bert_starttime),
                                 ('Heavy batch', args.batch_pid, args.batch_starttime)):
            row = table.get(pid) or {}
            if (row.get('starttime') != birth or row.get('state') not in ('R', 'S', 'D')
                    or row.get('pid') != pid):
                raise ValueError(label+' identity is missing, changed or not live; host namespace is required')
        if args.bert_pid not in _family(table, args.batch_pid):
            raise ValueError('BERT worker no longer belongs to the original heavy batch')
        owners = _lease_owners(paths.RESULTS_DIR/'autonomous-heavy-experiments.lock')
        if owners != [args.batch_pid]:
            raise ValueError('The original batch must remain the unique actual heavy-lease owner')
        own = _family(table, os.getpid())
        if os.getpid() not in table:
            raise ValueError('Own supervisor is absent from the same host process namespace')
        supervisor_rss = table[os.getpid()].get('vmrss_kib', 0)*1024
        current_tree = sum(table[pid].get('vmrss_kib', 0)*1024 for pid in own if pid in table)
        rss = max(current_tree, (rss or 0)+supervisor_rss)
        if rss > 4*GIB and 'Own child subtree RSS exceeds4GiB' not in reasons:
            reasons.append('Own supervisor and descendant RSS exceeds4GiB')
        for pid, row in table.items():
            if pid in own or row.get('state') in ('Z', 'X'):
                continue
            name = _process_name(pid)
            if name in COMPILERS:
                foreign.append(dict(pid=pid, starttime=row.get('starttime'), name=name))
        if foreign:
            reasons.append('Another native compiler is active outside this parser task')
    except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
        reasons.append(type(exc).__name__+': '+str(exc)[:1200])
    return dict(stop=bool(reasons), reasons=reasons, subtree_rss_bytes=rss,
        supervisor_rss_bytes=supervisor_rss,
        child_sample_subtree_rss_bytes=None if sample is None else sample['subtree_rss_bytes'],
        available_ram_bytes=available, free_disk_bytes=free,
        sample_ts_ns=None if sample is None else sample.get('sample_ts_ns'),
        bert_and_batch=summary, heavy_lease_owners=owners, foreign_compilers=foreign,
        monitored_worker=dict(pid=args.bert_pid, starttime=args.bert_starttime),
        monitored_batch=dict(pid=args.batch_pid, starttime=args.batch_starttime),
        scope=SCOPE)


def source_gate(args):
    checked = original.source_gate(args) if args.model == 'resnet20' else common.source_gate(args)
    params = checked['optimized_source']['optimizer_params']
    if params.get('max_slot') != 65536 or params.get('bts_ub') != 16:
        raise ValueError('Preserve the original 64k/level16 Orbit parser metadata')
    registry = args.registry or paths.RESULTS_DIR/'registry.jsonl'
    if registry.exists():
        for line in registry.read_text().splitlines():
            if not line.strip():
                continue
            row = original.json.loads(line)
            if (row.get('round') == ROUND and row.get('mode') == MODE
                    and row.get('model') == checked['optimized_source']['model']
                    and ((row.get('metrics') or {}).get('attempted') or {}).get('build', 0) > 0):
                raise ValueError('This explicit parallel variant already attempted this model once')
    checked.update(parallel_exception_scope=SCOPE,
        identified_bert=dict(worker_pid=args.bert_pid, worker_starttime=args.bert_starttime,
            source_case=str(args.bert_case), run_case=str(args.bert_run_case)),
        original_heavy_owner=dict(pid=args.batch_pid, starttime=args.batch_starttime,
            case=str(args.batch_case)), global_heavy_lease_acquired_by_this_entry=False)
    return checked


def _clone_plain(function, namespace):
    if not isinstance(function, FunctionType) or hasattr(function, '__wrapped__'):
        raise TypeError('Only undecorated ordinary functions may receive private globals')
    copied = FunctionType(function.__code__, namespace, function.__name__, function.__defaults__, function.__closure__)
    copied.__kwdefaults__ = function.__kwdefaults__
    return copied


def _namespace(args):
    namespace = dict(vars(original))
    factory = original.shared.MODEL if args.model == 'resnet20' else common.MODEL_RECORDS['vgg16']['factory']
    latest = [None]

    def observe(sample, available, free):
        latest[0] = parallel_observation(args, sample, available, free)
        return latest[0]

    def durable(path, record):
        if path.name == 'result.json' and record.get('round') == ROUND:
            record['limits'].update(AS_gib=4, subtree_RSS_gib=4, minimum_available_gib=2)
            record.update(parallel_exception_scope=SCOPE,
                global_heavy_lease_acquired_by_this_entry=False,
                existing_initializer_lock_retained=True,
                bert_signal_actions=0, latest_parallel_observation=latest[0])
            if record.get('status') != 'ok':
                record['backend_parser_consumption_verified'] = None
        original.durable_json(path, record)

    def process(command, **options):
        if options['mem_gib'] != 12 or options['fsize_gib'] != 8:
            raise ValueError('Original parser process resource interface changed')
        before = observe(None, original._available_ram_bytes(), shutil.disk_usage(args.out).free)
        if before['stop']:
            original.durable_json(args.out/'prelaunch-parallel-guard.json', before)
            raise RuntimeError('; '.join(before['reasons']))
        options['mem_gib'] = 4
        observation = original.run_process(command, **options)
        # Persist a real subprocess return before any after-call policy rejection.
        original.durable_json(args.out/(Path(options['log']).stem+'-process-return.json'), observation)
        return observation

    namespace.update(__file__=__file__, ROUND=ROUND, MODE=MODE, CLIENT=CLIENT,
        source_gate=source_gate, physical_observation=observe, durable_json=durable,
        run_process=process, shared=SimpleNamespace(MODEL=factory, read=original.shared.read,
            native_environment=original.shared.native_environment))
    for name in ('execute', 'prepare', 'append_registry', 'terminal'):
        namespace[name] = _clone_plain(getattr(original, name), namespace)
    prepare = namespace['prepare']

    def snapshot(call_args, out, checked):
        work = prepare(call_args, out, checked)
        shutil.copyfile(original.__file__, out/'source/underlying-parser-entry.py')
        shutil.copyfile(common.__file__, out/'source/underlying-common-parser-entry.py')
        return work

    namespace['prepare'] = snapshot
    return namespace


def execute(args):
    root = paths.RESULTS_DIR.resolve()
    if (args.bert_case.parent != root or args.bert_run_case.parent != root
            or args.batch_case.parent != root or args.out.parent != root
            or not args.out.name.startswith('orbit-'+args.model+'-static-parse-cpu-overlap-')):
        raise ValueError('Use explicit original cases and a fresh independent overlap case inside results')
    if len({args.out, args.bert_case, args.bert_run_case, args.batch_case, args.source_case, args.shared_build_case}) != 6:
        raise ValueError('The new overlap case must preserve every independent original source')
    return _namespace(args)['execute'](args)


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError('A positive original process identity is required')
    return number


def parser():
    command = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    command.add_argument('--model', choices=('resnet20', 'vgg16'), required=True)
    for name in ('out', 'source-case', 'shared-build-case', 'orbit-root', 'go-sdk',
                 'initialization-audit', 'bert-case', 'bert-run-case', 'batch-case'):
        command.add_argument('--'+name, type=Path, required=True)
    for name in ('bert-pid', 'bert-starttime', 'batch-pid', 'batch-starttime'):
        command.add_argument('--'+name, type=positive, required=True)
    command.add_argument('--registry', type=Path)
    command.add_argument('--run', action='store_true')
    return command


def main(argv=None):
    args = parser().parse_args(argv)
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.expanduser().resolve())
    result = execute(args)
    print(original.json.dumps({key: result[key] for key in ('model', 'status', 'error',
        'compiler_graph_completed', 'backend_parser_consumption_verified')}))
    return int(result['status'] not in ('ok', 'prepared'))


if __name__ == '__main__':
    raise SystemExit(main())
