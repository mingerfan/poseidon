"""One offline generic Qwen library build alongside an identified BERT tune.

The existing library builder and initializer lock are retained. The original
heavy lease is observed only; this entry never acquires/releases it or signals
BERT. A successful archive supplies no actual27-component proof or execution.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import threading
import time

from .. import paths
from . import qwen_component_client_build as original
from . import orbit_static_parse_cpu_overlap as overlap

ROUND = 'QWEN-COMPONENT-LIBRARY-CPU-OVERLAP'
MODE = 'offline-generic-library-only-during-identified-bert-tune3'
MODULE = 'pt_agent.experiments.qwen_component_library_cpu_overlap'
GIB = original.GIB
SCOPE = ('One offline generic52-call/3-boundary-bootstrap library archive build '
         'during the identified BERT CPU tune; no model payload, generated-code '
         'execution, context, keys, actual component proofs or encrypted generation. '
         'Observed build timing includes possible CPU contention.')


def source_gate(args):
    checked = original.source_gate(args)
    registry = args.registry or paths.RESULTS_DIR/'registry.jsonl'
    if registry.exists():
        for line in registry.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if (row.get('round') == ROUND and row.get('mode') == MODE
                    and (row.get('metrics') or {}).get('attempted_builds', 0) > 0):
                raise ValueError('This explicit generic-library overlap variant already attempted one build')
    checked.update(parallel_exception_scope=SCOPE,
        guard_implementation=str(Path(overlap.__file__).resolve()),
        identified_bert=dict(worker_pid=args.bert_pid, worker_starttime=args.bert_starttime,
            source_case=str(args.bert_case), run_case=str(args.bert_run_case)),
        original_heavy_owner=dict(pid=args.batch_pid, starttime=args.batch_starttime,
            case=str(args.batch_case)), global_heavy_lease_acquired_by_this_entry=False)
    return checked


def _namespace(args):
    """Private ordinary-function globals; original decorated lock stays intact."""
    namespace = dict(vars(original))
    latest = [None]
    write_lock = threading.Lock()

    def observe(sample, available):
        row = overlap.parallel_observation(args, sample, available, shutil.disk_usage(args.out).free)
        # Guard rules are exactly delegated; only the consumer's scope text changes.
        row['shared_guard_scope'] = row['scope']
        row['scope'] = SCOPE
        row['observed_perf_counter_ns'] = time.perf_counter_ns()
        latest[0] = row
        with write_lock, (args.out/'parallel-observations.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
        return row

    def durable(path, record):
        path = Path(path)
        if path.name == 'result.json' and record.get('round') == ROUND:
            record['limits'].update(monitor_poll_seconds=.25, subtree_RSS_gib=4)
            record.update(scope=SCOPE, parallel_exception_scope=SCOPE,
                global_heavy_lease_acquired_by_this_entry=False,
                existing_initializer_lock_retained=True, bert_signal_actions=0,
                latest_parallel_observation=latest[0],
                parallel_observations=str(args.out/'parallel-observations.jsonl'))
            if record.get('status') != 'ok':
                record.update(generic_library_build_passed=False, generic_library_compile_accepted=False)
            record.update(linked_native_program_verified=False, full_qwen_compile_verified=False,
                whole_generation_verified=False, actual_component_artifacts_admitted=0,
                actual_component_manifest=None, parameters_selected=False)
        try:
            original.durable_json(path, record)
        except BaseException as exc:
            if path.name == 'guard-failure.json':
                # The original watcher's next statement is its own stop signal.
                # If persistence fails here, still stop only this supervisor.
                if latest[0] is not None:
                    latest[0]['guard_persistence_error'] = type(exc).__name__+': '+str(exc)
                try:
                    raise
                finally:
                    os.kill(os.getpid(), signal.SIGTERM)
            raise

    def process(command, **options):
        if options['mem_gib'] != 4 or options['fsize_gib'] != 8:
            raise ValueError('Preserve the original library AS4/file8 process contract')
        before = observe(None, original._available_ram_bytes())
        if before['stop']:
            original.durable_json(args.out/'prelaunch-parallel-guard.json', before)
            raise RuntimeError('; '.join(before['reasons']))
        observed = original.run_process(command, **options)
        # Keep the real return even if BERT advances before the final check.
        original.durable_json(args.out/'build-process-return.json', observed)
        return observed

    namespace.update(__file__=__file__, ROUND=ROUND, MODE=MODE, MODULE=MODULE,
        source_gate=source_gate, physical=observe, durable_json=durable, run_process=process)
    for name in ('execute', 'close', 'prepare'):
        namespace[name] = overlap._clone_plain(getattr(original, name), namespace)
    prepare = namespace['prepare']

    def snapshot(out, checked):
        command = prepare(out, checked)
        shutil.copyfile(original.__file__, out/'source/underlying-library-builder.py')
        shutil.copyfile(overlap.__file__, out/'source/shared-cpu-overlap-guard.py')
        return command

    namespace['prepare'] = snapshot
    return namespace


def execute(args):
    root = paths.RESULTS_DIR.resolve()
    for name in ('out', 'prepared', 'go_sdk', 'bert_case', 'bert_run_case', 'batch_case', 'registry'):
        value = getattr(args, name, None)
        if value is not None:
            setattr(args, name, Path(value).expanduser().resolve())
    for name in ('bert_pid', 'bert_starttime', 'batch_pid', 'batch_starttime'):
        if type(getattr(args, name, None)) is not int or getattr(args, name) <= 0:
            raise ValueError('Explicit positive integer original process identities are required')
    if (any(getattr(args, name).parent != root for name in ('out', 'bert_case', 'bert_run_case', 'batch_case'))
            or not args.out.name.startswith('qwen-component-library-cpu-overlap-')):
        raise ValueError('Use original result cases and a fresh independent Qwen overlap output')
    if len({args.out, args.prepared, args.bert_case, args.bert_run_case, args.batch_case}) != 5:
        raise ValueError('Preserve each original preparation/BERT/batch case separately')
    return _namespace(args)['execute'](args)


def positive(value):
    if isinstance(value, bool):
        raise argparse.ArgumentTypeError('A positive original process identity is required')
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise argparse.ArgumentTypeError('A positive original process identity is required') from exc
    if number <= 0 or isinstance(value, float):
        raise argparse.ArgumentTypeError('A positive original process identity is required')
    return number


def parser():
    command = original.parser()
    command.description = __doc__
    for name in ('bert-case', 'bert-run-case', 'batch-case'):
        command.add_argument('--'+name, type=Path, required=True)
    for name in ('bert-pid', 'bert-starttime', 'batch-pid', 'batch-starttime'):
        command.add_argument('--'+name, type=positive, required=True)
    return command


def main(argv=None):
    result = execute(parser().parse_args(argv))
    print(json.dumps({key: result[key] for key in ('status', 'error', 'generic_library_build_passed',
        'linked_native_program_verified', 'whole_generation_verified')}))
    return int(result['status'] not in ('ok', 'prepared'))


if __name__ == '__main__':
    raise SystemExit(main())
