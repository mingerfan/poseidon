"""Bounded original VGG fusion/math/compile alongside one identified BERT tune.

This explicit parallel exception loads the original three saved training probes,
runs the unchanged adapter mathematics and invokes the unchanged HEIR compiler.
The original initializer lock remains; the existing heavy lease is observed only.
No BERT process is signalled and no model, library or compilation policy changes.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import threading
import time

from .. import paths
from . import heir_vgg_fused_pipeline as original
from . import orbit_static_parse_cpu_overlap as overlap

ROUND = 'HEIR-VGG-CPU-OVERLAP'
MODE = 'original-fused-vgg-training-math-native-compile-during-identified-bert-tune3'
MODULE = 'pt_agent.experiments.heir_vgg_cpu_overlap'
GIB = original.GIB
RSS_GIB = 10
RSS4_REASONS = frozenset(('Own child subtree RSS exceeds4GiB',
                         'Own supervisor and descendant RSS exceeds4GiB'))
SCOPE = ('Original VGG four-pool/conv adapter export, original three saved training '
         'samples and strict binary64 mathematics, followed by native HEIR lowering; '
         'restricted parallel exception during the identified BERT tune3. Timing '
         'overlaps CPU tune. No Go, context, keys, CKKS, report, confirmation or API.')


def source_gate(source, args):
    """Keep the worker's original source object exactly reproducible."""
    checked = original.source_gate(source)
    registry = args.registry or paths.RESULTS_DIR/'registry.jsonl'
    if registry.exists():
        for line in registry.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            counts = (row.get('metrics') or {}).get('attempted') or {}
            if (row.get('round') == ROUND and row.get('mode') == MODE
                    and any(counts.get(name, 0) > 0 for name in ('export_math', 'native_compile'))):
                raise ValueError('This explicit VGG overlap variant already attempted one model job')
    return checked


def parallel_observation(args, sample, available, free):
    """Use the original guard and replace only its two explicit RSS4 reasons."""
    row = overlap.parallel_observation(args, sample, available, free)
    original_reasons = list(row['reasons'])
    row['shared_guard_scope'] = row['scope']
    row['shared_guard_reasons_before_RSS_policy'] = original_reasons
    row['reasons'] = [reason for reason in original_reasons if reason not in RSS4_REASONS]
    rss = row['subtree_rss_bytes']
    if rss is not None and rss > RSS_GIB*GIB:
        row['reasons'].append('Own supervisor and descendant RSS exceeds10GiB')
    row.update(stop=bool(row['reasons']), scope=SCOPE, own_aggregate_RSS_gib=RSS_GIB,
               observed_perf_counter_ns=time.perf_counter_ns())
    return row


def accepted(result):
    return (result.get('status') == 'ok' and result.get('model_compiled') is True
            and result.get('strict_math_passed') is True
            and all(type((result.get('completed') or {}).get(name)) is int
                    and result['completed'][name] == 1 for name in ('export_math', 'native_compile')))


def append_registry(out, result, registry):
    """Retain the original eight-field row and distinguish fact from acceptance."""
    metrics = {key: result[key] for key in ('status', 'attempted', 'completed', 'strict_math_passed',
        'model_compiled', 'static_native_bootstraps', *original.FORMAL)}
    metrics.update(model_compiled_accepted=accepted(result), scope=SCOPE,
                   parallel_exception_scope=SCOPE, bert_signal_actions=0)
    row = dict(run_id=out.name, round=ROUND, model=original.MODEL, mode=MODE, git_rev=None,
               metrics=metrics, artifacts_dir=str(out), ts=datetime.now(timezone.utc).isoformat())
    with Path(registry).open('a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX); stream.seek(0)
        if any(json.loads(line).get('run_id') == out.name for line in stream if line.strip()):
            raise ValueError('A result identity may be registered once')
        stream.seek(0, os.SEEK_END); stream.write(json.dumps(row, allow_nan=False)+'\n')
        stream.flush(); os.fsync(stream.fileno())


def _namespace(args):
    """Clone only ordinary supervision functions; all numerical calls stay original."""
    namespace = dict(vars(original))
    latest = [None]
    write_lock = threading.Lock()

    def observe(sample, available, free):
        row = parallel_observation(args, sample, available, free)
        latest[0] = row
        with write_lock, (args.out/'parallel-observations.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
        return row

    def durable(path, result):
        path = Path(path)
        if path.name == 'result.json' and result.get('round') == ROUND:
            result['limits'].update(subtree_RSS_gib=RSS_GIB, minimum_available_gib=2,
                                    monitor_poll_seconds=.2)
            result.update(scope=SCOPE, parallel_exception_scope=SCOPE,
                global_heavy_lease_acquired_by_this_entry=False, existing_initializer_lock_retained=True,
                bert_signal_actions=0, model_compiled_accepted=accepted(result),
                latest_parallel_observation=latest[0],
                parallel_observations=str(args.out/'parallel-observations.jsonl'),
                identified_bert=dict(worker_pid=args.bert_pid, worker_starttime=args.bert_starttime,
                    source_case=str(args.bert_case), run_case=str(args.bert_run_case)),
                original_heavy_owner=dict(pid=args.batch_pid, starttime=args.batch_starttime,
                    case=str(args.batch_case)))
        original.durable_json(path, result)

    def process(command, **options):
        if options['mem_gib'] != 24 or options['fsize_gib'] != 16:
            raise ValueError('Keep the original explicit AS24/file16 VGG contract')
        before = observe(None, original._available_ram_bytes(), shutil.disk_usage(args.out).free)
        if before['stop']:
            original.durable_json(args.out/'prelaunch-parallel-guard.json', before)
            raise RuntimeError('; '.join(before['reasons']))
        actual = original.run_process(command, **options)
        # Preserve the real process outcome before a later policy rejection.
        original.durable_json(Path(options['log']).parent/'process-return.json', actual)
        return actual

    namespace.update(__file__=__file__, ROUND=ROUND, MODE=MODE,
        source_gate=lambda source: source_gate(source, args), physical_observation=observe,
        durable_json=durable, run_process=process, append_registry=append_registry)
    for name in ('execute', 'terminal_result'):
        namespace[name] = overlap._clone_plain(getattr(original, name), namespace)
    original_terminal = namespace['terminal_result']

    def terminal(out, result):
        # The original watcher is stopped before terminal persistence. Repeat the
        # same admission once here and retain a completed native compilation fact.
        try:
            row = observe(None, original._available_ram_bytes(), shutil.disk_usage(out).free)
            if row['stop']:
                result['terminal_parallel_guard'] = row
                if result['status'] in ('ok', 'prepared'):
                    result.update(status='resource_stopped', guard_trigger=row,
                                  error='; '.join(row['reasons']))
        except BaseException as exc:
            result['terminal_parallel_observation_error'] = type(exc).__name__+': '+str(exc)
            if result['status'] in ('ok', 'prepared'):
                result.update(status='supervision_error', error=result['terminal_parallel_observation_error'])
        original_terminal(out, result)

    namespace['terminal_result'] = terminal

    def snapshot(out):
        rows = original._snapshot(out)
        for module, label in ((__file__, 'heir-vgg-cpu-overlap.py'),
                              (overlap.__file__, 'shared-cpu-overlap-guard.py')):
            target = out/'source'/label
            shutil.copyfile(module, target)
            rows.append(dict(original=str(Path(module).resolve()), snapshot=str(target), bytes=target.stat().st_size))
        return rows

    namespace['_snapshot'] = snapshot
    return namespace


def execute(args):
    root = paths.RESULTS_DIR.resolve()
    for name, value in vars(args).items():
        if isinstance(value, Path):
            expanded = value.expanduser()
            setattr(args, name, expanded.absolute() if name == 'python' else expanded.resolve())
    if type(args.memory_gib) is not int or args.memory_gib != 24:
        raise ValueError('This explicit overlap variant retains AS24GiB')
    for name in ('bert_pid', 'bert_starttime', 'batch_pid', 'batch_starttime'):
        if type(getattr(args, name)) is not int or getattr(args, name) <= 0:
            raise ValueError('Explicit positive original BERT/batch identities are required')
    if (any(getattr(args, name).parent != root for name in ('out', 'bert_case', 'bert_run_case', 'batch_case'))
            or not args.out.name.startswith('heir-vgg16-fused-pool-conv-cpu-overlap-')
            or len({args.out, args.bert_case, args.bert_run_case, args.batch_case}) != 4):
        raise ValueError('Keep original BERT/batch cases and a fresh independent VGG overlap case')
    return _namespace(args)['execute'](args)


def parser():
    command = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ('out', 'bert-case', 'bert-run-case', 'batch-case'):
        command.add_argument('--'+name, type=Path, required=True)
    for name in ('bert-pid', 'bert-starttime', 'batch-pid', 'batch-starttime'):
        command.add_argument('--'+name, type=int, required=True)
    for name in original.SOURCE_FIELDS:
        command.add_argument('--'+name.replace('_', '-'), type=Path)
    command.add_argument('--python', type=Path, default=paths.INTAKE_PYTHON)
    command.add_argument('--heir-opt', type=Path,
        default=paths.WORKSPACE/'baselines/heir-2026.11.1-venv/bin/heir-opt')
    command.add_argument('--registry', type=Path)
    command.add_argument('--memory-gib', type=int, choices=(24,), default=24)
    command.add_argument('--run', action='store_true')
    return command


def main(argv=None):
    result = execute(parser().parse_args(argv))
    print(json.dumps({key: result.get(key) for key in
        ('status', 'error', 'strict_math_passed', 'model_compiled', 'model_compiled_accepted')}))
    return int(result['status'] not in ('ok', 'prepared'))


if __name__ == '__main__':
    raise SystemExit(main())
