"""Finite metadata wait followed by one reviewed, source-bound BERT pair.

No heavy lease or sample access occurs here. After the exact prior controller
and both jobs close, exec keeps this process identity and enters the separate
outer, which repeats source, producer, lease and resource admission itself.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time

from . import autonomous_serial_batch as batch
from . import bert_ordered_mock_pair as pair
from .archive_producer_closure import _process as birth_probe


SCHEMA = 'bert-ordered-original-mock-pair-launch-v1'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    require(path.is_file() and not path.is_symlink() and path.suffix == '.json'
        and path.stat().st_size <= 32*1024**2, 'Bounded regular launch metadata required')
    return json.loads(path.read_text())


def save(path, value):
    pair.save(path, value)


def validate(value):
    require(value.get('schema') == SCHEMA and value.get('automatic_restart') is False,
            'Finite reviewed one-attempt pair launch required')
    argv = value.get('argv')
    require(isinstance(argv, list) and all(isinstance(v, str) and v for v in argv)
        and argv[1:4] == ['-B', '-m', pair.MODULE]
        and Path(argv[0]).resolve() == Path(sys.executable).resolve(),
        'Only this existing system Python and the reviewed pair outer are executable')
    for flag in argv[4:]:
        if flag.startswith('--'):
            require(argv.count(flag) == 1, 'Keep one exact outer option: '+flag)
    args = pair.normalize_args(pair.parser().parse_args(argv[4:]))
    watch = value.get('watch') or {}
    require(watch.get('case') == str(args.prior_case)
        and watch.get('pid') == args.prior_pid and watch.get('starttime') == args.prior_starttime
        and type(watch.get('seconds')) is int and 0 < watch['seconds'] <= 90000
        and type(args.prior_pid) is int and args.prior_pid > 0
        and type(args.prior_starttime) is int and args.prior_starttime > 0,
        'Wait on the exact saved predecessor case and positive process birth')
    require(args.host_pid_namespace == os.readlink('/proc/self/ns/pid')
        and value.get('deadline_utc') == args.deadline_utc,
        'Keep the actual host namespace and publication deadline')
    pair.case_paths(args)
    return args, batch.deadline_timestamp(args.deadline_utc)


def limits(waiting):
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_AS, (512*1024**2 if waiting else 24*1024**3, 24*1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (64*1024**2 if waiting else 4*1024**3, 4*1024**3))


def gone(pid, birth, *, probe=birth_probe, sleep=time.sleep, clock=time.monotonic):
    began = clock()
    while True:
        value = probe(pid)
        if value is None or value.get('starttime') != birth:
            return dict(original_birth_gone=True, pid=pid, starttime=birth,
                        pid_reused=value is not None)
        if clock()-began >= 30:
            raise TimeoutError('Saved predecessor birth remains present after terminal persistence')
        sleep(.25)


def run(args, *, wait=batch.wait_closed, probe=birth_probe, executor=os.execv,
        set_limits=limits, now=time.time, sleep=time.sleep, clock=time.monotonic):
    manifest = read(args.manifest)
    outer, deadline = validate(manifest)
    ready = read(args.readiness)
    require(ready.get('schema') == 'bert-ordered-mock-pair-launch-readiness-v1'
        and ready.get('source_stopped') is True and ready.get('must_fix') == []
        and all(ready.get(key) is True for key in ('outer_tests_passed', 'worker_tests_passed',
            'waiter_tests_passed', 'old_source_preflight_passed', 'old_producer_preflight_passed')),
        'All reviewed source/test/old-source readiness must pass')
    require(not args.metadata_out.exists() and not outer.out.exists(), 'Preserve any existing metadata or pair attempt')
    require(not args.metadata_out.resolve().is_relative_to(pair.paths.RESULTS_DIR.resolve()),
            'Keep metadata waiting evidence outside the scientific result root')
    identity = probe(os.getpid())
    require(identity and identity.get('pid') == os.getpid()
        and type(identity.get('starttime')) is int and identity['starttime'] > 0,
        'The actual metadata-wait owner birth is required')
    args.metadata_out.mkdir(parents=True)
    result = dict(schema='bert-ordered-mock-pair-waiter-result-v1', status='waiting',
        started_utc=datetime.now(timezone.utc).isoformat(), process=identity,
        predecessor=manifest['watch'], pair_case=str(outer.out), manifest=str(args.manifest),
        readiness=str(args.readiness), deadline_utc=manifest['deadline_utc'],
        metadata_only_until_exec=True, actual_heavy_lease_acquired=False,
        numeric_source_gate_runs=0, model_or_sample_payload_reads=0, API=0, compiler=0, CKKS=0,
        original_process_declared_terminal=False, automatic_restart=False, error=None)
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}

    def interrupted(sig, frame):
        raise InterruptedError('The metadata pair waiter received signal '+str(sig))

    try:
        for sig in handlers:
            signal.signal(sig, interrupted)
        set_limits(True)
        save(args.metadata_out/'progress.json', result)
        remaining = deadline-now()-pair.SECONDS-30
        if remaining <= 0:
            result.update(status='skipped_deadline', reason='A full pair plus persistence no longer fits the fixed deadline')
        else:
            watch = dict(manifest['watch'], seconds=min(manifest['watch']['seconds'], remaining))
            result['wait'] = wait(watch)
            result['strict_predecessor_closure'] = gone(outer.prior_pid, outer.prior_starttime,
                probe=probe, sleep=sleep, clock=clock)
            result['original_process_declared_terminal'] = True
            require(now()+pair.SECONDS+30 < deadline, 'Full pair deadline reserve reached after waiting')
            require(not outer.out.exists(), 'The scientific pair case must still be new')
            plan = manifest.get('plan_observation')
            if plan is not None:
                path = Path(plan['path'])
                require(path.name == 'CODEX_TASKS.md' and type(plan.get('mtime_ns')) is int,
                        'Use the explicitly observed mainline plan update clock')
                if path.stat().st_mtime_ns != plan['mtime_ns']:
                    result.update(status='plan_changed_reconcile_automatically',
                        reason='The mainline plan changed during waiting; the active agent must read its latest instructions before numerical entry')
                    return result
            result.update(status='original_owner_closed_exec_requested',
                          source_gate_pending_in_new_outer=True)
            save(args.metadata_out/'progress.json', result)
            save(args.metadata_out/'launch-intent.json', dict(result, exact_argv=manifest['argv'],
                clock_scope='Metadata waiting is separate; warm numerical case clock begins in the new outer'))
            set_limits(False)
            # This remains the same PID/birth. The outer takes the shared heavy
            # lease itself and repeats every source/closure/resource check.
            executor(manifest['argv'][0], manifest['argv'])
            raise RuntimeError('The real exec unexpectedly returned without entering the pair outer')
    except TimeoutError as exc:
        result.update(status='wait_expired_owner_unproven', error=type(exc).__name__+': '+str(exc),
                      original_process_declared_terminal=False)
    except BaseException as exc:
        result.update(status='error', error=type(exc).__name__+': '+str(exc))
    finally:
        # Successful exec replaces this image and never reaches this block.
        for sig in handlers:
            signal.signal(sig, signal.SIG_IGN)
        try:
            result['ended_utc'] = datetime.now(timezone.utc).isoformat()
            save(args.metadata_out/'result.json', result)
            save(args.metadata_out/'completion.json', dict(status=result['status'],
                metadata_persisted=True, scientific_case_created=outer.out.exists(),
                model_or_sample_payload_reads=0, actual_heavy_lease_acquired=False,
                automatic_restart=False))
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'readiness', 'metadata-out'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(args)
    print(json.dumps(dict(status=result['status'], numeric_source_gate_runs=0)))
    return 0 if result['status'] == 'skipped_deadline' else 1


if __name__ == '__main__':
    raise SystemExit(main())
