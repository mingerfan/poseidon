"""Wait for the exact three-hour cases and publish saved scalar evidence once."""
import argparse
from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path
import resource
import signal
import time

from . import bert_three_hour_terminal_metadata as evidence


def validate(manifest):
    evidence.require(manifest.get('schema') == 'bert-three-hour-terminal-metadata-manifest-v1',
                     'Explicit three-hour metadata manifest required')
    names = dict(source_case=evidence.FRESH, native_case=evidence.SEARCH,
                 source_queue=evidence.QUEUE, report_case=evidence.REPORT,
                 report_queue=evidence.REPORT_QUEUE)
    cases = [Path(manifest[k]).resolve() for k in names]
    evidence.require(all(p.name == names[k] for k, p in zip(names, cases))
                     and len(set(cases)) == 5 and len({p.parent for p in cases}) == 1,
                     'Exact distinct three-hour source/queue/report family required')
    evidence.require(Path(manifest['registry']).resolve() == cases[0].parent/'registry.jsonl',
                     'Original common registry required')
    for key in ('comparison_watch', 'report_watch'):
        watch = manifest[key]
        evidence.require(all(type(watch.get(k)) is int and watch[k] > 0 for k in ('pid', 'starttime')),
                         'Actual positive controller PID and birth required')
    for key in ('attribution_helper', 'quality_helper'):
        evidence.require(Path(manifest[key]).is_absolute() and Path(manifest[key]).is_file(),
                         'Existing trusted metadata helper required')
    evidence.require(len(manifest['destinations']) == 2
                     and len({Path(p).resolve() for p in manifest['destinations']}) == 2
                     and all(Path(p).is_dir() for p in manifest['destinations']),
                     'Two distinct existing scalar delivery directories required')
    deadline = datetime.fromisoformat(manifest['deadline_utc'])
    evidence.require(deadline.tzinfo is not None, 'Explicit timezone for the metadata cutoff required')
    return deadline


def watched_closed(watch, *, process_reader):
    value = process_reader(watch['pid'])
    return value is None or value.get('starttime') != watch['starttime']


def publish(manifest, name, data):
    for directory in manifest['destinations']:
        evidence.save(Path(directory)/name, data)


def observe(manifest, out, *, now=lambda: datetime.now(timezone.utc), sleep=time.sleep,
            process_reader=evidence.closure._process):
    deadline = validate(manifest)
    out = Path(out)
    old = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM)}
    def interrupted(signum, _frame):
        raise InterruptedError('Metadata observer received signal '+str(signum))
    with out.with_name(out.name+'.metadata.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        out.mkdir(exist_ok=False)
        state = dict(schema='bert-three-hour-terminal-metadata-observation-v1', status='waiting',
                     started_utc=now().isoformat(), metadata_only=True, heavy_lease_acquired=False,
                     attribution_publications=0, comparison_publications=0, quality_publications=0,
                     report_followup_closed=False, new_scientific_actions=0,
                     scientific_rows_modified=False, signals_sent=[], proofs={}, error=None)
        try:
            for sig in old:
                signal.signal(sig, interrupted)
            while now() < deadline:
                for kind in ('agent', 'native'):
                    if kind in state['proofs']:
                        continue
                    try:
                        proof = evidence.source_proof(manifest, kind, process_reader=process_reader)
                    except evidence.Pending as exc:
                        state[kind+'_pending'] = str(exc)
                        continue
                    state['proofs'][kind] = proof
                    state.pop(kind+'_pending', None)
                if 'agent' in state['proofs'] and not state['attribution_publications']:
                    blobs = evidence.attribution(manifest, state['proofs']['agent'])
                    for name, text in blobs.items():
                        evidence.save(out/name, text)
                        publish(manifest, name, text)
                    state['attribution_publications'] = len(blobs)*len(manifest['destinations'])
                if (len(state['proofs']) == 2 and not state['comparison_publications']
                        and watched_closed(manifest['comparison_watch'], process_reader=process_reader)):
                    try:
                        queue = evidence.queue_proof(manifest, process_reader=process_reader)
                    except evidence.Pending as exc:
                        state['comparison_pending'] = str(exc)
                    else:
                        data = dict(schema='bert-three-hour-fresh-vs-independent-native-terminal-v1',
                                    observed_utc=now().isoformat(), model='bert_tiny',
                                    param_group='ours-N16-boot5', metadata_only=True,
                                    fresh_arranger=state['proofs']['agent'],
                                    dacapo_current_rule63_search=state['proofs']['native'],
                                    queue=queue, original_two_hour_records_preserved=True,
                                    original_9180_warm_record_preserved=True,
                                    native_and_cold_clocks_are_separate=True, new_scientific_actions=0)
                        evidence.save(out/'bert_3h_comparison_terminal.json', data)
                        publish(manifest, 'bert_3h_comparison_terminal.json', data)
                        state['comparison_publications'] = len(manifest['destinations'])
                        state.pop('comparison_pending', None)
                if not state['report_followup_closed'] and watched_closed(
                        manifest['report_watch'], process_reader=process_reader):
                    waiter = Path(manifest['report_waiter_case'])
                    result, done = evidence.read(waiter/'result.json'), evidence.read(waiter/'completion.json')
                    if result and done:
                        evidence.require(done.get('status') == result.get('status')
                                         and done.get('metadata_persisted') is True,
                                         'Original report waiter must finish durable metadata persistence')
                        if not Path(manifest['report_case']).exists():
                            state.update(report_followup_closed=True, report_status='not_created',
                                         report_reason=result.get('reason') or result.get('error'))
                        else:
                            try:
                                quality = evidence.final_quality(manifest, process_reader=process_reader)
                            except evidence.Pending as exc:
                                state['report_pending'] = str(exc)
                            else:
                                evidence.save(out/'bert_fresh_3h_final_quality.json', quality)
                                publish(manifest, 'bert_fresh_3h_final_quality.json', quality)
                                state.update(report_followup_closed=True, report_status=quality['status'],
                                             quality_publications=len(manifest['destinations']))
                                state.pop('report_pending', None)
                state['observed_utc'] = now().isoformat()
                evidence.save(out/'progress.json', state)
                if state['comparison_publications'] and state['report_followup_closed']:
                    state['status'] = 'ok'
                    break
                sleep(min(15, max(0, (deadline-now()).total_seconds())))
            else:
                state['status'] = 'deadline_without_all_terminal_sources'
        except BaseException as exc:
            state.update(status='error', error=type(exc).__name__+': '+str(exc))
        finally:
            for sig in old:
                signal.signal(sig, signal.SIG_IGN)
            state['ended_utc'] = now().isoformat()
            evidence.save(out/'result.json', state)
            evidence.save(out/'completion.json', dict(status=state['status'], metadata_only=True,
                          metadata_persisted=True, new_scientific_actions=0,
                          comparison_publications=state['comparison_publications'],
                          attribution_publications=state['attribution_publications'],
                          quality_publications=state['quality_publications']))
            for sig, handler in old.items():
                signal.signal(sig, handler)
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--host-pid-namespace', required=True)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args(argv)
    evidence.require(args.run and args.host_pid_namespace == os.readlink('/proc/self/ns/pid'),
                     'Explicit metadata observation in the actual host namespace required')
    limit = 512*1024**2
    hard = resource.getrlimit(resource.RLIMIT_AS)[1]
    resource.setrlimit(resource.RLIMIT_AS, (limit if hard == resource.RLIM_INFINITY else min(limit, hard), hard))
    result = observe(evidence.read(args.manifest), args.out)
    print(__import__('json').dumps(dict(status=result['status'], metadata_only=True,
                                      scientific_actions=0, error=result.get('error'))))
    return int(result['status'] == 'error')


if __name__ == '__main__':
    raise SystemExit(main())
