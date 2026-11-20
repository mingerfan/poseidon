"""Bind an in-process cold archive to its actual serial job producer births.

Only saved metadata and current host PID identities are inspected. Original
scientific source-admission births are deliberately not archive-producer proof.
"""
import fcntl
import json
import os
from pathlib import Path

REGISTRY_FIELDS = {'run_id', 'round', 'model', 'mode', 'git_rev', 'metrics', 'artifacts_dir', 'ts'}
ARCHIVE_MODULE = 'pt_agent.experiments.a_group_cold_closed_archive'
ARCHIVE_ROUND = 'A-GROUP-COLD-CLOSED-RESOURCE-ARCHIVE'


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 32 * 1024**2,
            'Bounded original metadata file required: ' + str(path))
    with path.open() as stream:
        return json.load(stream)


def _process(pid):
    try:
        fields = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
    except FileNotFoundError:
        return None
    return dict(pid=pid, starttime=int(fields[19]), state=fields[0])


def positive(value):
    return type(value) is int and value > 0


def registry_rows(path, identities):
    found = {key: [] for key in identities}
    require(Path(path).is_file() and not Path(path).is_symlink(), 'Original registry metadata file required')
    with Path(path).open() as stream:
        fcntl.flock(stream, fcntl.LOCK_SH)
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if row.get('run_id') in found:
                    found[row['run_id']].append(row)
    result = {}
    for identity, case in identities.items():
        rows = found[identity]
        require(len(rows) == 1 and set(rows[0]) == REGISTRY_FIELDS,
                'Unique eight-field registry required: ' + identity)
        row = rows[0]
        require(Path(row['artifacts_dir']).resolve() == case,
                'Registry artifact identity/status differs: ' + identity)
        result[identity] = row
    return result


def closed_case(case, statuses=('ok',)):
    result, completion = read(case / 'result.json'), read(case / 'completion.json')
    require(result.get('status') in statuses and result.get('status') == completion.get('status')
            and completion.get('round') == result.get('round')
            and completion.get('required_artifacts_persisted') is True
            and completion.get('registry_persisted') is True,
            'Successful durable result/completion required: ' + str(case))
    require(result.get('run_id') in (None, case.name), 'Result run identity differs')
    return result, completion


def clean(process):
    return (process.get('exit_code') == 0 and process.get('timed_out') is False
            and 'signal' in process and process['signal'] is None
            and process.get('remaining_pids') == [] and process.get('signals_sent') == []
            and positive(process.get('started_ns')) and positive(process.get('ended_ns'))
            and process['ended_ns'] >= process['started_ns'])


def command_options(argv, archive, source_kind, namespace):
    require(isinstance(argv, list) and all(isinstance(x, str) for x in argv), 'Original argv list required')
    def option(name):
        require(argv.count(name) == 1 and argv.index(name) + 1 < len(argv), 'Unique command option required: ' + name)
        return argv[argv.index(name) + 1]
    require(option('-m') == ARCHIVE_MODULE and option('--out') == str(archive)
            and option('--case') == source_kind and option('--host-pid-namespace') == namespace
            and argv.count('--run') == 1, 'Archive command/source/output/namespace binding differs')


def archive_event_pids(archive, result, completion, process):
    timing = result.get('timing') or {}
    spans = timing.get('spans') or []
    require(timing.get('status') == 'ok' and timing.get('attribution_ready') is True and bool(spans),
            'Original archive timing spans required')
    expected, pids = set(), set()
    for span in spans:
        event_id = span.get('event_id')
        require(isinstance(event_id, str) and len(event_id.split(':')) == 2, 'Typed archive event identity required')
        parts = event_id.split(':')
        require(all(part.isdecimal() and int(part) > 0 for part in parts), 'Positive archive event PID/sequence required')
        pid = int(parts[0])
        require(span.get('pid') == pid and span.get('status') == 'ok'
                and type(span.get('start_ns')) is int and type(span.get('end_ns')) is int
                and process['started_ns'] <= span['start_ns'] <= span['end_ns'] <= process['ended_ns'],
                'Archive span PID or job interval differs')
        require(event_id not in expected, 'Duplicate archive completed event identity')
        expected.add(event_id); pids.add(pid)
    require({span.get('event_id') for span in completion.get('phase_spans', [])} == expected,
            'Archive completion event identities differ')
    seen = {}
    event_path = archive / 'events.jsonl'
    require(event_path.is_file() and not event_path.is_symlink(), 'Original archive event metadata file required')
    with event_path.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            event = json.loads(line); identity = event.get('event_id')
            require(identity in expected and event.get('pid') == int(identity.split(':')[0]),
                    'Archive raw event identity/PID differs')
            kind = event.get('type')
            require(kind in ('begin', 'end') and kind not in seen.setdefault(identity, set()),
                    'Archive event must have one original begin/end pair')
            seen[identity].add(kind)
    require(set(seen) == expected and all(kinds == {'begin', 'end'} for kinds in seen.values()),
            'Archive begin/end events are incomplete')
    return pids, sorted(expected)


def verify_archive_producer(archive_case, owning_serial_case, *, host_pid_namespace, process_reader=_process):
    """Return actual archive-job births, all closed in the requested host namespace.

    The owning queue must be fully closed; its unrelated scientific failures remain preserved. Archive identity is
    checked independently across its registry/result/events and the owner's
    manifest/result/command/resource snapshots. A reused host PID has a different
    birth; a still-present original birth (including a zombie) is rejected.
    """
    archive, owner = Path(archive_case).absolute(), Path(owning_serial_case).absolute()
    require(archive == archive.resolve() and owner == owner.resolve()
            and archive.is_dir() and owner.is_dir() and archive != owner
            and archive.parent == owner.parent, 'Canonical distinct sibling cases required')
    require(host_pid_namespace == os.readlink('/proc/self/ns/pid'), 'Approved host PID namespace required')
    ar, ac = closed_case(archive)
    sr, sc = closed_case(owner, ('ok', 'error', 'completed_with_observations'))
    require(sr.get('round') == 'AUTONOMOUS-SERIAL' and sr.get('mode') == 'finite-trusted-serial-experiments'
            and ar.get('round') == ARCHIVE_ROUND and ar.get('resource_status') == 'archived_or_kept',
            'Original serial/archive scope differs')
    source_kind = ar.get('source_kind')
    require(isinstance(source_kind, str) and source_kind == Path(source_kind).name
            and ar.get('source_case') == str(archive.parent / source_kind), 'Archive original source identity differs')
    registries = registry_rows(owner.parent / 'registry.jsonl', {archive.name: archive, owner.name: owner})
    areg, sreg = registries[archive.name], registries[owner.name]
    require(areg['round'] == ar['round'] and areg['model'] == source_kind
            and areg['mode'] == 'resource-only-reversible-gzip'
            and sreg['round'] == sr['round'] and sreg['mode'] == sr['mode']
            and sreg['model'] == 'original-model-experiment-queue'
            and areg.get('metrics', {}).get('status') == ar['status']
            and sreg.get('metrics', {}).get('status') == sr['status'], 'Registry round/model/mode/status differs')
    jobs = sr.get('jobs') or []
    require(all(not row.get('cleanup_unknown') and not row.get('signal')
                and not (row.get('process') or {}).get('remaining_pids') for row in jobs),
            'Owning serial descendant cleanup remains unresolved')
    matching = [row for row in jobs if row.get('out') == str(archive)]
    require(len(matching) == 1, 'Exactly one owning serial job must produce archive case')
    job = matching[0]; job_id = job.get('id')
    require(isinstance(job_id, str) and job_id == Path(job_id).name
            and job.get('status') == job.get('experiment_status') == 'ok'
            and job.get('cleanup_unknown') is False and clean(job.get('process') or {}),
            'Original archive job must have one clean successful process observation')
    command_options(job.get('argv'), archive, source_kind, host_pid_namespace)
    manifest = read(owner / 'manifest.json')
    declared = [row for row in manifest.get('jobs', []) if row.get('out') == str(archive)]
    require(len(declared) == 1 and declared[0].get('id') == job_id
            and declared[0].get('argv') == job['argv']
            and read(owner / (job_id + '-command.json')) == job['argv'],
            'Manifest/result/saved actual archive command differ')
    recorded = [row for row in sreg['metrics'].get('jobs', []) if row.get('out') == str(archive)]
    require(recorded == [job], 'Owning registry matching actual job differs')
    supervisor = read(owner / 'supervisor.json')
    outer = supervisor.get('process') or {}
    require(positive(supervisor.get('pid')) and outer.get('pid') == supervisor['pid']
            and positive(outer.get('starttime')), 'Owning controller saved birth required')
    controller = (outer['pid'], outer['starttime']); process = job['process']
    event_pids, event_ids = archive_event_pids(archive, ar, ac, process)
    resource_path = owner / (job_id + '-resources.jsonl')
    require(resource_path.is_file() and not resource_path.is_symlink(), 'Original matching-job resource metadata required')
    births, event_births, sampled_outer, roots, snapshots = set(), {}, set(), set(), 0
    with resource_path.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            sample = json.loads(line); timestamp = sample.get('ts_ns')
            require(type(timestamp) is int and process['started_ns'] <= timestamp <= process['ended_ns'],
                    'Actual archive resource snapshot is outside matching job interval')
            observed_outer = sample.get('outer_process') or {}
            require((observed_outer.get('pid'), observed_outer.get('starttime')) == controller,
                    'Archive resource snapshot owner birth differs')
            sampled_outer.add(controller); snapshots += 1
            require(isinstance(sample.get('processes'), list), 'Original archive process snapshots required')
            for item in sample['processes']:
                require(positive(item.get('pid')) and positive(item.get('starttime')),
                        'Positive actual archive producer PID/birth required')
                birth = (item['pid'], item['starttime']); births.add(birth)
                if item['pid'] in event_pids:
                    if item.get('ppid') == controller[0] and item.get('pgid') == item['pid']:
                        roots.add(birth)
                    event_births.setdefault(item['pid'], set()).add(item['starttime'])
    require(snapshots > 0 and births and sampled_outer == {controller},
            'Actual matching-job archive producer snapshots/births required')
    require(len(roots) == 1, 'One actual archive event process must be a direct owning-controller child/group leader')
    require(set(event_births) == event_pids and all(len(values) == 1 for values in event_births.values()),
            'Archive event PID is unbound or has ambiguous actual resource birth')
    require(all(pid != controller[0] for pid, _ in births),
            'Owning controller cannot substitute for archive producer')
    closed = []
    for pid, birth in sorted(births | {controller}):
        live = process_reader(pid)
        require(live is None or (positive(live.get('starttime')) and live['starttime'] != birth),
                'Actual archive/controller producer birth is still present or unresolved')
        closed.append(dict(pid=pid, starttime=birth, host_pid_reused=live is not None))
    return dict(closed_producer_births=len(births), owning_serial_case=str(owner), archive_case=str(archive),
        archive_producer_births=[row for row in closed if row['pid'] != controller[0]],
        owning_controller_birth=next(row for row in closed if row['pid'] == controller[0]),
        host_pid_namespace=host_pid_namespace, matching_job_id=job_id, exact_archive_argv=job['argv'],
        actual_resource_file=str(resource_path), actual_resource_snapshots=snapshots,
        archive_event_pids=sorted(event_pids), archive_event_ids=event_ids,
        registry_file=str(owner.parent/'registry.jsonl'), unique_eight_field_registry_run_ids=sorted(registries),
        source_scientific_births_used_as_archive_proof=False, scientific_activity=0,
        closure_scope='Actual matched serial archive job tree plus owning controller; original source calibration producers excluded')
