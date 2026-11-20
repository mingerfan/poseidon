"""Recover interrupted BERT budgets with explicit supervisor log-file limits.

The original interrupted case remains intact. Only its saved process identities
and terminal metadata are read before running the unused native trial followed
by a new, independent cold Agent trial. Scientific modules are reused unchanged.
"""
import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
from types import FunctionType

from .. import paths
from . import archive_producer_closure as closure
from . import autonomous_serial_batch as original
from . import bert_fresh_cold_3h as agent
from . import bert_rule63_shared_optimized_compile_3h as native
from . import bert_three_hour_comparison_queue as prior

MODULE = 'pt_agent.experiments.bert_supervisor_recovery'
INTERRUPTED = 'bert-fresh-10800-then-independent-rule63-search-serial-20261117'
FRESH = 'bert-sst2-fresh-cold-agent-3h-20261117'
SCHEMA = 'bert-supervisor-file-quota-recovery-v1'
GIB = 1024**3
require = closure.require


def resource_births(case, *, truncated_path=None):
    """Read full snapshots; identify one final incomplete write explicitly."""
    births, snapshots, tails = set(), 0, []
    supervisor = case/'supervisor.json'
    if supervisor.is_file():
        saved = closure.read(supervisor); row = saved.get('process') or {}
        require(row.get('pid') == saved.get('pid') and closure.positive(row.get('starttime')),
                'Saved supervisor identity required')
        births.add((row['pid'], row['starttime']))
    for path in sorted(case.rglob('*resources.jsonl')):
        require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 512*1024**2,
                'Bounded original process metadata required')
        with path.open('rb') as stream:
            for line in stream:
                if not line.strip():
                    continue
                if not line.endswith(b'\n'):
                    require(path == truncated_path and stream.read(1) == b'',
                            'Only the identified quota-interrupted final write may be incomplete')
                    # Recover no guessed identities from an incomplete JSON row.
                    tails.append(dict(path=str(path), bytes=len(line), metadata_incomplete=True))
                    continue
                sample = json.loads(line)
                rows = list(sample.get('processes') or [])
                if sample.get('outer_process'):
                    rows.append(sample['outer_process'])
                for row in rows:
                    require(closure.positive(row.get('pid')) and closure.positive(row.get('starttime')),
                            'Positive recorded process births required')
                    births.add((row['pid'], row['starttime']))
                snapshots += 1
    require(births and snapshots, 'Saved original process snapshots required')
    return births, dict(resource_snapshots=snapshots, incomplete_final_writes=tails)


def verify_gone(births, *, probe=closure._process):
    for pid, birth in sorted(births):
        current = probe(pid)
        require(current is None or current.get('starttime') != birth,
                'An original process birth remains present, including a possible zombie')


def interrupted_closure(root, *, host_namespace, probe=closure._process):
    require(host_namespace == os.readlink('/proc/self/ns/pid'), 'Actual host PID namespace required')
    case, source = root/INTERRUPTED, root/FRESH
    result, done = closure.closed_case(case, ('error',))
    sr, sd = closure.closed_case(source, ('error',))
    registry = closure.registry_rows(root/'registry.jsonl', {case.name:case, source.name:source})
    qr, ar = registry[case.name], registry[source.name]
    require(result.get('round') == 'AUTONOMOUS-SERIAL'
            and result.get('mode') == 'finite-trusted-serial-experiments'
            and qr.get('model') == 'original-model-experiment-queue'
            and qr.get('round') == result['round'] and qr.get('mode') == result['mode']
            and qr.get('metrics') == dict(status='error', jobs=result.get('jobs')),
            'Unchanged interrupted owner registry/result required')
    require((sr.get('round'), sr.get('model'), sr.get('mode')) == (agent.ROUND, agent.MODEL, agent.MODE)
            and ar.get('round') == sr['round'] and ar.get('model') == sr['model']
            and ar.get('mode') == sr['mode'] and ar.get('metrics', {}).get('status') == 'error'
            and sr.get('monitor_stopped') is True and sr.get('compile_accepted') is False
            and (sr.get('completed') or {}).get('compile') == 0,
            'Original interrupted fresh trial must remain uncompiled and terminal')
    declared = closure.read(case/'manifest.json')
    prior.namespace()['validate'](declared, root)
    jobs = result.get('jobs') or []
    require(len(jobs) == 2 and jobs[0].get('status') == 'ok'
            and jobs[0].get('cleanup_unknown') is False
            and jobs[1].get('out') == str(source) and jobs[1].get('cleanup_unknown') is True
            and jobs[1].get('error') == 'OSError: [Errno 27] File too large'
            and jobs[1].get('argv') == declared['jobs'][1]['argv']
            and closure.read(case/(jobs[1]['id']+'-command.json')) == jobs[1]['argv'],
            'Exact interrupted EFBIG job and untouched original command required')
    primary = case/(jobs[1]['id']+'-resources.jsonl')
    require(primary.stat().st_size == 64*1024**2, 'Original64MiB resource-write interruption required')
    birthsets, details = [], []
    for item in (case, source):
        births, detail = resource_births(item, truncated_path=primary)
        birthsets.append(births); details.append(dict(case=str(item), **detail))
    births = set().union(*birthsets)
    verify_gone(births, probe=probe)
    for process in sr.get('processes') or []:
        require((process.get('observation') or {}).get('remaining_pids') == [],
                'Source cleanup must have a recorded empty residual family')
    record = closure.read(source/'three-hour-record.json')
    require(record.get('budget_seconds') == 10800 and record.get('compile_accepted') is False
            and record.get('persisted_pipeline_within_three_hours') is False
            and 0 < sd.get('complete_case_elapsed_seconds', 0) < 10800,
            'Interrupted source did not consume its complete three-hour policy')
    return dict(schema=SCHEMA, observed_utc=datetime.now(timezone.utc).isoformat(),
        host_pid_namespace=host_namespace, interrupted_case=str(case), interrupted_source=str(source),
        original_cleanup_unknown_preserved=True, original_status='error',
        recorded_parent_resource_bytes=primary.stat().st_size,
        original_owner_file_limit_not_recorded=True,
        failure='OSError: [Errno 27] File too large',
        complete_case_elapsed_seconds=sd['complete_case_elapsed_seconds'],
        full_three_hour_allocation_consumed=False,
        saved_process_births=[dict(pid=p,starttime=b) for p,b in sorted(births)],
        all_saved_births_currently_gone=True, resource_metadata=details,
        native_job_previously_dispatched=False, original_registry_fields=8,
        required_artifacts_and_registry_persisted=True)


def set_supervisor_limits(*, file_gib=16):
    before = resource.getrlimit(resource.RLIMIT_FSIZE)
    wanted = file_gib*GIB
    hard = before[1]
    require(hard == resource.RLIM_INFINITY or hard >= wanted,
            'Supervisor hard file allowance must support its explicit log budget')
    resource.setrlimit(resource.RLIMIT_FSIZE, (wanted, hard))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    after = resource.getrlimit(resource.RLIMIT_FSIZE)
    require(after[0] == wanted, 'Supervisor log budget was not applied')
    return dict(before=list(before), after=list(after), file_bytes=wanted,
                scope='Actual queue supervisor and its resource log writes; children keep original limits')


def fixed_recovery(manifest, root, *, base):
    value = base(manifest, root)
    require(value.get('recovery_schema') == SCHEMA and value.get('direct_human_budget_seconds') == 10800,
            'Explicit finite three-hour recovery policy required')
    proof = interrupted_closure(Path(root).resolve(), host_namespace=value['host_pid_namespace'])
    original_manifest = closure.read(Path(root)/INTERRUPTED/'manifest.json')
    require(value['watch'] == original_manifest['watch'] and value.get('deadline_utc') == original_manifest.get('deadline_utc'),
            'Preserve the original verified predecessor and submission deadline')
    jobs = value['jobs']
    require(len(jobs) == 2 and all(row.get('requires', []) == [] for row in jobs),
            'Run two independent scientific trials serially without success dependencies')
    native_job = copy.deepcopy(original_manifest['jobs'][2]); native_job['requires'] = []
    require(jobs[0] == native_job, 'The previously unused native trial and its full10800s must remain unchanged')
    fresh_job = copy.deepcopy(original_manifest['jobs'][1]); fresh_job['requires'] = []
    old_out = fresh_job['out']; new_out = jobs[1]['out']
    require(Path(new_out).name == FRESH.replace('-20261117','-logquota-recovery-20261117')
            and Path(new_out).parent == Path(root), 'New independent cold Agent case required')
    fresh_job['out'] = new_out
    argv = fresh_job['argv']; argv[argv.index('--out')+1] = new_out
    argv[argv.index('--checkpoint-out')+1] = str(Path(root)/'bert-sst2-fresh-cold-agent-3h-checkpoint-logquota-recovery-20261117')
    require(jobs[1] == fresh_job and old_out != new_out, 'Only fresh output/checkpoint locations may differ')
    return value, proof


def namespace(*, limits):
    source = vars(original)
    values = dict(source, MODULES=set(original.MODULES)|
        {agent.MODULE.removeprefix(original.PREFIX), native.MODULE.removeprefix(original.PREFIX)})
    for name, fn in source.items():
        if isinstance(fn, FunctionType) and fn.__globals__ is source:
            clone = FunctionType(fn.__code__, values, fn.__name__, fn.__defaults__, fn.__closure__)
            clone.__kwdefaults__ = copy.copy(fn.__kwdefaults__); values[name] = clone
    base, writer = values['validate'], values['write_json']
    captured = {}
    def validate(manifest, root):
        value, proof = fixed_recovery(manifest, root, base=base)
        captured['closure'] = proof
        return value
    def write_json(path, value):
        writer(path, value)
        if path.name == 'manifest.json':
            writer(path.parent/'supervisor-file-limits.json', limits)
            writer(path.parent/'interrupted-predecessor-current-closure.json', captured['closure'])
    values.update(validate=validate, write_json=write_json)
    return values


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--manifest', type=Path, required=True)
    cli.add_argument('--out', type=Path, required=True)
    cli.add_argument('--lease-wait-seconds', type=int, default=60)
    args = cli.parse_args(argv)
    if not 0 <= args.lease_wait_seconds <= 604800:
        cli.error('Finite serial lease wait required')
    args.manifest = args.manifest.absolute(); args.out = args.out.absolute()
    return namespace(limits=set_supervisor_limits())['run'](args)


if __name__ == '__main__':
    raise SystemExit(main())
