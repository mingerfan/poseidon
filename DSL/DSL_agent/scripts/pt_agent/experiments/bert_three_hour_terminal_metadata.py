"""Read saved three-hour BERT evidence and publish separate scalar artifacts.

This module performs no model, Agent, compiler, solver or encrypted execution.
Existing attribution and report scalar functions are supplied explicitly by path.
"""
import csv
from datetime import datetime, timezone
import fcntl
import importlib.util
import io
import json
import os
from pathlib import Path

from . import archive_producer_closure as closure
from . import bert_fresh_cold_3h as fresh
from . import bert_rule63_shared_optimized_compile_3h as native
from . import bert_fresh_frozen_report_3h as report_entry

FRESH = 'bert-sst2-fresh-cold-agent-3h-20261117'
SEARCH = 'bert-rule63-current-shared-optimized-qwen48-3h-20261117'
QUEUE = 'bert-fresh-10800-then-independent-rule63-search-serial-20261117'
REPORT = 'bert-sst2-fresh-cold-agent-3h-isolated-report-20261117'
REPORT_QUEUE = 'bert-fresh-3h-isolated-report-serial-20261117'
ATTRIBUTION_OUTPUTS = ('bert_3h_agent_attribution.json', 'bert_3h_agent_stage_times.csv',
                       'bert_3h_agent_iteration_observations.jsonl')
require = closure.require


class Pending(RuntimeError):
    pass


def read(path):
    path = Path(path)
    if not path.exists():
        return None
    require(path.name not in ('SESSION.json', 'AGENT.json') and path.suffix not in
            ('.npz', '.npy', '.pt', '.ptdsl', '.mlir'), 'Saved metadata paths only')
    return closure.read(path)


def load_helper(path, name):
    path = Path(path)
    require(path.is_absolute() and path.is_file() and not path.is_symlink(),
            'An explicit existing trusted metadata helper is required')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def clean_tree(value):
    if isinstance(value, dict):
        require(not value.get('remaining_pids') and not value.get('cleanup_unknown'),
                'Original descendant cleanup remains unresolved')
        for child in value.values():
            clean_tree(child)
    elif isinstance(value, list):
        for child in value:
            clean_tree(child)


def closed(case, registry, *, identity=None):
    result, done = read(case/'result.json'), read(case/'completion.json')
    if not result or not done or result.get('status') == 'running':
        raise Pending('Original terminal persistence pending: '+case.name)
    require(result.get('status') in ('ok', 'error', 'completed_with_observations')
            and done.get('status') == result['status'] and done.get('round') == result.get('round')
            and done.get('registry_persisted') is True
            and done.get('required_artifacts_persisted') is True,
            'Matching durable original result/completion required')
    row = registry[case.name]
    require(row['round'] == result['round'] and row['mode'] == result['mode']
            and row.get('metrics', {}).get('status') == result['status'],
            'Original registry identity/status differs')
    if result.get('round') == 'AUTONOMOUS-SERIAL':
        require(row.get('model') == 'original-model-experiment-queue'
                and row.get('metrics') == dict(status=result['status'], jobs=result.get('jobs')),
                'Actual queue registry must match its original schema and jobs')
    else:
        require(row.get('model') == result.get('model'), 'Original registry model differs')
        if result.get('metrics') is not None:
            require(row.get('metrics') == result['metrics'], 'Original source metrics differ')
        require(result.get('monitor_stopped') is True, 'Original source monitor must stop')
    if identity:
        require(tuple(result.get(k) for k in ('round', 'model', 'mode')) == identity,
                'Exact original scientific source identity required')
    clean_tree(result)
    return result, done


def births_closed(case, *, process_reader=closure._process, extra_resources=()):
    births, snapshots = set(), 0
    supervisor = read(case/'supervisor.json')
    if supervisor:
        row = supervisor.get('process') or {}
        require(row.get('pid') == supervisor.get('pid') and closure.positive(row.get('starttime')),
                'Actual saved controller birth required')
        births.add((row['pid'], row['starttime']))
    if case.name in (QUEUE, REPORT_QUEUE):
        require(supervisor is not None, 'The owning queue supervisor must be saved')
    for path in sorted(set(case.rglob('*resources.jsonl')) | set(extra_resources)):
        require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 512*1024**2,
                'Bounded original resource metadata required')
        with path.open() as stream:
            for line in stream:
                if not line.strip():
                    continue
                sample = json.loads(line)
                rows = list(sample.get('processes') or [])
                if sample.get('outer_process'):
                    rows.append(sample['outer_process'])
                for row in rows:
                    require(closure.positive(row.get('pid')) and closure.positive(row.get('starttime')),
                            'Explicit positive saved process identities required')
                    births.add((row['pid'], row['starttime']))
                snapshots += 1
    require(births and snapshots, 'Original saved producer evidence required')
    for pid, birth in sorted(births):
        current = process_reader(pid)
        if current is not None and current.get('starttime') == birth:
            raise Pending('An original producer birth is still present')
    return dict(saved_births=len(births), resource_snapshots=snapshots, all_saved_births_closed=True)


def paper_row(path, case):
    found = []
    with Path(path).open() as stream:
        fcntl.flock(stream, fcntl.LOCK_SH)
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if row.get('run_id') == case.name:
                    found.append(row)
    if not found:
        raise Pending('Original paper append follows completion persistence')
    require(len(found) == 1 and Path(found[0].get('source', '')).resolve() == case/'result.json',
            'One unchanged paper row bound to this source is required')
    return found[0]


def clock_binding(result, done, record):
    require(record and record.get('round') == fresh.ROUND and record.get('mode') == fresh.MODE
            and record.get('model') == fresh.MODEL and record.get('budget_seconds') == 10800
            and record.get('started_ns') == done.get('started_ns')
            and record.get('required_artifacts_and_registry_completed_ns') == done.get('ended_ns')
            and record.get('completion_elapsed_seconds') == done.get('complete_case_elapsed_seconds')
            and all(type(record.get(k)) is bool for k in
                    ('persisted_pipeline_within_three_hours', 'persisted_pipeline_within_two_hours')),
            'Same-source authoritative10800second record required')
    start, end, observed = (record['started_ns'], done['ended_ns'], record.get('observed_ns'))
    require(all(type(x) is int for x in (start, end, observed)) and start <= end <= observed
            and record.get('elapsed_seconds') == (observed-start)/1e9,
            'Original complete and observation clocks must agree')
    accepted = record['persisted_pipeline_within_three_hours']
    if accepted:
        counters = result.get('completed') or {}
        require(result.get('status') == 'ok' and record.get('status') == 'ok'
                and result.get('compile_accepted') is True and result.get('frozen') is True
                and record.get('compile_accepted') is True and observed-start <= 10800*10**9
                and all(counters.get(k) == 1 for k in
                        ('prepare', 'checkpoint', 'confirm', 'math', 'emit', 'trace', 'compile'))
                and type(counters.get('step')) is int and 1 <= counters['step'] <= 25
                and result.get('attempted') == counters,
                'Positive three-hour usability requires every accepted native pipeline phase')
    require(record['persisted_pipeline_within_two_hours'] is
            bool(accepted and observed-start <= 7200*10**9),
            'The first7200seconds flag must describe this actual separately allocated case')
    return dict(usable_within_three_hours=accepted,
                usable_within_two_hours=record['persisted_pipeline_within_two_hours'])


def source_proof(manifest, kind, *, process_reader=closure._process):
    case = Path(manifest['source_case' if kind == 'agent' else 'native_case']).resolve()
    identity = ((fresh.ROUND, fresh.MODEL, fresh.MODE) if kind == 'agent'
                else (native.ROUND, native.MODEL, native.MODE))
    if not (case/'completion.json').is_file():
        raise Pending('Original '+kind+' completion pending')
    registry = closure.registry_rows(manifest['registry'], {case.name: case})
    result, done = closed(case, registry, identity=identity)
    original = paper_row(manifest['runs'], case)
    require(original.get('budget_s') == 10800, 'Original three-hour paper allocation required')
    producer = births_closed(case, process_reader=process_reader)
    proof = dict(source_result=str(case/'result.json'), source_completion=str(case/'completion.json'),
                 registry_fields=8, registry_run_id=case.name, scientific_status=result['status'],
                 producer_closure=producer, paper_record_original=original,
                 complete_case_elapsed_s=done['complete_case_elapsed_seconds'])
    if kind == 'agent':
        proof.update(clock_binding(result, done, read(case/'three-hour-record.json')))
        if proof['usable_within_three_hours']:
            require(original.get('status') == 'ok', 'Positive three-hour proof requires the unchanged successful paper row')
        proof['authoritative_record'] = str(case/'three-hour-record.json')
    else:
        compile_phase = (result.get('phases') or {}).get('compile') or {}
        observed = compile_phase.get('native_process') or {}
        proof.update(native_compile_budget_s=10800, outer_restore_compile_budget_s=13200,
                     native_timed_out=observed.get('timed_out'), native_exit_code=observed.get('exit_code'),
                     native_signal=observed.get('signal'), original_math_quality_unmeasured=True)
    return proof


def queue_proof(manifest, *, process_reader=closure._process):
    owner = Path(manifest['source_queue']).resolve()
    if not (owner/'completion.json').is_file():
        raise Pending('Original independent comparison queue is still running')
    registry = closure.registry_rows(manifest['registry'], {owner.name: owner})
    result, done = closed(owner, registry)
    require(result.get('round') == 'AUTONOMOUS-SERIAL'
            and result.get('mode') == 'finite-trusted-serial-experiments',
            'Original independent comparison controller required')
    from .bert_three_hour_comparison_queue import namespace
    declared = namespace()['validate'](read(owner/'manifest.json'), owner.parent)
    require(len(result.get('jobs') or []) == len(declared['jobs']), 'Every original job must be terminal')
    for job, spec in zip(result['jobs'], declared['jobs']):
        require(job.get('id') == spec['id'] and job.get('out') == spec['out'],
                'Original serial job order differs')
        if job.get('argv'):
            require(job['argv'] == spec['argv'] == read(owner/(spec['id']+'-command.json')),
                    'Actual command and original declared command differ')
    return dict(status=result['status'], case=str(owner), complete_case_elapsed_s=done['complete_case_elapsed_seconds'],
                producer_closure=births_closed(owner, process_reader=process_reader),
                original_job_outcomes=[dict(id=j['id'], status=j['status'], experiment_status=j.get('experiment_status'))
                                       for j in result['jobs']], unrelated_failures_preserved=True)


def attribution(manifest, proof):
    helper = load_helper(manifest['attribution_helper'], 'bert_three_hour_original_attribution')
    case = Path(manifest['source_case']).resolve()
    data = helper.build(case)
    require(not data['inflight'], 'Terminal attribution must bind the closed original source')
    data['schema'] = 'bert-fresh-3h-payload-free-attribution-v1'
    data.pop('source_two_hour_record', None)
    data['source_three_hour_record'] = str(case/'three-hour-record.json')
    data['terminal_source_admission'] = proof
    csvfile = io.StringIO()
    writer = csv.DictWriter(csvfile, fieldnames=('scope', 'stage', 'event_id', 'exclusive_ns',
                            'exclusive_s', 'additive_with_cold', 'source', 'note'))
    writer.writeheader(); writer.writerows(data['stage_rows'])
    return {ATTRIBUTION_OUTPUTS[0]: json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n',
            ATTRIBUTION_OUTPUTS[1]: csvfile.getvalue(),
            ATTRIBUTION_OUTPUTS[2]: ''.join(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n'
                                          for row in data['iteration_observations'])}


def final_quality(manifest, *, process_reader=closure._process):
    source, report, owner, queue = (Path(manifest[k]).resolve() for k in
                                  ('source_case', 'report_case', 'source_queue', 'report_queue'))
    if not (report/'result.json').is_file():
        raise Pending('No independent three-hour report result exists')
    if any(not (p/'completion.json').is_file() for p in (source, report, owner, queue)):
        raise Pending('Original source/report/queue terminal persistence is pending')
    registry = closure.registry_rows(manifest['registry'], {p.name: p for p in (source, report, owner, queue)})
    originals = {p: closed(p, registry) for p in (source, report, owner, queue)}
    fresh_result, fresh_done = originals[source]
    require(tuple(fresh_result.get(k) for k in ('round', 'model', 'mode')) ==
            (fresh.ROUND, fresh.MODEL, fresh.MODE), 'Exact original fresh source identity required')
    record = read(source/'three-hour-record.json')
    flags = clock_binding(fresh_result, fresh_done, record)
    require(flags['usable_within_three_hours'] is True, 'Only an accepted three-hour source may report')
    run, done = originals[report]
    require(tuple(run.get(k) for k in ('round', 'model', 'mode')) ==
            (report_entry.ROUND, fresh.MODEL, report_entry.MODE)
            and Path(run.get('source_case', '')).resolve() == source
            and Path(run.get('compile_case', '')).resolve() == source,
            'Exact isolated three-hour report source required')
    helper = load_helper(manifest['quality_helper'], 'bert_three_hour_original_report_scalars')
    bindings = [helper.queue_binding(owner, originals[owner][0], source, fresh.MODULE),
                helper.queue_binding(queue, originals[queue][0], report, report_entry.MODULE,
                                     source=source, owner=owner)]
    extra = {source: (Path(bindings[0]['job_resources']),), report: (Path(bindings[1]['job_resources']),)}
    closures = {p.name: births_closed(p, process_reader=process_reader, extra_resources=extra.get(p, ()))
                for p in (source, report, owner, queue)}
    admission = read(report/'admission.json')
    if run.get('admission_passed') is True:
        lineage = (admission or {}).get('source_producer_closure') or {}
        require(admission == run.get('source') and admission.get('fresh_three_hour_accepted') is True
                and admission.get('fresh_three_hour_record') == str(source/'three-hour-record.json')
                and admission.get('source_case') == str(source) and admission.get('compile_case') == str(source)
                and admission.get('report_feedback_permitted') is False
                and lineage.get('supervisor_case') == str(owner)
                and lineage.get('all_serial_jobs_terminal') is True,
                'Saved original isolated source admission must match')
    worker = read(report/'worker/worker_result.json') or {}
    if worker:
        require(worker.get('phase') == 'report' and worker.get('model') == fresh.MODEL
                and all(worker.get(k) == 0 for k in ('llm_calls', 'native_test_reads', 'compiler_runs',
                    'keygen_runs', 'real_ckks_attempts', 'real_ckks_completed')),
                'Original report must contain no unrelated scientific activity')
    access = read(report/'session/REPORT_ACCESS.json')
    if access:
        require(access.get('program') == str(report/'session/program.ptdsl')
                and (not access.get('complete') or access.get('result') == str(report/'worker/worker_result.json')),
                'Report capability must bind the private report shell')
    numeric = helper.quality(worker, report, access)
    if numeric['complete_scalar_quality']:
        require(run.get('admission_passed') is True and run.get('attempted_report') == 1,
                'Report quality requires one admitted isolated attempt')
        for name in ('shell', 'shell_recheck') if run.get('report_accepted') else ('shell',):
            shell = run.get(name) or {}
            require(shell.get('frozen_program_path_rebased_only') is True and shell.get('original_case_writes') == 0,
                    'Independent report shell provenance required')
    return dict(schema='bert-fresh-3h-final-quality-v1', observed_utc=datetime.now(timezone.utc).isoformat(),
                status='terminal_report', pending=False, source_case=str(source), report_case=str(report),
                scientific_report_status=run['status'], report_accepted=run.get('report_accepted'),
                report_error=run.get('error'), report_quality=numeric,
                **helper.quality_admission(numeric, run), **{'fresh_'+k: v for k, v in flags.items()},
                independent_report_complete_case_s=done['complete_case_elapsed_seconds'],
                report_clock_scope='Independent post-compile872 report; excluded from the original10800second pipeline',
                report_feedback_to_Agent=False, original_scientific_rows_modified=False,
                provenance=dict(registry_fields=8, queue_bindings=bindings, producer_closures=closures),
                CKKS_accuracy=None, CKKS_inference_s=None, new_scientific_actions=0)


def save(path, data):
    path = Path(path)
    require(path.parent.is_dir(), 'Existing metadata output directory required')
    temporary = path.with_name(path.name+'.bert-three-hour.tmp')
    with temporary.open('x') as stream:
        stream.write(data if isinstance(data, str) else json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
        stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)
