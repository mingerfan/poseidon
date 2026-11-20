"""Saved scalar evidence for the recovered comparison; originals stay intact."""
import copy
from pathlib import Path
from types import FunctionType, SimpleNamespace

from . import bert_three_hour_terminal_metadata as original
from . import bert_recovery_frozen_report as report
from . import bert_supervisor_recovery as recovery

FRESH = recovery.FRESH.replace('-20261117','-logquota-recovery-20261117')
SEARCH = original.SEARCH
QUEUE = 'bert-native10800-then-fresh-agent-logquota-recovery-serial-20261117'
REPORT = 'bert-sst2-fresh-cold-agent-3h-recovery-isolated-report-20261117'
REPORT_QUEUE = 'bert-fresh-3h-recovery-isolated-report-serial-20261117'
ATTRIBUTION_OUTPUTS = ('bert_3h_recovery_agent_attribution.json',
    'bert_3h_recovery_agent_stage_times.csv', 'bert_3h_recovery_agent_iteration_observations.jsonl')


def namespace():
    source=vars(original)
    values=dict(source,FRESH=FRESH,SEARCH=SEARCH,QUEUE=QUEUE,REPORT=REPORT,
                REPORT_QUEUE=REPORT_QUEUE,ATTRIBUTION_OUTPUTS=ATTRIBUTION_OUTPUTS,report_entry=report)
    for name,fn in source.items():
        if isinstance(fn,FunctionType) and fn.__globals__ is source:
            clone=FunctionType(fn.__code__,values,fn.__name__,fn.__defaults__,fn.__closure__)
            clone.__kwdefaults__=copy.copy(fn.__kwdefaults__);values[name]=clone
    def queue_proof(manifest, *, process_reader=original.closure._process):
        owner=Path(manifest['source_queue']).resolve()
        if not (owner/'completion.json').is_file():
            raise original.Pending('Recovered independent comparison queue is still running')
        registry=original.closure.registry_rows(manifest['registry'],{owner.name:owner})
        result,done=values['closed'](owner,registry)
        original.require(owner.name==QUEUE and result.get('round')=='AUTONOMOUS-SERIAL'
            and result.get('mode')=='finite-trusted-serial-experiments',
            'Exact recovered serial owner required')
        declared=recovery.namespace(limits={})['validate'](values['read'](owner/'manifest.json'),owner.parent)
        original.require(len(result.get('jobs')or[])==len(declared['jobs']),
                         'Every recovered independent job must be terminal')
        for job,spec in zip(result['jobs'],declared['jobs']):
            original.require(job.get('id')==spec['id'] and job.get('out')==spec['out'],
                             'Recovered job identity/order differs')
            if job.get('argv'):
                original.require(job['argv']==spec['argv']==values['read'](owner/(spec['id']+'-command.json')),
                                 'Recovered actual and declared commands differ')
        limits=values['read'](owner/'supervisor-file-limits.json')
        original.require(limits.get('file_bytes')==16*recovery.GIB and limits.get('after',[None])[0]==16*recovery.GIB,
                         'Actual recovered supervisor own log allowance required')
        return dict(status=result['status'],case=str(owner),
            complete_case_elapsed_s=done['complete_case_elapsed_seconds'],
            producer_closure=values['births_closed'](owner,process_reader=process_reader),
            original_job_outcomes=[dict(id=j['id'],status=j['status'],experiment_status=j.get('experiment_status'))for j in result['jobs']],
            unrelated_failures_preserved=True,original_interrupted_case_preserved=True,
            supervisor_file_limits=limits)
    values['queue_proof']=queue_proof
    return SimpleNamespace(**values)
