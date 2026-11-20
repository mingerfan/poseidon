"""Wait for the three-hour comparison, then conditionally run one frozen report.

Waiting reads metadata only. A new private serial batch takes the shared heavy
lease after the accepted source and every prior owner have actually closed.
"""
import argparse
import copy
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import resource
import signal
import time
from types import FunctionType

from .. import paths
from . import autonomous_serial_batch as batch
from . import bert_fresh_cold_3h as fresh
from . import bert_fresh_frozen_report_3h as report
from .archive_producer_closure import _process


SCHEMA='bert-three-hour-isolated-report-launch-v1'
read,save=report.base.read,report.base.save


def require(value,message):
    if not value:raise ValueError(message)


def option(argv,flag):return report._option(argv,flag)


def validate(value):
    require(value.get('schema')==SCHEMA and value.get('automatic_retry')is False,
            'One explicit conditional three-hour report launch is required')
    root=paths.RESULTS_DIR.resolve();serial=Path(value['serial_case']).resolve()
    source=Path(value['source_case']).resolve();target=Path(value['report_case']).resolve()
    manifest=value['batch'];watch=manifest['watch'];owner=Path(watch['case']).resolve()
    require(batch.deadline_timestamp(manifest.get('deadline_utc'))is not None,
            'Use the explicit timezone-aware paper deadline')
    require(len({serial,source,target,owner})==4 and all(p.parent==root for p in(serial,source,target,owner)),
            'Use distinct canonical direct source, original owner, report and new serial cases')
    jobs=manifest.get('jobs')or[]
    require(len(jobs)==1 and type(watch.get('pid'))is int and watch['pid']>0
        and type(watch.get('starttime'))is int and watch['starttime']>0,
        'One report and the actual positive original owner birth are required')
    job=jobs[0];argv=job['argv']
    require(argv[1:4]==['-B','-m',report.MODULE] and Path(option(argv,'--out')).resolve()==target
        and job['out']==str(target) and job['seconds']==1920
        and job.get('outer_address_space_gib')==4 and job.get('outer_file_gib')==8
        and Path(option(argv,'--source-case')).resolve()==source
        and Path(option(argv,'--supervisor-case')).resolve()==owner
        and job.get('requires_case_statuses')=={str(source):'ok'},
        'Keep the exact new accepted-source872 report and original4/8 parent limits')
    return value


def private_batch(value):
    validate(value);original=vars(batch);values=dict(original)
    values['MODULES']=set(batch.MODULES)|{report.MODULE.removeprefix(batch.PREFIX)}
    for name,function in original.items():
        if isinstance(function,FunctionType) and function.__globals__ is original:
            clone=FunctionType(function.__code__,values,function.__name__,function.__defaults__,function.__closure__)
            clone.__kwdefaults__=copy.copy(function.__kwdefaults__);values[name]=clone
    expected=copy.deepcopy(value['batch']['jobs'][0])
    def parent_limits(job):
        return(4,8)if job==expected else batch.parent_limits(job)
    values['parent_limits']=parent_limits
    values['validate'](value['batch'],paths.RESULTS_DIR)
    return values


def source_record(source):
    result=read(source/'result.json');completion=read(source/'completion.json')
    record=read(source/'three-hour-record.json')
    require(result.get('round')==fresh.ROUND and result.get('mode')==fresh.MODE
        and result.get('model')==fresh.MODEL and completion.get('round')==fresh.ROUND
        and completion.get('status')==result.get('status')
        and completion.get('required_artifacts_persisted')is True and completion.get('registry_persisted')is True
        and record.get('round')==fresh.ROUND and record.get('mode')==fresh.MODE
        and record.get('budget_seconds')==10800
        and record.get('started_ns')==completion.get('started_ns')
        and record.get('required_artifacts_and_registry_completed_ns')==completion.get('ended_ns'),
        'Bind the actual original three-hour source result, clock and persistence')
    accepted=record.get('persisted_pipeline_within_three_hours')
    require(type(accepted)is bool,'One actual authoritative terminal usability boolean is required')
    if accepted:
        require(result.get('status')=='ok' and result.get('compile_accepted')is True
            and record.get('status')=='ok' and record.get('compile_accepted')is True
            and completion['ended_ns']<=record['observed_ns']<=record['started_ns']+10800*10**9,
            'Positive acceptance must include actual native output and persistence by10800s')
    return dict(accepted=accepted,record=str(source/'three-hour-record.json'),
        usable_within_two_hours=record.get('persisted_pipeline_within_two_hours'),
        usable_within_three_hours=accepted,elapsed_seconds=record.get('elapsed_seconds'),
        report_feedback_to_Agent=False)


def owner_gone(pid,birth,*,probe=_process,sleep=time.sleep,clock=time.monotonic):
    start=clock()
    while True:
        actual=probe(pid)
        if actual is None or actual.get('starttime')!=birth:return
        if clock()-start>=30:raise TimeoutError('Original three-hour controller birth remains present')
        sleep(.25)


def limits(waiting):
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    resource.setrlimit(resource.RLIMIT_AS,(512*1024**2 if waiting else 4*1024**3,4*1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE,(64*1024**2 if waiting else 8*1024**3,8*1024**3))


def run(args,*,wait=batch.wait_closed,probe=_process,set_limits=limits,now=time.time):
    value=validate(read(args.manifest));scope=private_batch(value)
    ready=read(args.readiness)
    require(ready.get('status')=='ok' and ready.get('must_fix')==[] and ready.get('source_stopped')is True,
            'Frozen report and waiter tests and source review must pass')
    source=Path(value['source_case']);target=Path(value['report_case']);serial=Path(value['serial_case'])
    require(not args.metadata_out.exists() and not target.exists() and not serial.exists()
        and not args.metadata_out.resolve().is_relative_to(paths.RESULTS_DIR.resolve()),
        'Preserve all prior cases and keep waiting evidence outside scientific results')
    identity=probe(os.getpid())
    require(identity and identity.get('pid')==os.getpid()
        and type(identity.get('starttime'))is int and identity['starttime']>0,
        'The actual metadata-wait owner birth is required')
    args.metadata_out.mkdir(parents=True)
    started=time.perf_counter_ns();watch=value['batch']['watch']
    deadline=batch.deadline_timestamp(value['batch']['deadline_utc'])
    result=dict(schema='bert-three-hour-report-waiter-result-v1',status='waiting',
        started_utc=datetime.now(timezone.utc).isoformat(),source_case=str(source),report_case=str(target),
        original_queue=watch['case'],serial_case=str(serial),process=identity,
        report_dispatches=0,scientific_case_created=False,report_samples_read_by_waiter=0,
        original_clock_unchanged=True,API=0,compiler=0,CKKS=0,automatic_retry=False,error=None)
    handlers={sig:signal.getsignal(sig)for sig in(signal.SIGTERM,signal.SIGINT)}
    def interrupt(sig,frame):raise InterruptedError('Report metadata waiter received signal'+str(sig))
    try:
        for sig in handlers:signal.signal(sig,interrupt)
        set_limits(True);save(args.metadata_out/'progress.json',result)
        remaining=deadline-now()-1950
        require(remaining>0,'A complete finite report and publication must fit the paper deadline')
        result['wait']=wait(dict(watch,seconds=min(watch['seconds'],remaining)))
        owner_gone(watch['pid'],watch['starttime'],probe=probe)
        require(now()+1950<deadline,'Full report allowance no longer fits the paper deadline')
        if not(source/'three-hour-record.json').is_file():
            original=read(Path(watch['case'])/'result.json')
            matched=[job for job in original.get('jobs',[])if job.get('out')==str(source)]
            require(len(matched)==1 and matched[0].get('status')!='ok',
                    'An absent source record requires a saved unsuccessful original source job')
            result.update(status='skipped_unaccepted_source',reason='The original source never produced an accepted complete pipeline')
            return result
        result['source_admission']=source_record(source)
        if not result['source_admission']['accepted']:
            result.update(status='skipped_unaccepted_source',reason='Actual three-hour acceptance was false')
            return result
        require(not target.exists() and not serial.exists(),'Keep the new report and serial output unused')
        set_limits(False)
        batch_file=args.metadata_out/'report-batch.json';save(batch_file,value['batch'])
        save(args.metadata_out/'launch-intent.json',dict(source_admission=result['source_admission'],
            batch=str(batch_file),serial_case=str(serial),actual_heavy_lease_acquired=False))
        result['batch_returncode']=scope['run'](argparse.Namespace(manifest=batch_file,out=serial,lease_wait_seconds=60))
        actual=read(serial/'result.json');result['serial_result']=actual
        result['report_dispatches']=sum('process'in job for job in actual.get('jobs',[]))
        result.update(status='completed',scientific_case_created=target.exists())
    except BaseException as exc:
        result.update(status='error',error=type(exc).__name__+': '+str(exc))
    finally:
        for sig in handlers:signal.signal(sig,signal.SIG_IGN)
        try:
            result['elapsed_seconds']=(time.perf_counter_ns()-started)/1e9
            save(args.metadata_out/'result.json',result)
            save(args.metadata_out/'completion.json',dict(status=result['status'],metadata_persisted=True,
                report_dispatches=result['report_dispatches'],scientific_case_created=target.exists()))
        finally:
            for sig,handler in handlers.items():signal.signal(sig,handler)
    return result


def main(argv=None):
    cli=argparse.ArgumentParser(description=__doc__)
    for name in('manifest','readiness','metadata-out'):cli.add_argument('--'+name,type=Path,required=True)
    args=cli.parse_args(argv)
    for name,value in vars(args).items():setattr(args,name,value.expanduser().resolve())
    result=run(args)
    print(json.dumps({key:result.get(key)for key in('status','report_dispatches','scientific_case_created','error')}))
    return int(result['status']=='error')


if __name__=='__main__':raise SystemExit(main())
