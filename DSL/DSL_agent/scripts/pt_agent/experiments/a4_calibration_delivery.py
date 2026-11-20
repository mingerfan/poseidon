"""Deliver closed A4 calibration metadata without rerunning an experiment.

The finite watcher reads JSON records, never a model, weight, key or ciphertext.
Public fixture costs remain separate from full-model benchmark observations.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import time

from .. import paths
from .r10_paper_metrics import summarize_events, write_json
from .completion_metrics import artifact_bytes, persist_case_completion

MAX_JSON_BYTES=64*1024**2
MAX_REGISTRY_BYTES=128*1024**2
REGISTRY_FIELDS={'run_id','round','model','mode','git_rev','metrics','artifacts_dir','ts'}
CUTOFF=datetime(2026,11,18,12,tzinfo=timezone.utc)
NATIVE_CASE='a4-resnet20-123-prepared-disk-keyfree-as4-r31-20261117'
CALIBRATION_CASE='a4-independent39-repeated-calibration-20261117'


def finite(value):
    if isinstance(value,float)and not math.isfinite(value):raise ValueError('Nonfinite source JSON number')
    if isinstance(value,dict):
        for item in value.values():finite(item)
    elif isinstance(value,list):
        for item in value:finite(item)
    return value


def parse(text):
    def reject(value):raise ValueError('Nonfinite JSON literal: '+value)
    return finite(json.loads(text,parse_constant=reject))


def read(path):
    path=Path(path)
    if path.stat().st_size>MAX_JSON_BYTES:raise ValueError('Source metadata JSON exceeds64MiB')
    return parse(path.read_text())


def inspect_case(case,registry):
    """Require the actual case result, completion and unique registry closure."""
    case=Path(case);result=case/'result.json';completion=case/'completion.json'
    reference=dict(case=str(case),result=str(result),completion=str(completion),registry=str(registry))
    if not result.is_file()or not completion.is_file():
        return dict(state='pending',source=reference,detail='Result or completion not yet persisted')
    record=read(result);closed=read(completion)
    if not Path(registry).is_file():return dict(state='pending',source=reference,detail='Registry not yet present')
    if Path(registry).stat().st_size>MAX_REGISTRY_BYTES:raise ValueError('Source registry exceeds128MiB metadata allowance')
    matches=[]
    with Path(registry).open()as stream:
        fcntl.flock(stream,fcntl.LOCK_SH)
        for line in stream:
            if not line.strip():continue
            if len(line)>MAX_JSON_BYTES:raise ValueError('Source registry row exceeds metadata allowance')
            row=parse(line)
            if row.get('run_id')==case.name:matches.append(row)
    if not matches:return dict(state='pending',source=reference,detail='Case registry row not yet persisted')
    if len(matches)!=1:raise ValueError('Case registry identity is not unique: '+case.name)
    row=matches[0]
    if (set(row)!=REGISTRY_FIELDS or Path(row.get('artifacts_dir','')).resolve()!=case.resolve()
            or closed.get('registry_persisted')is not True or closed.get('required_artifacts_persisted')is not True
            or record.get('status')not in('ok','error')or closed.get('status')!=record['status']
            or (row.get('metrics')or{}).get('status')!=record['status']):
        raise ValueError('Case result/completion/registry contract differs: '+case.name)
    reference.update(run_id=row['run_id'],round=row['round'],registry_ts=row['ts'])
    return dict(state='closed',status=record['status'],source=reference,record=record,
                completion_record=closed,registry_record=row)


def source_status(case):
    return dict(state=case.get('state','unknown'),status=case.get('status'),
                detail=case.get('detail')or(case.get('record')or{}).get('error'),source=case.get('source'))


def validate_calibration(summary,projection,native):
    if (summary.get('status')!='ok'or summary.get('cell_count')!=39 or len(summary.get('cells',[]))!=39
            or summary.get('target_count')!=200 or summary.get('old_durations_reused')is not False
            or summary.get('parameter_group')!='r31-N16-C2S52-P61-scale50'):
        raise ValueError('Require the fresh complete39/200-target calibration table')
    repeats=[row.get('repetitions')for row in summary['cells']]
    if repeats.count(5)!=38 or repeats.count(10)!=1:raise ValueError('Calibration repetition counts differ')
    if (projection.get('execution_count')!=17863 or projection.get('bootstrap_count')!=123
            or projection.get('old_durations_reused')is not False
            or projection.get('fit_to_whole_model_clock')is not False
            or projection.get('full_inference_prediction_s')is not None):
        raise ValueError('Require original123-plan independent operator-only projection')
    source_plan=((native.get('record')or{}).get('source')or{}).get('plan')
    if source_plan is not None and Path(projection.get('source_plan','')).resolve()!=Path(source_plan).resolve():
        raise ValueError('Projection is bound to a different full-model RuntimePlan')


def build_payloads(calibration,native,*,stopped_reason=None,now=None):
    now=now or datetime.now(timezone.utc)
    refs=dict(calibration=source_status(calibration),native=source_status(native))
    status='ok' if calibration.get('state')=='closed'and calibration.get('status')=='ok'else 'driver_error'
    detail=stopped_reason or refs['calibration'].get('detail')
    summary=projection=None
    source_case=Path((calibration.get('source')or{}).get('case','.'))
    if calibration.get('state')=='closed':
        summary_path=source_case/'calibration-summary.json';projection_path=source_case/'operator-projection.json'
        refs['calibration'].update(summary=str(summary_path),operator_projection=str(projection_path))
        if summary_path.is_file():summary=read(summary_path)
        if projection_path.is_file():projection=read(projection_path)
        if status=='ok':
            if summary is None or projection is None:
                status='driver_error';detail='Closed calibration success lacks required table/projection metadata'
            else:
                try:validate_calibration(summary,projection,native)
                except ValueError as exc:status='driver_error';detail=str(exc)
    if stopped_reason is not None:status='timeout'
    common=dict(schema_version=1,status=status,status_detail=detail,source=refs,
        delivered_utc=now.isoformat(),extra_integrity_checks=0,output_readback_checks=0,
        new_api_calls=0,new_native_executions=0,public_fixture_is_whole_model_benchmark=False)
    raw_summary=dict(common,data=summary)
    raw_projection=dict(common,data=projection,full_formal_prediction_s=None)
    native_record=native.get('record')or{};native_summary=native_record.get('client_summary')or{}
    actual_s=None;actual_scope=native_summary.get('prepared_inference_scope')
    if native.get('state')=='closed'and native.get('status')=='ok':
        candidate=native_record.get('ckks_inference_s')
        comparison=native_record.get('comparison')or{}
        if (type(candidate)in(int,float)and math.isfinite(candidate)and candidate>0
                and comparison.get('passed')is True and comparison.get('bootstrap_count')==123
                and comparison.get('execution_count')==17863
                and native_summary.get('ckks_inference_completed')is True
                and native_summary.get('no_keygen_or_constant_encoding_during_inference')is True
                and isinstance(actual_scope,str)and actual_scope
                and native_summary.get('ckks_inference_s')==candidate):
            actual_s=candidate
    predicted_s=None
    if status=='ok'and projection is not None and projection.get('status')=='complete_operator_projection':
        candidate=projection.get('operator_component_prediction_s')
        if type(candidate)in(int,float)and math.isfinite(candidate)and candidate>0:predicted_s=candidate
    ratio=predicted_s/actual_s if predicted_s is not None and actual_s is not None else None
    diagnostic=dict(common,native=dict(status=refs['native'],ckks_inference_s=actual_s,scope=actual_scope),
        operator_projection=dict(operator_component_prediction_s=predicted_s,
            scope=(projection or{}).get('scope'),unmeasured_components=(projection or{}).get('unmeasured_components')),
        operator_projection_over_actual_ckks_ratio=ratio,ratio_label='Scope diagnostic only',
        full_formal_prediction_s=None,fit_to_whole_model_clock=False,
        interpretation='Independent public operator projection and full-model encrypt-to-decrypt clock have different coverage and disk/cache working sets; the ratio does not measure full-inference prediction accuracy')
    return dict(a4_prepared_disk_calibration=raw_summary,
                a4_prepared_disk_operator_projection=raw_projection,
                a4_prepared_disk_scope_diagnostic=diagnostic)


def publish_new(destination,payloads):
    """Publish fresh finite JSON once, with no readback or overwrite."""
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    targets={name:destination/(name+'.json')for name in payloads}
    if any(path.exists()for path in targets.values()):raise ValueError('Preserve an earlier A4 metadata publication')
    published=[]
    for name,payload in payloads.items():
        finite(payload);target=targets[name];temporary=target.with_name('.'+target.name+'.partial-'+str(os.getpid()))
        with temporary.open('x')as stream:
            json.dump(payload,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
        os.link(temporary,target);temporary.unlink();published.append(str(target))
    return published


def watch(args):
    out=args.out.absolute()
    if out.exists():raise ValueError('Fresh metadata watcher evidence directory required')
    out.mkdir(parents=True)
    started=time.perf_counter_ns();limit=time.monotonic()+args.seconds
    cache={};result=dict(status='running',error=None,polls=0,new_api_calls=0,
        new_native_executions=0,extra_integrity_checks=0,output_readback_checks=0,
        no_benchmark_runs_appended=True,published=[])
    latest=dict(calibration=dict(state='pending',source=dict(case=str(args.calibration))),
                native=dict(state='pending',source=dict(case=str(args.native))))
    stopped_reason=None
    try:
        while True:
            now=datetime.now(timezone.utc)
            if time.monotonic()>=limit or now>=args.deadline:
                stopped_reason='Finite metadata watcher allowance or submission cutoff reached'
                break
            for key,case in(('calibration',args.calibration),('native',args.native)):
                if key not in cache:
                    latest[key]=inspect_case(case,args.registry)
                    if latest[key]['state']=='closed':cache[key]=latest[key]
                else:latest[key]=cache[key]
            result['polls']+=1
            with(out/'observations.jsonl').open('a')as stream:
                stream.write(json.dumps(dict(ts=now.isoformat(),poll=result['polls'],
                    calibration=source_status(latest['calibration']),native=source_status(latest['native'])),allow_nan=False)+'\n')
            if all(row.get('state')=='closed'for row in latest.values()):break
            time.sleep(max(0,min(10,limit-time.monotonic(),(args.deadline-now).total_seconds())))
        payloads=build_payloads(latest['calibration'],latest['native'],stopped_reason=stopped_reason)
        write_json(out/'source-observations.json',latest)
        for name,payload in payloads.items():write_json(out/(name+'.json'),payload)
        result['published']=publish_new(args.destination,payloads)
        result.update(status='ok',delivery_status=payloads['a4_prepared_disk_calibration']['status'],
            stopped_reason=stopped_reason,source_status={key:source_status(value)for key,value in latest.items()})
    except BaseException as exc:
        result.update(status='error',error=type(exc).__name__+': '+str(exc))
        write_json(out/'source-observations.json',latest)
        try:
            failed=dict(state='pending',detail=result['error'],source=dict(case=str(args.calibration)))
            payloads=build_payloads(failed,latest['native'])
            for name,payload in payloads.items():write_json(out/(name+'-failure.json'),payload)
            result['published']=publish_new(args.destination,payloads)
            result['delivery_status']='driver_error'
        except BaseException as publication_error:
            result['failure_publication_error']=type(publication_error).__name__+': '+str(publication_error)
    finally:
        result['elapsed_s']=(time.perf_counter_ns()-started)/1e9
        write_json(out/'result.json',result)
        entry=dict(run_id=out.name,round='A4-METADATA-DELIVERY',model='metadata-only',mode='finite-closed-case-watch',
            git_rev=None,metrics=dict(status=result['status'],delivery_status=result.get('delivery_status'),
                new_api_calls=0,new_native_executions=0),artifacts_dir=str(out),ts=datetime.now(timezone.utc).isoformat())
        with args.registry.open('a+')as stream:
            fcntl.flock(stream,fcntl.LOCK_EX);stream.seek(0)
            if any(parse(line).get('run_id')==out.name for line in stream if line.strip()):raise ValueError('Preserve previous watcher registry identity')
            stream.seek(0,os.SEEK_END);stream.write(json.dumps(entry,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
        persist_case_completion(out,round_label=entry['round'],started_ns=started,status=result['status'],
            timing=summarize_events([]),required_artifact_bytes=artifact_bytes(out))
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--calibration',type=Path,default=paths.RESULTS_DIR/CALIBRATION_CASE)
    parser.add_argument('--native',type=Path,default=paths.RESULTS_DIR/NATIVE_CASE)
    parser.add_argument('--registry',type=Path,default=paths.RESULTS_DIR/'registry.jsonl')
    parser.add_argument('--destination',type=Path,default=paths.WORKSPACE/'arranger-dac/data/codex')
    parser.add_argument('--seconds',type=int,default=18000)
    parser.add_argument('--deadline',type=datetime.fromisoformat,default=CUTOFF)
    args=parser.parse_args(argv)
    if not 1<=args.seconds<=18000:parser.error('Finite watcher allowance1..18000s required')
    if args.deadline.tzinfo is None or args.deadline>CUTOFF:parser.error('Explicit timezone and cutoff no later than18 November12UTC required')
    return 0 if watch(args)['status']=='ok'else 1


if __name__=='__main__':raise SystemExit(main())
