"""Verify saved successful VGG worker evidence after a JSON readback rejection.

No model, candidate, confirmation or report is evaluated. The original outer
error and its clocks remain untouched; this case owns only a metadata audit.
"""
import argparse
from datetime import datetime,timezone
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import time

from .. import paths
from ..autonomous_budget import call_records
from . import r30_continue_common_session as continuation
from .r30_continue_alex_session import same_state,_usage
from .r10_paper_metrics import StageRecorder,read_events,summarize_events,write_json
from .completion_metrics import artifact_bytes,persist_case_completion

ROUND='VGG-CONTINUATION-TERMINAL-AUDIT'
MODE='metadata-only-vgg-continuation-json-readback-recovery'
ERROR='ValueError: Worker terminal result differs from the restored session'
FIELDS={'run_id','round','model','mode','git_rev','metrics','artifacts_dir','ts'}
OLD_METRICS=('status','variant','cold_start','original_api_calls','actual_new_api_calls',
             'worker_attempts','worker_completed','original_time_seconds')


def read(path):
    path=Path(path)
    if path.is_symlink()or not path.is_file():raise ValueError('A retained regular metadata file is required')
    value=json.loads(path.read_text())
    if not isinstance(value,dict):raise ValueError('Metadata must be an object')
    return value


def _registry_rows(registry,case):
    with Path(registry).open()as stream:
        return [value for line in stream if line.strip()
                for value in [json.loads(line)]if value.get('artifacts_dir')==str(case)]


def audit_source(case,*,ledger,registry):
    """Check public saved source/SESSION/ledger metadata only, with no loader."""
    from ..planners.transformer_session import load_state
    case=Path(case).resolve();record=read(case/'result.json');closed=read(case/'completion.json')
    if (record.get('round')!=continuation.ROUND or record.get('mode')!=continuation.MODE
            or record.get('model')!=continuation.MODEL or record.get('status')!='error'
            or record.get('error')!=ERROR or record.get('worker_attempts')!=1
            or record.get('worker_completed')!=0 or record.get('original_api_calls')!=4
            or record.get('actual_new_api_calls')!=2 or record.get('cold_start')is not False
            or record.get('resource_monitor_error')is not None
            or closed.get('status')!='error' or closed.get('required_artifacts_persisted')is not True
            or closed.get('registry_persisted')is not True or not continuation._clean(record.get('process')or{})):
        raise ValueError('Only the exact closed readback rejection with one clean worker is eligible')
    for key in ('complete_case_elapsed_seconds',):
        if type(closed.get(key))not in(int,float)or not math.isfinite(closed[key])or closed[key]<0:
            raise ValueError('Preserve a finite original continuation clock')
    rows=_registry_rows(registry,case)
    if (len(rows)!=1 or set(rows[0])!=FIELDS or rows[0]['run_id']!=case.name
            or rows[0]['round']!=continuation.ROUND or rows[0]['model']!=continuation.MODEL
            or rows[0]['mode']!=continuation.MODE or rows[0]['metrics']!={k:record.get(k)for k in OLD_METRICS}):
        raise ValueError('The original unique registry row must attest the saved outer error')
    source=record.get('source')or{}
    initial,verified_source=continuation.check_source(source['original_case'],ledger=ledger)
    if not same_state(source,verified_source):raise ValueError('Original retained source admission differs')
    inherited=load_state(case/'inherited-agent/SESSION.json')
    if not same_state(inherited,initial):raise ValueError('The inherited whole SESSION differs from the original')
    agent=case/'work/agent';session=load_state(agent/'SESSION.json');worker=read(agent/'worker-result.json')
    if not same_state(record.get('worker'),worker):raise ValueError('Outer saved worker differs from its retained worker file')
    before,after=record.get('api_before'),record.get('api_after');calls=call_records(ledger)
    if (type(before)is not int or type(after)is not int or before<0 or after-before!=2 or after>len(calls)):
        raise ValueError('Exactly two retained new ledger attempts are required')
    provider=continuation.check_provider_history(initial,session,agent,2)
    requests=sorted((agent/'api').glob('*.request.json'))
    for index,request in enumerate(requests):
        response=read(request.with_name(request.name.removesuffix('.request.json')+'.response.json'))
        entry=calls[before+index]
        if (entry.get('status')!=200 or entry.get('error')is not None
                or entry.get('purpose')!='agent-iter-'+str(5+index)):
            raise ValueError('Both new calls must retain their actual successful ledger records')
        _usage(response,entry)
    confirms=[event for event in read_events(agent/'events.jsonl')if event.get('stage_id')=='agent_confirm_select']
    begins=[event for event in confirms if event.get('type')=='begin']
    ends=[event for event in confirms if event.get('type')=='end']
    if (len(begins)!=1 or len(ends)!=1 or begins[0].get('event_id')!=ends[0].get('event_id')
            or ends[0].get('status')!='ok' or ends[0].get('error')is not None
            or type(ends[0].get('end_ns'))is not int or ends[0]['end_ns']<begins[0]['start_ns']):
        raise ValueError('One already-completed original confirmation span is required')
    if continuation.check_worker(worker,session,source,2,agent_dir=agent)!='ok':
        raise ValueError('The saved worker must verify one accepted frozen selection and all six strict checks')
    selected=worker['agent']['confirm']['selected']
    confirmed=[row for row in worker['agent']['confirm']['evals']
               if(row.get('iteration'),row.get('origin'))==(selected['iteration'],selected['origin'])]
    return dict(source_case=str(case),source_result=str(case/'result.json'),worker_result=str(agent/'worker-result.json'),
        selected_program=str(Path(worker['selected_program']).resolve()),original_case=source['original_case'],
        model_info=source['model_info'],calibration=source['calibration'],model=continuation.MODEL,
        factory=continuation.FACTORY,source_outer_status='error',source_outer_error=ERROR,
        saved_worker_exit_code=0,saved_worker_status='ok',formal_frozen=True,quality_acceptance_passed=True,
        strict_math_passed=True,strict_math_checks=6,confirmation_runs_verified=1,
        original_api_calls=4,continued_api_calls=2,total_session_calls=6,provider_history=provider,
        ledger_interval=dict(before=before,after=after),selected_iteration=selected['iteration'],
        selected_origin=selected['origin'],confirmation_metrics={key:confirmed[0].get(key)for key in
            ('total','passed','diverged','n_diverged','acc_drop_pp','max_u_calibrated','max_u_analytic')},
        boots_pred=worker.get('boots_pred'),latency_pred_us=worker.get('latency_pred_us'),
        original_cold_complete_seconds=source['original_complete_seconds'],
        original_continuation_complete_seconds=closed['complete_case_elapsed_seconds'],
        clock_scope='Original cold and continuation clocks stay separate; this new clock covers metadata audit only',
        selected_AST_verified=True,JSON_confirmation_semantics_verified=True,
        source_git_rev=rows[0]['git_rev'],source_case_modified=False)


def execute(args):
    out=Path(args.out).resolve();source=Path(args.source_case).resolve();root=paths.RESULTS_DIR.resolve()
    registry=Path(args.registry or root/'registry.jsonl')
    if(out.exists()or not out.is_relative_to(root)or out==root or out.is_relative_to(source)
            or source==root or not source.is_relative_to(root)):
        raise ValueError('Choose a fresh audit case under RESULTS, separate from its source')
    started=time.perf_counter_ns();out.mkdir();recorder=StageRecorder(out/'events.jsonl')
    result=dict(round=ROUND,mode=MODE,model=continuation.MODEL,status='running',error=None,
        source_case=str(source),model_runs=0,api_calls=0,new_confirmations=0,report_samples=0,
        compiler_runs=0,real_ckks_runs=0,extra_integrity_checks=0,source_case_modified=False,
        formal_frozen=False,quality_acceptance_passed=False,strict_math_passed=False,
        compiled_bootstraps=None,CKKS_latency=None,accuracy_acceptance=None,security_acceptance=None)
    try:
        with recorder.span('saved_metadata_source_and_terminal_checks'):
            proof=audit_source(source,ledger=args.ledger,registry=registry)
            result.update(status='ok',outcome='quality_accepted',proof=proof,
                formal_frozen=True,quality_acceptance_passed=True,strict_math_passed=True,
                worker_result=proof['worker_result'],selected_program=proof['selected_program'])
            shutil.copyfile(__file__,out/'audit-source.py')
            shutil.copyfile(continuation.__file__,out/'readback-source.py')
    except Exception as exc:
        result.update(status='error',error=type(exc).__name__+': '+str(exc))
    timing=summarize_events(read_events(out/'events.jsonl'),time.perf_counter_ns())
    result['timing']=timing;write_json(out/'result.json',result)
    row=dict(run_id=out.name,round=ROUND,model=continuation.MODEL,mode=MODE,git_rev=None,
        metrics={key:result.get(key)for key in ('status','formal_frozen','quality_acceptance_passed',
            'strict_math_passed','api_calls','model_runs','new_confirmations','report_samples')},
        artifacts_dir=str(out),ts=datetime.now(timezone.utc).isoformat())
    with registry.open('a+')as stream:
        fcntl.flock(stream,fcntl.LOCK_EX);stream.seek(0)
        if any(json.loads(line).get('run_id')==out.name for line in stream if line.strip()):
            raise ValueError('Preserve existing audit registry identities')
        stream.seek(0,os.SEEK_END);stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
    persist_case_completion(out,round_label=ROUND,started_ns=started,status=result['status'],
        timing=timing,required_artifact_bytes=artifact_bytes(out))
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-case',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--ledger',type=Path,default=paths.RESULTS_DIR/'llm_calls.jsonl');parser.add_argument('--registry',type=Path)
    args=parser.parse_args(argv)
    return 0 if execute(args)['status']=='ok'else 1


if __name__=='__main__':raise SystemExit(main())
