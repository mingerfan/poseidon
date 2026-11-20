"""Frozen-only BERT continuation; every original Agent step is forbidden.

Admission reads retained metadata first. A session still evaluating/tuning or
confirming closes as an ok/observed_not_eligible metadata observation before the original source/audit
orchestration. Accepted done sessions reuse the original report and compiler
protocols, including their original task/math criteria and terminal accounting.
"""
import argparse
import copy
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import time
import types

from . import bert_serial_continuation as base

ROUND='BERT-POSTFREEZE-COMPILE-CONTINUATION'
MODEL=base.MODEL
MODE='retained-complete-c2-freeze-report-compile-zero-agent-steps'
MODULE='pt_agent.experiments.bert_postfreeze_compile_continuation'
OVERALL_SECONDS=21600
ZERO=('new_agent_steps','new_tune_commands','new_confirmation_commands','new_API_requests')


def validate(args):
    if type(args.overall_seconds) is not int or not 0<args.overall_seconds<=OVERALL_SECONDS:
        raise ValueError('Use the finite frozen-only allowance1..21600seconds')
    root=base.paths.RESULTS_DIR.resolve()
    cases=[args.source_case,args.out,args.checkpoint_out,args.compile_out]
    if any(case.parent!=root for case in cases) or len(set(cases))!=4:
        raise ValueError('Keep four separate direct result directories')
    if args.out.exists() or args.compile_out.exists():
        raise ValueError('Use explicitly fresh outer and compile cases; retain all prior attempts')
    args.max_steps=0;args.step_seconds=base.DEFAULT_STEP_SECONDS;args.launched=False


def admission(args):
    """Complete original metadata gates, with no model/source audit command."""
    state=base.session_state(args.source_case)
    if state['phase']!='done' or state['failed'] is not None:
        return dict(eligible=False,reason='Original SESSION is not done with failed=None',session=state)
    outcome=base.frozen_state(args.source_case)
    if outcome.get('frozen') is not True:
        return dict(eligible=False,reason='Original completed session selected no accepted frozen program',
                    session=state,frozen=outcome)
    marker=args.source_case/'session/REPORT_ACCESS.json'
    report=base.read(marker) if marker.exists() else None
    if report is not None:
        if report.get('complete') is not True:
            raise ValueError('Preserve incomplete report access; refuse another report attempt')
        path=Path(report['result']).resolve()
        if (not base.same(report.get('program'),outcome['program'])
                or path.parent.parent!=args.source_case/'phases' or not path.parent.name.endswith('-report')
                or not path.is_file()):
            raise ValueError('The exact complete frozen report marker/result path differs')
    return dict(eligible=True,session=state,frozen=outcome,report_access=report,
        frozen_records={name:base.read(args.source_case/'session'/name) for name in ('AGENT.json','FROZEN.json')},
        program_text=Path(outcome['program']).read_text())


def _same_freeze(args,accepted):
    current=admission(args)
    # Report access may advance absent->complete once. The candidate and its
    # original independent confirmation must remain identical throughout.
    keys=('eligible','session','frozen','frozen_records','program_text')
    if current.get('eligible') is not True or any(current.get(key)!=accepted.get(key) for key in keys):
        raise ValueError('The admitted original frozen candidate or independent C2 metadata changed')
    return current


def command(args,source,phase,*,accepted):
    if phase not in ('checkpoint','report','compile'):
        raise ValueError('Frozen-only continuation forbids Agent step, tune, confirmation and Provider commands')
    current=_same_freeze(args,accepted)
    if phase=='report' and current['report_access'] is not None:
        raise ValueError('A complete report already exists; use the exact original saved report')
    return base.command(args,source,phase)


def namespace(accepted,source):
    """Private trusted plain-function globals; context managers remain original."""
    values=dict(vars(base),ROUND=ROUND,MODEL=MODEL,MODE=MODE,MODULE=MODULE,__file__=__file__,
                ZERO_FIELDS=base.ZERO_FIELDS+ZERO)
    for name,function in vars(base).items():
        if (isinstance(function,types.FunctionType) and function.__module__==base.__name__
                and function.__closure__ is None):
            cloned=types.FunctionType(function.__code__,values,name,function.__defaults__,None)
            cloned.__kwdefaults__=copy.copy(function.__kwdefaults__);values[name]=cloned
    def source_gate(args):
        _same_freeze(args,accepted)
        return copy.deepcopy(source)
    def original_command(args,record,phase):return command(args,record,phase,accepted=accepted)
    def snapshot(args,folder):
        base._snapshot_state(args,folder)
        target=args.out/'original-serial-driver-source.py'
        if not target.exists():shutil.copyfile(Path(base.__file__),target)
    values.update(source_gate=source_gate,command=original_command,_snapshot_state=snapshot)
    return values


def observed_terminal(args,*,started_ns,status,error=None,observation=None):
    """Close metadata-only admission using original reserve/registry/completion."""
    args.out.mkdir();recorder=base.StageRecorder(args.out/'events.jsonl')
    with recorder.span('postfreeze_admission_observation'):
        shutil.copyfile(Path(__file__),args.out/'driver-source.py')
        shutil.copyfile(Path(base.__file__),args.out/'original-serial-driver-source.py')
    result=dict(round=ROUND,model=MODEL,mode=MODE,status=status,error=error,admission=observation,
        outcome='observed_not_eligible' if status=='ok' else 'admission_error',eligible=False,
        postfreeze_compile_completed=False,
        processes=[],attempted=dict.fromkeys(base.PHASE_LIMITS,0),completed=dict.fromkeys(base.PHASE_LIMITS,0),
        frozen=False,report_completed=False,task_report_acceptance=None,compile_completed=False,
        compiled_bootstraps=None,predicted_latency_us=None,CKKS_latency=None,security_acceptance=None,
        source_case=str(args.source_case),overall_seconds=args.overall_seconds,max_step_commands=0,
        timing=base.summarize_events(base.read_events(args.out/'events.jsonl')),
        **dict.fromkeys(base.ZERO_FIELDS+ZERO,0))
    result['clock_scope']='Metadata-only entry/admission/required persistence; no checkpoint/model/report/compiler phase started'
    for _ in range(20):
        size=base.artifact_bytes(args.out)
        result['terminal_budget']=base.terminal_budget(size,shutil.disk_usage(args.out).free)
        if result['terminal_budget']['reasons']:
            result['status']='error';result['error']='Admission metadata resource allowance exhausted'
        result['driver_artifact_bytes_before_completion']=size
        result['metrics']={key:result[key] for key in ('status','outcome','eligible','postfreeze_compile_completed','attempted','completed','frozen','report_completed',
            'compile_completed','compiled_bootstraps','predicted_latency_us','CKKS_latency','security_acceptance',*ZERO)}
        base.durable_json(args.out/'result.json',result)
        if base.artifact_bytes(args.out)==size:break
    else:raise RuntimeError('Frozen-only observation inventory did not converge')
    from ..registry import git_rev
    row=dict(run_id=args.out.name,round=ROUND,model=MODEL,mode=MODE,git_rev=git_rev(),metrics=result['metrics'],
        artifacts_dir=str(args.out),ts=datetime.now(timezone.utc).isoformat())
    with (base.paths.RESULTS_DIR/'registry.jsonl').open('a') as stream:
        fcntl.flock(stream.fileno(),fcntl.LOCK_EX)
        stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
    values=namespace({},{});values['persist_terminal_completion'](args.out,started_ns=started_ns,result=result,
        allow_resource_failure=bool(result['terminal_budget']['reasons']))
    return result


def execute(args):
    validate(args);started=time.perf_counter_ns()
    try:
        accepted=admission(args)
        if not accepted['eligible']:
            return observed_terminal(args,started_ns=started,status='ok',observation=accepted)
        # This first original source preflight occurs only after complete freeze.
        source=base.source_gate(args)
        source['postfreeze_contract']=dict(variant=MODE,frozen=accepted['frozen'],
            zero_additional_commands=dict.fromkeys(ZERO,0),maximum_step_commands=0,
            original_report_reuse='Absent permits one original report; complete is strictly reused; incomplete refuses',
            selected_by_report=False,selected_by_compiler=False,
            admission_started_ns=started,admission_ended_ns=time.perf_counter_ns(),
            clock_scope='Admission timestamps precede the original delegated serial root clock; queue process records include both')
    except Exception as exc:
        return observed_terminal(args,started_ns=started,status='error',error=type(exc).__name__+': '+str(exc))
    # No max-steps CLI exists. Original loop has zero allowance and an explicit
    # phase-command rejection provides a second independent barrier to steps.
    return namespace(accepted,source)['execute'](args)


def parser():
    result=argparse.ArgumentParser(description=__doc__)
    for name in ('source-case','out','checkpoint-out','compile-out'):
        result.add_argument('--'+name,type=Path,required=True)
    result.add_argument('--intake-python',type=Path,default=base.paths.INTAKE_PYTHON)
    result.add_argument('--data-python',type=Path,default=base.paths.PAPER_DATA_PYTHON or base.paths.ORBIT_ROOT.parent/'paper-data-venv/bin/python')
    result.add_argument('--ledger',type=Path,default=base.paths.RESULTS_DIR/'llm_calls.jsonl')
    result.add_argument('--allowance',type=Path,default=base.paths.RESULTS_DIR/'autonomous-api-allowance.json')
    result.add_argument('--metadata-template',type=Path,default=base.paths.RESULTS_DIR/'E5d-agent-sessions-20261106/bert/acceptance/pars/export/Bert_PTAgent_DSL/config/compile_argv.json')
    result.add_argument('--profile',type=Path,default=base.paths.PROFILES_DIR/'qwen48/compiler-config.json')
    result.add_argument('--frontend',type=Path,default=base.paths.DACAPO_ROOT/'python/hecate/hecate/expr.py')
    result.add_argument('--overall-seconds',type=int,default=OVERALL_SECONDS)
    return result


def main(argv=None):
    cli=parser();args=cli.parse_args(argv)
    for name,value in vars(args).items():
        if isinstance(value,Path):setattr(args,name,value.expanduser().absolute() if name in ('intake_python','data_python') else value.expanduser().resolve())
    try:validate(args)
    except ValueError as exc:cli.error(str(exc))
    result=execute(args)
    print(json.dumps({key:result.get(key) for key in ('status','outcome','eligible','frozen','report_completed','compile_completed','error')}),flush=True)
    return 0 if result['status'] in ('ok','quality_rejected') else 1


if __name__=='__main__':raise SystemExit(main())
