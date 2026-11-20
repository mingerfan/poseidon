"""Admit three closed ResNet RuntimePlans and reuse the existing Fig7 exporter.

Only JSON compiler metadata is read. Source labels stay unmapped where the
actual RuntimePlan has no DSL locations. No bundle or raw MLIR is opened.
"""
import argparse
from collections import Counter
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import time

from .. import paths
from .a_group_artifacts import export_plan
from .a_group_future_closed_archive import same_birth
from .completion_metrics import artifact_bytes,persist_case_completion
from .native_component_matrix import array_items
from .r10_paper_metrics import StageRecorder,read_events,summarize_events,write_json

ROUND='P1-D-RESNET-COMPILED-ARTIFACTS'
FIELDS={'run_id','round','model','mode','git_rev','metrics','artifacts_dir','ts'}
OURS=dict(bootstrapLevelLowerBound=1,bootstrapLevelUpperBound=5,levelLowerBound=1,
    levelUpperBound=8,polynomialDegree=65536,rescalingFactor=43,runtime='HEAAN-HEVM')
NATIVE=dict(bootstrapLevelLowerBound=3,bootstrapLevelUpperBound=16,levelLowerBound=1,
    levelUpperBound=29,polynomialDegree=131072,rescalingFactor=51,runtime='HEAAN-HEVM')
SOURCES=(
    dict(name='fixed',case='local-compile-depth-dp-no-debug-r29-20261116/dacapo-depth-dp',
        run_id='local-compile-depth-dp-no-debug-r29-20261116-dacapo-depth-dp',
        plan='work/emit/lowered._hecate_golden.runtime-plan.json',profile='work/compiler-profile.json',
        model='pt_agent.experiments.r13_common_models:resnet20_silu',system='fixed_rule_depth',
        param_group='ours-N16-boot5',boots=230,ops=20490,parameters=OURS),
    dict(name='arranger',case='fresh-resnet20-agent-e2e-r30-authorized-20261116',
        run_id='fresh-resnet20-agent-e2e-r30-authorized-20261116',
        plan='post-freeze/emit/lowered._hecate_golden.runtime-plan.json',profile='source/compiler-profile.json',
        model='resnet20_silu',system='arranger',param_group='ours-N16-boot5',
        boots=123,ops=17863,parameters=OURS),
    dict(name='dacapo_native',case='dacapo-original-resnet20-native-profile-20261116',
        run_id='dacapo-original-resnet20-native-profile-20261116',
        plan='work/compile/lowered._hecate_ResNet.runtime-plan.json',profile='compiler-config.json',
        model='dacapo-original-resnet20-silu',system='dacapo_native',param_group='dacapo-native',
        boots=19,ops=12661,parameters=NATIVE),
)


def require(ok,message):
    if not ok:raise ValueError(message)


def read(path):return json.loads(Path(path).read_text())


def clean(process):
    return (process.get('exit_code')==0 and process.get('timed_out')is False
        and process.get('signal')is None and process.get('remaining_pids')==[]
        and process.get('signals_sent')==[])


def validate_record(source,record,profile):
    require(record.get('status')=='ok'and record.get('model')==source['model'],
            'Require the exact successful original ResNet compiler source')
    require({k:profile.get(k)for k in source['parameters']}==source['parameters'],
            'Preserve the saved source-specific compiler parameter group')
    case=paths.RESULTS_DIR/source['case'];wanted=case/source['plan']
    if source['name']=='dacapo_native':
        plan=Path(record['plan']['path']);ops=record.get('execution_instructions');boots=record.get('bootstrap_count')
        worker=record.get('compile_worker')or{}
        require(record.get('source_checked')is True and record.get('original_plan_modified')is False
            and worker.get('status')=='ok'and clean(worker.get('process')or{})
            and clean(record.get('compile_process')or{}),'Original native compiler did not close cleanly')
    else:
        compiled=record['phases']['compile'];plan=Path(compiled['runtime_plan'])
        ops=compiled.get('execution_steps');boots=compiled.get('compiled_bootstraps')
        require(compiled.get('status')=='ok'and clean(compiled.get('process')or{})
            and record.get('compiled_bootstraps')==boots,'Original controlled compiler did not close cleanly')
    require(plan.resolve()==wanted.resolve()and ops==source['ops']and boots==source['boots'],
            'Original RuntimePlan path or recorded actual counts differ')
    return wanted


def registry_record(source,case):
    rows=[]
    with(paths.RESULTS_DIR/'registry.jsonl').open()as stream:
        fcntl.flock(stream,fcntl.LOCK_SH)
        for line in stream:
            if line.strip():
                row=json.loads(line)
                if row.get('run_id')==source['run_id']:rows.append(row)
    require(len(rows)==1 and set(rows[0])==FIELDS
        and Path(rows[0].get('artifacts_dir','')).resolve()==case.resolve()
        and rows[0].get('metrics',{}).get('status')=='ok',
        'Unique eight-field successful original registry required')
    return rows[0]


def saved_births(case):
    births=set()
    for path in case.rglob('*resources.jsonl'):
        with path.open()as stream:
            for line in stream:
                if not line.strip():continue
                row=json.loads(line);items=list(row.get('processes')or[])
                if row.get('outer_process'):items.append(row['outer_process'])
                for item in items:
                    if type(item.get('pid'))is int and type(item.get('starttime'))is int:
                        births.add((item['pid'],item['starttime']))
    require(births,'Original saved producer birth metadata required')
    return sorted(births)


def plan_metadata(plan):
    levels={row['id']:row['level']for row in array_items(plan,'values')}
    counts=Counter();total=0;boot_levels=Counter()
    for op in array_items(plan,'execution'):
        counts[op['op']]+=1;total+=1
        if op['op']=='boot':
            boot_levels[(tuple(levels[name]for name in op['inputs']),levels[op['output']])]+=1
    return dict(exec_ops=total,boots=counts['boot'],ops_by_type=dict(counts),
        value_count=len(levels),value_levels=sorted(set(levels.values())),
        bootstrap_level_pairs=[dict(input_levels=list(a),output_level=b,count=n)
            for(a,b),n in sorted(boot_levels.items())],external_payloads_read=False)


def admit_source(source,destination,*,check_births):
    case=paths.RESULTS_DIR/source['case'];record=read(case/'result.json');completion=read(case/'completion.json')
    require(completion.get('status')=='ok'and completion.get('required_artifacts_persisted')is True
        and completion.get('registry_persisted')is True,'Original result/completion must be durably closed')
    registry=registry_record(source,case)
    plan=validate_record(source,record,read(case/source['profile']))
    observed=plan_metadata(plan)
    require(observed['boots']==source['boots']and observed['exec_ops']==source['ops'],
        'Full actual RuntimePlan metadata counts differ')
    births=saved_births(case)
    if check_births:require(not any(same_birth(pid,birth)for pid,birth in births),
        'A saved original producer birth remains active in the host namespace')
    target=Path(destination)/('resnet20_'+source['name']+'.json')
    require(not os.path.lexists(target)and not os.path.lexists(target.with_name(target.name+'.new')),
        'Preserve earlier artifact publication or partial')
    argv=['--plan',str(plan),'--out',str(target),'--model','resnet20',
        '--system',source['system'],'--param-group',source['param_group'],
        '--expected-boots',str(source['boots']),'--expected-ops',str(source['ops'])]
    return dict(name=source['name'],system=source['system'],param_group=source['param_group'],
        compiler_parameters=source['parameters'],case=str(case),result=str(case/'result.json'),
        completion=str(case/'completion.json'),registry_run_id=source['run_id'],registry_ts=registry['ts'],
        plan=str(plan),saved_profile=str(case/source['profile']),observed=observed,
        saved_producer_births=[dict(pid=pid,starttime=birth)for pid,birth in births],
        host_births_checked=check_births,host_births_pending=not check_births,
        target=str(target),exporter_cli_suffix=argv,source_mapping='unmapped fixed operation intervals; no inferred layer names',
        source_scope=(record.get('source')or{}).get('scope')if source['name']=='fixed'else None)


def run(args):
    require(args.publish,'Explicit metadata publication --publish required')
    out=args.out.absolute()
    require(out.parent==paths.RESULTS_DIR.resolve()and not out.exists(),'Fresh direct metadata result child required')
    out.mkdir();started=time.perf_counter_ns();rec=StageRecorder(out/'events.jsonl')
    result=dict(round=ROUND,status='running',error=None,source_table=[],exports=[],
        new_model_calls=0,new_Count_calls=0,new_native_calls=0,new_CKKS_calls=0,new_API_calls=0,
        bundle_reads=0,raw_MLIR_reads=0,integrity_checks=0,output_readback_checks=0)
    try:
        require(args.host_pid_namespace==os.readlink('/proc/self/ns/pid'),
            'Producer birth checks require the approved host process namespace')
        with rec.span('admit_three_closed_actual_RuntimePlans'):
            result['source_table']=[admit_source(source,args.destination,check_births=True)for source in SOURCES]
            write_json(out/'source-table.json',result['source_table'])
        for row in result['source_table']:
            with rec.span('export_actual_unmapped_intervals',system=row['system']):
                result['exports'].append(export_plan(row['plan'],row['target'],model='resnet20',
                    system=row['system'],param_group=row['param_group'],
                    expected_boots=row['observed']['boots'],expected_ops=row['observed']['exec_ops']))
        result['status']='ok'
    except BaseException as exc:result.update(status='error',error=type(exc).__name__+': '+str(exc))
    finally:
        result['timing']=summarize_events(read_events(out/'events.jsonl'));write_json(out/'result.json',result)
        entry=dict(run_id=out.name,round=ROUND,model='resnet20',mode='actual-compiler-metadata-export',git_rev=None,
            metrics=dict(status=result['status'],exported=len(result['exports']),API_calls=0),
            artifacts_dir=str(out),ts=datetime.now(timezone.utc).isoformat())
        with(paths.RESULTS_DIR/'registry.jsonl').open('a+')as stream:
            fcntl.flock(stream,fcntl.LOCK_EX);stream.seek(0)
            require(not any(json.loads(line).get('run_id')==out.name for line in stream if line.strip()),'Preserve previous metadata export identity')
            stream.seek(0,os.SEEK_END);stream.write(json.dumps(entry,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
        persist_case_completion(out,round_label=ROUND,started_ns=started,status=result['status'],
            timing=result['timing'],required_artifact_bytes=artifact_bytes(out))
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--destination',type=Path,default=paths.WORKSPACE/'arranger-dac/data/codex/artifacts')
    parser.add_argument('--host-pid-namespace',required=True)
    parser.add_argument('--publish',action='store_true')
    result=run(parser.parse_args(argv));print(json.dumps(dict(status=result['status'],error=result['error'],exports=result['exports'])))
    return 0 if result['status']=='ok'else 1


if __name__=='__main__':raise SystemExit(main())
