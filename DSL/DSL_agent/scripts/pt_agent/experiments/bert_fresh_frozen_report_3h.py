"""One isolated872-row report after a new accepted fresh three-hour BERT run.

Only this report shell gains report access. The original model/session/compiler
case and its three-hour record remain read-only; this has its own1800second clock.
"""
import argparse
import ast
import copy
import inspect
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import textwrap
import types

from .. import paths
from . import bert_fresh_cold_3h as fresh
from . import bert_frozen_fork_report as base
from . import bert_compile_first_deferred_report as deferred
from . import bert_serial_continuation as serial
from .archive_producer_closure import closed_case
from .archive_producer_closure import _process, registry_rows

ROUND='BERT-FRESH-3H-FROZEN-REPORT'
MODEL=fresh.MODEL
MODE='isolated-new-fresh-three-hour-accepted-native-validation872-once'
MODULE='pt_agent.experiments.bert_fresh_frozen_report_3h'


def _option(argv,flag):
    if argv.count(flag)!=1 or argv.index(flag)+1==len(argv):
        raise ValueError('One saved original command option required: '+flag)
    return argv[argv.index(flag)+1]


def queue_registry_row(owner, run):
    """Bind the actual queue schema, which has no top-level model or metrics."""
    row = registry_rows(owner.parent/'registry.jsonl', {owner.name:owner})[owner.name]
    if (row.get('round') != run.get('round') or row.get('mode') != run.get('mode')
            or row.get('model') != 'original-model-experiment-queue'
            or row.get('metrics') != dict(status=run.get('status'), jobs=run.get('jobs'))):
        raise ValueError('The actual closed queue registry and original jobs differ')
    from .bert_three_hour_comparison_queue import namespace as queue_namespace
    queue_namespace()['validate'](base.read(owner/'manifest.json'), paths.RESULTS_DIR)
    return row


def producer_closure(case,owner,namespace,*,process_reader=_process):
    """Prove the original serial job and every saved numeric/compiler birth gone."""
    if namespace!=os.readlink('/proc/self/ns/pid'):
        raise ValueError('Inspect the actual approved host PID namespace')
    run,completion=closed_case(owner,('ok','error','completed_with_observations'))
    if run.get('round')!='AUTONOMOUS-SERIAL' or run.get('mode')!='finite-trusted-serial-experiments':
        raise ValueError('Original owning serial controller required')
    queue_registry_row(owner,run)
    jobs=run.get('jobs')or[]
    if any(row.get('cleanup_unknown') or row.get('signal') or(row.get('process')or{}).get('remaining_pids')for row in jobs):
        raise ValueError('Finish every earlier serial job and its descendant cleanup first')
    matched=[row for row in jobs if row.get('out')==str(case)]
    if len(matched)!=1:raise ValueError('One actual owning fresh BERT job is required')
    job=matched[0];observation=job.get('process')or{}
    if job.get('status')!='ok' or job.get('experiment_status')!='ok' or not fresh._clean(observation):
        raise ValueError('Fresh BERT job must have clean accepted actual process closure')
    argv=job.get('argv')or[]
    if _option(argv,'-m')!=fresh.MODULE or Path(_option(argv,'--out')).resolve()!=case:
        raise ValueError('Owning command must invoke this exact fresh source')
    declared=[row for row in base.read(owner/'manifest.json').get('jobs',[])if row.get('out')==str(case)]
    if len(declared)!=1 or declared[0].get('id')!=job.get('id') or declared[0].get('argv')!=argv:
        raise ValueError('Original manifest and actual fresh source command differ')
    if base.read(owner/(job['id']+'-command.json'))!=argv:
        raise ValueError('Saved original fresh command differs')
    supervisor=base.read(owner/'supervisor.json');controller=supervisor.get('process')or{}
    if controller.get('pid')!=supervisor.get('pid') or type(controller.get('starttime'))is not int:
        raise ValueError('The owning controller birth is required')
    births={(controller['pid'],controller['starttime'])};snapshots=0
    primary=owner/(job['id']+'-resources.jsonl')
    resources=[primary]+sorted(case.rglob('*resources.jsonl'))
    for path in resources:
        if not path.is_file()or path.is_symlink():raise ValueError('Original regular resource metadata required')
        with path.open()as stream:
            for line in stream:
                if not line.strip():continue
                sample=json.loads(line);outer=sample.get('outer_process')or{}
                rows=sample.get('processes')
                if not isinstance(rows,list):raise ValueError('Original process snapshots are required')
                if path==primary:
                    if (outer.get('pid'),outer.get('starttime'))!=(controller['pid'],controller['starttime']):
                        raise ValueError('The original job resource owner birth differs')
                    if not observation['started_ns']<=sample['ts_ns']<=observation['ended_ns']:
                        raise ValueError('Original fresh job resource snapshot is outside its interval')
                    snapshots+=1
                for row in [outer,*rows]:
                    if type(row.get('pid'))is not int or type(row.get('starttime'))is not int:
                        raise ValueError('Explicit saved process PID/birth required')
                    births.add((row['pid'],row['starttime']))
    if snapshots<1 or len(births)<2:raise ValueError('Original fresh worker and controller samples are required')
    closed=[]
    for pid,birth in sorted(births):
        current=process_reader(pid)
        if current is not None and current.get('starttime')==birth:
            raise ValueError('An original BERT/queue producer birth remains present')
        closed.append(dict(pid=pid,starttime=birth,pid_reused=current is not None))
    return dict(supervisor_case=str(owner),matching_job_id=job['id'],closed_births=closed,
                saved_births_checked=len(closed),resource_snapshots=snapshots,host_pid_namespace=namespace,
                all_serial_jobs_terminal=True,unrelated_serial_failures_preserved=True)


def admission(args):
    case=args.source_case;root=paths.RESULTS_DIR.resolve()
    if (any(path.parent!=root for path in(case,args.out,args.supervisor_case))
            or len({case,args.out,args.supervisor_case})!=3):
        raise ValueError('Use distinct canonical source/report/owning serial cases')
    result,registry=base.closed_case(case)
    counters=result.get('completed')or{}
    if (result.get('round')!=fresh.ROUND or result.get('mode')!=fresh.MODE or result.get('model')!=MODEL
            or result.get('compile_accepted')is not True or result.get('frozen')is not True
            or result.get('monitor_stopped')is not True or result.get('termination')is not None
            or any(counters.get(key)!=1 for key in('prepare','checkpoint','confirm','math','emit','trace','compile'))
            or type(counters.get('step'))is not int or not 1<=counters['step']<=25
            or result.get('attempted')!=counters
            or any(result.get(key)!=0 for key in('report_samples','keygen_runs','ckks_runs','downloads','installations','integrity_checks'))):
        raise ValueError('One fully accepted new cold prepare/C2/math/native pipeline is required')
    completion=base.read(case/'completion.json');record=base.read(case/'three-hour-record.json')
    if (record.get('round')!=fresh.ROUND or record.get('mode')!=fresh.MODE or record.get('status')!='ok'
            or record.get('compile_accepted')is not True or record.get('persisted_pipeline_within_three_hours')is not True
            or record.get('started_ns')!=completion.get('started_ns')
            or record.get('required_artifacts_and_registry_completed_ns')!=completion.get('ended_ns')
            or not completion['ended_ns']<=record['observed_ns']<=record['started_ns']+10800*10**9):
        raise ValueError('The authoritative actual fresh three-hour record must already accept this program')
    for row in result.get('processes',[]):
        if not fresh._clean(row.get('observation')or{}):raise ValueError('Every original fresh phase must have clean closure')
    phases=result.get('phases')or{}
    proof=case/'math/math-result.json';math=base.read(proof)
    if math!=phases.get('math'):raise ValueError('Original fresh mathematical proof differs from saved phase')
    checkpoint=Path(math['source']['checkpoint_audit']).resolve()
    expected=deferred.compile_source(case,checkpoint,proof)
    for phase in('emit','trace','compile'):
        worker=phases.get(phase)or{}
        if worker.get('status')!='ok' or worker.get('source')!=expected:
            raise ValueError('Fresh compiler phase differs from the same frozen source/math: '+phase)
    native=fresh.native_artifacts(phases['compile'])
    if native!=result.get('native_artifacts') or phases['compile'].get('runtime_plan')!=result.get('runtime_plan'):
        raise ValueError('Actual accepted native plan/bundle provenance changed')
    if (case/'session/REPORT_ACCESS.json').exists()or(case/'model/samples/report.npz').exists():
        raise ValueError('Original fresh source report access must remain unused')
    if args.metadata_template!=Path(result['source']['metadata_template']).resolve():
        raise ValueError('Keep the actual original compiler metadata template')
    sample=base.read(case/'model/samples/source.json')
    if args.data_python.resolve()!=Path(sample['reader_python']).resolve():
        raise ValueError('Keep the original independent SST2 reader')
    prepared=[row for row in result['processes']if row['phase']=='prepare']
    if len(prepared)!=1:raise ValueError('One original fresh prepare command required')
    for flag,path in(('--python',args.intake_python),('--ledger',args.ledger),('--allowance',args.allowance)):
        if Path(_option(prepared[0]['argv'],flag)).resolve()!=path.resolve():
            raise ValueError('Original report runtime/accounting path differs: '+flag)
    closure=producer_closure(case,args.supervisor_case,args.host_pid_namespace)
    return dict(source_case=str(case),compile_case=str(case),model=expected['original_model'],
        inputs=sample['source'],model_info=expected['model_info'],program=expected['program'],
        frozen=expected['frozen'],task_policy=expected['task_policy'],split_version=expected['split_version'],
        report_samples=872,source=expected,compiler_registry=registry,native_artifacts=native,
        original_report_access_present=False,report_feedback_permitted=False,algorithm_changed=False,
        fresh_three_hour_record=str(case/'three-hour-record.json'),fresh_three_hour_accepted=True,
        source_producer_closure=closure,report_clock_scope='Independent post-compile872 report; excluded from original10800second pipeline')


def _safe_execute_tree():
    """Change only the report supervisor's private finalization ordering."""
    tree=ast.parse(textwrap.dedent(inspect.getsource(base.execute)));function=tree.body[0]
    outer=[node for node in function.body if isinstance(node,ast.Try)]
    if len(outer)!=1:raise ValueError('Original isolated report terminal structure changed')
    final=outer[0].finalbody
    terminal=[node for node in final if isinstance(node,ast.Try)]
    if len(terminal)!=1:raise ValueError('Original isolated report persistence structure changed')
    persist=terminal[0]
    before=ast.parse('''
for signum in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM):
    if signum not in handlers:handlers[signum]=signal.getsignal(signum)
    signal.signal(signum,signal.SIG_IGN)
signal.setitimer(signal.ITIMER_REAL,0)
stop.set()
if watcher is not None:watcher.join(timeout=3)
monitor_stopped=watcher is None or not watcher.is_alive()
result['monitor_stopped']=monitor_stopped
if not monitor_stopped:result.update(status='error',report_accepted=False,error='Report monitor remains active')
persistence_closed=False
''').body
    persist.body+=ast.parse('persistence_closed=True').body
    persist.finalbody=ast.parse('''
if monitor_stopped and persistence_closed:
    for signum,handler in handlers.items():signal.signal(signum,handler)
''').body
    outer[0].finalbody=[*before,persist]
    ast.fix_missing_locations(tree);return tree


def namespace():
    original=vars(base);values=dict(original,ROUND=ROUND,MODE=MODE,MODULE=MODULE,__name__=__name__)
    for name,value in original.items():
        if isinstance(value,types.FunctionType)and value.__globals__ is original:
            clone=types.FunctionType(value.__code__,values,name,value.__defaults__,value.__closure__)
            clone.__kwdefaults__=copy.copy(value.__kwdefaults__);values[name]=clone
    values['admission']=admission
    original_shell=values['prepare_shell']
    def shell(args,source):
        proof=original_shell(args,source)
        for module,name in((base,'original-isolated-report-source.py'),(fresh,'accepted-fresh-driver-source.py'),
                           (deferred,'original-deferred-math-source.py')):
            shutil.copyfile(module.__file__,args.out/name)
        return proof
    values['prepare_shell']=shell
    exec(compile(_safe_execute_tree(),str(Path(__file__)),'exec'),values)
    return values


def parser():
    cli=argparse.ArgumentParser(description=__doc__)
    for name in('source-case','supervisor-case','out','metadata-template'):
        cli.add_argument('--'+name,type=Path,required=True)
    cli.add_argument('--host-pid-namespace',required=True)
    cli.add_argument('--intake-python',type=Path,default=paths.INTAKE_PYTHON)
    cli.add_argument('--data-python',type=Path,default=paths.PAPER_DATA_PYTHON or paths.ORBIT_ROOT.parent/'paper-data-venv/bin/python')
    cli.add_argument('--ledger',type=Path,default=paths.RESULTS_DIR/'llm_calls.jsonl')
    cli.add_argument('--allowance',type=Path,default=paths.RESULTS_DIR/'autonomous-api-allowance.json')
    return cli


def main(argv=None):
    args=parser().parse_args(argv)
    for name,value in vars(args).items():
        if isinstance(value,Path):setattr(args,name,value.expanduser().absolute()if name.endswith('python')else value.expanduser().resolve())
    args.compile_case=args.source_case
    result=namespace()['execute'](args)
    print(json.dumps({key:result.get(key)for key in('status','report_accepted','report_samples','error')}),flush=True)
    return 0 if result['status']=='ok'else 1


if __name__=='__main__':raise SystemExit(main())
