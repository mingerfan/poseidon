"""One finite serial archive of exactly three inactive A2 raw compiler traces.

Compression reuses the original R34 primitive, including its durable rename
before raw removal. This job has its own clocks and records; original model,
RuntimePlan, bundle, constants, source, plans and scientific clocks stay intact.
The existing scheduler owns the heavy-job lease. This module acquires none.
"""
import argparse
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import stat
import time

from .. import paths
from .r10_paper_metrics import StageRecorder,read_events,summarize_events,write_json
from .r34_archive_text import archive_one,archive_state
from .completion_metrics import artifact_bytes,persist_case_completion

GIB=1024**3
ROUND='A-GROUP-CLOSED-A2-RAW-ARCHIVE'
CASES=('a2-resnet20-frozen-native-profile-four-placement-pars-20261117','a2-vgg16-frozen-native-profile-four-placement-pars-20261117','a2-alexnet-frozen-native-profile-four-placement-pars-20261117')
MODELS=('resnet20_silu','vgg16_silu_avgpool','alexnet_silu_avgpool')
QUEUE='a-group-a2-a3-a4-serial-20261117'
FIELDS={'run_id','round','model','mode','git_rev','metrics','artifacts_dir','ts'}


def read(path):return json.loads(Path(path).read_text())


def require(ok,message):
    if not ok:raise ValueError(message)


def durable(path,value):
    write_json(path,value)
    with Path(path).open('rb')as stream:os.fsync(stream.fileno())
    fd=os.open(str(Path(path).parent),os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def same_birth(pid,birth):
    try:fields=(Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()
    except FileNotFoundError:return False
    return int(fields[19])==birth


def saved_births(case):
    births=set()
    for path in Path(case).rglob('*resources.jsonl'):
        with path.open()as stream:
            for line in stream:
                if not line.strip():continue
                row=json.loads(line);items=list(row.get('processes')or[])
                if row.get('outer_process'):items.append(row['outer_process'])
                for item in items:
                    if type(item.get('pid'))is int and type(item.get('starttime'))is int:births.add((item['pid'],item['starttime']))
    require(births,'Saved producer birth metadata required')
    require(not any(same_birth(p,b)for p,b in births),'A saved producer process birth is still present')
    return len(births)


def registry(case,result):
    rows=[]
    with(paths.RESULTS_DIR/'registry.jsonl').open()as stream:
        for line in stream:
            if line.strip():
                row=json.loads(line)
                if row.get('run_id')==case.name:rows.append(row)
    require(len(rows)==1 and set(rows[0])==FIELDS and rows[0]['artifacts_dir']==str(case)and rows[0]['metrics']['status']==result['status'],'Unique matching original eight-field registry required')
    return rows[0]


def completed(case,allowed):
    result=read(case/'result.json');completion=read(case/'completion.json')
    require(result.get('status')in allowed and completion.get('status')==result['status'],'Durably closed original case required')
    require(completion.get('registry_persisted')is True and completion.get('required_artifacts_persisted')is True,'Original required persistence is incomplete')
    registry(case,result)
    return result,completion


def file_stat(path):
    path=Path(path);value=path.lstat()
    require(not path.is_symlink()and path.resolve()==path.absolute()and stat.S_ISREG(value.st_mode)and value.st_nlink==1,'Only original regular unlinked files eligible')
    return dict(path=str(path),bytes=value.st_size,inode=value.st_ino,device=value.st_dev,mtime_ns=value.st_mtime_ns)


def preservation(case,result):
    observed=next(row['worker']for row in result['phases']if row['phase']=='observe')
    plan=Path(observed['runtime_plan']);emit=case/'emit'
    require(plan.parent==emit and plan.name.endswith('.runtime-plan.json'),'Original actual RuntimePlan identity differs')
    bundle=plan.with_name(plan.name.removesuffix('.runtime-plan.json')+'.bundle')
    require(bundle.is_dir()and not bundle.is_symlink()and(bundle/'data').is_dir(),'Original compiled bundle required')
    selected=[case/name for name in('result.json','completion.json','events.jsonl','program.ptdsl','source.json','admission.json','compile-argv.json','placement.json')]
    selected+=[case/'plans'/(name+'.json')for name in('depth-dp','depth-dp+prune','stage-cut','stage-cut+prune')]
    selected+=[emit/'candidate_trace.py',emit/'constants.npz',emit/'lowered.mlir',plan,bundle/'manifest.json']
    return[file_stat(p)for p in selected]


def validate_manifest(manifest):
    root=paths.RESULTS_DIR.resolve()
    require(manifest.get('round')==ROUND and manifest.get('no_future_A2_raw_consumers')is True,'Explicit inactive-source authority required')
    require(manifest.get('host_pid_namespace')==os.readlink('/proc/self/ns/pid'),'Execute in the approved host process namespace')
    queue=root/QUEUE;q,qc=completed(queue,('ok','completed_with_observations'))
    supervisor=read(queue/'supervisor.json');require((supervisor['pid'],supervisor['process']['starttime'])==(231062,5241634),'Original queue producer identity differs')
    require(not same_birth(231062,5241634),'Original producer queue is still alive')
    wait=manifest['wait_closed_case'];waitcase=Path(wait['path'])
    require(waitcase.parent==root and waitcase.name=='a4-resnet20-123-prepared-disk-keyfree-as4-r31-20261117','Only the explicitly scheduled current A4 predecessor may be awaited')
    completed(waitcase,('ok','error'));wait_births=saved_births(waitcase)
    require((wait.get('owner_pid'),wait.get('owner_starttime'))==(236794,5280683)and not same_birth(236794,5280683),'Scheduled A4 predecessor owner must be closed')
    rows=manifest.get('files');require(isinstance(rows,list)and len(rows)==3,'Exactly three explicit A2 raw traces required')
    jobs=read(queue/'manifest.json')['jobs'];require(len(jobs)==5,'Original five-job producer family required')
    selected=[]
    for index,(name,model,row)in enumerate(zip(CASES,MODELS,rows)):
        case=root/name;source=case/'emit/candidate_trace.mlir'
        require(row==file_stat(source),'Exact approved original source stat/path changed')
        target=source.with_name(source.name+'.gz');temporary=target.with_name(target.name+'.partial')
        require(not os.path.lexists(target)and not os.path.lexists(temporary),'Archive output must be fresh')
        result,completion=completed(case,('ok',))
        require(result.get('round')=='A2-NATIVE-PROFILE'and result.get('model')==model and result.get('monitor_stopped')is True and result.get('error')is None,'Clean accepted A2 producer required')
        expected=dict.fromkeys(('planning','emit','trace','compile','observe'),1)
        require(result.get('attempted')==expected==result.get('completed'),'Every original source phase must have completed once')
        require(Path(jobs[index]['out'])==case,'Original producer job binding differs')
        require(not any(str(case)in arg for job in jobs[3:]for arg in job['argv']),'Original family still declares an A2 source consumer')
        births=saved_births(case);preserved=preservation(case,result)
        selected.append(dict(source=source,target=target,temporary=temporary,stat=row,preserved=preserved,births=births,scientific_status=result['status'],compiled_bootstraps=result['compiled_bootstraps'],scientific_compile_s=result['compile_s']))
    return selected,dict(queue_status=q['status'],A4_predecessor_saved_births_closed=wait_births,host_pid_namespace=manifest['host_pid_namespace'])


def append_registry(out,result):
    row=dict(run_id=out.name,round=ROUND,model='three-closed-A2-raw-traces',mode='reversible-gzip-no-readback',git_rev=None,metrics={key:result[key]for key in('status','completed_files','released_bytes')},artifacts_dir=str(out),ts=result['ended_utc'])
    with(paths.RESULTS_DIR/'registry.jsonl').open('a+')as stream:
        fcntl.flock(stream,fcntl.LOCK_EX);stream.seek(0)
        require(not any(json.loads(line).get('run_id')==out.name for line in stream if line.strip()),'Preserve previous archive identity')
        stream.seek(0,os.SEEK_END);stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())


def run(args):
    require(args.run,'Explicit archive --run required');require(1<=args.seconds<=1200,'Finite archive wall allowance1..1200 required')
    out=args.out.absolute();require(out.parent==paths.RESULTS_DIR.resolve()and not out.exists(),'Fresh direct result child required')
    out.mkdir();started=time.perf_counter_ns();events=StageRecorder(out/'events.jsonl')
    result=dict(round=ROUND,status='running',error=None,files=[],in_progress=None,completed_files=0,released_bytes=0,api_calls=0,model_Count_compile_CKKS=0,integrity_checks=0,scientific_clocks_modified=False,heavy_lease_acquired=False,archive_clock_scope='Independent resource-reclamation invocation; includes metadata admission and compression. Pre-persistence elapsed excludes final result and registry persistence; completion includes required persistence. Original scientific clocks remain unchanged.',limits=dict(seconds=args.seconds,AS_gib=1,minimum_free_gib=10,compression_concurrency=1))
    old={sig:signal.getsignal(sig)for sig in(signal.SIGTERM,signal.SIGINT,signal.SIGALRM)}
    def interrupt(signum,frame):result['incoming_signal']=signum;raise InterruptedError('Archive interrupted by signal'+str(signum))
    for sig in old:signal.signal(sig,interrupt)
    try:
        with events.span('archive_three_closed_A2_raw_traces'):
            hard=resource.getrlimit(resource.RLIMIT_AS)[1];limit=GIB if hard==resource.RLIM_INFINITY else min(GIB,hard);resource.setrlimit(resource.RLIMIT_AS,(limit,limit));signal.setitimer(signal.ITIMER_REAL,args.seconds)
            result['AS_limit_applied_bytes']=limit
            manifest=read(args.manifest);durable(out/'manifest.json',manifest)
            with events.span('closed_metadata_admission'):selected,result['admission']=validate_manifest(manifest)
            for row in selected:
                source,target,temporary=row['source'],row['target'],row['temporary']
                require(file_stat(source)==row['stat'],'Raw source changed since admission')
                require(shutil.disk_usage(source.parent).free>=10*GIB+row['stat']['bytes']*101//100+1024**2,'Free disk must cover10GiB plus worst temporary gzip size')
                require(all(file_stat(Path(item['path']))==item for item in row['preserved']),'Required scientific artifacts changed before compression')
                result['in_progress']=archive_state(source,target,temporary);durable(out/'progress.json',result)
                with events.span('gzip_one_raw_trace',source=str(source)):
                    observed=archive_one(source,target,temporary)
                require(all(file_stat(Path(item['path']))==item for item in row['preserved']),'Required scientific artifact stat changed during archive')
                require(observed['status']in('compressed','kept_uncompressed'),'Original archive primitive did not complete')
                result['files'].append(dict(observed,producer_births_closed=row['births'],preserved_artifacts=len(row['preserved']),original_compiled_bootstraps=row['compiled_bootstraps'],original_scientific_compile_s=row['scientific_compile_s']))
                result['completed_files']=len(result['files']);result['released_bytes']=sum(item['released_bytes']for item in result['files']);result['in_progress']=None;durable(out/'progress.json',result)
            result['status']='ok'
    except BaseException as exc:
        result.update(status='error',error=type(exc).__name__+': '+str(exc))
        if result['in_progress']:result['interrupted_state']=archive_state(source,target,temporary)
    finally:
        for sig in old:signal.signal(sig,signal.SIG_IGN)
        signal.setitimer(signal.ITIMER_REAL,0)
        result['ended_utc']=datetime.now(timezone.utc).isoformat();result['timing']=summarize_events(read_events(out/'events.jsonl'),time.perf_counter_ns())
        result['archive_only_elapsed_s']=(time.perf_counter_ns()-started)/1e9;result['max_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        durable(out/'result.json',result);append_registry(out,result)
        marker=persist_case_completion(out,round_label=ROUND,started_ns=started,status=result['status'],timing=result['timing'],required_artifact_bytes=artifact_bytes(out));durable(out/'completion.json',marker)
        for sig,handler in old.items():signal.signal(sig,handler)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--seconds',type=int,default=1200);parser.add_argument('--run',action='store_true');args=parser.parse_args(argv)
    result=run(args);print(json.dumps({key:result[key]for key in('status','error','completed_files','released_bytes')}));return 0 if result['status']=='ok' else 1


if __name__=='__main__':raise SystemExit(main())
