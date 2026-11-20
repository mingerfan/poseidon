"""Reversible resource-only archives for two exact future closed A-group cases.

The scheduler owns the heavy-job lease. This entry archives only the raw trace
and adjacent redundant NPZ after producer closure; scientific records and all
other files remain intact. No model, compiler, key or ciphertext is executed.
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
from .completion_metrics import artifact_bytes,persist_case_completion
from .r10_paper_metrics import StageRecorder,read_events,summarize_events,write_json
from .r34_archive_text import archive_one,archive_state

GIB=1024**3
ROUND='A-GROUP-FUTURE-CLOSED-RESOURCE-ARCHIVE'
A2='a2-alexnet-frozen-native-output-price-four-placement-pars-20261117'
A5='a5-vgg16-fixed63-optimized-current-rss14-20261117'
KINDS=('a2-alexnet-output-price','a5-vgg16-search')
REGISTRY_FIELDS={'run_id','round','model','mode','git_rev','metrics','artifacts_dir','ts'}


def require(condition,message):
    if not condition:raise ValueError(message)


def read(path):return json.loads(Path(path).read_text())


def durable(path,data):
    write_json(path,data)
    with Path(path).open('rb')as stream:os.fsync(stream.fileno())
    fd=os.open(str(Path(path).parent),os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def same_birth(pid,birth):
    try:fields=(Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()
    except FileNotFoundError:return False
    return int(fields[19])==birth


def closed(case,run_id):
    result=read(case/'result.json');completion=read(case/'completion.json')
    require(result.get('status')in('ok','error')and completion.get('status')==result['status'],
            'Source scientific result/completion must be closed ok or error')
    require(completion.get('registry_persisted')is True and completion.get('required_artifacts_persisted')is True,
            'Source required persistence must be complete')
    rows=[]
    with(paths.RESULTS_DIR/'registry.jsonl').open()as stream:
        fcntl.flock(stream,fcntl.LOCK_SH)
        for line in stream:
            if line.strip():
                row=json.loads(line)
                if row.get('run_id')==run_id:rows.append(row)
    require(len(rows)==1 and set(rows[0])==REGISTRY_FIELDS
        and Path(rows[0].get('artifacts_dir','')).resolve()==case.resolve()
        and rows[0].get('metrics',{}).get('status')==result['status'],
        'Unique matching source eight-field registry required')
    if result.get('run_id')is not None:require(result['run_id']==run_id,'Source result run identity differs')
    return dict(status=result['status'],result=str(case/'result.json'),completion=str(case/'completion.json'),
                run_id=run_id),result


def snapshot(path,*,archive_source=False):
    path=Path(path);s=path.lstat()
    if archive_source:
        require(not path.is_symlink()and path.resolve()==path.absolute()
            and stat.S_ISREG(s.st_mode)and s.st_nlink==1,
            'Only original regular unlinked raw trace/NPZ files eligible')
    row=dict(path=str(path),inode=s.st_ino,device=s.st_dev,mode=s.st_mode,links=s.st_nlink)
    if not stat.S_ISDIR(s.st_mode):row.update(bytes=s.st_size,mtime_ns=s.st_mtime_ns)
    if stat.S_ISLNK(s.st_mode):row['target']=os.readlink(path)
    return row


def producer_births(case):
    births=set()
    for p in case.rglob('*resources.jsonl'):
        with p.open()as stream:
            for line in stream:
                if not line.strip():continue
                data=json.loads(line);items=list(data.get('processes')or[])
                if data.get('outer_process'):items.append(data['outer_process'])
                for item in items:
                    if type(item.get('pid'))is int and type(item.get('starttime'))is int:
                        births.add((item['pid'],item['starttime']))
    require(not any(same_birth(pid,birth)for pid,birth in births),'A saved source producer birth is still present')
    return len(births)


def admit(args):
    require(args.no_future_consumers,'Explicit no-future-raw/NPZ-consumer authority required')
    require(args.host_pid_namespace==os.readlink('/proc/self/ns/pid'),
            'Archive must run in the approved host process namespace')
    root=paths.RESULTS_DIR.resolve()
    if args.case=='a2-alexnet-output-price':
        parent=root/A2;source=parent;emit=source/'emit';run_id=A2
    elif args.case=='a5-vgg16-search':
        parent=root/A5;source=parent/'dacapo-optimized';emit=source/'work/emit';run_id=A5+'-dacapo-optimized'
    else:raise ValueError('Only the two exact approved future source cases are eligible')
    parent_closed,parent_result=closed(parent,parent.name)
    if source==parent:source_closed,source_result=parent_closed,parent_result
    else:
        source_closed,source_result=closed(source,run_id)
        require(Path(parent_result.get('child_case','')).resolve()==source.resolve()
            and Path(parent_result.get('child_result','')).resolve()==(source/'result.json').resolve()
            and Path(parent_result.get('child_completion','')).resolve()==(source/'completion.json').resolve()
            and parent_result.get('child_status')==source_result['status'],
            'A5 parent summary must bind the original child result/completion')
    births=producer_births(parent)
    raw=emit/'candidate_trace.mlir';npz=emit/'constants.npz'
    raw_exists=os.path.lexists(raw)
    if raw_exists:require(births>0,'Existing raw trace requires saved producer birth metadata')
    candidates=[raw,npz]if raw_exists else []
    selected=[]
    for p in candidates:
        if not os.path.lexists(p):continue
        observation=snapshot(p,archive_source=True)
        target=p.with_name(p.name+'.gz');temporary=target.with_name(target.name+'.partial')
        require(not os.path.lexists(target)and not os.path.lexists(temporary),'Preserve every previous archive/partial/link')
        selected.append(dict(source=str(p),target=str(target),temporary=str(temporary),stat=observation))
    excluded={row['source']for row in selected}
    preserved=[snapshot(p)for p in parent.rglob('*')if str(p)not in excluded]
    return dict(parent=str(parent),source_case=str(source),parent_closure=parent_closed,
        source_closure=source_closed,scientific_status=source_result['status'],
        scientific_error=source_result.get('error'),producer_births_closed=births,
        host_pid_namespace=args.host_pid_namespace,no_future_consumers=True,
        skip_reason=None if raw_exists else 'Raw trace was not generated; raw/NPZ both skipped without changing scientific status',
        selected=selected,preserved=preserved,scope='Only two exact raw/NPZ pairs; all other source files and directory identities retained')


def preserve(rows):
    require(all(snapshot(Path(row['path']))==row for row in rows),
            'A retained scientific artifact changed during the resource-only job')


def append_registry(out,result):
    row=dict(run_id=out.name,round=ROUND,model=result['source_kind'],mode='resource-only-reversible-gzip',
        git_rev=None,metrics={key:result.get(key)for key in('status','resource_status','scientific_status','completed_files','released_bytes')},
        artifacts_dir=str(out),ts=datetime.now(timezone.utc).isoformat())
    with(paths.RESULTS_DIR/'registry.jsonl').open('a+')as stream:
        fcntl.flock(stream,fcntl.LOCK_EX);stream.seek(0)
        require(not any(json.loads(line).get('run_id')==out.name for line in stream if line.strip()),
                'Preserve previous resource archive identity')
        stream.seek(0,os.SEEK_END);stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())


def run(args):
    require(args.run,'Explicit resource archive --run required')
    require(1<=args.seconds<=1200,'Finite archive wall allowance1..1200s required')
    out=args.out.absolute()
    require(out.parent==paths.RESULTS_DIR.resolve()and not out.exists(),'Fresh direct resource-only results child required')
    out.mkdir();started=time.perf_counter_ns();rec=StageRecorder(out/'events.jsonl')
    result=dict(round=ROUND,source_kind=args.case,status='running',error=None,resource_status='pending',
        scientific_status=None,scientific_status_upgraded=False,files=[],completed_files=0,released_bytes=0,
        in_progress=None,api_calls=0,model_Count_compile_CKKS=0,integrity_checks=0,output_readback_checks=0,
        heavy_lease_acquired=False,limits=dict(address_space_gib=1,seconds=args.seconds,minimum_free_gib=10,concurrency=1),
        clock_scope='Independent resource reclamation only; original scientific clocks remain untouched')
    old={sig:signal.getsignal(sig)for sig in(signal.SIGTERM,signal.SIGINT,signal.SIGALRM)}
    def interrupted(sig,frame):result['incoming_signal']=sig;raise InterruptedError('Resource archive interrupted by signal '+str(sig))
    for sig in old:signal.signal(sig,interrupted)
    try:
        with rec.span('future_closed_source_resource_archive'):
            hard=resource.getrlimit(resource.RLIMIT_AS)[1]
            limit=GIB if hard==resource.RLIM_INFINITY else min(GIB,hard)
            resource.setrlimit(resource.RLIMIT_AS,(limit,limit));signal.setitimer(signal.ITIMER_REAL,args.seconds)
            with rec.span('closed_metadata_admission'):admission=admit(args)
            result['source_admission']=admission;result['scientific_status']=admission['scientific_status']
            durable(out/'admission.json',admission)
            if admission['skip_reason']:
                result.update(status='ok',resource_status='metadata_skip',skip_reason=admission['skip_reason'])
            else:
                for row in admission['selected']:
                    source=Path(row['source']);target=Path(row['target']);temporary=Path(row['temporary'])
                    require(snapshot(source,archive_source=True)==row['stat'],'Original raw/NPZ source stat changed since admission')
                    require(not os.path.lexists(target)and not os.path.lexists(temporary),'Archive destinations must remain fresh')
                    preserve(admission['preserved'])
                    required=10*GIB+row['stat']['bytes']*101//100+1024**2
                    require(shutil.disk_usage(source.parent).free>=required,'resource_limit: free10GiB plus101% raw/NPZ temporary allowance')
                    result['in_progress']=archive_state(source,target,temporary);durable(out/'progress.json',result)
                    with rec.span('gzip_one_redundant_file',source=str(source)):
                        compressed=archive_one(source,target,temporary)
                    require(compressed['status']in('compressed','kept_uncompressed'),'Original archive primitive did not complete')
                    preserve(admission['preserved'])
                    result['files'].append(compressed);result['completed_files']=len(result['files'])
                    result['released_bytes']=sum(item['released_bytes']for item in result['files'])
                    result['in_progress']=None;durable(out/'progress.json',result)
                result.update(status='ok',resource_status='archived_or_kept')
    except BaseException as exc:
        result.update(status='error',error=type(exc).__name__+': '+str(exc),resource_status='failed')
        if result['in_progress']:result['interrupted_state']=archive_state(source,target,temporary)
    finally:
        for sig in old:signal.signal(sig,signal.SIG_IGN)
        signal.setitimer(signal.ITIMER_REAL,0)
        result['timing']=summarize_events(read_events(out/'events.jsonl'),time.perf_counter_ns())
        result['resource_only_elapsed_s']=(time.perf_counter_ns()-started)/1e9
        durable(out/'result.json',result);append_registry(out,result)
        marker=persist_case_completion(out,round_label=ROUND,started_ns=started,status=result['status'],
            timing=result['timing'],required_artifact_bytes=artifact_bytes(out))
        durable(out/'completion.json',marker)
        for sig,handler in old.items():signal.signal(sig,handler)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=KINDS,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--host-pid-namespace',required=True)
    parser.add_argument('--seconds',type=int,default=1140)
    parser.add_argument('--no-future-consumers',action='store_true')
    parser.add_argument('--run',action='store_true')
    result=run(parser.parse_args(argv))
    print(json.dumps({key:result[key]for key in('status','resource_status','scientific_status','completed_files','released_bytes')}))
    return 0 if result['status']=='ok'else 1


if __name__=='__main__':raise SystemExit(main())
