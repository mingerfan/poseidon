#!/usr/bin/env python3
"""Analyze all GPU activity, retaining idle gaps and separating host API time."""
from collections import defaultdict
import csv
import json
from pathlib import Path
import sqlite3
import statistics
import sys

ROOT=Path(__file__).resolve().parent
label=sys.argv[1] if len(sys.argv)>1 else 'nsys'
db=sqlite3.connect(ROOT/label/'profile.sqlite')
strings=dict(db.execute('select id,value from StringIds'))
window=db.execute('select duration from ANALYSIS_DETAILS').fetchone()[0]

def distribution(values):
    a=sorted(values)
    return dict(count=len(a),total_ns=sum(a),mean_ns=statistics.mean(a) if a else 0,
        p50_ns=a[len(a)//2] if a else 0,p95_ns=a[min(len(a)-1,int(.95*len(a)))] if a else 0,
        p99_ns=a[min(len(a)-1,int(.99*len(a)))] if a else 0,max_ns=a[-1] if a else 0)

def merged(intervals):
    result=[]
    for start,end in sorted(intervals):
        start=max(0,start);end=min(window,end)
        if end<=start:continue
        if result and start<=result[-1][1]:result[-1][1]=max(end,result[-1][1])
        else:result.append([start,end])
    return result

def coverage(intervals):return sum(b-a for a,b in merged(intervals))

def overlap(left,right):
    i=j=total=0
    while i<len(left) and j<len(right):
        a,b=left[i];c,d=right[j]
        total+=max(0,min(b,d)-max(a,c))
        if b<d:i+=1
        else:j+=1
    return total

host=[]
if db.execute("select 1 from sqlite_master where name='NVTX_EVENTS'").fetchone():
    for a,b,text,text_id in db.execute('select start,end,text,textId from NVTX_EVENTS'):
        if (text or strings.get(text_id))=='HostRefresh':host.append((a,b if b is not None else window))
host=merged(host)

apis={};api_by_name=defaultdict(list);api_by_thread=defaultdict(lambda:defaultdict(list))
for a,b,tid,corr,name in db.execute('select start,end,globalTid,correlationId,nameId from CUPTI_ACTIVITY_KIND_RUNTIME'):
    apis[corr]=(a,b,strings[name],tid)
    api_by_name[strings[name]].append(b-a)
    api_by_thread[tid][strings[name]].append(b-a)
activities=defaultdict(list);kernels=defaultdict(list);copies=defaultdict(list)
kernel_names=defaultdict(list);kernel_bytes=defaultdict(int)
queue=[]
for a,b,d,stream,corr,name in db.execute('select start,end,deviceId,streamId,correlationId,shortName from CUPTI_ACTIVITY_KIND_KERNEL'):
    activities[d].append((a,b,corr,'kernel'));kernels[d].append((a,b))
    kernel_names[strings[name]].append(b-a)
    if corr in apis:queue.append(a-apis[corr][1])
copy_kinds=dict(db.execute('select id,label from ENUM_CUDA_MEMCPY_OPER'))
copy_stats=defaultdict(lambda:dict(count=0,bytes=0,duration_ns=0))
for a,b,d,corr,size,kind in db.execute('select start,end,deviceId,correlationId,bytes,copyKind from CUPTI_ACTIVITY_KIND_MEMCPY'):
    activities[d].append((a,b,corr,'copy'));copies[d].append((a,b))
    s=copy_stats[copy_kinds[kind]];s['count']+=1;s['bytes']+=size;s['duration_ns']+=b-a
for a,b,d,corr in db.execute('select start,end,deviceId,correlationId from CUPTI_ACTIVITY_KIND_MEMSET'):
    activities[d].append((a,b,corr,'memset'))

devices=[];bins=[];gap_records=[]
for d,records in sorted(activities.items()):
    ordered=sorted(records);tail=0;gaps=[];parts=defaultdict(int)
    for start,end,corr,kind in ordered:
        if start>tail:
            lo=max(0,tail);hi=min(window,start)
            if hi>lo:
                gap=hi-lo;gaps.append(gap)
                if corr in apis:
                    a,b,name,tid=apis[corr]
                    before=max(0,min(hi,a)-lo)
                    inside=max(0,min(hi,b)-max(lo,a))
                    after=max(0,hi-max(lo,b))
                    assert before+inside+after==gap
                    parts['before_corresponding_api_entry_ns']+=before
                    parts['inside_corresponding_api_ns']+=inside
                    parts['after_corresponding_api_return_ns']+=after
                else:
                    name='unmatched';tid=None
                    parts['unmatched_ns']+=gap
                if gap>=1e6:gap_records.append(dict(device=d,start_ns=lo,end_ns=hi,duration_ns=gap,next_kind=kind,next_api=name,tid=tid))
        tail=max(tail,end)
    if tail<window:parts['unmatched_ns']+=window-max(0,tail)
    all_busy=coverage([(a,b) for a,b,_,_ in records])
    devices.append(dict(device=d,kernel_duration=distribution([b-a for a,b in kernels[d]]),
        kernel_union_ns=coverage(kernels[d]),copy_union_ns=coverage(copies[d]),
        any_gpu_activity_union_ns=all_busy,any_gpu_activity_percent=100*all_busy/window,
        kernel_active_percent=100*coverage(kernels[d])/window,
        idle_ns=window-all_busy,gap_parts=parts,gap_distribution=distribution(gaps),
        idle_during_host_refresh_ns=coverage(host)-overlap(merged([(a,b) for a,b,_,_ in records]),host),
        gaps_over_100us_total_ns=sum(g for g in gaps if g>1e5),
        gaps_over_1ms_total_ns=sum(g for g in gaps if g>1e6)))
    assert abs(sum(parts.values())-(window-all_busy))<1
    kernel_intervals=merged(kernels[d])
    activity_intervals=merged([(a,b) for a,b,_,_ in records])
    for second in range((window+999999999)//1000000000):
        lo,hi=second*1000000000,min(window,(second+1)*1000000000)
        def within(items):return sum(max(0,min(hi,b)-max(lo,a)) for a,b in items)
        bins.append(dict(device=d,second=second,kernel_fraction=within(kernel_intervals)/(hi-lo),
            any_activity_fraction=within(activity_intervals)/(hi-lo),host_refresh_fraction=within(host)/(hi-lo)))

chains=defaultdict(list)
for chain,symbol,module,depth in db.execute('select id,symbol,module,stackDepth from OSRT_CALLCHAINS order by id,stackDepth'):
    chains[chain].append(dict(symbol=strings.get(symbol,str(symbol)),module=strings.get(module,str(module))))
osrt=defaultdict(list);locks=defaultdict(lambda:dict(count=0,duration_ns=0))
osrt_app=defaultdict(lambda:dict(count=0,duration_ns=0))
for a,b,name,chain,tid in db.execute('select start,end,nameId,callchainId,globalTid from OSRT_API'):
    name=strings[name];osrt[name].append(b-a)
    app=next((f['symbol'] for f in chains[chain] if f['symbol'].startswith(('poseidon::','fhegpu::'))),'unresolved/system')
    s=osrt_app[(name,app)];s['count']+=1;s['duration_ns']+=b-a
    if name in ('pthread_mutex_lock','pthread_rwlock_wrlock','pthread_rwlock_rdlock'):
        s=locks[(name,chain,tid)];s['count']+=1;s['duration_ns']+=b-a
lock_rows=[]
for (name,chain,tid),s in sorted(locks.items(),key=lambda x:x[1]['duration_ns'],reverse=True)[:20]:
    frames=chains[chain]
    lock_rows.append(dict(api=name,tid=tid,**s,frames=frames))
result=dict(window_ns=window,devices=devices,host_refresh_intervals=host,
    host_refresh_union_ns=coverage(host),
    all_devices_idle_ns=window-coverage([(a,b) for records in activities.values() for a,b,_,_ in records]),
    api={k:distribution(v) for k,v in api_by_name.items()},
    top_kernel_names=sorted([dict(name=k,**distribution(v)) for k,v in kernel_names.items()],key=lambda v:v['total_ns'],reverse=True)[:30],
    copy_stats=copy_stats,osrt={k:distribution(v) for k,v in osrt.items()},top_lock_callchains=lock_rows,
    osrt_by_first_application_frame=[dict(api=name,frame=frame,**stats) for (name,frame),stats in
        sorted(osrt_app.items(),key=lambda x:x[1]['duration_ns'],reverse=True)],
    api_by_thread={str(tid):{k:distribution(v) for k,v in byname.items()} for tid,byname in api_by_thread.items()},
    launch_return_to_kernel_start_ns=distribution(queue),largest_idle_gaps=sorted(gap_records,key=lambda x:x['duration_ns'],reverse=True)[:50],
    notes=['CUDA runtime API times and OSRT times overlap and are summed across threads; do not add them to wall time.',
        'GPU interval unions include all streams. Kernel busy differs from NVML GPU busy. Copy union is not SM computation.',
        'Idle-before-API means the next recorded GPU activity was not submitted through its matched runtime API yet; may include dependency waits or host work.',
        '20-second capture is a diagnostic segment, not a replacement for the full 2-million-record speed comparison.'])
(ROOT/f'{label}-analysis.json').write_text(json.dumps(result,indent=2)+'\n')
with (ROOT/f'{label}-gpu-seconds.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(bins[0]));w.writeheader();w.writerows(bins)
print('window',window/1e9)
for d in devices:print('GPU',d['device'],'kernel %',d['kernel_active_percent'],'kernel/copy/memset %',d['any_gpu_activity_percent'],'gap parts',dict(d['gap_parts']))
print('top APIs',[(k,v['count'],v['total_ns']/1e9) for k,v in sorted(result['api'].items(),key=lambda x:x[1]['total_ns'],reverse=True)[:12]])
print('top locks',[(r['api'],r['duration_ns']/1e9,[f['symbol'] for f in r['frames'][:6]]) for r in lock_rows[:5]])
