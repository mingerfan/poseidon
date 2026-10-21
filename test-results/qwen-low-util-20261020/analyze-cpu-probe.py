#!/usr/bin/env python3
"""Summarize direct CPU API counters and passive process samples."""
from collections import Counter
import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def summarize(name):
    folder=ROOT/name
    threads=[json.loads(s) for s in (folder/'cuda-api.jsonl').read_text().splitlines()]
    begin,end=threads[0]['window_begin_ns'],threads[0]['window_end_ns']
    assert all(t['window_begin_ns']==begin and t['window_end_ns']==end for t in threads)
    apis={};launch_gaps=[]
    for t in threads:
        for r in t['rows']:
            s=apis.setdefault(r['api'],dict(count=0,total_ns=0,maximum_ns=0,errors=0,log2_ns=[0]*32))
            s['count']+=r['count'];s['total_ns']+=r['total_ns'];s['maximum_ns']=max(s['maximum_ns'],r['maximum_ns'])
            s['errors']+=r['errors_except_not_ready']
            s['log2_ns']=[a+b for a,b in zip(s['log2_ns'],r['log2_ns'])]
        launch_gaps.extend(t['runtime_launch_start_intervals'])
    launches=sum(v['count'] for k,v in apis.items() if k.startswith('cudaLaunchKernel'))
    assert launches and all(s['errors']==0 for s in apis.values())
    for s in apis.values():
        s['mean_us']=s['total_ns']/s['count']/1000
        s['calls_per_runtime_kernel_launch']=s['count']/launches
        s['cumulative_seconds']=s['total_ns']/1e9
    samples=[json.loads(s) for s in (folder/'samples.jsonl').read_text().splitlines()]
    selected=[s for s in samples if begin<=s['monotonic']*1e9<=end and s['process']]
    a,b=selected[0],selected[-1];duration=b['monotonic']-a['monotonic']
    wchan=Counter(t['wchan'] for s in selected for t in s['process']['threads'])
    ticks=[]
    first={t['tid']:t['utime_ticks']+t['stime_ticks'] for t in a['process']['threads']}
    last={t['tid']:t['utime_ticks']+t['stime_ticks'] for t in b['process']['threads']}
    ticks_per_second=json.loads((folder/'status.json').read_text())['proc_ticks_per_second']
    for tid,value in last.items():
        if tid in first:ticks.append(dict(tid=tid,cpu_seconds=(value-first[tid])/ticks_per_second))
    # NVML samples are coarse and this is only a segment diagnostic. It is not
    # substituted for the full-run time-weighted metric in the previous report.
    gpu=[s['nvml'] for s in selected if 'nvml' in s]
    boot=[]
    if (folder/'boot.rank0.csv').exists():
        for row in csv.DictReader((folder/'boot.rank0.csv').open()):
            finish=int(row['end_monotonic_ns']);start=finish-int(row['duration_ns'])
            overlap=max(0,min(finish,end)-max(start,begin))
            if overlap:boot.append(dict(ordinal=int(row['ordinal']),start_ns=start,end_ns=finish,overlap_ns=overlap))
    callers=folder/'symbolized-callers.json'
    result=dict(case=name,window_begin_ns=begin,window_end_ns=end,window_seconds=(end-begin)/1e9,
        counted_threads=len(threads),runtime_kernel_launch_calls=launches,apis=apis,
        runtime_launch_start_intervals=launch_gaps,
        cpu_core_seconds=sum(t['cpu_seconds'] for t in ticks),cpu_seconds_by_thread=ticks,
        observed_host_cpu_core_equivalents=sum(t['cpu_seconds'] for t in ticks)/duration,
        rchar_progress_mib_per_second=(b['process']['io']['rchar']-a['process']['io']['rchar'])/duration/2**20,
        sampled_wchan_counts=dict(wchan),
        sampled_segment_four_gpu_busy_percent=sum(d['gpu_util'] for s in gpu for d in s['gpu'])/(len(gpu)*4),
        boot_calls_overlapping_window=boot,boot_overlap_seconds=sum(b['overlap_ns'] for b in boot)/1e9,
        sampled_setdevice_callers=json.loads(callers.read_text()) if callers.exists() else None,
        notes=['No nsys, CUPTI, extra CUDA event, stream wait or synchronization is used by this interposer.',
            'API duration is CPU wall time of the real function, not GPU kernel time or exclusive CPU execution.',
            'Calls whose entry falls within the window are counted with full return duration; a call may cross the end.',
            'Counts cover named intercepted symbols, including PTDS runtime entry points. Driver calls resolved through dispatch tables may bypass interposition.',
            'cudaSetDevice count is not a count of actual device/context changes.',
            'Host window read progress is a proxy, not a completed-work throughput measurement. Different cases can reach different graph phases.'])
    (ROOT/f'{name}-analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(name,'runtime launches',launches,'CPU cores',result['observed_host_cpu_core_equivalents'],
        'segment GPU %',result['sampled_segment_four_gpu_busy_percent'],'Boot overlap',result['boot_overlap_seconds'])
    for k,s in sorted(apis.items(),key=lambda kv:kv[1]['total_ns'],reverse=True)[:12]:
        print(k,s['count'],'mean us',round(s['mean_us'],3),'cumulative s',round(s['cumulative_seconds'],3))
    return result

if __name__=='__main__':
    for name in ('api-probe','api-probe-late'):
        if (ROOT/name/'status.json').exists():summarize(name)
