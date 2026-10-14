"""Measure H2D/kernel overlap on the same GPU within online NVTX ranges."""
import argparse
import json
import sqlite3
from pathlib import Path


def union(intervals):
    result = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result


def analyze(path):
    connection = sqlite3.connect(path)
    ranges = connection.execute("select text,start,end from NVTX_EVENTS where text like 'online.iteration.%' order by start").fetchall()
    rows=[]
    for name, begin, end in ranges:
        copy_ns=overlap_ns=copy_bytes=copy_count=0
        for device in range(4):
            kernels=union(connection.execute('select start,end from CUPTI_ACTIVITY_KIND_KERNEL where deviceId=? and start>=? and end<=?',(device,begin,end)))
            copies=connection.execute('select start,end,bytes from CUPTI_ACTIVITY_KIND_MEMCPY where deviceId=? and copyKind=1 and start>=? and end<=? order by start',(device,begin,end)).fetchall()
            cursor=0
            for start,stop,size in copies:
                copy_ns += stop-start;copy_bytes+=size;copy_count+=1
                while cursor<len(kernels) and kernels[cursor][1]<=start:
                    cursor+=1
                i=cursor
                while i<len(kernels) and kernels[i][0]<stop:
                    overlap_ns+=max(0,min(stop,kernels[i][1])-max(start,kernels[i][0]));i+=1
        rows.append({'range':name,'wall_seconds':(end-begin)*1e-9,'h2d_count':copy_count,'h2d_bytes':copy_bytes,
                     'h2d_milliseconds':copy_ns*1e-6,'same_gpu_kernel_overlap_milliseconds':overlap_ns*1e-6,
                     'overlap_fraction':overlap_ns/copy_ns if copy_ns else 0})
    return {'source':str(path),'scope':'all H2D copies in online iterations; overlap with union of kernels on the same GPU; NVTX range includes output check', 'iterations':rows}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('database',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    a.output.write_text(json.dumps(analyze(a.database),indent=2)+'\n')
