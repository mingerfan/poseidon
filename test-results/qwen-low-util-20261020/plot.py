#!/usr/bin/env python3
"""Standalone static figure for the two independent diagnostic measurements."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root=Path(__file__).resolve().parent
n=json.loads((root/'nsys-late-analysis.json').read_text())
c=json.loads((root/'api-probe-late-analysis.json').read_text())
fig,axes=plt.subplots(3,1,figsize=(12,10),gridspec_kw={'height_ratios':[1.1,1.3,1.5]})
fig.subplots_adjust(top=.91,bottom=.13,hspace=.78,left=.23,right=.97)
w=n['window_ns'];left=np.zeros(4)
rows=n['devices']
categories=[('Kernel','#2a9d78',[d['kernel_union_ns'] for d in rows]),
    ('Copy/memset only','#457b9d',[d['any_gpu_activity_union_ns']-d['kernel_union_ns'] for d in rows]),
    ('Idle: before next CUDA API','#e9b35e',[d['gap_parts'].get('before_corresponding_api_entry_ns',0) for d in rows]),
    ('Idle: in CUDA API','#bf6c69',[d['gap_parts'].get('inside_corresponding_api_ns',0) for d in rows]),
    ('Idle: after API / unmatched','#b6bbc4',[d['gap_parts'].get('after_corresponding_api_return_ns',0)+d['gap_parts'].get('unmatched_ns',0) for d in rows])]
for label,color,values in categories:
    widths=np.array(values)*100/w
    axes[0].barh(range(4),widths,left=left,label=label,color=color)
    left+=widths
assert np.allclose(left,100)
axes[0].set(yticks=range(4),yticklabels=[f'GPU {d}' for d in range(4)],xlim=(0,100),xlabel='Percent of the 20-second capture',title='Nsight: most idle time occurs before the next CUDA API is entered')
axes[0].legend(loc='upper center',bbox_to_anchor=(.5,-.30),ncol=3,fontsize=8,frameon=False)
bins=list(csv.DictReader((root/'nsys-late-gpu-seconds.csv').open()))
seconds=sorted({int(b['second']) for b in bins})
for d,color in enumerate(('#376bc5','#e0872e','#2a9d78','#bf5f97')):
    selected=[b for b in bins if int(b['device'])==d and int(b['second'])<int(w/1e9)]
    axes[1].plot([int(b['second'])+.5 for b in selected],
        [100*float(b['any_activity_fraction']) for b in selected],label=f'GPU {d}',color=color,lw=1.3)
for a,b in n['host_refresh_intervals']:
    axes[1].axvspan(a/1e9,b/1e9,color='#66717c',alpha=.15,lw=0)
axes[1].set(xlim=(0,w/1e9),ylim=(0,100),ylabel='Kernel/copy active (%)',xlabel='Seconds within capture',title='Shaded intervals: CPU Host refresh; curves: GPU activity in 1-second bins')
axes[1].legend(ncol=4,loc='upper right',fontsize=8)
axes[1].grid(alpha=.2)
names=['cudaSetDevice','cudaEventQuery','cudaLaunchKernel_ptsz','cudaMemcpyAsync_ptsz','cudaEventRecord','cudaEventCreateWithFlags']
counts=[c['apis'][name]['count']/c['runtime_kernel_launch_calls'] for name in names]
axes[2].barh(range(len(names)),counts,color='#457b9d')
axes[2].set(yticks=range(len(names)),yticklabels=[n.replace('_ptsz','') for n in names],xlabel='Calls per intercepted runtime kernel launch',title='Independent run without nsys: completion/management calls per kernel launch')
axes[2].invert_yaxis()
for i,(name,value) in enumerate(zip(names,counts)):
    stat=c['apis'][name]
    axes[2].text(value,i,f'  {value:.2f} ({stat["mean_us"]:.2f} us/call)',va='center',fontsize=8)
axes[2].set_xlim(0,max(counts)*1.5)
axes[2].grid(axis='x',alpha=.2)
fig.suptitle('Qwen on four V100s: later 20-second diagnostic segments',fontsize=14)
fig.text(.23,.025,'Profiled and unprofiled segments can reach different graph positions. These are diagnostics, not the full-run NVML metric.\nAPI times are CPU wall time; activity intervals include all GPU streams.',fontsize=8)
fig.savefig(root/'low-util-diagnosis.png',dpi=170)
plt.close(fig)
