#!/usr/bin/env python3
"""Standalone benchmark figure; bar whiskers show sample minima/maxima."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

out=Path(__file__).resolve().parent
data=json.loads((out/'comparison.json').read_text())
lookup={(r['group'],r['name']):r for r in data['comparisons']}
fig,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained')
titles={'local-mlp':'RTX 4060 / MLP','local-qproj':'RTX 4060 / q_proj proxy',
        'local-pressure':'RTX 4060 / 3.125 GiB weights','server-mlp1':'V100 x1 / MLP',
        'server-mlp2':'V100 x2 / MLP','server-pressure':'V100 / 12.5 GiB weights'}
labels=['Original\nfull','Previous\nfull','Updated\nfull','Unfixed\nWL','Previous\nWL','Updated\nWL']
colors=['#577590','#e5a34d','#35946b','#b35b63','#e5a34d','#35946b']
for ax,group in zip(axes.flat,titles):
    metrics=[lookup[group,name]['online_ms'] for name in ('legacy-eager','previous-eager','updated-eager')]
    metrics+=[lookup[group,'unfixed-waterline']['online_ms']]
    metrics+=[lookup[group,name]['online_ms'] for name in ('previous-waterline','updated-waterline')]
    values=[m['median']/1000 for m in metrics]
    errors=[[v-m['min']/1000 for v,m in zip(values,metrics)],
            [m['max']/1000-v for v,m in zip(values,metrics)]]
    bars=ax.bar(range(6),values,color=colors,width=.7,yerr=errors,capsize=3,
                error_kw=dict(elinewidth=.9,capthick=.9),zorder=3)
    for bar in bars[:3]:bar.set_hatch('//');bar.set_alpha(.75)
    for i,(value,m) in enumerate(zip(values,metrics)):
        ax.text(i,m['max']/1000*1.08,f'{value:.3f}',ha='center',va='bottom',fontsize=8)
    ax.set_yscale('log');ax.minorticks_off()
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value,pos:f'{value:g}'))
    ax.set_ylim(min(m['min']/1000 for m in metrics)*.72,max(m['max']/1000 for m in metrics)*1.55)
    ax.set_xticks(range(6),labels,fontsize=8)
    ax.set_title(titles[group],fontsize=11)
    ax.grid(axis='y',alpha=.25,zorder=0)
    ax.spines[['top','right']].set_visible(False)
    ax.set_ylabel('Online time (s, log scale)',fontsize=9)
fig.suptitle('Continuous submission and GPU-first preparation',fontsize=15)
fig.supxlabel('Online includes runtime preparation for WL; full-resident preparation is upfront. All Runtime versions share the same corrected GPU backend.\nWhiskers: min/max; 3 samples, or 10 for the main RTX 4060 MLP cases.',fontsize=8)
fig.savefig(out/'comparison.png',dpi=180)
fig.savefig(out/'comparison.svg')
print(out/'comparison.png')
