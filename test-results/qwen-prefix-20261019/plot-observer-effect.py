#!/usr/bin/env python3
"""Plot the separate observer diagnostic without calling it model throughput."""
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
source = json.loads((root/'observer-query-aba-diagnostic.json').read_text())
rates = []
for name in ('a', 'b', 'c'):
    begin, end = source[name+'_begin'], source[name+'_end']
    seconds = end['monotonic']-begin['monotonic']
    delta = int(end['io'].splitlines()[0].split()[1])-int(begin['io'].splitlines()[0].split()[1])
    rates.append(dict(phase=name,seconds=seconds,bytes=delta,mib_per_second=delta/seconds/2**20))
(root/'smi-observer-effect-summary.json').write_text(json.dumps(dict(phases=rates,
    query_median_seconds=statistics.median(q['seconds'] for q in source['queries']),
    metric='Model /proc rchar progress proxy; not instruction throughput'),indent=2)+'\n')
fig, ax = plt.subplots(figsize=(8,4.8),layout='constrained')
values = [r['mib_per_second'] for r in rates]
ax.bar(range(3),values,color=['#376bc5','#cc5546','#376bc5'],width=.6)
ax.set_xticks(range(3),['A: No query restarts\n12s','B: Repeated SMI queries\n12.9s',
                       'C: No query restarts\n12s'],fontsize=10)
for index, value in enumerate(values):
    ax.text(index,value+.2,f'{value:.2f}',ha='center',fontsize=12)
fig.suptitle('Observer interference: short A/B/A diagnostic',fontsize=14)
ax.set(ylabel='Model read progress (MiB/s)',ylim=(0,max(values)*1.2))
ax.set_title('Persistent pmon unchanged; this is not a whole-model speed comparison.',fontsize=9,pad=12)
ax.grid(axis='y',alpha=.2)
fig.savefig(root/'observer-effect.png',dpi=160)
plt.close(fig)
