import json,math
from pathlib import Path
root=Path(__file__).resolve().parent
results=root/'188-results'
read=lambda p:json.loads(p.read_text())
summary={'remote_checks':{},'comparisons':{},'memory':{},'scope':'MLP degree 8192 without Boot; four physical GPUs on one host. RMM peak increments include asynchronous work and workspaces, with cached allocations recorded separately.'}
for report in sorted(results.glob('*.json')):
 if '.checks.' in report.name or report.name.endswith('.checks.json'):continue
 d=read(report);assert d['passed'],report
 summary['remote_checks'][report.name]=True
for check in results.glob('*.checks.json'):
 d=read(check)
 for iteration in d['iterations']:
  for j,device in enumerate(iteration['devices']):
   assert device['net_change_bytes']==0 and device['active_bytes']==d['memory_baseline_bytes'][j],check
 for numeric in d['numerical_checks']:assert numeric['repeat_max_error']<=1e-4,check
 assert not d['oracle_provided'] or all(x['expected_max_error']<=1e-4 for x in d['numerical_checks']),check
for topology in ['1gpu','4gpu','2x2']:
 def output(variant):
  checks=[read(p) for p in results.glob(f'mlp-{topology}-{variant}-repeat.json.rank*.checks.json')]
  with_output=[d['decoded_output'] for d in checks if d['decoded_output']]
  assert len(with_output)==1,(topology,variant)
  return [complex(*x) for x in with_output[0]]
 baseline=output('baseline')
 for variant in ['release','reuse']:
  actual=output(variant);assert len(actual)==len(baseline)
  error=max(abs(a-b) for a,b in zip(actual,baseline))
  assert math.isfinite(error) and error<1e-4,(topology,variant,error)
  summary['comparisons'][f'{topology}-{variant}']={'full_slot_max_error':error,'tolerance':1e-4,'slots':len(actual)}
 estimated={v:read(root/topology/f'mlp.{v}._hecate_MLP.memory.json') for v in ['baseline','release','reuse']}
 measured={v:[read(p) for p in sorted(results.glob(f'mlp-{topology}-{v}-repeat.json.rank*.checks.json'))] for v in ['baseline','release','reuse']}
 rows=[]
 for place in estimated['reuse']['places']:
  if place['place']['kind']!='device':continue
  rank,index=place['place']['rank'],place['place']['index']
  row={'rank':rank,'index':index,'estimate_peak_bytes':{},'rmm_warmup_baseline_bytes':{},'rmm_peak_increment_bytes':{}}
  for variant in ['baseline','release','reuse']:
   row['estimate_peak_bytes'][variant]=next(p['peak_bytes'] for p in estimated[variant]['places'] if p['place']==place['place'])
   checks=next(d for d in measured[variant] if d['rank']==rank)
   row['rmm_warmup_baseline_bytes'][variant]=checks['memory_baseline_bytes'][index]
   row['rmm_peak_increment_bytes'][variant]=max(it['devices'][index]['peak_increment_bytes'] for it in checks['iterations'])
  rows.append(row)
 summary['memory'][topology]=rows
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print('Summary checks passed:',len(summary['remote_checks']),'reports; all repetitions return to baseline')
for name,d in summary['comparisons'].items():print(name,'max error',d['full_slot_max_error'])
for name,rows in summary['memory'].items():
 print(name,'estimate reuse MiB',[r['estimate_peak_bytes']['reuse']/2**20 for r in rows], 'RMM reuse peak increment MiB',[r['rmm_peak_increment_bytes']['reuse']/2**20 for r in rows])
