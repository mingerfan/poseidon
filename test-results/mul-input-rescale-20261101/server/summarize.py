from pathlib import Path
from collections import Counter
import json,re
r=Path(__file__).resolve().parent
rows={}
for name in ('qwen-before','qwen-after','resnet-after'):
    d=r/name
    record=json.loads((d/'result.json').read_text())
    log=(d/'compile.log').read_text()
    count=re.findall(r'Number of Bootstrapping:\s*(\d+)',log)
    record.pop('trace_boots',None)
    if count:record['boots']=int(count[-1])
    if name.startswith('qwen'):
        trace=json.loads((d/'scale-trace.json').read_text())
        boots=[b for row in trace for b in row.get('boot',[])]
        record['regional_trace_boots']=len(boots)
        record['boot_input_q_histogram']=dict(sorted(Counter(b['q'] for b in boots).items()))
        record['boot_input_q_ge21']=sum(b['q']>=21 for b in boots)
        timings=re.findall(r'^\s*([\d.]+)\s*\([^\n]*\)\s+GreedyBootstrapPlacement\s*$',log,re.M)
        if timings:record['planner_seconds']=float(timings[-1])
    else:
        audit=json.loads((d/'audit.json').read_text())
        record.update(boots=audit['counts']['boot'],rescales=audit['counts']['rescale'],
                      output_q=audit['output_q'],output_s=audit['output_s'],verifier_passed=audit['verifier_passed'],
                      content_set_sha256=audit['content_set_sha256'])
    time_log=(d/'time.log').read_text()
    peak=re.search(r'Maximum resident set size \(kbytes\): (\d+)',time_log)
    if peak:record['peak_rss_kib']=int(peak[1])
    rows[name]=record
rows['scope']='Qwen3-0.6B, 28 layers, 2-token prefill; same trace/profile/output s40 and greedy beam4. Full precompile including scale materialization; no full runtime export or encrypted GPU inference. ResNet20 uses optimized beam32 and verified physical runtime plan.'
rows['historical_qwen']={'boots':12315,'seconds':369.7507575158961,'planner_seconds':157.9938,'scope':'Before both linear reduction and multiplication input fixes; same Qwen trace/profile/beam4/output s40.'}
rows['micro']=json.loads((r/'verification.json').read_text())
rows['tests']=json.loads((r/'remote-tests-result.json').read_text())
(r/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps({k:{a:v for a,v in row.items() if a in ('boots','seconds','planner_seconds','peak_rss_kib','output_q','output_s','rescales')} for k,row in rows.items() if k in ('qwen-before','qwen-after','resnet-after')},indent=2))
