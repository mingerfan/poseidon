import json
from pathlib import Path
from benchmark_runner import load,DEFAULT,dump
from benchmark_graph import digest
from compiler_configuration import PROFILE_SHA256,configuration
from unified_graph_contract import prepare,validate_candidate
from unified_graph_lowering import lower
from unified_public_exercises import golden_variant
from hecate_python_env import WORK
rows,index=load(DEFAULT);models={r['model']['id']:r['model'] for r in rows}
coverage=json.loads((DEFAULT/'coverage.json').read_text());results=[]
for t in coverage['directed_tasks']:
 if t['model_id'] not in ('bench_boundary_0044','bench_boundary_0061','bench_boundary_0097'):continue
 row=dict(task=t['id'],model=t['model_id'],exercise=t['exercise'])
 try:
  r=prepare(models[t['model_id']],PROFILE_SHA256,configuration('seal-cpu-eva-w45-v1'),t['exercise'],construction_profile=t['profile'])
  src=golden_variant(lower(r),t['exercise'],r)
  result=validate_candidate(dict(schema=1,request_id=r['request_id'],hecate_source=src),r)
  row.update(status='validated',source_bytes=len(src),check=result)
 except ValueError as e:row.update(status='blocked',reason=str(e))
 results.append(row)
dump(WORK/'results/baseline-boundaries-r27-directed-diagnostic.json',dict(rows=results,agent_calls=0))
print(json.dumps(results,default=str)[:5000])
