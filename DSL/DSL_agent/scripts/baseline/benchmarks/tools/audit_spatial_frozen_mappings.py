"""Static per-node helper eligibility, not a compilation/execution result."""
from collections import Counter
import hashlib,json
from benchmark_runner import load,DEFAULT,dump
from benchmark_graph import require
from upstream_candidate_helpers import manifest,SPATIAL_PROFILE
from hecate_python_env import WORK
rows,index=load(DEFAULT);records=[];families=Counter();blocked=Counter()
for row in rows:
    model=row['model'];cap=manifest(model,SPATIAL_PROFILE)
    bound={name:spec['binding'] for name,spec in cap['helpers'].items() if name.startswith(('HE_Conv','HE_Avg','HE_Pool'))}
    unavailable=[r for r in cap['unavailable_bindings'] if r.get('helper')=='spatial']
    if bound or unavailable:
        families.update(b['helper'] for b in bound.values());blocked.update(r['reason'] for r in unavailable)
        records.append(dict(id=model['id'],model_sha256=row['model_sha256'],metadata=row['metadata'],
            bindings={n:dict(node_id=b['node_id'],helper=b['helper'],period=b['period'],work=b['work']) for n,b in bound.items()},
            unavailable=unavailable,status='eligibility_only_not_executed'))
out=WORK/'results/upstream-spatial-r22-frozen-mappings.json';require(not out.exists(),'Preserve mapping')
dump(out,dict(schema=1,model_count=len(rows),model_set_sha256=index['model_set_sha256'],mapped_models=len(records),
    bound_node_counts=dict(families),blocked_node_counts=dict(blocked),records=records,new_executions=0,agent_calls=0))
print(dict(mapped_models=len(records),bound_node_counts=dict(families),blocked_node_counts=dict(blocked)))
