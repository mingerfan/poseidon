
"""Audit original frozen model identity and nested actual helper evidence."""
import json,hashlib
from pathlib import Path
from collections import Counter
from benchmark_runner import DEFAULT,load,dump
from benchmark_graph import digest,require
from hecate_python_env import WORK
from audit_unified_candidate import verify_candidate
from semantic_benchmark_execution import runtime_sources
base=WORK/'results';report=json.loads((base/'upstream-virtual-r26-verified-native/report.json').read_text())
models,_=load(DEFAULT)
wanted={r['model']['id']:r for r in models if r['metadata'].get('helper') in ('HE_MPBN','HE_Linear','HE_ReshapeLinear')}
require(len(wanted)==24,'Frozen helper denominator changed')
rows={r['id']:r for r in report['rows']};require(set(wanted)<=set(rows),'Frozen cases missing')
records=[];families=Counter();splits=Counter();eliminations=0;inner_calls=0
for name,row in wanted.items():
    run=rows[name];require(run['status']=='passed','Frozen case did not pass')
    folder=Path(run['evidence']);request=json.loads((folder/'request.json').read_text())
    require(digest(request['model'])==row['model_sha256'],'Changed original mathematical model')
    checked=verify_candidate(folder)
    require(checked['helper_coverage']['finite_return_influence_checked'],'No directed witness')
    actual=json.loads((folder/'attempt-00/output/upstream-calls.json').read_text())
    count=0;zero=0
    for call in actual['bound_calls']:
        a=call['actual'];calls=a.get('inner_calls',[a])
        count+=len(calls)
        zero+=sum(c['virtual']['public_support_zero_elisions'] for c in calls)
    inner_calls+=count;eliminations+=zero;families.update([row['metadata']['helper']]);splits.update([row['split']])
    records.append(dict(id=name,model_sha256=row['model_sha256'],topology=row['topology'],split=row['split'],
        helper=row['metadata']['helper'],evidence=str(folder),request_id=request['request_id'],
        actual_inner_virtual_calls=count,public_zero_eliminations=zero,comparison=checked['comparison']))
out=base/'upstream-virtual-r26-verified-original-corpus-audit.json';require(not out.exists(),'Preserve audit')
dump(out,dict(schema=1,status='passed',planned=24,passed=24,failed=0,blocked=0,skipped=0,
    families=dict(families),splits=dict(splits),records=records,models_unchanged=True,
    total_actual_inner_virtual_calls=inner_calls,total_public_zero_eliminations=eliminations,
    agent_calls=0,manual_reference_and_backend_acceptance_not_agent_holdout_score=True,
    source_sha256=digest(runtime_sources()),checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
print(dict(families=dict(families),splits=dict(splits),actual_inner_calls=inner_calls,public_zero_eliminations=eliminations))
