"""Compare old immutable requests and fresh chunk runtime I/O metadata."""
import json,hashlib
from pathlib import Path
from unified_graph_contract import prepare
from benchmark_graph import canonical,require
from benchmark_runner import DEFAULT,load
from hecate_python_env import WORK
base=WORK/'results'
paths=[]
for name in ('upstream-concat-r18','upstream-helper-r18-catalog','upstream-concat-r18-controls',
'upstream-spatial-r22-native-initial','upstream-spatial-r22-helper-directed','upstream-spatial-r22-regression',
'upstream-spatial-r22-chunk-regression','upstream-spatial-r22-frozen-corpus-validated',
'upstream-spatial-r23-native-final','upstream-spatial-r23-helper-directed','upstream-spatial-r23-regression','upstream-spatial-r23-chunk-regression',
'upstream-fused-r24-verified-native','upstream-fused-r24-verified-helper-directed','upstream-fused-r24-verified-regression','upstream-fused-r24-verified-chunk-regression',
'upstream-ds-r25-native','upstream-ds-r25-helper-directed','upstream-ds-r25-regression','upstream-ds-r25-chunk-regression'):
    report=json.loads((base/name/'report.json').read_text())
    for row in report.get('rows',[])+report.get('records',[]):
        if row.get('evidence'):paths.append(Path(row['evidence'])/'request.json')
paths.append(base/'candidate-replay-g2fr2jh4/request.json')
records=[]
for path in dict.fromkeys(paths):
    r=json.loads(path.read_text())
    new=prepare(r['model'],r['compiler_profile_sha256'],r.get('compiler_configuration'),
        r.get('construction_exercise',{}).get('id'),construction_profile=r.get('construction_profile'),
        constant_policy=r['constant_origins'].get('policy'),helper_profile=r.get('upstream_helpers',{}).get('profile'),
        helper_exercise=r.get('upstream_exercise',{}).get('required_helpers'),
        chunk_period=r['layout']['input_slot_period'] if r['layout']['execution_abi']=='unified-chunked-inputs-v1' else None)
    require(canonical(new)==canonical(r),'Changed legacy request '+str(path))
    records.append(dict(path=str(path),request_id=r['request_id'],identical=True))
rows,index=load(DEFAULT);old=Path('scripts/baseline/benchmarks/semantic-v1-downsample-r25')
oi=json.loads((old/'index.json').read_text());oc=json.loads((old/'coverage.json').read_text())
nc=json.loads((DEFAULT/'coverage.json').read_text())
require(index['shards']==oi['shards'],'Changed models')
for name in ('directed_tasks','rejection_tasks'):
    require(nc[name]==oc[name],'Changed old task '+name)

old_helpers=oc['helper_directed_tasks'];current={t['id']:t for t in nc['helper_directed_tasks']}
require(all(current[t['id']]==t for t in old_helpers),'Changed old helper tasks')
out=base/'upstream-virtual-r26-verified-compatibility.json';require(not out.exists(),'Preserve audit')
from benchmark_runner import dump
dump(out,dict(schema=1,old_requests=len(records),all_identical=True,records=records,
    old_shards_and_tasks_identical=True,old_helpers_identical=len(old_helpers),new_helpers=len(current)-len(old_helpers),
    model_set_sha256=index['model_set_sha256'],checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    new_executions=0))
print('Old requests byte-identical:',len(records))
