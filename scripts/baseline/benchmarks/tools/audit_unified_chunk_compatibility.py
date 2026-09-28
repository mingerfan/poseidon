"""Compare old immutable requests and fresh chunk runtime I/O metadata."""
import json,hashlib
from pathlib import Path
from unified_graph_contract import prepare
from benchmark_graph import canonical,require
from benchmark_runner import DEFAULT,load
from hecate_python_env import WORK
base=WORK/'results'
paths=[]
for name in ('upstream-concat-r18','upstream-helper-r18-catalog','upstream-concat-r18-controls'):
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
        helper_exercise=r.get('upstream_exercise',{}).get('required_helpers'))
    require(canonical(new)==canonical(r),'Changed legacy request '+str(path))
    records.append(dict(path=str(path),request_id=r['request_id'],identical=True))
rows,index=load(DEFAULT);old=Path('scripts/baseline/benchmarks/semantic-v1-rejection-r20')
oi=json.loads((old/'index.json').read_text());oc=json.loads((old/'coverage.json').read_text())
nc=json.loads((DEFAULT/'coverage.json').read_text())
require(index['shards']==oi['shards'],'Changed models')
for name in ('directed_tasks','helper_directed_tasks','rejection_tasks'):
    require(nc[name]==oc[name],'Changed old task '+name)
report=json.loads((base/'unified-chunk-r21/report.json').read_text())
from audit_unified_candidate import verify_candidate
checked=[]
for row in report['rows']:
    folder=Path(row['evidence']);result=verify_candidate(folder)
    req=json.loads((folder/'request.json').read_text())
    bound=json.loads((folder/'logical-bindings.json').read_text())
    require(bound==req['layout'],'Written bindings differ')
    original=json.loads((folder/'report.json').read_text())
    execution=original['attempts'][0]['execution']
    require(execution['logical_tensor_count']==len(req['model']['inputs']),'Logical tensor count')
    require(execution['model_chunk_count']==len(req['layout']['inputs']),'Physical model input count')
    require(execution['encrypted_input_count']==len(req['layout']['inputs'])+1,'Physical encrypted input count')
    require(len(execution['ciphertext_metadata'])==4,'Four batches')
    for observation in execution['ciphertext_metadata']:
        require(len(observation['inputs'])==execution['encrypted_input_count'],'Actual input metadata')
        require(len(observation['outputs'])==req['layout']['output_ciphertexts'],'Actual chunk result metadata')
    checked.append(dict(id=row['id'],request_id=req['request_id'],inputs=execution['logical_tensor_count'],
        model_chunks=execution['model_chunk_count'],output_chunks=req['layout']['output_ciphertexts'],
        physical_metadata_verified=True,numerical_passed=result['comparison']['passed'],
        coverage=result['coverage']))
out=base/'unified-chunk-r21-compatibility-final.json';require(not out.exists(),'Preserve audit')
out.write_text(json.dumps(dict(schema=1,old_requests=len(records),all_identical=True,records=records,
    chunk_execution_audit=checked,old_shards_and_tasks_identical=True,model_set_sha256=index['model_set_sha256'],
    checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),new_executions=0),indent=2)+chr(10))
print('Old requests byte-identical:',len(records))
print('Audited actual chunk bindings:',len(checked))
