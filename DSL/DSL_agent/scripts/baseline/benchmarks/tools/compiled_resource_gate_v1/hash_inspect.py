import json,hashlib,pathlib
from public_weight_context import read_request,request_digest
from compiler_configuration import canonical,request_configuration
from candidate_contract import strict_json
root=pathlib.Path('/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-v12-source-live-rebound-v2')
for name in ['linear','norm','split_weight']:
 file=next((root/name).glob('tasks/*/program/request.json'))
 raw=file.read_text()
 for method in [json.loads,read_request]:
  r=method(raw)
  h=request_digest(r)
  print(name,method.__name__,r['request_id'],h,'same',h==r['request_id'])
  print('schema_request_id',r['response_schema'])
  print('backend',r['compiler_configuration']['name'],'prefix',raw[:100])
