"""Regression: 40,000 same-rank transfers must complete without retained threads.

This synthetic copy-only plan tests resource cleanup, not ResNet inference.
The model itself always uses the compiler-emitted plan.
"""
import copy
import hashlib
import json
import subprocess
from pathlib import Path

if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    work = root/'transfer-stress'
    work.mkdir()
    template = json.loads(next((root/'compiled-jit-prefix').glob('*.runtime-plan.json')).read_text())
    original = {v['id']:v for v in template['values']}
    ids = template['external_inputs']
    values = [copy.deepcopy(original[i]) for i in ids]
    execution = []
    previous = ids[0]
    source = dict(kind='host',rank=0)
    for i in range(40000):
        output = str(i+8)
        destination = dict(kind='device',rank=0,index=0) if i%2==0 else dict(kind='host',rank=0)
        desc = copy.deepcopy(original[ids[0]])
        desc.update(id=output,place=destination)
        values.append(desc)
        execution.append(dict(kind='transfer',ordinal=len(execution),transfer_id=str(i),
                              hint='point_to_point',inputs=[previous],outputs=[output],
                              sources=[source],destinations=[destination],output_kinds=['ciphertext']))
        if i:
            execution.append(dict(kind='release',ordinal=len(execution),value=previous))
        previous,source = output,destination
    plan = {k:template[k] for k in ('target','external_inputs')}
    plan.update(format_version=2,plan_id='99018',values=values,initialization=[],
                execution=execution,finalization=[],final_outputs=[previous])
    (work/'plan.json').write_text(json.dumps(plan))
    fixture = json.loads((root/'fixture-final.json').read_text())
    fixture.update(python_output=fixture['packed_inputs'][0][:1000],
                   model_sha256=hashlib.sha256(b'copy-only-40000-transfer-resource-test').hexdigest(),
                   scope='resource stress test; not model inference')
    (work/'fixture.json').write_text(json.dumps(fixture))
    bundle = work/'bundle'
    bundle.mkdir()
    profiles = root.parent/'resnet20-framework-v1/profiles-native-v4'
    with (work/'run.log').open('x') as log:
        subprocess.run(['/usr/bin/time','-v',str(root/'runtime-build-final/dacapo_mlp_vec_e2e'),
                        str(work/'plan.json'),str(profiles/'operator-spec.json'),str(bundle),
                        str(work/'fixture.json'),str(work/'result.json'),str(work/'diff.log')],
                       stdout=log,stderr=subprocess.STDOUT,check=True)
    result = json.loads((work/'result.json').read_text())
    assert result['passed'] and result['python_max_abs_diff']==0
    print('TRANSFER_RESOURCE_STRESS_PASS 40000 copies; not model inference',flush=True)
