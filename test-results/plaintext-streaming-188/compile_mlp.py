"""Compile the existing placed MLP with two bounded preparation schedules."""
import hashlib
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[2]
compiler = root/'third_party/ckks-runtime/third_party/dacapo'
spec_path = root/'test-results/e2e-new-profile-e63ec18/profiles/operator-spec.json'
spec = json.loads(spec_path.read_text())
digest = 'sha256:' + hashlib.sha256(spec_path.read_bytes()).hexdigest()
out = root/'build-plaintext-models'
out.mkdir(exist_ok=True)
for topology in ['1gpu', '4gpu']:
    for variant, prefetch in [('stream', 0), ('prefetch', 8*2**20)]:
        prefix = out/f'{topology}-{variant}'
        passes = [f'plan-plaintext-streaming{{prefix={prefix} operator-spec={spec_path} gpu-budget-bytes={256*2**20} host-budget-bytes={16*2**20} pinned-budget-bytes={8*2**20} raw-budget-bytes={2**20} workspace-bytes-per-op={2**20} prefetch-bytes={prefetch}}}',
                  'plan-runtime-memory',
                  f'estimate-runtime-memory{{prefix={prefix} operator-spec={spec_path}}}',
                  f'emit-runtime-plan{{prefix={prefix} plan-id=991 target-id={spec["target_id"]} operator-spec-id={spec["spec_id"]} operator-spec-version={spec["version"]} operator-spec-sha256={digest} context-id={spec["context"]["context_id"]} boot-profile={spec["boot_profiles"][0]["profile_id"]} boot-implementation=decrypt_reencrypt}}']
        result = subprocess.run([str(compiler/'build/nix/bin/hecate-opt'),
                str(root/f'test-results/memory-v1-review/{topology}/mlp.input.mlir'),
                '-p=builtin.module(func.func('+','.join(passes)+'))', '-o', str(prefix)+'.mlir'],
                cwd=compiler, capture_output=True, text=True)
        (out/f'{topology}-{variant}.compile.log').write_text(result.stderr)
        result.check_returncode()
        report = json.loads(Path(str(prefix)+'._hecate_MLP.streaming.json').read_text())
        print(topology, variant, report['batches'], report['object_peak_bytes_host_then_gpu'])

# Exercise the generic full-model smoke runner with multiple external/final IDs.
plan=json.loads((out/'4gpu-stream._hecate_MLP.runtime-plan.json').read_text())
external=plan['external_inputs'][0]
plan['final_outputs'].append(external)
for phase in ['initialization','execution','finalization']:
    plan[phase]=[op for op in plan[phase] if not(op['kind']=='release' and op['value']==external)]
ordinal=0
for phase in ['initialization','execution','finalization']:
    for op in plan[phase]:
        op['ordinal']=ordinal
        ordinal+=1
value=dict(next(v for v in plan['values'] if v['id']==external))
value['id']='9000000'
plan['values'].append(value)
plan['external_inputs'].append(value['id'])
plan['final_outputs'].append(value['id'])
(out/'multi-io.json').write_text(json.dumps(plan))
