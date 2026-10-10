import json
from pathlib import Path
root=Path(__file__).resolve().parent
baseline=json.loads((root.parent/'memory-v1-review/1gpu/mlp.baseline.numerical.json').read_text())
models=[]
for path in sorted((root/'mlp').glob('*.json')):
    row=json.loads(path.read_text())
    error=max(abs(a-b) for a,b in zip(row['poseidon_output'],baseline['poseidon_output']))
    assert error<1e-4 and row['passed']
    assert all(row[k]==baseline[k] for k in ['input_level','input_scale_log2','output_level','output_scale_log2'])
    models.append({'name':path.stem,'passed':row['passed'],'runtime_seconds':row['runtime_seconds'],
                   'max_error_vs_eager_ckks':error,'metadata_matches_eager':True,'timing':row['runtime_timing']})
memory=[]
for path in sorted((root/'memory').glob('*.checks.json')):
    row=json.loads(path.read_text())
    assert row['pinned_live_bytes']==0
    baseline_bytes=row['memory_baseline_bytes']
    peaks=[0]*len(baseline_bytes)
    for iteration in row['iterations']:
        for i,d in enumerate(iteration['devices']):
            assert d['active_bytes']==baseline_bytes[i]
            peaks[i]=max(peaks[i],d['peak_increment_bytes'])
    memory.append({'name':path.name,'baseline_bytes':baseline_bytes,'peak_increment_bytes':peaks,
                   'pinned_peak_bytes':row['pinned_process_peak_bytes'],'pinned_final_bytes':row['pinned_live_bytes'],
                   'all_iterations_returned_to_baseline':True})
summary={'model_numerical':models,'memory':memory,
         'trace':{kind:json.loads((root/f'trace-{kind}.json').read_text()) for kind in ['stream','prefetch']}}
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print('models',len(models),'memory cases',len(memory),'max CKKS eager difference',max(m['max_error_vs_eager_ckks'] for m in models))
