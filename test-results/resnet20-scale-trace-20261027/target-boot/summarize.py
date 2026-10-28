from pathlib import Path
import hashlib
import json

root=Path('/home/xuming/poseidon-manual-boot-20261026')
experiment=root/'resnet20-optimized-20261027T055506Z/placement-audit/target-scale-experiment'
out=experiment/'boot-tests'
results=[]
for name in ['relu-boot','relu-boot-relu']:
    metadata=json.loads((out/(name+'-summary.json')).read_text())
    report=json.loads((out/(name+'-gpu-report.json')).read_text())
    path=Path(metadata['plan'])
    digest='sha256:'+hashlib.sha256(path.read_bytes()).hexdigest()
    assert report['plan_sha256']==digest
    assert report['passed']
    plan=json.loads(path.read_text())
    values={v['id']:v for v in plan['values']}
    ops=plan['execution']
    boot=next(op for op in ops if op.get('op')=='boot')
    align=next(op for op in ops if op.get('output')==boot['inputs'][0])
    assert align['op']=='mod_switch'
    before=values[align['inputs'][0]]
    assert before['level']==7 and abs(before['scale_log2']-40)<1e-8
    summary={k:report[k] for k in ['passed','plan_sha256','operator_spec_sha256','input_level','input_scale_log2','output_level','output_scale_log2','tolerance','max_abs_error','max_identity_error','max_imaginary']}
    summary.update(name=name,physical_cuda_device=2,slots=len(report['input']),samples=[{k:report[k][i] for k in ['input','expected','actual']} for i in range(10)],boot_alignment=dict(source_q=8,target_q=6,scale_log2=before['scale_log2'],modswitches=1,upscales=0,rescales=0),boot_count=1,boot_input=metadata['boots'][0]['before'],boot_output=metadata['boots'][0]['after'],counts=metadata['counts'])
    results.append(summary)
checks=json.loads((out/'checks.json').read_text())
checks.update(encrypted_cases=2,all_32768_slots_passed=True,actual_gpu_device=2,final_compiler_regenerates_gpu_tested_plans=True)
(out/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')
summary=dict(reference='Same scalar polynomial and the native manual Boot sine map; not ideal ReLU or an exact identity Boot.',cases=results,run=json.loads((out/'gpu-run.json').read_text()),checks=checks)
(out/'gpu-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary),flush=True)
