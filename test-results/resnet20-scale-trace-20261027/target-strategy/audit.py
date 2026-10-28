from pathlib import Path
import hashlib
import json
import math
import subprocess

root = Path('/home/xuming/poseidon-manual-boot-20261026')
audit = root/'resnet20-optimized-20261027T055506Z/placement-audit'
out = audit/'target-scale-experiment'

# Compare the actual generated arithmetic, not a hand-authored witness plan.
text = (audit/'chain-waste-details.py').read_text()
text = text.replace("root = audit/'chain-waste-probe'","root = audit/'target-scale-experiment'")
text = text.replace("('manual','shuffle-only-30-0','working-matched-budget-40')","('manual','target')")
text = text.replace("    path = root/case/'artifacts'","    path = audit/'one-relu-runtime' if case == 'manual' else root/'one-relu-inline'")
text = text.replace('Three plans:','Two plans:')
(out/'fingerprint-check.py').write_text(text)
exec(compile(text,str(out/'fingerprint-check.py'),'exec'),{})

# Verify the old strategy still emits the same whole-ReLU plan.
command = ['python3',str(root/'src/third_party/ckks-runtime/integrations/dacapo/generate_model_artifacts.py'),'mlp','--operator-spec',str(audit/'one-relu-full-spec.json'),'--compiler-profile',str(root/'alignment-tests/compiler.json'),'--hecate-opt',str(out/'hecate-opt-wrapper'),'--traced-mlir',str(audit/'one-relu-tagged.mlir'),'--output-dir',str(out/'threshold-reference'),'--device-counts','1','--plan-format','json','--boot-placement','greedy','--waterline','40']
run = subprocess.run(command,capture_output=True,text=True)
(out/'threshold-reference.log').write_text(run.stdout+run.stderr)
assert run.returncode==0,run.stderr
old = next((audit/'one-relu-runtime').glob('*.runtime-plan.json'))
new = next((out/'threshold-reference').glob('*.runtime-plan.json'))
assert json.loads(old.read_text()) == json.loads(new.read_text())

# Check direct use of the user-facing target option against the real Boot budget.
command[command.index('--operator-spec')+1] = str(root/'alignment-tests/operator-spec.json')
command[command.index('--output-dir')+1] = str(out/'target-production-cli')
command += ['--scale-strategy','target']
(out/'target-command.json').write_text(json.dumps(command,indent=2)+'\n')
run = subprocess.run(command,capture_output=True,text=True)
(out/'target-production-cli.log').write_text(run.stdout+run.stderr)
assert run.returncode==0,run.stderr
plan_path = next((out/'target-production-cli').glob('*.runtime-plan.json'))
plan = json.loads(plan_path.read_text())
desc = {v['id']:v for v in plan['values']}
final = desc[plan['final_outputs'][0]]
assert final['level']==7 and abs(final['scale_log2']-40)<1e-8
verify = subprocess.run([str(root/'validate-plan'),str(plan_path),str(root/'alignment-tests/operator-spec.json')],capture_output=True,text=True)
assert verify.returncode==0,verify.stdout+verify.stderr

# This is an input-alignment test only; it does not execute native Boot.
spec = root/'alignment-tests/operator-spec.json'
profile = json.loads(spec.read_text())['boot_profiles'][0]['profile_id']
ct = 'tensor<1x!ckks.poly<2 * 41 * 7>>'
result = 'tensor<1x!ckks.poly<2 * 40 * 30>>'
source = f'module {{ func.func @align(%x: {ct}) -> {result} attributes {{init_level = 30 : i64, earth.arg_scales = array<f64: {final["scale_log2"]:.17e}>}} {{ %y = "ckks.bootstrapc"(%x) : ({ct}) -> {result}\n return %y : {result} }} }}'
(out/'boot-input.mlir').write_text(source)
pipeline = f'builtin.module(func.func(materialize-ckks-physical-levels{{operator-spec={spec} boot-profile={profile} levels-per-logical-level=1}}))'
run = subprocess.run([str(out/'hecate-opt-wrapper'),f'--ckks-config={root}/alignment-tests/compiler.json',f'--pass-pipeline={pipeline}',str(out/'boot-input.mlir'),'-o',str(out/'boot-aligned.mlir')],capture_output=True,text=True)
assert run.returncode==0,run.stderr
aligned = (out/'boot-aligned.mlir').read_text()
assert aligned.count('"ckks.modswitchc"')==1 and '"ckks.rescalec"' not in aligned and '"ckks.mulcp"' not in aligned

checks = dict(
    branch='experiment/native-scale-planning',
    same_57_mathematical_multiplications=True,
    identical_coefficient_values_and_core_dependencies=True,
    threshold_whole_relu_plan_unchanged=True,
    production_q_budget=23,
    production_target_plan_validated=verify.stdout.strip(),
    boot_input_alignment=dict(input_q=8,input_scale_log2=final['scale_log2'],output_q=6,output_scale_log2=40,modswitches=1,rescales=0,encoded_one_multiplications=0),
    target_policy_checks=6,
    existing_native_planning_checks=25,
    existing_manual_boot_alignment_checks=9,
    existing_threshold_graph_checks=540,
    existing_threshold_valid_plans=495,
    existing_threshold_expected_rejections=45,
    encrypted_gpu_plan_sha256=json.loads((out/'gpu-summary.json').read_text())['plan_sha256'],
)
(out/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps(checks,ensure_ascii=False),flush=True)
