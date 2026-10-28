from pathlib import Path
import collections
import json
import math
import re
import subprocess

root = Path('/home/xuming/poseidon-manual-boot-20261026')
audit = root/'resnet20-optimized-20261027T055506Z/placement-audit'
experiment = audit/'target-scale-experiment'
out = experiment/'boot-tests'
out.mkdir(exist_ok=True)
ct = 'tensor<1x!earth.ci<0 * 0>>'
source = (audit/'one-relu-tagged.mlir').read_text()
body = source[source.index('\n')+1:source.index(' return ')]
renamed = re.sub(r'%(arg0|[0-9]+)', lambda m: '%refresh' if m[1] == 'arg0' else '%second'+m[1], body)
boot = f' %refresh = "earth.bootstrap"(%300) : ({ct}) -> {ct} loc("explicit-boot")\n'
for name, rest, output in [('relu-boot', '', 'refresh'),('relu-boot-relu', renamed, 'second300')]:
    trace = out/(name+'.mlir')
    trace.write_text(f'module {{ func.func @{name.replace("-","_")}(%arg0: {ct}) -> {ct} {{\n'+body+boot+rest+f' return %{output} : {ct}\n}}}}\n')
    command = json.loads((experiment/'target-command.json').read_text())
    command[command.index('--traced-mlir')+1] = str(trace)
    command[command.index('--output-dir')+1] = str(out/name)
    command += ['--inline-payload-max-bytes','1048576']
    (out/(name+'-command.json')).write_text(json.dumps(command,indent=2)+'\n')
    run = subprocess.run(command,capture_output=True,text=True,timeout=120)
    (out/(name+'.log')).write_text(run.stdout+run.stderr)
    assert run.returncode == 0, run.stderr[-8000:]
    path = next((out/name).glob('*.runtime-plan.json'))
    check = subprocess.run([str(root/'validate-plan'),str(path),str(root/'alignment-tests/operator-spec.json')],capture_output=True,text=True)
    assert check.returncode == 0, check.stdout+check.stderr
    plan = json.loads(path.read_text())
    desc = {v['id']:v for v in plan['values']}
    ops = plan['initialization']+plan['execution']+plan['finalization']
    boots = []
    counts = collections.Counter()
    for op in ops:
        kind = op.get('op',op['kind'])
        counts[kind] += 1
        if kind == 'boot':
            before, after = desc[op['inputs'][0]], desc[op['output']]
            assert before['level']==5 and abs(before['scale_log2']-40)<1e-8
            assert after['level']==30 and after['scale_log2']==40
            boots.append(dict(before=before,after=after))
    assert len(boots)==1
    report = dict(name=name,plan=str(path),validation=check.stdout.strip(),counts=dict(counts),boots=boots,output=desc[plan['final_outputs'][0]])
    (out/(name+'-summary.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
