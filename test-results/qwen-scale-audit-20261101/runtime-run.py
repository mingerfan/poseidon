import json,subprocess,pathlib,time
root=pathlib.Path(__file__).resolve().parent
base=json.load(open('/home/xuming/poseidon-qwen-regional-20261101/repair-targets/regional-greedy/command.json'))
rows=[]
for beam in (4,32):
 name=f'paired-8-runtime-beam{beam}'
 cmd=base.copy();cmd[1]=str(root/'paired-8.mlir');cmd[-1]=str(root/(name+'.mlir'))
 cmd=[ '--dacapo' if a=='--dacapo-precompile' else '--runtime-plan-format=json' if a.startswith('--runtime-plan-format=') else '--runtime-plan-inline-payload-max-bytes=131072' if a.startswith('--runtime-plan-inline-payload-max-bytes=') else f'--regional-beam-width={beam}' if a.startswith('--regional-beam-width=') else f'--regional-scale-report={root/(name+".trace.json")}' if a.startswith('--regional-scale-report=') else f'--regional-failure-report={root/(name+".failure.json")}' if a.startswith('--regional-failure-report=') else a for a in cmd if a!='--emit-bytecode']
 start=time.monotonic();p=subprocess.run(cmd,capture_output=True,text=True,timeout=60)
 (root/(name+'.log')).write_text(p.stdout+p.stderr);(root/(name+'.command.json')).write_text(json.dumps(cmd,indent=2))
 row=dict(name=name,returncode=p.returncode,seconds=time.monotonic()-start);rows.append(row);print(row,flush=True)
(root/'runtime-results.json').write_text(json.dumps(rows,indent=2)+'\n')
