import json, subprocess, pathlib, time
root=pathlib.Path(__file__).resolve().parent
base=json.load(open('/home/xuming/poseidon-qwen-regional-20261101/repair-targets/regional-greedy/command.json'))
base[0]=str(root/'hecate-opt')
results=[]
ct='tensor<1x!earth.ci<0 * 0>>';pt='tensor<1x!earth.pl<0 * 0>>'
for terms in (64,256,1024):
 ops=[f'module {{ func.func @graph(%x: {ct}) -> ({ct}, {ct}) {{']
 for j in range(32):
  ops.append(f'%rot{j} = "earth.rotate"(%x) <{{offset = array<i64: {j+1}>}}> : ({ct}) -> {ct}')
 outputs=[]
 for branch in range(2):
  result=None
  for j in range(terms):
   coef=(1+branch*.2+j/(terms*10))/terms
   n=f'{branch}_{j}'
   ops.append(f'%p{n} = "earth.constant"() {{value = dense<{coef}> : tensor<8192xf64>, rms_var = {coef} : f64}} : () -> {pt}')
   ops.append(f'%m{n} = "earth.mul"(%rot{j%32}, %p{n}) : ({ct}, {pt}) -> {ct}')
   if result is None: result=f'm{n}'
   else:
    ops.append(f'%a{n} = "earth.add"(%{result}, %m{n}) : ({ct}, {ct}) -> {ct}')
    result=f'a{n}'
  outputs.append(result)
 ops += [f'return %{outputs[0]}, %{outputs[1]} : {ct}, {ct}', '}}']
 inp=root/f'linear-{terms}.mlir'; inp.write_text('\n'.join(ops)+'\n')
 for beam in (4,32):
  name=f'linear-{terms}-beam{beam}';trace=root/(name+'.trace.json')
  cmd=base.copy();cmd[1]=str(inp)
  cmd=[f'--regional-beam-width={beam}' if a.startswith('--regional-beam-width=') else f'--regional-scale-report={trace}' if a.startswith('--regional-scale-report=') else f'--regional-failure-report={root/(name+".failure.json")}' if a.startswith('--regional-failure-report=') else a for a in cmd]
  start=time.monotonic();p=subprocess.run(cmd,capture_output=True,text=True,timeout=45)
  (root/(name+'.log')).write_text(p.stdout+p.stderr)
  row=dict(name=name,seconds=time.monotonic()-start,returncode=p.returncode)
  if trace.exists():
   tr=json.loads(trace.read_text());row['boots']=sum(len(r.get('boot',[])) for r in tr);row['tail']=tr[-3:]
  results.append(row);print(json.dumps(row),flush=True)
(root/'linear-results.json').write_text(json.dumps(results,indent=2)+'\n')
