import json, subprocess, pathlib, re, time
root=pathlib.Path(__file__).resolve().parent
base=json.load(open('/home/xuming/poseidon-qwen-regional-20261101/repair-targets/regional-greedy/command.json'))
results=[]
for paired in (False,True):
 for iterations in (1,3,8):
  ct='tensor<1x!earth.ci<0 * 0>>';pt='tensor<1x!earth.pl<0 * 0>>'
  ops=[f'module {{ func.func @graph(%x: {ct}, %y: {ct}) -> {ct} {{']
  for n,v in [('quarter',.25),('threequarter',.75),('half',.5),('one',1.)]:
   ops.append(f'%{n} = "earth.constant"() {{value = dense<{v}> : tensor<8192xf64>, rms_var = {v} : f64}} : () -> {pt}')
  def op(n,kind,a,b,plain=False):
   ops.append(f'%{n} = "earth.{kind}"(%{a}, %{b}) : ({ct}, {pt if plain else ct}) -> {ct}')
   return n
  r='x'; y='y'
  for i in range(iterations):
   sq=op(f'sq{i}','mul',r,r)
   aff=op(f'a{i}','mul',r,'quarter',True)
   aff=op(f'b{i}','add',aff,'threequarter',True)
   r=op(f'r{i}','mul',sq,aff)
   if paired:
    z=op(f'c{i}','mul',r,'half',True)
    z=op(f'd{i}','add',z,'one',True)
    y=op(f'y{i}','mul',y,z)
  ops += [f'return %{y if paired else r} : {ct}', '}}']
  stem=f'{"paired" if paired else "residual"}-{iterations}'
  inp=root/(stem+'.mlir'); inp.write_text('\n'.join(ops)+'\n')
  for beam in (4,32):
   name=stem+f'-beam{beam}'; trace=root/(name+'.trace.json')
   cmd=base.copy();cmd[1]=str(inp)
   cmd=[f'--regional-beam-width={beam}' if a.startswith('--regional-beam-width=') else f'--regional-scale-report={trace}' if a.startswith('--regional-scale-report=') else f'--regional-failure-report={root/(name+".failure.json")}' if a.startswith('--regional-failure-report=') else a for a in cmd]
   start=time.monotonic();r0=subprocess.run(cmd,capture_output=True,text=True,timeout=45)
   (root/(name+'.log')).write_text(r0.stdout+r0.stderr)
   row=dict(name=name,seconds=time.monotonic()-start,returncode=r0.returncode)
   if trace.exists():
    tr=json.loads(trace.read_text());bs=[b for row in tr for b in row.get('boot',[])];row['boots']=len(bs);row['boot_states']=bs
   results.append(row);print(json.dumps(row),flush=True)
(root/'micro-results.json').write_text(json.dumps(results,indent=2)+'\n')
