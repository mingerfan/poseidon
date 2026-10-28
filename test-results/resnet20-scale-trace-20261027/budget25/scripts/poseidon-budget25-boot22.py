from pathlib import Path
import collections,hashlib,json,re,subprocess,time
root=Path('/home/xuming/poseidon-manual-boot-20261026')
audit=root/'resnet20-optimized-20261027T055506Z/placement-audit'
e=audit/'target-scale-experiment';out=e/'budget25/boot22';out.mkdir(exist_ok=True)
ct='tensor<1x!earth.ci<0 * 0>>'
s=(audit/'one-relu-tagged.mlir').read_text();body=s[s.index('\n')+1:s.index(' return ')]
second=re.sub(r'%(arg0|[0-9]+)',lambda m:'%refresh' if m[1]=='arg0' else '%second'+m[1],body).replace('loc("raw-','loc("second-raw-')
boot=f' %refresh = "earth.bootstrap"(%300) : ({ct}) -> {ct} loc("explicit-boot")\n'
base=json.loads((e/'budget25/one-relu-manual/command.json').read_text());base=[x for x in base if not x.startswith('--native-target-bootstrap-schedule=') and not x.startswith('--native-target-scale-goals=')]
for name,rest,output in [('relu-boot','', 'refresh'),('relu-boot-relu',second,'second300')]:
 case=out/name;case.mkdir(exist_ok=True);trace=case/'trace.mlir';trace.write_text(f'module {{ func.func @{name.replace("-","_")}(%arg0: {ct}) -> {ct} {{\n'+body+boot+rest+f' return %{output} : {ct}\n}}}}\n')
 c=base[:];c[c.index(str(audit/'one-relu-tagged.mlir'))]=str(trace);c[c.index('-o')+1]=str(case/'model.optimized.mlirbc')
 prep=subprocess.run(c+['--mlir-print-ir-before=greedy-bootstrap-placement'],capture_output=True,text=True,timeout=120)
 (case/'before-planner.log').write_text(prep.stderr)
 rows=[]
 for line in prep.stderr.splitlines():
  m=re.match(r'\s*%\w+ = "earth\.(\w+)"',line)
  if m and m[1] not in ('bootstrap','upscale','rescale','modswitch'):
   loc=re.search(r'((?:second-)?raw-\d+)',line);rows.append((len(rows)+2,loc[1] if loc else '',line))
 assert len(rows)>150,prep.stderr[-3000:]
 wanted={i:next(loc for j,loc,line in rows if i==j) for i in (43,85,157)}
 assert len(set(wanted.values()))==3 and all(wanted.values()),wanted
 goals={str(i):v for i,v in ((43,40.1),(85,44),(157,40))}
 if rest:
  for i,scale in ((43,40.1),(85,44),(157,40)):
   target='second-'+wanted[i];matches=[j for j,loc,line in rows if loc==target];assert len(matches)==1,(target,matches);goals[str(matches[0])]=scale
 (case/'goals.json').write_text(json.dumps(goals,indent=2)+'\n');c+=['--native-target-scale-goals='+str(case/'goals.json')]
 t=time.monotonic();r=subprocess.run(c,capture_output=True,text=True,timeout=120);(case/'compile.log').write_text(r.stdout+r.stderr);(case/'command.json').write_text(json.dumps(c,indent=2)+'\n');assert not r.returncode,r.stderr[-3000:]
 p=next(case.glob('*.runtime-plan.json'));plan=json.loads(p.read_text());d={v['id']:v for v in plan['values']};boots=[o for o in plan['execution'] if o.get('op')=='boot'];assert len(boots)==1,len(boots)
 check=subprocess.run([str(root/'validate-plan'),str(p),str(root/'alignment-tests/operator-spec.json')],capture_output=True,text=True);assert not check.returncode,check.stderr
 report=dict(name=name,command=c,compiler_sha256=(e/'binary.sha256').read_text().strip(),plan=str(p),plan_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),counts=dict(collections.Counter(o.get('op',o['kind']) for o in plan['execution'])),boots=[dict(before=d[o['inputs'][0]],after=d[o['output']]) for o in boots],output=d[plan['final_outputs'][0]],validation=check.stdout,seconds=time.monotonic()-t)
 (case/'result.json').write_text(json.dumps(report,indent=2)+'\n');print({k:v for k,v in report.items() if k!='command'},flush=True)
