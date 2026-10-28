from pathlib import Path
import collections,json,resource,subprocess,sys,time
root=Path('/home/xuming/poseidon-manual-boot-20261026');e=root/'resnet20-optimized-20261027T055506Z/placement-audit/target-scale-experiment'
name=sys.argv[1];out=e/'budget25'/name/'full';out.mkdir(parents=True,exist_ok=True)
c=json.loads((e/'budget25/full/threshold/run.json').read_text())['command'];c[0]=str(e/'budget25'/name/'hecate-opt-wrapper');c=[x.replace('--boot-placement=greedy','--boot-placement=optimized') for x in c];c[c.index('-o')+1]=str(out/'resnet20.optimized.mlirbc')
r=dict(name=name,command=c,state='running');(out/'run.json').write_text(json.dumps(r,indent=2)+'\n');t=time.monotonic()
with (out/'compile.log').open('w') as log:
 p=subprocess.Popen(c,stdout=log,stderr=subprocess.STDOUT);r['pid']=p.pid;(out/'run.json').write_text(json.dumps(r,indent=2)+'\n');print(name,p.pid,flush=True)
 try:code=p.wait(timeout=1800)
 except subprocess.TimeoutExpired:p.terminate();code=p.wait();r['timed_out']=True
r.update(returncode=code,seconds=time.monotonic()-t,state='passed' if code==0 else 'failed',peak_child_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
if code==0:
 path=next(out.glob('*.runtime-plan.json'));plan=json.loads(path.read_text());desc={v['id']:v for v in plan['values']};boots=[op for op in plan['execution'] if op.get('op')=='boot'];r.update(boots=len(boots),counts=dict(collections.Counter(o.get('op',o['kind']) for o in plan['execution'])))
 assert all(desc[o['inputs'][0]]['level']==5 and abs(desc[o['inputs'][0]]['scale_log2']-40)<1e-8 for o in boots)
 v=subprocess.run([str(root/'validate-plan'),str(path),str(root/'alignment-tests/operator-spec.json')],capture_output=True,text=True);r['validation']=v.stdout+v.stderr;r['validation_code']=v.returncode
(out/'run.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='command'}),flush=True)
if code:print((out/'compile.log').read_text()[-2400:],flush=True)
