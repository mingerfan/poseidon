from pathlib import Path
from collections import Counter
import argparse,hashlib,json,resource,subprocess,time
root=Path('/home/xuming/poseidon-manual-boot-20261026');e=root/'resnet20-optimized-20261027T055506Z/placement-audit/target-scale-experiment';parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,required=True);args=parser.parse_args();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
c=json.loads((e/'auto25-unhinted-20261028T011542Z/q31/result.json').read_text())['command'];c=[a for a in c if a!='--native-target-scale-planning'];c[0]=str(e/'hecate-opt-wrapper');c[c.index('--boot-placement=greedy')]='--boot-placement=optimized';c[c.index('-o')+1]=str(out/'resnet20.optimized.mlirbc');c+=['--mlir-timing','--mlir-timing-display=list'];assert not any('scale-goals' in a or 'bootstrap-schedule' in a for a in c)
r=dict(command=c,state_policy='one-cheapest-plan-per-cut',scale_strategy='threshold',initial_q=31,boot_q_budget=25,compiler_sha256=(e/'binary.sha256').read_text().strip());t=time.monotonic()
with (out/'compile.log').open('w') as log:
 p=subprocess.Popen(c,stdout=log,stderr=subprocess.STDOUT);r['pid']=p.pid;(out/'result.json').write_text(json.dumps(r,indent=2)+'\n');print('full optimized PID',p.pid,flush=True)
 try:code=p.wait(timeout=600)
 except subprocess.TimeoutExpired:p.kill();p.wait();raise
r.update(returncode=code,seconds=time.monotonic()-t,peak_child_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
if not code:
 path=next(out.glob('*.runtime-plan.json'));plan=json.loads(path.read_text());d={v['id']:v for v in plan['values']};boots=[s for s in plan['execution'] if s.get('op')=='boot'];assert all(d[s['inputs'][0]]['level']==5 and abs(d[s['inputs'][0]]['scale_log2']-40)<1e-8 and d[s['output']]['level']==30 and d[s['output']]['scale_log2']==40 for s in boots)
 v=subprocess.run([str(root/'validate-plan'),str(path),str(root/'alignment-tests/operator-spec.json')],capture_output=True,text=True);r.update(boots=len(boots),counts=dict(Counter(s.get('op',s['kind']) for s in plan['execution'])),validation=v.stdout+v.stderr,validation_code=v.returncode,plan_sha256=hashlib.sha256(path.read_bytes()).hexdigest());assert not v.returncode,v.stderr
(out/'result.json').write_text(json.dumps(r,indent=2)+'\n');print({k:v for k,v in r.items() if k!='command'},flush=True)
if code:print((out/'compile.log').read_text()[-2500:],flush=True)
