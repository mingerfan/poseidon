import argparse, hashlib, json, subprocess, time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('model',choices=['qwen-before','qwen-after','resnet-after']);a=p.parse_args()
root=Path(__file__).resolve().parent
old=root.parent/'linear-reduction-fix'
if a.model.startswith('qwen'):
    cmd=json.load(open(root.parent/'repair-targets/regional-greedy/command.json'))
else:
    cmd=json.load(open(old/'resnet-default32-command.json'))
out=root/a.model;out.mkdir(exist_ok=False)
cmd[0]=str((old if a.model=='qwen-before' else root)/'hecate-opt')
for i,v in enumerate(cmd):
    if v.startswith(('--regional-scale-report=','--regional-failure-report=','--regional-search-report=')):
        cmd[i]=v.split('=')[0]+'='+str(out/Path(v.split('=',1)[1]).name)
    if v.startswith(str(old/'resnet-default32/')):
        cmd[i]=str(out/Path(v).name)
if '--compilation-progress' not in cmd:cmd.append('--compilation-progress')
(out/'command.json').write_text(json.dumps(cmd,indent=2))
elf=Path(cmd[0]).parent/'hecate-opt.real'
record=dict(model=a.model,compiler_sha256=hashlib.sha256(elf.read_bytes()).hexdigest(),started=time.time())
start=time.monotonic()
with (out/'compile.log').open('w') as log:
    process=subprocess.Popen(['/usr/bin/time','-v','-o',str(out/'time.log')]+cmd,stdout=log,stderr=subprocess.STDOUT)
    (out/'pid').write_text(str(process.pid))
    try:code=process.wait(timeout=1200)
    except subprocess.TimeoutExpired:
        subprocess.run(['pkill','-TERM','-P',str(process.pid)],check=False)
        process.terminate();code=process.wait();record['timed_out']=True
record.update(returncode=code,seconds=time.monotonic()-start)
trace=out/'scale-trace.json'
if code==0 and trace.exists():
    rows=json.loads(trace.read_text());boots=[b for row in rows for b in row.get('boot',[])]
    record.update(trace_boots=len(boots),regions=len(rows),last_state=rows[-1])
(out/'result.json').write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)
raise SystemExit(code)
