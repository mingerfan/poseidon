#!/usr/bin/env python3
"""One existing-master SSH; bounded read-only terminal wait and public aggregate capture.
No remote writes, apps, retries, authentication changes, cleanup, or signals.
"""
import datetime
import hashlib
import json
import pathlib
import shlex
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
REMOTE = r'''
import csv, datetime, hashlib, json, pathlib, subprocess, time
R=pathlib.Path('/root/private_data/llama-manual-framework-node2-20261116-v1')
CASE='public-silu-baseline-v1'
UUID='GPU-499236d3-b46c-5d25-4a22-530d47ed5112'
OUT=R/'results'/CASE
start=time.monotonic()
checks=0
# At most 45 minutes total including the two bounded postflight NVML commands.
while True:
    checks+=1
    if (OUT/'terminal.json').is_file():
        status='terminal_observed'
        break
    if time.monotonic()-start>=2640:
        status='bounded_wait_expired'
        break
    time.sleep(min(30,2640-(time.monotonic()-start)))
files={}
allowed={
 'command.json':OUT/'command.json', 'pid-map.json':OUT/'pid-map.json',
 'stdout.log':OUT/'stdout.log', 'gpu.jsonl':OUT/'gpu.jsonl',
 'terminal.json':OUT/'terminal.json', 'SUMMARY.json':R/'reports'/CASE/'SUMMARY.json',
 'guard-launch.log':R/(CASE+'-guard-launch.log')}
for name,p in allowed.items():
    try:
        if p.is_symlink() or p.resolve()!=p: raise ValueError('not canonical nonsymlink')
        with p.open('rb') as f:
            b=f.read(8*1024*1024+1)
        if len(b)>8*1024*1024: raise ValueError('aggregate capture exceeds 8MiB bound')
        files[name]={'remote_path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'text':b.decode('utf8')}
    except FileNotFoundError:
        files[name]={'remote_path':str(p),'state':'absent'}
    except Exception as e:
        files[name]={'remote_path':str(p),'state':'capture_error','error':type(e).__name__+': '+str(e)}
def identity(pid):
    p=pathlib.Path('/proc')/str(pid)
    try:
        s=(p/'stat').read_text().rsplit(')',1)[1].split()
        return {'pid':pid,'pgid':int(s[2]),'sid':int(s[3]),'start_ticks':s[19],'uid':p.stat().st_uid,'state':s[0]}
    except FileNotFoundError: return None
    except Exception as e: return {'error':type(e).__name__+': '+str(e)}
expected={'pid':95575,'pgid':95575,'start_ticks':'1373031093'}
post={'timestamp':datetime.datetime.now().astimezone().isoformat(),'expected_gpu_uuid':UUID,
      'expected_guard_identity':expected,'guard_identity_once':identity(95575),
      'scope':'single read-only sample; no continuous exclusivity or lease-release proof'}
try:
    mapping=json.loads(files.get('pid-map.json',{}).get('text','null'))
    pid=mapping.get('namespace_pid') if isinstance(mapping,dict) else None
    if type(pid) is int and pid>0: post['registered_namespace_identity_once']=identity(pid)
except Exception as e:
    post['pid_map_parse_error']=type(e).__name__+': '+str(e)
for name,kind,fields in [('gpu','gpu','uuid,name,memory.total,memory.used,temperature.gpu'),
                         ('compute_apps','compute-apps','gpu_uuid,pid,process_name,used_gpu_memory')]:
    try:
        p=subprocess.run(['nvidia-smi','--query-'+kind+'='+fields,'--format=csv,noheader,nounits'],
                         text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=10)
        post[name]={'returncode':p.returncode,'rows':[[v.strip() for v in r] for r in csv.reader(p.stdout.splitlines()) if r],
                    'stderr':p.stderr}
    except Exception as e:
        post[name]={'error':type(e).__name__+': '+str(e)}
print(json.dumps({'schema':'node2-public-silu-readonly-capture-v1','case':CASE,'gpu_uuid':UUID,
                  'wait_status':status,'terminal_checks_30s_max_cadence':checks,
                  'monitor_wall_seconds':time.monotonic()-start,'files':files,'postflight':post},allow_nan=False))
'''

def main():
    # Deliberately do not print, save, or reveal the existing auth directory/socket.
    with pathlib.Path('/tmp/llama-a800-node2-active-auth').open() as f:
        auth_dir=f.readline().rstrip('\r\n')
    socket=pathlib.Path(auth_dir)/'socket'
    if not pathlib.Path(auth_dir).is_dir() or not socket.is_socket():
        raise RuntimeError('existing SSH control socket unavailable; no reconnect attempted')
    argv=['ssh','-T','-p','58222','-S',str(socket),'-oControlMaster=no','-oProxyCommand=false',
          '-oBatchMode=yes','hpuict.2121267849739771906@ssh.sd5ai.scnet.cn',
          '/opt/conda/bin/python3 -I -B -c '+shlex.quote(REMOTE)]
    begun=datetime.datetime.now().astimezone().isoformat()
    # One invocation only; timeout kills only this local observer, never remote jobs.
    try:
        p=subprocess.run(argv,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=2700)
        stdout,stderr,rc=p.stdout,p.stderr,p.returncode
    except subprocess.TimeoutExpired as e:
        stdout,stderr,rc=e.stdout or b'',e.stderr or b'',None
    (HERE/'capture.stdout.json').write_bytes(stdout)
    (HERE/'capture.stderr.log').write_bytes(stderr)
    record={'started_local':begun,'finished_local':datetime.datetime.now().astimezone().isoformat(),
            'ssh_invocations':1,'ssh_returncode':rc,'retry_attempted':False,
            'remote_writes':False,'remote_application_launches':False,
            'stdout_sha256':hashlib.sha256(stdout).hexdigest(),
            'stderr_sha256':hashlib.sha256(stderr).hexdigest()}
    (HERE/'COLLECTION.json').write_text(json.dumps(record,indent=2)+'\n')
    if rc!=0:
        print(json.dumps({'collection':'failed-frozen-no-retry','ssh_returncode':rc}))
        return 1
    v=json.loads(stdout)
    manifest={k:val for k,val in v.items() if k not in ('files','postflight')}
    manifest['files']={}
    for name,f in v['files'].items():
        manifest['files'][name]={k:value for k,value in f.items() if k!='text'}
        if 'text' in f:
            b=f['text'].encode('utf8')
            if len(b)!=f['bytes'] or hashlib.sha256(b).hexdigest()!=f['sha256']:
                raise RuntimeError('capture transport hash mismatch; frozen; no retry')
            (HERE/name).write_bytes(b)
    (HERE/'postflight.json').write_text(json.dumps(v['postflight'],indent=2)+'\n')
    (HERE/'CAPTURE-MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'collection':v['wait_status'],'ssh_returncode':rc,
                      'monitor_wall_seconds':v['monitor_wall_seconds'],
                      'file_states':{k:f.get('state','captured') for k,f in v['files'].items()}}))
    return 0

if __name__=='__main__':
    raise SystemExit(main())
