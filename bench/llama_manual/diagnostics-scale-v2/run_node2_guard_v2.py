#!/usr/bin/env python3
"""Pinned node2 single-process CUDA admission. Trusted frozen root/toolchain assumed.
No credential management, retries, foreign signals, system changes or profiling.
"""
import argparse, csv, datetime, fcntl, hashlib, json, os, pathlib, re, signal, subprocess, sys, time
ROOT=pathlib.Path('/root/private_data/llama-manual-framework-node2-20261116-v1')
UUID='GPU-499236d3-b46c-5d25-4a22-530d47ed5112'
PREFIX='/root/private_data/FHE_manual_20261114/deps/prefix/lib'
CUDART='/usr/local/cuda-11.8/targets/x86_64-linux/lib/libcudart.so.11.0'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def pairs(xs):
 d={}
 for k,v in xs:
  if k in d:raise ValueError('duplicate JSON key')
  d[k]=v
 return d
def regular(p):
 if p.resolve()!=p or p.is_symlink() or not p.is_file() or p.stat().st_uid!=os.getuid():raise RuntimeError('canonical owned file required')
 return p
def contained(name):
 p=pathlib.Path(name)
 if p.is_absolute() or '..' in p.parts:raise RuntimeError('contained relative role required')
 return ROOT/p

def smi(fields,kind='gpu'):
 return subprocess.check_output(['nvidia-smi','--query-'+kind+'='+fields,'--format=csv,noheader,nounits'],text=True,timeout=10).strip()
def active():
 return [dict(zip(['uuid','pid','name','MiB'],[x.strip() for x in row])) for row in csv.reader(smi('gpu_uuid,pid,process_name,used_gpu_memory','compute-apps').splitlines()) if row and row[0].strip()==UUID]
def ident(pid):
 p=pathlib.Path('/proc')/str(pid)
 try:
  s=(p/'stat').read_text().rsplit(')',1)[1].split()
  return {'pid':pid,'pgid':int(s[2]),'sid':int(s[3]),'start_ticks':s[19],'uid':p.stat().st_uid,'state':s[0]}
 except FileNotFoundError:return None

def read_registered_map(path,identity,supervisor):
 if not path.is_file():return None
 m=json.loads(path.read_text(),object_pairs_hook=pairs)
 if identity is None or m['namespace_pid']!=identity['pid'] or m['pgid']!=identity['pgid'] or m['start_ticks']!=identity['start_ticks'] or m['supervisor_pid']!=supervisor or m['gpu_uuid']!=UUID or type(m['nvml_host_pid']) is not int or m['nvml_host_pid']<=0:raise RuntimeError('registered PID ownership mismatch')
 return m

def main():
 ap=argparse.ArgumentParser();ap.add_argument('authorization');ap.add_argument('authorization_sha256');args=ap.parse_args()
 if pathlib.Path(__file__).resolve()!=ROOT/'tools/run_node2_guard_v2.py':raise RuntimeError('execute actual root guard')
 auth=regular(pathlib.Path(args.authorization))
 if auth.parent!=ROOT/'authorizations' or sha(auth)!=args.authorization_sha256:raise RuntimeError('Main authorization hash/path mismatch')
 a=json.loads(auth.read_text(),object_pairs_hook=pairs)
 expected={'schema','root','gpu_uuid','case','GPU_execution_authorized','command','artifacts','source_manifest','leases','environment','timeout_seconds'}
 if set(a)!=expected or a['schema']!='LLAMA_NODE2_GPU_V1' or a['root']!=str(ROOT) or a['gpu_uuid']!=UUID or a['GPU_execution_authorized'] is not True:raise RuntimeError('closed Main GPU authorization required')
 if not re.fullmatch(r'(public-silu|manual-(real|paired)-b(01|32))-(baseline|cpu-reuse)-v[1-9][0-9]*',a['case']):raise RuntimeError('closed case')
 if not isinstance(a['timeout_seconds'],int) or not 60<=a['timeout_seconds']<=18000:raise RuntimeError('deadline')
 required={'tools/run_node2_guard_v2.py','tools/cuda_registered_exec.py','build/libposeidon_gpu.a','build/libposeidon_shared.so'}
 if not required<=set(a['artifacts']):raise RuntimeError('missing mandatory pins')
 for n,h in a['artifacts'].items():
  if not re.fullmatch('[0-9a-f]{64}',h) or sha(regular(contained(n)))!=h:raise RuntimeError('artifact hash mismatch')
 cmd=a['command'];binary=ROOT/('diagnostics-scale-v2/llama31_public_paired_silu_metadata' if a['case']=='public-silu-baseline-v2' else ('build/bin/llama31_public_paired_silu' if a['case'].startswith('public-silu') else 'build/bin/llama31_poseidon_manual'))
 if not isinstance(cmd,list) or not all(isinstance(x,str) for x in cmd) or cmd[0]!=str(binary) or str(binary.relative_to(ROOT)) not in a['artifacts']:raise RuntimeError('closed binary')
 if a['case'].startswith('public-silu'):
  if len(cmd)!=5 or cmd[1]!='--public-silu-pair-v1':raise RuntimeError('closed SiLU command')
  manifest=cmd[3];fixture=cmd[2];report=cmd[4]
 else:
  if len(cmd)!=12 or cmd[1] not in ('--run','--run-v2') or cmd[2] not in ('manual-highprecision-real-v1','manual-highprecision-paired-silu-candidate-v1') or cmd[8] not in ('1','32') or cmd[9]!='38':raise RuntimeError('closed model command')
  manifest=cmd[10];fixture=cmd[3];report=cmd[11]
 for p in [manifest,fixture]:
  regular(pathlib.Path(p));n=str(pathlib.Path(p).relative_to(ROOT))
  if n not in a['artifacts']:raise RuntimeError('command input not pinned')
 report=pathlib.Path(report)
 if report!=ROOT/'reports'/a['case']/'SUMMARY.json' or report.parent.resolve()!=report.parent or not report.parent.is_dir() or os.path.lexists(report):raise RuntimeError('exclusive separate report required')
 sources=json.loads(regular(contained(a['source_manifest'])).read_text(),object_pairs_hook=pairs)
 if a['source_manifest'] not in a['artifacts']:raise RuntimeError('source inventory unpinned')
 for n,h in sources.items():
  if sha(regular(contained('source/'+n)))!=h:raise RuntimeError('source mutation')
 for k in ('LD_PRELOAD','LD_AUDIT'):
  if os.environ.get(k):raise RuntimeError('injected loader')
 allowed={'OMP_NUM_THREADS':{'1','6'},'OPENBLAS_NUM_THREADS':{'1'},'MKL_NUM_THREADS':{'1'},'POSEIDON_GPU_QP_PREPARE':{'optimized','reference'},'POSEIDON_GPU_QP_PREPARE_THREADS':{'1','4','6'},'POSEIDON_GPU_QP_VERIFY_REFERENCE':{'0'},'LLAMA_MANUAL_IMPLEMENTATION_PRESET':{'baseline','cpu-reuse'}}
 if not isinstance(a['environment'],dict) or any(k not in allowed or v not in allowed[k] for k,v in a['environment'].items()):raise RuntimeError('closed environment')
 locks=[]
 for name,v in a['leases'].items():
  p=regular(pathlib.Path(name));fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW|os.O_CLOEXEC);st=os.fstat(fd)
  if st.st_ino!=v['inode'] or st.st_mode!=v['mode'] or st.st_uid!=0:raise RuntimeError('shared lease identity')
  fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(fd)
 if len(locks)!=8:raise RuntimeError('all existing cooperative node2 leases required')
 claim=ROOT/('claim-'+a['case']);fd=os.open(claim,os.O_CREAT|os.O_EXCL|os.O_RDWR|os.O_NOFOLLOW,0o600);fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(fd)
 rows=list(csv.reader(smi('uuid,name,memory.total,memory.used,temperature.gpu').splitlines()))
 if len(rows)!=1:raise RuntimeError('single GPU only')
 row=[x.strip() for x in rows[0]]
 if row[0]!=UUID or not row[1].startswith('NVIDIA A800') or int(row[2])<80000 or int(row[3])!=0 or int(row[4])>55 or active():raise RuntimeError('fresh idle node2 admission failed')
 out=ROOT/'results'/a['case']
 if out.parent.resolve()!=out.parent or os.path.lexists(out):raise RuntimeError('exclusive owned result directory')
 out.mkdir(mode=0o700)
 env=os.environ.copy()
 for k in list(env):
  if k.startswith(('LLAMA_','POSEIDON_','CERIUM_')) or k in ('PYTHONPATH','PYTHONHOME','LD_PRELOAD','LD_AUDIT'):env.pop(k,None)
 env.update(a['environment']);env.update(CUDA_VISIBLE_DEVICES=UUID,LD_LIBRARY_PATH=PREFIX+':/usr/local/cuda-11.8/lib64:/usr/local/nvidia/lib64',CUDA_CACHE_PATH=str(out/'cuda-cache'),XDG_CACHE_HOME=str(out/'cache'))
 wrapped=['taskset','-c','0-6',sys.executable,'-I','-B',str(ROOT/'tools/cuda_registered_exec.py'),'--supervisor-pid',str(os.getpid()),'--map',str(out/'pid-map.json'),'--cudart',CUDART,'--',*cmd]
 record={'start':datetime.datetime.now().astimezone().isoformat(),'initial_gpu':row,'authorization_sha256':args.authorization_sha256,'command':cmd,'env':{k:env[k] for k in ['CUDA_VISIBLE_DEVICES','LD_LIBRARY_PATH',*a['environment']]},'guard_pid':os.getpid(),'owned_shared_leases':a['leases']}
 (out/'command.json').write_text(json.dumps(record,indent=2)+'\n');requested=[]
 def request(s,_):requested.append(s)
 for s in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(s,request)
 p=None;identity=None;mapping=None;reason=None;start=time.monotonic()
 def stop():
  if p is None:return
  now=ident(p.pid)
  if now and (identity is None or any(now[k]!=identity[k] for k in ['pgid','sid','start_ticks','uid'])):raise RuntimeError('refuse unproved group cleanup')
  if now:
   os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(timeout=15)
   except subprocess.TimeoutExpired:
    now=ident(p.pid)
    if not now or any(now[k]!=identity[k] for k in ['pgid','sid','start_ticks','uid']):raise RuntimeError('owned PID changed during stop')
    os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=5)
 with (out/'stdout.log').open('x') as log,(out/'gpu.jsonl').open('x') as audit:
  try:
   p=subprocess.Popen(wrapped,cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=tuple(locks));identity=ident(p.pid)
   if not identity or identity['pgid']!=p.pid or identity['sid']!=p.pid or identity['uid']!=os.getuid():raise RuntimeError('owned CUDA process identity missing')
   record['child_identity']=identity
   while p.poll() is None:
    if requested:reason='guard received signal';stop();break
    if mapping is None:mapping=read_registered_map(out/'pid-map.json',identity,os.getpid())
    processes=active();sample=[x.strip() for x in next(csv.reader(smi('temperature.gpu,memory.used,utilization.gpu,power.draw').splitlines()))]
    foreign=[x for x in processes if mapping is not None and int(x['pid'])!=mapping['nvml_host_pid']]
    audit.write(json.dumps({'elapsed_s':time.monotonic()-start,'registered':mapping,'processes':processes,'foreign':foreign,'sample':sample})+'\n');audit.flush()
    if foreign or (mapping is None and len(processes)>1):reason='foreign GPU process';stop();break
    if int(sample[0])>75:reason='temperature';stop();break
    if time.monotonic()-start>a['timeout_seconds']:reason='deadline';stop();break
    if mapping is None and time.monotonic()-start>45:reason='registration timeout';stop();break
    time.sleep(2)
   rc=p.wait()
   # The target can exit after publishing the map between monitor samples.
   if mapping is None:mapping=read_registered_map(out/'pid-map.json',identity,os.getpid())
   if rc in (0,2) and mapping is None:raise RuntimeError('completed target without registration')
  except BaseException as e:
   reason=type(e).__name__+': '+str(e)
   try:stop()
   except BaseException as cleanup_error:
    record['cleanup_error']=type(cleanup_error).__name__+': '+str(cleanup_error)
   rc=p.poll() if p else 1
   if p and rc is None:record['owned_process_may_still_be_live']=True
 record.update(returncode=rc,abort_reason=reason,requested_signals=requested,registered=mapping,wall_seconds=time.monotonic()-start,end=datetime.datetime.now().astimezone().isoformat(),classification='numerical_negative' if rc==2 and reason is None else ('completed' if rc==0 and reason is None else 'structural_or_resource_failure'))
 (out/'terminal.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record));return rc if reason is None and rc in (0,2) else 1
if __name__=='__main__':raise SystemExit(main())
