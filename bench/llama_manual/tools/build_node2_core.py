#!/usr/bin/env python3
"""One pinned node2 source build. CPU only; no application or CUDA execution."""
import hashlib, json, os, pathlib, subprocess, sys, time, fcntl, signal, ctypes
ROOT=pathlib.Path('/root/private_data/llama-manual-framework-node2-20261116-v1')
UUID='GPU-499236d3-b46c-5d25-4a22-530d47ed5112'
PREFIX=pathlib.Path('/root/private_data/FHE_manual_20261114/deps/prefix')
CMAKE='/root/private_data/FHE_manual_20261114/deps/cmake-wheel/cmake/data/bin/cmake'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def regular(p):
 if p.is_symlink() or p.resolve()!=p or not p.is_file():raise RuntimeError('noncanonical file')
 return p

def main():
 if pathlib.Path(__file__).resolve()!=ROOT/'tools/build_node2_core.py':raise RuntimeError('execute actual pinned root builder')
 authorization=json.loads(regular(ROOT/'build-authorization.json').read_text())
 if authorization['GPU_execution_authorized'] is not False:raise RuntimeError('CPU authorization only')
 if authorization['root']!=str(ROOT) or authorization['gpu_uuid']!=UUID:raise RuntimeError('closed root/UUID')
 for name,h in authorization['artifacts'].items():
  p=ROOT/name
  if '..' in pathlib.Path(name).parts or pathlib.Path(name).is_absolute() or sha(regular(p))!=h:raise RuntimeError('artifact mismatch')
 sources=json.loads((ROOT/'source-sha256.json').read_text())
 if len(sources)!=authorization['source_count']:raise RuntimeError('source count')
 def verify():
  for name,h in sources.items():
   if pathlib.Path(name).is_absolute() or '..' in pathlib.Path(name).parts or sha(regular(ROOT/'source'/name))!=h:raise RuntimeError('source mutation')
 verify();locks=[]
 for name,expected in authorization['leases'].items():
  p=pathlib.Path(name);regular(p);fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW|os.O_CLOEXEC);st=os.fstat(fd)
  if st.st_ino!=expected['inode'] or st.st_uid!=0 or st.st_mode!=expected['mode']:raise RuntimeError('lease identity')
  fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(fd)
 lock=ROOT/'host-build.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_RDWR|os.O_NOFOLLOW,0o600);fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(fd)
 q=subprocess.run(['nvidia-smi','--query-gpu=uuid,memory.used,temperature.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=10)
 row=[x.strip() for x in q.stdout.strip().split(',')]
 if row[0]!=UUID or int(row[1])!=0 or int(row[2])>55:raise RuntimeError('CPU build admission GPU not idle')
 p=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=10)
 if p.stdout.strip():raise RuntimeError('foreign GPU process')
 if os.environ.get('LD_PRELOAD') or os.environ.get('LD_AUDIT'):raise RuntimeError('injected runtime')
 helper=ROOT/'tools/owned_cpu_group.py';namespace={};exec(compile(helper.read_bytes(),str(helper),'exec'),namespace)
 OwnedCPUGroup=namespace['OwnedCPUGroup'];os.sched_setaffinity(0,{6})
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CMAKE_BUILD_PARALLEL_LEVEL='1',GMP_ROOT=str(PREFIX),CPM_LOCAL_PACKAGES_ONLY='ON')
 env['LD_LIBRARY_PATH']=str(PREFIX/'lib')+':/usr/local/cuda/lib64:/usr/local/nvidia/lib64'
 result={'scope':'fresh framework core source CPU/CUDA compilation, no GPU or app execution','commands':[],'initial_gpu':row,'source_count':len(sources)}
 deadline=time.monotonic()+900
 caught=[]
 def stop(sig,_):caught.append(sig);raise RuntimeError('CPU build interrupted')
 for s in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(s,stop)
 wrap="import os,sys,ctypes,signal; p=int(sys.argv[1]); k=ctypes.CDLL(None,use_errno=True); assert os.getppid()==p; assert k.prctl(1,int(signal.SIGTERM),0,0,0)==0; assert os.getppid()==p; assert sys.stdin.buffer.read(1)==b'1'; os.execvpe(sys.argv[2],sys.argv[2:],os.environ)"
 def run(name,argv):
  row={'stage':name,'argv':argv};result['commands'].append(row);start=time.monotonic();owner=None;p=None
  with (ROOT/(name+'.stdout.log')).open('xb') as out,(ROOT/(name+'.stderr.log')).open('xb') as err:
   try:
    p=subprocess.Popen([sys.executable,'-c',wrap,str(os.getpid()),*argv],stdin=subprocess.PIPE,stdout=out,stderr=err,env=env,cwd=ROOT,start_new_session=True,pass_fds=tuple(locks))
    owner=OwnedCPUGroup(p);row['identity']=owner.leader;p.stdin.write(b'1');p.stdin.close();rc=p.wait(timeout=max(1,deadline-time.monotonic()));row['returncode']=rc
   finally:
    if owner:row['cleanup']=owner.cleanup()
    elif p:p.wait(timeout=3)
    row['seconds']=time.monotonic()-start
  if row['returncode']!=0:raise RuntimeError(name+' failed; stop, no retry')
 try:
  run('configure-core',[CMAKE,'-S',str(ROOT/'source'),'-B',str(ROOT/'build'),'-DCMAKE_BUILD_TYPE=Release','-DCMAKE_CUDA_ARCHITECTURES=80','-DPOSEIDON_BUILD_GPU=ON','-DPOSEIDON_BUILD_LLAMA_MANUAL=OFF','-DPOSEIDON_BUILD_CKKS_RUNTIME_API=OFF','-DPOSEIDON_BUILD_EXAMPLES=OFF','-DPOSEIDON_BUILD_DEPS=OFF','-DPOSEIDON_USE_ZLIB=OFF','-DPOSEIDON_USE_ZSTD=OFF','-DPOSEIDON_USE_MSGSL=OFF','-DPOSEIDON_USE_OPENMP=OFF','-DFETCHCONTENT_FULLY_DISCONNECTED=ON','-DCPM_LOCAL_PACKAGES_ONLY=ON','-DCMAKE_PREFIX_PATH='+str(PREFIX),'-DGMP_ROOT='+str(PREFIX),'-DCMAKE_EXPORT_COMPILE_COMMANDS=ON','-DCMAKE_CXX_FLAGS=-fopenmp -fno-fast-math'])
  run('compile-core',[CMAKE,'--build',str(ROOT/'build'),'--target','poseidon_gpu','--parallel','1'])
  verify();result.update(status='passed',archive_sha256=sha(ROOT/'build/libposeidon_gpu.a'),CPU_library_sha256=sha(ROOT/'build/libposeidon_shared.so'))
 except BaseException as e:result.update(status='failed_stop',error=type(e).__name__+': '+str(e));raise
 finally:
  result['caught_signals']=caught;(ROOT/'core-build-result.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
