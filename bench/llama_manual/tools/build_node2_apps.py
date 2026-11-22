#!/usr/bin/env python3
"""One app-only target build/CPU closure; never runs an admitted GPU form."""
import os,sys,pathlib,json,hashlib,subprocess,time,fcntl,signal
ROOT=pathlib.Path('/root/private_data/llama-manual-framework-node2-20261116-v1')
PREFIX='/root/private_data/FHE_manual_20261114/deps/prefix'
CMAKE='/root/private_data/FHE_manual_20261114/deps/cmake-wheel/cmake/data/bin/cmake'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def regular(p):
 if p.resolve()!=p or p.is_symlink() or not p.is_file():raise RuntimeError('canonical file required')
 return p

def main():
 if pathlib.Path(__file__).resolve()!=ROOT/'tools/build_node2_apps.py':raise RuntimeError('actual root builder required')
 a=json.loads(regular(ROOT/'app-build-authorization.json').read_text())
 if a['GPU_execution_authorized'] is not False or a['root']!=str(ROOT):raise RuntimeError('CPU authorization required')
 for n,h in a['artifacts'].items():
  p=pathlib.Path(n)
  if p.is_absolute() or '..' in p.parts or sha(regular(ROOT/p))!=h:raise RuntimeError('artifact mismatch')
 result=json.loads(regular(ROOT/'core-build-result.json').read_text())
 if result['status']!='passed' or result['archive_sha256']!=sha(ROOT/'build/libposeidon_gpu.a') or result['CPU_library_sha256']!=sha(ROOT/'build/libposeidon_shared.so'):raise RuntimeError('passed unchanged core required')
 sources=json.loads((ROOT/a['source_manifest']).read_text());before_core=(sha(ROOT/'build/libposeidon_gpu.a'),sha(ROOT/'build/libposeidon_shared.so'))
 def verify():
  for n,h in sources.items():
   p=pathlib.Path(n)
   if p.is_absolute() or '..' in p.parts or sha(regular(ROOT/'source'/p))!=h:raise RuntimeError('source mutation')
 verify();fds=[]
 for n,v in a['leases'].items():
  p=regular(pathlib.Path(n));fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW|os.O_CLOEXEC);s=os.fstat(fd)
  if s.st_ino!=v['inode'] or s.st_mode!=v['mode'] or s.st_uid!=0:raise RuntimeError('shared lease changed')
  fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);fds.append(fd)
 fd=os.open(ROOT/'app-build.lock',os.O_CREAT|os.O_EXCL|os.O_RDWR|os.O_NOFOLLOW,0o600);fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);fds.append(fd)
 uuid=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used,temperature.gpu','--format=csv,noheader,nounits'],text=True,timeout=10).strip().split(',')
 if uuid[0].strip()!='GPU-499236d3-b46c-5d25-4a22-530d47ed5112' or int(uuid[1])!=0 or int(uuid[2])>55 or subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=10).strip():raise RuntimeError('fresh idle CPU admission failed')
 if os.environ.get('LD_PRELOAD') or os.environ.get('LD_AUDIT'):raise RuntimeError('injected runtime')
 ns={};helper=ROOT/'tools/owned_cpu_group.py';exec(compile(helper.read_bytes(),str(helper),'exec'),ns);Owned=ns['OwnedCPUGroup'];os.sched_setaffinity(0,{6})
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',LD_LIBRARY_PATH=PREFIX+'/lib:/usr/local/cuda-11.8/lib64:/usr/local/nvidia/lib64');deadline=time.monotonic()+900
 record={'commands':[],'scope':'one app-only configure/link and CPU contract closure; no GPU form with valid inputs','before_core':before_core}
 wrap="import os,sys,ctypes,signal; p=int(sys.argv[1]); k=ctypes.CDLL(None); assert os.getppid()==p; assert k.prctl(1,int(signal.SIGTERM),0,0,0)==0; assert os.getppid()==p; assert sys.stdin.buffer.read(1)==b'1'; os.execvpe(sys.argv[2],sys.argv[2:],os.environ)"
 def interrupted(sig,_):raise RuntimeError('app build interrupted')
 for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,interrupted)
 def run(name,argv,wanted=0):
  row={'name':name,'argv':argv,'expected':wanted};record['commands'].append(row);start=time.monotonic();owner=None;p=None
  with (ROOT/(name+'.stdout.log')).open('xb') as out,(ROOT/(name+'.stderr.log')).open('xb') as err:
   try:
    p=subprocess.Popen([sys.executable,'-c',wrap,str(os.getpid()),*argv],env=env,cwd=ROOT,stdin=subprocess.PIPE,stdout=out,stderr=err,start_new_session=True,pass_fds=tuple(fds));owner=Owned(p);row['identity']=owner.leader;p.stdin.write(b'1');p.stdin.close();row['returncode']=p.wait(timeout=max(1,deadline-time.monotonic()))
   finally:
    if owner:row['cleanup']=owner.cleanup()
    elif p:p.wait(timeout=3)
    row['seconds']=time.monotonic()-start
  if row['returncode']!=wanted:raise RuntimeError(name+' failed; stop')
 try:
  run('configure-apps',[CMAKE,'-S',str(ROOT/'source'),'-B',str(ROOT/'build'),'-DPOSEIDON_BUILD_LLAMA_MANUAL=ON'])
  run('compile-apps',[CMAKE,'--build',str(ROOT/'build'),'--target','llama31_poseidon_manual','llama31_manual_plan_tests','llama31_public_paired_silu','--parallel','1'])
  bin=ROOT/'build/bin';manual=bin/'llama31_poseidon_manual';control=bin/'llama31_public_paired_silu'
  run('cpu-plan-tests',[str(bin/'llama31_manual_plan_tests')]);run('cpu-plan-schema',[str(manual),'--test-plan-cpu']);run('cpu-silu-schema',[str(control),'--test-cpu'])
  run('cpu-inputs-manual',[str(manual),'--test-inputs-cpu','manual-highprecision-real-v1',str(ROOT/'assets/r20_fixture.txt'),str(ROOT/'assets/layer0/layer.weights.bin'),str(ROOT/'assets/layer0/index.tsv'),str(ROOT/'assets/embeddings.bin'),str(ROOT/'assets/reference-block01.bin'),'1','38',str(ROOT/'manifests/manual-real-b01-baseline-v1.tsv'),str(ROOT/'reports/manual-real-b01-baseline-v1/SUMMARY.json')])
  run('cpu-inputs-silu',[str(control),'--test-inputs-cpu',str(ROOT/'assets/r20_fixture.txt'),str(ROOT/'manifests/public-silu-baseline-v1.tsv'),str(ROOT/'reports/public-silu-baseline-v1/SUMMARY.json')])
  run('cpu-invalid-manual',[str(manual),'--invalid'],1);run('cpu-invalid-silu',[str(control),'--invalid'],1)
  run('app-ldd',['ldd',str(manual)]);run('app-symbols',['nm','-C',str(manual)]);run('backend-symbols',['nm','-C',str(ROOT/'build/libposeidon_gpu.a')])
  if ' T poseidon::gpu::GpuEvaluator::bootstrap_manual_projections(' not in (ROOT/'app-symbols.stdout.log').read_text():raise RuntimeError('binary projection definition missing')
  after_core=(sha(ROOT/'build/libposeidon_gpu.a'),sha(ROOT/'build/libposeidon_shared.so'))
  if before_core!=after_core:raise RuntimeError('app build changed core')
  verify();record.update(status='passed',after_core=after_core,binaries={p.name:sha(p) for p in [manual,control,bin/'llama31_manual_plan_tests']})
 except BaseException as e:record.update(status='failed_stop',error=type(e).__name__+': '+str(e));raise
 finally:(ROOT/'app-build-result.json').write_text(json.dumps(record,indent=2)+'\n')
 print(json.dumps(record,indent=2))
if __name__=='__main__':main()
