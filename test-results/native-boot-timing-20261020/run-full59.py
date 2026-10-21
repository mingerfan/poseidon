from pathlib import Path
import datetime,hashlib,json,os,subprocess,sys,time
r=Path(__file__).resolve().parent
sys.path.insert(0,str(r.parent/'poseidon-qwen-prefix-20261019'))
from nvml_sampler import Observer
observer=Observer();results=[]
try:
 for mode in ('runtime-mixed','qwen30'):
  assert not observer.applications(),'Other compute processes active'
  p=r/('full59-'+mode);p.mkdir(exist_ok=False)
  env={k:v for k,v in os.environ.items() if not k.startswith('POSEIDON_')}
  env.update(CUDA_VISIBLE_DEVICES='0',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',POSEIDON_GPU_RUNTIME_INITIAL_POOL_MB='24576',POSEIDON_GPU_RUNTIME_MAXIMUM_POOL_MB='24576')
  command=[str(r/'native-boot-bench-full59'),str(p/'report.json'),'16384',mode,'10']
  (p/'command.json').write_text(json.dumps(command,indent=2)+'\n')
  (p/'environment.json').write_text(json.dumps({k:v for k,v in env.items() if k.startswith(('POSEIDON_','CUDA_','OMP_','OPENBLAS_'))},indent=2)+'\n')
  started=time.monotonic()
  with (p/'run.log').open('w') as f:
   proc=subprocess.Popen(command,env=env,stdout=f,stderr=subprocess.STDOUT)
   (p/'launch.json').write_text(json.dumps(dict(pid=proc.pid,timestamp=datetime.datetime.now().astimezone().isoformat()),indent=2)+'\n')
   code=proc.wait()
  status=dict(context=mode,degree=16384,exit_code=code,wall_seconds=time.monotonic()-started)
  (p/'status.json').write_text(json.dumps(status,indent=2)+'\n');results.append(status)
  print(json.dumps(status),flush=True)
finally:observer.close()
(r/'full59-run-status.json').write_text(json.dumps(results,indent=2)+'\n')
