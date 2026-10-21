from pathlib import Path
import datetime,hashlib,json,os,subprocess,sys,time
r=Path(__file__).resolve().parent
sys.path.insert(0,str(r.parent/'poseidon-qwen-prefix-20261019'))
from nvml_sampler import Observer
o=Observer()
try:assert not o.applications(),'Other compute processes active'
finally:o.close()
p=r/'existing-dynamic32';p.mkdir(exist_ok=False)
env={k:v for k,v in os.environ.items() if not k.startswith('POSEIDON_')}
env.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',POSEIDON_BOOTSTRAP_PROFILE='dynamic32',POSEIDON_BOOTSTRAP_DEGREE='16384',POSEIDON_BOOTSTRAP_GPU_ONLY_TIMING='1',POSEIDON_BOOTSTRAP_IGNORE_CORRECTNESS_FAILURE='0',POSEIDON_BOOTSTRAP_ITERATIONS='1',POSEIDON_BOOTSTRAP_WARMUP='1',POSEIDON_BOOTSTRAP_FULL_WARMUP='2',POSEIDON_BOOTSTRAP_FULL_ITERATIONS='10')
command=['bash',str(r/'run-existing-dynamic32.sh')]
(p/'command.json').write_text(json.dumps(command,indent=2)+'\n')
(p/'environment-input.json').write_text(json.dumps({k:v for k,v in env.items() if k.startswith(('POSEIDON_','CUDA_','OMP_','OPENBLAS_'))},indent=2)+'\n')
start=time.monotonic()
with (p/'run.log').open('w') as f:
 proc=subprocess.Popen(command,env=env,stdout=f,stderr=subprocess.STDOUT)
 (p/'launch.json').write_text(json.dumps(dict(pid=proc.pid,timestamp=datetime.datetime.now().astimezone().isoformat()),indent=2)+'\n')
 try:code=proc.wait(timeout=300)
 except subprocess.TimeoutExpired:
  proc.terminate();code=proc.wait(timeout=30)
  (p/'timeout.json').write_text(json.dumps(dict(reason='300-second diagnostic cap; terminated this benchmark process only'))+'\n')
(p/'status.json').write_text(json.dumps(dict(exit_code=code,wall_seconds=time.monotonic()-start),indent=2)+'\n')
print('Existing standalone native benchmark exit',code,flush=True)
