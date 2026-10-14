"""Execute the one separately authorized, frozen r235 bootstrap trial."""
import argparse,hashlib,json,math,os,pathlib,resource,shlex,shutil,subprocess,sys,time
ROOT=(Path(__file__).resolve().parents[6] if Path(__file__).resolve().parents[4].name == "DSL_agent" and Path(__file__).resolve().parents[5].name == "DSL" else Path(__file__).resolve().parents[4])
sys.path.insert(0,str(ROOT/'scripts/baseline'))
WORK=pathlib.Path('/home/lhohy/poseidon-work/platforms/aarch64-linux')
OUT=WORK/'results/poseidon-bootstrap-ratio128-r236'
BUILD=WORK/'build-poseidon/bootstrap-ratio128-r236'
SRC=ROOT/'src/poseidon/tests/bootstrapping/agent_bootstrap_experiment_ratio128'
LIBBUILD=WORK/'build-poseidon/bootstrap-cpu-r217/poseidon_build'
LIB=LIBBUILD/'libposeidon_shared.so'
PROPOSAL=ROOT/'docs/baseline/poseidon-native-bootstrap-proposal-r235.json'
LIB_SHA='dcfb32471bbadf12e66835fc9b8dac4c06106582cf884862f3ac8a9612e154e4'
PROPOSAL_FILE_SHA='7650c13d8fd048fad55b5badba361ae4b96baa15f69f6b93db85fce405115f18'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--inside',action='store_true');a=p.parse_args()
 auth=json.loads((OUT/'authorization.json').read_text())
 assert auth['approved'] is True and sha(PROPOSAL)==auth['proposal_file_sha256']==PROPOSAL_FILE_SHA
 cfg=json.loads(PROPOSAL.read_text());assert auth['approved_scope']==cfg
 limits=cfg['limits'];deadline=auth['started_monotonic']+limits['max_wall_seconds']
 assert cfg['bootstrap']['log_message_ratio']==7 and cfg['bootstrap']['output_ratio']==128
 assert cfg['numeric_tolerance']==dict(atol=1e-5,rtol=1e-4)
 assert math.prod(cfg['q']+cfg['p']).bit_length()==876<=881
 assert sha(LIB)==LIB_SHA
 for rel,h in cfg['probe_source_sha256'].items():assert sha(ROOT/rel)==h,('Probe changed',rel)
 def remaining():
  n=int(deadline-time.monotonic())-5
  if n<=0:raise RuntimeError('Approved wall budget exhausted')
  return n
 if not a.inside:
  from hecate_python_env import enter_nix
  try:
   return enter_nix(shlex.join(['python3','-B',str(pathlib.Path(__file__).resolve()),'--inside']),
    seconds=remaining(),environment='poseidon-cpu')
  except Exception as exc:
   if not (OUT/'report.json').exists():
    (OUT/'report.json').write_text(json.dumps(dict(format='poseidon-bootstrap-experiment-result-v2',
     status='blocked',layer='environment_startup',error=str(exc),paid_calls=0,download_bytes=0,
     authorization_sha256=sha(OUT/'authorization.json')),indent=2)+'\n')
   raise
 assert os.environ.get('IN_NIX_SHELL') and sys.version_info[:3]==(3,10,14)
 assert not BUILD.exists() and not (OUT/'report.json').exists()
 assert shutil.disk_usage(WORK).free>=4*1024**3
 available=int(next(x.split()[1] for x in pathlib.Path('/proc/meminfo').read_text().splitlines()
     if x.startswith('MemAvailable:')))*1024
 assert available>7*1024**3,'Less than 7 GiB currently available; preserve trial'
 shutil.copytree(SRC,OUT/'source-snapshot');shutil.copy2(__file__,OUT/'runner.py')
 report=dict(format='poseidon-bootstrap-experiment-result-v2',status='running',
  authorization_sha256=sha(OUT/'authorization.json'),proposal_file_sha256=sha(PROPOSAL),
  proposal_sha256=cfg['proposal_sha256'],parameters=cfg,library_sha256=sha(LIB),
  source_hashes=cfg['probe_source_sha256'],steps=[],paid_calls=0,download_bytes=0,
  runtime_integration=False,agent_generated=False,stop_on_first_failure=True)
 def size():
  return sum(p.stat().st_size for base in (BUILD,OUT) if base.exists() for p in base.rglob('*') if p.is_file())
 def save():
  report.update(elapsed_seconds=time.monotonic()-auth['started_monotonic'],disk_bytes=size())
  (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 def boundary():
  remaining()
  assert size()<=limits['max_disk_bytes'],'Approved disk budget exhausted'
  assert sha(LIB)==LIB_SHA and sha(PROPOSAL)==PROPOSAL_FILE_SHA,'Dependency changed'
  for rel,h in cfg['probe_source_sha256'].items():assert sha(ROOT/rel)==h,('Probe changed',rel)
 def preexec():
  resource.setrlimit(resource.RLIMIT_CORE,(0,0))
  resource.setrlimit(resource.RLIMIT_AS,(limits['max_address_space_bytes'],)*2)
  seconds=remaining();resource.setrlimit(resource.RLIMIT_CPU,(seconds,seconds))
  resource.setrlimit(resource.RLIMIT_FSIZE,(64*1024**2,)*2)
  resource.setrlimit(resource.RLIMIT_NOFILE,(128,128))
 def bounded(cmd,name,env=None):
  boundary();stamp=time.monotonic();seconds=remaining()
  with (OUT/(name+'.log')).open('w') as log:
   result=subprocess.run(['/usr/bin/timeout','-k','3s',str(seconds),*cmd],stdout=log,
    stderr=subprocess.STDOUT,timeout=seconds+4,preexec_fn=preexec,env=env)
  report['steps'].append(dict(name=name,returncode=result.returncode,seconds=time.monotonic()-stamp))
  save();print(name,result.returncode,flush=True);boundary()
  return result.returncode
 try:
  save()
  if bounded(['cmake','-G','Ninja','-S',str(SRC),'-B',str(BUILD),'-DCMAKE_BUILD_TYPE=Release',
   '-DPOSEIDON_ROOT='+str(ROOT),'-DPOSEIDON_CPU_BUILD='+str(LIBBUILD)],'configure'):
   raise RuntimeError('configure failed')
  if bounded(['cmake','--build',str(BUILD),'-j2'],'build'):raise RuntimeError('build failed')
  binary=BUILD/'agent_bootstrap_experiment'
  report['binary_sha256']=sha(binary);shutil.copy2(binary,OUT/'probe-binary')
  library=os.environ['HECATE_PYTHON_LIBRARY_PATH']
  assert all(pathlib.Path(p).is_relative_to('/nix/store') for p in library.split(':'))
  cmd=['/usr/bin/bwrap','--unshare-all','--new-session','--die-with-parent','--cap-drop','ALL','--clearenv',
   '--ro-bind','/nix/store','/nix/store','--ro-bind',str(binary),'/probe','--ro-bind',str(LIB),str(LIB),
   '--proc','/proc','--dev','/dev','--tmpfs','/tmp','--setenv','HOME','/nonexistent',
   '--setenv','PATH','/nonexistent','--setenv','OMP_NUM_THREADS','1','--setenv','OPENBLAS_NUM_THREADS','1',
   '--setenv','LD_LIBRARY_PATH',library,'--chdir','/tmp','--','/probe']
  code=bounded(cmd,'execute',{'PATH':'/usr/bin:/bin'})
  log=(OUT/'execute.log').read_text()
  report.update(context_admitted='context_admitted=true' in log,
   comparisons=[line for line in log.splitlines() if line.startswith('comparison ')],
   diagnostics=[line for line in log.splitlines() if line.startswith(('state=','blocked_','bootstrap_config '))])
  report['status']='passed' if code==0 and 'all_four_groups_passed=true' in log else 'failed'
  report['numerical_passed']=report['status']=='passed'
  report['launcher_peak_child_rss_kib']=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
  report['memory_note']='Launcher accounting does not measure the isolated probe peak; RLIMIT_AS enforces 6 GiB.'
 except Exception as exc:
  report.update(status='blocked',error=str(exc))
 finally:save()
 print(json.dumps({k:v for k,v in report.items() if k not in ('parameters','source_hashes')},indent=2),flush=True)
 return 0 if report['status']=='passed' else 1
if __name__=='__main__':raise SystemExit(main())
