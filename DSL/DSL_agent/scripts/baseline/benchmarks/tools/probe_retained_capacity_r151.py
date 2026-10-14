"""Replay only retained Earth IR to diagnose capacity; no API, keys or execution."""
import argparse,json,os,shlex,shutil,sys,time,re
from pathlib import Path
from stage2_agent_campaign import ROOT,BASE
from benchmark_graph import digest
from benchmark_runner import dump
from stage2_agent_pilot_plan import sha
from campaign_live_state import sealed

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--inside',action='store_true');a=p.parse_args()
 from hecate_python_env import VENV,enter_nix
 if not a.inside:
  return enter_nix('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/'bin/python'),'-B',str(Path(__file__).resolve()),*sys.argv[1:],'--inside']),seconds=300)
 if os.environ.get('IN_NIX_SHELL')!='pure' or Path(sys.prefix)!=VENV:raise ValueError('Pinned environment required')
 from workspace_paths import RESULTS
 from semantic_benchmark_execution import runtime_sources
 from candidate_sandbox import run
 from seal_cpu_golden import PROFILE
 from native_execution_slots import native_slot
 from hecate_python_env import WORK
 if a.output.exists() or a.output.parent.resolve()!=RESULTS.resolve():raise ValueError('Fresh result directory required')
 before=runtime_sources()
 cases=[('free_bench_helper_0032','agent-deepseek-0l5ucnop'),('free_bench_helper_0033','agent-deepseek-2sfh9k00'),('free_bench_helper_0034','agent-deepseek-vfebar7z')]
 profile=json.loads(PROFILE.read_text())
 factor=profile['rescalingFactor'];upper=profile['bootstrapLevelUpperBound']
 a.output.mkdir();parents={str(PROFILE):sha(PROFILE)}
 sourcefile=ROOT/'third_party/dacapo/lib/Dialect/Earth/IR/EarthDialect.cpp'
 parents[str(sourcefile)]=sha(sourcefile)
 plan=sealed(dict(format='poseidon-retained-capacity-probe-r151',source_hashes=before,profile=profile,parents=parents,cases=cases,
  runner_sha256=sha(Path(__file__)),max_seconds=300,case_seconds=60,paid_calls=0,encrypted_execution=False,compile_only=True))
 dump(a.output/'plan.json',plan);rows=[];start=time.monotonic()
 for ident,name in cases:
  old=RESULTS/name/'attempt-00';folder=a.output/ident;folder.mkdir();out=folder/'output';out.mkdir()
  payload=old/'trace-payload.json';ir=old/'output/candidate_trace.mlir'
  for src in (payload,ir,old/'compile.log'):parents[str(src)]=sha(src)
  shutil.copyfile(ir,out/'candidate_trace.mlir')
  cmd=['/hecate-opt','/out/candidate_trace.mlir','--eva','--ckks-config=/profile.json','--waterline=45',
    '--mlir-disable-threading','--verify-each','--mlir-print-ir-after-failure','-o','/out/lowered.mlir']
  with native_slot(WORK/'cache/agent-native-slots'):
   code=run(payload,out,cmd,folder/'compile.log',seconds=60)
  text=(folder/'compile.log').read_text()
  matches=re.findall(r'note: see current operation:.*?earth\.mul.*? : \(tensor<(\d+)x!earth\.(?:ci|pl)<(\d+) \* (\d+)>>, tensor<(\d+)x!earth\.(?:ci|pl)<(\d+) \* (\d+)>>',text)
  # Match only compiler diagnostics, not arbitrary IR operations in the dump.
  errors=[]
  for m in matches:
   n1,s1,l1,n2,s2,l2=map(int,m)
   errors.append(dict(lhs_shape=n1,rhs_shape=n2,lhs_scale=s1,rhs_scale=s2,lhs_level=l1,rhs_level=l2,
    shapes_equal=n1==n2,levels_equal=l1==l2,capacity_limit=upper*factor,lhs_capacity=l1*factor+s1,
    capacity_exceeded=l1*factor+s1>upper*factor))
  if code==0 or not errors or not all(x['shapes_equal'] and x['levels_equal'] and x['capacity_exceeded'] for x in errors):
   raise ValueError('Capacity hypothesis not proven by replay: '+ident)
  row=dict(id=ident,original_evidence=str(old),compile_exit=code,expected_capacity_rejection_verified=True,
   diagnosed_failure='configured_scale_level_capacity_exceeded',diagnostics=errors,
   log_sha256=sha(folder/'compile.log'),original_ir_sha256=sha(ir),compiled_successfully=False,encrypted_execution=False)
  rows.append(row);dump(a.output/'progress.json',rows);print(ident,row['diagnosed_failure'],flush=True)
 if runtime_sources()!=before or any(sha(Path(n))!=h for n,h in parents.items()):raise ValueError('Source/evidence drift')
 result=sealed(dict(format='poseidon-retained-capacity-result-r151',plan_binding=plan['binding'],parents=parents,
  rows=rows,confirmed_capacity_failures=len(rows),new_compiler_invocations=len(rows),new_successful_compilations=0,
  paid_calls=0,new_encrypted_executions=0,stage2_complete=False,seconds=time.monotonic()-start,
  scope='Explains these exact candidates under unchanged profile; does not prove all equivalent DSL programs impossible or validate their math.'))
 dump(a.output/'report.json',result);print(json.dumps({'binding':result['binding'],'confirmed_capacity_failures':len(rows)}))
 return 0
if __name__=='__main__':raise SystemExit(main())
