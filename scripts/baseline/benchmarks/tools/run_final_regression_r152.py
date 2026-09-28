"""Run only current focused compatibility and audit-gate tests in the pinned environment."""
import argparse,json,os,shlex,sys,unittest
from pathlib import Path
from stage2_agent_campaign import ROOT,BASE
from benchmark_graph import digest
from stage2_agent_pilot_plan import sha
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--inside',action='store_true');a=p.parse_args()
 from hecate_python_env import VENV,enter_nix
 if not a.inside:return enter_nix('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" '+shlex.join([str(VENV/'bin/python'),'-B',str(Path(__file__).resolve()),*sys.argv[1:],'--inside']),seconds=120)
 if os.environ.get('IN_NIX_SHELL')!='pure' or Path(sys.prefix)!=VENV:raise ValueError('Pinned environment required')
 from workspace_paths import RESULTS
 from semantic_benchmark_execution import runtime_sources
 if a.output.exists() or a.output.parent.resolve()!=RESULTS.resolve():raise ValueError('Fresh results directory required')
 a.output.mkdir();before=runtime_sources()
 names=['test_unified_generation_guidance','test_unified_directed_guidance','test_retained_acceptance_context_r152']
 suite=unittest.defaultTestLoader.loadTestsFromNames(names)
 with (a.output/'tests.log').open('x') as f:res=unittest.TextTestRunner(stream=f,verbosity=2).run(suite)
 if runtime_sources()!=before:raise ValueError('Source drift')
 result=dict(format='poseidon-stage2-final-focused-regression-r152',tests=res.testsRun,passed=res.testsRun-len(res.failures)-len(res.errors)-len(res.skipped),
  failed=len(res.failures),errors=len(res.errors),skipped=len(res.skipped),source_sha256=digest(before),modules=names,
  log_sha256=sha(a.output/'tests.log'),runner_sha256=sha(Path(__file__)),audit_test_sha256=sha(Path(__file__).with_name('test_retained_acceptance_context_r152.py')),
  new_paid_calls=0,new_compilations=0,new_encrypted_executions=0)
 result['binding']=digest(result)
 with (a.output/'report.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(result));return int(not res.wasSuccessful())
if __name__=='__main__':raise SystemExit(main())
