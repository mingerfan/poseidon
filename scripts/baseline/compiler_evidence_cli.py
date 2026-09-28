"""Audit retained compiler/runtime evidence; no compilation, keygen or execution."""
import argparse,json,os,shlex,sys,hashlib
from pathlib import Path
from benchmark_graph import require,digest
from benchmark_runner import ROOT,DEFAULT,load,strict_file,dump

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['compiler-evidence'])
    p.add_argument('--candidate',action='append',type=Path,default=[])
    p.add_argument('--batch-report',action='append',type=Path,default=[])
    p.add_argument('--output',type=Path,required=True);p.add_argument('--inside',action='store_true',help=argparse.SUPPRESS)
    a=p.parse_args()
    from hecate_python_env import WORK,VENV,enter_nix
    def path_gate(path):
        require(path.resolve().is_relative_to(WORK/'results'),'Evidence must be in selected platform results')
        require(not path.is_symlink(),'Indirect evidence path')
    path_gate(a.output);require(not a.output.exists() and a.output.parent.is_dir(),'New audit file required')
    require(len(a.batch_report)<=8 and len(a.candidate)<=48,'Audit manifest bounds')
    candidates=list(a.candidate);batch_hashes={}
    for report_path in a.batch_report:
        path_gate(report_path);r=strict_file(report_path,16*1024**2)
        batch_hashes[str(report_path)]=hashlib.sha256(report_path.read_bytes()).hexdigest()
        for row in [*r.get('rows',[]),*r.get('records',[])]:
            if 'evidence' in row:candidates.append(Path(row['evidence']))
    candidates=list(dict.fromkeys(candidates));require(1<=len(candidates)<=48,'Audit shard must contain 1 to 48 distinct executions')
    for path in candidates:path_gate(path)
    if not a.inside:
        cmd=[str(VENV/'bin/python'),'-B',str(ROOT/'scripts/benchmark.py'),*sys.argv[1:],'--inside']
        return enter_nix('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" '+shlex.join(cmd),seconds=300)
    require(os.environ.get('IN_NIX_SHELL') and Path(sys.prefix)==VENV,'Locked Nix Python required')
    from compiler_artifact_evidence import audit
    load(DEFAULT);coverage=strict_file(DEFAULT/'coverage.json',4*1024**2)
    ids={x['id'] for x in coverage['requirements'] if x['layer']=='compiler'}
    require(ids=={'compiler.rescale','compiler.modswitch','compiler.relinearization','compiler.security_parameters','compiler.rotation_keys'},'Compiler ledger drift')
    records=[]
    try:
        for folder in candidates:records.append(audit(folder))
    except Exception as error:
        dump(a.output,dict(status='incomplete_audit_failure',error=str(error),completed=len(records),planned=len(candidates),records=records,new_encrypted_executions=0,agent_calls=0));raise
    observed={name:[] for name in ids}
    for record in records:
        evidence=record['compiler_evidence']
        for op in ('rescale','modswitch'):
            if evidence[op]['artifact_operations']:observed['compiler.'+op].append(record['evidence'])
        if evidence['relinearization']['fused_mulcc_operations']:observed['compiler.relinearization'].append(record['evidence'])
        observed['compiler.security_parameters'].append(record['evidence'])
        if evidence['rotation_keys']['required_steps']:observed['compiler.rotation_keys'].append(record['evidence'])
    counts={name:len(values) for name,values in observed.items()}
    result=dict(schema=1,contract='hevm-runtime-evidence-v1',status='audited',planned=len(candidates),audited=len(records),records=records,
        observed_requirements=observed,observed_execution_counts=counts,unique_model_count_not_claimed=True,
        numerical_passed=sum(r['numerical_passed'] for r in records),numerical_failed=sum(not r['numerical_passed'] for r in records),
        suite=str(DEFAULT),coverage_sha256=hashlib.sha256((DEFAULT/'coverage.json').read_bytes()).hexdigest(),
        batch_report_hashes=batch_hashes,new_encrypted_executions=0,agent_calls=0,
        checker_sources={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in [Path(__file__),ROOT/'scripts/baseline/compiler_artifact_evidence.py']},
        scope='retained execution artifacts and final runtime metadata; not new execution or formal numerical error bound')
    dump(a.output,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('records','observed_requirements','batch_report_hashes')},indent=2));return 0
