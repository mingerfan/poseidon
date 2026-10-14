"""Three real sandbox rejection probes; never generate keys or call an API.

Use completed explicit45 artifacts and valid source hashes. The worker must
reject mismatched identities before it attempts to open keys or execute HEVM.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

from hecate_python_env import ROOT,WORK,VENV,digest
from seal_cpu_golden import dump
from seal_artifact_gate import require
from native_execution_slots import native_slot
from compiler_configuration import configuration,canonical,request_configuration
import candidate_sandbox as sandbox


def probe(source):
    require(os.environ.get('IN_NIX_SHELL') and Path(sys.prefix)==VENV,'Requires pinned Nix/venv')
    source=Path(source).resolve();results=WORK/'results'
    require(source.parent==results and source.name.startswith('candidate-replay-'),'Requires local manual candidate evidence')
    original=json.loads((source/'report.json').read_text())
    require(original['status']=='passed' and original['agent_calls']==0 and original['waterline']==45,
            'Requires completed non-Agent explicit45 evidence')
    require(original['compiler_configuration']==configuration('seal-cpu-eva-w45-v1'),'Unexpected source configuration')
    attempt=original['attempts'][0];old=source/'attempt-00/output'
    for name,value in original['frozen_hashes'].items(): require(digest(source/name)==value,'Source input mutated')
    for name,value in attempt['artifact_hashes'].items(): require(digest(old/name)==value,'Source artifact mutated')
    payload=json.loads((source/'attempt-00/trace-payload.json').read_text())
    request_configuration(payload['request'])
    os.umask(0o077)
    root=Path(tempfile.mkdtemp(prefix='compiler-configuration-guard-',dir=results))
    print('Compiler configuration guard evidence: '+str(root),flush=True)
    report=dict(status='running',source_run=str(source),source_report_sha256=digest(source/'report.json'),
                source_hashes={str(old/n):digest(old/n) for n in ('lowered._hecate_golden.hevm','_hecate_golden.cst')},
                worker_sha256=digest(ROOT/'scripts/baseline/candidate_worker.py'),
                configuration_source_sha256=digest(ROOT/'scripts/baseline/compiler_configuration.py'),
                api_calls=0,fhe_executions=0,keys_mounted=False,cases=[])
    mutations=(('different_valid_waterline','Compiled artifact input precision differs from immutable request'),
               ('changed_configuration','Changed compiler configuration'),
               ('changed_json_hash','Compiler configuration profile hash mismatch'))
    for name,expected in mutations:
        case=root/name;case.mkdir();out=case/'output';out.mkdir()
        for filename in ('lowered._hecate_golden.hevm','_hecate_golden.cst'):
            shutil.copyfile(old/filename,out/filename)
        changed=copy.deepcopy(payload);r=changed['request']
        if name=='different_valid_waterline': r['compiler_configuration']=configuration('seal-cpu-eva-w40-v1')
        elif name=='changed_configuration': r['compiler_configuration']['waterline']=40
        else:
            r['compiler_profile_sha256']='0'*64
            r['compiler_configuration']['ckks_config_sha256']='0'*64
        r['request_id']=hashlib.sha256(canonical({k:v for k,v in r.items() if k!='request_id'})).hexdigest()
        changed['candidate']['request_id']=r['request_id']
        path=case/'payload.json';dump(path,changed)
        command=[str(VENV/'bin/python'),'/app/candidate_worker.py','execute']
        with native_slot(WORK/'cache/agent-native-slots',timeout=120):
            code=sandbox.run(path,out,command,case/'worker.log',seconds=30)
        log=(case/'worker.log').read_text()
        matched=(code!=0 and expected in log and 'Traceback' in log and
                 not (out/'execution.json').exists() and not (out/'decrypted.npy').exists())
        report['cases'].append(dict(name=name,exit_code=code,expected_diagnostic=expected,
            matched_expected=matched,hashes={str(p.relative_to(root)):digest(p)
                for p in (path,case/'worker.log',out/'lowered._hecate_golden.hevm',out/'_hecate_golden.cst')}))
        dump(root/'report.json',report)
    require(digest(source/'report.json')==report['source_report_sha256'],'Source report changed')
    report['status']='passed' if all(c['matched_expected'] for c in report['cases']) else 'failed'
    dump(root/'report.json',report)
    return 0 if report['status']=='passed' else 1


if __name__=='__main__':
    require(len(sys.argv)==2,'Pass one completed explicit45 manual run')
    raise SystemExit(probe(sys.argv[1]))
