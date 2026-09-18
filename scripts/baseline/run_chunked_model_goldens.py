"""Deterministic schema-4 baseline through real isolated Dacapo/SEAL, no API.

Correct programs are FX rule answers, never labelled Agent or hand-written.
Two faulty programs reuse the wrong chunk to verify numerical rejection.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile

from hecate_python_env import ROOT, WORK, VENV, enter_nix, digest
from seal_artifact_gate import require


def inside():
    import torch
    from chunked_model_cases import cases
    from run_model_batch import prepare_case
    from fx_to_hecate import translate
    from candidate_contract import make_request
    from compiler_configuration import configuration
    from seal_cpu_golden import PROFILE, dump
    torch.set_num_threads(2)
    os.umask(0o077)
    root=Path(tempfile.mkdtemp(prefix='chunked-model-goldens-',dir=WORK/'results'))
    print('Chunked model evidence:',root,flush=True)
    config=configuration('seal-cpu-eva-w45-v1',digest(PROFILE))
    report=dict(status='running',generator='deterministic_chunked_fx',agent_calls=0,
        backend='upstream_SEAL_HEVM_CPU',poseidon_gpu_validated=False,compiler_configuration=config,
        source_hashes={n:digest(ROOT/'scripts/baseline'/n) for n in
                      ('chunked_model.py','chunked_input_abi.py','chunked_model_cases.py',Path(__file__).name)},
        cases=[])
    examples=cases()
    plans=[(d,False) for d in examples]+[(examples[1],True),(examples[5],True)]
    for i,(descriptor,wrong) in enumerate(plans):
        folder=root/f'case-{i:02d}';folder.mkdir()
        dump(folder/'model.json',descriptor)
        model,shape=prepare_case(descriptor,folder)
        payload=translate(model,shape)
        request=make_request(payload,descriptor,digest(PROFILE),str(model),compiler_configuration=config)
        source=payload['hecate_source']
        if wrong:
            lines=source.splitlines(keepends=True)
            last=payload['layout']['inputs'][-1]['dsl_name']
            lines[2:]=[re.sub(r'\b'+last+r'\b','x',s) for s in lines[2:]]
            source=''.join(lines)
            require(source != payload['hecate_source'],'Counterexample did not alter source')
        dump(folder/'request.json',request)
        dump(folder/'responses.json',[json.dumps(dict(schema=1,request_id=request['request_id'],hecate_source=source))])
        report['cases'].append(dict(id=descriptor['id'],counterexample=wrong,folder=folder.name,status='pending'))
    dump(root/'report.json',report)
    def run(i):
        item=dict(report['cases'][i]);folder=root/item['folder']
        command=[str(VENV/'bin/python'),str(ROOT/'scripts/baseline/run_candidate.py'),'--inside',
                 '--case',str(folder/'model.json'),'--replay',str(folder/'responses.json'),
                 '--max-repairs','0','--compiler-configuration','seal-cpu-eva-w45-v1']
        item['command']=command
        try:
            with (folder/'run.log').open('w') as stream:
                code=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=360).returncode
            item['exit_code']=code
            log=(folder/'run.log').read_text()
            matches=re.findall(r'^Candidate evidence: (.+)$',log,re.MULTILINE)
            require(len(matches)==1,'Missing candidate evidence')
            run=Path(matches[0]).resolve()
            require(run.parent == (WORK/'results').resolve(),'Unexpected evidence path')
            child=json.loads((run/'report.json').read_text())
            item.update(run=str(run),candidate_status=child['status'],report_sha256=digest(run/'report.json'))
            require(child['agent_calls']==0 and child['provider']=='scripted_replay','Unexpected generation source')
            attempt=child['attempts'][0] if child['attempts'] else {}
            item.update(failure_layer=attempt.get('failure_layer',child.get('failure_layer')),
                        diagnostic=attempt.get('diagnostic',child.get('diagnostic')),
                        comparison=attempt.get('comparison'))
            require(attempt.get('execution',{}).get('encrypted_execution') is True,'Missing encrypted execution')
            matched=(code==1 and item['failure_layer']=='numerical_comparison' and
                     not item['comparison']['passed']) if item['counterexample'] else code==0 and child['status']=='passed'
            require(matched,'Unexpected numerical outcome')
            cleanup=json.loads((run/'key-cleanup-outcome.json').read_text())
            require(cleanup['complete'] and not (run/'private-keys').exists(),'Key cleanup incomplete')
            item.update(status='passed',freed_key_bytes=cleanup['freed_bytes'])
        except Exception as error:
            item.update(status='failed',runner_diagnostic=str(error))
        return i,item
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(run,i) for i in range(len(plans))]
        for future in as_completed(futures):
            i,item=future.result();report['cases'][i]=item
            dump(root/'report.json',report)
            print(f"{i+1}/{len(plans)} {item['id']} wrong={item['counterexample']}: {item['status']} {item.get('failure_layer')}",flush=True)
    report['status']='passed' if all(c['status']=='passed' for c in report['cases']) else 'failed'
    dump(root/'report.json',report)
    return 0 if report['status']=='passed' else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--inside',action='store_true')
    args=parser.parse_args()
    require(Path.cwd().resolve()==ROOT,'Wrong source root')
    if args.inside:
        require(Path(sys.prefix)==VENV and os.environ.get('IN_NIX_SHELL'),'Pinned environment required')
        return inside()
    cmd='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" PYTHONDONTWRITEBYTECODE=1 '+shlex.join(
        [str(VENV/'bin/python'),str(Path(__file__).resolve()),'--inside'])
    return enter_nix(cmd,seconds=2200)


if __name__=='__main__':raise SystemExit(main())
