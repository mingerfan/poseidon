"""Rule-generated schema5 FHE baseline plus wrong-reduction counterexamples."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile

from hecate_python_env import ROOT,WORK,VENV,enter_nix,digest
from seal_artifact_gate import require


def inside(smoke=False,tensor=False,spatial=False,spatial_max=False,composition=False,permutation=False,layout_guidance=False,native_packed=False):
    import torch
    if native_packed:
        from packed_native_cases import cases,candidate
    elif layout_guidance:
        from layout_guidance_cases import cases
    elif permutation:
        from permutation_model_cases import cases
    elif composition:
        from packed_composition_cases import cases
    elif spatial_max:
        from packed_spatial_cases import large_cases as cases
    elif spatial:
        from packed_spatial_cases import cases
    elif tensor:
        from tensor_model_cases import cases
    else:
        from packed_model_cases import cases
    from run_model_batch import prepare_case
    from fx_to_hecate import translate
    from candidate_contract import make_request
    from compiler_configuration import configuration
    from seal_cpu_golden import PROFILE,dump
    torch.set_num_threads(2);os.umask(0o077)
    prefix='tensor-permutation-goldens-' if permutation else 'packed-composition-goldens-' if composition else 'packed-spatial-max-goldens-' if spatial_max else 'packed-spatial-goldens-' if spatial else 'tensor-model-goldens-' if tensor else 'packed-model-goldens-'
    if layout_guidance:prefix='layout-guidance-goldens-'
    if native_packed:prefix='packed-native-goldens-'
    root=Path(tempfile.mkdtemp(prefix=prefix,dir=WORK/'results'))
    print('Packed model evidence:',root,flush=True)
    config=configuration('seal-cpu-eva-w45-v1',digest(PROFILE))
    report=dict(status='running',generator='deterministic_tensor_permutation_fx' if permutation else 'deterministic_packed_composition_fx' if composition else 'deterministic_packed_spatial_fx' if spatial or spatial_max else 'deterministic_tensor_fx' if tensor else 'deterministic_packed_fx',agent_calls=0,
        backend='upstream_SEAL_HEVM_CPU',poseidon_gpu_validated=False,compiler_configuration=config,
        source_hashes={n:digest(ROOT/'scripts/baseline'/n) for n in
            ('packed_model.py','packed_input_abi.py','permutation_model_cases.py' if permutation else 'packed_composition_cases.py' if composition else 'packed_spatial_cases.py' if spatial or spatial_max else 'tensor_model_cases.py' if tensor else 'packed_model_cases.py',Path(__file__).name)},cases=[])
    if layout_guidance:
        report['generator']='deterministic_layout_guidance_fx'
        report['source_hashes'].update({n:digest(ROOT/'scripts/baseline'/n) for n in
            ('layout_guidance_cases.py','candidate_contract.py')})
    if native_packed:
        report['generator']='deterministic_packed_native_fixture'
        report['source_hashes'].update({n:digest(ROOT/'scripts/baseline'/n) for n in
            ('packed_native_cases.py','candidate_contract.py','decorated_functions.py')})
    examples=cases()
    plans=([(d,False) for d in examples]+[(examples[0],True),(examples[2],True)] if native_packed else
           [(d,False) for d in examples] if layout_guidance else
           [(d,False) for d in examples]+[(examples[0],True),(examples[7],True)] if permutation else
           [(d,False) for d in examples]+[(examples[1],True),(examples[8],True)] if composition else
           [(d,False) for d in examples] if spatial_max else [(examples[0],False),(examples[9],False)] if smoke else
           [(d,False) for d in examples]+([(examples[1],True),(examples[7],True)] if spatial else [(examples[4],True),(examples[6],True)] if tensor else
                                       [(examples[0],True),(examples[8],True)]))
    for i,(d,wrong) in enumerate(plans):
        folder=root/f'case-{i:02d}';folder.mkdir();dump(folder/'model.json',d)
        model,shape=prepare_case(d,folder);payload=translate(model,shape)
        request=make_request(payload,d,digest(PROFILE),str(model),compiler_configuration=config,native_array_mutation=native_packed)
        source=payload['hecate_source']
        if native_packed:
            source=candidate(payload,examples.index(d),wrong)
        elif wrong:
            if permutation and d['id']==examples[0]['id']:
                # Legal key step but wrong slot permutation.
                source=source.replace('.rotate(1)','.rotate(2)')
            elif permutation:
                line=source.splitlines()[-1];prefix,tail=line.split('[',1)
                names=tail.rstrip(']').split(', ');names[0],names[-1]=names[-1],names[0]
                source=source[:source.rfind(line)]+prefix+'['+', '.join(names)+']\n'
            elif composition and d['id']==examples[1]['id']:
                # Use the folded offset as gain: same type/shape, wrong BN.
                origins=payload['constant_origins']
                gain=next(k for k,v in origins.items() if v.endswith('.gain'))
                offset=next(k for k,v in origins.items() if v.endswith('.offset'))
                source=re.sub(r'\b'+gain+r'\b',offset,source)
            elif composition or spatial and d['id']==examples[7]['id']:
                # Wrong first pooling window while preserving ciphertext count.
                line=source.splitlines()[-1];prefix,tail=line.split('[',1)
                names=tail.rstrip(']').split(', ')
                if composition:names[0],names[-1]=names[-1],names[0]
                else:names[0]=names[1]
                source=source[:source.rfind(line)]+prefix+'['+', '.join(names)+']\n'
            elif spatial or tensor and d['id']==examples[6]['id']:
                # Reuse group0's masked rows for group1: legal DSL, wrong model.
                origins=payload['constant_origins']
                first,stop=(5,10) if spatial else (2,4)
                for row in range(first,stop):
                    old=next(k for k,v in origins.items() if v.endswith(f'.weight[{row}]'))
                    new=next(k for k,v in origins.items() if v.endswith(f'.weight[{row-first}]'))
                    source=re.sub(r'\b'+old+r'\b',new,source)
            else:
                step=payload['layout']['input_slot_period']//2
                source=source.replace(f'.rotate({step})',f'.rotate({step//2})')
            require(source!=payload['hecate_source'],'Unchanged counterexample')
        dump(folder/'request.json',request)
        dump(folder/'responses.json',[json.dumps(dict(schema=1,request_id=request['request_id'],hecate_source=source))])
        report['cases'].append(dict(id=d['id'],counterexample=wrong,folder=folder.name,status='pending'))
    dump(root/'report.json',report)
    def run(i):
        item=dict(report['cases'][i]);folder=root/item['folder']
        command=[str(VENV/'bin/python'),str(ROOT/'scripts/baseline/run_candidate.py'),'--inside',
            '--case',str(folder/'model.json'),'--replay',str(folder/'responses.json'),'--max-repairs','0',
            '--compiler-configuration','seal-cpu-eva-w45-v1']
        if native_packed:command+=['--native-array-mutation']
        item['command']=command
        try:
            with (folder/'run.log').open('w') as stream:
                code=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=480).returncode
            item['exit_code']=code
            matches=re.findall(r'^Candidate evidence: (.+)$',(folder/'run.log').read_text(),re.MULTILINE)
            require(len(matches)==1,'Missing candidate evidence')
            run=Path(matches[0]).resolve();require(run.parent==(WORK/'results').resolve(),'Unexpected evidence path')
            child=json.loads((run/'report.json').read_text())
            item.update(run=str(run),candidate_status=child['status'],report_sha256=digest(run/'report.json'))
            require(child['agent_calls']==0 and child['provider']=='scripted_replay','Unexpected generator')
            attempt=child['attempts'][0] if child['attempts'] else {}
            item.update(failure_layer=attempt.get('failure_layer',child.get('failure_layer')),
                diagnostic=attempt.get('diagnostic',child.get('diagnostic')),comparison=attempt.get('comparison'))
            require(attempt.get('execution',{}).get('encrypted_execution') is True,'Missing encrypted execution')
            require((code==1 and item['failure_layer']=='numerical_comparison' and not item['comparison']['passed'])
                if item['counterexample'] else code==0 and child['status']=='passed','Unexpected numerical outcome')
            cleanup=json.loads((run/'key-cleanup-outcome.json').read_text())
            require(cleanup['complete'] and not (run/'private-keys').exists(),'Key cleanup incomplete')
            item.update(status='passed',freed_key_bytes=cleanup['freed_bytes'])
        except Exception as error:item.update(status='failed',runner_diagnostic=str(error))
        return i,item
    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(run,i) for i in range(len(plans))]):
            i,item=future.result();report['cases'][i]=item;dump(root/'report.json',report)
            print(f"{i+1}/{len(plans)} {item['id']} wrong={item['counterexample']}: {item['status']} {item.get('failure_layer')} {item.get('diagnostic')}",flush=True)
    report['status']='passed' if all(c['status']=='passed' for c in report['cases']) else 'failed'
    dump(root/'report.json',report)
    return 0 if report['status']=='passed' else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inside',action='store_true');parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--tensor',action='store_true',help='Rank-preserving broadcast/reshape/last-axis Linear cohort')
    parser.add_argument('--spatial',action='store_true',help='Packed batched Conv/average-pool cohort')
    parser.add_argument('--spatial-max',action='store_true',help='Two 256-input/16-output Conv boundary cases')
    parser.add_argument('--composition',action='store_true',help='Packed BatchNorm/concat and mixed-layout compositions')
    parser.add_argument('--permutation',action='store_true',help='Static axis permutation with real packed slot routing')
    parser.add_argument('--layout-guidance',action='store_true',help='Two prior layout failures and four new compositions with versioned guidance')
    parser.add_argument('--native-packed',action='store_true',help='Native decorated functions/arrays/loops with schema5 variable-period layout')
    args=parser.parse_args();require(Path.cwd().resolve()==ROOT,'Wrong source root')
    if args.inside:
        require(Path(sys.prefix)==VENV and os.environ.get('IN_NIX_SHELL'),'Pinned environment required')
        require(sum((args.tensor,args.spatial,args.spatial_max,args.composition,args.permutation,args.layout_guidance,args.native_packed))<=1,'Choose one cohort')
        return inside(args.smoke,args.tensor,args.spatial,args.spatial_max,args.composition,args.permutation,args.layout_guidance,args.native_packed)
    cmd='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" PYTHONDONTWRITEBYTECODE=1 '+shlex.join(
        [str(VENV/'bin/python'),str(Path(__file__).resolve()),'--inside',*(['--smoke'] if args.smoke else []),
         *(['--tensor'] if args.tensor else []),*(['--spatial'] if args.spatial else []),
         *(['--spatial-max'] if args.spatial_max else []),*(['--composition'] if args.composition else []),
         *(['--permutation'] if args.permutation else []),*(['--layout-guidance'] if args.layout_guidance else []),
         *(['--native-packed'] if args.native_packed else [])])
    return enter_nix(cmd,seconds=3600)


if __name__=='__main__':raise SystemExit(main())
