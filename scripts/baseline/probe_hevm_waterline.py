"""Frozen three-key waterline experiment through existing compiler and SEAL HEVM.

No Agent API, no change to default profile/tolerance, no retry-until-pass.
The same archived Earth IR, input arrays and key set are used at 40/45/50.
Only the waterline option changes; the normal Agent launcher remains at 40.
"""
import json
import os
from pathlib import Path
import shlex
import shutil
import struct
import subprocess
import sys
import tempfile
from hecate_python_env import ROOT,WORK,VENV,enter_nix,digest
from python_compiler_smoke import BUILD
from seal_cpu_golden import KEY_BUILD,PROFILE,WATERLINE,compare,dump
from seal_artifact_gate import inspect_artifacts,require
from native_execution_slots import native_slot
import candidate_sandbox as sandbox

SOURCE_RUN=WORK/'results/candidate-replay-t9aglvbt'
SCALES=(40,45,50)
TRIALS=3
SCRIPT=Path(__file__).resolve()
SOURCE_REPORT_SHA='4ea525b5baa833cf40df36050a00b01f683a86c5a6e307fc28d3c5e05f1f00e0'

def semantic_instructions(raw):
    # Body size lives at byte 24; ignore unspecified tensor.empty operands.
    body_size=struct.unpack_from('<Q',raw,24)[0]
    return [op for op in struct.iter_unpack('<4H',raw[24+body_size:]) if op[0]!=65535]

def checked_original():
    require(digest(SOURCE_RUN/'report.json')==SOURCE_REPORT_SHA,'Frozen source report changed')
    report=json.loads((SOURCE_RUN/'report.json').read_text())
    require(report['waterline']==40 and report['backend']=='upstream_SEAL_HEVM_CPU',
            'Unexpected source backend/profile')
    require(report['attempts'][0]['failure_layer']=='numerical_comparison' and
            not report['attempts'][0]['comparison']['passed'],'Source must retain original numerical failure')
    for name,value in report['frozen_hashes'].items():
        require(digest(SOURCE_RUN/name)==value,'Original reference/source mutated')
    output=SOURCE_RUN/'attempt-00/output'
    for name,value in report['attempts'][0]['artifact_hashes'].items():
        require(digest(output/name)==value,'Original compiled artifact mutated')
    return report,output

def native(*args,**kwargs):
    with native_slot(WORK/'cache/agent-native-slots',timeout=120):
        return sandbox.run(*args,**kwargs)

def inside(root):
    import numpy as np
    require(os.environ.get('IN_NIX_SHELL') and Path(sys.prefix)==VENV,'Requires pinned isolation')
    original,old_output=checked_original()
    require(WATERLINE==40,'Normal Agent waterline must remain unchanged')
    originals={str(p):digest(p) for p in (
        SOURCE_RUN/'report.json',SOURCE_RUN/'request.json',SOURCE_RUN/'arrays.npz',
        SOURCE_RUN/'attempt-00/trace-payload.json',old_output/'candidate_trace.mlir',
        old_output/'lowered._hecate_golden.hevm',PROFILE)}
    producer_paths=[SCRIPT,Path(sandbox.__file__),SCRIPT.with_name('seal_cpu_golden.py'),
                    SCRIPT.with_name('seal_artifact_gate.py'),SCRIPT.with_name('candidate_worker.py')]
    (root/'producer-snapshots').mkdir()
    snapshots={}
    for path in producer_paths:
        target=root/'producer-snapshots'/path.name
        shutil.copyfile(path,target);snapshots[str(target.relative_to(root))]=digest(target)
    runtime_hashes={str(p):digest(p) for p in (BUILD/'bin/hecate-opt',BUILD/'lib/libSEAL_HEVM.so',
                                              KEY_BUILD/'seal_golden_keys',KEY_BUILD/'libseal_golden_metadata.so')}
    with np.load(SOURCE_RUN/'arrays.npz',allow_pickle=False) as data:
        inputs,reference=data['inputs'].copy(),data['reference'].copy()
    require(inputs.shape==(4,4) and reference.shape==(4,4),'Unexpected source arrays')
    independent=sum(np.roll(inputs,-i,axis=-1) for i in range(4))
    np.testing.assert_allclose(reference,independent,atol=1e-15,rtol=0)
    shutil.copyfile(SOURCE_RUN/'arrays.npz',root/'reference-arrays.npz')
    payload=root/'trace-payload.json'
    shutil.copyfile(SOURCE_RUN/'attempt-00/trace-payload.json',payload)
    require(json.loads(payload.read_text())['request']['model']['id']=='construction-sumslots4',
            'Only the frozen sumslots4 program is in this diagnostic')
    report=dict(status='running',purpose='controlled_waterline_experiment',
        source_run=str(SOURCE_RUN),source_hashes=originals,producer_snapshots=snapshots,
        runtime_hashes=runtime_hashes,waterlines=list(SCALES),planned_key_sets=TRIALS,
        original_failure_preserved=True,production_waterline=40,production_profile_changed=False,
        threshold=dict(atol=1e-5,rtol=1e-4),threshold_changed=False,agent_calls=0,
        reference_arrays_sha256=digest(root/'reference-arrays.npz'),payload_sha256=digest(payload),
        source_request_reused_for_io_only=True,
        llm_generation_validated=False,poseidon_gpu_validated=False,
        all_semantics_proven=False,compiled=[],trials=[])
    dump(root/'report.json',report)
    baseline_ops=None
    for scale in SCALES:
        out=root/('compiled-'+str(scale));out.mkdir()
        shutil.copyfile(old_output/'candidate_trace.mlir',out/'candidate_trace.mlir')
        for file in old_output.glob('*.cst'):
            require(file.name in original['attempts'][0]['artifact_hashes'],'Unfrozen constants')
            shutil.copyfile(file,out/file.name)
        command=['/hecate-opt','/out/candidate_trace.mlir','--eva','--ckks-config=/profile.json',
                 '--waterline='+str(scale),'--enable-debug-printer','--mlir-disable-threading',
                 '--verify-each','-o','/out/lowered.mlir']
        require(native(payload,out,command,root/('compile-'+str(scale)+'.log'),seconds=90)==0,
                'Experimental compile failed')
        raw=(out/'lowered._hecate_golden.hevm').read_bytes()
        gate=inspect_artifacts(raw,(out/'_hecate_golden.cst').read_bytes(),rotation_steps=(-3,-2,-1,1,2,3))
        require(gate['arg_scale']==[scale] and gate['res_scale']==[scale] and
                gate['arg_level']==[13] and gate['res_level']==[1] and gate['initial_level']==13 and
                gate['rotation_steps']==[1,2,3] and gate['plaintext_buffers']==0,
                'Compiler changed more than the intended precision contract')
        ops=semantic_instructions(raw)
        if baseline_ops is None: baseline_ops=ops
        require(ops==baseline_ops,'Instruction sequence changed between waterlines')
        np.savez(out/'arrays.npz',inputs=inputs)  # Never expose reference to runtime.
        artifacts={p.name:digest(p) for p in out.iterdir() if p.is_file()}
        report['compiled'].append(dict(waterline=scale,directory=str(out),command=command,
                                       gate=gate,semantic_instructions=ops,hashes=artifacts))
        dump(root/'report.json',report)
    for trial in range(TRIALS):
        run=Path(tempfile.mkdtemp(prefix='seal-cpu-golden-loop-precision-',dir=WORK/'results'))
        keys=run/'private-keys';keys.mkdir(mode=0o700)
        row=dict(index=trial,run=str(run),status='running',scales=[])
        report['trials'].append(row);dump(root/'report.json',report)
        trial_report=dict(status='running',purpose=report['purpose'],key_set_index=trial,
                          source_run=str(SOURCE_RUN),agent_calls=0,waterlines=list(SCALES),scales=[])
        try:
            with native_slot(WORK/'cache/agent-native-slots',timeout=120):
                with (run/'parameters.json').open('x') as stream,(run/'keygen.log').open('x') as log:
                    code=subprocess.run([str(KEY_BUILD/'seal_golden_keys'),str(keys),
                        *map(str,original['parameters']['rotation_steps'])],
                        stdout=stream,stderr=log,timeout=90).returncode
            require(code==0,'Key generation failed')
            params=json.loads((run/'parameters.json').read_text())
            require(params==original['parameters'],'Security parameters or rotation key policy changed')
            trial_report['parameters']=params
            key_hashes={p.name:digest(p) for p in keys.iterdir()}
            trial_report['same_key_set_hashes']=key_hashes
            for template in report['compiled']:
                scale=template['waterline'];out=run/('scale-'+str(scale));out.mkdir()
                for name,value in template['hashes'].items():
                    path=Path(template['directory'])/name
                    require(digest(path)==value,'Compiled template changed')
                    shutil.copyfile(path,out/name)
                require(all(digest(keys/n)==h for n,h in key_hashes.items()),'Key set changed within a trial')
                command=[str(VENV/'bin/python'),'/app/candidate_worker.py','execute']
                require(native(payload,out,command,run/('execute-'+str(scale)+'.log'),seconds=150,keys=keys)==0,
                        'Existing SEAL HEVM execution failed')
                execution=json.loads((out/'execution.json').read_text())
                require(execution['encrypted_execution'] and not execution['bootstrap_executed'] and
                        execution['input_batches']==4,'Missing actual four-input encrypted execution')
                actual=np.load(out/'decrypted.npy',allow_pickle=False)
                comparison=compare(actual,reference,1e-5,1e-4)
                item=dict(waterline=scale,directory=str(out),execution=execution,comparison=comparison,
                          decrypted_sha256=digest(out/'decrypted.npy'),execution_sha256=digest(out/'execution.json'))
                for name,value in template['hashes'].items():
                    require(digest(out/name)==value,'Runtime mutated compiled artifact/input')
                require(all(digest(keys/n)==h for n,h in key_hashes.items()),'Runtime mutated key set')
                trial_report['scales'].append(item);row['scales'].append(item)
                print('trial',trial,'waterline',scale,'passed',comparison['passed'],
                      'max_abs',comparison['max_absolute_error'],flush=True)
            trial_report['status']='passed' if all(x['comparison']['passed'] for x in row['scales']) else 'failed'
        except Exception as error:
            trial_report.update(status='failed',diagnostic=str(error))
            raise
        finally:
            row['status']=trial_report['status']
            dump(run/'report.json',trial_report)
            if (run/'parameters.json').exists() and trial_report.get('parameters'):
                from result_retention import cleanup_run
                cleanup_run(run,WORK/'results')
                row['cleanup']=json.loads((run/'key-cleanup-outcome.json').read_text())
            dump(root/'report.json',report)
    require(all(digest(Path(p))==h for p,h in originals.items()),'Original evidence/default profile changed')
    require(all(digest(Path(p))==h for p,h in runtime_hashes.items()),'Runtime changed during experiment')
    for scale in SCALES:
        comparisons=[x['comparison'] for t in report['trials'] for x in t['scales'] if x['waterline']==scale]
        report.setdefault('summary',[]).append(dict(waterline=scale,key_sets=len(comparisons),
            passed_key_sets=sum(c['passed'] for c in comparisons),input_executions=4*len(comparisons),
            compared_values=sum(c['compared_values'] for c in comparisons),
            max_absolute_error=max(c['max_absolute_error'] for c in comparisons),
            mae=sum(c['mae']*c['compared_values'] for c in comparisons)/sum(c['compared_values'] for c in comparisons)))
    report['status']='diagnostic_completed'
    dump(root/'report.json',report)
    print(json.dumps(report['summary'],indent=2))
    return 0

def main():
    require(Path.cwd().resolve()==ROOT,'Requires source root')
    if sys.argv[1:2]!=['--inside']:
        require(not sys.argv[1:],'This fixed diagnostic has no parameter overrides')
        os.umask(0o077)
        root=Path(tempfile.mkdtemp(prefix='hevm-waterline-diagnostic-',dir=WORK/'results'))
        print('Waterline diagnostic: '+str(root),flush=True)
        command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" PYTHONDONTWRITEBYTECODE=1 '+shlex.join(
            [str(VENV/'bin/python'),str(SCRIPT),'--inside',str(root)])
        return enter_nix(command,seconds=900)
    root=Path(sys.argv[2]).resolve()
    require(root.parent==WORK/'results' and root.name.startswith('hevm-waterline-diagnostic-'),
            'Invalid diagnostic output')
    return inside(root)

if __name__=='__main__': raise SystemExit(main())
