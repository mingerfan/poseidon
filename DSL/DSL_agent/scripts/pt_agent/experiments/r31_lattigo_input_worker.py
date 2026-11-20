"""Prepare one saved real training input for the native N16 bootstrap probe.

All sources are explicit argv paths. This worker reads no registry, RESULTS
configuration, weights, heldout/confirmation/report samples or credentials.
The reference is the exact bootstrap identity followed by rotate(+1) and add.
"""
import argparse
import ast
import json
import os
from pathlib import Path
import time

from .r10_paper_metrics import StageRecorder, read_events, summarize_events, write_json

MODEL='pt_agent.experiments.r13_common_models:resnet20_silu'


def _same_path(left,right):
    return Path(left).resolve()==Path(right).resolve()


def check_sources(args):
    r13=json.loads(args.r13_result.read_text());compile_result=json.loads(args.compilation_result.read_text())
    if r13.get('round')!='R13' or r13.get('status')!='ok' or r13.get('model')!=MODEL:
        raise ValueError('Original successful R13 ResNet source is required')
    artifacts=r13['artifacts']
    for field,path in (('training_probes',args.probes),('training_probe_source',args.probe_source)):
        if not _same_path(artifacts[field],path):raise ValueError('Saved R13 training source path differs')
    if compile_result.get('round')!='R29' or compile_result.get('system')!='agent-pars' or compile_result.get('status')!='ok':
        raise ValueError('Original successful frozen Agent compilation is required')
    if not _same_path(compile_result['source_model_info'],Path(artifacts['model_info'])/'model_info.json'):
        raise ValueError('Compiled model and R13 ModelInfo differ')
    source=compile_result['source']
    if source.get('independent_audit_passed') is not True:
        raise ValueError('Compiled source lacks its independent frozen quality audit')
    retained=Path(source['program']);original=json.loads(Path(source['source_result']).read_text())
    audit_path=Path(source['source_result']).parent/'post-run-quality-audit.json'
    audit=json.loads(audit_path.read_text());worker=original.get('worker') or {}
    if (original.get('status')!='ok' or original.get('model')!='resnet20_silu'
            or worker.get('model')!='resnet20_silu' or worker.get('frozen') is not True
            or (worker.get('quality_acceptance') or {}).get('passed') is not True
            or audit.get('status')!='quality_accepted' or audit.get('formal_frozen') is not True
            or audit.get('model')!='resnet20_silu' or audit.get('strict_math_passed') is not True
            or audit.get('quality_acceptance_passed') is not True
            or not _same_path(worker.get('source_model_info',''),compile_result['source_model_info'])
            or not _same_path(audit['source_result'],source['source_result'])
            or not _same_path(audit['selected_program'],retained)):
        raise ValueError('Original Agent quality/freeze provenance is incomplete')
    if ast.dump(ast.parse(args.program.read_text()),include_attributes=False)!=ast.dump(ast.parse(retained.read_text()),include_attributes=False):
        raise ValueError('Compiled program semantics differ from the frozen program')
    probe_source=json.loads(args.probe_source.read_text())
    if (probe_source.get('split')!='train' or probe_source.get('indices')!=[0,1,2]
            or probe_source.get('sample_ids')!=['train:0','train:1','train:2']):
        raise ValueError('The three original saved training probe identifiers differ')
    return dict(model=MODEL,r13_result=str(args.r13_result),compilation_result=str(args.compilation_result),
        model_info=compile_result['source_model_info'],program=str(args.program),frozen_program=str(retained),
        independent_audit=str(audit_path),probes=str(args.probes),probe_source=str(args.probe_source),
        sample_id='train:0',sample_index=0,split='train',normalization=probe_source['normalization'],
        normalization_reapplied=False,source_samples_reconstructed=False,
        reference_scope='Native bootstrap identity on the unchanged saved normalized training input; then logical rotate(+1) and add 0.125. No model accuracy or candidate-selection measurement.')


def make_bootstrap_plan(original_target, *, bootstraps=1, branch=False):
    if bootstraps not in (1,2,3):raise ValueError('Native probe uses one to three bootstraps')
    if branch and bootstraps!=1:raise ValueError('The SSA ownership probe uses one native bootstrap')
    def value(name,level,kind='ciphertext'):
        return dict(id=name,kind=kind,context='qwen-bootstrap48',level=level,scale_log2=43,
            components=1 if kind=='plaintext' else 2,ntt=True,place=dict(kind='host',rank=0))
    place=dict(kind='host',rank=0)
    values=[value('x',1),value('offset',5,'plaintext')];execution=[];current='x'
    for index in range(bootstraps):
        if index:
            down='down'+str(index);values.append(value(down,1))
            execution.append(dict(kind='compute',ordinal=len(execution)+1,op='mod_switch',inputs=[current],
                output=down,place=place,attrs=dict(target_level=1)))
            current=down
        refreshed='refreshed'+str(index);values.append(value(refreshed,5))
        execution.append(dict(kind='compute',ordinal=len(execution)+1,op='boot',inputs=[current],
            output=refreshed,place=place,attrs=dict(implementation='native',
            operator_profile='original-plan-level1-to5-adapter-probe',target_level=5,
            target_scale_log2=43,target_components=2)))
        current=refreshed
    values.append(value('rotated',5))
    execution.append(dict(kind='compute',ordinal=len(execution)+1,op='rotate',inputs=[current],
        output='rotated',place=place,attrs=dict(steps=1)))
    current='rotated';final_level=5
    if branch:
        # Read the exact original SSA input after bootstrap has completed.
        # Bring the refreshed branch down to its original level before addition.
        values.extend([value('refreshed-down',1),value('branch-sum',1)])
        execution.extend([dict(kind='compute',ordinal=len(execution)+1,op='mod_switch',inputs=[current],
            output='refreshed-down',place=place,attrs=dict(target_level=1)),
            dict(kind='compute',ordinal=len(execution)+2,op='add_cc',inputs=['refreshed-down','x'],
            output='branch-sum',place=place)])
        current='branch-sum';final_level=1;values[1]['level']=1
    values.append(value('y',final_level))
    execution.append(dict(kind='compute',ordinal=len(execution)+1,op='add_cp',inputs=[current,'offset'],output='y',place=place))
    return dict(format_version=1,plan_id=1,target=original_target,values=values,
        external_inputs=['x'],initialization=[dict(kind='encode',ordinal=0,output='offset',
            payload=dict(kind='bundle',content='fixture:offset'))],execution=execution,
        finalization=[],final_outputs=['y'],plaintext_bundle=dict(id='r31-training-bootstrap-probe',version=1))


def prepare(args):
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[name]='1'
    args.work_dir.mkdir(parents=True,exist_ok=False)
    recorder=StageRecorder(args.work_dir/'events.jsonl');began=time.perf_counter_ns()
    result=dict(status='running',mode=args.mode,source_checked=False,training_samples=0,
        probe_load_attempts=0,probe_load_completed=0,pack_attempts=0,pack_completed=0,
        independent_reference_attempts=0,independent_reference_completed=0,
        model_weight_reads=0,model_forwards=0,llm_calls=0,keygen_runs=0,real_ckks_runs=0,
        heldout_samples=0,confirm_samples=0,report_samples=0,extra_integrity_checks=0,
        security_acceptance=None,accuracy_acceptance=None,error=None)
    try:
        with recorder.span('original_source_provenance'):
            result['source']=check_sources(args);result['source_checked']=True
            if args.branch:
                result['source']['reference_scope']='Native bootstrap identity on the unchanged saved normalized train:0; rotate(+1), read the original input again, add both branches and 0.125. Only final decrypt.'
        import numpy as np
        from ..dsl.parser import parse
        from ..lower import pack_inputs
        with recorder.span('parse_original_pack'):
            program=parse(args.program.read_text())
            if (program.header.slots!=16384 or len(program.params)!=1
                    or program.params[0].shape!=(1,3,32,32)):
                raise ValueError('The saved original ResNet input signature differs')
            packs=[op for op in program.ops if op.kind=='pack']
            if len(packs)!=1 or packs[0].attrs.get('layout')!='mux' or packs[0].attrs.get('gap')!=1:
                raise ValueError('The original mux/gap1 input packing differs')
        result['probe_load_attempts']+=1
        with recorder.span('saved_train0_read'):
            with np.load(args.probes,allow_pickle=False) as archive:
                samples=archive['input'];indices=archive['indices']
            if (samples.dtype!=np.float64 or samples.shape!=(3,3,32,32)
                    or indices.tolist()!=[0,1,2] or not np.isfinite(samples[0]).all()):
                raise ValueError('Saved original training input dtype/shape/indices/finite check failed')
            image=samples[0:1].copy();del samples
            result['probe_load_completed']+=1;result['training_samples']=1
        result['pack_attempts']+=1
        with recorder.span('original_pack_inputs'):
            blocks=pack_inputs(program,[image])
            if len(blocks)!=1 or blocks[0].shape!=(1,16384) or not np.isfinite(blocks[0]).all():
                raise ValueError('Original packing produced an unsupported ciphertext input')
            packed=np.asarray(blocks[0][0],dtype='<f8');input_path=args.work_dir/'input.bin'
            input_path.write_bytes(packed.tobytes());result['pack_completed']+=1
            result['packing']=dict(function='pt_agent.lower.pack_inputs',layout='mux',gap=1,logical_slots=16384,
                shape=list(image.shape),scalar_multiplier=1,period=16384)
        result['independent_reference_attempts']+=1
        with recorder.span('independent_identity_rotate_add_reference'):
            expected=np.roll(packed,-1)+.125
            if args.branch:expected=np.roll(packed,-1)+packed+.125
            if not np.isfinite(expected).all():raise ValueError('Nonfinite independent bootstrap reference')
            expected_path=args.work_dir/'expected.json'
            write_json(expected_path,dict(indices=list(range(16384)),values=expected.tolist(),scope=result['source']['reference_scope']))
            result['independent_reference_completed']+=1
        with recorder.span('primitive_plan_and_plaintext'):
            # This generated primitive target carries the original target ID as provenance.
            original_target=dict(target_id='poseidon-ckks-cpu',adapter_probe=True,
                source_compilation_result=str(args.compilation_result))
            plan=make_bootstrap_plan(original_target,bootstraps=args.bootstrap_count,branch=args.branch);plan_path=args.work_dir/'bootstrap-plan.json'
            bundle=args.work_dir/'fixture.bundle';(bundle/'data').mkdir(parents=True)
            raw=np.full(16384,.125,dtype='<f8').tobytes();(bundle/'data/offset.bin').write_bytes(raw)
            write_json(bundle/'manifest.json',dict(bundle_format_version=1,bundle_id=plan['plaintext_bundle']['id'],
                version=1,blobs=[dict(content='fixture:offset',byte_length=len(raw))]))
            write_json(plan_path,plan)
            result.update(bootstrap_count=args.bootstrap_count,ssa_branch_probe=args.branch,
                plan=str(plan_path),bundle=str(bundle),input=str(input_path),expected=str(expected_path))
        result['status']='ok'
    except BaseException as exc:result.update(status='error',error=type(exc).__name__+': '+str(exc))
    result['elapsed_ns']=time.perf_counter_ns()-began
    result['timing']=summarize_events(read_events(args.work_dir/'events.jsonl'))
    write_json(args.work_dir/'input-result.json',result)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=('bootstrap-identity',),required=True)
    p.add_argument('--bootstrap-count',type=int,choices=(1,2,3),default=1)
    p.add_argument('--branch',action='store_true',help='Read original input again after the native bootstrap')
    for name in ('program','r13-result','probes','probe-source','compilation-result','work-dir'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args(argv);result=prepare(args)
    print(json.dumps(dict(status=result['status'],error=result['error'],out=str(args.work_dir))))
    return 0 if result['status']=='ok' else 1

if __name__=='__main__':raise SystemExit(main())
