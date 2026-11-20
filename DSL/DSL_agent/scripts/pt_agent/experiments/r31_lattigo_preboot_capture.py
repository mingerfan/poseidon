"""Extract an independently replayed pre-bootstrap vector for a finite probe.

This consumes only saved packed training input and the frozen plan/bundle.
No ciphertext, model weights, Agent state or additional samples are read.
"""
import argparse
import json
import os
from pathlib import Path
import time

from .r10_paper_metrics import StageRecorder,read_events,summarize_events,write_json
from .r31_lattigo_math_worker import replay_plan


def capture_vector(plan,manifest,bundle,packed,expected_row,output):
    import numpy as np
    captured=[]
    def observer(row,vector):
        if row['ordinal']!=expected_row['ordinal']:return
        for key in ('ordinal','input','output','input_level','target_level','minimum','maximum','absmax'):
            if row[key]!=expected_row[key]:raise ValueError('Captured pre-bootstrap metadata differs: '+key)
        if vector.flags.writeable:raise ValueError('The capture callback requires a read-only view')
        vector.astype('<f8',copy=False).tofile(output)
        captured.append(dict(row))
    _,summary=replay_plan(plan,manifest,bundle,packed,bootstrap_vector_observer=observer)
    if len(captured)!=1:raise ValueError('Expected exactly one selected pre-bootstrap vector')
    return captured[0],summary


def prepare(args):
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
    args.work_dir.mkdir(parents=True,exist_ok=False);started=time.perf_counter_ns()
    rec=StageRecorder(args.work_dir/'events.jsonl')
    result=dict(round='R31',mode='independent-largest-preboot-reference-capture',status='running',error=None,
        source_checked=False,training_samples=1,heldout_samples=0,confirm_samples=0,report_samples=0,
        llm_calls=0,keygen_runs=0,real_ckks_runs=0,intermediate_decryptions=0,extra_integrity_checks=0,
        bootstrap_domain_acceptance=None,security_acceptance=None,accuracy_acceptance=None)
    try:
        import numpy as np
        with rec.span('verified_original_training_math_source'):
            math_record=json.loads(args.math_result.read_text())
            if (math_record.get('status')!='ok' or math_record.get('mathematical_equivalence_verified') is not True
                    or math_record.get('samples_completed')!=3 or len(math_record.get('comparisons',[]))!=9
                    or not all(x.get('passed') is True for x in math_record['comparisons'])):
                raise ValueError('Original three-training-probe nine-pair math acceptance required')
            sample=math_record['samples'][0]
            if sample['sample_id']!='train:0' or Path(sample['input']).resolve()!=args.input.resolve():
                raise ValueError('Use only the unchanged saved original train:0 packed input')
            rows=sample['replay_summary']['bootstrap_inputs'];selected=max(rows,key=lambda x:x['absmax'])
            if (len(rows)!=90 or selected['ordinal']!=21205 or selected['absmax']!=18.655545093784752
                    or selected['input_level']!=1 or selected['target_level']!=5):
                raise ValueError('Original selected pre-bootstrap input/range differs')
            plan=json.loads(args.plan.read_text());manifest=json.loads((args.bundle/'manifest.json').read_text())
            result['source_checked']=True
            result['source']=dict(math_result=str(args.math_result),plan=str(args.plan),bundle=str(args.bundle),
                original_packed_input=str(args.input),sample_id='train:0',selected_math_row=selected,
                ciphertext_intermediate_values_read=False,original_math_modified=False,
                scope='Independent mathematical pre-bootstrap value; the actual full ciphertext carries its own unobserved accumulated error')
        with rec.span('independent_float_preboot_capture'):
            packed=np.fromfile(args.input,dtype='<f8',count=16384)
            input_path=args.work_dir/'input.bin'
            captured,summary=capture_vector(plan,manifest,args.bundle,packed,selected,input_path)
            result['captured']=captured;result['replayed_operations']=summary['execution_completed']
            vector=np.fromfile(input_path,dtype='<f8')
            if vector.shape!=(16384,) or not np.isfinite(vector).all():raise ValueError('Invalid captured values')
        with rec.span('single_native_bootstrap_probe_protocol'):
            # Preserve the native algorithm and level1 -> level5 contract.
            name=selected['input'];out=selected['output']
            by_id={v['id']:v for v in plan['values']}
            boot=next(op for op in plan['execution'] if op['ordinal']==selected['ordinal'])
            probe=dict(format_version=1,plan_id=1,target=dict(plan['target'],independent_preboot_probe=True),
                values=[by_id[name],by_id[out]],external_inputs=[name],initialization=[],execution=[boot],
                finalization=[],final_outputs=[out],plaintext_bundle=dict(id='r31-largest-preboot-probe',version=1))
            bundle=args.work_dir/'fixture.bundle';(bundle/'data').mkdir(parents=True)
            write_json(bundle/'manifest.json',dict(bundle_format_version=1,version=1,
                bundle_id=probe['plaintext_bundle']['id'],blobs=[]))
            plan_path=args.work_dir/'bootstrap-plan.json';write_json(plan_path,probe)
            expected=args.work_dir/'expected.json'
            write_json(expected,dict(indices=list(range(16384)),values=vector.tolist(),
                scope='Independent original train:0 mathematical vector immediately before ordinal21205; one native bootstrap identity, final decrypt only. Fresh input scale43; full-run encrypted input values were not read.'))
            result.update(plan=str(plan_path),bundle=str(bundle),input=str(input_path),expected=str(expected),
                expected_encryption_attempts=1,expected_bootstrap_attempts=1,expected_final_decryption_attempts=1)
        result['status']='ok'
    except BaseException as exc:result.update(status='error',error=type(exc).__name__+': '+str(exc))
    result['elapsed_ns']=time.perf_counter_ns()-started;result['timing']=summarize_events(read_events(args.work_dir/'events.jsonl'))
    write_json(args.work_dir/'capture-result.json',result);return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','bundle','input','math-result','work-dir'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args(argv);r=prepare(args);print(json.dumps(dict(status=r['status'],error=r['error'])))
    return 0 if r['status']=='ok' else 1


if __name__=='__main__':raise SystemExit(main())
