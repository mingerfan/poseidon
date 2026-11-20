"""Replay saved mathematical probes and expose all native-bootstrap messages.

No ModelInfo/checkpoint/sample import, compiler, key or encrypted operation.
"""
import argparse
import json
import os
from pathlib import Path
import time

from .r10_paper_metrics import StageRecorder,read_events,summarize_events,write_json
from .r31_lattigo_math_worker import replay_plan,check_math


def run(args):
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[name]='1'
    import numpy as np
    args.work_dir.mkdir(parents=True,exist_ok=False);vectors=args.work_dir/'vectors';vectors.mkdir()
    started=time.perf_counter_ns();rec=StageRecorder(args.work_dir/'events.jsonl')
    result=dict(round='R31',mode='all-prebootstrap-plaintext-message-domain',status='running',
        error=None,replay_attempts=0,replay_completed=0,vectors_completed=0,comparisons=[],
        rows=[],keygen_runs=0,encryptions=0,bootstraps=0,decryptions=0,
        llm_calls=0,extra_integrity_checks=0,bootstrap_domain_acceptance=None)
    def save():write_json(args.work_dir/'domain-math-result.json',result)
    def check():
        if time.perf_counter_ns()-started>args.seconds*1e9:raise TimeoutError('Plaintext-domain replay time exhausted')
    save()
    try:
        with rec.span('original_saved_math_sources'):
            old=json.loads(args.math_result.read_text());plan=json.loads(args.plan.read_text())
            manifest=json.loads((args.bundle/'manifest.json').read_text())
            if old.get('status')!='ok' or old.get('mathematical_equivalence_verified') is not True:
                raise ValueError('Successful original independent three-layer math source required')
            samples=old['samples']
            if [s['sample_id'] for s in samples]!=['train:0','train:1','train:2']:
                raise ValueError('Expected only original three saved training inputs')
            if len(old.get('comparisons',[]))!=9 or not all(x.get('passed') is True for x in old['comparisons']):
                raise ValueError('Original nine mathematical comparisons are incomplete')
            if len(plan['execution'])!=17063:raise ValueError('Original frozen compiled plan differs')
            result['source']=dict(plan=str(args.plan),bundle=str(args.bundle),math_result=str(args.math_result),
                frozen_program=old['source']['frozen_program'],original_normalization=old['source']['normalization'],
                source_inputs=[s['input'] for s in samples],normalization_reapplied=False)
        for sample in samples:
            check();sample_id=sample['sample_id'];original_rows=sample['replay_summary']['bootstrap_inputs']
            if len(original_rows)!=90:raise ValueError('Original sample did not contain ninety bootstraps')
            raw=np.fromfile(sample['input'],dtype='<f8')
            if raw.shape!=(16384,) or not np.isfinite(raw).all():raise ValueError('Saved packed training input differs')
            local=[]
            def capture(row,vector):
                check();index=len(local);original=original_rows[index]
                if any(row[k]!=original[k] for k in ('ordinal','input','output','input_level','target_level','minimum','maximum','absmax')):
                    raise ValueError('Replayed original preboot statistics differ')
                path=vectors/(sample_id.replace(':','-')+'-'+str(row['ordinal'])+'.bin')
                vector.astype('<f8',copy=False).tofile(path)
                mean=float(np.mean(vector));mean_abs=float(np.mean(np.abs(vector)))
                saved=dict(path=str(path.resolve()),sample=sample_id,ordinal=row['ordinal'],
                    input=row['input'],output=row['output'],input_level=row['input_level'],
                    absmax=row['absmax'],mean=mean,mean_abs=mean_abs,
                    rms=float(np.sqrt(np.mean(vector*vector))),nonzero=int(np.count_nonzero(vector)),
                    normalized_fourier_coefficient_abs_bound=mean_abs,
                    normalized_fourier_dc=mean,
                    bound_scope='Every normalized canonical inverse-embedding coefficient has abs <= mean(abs(slots)); DC equals mean(slots). Public native Encode checked separately.')
                local.append(saved);result['rows'].append(saved);result['vectors_completed']+=1;save()
            result['replay_attempts']+=1;save()
            with rec.span('original_runtime_plan_float_replay',sample=sample_id):
                output,summary=replay_plan(plan,manifest,args.bundle,raw,checkpoint=check,
                    bootstrap_vector_observer=capture)
                result['replay_completed']+=1
            if len(local)!=90 or summary['execution_completed']!=17063:raise ValueError('Full original replay incomplete')
            reference=json.loads(Path(sample['expected']).read_text());indices=reference['indices']
            comparison=check_math(np.asarray(reference['values'],dtype=np.float64),output[indices])
            comparison['sample_id']=sample_id;result['comparisons'].append(comparison);save()
            if not comparison['passed']:raise ValueError('Original strict final mathematical comparison failed')
        if result['vectors_completed']!=270:raise ValueError('Expected all270 preboot messages')
        write_json(args.work_dir/'vectors.json',dict(vectors=result['rows'],count=270,
            scope='Original three saved training inputs, original frozen RuntimePlan/bundle, before each of90 mathematical identity bootstraps; no ciphertext decryption'))
        result.update(status='ok',vectors_manifest=str(args.work_dir/'vectors.json'),
            mean_abs_max=max(r['mean_abs'] for r in result['rows']),
            mean_abs_worst=max(result['rows'],key=lambda r:r['mean_abs']),
            abs_mean_max=max(abs(r['mean']) for r in result['rows']),
            conservative_coefficient_message_domain_all=(max(r['mean_abs'] for r in result['rows'])<1))
    except BaseException as exc:result.update(status='error',error=type(exc).__name__+': '+str(exc))
    result['elapsed_ns']=time.perf_counter_ns()-started
    result['timing']=summarize_events(read_events(args.work_dir/'events.jsonl'));save();return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','bundle','math-result','work-dir'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--seconds',type=float,default=300);args=p.parse_args(argv);r=run(args)
    print(json.dumps(dict(status=r['status'],error=r['error'],vectors_completed=r['vectors_completed'])))
    return 0 if r['status']=='ok' else 1


if __name__=='__main__':raise SystemExit(main())
