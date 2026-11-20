"""Independent original43 RuntimePlan math against current fresh123 train:0.

The saved input and independently prepared Tensor logits are the only numeric
inputs. Boot/modswitch/rescale/relinearize are ideal identities in this oracle.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import resource
import shutil
import signal
import sys
import time

from . import r31_fresh_cnn_input as fresh
from .completion_metrics import artifact_bytes, persist_case_completion
from .r10_paper_metrics import StageRecorder, read_events, run_process, summarize_events, write_json

ROUND='R31-FRESH-PLAN-MATH'
MODE='fresh123-original43-single-training-plan-math'
MODULE='pt_agent.experiments.r31_fresh_plan_math'
SCOPE='Current independently frozen freshR30 ResNet20 and original43 RuntimePlan on saved train:0 only; boot/mod_switch/rescale/relinearize are mathematical identities; no new model forward, CKKS or quality selection.'
SECONDS=900
OUTER_SECONDS=960
CASE_BYTES=4*1024**3
EXPECTED_EXECUTION=17863
EXPECTED_BOOTS=123
ZERO_FIELDS=('api_calls','model_forwards','npz_reads','checkpoint_reads','source_image_reads',
    'compiler_runs','keygen_runs','encryption_runs','native_bootstrap_runs','decryption_runs',
    'heldout_samples','confirm_samples','report_samples','downloads','installations','extra_integrity_checks')


def read(path):return json.loads(Path(path).read_text())


def same(a,b):return isinstance(a,(str,Path)) and isinstance(b,(str,Path)) and Path(a).resolve()==Path(b).resolve()


def source_gate(input_prepare,source_case,archive_manifest):
    prepared=read(input_prepare);worker=prepared.get('worker') or {}
    if (prepared.get('round')!=fresh.ROUND or prepared.get('mode')!=fresh.MODE or prepared.get('model')!=fresh.MODEL
            or prepared.get('status')!='ok' or not fresh._success(worker)):
        raise ValueError('The accepted fresh input and new Tensor preparation are required')
    current=fresh.prepare_source(Path(source_case),fresh.MODEL);gate=current['gate']
    retained=prepared.get('source') or {};old=retained.get('gate') or {};inner=(worker.get('source') or {}).get('gate') or {}
    for key in ('source_case','model_info','program','probes','probe_source'):
        if not same(old.get(key),gate[key]) or not same(inner.get(key),gate[key]):raise ValueError('Fresh preparation source identity differs: '+key)
    if (old.get('factory')!=gate['factory'] or gate['factory']!='pt_agent.experiments.r13_common_models:resnet20_silu'
            or old.get('normalization')!=gate['normalization'] or old.get('original_sources')!=gate['original_sources']
            or not same(retained.get('runtime_plan'),current['runtime_plan'])
            or not same(worker['source'].get('runtime_plan'),current['runtime_plan'])):
        raise ValueError('Current factory/program/normalization or original plan differs')
    plan=Path(current['runtime_plan']).resolve();bundle=plan.with_name(plan.name.removesuffix('.runtime-plan.json')+'.bundle')
    archive=read(archive_manifest);entries=[x for x in archive.get('files',[]) if x.get('kind')=='runtime_bundle']
    if (len(entries)!=1 or not same(archive.get('case_result'),Path(source_case)/'result.json')
            or not same(entries[0].get('path'),bundle)):
        raise ValueError('The closed recovery inventory must identify this canonical fresh bundle')
    work=Path(input_prepare).resolve().parent/'work'
    if not same(worker.get('input'),work/'input.bin') or not same(worker.get('expected'),work/'expected.json'):
        raise ValueError('Use only the accepted preparation input and expected artifacts')
    fresh.check_outputs(worker,work)
    expected=read(work/'expected.json')
    if (expected.get('sample_id')!='train:0' or expected.get('old_expected_values_used') is not False
            or expected.get('criterion')!=fresh.CRITERION or expected.get('values')!=worker.get('tensor_values')
            or expected.get('tensor_values')!=worker.get('tensor_values') or expected.get('indices')!=list(range(10))):
        raise ValueError('Current independently prepared train:0 Tensor reference differs')
    inventory=current['source_metadata']['plan']
    if inventory['execution_count']!=EXPECTED_EXECUTION or inventory['bootstrap_count']!=EXPECTED_BOOTS:
        raise ValueError('Require the current original123-bootstrap plan')
    return dict(input_prepare=str(Path(input_prepare).resolve()),source_case=str(Path(source_case).resolve()),
        plan=str(plan),bundle=str(bundle),archive_manifest=str(Path(archive_manifest).resolve()),
        input=str(work/'input.bin'),expected=str(work/'expected.json'),program=gate['program'],factory=gate['factory'],
        model_info=gate['model_info'],normalization=gate['normalization'],inventory=inventory,
        reference_values=expected['values'],decode_indices=expected['indices'],sample_id='train:0',scope=SCOPE)


def worker(args):
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
    args.work_dir.mkdir(parents=True,exist_ok=False);began=time.perf_counter_ns();rec=StageRecorder(args.work_dir/'events.jsonl')
    result=dict(status='running',scope=SCOPE,error=None,source_checked=False,replay_attempts=0,replay_completed=0,
        checkpoint_callbacks=0,bootstrap_observations=[],op_completed=None,op_counts=None,math_passed=False,
        mathematical_equivalence_verified=False,training_samples=0,input_read_attempts=0,input_read_completed=0,
        reference_read_attempts=0,reference_read_completed=0,bootstrap_domain_acceptance=None,
        accuracy_acceptance=None,security_acceptance=None,actual_ckks_latency=None)
    result.update(dict.fromkeys(ZERO_FIELDS,0))
    def checkpoint():
        result['checkpoint_callbacks']+=1
        if time.perf_counter_ns()-began>SECONDS*10**9:raise TimeoutError('Fresh plan math wall allowance exhausted')
        if artifact_bytes(args.work_dir.parent)>CASE_BYTES or shutil.disk_usage(args.work_dir).free<10*1024**3:raise RuntimeError('Fresh plan math disk allowance exhausted')
        write_json(args.work_dir/'worker-result.json',result)
    def observe(row):result['bootstrap_observations'].append(dict(row));checkpoint()
    old=signal.getsignal(signal.SIGALRM)
    def expired(signum,frame):raise TimeoutError('Fresh plan math expired')
    signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,SECONDS)
    try:
        with rec.span('fresh_original_plan_independent_math'):
            source=read(args.source_record)
            for key in ('plan','bundle','input','expected','archive_manifest','program'):
                if not same(getattr(args,key),source[key]):raise ValueError('Explicit worker source path differs: '+key)
            import numpy as np
            from .r31_lattigo_math_worker import replay_plan,check_math
            from .r31_lattigo_runtime_plan import plan_inventory
            from .dacapo_native_plan_math import check_restored_bundle
            plan=read(args.plan);manifest=read(args.bundle/'manifest.json');inventory=plan_inventory(plan,manifest)
            if (inventory['execution_count']!=EXPECTED_EXECUTION or inventory['bootstrap_count']!=EXPECTED_BOOTS
                    or {v['scale_log2'] for v in plan['values']}!={43,86}):raise ValueError('Only unchanged original43 fresh123 plan is accepted')
            result['restored_bundle']=check_restored_bundle(dict(manifest=manifest),args.bundle,args.archive_manifest)
            result.update(source_checked=True,source=source,inventory=inventory);checkpoint()
            result['input_read_attempts']=1;checkpoint();packed=np.fromfile(args.input,dtype='<f8')
            if packed.shape!=(16384,) or not np.isfinite(packed).all():raise ValueError('Saved input must be finite original16384slots')
            result.update(input_read_completed=1,training_samples=1,reference_read_attempts=1);checkpoint()
            expected=read(args.expected)
            if (expected.get('values')!=source['reference_values'] or expected.get('indices')!=source['decode_indices']
                    or expected.get('sample_id')!='train:0' or expected.get('old_expected_values_used') is not False):
                raise ValueError('Worker reference differs from the accepted new frozen Tensor output')
            result['reference_read_completed']=1
            result['replay_attempts']=1;checkpoint()
            with rec.span('original43_last_use_replay'):
                output,summary=replay_plan(plan,manifest,args.bundle,packed,bootstrap_observer=observe,checkpoint=checkpoint)
            summary['reference_scope']=SCOPE
            result.update(replay_completed=1,op_completed=summary['execution_completed'],op_counts=summary['op_counts'],replay_summary=summary)
            if (summary['execution_completed']!=EXPECTED_EXECUTION or len(summary['bootstrap_inputs'])!=EXPECTED_BOOTS
                    or summary['op_counts']!=source['inventory']['op_counts']):raise ValueError('Every current original operation and ideal bootstrap must complete')
            values=output[expected['indices']][None,:];reference=np.asarray(expected['values'],dtype=np.float64)[None,:]
            compared=check_math(reference,values);result.update(comparison=compared,decoded_values=values.tolist(),math_passed=compared['passed'])
            write_json(args.work_dir/'selected-logits.json',dict(sample_id='train:0',shape=[1,10],dtype='float64',values=values.tolist(),reference=reference.tolist(),comparison=compared,scope=SCOPE))
            if not compared['passed']:raise ValueError('Current frozen Tensor and original RuntimePlan math differ')
            proof=dict(status='strict_math_passed',source_plan=source['plan'],current_program=source['program'],
                input_prepare=source['input_prepare'],input=source['input'],reference=source['expected'],
                source_case=source['source_case'],source_factory=source['factory'],sample_ids=['train:0'],
                operation_count=EXPECTED_EXECUTION,ideal_bootstrap_count=EXPECTED_BOOTS,criterion=fresh.CRITERION,
                absolute_tolerance=1e-12,relative_tolerance=1e-12,original_plan_modified=False,
                precision_mapping_applied=False,scope=SCOPE)
            write_json(args.work_dir/'source-math-proof.json',proof)
            result.update(status='ok',mathematical_equivalence_verified=True,source_math_proof=str(args.work_dir/'source-math-proof.json'))
    except BaseException as exc:result.update(status='error',error=type(exc).__name__+': '+str(exc))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)
        result['elapsed_ns']=time.perf_counter_ns()-began;result['worker_maxrss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        write_json(args.work_dir/'worker-result.json',result)
    return result


def execute(args):
    from .. import paths
    from .r30_frozen_cnn_report import clean
    out=args.out.resolve()
    if out.parent!=paths.RESULTS_DIR.resolve() or out.exists():raise ValueError('Use a fresh configured results case')
    began=time.perf_counter_ns();out.mkdir();rec=StageRecorder(out/'events.jsonl')
    result=dict(round=ROUND,model=fresh.MODEL,mode=MODE,status='running',error=None,worker=None,process=None,
        scope=SCOPE,security_acceptance=None,accuracy_acceptance=None,formal_latency=None,artifact_bytes=0)
    try:
        with rec.span('fresh_plan_math_preparation'):
            source=source_gate(args.input_prepare,args.source_case,args.archive_manifest);result['source']=source
            write_json(out/'source-record.json',source);shutil.copyfile(Path(__file__),out/'entry-source.py')
            shutil.copyfile(Path(__file__).with_name('r31_lattigo_math_worker.py'),out/'replay-source.py')
            command=[str(args.python),'-B','-m',MODULE,'worker','--work-dir',str(out/'worker'),'--source-record',str(out/'source-record.json')]
            for key in ('plan','bundle','input','expected','archive_manifest','program'):command+=['--'+key.replace('_','-'),source[key]]
            result['argv']=command
            with rec.span('bounded_single_training_replay'):
                result['process']=run_process(command,cwd=paths.REPO/'DSL/DSL_agent/scripts',log=out/'worker.log',
                    resources=out/'resources.jsonl',timeout=930,mem_gib=4,fsize_gib=8,
                    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1'))
                if (out/'worker/worker-result.json').is_file():result['worker']=read(out/'worker/worker-result.json')
                if not clean(result['process']) or not result['worker'] or result['worker']['status']!='ok' or not result['worker']['mathematical_equivalence_verified']:
                    raise RuntimeError('Current original plan mathematical acceptance failed')
            result['status']='ok'
    except BaseException as exc:result.update(status='error',error=type(exc).__name__+': '+str(exc))
    finally:
        if (out/'worker/worker-result.json').is_file():result['worker']=read(out/'worker/worker-result.json')
        events=read_events(out/'events.jsonl');parent=next((r['event_id'] for r in events if r.get('type')=='begin' and r.get('stage_id')=='bounded_single_training_replay'),None)
        child=read_events(out/'worker/events.jsonl')
        for row in child:
            if row.get('parent_id') is None:row['parent_id']=parent
        result['timing']=summarize_events(events+child);result['recorded_pre_persistence_seconds']=(time.perf_counter_ns()-began)/1e9
        result['metrics']=dict(status=result['status'],worker=result['worker'],security_acceptance=None,formal_latency=None)
        for _ in range(20):
            write_json(out/'result.json',result);size=artifact_bytes(out)
            if size+65536>CASE_BYTES:result.update(status='error',error='Terminal case allowance exhausted');result['metrics']['status']='error'
            if size==result['artifact_bytes']:break
            result['artifact_bytes']=size
        else:raise RuntimeError('Fresh math terminal size did not converge')
        from ..registry import git_rev
        row=dict(run_id=out.name,round=ROUND,model=fresh.MODEL,mode=MODE,git_rev=git_rev(),metrics=result['metrics'],artifacts_dir=str(out),ts=datetime.now(timezone.utc).isoformat())
        with (paths.RESULTS_DIR/'registry.jsonl').open('a') as stream:
            fcntl.flock(stream.fileno(),fcntl.LOCK_EX);stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
        persist_case_completion(out,round_label=ROUND,started_ns=began,status=result['status'],timing=result['timing'],required_artifact_bytes=artifact_bytes(out))
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='phase',required=True)
    outer=sub.add_parser('prepare')
    for name in ('out','input-prepare','source-case','archive-manifest'):outer.add_argument('--'+name,type=Path,required=True)
    outer.add_argument('--python',type=Path)
    inner=sub.add_parser('worker')
    for name in ('work-dir','source-record','plan','bundle','input','expected','archive-manifest','program'):inner.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args(argv)
    if args.phase=='worker':
        resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3));result=worker(args)
    else:
        from .. import paths
        if args.python is None:args.python=paths.ORBIT_ROOT.parent/'orbit-venv/bin/python'
        old=signal.getsignal(signal.SIGALRM)
        def expire(signum,frame):raise TimeoutError('Fresh plan math outer allowance exhausted')
        signal.signal(signal.SIGALRM,expire);signal.setitimer(signal.ITIMER_REAL,OUTER_SECONDS)
        try:result=execute(args)
        finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)
    print(json.dumps(dict(status=result['status'],error=result['error'])));return int(result['status']!='ok')


if __name__=='__main__':raise SystemExit(main())
