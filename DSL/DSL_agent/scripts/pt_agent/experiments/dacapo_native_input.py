"""Prepare original N17 input and physical-unit references from accepted math.

Only saved packed train:0 and its independently checked physical logits are
reused. No image, model, weight, bundle payload or native runtime is evaluated.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
from pathlib import Path
import resource
import shutil
import signal
import time

from .. import paths
from . import dacapo_native_binding as binding
from .completion_metrics import persist_case_completion
from .r10_paper_metrics import StageRecorder, read_events, summarize_events, write_json

ROUND='DACAPO-NATIVE-INPUT'
MODEL='dacapo-original-resnet20-silu'
MODE='original-plan-saved-train0-physical32-reference'
DECODER='original-physical32-first10'
INPUT_BYTES=65536*8
CASE_BYTES=4*1024**2
FREE_BYTES=10*1024**3
SECONDS=60
ZERO_FIELDS=('api_calls','model_forwards','source_image_reads','checkpoint_reads','new_packing_runs',
    'compiler_runs','keygen_runs','ckks_executions','intermediate_decryptions','downloads',
    'installations','integrity_checks','heldout_samples','confirm_samples','report_samples')


def read(path):return json.loads(Path(path).read_text())


def same(a,b):
    return isinstance(a,(str,Path)) and isinstance(b,(str,Path)) and Path(a).resolve()==Path(b).resolve()


def _logits(value):
    if (not isinstance(value,list) or len(value)!=1 or not isinstance(value[0],list)
            or len(value[0])!=10 or any(type(x) not in (int,float) or not math.isfinite(x) for x in value[0])):
        raise ValueError('Accepted physical logits must be finite [1,10]')
    return [float(x) for x in value[0]]


def source_gate(math_result,plan):
    math_result=Path(math_result).resolve();plan=Path(plan).resolve()
    record=read(math_result);worker=record.get('worker') or {};selected=record.get('selected_paths') or {}
    if (record.get('round')!='DACAPO-NATIVE-MATH' or record.get('model')!=MODEL
            or record.get('mode')!='original-dacapo-decoded-ideal-three-probe-math' or record.get('status')!='ok'
            or record.get('original_training_math_verified') is not True or record.get('full_execution_scope_ready') is not True
            or worker.get('status')!='ok' or worker.get('source_checked') is not True
            or worker.get('source_factory')!=binding.SOURCE_FACTORY or worker.get('original_plan_modified') is not False
            or worker.get('original_training_math_verified') is not True or worker.get('full_execution_scope_ready') is not True
            or worker.get('packed_forward_attempts')!=3 or worker.get('packed_forward_completed')!=3
            or worker.get('math_check_attempts')!=3 or worker.get('math_check_completed')!=3
            or worker.get('op_attempts')!=37983 or worker.get('op_completed')!=37983
            or any(worker.get(key)!=0 for key in ('api_calls','llm_calls','model_forwards','torch_model_forwards',
                'checkpoint_reads','raw_cifar_reads','heldout_samples','confirm_samples','report_samples',
                'compiler_runs','keygen_runs','ckks_executions','integrity_checks'))
            or not same(selected.get('plan'),plan)):
        raise ValueError('An accepted original-plan three-training-probe mathematical run is required')
    summary=worker.get('source_summary') or {};metadata=worker.get('program_metadata') or {}
    if (not same(summary.get('paths',{}).get('plan',{}).get('path'),plan)
            or metadata.get('execution_count')!=12661 or metadata.get('bootstrap_count')!=19
            or metadata.get('original_plan_modified') is not False or metadata.get('inline_scalar_count')!=122
            or summary.get('packing_provenance',{}).get('output_formula')!='first10 decoded slots times32, shape[1,10]'
            or summary.get('original_pack_bb32_preserved') is not True
            or summary.get('original_poly_preserved') is not True):
        raise ValueError('Original plan and physical32 decoder provenance differ')
    proof_path=Path(worker['source_math_proof']).resolve();proof=read(proof_path)
    scalar_path=Path(selected['scalar_source_proof']).resolve();scalar=read(scalar_path)
    if (proof_path!=math_result.parent/'worker/source-math-proof.json'
            or not same(summary['paths']['scalar_source_proof']['path'],scalar_path)):
        raise ValueError('Use the accepted case mathematical proof and scalar provenance paths')
    gates=binding._source_gates({},metadata.get('inline_scalar_ids',[]),scalar,proof,
        public_fixture=False,plan_path=plan)
    binding.require_execution_ready(dict(execution_scope_ready=gates['execution_scope_ready'],source_gates=gates))
    if (proof.get('status')!='strict_math_passed' or proof.get('source_factory')!=binding.SOURCE_FACTORY
            or proof.get('sample_ids')!=['train:0','train:1','train:2'] or proof.get('all_three_passed') is not True
            or proof.get('original_poly_preserved') is not True or proof.get('original_pack_bb32_preserved') is not True
            or proof.get('original_plan_modified') is not False or proof.get('absolute_tolerance')!=1e-12
            or proof.get('relative_tolerance')!=1e-12 or not binding._same_source_plan(proof.get('source_plan'),plan)
            or scalar.get('status')!='scalar_sites_verified' or scalar.get('source_factory')!=binding.SOURCE_FACTORY
            or not binding._same_source_plan(scalar.get('source_plan'),plan)):
        raise ValueError('Scalar and mathematical proofs must bind the same original plan')
    rows=worker.get('probe_rows') or []
    if len(rows)!=3:raise ValueError('Exactly three completed training probes are required')
    for index,row in enumerate(rows):
        if (row.get('sample_id')!='train:'+str(index) or row.get('probe_index')!=index
                or row.get('math',{}).get('passed') is not True
                or row.get('math',{}).get('criterion')!='1e-12+1e-12*abs(reference)'):
            raise ValueError('Every fixed training probe must pass the original mathematical criterion')
        actual,expected=_logits(row['decoded']),_logits(row['reference'])
        if any(abs(a-e)>1e-12+1e-12*abs(e) for a,e in zip(actual,expected)):
            raise ValueError('Saved ideal output and independent physical reference disagree')
    first=rows[0];packed=Path(first['packed_input']['path']).resolve()
    if (packed!=math_result.parent/'worker/packed-input-train-0.f64'
            or first['packed_input'].get('bytes')!=INPUT_BYTES or packed.stat().st_size!=INPUT_BYTES):
        raise ValueError('Reuse only the accepted saved packed train:0 input')
    return dict(math_result=str(math_result),plan=str(plan),bundle=str(Path(selected['bundle']).resolve()),
        scalar_source_proof=str(scalar_path),source_math_proof=str(proof_path),packed_input=str(packed),
        sample_id='train:0',reference=_logits(first['reference']),ideal_decoded=_logits(first['decoded']),
        decoder=DECODER,output_multiplier=32.0,
        decoder_provenance=summary['packing_provenance'],new_source_images_read=False,
        expected_reference_source='Accepted math.worker.probe_rows[0].reference from independent R24 physical polynomial output',
        old_native_expected_values_used=False,source_plan_modified=False)


def make_expected(source):
    return dict(indices=list(range(10)),values=list(source['reference']),output_multiplier=32.0,
        output_decoder=DECODER,source_plan=source['plan'],source_math_proof=source['source_math_proof'],
        source_factory=binding.SOURCE_FACTORY,sample_id='train:0',shape=[1,10],dtype='float64',
        scope='Original DaCapo first10 final decoded slots times32; accepted saved train:0 physical logits; no new model or native execution',
        expected_reference_source=source['expected_reference_source'])


def check_native_decoder(client):
    """Require observed capability from the new parameter-only binary."""
    if (client.get('status')!='ok' or client.get('mode')!='parameters'
            or client.get('final_decoder_protocol')!=1
            or client.get('supported_final_output_multipliers')!=[1,32]
            or client.get('original_final_decoder')!=DECODER
            or any(client.get(key)!=0 for key in ('keygen_api_attempts','encryption_attempts',
                'compute_attempts','final_decryption_attempts'))):
        raise ValueError('A new zero-key parameter observation with physical32 decoder capability is required')
    return dict(final_decoder_protocol=1,supported_final_output_multipliers=[1,32],
        original_final_decoder=DECODER,actual_parameter_metadata_verified=True)


def artifact_bytes(out):return sum(p.stat().st_size for p in out.rglob('*') if p.is_file())


def execute(args):
    out=Path(args.out).resolve();math_result=Path(args.math_result).resolve();plan=Path(args.plan).resolve()
    if (out.parent!=paths.RESULTS_DIR.resolve() or out.exists()
            or out==math_result.parent or out in math_result.parents or math_result.parent in out.parents
            or out==plan.parent or plan.parent in out.parents):
        raise ValueError('A fresh preparation case under the configured results root is required')
    started=time.perf_counter_ns();out.mkdir();recorder=StageRecorder(out/'events.jsonl')
    result=dict(round=ROUND,model=MODEL,mode=MODE,status='running',error=None,
        saved_input_copy_attempts=0,saved_input_copy_completed=0,reference_prepared=False,
        source_plan_modified=False,final_decoder_application_phase='Only after sole final decrypt/decode',
        security_acceptance=None,accuracy_acceptance=None,formal_latency=None,artifact_bytes=0)
    result.update(dict.fromkeys(ZERO_FIELDS,0))
    result['required_native_client']=dict(final_decoder_protocol=1,
        supported_final_output_multipliers=[1,32],original_final_decoder=DECODER,
        acceptance_scope='Require actual new binary parameter metadata; old N17 prepared binary remains historical')
    def check():
        if time.perf_counter_ns()-started>SECONDS*10**9:raise TimeoutError('Saved reference preparation wall allowance exhausted')
        if artifact_bytes(out)+16384>CASE_BYTES or shutil.disk_usage(out).free<FREE_BYTES:
            raise RuntimeError('Saved reference preparation disk allowance exhausted')
    try:
        with recorder.span('saved_original_native_reference_preparation'):
            with recorder.span('same_source_math_and_decoder_contract'):
                source=source_gate(math_result,plan);result['source']=source;check()
                write_json(out/'source-record.json',source)
            with recorder.span('copy_original_packed_training_input'):
                result['saved_input_copy_attempts']+=1
                shutil.copyfile(source['packed_input'],out/'input.bin')
                if (out/'input.bin').stat().st_size!=INPUT_BYTES:raise ValueError('Copied packed input size differs')
                result['saved_input_copy_completed']+=1;check()
            with recorder.span('save_original_physical_logits'):
                write_json(out/'expected.json',make_expected(source))
                shutil.copyfile(Path(__file__),out/'preparation-source.py')
                shutil.copyfile(Path(__file__).with_name('dacapo_native_runtime_plan.go'),out/'native-client-source.go')
                result.update(reference_prepared=True,input=str(out/'input.bin'),expected=str(out/'expected.json'))
                result['native_arguments']=['-mode','execute','-plan',source['plan'],'-bundle',source['bundle'],
                    '-input',result['input'],'-expected',result['expected'],'-scalar-site-proof',source['scalar_source_proof'],
                    '-source-math-proof',source['source_math_proof'],'-allow-bootstrap','-precision-bits','51',
                    '-bootstrap-cache','1','-check-input-ssa','-trace-key-requests','-checkpoint-interval','1000']
                check()
        result['status']='ok'
    except BaseException as exc:result.update(status='error',error=type(exc).__name__+': '+str(exc),reference_prepared=False)
    finally:
        result['timing']=summarize_events(read_events(out/'events.jsonl'))
        result['recorded_pre_persistence_seconds']=(time.perf_counter_ns()-started)/1e9
        result['worker_maxrss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result['metrics']={key:result[key] for key in ('status','reference_prepared','saved_input_copy_attempts',
            'saved_input_copy_completed','security_acceptance','accuracy_acceptance','formal_latency')+ZERO_FIELDS}
        for _ in range(20):
            write_json(out/'result.json',result);size=artifact_bytes(out)
            if size+16384>CASE_BYTES:
                result.update(status='error',error='Terminal reference preparation exceeds case allowance');result['metrics']['status']='error'
            if size==result['artifact_bytes']:break
            result['artifact_bytes']=size
        else:raise RuntimeError('Terminal reference artifact size did not converge')
        from ..registry import git_rev
        row=dict(run_id=out.name,round=ROUND,model=MODEL,mode=MODE,git_rev=git_rev(),metrics=result['metrics'],
            artifacts_dir=str(out),ts=datetime.now(timezone.utc).isoformat())
        with (paths.RESULTS_DIR/'registry.jsonl').open('a') as stream:
            fcntl.flock(stream.fileno(),fcntl.LOCK_EX);stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
        persist_case_completion(out,round_label=ROUND,started_ns=started,status=result['status'],
            timing=result['timing'],required_artifact_bytes=artifact_bytes(out))
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('out','math-result','plan'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args(argv)
    resource.setrlimit(resource.RLIMIT_AS,(1024**3,1024**3))
    def expire(signum,frame):raise TimeoutError('Saved reference preparation expired')
    signal.signal(signal.SIGALRM,expire);signal.setitimer(signal.ITIMER_REAL,SECONDS)
    try:result=execute(args)
    finally:signal.setitimer(signal.ITIMER_REAL,0)
    print(json.dumps(dict(status=result['status'],reference_prepared=result['reference_prepared'],error=result['error'])))
    return 0 if result['status']=='ok' else 1


if __name__=='__main__':raise SystemExit(main())
