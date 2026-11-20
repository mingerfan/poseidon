"""Audit local trained BERT checkpoint key coverage without inference or hashes.

The independent worker records restricted state loading and the standard HF
loading report. Tensor values stay local; result rows contain semantic metadata.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import time

from .. import paths
from .completion_metrics import artifact_bytes, persist_case_completion
from .r10_paper_metrics import StageRecorder, read_events, run_process, summarize_events, write_json

ROUND='BERT-CHECKPOINT-AUDIT'
MODEL='bert-tiny-sst2'
MODE='original-trained-state-semantic-coverage'
SECONDS=300
CASE_BYTES=16*1024**2
FREE_BYTES=10*1024**3
ZERO_FIELDS=('model_forwards','raw_sample_reads','report_samples','confirm_samples','api_calls',
    'compiler_runs','keygen_runs','ckks_executions','downloads','installations','integrity_checks')


def _read(path):return json.loads(Path(path).read_text())


def source_gate(model,prepare_result,model_info):
    model=Path(model).resolve();prepare_result=Path(prepare_result).resolve();model_info=Path(model_info).resolve()
    prepared=_read(prepare_result);info=_read(model_info);config=_read(model/'config.json')
    metadata=info.get('source',{}).get('config',{})
    checks=prepared.get('reference_checks') or []
    if (prepared.get('status')!='ok' or prepared.get('phase')!='prepare' or prepared.get('model')!=MODEL
            or prepared.get('import_completed') is not True
            or Path(prepared.get('model_dir','')).resolve()!=model_info.parent
            or Path(prepared.get('model_metadata',{}).get('unchanged_original_model','')).resolve()!=model
            or Path(info.get('source',{}).get('local_path','')).resolve()!=model
            or info.get('task')!='classification' or info.get('metric')!='classification_top1'
            or info.get('threshold')!={'acc_drop_pp':.5} or metadata.get('model_type')!='bert'
            or metadata.get('num_labels')!=2 or len(checks)!=3 or not all(r.get('passed') is True for r in checks)
            or any(prepared.get(k)!=0 for k in ('native_test_reads','report_samples','confirm_samples','llm_calls',
                'compiler_runs','keygen_runs','real_ckks_attempts','real_ckks_completed'))):
        raise ValueError('Accepted original BERT/SST-2 intake and three strict reference checks are required')
    if (config.get('model_type')!='bert' or config.get('architectures')!=['BertForSequenceClassification']
            or config.get('hidden_size')!=metadata.get('hidden_size')
            or config.get('num_hidden_layers')!=metadata.get('num_hidden_layers')):
        raise ValueError('Original local BERT config differs from the accepted intake')
    checkpoint=model/'pytorch_model.bin'
    if (not checkpoint.is_file() or checkpoint.stat().st_size<=0
            or (model/'model.safetensors').exists() or (model/'model.safetensors.index.json').exists()):
        raise ValueError('This audit follows the accepted restricted local .bin intake branch')
    return dict(model=str(model),checkpoint=str(checkpoint),checkpoint_bytes=checkpoint.stat().st_size,
        prepare_result=str(prepare_result),model_info=str(model_info),source_config=metadata,
        task=info['task'],metric=info['metric'],task_policy=info['threshold'],
        original_normalized_reference_checks=checks,original_model_changed=False)


def validate_coverage(checkpoint,loaded,loading_info,*,hidden_size,num_labels,nonpersistent=None):
    """Check complete parameter/buffer metadata; value equality is computed locally."""
    nonpersistent=nonpersistent or {}
    if not checkpoint or not loaded or not all(isinstance(k,str) for k in checkpoint.keys()|loaded.keys()):
        raise ValueError('Nonempty named state dictionaries are required')
    required=set(loaded);missing=required-set(checkpoint);extra=set(checkpoint)-required
    if missing:raise ValueError('Checkpoint misses model state keys: '+', '.join(sorted(missing)))
    if not extra<=set(nonpersistent):raise ValueError('Checkpoint contains unknown state keys: '+', '.join(sorted(extra-set(nonpersistent))))
    rows=[]
    for key in sorted(checkpoint):
        original=checkpoint[key];target=loaded[key] if key in loaded else nonpersistent[key]
        if (original.get('shape')!=target.get('shape') or original.get('dtype')!=target.get('dtype')
                or original.get('finite') is not True or target.get('finite') is not True
                or target.get('numerically_equal') is not True):
            raise ValueError('Checkpoint semantic shape/dtype/finite/value coverage failed: '+key)
        rows.append(dict(name=key,shape=original['shape'],dtype=original['dtype'],finite=True,
            numerically_equal=True,role='persistent_model_state' if key in loaded else 'verified_nonpersistent_buffer'))
    for field in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'):
        if not isinstance(loading_info.get(field),list):raise ValueError('HF loading report field is absent: '+field)
    if (loading_info['missing_keys'] or loading_info['mismatched_keys'] or loading_info['error_msgs']
            or not set(loading_info['unexpected_keys'])<=extra):
        raise ValueError('HF loading report contains unresolved missing/unexpected/mismatched/error entries')
    for key,shape in (('classifier.weight',[num_labels,hidden_size]),('classifier.bias',[num_labels])):
        if key not in loaded or key not in checkpoint or checkpoint[key].get('shape')!=shape:
            raise ValueError('Original checkpoint classifier shape/coverage failed: '+key)
    if num_labels!=2:raise ValueError('Original SST-2 classifier must retain two outputs')
    return dict(passed=True,checkpoint_key_count=len(checkpoint),persistent_model_state_count=len(loaded),
        missing_keys=[],unknown_keys=[],shape_mismatches=[],dtype_mismatches=[],nonfinite_keys=[],
        checkpoint_keys=sorted(checkpoint),model_state_keys=sorted(loaded),
        verified_nonpersistent_keys=sorted(extra),loading_info=loading_info,rows=rows,
        classifier=dict(weight='classifier.weight',bias='classifier.bias',weight_shape=[num_labels,hidden_size],
            bias_shape=[num_labels],both_original_checkpoint_keys=True,numerically_equal=True),
        semantic_scope='Every standard model parameter/persistent buffer is covered by the original state; extra legacy buffers require a same-name nonpersistent model buffer and exact shape/dtype/value equality. No digest or byte integrity assertion.')


def _runtime():
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
    import torch
    import transformers
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    return torch,transformers


def audit(args,recorder,result):
    with recorder.span('accepted_intake_source_gate'):
        source=source_gate(args.model,args.prepare_result,args.model_info);result['source']=source
    torch,transformers=_runtime()
    with recorder.span('restricted_original_checkpoint'):
        result['checkpoint_load_attempts']=1
        state=torch.load(source['checkpoint'],map_location='cpu',weights_only=True)
        if not isinstance(state,dict) or not state or not all(isinstance(k,str) and torch.is_tensor(v) for k,v in state.items()):
            raise ValueError('Original checkpoint must be a named tensor-only state_dict')
        result['checkpoint_load_completed']=1
    local=dict(local_files_only=True,trust_remote_code=False)
    with recorder.span('standard_original_hf_load_with_info'):
        config=transformers.AutoConfig.from_pretrained(source['model'],**local)
        if (config.model_type!='bert' or config.num_labels!=2
                or config.hidden_size!=source['source_config']['hidden_size']):
            raise ValueError('Standard HF config differs from accepted original classification metadata')
        result['from_pretrained_attempts']=1
        module,info=transformers.AutoModelForSequenceClassification.from_pretrained(source['model'],config=config,
            state_dict=state,use_safetensors=False,attn_implementation='eager',output_loading_info=True,
            ignore_mismatched_sizes=False,**local)
        result['from_pretrained_completed']=1
        module.eval();result['loading_info']=info
    def describe(value,original=None):
        value=value.detach().cpu()
        return dict(shape=list(value.shape),dtype=str(value.dtype).removeprefix('torch.'),
            finite=bool(torch.isfinite(value).all().item()),
            **(dict(numerically_equal=bool(torch.equal(value,original.detach().cpu()))) if original is not None else {}))
    with recorder.span('complete_state_and_classifier_semantic_coverage'):
        original={k:describe(v) for k,v in state.items()}
        current=module.state_dict()
        loaded={k:describe(v,state.get(k)) for k,v in current.items()}
        buffers={k:describe(v,state.get(k)) for k,v in module.named_buffers() if k not in current and k in state}
        result['state_coverage']=validate_coverage(original,loaded,info,hidden_size=config.hidden_size,
            num_labels=config.num_labels,nonpersistent=buffers)
        result['checkpoint_semantic_coverage_passed']=True


def worker(args):
    args.work_dir.mkdir(parents=True,exist_ok=False)
    result=dict(round=ROUND,model=MODEL,mode=MODE,status='running',error=None,
        checkpoint_semantic_coverage_passed=False,checkpoint_load_attempts=0,checkpoint_load_completed=0,
        from_pretrained_attempts=0,from_pretrained_completed=0,**dict.fromkeys(ZERO_FIELDS,0))
    recorder=StageRecorder(args.work_dir/'events.jsonl')
    try:
        with recorder.span('bert_checkpoint_semantic_audit'):audit(args,recorder,result)
        result['status']='ok'
    except BaseException as exc:result.update(status='error',error=type(exc).__name__+': '+str(exc))
    write_json(args.work_dir/'worker-result.json',result)
    return result


def execute(args):
    out=args.out.resolve()
    if (out.parent!=paths.RESULTS_DIR.resolve() or out.exists()
            or any(out==p or out in p.parents or p in out.parents for p in
                (args.model.resolve(),args.prepare_result.resolve().parent,args.model_info.resolve().parent))):
        raise ValueError('A fresh separate checkpoint audit case is required')
    started=time.perf_counter_ns();out.mkdir();recorder=StageRecorder(out/'events.jsonl')
    result=dict(round=ROUND,model=MODEL,mode=MODE,status='running',error=None,worker=None,process=None,
        checkpoint_semantic_coverage_passed=False,security_acceptance=None,accuracy_acceptance=None,
        compiled_bootstraps=None,CKKS_latency=None,**dict.fromkeys(ZERO_FIELDS,0))
    try:
        with recorder.span('complete_bert_checkpoint_audit'):
            source_gate(args.model,args.prepare_result,args.model_info)
            if shutil.disk_usage(out).free<FREE_BYTES:raise RuntimeError('Checkpoint audit free disk reserve exhausted')
            shutil.copyfile(Path(__file__),out/'audit-source.py')
            command=[sys.executable,'-B','-m','pt_agent.tools.nixrun','--seconds',str(SECONDS+60),
                '--python',str(args.intake_python),'--','-B','-m',__spec__.name,'--worker',
                '--model',str(args.model),'--prepare-result',str(args.prepare_result),'--model-info',str(args.model_info),
                '--work-dir',str(out/'worker')]
            result['command']=command
            with recorder.span('independent_checkpoint_load_and_coverage'):
                result['process']=run_process(command,cwd=paths.REPO/'DSL/DSL_agent/scripts',log=out/'worker.log',
                    resources=out/'resources.jsonl',timeout=SECONDS+120,mem_gib=4,fsize_gib=1)
                proc=result['process']
                if (out/'worker/worker-result.json').is_file():result['worker']=_read(out/'worker/worker-result.json')
                if proc.get('exit_code')!=0 or proc.get('timed_out') or proc.get('signal') or proc.get('remaining_pids'):
                    raise RuntimeError('Checkpoint audit child did not finish cleanly')
                worker_result=result['worker'] or {}
                if (worker_result.get('status')!='ok' or worker_result.get('checkpoint_semantic_coverage_passed') is not True
                        or any(worker_result.get(k)!=1 for k in ('checkpoint_load_attempts','checkpoint_load_completed',
                            'from_pretrained_attempts','from_pretrained_completed'))
                        or any(worker_result.get(k)!=0 for k in ZERO_FIELDS)):
                    raise ValueError('Checkpoint audit coverage or action counts are incomplete')
                result['checkpoint_semantic_coverage_passed']=True
        result['status']='ok'
    except BaseException as exc:result.update(status='error',error=type(exc).__name__+': '+str(exc))
    finally:
        result['timing']=summarize_events(read_events(out/'events.jsonl'))
        result['recorded_pre_persistence_seconds']=(time.perf_counter_ns()-started)/1e9
        result['metrics']={k:result[k] for k in ('status','checkpoint_semantic_coverage_passed','security_acceptance',
            'accuracy_acceptance','compiled_bootstraps','CKKS_latency')+ZERO_FIELDS}
        if artifact_bytes(out)+len(json.dumps(result).encode())+65536>CASE_BYTES:
            result.update(status='error',error='Checkpoint audit case allowance exceeded');result['metrics']['status']='error'
        write_json(out/'result.json',result)
        from ..registry import git_rev
        row=dict(run_id=out.name,round=ROUND,model=MODEL,mode=MODE,git_rev=git_rev(),metrics=result['metrics'],
            artifacts_dir=str(out),ts=datetime.now(timezone.utc).isoformat())
        with (paths.RESULTS_DIR/'registry.jsonl').open('a') as stream:
            fcntl.flock(stream.fileno(),fcntl.LOCK_EX);stream.write(json.dumps(row)+'\n');stream.flush()
        persist_case_completion(out,round_label=ROUND,started_ns=started,status=result['status'],
            timing=result['timing'],required_artifact_bytes=artifact_bytes(out))
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--worker',action='store_true')
    for name in ('model','prepare-result','model-info'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--work-dir',type=Path);parser.add_argument('--out',type=Path)
    parser.add_argument('--intake-python',type=Path,default=paths.INTAKE_PYTHON)
    args=parser.parse_args(argv)
    for name in ('model','prepare_result','model_info'):setattr(args,name,getattr(args,name).resolve())
    if args.worker and args.work_dir is None:parser.error('--worker requires --work-dir')
    if not args.worker and args.out is None:parser.error('Audit outer requires --out')
    result=worker(args) if args.worker else execute(args)
    print(json.dumps(dict(status=result['status'],checkpoint_semantic_coverage_passed=result['checkpoint_semantic_coverage_passed'])))
    return 0 if result['status']=='ok' else 1


if __name__=='__main__':raise SystemExit(main())
