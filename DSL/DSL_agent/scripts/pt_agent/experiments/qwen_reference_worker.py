"""Original trained Qwen prefill/own-token references and mapped calibration.

Both phases are plaintext reference diagnostics on original training texts.
Their greedy tokens never enter the encrypted generation interface.  The HF
phase and the mapped phase run separately so their complete weights coexist
only within one implementation at a time.
"""

import argparse
import json
import os
from pathlib import Path
import resource

from .r10_paper_metrics import StageRecorder,write_json


def _numeric():
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[name]='1'
    import numpy as np
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    return np,torch


def _identity(info,local_path):
    if (Path(info['source']['local_path']).resolve()!=Path(local_path).resolve()
            or info['source']['config']['num_hidden_layers']!=24
            or info['source']['config']['hidden_size']!=896
            or info['source']['config']['vocab_size']!=151936
            or info['sequence']['length']!=32):
        raise ValueError('The saved full original Qwen2-0.5B interface differs')
    shapes={item['name']:tuple(item['shape']) for item in info['inputs']}
    if shapes!={'inputs_embeds':(1,32,896),'attention_mask':(1,32),'sel':(1,32)}:
        raise ValueError('The original saved input signature differs')


def _texts(args):
    source=json.loads(args.source_rows.read_text())
    if not Path(source['source']).name.startswith('train-') or source['column']!='sentence':
        raise ValueError('Only the already saved real training texts are accepted')
    texts=[json.loads(line)['text'] for line in args.samples.read_text().splitlines()]
    if len(texts)!=len(source['source_rows']) or any(not isinstance(text,str) or not text.strip() for text in texts):
        raise ValueError('Original training text/provenance rows differ')
    return texts,['train:'+str(index) for index in source['source_rows']],source


def hf_reference(args,recorder,result,np,torch):
    from ..intake.hf_frontend import load_local
    from ..kernels.encrypted_generation import eligible_prompts
    info=json.loads((args.model_info/'model_info.json').read_text())
    _identity(info,args.local_model)
    with recorder.span('load_original_hf_float64'):
        result['original_model_load_attempts']=1
        local=load_local(args.local_model,attn_implementation='eager',observer=recorder)
        if len(local.module.model.layers)!=24 or local.module.config.num_hidden_layers!=24:
            raise ValueError('The original HF model must retain every decoder layer')
        result['original_model_load_completed']=1
    with recorder.span('select_original_training_prompts'):
        texts,ids,source=_texts(args)
        chosen=eligible_prompts(local.processor,texts,ids,limit=3)
        if local.processor.pad_token_id is None:
            local.processor.pad_token=local.processor.eos_token
        local.processor.padding_side=info['sequence']['padding_side']
        if local.processor.padding_side!='right':
            raise ValueError('The saved fixed-prefix interface requires its original right padding')
        result['training_text_rows_read']=len(texts)
        result['prompt_selection']=chosen
        result['training_source']=source['source']
        result['original_model_identity']=dict(layers=24,hidden_size=896,vocabulary=151936,
            use_cache=False,sequence_length=info['sequence']['length'],dtype='float64',attention='eager')
        result['model_info_source']=str(args.model_info.resolve())
        result['original_hf_source']=str(args.local_model.resolve())
    trajectories=[]
    for index in chosen['indices']:
        original_ids=local.processor(texts[index],padding=False,truncation=False)['input_ids']
        length=len(original_ids)
        ids32=list(original_ids)+[local.processor.pad_token_id]*(32-length)
        trajectory=dict(sample_id=ids[index],original_text_index=index,text=texts[index],
            original_token_ids=original_ids,steps=[],greedy_tokens=[])
        for step in range(2):
            attention=np.zeros((1,32),dtype=np.float64);attention[0,:length]=1.
            selected=np.zeros_like(attention);selected[0,length-1]=1.
            with recorder.span('original_hf_prefill' if step==0 else 'original_hf_own_token_recompute',sample_id=ids[index],step=step):
                result['hf_forward_attempts']+=1
                write_json(args.work_dir/'worker-result.json',result)
                with torch.no_grad():
                    token_tensor=torch.tensor([ids32],dtype=torch.long)
                    embeddings=local.module.get_input_embeddings()(token_tensor)
                    all_logits=local.module(inputs_embeds=embeddings,attention_mask=torch.from_numpy(attention),
                                            use_cache=False,return_dict=False)[0]
                    logits=all_logits[:,length-1:length,:].detach().cpu().numpy().copy()
                if logits.shape!=(1,1,151936) or logits.dtype!=np.float64 or not np.isfinite(logits).all():
                    raise ValueError('Original full-model logits are nonfinite or have changed dimensions')
                next_token=int(np.argmax(logits.reshape(-1)))
                filename=f'prompt-{len(trajectories)}-step-{step}-logits.npy'
                np.save(args.work_dir/filename,logits)
                row=dict(step=step,prefix_length=length,prefix_token_ids=list(ids32),
                    attention_mask=attention.tolist(),sel=selected.tolist(),logits_file=filename,
                    greedy_token=next_token,all_finite=True)
                trajectory['steps'].append(row);trajectory['greedy_tokens'].append(next_token)
                ids32[length]=next_token;length+=1
                result['hf_forward_completed']+=1
                result['trajectories']=trajectories+[trajectory]
                write_json(args.work_dir/'worker-result.json',result)
        decoded=[]
        for token in trajectory['greedy_tokens']:
            decoded.append(token)
            if token==local.processor.eos_token_id:break
        trajectory['client_eos_truncated_tokens']=decoded
        trajectories.append(trajectory)
    result.update(status='ok',trajectories=trajectories,hf_reference_completed=True,
        original_prefill_forwards=3,original_own_token_forwards=3,
        plaintext_reference_only=True,kv_cache_used=False,ciphertext_generation_executed=False)


def _stats_update(stats,name,value,np):
    array=np.asarray(value)
    if not np.isfinite(array).all():raise ValueError('Original reference calibration is nonfinite: '+name)
    if not array.size:raise ValueError('Original reference calibration is empty: '+name)
    low,high=float(array.min()),float(array.max())
    observation=dict(min=low,max=high,absmax=max(abs(low),abs(high)),p999=float(np.percentile(np.abs(array),99.9)),count=int(array.size))
    if name not in stats:stats[name]=observation
    else:
        old=stats[name]
        old.update(min=min(old['min'],low),max=max(old['max'],high),absmax=max(old['absmax'],observation['absmax']),
                   p999=max(old['p999'],observation['p999']),count=old['count']+observation['count'])


def mapped_reference(args,recorder,result,np,torch):
    from ..models.mapped_info import load_mapped_info
    from ..intake.mapped_reference import MappedReference
    from ..analyzer import approximation_values,approximation_points
    source=json.loads(args.reference.read_text())
    if (source['status']!='ok' or source.get('hf_reference_completed') is not True
            or source['hf_forward_completed']!=6 or len(source['trajectories'])!=3
            or source['original_hf_source']!=str(args.local_model.resolve())):
        raise ValueError('All original full-model reference trajectories are required')
    metadata=json.loads((args.model_info/'model_info.json').read_text());_identity(metadata,args.local_model)
    original_interface=json.loads((Path(source['model_info_source'])/'model_info.json').read_text())
    _identity(original_interface,args.local_model)
    if any(metadata[key]!=original_interface[key] for key in ('inputs','sequence')) or metadata['source']['config']!=original_interface['source']['config']:
        raise ValueError('The saved original HF reference and normalized graph interfaces differ')
    result['reference_saved_model_info']=source['model_info_source']
    result['normalized_model_info_source']=str(args.model_info.resolve())
    with recorder.span('map_original_full_graph'):
        info=load_mapped_info(args.model_info)
        result['mapping']=info.mapped_weights
        reference_graph=info.graph
        if args.phase=='framework-reference':
            import inspect
            import transformers
            from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm,Qwen2Attention
            from ..intake.framework_precision import expand_reference_precision
            evidence=[]
            for function in (Qwen2RMSNorm.forward,Qwen2Attention.forward):
                lines,start=inspect.getsourcelines(function)
                if 'torch.float32' not in ''.join(lines):
                    raise ValueError('The original framework intermediate precision source differs')
                evidence.append(dict(function=function.__qualname__,path=inspect.getsourcefile(function),
                                     first_line=start,intermediate_dtype='torch.float32'))
            policies={node.id:'torch.float32' for node in info.graph.nodes if node.op in ('rmsnorm','softmax')}
            if sum(node.op=='rmsnorm' for node in info.graph.nodes)!=49 or sum(node.op=='softmax' for node in info.graph.nodes)!=24:
                raise ValueError('The original full-model normalization occurrences differ')
            reference_graph=expand_reference_precision(info.graph,policies)
            result['reference_precision_policy']=dict(source=evidence,torch_version=torch.__version__,
                transformers_version=transformers.__version__,rmsnorm_occurrences=49,softmax_occurrences=24,
                original_hf_reference_preserved=True,reference_only=True,
                canonical_candidate_graph_changed=False,weights_changed=False)
        reference=MappedReference(reference_graph)
        if reference.observation['private_weight_copies']!=0:
            raise ValueError('The complete saved graph must use its original read-only mappings')
        embedding=info.graph.weights['original.model.embed_tokens.weight']
    calibration={};checks=[]
    special={name for node,name in approximation_points(info.graph) if node.op!='act'}
    for trajectory in source['trajectories']:
        for row in trajectory['steps']:
            with recorder.span('mapped_reference_step',sample_id=trajectory['sample_id'],step=row['step']):
                result['mapped_forward_attempts']+=1
                ids=np.asarray(row['prefix_token_ids'],dtype=np.int64)
                arrays=dict(inputs_embeds=np.asarray(embedding[ids])[None,:,:],
                    attention_mask=np.asarray(row['attention_mask'],dtype=np.float64),sel=np.asarray(row['sel'],dtype=np.float64))
                tensors=reference.forward(arrays,keep=True)
                values={name:value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
                        for name,value in tensors.items()}
                observed=values[info.graph.outputs[0]]
                expected=np.load(args.reference.parent/row['logits_file'],mmap_mode='r',allow_pickle=False)
                if observed.dtype!=np.float64 or observed.shape!=expected.shape or not np.isfinite(observed).all():
                    raise ValueError('Mapped full-model reference output dimensions or finite values differ')
                error=np.abs(observed-expected);tolerance=1e-12+1e-12*np.abs(expected)
                passed=bool(np.all(error<=tolerance))
                check=dict(sample_id=trajectory['sample_id'],step=row['step'],passed=passed,
                    max_absolute_difference=float(error.max()),max_tolerance_ratio=float((error/tolerance).max()),
                    greedy_token=int(np.argmax(observed.reshape(-1))),original_greedy_token=row['greedy_token'],
                    finite_elements=int(observed.size))
                check['token_agreement']=check['greedy_token']==check['original_greedy_token']
                checks.append(check);result['math_checks']=checks
                if not passed:raise ValueError('Original eager and mapped full-model logits differ under the original mathematical policy')
                if not check['token_agreement']:raise ValueError('Original eager and mapped greedy tokens differ')
                observation_callback=getattr(args,'_observation_callback',None)
                if observation_callback is not None:
                    observation_callback(info.graph,values,trajectory,row,check)
                for name,tensor in info.graph.tensors.items():
                    if tensor.kind=='act':_stats_update(calibration,name,values[name],np)
                with np.errstate(over='ignore',invalid='ignore',under='ignore'):
                    analytic=approximation_values(info.graph,values)
                for name,value in analytic.items():
                    if name not in special:
                        continue
                    if name.endswith('::inv') and not np.isfinite(value).all():
                        result.setdefault('calibration_issues',[]).append(dict(
                            sample_id=trajectory['sample_id'],step=row['step'],point=name,
                            reason='Unshifted analytic exp-sum is nonfinite; no finite domain was substituted',
                            elements=int(np.asarray(value).size),finite_elements=int(np.isfinite(value).sum())))
                        continue
                    _stats_update(calibration,name,value,np)
                result['mapped_forward_completed']+=1
                result['reference_ownership']=reference.observation
                write_json(args.work_dir/'worker-result.json',result)
    write_json(args.work_dir/'calibration.json',calibration)
    result.update(status='ok',mapped_reference_completed=True,calibration_points=len(calibration),
        approximation_points=len(approximation_points(info.graph)),calibration_trajectories=3,calibration_forward_states=6,
        calibration=str(args.work_dir/'calibration.json'),prefill_and_own_token_math_passed=True,
        plaintext_reference_only=True,ciphertext_generation_executed=False)
    result['calibration_complete']=not bool(result.get('calibration_issues'))
    if args.phase=='framework-reference':
        result['framework_precision_reference_passed']=True
        result['calibration_scope']='Original framework activations and canonical analytic approximation domains on the same real training trajectories'


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('hf-reference','mapped-reference','framework-reference'),required=True)
    for flag in ('work-dir','model-info','local-model'):parser.add_argument('--'+flag,type=Path,required=True)
    for flag in ('samples','source-rows','reference'):parser.add_argument('--'+flag,type=Path)
    args=parser.parse_args(argv);args.work_dir.mkdir(parents=True,exist_ok=True)
    result=dict(status='running',phase=args.phase,hf_forward_attempts=0,hf_forward_completed=0,
        mapped_forward_attempts=0,mapped_forward_completed=0,hf_reference_completed=False,mapped_reference_completed=False,
        api_calls=0,compiler_runs=0,keygen_runs=0,ckks_executions=0,heldout_samples=0,confirm_samples=0,report_samples=0,
        candidate_frozen=False,compiled_bootstraps=None,predicted_latency=None,CKKS_latency=None,accuracy_acceptance=None)
    recorder=StageRecorder(args.work_dir/'events.jsonl')
    try:
        np,torch=_numeric()
        with recorder.span('qwen_original_reference'):
            (hf_reference if args.phase=='hf-reference' else mapped_reference)(args,recorder,result,np,torch)
    except BaseException as exc:result.update(status='error',error_type=type(exc).__name__,error=str(exc)[:2000])
    result['worker_maxrss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    write_json(args.work_dir/'worker-result.json',result)
    print(json.dumps({name:result[name] for name in ('status','phase','hf_forward_completed','mapped_forward_completed','worker_maxrss_kib')}))
    return 0 if result['status']=='ok' else 1


if __name__=='__main__':raise SystemExit(main())
