"""Three saved training probes check only arithmetic of the trusted HEIR adapter.

No fitting, quality selection, heldout/confirmation/report access or CKKS occurs.
All source paths are supplied by the outer process.
"""
import argparse
import ast
import json
import math
import os
from pathlib import Path
import re
import resource
import sys


def validate_memory_gib(value):
    """Accept only the explicit finite address-space allowance, in GiB."""
    if (isinstance(value,bool) or not isinstance(value,(int,float))
            or not math.isfinite(value) or not 2<=value<=12):
        raise ValueError('Math memory allowance must be finite and within 2..12 GiB')
    return value


def supervised(args):
    """The host records results; the pure child receives every source path."""
    from datetime import datetime,timezone
    import fcntl
    from .. import paths
    from .r10_paper_metrics import StageRecorder,run_process,read_events,summarize_events,write_json
    from .heir_cnn_preflight import finalize_size
    memory_gib=validate_memory_gib(getattr(args,'memory_gib',2))
    out=args.out.absolute();out.mkdir(parents=True,exist_ok=False)
    command=[sys.executable,'-B','-m','pt_agent.tools.nixrun','--seconds','300',
             '--python',str(paths.INTAKE_PYTHON),'--','-m',__name__.replace('__main__','pt_agent.experiments.heir_cnn_math_check')]
    for name in ('graph_dir','program','mlir','probes','probe_source'):
        command+=['--'+name.replace('_','-'),str(getattr(args,name).absolute())]
    command+=['--out',str(out/'worker'),'--memory-gib',str(memory_gib)]
    if getattr(args,'exact_optiona_selector',False):command+=['--exact-optiona-selector']
    if getattr(args,'exact_spatial_mean_matvec',False):command+=['--exact-spatial-mean-matvec']
    if getattr(args,'common_model',None):command+=['--common-model',args.common_model]
    result=dict(round='HEIR-CNN-MATH',mode='trusted-heir-cnn-three-probe-arithmetic',model=getattr(args,'common_model',None) or 'resnet20_silu',
                status='running',error=None,worker=None,process=None,artifact_bytes=None,
                limits=dict(AS_gib=memory_gib,nix_seconds=300,supervisor_seconds=360),
                compiled_bootstraps=None,predicted_latency=None,CKKS_latency=None,accuracy_acceptance=None,
                scope='Saved train:0,1,2 mathematical regression only; no quality selection or CKKS')
    recorder=StageRecorder(out/'events.jsonl')
    try:
        with recorder.span('saved-probe-arithmetic',argv=command):
            obs=run_process(command,cwd=paths.REPO/'DSL/DSL_agent/scripts',log=out/'checks.log',
                            resources=out/'resources.jsonl',timeout=360,mem_gib=memory_gib)
        result['process']=dict(argv=command,observation=obs)
        if (out/'worker/result.json').is_file():result['worker']=json.loads((out/'worker/result.json').read_text())
        result['status']='ok' if obs['exit_code']==0 and not obs['timed_out'] and not obs['remaining_pids'] and (result['worker'] or {}).get('status')=='ok' else 'error'
    except BaseException as exc:
        result['status']='error';result['error']=type(exc).__name__+': '+str(exc)
    finally:
        write_json(out/'timing.json',summarize_events(read_events(out/'events.jsonl')))
        result['ts']=datetime.now(timezone.utc).isoformat()
        finalize_size(out,result)
        entry=dict(run_id=out.name,round=result['round'],model=result['model'],mode=result['mode'],git_rev='working-tree',
                   metrics={k:result[k] for k in ('status','compiled_bootstraps','predicted_latency','CKKS_latency','accuracy_acceptance')},
                   artifacts_dir=str(out),ts=result['ts'])
        with (paths.RESULTS_DIR/'registry.jsonl').open('a+') as f:
            fcntl.flock(f,fcntl.LOCK_EX);f.seek(0)
            if any(json.loads(l).get('run_id')==out.name for l in f if l.strip()):raise ValueError('Result identity already exists')
            f.seek(0,2);f.write(json.dumps(entry)+'\n');f.flush();fcntl.flock(f,fcntl.LOCK_UN)
    print(json.dumps({k:result[k] for k in ('status','error','artifact_bytes')}))
    return 0 if result['status']=='ok' else 1


def check_math(reference, observed):
    import numpy as np
    if (not isinstance(observed,np.ndarray) or observed.dtype!=np.float64
            or reference.shape!=observed.shape or not np.isfinite(observed).all()
            or not np.isfinite(reference).all()):
        raise ValueError('Original float64 shape/finite contract differs')
    error=np.abs(reference-observed)
    ratio=error/(1e-12+1e-12*np.abs(reference))
    return dict(passed=bool(np.all(ratio<=1)),shape=list(reference.shape),
                max_absolute_error=float(error.max()),max_error_ratio=float(ratio.max()),
                criterion='abs(error) <= 1e-12 + 1e-12*abs(reference)')


def check_constants(graph, text,*,exact_optiona_selector=False,exact_spatial_mean_matvec=False,common_model=None):
    import numpy as np
    from ..kernels.conv import fold_bn
    expected=[]
    if common_model is not None:
        from .heir_common_models import graph_contract
        graph_contract(graph,common_model)
    from .heir_cnn_export import exact_optiona_selectors,optiona_selector_weights,spatial_mean_matrix
    selectors=exact_optiona_selectors(graph) if exact_optiona_selector else {}
    for node in graph.nodes:
        a=node.attrs
        if node.id in selectors:
            spec=selectors[node.id]
            expected.append(('derived-exact-optiona:'+spec['original_nodes'][-1],optiona_selector_weights(spec['input_shape'][1],spec['pad_low'],spec['pad_high'])))
        elif node.op=='conv2d':
            expected.append((a['weight'],graph.weights[a['weight']]))
            if a.get('bias'):expected.append((a['bias'],graph.weights[a['bias']]))
        elif node.op=='batchnorm':
            scale,shift=fold_bn(*(graph.weights[a[k]] for k in ('weight','bias','mean','var')),a['eps'])
            expected.extend(((node.id+':scale',scale),(node.id+':shift',shift)))
        elif node.op=='linear':
            expected.append((a['weight'],graph.weights[a['weight']]))
            if a.get('bias'):expected.append((a['bias'],graph.weights[a['bias']]))
        elif node.op=='mean' and exact_spatial_mean_matvec:
            if tuple(a['axes'])!=(2,3):raise ValueError('Fixed mean constant axes differ')
            expected.append(('derived-exact-spatial-mean:'+node.id,spatial_mean_matrix(tuple(graph.tensors[node.inputs[0]].shape))))
        elif node.op=='avgpool' and common_model is not None:
            from .heir_common_models import avgpool_weights
            weight,_=avgpool_weights(tuple(graph.tensors[node.inputs[0]].shape),a,tuple(graph.tensors[node.outputs[0]].shape))
            expected.append(('derived-exact-local-avgpool:'+node.id,weight))
    actual=re.findall(r'^\s*%\w+ = arith.constant dense<(.+)> : tensor<([^>]+)>$',text,re.MULTILINE)
    if len(actual)!=len(expected):raise ValueError('Emitted tensor constant inventory differs')
    rows=[]
    for (name,value),(literal,ty) in zip(expected,actual):
        array=np.asarray(ast.literal_eval(literal),dtype=np.float64)
        row=check_math(value,array)
        row.update(name=name,source_dtype=str(value.dtype),emitted_type=ty,
                   numerically_equal=bool(np.array_equal(value,array)))
        rows.append(row)
        if not row['passed'] or not row['numerically_equal']:
            raise ValueError('Emitted float64 constant changed numerically: '+name)
    return rows


def evaluate_graph(graph, configs, image,*,common_model=None):
    import numpy as np
    from ..kernels.conv import fold_bn
    from .heir_cnn_export import chebyshev_tape
    if common_model is not None:
        from .heir_common_models import graph_contract
        graph_contract(graph,common_model)
    values={graph.inputs[0]:image}
    observations=[]
    activation=0
    remaining={name:sum(n.inputs.count(name) for n in graph.nodes)+graph.outputs.count(name)
               for name in graph.tensors}
    for node in graph.nodes:
        x=values[node.inputs[0]];a=node.attrs
        if node.op=='conv2d':
            import torch
            import torch.nn.functional as F
            value=F.conv2d(torch.tensor(x,dtype=torch.float64),
                           torch.tensor(graph.weights[a['weight']],dtype=torch.float64),
                           bias=torch.tensor(graph.weights[a['bias']],dtype=torch.float64) if a.get('bias') else None,
                           stride=a['stride'],padding=a['padding'],groups=a.get('groups',1)).numpy()
        elif node.op=='batchnorm':
            scale,shift=fold_bn(*(graph.weights[a[k]] for k in ('weight','bias','mean','var')),a['eps'])
            if common_model is None:value=x*scale[None,:,None,None]+shift[None,:,None,None]
            else:
                if x.ndim<2 or scale.shape!=(x.shape[1],) or shift.shape!=scale.shape:raise ValueError('Common-model batchnorm channel shape differs')
                broadcast=(1,x.shape[1])+((1,)*(x.ndim-2))
                value=x*scale.reshape(broadcast)+shift.reshape(broadcast)
        elif node.op=='act':
            cfg=configs[activation]
            observations.append(x.copy())
            tape,out,_=chebyshev_tape(cfg['fn'],cfg['deg'],cfg['domain'])
            value=tape.evaluate(x,out)
            activation+=1
        elif node.op=='add':value=x+values[node.inputs[1]]
        elif node.op=='slice':
            selectors=[slice(None)]*x.ndim
            selectors[a['axis']]=slice(a['start'],a['end'],a['step'])
            value=x[tuple(selectors)]
        elif node.op=='pad':
            pads=[(0,0)]*x.ndim
            for i in range(len(a['pads'])//2):pads[-1-i]=tuple(a['pads'][i*2:i*2+2])
            value=np.pad(x,pads)
        elif node.op=='mean':value=x.reshape(*x.shape[:2],-1).sum(-1)*(1/(x.shape[2]*x.shape[3]))
        elif node.op=='linear':
            value=x@graph.weights[a['weight']].T
            if a.get('bias'):value=value+graph.weights[a['bias']]
        elif node.op=='avgpool' and common_model is not None:
            from .heir_common_models import avgpool_spec
            spec=avgpool_spec(x.shape,a,graph.tensors[node.outputs[0]].shape)
            kh,kw=spec['kernel'];sh,sw=spec['stride'];ph,pw=spec['padding']
            padded=np.pad(x,((0,0),(0,0),(ph,ph),(pw,pw)))
            value=np.empty(spec['output_shape'],dtype=np.float64)
            for h in range(value.shape[2]):
                for w in range(value.shape[3]):
                    value[:,:,h,w]=padded[:,:,h*sh:h*sh+kh,w*sw:w*sw+kw].sum(axis=(2,3))*(1/spec['divisor'])
        elif node.op=='reshape' and common_model is not None:
            from .heir_common_models import reshape_spec
            spec=reshape_spec(x.shape,a,graph.tensors[node.outputs[0]].shape)
            value=x.reshape(spec['output_shape'],order='C')
        else:raise ValueError('Unexpected graph operation')
        if common_model is not None and value.shape!=tuple(graph.tensors[node.outputs[0]].shape):raise ValueError('Common-model arithmetic output shape differs')
        if value.dtype!=np.float64 or not np.isfinite(value).all():raise ValueError('Adapter arithmetic became nonfinite')
        values[node.outputs[0]]=value
        for name in node.inputs:
            remaining[name]-=1
            if not remaining[name]:values.pop(name,None)
    return values[graph.outputs[0]],observations


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('graph-dir','program','mlir','probes','probe-source','out'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--supervise',action='store_true')
    p.add_argument('--memory-gib',type=float,default=2,
                   help='Supervised math address-space allowance in GiB (finite 2..12; default 2)')
    p.add_argument('--exact-optiona-selector',action='store_true')
    p.add_argument('--exact-spatial-mean-matvec',action='store_true')
    from .heir_common_models import COMMON_MODELS
    p.add_argument('--common-model',choices=COMMON_MODELS)
    a=p.parse_args(argv)
    try:validate_memory_gib(a.memory_gib)
    except ValueError as exc:p.error(str(exc))
    if a.supervise:return supervised(a)
    a.out.mkdir(parents=True,exist_ok=False)
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
    result=dict(status='running',scope='Three R13 saved training probes; trusted adapter arithmetic and numerical constants only',
                source=dict(graph=str(a.graph_dir),program=str(a.program),mlir=str(a.mlir),probes=str(a.probes),probe_source=str(a.probe_source)),
                configured_memory_gib=a.memory_gib,
                graph_load_attempts=0,graph_load_completed=0,probe_load_attempts=0,probe_load_completed=0,
                adapter_forward_attempts=0,adapter_forward_completed=0,reference_forward_attempts=0,reference_forward_completed=0,
                activation_pairs=0,probes=[],constants=[],error=None,quality_selection=False,
                confirm_samples=0,report_samples=0,heldout_samples=0,training_samples_for_selection=0,
                llm_calls=0,calibration_attempts=0,plan_attempts=0,count_attempts=0,emit_attempts=0,
                compiler_runs=0,keygen_runs=0,real_ckks_attempts=0,real_ckks_completed=0,
                compiled_bootstraps=None,predicted_latency=None,CKKS_latency=None,accuracy_acceptance=None)
    def save():(a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    try:
        import numpy as np
        import torch
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        from .heir_cnn_export import load_graph,saved_activations
        from ..tensor_sim import run_tensor
        source=json.loads(a.probe_source.read_text())
        if a.common_model:
            from .heir_common_models import checked_saved_frontend_identity
            checked_saved_frontend_identity(json.loads((a.program.parent.parent/'result.json').read_text()),a.program,a.common_model)
        if source.get('split')!='train' or source.get('indices')!=[0,1,2] or source.get('sample_ids')!=['train:0','train:1','train:2']:
            raise ValueError('Only the three saved R13 training probes are in scope')
        result['graph_load_attempts']=1;save()
        graph=load_graph(a.graph_dir);result['graph_load_completed']=1
        if a.common_model is not None:
            from .heir_common_models import graph_contract
            result['common_model']=a.common_model
            result['common_model_contract']=graph_contract(graph,a.common_model)
        program,configs=saved_activations(a.program.read_text())
        result['exact_optiona_selector']=a.exact_optiona_selector
        result['exact_spatial_mean_matvec']=a.exact_spatial_mean_matvec
        result['constants']=check_constants(graph,a.mlir.read_text(),exact_optiona_selector=a.exact_optiona_selector,exact_spatial_mean_matvec=a.exact_spatial_mean_matvec,common_model=a.common_model);save()
        result['probe_load_attempts']=1;save()
        with np.load(a.probes,allow_pickle=False) as archive:
            images=archive['input'];indices=archive['indices']
        probe_shape=(3,*result['common_model_contract']['input_shape'][1:]) if a.common_model else (3,3,32,32)
        if images.dtype!=np.float64 or images.shape!=probe_shape or indices.tolist()!=[0,1,2] or not np.isfinite(images).all():
            raise ValueError('Original probe dtype/shape/indices differ')
        result['probe_load_completed']=1
        for i in range(3):
            result['reference_forward_attempts']+=1;save()
            observations=[]
            reference=run_tensor(program,graph.weights,images[i:i+1],cache=False,
                                 observer=lambda name,value,cfg:observations.append(value.copy()))
            result['reference_forward_completed']+=1
            result['adapter_forward_attempts']+=1;save()
            observed,activation_values=evaluate_graph(graph,configs,images[i:i+1],common_model=a.common_model)
            result['adapter_forward_completed']+=1
            expected_acts=result['common_model_contract']['activation_count'] if a.common_model else 19
            if len(observations)!=expected_acts or len(activation_values)!=expected_acts:raise ValueError('Original activation inventory differs')
            pairs=[check_math(x,y) for x,y in zip(observations,activation_values)]
            result['activation_pairs']+=len(pairs)
            check=check_math(reference,observed)
            result['probes'].append(dict(sample_id=source['sample_ids'][i],reference=reference.tolist(),observed=observed.tolist(),
                                         activation_checks=pairs,logit_check=check))
            save()
            if not check['passed'] or not all(x['passed'] for x in pairs):raise ValueError('Original strict mathematical criterion failed')
        result['status']='ok'
    except BaseException as exc:
        result['status']='error';result['error']=type(exc).__name__+': '+str(exc)
    finally:
        result['resource_ru_maxrss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        save()
    print(json.dumps({k:result[k] for k in ('status','error','reference_forward_completed','adapter_forward_completed','activation_pairs')}))
    return 0 if result['status']=='ok' else 1


if __name__=='__main__':raise SystemExit(main())
