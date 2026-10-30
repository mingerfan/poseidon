"""U4 §11.7: complete finite choices and read-only calibration score ranges."""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
from unittest.mock import patch

from ..paths import REPO, RESULTS_DIR
from .u4_point_diagnosis import write


def score_categories(scores, mask):
    import numpy as np
    mask=np.asarray(mask,dtype=bool)
    query=mask[:,None,:,None];key=mask[:,None,None,:]
    selectors=dict(valid_query_valid_key=query & key,
                   padding_key=np.broadcast_to(~key,scores.shape),
                   padding_query=np.broadcast_to(~query,scores.shape))
    result={}
    for name, selected in selectors.items():
        values=scores[np.broadcast_to(selected,scores.shape)]
        result[name]=dict(count=int(values.size),minimum=float(values.min()) if values.size else None,
                          maximum=float(values.max()) if values.size else None)
    return result


def calibration_scores(info, arrays):
    from ..analyzer import reference_forward
    nodes=[n for n in info.graph.nodes if n.op=='softmax']
    result={n.id:dict(node=n.id,value=n.inputs[0],categories={}) for n in nodes}
    count=len(next(iter(arrays.values())))
    for i in range(count):
        values=reference_forward(info.graph,{k:v[i:i+1] for k,v in arrays.items()},keep=True)
        for node in nodes:
            scores=values[node.inputs[0]]
            mask=arrays[node.attrs['mask']][i:i+1]
            for key,row in score_categories(scores,mask).items():
                total=result[node.id]['categories'].setdefault(key,dict(count=0,minimum=None,maximum=None))
                total['count']+=row['count']
                if row['count']:
                    total['minimum']=row['minimum'] if total['minimum'] is None else min(total['minimum'],row['minimum'])
                    total['maximum']=row['maximum'] if total['maximum'] is None else max(total['maximum'],row['maximum'])
    return dict(samples=count,split='calibration',engine='exact',affects_rule=False,
                scope='Pre-shift scores before causal exclusion. padding_key includes all queries; '
                      'padding_query includes all keys, so these two categories overlap.',
                nodes=list(result.values()))


def diagnose(args):
    from ..models.info import ModelInfo
    from ..data.samples import StoredSplits
    from ..planners import transformer_rule as rule
    from ..dsl.parser import parse
    from ..dsl.typecheck import check
    info=ModelInfo.load(args.prepared/'model')
    calibration=json.loads((args.prepared/'model/calibration.json').read_text())
    ranges=None
    if args.model=='qwen':
        if args.score_ranges is not None:
            ranges=json.loads(args.score_ranges.read_text())
        else:
            arrays,_=StoredSplits(args.prepared/'model/samples',info).get('calibration')
            ranges=calibration_scores(info,arrays)
        write(args.out/'SCORE_RANGES.json',ranges)
    keys=[];softmax=[]
    for node in info.graph.nodes:
        out=node.outputs[0]
        if node.op in ('layernorm','rmsnorm'):keys.append((node.id,out+'::rsqrt'))
        elif node.op=='softmax':
            cfg=rule.squared_softmax_adjustments(calibration[out+'::exp'],info.graph.tensors[node.inputs[0]].shape[-1])
            lo,hi=(float(calibration[out+'::exp'][k]) for k in ('min','max'))
            width=(hi-lo)*1.5
            softmax.append(dict(node=node.id,k0=max(0,math.ceil(math.log2(width))),k=cfg['steps'],
                unscaled_width=width,scaled_width=cfg['exp_domain'][1]-cfg['exp_domain'][0],
                inv_ratio=cfg['inv_domain'][1]/cfg['inv_domain'][0],configuration=cfg))
            keys.extend((node.id,out+'::'+role) for role in ('exp','inv','step_inv') if role!='step_inv' or cfg['steps'])
    rows=[];select=rule.select_approximation
    def recorded(fn,domain,**kwargs):
        node,key=keys[len(rows)]
        print(f'CHOOSING {node} {key} {fn} {domain}',flush=True)
        report=select(fn,domain,**kwargs)
        rows.append(dict(node=node,key=key,
            fit=report['selected'].get('fit','absolute') if report['selected'] is not None else None,**report))
        write(args.out/'CHOICES.partial.json',rows)
        print(f'CHOICE {node} {key} {report["status"]} {report["selected"]}',flush=True)
        return report
    source=None;issues=[]
    with patch.object(rule,'select_approximation',side_effect=recorded):
        try:source=rule.plan(info.graph,calibration)
        except rule.ApproximationLimitError as exc:issues=exc.issues
    if len(rows)!=len(keys):raise RuntimeError('Incomplete finite point table')
    for node in softmax:
        node['points']=[dict(key=r['key'],status=r['status'],selected=r['selected'],depth=r['selected_depth'],
                             fit=None if r['selected'] is None else r['selected'].get('fit','absolute'),
                             required=r['required']) for r in rows if r['node']==node['node']]
    result=dict(complete=not issues,table_complete=True,split='calibration',prepared=str(args.prepared),
                softmax=softmax,points=rows,issues=issues,model=args.model,
                score_ranges=str(args.out/'SCORE_RANGES.json') if ranges is not None else None,
                planned_program=None,llm_calls=0,report_images=0)
    if source is not None:
        path=args.out/'program.ptdsl';path.write_text(source)
        checked=check(parse(source),{k:v.shape for k,v in info.graph.weights.items()})
        if not checked.ok:raise RuntimeError('Planned Qwen program typecheck failed: '+str(checked.diags))
        result.update(planned_program=str(path),typecheck=True)
        if args.old_program is not None:
            from ..transformer_validation import points
            from ..kernels import approx
            from ..backends import CountBackend
            def choices(text):
                result=[]
                for point in points(parse(text)):
                    if point.op.kind=='act':continue
                    backend=CountBackend(8)
                    value=approx.activation(backend,backend.input(0),point.attrs)
                    result.append(dict(role=point.role,method=point.attrs['method'],fit=point.attrs.get('fit','absolute'),
                        deg=point.attrs.get('deg'),deg0=point.attrs.get('deg0'),iters=point.attrs.get('iters'),depth=value.depth))
                return result
            before=choices(args.old_program.read_text());after=choices(source)
            if len(before)!=len(after):raise RuntimeError('Old/new point inventory differs')
            result['old_program']=str(args.old_program)
            result['program_text_changed']=args.old_program.read_text()!=source
            result['old_new_choices']=[dict(index=i,before=a,after=b) for i,(a,b) in enumerate(zip(before,after))]
    else:
        result.update(status='infeasible',stop_rule='U4 §11.3–§11.4 and §11.7 after complete table')
        write(args.out/'STOP.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared',type=Path,required=True)
    parser.add_argument('--model',choices=('bert','qwen'),default='qwen')
    parser.add_argument('--old-program',type=Path)
    parser.add_argument('--score-ranges',type=Path)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--inside',action='store_true')
    args=parser.parse_args()
    if args.inside:
        resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,6*1024**3))
        resource.setrlimit(resource.RLIMIT_FSIZE,(8*1024**3,8*1024**3))
        resource.setrlimit(resource.RLIMIT_CORE,(0,0))
        try:report=diagnose(args)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            report=dict(complete=False,error=f'{type(exc).__name__}: {exc}')
            write(args.out/'STOP.json',report)
        write(args.out/'REPORT.json',report)
        return 0 if report['complete'] else 1
    stamp=datetime.now(timezone.utc);started=time.monotonic()
    run_id='u4-'+args.model+'-relative-'+stamp.strftime('%Y%m%dT%H%M%S%f')
    args.out=args.out or RESULTS_DIR/run_id
    args.out.mkdir(parents=True,exist_ok=False)
    print('RUN='+str(args.out),flush=True)
    command=[sys.executable,'-B','-m','pt_agent.tools.nixrun','--seconds','3600','--',
             '-m','pt_agent.experiments.u4_wide_softmax','--inside','--prepared',str(args.prepared),
             '--model',args.model,'--out',str(args.out)]
    for key in ('old_program','score_ranges'):
        if getattr(args,key) is not None:command+=['--'+key.replace('_','-'),str(getattr(args,key))]
    with (args.out/'run.log').open('w') as log:
        status=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=3630)
    path=args.out/'REPORT.json'
    report=json.loads(path.read_text()) if path.exists() else dict(complete=False,error='see run.log')
    rev=subprocess.check_output(['git','rev-parse','--short','HEAD'],cwd=REPO,text=True).strip()
    report.update(round='U4',section='11.8',run_id=run_id,git_rev=rev,seconds=time.monotonic()-started,
                  exit=status.returncode,llm_calls=0,report_images=0)
    write(path,report)
    from ..registry import append
    os.environ.setdefault('PT_RESULTS',str(RESULTS_DIR))
    append(dict(run_id=run_id,round='U4',model=args.model,mode='relative-fit-planning',git_rev=rev,
                metrics=dict(complete=report['complete'],table_complete=report.get('table_complete'),llm_calls=0),
                artifacts_dir=str(args.out),ts=stamp.isoformat()))
    print(json.dumps(dict(report=str(path),complete=report['complete'],table_complete=report.get('table_complete'))),flush=True)
    return 0 if report['complete'] else 1


if __name__=='__main__':raise SystemExit(main())
