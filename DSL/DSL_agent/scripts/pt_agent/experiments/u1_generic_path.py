"""U1 generic split and real unlabelled ResNet20 tool-path acceptance."""

from dataclasses import asdict
import json


def run(args):
    import numpy as np
    from .u1_cli_acceptance import prepare
    from types import SimpleNamespace
    from ..analyzer import calibrate
    from ..data.samples import read
    from ..intake.probe import import_torch
    from ..metrics import passes
    from ..models.builders import resnet20
    from ..planners.agent_tools import build_context, evaluate
    from ..planners.confirm import Candidate, confirm_select
    from ..planners.prompts import build_system_prompt

    prepare(SimpleNamespace(worker='prepare',out=args.out))
    raw=read(args.out/'cifar-unlabelled.npz')
    imported=import_torch(resnet20(),raw,name='resnet20',source='pt_agent.models.builders:resnet20')
    info=imported.info
    info.save(args.out/'model')
    arrays,labels=imported.samples.get('calibration')
    assert labels is None
    calibration=calibrate(info.graph,arrays)
    ctx=build_context(info,calibration,samples=imported.samples)
    assert len(ctx.val_images)==20 and ctx.val_labels is None
    expected=raw.tensors(info.split['heldout'],names=info.graph.inputs)[0]
    assert all(np.array_equal(ctx.val_inputs[key],value) for key,value in expected.items())
    source=(args.reference/'models/resnet20/program.ptdsl').read_text()
    feedback=evaluate(source,ctx)
    assert feedback['stage']=='complete',feedback
    validation=feedback['validation']
    assert 'agreement_pct' in validation and 'top1' not in validation and 'acc_drop_pp' not in validation
    expected_feasible=passes(validation,info.threshold) and validation['max_u']<=0.9 and not validation['diverged']
    assert feedback['feasible']==expected_feasible
    cost=feedback['cost']
    candidate=Candidate(source,'frozen-regression',0,cost['latency_pred_us'],cost['boots_pred'])
    confirmed=confirm_select([candidate],ctx)
    confirmed['selected']=asdict(confirmed['selected']) if confirmed['selected'] is not None else None
    prompt=build_system_prompt(ctx)
    assert '99.5% (agreement_pct)' in prompt
    (args.out/'system_prompt.txt').write_text(prompt)
    result=dict(model='resnet20-unlabelled',split_counts={key:len(info.split[key]) for key in ('calibration','heldout','confirm','report')},
        context_samples_equal=True,validation=validation,feasible=feedback['feasible'],confirmation=confirmed,
        metric=info.metric,threshold=info.threshold,report_samples_evaluated=0,llm_calls=0)
    (args.out/'GENERIC.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return [result]
