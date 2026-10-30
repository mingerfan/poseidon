"""A1 6.17 independent heldout Mock memory cases and alignment diagnosis."""

import gc
import json
from unittest.mock import patch


def estimate_caches():
    from ..backends.dag import DagBackend
    from ..backends.count import CountBackend
    from .. import tensor_sim
    from .a1_mock_memory import retained_size
    rows=[]
    for obj in gc.get_objects():
        if isinstance(obj,(DagBackend,CountBackend)):
            rows.append(dict(backend=type(obj).__name__,
                constants=len(obj._constants),identities=len(obj._constant_objects),
                bytes=retained_size((obj._constants,obj._constant_objects))))
    return dict(backends=rows,tensor=tensor_sim.cache_info(),
                cost_result_bytes=retained_size(tensor_sim._estimates))


def context(args):
    from ..models.info import ModelInfo
    from ..dsl.parser import parse
    from ..planners.agent_tools import build_context
    from .a1_transformer_agent import read,samples_for
    info=ModelInfo.load(args.model_dir)
    ctx=build_context(info,read(args.model_dir/'calibration.json'),samples=samples_for(args.model_dir,info))
    ctx.prompt_version=args.prompt_version
    return parse(args.program.read_text()),ctx


def diagnostic(args):
    import numpy as np
    from .. import transformer_validation as validation
    from ..planners import agent_tools
    from ..planners.transformer_session import json_safe
    from ..tensor_sim import cached_estimate
    from ..cost import estimate
    from .a1_mock_memory import snapshot
    program,ctx=context(args)
    result=dict(complete=False,program=str(args.program),case=args.part,
                batch=10 if args.part=='mock-case10' else 20,llm_calls=0,report_images=0)
    if args.part in ('mock-estimate20','mock-release20'):
        result['cost']=cached_estimate(program,ctx.weights,ctx.profile,estimate)
    result.update(before_mock=snapshot(),estimate_caches=estimate_caches())
    if args.part=='mock-release20':
        from ..cost import release_estimate_caches
        release_estimate_caches()
        result.update(after_release=snapshot(),released_estimate_caches=estimate_caches())
    def save():
        (args.out/'MEASUREMENTS.json').write_text(json.dumps(json_safe(result),indent=2)+'\n')
    save();outputs=[];diagnostics=[]
    def batch(observer):
        result.setdefault('batches',[]).append(snapshot(observer));save()
    try:
        with patch.object(agent_tools,'VAL_BATCH',result['batch']):
            result['validation']=validation.validate(program,ctx,n=20,engine='mock',
                stop_on_divergence=False,output_callback=outputs.append,
                diagnostic_callback=diagnostics.append,batch_callback=batch)
        np.savez_compressed(args.out/'outputs.npz',mock=outputs[0])
        result.update(complete=True,diagnostics=diagnostics)
    except MemoryError as exc:
        result['error']=f'MemoryError: {exc}'
    result['after_mock']=snapshot();save()
    result['peak_rss_bytes']=result['after_mock']['peak_rss_bytes']
    return json_safe(result)


def alignment(args):
    """Run unchanged alignment, recording tensor results; omit final Mock only."""
    from ..planners import transformer_tuner as tuner
    from ..planners.transformer_session import json_safe
    program,ctx=context(args)
    original=tuner.validate;rows=[]
    def validate(*values,**options):
        if options.get('engine')=='mock':
            return rows[0]['validation']
        value=original(*values,**options)
        rows.append(dict(validation=value))
        return value
    with patch.object(tuner,'validate',validate),tuner.tune_rule_scope('v8-6.14'):
        result=tuner.tune(program,ctx,budget_s=0,max_sweeps=1)
    return json_safe(dict(complete=True,llm_calls=0,report_images=0,program=str(args.program),
        final_mock_omitted=True,alignment=result.history[0],tensor_evaluations=rows,
        changes=result.changes,feasible=result.feasible))
