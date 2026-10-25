"""U3b serial CNN Agent acceptance with tensor profiles and external Mock audits."""

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import fcntl
import gc
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

from ..paths import RESULTS_DIR, REPO, WORK_ROOT, PLATFORM_DIR, DACAPO_ROOT, DACAPO_UPSTREAM, HECATE_BUILD
from ..run import _write
from .r8g_mlp_rerun import _calls, _stop, INNER_SECONDS
from .r8h_rerun import trace_summary


ORDER = ('resnet20_relu', 'squeezenet_silu_avgpool', 'squeezenet_relu_maxpool',
         'mobilenet_silu', 'mobilenet_relu', 'vgg16_silu_avgpool',
         'vgg16_relu_maxpool', 'alexnet_silu_avgpool', 'alexnet_relu_maxpool')
API_BUDGET = 50
GLOBAL_LIMIT = 113
BASELINE_REPORT = 'u3a-complete-20261024T161354727454/REPORT.json'


def read(path):
    return json.loads(Path(path).read_text())


def limits():
    resource.setrlimit(resource.RLIMIT_AS, (4*1024**3, 6*1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (8*1024**3, 8*1024**3))


def context(source_case):
    from ..data.samples import LegacyCifar, StoredSplits
    from ..models.info import ModelInfo
    from ..planners.agent_tools import build_context
    case = Path(source_case)
    retained = read(case/'ACCEPTANCE.json')
    info = ModelInfo.load(case)
    samples = StoredSplits(Path(retained.get('sample_source_dir', str(case/'samples'))), info)
    ctx = build_context(info, read(case/'calibration.json'), samples=samples)
    # U3a retained calibration/heldout only; later splits use the same lazy source.
    if info.input_handling.get('internal_regression_source')!='legacy_cifar':
        raise ValueError('U3b requires the fixed legacy_cifar source')
    ctx.samples = LegacyCifar(info.graph)
    ctx.prompt_version = 'r8-v5'
    return ctx


def split_context(ctx, split, directory):
    """Compute report-only references once; never expose them to the Agent."""
    import numpy as np
    from ..analyzer import reference_forward
    arrays, labels = ctx.samples.get(split)
    images = arrays[ctx.graph.inputs[0]]
    path = Path(directory)/(split+'-reference.npz')
    if path.exists():
        with np.load(path) as archive:
            exact = archive['logits']
    else:
        rows = []
        for start in range(0, len(images), 20):
            values = reference_forward(ctx.graph, {k:v[start:start+20] for k,v in arrays.items()})
            rows.append(values[ctx.graph.outputs[0]].copy())
        exact = np.concatenate(rows)
        np.savez_compressed(path, logits=exact)
    return replace(ctx, val_images=images, val_labels=labels, val_exact_logits=exact,
                   val_exact_top1=float(100*np.mean(exact.argmax(axis=1)==labels)),
                   val_inputs=arrays, val_split='test[0:8000]' if split=='report' else split)


def report_evaluation(program, ctx, *, engine_profile='mock'):
    from ..planners.agent_tools import validate
    if engine_profile not in ('mock', 'tensor'):
        raise ValueError('invalid engine_profile')
    return validate(program, ctx, stop_on_divergence=False,
                    **(dict(engine='tensor') if engine_profile=='tensor' else {}))


def mock_spot(program, ctx, directory, role, split, count):
    import numpy as np
    from ..planners.agent_tools import validate
    path = directory/(role+'-'+split+'-spot.json')
    if path.exists():
        saved = read(path)
        if saved.get('complete'):
            return saved
        raise RuntimeError('Prior Mock audit did not pass; retain its evidence')
    audit_ctx = replace(ctx, val_images=ctx.val_images[:count], val_labels=ctx.val_labels[:count],
        val_exact_logits=ctx.val_exact_logits[:count],
        val_inputs={k:v[:count] for k,v in ctx.val_inputs.items()})
    results, outputs = {}, {}
    for engine in ('tensor', 'mock'):
        captured = []
        results[engine] = validate(program, audit_ctx, stop_on_divergence=False,
            engine=engine, output_callback=lambda values:captured.append(values))
        outputs[engine] = captured[0]
        gc.collect()
    tensor, mock = outputs['tensor'], outputs['mock']
    indices = results['tensor']['diverged_indices']
    same_divergence = indices == results['mock']['diverged_indices']
    mask = np.ones(count, dtype=bool)
    mask[indices] = False
    error = np.abs(mock[mask]-tensor[mask])
    finite = np.isfinite(mock[mask]).all() and np.isfinite(tensor[mask]).all()
    logits_equal = bool(finite and np.all(error<=1e-9*(1+np.abs(tensor[mask]))))
    classes_equal = bool(np.array_equal(mock[mask].argmax(axis=1), tensor[mask].argmax(axis=1)))
    result = dict(complete=same_divergence and logits_equal and classes_equal,
        role=role, split=split, images=count,
        max_abs_diff=(float(error.max()) if error.size else 0.) if finite else None,
        max_normalized_diff=(float((error/(1+np.abs(tensor[mask]))).max()) if error.size else 0.) if finite else None,
        logits_equal=logits_equal, classes_equal=classes_equal, divergence_equal=same_divergence,
        diverged_indices={k:v['diverged_indices'] for k,v in results.items()})
    np.savez_compressed(directory/(role+'-'+split+'-spot.npz'), **outputs)
    _write(path, result)
    if not result['complete']:
        raise RuntimeError('Mock/tensor audit mismatch: '+role+'/'+split)
    return result


def summarize(root):
    """Compare only frozen programs; report data never enters candidate selection."""
    from collections import Counter
    from statistics import median
    from ..dsl.parser import parse
    batch=read(root/'REPORT.json')
    rows=[]
    for name in ORDER:
        entry=batch['models'][name]
        case=root/'models'/name
        current=read(case/'REPORT.json') if (case/'REPORT.json').exists() else {}
        retained=read(Path(batch['source_cases'][name])/'ACCEPTANCE.json')
        result=read(case/'AGENT.json') if (case/'AGENT.json').exists() else {}
        selected=(result.get('confirm') or {}).get('selected')
        agent_cost=current.get('cost') or result.get('best') or {}
        settings=[]
        if selected is not None:
            for index,op in enumerate(op for op in parse(selected['source']).ops if op.kind in ('act','maxpool')):
                settings.append(dict(index=index,kind=op.kind,fn=op.attrs.get('fn'),method=op.attrs['method'],
                    deg=op.attrs.get('deg'),degrees=op.attrs.get('degrees'),domain=op.attrs['domain'],
                    line=op.lineno,call_lines=list(op.call_lines)))
        baseline=parse((Path(batch['source_cases'][name])/'program.ptdsl').read_text())
        rule_settings=[dict(index=i,kind=op.kind,fn=op.attrs.get('fn'),method=op.attrs['method'],
            deg=op.attrs.get('deg'),degrees=op.attrs.get('degrees'),domain=op.attrs['domain'])
            for i,op in enumerate(op for op in baseline.ops if op.kind in ('act','maxpool'))]
        distribution=Counter(json.dumps({k:r[k] for k in ('kind','fn','method','deg','degrees','domain')},sort_keys=True)
                             for r in settings)
        confirmation=[r for r in (result.get('confirm') or {}).get('evals',[]) if selected and r['passed']
            and r['iteration']==selected['iteration'] and r['origin']==selected['origin']]
        trace=current.get('trace_summary') or {}
        report_metrics=current.get('report_metrics') or {}
        compiled=current.get('compile_result') or {}
        agent_boots=agent_cost.get('boots_pred') if selected else None
        latency=agent_cost.get('latency_pred_us') if selected else None
        rows.append(dict(model=name,complete=entry['complete'],stage=entry['stage'],
            agent_status=result.get('status'),report_path=str(case/'REPORT.json'),
            selected=selected,approximation_points=settings,rule_approximation_points=rule_settings,
            approximation_distribution=[dict(**json.loads(k),count=v) for k,v in distribution.items()],
            boots_pred=agent_boots,boots_compiled=compiled.get('boots_compiled'),
            rotations=(current.get('cost') or {}).get('counts',{}).get('rot'),latency_pred_us=latency,
            rule_boots=retained['boots_pred'],rule_boots_compiled=retained.get('boots_compiled'),
            rule_rotations=retained['rotations'],rule_latency_pred_us=retained['latency_pred_us'],
            boot_reduction_pct=100*(retained['boots_pred']-agent_boots)/retained['boots_pred'] if agent_boots is not None else None,
            latency_reduction_pct=100*(retained['latency_pred_us']-latency)/retained['latency_pred_us'] if latency is not None else None,
            confirm_acc_drop_pp=confirmation[0]['acc_drop_pp'] if confirmation else None,
            confirm_evals=(result.get('confirm') or {}).get('evals',[]),
            rule_confirm=current.get('rule_confirm_metrics'),
            agent_report=report_metrics.get('agent'),rule_report=report_metrics.get('rule'),
            api_calls=entry.get('api_calls',0),iterations=result.get('iterations',0),
            tune_calls=trace.get('tune_calls',0),tuner_seconds=trace.get('tuner_seconds',0),
            seconds=entry.get('seconds',0),native_compiled=compiled.get('native_compiled',False),
            server_job=compiled.get('server_job'),compile_result=compiled,spots=current.get('spots',{})))
    improvements=[r['boot_reduction_pct'] for r in rows if r['boot_reduction_pct'] is not None and r['boot_reduction_pct']>0]
    summary=dict(round='U3b',complete=batch['complete'],models=rows,model_denominator=9,
        agent_report_accuracy_pass=sum(r['agent_report'] is not None and r['agent_report']['acc_drop_pp']<=.5 for r in rows),
        rule_report_accuracy_pass=sum(r['rule_report'] is not None and r['rule_report']['acc_drop_pp']<=.5 for r in rows),
        fewer_boots_models=len(improvements),median_boot_reduction_pct=median(improvements) if improvements else None,
        api_calls=batch['api_calls'],api_total=batch['api_total'],api_limit=batch['api_limit'],
        prompt_version='r8-v5',engine_profile='tensor',tune_policy='new_best',
        maxpool_comparisons=[dict(model=r['model'],agent=[p for p in r['approximation_points'] if p['kind']=='maxpool'],
            rule=[p for p in r['rule_approximation_points'] if p['kind']=='maxpool']) for r in rows
            if any(p['kind']=='maxpool' for p in r['rule_approximation_points'])])
    _write(root/'SUMMARY.json',summary)
    return summary


class RetainedClient:
    """Keep each provider response before parsing or numerical feedback begins."""
    def __init__(self, client, directory):
        self.client, self.directory = client, directory

    def chat(self, messages, *, purpose):
        path = self.directory/'responses'/purpose
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.with_suffix('.json').exists():
            raise RuntimeError('Existing response retained; generation is not replayed')
        result = self.client.chat(messages, purpose=purpose)
        _write(path.with_suffix('.json'), result)
        try:
            source = json.loads(result['content'])['program']
        except (ValueError, KeyError, TypeError):
            pass
        else:
            if isinstance(source, str):
                path.with_suffix('.ptdsl').write_text(source)
        return result


def complete_retained_confirmation(ctx, case, *, prior_seconds):
    """Finish a completed six-round loop using its preserved candidate pool."""
    from .r8e6_reconfirm import reconstruct_candidates
    from ..planners.confirm import confirm_select
    rows=[json.loads(line) for line in (case/'agent_trace.jsonl').read_text().splitlines()]
    if [r['iteration'] for r in rows]!=list(range(1,7)) or any(r['fb']['stage']=='llm' for r in rows):
        raise RuntimeError('Generation is incomplete; no automatic replay')
    candidates,comparisons=reconstruct_candidates(case/'agent_trace.jsonl',ctx.weights,profile=ctx.profile)
    if not all(row['matches'] for row in comparisons):
        raise RuntimeError('Retained candidate costs differ')
    marker=case/'retained-confirmation-started.json'
    if marker.exists():
        raise RuntimeError('Retained confirmation already started; no repeat')
    _write(marker,dict(candidates=[asdict(c) for c in candidates],cost_comparisons=comparisons,
        api_calls=0,retuning=False,prior_seconds=prior_seconds))
    started=time.monotonic()
    confirmation=confirm_select(candidates,ctx,engine_profile='tensor')
    selected=confirmation['selected']
    best=None
    lines=None
    if selected is not None:
        row=rows[selected.iteration-1]
        original=json.loads(next(m['content'] for m in row['messages'] if m['role']=='assistant'))['program']
        tuned=row.get('tuned') or {}
        cost=row['fb']['cost'] if selected.origin=='llm' else tuned['cost_after']
        values=row['fb']['validation'] if selected.origin=='llm' else tuned['history'][-1]
        best=dict(cost,iteration=selected.iteration,source=selected.origin,agree=values['agree'],
            max_u=values['max_u'],boot_input_max_abs=values.get('boot_input_max_abs'),
            worst_acts=values.get('worst_acts',[]))
        lines=sum(bool(line.strip()) for line in original.splitlines())
    return dict(status='feasible' if selected else 'confirm_failed',error=None,
        best_program=selected.source if selected else None,best_iteration=selected.iteration if selected else None,
        best_source=selected.origin if selected else None,best=best,llm_source_lines=lines,
        frozen_source_lines=sum(bool(s.strip()) for s in selected.source.splitlines()) if selected else None,
        first_feasible_iteration=min((c.iteration for c in candidates),default=None),
        iterations=len(rows),llm_calls=len(rows),pool_size=len(candidates),tune_policy='new_best',
        prompt_tokens=sum(r['usage'].get('prompt_tokens',0) for r in rows),
        completion_tokens=sum(r['usage'].get('completion_tokens',0) for r in rows),
        seconds=prior_seconds+time.monotonic()-started,confirm=confirmation,
        restored_completed_loop=True,regenerated=False,retuned=False)


def model_inside(args):
    from ..cost import estimate
    from ..dsl.parser import parse
    from ..export import export_package
    from ..llm import LLMClient
    from ..planners.agent import run_agent
    from ..planners.confirm import build_confirm_context
    from ..run import _server_job
    from ..tensor_sim import clear_caches
    from ..toolchain import TRUSTED_LIMITS
    from .u2_multi_layout import compile_case
    case = args.out
    path = case/'REPORT.json'
    record = read(path) if path.exists() else dict(round='U3b', model=args.model, complete=False,
        prompt_version='r8-v5', engine_profile='tensor', tune_policy='new_best', max_iters=6,
        source_case=str(args.source_case), stage='ready', spots={}, report_metrics={})
    if record.get('complete'):
        return record
    ctx = context(args.source_case)
    agent_path = case/'AGENT.json'
    if not agent_path.exists():
        if record.get('error')=='ValueError: Original samples for confirm have not been prepared':
            agent=complete_retained_confirmation(ctx,case,prior_seconds=args.prior_seconds)
        elif (case/'agent_trace.jsonl').exists() or list((case/'responses').glob('*.json')):
            raise RuntimeError('Interrupted generation retained; no automatic replay of a partial model')
        else:
            record['stage'] = 'agent'; _write(path, record)
            client = RetainedClient(LLMClient(args.budget_path, model='deepseek-flash',
                reasoning_effort='high', max_tokens=32768, max_total_calls=args.call_limit), case)
            agent = run_agent(ctx, client, max_iters=6, trace_path=case/'agent_trace.jsonl',
                tune_policy='new_best', tuner_budget_s=2400, engine_profile='tensor')
        agent['confirm']['selected'] = asdict(agent['confirm']['selected']) if agent['confirm']['selected'] else None
        _write(agent_path, agent)
    agent = read(agent_path)
    record.pop('error',None)
    record.update(agent_status=agent['status'], confirm=agent['confirm'],
                  trace_summary=trace_summary(case/'agent_trace.jsonl'))
    _write(path, record)
    if agent['status']=='error':
        raise RuntimeError('Agent error retained: '+str(agent['error']))
    source = agent.get('best_program')
    if source is not None:
        frozen = case/'program.ptdsl'
        if frozen.exists() and frozen.read_text()!=source:
            raise RuntimeError('Frozen program differs from selected candidate')
        if not frozen.exists():frozen.write_text(source)
    # External measurements start after the generation/selection caches are released.
    clear_caches();gc.collect()
    roles = [('agent', source)] if source is not None else []
    roles.append(('rule', (args.source_case/'program.ptdsl').read_text()))
    if 'rule_confirm_metrics' not in record:
        # This fixed baseline measurement follows the Agent's completed confirmation.
        record['stage'] = 'rule-confirm-report'; _write(path, record)
        confirmed = report_evaluation(parse(roles[-1][1]),build_confirm_context(ctx),
                                      engine_profile='tensor')
        record['rule_confirm_metrics'] = {k:v for k,v in confirmed.items() if k!='rows'}
        _write(path, record)
    for role, text in roles:
        clear_caches();gc.collect()
        program = parse(text)
        record['stage'] = role+'-heldout-spot'; _write(path, record)
        record['spots'][role+'-heldout'] = mock_spot(program,ctx,case,role,'heldout',20)
        _write(path, record)
    # This happens after run_agent and its one confirmation have returned.
    report_ctx = split_context(ctx,'report',case)
    for role, text in roles:
        clear_caches();gc.collect()
        program = parse(text)
        record['stage'] = role+'-report-spot'; _write(path, record)
        record['spots'][role+'-report'] = mock_spot(program,report_ctx,case,role,'report',100)
        _write(path, record)
        if role not in record['report_metrics']:
            record['stage'] = role+'-report'; _write(path, record)
            clear_caches();gc.collect()
            metrics = report_evaluation(program,report_ctx,engine_profile='tensor')
            record['report_metrics'][role] = {k:v for k,v in metrics.items() if k!='rows'}
            _write(path, record)
    if source is not None and 'compile_result' not in record:
        record['stage'] = 'export'; _write(path, record)
        program = parse(source)
        cost = estimate(program,ctx.weights,profile=ctx.profile)
        record['cost'] = cost
        export = case/'export'
        package = export_package(source,program,ctx.weights,export)
        weights = export/'weights'/(program.name+'.npz')
        result = dict(slots=program.header.slots,boots_pred=cost['boots_pred'],
                      latency_pred_us=cost['latency_pred_us'],model=args.model)
        reason = ('constants exceed local 1.5 GiB allowance' if cost['const_bytes']>1.5*1024**3 else
                  'free disk is below the trusted runner requirement' if
                  shutil.disk_usage(case).free<TRUSTED_LIMITS['min_free_gib']*1024**3 else None)
        if reason is not None:
            result.update(native_compiled=False,boots_compiled=None,
                reason=reason,server_job=str(_server_job(case,package,weights)))
        else:
            record['stage']='compile'; _write(path,record)
            native=case/'native';native.mkdir(exist_ok=True)
            if (native/'REPORT.json').exists():
                saved=read(native/'REPORT.json')
                if not saved.get('pruned'):
                    raise RuntimeError('Incomplete prior compile retained; no automatic recompile')
                result=saved
            else:
                compile_case(native,result,package,weights,program.name)
            result['native_compiled']=True
        record['compile_result']=result
        _write(path,record)
    record.update(complete=True,stage='confirm_failed' if source is None else 'complete',
                  peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    _write(path,record)
    clear_caches();gc.collect()
    return record


def prepare_inside(args):
    from ..llm import Config
    from ..planners.prompts import build_system_prompt, initial_message
    retained = read(args.baseline_report)
    cases = {r['model']:str(Path(r['acceptance']).parent) for r in retained['models']}
    manifest = dict(round='U3b',order=list(ORDER),source_cases=cases,
        endpoint='https://api.deepseek.com/chat/completions',requests=[],
        prompt_version='r8-v5',engine_profile='tensor',tune_policy='new_best',
        max_iters=6,max_confirm_evals=6,max_sweeps=1,tuner_budget_s=2400,
        llm_model='deepseek-flash',reasoning_effort='high',max_tokens=32768,
        api_budget=50,global_limit=113,retries=0,extra_connectivity_requests=0,
        initial_payload='Model structure, shapes, parameter names, calibration/heldout statistics, r8-v5 prompts',
        followup_payload='Generated DSL, internal parse/exact/validation, tuning and predicted cost feedback',
        excluded_payload='Raw images, raw weights, confirmation/report results, compilation/CKKS results',
        credentials_read=False,network_calls=0)
    for name in ORDER:
        ctx=context(cases[name])
        body=Config(model='deepseek-flash',reasoning_effort='high',max_tokens=32768,timeout_seconds=600).settings()
        body['messages']=[dict(role='system',content=build_system_prompt(ctx)),dict(role='user',content=initial_message(ctx))]
        path=args.out/(name+'.first-request.pretty.json')
        if path.exists():raise FileExistsError(path)
        _write(path,body)
        manifest['requests'].append(dict(model=name,path=str(path),bytes=path.stat().st_size))
        _write(args.out/'manifest.json',manifest)
        del ctx,body;gc.collect()
    manifest['complete']=True;_write(args.out/'manifest.json',manifest)
    return manifest


def test_inside(args):
    import unittest
    groups={'packaging':{'test_p2_export.py','test_toolchain_decoupled.py'},
            'tensor':{'test_r8f_tensor_validate.py'}}
    files=sorted(Path('pt_agent/tests').glob('test_*.py'))
    if args.part=='unit':files=[p for p in files if p.name.startswith('test_u3b_')]
    elif args.part=='intake':
        python=PLATFORM_DIR/'venvs/intake-2.0.1-cpu/bin/python'
        if Path(sys.executable).absolute()!=python.absolute():
            os.execv(str(python),[str(python),'-B','-m','pt_agent.experiments.u3b_agent_cnn',*sys.argv[1:]])
        files=[p for p in files if p.name=='test_u1_onnx.py']
    else:
        files=[p for p in files if not p.name.startswith('test_u3b_')]
        excluded=groups['packaging']|groups['tensor']
        files=([p for p in files if p.name in groups[args.part]] if args.part in groups else
               [p for p in files if p.name not in excluded])
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName('pt_agent.tests.'+p.stem) for p in files)
    expected={'main':421,'packaging':17,'tensor':17,'intake':4}
    if args.part in expected and suite.countTestCases()!=expected[args.part]:
        raise RuntimeError('455-test inventory differs')
    result=unittest.TextTestRunner(verbosity=2,failfast=True).run(suite)
    report=dict(complete=result.wasSuccessful(),tests=result.testsRun,failures=len(result.failures),
                errors=len(result.errors),skipped=len(result.skipped),llm_calls=0)
    if not report['complete']:report['error']='Test failure; see run.log'
    return report


def inside(args):
    limits()
    if args.part=='summary':
        summarize(args.out)
        return 0
    import torch
    torch.set_num_threads(1)
    report=dict(complete=False,part=args.part,llm_calls=0)
    try:
        if args.part=='prepare':report.update(prepare_inside(args))
        elif args.part=='model':report=model_inside(args)
        else:report.update(test_inside(args))
        if 'complete' not in report:report['complete']=True
        if args.part=='prepare':report['complete']=True
    except Exception as exc:
        import traceback
        traceback.print_exc()
        if args.part=='model' and (args.out/'REPORT.json').exists():
            report=read(args.out/'REPORT.json')
        report.update(complete=False,error=f'{type(exc).__name__}: {exc}',
                      peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    _write(args.out/'REPORT.json',report)
    return 0 if report['complete'] else 1


def environment():
    return dict(os.environ,POSEIDON_WORK_ROOT=str(WORK_ROOT),PT_DACAPO_ROOT=str(DACAPO_ROOT),
        PT_DACAPO_UPSTREAM=str(DACAPO_UPSTREAM),PT_HECATE_BUILD=str(HECATE_BUILD),
        PT_AGENT_RESULTS_ROOT=str(RESULTS_DIR))


def outer_part(args):
    from ..registry import append,git_rev,new_run_id
    args.out=(args.out or RESULTS_DIR/('u3b-'+args.part+'-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f'))).resolve()
    args.out.mkdir(parents=True,exist_ok=False)
    seconds=3600
    command=[sys.executable,'-B','-m','pt_agent.tools.nixrun','--seconds',str(seconds),'--',
        '-m','pt_agent.experiments.u3b_agent_cnn','--inside','--part',args.part,'--out',str(args.out),
        '--baseline-report',str(args.baseline_report.resolve())]
    _write(args.out/'argv.json',command);print('RUN='+str(args.out),flush=True)
    with (args.out/'run.log').open('w') as stream:
        status=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,env=environment(),preexec_fn=limits).returncode
    report=read(args.out/'REPORT.json') if (args.out/'REPORT.json').exists() else dict(complete=False)
    report['exit']=status;_write(args.out/'REPORT.json',report)
    os.environ['PT_RESULTS']=str(RESULTS_DIR)
    append(dict(run_id=new_run_id('U3b',args.part,'offline'),round='U3b',model=args.part,mode='offline',
        git_rev=git_rev(),metrics=report,artifacts_dir=str(args.out),ts=datetime.now(timezone.utc).isoformat()))
    print('REPORT='+str(args.out/'REPORT.json'),flush=True)
    return status


def outer(args):
    from ..registry import append,git_rev,load,new_run_id
    root=(args.resume or args.out or RESULTS_DIR/('u3b-agent-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S'))).resolve()
    root.mkdir(parents=True,exist_ok=True)
    os.environ['PT_RESULTS']=str(RESULTS_DIR)
    with (root/'run.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=root/'REPORT.json';budget=RESULTS_DIR/'llm_calls.jsonl'
        if args.resume:batch=read(path)
        else:
            if path.exists():raise FileExistsError('Use --resume')
            manifest=read(args.review/'manifest.json')
            if not manifest['complete'] or manifest['order']!=list(ORDER):raise ValueError('Incomplete review manifest')
            before=len(_calls(budget))
            batch=dict(round='U3b',complete=False,stage='ready',git_rev=git_rev(),order=list(ORDER),
                api_before=before,api_total=before,api_calls=0,api_budget=50,api_limit=min(before+50,113,300),
                review=str(args.review),source_cases=manifest['source_cases'],models={name:dict(complete=False,stage='not_started') for name in ORDER})
            _write(path,batch)
        recorded={r['metrics'].get('global_call') for r in load() if r['mode']=='llm-call'}
        for name in ORDER:
            entry=batch['models'][name]
            if entry.get('complete'):continue
            case=root/'models'/name;case.mkdir(parents=True,exist_ok=True)
            entry['report_path']=str(case/'REPORT.json');batch['active_model']=name
            api_before=entry.setdefault('api_before',len(_calls(budget)))
            if len(_calls(budget))>=batch['api_limit'] and not (case/'AGENT.json').exists():
                batch['stage']='budget_exhausted';break
            if entry.get('worker_pid'):
                try:os.kill(entry['worker_pid'],0)
                except ProcessLookupError:pass
                else:raise RuntimeError('Previous model worker still running')
            prior_seconds=entry.get('seconds',0)
            remaining=INNER_SECONDS-int(prior_seconds)
            if remaining<=0:raise RuntimeError('Per-model allowance exhausted')
            command=[sys.executable,'-B','-m','pt_agent.tools.nixrun','--seconds',str(remaining),'--',
                '-m','pt_agent.experiments.u3b_agent_cnn','--inside','--part','model','--model',name,
                '--out',str(case),'--source-case',batch['source_cases'][name],
                '--budget-path',str(budget),'--call-limit',str(batch['api_limit']),
                '--prior-seconds',str(prior_seconds)]
            entry.pop('error',None)
            entry.update(stage='running',command=command);batch['stage']='running';_write(path,batch)
            begin=time.monotonic();process=None
            def account(*,wait=False):
                calls=_calls(budget,wait=wait)
                if calls is None:return
                for index in range(api_before+1,len(calls)+1):
                    if index not in recorded:
                        call=calls[index-1]
                        append(dict(run_id=new_run_id('U3b',name,f'api-{index}'),round='U3b',model=name,
                            mode='llm-call',git_rev=batch['git_rev'],metrics=dict(call,global_call=index),
                            artifacts_dir=str(case),ts=call['ts']))
                        recorded.add(index)
                entry['api_calls']=len(calls)-api_before
                batch.update(api_calls=len(calls)-batch['api_before'],api_total=len(calls))
            try:
                with (case/'stdout.txt').open('a') as stdout,(case/'stderr.txt').open('a') as stderr:
                    process=subprocess.Popen(command,stdout=stdout,stderr=stderr,env=environment(),start_new_session=True)
                    entry['worker_pid']=process.pid;_write(path,batch)
                    while True:
                        try:code=process.wait(timeout=2);break
                        except subprocess.TimeoutExpired:
                            account();entry['seconds']=prior_seconds+time.monotonic()-begin;_write(path,batch)
                            if entry['seconds']>INNER_SECONDS+120:raise TimeoutError('Per-model allowance exhausted')
            except (Exception,KeyboardInterrupt) as exc:
                _stop(process);code=1;entry['error']=f'{type(exc).__name__}: {exc}'
            account(wait=True)
            current=read(case/'REPORT.json') if (case/'REPORT.json').exists() else {}
            entry.update(stage=current.get('stage','error'),complete=current.get('complete',False) and code==0,
                seconds=prior_seconds+time.monotonic()-begin,error=current.get('error') or entry.get('error'))
            entry.pop('worker_pid',None);_write(path,batch)
            append(dict(run_id=new_run_id('U3b',name,'model'),round='U3b',model=name,mode='agent-cnn',
                git_rev=batch['git_rev'],metrics=entry,artifacts_dir=str(case),ts=datetime.now(timezone.utc).isoformat()))
            if not entry['complete']:
                batch['stage']='error';break
        batch['complete']=all(r['complete'] for r in batch['models'].values())
        if batch['complete']:batch['stage']='complete'
        batch['active_model']=None;_write(path,batch)
        # Parsing summary programs needs the pinned numerical Python environment.
        command=[sys.executable,'-B','-m','pt_agent.tools.nixrun','--seconds','3600','--',
            '-m','pt_agent.experiments.u3b_agent_cnn','--inside','--part','summary','--out',str(root)]
        with (root/'summary.log').open('a') as stream:
            subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,env=environment(),
                           preexec_fn=limits,check=True)
        summary=read(root/'SUMMARY.json')
        append(dict(run_id=new_run_id('U3b','nine-models','batch'),round='U3b',model='nine-models',
            mode='agent-cnn-summary',git_rev=batch['git_rev'],metrics=summary,artifacts_dir=str(root),
            ts=datetime.now(timezone.utc).isoformat()))
        print('REPORT='+str(path),flush=True)
        return 0 if batch['complete'] else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--part',choices=('unit','main','packaging','tensor','intake','prepare','model','summary','run'),default='run')
    parser.add_argument('--out',type=Path)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--review',type=Path)
    parser.add_argument('--baseline-report',type=Path,default=RESULTS_DIR/BASELINE_REPORT)
    parser.add_argument('--inside',action='store_true')
    parser.add_argument('--model',choices=ORDER)
    parser.add_argument('--source-case',type=Path)
    parser.add_argument('--budget-path',type=Path)
    parser.add_argument('--call-limit',type=int)
    parser.add_argument('--prior-seconds',type=float,default=0.)
    args=parser.parse_args()
    if args.inside:return inside(args)
    if args.part=='summary':parser.error('summary is an internal stage')
    if args.part=='run':
        if not args.review and not args.resume:parser.error('run requires --review or --resume')
        return outer(args)
    return outer_part(args)


if __name__=='__main__':
    raise SystemExit(main())
