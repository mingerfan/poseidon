"""U5 finite compatibility, held-out study, native package and regression checks."""

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
from types import SimpleNamespace
import unittest

from ..paths import RESULTS_DIR, REPO, PLATFORM_DIR
from .u5_fitting import write


BASE = 'afcc7654'
MODELS = ('resnet20','resnet20_silu','cnn_plain','mlp')


def compatibility(args):
    from . import s1_acceptance
    from .u2_multi_layout import baseline_package
    from ..models.zoo import get
    from ..planners import agent_tools, prompts
    from . import s1_managed
    from ..tests import test_p2_export
    s1_acceptance.BASE = BASE
    # Reuse the existing read-only comparisons with the outer process's paths.
    s1_acceptance.RESULTS_DIR = s1_managed.RESULTS_DIR = args.results_root
    test_p2_export.FROZEN = args.frozen_dir
    test_p2_export.MODELS = args.models_dir
    test_p2_export.R6B = args.results_root/'r6b-refresh-20261021T025529/REPORT.json'
    result = s1_acceptance.compatibility(args)
    result.update(round='U5',prompts=[])
    old = baseline_package(args.snapshot)
    for name in MODELS:
        model=get(name,args.models_dir)
        graph=model.import_graph()
        calibration=json.loads((args.frozen_dir/'models'/name/'calibration.json').read_text())
        before=old('planners.agent_tools').build_context(graph,calibration,n_val=20,normalization=model.normalization)
        after=agent_tools.build_context(graph,calibration,n_val=20,normalization=model.normalization)
        for version in ('r8-v4','r8-v5'):
            before.prompt_version=after.prompt_version=version
            system_a=old('planners.prompts').build_system_prompt(before)
            system_b=prompts.build_system_prompt(after)
            input_a=old('planners.prompts').initial_message(before)
            input_b=prompts.initial_message(after)
            if system_a!=system_b or input_a!=input_b:
                raise AssertionError(name+'/'+version+': frozen prompt text changed')
            result['prompts'].append(dict(model=name,version=version,system_equal=True,input_equal=True))
        del model,graph,before,after
        gc.collect()
    result['complete']=True
    return result


def study(args):
    from ..dsl.parser import parse
    from ..dsl.printer import print_program
    from ..models.zoo import get
    from ..planners.agent_tools import build_context,validate
    from ..planners.tuner import LADDER
    from ..tensor_sim import clear_caches
    names=MODELS if args.model is None else (args.model,)
    result=dict(round='U5',part='study',complete=False,models=[],llm_calls=0,report_images=0,
                split='test[9000:10000]',inputs_per_variant=1000)
    for name in names:
        model=get(name,args.models_dir)
        graph=model.import_graph()
        calibration=json.loads((args.frozen_dir/'models'/name/'calibration.json').read_text())
        program=parse((args.frozen_dir/'models'/name/'program.ptdsl').read_text())
        ctx=build_context(graph,calibration,n_val=1000,normalization=model.normalization)
        row=dict(model=name,variants=[],cheb_points=sum(op.kind in ('act','maxpool') and
                 op.attrs.get('method')=='cheb' for op in program.ops))
        result['models'].append(row)
        folder=args.out/name
        folder.mkdir()
        def record(label,candidate,**details):
            started=time.monotonic()
            (folder/(label+'.ptdsl')).write_text(print_program(candidate))
            measured=validate(candidate,ctx,engine='tensor',cache=True,stop_on_divergence=False)
            item=dict(label=label,metrics=measured,seconds=time.monotonic()-started,**details)
            row['variants'].append(item)
            write(args.out/'REPORT.json',result)
            print('STUDY '+name+' '+label+' '+json.dumps({k:measured.get(k) for k in
                  ('acc_drop_pp','max_u','n_diverged','feasible')}),flush=True)
        record('original',program)
        changed=replace(program,ops=[replace(op,attrs={**op.attrs,'method':'remez'})
            if op.kind in ('act','maxpool') and op.attrs.get('method')=='cheb' else op for op in program.ops])
        if row['cheb_points']:
            record('same-degree-remez',changed)
        else:
            row['no_cheb_points']=True
            row['same_degree_reused_original']=True
        candidates=[]
        lowest=[]
        for index,op in enumerate(changed.ops):
            if op.kind not in ('act','maxpool') or op.attrs.get('method')!='remez':
                continue
            degree=op.attrs['deg']
            lower_degree=next((value for value in LADDER if value < degree),None)
            if lower_degree is None:
                lowest.append(dict(line=op.lineno,degree=degree))
                continue
            ops=list(changed.ops)
            ops[index]=replace(op,attrs={**op.attrs,'deg':lower_degree})
            label='lower-line-'+str(op.lineno)
            record(label,replace(changed,ops=ops),line=op.lineno,degree_before=degree,degree_after=lower_degree)
            candidates.append(row['variants'][-1])
        row.update(lowered_candidates=len(candidates),lowered_feasible=sum(v['metrics']['feasible'] for v in candidates),
                   already_lowest=lowest,complete=True)
        clear_caches()
        del model,graph,ctx,program,changed
        gc.collect()
        write(args.out/'REPORT.json',result)
    result['complete']=True
    return result


def native(args):
    from .. import toolchain
    from ..dsl.parser import parse
    from ..export import export_package
    from ..manage import read
    from ..run import _compile_export
    from ..tests.test_u5_package import MIXED
    from .s1_managed import compare_ir
    from .s0_pipeline_spike import compare_plans
    program=parse(MIXED)
    result=dict(round='U5',part='native',complete=False,modes={},llm_calls=0,report_images=0)
    toolchain.setup_env()
    from hecate_python_env import VENV
    for mode in ('compiler','managed'):
        directory=args.out/mode
        package=export_package(MIXED,program,{},directory)
        weights=directory/'weights'/(program.name+'.npz')
        if mode=='compiler':
            status=_compile_export(dict(package=str(package),weights=str(weights),mode=mode))
            result['modes'][mode]=status
            write(args.out/'REPORT.json',result)
            if status['compile'] is None or status['compile']['exit']:
                raise RuntimeError('Mixed P2 compiler mode failed')
            toolchain.prune(directory/'compiled',keep=('lowered._hecate_'+program.name+'.bundle',))
            continue
        trace=directory/'trace'
        traced=toolchain.trusted_run(directory,[str(VENV/'bin/python'),'-B',str(package/'trace_model.py'),
            '--mode','managed','--hecate-opt',str(toolchain.host_paths()['hecate_opt']),
            '--weights',str(weights),'--output-dir',str(trace)],directory/'trace.log',
            env_extra=toolchain.package_trace_env())
        result['modes'][mode]=dict(trace=traced)
        write(args.out/'REPORT.json',result)
        if traced['exit']:
            raise RuntimeError('Mixed P2 managed trace failed')
        profile=toolchain.management_profile()
        a=read(args.out/'compiler/trace/trace_model.mlir',profile)
        b=read(trace/'preprocessed.earth.mlir',profile)
        result['stage0']=compare_ir(a,b)
        del a,b
        gc.collect()
        if not result['stage0']['equal'] or result['stage0']['conclusion']!='相同':
            raise AssertionError('Mixed P2 stage 0 differs')
        toolchain.prune(args.out/'compiler/trace')
        toolchain.prune(trace,keep=('managed.mlir',))
        status=toolchain.trusted_run(directory,[str(VENV/'bin/python'),'-B',str(package/'compile_model.py'),
            '--mode','managed','--hecate-opt',str(toolchain.host_paths()['hecate_opt']),
            '--input',str(trace/'managed.mlir'),'--output-dir',str(directory/'compiled')],
            directory/'compile.log',env_extra=dict(LD_LIBRARY_PATH=os.environ['HECATE_PYTHON_LIBRARY_PATH']))
        result['modes'][mode]['compile']=status
        write(args.out/'REPORT.json',result)
        if status['exit']:
            raise RuntimeError('Mixed P2 managed compile failed')
    suffix='lowered._hecate_'+program.name+'.runtime-plan.json'
    result['plan']=compare_plans(args.out/'compiler/compiled'/suffix,args.out/'managed/compiled'/suffix)
    if not result['plan']['equal']:
        raise AssertionError('Mixed P2 RuntimePlan differs')
    result['pruned']={mode:{part:toolchain.prune(args.out/mode/part) for part in ('trace','compiled')}
                      for mode in ('compiler','managed')}
    result['complete']=True
    return result


def regression(args):
    if args.group=='intake':
        python=PLATFORM_DIR/'venvs/intake-2.0.1-cpu/bin/python'
        if Path(sys.executable).absolute()!=python.absolute():
            os.execv(str(python),[str(python),'-B','-m','pt_agent.experiments.u5_acceptance',*sys.argv[1:]])
    files=sorted(Path('pt_agent/tests').glob('test_*.py'))
    groups=dict(packaging={'test_p2_export.py','test_toolchain_decoupled.py'},
                tensor={'test_r8f_tensor_validate.py'},weight={'test_u3b2_weight_usage.py'},
                intake={'test_u1_onnx.py'})
    if args.group=='main':
        excluded=groups['packaging']|groups['tensor']|groups['weight']
        files=[p for p in files if p.name not in excluded and not p.name.startswith(('test_s1_','test_u5_'))]
    elif args.group=='s1':
        from ..tests import test_s1_stages
        test_s1_stages.OUTPUT_DIR=args.out/'native-shared-fanout'
        files=[p for p in files if p.name.startswith('test_s1_')]
    else:
        files=[p for p in files if p.name in groups[args.group]]
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName('pt_agent.tests.'+p.stem) for p in files)
    expected=dict(main=435,packaging=17,tensor=17,weight=17,intake=4,s1=34)
    if suite.countTestCases()!=expected[args.group]:
        raise AssertionError(f'Existing regression inventory differs: {suite.countTestCases()} vs {expected[args.group]}')
    result=unittest.TextTestRunner(verbosity=2,failfast=True).run(suite)
    return dict(round='U5',part='regression',group=args.group,complete=result.wasSuccessful(),
                tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
                skipped=len(result.skipped),llm_calls=0,report_images=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--part',choices=('compat','study','native','regression'),required=True)
    parser.add_argument('--model',choices=MODELS)
    parser.add_argument('--group',choices=('main','packaging','tensor','weight','intake','s1'))
    parser.add_argument('--inside',action='store_true')
    parser.add_argument('--out',type=Path)
    parser.add_argument('--snapshot',type=Path)
    parser.add_argument('--results-root',type=Path)
    parser.add_argument('--models-dir',type=Path)
    parser.add_argument('--frozen-dir',type=Path)
    args=parser.parse_args()
    if args.inside:
        resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,6*1024**3))
        resource.setrlimit(resource.RLIMIT_FSIZE,(8*1024**3,8*1024**3))
        resource.setrlimit(resource.RLIMIT_CORE,(0,0))
        result=dict(round='U5',part=args.part,complete=False,llm_calls=0,report_images=0)
        try:
            result=dict(compat=compatibility,study=study,native=native,regression=regression)[args.part](args)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            path=args.out/'REPORT.json'
            if path.exists():
                result.update(json.loads(path.read_text()))
            result.update(complete=False,error=f'{type(exc).__name__}: {exc}')
            if args.part=='native':
                from .. import toolchain
                result['pruned_after_stop']={str(p.relative_to(args.out)):toolchain.prune(p)
                    for p in args.out.rglob('*') if p.is_dir()}
        write(args.out/'REPORT.json',result)
        return 0 if result['complete'] else 1
    now=datetime.now(timezone.utc)
    args.results_root=args.results_root or RESULTS_DIR
    args.models_dir=args.models_dir or args.results_root/'models-v2'
    args.frozen_dir=args.frozen_dir or args.results_root/'r8e7-reconfirm-20261022T122558'
    run_id='u5-'+(args.model or args.group or args.part)+'-'+now.strftime('%Y%m%dT%H%M%S%f')
    directory=args.out or RESULTS_DIR/run_id
    directory.mkdir(parents=True,exist_ok=False)
    if args.part=='compat':
        import tarfile
        archive=directory/'baseline.tar'
        with archive.open('wb') as stream:
            subprocess.run(['git','archive',BASE,'DSL/DSL_agent/scripts/pt_agent'],cwd=REPO,stdout=stream,check=True)
        args.snapshot=directory/'baseline-tree'
        args.snapshot.mkdir()
        with tarfile.open(archive) as stream:
            stream.extractall(args.snapshot)
    command=[sys.executable,'-B','-m','pt_agent.tools.nixrun','--seconds','3600','--',
             '-m','pt_agent.experiments.u5_acceptance','--inside','--part',args.part,'--out',str(directory)]
    for key in ('model','group','snapshot','results_root','models_dir','frozen_dir'):
        if getattr(args,key) is not None:
            command += ['--'+key.replace('_','-'),str(getattr(args,key))]
    write(directory/'argv.json',command)
    print('RUN='+str(directory),flush=True)
    started=time.monotonic()
    with (directory/'run.log').open('w') as stream:
        status=subprocess.run(command,cwd=REPO/'DSL/DSL_agent/scripts',stdout=stream,stderr=subprocess.STDOUT)
    path=directory/'REPORT.json'
    result=json.loads(path.read_text()) if path.exists() else dict(complete=False,error='See run.log')
    rev=subprocess.check_output(['git','rev-parse','--short','HEAD'],text=True).strip()
    result.update(run_id=run_id,git_rev=rev,exit=status.returncode,seconds=time.monotonic()-started)
    write(path,result)
    with (RESULTS_DIR/'registry.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(run_id=run_id,round='U5',model=args.model or args.group or 'compatibility',
            mode=args.part,git_rev=rev,metrics=dict(complete=result['complete'],exit=status.returncode,
              tests=result.get('tests'),group=args.group,llm_calls=0),artifacts_dir=str(directory),ts=now.isoformat()))+'\n')
    print(json.dumps(dict(report=str(path),exit=status.returncode,complete=result['complete']),indent=2))
    return status.returncode


if __name__=='__main__':
    raise SystemExit(main())
