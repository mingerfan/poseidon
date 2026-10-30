"""Finite U4 external package/compiler acceptance and retained regressions."""

import argparse
from datetime import datetime,timezone
import gc
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import unittest

from ..paths import REPO,RESULTS_DIR,PLATFORM_DIR
from .u4_point_diagnosis import write


def native(args):
    from .. import toolchain
    from ..cost import estimate
    from ..dsl.parser import parse
    from ..export import export_package
    from ..models.info import ModelInfo
    from ..run import _compile_export,_server_job
    from .s0_pipeline_spike import compare_plans
    numerical=json.loads(args.evaluation.read_text())
    if not numerical['complete'] or numerical['program']!=str(args.program):
        raise ValueError('Native acceptance requires the completed Mock/tensor report for the same program')
    info=ModelInfo.load(args.prepared/'model');text=args.program.read_text();program=parse(text)
    result=dict(round='U4',part='native',model=args.model,complete=False,acceptance_complete=False,
                program=str(args.program),evaluation=str(args.evaluation),modes={},llm_calls=0,report_images=0)
    result['cost']=estimate(program,info.graph.weights)
    write(args.out/'REPORT.json',result)
    toolchain.setup_env()
    for mode in ('compiler','managed'):
        directory=args.out/mode
        package=export_package(text,program,info.graph.weights,directory)
        weights=directory/'weights'/(program.name+'.npz')
        record=dict(package=str(package),weights=str(weights),mode=mode)
        result['modes'][mode]=dict(export=record)
        write(args.out/'REPORT.json',result)
        if result['cost']['const_bytes']>1.5*1024**3:
            job=_server_job(directory,package,weights,mode=mode)
            subprocess.run(['bash','-n',str(job)],check=True)
            result['modes'][mode].update(server_job=str(job),local_compiled=False)
            continue
        status=_compile_export(record)
        result['modes'][mode].update(status)
        write(args.out/'REPORT.json',result)
        if status['compile'] is None or status['compile']['exit'] or status['plan'] is None:
            raise RuntimeError(f'{mode}: package trace/compile failed')
        same=status['plan']['boots']==result['cost']['boots_pred']
        result['modes'][mode]['boots_equal']=same
        result['modes'][mode]['pruned']=dict(trace=toolchain.prune(directory/'trace'),
            compiled=toolchain.prune(directory/'compiled',keep=('lowered._hecate_'+program.name+'.bundle',)))
        write(args.out/'REPORT.json',result)
        if not same:raise RuntimeError(f'{mode}: predicted/native bootstrap count differs')
    if any('server_job' in r for r in result['modes'].values()):
        result.update(complete=True,local_compiled=False,pending='server execution and full RuntimePlan comparison')
        return result
    suffix='lowered._hecate_'+program.name+'.runtime-plan.json'
    result['runtime_plan']=compare_plans(args.out/'compiler/compiled'/suffix,args.out/'managed/compiled'/suffix)
    write(args.out/'REPORT.json',result)
    if not result['runtime_plan']['equal']:raise RuntimeError('Compiler/managed RuntimePlan differs')
    for mode in ('compiler','managed'):
        result['modes'][mode]['final_prune']=toolchain.prune(args.out/mode/'compiled')
    result.update(complete=True,acceptance_complete=True,local_compiled=True)
    return result


def compatibility(args):
    from . import s1_acceptance,s1_managed
    from .u2_multi_layout import baseline_package
    from ..models.zoo import get
    from ..planners import agent_tools,prompts
    from ..tests import test_p2_export
    s1_acceptance.BASE='62ecf194'
    s1_acceptance.RESULTS_DIR=s1_managed.RESULTS_DIR=args.results_root
    test_p2_export.FROZEN=args.frozen_dir;test_p2_export.MODELS=args.models_dir
    test_p2_export.R6B=args.results_root/'r6b-refresh-20261021T025529/REPORT.json'
    result=s1_acceptance.compatibility(args)
    result.update(round='U4',prompts=[])
    previous=baseline_package(args.snapshot)
    for name in ('resnet20','resnet20_silu','cnn_plain','mlp'):
        model=get(name,args.models_dir);graph=model.import_graph()
        calibration=json.loads((args.frozen_dir/'models'/name/'calibration.json').read_text())
        before=previous('planners.agent_tools').build_context(graph,calibration,n_val=20,normalization=model.normalization)
        after=agent_tools.build_context(graph,calibration,n_val=20,normalization=model.normalization)
        for version in ('r8-v4','r8-v5','r8-v6'):
            before.prompt_version=after.prompt_version=version
            if (previous('planners.prompts').build_system_prompt(before)!=prompts.build_system_prompt(after) or
                previous('planners.prompts').initial_message(before)!=prompts.initial_message(after)):
                raise AssertionError(name+'/'+version+': frozen prompt changed')
            result['prompts'].append(dict(model=name,version=version,system_equal=True,input_equal=True))
        del model,graph,before,after
        gc.collect()
        write(args.out/'REPORT.json',result)
    return result


def regression(args):
    if args.group=='intake':
        python=PLATFORM_DIR/'venvs/intake-2.0.1-cpu/bin/python'
        if Path(sys.executable).absolute()!=python.absolute():
            os.execv(str(python),[str(python),'-B','-m','pt_agent.experiments.u4_acceptance',*sys.argv[1:]])
    files=sorted(Path('pt_agent/tests').glob('test_*.py'))
    groups=dict(packaging={'test_p2_export.py','test_toolchain_decoupled.py'},
                tensor={'test_r8f_tensor_validate.py'},weight={'test_u3b2_weight_usage.py'},intake={'test_u1_onnx.py'})
    if args.group=='main':
        excluded=groups['packaging']|groups['tensor']|groups['weight']|{'test_r1_parser.py'}
        files=[p for p in files if p.name not in excluded and not p.name.startswith(('test_s1_','test_u5_','test_u4_'))]
    elif args.group=='s1':
        from ..tests import test_s1_stages
        test_s1_stages.OUTPUT_DIR=args.out/'native-shared-fanout'
        files=[p for p in files if p.name.startswith('test_s1_')]
    elif args.group=='u5':files=[p for p in files if p.name.startswith('test_u5_')]
    else:files=[p for p in files if p.name in groups[args.group]]
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName('pt_agent.tests.'+p.stem) for p in files)
    expected=dict(main=429,packaging=17,tensor=17,weight=17,intake=4,s1=34,u5=25)
    if suite.countTestCases()!=expected[args.group]:
        raise AssertionError(f'Regression inventory differs: {suite.countTestCases()} vs {expected[args.group]}')
    result=unittest.TextTestRunner(verbosity=2,failfast=True).run(suite)
    return dict(round='U4',part='regression',group=args.group,complete=result.wasSuccessful(),
                tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
                reused_parser_and_u4_tests=str(args.unit_report),llm_calls=0,report_images=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--part',choices=('native','compat','regression'),required=True)
    parser.add_argument('--model',choices=('bert','qwen'),default='bert')
    parser.add_argument('--group',choices=('main','packaging','tensor','weight','intake','s1','u5'))
    parser.add_argument('--inside',action='store_true')
    for field in ('out','prepared','program','evaluation','snapshot','results-root','models-dir','frozen-dir','unit-report'):
        parser.add_argument('--'+field,type=Path)
    args=parser.parse_args()
    if args.inside:
        resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,6*1024**3))
        resource.setrlimit(resource.RLIMIT_FSIZE,(8*1024**3,8*1024**3))
        resource.setrlimit(resource.RLIMIT_CORE,(0,0))
        result=dict(round='U4',part=args.part,complete=False,llm_calls=0,report_images=0)
        try:result={'native':native,'compat':compatibility,'regression':regression}[args.part](args)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            path=args.out/'REPORT.json'
            if path.exists():result.update(json.loads(path.read_text()))
            result.update(complete=False,error=f'{type(exc).__name__}: {exc}')
            if args.part=='native':
                from .. import toolchain
                result['pruned_after_stop']={str(folder.relative_to(args.out)):toolchain.prune(folder)
                    for folder in [args.out/mode/part for mode in ('compiler','managed')
                                   for part in ('trace','compiled')] if folder.is_dir()}
        write(args.out/'REPORT.json',result)
        return 0 if result['complete'] else 1
    stamp=datetime.now(timezone.utc)
    run_id='u4-'+args.model+'-'+(args.group or args.part)+'-'+stamp.strftime('%Y%m%dT%H%M%S%f')
    args.out=args.out or RESULTS_DIR/run_id;args.out.mkdir(parents=True,exist_ok=False)
    args.results_root=args.results_root or RESULTS_DIR
    args.models_dir=args.models_dir or args.results_root/'models-v2'
    args.frozen_dir=args.frozen_dir or args.results_root/'r8e7-reconfirm-20261022T122558'
    command=[sys.executable,'-B','-m','pt_agent.tools.nixrun','--seconds','3600','--',
             '-m','pt_agent.experiments.u4_acceptance','--inside','--part',args.part,'--model',args.model,'--out',str(args.out)]
    for field in ('prepared','program','evaluation','snapshot','results_root','models_dir','frozen_dir','unit_report','group'):
        value=getattr(args,field)
        if value is not None:command.extend(['--'+field.replace('_','-'),str(value)])
    write(args.out/'argv.json',command);print('RUN='+str(args.out),flush=True)
    started=time.monotonic()
    with (args.out/'run.log').open('w') as log:
        status=subprocess.run(command,cwd=REPO/'DSL/DSL_agent/scripts',stdout=log,stderr=subprocess.STDOUT)
    path=args.out/'REPORT.json'
    result=json.loads(path.read_text()) if path.exists() else dict(complete=False,error='see run.log')
    rev=subprocess.check_output(['git','rev-parse','--short','HEAD'],cwd=REPO,text=True).strip()
    result.update(run_id=run_id,git_rev=rev,exit=status.returncode,seconds=time.monotonic()-started)
    write(path,result)
    from ..registry import append
    os.environ.setdefault('PT_RESULTS',str(RESULTS_DIR))
    append(dict(run_id=run_id,round='U4',model=args.model,mode=args.part+('/'+args.group if args.group else ''),git_rev=rev,
        metrics=dict(complete=result['complete'],tests=result.get('tests'),llm_calls=0),artifacts_dir=str(args.out),ts=stamp.isoformat()))
    print(json.dumps(dict(report=str(path),complete=result['complete'],exit=status.returncode),indent=2),flush=True)
    return status.returncode


if __name__=='__main__':raise SystemExit(main())
