"""U4 §7.9.3: only the three largest already-passed managed members."""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from ..paths import REPO, RESULTS_DIR
from .u4_managed_payloads import write


def prepare(args):
    source=json.loads(args.manifest.read_text())
    baseline=json.loads(args.baseline.read_text())
    if not baseline['complete'] or baseline['local_passed']!=19 or baseline['server_pending']!=1:
        raise ValueError('Expected the completed 19 local members and retained one server-pending member')
    originals={row['label']:row for row in source['cases']}
    ordered=[]
    for index,item in enumerate(baseline['cases']):
        report=json.loads(Path(item['report']).read_text())
        if not report['complete']:raise ValueError('Incomplete local baseline '+item['case'])
        boots=report['plan']['candidate']['boots']
        row=deepcopy(originals[item['case']])
        row['baseline']['prior_s1_report']=row['baseline']['report']
        row['baseline']['report']=item['report']
        row['previous_boots']=boots
        ordered.append((boots,index,row))
    if len(ordered)!=19:raise ValueError('Local inventory differs from nineteen')
    chosen=[r for _,_,r in sorted(ordered,key=lambda item:(-item[0],item[1]))[:3]]
    result=dict(round='U4',section='7.9.3.2',source_manifest=str(args.manifest),
        previous_report=str(args.baseline),selection='three highest bootstrap counts among the nineteen local members',
        cases=chosen,denominator=3,retained_other_local=16,retained_server_pending=1,
        calls_per_case=dict(reference_trace=1,reference_compile=1,managed_compiler_tail=2,
                            preprocess=1,structure_print=1),llm_calls=0)
    write(args.out/'MANIFEST.json',result)
    return result


def inside(args):
    if args.resume is not None:
        return resume_inside(args)
    import importlib
    from unittest.mock import patch
    from ..manage.greedy import timing_log, timed_stage
    from ..tests._u4_legacy_manage import legacy_manage, compare_functions
    from .u4_managed_payloads import inside as shared_inside
    manager=importlib.import_module('pt_agent.manage')
    original=manager.manage
    # §7.9.5: retain the stopped case for the read-only disk inventory.
    args.preserve_stop=True
    def audited(recorded,profile,*,name='golden',stage_callback=None):
        with timed_stage('legacy_reference',nodes=len(recorded.nodes)) as timing:
            old=legacy_manage(recorded,profile,name=name)
            timing.update(output_nodes=len(old.nodes),boots=sum(n.kind=='bootstrap' for n in old.nodes))
        new=original(recorded,profile,name=name,stage_callback=stage_callback)
        comparison=compare_functions(old,new)
        comparison.update(legacy='tests/_u4_legacy_manage.py',scope='all Function and Node fields')
        write(args.out/'FUNCTION_COMPARISON.json',comparison)
        if not comparison['equal']:raise RuntimeError('Legacy/current complete Function differs')
        return new
    with timing_log(args.out/'trace-stages.json'),patch.object(manager,'manage',audited):
        status=shared_inside(args)
    result=json.loads((args.out/'REPORT.json').read_text())
    result.update(section='7.9.3.2',previous_shared_payload_baseline=True,
                  function_comparison=str(args.out/'FUNCTION_COMPARISON.json'))
    write(args.out/'REPORT.json',result)
    return status


def resume_inside(args):
    """Resume retained managed IR; never rewrite the earlier stopped report."""
    import gc
    import resource
    import shutil
    import traceback
    from .. import toolchain
    from ..manage import read, check, streams
    from . import s1_managed
    from .s0_pipeline_spike import compare_plans
    from .u4_bert_managed import compare_saved
    resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,6*1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE,(8*1024**3,8*1024**3))
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    previous=json.loads((args.resume/'REPORT.json').read_text())
    row=next(r for r in json.loads(args.manifest.read_text())['cases'] if r['label']==args.case)
    result=dict(round='U4',section='7.9.5',complete=False,case=row,resumed_from=str(args.resume),
        llm_calls=0,report_images=0,reused_function_comparison=str(args.resume/'FUNCTION_COMPARISON.json'),
        managed_compile=[],native_calls=dict(managed_normalize=0,managed_tail=0,reference_trace=0,reference_compile=0))
    def save():write(args.out/'REPORT.json',result)
    try:
        if previous['case']['label']!=row['label'] or previous['compiler']['exit']:
            raise ValueError('Resume requires the same case and a successful retained compiler reference')
        comparison=json.loads((args.resume/'FUNCTION_COMPARISON.json').read_text())
        if not comparison['equal']:raise ValueError('Retained complete Function comparison did not pass')
        old_plan=args.resume/'compiler/lowered._hecate_golden.runtime-plan.json'
        if not (args.resume/'compiler/lowered._hecate_golden.bundle/manifest.json').is_file():
            raise ValueError('The retained reference constant bundle is missing')
        toolchain.setup_env()
        profile=toolchain.management_profile()
        candidate=args.out/'dsl';candidate.mkdir()
        source=streams.resolve(args.resume/'dsl/managed.mlir')
        size=streams.uncompressed_size(source) if source.suffix=='.gz' else source.stat().st_size
        free=streams.admit(candidate,extra_bytes=size if size>streams.TEMP_INPUT_LIMIT else 0)
        result['input']=dict(source=str(source),uncompressed_bytes=size,free_bytes=free,
            route='temporary-file' if size>streams.TEMP_INPUT_LIMIT else 'stdin')
        managed=candidate/source.name
        shutil.copyfile(source,managed)
        env=dict(LD_LIBRARY_PATH=os.environ['HECATE_PYTHON_LIBRARY_PATH'])
        command=[str(toolchain.host_paths()['hecate_opt']),str(managed),
            '--ckks-config='+str(toolchain.BOOT_CONFIG_DIR/'compiler-config.json'),
            '--mlir-disable-threading','--verify-each',
            '--pass-pipeline=builtin.module('+toolchain.EARTH_TAIL_PASSES+')',
            '-o',str(candidate/'normalized.earth.mlir')]
        save()
        result['normalize']=toolchain.trusted_run(candidate,command,candidate/'normalize.log',env_extra=env)
        result['native_calls']['managed_normalize']+=1;save()
        if result['normalize']['exit']:raise RuntimeError('Resumed managed normalization failed')
        shutil.copyfile(toolchain.BOOT_CONFIG_DIR/'compiler-config.json',candidate/'boot-compiler-config.json')
        for index,argv in enumerate(toolchain.managed_compile_argv('dacapo-depth-dp','qwen48',candidate)):
            status=toolchain.trusted_run(candidate,toolchain.to_host_argv(argv,candidate),
                candidate/f'compile-{index+1}.log',env_extra=env)
            result['managed_compile'].append(status);result['native_calls']['managed_tail']+=1;save()
            if status['exit']:raise RuntimeError('Resumed managed conversion/export failed')
            if index==0:
                result.setdefault('pruned_stages',[]).append(toolchain.prune(candidate,
                    keep=('managed.ckks.mlir','normalized.earth.mlir')))
        new_plan=candidate/'lowered._hecate_golden.runtime-plan.json'
        result['plan']=compare_plans(old_plan,new_plan);save()
        if not result['plan']['equal']:raise RuntimeError('Retained compiler/managed full RuntimePlan differs')
        # The previous stop pruned compiler Earth text before emitting managed
        # IR. Recover only that independent reference via the existing protocol.
        reference=args.out/'reference-ir'
        s1_managed.RESULTS_DIR=args.models_root.parent
        result['reference_ir_recovery']=s1_managed.compiler_reference(row,reference)
        result['native_calls'].update(reference_trace=1,reference_compile=1);save()
        recovered=reference/'lowered._hecate_golden.runtime-plan.json'
        result['recovered_reference_comparison']=compare_plans(old_plan,recovered);save()
        if not result['recovered_reference_comparison']['equal']:
            raise RuntimeError('Recovered compiler reference differs from retained reference')
        a=read(reference/'lowered.earth.mlir',profile)
        b=read(candidate/'normalized.earth.mlir',profile)
        check(a);check(b)
        result['ir']=s1_managed.compare_ir(a,b)
        del a,b;gc.collect();save()
        if not result['ir']['equal']:raise RuntimeError('Resumed managed Earth IR differs')
        baseline=json.loads(Path(row['baseline']['report']).read_text())
        result['saved_baseline']=compare_saved(Path(baseline['plan']['new_plan']),new_plan);save()
        if not result['saved_baseline']['equal']:raise RuntimeError('Retained S1 baseline differs')
        result['pruned']={str(p):toolchain.prune(p) for p in
            (reference,candidate,args.resume/'compiler',args.resume/'dsl')}
        result['complete']=True;save()
    except Exception as exc:
        traceback.print_exc()
        result.update(error=f'{type(exc).__name__}: {exc}',artifacts_preserved_after_stop=True)
        save();write(args.out/'STOP.json',result)
    return 0 if result['complete'] else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--inside',action='store_true')
    parser.add_argument('--launch',action='store_true')
    parser.add_argument('--case')
    for field in ('out','manifest','baseline','models-root','resume'):
        parser.add_argument('--'+field,type=Path)
    args=parser.parse_args()
    if args.inside:return inside(args)
    fields=('case','out','manifest','models_root','resume')
    if args.launch:
        from ..tools.nixrun import main as nix_main
        sys.argv=[sys.argv[0],'--seconds','3600','--','-m',__spec__.name,'--inside']
        for name in fields:
            if getattr(args,name) is not None:sys.argv+=['--'+name.replace('_','-'),str(getattr(args,name))]
        try:return nix_main()
        except subprocess.CalledProcessError as exc:
            stderr=exc.stderr or '';(args.out/'nix.stderr.txt').write_text(stderr)
            write(args.out/'STOP.json',dict(complete=False,stage='nix-launch',exit=exc.returncode,stderr=stderr))
            sys.stderr.write(stderr);return exc.returncode
    stamp=datetime.now(timezone.utc)
    prefix='u4-manage-m3-prepare' if args.prepare else 'u4-manage-m3-'+args.case
    run_id=prefix+'-'+stamp.strftime('%Y%m%dT%H%M%S%f')
    args.out=args.out or RESULTS_DIR/run_id;args.out.mkdir(parents=True,exist_ok=False)
    if args.prepare:
        result=prepare(args)
        result.update(complete=True,run_id=run_id)
        write(args.out/'REPORT.json',result)
        print(json.dumps(dict(manifest=str(args.out/'MANIFEST.json'),cases=[(r['label'],r['previous_boots']) for r in result['cases']])),flush=True)
        mode='management-regression-prepared'
        status=0
    else:
        args.models_root=args.models_root or RESULTS_DIR/'models-v2'
        command=[sys.executable,'-B','-m',__spec__.name,'--launch']
        for name in fields:
            if getattr(args,name) is not None:command+=['--'+name.replace('_','-'),str(getattr(args,name))]
        write(args.out/'argv.json',command);print('RUN='+str(args.out),flush=True)
        began=time.monotonic()
        with (args.out/'run.log').open('w') as log:
            status=subprocess.run(command,cwd=REPO/'DSL/DSL_agent/scripts',stdout=log,stderr=subprocess.STDOUT).returncode
        path=args.out/'REPORT.json'
        result=json.loads(path.read_text()) if path.exists() else dict(complete=False,error='see run.log')
        result.update(run_id=run_id,exit=status,seconds=time.monotonic()-began)
        write(path,result);mode='management-regression'
        print(json.dumps(dict(report=str(path),complete=result['complete'],exit=status)),flush=True)
    from ..registry import append
    os.environ.setdefault('PT_RESULTS',str(RESULTS_DIR))
    rev=subprocess.check_output(['git','rev-parse','--short','HEAD'],cwd=REPO,text=True).strip()
    append(dict(run_id=run_id,round='U4',model=args.case or 'selected-three',mode=mode,git_rev=rev,
        metrics=dict(complete=result['complete'],llm_calls=0),artifacts_dir=str(args.out),ts=stamp.isoformat()))
    return status


if __name__=='__main__':raise SystemExit(main())
