"""Finite CLI for independently frozen compiler evaluation; never Agent inference."""
import argparse
import json
import os
from pathlib import Path
import resource
import shlex
import signal
import sys
import time

from delivery_component import bound,canonical,checked_bound,read_bytes,require,sha,write_once


def limits():
    for name,pair in [('CPU',(150,155)),('AS',(4*1024**3,)*2),('FSIZE',(16*1024**2,)*2),('NOFILE',(128,128)),('CORE',(0,0))]:
        resource.setrlimit(getattr(resource,'RLIMIT_'+name),pair)


def guard(co,reserve):
    """Reuse the approved resource64 controller; no replacement accounting."""
    import importlib.util, shutil, types
    from hecate_python_env import WORK
    root=WORK/'results/agent-evidence64-peer-proof-revision-r2812'
    expected={'migration_core.py':'21efe3dd1fcb5c8a6c06691cc5765f43441f5fece60a53e99eecb7b6a35c1f1f','controlled_apply.py':'5d5e179e5688d408711e79be310ae4a3a27b2b3ab17ef3c64903abe82ec6a5d1','manifest.json':'d55550d24d41277c5be311a02c76d306bf39bdd59f8644ae167add46fd3e694d'}
    require(all(sha(read_bytes(root/name))==digest for name,digest in expected.items()),'Original approved resource64 controller bytes')
    modules=[]
    for name in ['migration_core','controlled_apply']:
        spec=importlib.util.spec_from_file_location('compiler_eval64_'+name,root/(name+'.py'))
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);modules.append(module)
    proposal=checked_bound(json.loads(read_bytes(root/'manifest.json')))
    state,usage=co.snapshot()
    from compiler_managed_evaluation import RESOURCE_INVENTORY_ROOT
    acceptance=json.loads((RESOURCE_INVENTORY_ROOT/'HOST_IDENTITY_ACCEPTANCE.json').read_text())
    equivalence=json.loads((RESOURCE_INVENTORY_ROOT/'SOURCE_EQUIVALENCE_AND_SCOPE.json').read_text())
    require(acceptance['original_resource_cap_bytes']==64*1024**3 and
            equivalence['inventory_floor_not_decreased'] is True and
            equivalence['old_inventory_function_AST_equal_after_added_host_identity_call'] is True,
            'Published current-host intake preserves the original collector and resource scope')
    spec=importlib.util.spec_from_file_location('compiler_eval64_current_host',RESOURCE_INVENTORY_ROOT/'inventory_current_boot.py')
    inventory=importlib.util.module_from_spec(spec);spec.loader.exec_module(inventory)
    occupied=inventory.inventory(types.SimpleNamespace(inventory=modules[1]._inventory_original),
        modules[0],{k:v for k,v in state.items()if k!='binding'},proposal['extra_evidence_roots'])
    require(occupied+usage['remaining_reserved_bytes']+reserve+4*1024**3+256*1024**2+1024**2<=64*1024**3,
            'Original resource64 normal admission')
    require(shutil.disk_usage(WORK).free>=4*1024**3+usage['remaining_reserved_bytes']+reserve+
            4*1024**3+256*1024**2+1024**2,'Original physical reserve and normal protection')


def main(argv=None):
    limits()
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='action',required=True)
    freeze=sub.add_parser('freeze')
    for key in ['selector','canary-admission','help-evidence','help-text','output']:
        freeze.add_argument('--'+key,required=True)
    saved=sub.add_parser('read-complete',help='Read saved complete complex outputs; no compiler or native call')
    for key in ['profile','input-root','request','output']:saved.add_argument('--'+key,required=True)
    for name in ['diagnose','assess']:
        p=sub.add_parser(name)
        for key in ['profile','profile-sha256','output']:p.add_argument('--'+key,required=True)
        if name=='diagnose':
            p.add_argument('--request',required=True);p.add_argument('--source',required=True)
            p.add_argument('--level',choices=['compiled','numerical'],default='numerical')
        else:
            for key in ['delivery','binding','plan','mathematical-evidence','mathematical-evidence-sha256']:
                p.add_argument('--'+key,required=True)
        p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
        p.add_argument('--deadline',type=int,help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    if args.action=='read-complete':
        from ordinary_complete_output import read,validate_profile
        from compiler_managed_evaluation import read_complete_outputs
        from hecate_python_env import WORK
        from shared_execution import Coordinator,KINDS
        profile=validate_profile(read(args.profile));results=WORK/'results';root=Path(args.output).absolute()
        require(root.parent.resolve()==root.parent and root!=results and root.is_relative_to(results),
                'Fresh accounted saved-output reader directory')
        co=Coordinator(WORK.parents[1]/'coordination/v1',[results]);guard(co,1024**2)
        root.mkdir(mode=0o700)
        batch,_=co.register('complete-output-read-'+str(time.time_ns()),dict(branch='lhy-generalization',
            source_binding=profile['binding'],reserve_bytes=1024**2,wall_seconds=180,stages={k:0 for k in KINDS}),root)
        status='failed'
        previous=signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM,lambda *unused:(_ for _ in ()).throw(TimeoutError('Saved-output reader180-second wall')))
        signal.alarm(175)
        try:
            profile_bytes=canonical(profile);write_once(root/'profile.json',profile_bytes)
            result=read_complete_outputs(profile,args.input_root,read(args.request))
            result_bytes=canonical(result)
            require(len(result_bytes)+len(profile_bytes)+4096<=1024**2,'Saved-output reader1MiB evidence allocation')
            write_once(root/'REPORT.json',result_bytes);batch.checkpoint();status='succeeded'
            print(json.dumps(dict(report=str(root/'REPORT.json'),input_groups=result['input_groups'],
                physical_complex_cells=result['physical_complex_cells'],native_or_first_verified=False)))
            return 0
        except BaseException as error:
            write_once(root/'FAILURE.json',canonical(dict(type=type(error).__name__,diagnostic=str(error)[:1600],native_or_first_verified=False)))
            raise
        finally:
            signal.alarm(0);signal.signal(signal.SIGALRM,previous);batch.finish(status)
    from compiler_managed_evaluation import descriptor,freeze_profile,verify_profile,backend,classify_dependency,assess_frozen
    from hecate_python_env import WORK,VENV,enter_nix
    from shared_execution import Coordinator,KINDS,retained_bytes
    root=Path(args.output).absolute();results=WORK/'results'
    require(root.parent.resolve()==root.parent and root!=results and root.is_relative_to(results),
            'Fresh accounted results directory')
    co=Coordinator(WORK.parents[1]/'coordination/v1',[results])
    if args.action=='freeze':
        guard(co,1024**2);root.mkdir(mode=0o700)
        admission=checked_bound(json.loads(read_bytes(args.canary_admission)))
        profile=freeze_profile(descriptor(args.selector),admission['normal_CPU_canary'],
                               descriptor(args.help_evidence),descriptor(args.help_text))
        batch,_=co.register('compiler-profile-'+profile['binding'][:20],dict(branch='lhy-generalization',
            source_binding=profile['binding'],reserve_bytes=1024**2,wall_seconds=180,stages={k:0 for k in KINDS}),root)
        status='failed'
        try:write_once(root/'profile.json',canonical(profile));status='succeeded'
        finally:batch.finish(status)
        print(json.dumps(dict(profile=str(root/'profile.json'),binding=profile['binding'],numerically_certified=False)));return 0
    require(sha(read_bytes(args.profile))==args.profile_sha256,'Exact independently frozen profile bytes')
    profile=verify_profile(checked_bound(json.loads(read_bytes(args.profile))))
    require(os.environ.get('POSEIDON_SDK_MANIFEST')==profile['selector']['path'],'Select frozen SDK before imports')
    if not args.worker:
        guard(co,16*1024**2);root.mkdir(mode=0o700);deadline=time.monotonic_ns()+180*10**9
        write_once(root/'NIX_LAUNCH_ONCE.json',canonical(bound(dict(profile_binding=profile['binding'],
            Nix_attempts_consumed=1,maximum=1,retries=0,deadline_monotonic_ns=deadline))))
        original=list(sys.argv[1:]if argv is None else argv)
        worker=[*original,'--worker','--deadline',str(deadline)]
        entry=Path(__file__).resolve().parents[1]/'compiler_evaluation.py'
        command='env LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" '+shlex.join([str(VENV/'bin/python'),'-B',str(entry),*worker])
        code=enter_nix(command,seconds=180)
        write_once(root/'NIX_TERMINAL.json',canonical(bound(dict(returncode=code,Nix_attempts_consumed=1))))
        return code
    require(Path(sys.prefix)==VENV and os.environ.get('IN_NIX_SHELL')=='pure','Original pinned numerical environment')
    remaining=(args.deadline-time.monotonic_ns())/10**9
    require(0<remaining<=180,'Frozen overall deadline')
    signal.signal(signal.SIGALRM,lambda *unused:(_ for _ in ()).throw(TimeoutError('External evaluation deadline')))
    signal.setitimer(signal.ITIMER_REAL,remaining)
    guard(co,16*1024**2);run=root/'run';run.mkdir(mode=0o700)
    stages={k:0 for k in KINDS};stages['qualify']=1
    batch,_=co.register('compiler-evaluation-'+sha(str(root).encode())[:20],dict(branch='lhy-generalization',
        source_binding=profile['binding'],reserve_bytes=16*1024**2,wall_seconds=max(1,int(remaining)),stages=stages),run)
    status='failed';artifacts=[]
    try:
        write_once(run/'profile.json',canonical(profile))
        if args.action=='assess':
            result=assess_frozen(args.delivery,args.binding,profile,checked_bound(json.loads(read_bytes(args.plan))),run/'assessment',
                batch=batch,checkpoint=batch.checkpoint,mathematical_evidence=dict(path=str(Path(args.mathematical_evidence).absolute()),sha256=args.mathematical_evidence_sha256))
        else:
            from public_weight_context import read_request
            from candidate_contract import request_file_bytes
            from public_basis_storage import selected
            request=read_request(read_bytes(args.request).decode('utf-8'));source=read_bytes(args.source)
            write_once(run/'candidate.py',source);write_once(run/'request.json',request_file_bytes(request) if selected(request) else canonical(request))
            write_once(run/'before-call.json',canonical(bound(dict(source_sha256=sha(source),request_id=request['request_id'],
                compiler_profile_binding=profile['binding'],qualifier_calls_maximum=1,repairs=0,Agent_generation=False))))
            def register(path):batch.register_artifact(path);artifacts.append(Path(path))
            service=backend(profile)
            with batch.stage('qualify'):
                result=service.qualify(request,dict(schema=1,request_id=request['request_id'],hecate_source=source.decode()),
                    args.level,artifact_observer=register)
            require(read_bytes(run/'candidate.py')==source,'Original source changed')
            folder=Path(result['evidence']);report=json.loads(read_bytes(folder/'report.json'))
            diagnosis=classify_dependency(report,folder,profile)
            result=dict(result,mode='external_development_diagnostic',source_sha256=sha(source),
                dependency_classification=diagnosis,first_original_success=False,delivery_success=False,
                independent_mathematical_acceptance='not_executed_by_diagnostic')
        write_once(run/'REPORT.json',canonical(bound(result)))
        require(retained_bytes([root,*artifacts])<=16*1024**2,'Finite external method evidence')
        status='succeeded';print(json.dumps(dict(report=str(run/'REPORT.json'),status=result['status'],first_success=result.get('first_joint_passed',False))))
        return 0 if result['status']=='passed'else 1
    finally:batch.finish(status)
