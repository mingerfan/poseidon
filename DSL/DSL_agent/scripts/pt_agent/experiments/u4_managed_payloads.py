"""U4 §7.6: bounded S1 regressions through aliases, bytecode and checked structure."""

import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from ..paths import REPO,RESULTS_DIR


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def shared_preprocess(recorded,directory,profile,*,name='golden',compressed=False,legacy_compare=False):
    import shutil
    from .. import toolchain
    from ..manage import emit_raw
    from ..manage.mlir_read import read,graph_backend,RAW_OPS
    from ..manage.streams import compress_ir,resolve
    from .s1_managed import compare_ir
    directory.mkdir()
    metrics = {}
    raw = emit_raw(recorded,directory/'raw.earth.mlir',profile,name=name,metrics=metrics,compressed=compressed)
    constants = {v.name:v.data for v in recorded._constants.values()}
    shutil.copyfile(toolchain.BOOT_CONFIG_DIR/'compiler-config.json',directory/'boot-compiler-config.json')
    report = dict(emit=metrics,commands=[],format='bytecode-and-structure',legacy_comparison=None)

    def run(argv,label):
        command = toolchain.to_host_argv(argv,directory)
        status = toolchain.trusted_run(directory,command,directory/(label+'.log'),
            env_extra=dict(LD_LIBRARY_PATH=os.environ['HECATE_PYTHON_LIBRARY_PATH']))
        report['commands'].append(dict(label=label,argv=command,status=status))
        write(directory/'PREPROCESS.json',report)
        if status['exit']:
            raise RuntimeError(label+' failed; see '+str(directory/(label+'.log')))

    run(toolchain.frontend_preprocess_argv('dacapo-depth-dp','qwen48',directory,bytecode=True),'preprocess-bytecode')
    if legacy_compare:
        run(toolchain.frontend_preprocess_argv('dacapo-depth-dp','qwen48',directory),'preprocess-legacy-text')
    raw.unlink()
    run(toolchain.frontend_structure_argv('dacapo-depth-dp','qwen48',directory),'print-structure')
    structure = compress_ir(directory/'preprocessed.structure.mlir')
    function = read(structure,profile,allowed=RAW_OPS,constants=constants)
    report['bytecode_bytes'] = (directory/'preprocessed.mlirbc').stat().st_size
    report['structure_bytes'] = structure.stat().st_size
    report['constant_ops_checked'] = sum(n.kind=='constant' for n in function.nodes)
    report['distinct_restored_payloads'] = len({n.const_key for n in function.nodes if n.kind=='constant'})
    if legacy_compare:
        old = read(resolve(directory/'preprocessed.earth.mlir'),profile,allowed=RAW_OPS)
        report['legacy_comparison'] = compare_ir(old,function)
        if not report['legacy_comparison']['equal'] or report['legacy_comparison']['conclusion']!='相同':
            write(directory/'PREPROCESS.json',report)
            raise RuntimeError('Legacy text and checked structure differ')
    write(directory/'PREPROCESS.json',report)
    return graph_backend(function,recorded.slots),function,report


def inside(args):
    import resource
    import traceback
    from types import SimpleNamespace
    from unittest.mock import patch
    from . import s1_managed,s1_stages
    from .. import toolchain
    from ..manage import streams
    resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,6*1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE,(8*1024**3,8*1024**3))
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    manifest = json.loads(args.manifest.read_text())
    row = next(r for r in manifest['cases'] if r['label']==args.case)
    old = row['baseline']
    result = dict(round='U4',section='7.6.4',complete=False,case=row,llm_calls=0,report_images=0)
    try:
        attempt_pending=getattr(args,'attempt_pending',False)
        if attempt_pending:
            if old['status']!='server-pending':
                raise ValueError('The explicit pending-case continuation requires a retained server-pending member')
            args.preserve_stop=True
        if getattr(args,'size_only',False):
            from ..models.info import ModelInfo
            from ..dsl.parser import parse
            from ..lower import lower
            from ..manage import RecordingBackend,emit_raw
            info=ModelInfo.load(Path(row['info']))
            program=parse(Path(row['source']).read_text())
            recorded=RecordingBackend(program.header.slots,locations={op.id:op for op in program.ops})
            lower(program,info.graph.weights,recorded)
            metrics={}
            source=emit_raw(recorded,args.out/'raw.earth.mlir',toolchain.management_profile(),
                name='golden',metrics=metrics,compressed=True)
            size=streams.uncompressed_size(source)
            if size>8*1024**3:
                raise ValueError('Shared raw IR exceeds the unchanged 8 GiB temporary-file limit')
            free=streams.admit(args.out,extra_bytes=size if size>streams.TEMP_INPUT_LIMIT else 0)
            result.update(complete=True,size_only=True,local_complete=False,
                source=str(source),uncompressed_bytes=size,free_bytes=free,
                route='temporary-file' if size>streams.TEMP_INPUT_LIMIT else 'stdin',
                emit=metrics,native_calls=0)
        elif old['status']=='server-pending' and not attempt_pending:
            result.update(complete=True,local_complete=False,status='preserved-server-pending',
                          reason=old['reason'],server_job=old['server_job'],native_calls=0)
        else:
            # Reuse the exact S1 comparison protocol, including payload arrays,
            # to reconstruct its already-pruned native reference in a new dir.
            # All roots and selected cases arrive from the system-Python parent.
            legacy = row.get('unit') in ('chain','residual','constants')
            def preprocess(*values,**kwargs):
                return shared_preprocess(*values,**kwargs,legacy_compare=legacy)
            original_compress = streams.compress_ir
            def compress_after_reference_cleanup(path,**kwargs):
                path = Path(path)
                if (path.name=='candidate_trace.mlir' and path.is_file() and
                        path.parent.resolve()==(args.out/'compiler').resolve()):
                    import shutil
                    constant_file = path.parent/'constants.npz'
                    cleanup = dict(approval='user 2026-10-27: prune consumed constants before compression',
                        trace_bytes=path.stat().st_size,
                        constants_bytes=constant_file.stat().st_size if constant_file.exists() else 0,
                        free_before_bytes=shutil.disk_usage(path.parent).free)
                    # trace() calls compress_ir only after successful native
                    # completion. Preserve every file except its consumed NPZ.
                    keep = tuple(p.name for p in path.parent.iterdir() if p.name!='constants.npz')
                    cleanup['pruned'] = toolchain.prune(path.parent,keep=keep)
                    cleanup['free_after_bytes'] = shutil.disk_usage(path.parent).free
                    write(path.parent/'TRACE_CLEANUP.json',cleanup)
                return original_compress(path,**kwargs)
            options = SimpleNamespace(out=args.out,case=args.case,reference=None,proof_text=False,compressed=True)
            with patch.object(s1_managed,'inventory',lambda:[row]), \
                 patch.object(s1_managed,'RESULTS_DIR',args.models_root.parent), \
                 patch.object(s1_stages,'preprocess',preprocess), \
                 patch.object(streams,'compress_ir',compress_after_reference_cleanup):
                result = s1_managed.case_inside(options)
            result.update(round='U4',section='7.6.4',baseline=old,local_complete=True,status='passed')
            if attempt_pending:
                if not result['placement_cuts_equal'] or not result['placement_boots_equal']:
                    raise RuntimeError('Pending member depth-dp cuts/bootstrap count differs')
                result.update(resolved_server_pending=True,saved_baseline=dict(available=False,
                    reason='Prior member stopped before producing a RuntimePlan; current compiler/managed pair compared in full',
                    previous_report=old['report'],previous_server_job=old['server_job']))
            else:
                from .s0_pipeline_spike import step_states,stats
                before = json.loads(Path(old['report']).read_text())
                old_plan = Path(before['plan']['new_plan'])
                new_plan = Path(result['plan']['new_plan'])
                a,b = json.loads(old_plan.read_text()),json.loads(new_plan.read_text())
                result['saved_baseline'] = dict(plan=str(old_plan),
                    stats_equal=stats(a)==stats(b),step_states_equal=step_states(a)==step_states(b),
                    payload_descriptors_equal=[s.get('payload') for s in a['initialization']]==
                                              [s.get('payload') for s in b['initialization']])
                if not all(result['saved_baseline'][key] for key in
                           ('stats_equal','step_states_equal','payload_descriptors_equal')):
                    raise RuntimeError('New managed RuntimePlan differs from retained S1 baseline')
        write(args.out/'REPORT.json',result)
    except Exception as exc:
        traceback.print_exc()
        if (args.out/'REPORT.json').exists():
            result.update(json.loads((args.out/'REPORT.json').read_text()))
        result.update(complete=False,round='U4',section='7.6.4',error=f'{type(exc).__name__}: {exc}')
        if getattr(args,'preserve_stop',False):
            result['artifacts_preserved_after_stop'] = True
        else:
            result['pruned_after_stop'] = {d.name:toolchain.prune(d) for d in
                (args.out/'compiler',args.out/'dsl',args.out/'preprocessed') if d.is_dir()}
        write(args.out/'REPORT.json',result)
        write(args.out/'STOP.json',result)
    return 0 if result['complete'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inside',action='store_true')
    parser.add_argument('--launch',action='store_true')
    parser.add_argument('--attempt-pending',action='store_true')
    parser.add_argument('--size-only',action='store_true')
    parser.add_argument('--case',required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--models-root',type=Path,required=True)
    parser.add_argument('--out',type=Path)
    args = parser.parse_args()
    if args.inside:
        return inside(args)
    if args.launch:
        from ..tools.nixrun import main as nix_main
        sys.argv = [sys.argv[0],'--seconds','3600','--','-m',__spec__.name,'--inside',
            '--case',args.case,'--manifest',str(args.manifest),'--models-root',str(args.models_root),'--out',str(args.out)]
        for flag in ('attempt_pending','size_only'):
            if getattr(args,flag):sys.argv.append('--'+flag.replace('_','-'))
        try:
            return nix_main()
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr or ''
            (args.out/'nix.stderr.txt').write_text(stderr)
            write(args.out/'STOP.json',dict(complete=False,stage='nix-launch',exit=exc.returncode,stderr=stderr))
            sys.stderr.write(stderr)
            return exc.returncode
    stamp = datetime.now(timezone.utc)
    run_id = 'u4-shared-'+args.case+'-'+stamp.strftime('%Y%m%dT%H%M%S%f')
    args.out = args.out or RESULTS_DIR/run_id
    args.out.mkdir(parents=True,exist_ok=False)
    command = [sys.executable,'-B','-m',__spec__.name,'--launch','--case',args.case,
        '--manifest',str(args.manifest),'--models-root',str(args.models_root),'--out',str(args.out)]
    for flag in ('attempt_pending','size_only'):
        if getattr(args,flag):command.append('--'+flag.replace('_','-'))
    write(args.out/'argv.json',command)
    print('RUN='+str(args.out),flush=True)
    started = time.monotonic()
    with (args.out/'run.log').open('w') as log:
        status = subprocess.run(command,cwd=REPO/'DSL/DSL_agent/scripts',stdout=log,stderr=subprocess.STDOUT)
    report = args.out/'REPORT.json'
    result = json.loads(report.read_text()) if report.exists() else dict(complete=False,error='see run.log')
    if (args.out/'STOP.json').exists() and not report.exists():
        result.update(json.loads((args.out/'STOP.json').read_text()))
    rev = subprocess.check_output(['git','rev-parse','--short','HEAD'],cwd=REPO,text=True).strip()
    result.update(run_id=run_id,git_rev=rev,exit=status.returncode,seconds=time.monotonic()-started)
    write(report,result)
    os.environ.setdefault('PT_RESULTS',str(RESULTS_DIR))
    from ..registry import append
    append(dict(run_id=run_id,round='U4',model=args.case,
        mode='shared-input-size' if args.size_only else 'managed-shared-regression',git_rev=rev,
        metrics=dict(complete=result['complete'],llm_calls=0),artifacts_dir=str(args.out),ts=stamp.isoformat()))
    print(json.dumps(dict(report=str(report),complete=result['complete'],exit=status.returncode)),flush=True)
    return status.returncode


if __name__=='__main__':
    raise SystemExit(main())
