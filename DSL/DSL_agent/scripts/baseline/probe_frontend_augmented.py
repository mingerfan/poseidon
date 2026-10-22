"""Trusted frontend/compiler equivalence probe; no Agent, API, keys or FHE run.

Compare actual upstream +=/-=/*= against ordinary arithmetic through real
Hecate tracing and the existing Dacapo compiler. A recording ABI unit test is
separate. No candidate Python is executed by this fixed test harness.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shlex
import sys
import tempfile

from hecate_python_env import ROOT,WORK,VENV,enter_nix,digest
from python_compiler_smoke import BUILD,logged
from seal_cpu_golden import PROFILE,WATERLINE,dump
from seal_artifact_gate import inspect_artifacts,require

SOURCE=ROOT/'third_party/dacapo/python/hecate/hecate/expr.py'
SCRIPT=Path(__file__).resolve()


def trace(operation,style,output):
    spec=importlib.util.spec_from_file_location('hecate_operator_probe',SOURCE)
    hc=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=hc
    spec.loader.exec_module(hc)  # Trusted, pinned dependency, not generated code.
    if operation=='subtract':
        @hc.func('c')
        def golden(x):
            if style=='augmented':
                x-=.25
                return x
            return x-.25
    elif operation=='add':
        @hc.func('c')
        def golden(x):
            if style=='augmented':
                x+=.25
                return x
            return x+.25
    else:
        @hc.func('c')
        def golden(x):
            if style=='augmented':
                x*=.25
                return x
            return x*.25
    hc.save(str(output),str(output))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inside',action='store_true')
    parser.add_argument('--trace',choices=('add','subtract','multiply'))
    parser.add_argument('--style',choices=('direct','augmented'))
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    require(Path.cwd().resolve()==ROOT,'Run from source root')
    if args.trace:
        require(args.style and args.output and args.output.is_dir() and
                args.output.resolve().is_relative_to(WORK/'results'),'Invalid trace output')
        require(Path(sys.prefix)==VENV and os.environ.get('IN_NIX_SHELL'),'Pinned environment required')
        trace(args.trace,args.style,args.output)
        return 0
    require(not args.style and not args.output,'Trace-only arguments')
    if not args.inside:
        command='LD_LIBRARY_PATH='+chr(36)+'HECATE_PYTHON_LIBRARY_PATH PYTHONDONTWRITEBYTECODE=1 '+shlex.join(
            [str(VENV/'bin/python'),str(SCRIPT),'--inside'])
        return enter_nix(command,seconds=300)
    require(Path(sys.prefix)==VENV,'Pinned Python required')
    compatibility=WORK/'build-dacapo/hecate-python-root'
    require((compatibility/'build').resolve()==BUILD,'Existing frontend library link required')
    env=dict(os.environ,HECATE=str(compatibility),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2')
    out=Path(tempfile.mkdtemp(prefix='frontend-augmented-',dir=WORK/'results'))
    report=dict(status='running',cases=[],agent_calls=0,encrypted_execution=False,
                frontend_source_sha256=digest(SOURCE),probe_sha256=digest(SCRIPT),
                compiler_sha256=digest(BUILD/'bin/hecate-opt'),profile_sha256=digest(PROFILE),
                scope='trusted_actual_frontend_and_compiler_equivalence_not_Agent_or_FHE_execution')
    print('Frontend augmented evidence: '+str(out),flush=True)
    for operation in ('add','subtract','multiply'):
        pair={}
        for style in ('direct','augmented'):
            folder=out/(operation+'-'+style);folder.mkdir()
            cmd=[str(VENV/'bin/python'),str(SCRIPT),'--trace',operation,'--style',style,'--output',str(folder)]
            traced=logged(cmd,folder/'trace.log',seconds=45,env=env)
            entry=dict(trace_command=cmd,trace_exit=traced)
            pair[style]=entry
            if traced!=0: continue
            cmd=[str(BUILD/'bin/hecate-opt'),str(folder/'probe_frontend_augmented.mlir'),'--eva',
                 '--ckks-config='+str(PROFILE),'--waterline='+str(WATERLINE),'--enable-debug-printer',
                 '--mlir-disable-threading','--verify-each','-o',str(folder/'lowered.mlir')]
            compiled=logged(cmd,folder/'compile.log',seconds=60,env=env)
            entry.update(compile_command=cmd,compile_exit=compiled)
            if compiled!=0: continue
            hevm=folder/'lowered._hecate_golden.hevm';cst=folder/'_hecate_golden.cst'
            entry.update(hevm_sha256=digest(hevm),cst_sha256=digest(cst),
                         artifact=inspect_artifacts(hevm.read_bytes(),cst.read_bytes()),
                         earth_sha256=digest(folder/'probe_frontend_augmented.mlir'))
        same=all(key in pair['direct'] and pair['direct'].get(key)==pair['augmented'].get(key)
                 for key in ('hevm_sha256','cst_sha256'))
        report['cases'].append(dict(operation=operation,byte_identical_artifacts=same,**pair))
        dump(out/'report.json',report)
    report['status']='passed' if all(c['byte_identical_artifacts'] for c in report['cases']) else 'failed'
    require(digest(SOURCE)==report['frontend_source_sha256'],'Frontend changed during probe')
    dump(out/'report.json',report)
    print(json.dumps(dict(status=report['status'],cases=[dict(operation=c['operation'],
                     identical=c['byte_identical_artifacts']) for c in report['cases']]),indent=2))
    return 0 if report['status']=='passed' else 1


if __name__=='__main__':
    raise SystemExit(main())
