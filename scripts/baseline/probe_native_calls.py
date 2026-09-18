"""Trusted decorated-function calls versus equivalent direct Hecate programs.

Runs actual frontend/compiler, never Agent code, provider calls or FHE runtime.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import resource
import shlex
import sys
import tempfile

from hecate_python_env import ROOT, WORK, VENV, enter_nix, digest
from python_compiler_smoke import BUILD, logged
from seal_cpu_golden import PROFILE, WATERLINE
from seal_artifact_gate import require, inspect_artifacts

SOURCE=ROOT/'third_party/dacapo/python/hecate/hecate/expr.py'
SCRIPT=Path(__file__).resolve()
CASES=('scalar','pair','nested','forward','two_inputs','public_argument',
       'identity','repeated','array','zero_dim','zero_input','empty_helper','plain_return')


def trace(case,style,output):
    import numpy as np
    spec=importlib.util.spec_from_file_location('native_calls_probe',SOURCE)
    hc=importlib.util.module_from_spec(spec);sys.modules[spec.name]=hc
    spec.loader.exec_module(hc)
    if style=='direct':
        @hc.func('c,c' if case=='two_inputs' else 'c')
        def golden(x,y=None):
            if case=='identity':return x
            if case=='repeated':return (x+.25)+(x+.25)
            if case=='array':return np.array([[x+.25,x],[x+.75,x+.5]],dtype=object)
            if case=='zero_dim':return x+.25
            if case=='empty_helper':return x+.25
            if case in ('zero_input','plain_return'):return x*.5
            if case=='pair':return x+.25,x*.5
            if case=='nested':return (x*.5)*.5+.25
            if case=='two_inputs':return y-x
            if case=='public_argument':return x*.5
            return x*x+.25
    elif case=='identity':
        @hc.func('c')
        def helper(x):return x
        @hc.func('c')
        def golden(x):return helper(x)
    elif case=='repeated':
        @hc.func('c')
        def helper(x):return x+.25
        @hc.func('c')
        def golden(x):return helper(x)+helper(x)
    elif case=='array':
        @hc.func('c')
        def helper(x):return np.array([[x,x+.25],[x+.5,x+.75]],dtype=object)[:,::-1]
        @hc.func('c')
        def golden(x):return helper(x)
    elif case=='zero_dim':
        @hc.func('c')
        def helper(x):return np.array(x+.25,dtype=object)
        @hc.func('c')
        def golden(x):return helper(x).item()
    elif case=='zero_input':
        @hc.func('')
        def helper():return hc.resolveType(.5)
        @hc.func('c')
        def golden(x):return x*helper()
    elif case=='empty_helper':
        @hc.func('c')
        def helper(x):return []
        @hc.func('c')
        def golden(x):
            helper(x)
            return x+.25
    elif case=='plain_return':
        @hc.func('p')
        def helper(x):return x
        @hc.func('c')
        def golden(x):return x*helper(.5)
    elif case=='forward':
        @hc.func('c')
        def golden(x):return helper(x)
        @hc.func('c')
        def helper(x):return x*x+.25
    elif case=='two_inputs':
        @hc.func('c,c')
        def helper(x,y):return x-y
        @hc.func('c,c')
        def golden(x,y):return helper(y,x)
    elif case=='public_argument':
        @hc.func('c,p')
        def helper(x,weight):return x*weight
        @hc.func('c')
        def golden(x):return helper(x,.5)
    elif case=='nested':
        @hc.func('c')
        def first(x):return x*.5
        @hc.func('c')
        def second(x):return first(x)+.25
        @hc.func('c')
        def golden(x):return second(first(x))
    else:
        @hc.func('c')
        def helper(x):return (x+.25,x*.5) if case=='pair' else x*x+.25
        @hc.func('c')
        def golden(x):return helper(x)
    hc.save(str(output),str(output))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inside',action='store_true')
    parser.add_argument('--trace',choices=CASES)
    parser.add_argument('--style',choices=('direct','call'))
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    require(Path.cwd().resolve()==ROOT,'Wrong workspace')
    if args.trace:
        require(args.style and args.output and args.output.is_dir() and
                args.output.resolve().is_relative_to(WORK/'results'),'Invalid trace destination')
        require(Path(sys.prefix)==VENV and os.environ.get('IN_NIX_SHELL'),'Pinned environment required')
        trace(args.trace,args.style,args.output);return 0
    require(not args.style and not args.output,'Trace-only arguments')
    if not args.inside:
        cmd='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" PYTHONDONTWRITEBYTECODE=1 '+shlex.join(
            [str(VENV/'bin/python'),str(SCRIPT),'--inside'])
        return enter_nix(cmd,seconds=900)
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    compatibility=WORK/'build-dacapo/hecate-python-root'
    require((compatibility/'build').resolve()==BUILD,'Missing frontend library link')
    env=dict(os.environ,HECATE=str(compatibility),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2')
    out=Path(tempfile.mkdtemp(prefix='native-function-calls-',dir=WORK/'results'))
    report=dict(schema=1,status='running',cases=[],agent_calls=0,encrypted_execution=False,
                frontend_source_sha256=digest(SOURCE),probe_sha256=digest(SCRIPT),
                frontend_library_sha256=digest(BUILD/'lib/libHecateFrontend.so'),
                compiler_sha256=digest(BUILD/'bin/hecate-opt'),profile_sha256=digest(PROFILE))
    print('Native function call evidence:',out,flush=True)
    for case in CASES:
        pair={}
        for style in ('direct','call'):
            folder=out/(case+'-'+style);folder.mkdir()
            cmd=[str(VENV/'bin/python'),str(SCRIPT),'--trace',case,'--style',style,'--output',str(folder)]
            entry=dict(trace_exit=logged(cmd,folder/'trace.log',seconds=40,env=env));pair[style]=entry
            if entry['trace_exit']!=0:continue
            earth=folder/'probe_native_calls.mlir'
            entry['earth_sha256']=digest(earth)
            cmd=[str(BUILD/'bin/hecate-opt'),str(earth),'--eva','--ckks-config='+str(PROFILE),
                 '--waterline='+str(WATERLINE),'--enable-debug-printer','--mlir-disable-threading',
                 '--verify-each','-o',str(folder/'lowered.mlir')]
            entry.update(compile_exit=logged(cmd,folder/'compile.log',seconds=60,env=env),compile_command=cmd)
            if entry['compile_exit']!=0:continue
            hevm=folder/'lowered._hecate_golden.hevm';cst=folder/'_hecate_golden.cst'
            entry.update(hevm_sha256=digest(hevm),cst_sha256=digest(cst),
                artifact=inspect_artifacts(hevm.read_bytes(),cst.read_bytes(),expected_inputs=2 if case=='two_inputs' else 1))
        identical=all(k in pair['direct'] and pair['direct'].get(k)==pair['call'].get(k)
                      for k in ('hevm_sha256','cst_sha256'))
        report['cases'].append(dict(case=case,identical_artifacts=identical,**pair))
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(case,'BYTE_IDENTICAL' if identical else 'NOT_BYTE_IDENTICAL',flush=True)
    require(digest(SOURCE)==report['frontend_source_sha256'],'Frontend changed during probe')
    report['all_compiled']=all(c[s].get('trace_exit')==c[s].get('compile_exit')==0
        for c in report['cases'] for s in ('direct','call'))
    report['status']=('passed' if all(c['identical_artifacts'] for c in report['cases']) else
                      'compiled_nonidentical' if report['all_compiled'] else 'failed')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    # Different instruction/constant numbering is not a semantic verdict.
    # The separate real CPU experiment compares both forms to a Torch oracle.
    return 0 if report['all_compiled'] else 1


if __name__=='__main__':raise SystemExit(main())
