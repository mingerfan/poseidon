"""Actual Hecate bootstrap dataflow tracing, never encrypted execution.

Inspect whether each requested bootstrap is on the returned SSA dependency
path. Merely counting dead bootstrap operations is not a passing result.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import sys
import tempfile

from hecate_python_env import ROOT, WORK, VENV, enter_nix, digest
from python_compiler_smoke import BUILD, logged
from seal_artifact_gate import require
from seal_cpu_golden import PROFILE

SOURCE = ROOT/'third_party/dacapo/python/hecate/hecate/expr.py'
SCRIPT = Path(__file__).resolve()
STYLES = ('scalar','list','tuple','array','matrix','zero_dim')


def trace(style, output):
    import numpy as np
    spec=importlib.util.spec_from_file_location('bootstrap_frontend_probe',SOURCE)
    hc=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=hc
    spec.loader.exec_module(hc)
    @hc.func('c,c')
    def golden(x,y):
        if style=='scalar':
            return hc.bootstrap(x)+y
        if style=='zero_dim':
            value=np.empty((),dtype=object);value[()]=x
            return hc.bootstrap(value)[()]+y
        if style=='list': value=[x,y]
        elif style=='tuple': value=(x,y)
        else:
            value=np.empty((1,2) if style=='matrix' else (2,),dtype=object)
            value.flat[:]=[x,y]
        result=hc.bootstrap(value)
        if style=='matrix': return result[0,0]+result[0,1]
        return result[0]+result[1]
    hc.save(str(output),str(output))


def dataflow(source):
    definitions={};boots=set();returned=[]
    for line in source.splitlines():
        if '=' in line and re.match(r'\s*%[\w]+\s*=',line):
            lhs,rhs=line.split('=',1)
            result=re.findall(r'%\w+',lhs)[0]
            definitions[result]=re.findall(r'%\w+',rhs)
            if re.search(r'\bearth\.bootstrap\b',rhs): boots.add(result)
        if re.match(r'\s*"?(?:func\.)?return\b',line):
            returned.extend(re.findall(r'%\w+',line))
    require(returned,'No recognizable function return')
    seen=set();pending=list(returned)
    while pending:
        value=pending.pop()
        if value in seen:continue
        seen.add(value);pending.extend(definitions.get(value,()))
    return dict(bootstrap_results=sorted(boots),returned=returned,
                live_bootstrap_results=sorted(boots & seen),
                dead_bootstrap_results=sorted(boots-seen))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inside',action='store_true')
    parser.add_argument('--trace',choices=STYLES)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    require(Path.cwd().resolve()==ROOT,'Wrong source workspace')
    if args.trace:
        require(args.output and args.output.is_dir() and
                args.output.resolve().is_relative_to(WORK/'results'),'Invalid trace directory')
        require(Path(sys.prefix)==VENV and os.environ.get('IN_NIX_SHELL'),'Pinned environment required')
        trace(args.trace,args.output)
        return 0
    require(args.output is None,'Output is trace-only')
    if not args.inside:
        command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" PYTHONDONTWRITEBYTECODE=1 '+shlex.join(
            [str(VENV/'bin/python'),str(SCRIPT),'--inside'])
        return enter_nix(command,seconds=400)
    require(Path(sys.prefix)==VENV,'Pinned Python required')
    import resource
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    compatibility=WORK/'build-dacapo/hecate-python-root'
    require((compatibility/'build').resolve()==BUILD,'Frontend compatibility link missing')
    env=dict(os.environ,HECATE=str(compatibility),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2')
    out=Path(tempfile.mkdtemp(prefix='bootstrap-frontend-',dir=WORK/'results'))
    report=dict(schema=1,status='running',cases=[],agent_calls=0,encrypted_execution=False,
                bootstrap_execution_validated=False,frontend_sha256=digest(SOURCE),
                script_sha256=digest(SCRIPT),compiler_sha256=digest(BUILD/'bin/hecate-opt'),
                scope='actual_Hecate_Earth_IR_dataflow_not_bootstrap_backend')
    for style in STYLES:
        directory=out/style;directory.mkdir()
        cmd=[str(VENV/'bin/python'),str(SCRIPT),'--trace',style,'--output',str(directory)]
        code=logged(cmd,directory/'trace.log',seconds=35,env=env)
        row=dict(style=style,trace_exit=code,passed=False)
        if code==0:
            earth=directory/'probe_bootstrap_frontend.mlir'
            cmd=[str(BUILD/'bin/hecate-opt'),str(earth),'--ckks-config='+str(PROFILE),
                 '--verify-each','--mlir-disable-threading',
                 '-o',str(directory/'verified.mlir')]
            checked=logged(cmd,directory/'verify.log',seconds=30,env=env)
            info=dataflow(earth.read_text())
            expected=1 if style in ('scalar','zero_dim') else 2
            row.update(verify_exit=checked,dataflow=info,expected_live_bootstraps=expected,
                       earth_sha256=digest(earth),verify_command=cmd,
                       passed=checked==0 and len(info['live_bootstrap_results'])==expected)
        report['cases'].append(row)
        print(style,'PASS' if row['passed'] else 'FAIL',flush=True)
    report['status']='passed' if all(c['passed'] for c in report['cases']) else 'failed'
    require(digest(SOURCE)==report['frontend_sha256'],'Frontend changed during probe')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Bootstrap frontend evidence:',out,flush=True)
    return 0 if report['status']=='passed' else 1


if __name__=='__main__':raise SystemExit(main())
