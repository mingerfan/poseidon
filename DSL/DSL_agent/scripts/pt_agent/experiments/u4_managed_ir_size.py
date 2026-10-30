"""U4 §7.9.5: size alias-based managed text from the retained structural DAG."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import shutil
import struct
import subprocess
import sys
import time

from ..paths import REPO, RESULTS_DIR
from .u4_managed_payloads import write


def text_size(function, slots):
    """Count earth_emit's fixed-width f64 alias syntax without payload values."""
    from ..manage.earth_emit import type_text, float_text
    types=function.types()
    def value(a):return f'%arg{-a-1}' if a<0 else f'%v{a}'
    inputs=', '.join(f'%arg{i}: {type_text(t)} loc(unknown)' for i,t in enumerate(function.inputs))
    results=', '.join(type_text(t) for t in function.result_types)
    a=function.attrs
    attrs=', '.join((f'arg_scale = array<i64: {", ".join(map(str,a["arg_scale"]))}>',
        f'init_level = {a["init_level"]} : i64',f'res_scale = array<i64: {", ".join(map(str,a["res_scale"]))}>')) if a else ''
    suffix=f' attributes {{{attrs}}}' if attrs else ''
    size=len(f'module {{\n  func.func @_hecate_{function.name}({inputs}) -> ({results}){suffix} {{\n')
    size+=len('    return '+', '.join(value(v) for v in function.outputs)+' : '+results+
        ' loc(unknown)\n  } loc(unknown)\n} loc(unknown)\n')
    payloads=set()
    for n in function.nodes:
        operands=', '.join(type_text(types[v]) for v in n.args)
        args=', '.join(value(v) for v in n.args)
        props=attr=''
        if n.kind=='constant':
            if n.const_key is None:raise ValueError('Sizing requires retained constant names')
            payloads.add(n.const_key)
            attr=f' {{rms_var = {float_text(0)} : f64, value = #pt{n.const_key}}}'
        elif n.attrs:
            entries=[f'{key} = array<i64: {", ".join(map(str,v))}>' if key=='offset' else f'{key} = {v} : i64'
                for key,v in n.attrs.items()]
            props=' <{'+', '.join(entries)+'}>'
        loc=f' loc("ptdsl-op-{n.op_id}":{max(1,n.lineno)}:0)'
        if n.kind=='constant':loc=f' loc("ptdsl-const-{n.const_key}"("ptdsl-op-{n.op_id}":{max(1,n.lineno)}:0))'
        size+=len(f'    %v{n.id} = "earth.{n.kind}"({args}){props}{attr} : '
            f'({operands}) -> {type_text(n.type)}{loc}\n')
    size+=sum(len(f'#pt{key} = dense<"0x"> : tensor<{slots}xf64>\n')+slots*16 for key in payloads)
    return size


def inside(args):
    from .. import toolchain
    from ..manage import manage, emit, streams
    from ..manage.greedy import timing_log
    from ..manage.earth_emit import constant_table
    from ..manage.mlir_read import read, graph_backend, RAW_OPS
    from ..tests.test_s1_manage import record_case
    toolchain.setup_env()
    configs=list((args.saved/'managed').glob('*/config/model.json'))
    if len(configs)!=1:raise ValueError('Expected one retained package model configuration')
    config=configs[0].parent
    model=json.loads(configs[0].read_text())
    profile=json.loads((config/'management.json').read_text())['profile']
    controls=[]
    for case in ('chain','residual','constants'):
        recorded,_=record_case(case)
        function=manage(recorded,profile,name='size_control')
        constant_table(function)
        size=text_size(function,recorded.slots)
        directory=args.out/('control-'+case);directory.mkdir()
        metrics={};path=emit(function,directory/'managed.mlir',metrics=metrics,compressed=True)
        actual=streams.uncompressed_size(path)
        controls.append(dict(case=case,estimated_bytes=size,actual_bytes=actual,equal=size==actual))
        if size!=actual:raise RuntimeError('IR size control differs: '+case)
        toolchain.prune(directory)
    source=args.saved/'managed/trace/preprocessed.structure.mlir'
    with timing_log(args.out/'trace-stages.json'):
        function=read(source,profile,allowed=RAW_OPS,structure_only=True)
        recorded=graph_backend(function,model['slots'])
        function=manage(recorded,profile,name=model['name'])
        size=text_size(function,model['slots'])
    combinations={(n.const_key,n.type,struct.pack('>d',n.attrs['rms_var']))
        for n in function.nodes if n.kind=='constant'}
    report=dict(round='U4',section='7.9.5.2',complete=True,model=model['name'],
        controls=controls,saved_structure=str(source),diagnostic_structure_only=True,
        purpose='IR text size only; no executable payloads, compilation or numerical evaluation',
        uncompressed_bytes=size,route='stdin' if size<=streams.TEMP_INPUT_LIMIT else 'temporary-file',
        free_bytes=shutil.disk_usage(args.out).free,
        required_bytes=10*1024**3+(size if size>streams.TEMP_INPUT_LIMIT else 0),
        nodes=len(function.nodes),boots=sum(n.kind=='bootstrap' for n in function.nodes),
        cuts=function.cuts,encoding_combinations=len(combinations),estimated_cst_bytes=len(combinations)*model['slots']*8,
        llm_calls=0,new_trace_calls=0,new_compile_calls=0,report_images=0)
    write(args.out/'REPORT.json',report)
    return 0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inside',action='store_true');parser.add_argument('--launch',action='store_true')
    parser.add_argument('--saved',type=Path,required=True);parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    if args.inside:
        resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,6*1024**3))
        resource.setrlimit(resource.RLIMIT_FSIZE,(8*1024**3,8*1024**3))
        resource.setrlimit(resource.RLIMIT_CORE,(0,0))
        try:return inside(args)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            write(args.out/'STOP.json',dict(complete=False,error=f'{type(exc).__name__}: {exc}'))
            return 1
    if args.launch:
        from ..tools.nixrun import main as nix_main
        sys.argv=[sys.argv[0],'--seconds','1200','--','-m',__spec__.name,'--inside',
            '--saved',str(args.saved),'--out',str(args.out)]
        try:return nix_main()
        except subprocess.CalledProcessError as exc:
            stderr=exc.stderr or '';(args.out/'nix.stderr.txt').write_text(stderr)
            write(args.out/'STOP.json',dict(complete=False,stage='nix-launch',exit=exc.returncode,stderr=stderr))
            sys.stderr.write(stderr);return exc.returncode
    stamp=datetime.now(timezone.utc);run_id='u4-managed-ir-size-'+stamp.strftime('%Y%m%dT%H%M%S%f')
    args.out=args.out or RESULTS_DIR/run_id;args.out.mkdir(exist_ok=False)
    command=[sys.executable,'-B','-m',__spec__.name,'--launch','--saved',str(args.saved),'--out',str(args.out)]
    write(args.out/'argv.json',command);print('RUN='+str(args.out),flush=True);started=time.monotonic()
    with (args.out/'run.log').open('w') as log:
        status=subprocess.run(command,cwd=REPO/'DSL/DSL_agent/scripts',stdout=log,stderr=subprocess.STDOUT).returncode
    path=args.out/'REPORT.json'
    result=json.loads(path.read_text()) if path.exists() else dict(complete=False,error='see STOP.json and run.log')
    result.update(run_id=run_id,exit=status,seconds=time.monotonic()-started);write(path,result)
    from ..registry import append
    os.environ.setdefault('PT_RESULTS',str(RESULTS_DIR))
    rev=subprocess.check_output(['git','rev-parse','--short','HEAD'],cwd=REPO,text=True).strip()
    append(dict(run_id=run_id,round='U4',model='bert',mode='managed-ir-size',git_rev=rev,
        metrics=dict(complete=result['complete'],llm_calls=0),artifacts_dir=str(args.out),ts=stamp.isoformat()))
    print(json.dumps(dict(report=str(path),complete=result['complete'],exit=status)),flush=True)
    return status


if __name__=='__main__':raise SystemExit(main())
