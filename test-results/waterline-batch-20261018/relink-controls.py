#!/usr/bin/env python3
"""Relink frozen Runtime objects against the same corrected GPU backend."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

parser=argparse.ArgumentParser()
parser.add_argument('--historical',type=Path,required=True)
parser.add_argument('--updated',type=Path,required=True)
parser.add_argument('--backend',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
args.output=args.output.resolve();args.output.mkdir(exist_ok=False)
historical=json.loads((args.historical/'commands.json').read_text())
updated=json.loads((args.updated/'commands.json').read_text())
records=[]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
for variant,source,items in [('legacy','legacy',historical),('unfixed','unfixed',historical),
                              ('previous','current',historical),('updated','updated',updated)]:
    folder=args.output/variant;(folder/'bin').mkdir(parents=True)
    for runner in ('mlp','qwen'):
        record=next(r for r in items if r['variant']==source and r['runner']==runner)
        compile=record['compile']
        obj=Path(compile[compile.index('-o')+1])
        overlay=next(Path(t[2:]) for t in compile if t.startswith('-I') and (Path(t[2:])/'runtime/runtime.hpp').exists())
        header=overlay/'runtime/runtime.hpp'
        assert sha(header)==record['header_sha256']
        if not (folder/'runtime.hpp').exists():shutil.copyfile(header,folder/'runtime.hpp')
        command=record['link'].copy()
        command[command.index('-o')+1]=str(folder/'bin'/f'poseidon_runtime_gpu_{runner}_e2e')
        with (folder/(runner+'-link.log')).open('w') as log:
            subprocess.run(command,cwd=record['directory'],stdout=log,stderr=subprocess.STDOUT,check=True)
        records.append(dict(variant=variant,runner=runner,original_compile=compile,link=command,
            directory=record['directory'],object_sha256=sha(obj),header_sha256=sha(header),
            executable_sha256=sha(Path(command[command.index('-o')+1]))))
        print('Linked',variant,runner,flush=True)
(args.output/'commands.json').write_text(json.dumps(records,indent=2)+'\n')
libraries=[args.backend/name for name in ('libposeidon_runtime_gpu_api.a','libruntime_core.a','libruntime_plan_json.a','libposeidon_gpu.a','libposeidon_shared.so')]
(args.output/'libraries.json').write_text(json.dumps({str(p.resolve()):sha(p) for p in libraries},indent=2)+'\n')
