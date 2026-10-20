"""Build the validated deferred-plaintext optimizer driver against a built Dacapo."""
import argparse
import hashlib
import json
import shlex
import subprocess
from pathlib import Path

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hecate-build', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    cwd = args.hecate_build.resolve()/'tools'
    target = cwd/'CMakeFiles/hecate-opt.dir'
    out = args.output.resolve()
    out.mkdir()
    flags = {}
    for line in (target/'flags.make').read_text().splitlines():
        if line.startswith(('CXX_DEFINES =','CXX_INCLUDES =','CXX_FLAGS =')):
            key,value = line.split('=',1)
            flags[key.strip()] = shlex.split(value)
    source = Path(__file__).resolve().with_name('optimizer.cpp')
    obj = out/'optimizer.cpp.o'
    command = ['/usr/bin/c++',*flags['CXX_DEFINES'],*flags['CXX_INCLUDES'],
               *flags['CXX_FLAGS'],'-c',str(source),'-o',str(obj)]
    link = shlex.split((target/'link.txt').read_text())
    positions = [i for i,value in enumerate(link) if value.endswith('.cpp.o')]
    assert len(positions) == 1
    link[positions[0]] = str(obj)
    link[link.index('-o')+1] = str(out/'hecate-opt')
    with (out/'build.log').open('x') as log:
        for cmd in (command,link):
            subprocess.run(cmd,cwd=cwd,stdout=log,stderr=subprocess.STDOUT,check=True)
    (out/'provenance.json').write_text(json.dumps(dict(compile=command,link=link,cwd=str(cwd),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        binary_sha256=hashlib.sha256((out/'hecate-opt').read_bytes()).hexdigest()),indent=2)+'\n')
