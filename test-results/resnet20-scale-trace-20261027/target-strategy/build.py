from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import shlex
import shutil
import subprocess
import tarfile
import time

root = Path('/home/xuming/poseidon-manual-boot-20261026')
old = Path('/home/xuming/poseidon-mixed-q-20261026')
build = root/'compiler/compiler-build'
src = root/'compiler/dacapo'
perf = root/'resnet20-optimized-20261027T055506Z/performance'
audit = perf.parent/'placement-audit'
out = audit/'target-scale-experiment'
out.mkdir(exist_ok=True)
with tarfile.open(audit/'target-scale-src.tar') as archive:
    archive.extractall(out)
if not (out/'include').exists():
    shutil.copytree(perf/'include',out/'include')
local = (old/'source-root.txt').read_text()
def mapped(arg):
    return arg.replace(local+'/build/nix',str(build)).replace(local,str(src)).replace('/nix/store/',str(old/'deps/nix/store')+'/')
loader = old/'toolchain/lib/ld-linux-x86-64.so.2'
libraries = ':'.join(map(str,[old/'toolchain/lib',old/'tools',perf/'tools']))
def run(command,name):
    t = time.monotonic()
    with (out/(name+'.log')).open('w') as log:
        done = subprocess.run(command,cwd=build,stdout=log,stderr=subprocess.STDOUT)
    print(name,done.returncode,round(time.monotonic()-t,2),flush=True)
    if done.returncode:
        print((out/(name+'.log')).read_text()[-7000:],flush=True)
        raise RuntimeError(name)

includes = ['-I'+str(out/'src/include'),'-I'+str(out/'include'),'-I'+str(src/'include/hecate/Dialect/Earth/IR'),'-I'+str(src/'include'),'-I'+str(old/'deps/nix/store/v0vnvpp16nijl7pf2yj8k4bvfdcfp65r-llvm-18.1.8-dev/include'),'-I'+str(old/'deps/nix/store/6sgd3j8jw4624dnlwy4xldszblw4y0f0-mlir-18.1.8-dev/include')]
run([str(loader),'--library-path',libraries,str(perf/'tools/mlir-tblgen'),'-gen-rewriters',*includes,str(out/'src/include/hecate/Dialect/Earth/IR/EarthCanonicalizer.td'),'-o',str(out/'include/hecate/Dialect/Earth/IR/EarthCanonicalizerPattern.inc')],'tablegen')
flags = [mapped(a) for a in shlex.split(json.loads((root/'compiler/commands.json').read_text())[0]['command'])[1:]]
compiler = [str(loader),'--library-path',libraries,str(old/'tools/clang++'),'-nostdinc','-nostdinc++']
for path in ['/nix/store/qxaq7jz61a6zkr2mq49i0zvqip2m2jj8-gcc-15.2.0/include/c++/15.2.0','/nix/store/qxaq7jz61a6zkr2mq49i0zvqip2m2jj8-gcc-15.2.0/include/c++/15.2.0/x86_64-unknown-linux-gnu','/nix/store/i3v1x1v3hzcmsx45q2i60g0248n40r3h-clang-wrapper-18.1.8/resource-root/include','/nix/store/15h9askp4k1lx44d9871wid23j2a8ijp-glibc-2.42-61-dev/include']:
    compiler += ['-isystem',mapped(path)]
compiler += ['-I'+str(out/'src/include'),'-I'+str(out/'include')]
entries = [
 ('lib/Dialect/Earth/IR/EarthDialect.cpp','libEarthDialect.a'),
 ('lib/Dialect/Earth/Transforms/TargetScalePlanning.cpp','libEarthTransforms.a'),
 ('lib/Dialect/Earth/Transforms/GreedyBootstrapPlacement.cpp','libEarthTransforms.a'),
 ('lib/Dialect/Earth/Transforms/ProactiveRescaling.cpp','libEarthTransforms.a'),
 ('lib/Dialect/Earth/Transforms/EarlyModswitch.cpp','libEarthTransforms.a'),
 ('lib/Conversion/EarthToCKKS/EarthToCKKS.cpp','libHecateEarthToCKKS.a'),
 ('lib/Dialect/CKKS/Transforms/MaterializePhysicalLevels.cpp','libCKKSTransforms.a'),
]
def compile_one(entry):
    relative,_ = entry
    source = out/'src'/relative
    obj = out/(source.name+'.o')
    relevant = [source,out/'src/include/hecate/Dialect/Earth/Transforms/Common.h',out/'src/include/hecate/Dialect/Earth/IR/EarthOps.td']
    if source.stem == 'EarthDialect':
        relevant.append(out/'include/hecate/Dialect/Earth/IR/EarthCanonicalizerPattern.inc')
    digest = hashlib.sha256(b''.join(p.read_bytes() for p in relevant)).hexdigest()
    stamp = out/(source.name+'.compile-hash')
    if obj.exists() and stamp.exists() and stamp.read_text()==digest:
        print('unchanged',source.stem,flush=True)
        return
    args = flags[:]
    args[args.index('-c')+1] = str(source)
    args[args.index('-o')+1] = str(obj)
    run([*compiler,*args],'compile-'+source.stem)
    stamp.write_text(digest)
with ThreadPoolExecutor(max_workers=4) as pool:
    list(pool.map(compile_one,entries))
for name in sorted({archive for _,archive in entries}):
    origin = perf/'equivalent-states'/name if name == 'libEarthTransforms.a' else perf/name if name == 'libEarthDialect.a' else build/'lib'/name
    assert origin.exists(),origin
    shutil.copy2(origin,out/name)
    objects = [str(out/(Path(source).name+'.o')) for source,archive in entries if archive==name]
    subprocess.run(['ar','rcs',str(out/name),*objects],check=True)
link = [mapped(a) for a in json.loads((old/'link.json').read_text())[1:]]
overrides = {'lib/'+name:str(out/name) for _,name in entries}
link = [overrides.get(a,a) for a in link]
link[link.index('-o')+1] = str(out/'hecate-opt')
run([str(loader),'--library-path',libraries,str(old/'tools/ld'),'-rpath-link',str(old/'toolchain/lib'),*link],'link')
wrapper = out/'hecate-opt-wrapper'
wrapper.write_text('#!/bin/sh\nexec '+str(loader)+' --library-path '+str(old/'toolchain/lib')+' '+str(out/'hecate-opt')+' "$@"\n')
wrapper.chmod(0o755)
(out/'binary.sha256').write_text(hashlib.sha256((out/'hecate-opt').read_bytes()).hexdigest()+'\n')
print('ready',wrapper,flush=True)
