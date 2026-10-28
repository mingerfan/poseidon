from pathlib import Path
import json,shlex,shutil,subprocess,time
root=Path('/home/xuming/poseidon-manual-boot-20261026')
audit=root/'resnet20-optimized-20261027T055506Z/placement-audit'
script=(audit/'poseidon-target-scale-build.py').read_text()
g={};exec(compile(script[:script.index('def compile_one')],str(audit/'poseidon-target-scale-build.py'),'exec'),g)
e=g['out'];source=(e/'src/lib/Dialect/Earth/Transforms/DaCapoPlanner.cpp').read_text()
for name,boot_first in [('single-latency',False),('single-boots',True)]:
 out=e/'budget25'/name;out.mkdir(parents=True,exist_ok=True)
 text=source
 start=text.index('auto same = llvm::find_if(states,')
 end=text.index('*same = std::move(candidate);',start)+len('*same = std::move(candidate);')
 select='candidate.cost < states.front().cost'
 if boot_first:
  text=text.replace('#include <map>','#include <map>\n#include <tuple>')
  text=text.replace('SmallVector<double, 4> scales;','SmallVector<double, 4> scales;\n      unsigned boots = 0;')
  anchor='candidate.cost += copy->getAttrOfType<FloatAttr>("est_latency")\n                                      .getValueAsDouble();'
  assert anchor in text
  text=text.replace(anchor,anchor+'\n                copy.walk([&](hecate::earth::BootstrapOp) { ++candidate.boots; });')
  start=text.index('auto same = llvm::find_if(states,');end=text.index('*same = std::move(candidate);',start)+len('*same = std::move(candidate);')
  select='std::tie(candidate.boots, candidate.cost) < std::tie(states.front().boots, states.front().cost)'
  text=text.replace('return a.cost < b.cost;', 'return std::tie(a.boots, a.cost) < std::tie(b.boots, b.cost);')
 text=text[:start]+f'if (states.empty()) states.push_back(std::move(candidate));\n                else if ({select}) states.front() = std::move(candidate);'+text[end:]
 path=out/'DaCapoPlanner.cpp';path.write_text('// Experiment: one retained prefix per cut.\n'+text)
 flags=g['flags'][:];flags[flags.index('-c')+1]=str(path);flags[flags.index('-o')+1]=str(out/'DaCapoPlanner.cpp.o')
 t=time.monotonic()
 run=subprocess.run([*g['compiler'],*flags],cwd=g['build'],capture_output=True,text=True)
 (out/'compile.log').write_text(run.stdout+run.stderr);assert run.returncode==0,run.stderr
 shutil.copy2(e/'libEarthTransforms.a',out/'libEarthTransforms.a')
 subprocess.run(['ar','rcs',str(out/'libEarthTransforms.a'),str(out/'DaCapoPlanner.cpp.o')],check=True)
 link=[g['mapped'](a) for a in json.loads((g['old']/'link.json').read_text())[1:]]
 overrides={'lib/'+archive:str(e/archive) for _,archive in g['entries']};overrides['lib/libEarthTransforms.a']=str(out/'libEarthTransforms.a')
 link=[overrides.get(a,a) for a in link];link[link.index('-o')+1]=str(out/'hecate-opt')
 run=subprocess.run([str(g['loader']),'--library-path',g['libraries'],str(g['old']/'tools/ld'),'-rpath-link',str(g['old']/'toolchain/lib'),*link],cwd=g['build'],capture_output=True,text=True)
 (out/'link.log').write_text(run.stdout+run.stderr);assert run.returncode==0,run.stderr
 wrapper=out/'hecate-opt-wrapper';wrapper.write_text('#!/bin/sh\nexec '+str(g['loader'])+' --library-path '+str(g['old']/'toolchain/lib')+' '+str(out/'hecate-opt')+' "$@"\n');wrapper.chmod(0o755)
 print(name,'built',round(time.monotonic()-t,2),flush=True)
