#!/usr/bin/env python3
"""Private runner instrumentation; never edits the canonical source tree."""
import hashlib
import json
from pathlib import Path
import subprocess

root=Path(__file__).resolve().parent
old=root.parent/'poseidon-qwen-prefix-20261019'
folder=root/'build-host-markers'
(folder/'include/runtime').mkdir(parents=True,exist_ok=True)
assert not (folder/'qwen').exists()
source=old/'build-observed-v2/updated/include/runtime/runtime.hpp'
text=source.read_text()
original=hashlib.sha256(text.encode()).hexdigest()
text='#include <nvtx3/nvToolsExt.h>\n'+text
replacements=[
    ('        if (!runtime_trace_) return;',
     '        if (!runtime_trace_ || op != "Boot") return;',1),
    ('consumer_ordinal,op,duration_ns\\n','consumer_ordinal,op,duration_ns,end_monotonic_ns\\n',1),
    ("<< ',' << op << ',' << duration_nanoseconds << '\\n';",
     "<< ',' << op << ',' << duration_nanoseconds << ',' << ThreadTrace::timestamp_ns() << '\\n';",1),
    ('        Value output = compute_output(op, inputs);',
     '        if (op.kind == ComputeKind::Boot) nvtxRangePushA("HostRefresh");\n'
     '        Value output = compute_output(op, inputs);\n'
     '        if (op.kind == ComputeKind::Boot) nvtxRangePop();',2),
]
for before,after,count in replacements:
    assert text.count(before)==count,(before,text.count(before))
    text=text.replace(before,after)
(folder/'include/runtime/runtime.hpp').write_text(text)
command=next(x for x in json.loads((old/'build-observed-v2/commands.json').read_text()) if x['variant']=='updated')
compile=command['compile'];compile[1]='-I'+str(folder/'include')
compile[compile.index('-o')+1]=str(folder/'qwen.o')
compile[compile.index('-MF')+1]=str(folder/'qwen.d')
link=[x.replace(str(old/'build-observed-v2/updated/qwen.o'),str(folder/'qwen.o'))
      .replace(str(old/'build-observed-v2/updated/bin/poseidon_runtime_gpu_qwen_e2e'),str(folder/'qwen'))
      for x in command['link']]
data=dict(original_header_sha256=original,header_sha256=hashlib.sha256(text.encode()).hexdigest(),
    scope='Boot-only CSV trace with monotonic end time; NVTX HostRefresh around existing compute_output; no added math or synchronization',
    replacements=replacements,compile=compile,link=link)
(root/'host-marker-instrumentation.json').write_text(json.dumps(data,indent=2)+'\n')
with (folder/'build.log').open('w') as log:
    subprocess.run(compile,cwd=command['directory'],stdout=log,stderr=subprocess.STDOUT,check=True)
    subprocess.run(link,cwd=command['directory'],stdout=log,stderr=subprocess.STDOUT,check=True)
print('Built',folder/'qwen')
