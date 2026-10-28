from pathlib import Path
import shlex
import subprocess
import time

root = Path('/home/xuming/poseidon-manual-boot-20261026')
out = root/'resnet20-optimized-20261027T055506Z/placement-audit/variable-chain32/gpu'
out.mkdir(exist_ok=True)
source = root/'src/src/poseidon/tests/runtime_api/poseidon_gpu_plan_e2e.cpp'
text = source.read_text()
start = text.index('        if (std::getenv("POSEIDON_RUNTIME_BOOTSTRAP_MANUAL_Q50")) {',text.index('        PoseidonGpuApi api('))
end = text.index('        fhegpu::SequentialRuntime',start)
text = text[:start]+text[end:]
text = text.replace('for (std::size_t i = 0; i < input.size(); ++i) input[i] = double(i % 17 + 1) / 32;',
    'const std::vector<double> samples{-0.75,-0.5,-0.25,-0.125,0.0,0.125,0.25,0.5,0.75,0.875};\n            for (std::size_t i = 0; i < input.size(); ++i) input[i] = samples[i % samples.size()];')
text = text.replace('scalar plan with manual Boot sine map','single ReLU, no Boot, actual-prime context')
(out/'relu_gpu.cpp').write_text(text)
target = root/'build/src/poseidon/runtime_api/CMakeFiles/poseidon_runtime_gpu_plan_e2e.dir'
flags = {}
for line in (target/'flags.make').read_text().splitlines():
    if line.startswith('CXX_'):
        name,value = line.split(' = ',1)
        flags[name] = shlex.split(value)
command = ['/usr/bin/c++',*flags['CXX_DEFINES'],*flags['CXX_INCLUDES'],*flags['CXX_FLAGS'],'-I'+str(source.parent),'-c',str(out/'relu_gpu.cpp'),'-o',str(out/'relu_gpu.cpp.o')]
t = time.monotonic()
run = subprocess.run(command,capture_output=True,text=True)
(out/'compile.log').write_text(run.stdout+run.stderr)
assert run.returncode==0,run.stderr
link = shlex.split((target/'link.txt').read_text())
link[link.index('-o')+1] = str(out/'relu_gpu')
link = [str(out/'relu_gpu.cpp.o') if a.endswith('poseidon_gpu_plan_e2e.cpp.o') else a for a in link]
run = subprocess.run(link,cwd=root/'build/src/poseidon/runtime_api',capture_output=True,text=True)
(out/'link.log').write_text(run.stdout+run.stderr)
assert run.returncode==0,run.stderr
print('Built isolated no-Boot GPU probe in',round(time.monotonic()-t,2),'seconds',flush=True)
