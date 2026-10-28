from pathlib import Path
import shlex,subprocess,time
root=Path('/home/xuming/poseidon-manual-boot-20261026');e=root/'resnet20-optimized-20261027T055506Z/placement-audit/target-scale-experiment';out=e/'budget25/stem-gpu';out.mkdir(exist_ok=True)
original=root/'src/src/poseidon/tests/runtime_api/poseidon_gpu_plan_e2e.cpp';s=original.read_text()
start=s.index('        if (std::getenv("POSEIDON_RUNTIME_BOOTSTRAP_MANUAL_Q50")) {',s.index('        PoseidonGpuApi api('));end=s.index('        fhegpu::SequentialRuntime',start);s=s[:start]+s[end:]
s=s.replace('#include <vector>','#include <vector>\n#include <set>')
s=s.replace('        PoseidonGpuApi api(loaded_spec.spec.context_id, context, kCudaDevice, relin_keys);','''        auto galois = std::make_shared<poseidon::GaloisKeys>();
        std::set<int> rotations;
        for (const auto &instruction : plan.execution)
            if (const auto *op = std::get_if<fhegpu::ComputeOp>(&instruction.body))
                if (op->kind == fhegpu::ComputeKind::Rotate) {
                    int step = std::get<fhegpu::RotateAttrs>(op->attrs).steps;
                    const int slots = loaded_spec.spec.poly_degree / 2;
                    step = (step % slots + slots) % slots;
                    if (step) rotations.insert(step);
                }
        if (!rotations.empty()) key_generator.create_galois_keys(std::vector<int>(rotations.begin(), rotations.end()), *galois);
        std::cout << "Stem rotation keys ready: " << rotations.size() << std::endl;
        PoseidonGpuApi api(loaded_spec.spec.context_id, context, kCudaDevice, relin_keys, galois);''')
s=s.replace('scalar plan with manual Boot sine map','packed stem with rotations, no Boot')
(out/'stem_gpu.cpp').write_text(s)
h=(original.parent/'manual_bootstrap_reference.h').read_text();h=h.replace('case ComputeKind::Negate:', '''case ComputeKind::Rotate: {
                        const auto steps = std::get<RotateAttrs>(op->attrs).steps;
                        const auto size = static_cast<std::int64_t>(result.size());
                        const auto index = (static_cast<std::int64_t>(i) + steps % size + size) % size;
                        result[i] = values.at(op->inputs.at(0)).at(index);
                        break;
                    }
                    case ComputeKind::Negate:''')
(out/'manual_bootstrap_reference.h').write_text(h)
target=root/'build/src/poseidon/runtime_api/CMakeFiles/poseidon_runtime_gpu_plan_e2e.dir';flags={}
for line in (target/'flags.make').read_text().splitlines():
 if line.startswith('CXX_'):
  name,value=line.split(' = ',1);flags[name]=shlex.split(value)
c=['/usr/bin/c++',*flags['CXX_DEFINES'],*flags['CXX_INCLUDES'],*flags['CXX_FLAGS'],'-I'+str(original.parent),'-c',str(out/'stem_gpu.cpp'),'-o',str(out/'stem_gpu.cpp.o')]
t=time.monotonic();p=subprocess.run(c,capture_output=True,text=True);(out/'gpu-compile.log').write_text(p.stdout+p.stderr);assert p.returncode==0,p.stderr
link=shlex.split((target/'link.txt').read_text());link[link.index('-o')+1]=str(out/'stem_gpu');link=[str(out/'stem_gpu.cpp.o') if a.endswith('poseidon_gpu_plan_e2e.cpp.o') else a for a in link];p=subprocess.run(link,cwd=root/'build/src/poseidon/runtime_api',capture_output=True,text=True);(out/'gpu-link.log').write_text(p.stdout+p.stderr);assert p.returncode==0,p.stderr
print('Built stem GPU probe',round(time.monotonic()-t,2),flush=True)
