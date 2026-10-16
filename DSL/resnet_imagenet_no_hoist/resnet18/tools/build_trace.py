"""Rebuild the native-driven CPU recorder; never invokes a GPU."""
import shutil,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def build(output):
    output.mkdir(parents=True,exist_ok=False)
    native=ROOT/'runtime/native_snapshot'
    for model in (18,50):shutil.copytree(native/f'benchmark/resnet{model}_gpu',output/f'resnet{model}')
    inference=output/'resnet18/gpu_resnet18_inference.cpp';text=inference.read_text()
    hook='    const auto begin = std::chrono::steady_clock::now();';assert text.count(hook)==1
    text=text.replace(hook,'    poseidon::benchmark::resnet50_gpu::GpuCkksRuntime::stage(label);\n'+hook)
    hook='    result.logits.resize(encrypted_logits.size());';assert text.count(hook)==2
    text=text.replace(hook,'    runtime.finish(encrypted_logits);\n'+hook,1)
    inference.write_text(text)
    shutil.copy2(ROOT/'tools/cpu_trace_runtime.h',output/'resnet50/gpu_ckks_runtime.h')
    (output/'cuda_profiler_api.h').write_text('#pragma once\n#include <stdexcept>\ninline constexpr int cudaSuccess=0;\ninline int cudaProfilerStart(){throw std::runtime_error("CPU trace only");}\ninline int cudaProfilerStop(){throw std::runtime_error("CPU trace only");}\n')
    (output/'resnet50/gpu_activity_timing.h').write_text('''#pragma once
#include <string>
namespace poseidon::benchmark::s2c_first {
struct GpuActivityTimingResult {void print()const{}};
struct GpuActivityTiming {void stage(const std::string &,bool){} void begin_continuous(){}
GpuActivityTimingResult finish(){return {};}};}
''')
    (output/'main.cpp').write_text('''#include "resnet18/gpu_resnet18_inference.cpp"
int main(int argc,char **argv){
    using namespace poseidon::benchmark;
    using Runtime=resnet50_gpu::GpuCkksRuntime;
    if(argc!=2)throw std::runtime_error("trace path required");
    Runtime::output.open(argv[1]);Runtime::payload.open(std::string(argv[1])+".zlib",std::ios::binary);
    if(!Runtime::output || !Runtime::payload)throw std::runtime_error("output open failed");
    try {
      Runtime runtime(resnet50_gpu::make_resnet50_gpu_config());
      const auto topology=resnet18_gpu::make_resnet18_topology();
      const auto weights=resnet18_gpu::load_resnet18_weights();
      resnet18_gpu::run_gpu_resnet18_impl(0,topology,weights,8,runtime,false,false,false);
      throw std::runtime_error("missing complete output boundary");
    }catch(const resnet50_gpu::TraceComplete &){
      std::cout<<"CPU_NATIVE_TRACE COMPLETE inputs="<<Runtime::inputs<<" boots="<<Runtime::boots<<'\\n';
    }
}
''')
    cmd=['c++','-std=c++20','-O1','-ffunction-sections','-fdata-sections','-Wl,--gc-sections']
    cmd+=['-I'+str(output),'-I'+str(output/'resnet18'),'-I'+str(output/'resnet50')]
    cmd+=['-DPOSEIDON_GPU_RESNET18_SOURCE_DIR="'+str(native/'benchmark/resnet18_gpu')+'"',
          '-DPOSEIDON_GPU_RESNET50_SOURCE_DIR="'+str(native/'benchmark/resnet50_gpu')+'"',
          '-DPOSEIDON_RESNET20_RELU_FIXTURE="'+str(native/'Doc/analysis/resnet20_joint_chain_20260911/relu_precision_fixture.txt')+'"']
    cmd += [str(output/'main.cpp')]+[str(output/name) for name in
      ('resnet18/gpu_config.cpp','resnet18/resnet18_topology.cpp','resnet18/resnet18_weights.cpp',
       'resnet50/resnet50_config.cpp','resnet50/gpu_multiplexed_tensor.cpp','resnet50/gpu_relu_s2c_first.cpp')]
    subprocess.run(cmd+['-lz','-o',str(output/'native_trace')],check=True)
    return output/'native_trace'
