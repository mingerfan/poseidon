"""Isolate original ImageNet tensor/inference/ReLU code for CPU plan recording."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

def prepare(root, output):
    output.mkdir(parents=True,exist_ok=True)
    hashes={}
    for model in (18,50):
        source=root/f'benchmark/resnet{model}_gpu'
        target=output/f'resnet{model}'
        target.mkdir(exist_ok=True)
        for file in source.iterdir():
            if file.suffix not in ('.h','.cpp'):
                continue
            hashes[str(file.relative_to(root))]=hashlib.sha256(file.read_bytes()).hexdigest()
            shutil.copy2(file,target/file.name)
        inference=target/f'gpu_resnet{model}_inference.cpp'
        text=inference.read_text()
        hook='    const auto begin = std::chrono::steady_clock::now();'
        if text.count(hook)!=1:
            raise ValueError('Native stage timer layout changed')
        text=text.replace(hook,'    poseidon::benchmark::resnet50_gpu::GpuCkksRuntime::stage(label);\n'+hook)
        end='    result.logits.resize(encrypted_logits.size());'
        if end not in text:
            raise ValueError('Native final logits boundary changed')
        text=text.replace(end,'    runtime.finish(encrypted_logits);\n'+end,1)
        inference.write_text(text)
    shutil.copy2(Path(__file__).with_name('cpu_trace_runtime.h'),output/'resnet50/gpu_ckks_runtime.h')
    (output/'cuda_profiler_api.h').write_text('''#pragma once
#include <stdexcept>
inline constexpr int cudaSuccess=0;
inline int cudaProfilerStart(){throw std::runtime_error("GPU timing disabled in CPU trace");}
inline int cudaProfilerStop(){throw std::runtime_error("GPU timing disabled in CPU trace");}
''')
    (output/'resnet50/gpu_activity_timing.h').write_text('''#pragma once
#include <string>
namespace poseidon::benchmark::s2c_first {
struct GpuActivityTimingResult {void print()const{}};
struct GpuActivityTiming {
 void stage(const std::string &,bool){} void begin_continuous(){}
 GpuActivityTimingResult finish(){return {};}
};}
''')
    source50=(root/'benchmark/resnet50_gpu/gpu_resnet50_inference.cpp').read_text()
    head=source50[source50.index('GpuResNet50Result run_gpu_resnet50_head_check('):]
    begin=head.index('    std::vector<double> plain_averages(')
    end=head.index('    GpuCkksRuntime runtime(config);',begin)
    (output/'native_plain_head50.h').write_text('''// Extracted unchanged from the original native head-check oracle.
template<typename Features>
std::vector<double> native_plain_head50(const Features &features,
    const poseidon::benchmark::resnet50_gpu::ResNet50Weights &weights) {
    using namespace poseidon::benchmark::resnet50_gpu;
'''+head[begin:end]+'''    return plain_logits;
}
''')
    (output/'native-source-provenance.json').write_text(json.dumps({
        'scope':'CPU metadata adapter, no semantic/GPU acceptance',
        'source_hashes':hashes,
        'inference_edits':['record original timed stage name','export output IDs before final decrypt'],
        'tensor_and_relu_algorithms':'copied unchanged'},indent=2)+'\n')
    print('Prepared isolated CPU recorder; production native source unchanged')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('native_root',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();prepare(args.native_root,args.output)
