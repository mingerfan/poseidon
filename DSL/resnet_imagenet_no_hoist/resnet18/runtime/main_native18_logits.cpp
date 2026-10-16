// Output instrumentation only. Calls the same unmodified native inference entry.
#include "gpu_resnet18_inference.h"
#include <nlohmann/json.hpp>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
using namespace poseidon::benchmark::resnet18_gpu;
int main(int argc,char **argv){try{
    std::cout<<std::unitbuf;
    if(argc!=2 || std::ifstream(argv[1]).good())throw std::runtime_error("supply a new JSON result path");
    for(auto key:{"POSEIDON_RESCALE_META_BITS","POSEIDON_RESCALE_MODEL_BITS","POSEIDON_DSL_SCALE_SHIFT","POSEIDON_GPU_DNUM"})
        if(std::getenv(key))throw std::runtime_error(std::string("must be unset: ")+key);
    const char *mode=std::getenv("POSEIDON_GPU_LINEAR_TRANSFORM_MODE");
    if(!mode || std::string(mode)!="no_hoist")throw std::runtime_error("no_hoist required");
    const auto start=std::chrono::steady_clock::now();
    auto config=make_resnet18_gpu_config();auto topology=make_resnet18_topology();auto weights=load_resnet18_weights();
    const auto result=run_gpu_resnet18(0,config,topology,weights,topology.blocks.size());
    if(result.logits.size()!=1000 || result.completed_blocks!=8)throw std::runtime_error("native full output contract");
    nlohmann::json out={{"status","COMPLETE"},{"scope","original native no-hoist inference with final logits export"},
        {"image",result.image_id},{"label",result.true_label},{"prediction",result.predicted_label},
        {"logits",result.logits},{"seconds",std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()}};
    std::ofstream file(argv[1]);file<<out.dump(2)<<'\n';if(!file)throw std::runtime_error("write native logits");
    std::cout<<"NATIVE_LOGITS_EXPORT PASS prediction="<<result.predicted_label<<" label="<<result.true_label<<'\n';return 0;
}catch(const std::exception &e){std::cerr<<"NATIVE_LOGITS_ERROR "<<e.what()<<'\n';return 1;}}
