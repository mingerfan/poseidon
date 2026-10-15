#include "resnet18/gpu_resnet18_inference.cpp"
#include "resnet50/gpu_resnet50_inference.cpp"
#include <iomanip>
int main(int argc,char **argv){
    using namespace poseidon::benchmark;
    using Runtime=resnet50_gpu::GpuCkksRuntime;
    if(argc<3 || argc>4)throw std::runtime_error("usage: native_trace 18|50 output.jsonl [CPU-op-limit]");
    const int model=std::stoi(argv[1]);
    if(model!=18 && model!=50)throw std::runtime_error("model must be 18 or 50");
    Runtime::output.open(argv[2]);if(!Runtime::output)throw std::runtime_error("cannot create trace");
    Runtime::payload.open(std::string(argv[2])+".zlib",std::ios::binary);
    if(!Runtime::payload)throw std::runtime_error("cannot create lossless input/constant payload");
    if(argc==4)Runtime::limit=std::stoull(argv[3]);
    std::cout<<std::unitbuf<<std::setprecision(17);
    try {
        const auto config=resnet50_gpu::make_resnet50_gpu_config();
        Runtime runtime(config);
        if(model==18){
            const auto topology=resnet18_gpu::make_resnet18_topology();
            const auto weights=resnet18_gpu::load_resnet18_weights();
            resnet18_gpu::run_gpu_resnet18_impl(0,topology,weights,8,runtime,false,false,false);
        }else{
            const auto topology=resnet50_gpu::make_resnet50_topology();
            const auto weights=resnet50_gpu::load_resnet50_weights();
            resnet50_gpu::run_gpu_resnet50_impl(0,topology,weights,16,runtime,false,false,false);
        }
        throw std::runtime_error("trace returned without a complete output boundary");
    }catch(const resnet50_gpu::TraceComplete &){
        std::cout<<"CPU_NATIVE_TRACE COMPLETE model="<<model<<" ops="<<Runtime::next_id
            <<" inputs="<<Runtime::inputs<<" bootstrap_calls="<<Runtime::boots
            <<" gpu_executed=false semantics_validated=false\n";
    }catch(const resnet50_gpu::TraceLimit &){
        std::cout<<"CPU_NATIVE_TRACE TRUNCATED model="<<model<<" ops="<<Runtime::next_id
            <<" inputs="<<Runtime::inputs<<" bootstrap_calls="<<Runtime::boots
            <<" gpu_executed=false semantics_validated=false\n";
    }catch(const std::exception &error){
        std::cerr<<"CPU_NATIVE_TRACE ERROR "<<error.what()<<'\n';return 1;
    }
}
