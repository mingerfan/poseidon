#include "source_runtime.hpp"
#include "source_model.hpp"
#include "source_accuracy.hpp"
#include "llama_manual/benchmark_io.hpp"
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <rmm/mr/device/statistics_resource_adaptor.hpp>
#include <iomanip>

struct RestoreResource {
    rmm::mr::device_memory_resource *old;
    ~RestoreResource(){rmm::mr::set_current_device_resource(old);}
};
int main(int argc,char **argv) {
    using namespace cerium_source;
    using namespace poseidon::gpu;
    try {
        if(argc!=10)throw std::invalid_argument("fixture weights index embeddings reference output layers valid_tokens report_json");
        const auto layers=std::stoul(argv[7]),valid_tokens=std::stoul(argv[8]);
        if(layers<1||layers>32||!valid_tokens||valid_tokens>128)throw std::invalid_argument("model boundary");
        if(std::filesystem::exists(argv[6])||std::filesystem::exists(argv[9]))throw std::runtime_error("refusing to overwrite model result");
        const auto wall=Clock::now();std::cout<<std::setprecision(17);
        const auto enabled=[](const char *name){const char *v=std::getenv(name);return v&&std::string(v)=="1";};
        llama_manual::SafetensorWeights weights(argv[2],argv[3]);weights.validate_model_index(layers);
        const auto embeddings=llama_manual::benchmark::read_tensor(argv[4],"embeddings",0);
        const auto reference=llama_manual::benchmark::read_tensor(argv[5],"hidden",layers);
        gpu_check_cuda(cudaSetDevice(0),"source model device");
        auto *old=rmm::mr::get_current_device_resource();
        rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource> pool(old,64ULL<<20,60ULL<<30);
        rmm::mr::statistics_resource_adaptor<rmm::mr::device_memory_resource> memory(&pool);
        RestoreResource restore{old};rmm::mr::set_current_device_resource(&memory);
        Runtime runtime(argv[1]);auto &api=*runtime.api;
        NativeLinearAdapter linear(api,runtime.context);
        std::vector<CT> state;
        for(std::size_t tile=0;tile<layout::tiles(layout::hidden_width);++tile)
            state.push_back(runtime.encrypt(layout::pack_tile(embeddings,layout::hidden_width,tile),6));
        const auto mask=layout::causal_mask(valid_tokens);
        auto encrypted_mask=runtime.encrypt(mask,30);
        gpu_check_cuda(cudaDeviceSynchronize(),"source model encrypted inputs ready");
        const auto compute=Clock::now();
        std::cout<<"SOURCE_MODEL_BEGIN blocks="<<layers<<" valid_tokens="<<valid_tokens
            <<" N="<<runtime.context.parameters_literal()->degree()<<" complex_slots="<<runtime.encoder.slot_count()
            <<" capacity=128 hidden=4096 ffn=14336 q_heads=32 kv_heads=8"
            <<" precision_gate=false linear_adapter=native_dft vector_pairing=false"
            <<" group_accum="<<enabled("LLAMA_LINEAR_GROUP_ACCUM")
            <<" baby_cache="<<enabled("LLAMA_LINEAR_BABY_CACHE")
            <<" packed_upload="<<enabled("LLAMA_LINEAR_PACKED_UPLOAD")
            <<" persistent_prepare="<<enabled("LLAMA_LINEAR_PERSISTENT_PREP")
            <<" scalar_cache="<<enabled("LLAMA_SCALAR_CACHE")
            <<" add_prefix="<<enabled("LLAMA_ADD_PREFIX")
            <<" periodic_reduced="<<enabled("LLAMA_PERIODIC_REDUCED_FASTPATH")
            <<" periodic_gpu_encode="<<enabled("LLAMA_LINEAR_GPU_ENCODE")
            <<" periodic_gpu_pipeline="<<enabled("LLAMA_LINEAR_GPU_PIPELINE")
            <<" fused_relin_rs2="<<enabled("LLAMA_FUSED_RELIN_RS2")
            <<" weight_decode_fast="<<enabled("LLAMA_WEIGHT_DECODE_FASTPATH")<<std::endl;
        double weight_load_seconds=0;
        for(std::size_t layer=0;layer<layers;++layer) {
            const auto layer_start=Clock::now();auto w=weights.load_layer(layer);
            const auto load_seconds=seconds(layer_start);weight_load_seconds+=load_seconds;
            std::cout<<"SOURCE_WEIGHT_IO layer="<<layer<<" seconds="<<load_seconds<<std::endl;
            state=source_block(api,linear,state,w,encrypted_mask,layer);
            gpu_check_cuda(cudaDeviceSynchronize(),"source block complete");
            std::cout<<"SOURCE_BLOCK_FINISHED layer="<<layer<<" seconds="<<seconds(layer_start)
                <<" compute_elapsed="<<seconds(compute)<<" boots="<<api.boot_calls
                <<" output_q="<<state[0].meta.q_count<<std::endl;
        }
        const auto compute_seconds=seconds(compute);
        // No decryption or plaintext feedback occurs between model blocks.
        std::vector<double> output(128*4096);ComplexReferenceMetric metric;
        for(std::size_t tile=0;tile<16;++tile) {
            const auto decoded=runtime.decrypt_final(state[tile]);
            if(decoded.size()!=32768)throw std::runtime_error("wrong final slot count");
            for(std::size_t token=0;token<128;++token)for(std::size_t j=0;j<256;++j) {
                const auto value=decoded[token*256+j];
                const auto index=token*4096+tile*256+j;
                metric.observe(value,reference[index],token<valid_tokens);
                output[index]=value.real();
            }
        }
        const auto accuracy=metric.finish(); // finite aggregate validation precedes all output writes
        const auto real_error=accuracy.maximum_real_error,complex_error=accuracy.maximum_complex_error;
        const auto imaginary=accuracy.maximum_imaginary,maximum=accuracy.maximum_absolute_output;
        llama_manual::benchmark::write_tensor(argv[6],output,"hidden",layers);
        const auto peak=memory.get_bytes_counter().peak;
        std::ofstream report(argv[9]);report<<std::setprecision(17)
            <<"{\n  \"status\":\"completed_finite_output\",\n  \"blocks\":"<<layers
            <<",\n  \"valid_tokens\":"<<valid_tokens<<",\n  \"capacity\":128,\n  \"accuracy_gate\":false"
            <<",\n  \"ring_degree_N\":"<<runtime.context.parameters_literal()->degree()
            <<",\n  \"complex_slots\":"<<runtime.encoder.slot_count()
            <<",\n  \"layout_contract_version\":"<<layout::contract_version
            <<",\n  \"precision_metric\":\"active_token_maximum_complex_norm\""
            <<",\n  \"maximum_complex_error\":"<<complex_error
            <<",\n  \"precision_passed\":"<<(accuracy.passed()?"true":"false")
            <<",\n  \"maximum_real_error\":"<<real_error<<",\n  \"maximum_imaginary\":"<<imaginary
            <<",\n  \"rmse\":"<<accuracy.rmse<<",\n  \"maximum_absolute_output\":"<<maximum
            <<",\n  \"compute_seconds\":"<<compute_seconds<<",\n  \"whole_seconds\":"<<seconds(wall)<<",\n  \"weight_load_seconds\":"<<weight_load_seconds
            <<",\n  \"native_bootstrap_calls\":"<<api.boot_calls<<",\n  \"implicit_bootstrap_calls\":0"
            <<",\n  \"plain_cache_hits\":"<<api.plain_cache_hits<<",\n  \"plain_cache_misses\":"<<api.plain_cache_misses
            <<",\n  \"compact_plain_preparations\":"<<api.compact_plain_preparations
            <<",\n  \"rmm_live_peak_bytes\":"<<peak<<",\n  \"pool_reserved_bytes\":"<<pool.pool_size()
            <<",\n  \"reference_kind\":\"Cerium original_callpath_v2 float64 approximate packed reference with fixed jitter\""
            <<",\n  \"linear_adapter\":\""<<(enabled("LLAMA_LINEAR_GROUP_ACCUM")?"native_dft_group_sum":"native_dft_per_input_rescale")<<"\""
            <<",\n  \"linear_baby_cache\":"<<(enabled("LLAMA_LINEAR_BABY_CACHE")?"true":"false")
            <<",\n  \"linear_packed_upload\":"<<(enabled("LLAMA_LINEAR_PACKED_UPLOAD")?"true":"false")
            <<",\n  \"linear_persistent_prepare\":"<<(enabled("LLAMA_LINEAR_PERSISTENT_PREP")?"true":"false")
            <<",\n  \"scalar_cache\":"<<(enabled("LLAMA_SCALAR_CACHE")?"true":"false")
            <<",\n  \"add_q_prefix\":"<<(enabled("LLAMA_ADD_PREFIX")?"true":"false")
            <<",\n  \"periodic_reduced_fastpath\":"<<(enabled("LLAMA_PERIODIC_REDUCED_FASTPATH")?"true":"false")
            <<",\n  \"periodic_gpu_encoding\":"<<(enabled("LLAMA_LINEAR_GPU_ENCODE")?"true":"false")
            <<",\n  \"periodic_gpu_pipeline\":"<<(enabled("LLAMA_LINEAR_GPU_PIPELINE")?"true":"false")
            <<",\n  \"fused_relin_rs2\":"<<(enabled("LLAMA_FUSED_RELIN_RS2")?"true":"false")
            <<",\n  \"weight_decode_fastpath\":"<<(enabled("LLAMA_WEIGHT_DECODE_FASTPATH")?"true":"false")
            <<",\n  \"scalar_cache_hits\":"<<api.scalar_cache_hits<<",\n  \"scalar_cache_misses\":"<<api.scalar_cache_misses
            <<",\n  \"optimization_flag_evidence\":\"requested_environment_not_consumer_attestation\""
            <<",\n  \"original_scheduler\":false"
            <<",\n  \"vector_pairing\":false,\n  \"intermediate_decryption\":false\n}\n";
        if(!report)throw std::runtime_error("model report write failed");
        std::cout<<"SOURCE_MODEL_COMPLETE blocks="<<layers<<" finite=true precision_gate=false compute_seconds="<<compute_seconds
            <<" max_error="<<real_error<<" boots="<<api.boot_calls<<" live_peak_bytes="<<peak<<std::endl;
        return 0;
    }catch(const std::exception &e){std::cerr<<"SOURCE_MODEL_FAILED "<<e.what()<<std::endl;return 1;}
}
