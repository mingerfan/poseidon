// NEW experimental handwritten Llama entry. No DSL runtime, no plaintext
// replacement of encrypted operators, no intermediate decryption. Run only
// through a separately admitted owned GPU guard; this file is NOT a launch.
#include "source_manual_program.hpp"
#include "source_manual_io.hpp"
#include "source_accuracy.hpp"
#include "llama_manual/benchmark_io.hpp"
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <rmm/mr/device/statistics_resource_adaptor.hpp>
#include <cerrno>
#include <fcntl.h>
#include <iomanip>
#include <unistd.h>

namespace {
class Report {
    int fd_;
public:
    explicit Report(const std::string &path):fd_(::open(path.c_str(),O_CREAT|O_EXCL|O_WRONLY|O_NOFOLLOW|O_CLOEXEC,0600)) {
        if(fd_<0)throw std::runtime_error("exclusive manual report open failed");
    }
    ~Report(){::close(fd_);}
    void finish(const std::string &text) {
        std::size_t at=0;
        while(at<text.size()) {const auto n=::write(fd_,text.data()+at,text.size()-at);if(n<0&&errno==EINTR)continue;
            if(n<=0)throw std::runtime_error("manual report write failed");at+=std::size_t(n);}
        if(::fsync(fd_))throw std::runtime_error("manual report fsync failed");
    }
};
struct RestoreResource {rmm::mr::device_memory_resource *old;~RestoreResource(){rmm::mr::set_current_device_resource(old);}};
void configure_safe_implementation() {
    auto fixed=[](const char *name,const char *wanted) {
        if(const char *p=std::getenv(name);p&&*p&&std::string(p)!="0"&&std::string(p)!=wanted)
            throw std::invalid_argument(std::string("unsupported inherited manual option ")+name);
        if(::setenv(name,wanted,1))throw std::runtime_error("manual environment setup failed");
    };
    for(const auto *name:{"LLAMA_LINEAR_BABY_CACHE","LLAMA_LINEAR_PACKED_UPLOAD","LLAMA_LINEAR_GROUP_ACCUM",
        "LLAMA_LINEAR_PERSISTENT_PREP","LLAMA_PERIODIC_REDUCED_FASTPATH","LLAMA_WEIGHT_DECODE_FASTPATH",
        "LLAMA_FUSED_RELIN_RS2","LLAMA_LINEAR_GPU_ENCODE","LLAMA_LINEAR_GPU_PIPELINE",
        "LLAMA_OPERATOR_TIMING","LLAMA_LINEAR_PHASE_TIMING","LLAMA_SILU_RATIONAL_VR_SCALE","LLAMA_SILU_SIGN_GUARD_LOG2"})fixed(name,"0");
    fixed("LLAMA_SCALAR_CACHE","1");fixed("LLAMA_ADD_PREFIX","1");fixed("POSEIDON_NTT_ALGO","fourstep");
    fixed("POSEIDON_GPU_SSE_LOW_P","2");fixed("POSEIDON_RESCALE_X2","1");fixed("POSEIDON_MANUAL_BOOT_RANGE_EXPERIMENT","1");
    for(const auto *name:{"LLAMA_DIAG_DUMP_ACTIVE","LLAMA_PERIODIC_GPU_LIBRARY","POSEIDON_GPU_KEYLOAD_TRACE","POSEIDON_EVALMOD_QUOTIENT_MIN_LOG2"})
        if(const char *p=std::getenv(name);p&&*p&&std::string(p)!="0")throw std::invalid_argument(std::string("unsupported manual inherited ")+name);
    unsetenv("POSEIDON_EVALMOD_QUOTIENT_MIN_LOG2");
    // Precision is a constructor-frozen Api policy. Do not set an env flag and
    // present it as evidence that a consumer actually selected that branch.
}
void plan_cpu_test() {
    using namespace cerium_source::manual;
    for(auto mode:{Mode::HighPrecisionReal,Mode::HighPrecisionPairedSiluCandidate}) {
        Plan p{mode};BoundaryTrace trace(p);
        for(std::size_t i=0;i<boundaries.size();++i) {
            const auto &b=boundaries[i];trace.observe({b.name,b.tiles,b.q,std::exp2(40.),b.primary_boots,
                i>=12?p.secondary_boots():0,i>=12?p.projection_pairs():0,i>=5?16U:0U,0});
        }
        trace.finish();
    }
    std::cout<<"MANUAL_LLAMA_CPU_DONE plan_schema_only=true planned_Boots476_or364=true N65536=true no_Context_Runtime_keys_GPU=true"<<std::endl;
}
}

int main(int argc,char **argv)try {
    using namespace cerium_source;using namespace poseidon::gpu;
    std::cout<<std::setprecision(17);
    if(argc==2&&std::string(argv[1])=="--test-plan-cpu"){plan_cpu_test();return 0;}
    const auto input=manual::parse_inputs(argc,argv);manual::validate_manifest(input);
    const auto literal=parameters_from_fixture(input.payloads[0].c_str());require_cerium_parameters(literal);
    if(input.cpu_only) {
        std::cout<<"MANUAL_LLAMA_INPUTS_CPU_DONE input_version="<<manual::input_version_name(input.version)
            <<" model_weight_scope="<<manual::model_weight_scope(input.version)
            <<" pinned_manifest_declarations=true payload_hashing_external_preflight=true no_Context_Runtime_keys_GPU=true"<<std::endl;
        return 0;
    }
    configure_safe_implementation();Report report(input.report);const auto whole=Clock::now();
    llama_manual::SafetensorWeights weights(manual::model_weights_root(input),input.payloads[2]);weights.validate_model_index(input.layers);
    const auto embeddings=llama_manual::benchmark::read_tensor(input.payloads[3],"embeddings",0);
    const auto reference=llama_manual::benchmark::read_tensor(input.payloads[4],"hidden",input.layers);
    if(embeddings.size()!=128*4096||reference.size()!=128*4096)throw std::runtime_error("manual tensor geometry mismatch");
    gpu_check_cuda(cudaSetDevice(0),"manual model device0");auto *old=rmm::mr::get_current_device_resource();
    rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource> pool(old,64ULL<<20,64ULL<<30);
    rmm::mr::statistics_resource_adaptor<rmm::mr::device_memory_resource> memory(&pool);RestoreResource restore{old};
    rmm::mr::set_current_device_resource(&memory);
    const auto setup=Clock::now();Runtime runtime(literal,false,source_rotations());manual::Llama model(runtime,input.plan);
    std::vector<CT> state;
    for(std::size_t tile=0;tile<16;++tile)state.push_back(runtime.encrypt(layout::pack_tile(embeddings,4096,tile),6));
    auto mask=runtime.encrypt(layout::causal_mask(input.valid_tokens),27);
    gpu_check_cuda(cudaDeviceSynchronize(),"manual model encrypted inputs ready");
    std::size_t free=0,total=0;gpu_check_cuda(cudaMemGetInfo(&free,&total),"manual setup headroom");
    if(free<(8ULL<<30))throw std::runtime_error("manual setup headroom below8GiB");
    const auto setup_seconds=seconds(setup);double load_seconds=0,execution_seconds=0;
    std::cout<<"MANUAL_LLAMA_BEGIN plan="<<input.plan.name()<<" blocks="<<input.layers
        <<" input_version="<<manual::input_version_name(input.version)<<" model_weight_scope="<<manual::model_weight_scope(input.version)
        <<" N="<<runtime.context.parameters_literal()->degree()<<" slots="<<runtime.encoder.slot_count()
        <<" expected_Boots_per_block="<<input.plan.total_boots()<<" precision_policy=typed_integer_unit"
        <<" actual_seed_consumer_version="<<exp_unit_seed_consumer_version<<" model_qualification=false"<<std::endl;
    for(std::size_t layer=0;layer<input.layers;++layer) {
        const auto load=Clock::now();auto w=weights.load_layer(layer);load_seconds+=seconds(load);
        const auto begin=Clock::now();state=model.decoder_block(state,w,mask,layer);const auto wall=seconds(begin);execution_seconds+=wall;
        std::cout<<"MANUAL_LLAMA_BLOCK layer="<<layer<<" encrypted_execution_wall_seconds="<<wall
            <<" primary_Boots="<<model.primary_boots()<<" SiLU_Boots="<<model.silu_boots()
            <<" projection_pairs="<<model.projection_pairs()<<" selected_integer_seeds="<<model.integer_exp_seeds()<<std::endl;
    }
    // Only after ALL encrypted blocks. No hidden/plain tensor or key export.
    const auto final_decode=Clock::now();ComplexReferenceMetric metric;
    for(std::size_t tile=0;tile<16;++tile) {
        const auto values=runtime.decrypt_final(state[tile]);if(values.size()!=32768)throw std::runtime_error("manual final slot count");
        for(std::size_t token=0;token<128;++token)for(std::size_t feature=0;feature<256;++feature)
            metric.observe(values[token*256+feature],reference[token*4096+tile*256+feature],token<input.valid_tokens);
    }
    const auto accuracy=metric.finish();const auto decode_seconds=seconds(final_decode);const bool pass=accuracy.passed();
    std::ostringstream json;json<<std::setprecision(17)<<"{\n\"status\":\"completed_experimental_manual_model\",\"plan\":\""<<input.plan.name()
        <<"\",\"input_version\":\""<<manual::input_version_name(input.version)<<"\",\"model_weight_scope\":\""<<manual::model_weight_scope(input.version)
        <<"\",\"model_weight_layers_available\":"<<(input.version==manual::InputVersion::V1?1:32)
        <<",\"implementation\":\"handwritten_poseidon_api\",\"DSL_runtime\":false,\"N\":"<<runtime.context.parameters_literal()->degree()<<",\"complex_slots\":"<<runtime.encoder.slot_count()
        <<",\"blocks\":"<<model.completed_blocks()<<",\"valid_tokens\":38,\"capacity\":128,\"accuracy_gate\":true,\"maximum_complex_error\":"<<accuracy.maximum_complex_error
        <<",\"maximum_real_error\":"<<accuracy.maximum_real_error<<",\"maximum_imaginary\":"<<accuracy.maximum_imaginary<<",\"real_RMSE\":"<<accuracy.rmse
        <<",\"precision_passed\":"<<(pass?"true":"false")<<",\"all128_rows_finite\":true,\"production_model_qualification\":false,\"historical_pass_inherited\":false"
        <<",\"reference_kind\":\"frozen_Cerium_original_callpath_v2_float64_approx_packed_fixed_jitter_not_HF\",\"security_qualification\":false"
        <<",\"native_Cerium_full_FHE_plan_equal\":false,\"full32_component_domain_qualified\":false,\"precision_policy\":\"constructor_frozen_integer_unit\""
        <<",\"actual_integer_seed_calls\":"<<model.integer_exp_seeds()<<",\"actual_standard_seed_calls\":"<<model.standard_exp_seeds()
        <<",\"primary_ratio\":"<<model.primary_profile_numbers().ratio<<",\"SiLU_ratio\":"<<model.silu_profile_numbers().ratio
        <<",\"primary_m\":"<<model.primary_profile_numbers().m<<",\"SiLU_m\":"<<model.silu_profile_numbers().m
        <<",\"primary_eta\":"<<model.primary_profile_numbers().eta<<",\"SiLU_eta\":"<<model.silu_profile_numbers().eta
        <<",\"actual_Boot_output_q\":"<<model.actual_boot_output_q()<<",\"actual_Boot_output_scale\":"<<model.actual_boot_output_scale()
        <<",\"primary_Boot_calls\":"<<model.primary_boots()<<",\"SiLU_Boot_calls\":"<<model.silu_boots()
        <<",\"projection_pairs\":"<<model.projection_pairs()<<",\"boundary_metadata_and_consumed_counter_checks\":true,\"intermediate_decryption\":false"
        <<",\"setup_context_keys_profiles_inputs_seconds\":"<<setup_seconds<<",\"weight_load_seconds\":"<<load_seconds<<",\"encrypted_execution_wall_seconds\":"<<execution_seconds
        <<",\"execution_definition\":\"includes_CPU_diagonal_preparation_H2D_and_metadata_checks_excludes_weight_read_setup_final_decode\""
        <<",\"final_decode_seconds\":"<<decode_seconds<<",\"whole_seconds\":"<<seconds(whole)<<",\"GPU_busy_measured\":false,\"performance_qualification\":false"
        <<",\"rmm_live_peak_bytes\":"<<memory.get_bytes_counter().peak<<",\"report_contains_aggregates_only\":true,\"input_hash_evidence\":\"trusted_CPU_preflight_not_Cpp_rehash\"\n}\n";
    report.finish(json.str());std::cout<<"MANUAL_LLAMA_DONE precision_pass="<<pass<<" max_complex_error="<<accuracy.maximum_complex_error<<std::endl;
    return pass?0:2;
}catch(const std::exception &e){std::cerr<<"MANUAL_LLAMA_FAILED "<<e.what()<<std::endl;return 1;}
