// Real compiler HEVM -> native GPU evaluator. Metadata mode never touches GPU.
#define main strict_native_loader_probe_main
#include "main_strict_native_plan_probe.cpp"
#undef main
#include "strict_native_bootstrap_adapter.h"
#include "hevm_native_schedule.h"
#include "hevm_native_metadata_backend.h"

namespace strict_dsl {
using namespace poseidon::mgpu;
struct GpuBackend {
    const BaselineSession &s;
    const NativeHevmSchedule &schedule;
    const std::unordered_map<std::uint64_t,std::uint64_t> &rotation_levels;
    std::ifstream cst;
    std::vector<std::pair<std::uint64_t,std::uint64_t>> offsets;
    std::unordered_map<std::uint64_t,CT> ct,accumulate_reference;
    std::unordered_map<std::uint64_t,GpuPlaintextData> pt;
    std::optional<GpuGaloisKeysData> rotations;
    BootstrapOfflineCache bootstrap_cache;
    std::size_t boots=0,exact_plain=0,exact_mul=0,exact_accumulate=0;
    GpuBackend(const BaselineSession &session,const NativeHevmSchedule &program,
        const std::unordered_map<std::uint64_t,std::uint64_t> &levels,
        const char *constant_path,const std::string &input)
        :s(session),schedule(program),rotation_levels(levels),cst(constant_path,std::ios::binary) {
        auto word=[&](){std::uint64_t v=0;cst.read(reinterpret_cast<char*>(&v),8);if(!cst)throw std::runtime_error("truncated strict CST");return v;};
        auto count=word();std::uint64_t cursor=8;
        for(std::uint64_t i=0;i<count;++i){auto n=word();cursor+=8;offsets.emplace_back(cursor,n);cursor+=8*n;cst.seekg(cursor);}
        for(std::uint64_t i=0;i<27;++i){std::vector<double> slots(32768);
            for(std::size_t j=0;j<32768;++j)slots[j]=std::bit_cast<double>(strict_read_u64(input,8*(32768*i+j)));
            ct.emplace(i,s.encrypt_application_input(slots));}
    }
    double cipher_scale(std::uint64_t reg){return ct.at(reg).meta.scale;}
    void encode(std::uint64_t reg,std::uint64_t constant,std::uint64_t q,double scale) {
        const auto [offset,length]=offsets.at(constant);std::vector<double> values(length);
        cst.seekg(offset);cst.read(reinterpret_cast<char*>(values.data()),8*length);
        if(!cst)throw std::runtime_error("strict constant read failed");
        Plaintext plain;
        // Scalar encoding uses the original native scalar API (all slots).
        if(length==1)s.metadata.encoder.encode(values[0],s.metadata.ctx.crt_context()->parms_id_map().at(q-1),scale,plain);
        else s.metadata.encoder.encode(values,s.metadata.ctx.crt_context()->parms_id_map().at(q-1),scale,plain);
        pt.emplace(reg,GpuUploader::upload_plaintext(plain,0));
    }
    void rescale_group(const HevmNativeRescaleGroup &group) {
        CT out;execute_hevm_native_rescale_group(s.eval,ct.at(group.source),out,group);
        ct.emplace(group.destination,std::move(out));
    }
    auto next_action(std::uint64_t raw) const {
        auto it=std::lower_bound(schedule.actions.begin(),schedule.actions.end(),raw,
            [](const auto &a,auto index){return a.raw_index<index;});
        return it==schedule.actions.end()?it:std::next(it);
    }
    void prepare_rotations(std::uint64_t raw) {
        if(rotations)return;
        std::set<int> steps;std::set<std::size_t> levels;
        for(const auto &a:schedule.actions){if(a.raw_index<raw)continue;if(a.opcode==10)break;
            if(a.opcode==1){steps.insert(static_cast<int>(a.right));levels.insert(rotation_levels.at(a.raw_index));}}
        rotations.emplace(s.rotation_keys(std::vector<int>(steps.begin(),steps.end())));
        GpuUploader::prepare_key_views_for_q_counts(*rotations,std::vector<std::size_t>(levels.begin(),levels.end()));
    }
    void operation(const NativeHevmAction &a) {
        auto &left=ct.at(a.left);CT out;
        switch(a.opcode) {
        case 1:prepare_rotations(a.raw_index);s.eval.rotate(left,static_cast<int>(a.right),*rotations,out);break;
        case 2:s.eval.negate(left,out);break;
        case 4:s.eval.drop_modulus(left,out,s.metadata.ctx.crt_context()->parms_id_map().at(left.meta.q_count-a.right-1));break;
        case 6:{auto &right=ct.at(a.right);const auto saved=right.meta.scale;
            reconcile_hevm_native_scale(left,right);s.eval.add(left,right,out);right.meta.scale=saved;
            auto ref=accumulate_reference.find(a.right);
            if(ref!=accumulate_reference.end()) {require_exact_ciphertexts(s.metadata.ctx,out,ref->second,"strict native plain accumulate");
                ++exact_accumulate;accumulate_reference.erase(ref);}break;}
        case 7:s.eval.add_plain(left,pt.at(a.right),out);break;
        case 8:{auto &right=ct.at(a.right);CT raw,prefix;
            s.eval.multiply(left,right,raw);
            s.eval.multiply_q_prefix(left,right,prefix,left.meta.parms_id);
            require_exact_ciphertexts(s.metadata.ctx,raw,prefix,"strict native multiply prefix");++exact_mul;
            s.eval.relinearize(raw,s.keys,out);break;}
        case 9:{s.eval.multiply_plain(left,pt.at(a.right),out);CT prefix;
            s.eval.multiply_plain_q_prefix(left,pt.at(a.right),prefix);
            require_exact_ciphertexts(s.metadata.ctx,out,prefix,"strict native plain prefix");++exact_plain;
            auto next=next_action(a.raw_index);
            if(next!=schedule.actions.end() && next->opcode==6 && next->right==a.destination) {
                auto &destination=ct.at(next->left);CT fused;
                s.eval.drop_modulus(destination,fused,destination.meta.parms_id);
                s.eval.multiply_plain_accumulate(left,pt.at(a.right),fused);
                accumulate_reference.emplace(a.destination,std::move(fused));}break;}
        case 10:{gpu_check_cuda(cudaDeviceSynchronize(),"release strict Conv keys before bootstrap");rotations.reset();
            auto observer=decoded_values(s.replay,left);
            out=strict_native_bootstrap_only(s,left,observer,bootstrap_cache);++boots;
            std::cout<<"STRICT_DSL_BOOTSTRAP count="<<boots<<" q_in=6 q_out="<<out.meta.q_count
                <<" log_scale="<<std::log2(out.meta.scale)<<" mode=native_baseline_fold\n";break;}
        default:throw std::runtime_error("unsupported strict GPU action");
        }
        ct.emplace(a.destination,std::move(out));
    }
    void release_cipher(std::uint64_t reg){ct.erase(reg);}
    void release_plain(std::uint64_t reg){pt.erase(reg);}
};
}

int main(int argc,char **argv) {
    if(argc!=6)throw std::runtime_error("usage: --metadata-only|--unsafe-accuracy-only fixture HEVM CST inputs.f64");
    const std::string mode=argv[1];
    if(mode!="--metadata-only" && mode!="--unsafe-accuracy-only")throw std::runtime_error("invalid strict mode");
    // Reuse the successful real loader/encoder guard; it has no GPU calls.
    if(strict_native_loader_probe_main(5,argv+1))return 2;
    const auto binary=strict_read_file(argv[3]);
    const auto schedule=poseidon::mgpu::build_native_hevm_schedule(binary,
        strict_read_file(std::string(argv[3])+".native_scales"),
        strict_read_file(std::string(argv[3])+".native_encode_rules"),
        strict_read_file(std::string(argv[3])+".native_rescales"));
    ReluFixture fixture(argv[2]);
    std::ifstream fixture_stream(argv[2]);std::string version;std::size_t nq,np;fixture_stream>>version>>nq>>np;
    std::vector<std::uint64_t> primes(nq);for(auto &q:primes)fixture_stream>>q;
    strict_native_metadata::Backend metadata(primes);
    poseidon::mgpu::execute_native_hevm_schedule(schedule,metadata);
    if(metadata.boot!=18 || metadata.ct.size()!=1)throw std::runtime_error("strict metadata preflight failed");
    std::cout<<"STRICT_DSL_PREFLIGHT PASS actions="<<schedule.actions.size()<<" inputs=27 boots=18\n";
    if(mode=="--metadata-only")return 0;
    if(gpu_linear_transform_mode_from_environment(GpuLinearTransformMode::DoubleHoistBsgs)!=GpuLinearTransformMode::NoHoistBsgs)
        throw std::runtime_error("strict acceptance requires no-hoist environment");
    const auto input=strict_read_file(argv[5]);const auto oracle=load_application_network_fixture(0);
    std::string profile="q50-fixed45",dataset="grid";
    char *args[]={argv[0],argv[1],profile.data(),dataset.data(),argv[2]};
    return run_bootstrap_baseline(5,args,{}, {},nullptr,true,[&](const BaselineSession &s) {
        strict_dsl::GpuBackend backend(s,schedule,metadata.rotation_levels,argv[4],input);
        std::size_t limit=0;if(const char *raw=std::getenv("POSEIDON_DSL_MAX_OPS"))limit=std::stoull(raw);
        const auto done=poseidon::mgpu::execute_native_hevm_schedule(schedule,backend,limit);
        gpu_check_cuda(cudaDeviceSynchronize(),"strict HEVM completion");
        std::cout<<"STRICT_DSL_EXECUTION ops="<<done<<" bootstrap_calls="<<backend.boots
            <<" exact_plain="<<backend.exact_plain<<" exact_mul="<<backend.exact_mul
            <<" exact_accumulate="<<backend.exact_accumulate<<'\n';
        if(done<schedule.actions.size()){std::cout<<"DSL_TRUNCATED ops="<<done<<" bootstrap_calls="<<backend.boots<<'\n';return 0;}
        const auto decoded=decoded_values(s.replay,backend.ct.at(schedule.result_register));
        double worst=0;std::vector<double> logits;
        std::cout<<"STRICT_DSL_LOGITS";
        for(std::size_t i=0;i<10;++i){double x=decoded[i].real();logits.push_back(x);
            worst=std::max(worst,std::abs(x-oracle.reference_logits[i]));std::cout<<' '<<x;}
        auto argmax=[](const auto &values){return std::max_element(values.begin(),values.end())-values.begin();};
        std::cout<<"\nSTRICT_DSL_RESULT prediction="<<argmax(logits)
            <<" reference="<<argmax(oracle.reference_logits)<<" max_logit_error="<<worst<<'\n';
        return 0;
    });
}
