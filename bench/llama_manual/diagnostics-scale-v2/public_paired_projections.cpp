// Public synthetic, full-N, same-Context projection control. NOT a model or
// performance benchmark. No weights, old keys/PT/CT, saved arrays, or feedback.
#include "source_manual_boot.hpp"
namespace cerium_source { namespace mixed_block0 = manual_backend; }
#include "source_accuracy.hpp"
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <rmm/mr/device/statistics_resource_adaptor.hpp>
#include <array>
#include <cerrno>
#include <filesystem>
#include <fcntl.h>
#include <iomanip>
#include <sstream>
#include <unistd.h>

using namespace cerium_source;
using namespace poseidon::gpu;
namespace {
constexpr const char *fixture_sha="d9a79047125a6506c6b2d4c335737ccdb52a216f398a8150b133d47a44f72f56";
constexpr std::size_t active_slots=38*256;
struct RestoreResource {rmm::mr::device_memory_resource *old;~RestoreResource(){rmm::mr::set_current_device_resource(old);}};
class ExclusiveReport {
    int fd_=-1;
public:
    explicit ExclusiveReport(const std::filesystem::path &p) {
        fd_=::open(p.c_str(),O_CREAT|O_EXCL|O_WRONLY|O_CLOEXEC,0600);
        if(fd_<0)throw std::runtime_error("exclusive report creation failed");
    }
    ~ExclusiveReport(){if(fd_>=0)::close(fd_);}
    void finish(const std::string &text) {
        std::size_t at=0;
        while(at<text.size()) {
            const auto n=::write(fd_,text.data()+at,text.size()-at);
            if(n<0&&errno==EINTR)continue;
            if(n<=0)throw std::runtime_error("report write failed");
            at+=static_cast<std::size_t>(n);
        }
        if(::fsync(fd_))throw std::runtime_error("report fsync failed");
    }
};
bool same_meta(const GpuCiphertextMeta &a,const GpuCiphertextMeta &b) {
    return a.parms_id==b.parms_id&&a.scale==b.scale&&a.correction_factor==b.correction_factor&&
        a.is_ntt_form==b.is_ntt_form&&a.degree==b.degree&&a.q_count==b.q_count&&
        a.p_count==b.p_count&&a.component_count==b.component_count;
}
std::vector<GpuWord> words(const CT &x) {
    std::vector<GpuWord> out;
    for(const auto &poly:x.make_const_view().polys)for(const auto &shard:poly.shards) {
        const auto n=shard.limb_count*shard.coeff_count,at=out.size();out.resize(at+n);
        gpu_check_cuda(cudaMemcpy(out.data()+at,shard.ptr,n*sizeof(GpuWord),cudaMemcpyDeviceToHost),"projection const RNS snapshot");
    }
    if(out.size()!=x.meta.degree*x.meta.q_count*x.meta.component_count)
        throw std::runtime_error("projection RNS shape mismatch");
    return out;
}
struct Snapshot {GpuCiphertextMeta meta;std::vector<GpuWord> raw;};
Snapshot snapshot(const CT &x){return {x.meta,words(x)};}
bool same_snapshot(const Snapshot &a,const Snapshot &b){return same_meta(a.meta,b.meta)&&a.raw==b.raw;}
bool matches(const CT &x,const Snapshot &s){return same_meta(x.meta,s.meta)&&same_snapshot(snapshot(x),s);}
void shape(const Runtime &r,const CT &x,std::size_t q,double scale) {
    if(x.empty()||x.meta.degree!=65536||x.meta.q_count!=q||x.meta.p_count||x.meta.component_count!=2||
       !x.meta.is_ntt_form||x.meta.correction_factor!=1||x.meta.scale!=scale||
       x.meta.parms_id!=r.parameters.get_level_by_q_count(q).parms_id||x.fields_.size()!=1||x.fields_[0].device_id!=0)
        {
        std::ostringstream d;d<<std::setprecision(17)<<"projection actual N/Q/id/scale/components/device mismatch: {\"expected_q\":"<<q
          <<",\"expected_scale\":"<<scale<<",\"actual_scale\":"<<x.meta.scale<<",\"actual_log_scale\":"<<std::log2(x.meta.scale)
          <<",\"relative_scale_delta\":"<<(x.meta.scale/scale-1.)<<",\"N\":"<<x.meta.degree<<",\"Q\":"<<x.meta.q_count
          <<",\"P\":"<<x.meta.p_count<<",\"components\":"<<x.meta.component_count<<",\"actual_size\":"<<x.size()
          <<",\"NTT\":"<<x.meta.is_ntt_form<<",\"correction\":"<<x.meta.correction_factor
          <<",\"parms_match\":"<<(x.meta.parms_id==r.parameters.get_level_by_q_count(q).parms_id)
          <<",\"fields\":"<<x.fields_.size()<<",\"device\":"<<(x.fields_.empty()?-1:x.fields_[0].device_id)<<",\"empty\":"<<x.empty()<<"}";
        throw std::runtime_error(d.str());
    }
}
void configure() {
    auto set=[](const char *key,const char *wanted) {
        if(const char *v=std::getenv(key);v&&std::string(v)!=wanted&&std::string(v)!="0")
            throw std::invalid_argument(std::string("incompatible inherited ")+key);
        setenv(key,wanted,1);
    };
    for(const auto *key:{"LLAMA_LINEAR_GROUP_ACCUM","LLAMA_LINEAR_GPU_ENCODE","LLAMA_LINEAR_GPU_PIPELINE",
        "LLAMA_FUSED_RELIN_RS2","LLAMA_OPERATOR_TIMING","LLAMA_LINEAR_PHASE_TIMING"})set(key,"0");
    set("LLAMA_SCALAR_CACHE","1");set("LLAMA_ADD_PREFIX","1");
    set("POSEIDON_NTT_ALGO","fourstep");set("POSEIDON_GPU_SSE_LOW_P","2");
    set("POSEIDON_RESCALE_X2","1");set("POSEIDON_MANUAL_BOOT_RANGE_EXPERIMENT","1");
    for(const auto *key:{"LLAMA_DIAG_DUMP_ACTIVE","LLAMA_PERIODIC_GPU_LIBRARY","POSEIDON_GPU_KEYLOAD_TRACE"})
        if(const char *v=std::getenv(key);v&&*v&&std::string(v)!="0")throw std::invalid_argument(std::string("unsupported inherited ")+key);
    if(const char *v=std::getenv("POSEIDON_EVALMOD_QUOTIENT_MIN_LOG2");v&&std::string(v)!="0")
        throw std::invalid_argument("inherited quotient floor");
    unsetenv("POSEIDON_EVALMOD_QUOTIENT_MIN_LOG2");
}
std::string closed_output(const char *p) {
    const std::filesystem::path path(p);
    const auto name=path.filename().string();
    if(name.empty()||name=="."||name=="..")throw std::invalid_argument("report filename required");
    const auto result=(std::filesystem::canonical(path.parent_path())/name).string();
    if(result.find_first_of(" \t\r\n")!=std::string::npos)throw std::invalid_argument("report whitespace prohibited");
    if(std::filesystem::exists(result)||std::filesystem::is_symlink(result))throw std::runtime_error("report already exists or is a symlink");
    return result;
}
void manifest(const char *path,const char *fixture,const std::string &report) {
    std::ifstream in(path);std::string magic;
    if(!std::getline(in,magic)||magic!="PUBLIC_PAIRED_PROJECTIONS_INPUT_V1")
        throw std::runtime_error("public CPU fixture hash manifest required");
    std::string role,p,d,extra,line;
    if(!std::getline(in,line))throw std::runtime_error("missing fixture manifest row");
    std::istringstream row(line);
    if(!(row>>role>>p>>d)||(row>>extra)||role!="fixture"||d!=fixture_sha||p!=std::filesystem::canonical(fixture).string())
        throw std::runtime_error("pinned public fixture path/digest mismatch");
    if(!std::getline(in,line))throw std::runtime_error("missing report manifest row");
    std::istringstream reportrow(line);
    if(!(reportrow>>role>>p>>d)||(reportrow>>extra)||role!="report"||d!="-"||p!=report||std::getline(in,line))
        throw std::runtime_error("closed public report manifest mismatch");
    // Trusted CPU preflight hashes fixture bytes; this validates its pinned
    // manifest and paths, not a second hash. Payload remains frozen thereafter.
}
struct Case {const char *name;double amplitude;unsigned mode;};
const std::array<Case,10> cases{{
    {"a_zero_small",.1,0},{"zero_b_small",.1,1},{"a_b_small",.1,2},{"b_a_small",.1,3},{"a_minus_b_small",.1,4},
    {"a_zero_31",31.,0},{"zero_b_31",31.,1},{"a_b_31",31.,2},{"b_a_31",31.,3},{"a_minus_b_31",31.,4}}};
std::pair<std::vector<double>,std::vector<double>> patterns(const Case &c) {
    std::vector<double> a(layout::slots),b(layout::slots);
    const double pi=std::acos(-1.);
    for(std::size_t i=0;i<layout::slots;++i) {
        const double x=c.amplitude*double(int((37*i+11)%257)-128)/128.;
        const double y=c.amplitude*std::sin(2*pi*13.*double(i)/double(layout::slots));
        a[i]=c.mode==1?0.:(c.mode==3?y:x);
        b[i]=c.mode==0?0.:(c.mode==3?x:(c.mode==4?-y:y));
    }
    return {std::move(a),std::move(b)};
}
struct Errors {ComplexReferenceSummary active,full,sine;};
struct Gates {
    bool active=true,full=true,sine=true;
    bool all() const {return active&&full&&sine;}
};
Gates fixed_gates(const Errors &a,const Errors &b,const Errors &c,const Errors &d) {
    Gates g;
    for(const auto *e:{&a,&b,&c,&d}){g.active&=e->active.passed();g.full&=e->full.passed();g.sine&=e->sine.passed();}
    return g;
}
void cpu_tests() {
    layout::require_ring(16,15);
    if(cases.size()!=10||cases.size()*4!=40||active_slots!=9728)throw std::runtime_error("projection closed counts");
    for(const auto &c:cases) {
        const auto [a,b]=patterns(c);
        if(a.size()!=32768||b.size()!=32768)throw std::runtime_error("public full-slot shape");
        for(std::size_t i=0;i<a.size();++i)
            if(!std::isfinite(a[i])||!std::isfinite(b[i])||std::abs(a[i])>c.amplitude||std::abs(b[i])>c.amplitude)
                throw std::runtime_error("public fixed range");
    }
    for(unsigned group=0;group<2;++group) {
        const auto ab=patterns(cases[group*5+2]),a0=patterns(cases[group*5]),zero_b=patterns(cases[group*5+1]);
        const auto ba=patterns(cases[group*5+3]),negative=patterns(cases[group*5+4]);
        bool nonzero_a=false,nonzero_b=false,different=false;
        for(std::size_t i=0;i<32768;++i) {
            if(a0.first[i]!=ab.first[i]||a0.second[i]!=0||zero_b.first[i]!=0||zero_b.second[i]!=ab.second[i]||
               ba.first[i]!=ab.second[i]||ba.second[i]!=ab.first[i]||negative.first[i]!=ab.first[i]||negative.second[i]!=-ab.second[i])
                throw std::runtime_error("public zero/swap/sign pattern regression");
            nonzero_a|=ab.first[i]!=0;nonzero_b|=ab.second[i]!=0;different|=ab.first[i]!=ab.second[i];
        }
        if(!nonzero_a||!nonzero_b||!different)throw std::runtime_error("degenerate public patterns");
    }
    const auto small=patterns(cases[2]),large=patterns(cases[7]);
    for(std::size_t i=0;i<32768;++i)
        if(std::abs(large.first[i]-310*small.first[i])>1e-12||std::abs(large.second[i]-310*small.second[i])>1e-12)
            throw std::runtime_error("public amplitude relation regression");
    ComplexReferenceMetric n;n.observe({8e-4,8e-4},0,true);
    if(n.finish().passed())throw std::runtime_error("separate-threshold regression");
    ComplexReferenceMetric p;p.observe({7e-4,7e-4},0,true);
    if(!p.finish().passed())throw std::runtime_error("complex norm gate regression");
    auto m=GpuCiphertextMeta{};m.degree=65536;m.q_count=6;m.component_count=2;m.scale=std::exp2(40.);m.is_ntt_form=true;
    if(!same_meta(m,m))throw std::runtime_error("metadata comparison");
    for(unsigned field=0;field<8;++field) {
        auto changed=m;
        switch(field){case 0:changed.parms_id[0]^=1;break;case 1:changed.scale*=2;break;case 2:changed.correction_factor=2;break;
          case 3:changed.is_ntt_form=false;break;case 4:changed.degree*=2;break;case 5:++changed.q_count;break;
          case 6:changed.p_count=1;break;default:changed.component_count=3;}
        if(same_meta(m,changed))throw std::runtime_error("metadata mutation ignored");
    }
    const Snapshot frozen{m,{11,22,33}};
    auto packing_mutation=frozen;packing_mutation.raw[1]^=1;
    auto control_mutation=frozen;control_mutation.raw[0]^=1;
    if(same_snapshot(packing_mutation,frozen)||same_snapshot(control_mutation,frozen))throw std::runtime_error("pre-operation baseline mutation ignored");
    ComplexReferenceSummary valid;valid.active_entries=1;
    Errors clean{valid,valid,valid},bad=clean;
    if(!fixed_gates(clean,clean,clean,clean).all())throw std::runtime_error("clean gate failed");
    bad.full.maximum_complex_error=.002;auto gates=fixed_gates(clean,bad,clean,clean);
    if(!gates.active||gates.full||gates.all())throw std::runtime_error("inactive/full failure omitted from fixed gates");
    bad=clean;bad.sine.maximum_complex_error=.002;gates=fixed_gates(clean,clean,clean,bad);
    if(gates.sine||gates.all())throw std::runtime_error("sine failure omitted from fixed gates");
    char temporary[]="/tmp/llama-public-proj-cpu-XXXXXX";const auto directory=::mkdtemp(temporary);
    if(!directory)throw std::runtime_error("owned public CPU temp directory failed");
    const std::filesystem::path temp(directory),file=temp/"report.json",link=temp/"dangling.json";
    {ExclusiveReport exclusive(file);exclusive.finish("{\"cpu_only\":true}\n");}
    bool duplicate_rejected=false;try{ExclusiveReport duplicate(file);}catch(const std::runtime_error&){duplicate_rejected=true;}
    std::ifstream old(file);std::stringstream content;content<<old.rdbuf();old.close();
    if(!duplicate_rejected||content.str()!="{\"cpu_only\":true}\n")throw std::runtime_error("exclusive report overwrite regression");
    std::filesystem::create_symlink(temp/"missing-target",link);
    bool link_rejected=false;try{ExclusiveReport dangling(link);}catch(const std::runtime_error&){link_rejected=true;}
    bool path_rejected=false;try{(void)closed_output(link.c_str());}catch(const std::runtime_error&){path_rejected=true;}
    if(!link_rejected||!path_rejected||std::filesystem::exists(temp/"missing-target"))throw std::runtime_error("report symlink regression");
    std::filesystem::remove_all(temp); // only this mkdtemp-owned public test directory
    std::cout<<"PUBLIC_PROJECTIONS_CPU_DONE cases=10 physical_Boots=40 N=65536 slots=32768 pattern_zero_swap_sign_amplitude=true metadata8_word_mutations=true exclusive_report_symlink=true fixed_full_and_sine_gates=true active38_complex_norm=true no_Runtime_keys_GPU=true"<<std::endl;
}
Errors evaluate(Runtime &r,const CT &x,const std::vector<double> &expected,long double rfactor,long double eta,long double qdiff) {
    const auto before=snapshot(x);const auto decoded=r.decrypt_final(x);
    if(decoded.size()!=32768||expected.size()!=32768)throw std::runtime_error("decoded full-slot shape");
    ComplexReferenceMetric active,full,sine;
    const long double factor=rfactor*qdiff/(2*std::acos(-1.L)*eta);
    for(std::size_t i=0;i<decoded.size();++i) {
        const auto reference=static_cast<double>(factor*std::sin(static_cast<long double>(expected[i])/factor));
        active.observe(decoded[i],expected[i],i<active_slots);full.observe(decoded[i],expected[i],true);
        sine.observe(decoded[i],reference,true);
    }
    if(!matches(x,before))throw std::runtime_error("const output decode mutated metadata/RNS");
    return {active.finish(),full.finish(),sine.finish()};
}
std::string json_errors(const Errors &e) {
    std::ostringstream s;s<<std::setprecision(17)<<"{\"active38_max_complex_identity\":"<<e.active.maximum_complex_error
      <<",\"allslot_max_complex_identity\":"<<e.full.maximum_complex_error<<",\"allslot_max_complex_sine\":"<<e.sine.maximum_complex_error
      <<",\"allslot_max_imaginary\":"<<e.full.maximum_imaginary<<",\"active38_real_RMSE\":"<<e.active.rmse
      <<",\"active38_identity_pass\":"<<(e.active.passed()?"true":"false")<<"}";return s.str();
}
}
#ifndef LLAMA_PUBLIC_PROJECTION_HELPERS_ONLY
int main(int argc,char **argv)try {
    std::cout<<std::setprecision(17);
    if(argc==2&&std::string(argv[1])=="--test-cpu"){cpu_tests();return 0;}
    const bool inputs_cpu=argc==5&&std::string(argv[1])=="--test-inputs-cpu";
    if(argc!=5||(!inputs_cpu&&std::string(argv[1])!="--public-projections-v1"))
        throw std::invalid_argument("closed public-projections-v1 or test-inputs-cpu fixture manifest report CLI required");
    configure();cpu_tests();const auto report_path=closed_output(argv[4]);manifest(argv[3],argv[2],report_path);
    const auto literal=parameters_from_fixture(argv[2]);require_cerium_parameters(literal);
    if(inputs_cpu) {
        std::cout<<"PUBLIC_PROJECTIONS_INPUTS_CPU_DONE pinned_public_fixture_manifest=true N65536=true no_Context_Runtime_keys_GPU=true"<<std::endl;
        return 0;
    }
    ExclusiveReport report(report_path);
    mixed_block0::SubmissionOwner submission;submission.check();
    gpu_check_cuda(cudaSetDevice(0),"public projections device0");
    auto *old=rmm::mr::get_current_device_resource();
    rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource> pool(old,64ULL<<20,64ULL<<30);
    rmm::mr::statistics_resource_adaptor<rmm::mr::device_memory_resource> memory(&pool);RestoreResource restore{old};
    rmm::mr::set_current_device_resource(&memory);
    const auto setup=Clock::now();Runtime runtime(literal,false,{0});
    mixed_block0::validate_key(runtime.relin);mixed_block0::validate_key(runtime.galois);
    mixed_block0::BoundApi owner(runtime,submission,mixed_block0::make_profile(runtime,65536,submission),0);
    auto &api=owner.get();owner.verify();
    gpu_check_cuda(cudaDeviceSynchronize(),"public projection profile ready");
    std::size_t free=0,total=0;gpu_check_cuda(cudaMemGetInfo(&free,&total),"public projection setup headroom");
    if(free<(8ULL<<30))throw std::runtime_error("public projection setup headroom below8GiB");
    const auto setup_seconds=seconds(setup);
    const auto q=static_cast<long double>(literal.q()[0].value())*literal.q()[1].value();
    const long double qdiff=q/std::exp2(std::round(std::log2(q)));
    const auto config=owner.config();
    std::cout<<"PUBLIC_PROJECTIONS_SETUP {\"N\":"<<runtime.context.parameters_literal()->degree()<<",\"complex_slots\":"<<runtime.encoder.slot_count()
      <<",\"ratio\":65536,\"m\":"<<config.m<<",\"eta\":"<<config.eta<<",\"qdiff\":"<<qdiff
      <<",\"native_q\":27,\"output_log_scale\":50,\"runtime_profile_setup_seconds\":"<<setup_seconds
      <<",\"device_free_bytes\":"<<free<<",\"pool_cap_bytes\":68719476736,\"performance_valid\":false}"<<std::endl;
    bool numerical_pass=true;std::vector<std::string> rows;
    for(const auto &c:cases) {
        submission.check();owner.verify();std::cout<<"PUBLIC_PROJECTIONS_CASE_BEGIN name="<<c.name<<std::endl;
        const auto [a,b]=patterns(c);
        auto ca=runtime.encrypt(a,6),cb=runtime.encrypt(b,6);shape(runtime,ca,6,std::exp2(40.));shape(runtime,cb,6,std::exp2(40.));
        gpu_check_cuda(cudaDeviceSynchronize(),"public encrypted inputs ready before packing");
        const auto sa=snapshot(ca),sb=snapshot(cb);
        auto packed=api.pack_independent_reals(ca,cb);shape(runtime,packed.carrier(),6,std::exp2(40.));
        gpu_check_cuda(cudaDeviceSynchronize(),"public packed carrier ready");
        if(!matches(ca,sa)||!matches(cb,sb))throw std::runtime_error("packing mutated original input metadata/RNS");
        const auto sp=snapshot(packed.carrier());
        const auto before=api.boot_calls,pairs_before=api.manual_projection_pairs;const auto wall=Clock::now();
        auto legacy=api.source_bootstrap(packed.carrier(),"public:legacy_packed_real_control");
        gpu_check_cuda(cudaDeviceSynchronize(),"legacy real control ready before candidate");
        shape(runtime,legacy,27,std::exp2(50.));
        const auto old_output=snapshot(legacy); // before any candidate/control can touch it
        auto pair=api.source_bootstrap_projections_candidate(packed);
        auto separate_a=api.source_bootstrap(ca,"public:separate_a");
        auto separate_b=api.source_bootstrap(cb,"public:separate_b");
        gpu_check_cuda(cudaDeviceSynchronize(),"public four fixed Boot calls complete");
        const auto four_boot_seconds=seconds(wall);
        if(api.boot_calls-before!=4||api.manual_projection_pairs-pairs_before!=1)throw std::runtime_error("actual physical Boot/projection count mismatch");
        for(const auto *out:{&legacy,&pair.real,&pair.imaginary,&separate_a,&separate_b})shape(runtime,*out,27,std::exp2(50.));
        if(!matches(legacy,old_output)||!matches(pair.real,old_output))
            throw std::runtime_error("old real changed or same-packed-input old/new real metadata/RNS not exact");
        if(pair.real.fields_[0].data()==pair.imaginary.fields_[0].data()||
           legacy.fields_[0].data()==pair.real.fields_[0].data()||legacy.fields_[0].data()==pair.imaginary.fields_[0].data())
            throw std::runtime_error("owned candidate/legacy outputs alias");
        const auto ea=evaluate(runtime,pair.real,a,65536,config.eta,qdiff);
        const auto eb=evaluate(runtime,pair.imaginary,b,65536,config.eta,qdiff);
        const auto control_a=evaluate(runtime,separate_a,a,65536,config.eta,qdiff);
        const auto control_b=evaluate(runtime,separate_b,b,65536,config.eta,qdiff);
        // Legacy decode is also checked, without using it as any encrypted input.
        const auto legacy_error=evaluate(runtime,legacy,a,65536,config.eta,qdiff);(void)legacy_error;
        if(!matches(legacy,old_output)||!matches(ca,sa)||!matches(cb,sb)||!matches(packed.carrier(),sp))
            throw std::runtime_error("input/carrier/old real changed during candidate/control/decode");
        owner.verify();
        const auto gates=fixed_gates(ea,eb,control_a,control_b);
        const bool active_pass=gates.active,full_pass=gates.full,sine_pass=gates.sine,pass=gates.all();numerical_pass&=pass;
        std::ostringstream row;row<<std::setprecision(17)<<"{\"name\":\""<<c.name<<"\",\"public_amplitude\":"<<c.amplitude
          <<",\"physical_Boots\":4,\"derived_internal_EvalMod_calls\":8,\"same_packed_real_metadata_RNS_exact\":true"
          <<",\"input_carrier_and_const_decodes_immutable\":true,\"all32768slots_finite\":true,\"owned_outputs_nonaliasing\":true"
          <<",\"diagnostic_wall_includes_control_RNS_snapshot\":true,\"four_Boot_diagnostic_wall_seconds\":"<<four_boot_seconds<<",\"pair_real\":"<<json_errors(ea)
          <<",\"pair_imaginary_projection\":"<<json_errors(eb)<<",\"separate_a\":"<<json_errors(control_a)<<",\"separate_b\":"<<json_errors(control_b)
          <<",\"active38_complex_gate_pass\":"<<(active_pass?"true":"false")
          <<",\"fullslot_identity_complex_gate_pass\":"<<(full_pass?"true":"false")
          <<",\"fullslot_sine_complex_gate_pass\":"<<(sine_pass?"true":"false")
          <<",\"all_fixed_gates_pass\":"<<(pass?"true":"false")<<"}";
        rows.push_back(row.str());std::cout<<"PUBLIC_PROJECTIONS_CASE "<<row.str()<<std::endl;
    }
    if(api.boot_calls!=40||api.manual_projection_pairs!=10||rows.size()!=10)throw std::runtime_error("closed final counts failed");
    std::ostringstream final;final<<std::setprecision(17)<<"{\n\"status\":\"completed_public_synthetic_control\",\n\"N\":65536,\"complex_slots\":32768,\"active_tokens\":38,\"case_count\":10,\"physical_Boots\":40,\"projection_pairs\":10,\"derived_internal_EvalMod_calls\":80"
      <<",\n\"runtime_profile_setup_seconds\":"<<setup_seconds<<",\"rmm_live_peak_bytes\":"<<memory.get_bytes_counter().peak
      <<",\"same_Context_key_inputs_for_legacy_real_control\":true,\"different_refresh_plans_not_cross_RNS_assertion\":true"
      <<",\"EvalMod_count_evidence\":\"derived_two_branches_per_completed_physical_Boot_not_independent_counter\""
      <<",\"active38_complex_threshold\":0.001,\"fullslot_public_identity_threshold\":0.001,\"fullslot_public_sine_threshold\":0.001,\"all_numerical_gates_passed\":"<<(numerical_pass?"true":"false")
      <<",\"performance_valid\":false,\"model_qualification\":false,\"cases\":[";
    for(std::size_t i=0;i<rows.size();++i)final<<(i?",\n":"")<<rows[i];final<<"]\n}\n";report.finish(final.str());
    std::cout<<"PUBLIC_PROJECTIONS_DONE numerical_pass="<<numerical_pass<<" physical_Boots="<<api.boot_calls
      <<" no_model_weights_or_saved_keys_PT_CT_arrays=true"<<std::endl;
    return numerical_pass?0:2;
}catch(const std::exception &e){std::cerr<<"PUBLIC_PROJECTIONS_FAILED "<<e.what()<<'\n';return 1;}
#endif // LLAMA_PUBLIC_PROJECTION_HELPERS_ONLY
