// Separate NEXT-generation complete paired-SiLU control. NOT authorized to
// build/run remotely by this source; no model/private arrays or saved keys.
// Reuse only the reviewed public fixture/metadata/report/metric helpers.
#define LLAMA_PUBLIC_PROJECTION_HELPERS_ONLY
#include "public_paired_projections.cpp"
#include "source_manual_pair_silu.hpp"
#include "source_silu_cpu.hpp"
namespace {
using Z=silu_cpu_reference::Z;
using Decoded=std::vector<std::complex<double>>;
struct PublicSiluInputs {std::array<std::vector<double>,2> gate,up;};
PublicSiluInputs silu_patterns() {
    PublicSiluInputs p;for(auto &x:p.gate)x.resize(32768);for(auto &x:p.up)x.resize(32768);
    const auto pi=std::acos(-1.);
    for(std::size_t j=0;j<32768;++j) {
        const double dyadic=25.*double(int((37*j+11)%257)-128)/128.;
        p.gate[0][j]=j%8==1?-8.:(j%8==3?-4.:(j%8==5?4.:dyadic));
        p.gate[1][j]=25.*std::sin(2*pi*17.*double(j)/32768.);
        p.up[0][j]=8.*std::cos(2*pi*13.*double(j)/32768.);
        p.up[1][j]=8.*double(int((29*j+7)%257)-128)/128.;
    }
    return p;
}
Z finish_source(Z v,Z w,const std::array<Z,3> &s) {
    const auto v2=v*v,v4=v2*v2,v6=v2*v4;
    const auto P=(static_cast<long double>(-.0055465625580307)*v2+static_cast<long double>(-.0819767021525476)*v+static_cast<long double>(-.3067541139982155))/2.L;
    const auto Q=(static_cast<long double>(.0002743776353465)*v6+static_cast<long double>(-.011113046708173)*v4+static_cast<long double>(.2281430841728270)*v2+.5L*v+static_cast<long double>(.0085064025895951))/2.L;
    return w*((s[0]-s[1])*P+(s[1]-s[2])*Q+(s[2]+1.L)*v/2.L);
}
Z f_after_refresh(Z x) {
    constexpr std::array<double,4> f{35./16.,-35./16.,21./16.,-5./16.};
    return silu_cpu_reference::odd_project(silu_cpu_reference::odd_project(x,f),f);
}
std::complex<double> narrow(Z x) {
    const auto re=static_cast<double>(x.real()),im=static_cast<double>(x.imag());
    if(!std::isfinite(x.real())||!std::isfinite(x.imag())||!std::isfinite(re)||!std::isfinite(im))
        throw std::runtime_error("nonfinite/unrepresentable public source oracle");
    return {re,im};
}
Decoded decode_const(Runtime &r,const CT &x) {
    gpu_check_cuda(cudaDeviceSynchronize(),"paired SiLU const decode barrier");const auto frozen=snapshot(x);
    auto v=r.decrypt_final(x);
    if(v.size()!=32768||!matches(x,frozen))throw std::runtime_error("paired SiLU const decode shape/immutability");
    for(const auto z:v)if(!std::isfinite(z.real())||!std::isfinite(z.imag()))throw std::runtime_error("paired SiLU nonfinite decode");
    return v;
}
struct SourceErrors {ComplexReferenceSummary active,full,gate_conditioned,all_refresh_conditioned;bool conditioned=false;};
std::string source_errors_json(const SourceErrors &e) {
    std::ostringstream s;s<<std::setprecision(17)<<"{\"active38_max_complex_source\":"<<e.active.maximum_complex_error
      <<",\"fullslot_max_complex_source\":"<<e.full.maximum_complex_error<<",\"max_imaginary\":"<<e.full.maximum_imaginary
      <<",\"post_Gate_refresh_decodedUp_conditioned\":";
    if(e.conditioned)s<<e.gate_conditioned.maximum_complex_error;else s<<"null";
    s<<",\"post_all_refresh_decodedUp_conditioned\":";
    if(e.conditioned)s<<e.all_refresh_conditioned.maximum_complex_error;else s<<"null";
    s<<",\"active38_source_pass\":"<<(e.active.passed()?"true":"false")<<",\"fullslot_source_pass\":"<<(e.full.passed()?"true":"false")<<"}";return s.str();
}
class RefreshTrace {
    Runtime &r;const long double factor;
    std::array<Snapshot,2> input;Snapshot carrier;
    std::size_t before_count=0,packed_count=0,after_count=0;
public:
    std::array<std::array<Decoded,2>,4> outputs;
    std::vector<std::string> rows;
    bool numerical_pass=true;
    RefreshTrace(Runtime &runtime,long double eta,long double qdiff):r(runtime),factor(65536.L*qdiff/(2*std::acos(-1.L)*eta)){}
    ManualSiluRefreshObserver observer() {
        ManualSiluRefreshObserver o;
        o.before=[this](std::size_t n,const CT &a,const CT &b) {
            if(n!=before_count++||packed_count!=n||after_count!=n)throw std::runtime_error("refresh before ordinal");
            shape(r,a,6,std::exp2(40.));shape(r,b,6,std::exp2(40.));
            gpu_check_cuda(cudaDeviceSynchronize(),"paired SiLU BEFORE PACK baseline barrier");
            input={snapshot(a),snapshot(b)};
        };
        o.packed=[this](std::size_t n,const CT &a,const CT &b,const CT &p) {
            gpu_check_cuda(cudaDeviceSynchronize(),"paired SiLU immediate AFTER PACK barrier");
            if(n!=packed_count++||before_count!=n+1||after_count!=n||!matches(a,input[0])||!matches(b,input[1]))
                throw std::runtime_error("refresh packing mutated input/ordinal");
            shape(r,p,6,std::exp2(40.));carrier=snapshot(p);
        };
        o.after=[this](std::size_t n,const CT &a,const CT &b,const CT &p,const CT &x,const CT &y) {
            gpu_check_cuda(cudaDeviceSynchronize(),"paired SiLU AFTER BOOT barrier");
            if(n!=after_count++||before_count!=n+1||packed_count!=n+1||!matches(a,input[0])||!matches(b,input[1])||!matches(p,carrier))
                throw std::runtime_error("refresh input/carrier mutation or ordinal");
            shape(r,x,27,std::exp2(50.));shape(r,y,27,std::exp2(50.));
            const std::array<const CT*,5> storage{&a,&b,&p,&x,&y};
            for(std::size_t i=0;i<storage.size();++i)for(std::size_t j=i+1;j<storage.size();++j)
                if(storage[i]->fields_[0].data()==storage[j]->fields_[0].data())throw std::runtime_error("refresh storage aliases");
            const auto xa=snapshot(x),ya=snapshot(y);const auto da=decode_const(r,a),db=decode_const(r,b);
            outputs[n]={decode_const(r,x),decode_const(r,y)};
            for(unsigned lane=0;lane<2;++lane) {
                const auto &expected=lane?db:da;ComplexReferenceMetric identity,sine;
                for(std::size_t j=0;j<32768;++j) {
                    identity.observe(outputs[n][lane][j],expected[j],true);
                    sine.observe(outputs[n][lane][j],static_cast<double>(factor*std::sin(expected[j].real()/factor)),true);
                }
                const auto ie=identity.finish(),se=sine.finish();numerical_pass&=ie.passed()&&se.passed();
                std::ostringstream row;row<<std::setprecision(17)<<"{\"ordinal\":"<<n<<",\"lane\":"<<lane
                  <<",\"fullslot_identity_max_complex\":"<<ie.maximum_complex_error<<",\"fullslot_real_sine_max_complex\":"<<se.maximum_complex_error
                  <<",\"Q_in\":6,\"input_log_scale\":40,\"Q_out\":27,\"output_log_scale\":50,\"immutable\":true}";
                rows.push_back(row.str());std::cout<<"PUBLIC_SILU_PAIR_REFRESH "<<row.str()<<std::endl;
            }
            if(!matches(a,input[0])||!matches(b,input[1])||!matches(p,carrier)||!matches(x,xa)||!matches(y,ya))
                throw std::runtime_error("refresh observer changed metadata/RNS");
        };return o;
    }
    void finish() const {if(before_count!=4||packed_count!=4||after_count!=4||rows.size()!=8)throw std::runtime_error("closed refresh trace");}
};
void silu_pair_cpu_tests() {
    const auto p=silu_patterns();bool different=false;
    for(std::size_t j=0;j<32768;++j)for(unsigned lane=0;lane<2;++lane) {
        const auto v=p.gate[lane][j],w=p.up[lane][j];
        if(!std::isfinite(v)||!std::isfinite(w)||std::abs(v)>25||std::abs(w)>8)throw std::runtime_error("fixed public SiLU range");
        const Z vr=Z(v,0)/50.L;const std::array<Z,3>s{silu_cpu_reference::sign(vr+8.L/50),silu_cpu_reference::sign(vr+4.L/50),silu_cpu_reference::sign(vr-4.L/50)};
        if(std::abs(finish_source({v,0},{w,0},s)-silu_cpu_reference::apply({v,0},{w,0}))>1e-12L)
            throw std::runtime_error("conditioned source oracle recurrence regression");
        (void)narrow(silu_cpu_reference::apply({v,0},{w,0}));different|=p.gate[0][j]!=p.gate[1][j];
    }
    if(!different)throw std::runtime_error("degenerate SiLU lanes");
    std::cout<<"PUBLIC_SILU_PAIR_CPU_DONE cases=2 planned_physical_Boots=32 planned_projection_pairs=16 fixed_gate25_up8=true no_Runtime_keys_GPU=true"<<std::endl;
}
SourceErrors source_errors(Runtime &r,const CT &out,const std::vector<double> &gate,const std::vector<double> &up,
    const Decoded &decoded_up,const RefreshTrace *trace,unsigned lane) {
    const auto actual=decode_const(r,out);ComplexReferenceMetric active,full,conditioned,all;
    for(std::size_t j=0;j<32768;++j) {
        const auto nominal=silu_cpu_reference::apply({gate[j],0},{up[j],0});
        active.observe(actual[j],narrow(nominal),j<active_slots);full.observe(actual[j],narrow(nominal),true);
        if(trace) {
            const auto v=Z(trace->outputs[0][lane][j].real(),trace->outputs[0][lane][j].imag()),w=Z(decoded_up[j].real(),decoded_up[j].imag());
            conditioned.observe(actual[j],narrow(silu_cpu_reference::apply(v,w)),true);
            std::array<Z,3> signs;
            for(unsigned b=0;b<3;++b) {const auto z=trace->outputs[b+1][lane][j];signs[b]=f_after_refresh({z.real(),z.imag()});}
            all.observe(actual[j],narrow(finish_source(v,w,signs)),true);
        } else {conditioned.observe(actual[j],narrow(nominal),true);all.observe(actual[j],narrow(nominal),true);}
    }
    return {active.finish(),full.finish(),conditioned.finish(),all.finish(),trace!=nullptr};
}
}
int main(int argc,char **argv)try {
    std::cout<<std::setprecision(17);
    if(argc==2&&std::string(argv[1])=="--test-cpu"){cpu_tests();silu_pair_cpu_tests();return 0;}
    const bool cpu=argc==5&&std::string(argv[1])=="--test-inputs-cpu";
    if(argc!=5||(!cpu&&std::string(argv[1])!="--public-silu-pair-v1"))throw std::invalid_argument("closed public SiLU pair CLI required");
    configure();cpu_tests();silu_pair_cpu_tests();const auto report_path=closed_output(argv[4]);manifest(argv[3],argv[2],report_path);
    const auto literal=parameters_from_fixture(argv[2]);require_cerium_parameters(literal);
    if(cpu){std::cout<<"PUBLIC_SILU_PAIR_INPUTS_CPU_DONE pinned_public_fixture_manifest=true no_Context_Runtime_keys_GPU=true"<<std::endl;return 0;}
    ExclusiveReport report(report_path);mixed_block0::SubmissionOwner submission;submission.check();
    gpu_check_cuda(cudaSetDevice(0),"public SiLU pair device0");auto *old=rmm::mr::get_current_device_resource();
    rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource> pool(old,64ULL<<20,64ULL<<30);
    rmm::mr::statistics_resource_adaptor<rmm::mr::device_memory_resource> memory(&pool);RestoreResource restore{old};rmm::mr::set_current_device_resource(&memory);
    const auto setup=Clock::now();Runtime runtime(literal,false,{0});mixed_block0::validate_key(runtime.relin);mixed_block0::validate_key(runtime.galois);
    mixed_block0::BoundApi owner(runtime,submission,mixed_block0::make_profile(runtime,65536,submission),0);auto &api=owner.get();owner.verify();
    gpu_check_cuda(cudaDeviceSynchronize(),"paired SiLU profile ready");std::size_t free=0,total=0;gpu_check_cuda(cudaMemGetInfo(&free,&total),"paired SiLU headroom");
    if(free<(8ULL<<30))throw std::runtime_error("paired SiLU setup headroom below8GiB");
    const auto setup_seconds=seconds(setup);const auto config=owner.config();const auto q=static_cast<long double>(literal.q()[0].value())*literal.q()[1].value();
    const long double qdiff=q/std::exp2(std::round(std::log2(q)));const auto values=silu_patterns();
    auto g0=runtime.encrypt(values.gate[0],6),u0=runtime.encrypt(values.up[0],14),g1=runtime.encrypt(values.gate[1],6),u1=runtime.encrypt(values.up[1],14);
    shape(runtime,g0,6,std::exp2(40.));shape(runtime,g1,6,std::exp2(40.));shape(runtime,u0,14,std::exp2(40.));shape(runtime,u1,14,std::exp2(40.));
    gpu_check_cuda(cudaDeviceSynchronize(),"public SiLU inputs before any tested operations");
    const std::array<const CT*,4> inputs{&g0,&u0,&g1,&u1};const std::array<Snapshot,4> initial{snapshot(g0),snapshot(u0),snapshot(g1),snapshot(u1)};
    const std::array<Decoded,2> decoded_up{decode_const(runtime,u0),decode_const(runtime,u1)};
    long double source_vs_true=0;
    for(unsigned lane=0;lane<2;++lane)for(std::size_t j=0;j<32768;++j) {
        const auto v=static_cast<long double>(values.gate[lane][j]),w=static_cast<long double>(values.up[lane][j]);
        source_vs_true=std::max(source_vs_true,std::abs(silu_cpu_reference::apply({v,0},{w,0})-Z(w*v/(1+std::exp(-v)),0)));
    }
    std::cout<<"PUBLIC_SILU_PAIR_SETUP {\"N\":"<<runtime.context.parameters_literal()->degree()<<",\"complex_slots\":"<<runtime.encoder.slot_count()
      <<",\"ratio\":65536,\"m\":"<<config.m<<",\"eta\":"<<config.eta<<",\"qdiff\":"<<qdiff<<",\"native_q\":27,\"output_log_scale\":50"
      <<",\"runtime_profile_setup_seconds\":"<<setup_seconds<<",\"source_vs_true_SiLU_max_complex\":"<<static_cast<double>(source_vs_true)<<",\"performance_valid\":false}"<<std::endl;
    bool pass=true;std::vector<std::string> rows;
    for(unsigned swapped=0;swapped<2;++swapped) {
        owner.verify();submission.check();const unsigned a=swapped,b=1-swapped;const auto &ga=swapped?g1:g0,&gb=swapped?g0:g1,&ua=swapped?u1:u0,&ub=swapped?u0:u1;
        const auto before=api.boot_calls,pair_before=api.manual_projection_pairs;const auto wall=Clock::now();
        auto off=manual_silu_all60_pair_candidate(api,ga,ua,gb,ub);gpu_check_cuda(cudaDeviceSynchronize(),"paired SiLU OFF baseline before ON");
        const std::array<Snapshot,2> off_frozen{snapshot(off[0]),snapshot(off[1])};
        RefreshTrace trace(runtime,config.eta,qdiff);const auto observer=trace.observer();auto on=manual_silu_all60_pair_candidate(api,ga,ua,gb,ub,observer);trace.finish();
        gpu_check_cuda(cudaDeviceSynchronize(),"paired SiLU ON baseline before old controls");const std::array<Snapshot,2> on_frozen{snapshot(on[0]),snapshot(on[1])};
        auto old_a=manual_silu_all60_unpaired(api,ga,ua);gpu_check_cuda(cudaDeviceSynchronize(),"old SiLU A before B baseline");const auto old_a_frozen=snapshot(old_a);
        auto old_b=manual_silu_all60_unpaired(api,gb,ub);gpu_check_cuda(cudaDeviceSynchronize(),"paired and independent SiLU controls complete");const auto old_b_frozen=snapshot(old_b);
        const auto diagnostic_seconds=seconds(wall);
        if(api.boot_calls-before!=16||api.manual_projection_pairs-pair_before!=8)throw std::runtime_error("actual SiLU physical/projection counts");
        const std::array<const CT*,6> out{&off[0],&off[1],&on[0],&on[1],&old_a,&old_b};
        for(const auto *x:out)shape(runtime,*x,10,std::exp2(40.));
        for(std::size_t i=0;i<out.size();++i)for(std::size_t j=i+1;j<out.size();++j)
            if(out[i]->fields_[0].data()==out[j]->fields_[0].data())throw std::runtime_error("SiLU final owned outputs alias");
        if(!matches(off[0],off_frozen[0])||!matches(off[1],off_frozen[1])||!matches(on[0],off_frozen[0])||!matches(on[1],off_frozen[1]))
            throw std::runtime_error("paired SiLU observer OFF/ON metadata/RNS not exact or OFF changed");
        const auto ea=source_errors(runtime,on[0],values.gate[a],values.up[a],decoded_up[a],&trace,0);
        const auto eb=source_errors(runtime,on[1],values.gate[b],values.up[b],decoded_up[b],&trace,1);
        const auto old_ea=source_errors(runtime,old_a,values.gate[a],values.up[a],decoded_up[a],nullptr,0);
        const auto old_eb=source_errors(runtime,old_b,values.gate[b],values.up[b],decoded_up[b],nullptr,1);
        bool one=trace.numerical_pass;
        for(const auto *e:{&ea,&eb,&old_ea,&old_eb})one&=e->active.passed()&&e->full.passed();
        pass&=one;
        for(unsigned i=0;i<4;++i)if(!matches(*inputs[i],initial[i]))throw std::runtime_error("SiLU public original input metadata/RNS mutated");
        if(!matches(off[0],off_frozen[0])||!matches(off[1],off_frozen[1])||!matches(on[0],on_frozen[0])||!matches(on[1],on_frozen[1])||!matches(old_a,old_a_frozen)||!matches(old_b,old_b_frozen))
            throw std::runtime_error("SiLU final output mutated by controls/decodes");
        owner.verify();std::ostringstream row;row<<std::setprecision(17)<<"{\"name\":\""<<(swapped?"swapped_b_a":"a_b")<<"\",\"physical_Boots\":16,\"projection_pairs\":8,\"derived_EvalMod_calls\":32"
          <<",\"paired_lane0\":"<<source_errors_json(ea)<<",\"paired_lane1\":"<<source_errors_json(eb)<<",\"old_lane0\":"<<source_errors_json(old_ea)<<",\"old_lane1\":"<<source_errors_json(old_eb)
          <<",\"paired_observer_OFF_ON_metadata_RNS_exact\":true,\"all_inputs_const_decodes_and_outputs_immutable\":true,\"all_slots_finite\":true,\"final_Q\":10,\"final_log_scale\":40"
          <<",\"diagnostic_includes_control_snapshots_and_observer_decode\":true,\"diagnostic_seconds\":"<<diagnostic_seconds<<",\"maxima_not_additive\":true,\"all_fixed_gates_pass\":"<<(one?"true":"false")<<",\"refreshes\":[";
        for(std::size_t i=0;i<trace.rows.size();++i)row<<(i?",":"")<<trace.rows[i];row<<"]}";rows.push_back(row.str());std::cout<<"PUBLIC_SILU_PAIR_CASE "<<row.str()<<std::endl;
    }
    if(api.boot_calls!=32||api.manual_projection_pairs!=16||rows.size()!=2)throw std::runtime_error("final closed SiLU counts");
    std::ostringstream final;final<<std::setprecision(17)<<"{\"status\":\"completed_public_silu_pair_control\",\"N\":65536,\"complex_slots\":32768,\"active_tokens\":38,\"case_count\":2,\"physical_Boots\":32,\"projection_pairs\":16,\"derived_internal_EvalMod_calls\":64"
      <<",\"runtime_profile_setup_seconds\":"<<setup_seconds<<",\"source_vs_true_SiLU_max_complex\":"<<static_cast<double>(source_vs_true)<<",\"all_numerical_gates_passed\":"<<(pass?"true":"false")<<",\"model_qualification\":false,\"performance_valid\":false,\"cases\":[";
    for(std::size_t i=0;i<rows.size();++i)final<<(i?",":"")<<rows[i];final<<"]}\n";report.finish(final.str());
    std::cout<<"PUBLIC_SILU_PAIR_DONE numerical_pass="<<pass<<" physical_Boots="<<api.boot_calls<<" projection_pairs="<<api.manual_projection_pairs<<std::endl;return pass?0:2;
}catch(const std::exception &e){std::cerr<<"PUBLIC_SILU_PAIR_FAILED "<<e.what()<<'\n';return 1;}
