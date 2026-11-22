// Deliberately reuse the bench helper implementation without changing its source.
// The renamed legacy main is NOT a reachable command path of this executable.
#define main bert_legacy_single_main
#ifndef BERT_MULTI_PLAN_LEGACY_RUNNER
#error "Build via the coherent CMake target with an explicit frozen helper root"
#endif
#include BERT_MULTI_PLAN_LEGACY_RUNNER
#undef main
#include "compact_attention/qk_reference.h"
#include "audit.h"

namespace multi_plan {
// Runtime still runs its standard verifier and calls preflight on every run.
// After one real union preflight, these calls are fail-closed allowlist checks,
// not repeated GPU key materialization. The opaque engines own their real keys.
class MultiApi : public RunnerApi {
public:
    using RunnerApi::RunnerApi;
    std::set<std::string> allowed;
    fhegpu::PlanRequirements united;
    std::uint64_t runtime_preflight_checks=0;
    void preflight(std::string_view digest,bool skip,const fhegpu::TargetConfig &target,
                   const fhegpu::OperatorSpec &spec,const fhegpu::PlanRequirements &req) {
        require(!skip && allowed.count(std::string(digest)),"Runtime preflight identity not in audited library");
        require(target.world_size==1 && target.device_counts==std::vector<int>{1} && spec.context_id==context_id_for_check,"Runtime preflight context/target mismatch");
        for(const auto &k:req.keys)require(std::binary_search(united.keys.begin(),united.keys.end(),k),"Runtime requirement absent from real union preflight");
        for(auto cap:req.capabilities)require(std::find(united.capabilities.begin(),united.capabilities.end(),cap)!=united.capabilities.end(),"Runtime capability absent from union preflight");
        ++runtime_preflight_checks;
    }
    std::string context_id_for_check;
};
struct DrainGuard {
    MultiApi &api;Json &report;std::function<void()> snapshot;
    ~DrainGuard() noexcept {
        try{api.drain();}catch(const std::exception &e){try{report["failure_drain_error"]=e.what();}catch(...) {}}catch(...) {}
        try {
            report["actual_counts_observed_before_teardown"]={{"native_boot",api.native_boot_calls.load()},{"prepared_linear",api.prepared_linear_calls.load()},{"prepared_qk",api.prepared_qk_calls.load()},{"prepared_av",api.prepared_av_calls.load()},{"runtime_preflight_checks",api.runtime_preflight_checks},{"scope","successful API callbacks; prepared engine reports include their own partial evaluator calls"}};
            if(snapshot)snapshot();
        }catch(...) {}
    }
};
struct HostCache {
    std::unordered_map<fhegpu::ValueId,Value> values;
    fhegpu::ResourceUsage usage{1};
    Json report;
};
using Decoded = std::vector<std::vector<std::complex<double>>>;
Json differences(const Decoded &a,const Decoded &b) {
    require(a.size()==b.size(),"cross-run output tile count mismatch");Json rows=Json::array();
    for(std::size_t tile=0;tile<a.size();++tile) {
        require(a[tile].size()==b[tile].size(),"cross-run slot count mismatch");
        std::size_t finite=0,changed=0;double real=0,imag=0;long double squares=0;bool nonfinite_delta=false;
        for(std::size_t k=0;k<a[tile].size();++k) {
            auto x=a[tile][k],y=b[tile][k];
            if(!std::isfinite(x.real()) || !std::isfinite(x.imag()) || !std::isfinite(y.real()) || !std::isfinite(y.imag()))continue;
            ++finite;if(x!=y)++changed;auto d=x-y;
            if(!std::isfinite(d.real()) || !std::isfinite(d.imag())){nonfinite_delta=true;continue;}
            real=std::max(real,std::abs(d.real()));imag=std::max(imag,std::abs(d.imag()));
            squares+=static_cast<long double>(d.real())*d.real()+static_cast<long double>(d.imag())*d.imag();
        }
        auto rms=finite?std::sqrt(squares/finite):0;bool valid=!nonfinite_delta && finite==a[tile].size() && std::isfinite(real) && std::isfinite(imag) && std::isfinite(rms) && rms<=std::numeric_limits<double>::max();
        rows.push_back({{"tile",tile},{"slots",a[tile].size()},{"finite_compared_slots",finite},{"changed_slots",changed},{"valid",valid},{"max_abs_real_difference",valid?Json(real):Json(nullptr)},{"max_abs_imaginary_difference",valid?Json(imag):Json(nullptr)},{"complex_rms_difference",valid?Json(double(rms)):Json(nullptr)}});
    }
    return rows;
}
using QkOwners=std::vector<std::unique_ptr<PreparedQk>>;
using AvOwners=std::vector<std::unique_ptr<PreparedAv>>;
Json attention_report(const Library &lib,const QkOwners &qks,const AvOwners &avs,const PreparedLinear *linear) {
    Json result={{"plans",Json::array()},{"attention_persistent_allocated_bytes",0},{"qk_key_uploads",0},{"av_relin_key_upload_calls",0},{"av_galois_key_upload_calls",0},{"av_unique_mask_bytes_all_adapters",0},{"av_mask_bundle_count_all_adapters",0},{"shared_main_cpu_key_material",true},{"per_plan_gpu_copies_counted",true}};
    const auto add=[&](const char *key,std::uint64_t bytes){result[key]=result.at(key).get<std::uint64_t>()+bytes;};
    for(std::size_t i=0;i<lib.plans.size();++i) {
        Json plan={{"name",lib.plans[i].name},{"namespace",lib.plans[i].name}};
        if(qks[i]){auto j=qks[i]->report();plan["qk"]=j;add("attention_persistent_allocated_bytes",j.at("persistent_allocated_bytes"));add("qk_key_uploads",j.at("key_uploads"));}
        if(avs[i]){auto j=avs[i]->report();plan["av"]=j;add("attention_persistent_allocated_bytes",j.at("persistent_allocated_bytes"));add("av_relin_key_upload_calls",j.at("relin_key_upload_calls"));add("av_galois_key_upload_calls",j.at("galois_key_upload_calls"));add("av_unique_mask_bytes_all_adapters",j.at("unique_mask_bytes"));add("av_mask_bundle_count_all_adapters",j.at("mask_bundle_count"));}
        result["plans"].push_back(plan);
    }
    std::uint64_t linear_bytes=0;if(linear){auto j=linear->report();linear_bytes=j.at("parameter_bytes").get<std::uint64_t>()+j.at("key_bytes").get<std::uint64_t>();for(const auto &m:j.at("matrices"))linear_bytes+=m.at("compact_bytes").get<std::uint64_t>();}
    result["shared_linear_persistent_bytes_accounted"]=linear_bytes;result["all_prepared_persistent_bytes_accounted"]=linear_bytes+result.at("attention_persistent_allocated_bytes").get<std::uint64_t>();
    result["memory_scope"]="sum of every retained per-plan QK/AV adapter's real persistent allocation plus one shared Linear's parameter/key/compact storage; identical manifests are NOT deduplicated; includes repeated GPU key uploads, excludes native Boot/ordinary API keys, scratch, CTs and allocator pool reserve";
    return result;
}
void gpu_run(Library &lib,Json &report,std::string &stage,std::uint64_t workspace,std::uint64_t host_budget,const std::optional<fs::path> &decoded_dir) {
    stage="verify_audited_source_snapshots";verify_sources(lib);
    auto &first=lib.plans.front();const auto &plan=first.loaded.plan;const auto capacity=first.spec.spec.poly_degree/2;
    int visible=0;stage="gpu_device_gate";cuda_check(cudaGetDeviceCount(&visible),"cudaGetDeviceCount");require(visible==1,"exactly one visible CUDA device required");
    cudaDeviceProp prop{};cuda_check(cudaGetDeviceProperties(&prop,0),"cudaGetDeviceProperties");report["device"]={{"logical_device",0},{"name",prop.name},{"total_global_bytes",prop.totalGlobalMem}};
    auto setup_begin=Clock::now(),begin=setup_begin;
    stage="context_once";auto context=lib.cpu_context?*lib.cpu_context:make_context(first.fixture.at("context"),&first.spec.spec);context_report(report,context);report["context_setup_ms"]=ms(begin,Clock::now());report["context_construction_calls_total"]=1;report["cpu_validated_context_reused"]=bool(lib.cpu_context);
    stage="keys_once";begin=Clock::now();poseidon::KeyGenerator keys(context);
    auto secret=std::make_shared<poseidon::SecretKey>(keys.secret_key());auto public_key=std::make_shared<poseidon::PublicKey>();keys.create_public_key(*public_key);
    auto relin=std::make_shared<poseidon::RelinKeys>();auto galois=std::make_shared<poseidon::GaloisKeys>();
    if(lib.need_relin)keys.create_relin_keys(*relin);
    auto rotations=lib.ordinary_rotations;rotations.insert(lib.prepared_rotations.begin(),lib.prepared_rotations.end());
    auto policy=poseidon::runtime_api::rotation_key_policy_from_environment();auto basis=poseidon::runtime_api::rotation_key_steps(rotations,context.parameters_literal()->slot(),policy);
    basis.insert(lib.prepared_rotations.begin(),lib.prepared_rotations.end());
    if(!basis.empty())keys.create_galois_keys(std::vector<int>(basis.begin(),basis.end()),*galois);
    report["keys"]={{"generation_calls",1},{"public_key_generation_calls",1},{"relin_generation_calls",lib.need_relin?1:0},{"galois_generation_calls",basis.empty()?0:1},{"rotation_key_policy",poseidon::runtime_api::rotation_key_policy_name(policy)},{"actual_cpu_rotation_key_steps",basis},{"shared_across_all_plans",true},{"setup_ms",ms(begin,Clock::now())}};
    poseidon::CKKSEncoder encoder(context);poseidon::Encryptor encryptor(context,*public_key);poseidon::Decryptor decryptor(context,*secret);
    stage="encrypt_original_inputs_once";begin=Clock::now();std::unordered_map<fhegpu::ValueId,Value> inputs;
    for(std::size_t i=0;i<first.clear_inputs.size();++i) {
        auto value_id=plan.external_inputs[i];auto d=std::find_if(plan.values.begin(),plan.values.end(),[&](const auto &v){return v.id==value_id;});
        poseidon::Plaintext plain;poseidon::Ciphertext ct;
        encoder.encode(first.clear_inputs[i],context.crt_context()->parms_id_map().at(d->level),std::exp2(d->scale_log2),plain);encryptor.encrypt(plain,ct);
        inputs.emplace(value_id,Value::from_host_ciphertext(std::move(ct)));
    }
    report["input_encryption"]={{"generation_calls",1},{"input_ciphertext_count",inputs.size()},{"ciphertext_encrypt_calls",inputs.size()},{"shared_across_all_runs",true},{"refresh_or_feedback",false},{"setup_ms",ms(begin,Clock::now())}};
    stage="gpu_api_once";report["gpu_executed"]=true;MultiApi api(first.spec.spec.context_id,context,{0},relin,galois);api.context_id_for_check=first.spec.spec.context_id;api.united=lib.requirements;
    // Declared before the drain guard so they remain alive during failure drain.
    std::unique_ptr<PreparedLinear> linear;QkOwners qks(lib.plans.size());AvOwners avs(lib.plans.size());
    std::vector<HostCache> caches(lib.plans.size());std::optional<std::size_t> active;
    std::optional<fhegpu::RunArtifact<Value>> last;
    std::optional<poseidon::gpu::GpuBootstrapProfile> pending_profile;
    fhegpu::SequentialRuntime<MultiApi> runtime(0,1,1,api,fhegpu::DeviceExecutionMode::PerDeviceWorkers,1);
    std::vector<Decoded> decoded_runs;
    DrainGuard guard{api,report,[&]{if(linear)report["prepared_linear"]=linear->report();report["prepared_attention"]=attention_report(lib,qks,avs,linear.get());}}; // drain before owners; retain partial failure counters.
    stage="complete_native_profile_once";begin=Clock::now();
    const auto &declared=first.spec.spec.boot_profiles.front();auto config=boot_config(first.fixture,declared.profile_id);
    pending_profile.emplace(poseidon::gpu::GpuBootstrapProfileBuilder::build(context,keys,0,config));const auto &profile=*pending_profile;
    require(profile.output_level==declared.output_level && profile.output_scale_log2==declared.output_scale_log2 && declared.input_level_min>=profile.input_level_min && declared.input_level_max<=profile.input_level_max && profile.input_components==declared.input_components && profile.output_components==declared.output_components,"built native Boot profile disagrees with OperatorSpec");
    auto p=profile_report(profile,config);p["generation_calls"]=1;p["installation_calls"]=1;p["setup_ms"]=ms(begin,Clock::now());p["all_random_subcontext_material_shared"]=true;p["workspace_capacity_bytes"]=workspace?workspace:std::uint64_t(context.parameters_literal()->degree())*65536;
    api.configure_native_bootstrap(0,std::move(*pending_profile),workspace);pending_profile.reset();report["native_boot_profile"]=p;
    stage="real_union_gpu_preflight_once";begin=Clock::now();
    api.BaseApi::preflight(lib.schedule_sha,false,plan.target,first.spec.spec,lib.requirements);
    for(const auto &c:lib.plans)api.allowed.insert(c.loaded.preflight_identity.empty()?c.loaded.source_sha256:c.loaded.preflight_identity);
    report["gpu_preflight"]={{"actual_api_calls",1},{"union_of_all_verified_plan_requirements",true},{"prepared_engines_upload_their_own_real_cropped_keys",true},{"setup_ms",ms(begin,Clock::now())}};
    stage="prepared_linear_once";begin=Clock::now();linear=std::make_unique<PreparedLinear>(context,galois,first.manifest_paths.at("bert.compact_linear"),first.fixture.value("compact_periodic_encoding",false));
    report["prepared_linear_setup_ms"]=ms(begin,Clock::now());
    report["attention_adapter_setup"]=Json::array();
    for(std::size_t i=0;i<lib.plans.size();++i) {
        auto &c=lib.plans[i];Json setup={{"name",c.name},{"shares_main_cpu_key_objects",true},{"new_random_material_generated",false}};
        if(c.manifest_paths.count("bert.compact_qk")){stage="prepared_qk_"+c.name;begin=Clock::now();qks[i]=std::make_unique<PreparedQk>(context,relin,galois,c.manifest_paths.at("bert.compact_qk"),c.fixture.at("prepared_qk_manifest_sha256"),c.require_qk_pack);setup["qk_setup_ms"]=ms(begin,Clock::now());}
        if(c.manifest_paths.count("bert.compact_av")){stage="prepared_av_"+c.name;begin=Clock::now();avs[i]=std::make_unique<PreparedAv>(context,relin,galois,c.manifest_paths.at("bert.compact_av"),c.fixture.at("prepared_av_manifest_sha256"),c.require_av_partial);setup["av_setup_ms"]=ms(begin,Clock::now());}
        report["attention_adapter_setup"].push_back(setup);
    }
    report["prepared_attention"]=attention_report(lib,qks,avs,linear.get());
    api.prepared_compute=[&](const fhegpu::ComputeOp &op,const std::vector<Value> &values)->Value {
        const auto &a=std::get<fhegpu::ApiCallAttrs>(op.attrs);
        if(a.callee=="bert.compact_linear" && linear)return linear->compute(api,op,values);
        require(active.has_value(),"prepared dispatch has no active audited plan");
        if(a.callee=="bert.compact_qk" && qks.at(*active))return qks.at(*active)->compute(api,op,values);
        if((a.callee=="bert.compact_av" || a.callee=="bert.compact_av_packed_partial") && avs.at(*active))return avs.at(*active)->compute(api,op,values);
        throw std::runtime_error("unbound prepared callee: "+a.callee);
    };
    fhegpu::BundleReadOptions bundle_options; // no raw bundle residency: preserve phase I/O.
    stage="per_plan_real_host_caches";begin=Clock::now();
    auto estimate=lib.report.at("estimated_all_retained_host_cache_bytes").get<std::uint64_t>();
    auto available=available_host_bytes();if(!host_budget)host_budget=available/2;
    require(host_budget && estimate<=host_budget && estimate<=available,"total retained Host caches exceed available/budget");
    fhegpu::ResourceUsage total_usage(1);report["host_caches"]=Json::array();
    for(std::size_t i=0;i<lib.plans.size();++i) {
        auto &c=lib.plans[i];require(api.constants.empty(),"cache setup namespace must be empty");
        auto j=prepare_resident(api,c.loaded.plan,c.bundle,capacity,"on",0,"cpu","host",bundle_options);
        require(j.at("enabled")==true && j.at("cached_device_values")==0,"real Host-only cache setup failed");
        caches[i].values=std::move(api.constants);caches[i].usage=api.resident_usage;caches[i].report=j;total_usage+=caches[i].usage;
        j["name"]=c.name;report["host_caches"].push_back(j);
    }
    require(total_usage.host==estimate,"Host cache physical-byte accounting mismatch");
    report["all_retained_host_cache_bytes"]=total_usage.host;report["host_cache_budget_bytes"]=host_budget;report["host_caches_setup_ms"]=ms(begin,Clock::now());
    stage="confirm_setup_source_snapshots";verify_sources(lib);
    report["setup_ms_excluding_plan_library_read_audit"]=ms(setup_begin,Clock::now());report["setup_complete_before_schedule"]=true;
    report["runs"]=Json::array();report["cross_run_slot_differences"]=Json::array();
    for(std::size_t run=0;run<lib.schedule.size();++run) {
        stage="scheduled_run_"+std::to_string(run);auto index=lib.names.at(lib.schedule[run].at("name").get<std::string>());auto &c=lib.plans[index];
        report["active_run"]={{"index",run},{"name",c.name},{"runtime_completed",false},{"endpoint_completed",false}};
        // Run outputs are never inputs of a subsequent independent candidate.
        api.drain();last.reset(); // Runtime already returned; settle every owner before any plan switch.
        if(active)caches[*active].values=std::move(api.constants);
        api.constants=std::move(caches[index].values);active=index;
        api.resident_usage=total_usage; // ALL retained maps, not just active plan.
        linear->begin_run();if(qks[index])qks[index]->begin_run();if(avs[index])avs[index]->begin_run();
        auto boot_before=api.native_boot_calls.load(),linear_before=api.prepared_linear_calls.load(),qk_before=api.prepared_qk_calls.load(),av_before=api.prepared_av_calls.load(),hits_before=api.cache_hits.load();
        fhegpu::RuntimeResources resources{c.spec,c.bundle,false,bundle_options};begin=Clock::now();
        last.emplace(runtime.run(c.loaded,resources,inputs));api.drain();auto elapsed=ms(begin,Clock::now());report["active_run"]["runtime_completed"]=true;
        auto j=timing(last->timing);j["index"]=run;j["name"]=c.name;j["phase"]=lib.schedule[run].at("phase");j["runtime_wall_ms"]=elapsed;j["instrumented_diagnostic"]=true;j["formal_steady_sample"]=false;
        j["native_boot_calls"]=api.native_boot_calls.load()-boot_before;j["prepared_linear_calls"]=api.prepared_linear_calls.load()-linear_before;j["prepared_qk_calls"]=api.prepared_qk_calls.load()-qk_before;j["prepared_av_calls"]=api.prepared_av_calls.load()-av_before;j["resident_cache_hits"]=api.cache_hits.load()-hits_before;
        require(j.at("native_boot_calls")==c.counts["native_boot"] && j.at("prepared_linear_calls")==c.counts["bert.compact_linear"] && j.at("prepared_qk_calls")==c.counts["bert.compact_qk"] && j.at("prepared_av_calls")==c.counts["prepared_av_total"],"actual Boot/prepared counts disagree with audited plan");
        j["adapter_namespace"]=c.name;j["prepared_linear"]=linear->run_report();if(qks[index])j["prepared_qk"]=qks[index]->run_report();if(avs[index])j["prepared_av"]=avs[index]->run_report();
        j["endpoint_completed"]=false;report["runs"].push_back(j); // retain real Runtime timings even if endpoint diagnostics fail.
        stage="endpoint_download_decrypt_"+std::to_string(run);begin=Clock::now();Decoded decoded;Json decoded_rows=Json::array();j["outputs"]=Json::array();
        for(std::size_t tile=0;tile<c.loaded.plan.final_outputs.size();++tile) {
            auto v=c.loaded.plan.final_outputs[tile];auto d=std::find_if(c.loaded.plan.values.begin(),c.loaded.plan.values.end(),[&](const auto &x){return x.id==v;});
            auto host=download(api,v,*d,last->values.at(v).value);api.synchronize(host);poseidon::Plaintext plain;decryptor.decrypt(host.host_ciphertext(),plain);
            std::vector<std::complex<double>> actual;encoder.decode(plain,actual);std::vector<double> expected;
            if(c.fixture.contains("expected_outputs"))expected=slots(c.fixture.at("expected_outputs")[tile],v,capacity);
            auto stats=summary(actual,expected);stats["value_id"]=v;stats["level"]=d->level;stats["scale_log2"]=d->scale_log2;j["outputs"].push_back(stats);report["runs"].back()=j;
            if(decoded_dir) {Json real=Json::array(),imag=Json::array();for(auto z:actual){real.push_back(std::isfinite(z.real())?Json(z.real()):Json(nullptr));imag.push_back(std::isfinite(z.imag())?Json(z.imag()):Json(nullptr));}decoded_rows.push_back({{"value_id",v},{"real",real},{"imaginary",imag}});}
            decoded.push_back(std::move(actual));
        }
        api.drain();j["endpoint_download_decrypt_ms"]=ms(begin,Clock::now());
        if(decoded_dir) {auto path=*decoded_dir/("run-"+std::to_string(run)+".json");save_json(path,{{"format_version",1},{"run",run},{"name",c.name},{"fixture_sha256",c.fixture_sha},{"outputs",decoded_rows}});j["decoded_path"]=path.string();}
        for(std::size_t previous=0;previous<decoded_runs.size();++previous)report["cross_run_slot_differences"].push_back({{"earlier_run",previous},{"later_run",run},{"earlier_name",lib.schedule[previous].at("name")},{"later_name",c.name},{"outputs",differences(decoded_runs[previous],decoded)}});
        decoded_runs.push_back(std::move(decoded));j["endpoint_completed"]=true;report["runs"].back()=j;report["active_run"]["endpoint_completed"]=true;
        std::cout<<"RUN "<<run<<" name="<<c.name<<" phase="<<j["phase"]<<" online_ms="<<j["online_execution_ms"]<<" native_boot="<<j["native_boot_calls"]<<'\n'<<std::flush;
    }
    stage="final_drain";api.drain();report["prepared_linear"]=linear->report();report["prepared_attention"]=attention_report(lib,qks,avs,linear.get());
    report["gpu_preflight"]["runtime_allowlist_checks"]=api.runtime_preflight_checks;
    report["native_boot_calls_total"]=api.native_boot_calls.load();report["retained_endpoint_decoded_bytes"]=decoded_runs.size()*3*capacity*sizeof(std::complex<double>);
    auto m=api.waterline_measurements();report["device_measurements"]=Json::array();
    for(std::size_t i=0;i<m.devices.size();++i){const auto &d=m.devices[i];report["device_measurements"].push_back({{"logical_device",i},{"managed_pool",d.managed_pool},{"allocated_live_bytes",d.allocated_live},{"allocated_peak_bytes",d.allocated_peak},{"pool_reserved_bytes",d.pool_reserved},{"pool_capacity_bytes",d.pool_capacity}});}
}
} // namespace multi_plan

int main(int argc,char **argv) {
    const auto startup=Clock::now();
    const char *usage="usage: bert_multi_plan_runner --check-only SCHEDULE REPORT\n       bert_multi_plan_runner --run-gpu SCHEDULE REPORT [--native-boot-workspace-bytes N] [--host-cache-budget-bytes N] [--decoded-dir DIR]\nCPU default: no CUDA device calls. Paths inside SCHEDULE are relative to its directory.\nAll schedule phases are instrumented diagnostics, not formal steady samples.\n";
    if(argc==1){std::cout<<usage;return 0;}if(argc==2 && std::string(argv[1])=="--help"){std::cout<<usage;return 0;}
    if(argc<4 || (std::string(argv[1])!="--check-only" && std::string(argv[1])!="--run-gpu")){std::cerr<<usage;return 2;}
    const bool gpu=std::string(argv[1])=="--run-gpu";const std::filesystem::path output=argv[3];std::string stage="arguments";
    Json report={{"format_version",1},{"mode","controlled_multi_plan_diagnostic"},{"completed",false},{"gpu_requested",gpu},{"gpu_executed",false},{"secret_key_assisted_refresh",false},{"numerical_acceptance_enforced",false},{"old_new_binary_pairing",false},{"gpu_benefit_established",false},{"random_material_exported",false}};
    try {
        std::uint64_t workspace=0,host_budget=0;std::optional<std::filesystem::path> decoded;
        for(int n=4;n<argc;n+=2) {
            multi_plan::require(gpu && n+1<argc,"GPU options require --run-gpu and a value");std::string name=argv[n],value=argv[n+1];
            if(name=="--decoded-dir")decoded=value;
            else {multi_plan::require(!value.empty() && value.find_first_not_of("0123456789")==std::string::npos,"byte budget must be unsigned decimal");auto number=std::stoull(value);if(name=="--native-boot-workspace-bytes")workspace=number;else if(name=="--host-cache-budget-bytes")host_budget=number;else throw std::runtime_error("unknown option: "+name);}
        }
        stage="cpu_load_verify_compatibility";auto library=multi_plan::load(argv[2]);report["compatibility"]=library.report;report["plan_library_load_verify_ms"]=ms(startup,Clock::now());
        if(gpu){multi_plan::gpu_run(library,report,stage,workspace,host_budget,decoded);report["gpu_executed"]=true;}
        else {report["check_only"]=true;report["keys"]={{"generation_calls",0}};report["native_boot_profile"]={{"generation_calls",0},{"installation_calls",0}};report["input_encryption"]={{"generation_calls",0}};}
        report["timing_scope"]="Plan library read/standard verification/compatibility audit and setup are excluded from every Runtime online interval. Each Runtime online is execution+finalization; runtime_wall_ms includes per-run Runtime setup+initialization. Endpoint download/decrypt/save and all-pairs comparisons are outside both intervals. Even phase=measured is an instrumented diagnostic, not a formal steady sample. Ordinary Host caches are separate by plan ValueId namespace and ALL retained bytes are charged; original Device transfers remain real. Each plan consumes the same once-encrypted original hidden, never a preceding final output. No repeated fixed-seed noise, decrypt-refresh, binary pairing, or demonstrated GPU speedup claim.";
        report["security_level"]="none; experimental parameters";report["completed"]=true;report["stage"]="complete";report["main_wall_ms"]=ms(startup,Clock::now());save_json(output,report);std::cout<<"COMPLETE report="<<output<<" gpu_executed="<<gpu<<'\n';return 0;
    } catch(const std::exception &e) {
        report["stage"]=stage;report["error"]=e.what();report["main_wall_ms"]=ms(startup,Clock::now());std::cerr<<"FAIL stage="<<stage<<" "<<e.what()<<'\n';try{save_json(output,report);}catch(const std::exception &save){std::cerr<<save.what()<<'\n';}return 1;
    }
}
