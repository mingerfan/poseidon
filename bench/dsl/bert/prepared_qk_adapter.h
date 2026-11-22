// Bench-owned adapter included inside native_runner.cpp's private namespace.
// Json/RunnerApi/Value/Clock helpers are declared by that translation unit.
#pragma once

class PreparedQk {
    bert::attention::PreparedQkEngine engine_;
    std::unordered_map<std::uint64_t, std::string> calls_;
    std::unordered_map<std::string, double> pair_ms_;
    std::unordered_map<std::string, std::uint64_t> pair_calls_;
    bert::attention::EngineStats before_;
    Json setup_ = Json::array();

    struct Manifest {
        std::vector<bert::attention::QkSpec> specs;
        std::unordered_map<std::uint64_t, std::string> calls;
    };
    // Shared CPU-only parsing keeps key generation and engine preparation bound
    // to the same manifest contract. Preparation rechecks the digest as well.
    static Manifest read_manifest(const std::filesystem::path &path,const std::string &expected_digest,
            bool require_pack_inputs=false) {
        std::string digest;
        auto manifest=read_json(path,&digest);
        const auto &version=manifest.at("format_version");
        if(!version.is_number_integer() || (version!=1 && version!=2) ||
           expected_digest.empty() || digest!=expected_digest)
            throw std::runtime_error("prepared QK manifest version/SHA256 mismatch");
        Manifest result;
        bool has_packed_inputs=false;
        std::set<std::string> ids;
        for(const auto &row:manifest.at("calls")) {
            auto text=row.at("call_id").get<std::string>();std::size_t parsed=0;
            if(text.empty() || text.front()=='-')throw std::runtime_error("invalid prepared QK call_id");
            auto call_id=std::stoull(text,&parsed);
            if(parsed!=text.size())throw std::runtime_error("invalid prepared QK call_id");
            bert::attention::QkSpec s;
            s.id=row.at("id").get<std::string>();s.tokens=row.at("tokens").get<std::size_t>();
            s.tile=row.at("tile").get<std::size_t>();s.head_dim=row.at("head_dim").get<std::size_t>();
            s.baby=row.at("baby").get<std::size_t>();s.input_level=row.at("input_level").get<std::size_t>();
            s.input_scale_log2=row.at("input_scale_log2").get<double>();
            s.intermediate_scale_log2=row.at("intermediate_scale_log2").get<double>();
            s.output_level=row.at("output_level").get<std::size_t>();
            s.output_scale_log2=row.at("output_scale_log2").get<double>();
            for(const auto *field:{"pair_products","pack_inputs"})
                if(row.contains(field) && !row.at(field).is_boolean())
                    throw std::runtime_error(std::string("prepared QK mode must be a JSON boolean: ")+field);
            s.pair_products=row.value("pair_products",false);
            s.pack_inputs=row.value("pack_inputs",false);
            if(version==1 && (s.pair_products || s.pack_inputs))
                throw std::runtime_error("nonbaseline prepared QK modes require manifest version 2");
            if(s.pair_products && s.pack_inputs)
                throw std::runtime_error("QK pair_products and pack_inputs are mutually exclusive");
            if(!ids.insert(s.id).second || !result.calls.emplace(call_id,s.id).second)
                throw std::runtime_error("duplicate prepared QK id/call_id");
            has_packed_inputs=has_packed_inputs || s.pack_inputs;
            result.specs.push_back(std::move(s));
        }
        if(result.specs.empty())throw std::runtime_error("prepared QK manifest has no calls");
        if(require_pack_inputs && (version!=2 || !has_packed_inputs))
            throw std::runtime_error("prepared QK manifest lacks required pack_inputs capability (version 2 and packed calls required)");
        return result;
    }
    static Json counters(const bert::attention::EngineStats &s) {
        return {{"run_calls",s.run_calls},{"baseline_run_calls",s.baseline_run_calls},
            {"paired_run_calls",s.paired_run_calls},{"packed_input_run_calls",s.packed_input_run_calls},
            {"cc_multiply_calls",s.cc_multiply_calls},
            {"relinearize_calls",s.relinearize_calls},{"cp_multiply_calls",s.cp_multiply_calls},
            {"rescale2_calls",s.rescale2_calls},{"physical_rescale_primes",s.physical_rescale_primes},
            {"rotation_calls",s.rotation_calls},{"hoisted_rotation_batches",s.hoisted_rotation_batches},
            {"add_calls",s.add_calls}};
    }
public:
    static std::set<int> cpu_rotation_steps(const std::filesystem::path &path,
            const std::string &expected_digest,bool direct,bool require_pack_inputs=false) {
        const auto manifest=read_manifest(path,expected_digest,require_pack_inputs);
        std::set<int> result;
        for(const auto &s:manifest.specs) {
            using Engine=bert::attention::PreparedQkEngine;
            const auto steps=s.pack_inputs
                ? (direct?Engine::required_packed_input_rotation_steps():Engine::required_packed_input_binary_rotation_steps())
                : s.pair_products
                ? (direct?Engine::required_paired_rotation_steps():Engine::required_paired_binary_rotation_steps())
                : (direct?Engine::required_rotation_steps():Engine::required_binary_rotation_steps());
            result.insert(steps.begin(),steps.end());
        }
        return result;
    }
    PreparedQk(const poseidon::PoseidonContext &context,
               std::shared_ptr<const poseidon::RelinKeys> relin,
               std::shared_ptr<const poseidon::GaloisKeys> galois,
               const std::filesystem::path &path,const std::string &expected_digest,
               bool require_pack_inputs=false)
        : engine_(context,std::move(relin),std::move(galois),0) {
        auto manifest=read_manifest(path,expected_digest,require_pack_inputs);
        calls_=std::move(manifest.calls);
        auto &specs=manifest.specs;
        std::sort(specs.begin(),specs.end(),[](const auto &a,const auto &b){
            return a.input_level==b.input_level?a.id<b.id:a.input_level>b.input_level;
        });
        for(const auto &s:specs) {
            engine_.prepare(s);const auto &stats=engine_.qk_stats(s.id);
            Json pair{{"id",s.id},{"input_level",s.input_level},{"output_level",s.output_level},
                {"input_scale_log2",s.input_scale_log2},{"output_scale_log2",s.output_scale_log2},
                {"pair_products",stats.pair_products},{"pack_inputs",stats.pack_inputs},
                {"schedule",stats.schedule},{"encoding",stats.encoding},
                {"mask_scale_log2",stats.mask_scale_log2},{"mask_bytes",stats.mask_bytes},
                {"setup_seconds",stats.setup_seconds}};
            if(stats.pack_inputs)
                pair["noise_caveat"]="Pre-CC packing relies on ideal-message zero gaps, not bitwise-zero encrypted CKKS gaps; packing changes noise propagation and operation order, and encrypted equivalence is not asserted.";
            else if(stats.pair_products)
                pair["noise_caveat"]="Post-CC pairing relies on ideal-message zero gaps, not bitwise-zero encrypted CKKS gaps; pairing changes noise propagation, and encrypted equivalence is not asserted.";
            setup_.push_back(std::move(pair));
        }
    }
    Value compute(RunnerApi &api,const fhegpu::ComputeOp &op,const std::vector<Value> &values) {
        const auto *attrs=std::get_if<fhegpu::ApiCallAttrs>(&op.attrs);
        if(!attrs || attrs->callee!="bert.compact_qk" || op.place.index!=0 || values.size()!=2)
            throw std::runtime_error("unsupported prepared QK callee/device/operand count");
        const auto &id=calls_.at(attrs->call_id);const auto begin=Clock::now();
        // Value copies retain the actual immutable GPU inputs through the
        // complete private-stream invocation; no raw borrowed lifetime cache.
        auto owners=values;
        for(auto &value:owners)api.synchronize(value);
        auto result=engine_.run(id,owners[0].device_ciphertext(),owners[1].device_ciphertext());
        pair_ms_[id]+=ms(begin,Clock::now());++pair_calls_[id];
        return Value::from_device_ciphertext(std::move(result));
    }
    void begin_run() {before_=engine_.engine_stats();pair_ms_.clear();pair_calls_.clear();}
    Json run_report() const {
        auto count=counters(engine_.engine_stats()),old=counters(before_);
        for(auto it=count.begin();it!=count.end();++it)
            it.value()=it.value().get<std::uint64_t>()-old.at(it.key()).get<std::uint64_t>();
        return {{"actual_evaluator_calls",count},{"pair_synchronized_wall_ms",pair_ms_},
            {"pair_calls",pair_calls_},
            {"scope","prepared callback wall includes input-ready wait and final output synchronization; not isolated kernel timing"}};
    }
    Json report() const {
        const auto s=engine_.engine_stats();
        return {{"pairs",setup_},{"actual_evaluator_calls_including_warmup",counters(s)},
            {"parameter_bytes",s.parameter_bytes},{"galois_key_bytes",s.galois_key_bytes},
            {"relin_key_bytes",s.relin_key_bytes},{"mask_bytes",s.mask_bytes},
            {"persistent_allocated_bytes",s.allocated_bytes},{"key_storage_q_count",s.key_storage_q_count},
            {"key_storage_p_count",s.key_storage_p_count},{"key_storage_is_full_q",s.key_storage_is_full_q},
            {"key_uploads",s.key_uploads},{"uploaded_rotation_steps",s.uploaded_rotation_steps},
            {"setup_seconds",s.setup_seconds},
            {"memory_scope","persistent keys/parameters/masks only; excludes evaluator/rotation scratch, run-local CT and allocator pool"}};
    }
};
