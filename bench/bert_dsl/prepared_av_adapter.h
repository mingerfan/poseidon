// Bench-owned adapter, included inside native_runner.cpp's private namespace.
#pragma once

class PreparedAv {
    bert::attention::PreparedAvEngine engine_;
    std::unordered_map<std::uint64_t, std::string> calls_;
    std::unordered_map<std::uint64_t, std::string> callees_;
    std::unordered_map<std::string, double> pair_ms_;
    std::unordered_map<std::string, std::uint64_t> pair_calls_;
    bert::attention::AvEngineStats before_;
    Json setup_ = Json::array();
    struct Manifest {
        std::vector<bert::attention::AvSpec> specs;
        std::unordered_map<std::uint64_t, std::string> calls;
        std::unordered_map<std::uint64_t, std::string> callees;
    };
    static void validate_layout(const bert::attention::AvSpec &s) {
        if(s.id.empty() || s.tokens!=128 || s.tile!=256 || s.head_dim!=64 || s.baby!=8 ||
           s.score_input_level!=10 || s.value_input_level!=8 || s.common_level!=6 || s.output_level!=4)
            throw std::invalid_argument("prepared AV requires id/N65536/T128/tile256/D64/baby8 and physical L10/L8/L6/L4");
    }
    // Mirror the immutable engine's validate_spec equations/capacity envelope,
    // not the reference planner's stricter canonical input-scale relationship.
    // All work here is CPU-only and must precede the first prepare/GPU owner.
    static void validate_physical_spec(const bert::attention::AvSpec &s,
            const poseidon::PoseidonContext &context) {
        validate_layout(s);
        if(context.parameters_literal()->degree()!=65536 || context.parameters_literal()->slot()!=32768)
            throw std::invalid_argument("prepared AV requires id/N65536/T128/tile256/D64/baby8 and physical L10/L8/L6/L4");
        const auto &q=context.parameters_literal()->q();
        if(q.size()<=s.score_input_level || context.parameters_literal()->p().empty())
            throw std::invalid_argument("prepared AV requires sufficient Q and HYBRID P primes");
        for(const auto &prime:q)
            if(prime.value()>std::numeric_limits<std::uint32_t>::max())
                throw std::invalid_argument("prepared AV requires <=32-bit Q primes");
        for(const auto &prime:context.parameters_literal()->p())
            if(prime.value()>std::numeric_limits<std::uint32_t>::max())
                throw std::invalid_argument("prepared AV requires <=32-bit P primes");
        for(double scale:{s.score_input_scale_log2,s.value_input_scale_log2,
                          s.common_scale_log2,s.output_scale_log2})
            if(!std::isfinite(scale) || scale<=0 || scale>200)
                throw std::invalid_argument("invalid AV physical scale");
        const auto drop_log=[&](std::size_t from,std::size_t to) {
            double result=0;
            for(auto l=from;l>to;--l)result+=std::log2(double(q.at(l).value()));
            return result;
        };
        const double diagonal=s.common_scale_log2+drop_log(s.value_input_level,s.common_level)-s.value_input_scale_log2;
        const double score=s.common_scale_log2+drop_log(s.score_input_level,s.common_level)-s.score_input_scale_log2;
        const double output=2*s.common_scale_log2-drop_log(s.common_level,s.output_level);
        if(std::abs(output-s.output_scale_log2)>1e-8 || diagonal<=1 || score<=1)
            throw std::invalid_argument("AV scale plan does not match actual Q-prime drops");
        const auto require_capacity=[&](std::size_t level,double scale,const char *stage) {
            double capacity=0;
            for(std::size_t l=0;l<=level;++l)capacity+=std::log2(double(q.at(l).value()));
            if(!std::isfinite(scale) || scale+32.0>=capacity)
                throw std::invalid_argument(std::string("AV scale exceeds actual Q capacity/headroom at ")+stage);
        };
        require_capacity(s.score_input_level,s.score_input_scale_log2,"score input");
        require_capacity(s.value_input_level,s.value_input_scale_log2,"value input");
        require_capacity(s.score_input_level,score,"score mask plaintext");
        require_capacity(s.value_input_level,diagonal,"diagonal mask plaintext");
        require_capacity(s.score_input_level,s.score_input_scale_log2+score,"score CP");
        require_capacity(s.value_input_level,s.value_input_scale_log2+diagonal,"diagonal CP");
        require_capacity(s.common_level,s.common_scale_log2,"common ciphertext");
        require_capacity(s.common_level,2*s.common_scale_log2,"CC before rescale");
        require_capacity(s.output_level,s.output_scale_log2,"output ciphertext");
        for(std::size_t l=s.output_level;l<=s.score_input_level;++l) {
            bool found=false;
            for(auto data=context.crt_context()->first_context_data();data;data=data->next_context_data())
                if(data->parms().p().empty() && data->parms().q().size()==l+1) {found=true;break;}
            if(!found)throw std::invalid_argument("AvSpec physical level is absent from context");
        }
    }
    // CPU-only, shared by pre-device key/callee validation and preparation.
    // Re-reading here also rechecks SHA256 and required capabilities at setup.
    static Manifest read_manifest(const std::filesystem::path &path,const std::string &expected_digest,
            bool require_packed_partial=false) {
        std::string digest;auto manifest=read_json(path,&digest);
        const auto &version=manifest.at("format_version");
        if(!version.is_number_integer() || (version!=1 && version!=2) ||
           expected_digest.empty() || digest!=expected_digest)
            throw std::runtime_error("prepared AV manifest version/SHA256 mismatch");
        if(!manifest.at("calls").is_array())throw std::runtime_error("prepared AV calls must be an array");
        Manifest result;
        std::set<std::string> ids;
        bool has_packed_partial=false;
        for(const auto &row:manifest.at("calls")) {
            auto text=row.at("call_id").get<std::string>();std::size_t parsed=0;
            if(text.empty() || text.front()=='-')throw std::runtime_error("invalid prepared AV call_id");
            auto call_id=std::stoull(text,&parsed);
            if(parsed!=text.size())throw std::runtime_error("invalid prepared AV call_id");
            for(const auto *field:{"tokens","tile","head_dim","baby","score_input_level",
                    "value_input_level","common_level","output_level"})
                if(!row.at(field).is_number_integer())
                    throw std::runtime_error(std::string("prepared AV physical field must be a JSON integer: ")+field);
            for(const auto *field:{"score_input_scale_log2","value_input_scale_log2",
                    "common_scale_log2","output_scale_log2"})
                if(!row.at(field).is_number())
                    throw std::runtime_error(std::string("prepared AV physical scale must be a JSON number: ")+field);
            bert::attention::AvSpec s;
            s.id=row.at("id").get<std::string>();s.tokens=row.at("tokens").get<std::size_t>();
            s.tile=row.at("tile").get<std::size_t>();s.head_dim=row.at("head_dim").get<std::size_t>();
            s.baby=row.at("baby").get<std::size_t>();
            s.score_input_level=row.at("score_input_level").get<std::size_t>();
            s.score_input_scale_log2=row.at("score_input_scale_log2").get<double>();
            s.value_input_level=row.at("value_input_level").get<std::size_t>();
            s.value_input_scale_log2=row.at("value_input_scale_log2").get<double>();
            s.common_level=row.at("common_level").get<std::size_t>();
            s.common_scale_log2=row.at("common_scale_log2").get<double>();
            s.output_level=row.at("output_level").get<std::size_t>();
            s.output_scale_log2=row.at("output_scale_log2").get<double>();
            validate_layout(s);
            const auto algorithm=row.value("algorithm",std::string("Baseline"));
            if(algorithm==bert::attention::av_algorithm_name(bert::attention::AvAlgorithm::Baseline))
                s.algorithm=bert::attention::AvAlgorithm::Baseline;
            else if(algorithm==bert::attention::av_algorithm_name(bert::attention::AvAlgorithm::PackedInput2Partial))
                s.algorithm=bert::attention::AvAlgorithm::PackedInput2Partial;
            else throw std::runtime_error("unknown prepared AV algorithm");
            const bool packed=s.algorithm==bert::attention::AvAlgorithm::PackedInput2Partial;
            if(version==1 && packed)
                throw std::runtime_error("nonbaseline prepared AV algorithms require manifest version 2");
            for(const auto *field:{"pack_inputs","partial_output","client_value_gap_zero_required"}) {
                if(!row.contains(field))continue;
                if(!row.at(field).is_boolean())
                    throw std::runtime_error(std::string("prepared AV mode must be a JSON boolean: ")+field);
                if(row.at(field).get<bool>()!=packed)
                    throw std::runtime_error(std::string("prepared AV algorithm/mode mismatch: ")+field);
            }
            if(row.contains("pair_products")) {
                if(!row.at("pair_products").is_boolean() || row.at("pair_products").get<bool>())
                    throw std::runtime_error("prepared AV pair_products is unsupported");
            }
            const std::string expected_callee=packed?"bert.compact_av_packed_partial":"bert.compact_av";
            const auto callee=packed?row.at("callee").get<std::string>():row.value("callee",expected_callee);
            if(callee!=expected_callee)throw std::runtime_error("prepared AV algorithm/callee mismatch");
            const std::string expected_semantics=bert::attention::av_output_semantics(s.algorithm);
            const auto semantics=packed?row.at("output_semantics").get<std::string>():row.value("output_semantics",expected_semantics);
            if(semantics!=expected_semantics)throw std::runtime_error("prepared AV algorithm/output_semantics mismatch");
            if(s.id.empty() || !ids.insert(s.id).second || !result.calls.emplace(call_id,s.id).second)
                throw std::runtime_error("empty/duplicate prepared AV id/call_id");
            result.callees.emplace(call_id,callee);
            has_packed_partial=has_packed_partial || packed;
            result.specs.push_back(std::move(s));
        }
        if(result.specs.empty())throw std::runtime_error("prepared AV manifest has no calls");
        if(require_packed_partial && (version!=2 || !has_packed_partial))
            throw std::runtime_error("prepared AV manifest lacks required packed_input2_partial capability (version 2 and partial calls required)");
        return result;
    }
    static Json counters(const bert::attention::AvEngineStats &s) {
        return {{"run_calls",s.run_calls},{"baseline_run_calls",s.baseline_run_calls},
            {"packed_input2_partial_run_calls",s.packed_input2_partial_run_calls},
            {"cc_multiply_calls",s.cc_multiply_calls},
            {"relinearize_calls",s.relinearize_calls},{"cp_multiply_calls",s.cp_multiply_calls},
            {"rescale_calls",s.rescale_calls},{"rescale2_calls",s.rescale2_calls},
            {"rescale4_calls",s.rescale4_calls},{"physical_rescale_primes",s.physical_rescale_primes},
            {"rotation_calls",s.rotation_calls},{"hoisted_rotation_batches",s.hoisted_rotation_batches},
            {"add_calls",s.add_calls}};
    }
public:
    struct CpuPlan {
        std::set<int> rotation_steps;
        std::unordered_map<std::uint64_t, std::string> callees;
    };
    static void validate_context_contract(const std::filesystem::path &path,
            const std::string &expected_digest,const poseidon::PoseidonContext &context,
            bool require_packed_partial=false) {
        const auto manifest=read_manifest(path,expected_digest,require_packed_partial);
        for(const auto &s:manifest.specs)validate_physical_spec(s,context);
    }
    static CpuPlan cpu_plan(const std::filesystem::path &path,const std::string &expected_digest,
            bool direct,bool require_packed_partial=false) {
        auto manifest=read_manifest(path,expected_digest,require_packed_partial);
        CpuPlan result;
        result.callees=std::move(manifest.callees);
        for(const auto &s:manifest.specs) {
            using Engine=bert::attention::PreparedAvEngine;
            const auto steps=s.algorithm==bert::attention::AvAlgorithm::PackedInput2Partial
                ? (direct?Engine::required_packed_input_rotation_steps():Engine::required_packed_input_binary_rotation_steps())
                : (direct?Engine::required_rotation_steps():Engine::required_binary_rotation_steps());
            result.rotation_steps.insert(steps.begin(),steps.end());
        }
        return result;
    }
    static std::set<int> cpu_rotation_steps(const std::filesystem::path &path,
            const std::string &expected_digest,bool direct,bool require_packed_partial=false) {
        return cpu_plan(path,expected_digest,direct,require_packed_partial).rotation_steps;
    }
    PreparedAv(const poseidon::PoseidonContext &context,
               std::shared_ptr<const poseidon::RelinKeys> relin,
               std::shared_ptr<const poseidon::GaloisKeys> galois,
               const std::filesystem::path &path,const std::string &expected_digest,
               bool require_packed_partial=false)
        : engine_(context,std::move(relin),std::move(galois),0) {
        auto manifest=read_manifest(path,expected_digest,require_packed_partial);
        calls_=std::move(manifest.calls);callees_=std::move(manifest.callees);
        auto &specs=manifest.specs;
        for(const auto &s:specs)validate_physical_spec(s,context);
        std::sort(specs.begin(),specs.end(),[](const auto &a,const auto &b){
            const auto x=std::max(a.score_input_level,a.value_input_level);
            const auto y=std::max(b.score_input_level,b.value_input_level);
            return x==y?a.id<b.id:x>y;
        });
        for(const auto &s:specs) {
            engine_.prepare(s);const auto &m=engine_.av_stats(s.id);
            Json pair{{"id",s.id},{"score_input_level",s.score_input_level},
                {"value_input_level",s.value_input_level},{"common_level",s.common_level},
                {"output_level",s.output_level},{"shared_mask_bundle_bytes",m.mask_bytes},
                {"algorithm",bert::attention::av_algorithm_name(m.algorithm)},
                {"schedule",m.schedule},{"output_semantics",m.output_semantics},
                {"partial_output",m.partial_output},{"client_value_gap_zero_required",m.client_value_gap_zero_required},
                {"mask_plaintext_count",m.mask_plaintext_count},
                {"diagonal_mask_scale_log2",m.diagonal_mask_scale_log2},
                {"score_mask_scale_log2",m.score_mask_scale_log2},
                {"reused_mask_bundle",m.reused_mask_bundle},{"setup_seconds",m.setup_seconds}};
            if(m.partial_output)
                pair["noise_caveat"]="Partial AV has defined nonzero ideal gap sums and requires proven ideal value gaps zero. Model consumers must restore ideal zero gaps by multiplying with a separately proven ideal-gap-zero masked reciprocal before merge/lane-mixing/dense/finals. CKKS encoding/rotation/rescale gap noise is not bitwise zero; prepacking and masked reciprocal coefficients change noise propagation, so full-model HE precision/equivalence is not established.";
            setup_.push_back(std::move(pair));
        }
    }
    Value compute(RunnerApi &api,const fhegpu::ComputeOp &op,const std::vector<Value> &values) {
        const auto *attrs=std::get_if<fhegpu::ApiCallAttrs>(&op.attrs);
        if(!attrs || (attrs->callee!="bert.compact_av" && attrs->callee!="bert.compact_av_packed_partial") ||
           op.place.index!=0 || values.size()!=2)
            throw std::runtime_error("unsupported prepared AV callee/device/operand count");
        const auto &id=calls_.at(attrs->call_id);
        if(attrs->callee!=callees_.at(attrs->call_id))throw std::runtime_error("prepared AV call_id algorithm/callee mismatch");
        const auto begin=Clock::now();
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
            {"scope","callback includes input-ready wait and synchronized real GPU AV; not isolated kernel timing"}};
    }
    Json report() const {
        const auto s=engine_.engine_stats();
        return {{"pairs",setup_},{"actual_evaluator_calls_including_warmup",counters(s)},
            {"parameter_bytes",s.parameter_bytes},{"galois_key_bytes",s.galois_key_bytes},
            {"relin_key_bytes",s.relin_key_bytes},{"unique_mask_bytes",s.mask_bytes},
            {"mask_bundle_count",s.mask_bundle_count},{"mask_plaintext_count",s.mask_plaintext_count},
            {"relin_key_upload_calls",s.relin_key_upload_calls},{"galois_key_upload_calls",s.galois_key_upload_calls},
            {"preparation_locked",s.preparation_locked},{"persistent_allocated_bytes",s.allocated_bytes},
            {"key_storage_q_count",s.key_storage_q_count},{"key_storage_p_count",s.key_storage_p_count},
            {"key_storage_is_full_q",s.key_storage_is_full_q},{"setup_seconds",s.setup_seconds},
            {"memory_scope","unique shared masks/keys/parameters only; excludes scratch, run-local CT and pool; per-pair shared_mask_bundle_bytes and mask_plaintext_count must not be summed"}};
    }
};
