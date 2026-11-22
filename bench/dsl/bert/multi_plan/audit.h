// Included in runner.cpp after the legacy bench helpers; CPU-only validation.
#pragma once
namespace multi_plan {
namespace fs = std::filesystem;
inline void require(bool ok,const std::string &message) { if(!ok)throw std::runtime_error(message); }
inline std::string hash(const std::string &bytes) {return "sha256:"+fhegpu::sha256_hex(bytes);}
inline std::string file_bytes(const fs::path &path) {
    std::ifstream in(path,std::ios::binary);require(bool(in),"cannot open "+path.string());
    std::ostringstream out;out<<in.rdbuf();require(!in.bad(),"cannot read "+path.string());return out.str();
}
inline fs::path resolve(const fs::path &base,const Json &path) {
    auto p=fs::path(path.get<std::string>());return fs::absolute(p.is_absolute()?p:base/p).lexically_normal();
}
inline std::uint64_t id(const Json &j) {
    if(j.is_number_unsigned())return j.get<std::uint64_t>();
    if(j.is_number_integer()){auto n=j.get<std::int64_t>();require(n>=0,"negative id");return n;}
    auto s=j.get<std::string>();require(!s.empty() && s.find_first_not_of("0123456789")==std::string::npos,"invalid decimal id");
    std::size_t n=0;auto v=std::stoull(s,&n);require(n==s.size(),"invalid id");return v;
}
inline Json desc_contract(Json d,bool keep_id=false) {if(!keep_id)d.erase("id");return d;}
struct Candidate {
    std::string name,fixture_sha,plan_sha,spec_sha;
    fs::path plan_path,spec_path,fixture_path;
    std::optional<fs::path> bundle;
    fhegpu::LoadedRuntimePlan loaded;
    fhegpu::LoadedOperatorSpec spec;
    fhegpu::PlanRequirements requirements;
    Json fixture,raw_plan,raw_spec,source;
    std::map<std::uint64_t,Json> descriptors;
    std::map<std::string,Json> manifests;
    std::map<std::string,fs::path> manifest_paths;
    std::vector<std::vector<double>> clear_inputs;
    std::map<std::string,std::uint64_t> counts;
    std::uint64_t estimated_host_cache_bytes=0;
    bool require_qk_pack=false,require_av_partial=false;
    std::unordered_map<std::uint64_t,std::string> av_callees;
};
struct Library {
    std::vector<Candidate> plans;
    std::map<std::string,std::size_t> names;
    Json schedule,report;
    std::string schedule_sha;
    fhegpu::PlanRequirements requirements;
    std::set<int> ordinary_rotations,prepared_rotations;
    bool need_relin=false,mixed_attention_contracts=false;
    std::set<std::string> prepared_capabilities;
    std::shared_ptr<poseidon::PoseidonContext> cpu_context;
};
inline void same(const Json &a,const Json &b,const std::string &what) {require(a==b,"incompatible "+what);}
inline void valid_sha(const Json &j,const std::string &what) {
    auto s=j.get<std::string>();require(s.size()==71 && s.substr(0,7)=="sha256:" && s.substr(7).find_first_not_of("0123456789abcdef")==std::string::npos,"invalid SHA256: "+what);
}
inline void append_f64_le(std::string &out,double x) {
    require(std::isfinite(x),"nonfinite clear input");std::uint64_t bits;std::memcpy(&bits,&x,8);
    for(unsigned i=0;i<8;++i)out.push_back(char((bits>>(8*i))&255));
}
inline std::set<int> steps(const Json &j) {
    require(j.is_array(),"rotation steps must be an array");std::set<int> out;
    for(const auto &x:j){require(x.is_number_integer() && x.get<double>()>=std::numeric_limits<int>::min() && x.get<double>()<=std::numeric_limits<int>::max(),"rotation step must be a JSON int32 integer");require(out.insert(x.get<int>()).second,"duplicate rotation step");}return out;
}
inline std::set<int> linear_steps(const Json &manifest) {
    std::set<int> result;
    for(const auto &m:manifest.at("matrices")) {
        int tile=m.at("tile"),baby=m.at("baby_step");require(tile==256 && baby==16,"unsupported Linear tile/baby");
        for(int b=1;b<baby;++b)result.insert(b);
        for(int g=-((tile-1+baby-1)/baby);g<=(tile-1)/baby;++g)if(g)result.insert(g*baby);
    }
    return result;
}
inline std::uint64_t positive_config_integer(const Json &config,const std::string &key) {
    const auto &value=config.at(key);require(value.is_number_integer(),"model config "+key+" must be a positive integer");
    if(value.is_number_unsigned()){auto n=value.get<std::uint64_t>();require(n>0,"model config "+key+" must be a positive integer");return n;}
    auto n=value.get<std::int64_t>();require(n>0,"model config "+key+" must be a positive integer");return std::uint64_t(n);
}
inline void audit_model_config(const Json &cfg) {
    require(cfg.is_object(),"model config must be an object");
    const std::set<std::string> required={"tokens","hidden","heads","tile","slots","intermediate","layers","bsgs_baby_step","reciprocal_iterations","ln_epsilon","ln_variance_center","weight_std","attention_layout"};
    for(const auto &key:required)require(cfg.contains(key),"model config missing required field "+key);
    for(auto it=cfg.begin();it!=cfg.end();++it)require(required.count(it.key()) || it.key()=="head_dim","unsupported model config field "+it.key());
    for(const auto *key:{"tokens","hidden","heads","tile","slots","intermediate","layers","bsgs_baby_step","reciprocal_iterations"})(void)positive_config_integer(cfg,key);
    auto tokens=positive_config_integer(cfg,"tokens"),hidden=positive_config_integer(cfg,"hidden"),heads=positive_config_integer(cfg,"heads"),tile=positive_config_integer(cfg,"tile"),slots=positive_config_integer(cfg,"slots"),intermediate=positive_config_integer(cfg,"intermediate");
    require(tile==256 && hidden==3*tile && cfg.at("bsgs_baby_step")==16,"unsupported model config prepared Linear geometry");
    require(!(slots&(slots-1)) && tokens<=slots/tile && tokens*tile==slots && intermediate%tile==0 && hidden%heads==0,"invalid model config dimensions");
    auto head_dim=hidden/heads;require(tile%head_dim==0,"model config head_dim must divide tile");
    if(cfg.contains("head_dim"))require(positive_config_integer(cfg,"head_dim")==head_dim,"model config head_dim disagrees with hidden/heads");
    require(cfg.at("attention_layout").is_string(),"model config attention_layout must be a string");auto layout=cfg.at("attention_layout").get<std::string>();
    require(layout=="offset" || (layout=="cyclic" && tokens<=UINT64_MAX/2 && tile==2*tokens && tokens==2*head_dim),"unsupported model config attention layout/geometry");
    require(cfg.at("reciprocal_iterations")==3,"model config requires exactly three reciprocal iterations");
    for(const auto *key:{"ln_epsilon","ln_variance_center","weight_std"}){const auto &value=cfg.at(key);require(value.is_number() && std::isfinite(value.get<double>()) && value.get<double>()>0,std::string("model config ")+key+" must be finite and positive");}
}
inline void audit_data(Candidate &c) {
    const auto &f=c.fixture,&model=f.at("model"),&cfg=model.at("config");audit_model_config(cfg);
    c.source["model_config"]=cfg;c.source["model_config_sha256"]=hash(cfg.dump());
    require(model.at("seed").is_number_integer() && (model.at("seed").is_number_unsigned() || model.at("seed").get<std::int64_t>()>=0),"model seed must be a nonnegative JSON integer");c.source["model_seed"]=model.at("seed");
    const auto capacity=c.spec.spec.poly_degree/2;
    const auto &context=f.at("context");
    same(context.at("poly_degree"),c.raw_spec.at("context").at("poly_degree"),"fixture/spec poly_degree");
    same(context.at("q_moduli_bits"),c.raw_spec.at("context").at("rns_moduli_log2"),"fixture/spec Q bits");
    same(context.at("q_moduli"),c.raw_spec.at("context").at("rns_moduli"),"fixture/spec actual Q primes");
    same(context.at("default_scale_log2"),c.raw_spec.at("context").at("default_scale_log2"),"fixture/spec default scale");
    require(c.spec.spec.status!="placeholder","placeholder OperatorSpec is forbidden");
    auto n=context.at("poly_degree").get<std::uint64_t>();log2_exact(n);
    require(n<=UINT64_MAX/2,"context degree overflow");std::set<std::uint64_t> moduli;
    for(const auto &[values_key,bits_key]:std::vector<std::pair<std::string,std::string>>{{"q_moduli","q_moduli_bits"},{"p_moduli","p_moduli_bits"}}) {
        const auto &values=context.at(values_key),&bits=context.at(bits_key);require(!values.empty() && values.size()==bits.size(),"explicit Q/P modulus widths mismatch");
        for(std::size_t k=0;k<values.size();++k){auto p=values[k].get<std::uint64_t>();auto b=bits[k].get<int>();require(b>=2 && b<=32 && p>=(1ULL<<(b-1)) && p<(1ULL<<b) && (p-1)%(2*n)==0 && moduli.insert(p).second,"invalid explicit Q/P modulus/width/NTT or duplicate");}
    }
    require(context.value("q0_level",1U)<context.at("q_moduli").size()-1 && context.value("secret_hamming_weight",0ULL)<=n,"invalid q0 level/secret weight");
    require(model.at("layout")=="cerium-feature-tiled","only cerium-feature-tiled client input is supported");
    const auto tokens=cfg.at("tokens").get<std::size_t>(),tile=cfg.at("tile").get<std::size_t>(),hidden=cfg.at("hidden").get<std::size_t>();
    require(tokens && tile && tokens<=capacity/tile && tokens*tile==capacity && hidden==3*tile && cfg.at("slots")==capacity,"unsupported packed input geometry (exactly three tiles required)");
    require(c.loaded.plan.external_inputs.size()==3 && f.at("inputs").size()==3,"exactly three original Host inputs required");
    for(std::size_t i=0;i<3;++i) {
        auto v=c.loaded.plan.external_inputs[i];const auto &d=c.descriptors.at(v);
        require(d.at("kind")=="ciphertext" && d.at("place").at("kind")=="host" && d.at("place").at("rank")==0 && d.at("ntt")==true && d.at("components")==2,"original input must be rank0 Host size2 NTT ciphertext");
        c.clear_inputs.push_back(slots(f.at("inputs")[i],v,capacity));
        require(c.clear_inputs.back().size()==capacity,"input must explicitly contain every packed slot");
    }
    std::string original,packed;original.reserve(tokens*hidden*8);packed.reserve(3*capacity*8);
    for(const auto &row:c.clear_inputs)for(auto x:row)append_f64_le(packed,x);
    for(std::size_t t=0;t<tokens;++t)for(std::size_t group=0;group<3;++group)for(std::size_t x=0;x<tile;++x)append_f64_le(original,c.clear_inputs[group][t*tile+x]);
    valid_sha(model.at("input_sha256"),"model input");same(hash(original),model.at("input_sha256"),"actual original client input SHA256");
    c.source["actual_original_input_sha256"]=hash(original);c.source["packed_input_f64le_sha256"]=hash(packed);
    auto weights=model.at("weight_sha256");require(weights.is_object() && !weights.empty(),"missing all-parameter hashes");
    const auto layers=positive_config_integer(cfg,"layers");require(weights.size()==layers,"parameter hash layer set must equal model config layers");
    for(std::uint64_t layer=0;layer<layers;++layer)require(weights.contains(std::to_string(layer)),"parameter hash layer set must be canonical 0..layers-1");
    const std::set<std::string> fields={"q_weight","q_bias","k_weight","k_bias","v_weight","v_bias","o_weight","o_bias","ffn_up_weight","ffn_up_bias","ffn_down_weight","ffn_down_bias","attention_ln_gamma","attention_ln_beta","ffn_ln_gamma","ffn_ln_beta"};
    for(auto layer=weights.begin();layer!=weights.end();++layer) {
        require(layer.value().is_object(),"parameter hash layer must be an object");
        std::set<std::string> actual;for(auto p=layer.value().begin();p!=layer.value().end();++p){actual.insert(p.key());valid_sha(p.value(),"parameter "+p.key());}
        require(actual==fields,"all 16 parameter SHA fields are required for every layer");
    }
    c.source["all_parameter_hashes_sha256"]=hash(weights.dump());
    require(f.contains("compact_linear_manifest"),"prepared Linear manifest is required");
    auto path=resolve(c.fixture_path.parent_path(),f.at("compact_linear_manifest"));std::string digest;
    auto m=read_json(path,&digest);require(m.at("format_version")==1 && m.at("dtype")=="<f4","unsupported Linear manifest");
    auto bytes=file_bytes(resolve(path.parent_path(),m.at("weight_file")));same(hash(bytes),m.at("weight_sha256"),"actual Linear weight file SHA256");
    require(m.at("matrices").is_array() && !m.at("matrices").empty() && m.at("calls").is_array() && !m.at("calls").empty(),"Linear matrices/calls must be nonempty arrays");
    std::set<std::string> matrices;
    for(const auto &row:m.at("matrices")) {
        auto matrix=row.at("matrix_id").get<std::string>();require(matrices.insert(matrix).second,"duplicate Linear matrix namespace id");
        auto dot=matrix.find('.');require(matrix.rfind("layer",0)==0 && dot!=std::string::npos,"unsupported matrix parameter binding");
        auto layer=matrix.substr(5,dot-5),field=matrix.substr(dot+1);require(field.size()>7 && field.substr(field.size()-7)=="_weight","invalid matrix field");
        auto bias=field.substr(0,field.size()-7)+"_bias";
        auto width=row.at("input_width").get<std::uint64_t>(),height=row.at("output_width").get<std::uint64_t>();
        require(width && height && height<=bytes.size()/4/width,"matrix dimensions overflow asset");
        const auto intermediate=positive_config_integer(cfg,"intermediate");
        require(width==(field=="ffn_down_weight"?intermediate:hidden) && height==(field=="ffn_up_weight"?intermediate:hidden),"Linear matrix dimensions disagree with model config");
        const auto digest_range=[&](const char *key,std::uint64_t count) {
            auto start=row.at(key).get<std::uint64_t>();require(start<=bytes.size() && count<=(bytes.size()-start)/4,"Linear parameter byte range invalid");return hash(bytes.substr(start,count*4));
        };
        same(digest_range("weight_offset",width*height),weights.at(layer).at(field),"dense parameter "+matrix);
        same(digest_range("bias_offset",height),weights.at(layer).at(bias),"dense bias "+matrix);
    }
    c.manifests["bert.compact_linear"]=m;c.manifest_paths["bert.compact_linear"]=path;
    c.source["compact_linear_manifest_sha256"]=digest;c.source["linear_weights_sha256"]=hash(bytes);
    for(const auto &[callee,key]:std::vector<std::pair<std::string,std::string>>{{"bert.compact_qk","prepared_qk_manifest"},{"bert.compact_av","prepared_av_manifest"}})if(f.contains(key)) {
        auto p=resolve(c.fixture_path.parent_path(),f.at(key));auto j=read_json(p,&digest);
        same(digest,f.at(key+"_sha256"),"prepared manifest SHA256"); // authoritative adapter validates version/mode/schema below
        c.manifests[callee]=j;c.manifest_paths[callee]=p;c.source[key+"_sha256"]=digest;
    }
    if(f.contains("expected_outputs")) {
        require(f.at("expected_outputs").size()==c.loaded.plan.final_outputs.size(),"expected output count mismatch");
        for(std::size_t i=0;i<c.loaded.plan.final_outputs.size();++i) {
            auto v=slots(f.at("expected_outputs")[i],c.loaded.plan.final_outputs[i],capacity);
            require(v.size()==capacity && std::all_of(v.begin(),v.end(),[](double x){return std::isfinite(x);}),"invalid expected output slots");
        }
    }
}
inline void audit_calls(Candidate &c,Library &lib,std::map<std::pair<std::string,std::uint64_t>,Json> &namespace_contracts) {
    std::map<std::pair<std::string,std::uint64_t>,Json> calls;
    std::map<std::string,Json> matrix_contracts;
    std::set<int> manifest_rotations;
    for(const auto &[callee,m]:c.manifests) {
        std::set<std::string> ids;require(m.at("calls").is_array() && !m.at("calls").empty(),"prepared calls must be a nonempty array");
        std::set<int> actual;
        if(callee=="bert.compact_linear")actual=linear_steps(m);
        else if(callee=="bert.compact_qk")actual=PreparedQk::cpu_rotation_steps(c.manifest_paths.at(callee),c.source.at("prepared_qk_manifest_sha256"),true,c.require_qk_pack);
        else {
            auto av=PreparedAv::cpu_plan(c.manifest_paths.at(callee),c.source.at("prepared_av_manifest_sha256"),true,c.require_av_partial);
            actual=std::move(av.rotation_steps);c.av_callees=std::move(av.callees);
            if(!lib.cpu_context)lib.cpu_context=std::make_shared<poseidon::PoseidonContext>(make_context(c.fixture.at("context"),&c.spec.spec));
            // Reuse the frozen authoritative adapter's full mode/parser/physical/headroom validator.
            PreparedAv::validate_context_contract(c.manifest_paths.at(callee),c.source.at("prepared_av_manifest_sha256"),*lib.cpu_context,c.require_av_partial);
        }
        for(const auto &call:m.at("calls")) {
            const auto call_id=id(call.at("call_id"));const auto call_callee=callee=="bert.compact_av"?c.av_callees.at(call_id):callee;
            auto key=std::make_pair(call_callee,call_id);
            require(calls.emplace(key,call).second,"duplicate callee/call_id namespace");
            if(callee=="bert.compact_linear") {
                auto contract=call;contract.erase("call_id");contract.erase("output_tile");auto matrix=call.at("matrix_id").get<std::string>();
                if(matrix_contracts.count(matrix))same(contract,matrix_contracts.at(matrix),"one Linear matrix physical contracts");else matrix_contracts[matrix]=contract;
                // Only Linear has a shared engine namespace. Attention belongs to c.name.
                auto old=namespace_contracts.find(key);if(old!=namespace_contracts.end())same(old->second,call,"callee/call_id physical contract namespace collision: "+callee+"/"+std::to_string(key.second));else namespace_contracts.emplace(key,call);
            } else {
                require(c.spec.spec.poly_degree==65536,"prepared attention requires N65536");
                const auto &cfg=c.fixture.at("model").at("config");require(cfg.at("attention_layout")=="cyclic" && cfg.at("tokens")==128 && positive_config_integer(cfg,"hidden")/positive_config_integer(cfg,"heads")==64,"prepared attention disagrees with model config cyclic/head geometry");
                require(!call.at("id").get<std::string>().empty() && ids.insert(call.at("id").get<std::string>()).second,"empty/duplicate prepared physical id");
                if(callee=="bert.compact_qk") {
                    const std::set<std::string> fields={"tokens","tile","head_dim","baby","call_id","id","input_level","input_scale_log2","intermediate_scale_log2","output_level","output_scale_log2","pair_products","pack_inputs"};
                    for(auto it=call.begin();it!=call.end();++it)require(fields.count(it.key()),"unsupported prepared QK call field/mode: "+it.key());
                    for(const auto *field:{"tokens","tile","head_dim","baby","input_level","output_level"})require(call.at(field).is_number_integer(),std::string("prepared QK physical field must be a JSON integer: ")+field);
                    for(const auto *field:{"input_scale_log2","intermediate_scale_log2","output_scale_log2"})require(call.at(field).is_number(),std::string("prepared QK physical scale must be a JSON number: ")+field);
                    require(!call.value("pair_products",false),"paired-products QK is outside this bounded baseline/pack_inputs diagnostic");
                    bert::attention::QkSpec s;s.input_level=call.at("input_level");s.input_scale_log2=call.at("input_scale_log2");s.intermediate_scale_log2=call.at("intermediate_scale_log2");s.output_level=call.at("output_level");s.output_scale_log2=call.at("output_scale_log2");s.pack_inputs=call.value("pack_inputs",false);
                    (void)bert::attention::reference::derive_scale_plan(s,c.fixture.at("context").at("q_moduli").get<std::vector<std::uint64_t>>());
                    if(!lib.cpu_context)lib.cpu_context=std::make_shared<poseidon::PoseidonContext>(make_context(c.fixture.at("context"),&c.spec.spec));
                    for(auto level:{s.input_level,s.input_level-2,s.output_level}) {
                        bool found=false;for(auto data=lib.cpu_context->crt_context()->first_context_data();data;data=data->next_context_data())if(data->parms().p().empty() && data->parms().q().size()==level+1){found=true;break;}
                        require(found,"QK physical level is absent from context");
                    }
                    if(s.pack_inputs)lib.prepared_capabilities.insert("qk.pack_inputs");
                } else if(call_callee=="bert.compact_av_packed_partial")lib.prepared_capabilities.insert("av.packed_input2_partial");
            }
        }
        if(callee!="bert.compact_linear") {
            std::set<int> canonical;const int slots=int(c.spec.spec.poly_degree/2);
            for(auto step:actual){step%=slots;if(step<0)step+=slots;if(step>slots/2)step-=slots;if(step)canonical.insert(step);}actual=std::move(canonical);
        }
        require(steps(m.at("rotation_steps"))==actual,"prepared manifest rotation steps disagree with physical engine: "+callee);
        c.source["prepared_rotation_steps_by_engine"][callee]=actual;manifest_rotations.insert(actual.begin(),actual.end());
    }
    require(c.fixture.contains("prepared_direct_rotation_keys") && c.fixture.at("prepared_direct_rotation_keys").is_boolean() && c.fixture.at("prepared_direct_rotation_keys").get<bool>(),"prepared direct rotation keys must be a true JSON boolean");
    require(steps(c.fixture.at("prepared_rotation_steps"))==manifest_rotations,"fixture prepared rotations must equal actual manifest union");
    lib.prepared_rotations.insert(manifest_rotations.begin(),manifest_rotations.end());
    std::set<std::pair<std::string,std::uint64_t>> used;
    std::map<std::string,std::size_t> next_tile;
    std::optional<std::string> active_linear_matrix;std::vector<fhegpu::ValueId> active_linear_inputs;
    for(const auto *phase:{&c.loaded.plan.initialization,&c.loaded.plan.execution,&c.loaded.plan.finalization})for(const auto &ins:*phase) {
        if(std::holds_alternative<fhegpu::PreparePlaintextOp>(ins.body))throw std::runtime_error("Host-only cache does not support PreparePlaintext");
        if(const auto *enc=std::get_if<fhegpu::EncodeOp>(&ins.body)) {
            const auto &d=c.descriptors.at(enc->output);require(d.at("kind")=="plaintext" && d.at("place").at("kind")=="host","Host-only Encode cache required");
            auto level=d.at("level").get<std::uint64_t>();auto bytes=c.spec.spec.poly_degree*(level+1)*sizeof(std::uint64_t);
            require(c.estimated_host_cache_bytes<=UINT64_MAX-bytes,"Host cache byte overflow");c.estimated_host_cache_bytes+=bytes;
        }
        const auto *op=std::get_if<fhegpu::ComputeOp>(&ins.body);if(!op)continue;
        if(op->reuse_input)throw std::runtime_error("reuse_input is unsupported; ordinary add aliasing is forbidden");
        if(op->kind==fhegpu::ComputeKind::Boot) {
            const auto &a=std::get<fhegpu::BootAttrs>(op->attrs);require(a.implementation==fhegpu::BootImplementation::Native && op->place.kind==fhegpu::PlaceKind::Device && op->place.index==0,"only real native device0 Boot is allowed");++c.counts["native_boot"];
        }
        if(op->kind!=fhegpu::ComputeKind::ApiCall)continue;
        const auto &a=std::get<fhegpu::ApiCallAttrs>(op->attrs);auto key=std::make_pair(a.callee,a.call_id);
        if(a.callee=="bert.compact_av" || a.callee=="bert.compact_av_packed_partial") {
            auto expected=c.av_callees.find(a.call_id);require(expected!=c.av_callees.end() && expected->second==a.callee,"prepared AV Runtime call_id algorithm/callee mismatch or missing manifest");
        }
        require(calls.count(key),"unbound prepared callee/call_id");require(used.insert(key).second,"repeated callee/call_id within one plan is unsupported");
        require(op->place.kind==fhegpu::PlaceKind::Device && op->place.index==0,"prepared calls require device0");
        const auto &call=calls.at(key);auto output=c.descriptors.at(op->output);
        const auto check=[&](const Json &d,int level,double scale) {
            require(d.at("kind")=="ciphertext" && d.at("components")==2 && d.at("ntt")==true && d.at("place")==output.at("place") && d.at("context")==output.at("context") && d.at("level")==level && d.at("scale_log2").get<double>()==scale,"prepared plan/manifest physical descriptor mismatch");
        };
        check(output,call.at("output_level"),call.at("output_scale_log2"));
        if(a.callee=="bert.compact_linear") {
            auto matrix=call.at("matrix_id").get<std::string>();auto tile=call.at("tile").get<std::size_t>();auto b=call.at("baby_step").get<std::size_t>();
            require(tile==256 && b==16 && call.at("rescale_count")==2 && call.at("input_level").get<int>()-2==call.at("output_level").get<int>(),"unsupported Linear physical rescale contract");
            auto row=std::find_if(c.manifests.at(a.callee).at("matrices").begin(),c.manifests.at(a.callee).at("matrices").end(),[&](const auto &r){return r.at("matrix_id")==matrix;});
            require(row!=c.manifests.at(a.callee).at("matrices").end(),"Linear matrix missing");
            same(row->at("input_width"),call.at("input_width"),"Linear width");same(row->at("output_width"),call.at("output_width"),"Linear output width");
            require(op->inputs.size()*tile==call.at("input_width").get<std::size_t>(),"Linear operand count mismatch");
            auto output_tile=call.at("output_tile").get<std::size_t>();
            if(output_tile==0){require(!active_linear_matrix,"Linear matrix sequences must not interleave");active_linear_matrix=matrix;active_linear_inputs=op->inputs;}
            require(active_linear_matrix && *active_linear_matrix==matrix && active_linear_inputs==op->inputs,"Linear sequence matrix/exact operand ids changed");
            require(output_tile==next_tile[matrix]++,"Linear calls must execute output0 then contiguous tiles");
            if((output_tile+1)*tile==call.at("output_width").get<std::size_t>()){active_linear_matrix.reset();active_linear_inputs.clear();}
            for(auto v:op->inputs)check(c.descriptors.at(v),call.at("input_level"),call.at("input_scale_log2"));
        } else {
            require(op->inputs.size()==2,"prepared pair requires two operands");
            require(call.at("tokens")==128 && call.at("tile")==256 && call.at("head_dim")==64 && call.at("baby")==8,"unsupported prepared attention geometry");
            if(a.callee=="bert.compact_qk")for(auto v:op->inputs)check(c.descriptors.at(v),call.at("input_level"),call.at("input_scale_log2"));
            else {check(c.descriptors.at(op->inputs[0]),call.at("score_input_level"),call.at("score_input_scale_log2"));check(c.descriptors.at(op->inputs[1]),call.at("value_input_level"),call.at("value_input_scale_log2"));}
        }
        ++c.counts[a.callee];
        if(a.callee=="bert.compact_qk" && call.value("pack_inputs",false))++c.counts["qk_pack_inputs"];
        if(a.callee=="bert.compact_av" || a.callee=="bert.compact_av_packed_partial")++c.counts["prepared_av_total"];
    }
    require(!c.require_qk_pack || c.counts["qk_pack_inputs"]>0,"required pack_inputs capability has no matching Runtime QK call");
    require(!c.require_av_partial || c.counts["bert.compact_av_packed_partial"]>0,"required packed_input2_partial capability has no matching Runtime AV call");
    c.source["attention_adapter_namespace"]=c.name;c.source["required_attention_capabilities"]={{"qk_pack_inputs",c.require_qk_pack},{"av_packed_partial",c.require_av_partial}};
    require(used.size()==calls.size(),"prepared manifest has missing/unused plan calls");
    require(c.counts["native_boot"]>0,"at least one native Boot required");
    for(const auto &[matrix,count]:next_tile) {
        const auto &m=c.manifests.at("bert.compact_linear");auto row=std::find_if(m.at("matrices").begin(),m.at("matrices").end(),[&](const auto &r){return r.at("matrix_id")==matrix;});
        require(count*row->at("tile").get<std::size_t>()==row->at("output_width").get<std::size_t>(),"incomplete Linear output tile sequence");
    }
}
inline void verify_sources(const Library &lib) {
    for(const auto &c:lib.plans) {
        same(hash(file_bytes(c.plan_path)),c.plan_sha,"plan source snapshot");same(hash(file_bytes(c.spec_path)),c.spec_sha,"spec source snapshot");same(hash(file_bytes(c.fixture_path)),c.fixture_sha,"fixture source snapshot");
        for(const auto &[callee,path]:c.manifest_paths) {
            std::string digest;auto m=read_json(path,&digest);same(m,c.manifests.at(callee),"prepared manifest source snapshot");
            if(callee=="bert.compact_linear")same(hash(file_bytes(resolve(path.parent_path(),m.at("weight_file")))),c.source.at("linear_weights_sha256"),"dense source snapshot");
        }
    }
}
inline Library load(const fs::path &schedule_path) {
    Library lib;auto input=read_json(schedule_path,&lib.schedule_sha);lib.schedule=input.at("schedule");
    require(input.at("plans").is_array() && input.at("plans").size()>=2 && input.at("plans").size()<=3,"two or three plans required");
    require(lib.schedule.is_array() && !lib.schedule.empty() && lib.schedule.size()<=64,"schedule requires 1..64 diagnostic runs");
    std::map<std::pair<std::string,std::uint64_t>,Json> namespace_contracts;
    Json common;std::uint64_t host_total=0;std::map<std::string,Json> attention_contracts;
    for(const auto &entry:input.at("plans")) {
        Candidate c;c.name=entry.at("name");require(!c.name.empty() && lib.names.emplace(c.name,lib.plans.size()).second,"duplicate or empty plan name");
        auto base=fs::absolute(schedule_path).parent_path();c.plan_path=resolve(base,entry.at("plan"));c.spec_path=resolve(base,entry.at("spec"));c.fixture_path=resolve(base,entry.at("fixture"));
        if(entry.contains("bundle") && !entry.at("bundle").is_null())c.bundle=resolve(base,entry.at("bundle"));
        c.fixture=read_json(c.fixture_path,&c.fixture_sha);c.raw_plan=read_json(c.plan_path,&c.plan_sha);c.raw_spec=read_json(c.spec_path,&c.spec_sha);
        require(c.fixture.value("format_version",0)==1,"unsupported fixture version");
        c.require_qk_pack=prepared_qk_requires_pack_inputs(c.fixture);c.require_av_partial=prepared_av_requires_packed_partial(c.fixture);
        c.loaded=fhegpu::RuntimePlanReader::read_file(c.plan_path);c.spec=fhegpu::OperatorSpecReader::read_file(c.spec_path);
        same(c.plan_sha,c.loaded.source_sha256,"plan source changed during load");same(c.spec_sha,c.spec.source_sha256,"spec source changed during load");
        c.requirements=fhegpu::PlanVerifier::verify(c.loaded.plan,c.spec,false);
        require(!c.loaded.plan.waterline && !c.fixture.contains("waterline"),"waterline is not supported by retained multi-plan Host caches");
        require(c.loaded.plan.target.world_size==1 && c.loaded.plan.target.device_counts==std::vector<int>{1},"multi-plan supports exactly one rank/device");
        require(c.spec.spec.boot_profiles.size()==1 && c.raw_spec.at("boot_profiles")[0].at("implementation")=="native","exactly one complete native Boot profile is supported");
        require(c.loaded.plan.plaintext_bundle.has_value()==c.bundle.has_value(),"bundle path/reference presence mismatch");
        for(const auto &d:c.raw_plan.at("values"))require(c.descriptors.emplace(id(d.at("id")),d).second,"duplicate value id");
        c.source={{"name",c.name},{"plan",c.plan_path.string()},{"plan_sha256",c.plan_sha},{"spec",c.spec_path.string()},{"spec_sha256",c.spec_sha},{"fixture",c.fixture_path.string()},{"fixture_sha256",c.fixture_sha},{"standard_plan_verified",true}};
        audit_data(c);audit_calls(c,lib,namespace_contracts);
        Json target=c.raw_plan.at("target");target.erase("operator_spec");
        Json initial=Json::array(),final=Json::array();for(auto v:c.loaded.plan.external_inputs)initial.push_back(c.descriptors.at(v));
        for(auto v:c.loaded.plan.final_outputs)final.push_back(desc_contract(c.descriptors.at(v)));
        require(final.size()==3,"exactly three final ciphertext tiles required");
        for(const auto &d:final)require(d.at("kind")=="ciphertext","final artifacts must be ciphertexts");
        Json signature={{"model_seed",c.fixture.at("model").at("seed")},{"model_config",c.fixture.at("model").at("config")},{"context",c.fixture.at("context")},{"native_boot",c.fixture.at("native_boot")},{"native_boot_profiles",c.fixture.value("native_boot_profiles",Json::object())},{"context_spec",c.raw_spec.at("context")},{"rescale_mode",c.raw_spec.at("rescale_mode")},{"operators",c.raw_spec.at("operators")},{"levels",c.raw_spec.at("levels")},{"boot_profiles",c.raw_spec.at("boot_profiles")},{"target",target},{"initial",initial},{"final",final},{"original_input",c.source.at("actual_original_input_sha256")},{"packed_input",c.source.at("packed_input_f64le_sha256")},{"parameters",c.fixture.at("model").at("weight_sha256")},{"linear_weights",c.source.at("linear_weights_sha256")},{"linear_contracts",c.manifests.at("bert.compact_linear")},{"periodic_encoding",c.fixture.value("compact_periodic_encoding",false)}};
        // File location is not part of a physical Linear contract.
        signature["linear_contracts"].erase("weight_file");
        if(lib.plans.empty())common=signature;else {
            for(auto it=signature.begin();it!=signature.end();++it)same(it.value(),common.at(it.key()),it.key());
            // SHA plus explicit packed comparisons prevents a misleading hash-only compatibility claim.
            require(c.clear_inputs==lib.plans.front().clear_inputs,"packed clear input values mismatch");
        }
        for(const auto &[callee,m]:c.manifests)if(callee!="bert.compact_linear") {
            auto old=attention_contracts.find(callee);
            if(old==attention_contracts.end())attention_contracts.emplace(callee,m);
            else if(old->second!=m)lib.mixed_attention_contracts=true;
        }
        lib.requirements.capabilities.insert(lib.requirements.capabilities.end(),c.requirements.capabilities.begin(),c.requirements.capabilities.end());
        lib.requirements.keys.insert(lib.requirements.keys.end(),c.requirements.keys.begin(),c.requirements.keys.end());
        for(const auto &key:c.requirements.keys) {
            if(key.kind==fhegpu::KeyKind::Relin)lib.need_relin=true;
            if(key.kind==fhegpu::KeyKind::Galois){require(key.rotation_step.has_value(),"rotation key missing step");lib.ordinary_rotations.insert(*key.rotation_step);}
            require(key.kind!=fhegpu::KeyKind::Secret,"secret-key runtime requirements forbidden");
        }
        if(c.manifests.count("bert.compact_qk") || c.manifests.count("bert.compact_av"))lib.need_relin=true;
        c.source["planned_counts"]=c.counts;c.source["estimated_host_cache_bytes"]=c.estimated_host_cache_bytes;host_total+=c.estimated_host_cache_bytes;
        lib.plans.push_back(std::move(c));
    }
    if(!lib.prepared_capabilities.empty()) {
        auto initial=lib.plans.front().manifests;initial.erase("bert.compact_linear");
        for(const auto &c:lib.plans){auto contracts=c.manifests;contracts.erase("bert.compact_linear");if(contracts!=initial)lib.mixed_attention_contracts=true;}
    }
    if(lib.mixed_attention_contracts)for(const auto &c:lib.plans)require(positive_config_integer(c.fixture.at("model").at("config"),"layers")==1,"mixed attention contracts are restricted to layers=1");
    for(const auto &run:lib.schedule) {require(lib.names.count(run.at("name").get<std::string>()),"unknown scheduled plan");auto phase=run.at("phase").get<std::string>();require(phase=="warmup" || phase=="diagnostic" || phase=="measured","invalid schedule phase");}
    std::sort(lib.requirements.keys.begin(),lib.requirements.keys.end());lib.requirements.keys.erase(std::unique(lib.requirements.keys.begin(),lib.requirements.keys.end()),lib.requirements.keys.end());
    lib.report={{"compatible",true},{"schedule_sha256",lib.schedule_sha},{"plans",Json::array()},{"schedule",lib.schedule},{"union_requirements",{{"capabilities_count",lib.requirements.capabilities.size()},{"keys_count",lib.requirements.keys.size()},{"ordinary_rotation_steps",lib.ordinary_rotations},{"prepared_direct_rotation_steps",lib.prepared_rotations},{"needs_relin",lib.need_relin}}},{"estimated_all_retained_host_cache_bytes",host_total},{"host_cache_scope","sum of every plan's Host Encode cache; no Device constants cached; uint64 RNS coefficient storage estimate, allocator/object overhead excluded"},{"data_trust_scope","local fixture/manifest trust roots; dense fields and original f64 input rehashed; LN SHA values compared, no raw LN arrays available to rehash"}};
    lib.report["attention_adapter_namespace_policy"]="per-plan independent QK/AV engines and GPU uploads; one shared main CPU key material/context/Boot/input setup; Linear contracts and engine remain shared";
    lib.report["mixed_attention_contracts"]=lib.mixed_attention_contracts;lib.report["prepared_capabilities_union"]=lib.prepared_capabilities;lib.report["cpu_context_construction_calls"]=lib.cpu_context?1:0;
    for(const auto &c:lib.plans)lib.report["plans"].push_back(c.source);
    return lib;
}
} // namespace multi_plan
