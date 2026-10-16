#include "imagenet_hevm_plan.h"
#include "gpu_ckks_runtime.h"
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>

using namespace imagenet_dsl;
using namespace poseidon::benchmark::resnet50_gpu;
int main(int argc,char **argv){try{
    require(argc==8 || argc==9,"usage: imagenet_dsl MODEL HEVM CONSTANTS INPUTS PAYLOAD REPORT MAX_OPS [--metadata-only]");
    std::cout<<std::unitbuf;
    const int model=std::stoi(argv[1]);Plan plan(argv[2]);auto config=make_resnet50_gpu_config();
    const auto constants=jsonlines(argv[3]),inputs=jsonlines(argv[4]);
    require(constants.size()==plan.plain_uses.size(),"constant count/HEVM mismatch");
    require(inputs.size()==149 && inputs[0].at("format")=="POSEIDON_IMAGENET_NATIVE_INPUTS_V1" && inputs[0].at("count")==148,"native input manifest");
    std::ifstream payload_file(argv[5],std::ios::binary|std::ios::ate);require(bool(payload_file),"open native payload");
    const auto payload_bytes=std::uint64_t(payload_file.tellg());
    for(std::size_t i=0;i<constants.size();++i){const auto &r=constants[i];require(r.at("index")==i,"constant order");
        if(r.at("op")=="encode_vector")require(r.at("payload_offset").get<std::uint64_t>()+16+r.at("payload_compressed_bytes").get<std::uint64_t>()<=payload_bytes,"constant payload range");
        else require(r.at("op")=="encode_scalar","constant kind");}
    std::size_t layouts=0,ones=0;
    for(std::size_t i=0;i<148;++i){const auto &r=inputs[i+1];require(r.at("register")==i && r.at("q")==32 && hex(r.at("scale_bits"))==0x4270000000000000ULL,"input register Q32/40");
        if(r.at("kind")=="layout"){payload(payload_file,r);++layouts;}
        else {require(r.at("kind")=="encrypted_one" && hex(r.at("value_bits"))==0x3ff0000000000000ULL,"encrypted ONE");++ones;}}
    require(layouts==147 && ones==1,"147 native input layouts plus ONE");
    for(std::size_t i=0;i<plan.ops.size();++i){const auto o=plan.ops[i];if(o.code==0){require(o.lhs<constants.size(),"constant index");
        require(constants[o.lhs].at("scale_bits").get<std::uint64_t>()==std::bit_cast<std::uint64_t>(plan.exact.at(i)),"Encode constant/f64 binding");}}
    plan.metadata(config.q_moduli,model);
    std::size_t first_boot=0,actions=0;
    for(auto o:plan.ops)if(o.code!=0xffffffff){++actions;if(o.code==10 && !first_boot)first_boot=actions;}
    std::cout<<"IMAGENET_DSL_PREFLIGHT PASS model="<<model<<" inputs=148 outputs=1000 raw_ops="<<plan.ops.size()
        <<" actions="<<actions<<" boots="<<plan.boots<<" first_boot_action="<<first_boot<<" keys_generated=false gpu_executed=false\n";
    if(argc==9){require(std::string(argv[8])=="--metadata-only","unknown option");return 0;}
    for(auto key:{"POSEIDON_RESCALE_META_BITS","POSEIDON_RESCALE_MODEL_BITS","POSEIDON_DSL_SCALE_SHIFT","POSEIDON_GPU_DNUM"})require(!std::getenv(key),std::string("must be unset: ")+key);
    const char *mode=std::getenv("POSEIDON_GPU_LINEAR_TRANSFORM_MODE");require(mode && std::string(mode)=="no_hoist","native/DSL no_hoist required");
    require(!std::ifstream(argv[6]).good(),"refusing to overwrite result");
    const std::size_t limit=std::stoull(argv[7]);const auto start=std::chrono::steady_clock::now();
    GpuCkksRuntime runtime(config);runtime.initialize_inference_evaluation_keys();runtime.initialize_bootstrap();
    using CT=GpuCkksRuntime::DeviceCiphertext;using PT=GpuCkksRuntime::DevicePlaintext;
    std::vector<std::unique_ptr<CT>> c(plan.cipher_uses.size());std::vector<std::unique_ptr<PT>> p(plan.plain_uses.size());
    auto cu=plan.cipher_uses,pu=plan.plain_uses;
    for(std::size_t i=0;i<148;++i){const auto &r=inputs[i+1];c[i]=std::make_unique<CT>(r.at("kind")=="layout"?runtime.encrypt(payload(payload_file,r)):runtime.encrypt_constant(1.));
        require(c[i]->meta.q_count==32 && std::bit_cast<std::uint64_t>(c[i]->meta.scale)==0x4270000000000000ULL,"actual input Q/scale");if(!cu[i])c[i].reset();}
    std::size_t executed=0,boots=0,exact_vector=0,exact_scalar=0,exact_product=0,metadata_moves=0,metadata_copies=0;
    std::vector<std::size_t> plain_constant(p.size());std::unique_ptr<CT> product_reference;
    std::uint32_t product_register=UINT32_MAX;
    for(std::size_t i=0;i<plan.ops.size();++i){const auto o=plan.ops[i];if(o.code==0xffffffff)continue;
        if(limit && executed==limit){runtime.synchronize();std::cout<<"IMAGENET_DSL_TRUNCATED PASS actions="<<executed<<" boots="<<boots
            <<" exact_vector="<<exact_vector<<" exact_scalar="<<exact_scalar<<" exact_product="<<exact_product<<" gpu_executed=true\n";return 0;}
        try {
            Plan::sources(o,[&](bool plain,std::uint32_t r){require(plain?bool(p.at(r)):bool(c.at(r)),"released input used");});
            if(o.code==0){const auto &k=constants.at(o.lhs);const double scale=plan.exact.at(i);const std::size_t q=1+(o.rhs>>10);
                p[o.dst]=std::make_unique<PT>(k.at("op")=="encode_scalar"?runtime.dsl_encode_scalar(q,f64(k.at("value_bits").get<std::uint64_t>()),scale):runtime.dsl_encode(q,payload(payload_file,k),scale));
                plain_constant[o.dst]=o.lhs;
                require(p[o.dst]->meta.q_count==q && std::bit_cast<std::uint64_t>(p[o.dst]->meta.scale)==std::bit_cast<std::uint64_t>(scale),"actual Encode metadata");
                if(!pu[o.dst])p[o.dst].reset();
            } else {
                auto &a=*c.at(o.lhs);CT result;
                switch(o.code){
                case 13:
                    if(cu[o.lhs]==1){result=std::move(a);result.meta.scale=plan.exact.at(i);++metadata_moves;}
                    else {result=runtime.dsl_unary(13,a,0,plan.exact.at(i));++metadata_copies;}
                    break;
                case 6:result=runtime.add(a,*c.at(o.rhs));break;
                case 7:result=runtime.dsl_add_plain(a,*p.at(o.rhs));break;
                case 9:result=runtime.multiply_plain_preencoded(a,*p.at(o.rhs));break;
                case 15:result=runtime.dsl_raw_multiply(a,*c.at(o.rhs));break;
                default:result=runtime.dsl_unary(o.code,a,o.rhs,o.code==13?plan.exact.at(i):0);break;
                }
                c[o.dst]=std::make_unique<CT>(std::move(result));
                const auto expected=plan.expected_cipher[o.dst];
                const auto observed_bits=std::bit_cast<std::uint64_t>(c[o.dst]->meta.scale);
                require(c[o.dst]->meta.q_count==expected.q && c[o.dst]->size()==expected.components,"actual/plan Q or ciphertext components differ");
                require(observed_bits==std::bit_cast<std::uint64_t>(expected.scale),
                    "actual/plan f64 scale differs: observed_bits="+std::to_string(observed_bits)+" expected_bits="+std::to_string(std::bit_cast<std::uint64_t>(expected.scale)));
                if(o.code==9){const auto &k=constants.at(plain_constant[o.rhs]);const bool scalar=k.at("op")=="encode_scalar";
                    if((scalar && !exact_scalar) || (!scalar && !exact_vector)){
                        auto reference=scalar?runtime.multiply_plain_scalar(a,f64(k.at("value_bits").get<std::uint64_t>()),p[o.rhs]->meta.scale):runtime.multiply_plain(a,payload(payload_file,k),p[o.rhs]->meta.scale);
                        runtime.dsl_require_equal(*c[o.dst],reference);if(scalar)++exact_scalar;else ++exact_vector;
                    }}
                if(o.code==15 && !exact_product && !product_reference){product_reference=std::make_unique<CT>(runtime.multiply_relinearize(a,*c.at(o.rhs)));product_register=o.dst;}
                if(o.code==16 && o.lhs==product_register && product_reference){runtime.dsl_require_equal(*c[o.dst],*product_reference);product_reference.reset();++exact_product;}
                if(o.code==10){++boots;std::cout<<"DSL_NATIVE_BOOTSTRAP call="<<boots<<" action="<<(executed+1)<<" q="<<c[o.dst]->meta.q_count<<" logscale="<<std::log2(c[o.dst]->meta.scale)<<'\n';}
                if(!cu[o.dst])c[o.dst].reset();
            }
            Plan::sources(o,[&](bool plain,std::uint32_t r){auto &use=plain?pu:cu;require(use[r]>0,"liveness underflow");if(--use[r]==0){if(plain)p[r].reset();else c[r].reset();}});
        }catch(const std::exception &e){throw std::runtime_error("raw="+std::to_string(i)+" opcode="+std::to_string(o.code)+" action="+std::to_string(executed+1)+": "+e.what());}
        ++executed;if(executed%1000==0)std::cout<<"DSL_PROGRESS actions="<<executed<<" boots="<<boots<<'\n';
    }
    require(boots==plan.boots,"runtime bootstrap count");std::vector<double> logits;
    for(auto r:plan.returns){require(c.at(r) && c[r]->meta.q_count==4,"actual head Q");auto value=runtime.decrypt(*c[r]);logits.push_back(value.at(0).real());require(std::isfinite(logits.back()),"nonfinite logit");if(--cu[r]==0)c[r].reset();}
    runtime.synchronize();for(auto n:cu)require(!n,"unreleased ciphertext");for(auto n:pu)require(!n,"unreleased plaintext");
    const auto prediction=std::distance(logits.begin(),std::max_element(logits.begin(),logits.end()));
    Json report={{"status","COMPLETE"},{"model",model},{"prediction",prediction},{"label",65},{"logits",logits},{"bootstrap_calls",boots},{"gpu_executed",true},
        {"exact_vector",exact_vector},{"exact_scalar",exact_scalar},{"exact_product",exact_product},
        {"metadata_moves",metadata_moves},{"metadata_copies",metadata_copies},
        {"HEVM_fnv64",plan.hash},{"seconds",std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()}};
    std::ofstream out(argv[6]);out<<report.dump(2)<<'\n';require(bool(out),"write result");
    std::cout<<"IMAGENET_DSL_COMPLETE model="<<model<<" prediction="<<prediction<<" label=65 boots="<<boots<<'\n';return 0;
}catch(const std::exception &e){std::cerr<<"IMAGENET_DSL_ERROR "<<e.what()<<'\n';return 1;}}
