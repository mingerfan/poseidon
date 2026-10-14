// Standalone exact-equivalence test, never linked into online inference.
#define POSEIDON_S2C_FIRST_RESNET20_LIBRARY
#include "bootstrap_network_precision.cpp"

int main(int argc,char **argv) {
    try {
        if(argc!=3 || std::string(argv[1])!="--unsafe-accuracy-only")
            throw std::invalid_argument("usage: conv_plain_batch_precision --unsafe-accuracy-only fixture.txt");
        std::cout<<std::unitbuf;
        auto parms=baseline_parameters("q50-fixed45",argv[2]);
        PoseidonFactory::get_instance()->set_device_type(DEVICE_SOFTWARE);
        auto ctx=PoseidonFactory::get_instance()->create_poseidon_context(parms);
        CKKSEncoder encoder(ctx);
        EvalModPoly poly(ctx,CosDiscrete,std::exp2(45),0,5,2,25,0,59);
        GpuBootstrapData::EvalModData plan;
        BaselineMetadata metadata{ctx,encoder,poly,parms_id_zero,0,plan,true};
        gpu_check_cuda(cudaSetDevice(0),"Conv batch test device");
        BaselinePool pool(std::size_t(2)<<30);
        auto secret=nonzero_secret(ctx);
        KeyGenerator generator(ctx,secret);PublicKey pub;generator.create_public_key(pub);
        Encryptor encryptor(ctx,pub,secret);
        GpuParameterData parameters(ctx,0,{8,9,10});GpuEvaluator eval(parameters);
        const auto q9=parameters.get_level_by_q_count(9).parms_id;
        const auto q10=parameters.get_level_by_q_count(10).parms_id;
        std::vector<double> slots(32768),weights(32768);
        for(std::size_t i=0;i<slots.size();++i)slots[i]=(int(i%103)-51)/128.0;
        Plaintext plain;encoder.encode(slots,q10,std::exp2(40),plain);
        Ciphertext encrypted;encryptor.encrypt(plain,encrypted);
        auto high=GpuUploader::upload_ciphertext(encrypted,0);
        CT input,snapshot;eval.drop_modulus(high,input,q9);eval.drop_modulus(input,snapshot,q9);
        GpuGaloisKeysData no_keys;
        using Runtime=network_probe::Runtime;
        std::size_t exact_words=0,checks=0;
        const double scale=parms.q().at(8).value();
        // All lengths include zero/negative coefficients and irregular masks.
        // 1,4,5,8,9,10 cover tails and both full-batch boundaries.
        for(std::size_t length=1;length<=10;++length)for(bool alias:{false,true}) {
            Runtime runtime(metadata,&eval,&no_keys);runtime.plain_batch_enabled=true;
            CT source;eval.drop_modulus(input,source,q9);
            auto x=runtime.device(std::move(source));
            auto evaluate=[&](bool capture) {
                if(capture)runtime.begin_prepared_capture();else runtime.begin_prepared_replay();
                Runtime::DeviceCiphertext sum,shared;
                for(std::size_t term=0;term<length;++term) {
                    for(std::size_t i=0;i<weights.size();++i)
                        weights[i]=(term==1 || i%7==0)?0.0:(int((i+term)%13)-6)/8.0;
                    if(term==0)sum=runtime.multiply_plain(x,weights,scale);
                    else runtime.multiply_plain_accumulate(x,weights,scale,sum);
                    if(alias && term==0)shared=sum;
                }
                runtime.validate(sum);
                if(shared.gpu)runtime.validate(shared);
                if(capture)runtime.finish_prepared_capture();else runtime.finish_prepared_replay();
                CT reference;eval.multiply_plain(*x.gpu,*runtime.prepared_plains[0].data,reference);
                if(shared.gpu) {
                    exact_words+=require_exact_ciphertexts(ctx,*shared.gpu,reference,"Conv COW alias");++checks;
                }
                for(std::size_t term=1;term<length;++term)
                    eval.multiply_plain_accumulate(*x.gpu,*runtime.prepared_plains[term].data,reference);
                exact_words+=require_exact_ciphertexts(ctx,*sum.gpu,reference,"Conv batch boundary");++checks;
                const auto &stats=runtime.plain_batch_stats;
                if(stats.pending_terms || (!capture && stats.exact_checks))
                    throw std::runtime_error("pending work or online observer in batch test");
                if(!alias && (stats.terms!=length || stats.calls!=(length+3)/4))
                    throw std::runtime_error("batch test did not exercise expected grouping");
            };
            evaluate(true);evaluate(false);
        }
        // Accumulating API: mixed read-only Q prefixes and output aliasing.
        std::vector<GpuPlaintextData> coefficients;
        for(double coefficient:{0.0,-1.25,2.5,-0.75}) {
            Plaintext p;encoder.encode(coefficient,q9,1.0,p);
            coefficients.push_back(GpuUploader::upload_plaintext(p,0));
        }
        for(std::size_t count=1;count<=4;++count) {
            CT fused,reference;eval.drop_modulus(input,fused,q9);eval.drop_modulus(input,reference,q9);
            std::vector<const CT *> xs;
            std::vector<const GpuPlaintextData *> ps;
            for(std::size_t i=0;i<count;++i) {
                // The last input aliases output; all other inputs are immutable.
                xs.push_back(i+1==count?&fused:&high);ps.push_back(&coefficients[i]);
                eval.multiply_plain_accumulate(input,coefficients[i],reference);
            }
            eval.multiply_plain_sum_q_prefix(xs,ps,fused,true);
            exact_words+=require_exact_ciphertexts(ctx,fused,reference,"Conv accumulating API");++checks;
        }
        int rejected=0;
        auto reject=[&](auto operation) {
            try {operation();}catch(const std::invalid_argument &) {++rejected;return;}
            throw std::runtime_error("Conv accumulating API accepted invalid destination");
        };
        CT empty,wrong;eval.drop_modulus(input,wrong,q9);
        reject([&]{eval.multiply_plain_sum_q_prefix({&input},{&coefficients[0]},empty,true);});
        wrong.meta.scale*=2;
        reject([&]{eval.multiply_plain_sum_q_prefix({&input},{&coefficients[0]},wrong,true);});
        wrong.meta.scale/=2;wrong.meta.is_ntt_form=false;
        reject([&]{eval.multiply_plain_sum_q_prefix({&input},{&coefficients[0]},wrong,true);});
        eval.drop_modulus(input,wrong,parameters.get_level_by_q_count(8).parms_id);
        reject([&]{eval.multiply_plain_sum_q_prefix({&input},{&coefficients[0]},wrong,true);});
        exact_words+=require_exact_ciphertexts(ctx,input,snapshot,"Conv immutable source");++checks;
        std::cout<<"CONV_PLAIN_BATCH_EXACT result=PASS lengths=1..10 capture_and_replay=true"
            <<" zero_negative_masks=true copy_on_write=true mixed_q_prefixes=true alias_safe=true"
            <<" same_ciphertext=true checks="<<checks<<" exact_residues="<<exact_words
            <<" rejected="<<rejected<<" security_approved=false\n";
        return 0;
    }catch(const std::exception &e) {std::cerr<<"ERROR "<<e.what()<<'\n';return 1;}
}
