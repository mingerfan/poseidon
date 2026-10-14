#define main existing_gpu_api_regression_main
#include "poseidon/tests/runtime_api/poseidon_gpu_api_test.cpp"
#undef main

std::size_t key_storage_bytes(const poseidon::gpu::GpuEvaluationKeyData &keys) {
    std::size_t bytes=0;
    for (const auto &field:keys.fields_) bytes+=field.size()*sizeof(poseidon::gpu::GpuWord);
    bytes+=(keys.galois_key_q0_ptrs.size()+keys.galois_key_p0_ptrs.size()+
            keys.galois_key_q1_ptrs.size()+keys.galois_key_p1_ptrs.size())*sizeof(const poseidon::gpu::GpuWord *);
    return bytes;
}

int main() {
    try {
        int count=0;require(cudaGetDeviceCount(&count)==cudaSuccess && count==1,"select exactly one CUDA device for this boundary test");
        kDefaultScaleLog2=40;
        poseidon::ParametersLiteral parameters(CKKS,14,13,40,0,0,
            poseidon::Modulus(0),{},{},poseidon::sec_level_type::none);
        parameters.set_log_modulus(std::vector<std::uint32_t>(50,30),
                                   std::vector<std::uint32_t>(8,30));
        poseidon::PoseidonContext context(parameters);
        require(context.parameters_literal()->degree()==16384 && context.parameters_literal()->q().size()==50 && context.parameters_literal()->p().size()==8,"requested context parameters mismatch");
        std::cout<<"SCOPE real GPU Q50/P8 key backing/Rotate/Relin parameter regression; not full Qwen/model execution; sec_level_type=none, N=16384 Q=50x30 P=8x30 input_scale=40 product_scale=80 no_Rescale=true\n";
        RmmPoolScope pool(0);
        poseidon::KeyGenerator generator(context);
        auto relin=std::make_shared<poseidon::RelinKeys>();generator.create_relin_keys(*relin);
        auto rotations=std::make_shared<poseidon::GaloisKeys>();generator.create_galois_keys(std::vector<int>{1,2},*rotations);
        const auto spec=make_operator_spec(context);const auto device=device_place();
        const auto elt1=context.crt_context()->galois_tool()->get_elt_from_step(1);
        const auto elt2=context.crt_context()->galois_tool()->get_elt_from_step(2);
        std::size_t relin_bytes=0,galois_one_bytes=0,galois_two_bytes=0;
        {
            auto r=poseidon::gpu::GpuUploader::upload_relin_keys(*relin,0,5);
            auto g=poseidon::gpu::GpuUploader::upload_galois_keys(*rotations,0,5,{elt1});
            require(r.make_const_view(5).storage_q_count==50 && g.make_const_view(5).storage_q_count==50 && r.meta.decomposition_count==7 && g.meta.decomposition_count==7 && r.meta.p_count==8 && g.meta.p_count==8,
                    "uploader did not retain one full stored Q prefix");
            relin_bytes=key_storage_bytes(r);galois_one_bytes=key_storage_bytes(g);
            require(cudaDeviceSynchronize()==cudaSuccess,"direct key upload failed");
        }
        {
            auto g=poseidon::gpu::GpuUploader::upload_galois_keys(*rotations,0,50,{elt1,elt2});
            require(g.make_const_view(50).storage_q_count==50 && g.meta.decomposition_count==7 && g.meta.p_count==8,
                    "combined Galois key upload changed full Q50/P8 backing");
            galois_two_bytes=key_storage_bytes(g);require(cudaDeviceSynchronize()==cudaSuccess,"combined key upload failed");
        }
        PoseidonGpuApi api(kContextId,context,0,relin,rotations);
        fhegpu::PlanRequirements requirements;
        for (int level:{4,24,49}) {
            requirements.keys.push_back({fhegpu::KeyKind::Relin,device,std::nullopt,level});
            requirements.keys.push_back({fhegpu::KeyKind::Galois,device,1,level});
        }
        const auto before=pool.allocated_bytes();
        api.preflight(kPlanSha,false,make_target(spec),spec.spec,requirements);
        require(pool.allocated_bytes()-before==relin_bytes+galois_one_bytes,
                "preflight uploaded more than one backing per key family");
        const auto allocated=pool.allocated_bytes(),total=pool.total_allocated_bytes();
        api.preflight(kPlanSha,false,make_target(spec),spec.spec,requirements);
        require(pool.allocated_bytes()==allocated && pool.total_allocated_bytes()==total,"repeat preflight allocated duplicate keys");
        auto added=requirements;added.keys.push_back({fhegpu::KeyKind::Galois,device,2,4});
        api.preflight(kPlanSha,false,make_target(spec),spec.spec,added);
        require(pool.allocated_bytes()-before==relin_bytes+galois_two_bytes,"new rotation did not replace the owning Galois backing");
        const auto expanded=pool.allocated_bytes(),expanded_total=pool.total_allocated_bytes();
        api.preflight(kPlanSha,false,make_target(spec),spec.spec,requirements);
        require(pool.allocated_bytes()==expanded && pool.total_allocated_bytes()==expanded_total,"older run lost/reuploaded the Galois union");
        std::cout<<"PASS single key backing relin_bytes="<<relin_bytes<<" galois_one_bytes="<<galois_one_bytes
                 <<" galois_two_bytes="<<galois_two_bytes<<" total_three_keys_bytes="<<relin_bytes+galois_two_bytes
                 <<" active_levels=4,24,49 active_q=5,25,50 stored_q=50 p_count=8 decomposition_blocks=7 no_duplicate_backing=true\n";
        poseidon::PublicKey public_key;generator.create_public_key(public_key);
        poseidon::CKKSEncoder encoder(context);poseidon::Encryptor encryptor(context,public_key);
        poseidon::Decryptor decryptor(context,generator.secret_key());poseidon::EvaluatorCkksSoftware cpu(context);
        std::vector<double> values(context.parameters_literal()->slot());
        for(std::size_t i=0;i<values.size();++i)values[i]=.125*std::sin(i*.071)+.025*std::cos(i*.019);
        const auto decode=[&](const poseidon::Ciphertext &cipher) {
            poseidon::Plaintext p;decryptor.decrypt(cipher,p);std::vector<std::complex<double>> slots;encoder.decode(p,slots);return slots;
        };
        const auto compare=[&](const PoseidonGpuValue &value,const poseidon::Ciphertext &expected,const char *label,int rotation) {
            auto host=transfer_value(api,900,value,fhegpu::ValueKind::Ciphertext,device,host_place());
            auto actual=decode(host.host_ciphertext()),reference=decode(expected);double maximum=0,dense_maximum=0;
            require(actual.size()==8192 && reference.size()==8192,"native CPU/GPU comparison did not cover all complex slots");
            require(expected.scale()==std::ldexp(1.,80) && host.host_ciphertext().scale()==expected.scale(),
                    "key operation retagged or changed native product scale");
            for(std::size_t i=0;i<reference.size();++i) {
                const auto error=std::abs(actual[i]-reference[i]);
                require(std::isfinite(error),std::string(label)+" returned nonfinite complex values");
                maximum=std::max(maximum,error);
                const auto source=(i+rotation)%values.size();
                const auto dense_error=std::abs(actual[i]-std::complex<double>(values[source]*values[source],0));
                require(std::isfinite(dense_error),std::string(label)+" returned nonfinite dense-oracle error");
                dense_maximum=std::max(dense_maximum,dense_error);
            }
            require(maximum<1e-4,std::string(label)+" active-level numerical mismatch");
            require(dense_maximum<1e-4,std::string(label)+" independent dense square/rotation mismatch");
            std::cout<<"PASS "<<label<<" q="<<value.device_ciphertext().meta.q_count<<" slots_checked="<<reference.size()
                     <<" native_scale="<<expected.scale()<<" max_abs_error="<<maximum<<" independent_dense_max_abs="<<dense_maximum<<'\n';
        };
        for(int level:{4,24,49}) {
            poseidon::Plaintext plain;encoder.encode(values,context.crt_context()->parms_id_map().at(level),std::ldexp(1.,40),plain);
            poseidon::Ciphertext cipher;encryptor.encrypt(plain,cipher);
            auto input=transfer_value(api,800,PoseidonGpuValue::from_host_ciphertext(cipher),fhegpu::ValueKind::Ciphertext,host_place(),device);
            poseidon::Ciphertext cpu_mult,cpu_relin,cpu_rotate1,cpu_rotate3;
            cpu.multiply(cipher,cipher,cpu_mult);cpu.relinearize(cpu_mult,cpu_relin,*relin);
            require(cipher.scale()==std::ldexp(1.,40) && cpu_mult.scale()==std::ldexp(1.,80) && cpu_relin.scale()==cpu_mult.scale(),
                    "native CPU reference changed actual scale");
            fhegpu::ComputeOp multiply; multiply.kind=fhegpu::ComputeKind::MulCC;multiply.place=device;
            auto product=api.compute(multiply,{input,input});
            fhegpu::ComputeOp rel;rel.kind=fhegpu::ComputeKind::Relinearize;rel.place=device;
            auto result=api.compute(rel,{product});compare(result,cpu_relin,"Relinearize",0);
            cpu.rotate(cpu_relin,cpu_rotate1,1,*rotations);cpu.rotate(cpu_rotate1,cpu_rotate3,2,*rotations);
            fhegpu::ComputeOp rotate;rotate.kind=fhegpu::ComputeKind::Rotate;rotate.place=device;rotate.attrs=fhegpu::RotateAttrs{1};
            auto simple=api.compute(rotate,{result});compare(simple,cpu_rotate1,"Rotate",1);
            rotate.attrs=fhegpu::RotateAttrs{3};
            auto rotated=api.compute(rotate,{result});compare(rotated,cpu_rotate3,"composite Rotate",3);
            rotate.attrs=fhegpu::RotateAttrs{1};rotate.reuse_input=0;
            auto reused=api.compute_reuse(rotate,std::move(result),{});compare(reused,cpu_rotate1,"reuse Rotate",1);
            api.drain();
        }
        return 0;
    } catch(const std::exception &error) {std::cerr<<"FAIL key backing test: "<<error.what()<<'\n';return 1;}
}
