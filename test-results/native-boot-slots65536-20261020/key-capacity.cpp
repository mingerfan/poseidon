// CPU-only structural audit: exact DFT rotation indices and one real key shape.
#include "poseidon/advance/homomorphic_dft.h"
#include "poseidon/keygenerator.h"
#include "poseidon/parameters_literal.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <set>
#include <iostream>
using Json=nlohmann::json;
void save(const char *path,const Json &result) {
    std::ofstream stream(path);stream.exceptions(std::ios::badbit|std::ios::failbit);
    stream<<result.dump(2)<<'\n';
}
std::uint64_t packed_bytes(const poseidon::KSwitchKeys &keys) {
    std::uint64_t result=0;
    for(const auto &row:keys.data())for(const auto &key:row) {
        const auto &cipher=key.data();
        for(const auto &poly:cipher.polys())
            result+=std::uint64_t(poly.poly_degree())*
                (poly.rns_num_q()+poly.rns_num_p())*sizeof(std::uint32_t);
    }
    return result;
}
int main(int argc,char **argv) {
    if(argc!=2)return 2;
    try {
        constexpr int log_n=17,log_slots=16,slots=1<<log_slots;
        Json result={{"polynomial_degree",1<<log_n},{"complex_slots",slots},
                     {"scope","CPU-only exact key-backing lower bound; no GPU allocations or inference"}};
        std::set<int> steps{0};
        for(const auto type:{poseidon::encode,poseidon::decode}) {
            poseidon::HomomorphicDFTMatrixLiteral literal(type,log_n,log_slots,49,
                {1,1,1},true,1.0,false,1);
            auto matrices=literal.gen_matrices();
            Json rows=Json::array();
            for(const auto &matrix:matrices) {
                const auto n1=poseidon::find_best_bsgs_ratio(matrix,slots,1);
                std::set<int> local;
                for(const auto &diagonal:matrix) {
                    const int giant=((diagonal.first/n1)*n1)&(slots-1);
                    const int baby=diagonal.first&(n1-1);
                    if(giant)local.insert(giant);
                    if(baby)local.insert(baby);
                }
                steps.insert(local.begin(),local.end());
                rows.push_back({{"diagonals",matrix.size()},{"n1",n1},
                                {"rotation_steps",local}});
            }
            result[type==poseidon::encode?"c2s":"s2c"]=rows;
            save(argv[1],result);
        }
        result["all_rotation_steps_including_conjugation_zero"]=steps;
        result["galois_key_count"]=steps.size();save(argv[1],result);
        std::cout<<"Required Galois keys: "<<steps.size()<<std::endl;
        poseidon::ParametersLiteral parameters(CKKS,log_n,log_slots,40,0,1,
            poseidon::Modulus(0),{},{},poseidon::sec_level_type::none);
        parameters.set_log_modulus(std::vector<std::uint32_t>(50,30),
                                   std::vector<std::uint32_t>(8,30));
        poseidon::PoseidonContext context(parameters);
        poseidon::KeyGenerator generator(context);
        poseidon::GaloisKeys single;
        generator.create_galois_keys(std::vector<std::uint32_t>{
            context.crt_context()->galois_tool()->get_elt_from_step(0)},single);
        poseidon::RelinKeys relin;generator.create_relin_keys(relin);
        const auto one=packed_bytes(single),rel=packed_bytes(relin);
        result["packed_bytes_per_galois_key"]=one;
        result["packed_bytes_all_galois_keys"]=one*steps.size();
        result["packed_bytes_relin_keys"]=rel;
        result["packed_key_backing_lower_bound_bytes"]=one*steps.size()+rel;
        result["packed_key_backing_lower_bound_gib"]=double(one*steps.size()+rel)/(1ULL<<30);
        result["excluded"]="DFT matrices, EvalMod plaintexts, GPU context parameters, pointer views, inputs/outputs and workspace";
        save(argv[1],result);return 0;
    }catch(const std::exception &error){std::cerr<<error.what()<<std::endl;return 1;}
}
