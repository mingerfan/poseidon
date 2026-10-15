// CPU loader/encoder gate for real compiler artifacts; never executes GPU ops.
#define POSEIDON_S2C_FIRST_RESNET20_LIBRARY
#define POSEIDON_S2C_FIRST_ENABLE_TIMING
#include "native_snapshot/Doc/analysis/resnet20_joint_chain_20260911/bootstrap_network_precision.cpp"
#include "hevm_native_rescale_groups.h"
#include "hevm_native_scale_sidecar.h"
#include <bit>
#include <fstream>
#include <iterator>
#include <set>

// Compile against the actual evaluator and ciphertext types, not the CPU mock.
void instantiate_strict_native_arithmetic(GpuEvaluator &eval,const CT &source,
    CT &destination,const poseidon::mgpu::HevmNativeRescaleGroup &group) {
    poseidon::mgpu::execute_hevm_native_rescale_group(eval,source,destination,group);
    poseidon::mgpu::reconcile_hevm_native_scale(source,destination);
}

std::string strict_read_file(const std::string &path) {
    std::ifstream file(path,std::ios::binary);
    if(!file)throw std::runtime_error("missing strict artifact: "+path);
    return {std::istreambuf_iterator<char>(file),{}};
}
std::uint64_t strict_read_u64(const std::string &data,std::size_t at,unsigned bytes=8) {
    if(at>data.size() || bytes>data.size()-at)throw std::runtime_error("truncated strict artifact");
    std::uint64_t value=0;
    for(unsigned i=0;i<bytes;++i)value|=std::uint64_t(static_cast<unsigned char>(data[at+i]))<<(8*i);
    return value;
}

int main(int argc,char **argv) {
    if(argc!=5)throw std::runtime_error("usage: fixture HEVM CST inputs.f64");
    const auto binary=strict_read_file(argv[2]);
    const auto scales=poseidon::mgpu::parse_hevm_native_scales(binary,strict_read_file(std::string(argv[2])+".native_scales"));
    const auto groups=poseidon::mgpu::parse_hevm_native_rescale_groups(binary,strict_read_file(std::string(argv[2])+".native_rescales"));
    auto header=strict_read_u64(binary,4,4),body=strict_read_u64(binary,header),nops=strict_read_u64(binary,header+8);
    if(strict_read_u64(binary,8)!=27 || strict_read_u64(binary,16)!=1)
        throw std::runtime_error("strict program requires 27 inputs and one result");
    for(std::size_t i=0;i<27;++i)
        if(strict_read_u64(binary,header+40+8*i)!=40 || strict_read_u64(binary,header+40+8*(27+i))!=31)
            throw std::runtime_error("strict input Q32/scale40 mismatch");
    std::ifstream cst(argv[3],std::ios::binary|std::ios::ate);
    if(!cst)throw std::runtime_error("missing CST");
    const auto cst_size=std::uint64_t(cst.tellg());cst.seekg(0);
    auto read_word=[&](){std::uint64_t word=0;cst.read(reinterpret_cast<char*>(&word),8);
        if(!cst)throw std::runtime_error("truncated CST");return word;};
    auto constants=read_word();
    if(constants>100000)throw std::runtime_error("invalid constant count");
    std::vector<std::pair<std::uint64_t,std::uint64_t>> offsets;
    std::uint64_t cursor=8;
    for(std::uint64_t i=0;i<constants;++i) {
        auto length=read_word();cursor+=8;
        if(!length || length>32768 || cursor>cst_size || length>(cst_size-cursor)/8)
            throw std::runtime_error("invalid CST vector span");
        offsets.emplace_back(cursor,length);cursor+=8*length;cst.seekg(cursor);
    }
    if(cursor!=cst_size)throw std::runtime_error("CST trailing bytes");
    for(auto scale:scales)if(scale.constant_index>=constants)throw std::runtime_error("Encode constant out of range");
    const auto inputs=strict_read_file(argv[4]);
    if(inputs.size()!=27*32768*8)throw std::runtime_error("input layouts size mismatch");
    for(std::size_t i=0;i<inputs.size();i+=8) {
        auto x=std::bit_cast<double>(strict_read_u64(inputs,i));
        if(!std::isfinite(x) || std::abs(x)>1)throw std::runtime_error("invalid application input slot");
    }
    std::size_t boot_count=0;
    for(std::size_t i=0;i<nops;++i) {
        auto at=header+body+8*i;
        if(strict_read_u64(binary,at,2)==10){++boot_count;
            if(strict_read_u64(binary,at+6,2)!=30)throw std::runtime_error("bootstrap output level mismatch");}
    }
    if(boot_count!=18)throw std::runtime_error("bootstrap count mismatch");
    std::string mode="--metadata-only",profile="q50-fixed45",dataset="grid";
    char *args[]={argv[0],mode.data(),profile.data(),dataset.data(),argv[1]};
    return run_bootstrap_baseline(5,args,[&](const BaselineMetadata &m) {
        std::set<std::uint64_t> tested_levels;
        std::size_t checks=0;
        for(auto scale:scales) {
            if(!tested_levels.insert(scale.level).second)continue;
            const auto [offset,length]=offsets.at(scale.constant_index);
            std::vector<double> values(length);cst.seekg(offset);
            cst.read(reinterpret_cast<char*>(values.data()),8*length);
            if(!cst)throw std::runtime_error("cannot read Encode vector");
            Plaintext plain;m.encoder.encode(values,m.ctx.crt_context()->parms_id_map().at(scale.level),scale.scale,plain);
            if(std::bit_cast<std::uint64_t>(plain.scale())!=scale.bits ||
                plain.parms_id()!=m.ctx.crt_context()->parms_id_map().at(scale.level))
                throw std::runtime_error("strict Encode bits/Q mismatch");
            ++checks;
        }
        std::cout<<"STRICT_NATIVE_PLAN_PROBE PASS inputs=27 boots="<<boot_count
            <<" raw_ops="<<nops<<" constants="<<constants<<" encodes="<<scales.size()
            <<" rescale_groups="<<groups.size()<<" encoder_levels_checked="<<checks
            <<" gpu_executed=false native_arithmetic_template_compiled=true\n";
    });
}
