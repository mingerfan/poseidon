#include "llama_manual/checkpoint.hpp"
#include <cstring>
#include <fstream>
#include <sstream>
#include <cstdlib>

namespace llama_manual {
namespace {
std::size_t element_size(const std::string &dtype) {
    if(dtype=="BF16"||dtype=="F16")return 2;
    if(dtype=="F32")return 4;
    if(dtype=="F64")return 8;
    throw std::invalid_argument("unsupported checkpoint dtype");
}
double number(const unsigned char *data,const std::string &dtype) {
    std::uint64_t bits=0;
    for(std::size_t j=0;j<element_size(dtype);++j)bits|=std::uint64_t(data[j])<<(8*j);
    if(dtype=="F64") {double value;std::memcpy(&value,&bits,8);return value;}
    if(dtype=="F32"||dtype=="BF16") {
        auto word=static_cast<std::uint32_t>(dtype=="BF16"?bits<<16:bits);
        float value;std::memcpy(&value,&word,4);return value;
    }
    const int sign=(bits&0x8000)?-1:1,exponent=(bits>>10)&31,fraction=bits&1023;
    if(exponent==31)throw std::runtime_error("nonfinite F16 checkpoint weight");
    return sign*(exponent?std::ldexp(1.+fraction/1024.,exponent-15):std::ldexp(double(fraction),-24));
}
}

SafetensorWeights::SafetensorWeights(const std::filesystem::path &root,const std::filesystem::path &index)
    :root_(std::filesystem::canonical(root)) {
    std::ifstream file(index);std::string line;
    if(!std::getline(file,line)||line!="LLAMA31_HF_SAFETENSORS_V1")
        throw std::invalid_argument("checked Llama 3.1 HF index required");
    while(std::getline(file,line)) {
        std::istringstream parser(line);std::string name,extra;Record r;
        if(!(parser>>name>>r.file>>r.dtype>>r.offset>>r.bytes>>r.rows>>r.columns)||(parser>>extra))
            throw std::invalid_argument("malformed tensor index row");
        const auto width=element_size(r.dtype);
        if(!r.rows||!r.columns||r.rows>std::numeric_limits<std::size_t>::max()/r.columns/width||
           r.bytes!=r.rows*r.columns*width||std::filesystem::path(r.file).filename()!=r.file)
            throw std::invalid_argument("tensor shape, size or shard path invalid");
        auto path=std::filesystem::canonical(root_/r.file);
        if(path.parent_path()!=root_)throw std::invalid_argument("shard escapes checkpoint directory");
        const auto size=std::filesystem::file_size(path);
        if(r.offset>size||r.bytes>size-r.offset||r.offset>std::uint64_t(std::numeric_limits<std::streamoff>::max()))
            throw std::invalid_argument("tensor byte range exceeds shard");
        if(!records_.emplace(name,r).second)throw std::invalid_argument("duplicate indexed tensor");
    }
    if(records_.empty())throw std::invalid_argument("empty checkpoint index");
}

Matrix SafetensorWeights::read_matrix(const std::string &name,std::size_t rows,
                                     std::size_t columns,bool permute) const {
    const auto &r=records_.at(name);
    if(r.rows!=rows||r.columns!=columns||(permute && rows%128))
        throw std::invalid_argument("checkpoint tensor shape/layout mismatch: "+name);
    Matrix result{rows,columns,std::vector<double>(rows*columns)};
    const auto width=element_size(r.dtype);
    const char *fast_flag=std::getenv("LLAMA_WEIGHT_DECODE_FASTPATH");
    const bool fast_bf16=fast_flag&&std::string(fast_flag)=="1"&&r.dtype=="BF16";
    std::vector<unsigned char> buffer(columns*width);
    std::ifstream file(root_/r.file,std::ios::binary);
    for(std::size_t out=0;out<rows;++out) {
        const auto source=permute?hf_qk_row(out):out;
        file.seekg(static_cast<std::streamoff>(r.offset+source*columns*width));
        if(!file.read(reinterpret_cast<char*>(buffer.data()),buffer.size()))
            throw std::runtime_error("truncated checkpoint tensor: "+name);
        if(fast_bf16) {
            // Dtype dispatch once, not for every one of ~218M BF16 weights.
            // Preserve little-endian bit interpretation and finite checks.
            for(std::size_t in=0;in<columns;++in) {
                const std::uint32_t bits=(std::uint32_t(buffer[2*in])|
                    (std::uint32_t(buffer[2*in+1])<<8))<<16;
                float f;std::memcpy(&f,&bits,sizeof(f));const double value=f;
                if(!std::isfinite(value))throw std::runtime_error("nonfinite checkpoint weight: "+name);
                result.values[out*columns+in]=value;
            }
        } else for(std::size_t in=0;in<columns;++in) {
            const double value=number(buffer.data()+in*width,r.dtype);
            if(!std::isfinite(value))throw std::runtime_error("nonfinite checkpoint weight: "+name);
            result.values[out*columns+in]=value;
        }
    }
    return result;
}

void SafetensorWeights::validate_model_index(std::size_t layers) const {
    if(!layers||layers>32)throw std::invalid_argument("Llama index needs 1..32 blocks");
    auto require=[&](const std::string &name,std::size_t rows,std::size_t columns) {
        const auto found=records_.find(name);
        if(found==records_.end()||found->second.rows!=rows||found->second.columns!=columns)
            throw std::invalid_argument("missing/wrong model tensor in checked index: "+name);
    };
    for(std::size_t layer=0;layer<layers;++layer) {
        const auto p="model.layers."+std::to_string(layer)+".";
        require(p+"input_layernorm.weight",4096,1);
        require(p+"post_attention_layernorm.weight",4096,1);
        require(p+"self_attn.q_proj.weight",4096,4096);
        require(p+"self_attn.k_proj.weight",1024,4096);
        require(p+"self_attn.v_proj.weight",1024,4096);
        require(p+"self_attn.o_proj.weight",4096,4096);
        require(p+"mlp.gate_proj.weight",14336,4096);
        require(p+"mlp.up_proj.weight",14336,4096);
        require(p+"mlp.down_proj.weight",4096,14336);
    }
}

LayerWeights SafetensorWeights::load_layer(std::size_t layer) {
    if(layer>=32)throw std::invalid_argument("Llama-3.1-8B block index");
    const auto prefix="model.layers."+std::to_string(layer)+".";
    LayerWeights w;
    w.attention_norm=read_matrix(prefix+"input_layernorm.weight",4096,1).values;
    w.ffn_norm=read_matrix(prefix+"post_attention_layernorm.weight",4096,1).values;
    w.q=read_matrix(prefix+"self_attn.q_proj.weight",4096,4096,true);
    w.k=read_matrix(prefix+"self_attn.k_proj.weight",1024,4096,true);
    w.v=read_matrix(prefix+"self_attn.v_proj.weight",1024,4096);
    w.o=read_matrix(prefix+"self_attn.o_proj.weight",4096,4096);
    w.gate=read_matrix(prefix+"mlp.gate_proj.weight",14336,4096);
    w.up=read_matrix(prefix+"mlp.up_proj.weight",14336,4096);
    w.down=read_matrix(prefix+"mlp.down_proj.weight",4096,14336);
    w.validate(ModelConfig{});
    return w;
}

Bf16MatrixShard SafetensorWeights::read_matrix_bf16_rows(
    const std::string &name,std::size_t full_rows,std::size_t columns,
    std::size_t row_begin,std::size_t row_count,
    bool hf_to_meta_qk,bool apply_q_head_order) const {
    const auto &r=records_.at(name);
    if(r.dtype!="BF16")
        throw std::invalid_argument("BF16 fast weight shard loader requires BF16 dtype: "+name);
    if(r.rows!=full_rows||r.columns!=columns||row_begin>full_rows||row_count>full_rows-row_begin||
       ((hf_to_meta_qk||apply_q_head_order)&&(full_rows%128)))
        throw std::invalid_argument("BF16 shard shape/bounds mismatch: "+name);
    Bf16MatrixShard out{row_count,columns,std::vector<std::uint16_t>(row_count*columns)};
    std::ifstream file(root_/r.file,std::ios::binary);
    if(!hf_to_meta_qk&&!apply_q_head_order) {
        file.seekg(static_cast<std::streamoff>(r.offset+row_begin*columns*2));
        if(!file.read(reinterpret_cast<char*>(out.values.data()),
                      static_cast<std::streamsize>(out.values.size()*2)))
            throw std::runtime_error("truncated BF16 weight shard: "+name);
    } else {
        for(std::size_t local_r=0;local_r<row_count;++local_r) {
            const std::size_t logical_r=row_begin+local_r;
            const std::size_t head_r=apply_q_head_order?(q_head_order(logical_r/128)*128+logical_r%128):logical_r;
            const std::size_t src_r=hf_to_meta_qk?hf_qk_row(head_r):head_r;
            file.seekg(static_cast<std::streamoff>(r.offset+src_r*columns*2));
            if(!file.read(reinterpret_cast<char*>(out.values.data()+local_r*columns),
                          static_cast<std::streamsize>(columns*2)))
                throw std::runtime_error("truncated BF16 permuted weight row: "+name);
        }
    }
    for(const std::uint16_t bits:out.values) {
        if((bits&0x7F80U)==0x7F80U)throw std::runtime_error("nonfinite BF16 checkpoint weight: "+name);
    }
    return out;
}

LayerWeightsBf16Shard SafetensorWeights::load_layer_bf16_shard(
    std::size_t layer,int shard_rank,int world_size) const {
    if(layer>=32||(world_size!=1&&world_size!=2)||shard_rank<0||shard_rank>=world_size)
        throw std::invalid_argument("invalid BF16 layer shard arguments");
    const auto prefix="model.layers."+std::to_string(layer)+".";
    LayerWeightsBf16Shard w;
    w.shard_rank=shard_rank;w.world_size=world_size;
    w.attention_norm=read_matrix(prefix+"input_layernorm.weight",4096,1).values;
    w.ffn_norm=read_matrix(prefix+"post_attention_layernorm.weight",4096,1).values;
    const auto split=[&](std::size_t total_rows) -> std::pair<std::size_t,std::size_t> {
        const std::size_t half=total_rows/static_cast<std::size_t>(world_size);
        return {static_cast<std::size_t>(shard_rank)*half,half};
    };
    {const auto [b,c]=split(4096);  w.q=read_matrix_bf16_rows(prefix+"self_attn.q_proj.weight",4096,4096,b,c,true,true);}
    {const auto [b,c]=split(1024);  w.k=read_matrix_bf16_rows(prefix+"self_attn.k_proj.weight",1024,4096,b,c,true,false);}
    {const auto [b,c]=split(1024);  w.v=read_matrix_bf16_rows(prefix+"self_attn.v_proj.weight",1024,4096,b,c,false,false);}
    {const auto [b,c]=split(4096);  w.o=read_matrix_bf16_rows(prefix+"self_attn.o_proj.weight",4096,4096,b,c,false,false);}
    {const auto [b,c]=split(14336); w.gate=read_matrix_bf16_rows(prefix+"mlp.gate_proj.weight",14336,4096,b,c,false,false);}
    {const auto [b,c]=split(14336); w.up=read_matrix_bf16_rows(prefix+"mlp.up_proj.weight",14336,4096,b,c,false,false);}
    {const auto [b,c]=split(4096);  w.down=read_matrix_bf16_rows(prefix+"mlp.down_proj.weight",4096,14336,b,c,false,false);}
    return w;
}
} // namespace llama_manual
