#pragma once
#include "llama_manual/model.hpp"
#include <cstdint>
#include <filesystem>
#include <map>

namespace llama_manual {
struct Bf16MatrixShard {
    std::size_t rows=0,columns=0;
    std::vector<std::uint16_t> values; // row-major [rows][columns] BF16
    void validate(std::size_t r,std::size_t c) const {
        if(rows!=r||columns!=c||!c||r>std::numeric_limits<std::size_t>::max()/c||
           values.size()!=r*c)throw std::invalid_argument("BF16 weight shard shape mismatch");
    }
};
struct LayerWeightsBf16Shard {
    std::vector<double> attention_norm,ffn_norm;
    Bf16MatrixShard q,k,v,o,gate,up,down;
    int shard_rank=0,world_size=1;
};

// A checked text index addresses original safetensors byte ranges. It does
// not expand or duplicate the checkpoint on disk, and never loads pickle.
class SafetensorWeights final : public NativeWeightProvider {
public:
    SafetensorWeights(const std::filesystem::path &root,const std::filesystem::path &index);
    Matrix read_matrix(const std::string &name,std::size_t rows,std::size_t columns,
                       bool hf_to_meta_qk=false) const;
    Bf16MatrixShard read_matrix_bf16_rows(const std::string &name,std::size_t full_rows,
                                          std::size_t columns,std::size_t row_begin,
                                          std::size_t row_count,bool hf_to_meta_qk=false,
                                          bool apply_q_head_order=false) const;
    LayerWeights load_layer(std::size_t layer) override;
    LayerWeightsBf16Shard load_layer_bf16_shard(std::size_t layer,int shard_rank,int world_size) const;
    void validate_model_index(std::size_t layers) const;
    static std::size_t hf_qk_row(std::size_t native_row) {
        return (native_row/128)*128+(native_row%128)/2+(native_row%2)*64;
    }
    static std::size_t q_head_order(std::size_t head) {
        return (head/8)*8+(head%8)/2+(head%2)*4;
    }
private:
    struct Record {std::string file,dtype;std::uint64_t offset,bytes;std::size_t rows,columns;};
    std::filesystem::path root_;
    std::map<std::string,Record> records_;
};
} // namespace llama_manual
