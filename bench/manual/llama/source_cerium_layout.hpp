#pragma once
// Fixed geometry of the Cerium Llama source-order port. This is NOT the
// feature-major Layout in support/llama_manual/linear.hpp.
// Ring degree N, complex slots, token capacity, and tile width are distinct.
// No smaller-N performance fallback is permitted for this implementation.
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace cerium_source::layout {
inline constexpr unsigned log_ring_degree=16, log_slots=15;
inline constexpr std::size_t ring_degree=65536, slots=32768;
inline constexpr std::size_t token_capacity=128, feature_tile=256;
inline constexpr std::size_t head_width=128, query_heads=32, kv_heads=8;
inline constexpr std::size_t hidden_width=4096, ffn_width=14336;
inline constexpr unsigned contract_version=1;
static_assert(slots*2==ring_degree && token_capacity*feature_tile==slots);
static_assert(query_heads*head_width==hidden_width && query_heads%kv_heads==0);

inline void require_ring(unsigned actual_log_n,unsigned actual_log_slots) {
    if(actual_log_n!=log_ring_degree || actual_log_slots!=log_slots)
        throw std::invalid_argument("Cerium Llama requires N=65536 and 32768 complex slots; smaller/different rings are not comparable");
}
inline void require_encoder_slots(std::size_t actual) {
    if(actual!=slots)throw std::invalid_argument("Cerium Llama encoder must expose 32768 complex slots");
}
inline std::size_t tiles(std::size_t width) {
    if(!width || width%feature_tile || width>std::numeric_limits<std::size_t>::max()/token_capacity)
        throw std::invalid_argument("Cerium Llama dense width must be a positive multiple of 256 without size overflow");
    return width/feature_tile;
}
inline std::size_t slot(std::size_t token,std::size_t local_feature) {
    if(token>=token_capacity || local_feature>=feature_tile)
        throw std::out_of_range("Cerium Llama packed coordinate");
    return token*feature_tile+local_feature;
}
inline std::size_t dense_index(std::size_t tile,std::size_t packed_slot,std::size_t width) {
    if(tile>=tiles(width) || packed_slot>=slots)
        throw std::out_of_range("Cerium Llama packed tile/slot");
    return (packed_slot/feature_tile)*width+tile*feature_tile+packed_slot%feature_tile;
}
// Pure rearrangement, not quantization/compression/encoding. Preserve all128
// rows, including padding; do not silently zero values or change signed zero.
inline std::vector<double> pack_tile(const std::vector<double> &dense,
                                     std::size_t width,std::size_t tile) {
    if(tile>=tiles(width) || dense.size()!=token_capacity*width)
        throw std::invalid_argument("Cerium Llama dense tensor shape/tile");
    std::vector<double> packed(slots);
    for(std::size_t token=0;token<token_capacity;++token)
        for(std::size_t feature=0;feature<feature_tile;++feature)
            packed[token*feature_tile+feature]=dense[token*width+tile*feature_tile+feature];
    return packed;
}
inline std::size_t q_head_order(std::size_t head) {
    if(head>=query_heads)throw std::out_of_range("Cerium Llama query head");
    return (head/8)*8+(head%8)/2+(head%2)*4;
}
// Gather HF split-half Q/K into Meta interleaved order: given a destination
// Meta row, return its HF source row. V must NOT use this map.
inline std::size_t hf_source_qk_row_for_meta_row(std::size_t row,std::size_t width) {
    if((width!=hidden_width && width!=kv_heads*head_width) || row>=width)
        throw std::out_of_range("Cerium Llama Q/K row");
    return head_width*(row/head_width)+(row%head_width)/2+(head_width/2)*(row%2);
}
// Score ciphertext slot =256*query+128*lane+offset. Offset is cyclic:
// the corresponding key token is (query+offset)%128, not just offset.
inline std::size_t score_key(std::size_t packed_slot) {
    if(packed_slot>=slots)throw std::out_of_range("Cerium Llama score slot");
    return (packed_slot/feature_tile+packed_slot%head_width)%token_capacity;
}
inline std::vector<double> causal_mask(std::size_t valid_tokens) {
    if(!valid_tokens || valid_tokens>token_capacity)
        throw std::invalid_argument("Cerium Llama valid token count");
    std::vector<double> mask(slots);
    for(std::size_t s=0;s<slots;++s) {
        const auto query=s/feature_tile;
        mask[s]=query<valid_tokens && score_key(s)<=query;
    }
    return mask;
}
} // namespace cerium_source::layout
