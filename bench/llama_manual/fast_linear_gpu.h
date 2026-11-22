#pragma once
#include <cuda_runtime_api.h>
#include <cstdint>

namespace fast_linear {
struct EncodeArgs {
    const double *weights=nullptr;      // dense row-major [rows][columns], device
    long long columns=0;
    int ot=0,lane=0,it_begin=0;
    const int *diagonals=nullptr;       // signed diagonal d, giant-group order
    const int *slot_map=nullptr;        // 512 entries (Encoder::slot_map)
    const double2 *inverse_roots=nullptr; // 512 entries (Encoder::inverse_roots)
    double fix=0;                       // scale/512
    double coefficient_limit=0;
    const uint32_t *moduli=nullptr,*ntt_roots=nullptr,*ntt_shoup=nullptr,*permutation=nullptr;
    int full_q=0,nq=0,np=0;
    uint32_t *const *outputs=nullptr;   // per input tile: matrix residue base [pos][limb][512]
    int *overflow=nullptr;
};
cudaError_t launch_encode_diagonals(const EncodeArgs &args,int diagonal_count,int input_tiles,cudaStream_t stream);

struct BatchEncodeQArgs {
    const double *weights=nullptr;
    const uint16_t *weights_bf16=nullptr; // if non-null, decodes BF16 -> double on GPU
    const int *col_perm=nullptr;          // optional column permutation [columns]
    long long columns=0;
    int ots[8]{};
    int lanes[8]{};
    const int *slot_map=nullptr;        // 512 entries
    const double2 *inverse_roots=nullptr; // 512 entries
    double fix=0,coefficient_limit=0;
    const uint32_t *moduli=nullptr,*ntt_roots=nullptr,*ntt_shoup=nullptr,*permutation=nullptr;
    int nq=0,nin=0,batch_count=0;
    uint32_t *encoded_q=nullptr;        // [batch_count][nin][256][nq][512]
    int *overflow=nullptr;
};
struct BatchMacQArgs {
    const uint32_t *babies_q=nullptr;   // [nin][16][2][nq][65536]
    const uint32_t *encoded_q=nullptr;  // [batch_count][nin][256][nq][512]
    uint32_t *out_q_batch=nullptr;      // [batch_count][16][2][nq][65536]
    const uint32_t *q_primes=nullptr;   // [nq]
    const uint64_t *q_barrett=nullptr;  // [nq]
    int nq=0,nin=0,batch_count=0;
};
cudaError_t launch_batch_encode_q_diagonals(const BatchEncodeQArgs &args,cudaStream_t stream);
cudaError_t launch_batched_compact_mac_q(const BatchMacQArgs &args,cudaStream_t stream);
} // namespace fast_linear
