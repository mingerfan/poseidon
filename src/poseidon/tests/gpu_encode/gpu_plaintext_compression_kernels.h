#pragma once
#include <cuda_runtime_api.h>
#include <cstdint>
#include <cstddef>
cudaError_t launch_unpack_residues(const std::uint64_t *input,std::uint32_t *output,
    std::size_t count,unsigned bits,cudaStream_t stream);
cudaError_t launch_replicate_ntt(const std::uint32_t *input,std::uint32_t *output,
    std::size_t count,std::size_t degree,std::size_t group,cudaStream_t stream);
