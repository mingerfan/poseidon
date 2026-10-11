#include "gpu_plaintext_compression_kernels.h"
namespace {
__global__ void unpack_residues(const std::uint64_t *input,std::uint32_t *output,
    std::size_t count,unsigned bits) {
    const std::size_t i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=count) return;
    const std::size_t offset=i*bits;
    const unsigned shift=offset%64;
    std::uint64_t code=input[offset/64]>>shift;
    if(shift+bits>64) code|=input[offset/64+1]<<(64-shift);
    output[i]=code&((std::uint64_t(1)<<bits)-1);
}
__global__ void replicate_ntt(const std::uint32_t *input,std::uint32_t *output,
    std::size_t count,std::size_t n,std::size_t group) {
    const std::size_t i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count) output[i]=input[(i/n)*(n/group)+(i%n)/group];
}
}
cudaError_t launch_unpack_residues(const std::uint64_t *input,std::uint32_t *output,
    std::size_t count,unsigned bits,cudaStream_t stream) {
    unpack_residues<<<(count+255)/256,256,0,stream>>>(input,output,count,bits);
    return cudaGetLastError();
}
cudaError_t launch_replicate_ntt(const std::uint32_t *input,std::uint32_t *output,
    std::size_t count,std::size_t degree,std::size_t group,cudaStream_t stream) {
    replicate_ntt<<<(count+255)/256,256,0,stream>>>(input,output,count,degree,group);
    return cudaGetLastError();
}
