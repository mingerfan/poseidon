#include <cuda_runtime_api.h>
extern "C" cudaError_t cudaGetDevice(int *d){*d=0;return cudaSuccess;}
