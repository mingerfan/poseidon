#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// Toolchain/device gate only: this is not CKKS or a Poseidon execution test.
void checked(cudaError_t code, const char *where) {
    if (code != cudaSuccess) throw std::runtime_error(std::string(where)+": "+cudaGetErrorString(code));
}
__global__ void affine(const int *input, int *output, int count) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < count) output[i] = 3*input[i]-7;
}
int main() {
    int *input = nullptr, *output = nullptr;
    try {
        int count = 0;
        checked(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
        if (count < 1) throw std::runtime_error("No CUDA device");
        checked(cudaSetDevice(0), "cudaSetDevice");
        cudaDeviceProp prop{};
        checked(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
        constexpr int n = 1027;
        std::vector<int> source(n), actual(n);
        for (int i = 0; i < n; ++i) source[i] = (i%31)-15;
        checked(cudaMalloc(&input, n*sizeof(int)), "allocate input");
        checked(cudaMalloc(&output, n*sizeof(int)), "allocate output");
        checked(cudaMemcpy(input, source.data(), n*sizeof(int), cudaMemcpyHostToDevice), "upload");
        affine<<<(n+127)/128,128>>>(input, output, n);
        checked(cudaGetLastError(), "launch");
        checked(cudaDeviceSynchronize(), "synchronize");
        checked(cudaMemcpy(actual.data(), output, n*sizeof(int), cudaMemcpyDeviceToHost), "download");
        for (int i = 0; i < n; ++i)
            if (actual[i] != 3*source[i]-7) throw std::runtime_error("GPU result mismatch");
        checked(cudaFree(input), "free input"); input = nullptr;
        checked(cudaFree(output), "free output"); output = nullptr;
        std::cout << "{\"status\":\"passed\",\"gpu_kernel_executed\":true,\"fhe_executed\":false,"
                  << "\"device\":\"" << prop.name << "\",\"compute_major\":" << prop.major
                  << ",\"compute_minor\":" << prop.minor << ",\"compared_integers\":" << n << "}\n";
    } catch (const std::exception &error) {
        if (input) cudaFree(input);
        if (output) cudaFree(output);
        std::cerr << error.what() << '\n';
        return 1;
    }
}
