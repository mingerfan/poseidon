// Compile the original reference without editing application sources. Linker
// section GC discards its GPU runtime functions; no gated runtime is created.
#include "/home/liufuyao/Work/poseidon_gpu_other/resnet20-9.3/benchmark/resnet20_gpu/gpu_relu.cpp"

double original_application_relu(double x) {
    return poseidon::benchmark::resnet20_gpu::core::polynomial_relu_reference(x);
}
