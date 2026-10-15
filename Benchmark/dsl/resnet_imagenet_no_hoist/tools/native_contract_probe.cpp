// CPU-only: execute the original topology/config factories without CUDA.
#include "benchmark/resnet18_gpu/resnet18_topology.h"
#include "benchmark/resnet50_gpu/resnet50_topology.h"
#include "benchmark/resnet50_gpu/resnet50_config.h"
#include <iostream>
#include <stdexcept>

int main() {
    using namespace poseidon::benchmark;
    auto config = resnet50_gpu::make_resnet50_gpu_config();
    const auto r18 = resnet18_gpu::make_resnet18_topology();
    const auto r50 = resnet50_gpu::make_resnet50_topology();
    if (config.q_moduli.size()!=50 || config.p_moduli.size()!=25 ||
        config.slot_count()!=32768 || config.application_log_scale!=40)
        throw std::runtime_error("unexpected native CKKS profile");
    std::cout << "{\"scope\":\"native CPU factories, not encrypted validation\","
              << "\"N\":" << config.degree() << ",\"slots\":" << config.slot_count()
              << ",\"Q\":" << config.q_moduli.size() << ",\"P\":" << config.p_moduli.size()
              << ",\"application_q\":" << config.application_q_count()
              << ",\"input_q\":" << config.application_q_count()-4
              << ",\"application_logscale\":" << config.application_log_scale
              << ",\"bootstrap_logscale\":" << config.evalmod_log_scale
              << ",\"models\":[";
    auto emit = [](int model, const auto &topology) {
        std::cout << "{\"model\":\"resnet" << model << "\",\"activation_positions\":"
                  << topology.bootstrap_point_count() << ",\"blocks\":[";
        bool first=true;
        for (const auto &b: topology.blocks) {
            if (!first) std::cout << ',';
            first=false;
            std::cout << "{\"name\":\"layer" << b.stage << '.' << b.block
                      << "\",\"in_channels\":" << b.input_channels
                      << ",\"out_channels\":" << b.output_channels
                      << ",\"stride\":" << b.stride
                      << ",\"projection\":" << (b.projection?"true":"false") << '}';
        }
        std::cout << "]}";
    };
    emit(18,r18); std::cout << ','; emit(50,r50);
    std::cout << "]}\n";
}
