#include "gpu_config.h"
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
using poseidon::benchmark::resnet20_gpu::core::make_gpu_config;
int main()
{
    unsetenv("POSEIDON_GPU_RESNET20_BOOTSTRAP_DEGREE");
    unsetenv("RESNET20_CMP_DEGREE");
    const auto baseline = make_gpu_config();
    assert(baseline.evalmod_degree == 22 && baseline.double_angle == 5);
    assert(baseline.log_q.size() == 70 && baseline.log_p.size() == 35);
    assert(baseline.bootstrap_q_count == 70 && baseline.application_q_count() == 36);
    assert(baseline.evalmod_log_scale == 52 && baseline.application_log_scale == 40);
    for (const auto degree : {"59", "30", "22"}) {
        setenv("POSEIDON_GPU_RESNET20_BOOTSTRAP_DEGREE", degree, 1);
        const auto config = make_gpu_config();
        assert(config.log_q == baseline.log_q && config.log_p == baseline.log_p);
        assert(config.boundary_k == 25 && config.message_ratio == 32);
        assert(config.double_angle == (config.evalmod_degree == 59 ? 2u : config.evalmod_degree == 30 ? 4u : 5u));
    }
    setenv("RESNET20_CMP_DEGREE", "59", 1);
    assert(make_gpu_config().evalmod_degree == 22); // Canonical selector wins.
    unsetenv("POSEIDON_GPU_RESNET20_BOOTSTRAP_DEGREE");
    assert(make_gpu_config().evalmod_degree == 59);
    setenv("POSEIDON_GPU_RESNET20_BOOTSTRAP_DEGREE", "23", 1);
    bool rejected = false;
    try { make_gpu_config(); } catch (const std::invalid_argument &) { rejected = true; }
    assert(rejected);
    std::cout << "PASS: default22, common context, degree/DA pairs and invalid selector\n";
}
