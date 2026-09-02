#include "gpu_config.h"

namespace poseidon::benchmark::resnet18_gpu
{

ResNet18GpuConfig make_resnet18_gpu_config()
{
    ResNet18GpuConfig config;
    config.log_scale = 40;
    config.application_log_scale = 40;
    config.evalmod_log_scale = 45;
    config.bootstrap_output_log_scale = 40;
    config.bootstrap_q_count = 34;
    config.dnum = 2;
    config.application_levels = 17;
    config.application_prime_bits = 32;
    config.cipher_product_rescale_primes = 1;
    config.physical_primes_per_application_level = 2;

    // q[0..1] is the q0 base. q[2..33] is the verified Q34 bootstrap
    // prefix and q[34..35] extends the refreshed result to application Q36.
    config.log_q = {
        32, 32,
        32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
        28, 28, 31, 31, 32, 32, 30, 31, 32, 31, 32,
        32, 31, 31, 31, 32, 32, 31, 32, 32, 32, 30,
        32, 32};
    config.log_p.assign(18, 32);
    config.validate();
    return config;
}

}  // namespace poseidon::benchmark::resnet18_gpu
