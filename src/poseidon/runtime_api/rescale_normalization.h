#pragma once

#include "poseidon/basics/modulus.h"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace poseidon::runtime_api {

struct RescaleNormalization {
    std::size_t native_drop_count;
    std::uint64_t coefficient_multiplier;
    double relative_rounding_error;
};

// Native CKKS rescale divides coefficients and scale by actual primes.
// A power-of-two target needs a real integer coefficient multiplication,
// with enough scale left before multiplication to preserve CKKS precision.
// Remaining reserved levels are ordinary modulus drops, never extra rounding.
inline RescaleNormalization make_rescale_normalization(
    const std::vector<Modulus> &active_q, double input_scale,
    std::size_t drop_count, double target_scale)
{
    if(!(input_scale>0) || !std::isfinite(input_scale) ||
       !(target_scale>0) || !std::isfinite(target_scale) ||
       !drop_count || drop_count>=active_q.size())
        throw std::invalid_argument("invalid Rescale normalization scale or level range");
    double remaining_bits=0;
    for(std::size_t i=0;i<active_q.size()-drop_count;++i)
        remaining_bits+=std::log2(static_cast<double>(active_q[i].value()));
    if(std::log2(target_scale)>=remaining_bits-1)
        throw std::invalid_argument("Rescale target scale exceeds remaining coefficient modulus");
    double native_scale=input_scale;
    for(std::size_t count=1;count<=drop_count;++count) {
        native_scale/=static_cast<double>(active_q[active_q.size()-count].value());
        // Integer upscaling cannot restore precision already lost in rounding.
        if(native_scale<std::ldexp(1.0,40))break;
        const double ratio=target_scale/native_scale;
        if(!(ratio>0) || !std::isfinite(ratio) || ratio>=std::ldexp(1.0,64))continue;
        const double rounded=std::floor(ratio+0.5);
        if(rounded>=std::ldexp(1.0,64))continue;
        const auto multiplier=static_cast<std::uint64_t>(rounded);
        const double error=std::abs(static_cast<double>(multiplier)/ratio-1);
        if(multiplier && error<=1e-9)
            return {count,multiplier,error};
    }
    throw std::invalid_argument("Rescale target lacks modulus/precision space for integer scale normalization");
}

} // namespace poseidon::runtime_api
