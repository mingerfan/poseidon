#pragma once

#include "functional_wide_probe_keys.h"
#include "poseidon/ckks_encoder.h"
#include "poseidon/decryptor.h"
#include "poseidon/key/relinkeys.h"

#include <complex>
#include <map>
#include <memory>
#include <tuple>

namespace poseidon::gpu::probe
{

struct WideValue
{
    DeviceVector<std::uint64_t> residues;
    std::size_t level;
    double scale;
};

class WidePolynomial
{
public:
    using Complex = std::complex<double>;
    using Coefficients = std::vector<Complex>;

    WidePolynomial(const ParametersLiteral &params, CKKSEncoder &encoder,
                   const RelinKeys &relin, int device)
        : params_(params), encoder_(encoder), relin_(relin), device_(device)
    {
        for (const auto &[level, id] : encoder.context().crt_context()->parms_id_map())
            ids_.emplace(level, id);
    }

    WideValue evaluate(const WideValue &input, const Coefficients &coefficients,
                       double target_scale, bool chebyshev)
    {
        const auto n = coefficients.size();
        if (n < 2 || (n & (n - 1)) != 0 ||
            n > (std::size_t{1} << input.level))
            throw std::invalid_argument("polynomial requires a power-of-two coefficient count and enough levels");

        std::vector<WideValue> powers;
        const WideValue *previous = &input;
        for (std::size_t exponent = 2; exponent < n; exponent *= 2)
        {
            auto next = square(*previous);
            if (chebyshev)
            {
                auto &bv = at(next.level).bv;
                bv.add(next.residues, next.residues);
                add_constant(next, -1.0);
            }
            powers.push_back(std::move(next));
            previous = &powers.back();
        }
        return split_evaluate(input, coefficients, target_scale, chebyshev, powers);
    }

    WideValue square(const WideValue &input)
    {
        auto &layer = at(input.level);
        WideValue product{buffer(input.level), input.level, input.scale * input.scale};
        layer.bv.square_relinearize(input.residues, layer.relin_key, product.residues);
        return rescale(product);
    }

    std::vector<Complex> decode(const WideValue &input, Decryptor &decryptor)
    {
        Ciphertext ciphertext;
        ciphertext.resize(encoder_.context(), ids_.at(input.level), 2);
        ciphertext.scale() = input.scale;
        ciphertext.is_ntt_form() = true;
        input.residues.copy_to_host(ciphertext.data(0), input.residues.size());
        Plaintext plain;
        decryptor.decrypt(ciphertext, plain);
        std::vector<Complex> slots;
        encoder_.decode(plain, slots);
        return slots;
    }

private:
    struct Level
    {
        Level(const ParametersLiteral &params, std::size_t level,
              const RelinKeys &relin, int device)
            : bv(params.log_n(), {params.q().begin(), params.q().begin() + level + 1},
                 params.p().front(), device),
              relin_key(2 * (level + 1) * (level + 2) * params.degree(), device)
        {
            std::vector<std::uint64_t> packed(relin_key.size());
            upload_bv_key(relin.key(2), params.degree(), level + 1, packed, relin_key);
        }

        GpuWideBv bv;
        DeviceVector<std::uint64_t> relin_key;
    };

    Level &at(std::size_t level)
    {
        auto &entry = levels_[level];
        if (!entry)
            entry = std::make_unique<Level>(params_, level, relin_, device_);
        return *entry;
    }

    DeviceVector<std::uint64_t> buffer(std::size_t level) const
    {
        return DeviceVector<std::uint64_t>(2 * (level + 1) * params_.degree(), device_);
    }

    double modulus(std::size_t level) const
    {
        return static_cast<double>(params_.q().at(level).value());
    }

    const DeviceVector<std::uint64_t> &constant(Complex value, std::size_t level, double scale)
    {
        const auto key = std::make_tuple(level, scale, value.real(), value.imag());
        auto [entry, inserted] = constants_.try_emplace(key);
        if (inserted)
        {
            Plaintext plain;
            encoder_.encode(value, ids_.at(level), scale, plain);
            entry->second = std::make_unique<DeviceVector<std::uint64_t>>(
                (level + 1) * params_.degree(), device_);
            entry->second->copy_from_host(plain.data(), entry->second->size());
        }
        return *entry->second;
    }

    void add_constant(WideValue &input, Complex value)
    {
        at(input.level).bv.add_plain(constant(value, input.level, input.scale), input.residues);
    }

    WideValue rescale(const WideValue &input)
    {
        if (input.level == 0)
            throw std::invalid_argument("rescale requires at least two Q limbs");
        WideValue result{buffer(input.level - 1), input.level - 1,
                         input.scale / modulus(input.level)};
        at(input.level).bv.rescale_last(input.residues, result.residues);
        return result;
    }

    WideValue drop_last(const WideValue &input)
    {
        WideValue result{buffer(input.level - 1), input.level - 1, input.scale};
        const std::size_t from = input.residues.size() / 2;
        const std::size_t to = result.residues.size() / 2;
        for (std::size_t c = 0; c < 2; ++c)
            gpu_check_cuda(cudaMemcpyAsync(result.residues.data() + c * to,
                input.residues.data() + c * from, to * sizeof(std::uint64_t),
                cudaMemcpyDeviceToDevice, gpu_execution_stream()), "polynomial drop last");
        return result;
    }

    WideValue split_evaluate(const WideValue &input, const Coefficients &coefficients,
                             double target_scale, bool chebyshev,
                             const std::vector<WideValue> &powers)
    {
        const std::size_t half = coefficients.size() / 2;
        if (half == 1)
        {
            auto &layer = at(input.level);
            const double before_rescale = target_scale * modulus(input.level);
            WideValue result{buffer(input.level), input.level, before_rescale};
            gpu_check_cuda(cudaMemsetAsync(result.residues.data(), 0,
                result.residues.size() * sizeof(std::uint64_t), gpu_execution_stream()),
                "polynomial clear leaf");
            layer.bv.multiply_plain_add(input.residues,
                constant(coefficients[1], input.level, before_rescale / input.scale), result.residues);
            add_constant(result, coefficients[0]);
            return rescale(result);
        }

        Coefficients low(coefficients.begin(), coefficients.begin() + half);
        Coefficients high(coefficients.begin() + half, coefficients.end());
        if (chebyshev)
        {
            // 2 T_half T_j = T_(half+j) + T_(half-j).
            for (std::size_t j = 1; j < half; ++j)
            {
                low[half - j] -= high[j];
                high[j] *= 2;
            }
        }
        std::size_t power_index = 0;
        for (std::size_t exponent = half; exponent > 2; exponent /= 2)
            ++power_index;
        const auto &power = powers.at(power_index);
        const double high_scale = target_scale * modulus(power.level) / power.scale;
        auto a = split_evaluate(input, low, target_scale, chebyshev, powers);
        auto b = split_evaluate(input, high, high_scale, chebyshev, powers);
        if (a.level != power.level || b.level != power.level)
            throw std::runtime_error("polynomial merge level mismatch");

        auto &layer = at(power.level);
        WideValue product{buffer(power.level), power.level, b.scale * power.scale};
        layer.bv.multiply_relinearize(power.residues, b.residues, layer.relin_key, product.residues);
        auto result = rescale(product);
        auto aligned = drop_last(a);
        if (std::abs(aligned.scale / result.scale - 1) > 1e-10)
            throw std::runtime_error("polynomial merge scale mismatch");
        at(result.level).bv.add(aligned.residues, result.residues);
        return result;
    }

    const ParametersLiteral &params_;
    CKKSEncoder &encoder_;
    const RelinKeys &relin_;
    int device_;
    std::map<std::size_t, parms_id_type> ids_;
    std::map<std::size_t, std::unique_ptr<Level>> levels_;
    std::map<std::tuple<std::size_t, double, double, double>,
             std::unique_ptr<DeviceVector<std::uint64_t>>> constants_;
};

} // namespace poseidon::gpu::probe
