#pragma once

#include "functional_wide_probe_keys.h"
#include "poseidon/advance/homomorphic_linear_transform.h"
#include "poseidon/key/galoiskeys.h"

#include <map>
#include <cstdlib>
#include <functional>
#include <set>
#include <string>

namespace poseidon::gpu::probe
{
using Word = std::uint64_t;
using Buffer = DeviceVector<Word>;

inline void clear(Buffer &buffer)
{
    gpu_check_cuda(cudaMemsetAsync(buffer.data(), 0, buffer.size() * sizeof(Word),
                                  gpu_execution_stream()), "DFT clear accumulator");
}

class Layer
{
public:
    Layer(const ParametersLiteral &params, std::size_t q_count, int device_id = 0)
        : device_id_(device_id), q_words_(params.degree() * q_count), degree_(params.degree()), q_count_(q_count),
          bv_(params.log_n(), {params.q().begin(), params.q().begin() + q_count},
              params.p().front(), device_id),
          packed_key_(2 * (q_words_ + degree_) * q_count), key_(packed_key_.size(), device_id)
    {
        for (const auto &q : params.q())
            if (q.value() >= (std::uint64_t{1} << 56))
                throw std::invalid_argument("56-bit key storage requires Q below 2^56");
        if (params.p().front().value() >= (std::uint64_t{1} << 56))
            throw std::invalid_argument("56-bit key storage requires P below 2^56");
    }

    Buffer transform(const Buffer &input, const MatrixPlain &matrix, const IndexMap &index,
                     const std::vector<int> &baby_steps, const GaloisKeys &keys,
                     const util::GaloisTool &galois, bool stable_plaintexts = false,
                     const std::function<void(const char *)> &trace = {}, int hoist = 0)
    {
        if (hoist)
        {
            if (trace) trace("cts_rotation");
            if (hoist == 2) bv_.prepare_galois_digits(input);
            else bv_.prepare_galois(input);
        }
        std::map<int, Buffer> babies;
        for (int step : baby_steps)
        {
            if (step == 0)
                continue;
            Buffer rotated(input.size(), device_id_);
            if (trace) trace("cts_rotation");
            rotate(input, step, keys, galois, rotated, hoist);
            babies.emplace(step, std::move(rotated));
        }

        if (trace) trace("cts_other");
        Buffer group(input.size(), device_id_), rotated(input.size(), device_id_);
        Buffer total(input.size(), device_id_);
        clear(total);
        for (const auto &[giant_step, steps] : index)
        {
            clear(group);
            if (trace) trace("cts_plain_multiply");
            for (int baby_step : steps)
            {
                const auto &diagonal = matrix.plain_vec.at(giant_step + baby_step);
                if (diagonal.coeff_count() != q_words_ || !diagonal.is_ntt_form())
                    throw std::runtime_error("unexpected DFT plaintext");
                const auto &baby = baby_step == 0 ? input : babies.at(baby_step);
                bv_.multiply_plain_add(baby, device_plain(diagonal, stable_plaintexts), group);
            }
            if (giant_step == 0)
            {
                if (trace) trace("cts_other");
                bv_.add(group, total);
            }
            else
            {
                if (trace) trace("cts_rotation");
                rotate(group, giant_step, keys, galois, rotated);
                if (trace) trace("cts_other");
                bv_.add(rotated, total);
            }
        }
        if (trace) trace("cts_other");
        Buffer result(2 * (q_words_ - degree_), device_id_);
        bv_.rescale_last(total, result);
        return result;
    }

    void project_real(Buffer &input, const GaloisKeys &keys, const util::GaloisTool &galois)
    {
        Buffer conjugate(input.size(), device_id_);
        rotate(input, 0, keys, galois, conjugate);
        bv_.add(conjugate, input);
    }

    void add_plain(const Plaintext &plain, Buffer &input, bool stable_plaintext = false)
    {
        bv_.add_plain(device_plain(plain, stable_plaintext), input);
    }

    void add(const Buffer &source, Buffer &destination)
    {
        bv_.add(source, destination);
    }

    std::pair<std::size_t, std::size_t> resident_bytes() const
    {
        std::size_t plains = 0, keys = 0;
        for (const auto &entry : plains_) plains += entry.second.size() * sizeof(Word);
        for (const auto &entry : keys_) keys += entry.second.size();
        for (const auto &entry : full_keys_) keys += entry.second.size() * sizeof(Word);
        for (const auto &entry : hoisted_keys_) keys += entry.second.size() * sizeof(Word);
        return {plains, keys};
    }

    void rotate(const Buffer &input, int step, const GaloisKeys &keys,
                const util::GaloisTool &galois, Buffer &output, int hoisted = 0)
    {
        const auto element = galois.get_elt_from_step(step);
        if (std::getenv("POSEIDON_WIDE_FULL_KEYS"))
        {
            auto [entry, inserted] = full_keys_.try_emplace(element, packed_key_.size(), device_id_);
            if (inserted)
                probe::upload_bv_key(keys.key(element), degree_, q_count_, packed_key_, entry->second);
            if (hoisted == 2 && std::getenv("POSEIDON_WIDE_HOIST_KEYCACHE"))
            {
                auto [cached, created] = hoisted_keys_.try_emplace(element, packed_key_.size(), device_id_);
                if (created) bv_.inverse_rotate_key(entry->second, element, cached->second);
                bv_.apply_galois_digits(input, cached->second, element, output, true);
            }
            else if (hoisted == 2) bv_.apply_galois_digits(input, entry->second, element, output);
            else if (hoisted) bv_.apply_galois_prepared(input, entry->second, element, output);
            else bv_.apply_galois(input, entry->second, element, output);
            return;
        }
        auto &key = keys_.try_emplace(element, packed_key_.size() * 7, device_id_).first->second;
        if (!key_ready_.count(element))
        {
            probe::upload_bv_key(keys.key(element), degree_, q_count_, packed_key_, key_);
            wide_pack_56(key_, key);
            key_ready_.insert(element);
        }
        wide_unpack_56(key, key_);
        if (hoisted == 2) bv_.apply_galois_digits(input, key_, element, output);
        else if (hoisted) bv_.apply_galois_prepared(input, key_, element, output);
        else bv_.apply_galois(input, key_, element, output);
    }

private:
    const Buffer &device_plain(const Plaintext &plain, bool stable)
    {
        if (plain.coeff_count() != q_words_ || !plain.is_ntt_form())
            throw std::runtime_error("unexpected wide plaintext");
        if (stable)
        {
            const auto found = stable_plains_.find(&plain);
            if (found != stable_plains_.end()) return *found->second;
        }
        const auto key = std::string(reinterpret_cast<const char *>(plain.data()),
                                     q_words_ * sizeof(Word));
        auto [entry, inserted] = plains_.try_emplace(key, q_words_, device_id_);
        if (inserted)
            entry->second.copy_from_host(plain.data(), q_words_);
        if (stable) stable_plains_.emplace(&plain, &entry->second);
        return entry->second;
    }

    int device_id_;
    std::size_t q_words_;
    std::size_t degree_;
    std::size_t q_count_;
    GpuWideBv bv_;
    std::vector<Word> packed_key_;
    Buffer key_;
    std::map<std::string, Buffer> plains_;
    // Only plaintext objects owned for the entire Layer lifetime use this index.
    std::map<const Plaintext *, const Buffer *> stable_plains_;
    std::map<std::uint64_t, DeviceVector<std::uint8_t>> keys_;
    std::map<std::uint64_t, Buffer> full_keys_;
    std::map<std::uint64_t, Buffer> hoisted_keys_;
    std::set<std::uint64_t> key_ready_;
};

inline Buffer modraise(const Buffer &input, const ParametersLiteral &params, int device_id = 0,
                       GpuWideNtt *single_plan = nullptr, GpuWideNtt *full_plan = nullptr)
{
    const std::size_t degree = params.degree();
    const std::size_t q_words = degree * params.q().size();
    std::unique_ptr<GpuWideNtt> single_owned, full_owned;
    if (!single_plan) { single_owned = std::make_unique<GpuWideNtt>(params.log_n(), std::vector<Modulus>{params.q().front()}, device_id); single_plan = single_owned.get(); }
    if (!full_plan) { full_owned = std::make_unique<GpuWideNtt>(params.log_n(), params.q(), device_id); full_plan = full_owned.get(); }
    Buffer q0(degree, device_id), raised(q_words, device_id);
    Buffer result(2 * q_words, device_id);
    for (std::size_t component = 0; component < 2; ++component)
    {
        gpu_check_cuda(cudaMemcpyAsync(q0.data(), input.data() + component * degree,
            degree * sizeof(Word), cudaMemcpyDeviceToDevice, gpu_execution_stream()), "ModRaise input");
        single_plan->inverse(q0);
        full_plan->extend_centered_q0(q0, raised);
        full_plan->forward(raised);
        gpu_check_cuda(cudaMemcpyAsync(result.data() + component * q_words, raised.data(),
            q_words * sizeof(Word), cudaMemcpyDeviceToDevice, gpu_execution_stream()), "ModRaise output");
    }
    return result;
}
} // namespace poseidon::gpu::probe
