#include "poseidon/gpu/gpu_double_hoist.h"

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>

namespace poseidon
{
namespace gpu
{
namespace
{

std::size_t checked_mul(std::size_t left, std::size_t right, const char *what)
{
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
    {
        throw std::overflow_error(what);
    }
    return left * right;
}

}  // namespace

void GpuQPCiphertextBuffer::ensure_capacity(
    int requested_device_id,
    std::size_t requested_degree,
    std::size_t requested_q_count,
    std::size_t requested_p_count,
    std::size_t requested_batch_count)
{
    if (requested_degree == 0 || requested_q_count == 0 ||
        requested_p_count == 0 || requested_batch_count == 0)
    {
        throw std::invalid_argument(
            "GpuQPCiphertextBuffer::ensure_capacity requires a non-zero shape");
    }

    const auto required_q_words =
        requested_batch_count * 2 * requested_q_count * requested_degree;
    const auto required_p_words =
        requested_batch_count * 2 * requested_p_count * requested_degree;
    const bool fits =
        device_id == requested_device_id &&
        degree == requested_degree &&
        q.size() >= required_q_words &&
        p.size() >= required_p_words;
    if (fits)
    {
        q_count = requested_q_count;
        p_count = requested_p_count;
        batch_count = requested_batch_count;
        return;
    }

    const auto q_words = checked_mul(
        checked_mul(
            checked_mul(requested_batch_count, std::size_t{2},
                        "QP Q batch/component size overflow"),
            requested_q_count,
            "QP Q limb size overflow"),
        requested_degree,
        "QP Q coefficient size overflow");
    const auto p_words = checked_mul(
        checked_mul(
            checked_mul(requested_batch_count, std::size_t{2},
                        "QP P batch/component size overflow"),
            requested_p_count,
            "QP P limb size overflow"),
        requested_degree,
        "QP P coefficient size overflow");

    q.allocate(q_words, requested_device_id);
    p.allocate(p_words, requested_device_id);
    device_id = requested_device_id;
    degree = requested_degree;
    q_count = requested_q_count;
    p_count = requested_p_count;
    batch_count = requested_batch_count;
}

GpuWord *GpuQPCiphertextBuffer::q_component(
    std::size_t batch,
    std::size_t component)
{
    if (batch >= batch_count || component >= 2)
    {
        throw std::out_of_range("GpuQPCiphertextBuffer Q component is out of range");
    }
    return q.data() + (batch * 2 + component) * q_count * degree;
}

GpuWord *GpuQPCiphertextBuffer::p_component(
    std::size_t batch,
    std::size_t component)
{
    if (batch >= batch_count || component >= 2)
    {
        throw std::out_of_range("GpuQPCiphertextBuffer P component is out of range");
    }
    return p.data() + (batch * 2 + component) * p_count * degree;
}

const GpuWord *GpuQPCiphertextBuffer::q_component(
    std::size_t batch,
    std::size_t component) const
{
    return const_cast<GpuQPCiphertextBuffer *>(this)->q_component(batch, component);
}

const GpuWord *GpuQPCiphertextBuffer::p_component(
    std::size_t batch,
    std::size_t component) const
{
    return const_cast<GpuQPCiphertextBuffer *>(this)->p_component(batch, component);
}

void GpuQCiphertextBatchBuffer::ensure_capacity(
    int requested_device_id,
    std::size_t requested_degree,
    std::size_t requested_q_count,
    std::size_t requested_batch_count)
{
    if (requested_degree == 0 || requested_q_count == 0 ||
        requested_batch_count == 0)
    {
        throw std::invalid_argument(
            "GpuQCiphertextBatchBuffer::ensure_capacity requires a non-zero shape");
    }
    const auto required_words = checked_mul(
        checked_mul(
            checked_mul(
                requested_batch_count,
                std::size_t{2},
                "Q batch/component size overflow"),
            requested_q_count,
            "Q batch limb size overflow"),
        requested_degree,
        "Q batch coefficient size overflow");
    if (device_id != requested_device_id || q.size() < required_words)
    {
        q.allocate(required_words, requested_device_id);
    }
    device_id = requested_device_id;
    degree = requested_degree;
    q_count = requested_q_count;
    batch_count = requested_batch_count;
}

GpuWord *GpuQCiphertextBatchBuffer::q_component(
    std::size_t batch,
    std::size_t component)
{
    if (batch >= batch_count || component >= 2)
    {
        throw std::out_of_range(
            "GpuQCiphertextBatchBuffer component is out of range");
    }
    return q.data() +
        (batch * 2 + component) * q_count * degree;
}

const GpuWord *GpuQCiphertextBatchBuffer::q_component(
    std::size_t batch,
    std::size_t component) const
{
    return const_cast<GpuQCiphertextBatchBuffer *>(this)->q_component(
        batch,
        component);
}

void GpuHybridKeySwitchWorkspace::ensure_capacity(
    int requested_device_id,
    std::size_t requested_degree,
    std::size_t requested_q_count,
    std::size_t requested_p_count)
{
    if (requested_degree == 0 || requested_q_count == 0 ||
        requested_p_count == 0)
    {
        throw std::invalid_argument(
            "GpuHybridKeySwitchWorkspace::ensure_capacity requires a non-zero shape");
    }
    const auto requested_q_words =
        requested_q_count * requested_degree;
    const auto requested_p_words =
        requested_p_count * requested_degree;
    if (device_id == requested_device_id &&
        degree == requested_degree &&
        permuted_digit_q.size() >= requested_q_words &&
        permuted_digit_p.size() >= requested_p_words &&
        p_coeff0.size() >= requested_p_words &&
        converted_q0.size() >= requested_q_words)
    {
        q_count = requested_q_count;
        p_count = requested_p_count;
        return;
    }

    const auto q_words = checked_mul(
        requested_q_count,
        requested_degree,
        "HYBRID Q workspace size overflow");
    const auto p_words = checked_mul(
        requested_p_count,
        requested_degree,
        "HYBRID P workspace size overflow");

    permuted_digit_q.allocate(q_words, requested_device_id);
    permuted_digit_p.allocate(p_words, requested_device_id);
    p_coeff0.allocate(p_words, requested_device_id);
    p_coeff1.allocate(p_words, requested_device_id);
    converted_q0.allocate(q_words, requested_device_id);
    converted_q1.allocate(q_words, requested_device_id);

    device_id = requested_device_id;
    degree = requested_degree;
    q_count = requested_q_count;
    p_count = requested_p_count;
}

GpuLinearTransformMode gpu_linear_transform_mode_from_environment(
    GpuLinearTransformMode fallback)
{
    const char *raw = std::getenv("POSEIDON_GPU_LINEAR_TRANSFORM_MODE");
    if (raw == nullptr || raw[0] == '\0')
    {
        return fallback;
    }

    const std::string value(raw);
    if (value == "classic" || value == "classic_bsgs")
    {
        return GpuLinearTransformMode::ClassicBsgs;
    }
    if (value == "single_hoist")
    {
        return GpuLinearTransformMode::SingleHoistBsgs;
    }
    if (value == "double_hoist")
    {
        return GpuLinearTransformMode::DoubleHoistBsgs;
    }
    throw std::invalid_argument(
        "POSEIDON_GPU_LINEAR_TRANSFORM_MODE must be classic, single_hoist, or double_hoist");
}

}  // namespace gpu
}  // namespace poseidon
