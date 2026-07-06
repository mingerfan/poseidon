#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace poseidon::mgpu::test
{

struct HevmOpRecord
{
    std::uint16_t opcode = 0;
    std::uint16_t dst = 0;
    std::uint16_t lhs = 0;
    std::uint16_t rhs = 0;
};

struct HevmConfigMetadata
{
    std::vector<std::uint64_t> arg_scales;
    std::vector<std::uint64_t> arg_levels;
    std::vector<std::uint64_t> result_scales;
    std::vector<std::uint64_t> result_levels;
    std::uint64_t init_level = 0;
};

inline std::uint16_t make_hevm_encode_attr(std::uint16_t level, std::uint16_t scale)
{
    if (level > 0x3F || scale > 0x3FF)
    {
        throw std::invalid_argument("HEVM encode attributes exceed 16-bit test encoding");
    }
    return static_cast<std::uint16_t>((level << 10) | scale);
}

inline void append_u16(std::string &output, std::uint16_t value)
{
    output.push_back(static_cast<char>(value & 0xFF));
    output.push_back(static_cast<char>((value >> 8) & 0xFF));
}

inline void append_u32(std::string &output, std::uint32_t value)
{
    for (int i = 0; i < 4; ++i)
    {
        output.push_back(static_cast<char>((value >> (8 * i)) & 0xFF));
    }
}

inline void append_u64(std::string &output, std::uint64_t value)
{
    for (int i = 0; i < 8; ++i)
    {
        output.push_back(static_cast<char>((value >> (8 * i)) & 0xFF));
    }
}

inline std::string make_hevm_binary(
    std::uint64_t arg_length, std::uint64_t res_length,
    std::uint64_t num_ctxt_buffer, std::uint64_t num_ptxt_buffer,
    const std::vector<std::uint64_t> &result_registers,
    const std::vector<HevmOpRecord> &ops,
    HevmConfigMetadata metadata = {})
{
    if (result_registers.size() != res_length)
    {
        throw std::invalid_argument("HEVM result register count does not match res_length");
    }
    if (metadata.arg_scales.empty())
    {
        metadata.arg_scales.assign(static_cast<std::size_t>(arg_length), 0);
    }
    if (metadata.arg_levels.empty())
    {
        metadata.arg_levels.assign(static_cast<std::size_t>(arg_length), 0);
    }
    if (metadata.result_scales.empty())
    {
        metadata.result_scales.assign(static_cast<std::size_t>(res_length), 0);
    }
    if (metadata.result_levels.empty())
    {
        metadata.result_levels.assign(static_cast<std::size_t>(res_length), 0);
    }
    if (metadata.arg_scales.size() != arg_length || metadata.arg_levels.size() != arg_length)
    {
        throw std::invalid_argument("HEVM argument metadata count does not match arg_length");
    }
    if (metadata.result_scales.size() != res_length ||
        metadata.result_levels.size() != res_length)
    {
        throw std::invalid_argument("HEVM result metadata count does not match res_length");
    }

    const std::uint64_t config_array_count = arg_length * 2 + res_length * 3;
    const std::uint64_t config_body_length = 40 + config_array_count * 8;

    std::string output;
    append_u32(output, 0x4845564D);
    append_u32(output, 24);
    append_u64(output, arg_length);
    append_u64(output, res_length);

    append_u64(output, config_body_length);
    append_u64(output, ops.size());
    append_u64(output, num_ctxt_buffer);
    append_u64(output, num_ptxt_buffer);
    append_u64(output, metadata.init_level);

    for (std::uint64_t i = 0; i < arg_length; ++i)
    {
        append_u64(output, metadata.arg_scales[static_cast<std::size_t>(i)]);
    }
    for (std::uint64_t i = 0; i < arg_length; ++i)
    {
        append_u64(output, metadata.arg_levels[static_cast<std::size_t>(i)]);
    }
    for (std::uint64_t i = 0; i < res_length; ++i)
    {
        append_u64(output, metadata.result_scales[static_cast<std::size_t>(i)]);
    }
    for (std::uint64_t i = 0; i < res_length; ++i)
    {
        append_u64(output, metadata.result_levels[static_cast<std::size_t>(i)]);
    }
    for (std::uint64_t i = 0; i < res_length; ++i)
    {
        append_u64(output, result_registers[static_cast<std::size_t>(i)]);
    }

    for (const HevmOpRecord &op : ops)
    {
        append_u16(output, op.opcode);
        append_u16(output, op.dst);
        append_u16(output, op.lhs);
        append_u16(output, op.rhs);
    }

    return output;
}

}  // namespace poseidon::mgpu::test
