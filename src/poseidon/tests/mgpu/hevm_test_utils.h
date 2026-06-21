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
    const std::vector<HevmOpRecord> &ops)
{
    if (result_registers.size() != res_length)
    {
        throw std::invalid_argument("HEVM result register count does not match res_length");
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
    append_u64(output, 0);

    for (std::uint64_t i = 0; i < arg_length; ++i)
    {
        append_u64(output, 0);
    }
    for (std::uint64_t i = 0; i < arg_length; ++i)
    {
        append_u64(output, 0);
    }
    for (std::uint64_t i = 0; i < res_length; ++i)
    {
        append_u64(output, 0);
    }
    for (std::uint64_t i = 0; i < res_length; ++i)
    {
        append_u64(output, 0);
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
