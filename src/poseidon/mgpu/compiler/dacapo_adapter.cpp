#include "poseidon/mgpu/compiler/dacapo_adapter.h"

#include "poseidon/mgpu/ir/schedule_json.h"

#include <cstdint>
#include <limits>
#include <sstream>
#include <utility>
#include <vector>

namespace poseidon::mgpu
{
namespace
{

constexpr std::uint32_t kHevmMagic = 0x4845564D;
constexpr std::uint16_t kHevmAllocOpcode = 0xFFFF;
constexpr int kUnassignedDacapoDevice = -1;

void add_diagnostic(DacapoAdapterResult &result, std::size_t offset, const std::string &message)
{
    result.diagnostics.push_back(DacapoAdapterDiagnostic{ offset, message });
}

bool ensure_available(
    DacapoAdapterResult &result, std::string_view input, std::size_t offset,
    std::size_t bytes, const char *what)
{
    if (offset > input.size() || bytes > input.size() - offset)
    {
        std::ostringstream stream;
        stream << "truncated HEVM binary while reading " << what;
        add_diagnostic(result, offset, stream.str());
        return false;
    }
    return true;
}

std::uint16_t read_u16_le(std::string_view input, std::size_t offset)
{
    return static_cast<std::uint16_t>(
        static_cast<unsigned char>(input[offset]) |
        (static_cast<unsigned char>(input[offset + 1]) << 8));
}

std::uint32_t read_u32_le(std::string_view input, std::size_t offset)
{
    std::uint32_t value = 0;
    for (std::size_t i = 0; i < 4; ++i)
    {
        value |= static_cast<std::uint32_t>(
                     static_cast<unsigned char>(input[offset + i]))
                 << (8 * i);
    }
    return value;
}

std::uint64_t read_u64_le(std::string_view input, std::size_t offset)
{
    std::uint64_t value = 0;
    for (std::size_t i = 0; i < 8; ++i)
    {
        value |= static_cast<std::uint64_t>(
                     static_cast<unsigned char>(input[offset + i]))
                 << (8 * i);
    }
    return value;
}

bool checked_add(std::uint64_t left, std::uint64_t right, std::uint64_t &result)
{
    if (left > std::numeric_limits<std::uint64_t>::max() - right)
    {
        return false;
    }
    result = left + right;
    return true;
}

bool checked_mul(std::uint64_t left, std::uint64_t right, std::uint64_t &result)
{
    if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left)
    {
        return false;
    }
    result = left * right;
    return true;
}

ValueId value_id_for_register(
    DacapoAdapterResult &result, std::size_t offset, std::uint64_t base,
    std::uint64_t reg, const char *kind)
{
    std::uint64_t id = 0;
    if (!checked_add(base, reg, id) || id == 0)
    {
        std::ostringstream stream;
        stream << "HEVM " << kind << " register " << reg
               << " cannot be represented as a Poseidon value id";
        add_diagnostic(result, offset, stream.str());
        return 0;
    }
    return static_cast<ValueId>(id);
}

bool validate_register(
    DacapoAdapterResult &result, std::size_t offset, std::uint64_t reg,
    std::uint64_t count, const char *kind)
{
    if (reg < count)
    {
        return true;
    }

    std::ostringstream stream;
    stream << "HEVM " << kind << " register " << reg << " is out of range for "
           << count << " " << kind << " buffers";
    add_diagnostic(result, offset, stream.str());
    return false;
}

MgpuValueRef value(ValueId id)
{
    return MgpuValueRef{ id };
}

MgpuOp make_op(
    MgpuOpKind kind, std::vector<MgpuValueRef> inputs, std::vector<MgpuValueRef> outputs,
    std::string debug_name = {})
{
    return MgpuOp{
        kind,
        kUnassignedDacapoDevice,
        std::move(inputs),
        std::move(outputs),
        std::move(debug_name),
    };
}

DacapoAdapterResult translate_hevm_binary(std::string_view input)
{
    DacapoAdapterResult result;

    if (!ensure_available(result, input, 0, 24, "HEVM header"))
    {
        return result;
    }

    const std::uint32_t magic = read_u32_le(input, 0);
    if (magic != kHevmMagic)
    {
        add_diagnostic(result, 0, "invalid HEVM magic number");
        return result;
    }

    const std::uint32_t header_size = read_u32_le(input, 4);
    if (header_size < 24)
    {
        add_diagnostic(result, 4, "HEVM header size is smaller than the fixed header");
        return result;
    }
    if (!ensure_available(result, input, header_size, 40, "HEVM config body"))
    {
        return result;
    }

    const std::uint64_t arg_length = read_u64_le(input, 8);
    const std::uint64_t res_length = read_u64_le(input, 16);
    const std::uint64_t config_body_length = read_u64_le(input, header_size);
    const std::uint64_t num_operations = read_u64_le(input, header_size + 8);
    const std::uint64_t num_ctxt_buffer = read_u64_le(input, header_size + 16);
    const std::uint64_t num_ptxt_buffer = read_u64_le(input, header_size + 24);

    if (arg_length > num_ctxt_buffer)
    {
        add_diagnostic(
            result, 8,
            "HEVM adapter V1 expects every function argument to be a ciphertext register");
        return result;
    }

    std::uint64_t arg_array_count = 0;
    std::uint64_t result_prefix_array_count = 0;
    std::uint64_t array_entry_count = 0;
    std::uint64_t result_arrays = 0;
    if (!checked_add(res_length, res_length, result_arrays) ||
        !checked_add(result_arrays, res_length, result_arrays) ||
        !checked_add(arg_length, arg_length, arg_array_count) ||
        !checked_add(res_length, res_length, result_prefix_array_count) ||
        !checked_add(arg_array_count, result_arrays, array_entry_count))
    {
        add_diagnostic(result, header_size, "HEVM config array length overflow");
        return result;
    }

    std::uint64_t expected_config_body_length = 40;
    std::uint64_t array_bytes = 0;
    if (!checked_mul(array_entry_count, 8, array_bytes) ||
        !checked_add(40, array_bytes, expected_config_body_length))
    {
        add_diagnostic(result, header_size, "HEVM config body length overflow");
        return result;
    }

    if (config_body_length != expected_config_body_length)
    {
        std::ostringstream stream;
        stream << "unexpected HEVM config body length " << config_body_length
               << ", expected " << expected_config_body_length;
        add_diagnostic(result, header_size, stream.str());
        return result;
    }
    if (!ensure_available(
            result, input, header_size + 40, static_cast<std::size_t>(array_bytes),
            "HEVM config arrays"))
    {
        return result;
    }

    std::uint64_t plain_base = 0;
    if (!checked_add(num_ctxt_buffer, 1, plain_base))
    {
        add_diagnostic(result, header_size + 16, "HEVM ciphertext buffer count overflow");
        return result;
    }
    std::uint64_t res_dst_entry_offset = 0;
    std::uint64_t res_dst_byte_offset = 0;
    std::uint64_t header_and_fixed_config = 0;
    std::uint64_t res_dst_offset = 0;
    std::uint64_t operations_offset = 0;
    std::uint64_t operations_bytes = 0;
    if (!checked_add(arg_array_count, result_prefix_array_count, res_dst_entry_offset) ||
        !checked_mul(res_dst_entry_offset, 8, res_dst_byte_offset) ||
        !checked_add(header_size, 40, header_and_fixed_config) ||
        !checked_add(header_and_fixed_config, res_dst_byte_offset, res_dst_offset) ||
        !checked_add(header_size, config_body_length, operations_offset) ||
        !checked_mul(num_operations, 8, operations_bytes))
    {
        add_diagnostic(result, header_size, "HEVM section offset overflow");
        return result;
    }
    if (!ensure_available(
            result, input, static_cast<std::size_t>(operations_offset),
            static_cast<std::size_t>(operations_bytes), "HEVM operations"))
    {
        return result;
    }

    for (std::uint64_t arg = 0; arg < arg_length; ++arg)
    {
        const ValueId output_id = value_id_for_register(result, 8, 1, arg, "cipher");
        if (output_id == 0)
        {
            return result;
        }
        result.schedule.ops.push_back(
            make_op(MgpuOpKind::UploadCipher, {}, { value(output_id) }, "hevm_arg"));
    }

    for (std::uint64_t op_index = 0; op_index < num_operations; ++op_index)
    {
        const std::size_t offset =
            static_cast<std::size_t>(operations_offset + op_index * 8);
        const std::uint16_t opcode = read_u16_le(input, offset);
        const std::uint16_t dst = read_u16_le(input, offset + 2);
        const std::uint16_t lhs = read_u16_le(input, offset + 4);
        const std::uint16_t rhs = read_u16_le(input, offset + 6);

        if (opcode == kHevmAllocOpcode)
        {
            continue;
        }

        switch (opcode)
        {
        case 0: {
            if (!validate_register(result, offset + 2, dst, num_ptxt_buffer, "plain"))
            {
                result.schedule.ops.clear();
                return result;
            }
            const ValueId plain_dst =
                value_id_for_register(result, offset + 2, plain_base, dst, "plain");
            if (plain_dst == 0)
            {
                result.schedule.ops.clear();
                return result;
            }
            result.schedule.ops.push_back(
                make_op(MgpuOpKind::UploadPlain, {}, { value(plain_dst) }, "hevm_encode"));
            break;
        }
        case 1: {
            if (!validate_register(result, offset + 2, dst, num_ctxt_buffer, "cipher") ||
                !validate_register(result, offset + 4, lhs, num_ctxt_buffer, "cipher"))
            {
                result.schedule.ops.clear();
                return result;
            }
            const ValueId cipher_dst =
                value_id_for_register(result, offset + 2, 1, dst, "cipher");
            const ValueId cipher_lhs =
                value_id_for_register(result, offset + 4, 1, lhs, "cipher");
            result.schedule.ops.push_back(make_op(
                MgpuOpKind::Rotate, { value(cipher_lhs) }, { value(cipher_dst) },
                "hevm_rotate"));
            break;
        }
        case 3: {
            if (!validate_register(result, offset + 2, dst, num_ctxt_buffer, "cipher") ||
                !validate_register(result, offset + 4, lhs, num_ctxt_buffer, "cipher"))
            {
                result.schedule.ops.clear();
                return result;
            }
            const ValueId cipher_dst =
                value_id_for_register(result, offset + 2, 1, dst, "cipher");
            const ValueId cipher_lhs =
                value_id_for_register(result, offset + 4, 1, lhs, "cipher");
            result.schedule.ops.push_back(make_op(
                MgpuOpKind::Rescale, { value(cipher_lhs) }, { value(cipher_dst) },
                "hevm_rescale"));
            break;
        }
        case 6: {
            if (!validate_register(result, offset + 2, dst, num_ctxt_buffer, "cipher") ||
                !validate_register(result, offset + 4, lhs, num_ctxt_buffer, "cipher") ||
                !validate_register(result, offset + 6, rhs, num_ctxt_buffer, "cipher"))
            {
                result.schedule.ops.clear();
                return result;
            }
            const ValueId cipher_dst =
                value_id_for_register(result, offset + 2, 1, dst, "cipher");
            const ValueId cipher_lhs =
                value_id_for_register(result, offset + 4, 1, lhs, "cipher");
            const ValueId cipher_rhs =
                value_id_for_register(result, offset + 6, 1, rhs, "cipher");
            result.schedule.ops.push_back(make_op(
                MgpuOpKind::Add, { value(cipher_lhs), value(cipher_rhs) },
                { value(cipher_dst) }, "hevm_addcc"));
            break;
        }
        case 7: {
            if (!validate_register(result, offset + 2, dst, num_ctxt_buffer, "cipher") ||
                !validate_register(result, offset + 4, lhs, num_ctxt_buffer, "cipher") ||
                !validate_register(result, offset + 6, rhs, num_ptxt_buffer, "plain"))
            {
                result.schedule.ops.clear();
                return result;
            }
            const ValueId cipher_dst =
                value_id_for_register(result, offset + 2, 1, dst, "cipher");
            const ValueId cipher_lhs =
                value_id_for_register(result, offset + 4, 1, lhs, "cipher");
            const ValueId plain_rhs =
                value_id_for_register(result, offset + 6, plain_base, rhs, "plain");
            if (plain_rhs == 0)
            {
                result.schedule.ops.clear();
                return result;
            }
            result.schedule.ops.push_back(make_op(
                MgpuOpKind::AddPlain, { value(cipher_lhs), value(plain_rhs) },
                { value(cipher_dst) }, "hevm_addcp"));
            break;
        }
        case 8: {
            if (!validate_register(result, offset + 2, dst, num_ctxt_buffer, "cipher") ||
                !validate_register(result, offset + 4, lhs, num_ctxt_buffer, "cipher") ||
                !validate_register(result, offset + 6, rhs, num_ctxt_buffer, "cipher"))
            {
                result.schedule.ops.clear();
                return result;
            }
            const ValueId cipher_dst =
                value_id_for_register(result, offset + 2, 1, dst, "cipher");
            const ValueId cipher_lhs =
                value_id_for_register(result, offset + 4, 1, lhs, "cipher");
            const ValueId cipher_rhs =
                value_id_for_register(result, offset + 6, 1, rhs, "cipher");
            result.schedule.ops.push_back(make_op(
                MgpuOpKind::Multiply, { value(cipher_lhs), value(cipher_rhs) },
                { value(cipher_dst) }, "hevm_mulcc"));
            break;
        }
        case 9: {
            if (!validate_register(result, offset + 2, dst, num_ctxt_buffer, "cipher") ||
                !validate_register(result, offset + 4, lhs, num_ctxt_buffer, "cipher") ||
                !validate_register(result, offset + 6, rhs, num_ptxt_buffer, "plain"))
            {
                result.schedule.ops.clear();
                return result;
            }
            const ValueId cipher_dst =
                value_id_for_register(result, offset + 2, 1, dst, "cipher");
            const ValueId cipher_lhs =
                value_id_for_register(result, offset + 4, 1, lhs, "cipher");
            const ValueId plain_rhs =
                value_id_for_register(result, offset + 6, plain_base, rhs, "plain");
            if (plain_rhs == 0)
            {
                result.schedule.ops.clear();
                return result;
            }
            result.schedule.ops.push_back(make_op(
                MgpuOpKind::MultiplyPlain, { value(cipher_lhs), value(plain_rhs) },
                { value(cipher_dst) }, "hevm_mulcp"));
            break;
        }
        case 10: {
            if (!validate_register(result, offset + 2, dst, num_ctxt_buffer, "cipher") ||
                !validate_register(result, offset + 4, lhs, num_ctxt_buffer, "cipher"))
            {
                result.schedule.ops.clear();
                return result;
            }
            const ValueId cipher_dst =
                value_id_for_register(result, offset + 2, 1, dst, "cipher");
            const ValueId cipher_lhs =
                value_id_for_register(result, offset + 4, 1, lhs, "cipher");
            result.schedule.ops.push_back(make_op(
                MgpuOpKind::BootstrapFallback, { value(cipher_lhs) }, { value(cipher_dst) },
                "hevm_bootstrap"));
            break;
        }
        default: {
            std::ostringstream stream;
            stream << "unsupported HEVM opcode " << opcode;
            add_diagnostic(result, offset, stream.str());
            result.schedule.ops.clear();
            return result;
        }
        }
    }

    for (std::uint64_t result_index = 0; result_index < res_length; ++result_index)
    {
        const std::size_t offset = static_cast<std::size_t>(res_dst_offset + result_index * 8);
        const std::uint64_t result_register = read_u64_le(input, offset);
        if (!validate_register(result, offset, result_register, num_ctxt_buffer, "cipher"))
        {
            result.schedule.ops.clear();
            return result;
        }
        const ValueId input_id =
            value_id_for_register(result, offset, 1, result_register, "cipher");
        if (input_id == 0)
        {
            result.schedule.ops.clear();
            return result;
        }
        result.schedule.ops.push_back(
            make_op(MgpuOpKind::Download, { value(input_id) }, {}, "hevm_result"));
    }

    return result;
}

}  // namespace

const char *to_string(DacapoInputFormat format) noexcept
{
    switch (format)
    {
    case DacapoInputFormat::Unknown:
        return "unknown";
    case DacapoInputFormat::EarthMlirText:
        return "earth_mlir_text";
    case DacapoInputFormat::HevmBinary:
        return "hevm_binary";
    case DacapoInputFormat::Json:
        return "json";
    }
    return "unknown";
}

std::string DacapoAdapterResult::format_diagnostics() const
{
    std::ostringstream stream;
    for (std::size_t i = 0; i < diagnostics.size(); ++i)
    {
        if (i > 0)
        {
            stream << '\n';
        }
        stream << "offset " << diagnostics[i].offset << ": " << diagnostics[i].message;
    }
    return stream.str();
}

DacapoAdapterResult translate_dacapo_schedule(
    std::string_view input, const DacapoAdapterOptions &options)
{
    DacapoAdapterResult result;

    if (input.empty())
    {
        add_diagnostic(result, 0, "Dacapo adapter input is empty");
        return result;
    }

    if (options.input_format == DacapoInputFormat::Json)
    {
        ScheduleJsonParseResult parsed = parse_schedule_json(input);
        if (parsed.ok())
        {
            result.schedule = std::move(parsed.schedule);
            return result;
        }

        for (const ScheduleJsonDiagnostic &diagnostic : parsed.diagnostics)
        {
            add_diagnostic(
                result, 0,
                "JSON schedule " + diagnostic.path + ": " + diagnostic.message);
        }
        return result;
    }

    if (options.input_format == DacapoInputFormat::HevmBinary)
    {
        return translate_hevm_binary(input);
    }

    std::ostringstream stream;
    stream << "Dacapo " << to_string(options.input_format)
           << " translation is not implemented; capture the optimized Dacapo output and "
              "map it into poseidon::mgpu::MgpuSchedule before enabling execution";
    add_diagnostic(result, 0, stream.str());
    return result;
}

}  // namespace poseidon::mgpu
