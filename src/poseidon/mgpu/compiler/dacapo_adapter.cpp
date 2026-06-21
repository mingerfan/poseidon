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
constexpr std::uint64_t kMaxHevmRegisterCount =
    static_cast<std::uint64_t>(std::numeric_limits<std::uint16_t>::max()) + 1;
constexpr std::uint64_t kHevmFixedConfigBytes = 40;
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

bool read_u64_array(
    DacapoAdapterResult &result, std::string_view input, std::size_t offset,
    std::uint64_t count, std::vector<std::uint64_t> &values, const char *what)
{
    values.clear();
    values.reserve(static_cast<std::size_t>(count));

    std::uint64_t bytes = 0;
    if (!checked_mul(count, 8, bytes))
    {
        std::ostringstream stream;
        stream << "HEVM " << what << " byte length overflow";
        add_diagnostic(result, offset, stream.str());
        return false;
    }
    if (!ensure_available(result, input, offset, static_cast<std::size_t>(bytes), what))
    {
        return false;
    }

    for (std::uint64_t i = 0; i < count; ++i)
    {
        values.push_back(read_u64_le(input, offset + static_cast<std::size_t>(i * 8)));
    }
    return true;
}

bool set_u64_attr(
    DacapoAdapterResult &result, MgpuOp &op, std::size_t offset,
    const char *name, std::uint64_t value)
{
    if (value > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
    {
        std::ostringstream stream;
        stream << "HEVM metadata '" << name << "' exceeds int64_t";
        add_diagnostic(result, offset, stream.str());
        return false;
    }

    op.integer_attributes.emplace(name, static_cast<std::int64_t>(value));
    return true;
}

ValueId allocate_value_id(
    DacapoAdapterResult &result, std::size_t offset, ValueId &next_value_id)
{
    if (next_value_id == 0)
    {
        add_diagnostic(result, offset, "Poseidon value id 0 is reserved");
        return 0;
    }

    const ValueId id = next_value_id;
    if (next_value_id == std::numeric_limits<ValueId>::max())
    {
        add_diagnostic(result, offset, "Poseidon value id space exhausted");
        next_value_id = 0;
        return 0;
    }

    ++next_value_id;
    return id;
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

ValueId lookup_register_value(
    DacapoAdapterResult &result, std::size_t offset,
    const std::vector<ValueId> &values, std::uint64_t reg, const char *kind)
{
    if (!validate_register(result, offset, reg, values.size(), kind))
    {
        return 0;
    }

    const ValueId id = values[static_cast<std::size_t>(reg)];
    if (id == 0)
    {
        std::ostringstream stream;
        stream << "HEVM " << kind << " register " << reg << " is used before definition";
        add_diagnostic(result, offset, stream.str());
    }
    return id;
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

MgpuOp make_op_with_attr(
    MgpuOpKind kind, std::vector<MgpuValueRef> inputs, std::vector<MgpuValueRef> outputs,
    std::string debug_name, std::string attr_name, std::int64_t attr_value)
{
    MgpuOp op = make_op(kind, std::move(inputs), std::move(outputs), std::move(debug_name));
    op.integer_attributes.emplace(std::move(attr_name), attr_value);
    return op;
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
    if (!ensure_available(
            result, input, header_size, kHevmFixedConfigBytes, "HEVM config body"))
    {
        return result;
    }

    const std::uint64_t arg_length = read_u64_le(input, 8);
    const std::uint64_t res_length = read_u64_le(input, 16);
    const std::uint64_t config_body_length = read_u64_le(input, header_size);
    const std::uint64_t num_operations = read_u64_le(input, header_size + 8);
    const std::uint64_t num_ctxt_buffer = read_u64_le(input, header_size + 16);
    const std::uint64_t num_ptxt_buffer = read_u64_le(input, header_size + 24);
    const std::uint64_t init_level = read_u64_le(input, header_size + 32);

    if (arg_length > num_ctxt_buffer)
    {
        add_diagnostic(
            result, 8,
            "HEVM adapter V1 expects every function argument to be a ciphertext register");
        return result;
    }
    if (num_ctxt_buffer > kMaxHevmRegisterCount || num_ptxt_buffer > kMaxHevmRegisterCount)
    {
        add_diagnostic(
            result, header_size + 16,
            "HEVM register count exceeds the 16-bit operation register space");
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

    std::uint64_t expected_config_body_length = kHevmFixedConfigBytes;
    std::uint64_t array_bytes = 0;
    if (!checked_mul(array_entry_count, 8, array_bytes) ||
        !checked_add(kHevmFixedConfigBytes, array_bytes, expected_config_body_length))
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
            result, input, header_size + kHevmFixedConfigBytes,
            static_cast<std::size_t>(array_bytes), "HEVM config arrays"))
    {
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
        !checked_add(header_size, kHevmFixedConfigBytes, header_and_fixed_config) ||
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

    std::uint64_t arg_scale_offset = 0;
    std::uint64_t arg_metadata_bytes = 0;
    std::uint64_t arg_level_offset = 0;
    std::uint64_t result_scale_offset = 0;
    std::uint64_t result_metadata_bytes = 0;
    std::uint64_t result_level_offset = 0;
    if (!checked_add(header_size, kHevmFixedConfigBytes, arg_scale_offset) ||
        !checked_mul(arg_length, 8, arg_metadata_bytes) ||
        !checked_add(arg_scale_offset, arg_metadata_bytes, arg_level_offset) ||
        !checked_add(arg_level_offset, arg_metadata_bytes, result_scale_offset) ||
        !checked_mul(res_length, 8, result_metadata_bytes) ||
        !checked_add(result_scale_offset, result_metadata_bytes, result_level_offset))
    {
        add_diagnostic(result, header_size, "HEVM metadata offset overflow");
        return result;
    }

    std::vector<std::uint64_t> arg_scales;
    std::vector<std::uint64_t> arg_levels;
    std::vector<std::uint64_t> result_scales;
    std::vector<std::uint64_t> result_levels;
    if (!read_u64_array(
            result, input, static_cast<std::size_t>(arg_scale_offset), arg_length,
            arg_scales, "HEVM arg_scale array") ||
        !read_u64_array(
            result, input, static_cast<std::size_t>(arg_level_offset), arg_length,
            arg_levels, "HEVM arg_level array") ||
        !read_u64_array(
            result, input, static_cast<std::size_t>(result_scale_offset), res_length,
            result_scales, "HEVM res_scale array") ||
        !read_u64_array(
            result, input, static_cast<std::size_t>(result_level_offset), res_length,
            result_levels, "HEVM res_level array"))
    {
        result.schedule.ops.clear();
        return result;
    }

    std::vector<ValueId> cipher_values(static_cast<std::size_t>(num_ctxt_buffer), 0);
    std::vector<ValueId> plain_values(static_cast<std::size_t>(num_ptxt_buffer), 0);
    ValueId next_value_id = 1;

    for (std::uint64_t arg = 0; arg < arg_length; ++arg)
    {
        const ValueId output_id = allocate_value_id(result, 8, next_value_id);
        if (output_id == 0)
        {
            return result;
        }
        cipher_values[static_cast<std::size_t>(arg)] = output_id;
        MgpuOp op = make_op(MgpuOpKind::UploadCipher, {}, { value(output_id) }, "hevm_arg");
        if (!set_u64_attr(result, op, 8, "hevm_arg_index", arg) ||
            !set_u64_attr(
                result, op, static_cast<std::size_t>(arg_scale_offset + arg * 8),
                "hevm_arg_scale", arg_scales[static_cast<std::size_t>(arg)]) ||
            !set_u64_attr(
                result, op, static_cast<std::size_t>(arg_level_offset + arg * 8),
                "hevm_arg_level", arg_levels[static_cast<std::size_t>(arg)]) ||
            !set_u64_attr(result, op, header_size + 32, "hevm_init_level", init_level))
        {
            result.schedule.ops.clear();
            return result;
        }
        result.schedule.ops.push_back(std::move(op));
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
            const ValueId output_id = allocate_value_id(result, offset + 2, next_value_id);
            if (output_id == 0)
            {
                result.schedule.ops.clear();
                return result;
            }
            plain_values[dst] = output_id;
            MgpuOp op = make_op(
                MgpuOpKind::UploadPlain, {}, { value(output_id) }, "hevm_encode");
            op.integer_attributes.emplace("encode_level", rhs >> 10);
            op.integer_attributes.emplace("encode_scale", rhs & 0x3FF);
            result.schedule.ops.push_back(std::move(op));
            break;
        }
        case 1: {
            if (!validate_register(result, offset + 2, dst, num_ctxt_buffer, "cipher") ||
                !validate_register(result, offset + 4, lhs, num_ctxt_buffer, "cipher"))
            {
                result.schedule.ops.clear();
                return result;
            }
            const ValueId input_id =
                lookup_register_value(result, offset + 4, cipher_values, lhs, "cipher");
            const ValueId output_id = allocate_value_id(result, offset + 2, next_value_id);
            if (input_id == 0 || output_id == 0)
            {
                result.schedule.ops.clear();
                return result;
            }
            cipher_values[dst] = output_id;
            result.schedule.ops.push_back(make_op_with_attr(
                MgpuOpKind::Rotate, { value(input_id) }, { value(output_id) },
                "hevm_rotate", "rotate_step", static_cast<std::int16_t>(rhs)));
            break;
        }
        case 3: {
            if (!validate_register(result, offset + 2, dst, num_ctxt_buffer, "cipher") ||
                !validate_register(result, offset + 4, lhs, num_ctxt_buffer, "cipher"))
            {
                result.schedule.ops.clear();
                return result;
            }
            const ValueId input_id =
                lookup_register_value(result, offset + 4, cipher_values, lhs, "cipher");
            const ValueId output_id = allocate_value_id(result, offset + 2, next_value_id);
            if (input_id == 0 || output_id == 0)
            {
                result.schedule.ops.clear();
                return result;
            }
            cipher_values[dst] = output_id;
            result.schedule.ops.push_back(make_op(
                MgpuOpKind::Rescale, { value(input_id) }, { value(output_id) },
                "hevm_rescale"));
            break;
        }
        case 2: {
            if (!validate_register(result, offset + 2, dst, num_ctxt_buffer, "cipher") ||
                !validate_register(result, offset + 4, lhs, num_ctxt_buffer, "cipher"))
            {
                result.schedule.ops.clear();
                return result;
            }
            const ValueId input_id =
                lookup_register_value(result, offset + 4, cipher_values, lhs, "cipher");
            const ValueId output_id = allocate_value_id(result, offset + 2, next_value_id);
            if (input_id == 0 || output_id == 0)
            {
                result.schedule.ops.clear();
                return result;
            }
            cipher_values[dst] = output_id;
            result.schedule.ops.push_back(make_op(
                MgpuOpKind::Negate, { value(input_id) }, { value(output_id) },
                "hevm_negate"));
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
            const ValueId left_id =
                lookup_register_value(result, offset + 4, cipher_values, lhs, "cipher");
            const ValueId right_id =
                lookup_register_value(result, offset + 6, cipher_values, rhs, "cipher");
            const ValueId output_id = allocate_value_id(result, offset + 2, next_value_id);
            if (left_id == 0 || right_id == 0 || output_id == 0)
            {
                result.schedule.ops.clear();
                return result;
            }
            cipher_values[dst] = output_id;
            result.schedule.ops.push_back(make_op(
                MgpuOpKind::Add, { value(left_id), value(right_id) },
                { value(output_id) }, "hevm_addcc"));
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
            const ValueId left_id =
                lookup_register_value(result, offset + 4, cipher_values, lhs, "cipher");
            const ValueId right_id =
                lookup_register_value(result, offset + 6, plain_values, rhs, "plain");
            const ValueId output_id = allocate_value_id(result, offset + 2, next_value_id);
            if (left_id == 0 || right_id == 0 || output_id == 0)
            {
                result.schedule.ops.clear();
                return result;
            }
            cipher_values[dst] = output_id;
            result.schedule.ops.push_back(make_op(
                MgpuOpKind::AddPlain, { value(left_id), value(right_id) },
                { value(output_id) }, "hevm_addcp"));
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
            const ValueId left_id =
                lookup_register_value(result, offset + 4, cipher_values, lhs, "cipher");
            const ValueId right_id =
                lookup_register_value(result, offset + 6, cipher_values, rhs, "cipher");
            const ValueId output_id = allocate_value_id(result, offset + 2, next_value_id);
            if (left_id == 0 || right_id == 0 || output_id == 0)
            {
                result.schedule.ops.clear();
                return result;
            }
            cipher_values[dst] = output_id;
            result.schedule.ops.push_back(make_op(
                MgpuOpKind::Multiply, { value(left_id), value(right_id) },
                { value(output_id) }, "hevm_mulcc"));
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
            const ValueId left_id =
                lookup_register_value(result, offset + 4, cipher_values, lhs, "cipher");
            const ValueId right_id =
                lookup_register_value(result, offset + 6, plain_values, rhs, "plain");
            const ValueId output_id = allocate_value_id(result, offset + 2, next_value_id);
            if (left_id == 0 || right_id == 0 || output_id == 0)
            {
                result.schedule.ops.clear();
                return result;
            }
            cipher_values[dst] = output_id;
            result.schedule.ops.push_back(make_op(
                MgpuOpKind::MultiplyPlain, { value(left_id), value(right_id) },
                { value(output_id) }, "hevm_mulcp"));
            break;
        }
        case 10: {
            if (!validate_register(result, offset + 2, dst, num_ctxt_buffer, "cipher") ||
                !validate_register(result, offset + 4, lhs, num_ctxt_buffer, "cipher"))
            {
                result.schedule.ops.clear();
                return result;
            }
            const ValueId input_id =
                lookup_register_value(result, offset + 4, cipher_values, lhs, "cipher");
            const ValueId output_id = allocate_value_id(result, offset + 2, next_value_id);
            if (input_id == 0 || output_id == 0)
            {
                result.schedule.ops.clear();
                return result;
            }
            cipher_values[dst] = output_id;
            result.schedule.ops.push_back(make_op_with_attr(
                MgpuOpKind::BootstrapFallback, { value(input_id) }, { value(output_id) },
                "hevm_bootstrap", "target_level", rhs));
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
            lookup_register_value(result, offset, cipher_values, result_register, "cipher");
        if (input_id == 0)
        {
            result.schedule.ops.clear();
            return result;
        }
        MgpuOp op = make_op(MgpuOpKind::Download, { value(input_id) }, {}, "hevm_result");
        if (!set_u64_attr(result, op, offset, "hevm_result_index", result_index) ||
            !set_u64_attr(result, op, offset, "hevm_result_register", result_register) ||
            !set_u64_attr(
                result, op, static_cast<std::size_t>(result_scale_offset + result_index * 8),
                "hevm_result_scale",
                result_scales[static_cast<std::size_t>(result_index)]) ||
            !set_u64_attr(
                result, op, static_cast<std::size_t>(result_level_offset + result_index * 8),
                "hevm_result_level",
                result_levels[static_cast<std::size_t>(result_index)]))
        {
            result.schedule.ops.clear();
            return result;
        }
        result.schedule.ops.push_back(std::move(op));
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
