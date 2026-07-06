#include "poseidon/frontends/dacapo/dacapo_constants.h"

#include <cstdint>
#include <cstring>
#include <limits>
#include <sstream>
#include <utility>

namespace poseidon::mgpu
{
namespace
{

void add_diagnostic(
    DacapoConstantParseResult &result, std::size_t offset, std::string message)
{
    result.diagnostics.push_back(DacapoConstantDiagnostic{ offset, std::move(message) });
}

bool ensure_available(
    DacapoConstantParseResult &result, std::string_view input, std::size_t offset,
    std::size_t bytes, const char *what)
{
    if (offset > input.size() || bytes > input.size() - offset)
    {
        std::ostringstream stream;
        stream << "truncated Dacapo constant file while reading " << what;
        add_diagnostic(result, offset, stream.str());
        return false;
    }
    return true;
}

std::int64_t read_i64_le(std::string_view input, std::size_t offset)
{
    std::uint64_t value = 0;
    for (std::size_t i = 0; i < 8; ++i)
    {
        value |= static_cast<std::uint64_t>(
                     static_cast<unsigned char>(input[offset + i]))
                 << (8 * i);
    }
    return static_cast<std::int64_t>(value);
}

double read_double_le(std::string_view input, std::size_t offset)
{
    std::uint64_t bits = 0;
    for (std::size_t i = 0; i < 8; ++i)
    {
        bits |= static_cast<std::uint64_t>(
                    static_cast<unsigned char>(input[offset + i]))
                << (8 * i);
    }

    double value = 0.0;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

bool i64_to_size(
    DacapoConstantParseResult &result, std::size_t offset, std::int64_t value,
    const char *what, std::size_t &converted)
{
    if (value < 0)
    {
        std::ostringstream stream;
        stream << "negative Dacapo constant " << what << ' ' << value;
        add_diagnostic(result, offset, stream.str());
        return false;
    }
    if (static_cast<std::uint64_t>(value) >
        static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
    {
        std::ostringstream stream;
        stream << "Dacapo constant " << what << " exceeds size_t";
        add_diagnostic(result, offset, stream.str());
        return false;
    }

    converted = static_cast<std::size_t>(value);
    return true;
}

bool checked_mul_size(std::size_t left, std::size_t right, std::size_t &result)
{
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
    {
        return false;
    }

    result = left * right;
    return true;
}

bool ensure_minimum_record_bytes(
    DacapoConstantParseResult &result, std::string_view input, std::size_t offset,
    std::size_t count)
{
    std::size_t minimum_bytes = 0;
    if (!checked_mul_size(count, 8, minimum_bytes))
    {
        add_diagnostic(result, offset, "Dacapo constant record byte length overflow");
        return false;
    }
    if (minimum_bytes > input.size() - offset)
    {
        add_diagnostic(
            result, offset,
            "Dacapo constant count exceeds the remaining file length");
        return false;
    }
    return true;
}

}  // namespace

std::string DacapoConstantParseResult::format_diagnostics() const
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

DacapoConstantParseResult parse_dacapo_constant_file(std::string_view input)
{
    DacapoConstantParseResult result;
    if (!ensure_available(result, input, 0, 8, "constant count"))
    {
        return result;
    }

    std::size_t offset = 0;
    std::size_t constant_count = 0;
    if (!i64_to_size(
            result, offset, read_i64_le(input, offset), "count", constant_count))
    {
        return result;
    }
    offset += 8;

    if (!ensure_minimum_record_bytes(result, input, offset, constant_count))
    {
        return result;
    }

    for (std::size_t constant_index = 0; constant_index < constant_count; ++constant_index)
    {
        if (!ensure_available(result, input, offset, 8, "constant vector length"))
        {
            result.table.values.clear();
            return result;
        }

        std::size_t vector_length = 0;
        if (!i64_to_size(
                result, offset, read_i64_le(input, offset), "vector length",
                vector_length))
        {
            result.table.values.clear();
            return result;
        }
        offset += 8;

        std::size_t data_bytes = 0;
        if (!checked_mul_size(vector_length, sizeof(double), data_bytes))
        {
            add_diagnostic(result, offset, "Dacapo constant vector byte length overflow");
            result.table.values.clear();
            return result;
        }
        if (!ensure_available(result, input, offset, data_bytes, "constant vector data"))
        {
            result.table.values.clear();
            return result;
        }

        std::vector<double> values;
        values.reserve(vector_length);
        for (std::size_t value_index = 0; value_index < vector_length; ++value_index)
        {
            values.push_back(read_double_le(input, offset + value_index * sizeof(double)));
        }
        offset += data_bytes;
        result.table.values.push_back(std::move(values));
    }

    if (offset != input.size())
    {
        std::ostringstream stream;
        stream << "Dacapo constant file has " << (input.size() - offset)
               << " trailing bytes";
        add_diagnostic(result, offset, stream.str());
        result.table.values.clear();
    }

    return result;
}

}  // namespace poseidon::mgpu
