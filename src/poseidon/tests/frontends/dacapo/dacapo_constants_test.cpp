#include "poseidon/frontends/dacapo/dacapo_constants.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace poseidon::mgpu;

namespace
{

void append_i64(std::string &output, std::int64_t value)
{
    const auto bits = static_cast<std::uint64_t>(value);
    for (int i = 0; i < 8; ++i)
    {
        output.push_back(static_cast<char>((bits >> (8 * i)) & 0xFF));
    }
}

void append_double(std::string &output, double value)
{
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    for (int i = 0; i < 8; ++i)
    {
        output.push_back(static_cast<char>((bits >> (8 * i)) & 0xFF));
    }
}

void require(bool condition, const std::string &message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}

void require_contains(const std::string &text, const std::string &needle)
{
    if (text.find(needle) == std::string::npos)
    {
        throw std::runtime_error("expected text to contain: " + needle + "\ntext:\n" + text);
    }
}

void require_double_eq(double actual, double expected, const std::string &message)
{
    if (std::fabs(actual - expected) > 0.0)
    {
        throw std::runtime_error(message);
    }
}

std::string make_constant_file()
{
    std::string output;
    append_i64(output, 2);
    append_i64(output, 3);
    append_double(output, 1.25);
    append_double(output, -2.5);
    append_double(output, 3.75);
    append_i64(output, 1);
    append_double(output, 9.0);
    return output;
}

void test_parse_constant_file()
{
    const DacapoConstantParseResult result =
        parse_dacapo_constant_file(make_constant_file());

    require(result.ok(), "constant parse failed:\n" + result.format_diagnostics());
    require(result.table.values.size() == 2, "constant count mismatch");
    require(result.table.values[0].size() == 3, "first constant length mismatch");
    require_double_eq(result.table.values[0][0], 1.25, "first constant value mismatch");
    require_double_eq(result.table.values[0][1], -2.5, "second constant value mismatch");
    require_double_eq(result.table.values[0][2], 3.75, "third constant value mismatch");
    require(result.table.values[1].size() == 1, "second constant length mismatch");
    require_double_eq(result.table.values[1][0], 9.0, "second constant value mismatch");
}

void test_rejects_truncated_vector_data()
{
    std::string input;
    append_i64(input, 1);
    append_i64(input, 2);
    append_double(input, 1.0);

    const DacapoConstantParseResult result = parse_dacapo_constant_file(input);
    require(!result.ok(), "truncated vector data should fail");
    require(result.table.values.empty(), "failed parse should clear partial constants");
    require_contains(result.format_diagnostics(), "truncated Dacapo constant file");
    require_contains(result.format_diagnostics(), "constant vector data");
}

void test_rejects_negative_count()
{
    std::string input;
    append_i64(input, -1);

    const DacapoConstantParseResult result = parse_dacapo_constant_file(input);
    require(!result.ok(), "negative count should fail");
    require_contains(result.format_diagnostics(), "negative Dacapo constant count");
}

void test_rejects_count_without_allocating()
{
    std::string input;
    append_i64(input, 8);

    const DacapoConstantParseResult result = parse_dacapo_constant_file(input);
    require(!result.ok(), "count beyond remaining file length should fail");
    require(result.table.values.empty(), "failed parse should not allocate constants");
    require_contains(result.format_diagnostics(), "count exceeds the remaining file length");
}

void test_rejects_trailing_bytes()
{
    std::string input = make_constant_file();
    input.push_back('x');

    const DacapoConstantParseResult result = parse_dacapo_constant_file(input);
    require(!result.ok(), "trailing bytes should fail");
    require(result.table.values.empty(), "failed parse should clear constants");
    require_contains(result.format_diagnostics(), "trailing bytes");
}

}  // namespace

int main()
{
    try
    {
        test_parse_constant_file();
        test_rejects_truncated_vector_data();
        test_rejects_negative_count();
        test_rejects_count_without_allocating();
        test_rejects_trailing_bytes();
    }
    catch (const std::exception &ex)
    {
        std::cerr << "mgpu Dacapo constants test failed: " << ex.what() << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "mgpu Dacapo constants tests passed\n";
    return EXIT_SUCCESS;
}
