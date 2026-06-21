#include "poseidon/mgpu/compiler/dacapo_adapter.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace poseidon::mgpu;

namespace
{

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

void test_format_strings()
{
    require(std::string(to_string(DacapoInputFormat::Unknown)) == "unknown",
            "unknown format string mismatch");
    require(std::string(to_string(DacapoInputFormat::EarthMlirText)) == "earth_mlir_text",
            "MLIR format string mismatch");
    require(std::string(to_string(DacapoInputFormat::Json)) == "json",
            "JSON format string mismatch");
}

void test_empty_input_fails_clearly()
{
    const DacapoAdapterResult result = translate_dacapo_schedule(
        "", DacapoAdapterOptions{ DacapoInputFormat::EarthMlirText });
    require(!result.ok(), "empty input should fail");
    require_contains(result.format_diagnostics(), "input is empty");
}

void test_unknown_dacapo_format_does_not_guess()
{
    const DacapoAdapterResult result = translate_dacapo_schedule(
        "module { func.func @main() }",
        DacapoAdapterOptions{ DacapoInputFormat::EarthMlirText });
    require(!result.ok(), "unimplemented adapter should fail");
    require_contains(result.format_diagnostics(), "translation is not implemented");
    require(result.schedule.ops.empty(), "failed adapter should not produce a schedule");
}

}  // namespace

int main()
{
    try
    {
        test_format_strings();
        test_empty_input_fails_clearly();
        test_unknown_dacapo_format_does_not_guess();
    }
    catch (const std::exception &ex)
    {
        std::cerr << "mgpu Dacapo adapter test failed: " << ex.what() << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "mgpu Dacapo adapter tests passed\n";
    return EXIT_SUCCESS;
}
