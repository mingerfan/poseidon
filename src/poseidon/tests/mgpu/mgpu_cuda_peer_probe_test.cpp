#include "poseidon/mgpu/comm/cuda_peer_probe.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace poseidon::mgpu;

namespace
{

constexpr int kSkip = 77;

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

void test_probe_visible_devices()
{
    CudaPeerProbeResult result;
    try
    {
        result = probe_cuda_peer_access();
    }
    catch (const std::exception &ex)
    {
        std::cout << "CUDA runtime unavailable: " << ex.what()
                  << "; skipping CUDA peer probe test\n";
        std::exit(kSkip);
    }
    if (result.visible_device_count <= 0)
    {
        std::cout << "no CUDA devices visible; skipping CUDA peer probe test\n";
        std::exit(kSkip);
    }

    require(
        result.devices.size() == static_cast<std::size_t>(result.visible_device_count),
        "device info count mismatch");
    require(
        result.peer_access.size() == static_cast<std::size_t>(result.visible_device_count),
        "peer access row count mismatch");
    for (int device = 0; device < result.visible_device_count; ++device)
    {
        require(
            result.peer_access[static_cast<std::size_t>(device)].size() ==
                static_cast<std::size_t>(result.visible_device_count),
            "peer access column count mismatch");
        require(
            result.peer_access[static_cast<std::size_t>(device)]
                              [static_cast<std::size_t>(device)],
            "self peer access should be true");
    }

    const std::string text = dump_cuda_peer_probe(result);
    require_contains(text, "cuda_peer_probe:");
    require_contains(text, "visible_devices:");
    require_contains(text, "peer_access_matrix:");

    const std::string json = cuda_peer_probe_to_json(result);
    require_contains(json, "\"visible_device_count\"");
    require_contains(json, "\"peer_access_matrix\"");
}

void test_full_peer_access_requirement()
{
    CudaPeerProbeResult result;
    result.visible_device_count = 2;
    result.peer_access = {
        { true, true },
        { false, true },
    };

    require(
        !cuda_peer_probe_has_full_peer_access(result, 2),
        "missing reverse peer access should fail full peer requirement");
    result.peer_access[1][0] = true;
    require(
        cuda_peer_probe_has_full_peer_access(result, 2),
        "full bidirectional peer access should pass");
    require(
        !cuda_peer_probe_has_full_peer_access(result, 3),
        "requiring more devices than visible should fail");
}

}  // namespace

int main()
{
    try
    {
        test_full_peer_access_requirement();
        test_probe_visible_devices();
    }
    catch (const std::exception &ex)
    {
        std::cerr << "mgpu CUDA peer probe test failed: " << ex.what() << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "mgpu CUDA peer probe tests passed\n";
    return EXIT_SUCCESS;
}
