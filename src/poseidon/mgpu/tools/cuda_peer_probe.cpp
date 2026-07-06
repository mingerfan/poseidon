#include "poseidon/mgpu/comm/cuda_peer_probe.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace poseidon::mgpu;

namespace
{

struct ToolOptions
{
    bool summary_json = false;
    int require_devices = 0;
    bool require_full_peer_access = false;
};

void print_usage(std::ostream &stream)
{
    stream
        << "usage: poseidon_mgpu_cuda_peer_probe "
           "[--summary-json] [--require-devices N] [--require-full-peer-access]\n";
}

int parse_int_arg(const std::string &name, const char *value)
{
    if (value == nullptr)
    {
        throw std::invalid_argument("missing value for " + name);
    }

    std::size_t consumed = 0;
    const int parsed = std::stoi(value, &consumed);
    if (consumed != std::string(value).size())
    {
        throw std::invalid_argument("invalid integer for " + name + ": " + value);
    }
    return parsed;
}

ToolOptions parse_args(int argc, char **argv)
{
    ToolOptions options;
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h")
        {
            print_usage(std::cout);
            std::exit(EXIT_SUCCESS);
        }
        if (arg == "--summary-json")
        {
            options.summary_json = true;
            continue;
        }
        if (arg == "--require-devices")
        {
            if (++i >= argc)
            {
                throw std::invalid_argument("missing value for --require-devices");
            }
            options.require_devices = parse_int_arg("--require-devices", argv[i]);
            continue;
        }
        if (arg == "--require-full-peer-access")
        {
            options.require_full_peer_access = true;
            continue;
        }

        throw std::invalid_argument("unknown argument: " + arg);
    }

    if (options.require_devices < 0)
    {
        throw std::invalid_argument("--require-devices must be non-negative");
    }
    return options;
}

}  // namespace

int main(int argc, char **argv)
{
    try
    {
        const ToolOptions options = parse_args(argc, argv);
        const CudaPeerProbeResult result = probe_cuda_peer_access();
        if (options.summary_json)
        {
            std::cout << cuda_peer_probe_to_json(result, 2) << '\n';
        }
        else
        {
            dump_cuda_peer_probe(std::cout, result);
        }

        if (options.require_devices > 0 &&
            result.visible_device_count < options.require_devices)
        {
            std::cerr << "poseidon_mgpu_cuda_peer_probe: visible CUDA devices "
                      << result.visible_device_count << " < required "
                      << options.require_devices << '\n';
            return EXIT_FAILURE;
        }

        if (options.require_full_peer_access &&
            !cuda_peer_probe_has_full_peer_access(result, options.require_devices))
        {
            std::cerr
                << "poseidon_mgpu_cuda_peer_probe: required devices do not have "
                   "full bidirectional peer access\n";
            return EXIT_FAILURE;
        }
    }
    catch (const std::exception &ex)
    {
        std::cerr << "poseidon_mgpu_cuda_peer_probe: " << ex.what() << '\n';
        print_usage(std::cerr);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
