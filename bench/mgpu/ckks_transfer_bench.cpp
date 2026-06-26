#include "poseidon/gpu/gpu_ciphertext.h"
#include "poseidon/mgpu/comm/cuda_peer_comm.h"
#include "poseidon/mgpu/comm/gpu_object_materializer.h"
#include "poseidon/mgpu/comm/gpu_comm.h"

#include <cuda_runtime_api.h>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace poseidon;
using namespace poseidon::mgpu;

namespace
{

struct BenchOptions
{
    int source_device = 0;
    int destination_device = 1;
    int iterations = 50;
    int warmup_iterations = 5;
    std::size_t degree = 1 << 15;
    std::size_t component_count = 2;
    std::size_t p_count = 0;
    bool allow_same_device = false;
};

struct LevelShape
{
    const char *name = "";
    std::size_t q_count = 0;
};

struct Stats
{
    double average_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;
};

struct BenchCaseResult
{
    Stats stats;
    std::size_t ciphertext_bytes = 0;
    std::size_t total_bytes = 0;
};

class RmmPoolScope
{
public:
    explicit RmmPoolScope(const std::vector<int> &device_ids)
    {
        scopes_.reserve(device_ids.size());
        for (const int device_id : device_ids)
        {
            check_cuda(cudaSetDevice(device_id), "cudaSetDevice");

            auto scope = std::make_unique<DeviceScope>();
            scope->device_id = device_id;
            scope->upstream = std::make_unique<rmm::mr::cuda_memory_resource>();
            scope->pool =
                std::make_unique<rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>>(
                    scope->upstream.get(), 1 << 26, std::nullopt);
            scope->previous = rmm::mr::get_current_device_resource();
            rmm::mr::set_current_device_resource(scope->pool.get());
            scopes_.push_back(std::move(scope));
        }
    }

    RmmPoolScope(const RmmPoolScope &) = delete;
    RmmPoolScope &operator=(const RmmPoolScope &) = delete;

    ~RmmPoolScope()
    {
        for (auto iter = scopes_.rbegin(); iter != scopes_.rend(); ++iter)
        {
            try
            {
                (void)cudaSetDevice((*iter)->device_id);
                rmm::mr::set_current_device_resource((*iter)->previous);
            }
            catch (...)
            {}
        }
    }

private:
    static void check_cuda(cudaError_t status, const char *what)
    {
        if (status != cudaSuccess)
        {
            throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
        }
    }

    struct DeviceScope
    {
        int device_id = 0;
        std::unique_ptr<rmm::mr::cuda_memory_resource> upstream;
        std::unique_ptr<rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>> pool;
        rmm::mr::device_memory_resource *previous = nullptr;
    };

    std::vector<std::unique_ptr<DeviceScope>> scopes_;
};

void check_cuda(cudaError_t status, const char *what)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

void print_usage(const char *program)
{
    std::cerr
        << "usage: " << program
        << " [--source-device N] [--destination-device N]"
           " [--iterations N] [--warmup N] [--degree N]"
           " [--components N] [--p-count N] [--allow-same-device]\n";
}

int parse_int(const char *value, const char *name)
{
    char *end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0' ||
        parsed < std::numeric_limits<int>::min() ||
        parsed > std::numeric_limits<int>::max())
    {
        throw std::invalid_argument(std::string("invalid integer for ") + name);
    }
    return static_cast<int>(parsed);
}

std::size_t parse_size(const char *value, const char *name)
{
    if (value[0] == '-')
    {
        throw std::invalid_argument(std::string("invalid size for ") + name);
    }

    char *end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (end == value || *end != '\0')
    {
        throw std::invalid_argument(std::string("invalid size for ") + name);
    }
    return static_cast<std::size_t>(parsed);
}

BenchOptions parse_options(int argc, char **argv)
{
    BenchOptions options;
    for (int index = 1; index < argc; ++index)
    {
        const std::string arg = argv[index];
        auto require_value = [&](const char *name) -> const char * {
            if (index + 1 >= argc)
            {
                throw std::invalid_argument(std::string("missing value for ") + name);
            }
            return argv[++index];
        };

        if (arg == "--source-device")
        {
            options.source_device = parse_int(require_value("--source-device"), arg.c_str());
        }
        else if (arg == "--destination-device")
        {
            options.destination_device = parse_int(require_value("--destination-device"), arg.c_str());
        }
        else if (arg == "--iterations")
        {
            options.iterations = parse_int(require_value("--iterations"), arg.c_str());
        }
        else if (arg == "--warmup")
        {
            options.warmup_iterations = parse_int(require_value("--warmup"), arg.c_str());
        }
        else if (arg == "--degree")
        {
            options.degree = parse_size(require_value("--degree"), arg.c_str());
        }
        else if (arg == "--components")
        {
            options.component_count = parse_size(require_value("--components"), arg.c_str());
        }
        else if (arg == "--p-count")
        {
            options.p_count = parse_size(require_value("--p-count"), arg.c_str());
        }
        else if (arg == "--allow-same-device")
        {
            options.allow_same_device = true;
        }
        else if (arg == "--help" || arg == "-h")
        {
            print_usage(argv[0]);
            std::exit(EXIT_SUCCESS);
        }
        else
        {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }

    if (options.iterations <= 0)
    {
        throw std::invalid_argument("--iterations must be positive");
    }
    if (options.warmup_iterations < 0)
    {
        throw std::invalid_argument("--warmup must be non-negative");
    }
    if (options.source_device < 0 || options.destination_device < 0)
    {
        throw std::invalid_argument("device ids must be non-negative");
    }
    if (options.source_device == options.destination_device &&
        !options.allow_same_device)
    {
        throw std::invalid_argument(
            "source and destination devices must differ unless --allow-same-device is set");
    }
    if (options.degree == 0 || options.component_count == 0)
    {
        throw std::invalid_argument("degree and component count must be non-zero");
    }
    return options;
}

std::vector<int> unique_devices(std::initializer_list<int> devices)
{
    std::vector<int> result;
    for (const int device : devices)
    {
        if (std::find(result.begin(), result.end(), device) == result.end())
        {
            result.push_back(device);
        }
    }
    return result;
}

std::size_t checked_mul(std::size_t a, std::size_t b, const char *what)
{
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a)
    {
        throw std::overflow_error(what);
    }
    return a * b;
}

std::size_t checked_add(std::size_t a, std::size_t b, const char *what)
{
    if (b > std::numeric_limits<std::size_t>::max() - a)
    {
        throw std::overflow_error(what);
    }
    return a + b;
}

std::size_t copy_request_payload_bytes(const GpuObjectCopyRequest &request)
{
    std::size_t bytes = 0;
    for (const GpuObjectBufferCopy &buffer : request.buffers)
    {
        bytes = checked_add(bytes, buffer.bytes, "copy request byte count overflow");
    }
    return bytes;
}

std::size_t copy_requests_payload_bytes(
    const std::vector<GpuObjectCopyRequest> &requests)
{
    std::size_t bytes = 0;
    for (const GpuObjectCopyRequest &request : requests)
    {
        bytes = checked_add(
            bytes,
            copy_request_payload_bytes(request),
            "copy request batch byte count overflow");
    }
    return bytes;
}

std::size_t first_copy_request_payload_bytes(
    const std::vector<GpuObjectCopyRequest> &requests)
{
    if (requests.empty())
    {
        throw std::invalid_argument("cannot get ciphertext bytes from an empty copy batch");
    }
    return copy_request_payload_bytes(requests.front());
}

std::shared_ptr<gpu::GpuCiphertextData> make_ciphertext(
    const BenchOptions &options, const LevelShape &level, ValueId seed)
{
    auto ciphertext = std::make_shared<gpu::GpuCiphertextData>(
        gpu::GpuCiphertextData::allocate_single_device(
            options.degree,
            level.q_count,
            options.component_count,
            options.source_device,
            options.p_count));
    ciphertext->meta.is_ntt_form = true;
    ciphertext->meta.scale = static_cast<double>(1ULL << 40);

    std::vector<gpu::GpuWord> host(ciphertext->fields_[0].size());
    for (std::size_t index = 0; index < host.size(); ++index)
    {
        host[index] = static_cast<gpu::GpuWord>(
            0x9e3779b9u + static_cast<gpu::GpuWord>(seed * 131) +
            static_cast<gpu::GpuWord>(index));
    }
    ciphertext->fields_[0].buffer.copy_from_host(host.data(), host.size());
    return ciphertext;
}

std::vector<GpuCommCopyRequest> make_requests(
    const BenchOptions &options,
    const LevelShape &level,
    std::size_t batch_size,
    std::vector<std::shared_ptr<gpu::GpuCiphertextData>> &sources)
{
    sources.clear();
    sources.reserve(batch_size);
    std::vector<GpuCommCopyRequest> requests;
    requests.reserve(batch_size);

    for (std::size_t index = 0; index < batch_size; ++index)
    {
        const ValueId source_id = static_cast<ValueId>(1000 + index);
        const ValueId destination_id = static_cast<ValueId>(2000 + index);
        sources.push_back(make_ciphertext(options, level, source_id));
        requests.push_back(GpuCommCopyRequest{
            source_id,
            destination_id,
            MgpuValueKind::Ciphertext,
            options.source_device,
            options.destination_device,
            sources.back(),
        });
    }
    return requests;
}

gpu::GpuWord expected_word(ValueId seed, std::size_t index)
{
    return static_cast<gpu::GpuWord>(
        0x9e3779b9u + static_cast<gpu::GpuWord>(seed * 131) +
        static_cast<gpu::GpuWord>(index));
}

std::vector<std::size_t> sample_indices(std::size_t word_count)
{
    if (word_count == 0)
    {
        throw std::invalid_argument("cannot sample an empty ciphertext");
    }

    std::vector<std::size_t> samples{
        0,
        word_count / 7,
        word_count / 3,
        word_count / 2,
        (word_count * 5) / 7,
        word_count - 1,
    };
    std::sort(samples.begin(), samples.end());
    samples.erase(std::unique(samples.begin(), samples.end()), samples.end());
    return samples;
}

gpu::GpuWord read_device_word(
    const gpu::GpuCiphertextData &ciphertext, std::size_t word_index)
{
    if (ciphertext.fields_.size() != 1)
    {
        throw std::invalid_argument("benchmark ciphertext must have exactly one GPU field");
    }

    gpu::GpuWord value = 0;
    const gpu::GpuFieldData &field = ciphertext.fields_[0];
    check_cuda(cudaSetDevice(field.device_id), "cudaSetDevice read ciphertext");
    check_cuda(
        cudaMemcpy(
            &value,
            field.data() + word_index,
            sizeof(gpu::GpuWord),
            cudaMemcpyDeviceToHost),
        "cudaMemcpy read ciphertext");
    return value;
}

void validate_destinations(const MaterializedGpuObjectBatchCopy &materialized)
{
    for (std::size_t object_index = 0;
         object_index < materialized.destination_objects.size();
         ++object_index)
    {
        const auto destination =
            std::static_pointer_cast<gpu::GpuCiphertextData>(
                materialized.destination_objects[object_index]);
        if (destination->fields_.empty())
        {
            throw std::invalid_argument("destination ciphertext has no GPU field");
        }

        const std::vector<std::size_t> samples =
            sample_indices(destination->fields_[0].size());
        const ValueId source_id = static_cast<ValueId>(1000 + object_index);
        for (const std::size_t word_index : samples)
        {
            const gpu::GpuWord actual = read_device_word(*destination, word_index);
            const gpu::GpuWord expected = expected_word(source_id, word_index);
            if (actual != expected)
            {
                std::ostringstream stream;
                stream << "copy validation failed for ciphertext " << object_index
                       << " at word " << word_index
                       << ": expected 0x" << std::hex << expected
                       << " got 0x" << actual;
                throw std::runtime_error(stream.str());
            }
        }
    }
}

void synchronize_devices(const BenchOptions &options)
{
    check_cuda(cudaSetDevice(options.source_device), "cudaSetDevice source synchronize");
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize source");
    if (options.destination_device != options.source_device)
    {
        check_cuda(cudaSetDevice(options.destination_device), "cudaSetDevice destination synchronize");
        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize destination");
    }
}

template <typename Operation>
double measure_once(const BenchOptions &options, Operation &&operation)
{
    synchronize_devices(options);
    const auto start = std::chrono::steady_clock::now();
    operation();
    synchronize_devices(options);
    const auto stop = std::chrono::steady_clock::now();
    const auto elapsed =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
            stop - start);
    return elapsed.count();
}

Stats summarize(const std::vector<double> &samples)
{
    if (samples.empty())
    {
        throw std::invalid_argument("cannot summarize empty sample set");
    }

    Stats stats;
    stats.average_ms =
        std::accumulate(samples.begin(), samples.end(), 0.0) /
        static_cast<double>(samples.size());
    stats.min_ms = *std::min_element(samples.begin(), samples.end());
    stats.max_ms = *std::max_element(samples.begin(), samples.end());
    return stats;
}

BenchCaseResult run_case(
    const BenchOptions &options,
    const LevelShape &level,
    std::size_t batch_size,
    bool batch_transfer,
    PoseidonGpuObjectCopyMaterializer &materializer,
    CudaPeerComm &peer_backend)
{
    std::vector<std::shared_ptr<gpu::GpuCiphertextData>> sources;
    const std::vector<GpuCommCopyRequest> requests =
        make_requests(options, level, batch_size, sources);
    MaterializedGpuObjectBatchCopy materialized =
        materializer.materialize_copy_batch(requests);
    const std::size_t ciphertext_bytes =
        first_copy_request_payload_bytes(materialized.object_copies);
    const std::size_t total_bytes =
        copy_requests_payload_bytes(materialized.object_copies);

    auto operation = [&]() {
        if (batch_transfer)
        {
            peer_backend.copy_objects(materialized.object_copies);
        }
        else
        {
            for (const GpuObjectCopyRequest &request : materialized.object_copies)
            {
                peer_backend.copy_object(request);
            }
        }
    };

    for (int iter = 0; iter < options.warmup_iterations; ++iter)
    {
        operation();
        synchronize_devices(options);
    }

    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(options.iterations));
    for (int iter = 0; iter < options.iterations; ++iter)
    {
        samples.push_back(measure_once(options, operation));
    }
    validate_destinations(materialized);
    return BenchCaseResult{ summarize(samples), ciphertext_bytes, total_bytes };
}

void print_header(const BenchOptions &options)
{
    int device_count = 0;
    check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count <= std::max(options.source_device, options.destination_device))
    {
        throw std::runtime_error("requested CUDA devices are not visible");
    }

    int can_access = 0;
    if (options.source_device == options.destination_device)
    {
        can_access = 1;
    }
    else
    {
        check_cuda(
            cudaDeviceCanAccessPeer(
                &can_access,
                options.destination_device,
                options.source_device),
            "cudaDeviceCanAccessPeer");
    }

    std::cout << "Poseidon mgpu CKKS ciphertext transfer bench\n";
    std::cout << "source_device=" << options.source_device
              << " destination_device=" << options.destination_device
              << " peer_access=" << (can_access ? "yes" : "no_host_staging")
              << " iterations=" << options.iterations
              << " warmup=" << options.warmup_iterations
              << " degree=" << options.degree
              << " components=" << options.component_count
              << " p_count=" << options.p_count << "\n\n";
    std::cout
        << std::left
        << std::setw(12) << "mode"
        << std::setw(8) << "count"
        << std::setw(10) << "level"
        << std::setw(10) << "q_count"
        << std::setw(14) << "ct_bytes"
        << std::setw(14) << "total_bytes"
        << std::setw(14) << "avg_ms"
        << std::setw(14) << "min_ms"
        << std::setw(14) << "max_ms"
        << std::setw(14) << "GBps"
        << "\n";
}

void print_result(
    const char *mode,
    std::size_t batch_size,
    const LevelShape &level,
    std::size_t ciphertext_bytes,
    std::size_t total_bytes,
    const Stats &stats)
{
    const double gbps =
        (static_cast<double>(total_bytes) / 1.0e9) / (stats.average_ms / 1000.0);
    std::cout
        << std::left
        << std::setw(12) << mode
        << std::setw(8) << batch_size
        << std::setw(10) << level.name
        << std::setw(10) << level.q_count
        << std::setw(14) << ciphertext_bytes
        << std::setw(14) << total_bytes
        << std::setw(14) << std::fixed << std::setprecision(4) << stats.average_ms
        << std::setw(14) << std::fixed << std::setprecision(4) << stats.min_ms
        << std::setw(14) << std::fixed << std::setprecision(4) << stats.max_ms
        << std::setw(14) << std::fixed << std::setprecision(3) << gbps
        << "\n";
}

}  // namespace

int main(int argc, char **argv)
{
    try
    {
        const BenchOptions options = parse_options(argc, argv);
        print_header(options);

        RmmPoolScope rmm_scope(
            unique_devices({ options.source_device, options.destination_device }));
        PoseidonGpuObjectCopyMaterializer materializer;
        CudaPeerComm peer_backend;

        const std::vector<std::size_t> batch_sizes{ 1, 5, 10 };
        const std::vector<LevelShape> levels{
            { "L4", 4 },
            { "L8", 8 },
            { "L12", 12 },
            { "L16", 16 },
            { "L20", 20 },
        };

        for (const std::size_t batch_size : batch_sizes)
        {
            for (const LevelShape &level : levels)
            {
                const BenchCaseResult single_result = run_case(
                    options,
                    level,
                    batch_size,
                    false,
                    materializer,
                    peer_backend);
                print_result(
                    "single",
                    batch_size,
                    level,
                    single_result.ciphertext_bytes,
                    single_result.total_bytes,
                    single_result.stats);

                const BenchCaseResult batch_result = run_case(
                    options,
                    level,
                    batch_size,
                    true,
                    materializer,
                    peer_backend);
                print_result(
                    "batch",
                    batch_size,
                    level,
                    batch_result.ciphertext_bytes,
                    batch_result.total_bytes,
                    batch_result.stats);
            }
        }
    }
    catch (const std::exception &ex)
    {
        std::cerr << "poseidon_mgpu_ckks_transfer_bench: " << ex.what() << '\n';
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
