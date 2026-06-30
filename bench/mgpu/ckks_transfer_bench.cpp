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
#include <ctime>
#include <fstream>
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

#include <unistd.h>

using namespace poseidon;
using namespace poseidon::mgpu;

namespace
{

enum class TransferMode
{
    ObjectLoop,
    ObjectLoopE2E,
    CopyObjects,
    AsyncObjectLoop,
    ContiguousBuffer,
};

enum class LogFormat
{
    Csv,
    Jsonl,
};

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
    std::vector<TransferMode> modes{
        TransferMode::ObjectLoop,
        TransferMode::CopyObjects,
    };
    std::vector<std::size_t> counts{ 1, 5, 10 };
    std::vector<std::size_t> levels{ 4, 8, 12, 16, 20 };
    std::optional<std::size_t> min_count;
    std::optional<std::size_t> max_count;
    std::optional<std::size_t> min_level;
    std::optional<std::size_t> max_level;
    bool explicit_counts = false;
    bool explicit_levels = false;
    std::string log_path;
    bool append_log = false;
    LogFormat log_format = LogFormat::Csv;
};

struct LevelShape
{
    std::string name;
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

struct RuntimeMetadata
{
    std::string hostname;
    std::string visible_devices;
    std::string peer_access;
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

class CudaStreamScope
{
public:
    explicit CudaStreamScope(int device_id) : device_id_(device_id)
    {
        check_cuda(cudaSetDevice(device_id_), "cudaSetDevice create stream");
        check_cuda(
            cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
            "cudaStreamCreateWithFlags");
    }

    CudaStreamScope(const CudaStreamScope &) = delete;
    CudaStreamScope &operator=(const CudaStreamScope &) = delete;

    ~CudaStreamScope()
    {
        if (stream_ != nullptr)
        {
            (void)cudaSetDevice(device_id_);
            (void)cudaStreamDestroy(stream_);
        }
    }

    cudaStream_t get() const noexcept
    {
        return stream_;
    }

private:
    int device_id_ = 0;
    cudaStream_t stream_ = nullptr;
};

class DeviceBuffer
{
public:
    DeviceBuffer(int device_id, std::size_t bytes)
        : device_id_(device_id), bytes_(bytes)
    {
        if (bytes_ == 0)
        {
            throw std::invalid_argument("device buffer byte count must be non-zero");
        }
        check_cuda(cudaSetDevice(device_id_), "cudaSetDevice allocate buffer");
        check_cuda(cudaMalloc(&ptr_, bytes_), "cudaMalloc benchmark buffer");
    }

    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;

    ~DeviceBuffer()
    {
        if (ptr_ != nullptr)
        {
            (void)cudaSetDevice(device_id_);
            (void)cudaFree(ptr_);
        }
    }

    void *data() const noexcept
    {
        return ptr_;
    }

    std::size_t bytes() const noexcept
    {
        return bytes_;
    }

private:
    int device_id_ = 0;
    std::size_t bytes_ = 0;
    void *ptr_ = nullptr;
};

void print_usage(const char *program)
{
    std::cerr
        << "usage: " << program
        << " [--source-device N] [--destination-device N]"
           " [--iterations N] [--warmup N] [--degree N]"
           " [--components N] [--p-count N] [--modes LIST]"
           " [--counts LIST | --min-count N --max-count N]"
           " [--levels LIST | --min-level N --max-level N]"
           " [--log PATH] [--append-log] [--log-format csv|jsonl]"
           " [--allow-same-device]\n"
           "modes: object_loop,object_loop_e2e,copy_objects,"
           "async_object_loop,contiguous_buffer\n";
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

std::string trim(std::string value)
{
    const auto first = value.find_first_not_of(" \t\n\r");
    if (first == std::string::npos)
    {
        return "";
    }
    const auto last = value.find_last_not_of(" \t\n\r");
    return value.substr(first, last - first + 1);
}

std::vector<std::string> split_csv(const std::string &value)
{
    std::vector<std::string> result;
    std::size_t start = 0;
    while (start <= value.size())
    {
        const std::size_t comma = value.find(',', start);
        const std::size_t end = comma == std::string::npos ? value.size() : comma;
        result.push_back(trim(value.substr(start, end - start)));
        if (comma == std::string::npos)
        {
            break;
        }
        start = comma + 1;
    }
    return result;
}

std::vector<std::size_t> parse_size_list(const char *value, const char *name)
{
    std::vector<std::size_t> result;
    for (const std::string &token : split_csv(value))
    {
        if (token.empty())
        {
            throw std::invalid_argument(std::string(name) + " contains an empty entry");
        }
        const std::size_t parsed = parse_size(token.c_str(), name);
        if (parsed == 0)
        {
            throw std::invalid_argument(std::string(name) + " entries must be non-zero");
        }
        result.push_back(parsed);
    }
    if (result.empty())
    {
        throw std::invalid_argument(std::string(name) + " must not be empty");
    }
    return result;
}

std::vector<std::size_t> make_range(
    std::size_t min_value, std::size_t max_value, const char *name)
{
    if (min_value == 0 || max_value == 0)
    {
        throw std::invalid_argument(std::string(name) + " range endpoints must be non-zero");
    }
    if (min_value > max_value)
    {
        throw std::invalid_argument(std::string(name) + " min must be <= max");
    }

    std::vector<std::size_t> values;
    values.reserve(max_value - min_value + 1);
    for (std::size_t value = min_value; value <= max_value; ++value)
    {
        values.push_back(value);
    }
    return values;
}

LogFormat parse_log_format(const char *value)
{
    const std::string token = value;
    if (token == "csv")
    {
        return LogFormat::Csv;
    }
    if (token == "jsonl")
    {
        return LogFormat::Jsonl;
    }
    throw std::invalid_argument("unknown log format: " + token);
}

TransferMode parse_mode_token(const std::string &token)
{
    if (token == "object_loop" || token == "single")
    {
        return TransferMode::ObjectLoop;
    }
    if (token == "object_loop_e2e" || token == "e2e" ||
        token == "materialized_object_loop")
    {
        return TransferMode::ObjectLoopE2E;
    }
    if (token == "copy_objects" || token == "batch")
    {
        return TransferMode::CopyObjects;
    }
    if (token == "async_object_loop" || token == "async_loop" || token == "async")
    {
        return TransferMode::AsyncObjectLoop;
    }
    if (token == "contiguous_buffer" || token == "raw_contiguous" || token == "contiguous")
    {
        return TransferMode::ContiguousBuffer;
    }
    throw std::invalid_argument("unknown transfer mode: " + token);
}

std::vector<TransferMode> parse_modes(const char *value)
{
    std::vector<TransferMode> modes;
    std::stringstream stream(value);
    std::string token;
    while (std::getline(stream, token, ','))
    {
        token = trim(token);
        if (token.empty())
        {
            throw std::invalid_argument("empty transfer mode in --modes");
        }
        modes.push_back(parse_mode_token(token));
    }
    if (modes.empty())
    {
        throw std::invalid_argument("--modes must not be empty");
    }
    return modes;
}

const char *mode_name(TransferMode mode)
{
    switch (mode)
    {
    case TransferMode::ObjectLoop:
        return "object_loop";
    case TransferMode::ObjectLoopE2E:
        return "object_loop_e2e";
    case TransferMode::CopyObjects:
        return "copy_objects";
    case TransferMode::AsyncObjectLoop:
        return "async_object_loop";
    case TransferMode::ContiguousBuffer:
        return "contiguous_buffer";
    }
    return "unknown";
}

std::string format_modes(const std::vector<TransferMode> &modes)
{
    std::ostringstream stream;
    for (std::size_t index = 0; index < modes.size(); ++index)
    {
        if (index != 0)
        {
            stream << ',';
        }
        stream << mode_name(modes[index]);
    }
    return stream.str();
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
        else if (arg == "--modes")
        {
            options.modes = parse_modes(require_value("--modes"));
        }
        else if (arg == "--counts")
        {
            options.counts = parse_size_list(require_value("--counts"), "--counts");
            options.explicit_counts = true;
        }
        else if (arg == "--levels")
        {
            options.levels = parse_size_list(require_value("--levels"), "--levels");
            options.explicit_levels = true;
        }
        else if (arg == "--min-count")
        {
            options.min_count = parse_size(require_value("--min-count"), arg.c_str());
        }
        else if (arg == "--max-count")
        {
            options.max_count = parse_size(require_value("--max-count"), arg.c_str());
        }
        else if (arg == "--min-level")
        {
            options.min_level = parse_size(require_value("--min-level"), arg.c_str());
        }
        else if (arg == "--max-level")
        {
            options.max_level = parse_size(require_value("--max-level"), arg.c_str());
        }
        else if (arg == "--log")
        {
            options.log_path = require_value("--log");
        }
        else if (arg == "--append-log")
        {
            options.append_log = true;
        }
        else if (arg == "--log-format")
        {
            options.log_format = parse_log_format(require_value("--log-format"));
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
    if (options.modes.empty())
    {
        throw std::invalid_argument("--modes must not be empty");
    }
    if (options.explicit_counts && (options.min_count || options.max_count))
    {
        throw std::invalid_argument("--counts cannot be combined with --min-count/--max-count");
    }
    if (options.explicit_levels && (options.min_level || options.max_level))
    {
        throw std::invalid_argument("--levels cannot be combined with --min-level/--max-level");
    }
    if (static_cast<bool>(options.min_count) != static_cast<bool>(options.max_count))
    {
        throw std::invalid_argument("--min-count and --max-count must be provided together");
    }
    if (static_cast<bool>(options.min_level) != static_cast<bool>(options.max_level))
    {
        throw std::invalid_argument("--min-level and --max-level must be provided together");
    }
    if (options.min_count && options.max_count)
    {
        options.counts = make_range(*options.min_count, *options.max_count, "count");
    }
    if (options.min_level && options.max_level)
    {
        options.levels = make_range(*options.min_level, *options.max_level, "level");
    }
    if (options.counts.empty() || options.levels.empty())
    {
        throw std::invalid_argument("benchmark count and level grids must not be empty");
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

std::string join_sizes(const std::vector<std::size_t> &values)
{
    std::ostringstream stream;
    for (std::size_t index = 0; index < values.size(); ++index)
    {
        if (index != 0)
        {
            stream << ',';
        }
        stream << values[index];
    }
    return stream.str();
}

std::vector<LevelShape> make_levels(const std::vector<std::size_t> &q_counts)
{
    std::vector<LevelShape> result;
    result.reserve(q_counts.size());
    for (const std::size_t q_count : q_counts)
    {
        result.push_back(LevelShape{ "L" + std::to_string(q_count), q_count });
    }
    return result;
}

std::string current_timestamp()
{
    const std::time_t now = std::time(nullptr);
    std::tm local_time{};
    localtime_r(&now, &local_time);

    char buffer[32] = {};
    if (std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%S%z", &local_time) == 0)
    {
        return "";
    }
    return buffer;
}

std::string hostname()
{
    char buffer[256] = {};
    if (gethostname(buffer, sizeof(buffer) - 1) != 0)
    {
        return "unknown";
    }
    return buffer;
}

std::string visible_device_list()
{
    int device_count = 0;
    check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");

    std::ostringstream stream;
    for (int device = 0; device < device_count; ++device)
    {
        if (device != 0)
        {
            stream << ',';
        }
        stream << device;
    }
    return stream.str();
}

std::string peer_access_label(int source_device, int destination_device)
{
    if (source_device == destination_device)
    {
        return "same_device";
    }

    int can_access = 0;
    check_cuda(
        cudaDeviceCanAccessPeer(&can_access, destination_device, source_device),
        "cudaDeviceCanAccessPeer");
    return can_access ? "yes" : "no_host_staging";
}

std::size_t checked_add(std::size_t a, std::size_t b, const char *what)
{
    if (b > std::numeric_limits<std::size_t>::max() - a)
    {
        throw std::overflow_error(what);
    }
    return a + b;
}

std::size_t checked_mul(std::size_t a, std::size_t b, const char *what)
{
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a)
    {
        throw std::overflow_error(what);
    }
    return a * b;
}

std::size_t ciphertext_payload_bytes(
    const BenchOptions &options, const LevelShape &level)
{
    const std::size_t limb_count =
        checked_add(level.q_count, options.p_count, "ciphertext limb count overflow");
    const std::size_t words_per_component =
        checked_mul(options.degree, limb_count, "ciphertext word count overflow");
    const std::size_t words =
        checked_mul(
            words_per_component,
            options.component_count,
            "ciphertext component word count overflow");
    return checked_mul(words, sizeof(gpu::GpuWord), "ciphertext byte count overflow");
}

std::size_t total_payload_bytes(
    const BenchOptions &options, const LevelShape &level, std::size_t batch_size)
{
    return checked_mul(
        ciphertext_payload_bytes(options, level),
        batch_size,
        "benchmark total byte count overflow");
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
    const BenchOptions &options, const LevelShape &level, ValueId seed,
    bool fill_payload = true)
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

    if (!fill_payload)
    {
        return ciphertext;
    }

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
    std::vector<std::shared_ptr<gpu::GpuCiphertextData>> &sources,
    bool fill_payload = true)
{
    sources.clear();
    sources.reserve(batch_size);
    std::vector<GpuCommCopyRequest> requests;
    requests.reserve(batch_size);

    for (std::size_t index = 0; index < batch_size; ++index)
    {
        const ValueId source_id = static_cast<ValueId>(1000 + index);
        const ValueId destination_id = static_cast<ValueId>(2000 + index);
        sources.push_back(make_ciphertext(options, level, source_id, fill_payload));
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

gpu::GpuWord expected_raw_word(std::size_t index)
{
    return static_cast<gpu::GpuWord>(
        0x6d2b79f5u + static_cast<gpu::GpuWord>(index * 97));
}

std::vector<gpu::GpuWord> make_raw_payload(std::size_t bytes)
{
    if (bytes % sizeof(gpu::GpuWord) != 0)
    {
        throw std::invalid_argument("raw benchmark byte count must be GpuWord-aligned");
    }

    std::vector<gpu::GpuWord> host(bytes / sizeof(gpu::GpuWord));
    for (std::size_t index = 0; index < host.size(); ++index)
    {
        host[index] = expected_raw_word(index);
    }
    return host;
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

gpu::GpuWord read_raw_device_word(
    int device_id, const void *buffer, std::size_t word_index)
{
    gpu::GpuWord value = 0;
    const auto *words = static_cast<const gpu::GpuWord *>(buffer);
    check_cuda(cudaSetDevice(device_id), "cudaSetDevice read raw buffer");
    check_cuda(
        cudaMemcpy(
            &value,
            words + word_index,
            sizeof(gpu::GpuWord),
            cudaMemcpyDeviceToHost),
        "cudaMemcpy read raw buffer");
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

void validate_raw_destination(
    int destination_device, const DeviceBuffer &destination)
{
    const std::size_t word_count = destination.bytes() / sizeof(gpu::GpuWord);
    const std::vector<std::size_t> samples = sample_indices(word_count);
    for (const std::size_t word_index : samples)
    {
        const gpu::GpuWord actual =
            read_raw_device_word(destination_device, destination.data(), word_index);
        const gpu::GpuWord expected = expected_raw_word(word_index);
        if (actual != expected)
        {
            std::ostringstream stream;
            stream << "raw buffer validation failed at word " << word_index
                   << ": expected 0x" << std::hex << expected
                   << " got 0x" << actual;
            throw std::runtime_error(stream.str());
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

template <typename Operation>
double measure_once_with_operation_sync(
    const BenchOptions &options, Operation &&operation)
{
    synchronize_devices(options);
    const auto start = std::chrono::steady_clock::now();
    operation();
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

BenchCaseResult run_object_copy_case(
    const BenchOptions &options,
    const LevelShape &level,
    std::size_t batch_size,
    TransferMode mode,
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

    BenchCaseResult result;
    result.ciphertext_bytes = ciphertext_bytes;
    result.total_bytes = total_bytes;

    auto operation = [&]() {
        switch (mode)
        {
        case TransferMode::ObjectLoop:
            for (const GpuObjectCopyRequest &request : materialized.object_copies)
            {
                peer_backend.copy_object(request);
            }
            return;
        case TransferMode::CopyObjects:
            peer_backend.copy_objects(materialized.object_copies);
            return;
        case TransferMode::ObjectLoopE2E:
        case TransferMode::AsyncObjectLoop:
        case TransferMode::ContiguousBuffer:
            throw std::invalid_argument(
                std::string("invalid object copy mode: ") + mode_name(mode));
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
    result.stats = summarize(samples);
    return result;
}

BenchCaseResult run_object_loop_e2e_case(
    const BenchOptions &options,
    const LevelShape &level,
    std::size_t batch_size,
    PoseidonGpuObjectCopyMaterializer &materializer,
    CudaPeerComm &peer_backend)
{
    std::vector<std::shared_ptr<gpu::GpuCiphertextData>> sources;
    const std::vector<GpuCommCopyRequest> requests =
        make_requests(options, level, batch_size, sources);

    BenchCaseResult result;
    result.ciphertext_bytes = ciphertext_payload_bytes(options, level);
    result.total_bytes = total_payload_bytes(options, level, batch_size);

    MaterializedGpuObjectBatchCopy last_materialized;
    auto operation = [&]() {
        MaterializedGpuObjectBatchCopy materialized =
            materializer.materialize_copy_batch(requests);
        if (materialized.object_copies.size() != requests.size() ||
            materialized.destination_objects.size() != requests.size())
        {
            throw std::logic_error("materialized copy batch size mismatch");
        }

        for (const GpuObjectCopyRequest &request : materialized.object_copies)
        {
            peer_backend.copy_object(request);
        }
        last_materialized = std::move(materialized);
    };

    for (int iter = 0; iter < options.warmup_iterations; ++iter)
    {
        last_materialized = {};
        operation();
        synchronize_devices(options);
    }

    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(options.iterations));
    for (int iter = 0; iter < options.iterations; ++iter)
    {
        last_materialized = {};
        samples.push_back(measure_once(options, operation));
    }
    validate_destinations(last_materialized);
    result.stats = summarize(samples);
    return result;
}

BenchCaseResult run_async_object_loop_case(
    const BenchOptions &options,
    const LevelShape &level,
    std::size_t batch_size,
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

    BenchCaseResult result;
    result.ciphertext_bytes = ciphertext_bytes;
    result.total_bytes = total_bytes;

    CudaStreamScope copy_stream(options.destination_device);
    auto operation = [&]() {
        for (const GpuObjectCopyRequest &request : materialized.object_copies)
        {
            peer_backend.copy_object_peer_async(request, copy_stream.get());
        }
        check_cuda(
            cudaStreamSynchronize(copy_stream.get()),
            "cudaStreamSynchronize async object loop");
    };

    for (int iter = 0; iter < options.warmup_iterations; ++iter)
    {
        operation();
    }

    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(options.iterations));
    for (int iter = 0; iter < options.iterations; ++iter)
    {
        samples.push_back(measure_once_with_operation_sync(options, operation));
    }
    validate_destinations(materialized);
    result.stats = summarize(samples);
    return result;
}

BenchCaseResult run_contiguous_buffer_case(
    const BenchOptions &options,
    const LevelShape &level,
    std::size_t batch_size,
    CudaPeerComm &peer_backend)
{
    BenchCaseResult result;
    result.ciphertext_bytes = ciphertext_payload_bytes(options, level);
    result.total_bytes = total_payload_bytes(options, level, batch_size);

    const std::vector<gpu::GpuWord> host_payload =
        make_raw_payload(result.total_bytes);
    DeviceBuffer source(options.source_device, result.total_bytes);
    DeviceBuffer destination(options.destination_device, result.total_bytes);

    check_cuda(cudaSetDevice(options.source_device), "cudaSetDevice raw source");
    check_cuda(
        cudaMemcpy(
            source.data(),
            host_payload.data(),
            result.total_bytes,
            cudaMemcpyHostToDevice),
        "cudaMemcpy raw payload to source");

    check_cuda(cudaSetDevice(options.destination_device), "cudaSetDevice raw destination");
    check_cuda(
        cudaMemset(destination.data(), 0, destination.bytes()),
        "cudaMemset raw destination");

    const CudaPeerCopyRequest copy_request{
        source.data(),
        destination.data(),
        result.total_bytes,
        options.source_device,
        options.destination_device,
    };
    auto operation = [&]() {
        peer_backend.copy_buffer(copy_request);
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
    validate_raw_destination(options.destination_device, destination);
    result.stats = summarize(samples);
    return result;
}

BenchCaseResult run_case(
    const BenchOptions &options,
    const LevelShape &level,
    std::size_t batch_size,
    TransferMode mode,
    PoseidonGpuObjectCopyMaterializer &materializer,
    CudaPeerComm &peer_backend)
{
    switch (mode)
    {
    case TransferMode::ObjectLoop:
    case TransferMode::CopyObjects:
        return run_object_copy_case(
            options,
            level,
            batch_size,
            mode,
            materializer,
            peer_backend);
    case TransferMode::ObjectLoopE2E:
        return run_object_loop_e2e_case(
            options,
            level,
            batch_size,
            materializer,
            peer_backend);
    case TransferMode::AsyncObjectLoop:
        return run_async_object_loop_case(
            options,
            level,
            batch_size,
            materializer,
            peer_backend);
    case TransferMode::ContiguousBuffer:
        return run_contiguous_buffer_case(
            options,
            level,
            batch_size,
            peer_backend);
    }

    throw std::invalid_argument(
        std::string("unknown transfer mode: ") + mode_name(mode));
}

bool has_object_mode(const BenchOptions &options)
{
    return std::any_of(
        options.modes.begin(),
        options.modes.end(),
        [](TransferMode mode) {
            return mode == TransferMode::ObjectLoop ||
                   mode == TransferMode::ObjectLoopE2E ||
                   mode == TransferMode::CopyObjects ||
                   mode == TransferMode::AsyncObjectLoop;
        });
}

bool has_contiguous_mode(const BenchOptions &options)
{
    return std::find(
        options.modes.begin(),
        options.modes.end(),
        TransferMode::ContiguousBuffer) != options.modes.end();
}

std::size_t max_value(const std::vector<std::size_t> &values, const char *name)
{
    if (values.empty())
    {
        throw std::invalid_argument(std::string(name) + " must not be empty");
    }
    return *std::max_element(values.begin(), values.end());
}

void preflight_largest_case(
    const BenchOptions &options,
    PoseidonGpuObjectCopyMaterializer &materializer)
{
    const LevelShape max_level{
        "L" + std::to_string(max_value(options.levels, "levels")),
        max_value(options.levels, "levels"),
    };
    const std::size_t max_count = max_value(options.counts, "counts");

    try
    {
        if (has_object_mode(options))
        {
            std::vector<std::shared_ptr<gpu::GpuCiphertextData>> sources;
            const std::vector<GpuCommCopyRequest> requests =
                make_requests(options, max_level, max_count, sources, false);
            MaterializedGpuObjectBatchCopy materialized =
                materializer.materialize_copy_batch(requests);
            if (materialized.object_copies.size() != max_count ||
                materialized.destination_objects.size() != max_count)
            {
                throw std::logic_error("preflight materialized copy batch size mismatch");
            }
        }

        if (has_contiguous_mode(options))
        {
            const std::size_t total_bytes =
                total_payload_bytes(options, max_level, max_count);
            DeviceBuffer source(options.source_device, total_bytes);
            DeviceBuffer destination(options.destination_device, total_bytes);
            (void)source;
            (void)destination;
        }
    }
    catch (const std::exception &ex)
    {
        std::ostringstream stream;
        stream << "largest-case preflight failed for degree=" << options.degree
               << " level=" << max_level.name
               << " count=" << max_count
               << " components=" << options.component_count
               << " p_count=" << options.p_count
               << ": " << ex.what()
               << ". Reduce --degree, --max-level, --max-count, or --p-count.";
        throw std::runtime_error(stream.str());
    }
}

double gbps_for_bytes(std::size_t bytes, const Stats &stats)
{
    return (static_cast<double>(bytes) / 1.0e9) / (stats.average_ms / 1000.0);
}

bool file_has_content(const std::string &path)
{
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    return input.good() && input.tellg() > 0;
}

std::string csv_escape(const std::string &value)
{
    bool needs_quotes = false;
    for (const char ch : value)
    {
        if (ch == '"' || ch == ',' || ch == '\n' || ch == '\r')
        {
            needs_quotes = true;
            break;
        }
    }
    if (!needs_quotes)
    {
        return value;
    }

    std::string escaped = "\"";
    for (const char ch : value)
    {
        if (ch == '"')
        {
            escaped += "\"\"";
        }
        else
        {
            escaped += ch;
        }
    }
    escaped += '"';
    return escaped;
}

std::string json_escape(const std::string &value)
{
    std::ostringstream stream;
    for (const char ch : value)
    {
        switch (ch)
        {
        case '"':
            stream << "\\\"";
            break;
        case '\\':
            stream << "\\\\";
            break;
        case '\n':
            stream << "\\n";
            break;
        case '\r':
            stream << "\\r";
            break;
        case '\t':
            stream << "\\t";
            break;
        default:
            stream << ch;
            break;
        }
    }
    return stream.str();
}

class ResultLogger
{
public:
    ResultLogger(const BenchOptions &options, const RuntimeMetadata &metadata)
        : options_(options), metadata_(metadata)
    {
        if (options_.log_path.empty())
        {
            return;
        }

        const bool write_header =
            options_.log_format == LogFormat::Csv &&
            (!options_.append_log || !file_has_content(options_.log_path));
        const auto mode =
            std::ios::out | (options_.append_log ? std::ios::app : std::ios::trunc);
        output_.open(options_.log_path, mode);
        if (!output_)
        {
            throw std::runtime_error("failed to open log file: " + options_.log_path);
        }

        if (write_header)
        {
            output_
                << "timestamp,hostname,visible_devices,source_device,"
                   "destination_device,peer_access,degree,level,q_count,count,"
                   "components,p_count,mode,ct_bytes,total_bytes,xfer_bytes,"
                   "warmup,iterations,avg_ms,min_ms,max_ms,gbps,synthetic_shape\n";
        }
    }

    void write(
        const char *mode,
        std::size_t count,
        const LevelShape &level,
        const BenchCaseResult &result)
    {
        if (!output_)
        {
            return;
        }

        if (options_.log_format == LogFormat::Csv)
        {
            write_csv(mode, count, level, result);
        }
        else
        {
            write_jsonl(mode, count, level, result);
        }
        output_.flush();
    }

private:
    void write_csv(
        const char *mode,
        std::size_t count,
        const LevelShape &level,
        const BenchCaseResult &result)
    {
        output_
            << csv_escape(current_timestamp()) << ','
            << csv_escape(metadata_.hostname) << ','
            << csv_escape(metadata_.visible_devices) << ','
            << options_.source_device << ','
            << options_.destination_device << ','
            << csv_escape(metadata_.peer_access) << ','
            << options_.degree << ','
            << csv_escape(level.name) << ','
            << level.q_count << ','
            << count << ','
            << options_.component_count << ','
            << options_.p_count << ','
            << csv_escape(mode) << ','
            << result.ciphertext_bytes << ','
            << result.total_bytes << ','
            << result.total_bytes << ','
            << options_.warmup_iterations << ','
            << options_.iterations << ','
            << std::fixed << std::setprecision(6) << result.stats.average_ms << ','
            << std::fixed << std::setprecision(6) << result.stats.min_ms << ','
            << std::fixed << std::setprecision(6) << result.stats.max_ms << ','
            << std::fixed << std::setprecision(6)
            << gbps_for_bytes(result.total_bytes, result.stats) << ','
            << "true\n";
    }

    void write_jsonl(
        const char *mode,
        std::size_t count,
        const LevelShape &level,
        const BenchCaseResult &result)
    {
        output_
            << "{\"timestamp\":\"" << json_escape(current_timestamp()) << "\""
            << ",\"hostname\":\"" << json_escape(metadata_.hostname) << "\""
            << ",\"visible_devices\":\"" << json_escape(metadata_.visible_devices) << "\""
            << ",\"source_device\":" << options_.source_device
            << ",\"destination_device\":" << options_.destination_device
            << ",\"peer_access\":\"" << json_escape(metadata_.peer_access) << "\""
            << ",\"degree\":" << options_.degree
            << ",\"level\":\"" << json_escape(level.name) << "\""
            << ",\"q_count\":" << level.q_count
            << ",\"count\":" << count
            << ",\"components\":" << options_.component_count
            << ",\"p_count\":" << options_.p_count
            << ",\"mode\":\"" << json_escape(mode) << "\""
            << ",\"ct_bytes\":" << result.ciphertext_bytes
            << ",\"total_bytes\":" << result.total_bytes
            << ",\"xfer_bytes\":" << result.total_bytes
            << ",\"warmup\":" << options_.warmup_iterations
            << ",\"iterations\":" << options_.iterations
            << ",\"avg_ms\":" << std::fixed << std::setprecision(6)
            << result.stats.average_ms
            << ",\"min_ms\":" << std::fixed << std::setprecision(6)
            << result.stats.min_ms
            << ",\"max_ms\":" << std::fixed << std::setprecision(6)
            << result.stats.max_ms
            << ",\"gbps\":" << std::fixed << std::setprecision(6)
            << gbps_for_bytes(result.total_bytes, result.stats)
            << ",\"synthetic_shape\":true}\n";
    }

    const BenchOptions &options_;
    const RuntimeMetadata &metadata_;
    std::ofstream output_;
};

void print_header(const BenchOptions &options, const RuntimeMetadata &metadata)
{
    int device_count = 0;
    check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count <= std::max(options.source_device, options.destination_device))
    {
        throw std::runtime_error("requested CUDA devices are not visible");
    }

    std::cout << "Poseidon mgpu CKKS ciphertext transfer bench\n";
    std::cout << "source_device=" << options.source_device
              << " destination_device=" << options.destination_device
              << " peer_access=" << metadata.peer_access
              << " iterations=" << options.iterations
              << " warmup=" << options.warmup_iterations
              << " degree=" << options.degree
              << " components=" << options.component_count
              << " p_count=" << options.p_count
              << " levels=" << join_sizes(options.levels)
              << " counts=" << join_sizes(options.counts)
              << " modes=" << format_modes(options.modes) << "\n\n";
    if (!options.log_path.empty())
    {
        std::cout << "log=" << options.log_path
                  << " format="
                  << (options.log_format == LogFormat::Csv ? "csv" : "jsonl")
                  << (options.append_log ? " append" : "")
                  << "\n\n";
    }
    std::cout
        << std::left
        << std::setw(22) << "mode"
        << std::setw(8) << "count"
        << std::setw(10) << "level"
        << std::setw(10) << "q_count"
        << std::setw(14) << "ct_bytes"
        << std::setw(14) << "total_bytes"
        << std::setw(14) << "xfer_bytes"
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
    const double gbps = gbps_for_bytes(total_bytes, stats);
    std::cout
        << std::left
        << std::setw(22) << mode
        << std::setw(8) << batch_size
        << std::setw(10) << level.name
        << std::setw(10) << level.q_count
        << std::setw(14) << ciphertext_bytes
        << std::setw(14) << total_bytes
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
        const RuntimeMetadata metadata{
            hostname(),
            visible_device_list(),
            peer_access_label(options.source_device, options.destination_device),
        };
        print_header(options, metadata);

        RmmPoolScope rmm_scope(
            unique_devices({ options.source_device, options.destination_device }));
        PoseidonGpuObjectCopyMaterializer materializer;
        CudaPeerComm peer_backend;
        const std::vector<LevelShape> levels = make_levels(options.levels);
        preflight_largest_case(options, materializer);
        ResultLogger logger(options, metadata);

        for (const std::size_t batch_size : options.counts)
        {
            for (const LevelShape &level : levels)
            {
                for (const TransferMode mode : options.modes)
                {
                    const BenchCaseResult result = run_case(
                        options,
                        level,
                        batch_size,
                        mode,
                        materializer,
                        peer_backend);
                    print_result(
                        mode_name(mode),
                        batch_size,
                        level,
                        result.ciphertext_bytes,
                        result.total_bytes,
                        result.stats);
                    logger.write(mode_name(mode), batch_size, level, result);
                }
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
