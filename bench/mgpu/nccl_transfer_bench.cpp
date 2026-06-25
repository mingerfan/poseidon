#include "poseidon/gpu/gpu_ciphertext.h"
#include "poseidon/mgpu/comm/nccl_comm.h"

#include <cuda_runtime_api.h>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
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

using namespace poseidon::mgpu;

namespace
{

namespace gpu = poseidon::gpu;

using GpuWord = gpu::GpuWord;

static_assert(sizeof(GpuWord) == 4, "Poseidon GPU word size changed");

enum class Mode
{
    CudaPeerBroadcast,
    CudaPeerGather,
    NcclBroadcast,
    NcclGather,
    NcclSendRecv,
};

struct BenchOptions
{
    std::vector<int> devices{ 0, 1, 2, 3, 4, 5, 6, 7 };
    int root_device = 0;
    int target_device = -1;
    std::vector<Mode> modes{
        Mode::CudaPeerBroadcast,
        Mode::CudaPeerGather,
        Mode::NcclBroadcast,
        Mode::NcclGather,
        Mode::NcclSendRecv,
    };
    int iterations = 50;
    int warmup_iterations = 5;
    std::size_t degree = 1 << 15;
    std::size_t component_count = 2;
    std::size_t p_count = 0;
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

struct ModeResult
{
    Mode mode = Mode::CudaPeerBroadcast;
    Stats stats;
};

struct CaseShape
{
    std::size_t ciphertext_bytes = 0;
    std::size_t total_bytes = 0;
    std::size_t words_per_ciphertext = 0;
};

struct CiphertextBatch
{
    std::vector<gpu::GpuCiphertextData> ciphertexts;
    std::size_t ciphertext_bytes = 0;
    std::size_t total_bytes = 0;
    std::size_t words_per_ciphertext = 0;
};

void check_cuda(cudaError_t status, const char *what)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

class RmmPoolScope
{
public:
    explicit RmmPoolScope(const std::vector<int> &device_ids)
    {
        scopes_.reserve(device_ids.size());
        for (const int device_id : device_ids)
        {
            check_cuda(cudaSetDevice(device_id), "cudaSetDevice RMM scope");

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
    struct DeviceScope
    {
        int device_id = 0;
        std::unique_ptr<rmm::mr::cuda_memory_resource> upstream;
        std::unique_ptr<rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>> pool;
        rmm::mr::device_memory_resource *previous = nullptr;
    };

    std::vector<std::unique_ptr<DeviceScope>> scopes_;
};

void print_usage(const char *program)
{
    std::cerr
        << "usage: " << program
        << " [--devices 0,1,2,3,4,5,6,7] [--root-device N]"
           " [--target-device N] [--modes MODE[,MODE...]]"
           " [--iterations N] [--warmup N] [--degree N]"
           " [--components N] [--p-count N]\n\n"
        << "modes: cuda_peer_broadcast,cuda_peer_gather,nccl_broadcast,"
           "nccl_gather,nccl_sendrecv\n";
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

Mode parse_mode(const std::string &value)
{
    if (value == "cuda_peer_broadcast")
    {
        return Mode::CudaPeerBroadcast;
    }
    if (value == "cuda_peer_gather")
    {
        return Mode::CudaPeerGather;
    }
    if (value == "nccl_broadcast")
    {
        return Mode::NcclBroadcast;
    }
    if (value == "nccl_gather")
    {
        return Mode::NcclGather;
    }
    if (value == "nccl_sendrecv")
    {
        return Mode::NcclSendRecv;
    }
    throw std::invalid_argument("unknown mode: " + value);
}

const char *mode_name(Mode mode)
{
    switch (mode)
    {
    case Mode::CudaPeerBroadcast:
        return "cuda_peer_broadcast";
    case Mode::CudaPeerGather:
        return "cuda_peer_gather";
    case Mode::NcclBroadcast:
        return "nccl_broadcast";
    case Mode::NcclGather:
        return "nccl_gather";
    case Mode::NcclSendRecv:
        return "nccl_sendrecv";
    }
    return "unknown";
}

std::vector<int> parse_devices(const char *value)
{
    std::vector<int> devices;
    for (const std::string &part : split_csv(value))
    {
        if (part.empty())
        {
            throw std::invalid_argument("--devices contains an empty entry");
        }
        devices.push_back(parse_int(part.c_str(), "--devices"));
    }
    return devices;
}

std::vector<Mode> parse_modes(const char *value)
{
    std::vector<Mode> modes;
    for (const std::string &part : split_csv(value))
    {
        if (part.empty())
        {
            throw std::invalid_argument("--modes contains an empty entry");
        }
        modes.push_back(parse_mode(part));
    }
    return modes;
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

        if (arg == "--devices")
        {
            options.devices = parse_devices(require_value("--devices"));
        }
        else if (arg == "--root-device")
        {
            options.root_device = parse_int(require_value("--root-device"), arg.c_str());
        }
        else if (arg == "--target-device")
        {
            options.target_device = parse_int(require_value("--target-device"), arg.c_str());
        }
        else if (arg == "--modes")
        {
            options.modes = parse_modes(require_value("--modes"));
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
    return options;
}

bool contains_device(const std::vector<int> &devices, int device)
{
    return std::find(devices.begin(), devices.end(), device) != devices.end();
}

void validate_options(BenchOptions &options)
{
    if (options.devices.size() < 2)
    {
        throw std::invalid_argument("--devices must contain at least two devices");
    }
    if (options.modes.empty())
    {
        throw std::invalid_argument("--modes must contain at least one mode");
    }
    if (options.iterations <= 0)
    {
        throw std::invalid_argument("--iterations must be positive");
    }
    if (options.warmup_iterations < 0)
    {
        throw std::invalid_argument("--warmup must be non-negative");
    }
    if (options.degree == 0 || options.component_count == 0)
    {
        throw std::invalid_argument("degree and component count must be non-zero");
    }
    if (std::any_of(options.devices.begin(), options.devices.end(), [](int device) { return device < 0; }))
    {
        throw std::invalid_argument("device ids must be non-negative");
    }

    std::vector<int> sorted = options.devices;
    std::sort(sorted.begin(), sorted.end());
    if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end())
    {
        throw std::invalid_argument("--devices must not contain duplicates");
    }
    if (!contains_device(options.devices, options.root_device))
    {
        throw std::invalid_argument("--root-device must be listed in --devices");
    }

    if (options.target_device < 0)
    {
        for (const int device : options.devices)
        {
            if (device != options.root_device)
            {
                options.target_device = device;
                break;
            }
        }
    }
    if (!contains_device(options.devices, options.target_device))
    {
        throw std::invalid_argument("--target-device must be listed in --devices");
    }
    if (options.target_device == options.root_device)
    {
        throw std::invalid_argument("--target-device must differ from --root-device");
    }

    int visible_devices = 0;
    check_cuda(cudaGetDeviceCount(&visible_devices), "cudaGetDeviceCount");
    for (const int device : options.devices)
    {
        if (device >= visible_devices)
        {
            std::ostringstream stream;
            stream << "requested CUDA device " << device
                   << " but only " << visible_devices << " CUDA devices are visible";
            throw std::runtime_error(stream.str());
        }
    }
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

std::string join_devices(const std::vector<int> &devices)
{
    std::ostringstream stream;
    for (std::size_t index = 0; index < devices.size(); ++index)
    {
        if (index != 0)
        {
            stream << ',';
        }
        stream << devices[index];
    }
    return stream.str();
}

int rank_for_device(const std::vector<int> &devices, int device)
{
    const auto iter = std::find(devices.begin(), devices.end(), device);
    if (iter == devices.end())
    {
        throw std::invalid_argument("device is not listed in --devices");
    }
    return static_cast<int>(std::distance(devices.begin(), iter));
}

gpu::GpuFieldData &single_field(gpu::GpuCiphertextData &ciphertext)
{
    if (ciphertext.fields_.size() != 1)
    {
        throw std::invalid_argument("benchmark ciphertext must have exactly one GPU field");
    }
    return ciphertext.fields_[0];
}

const gpu::GpuFieldData &single_field(const gpu::GpuCiphertextData &ciphertext)
{
    if (ciphertext.fields_.size() != 1)
    {
        throw std::invalid_argument("benchmark ciphertext must have exactly one GPU field");
    }
    return ciphertext.fields_[0];
}

std::size_t field_bytes(const gpu::GpuFieldData &field)
{
    return checked_mul(
        field.size(),
        sizeof(GpuWord),
        "GPU ciphertext field byte count overflow");
}

std::size_t ciphertext_payload_bytes(const gpu::GpuCiphertextData &ciphertext)
{
    std::size_t bytes = 0;
    for (const gpu::GpuFieldData &field : ciphertext.fields_)
    {
        bytes = checked_add(
            bytes,
            field_bytes(field),
            "GPU ciphertext byte count overflow");
    }
    return bytes;
}

std::size_t ciphertext_word_count(const gpu::GpuCiphertextData &ciphertext)
{
    std::size_t words = 0;
    for (const gpu::GpuFieldData &field : ciphertext.fields_)
    {
        words = checked_add(
            words,
            field.size(),
            "GPU ciphertext word count overflow");
    }
    return words;
}

void *ciphertext_data(gpu::GpuCiphertextData &ciphertext)
{
    return single_field(ciphertext).data();
}

const void *ciphertext_data(const gpu::GpuCiphertextData &ciphertext)
{
    return single_field(ciphertext).data();
}

int ciphertext_device(const gpu::GpuCiphertextData &ciphertext)
{
    return single_field(ciphertext).device_id;
}

gpu::GpuCiphertextData make_ciphertext(
    const BenchOptions &options, const LevelShape &level, int device)
{
    gpu::GpuCiphertextData ciphertext =
        gpu::GpuCiphertextData::allocate_single_device(
            options.degree,
            level.q_count,
            options.component_count,
            device,
            options.p_count);
    ciphertext.meta.is_ntt_form = true;
    ciphertext.meta.scale = static_cast<double>(1ULL << 40);
    return ciphertext;
}

gpu::GpuCiphertextData make_gather_aggregate_ciphertext(
    const BenchOptions &options, const LevelShape &level, int device)
{
    const std::size_t aggregate_components = checked_mul(
        options.component_count,
        options.devices.size(),
        "gather aggregate component count overflow");
    gpu::GpuCiphertextData ciphertext =
        gpu::GpuCiphertextData::allocate_single_device(
            options.degree,
            level.q_count,
            aggregate_components,
            device,
            options.p_count);
    ciphertext.meta.is_ntt_form = true;
    ciphertext.meta.scale = static_cast<double>(1ULL << 40);
    return ciphertext;
}

CiphertextBatch make_ciphertext_batch(
    const BenchOptions &options, const LevelShape &level,
    std::size_t count, int device)
{
    if (count == 0)
    {
        throw std::invalid_argument("ciphertext count must be non-zero");
    }

    CiphertextBatch batch;
    batch.ciphertexts.reserve(count);
    for (std::size_t index = 0; index < count; ++index)
    {
        gpu::GpuCiphertextData ciphertext = make_ciphertext(options, level, device);
        const std::size_t bytes = ciphertext_payload_bytes(ciphertext);
        const std::size_t words = ciphertext_word_count(ciphertext);
        if (index == 0)
        {
            batch.ciphertext_bytes = bytes;
            batch.words_per_ciphertext = words;
        }
        else if (bytes != batch.ciphertext_bytes || words != batch.words_per_ciphertext)
        {
            throw std::logic_error("benchmark ciphertext batch shape mismatch");
        }
        batch.total_bytes = checked_add(
            batch.total_bytes,
            bytes,
            "ciphertext batch byte count overflow");
        batch.ciphertexts.push_back(std::move(ciphertext));
    }
    return batch;
}

std::vector<CiphertextBatch> make_rank_batches(
    const BenchOptions &options, const LevelShape &level, std::size_t count)
{
    std::vector<CiphertextBatch> batches;
    batches.reserve(options.devices.size());
    for (const int device : options.devices)
    {
        batches.push_back(make_ciphertext_batch(options, level, count, device));
    }
    return batches;
}

std::vector<gpu::GpuCiphertextData> make_gather_aggregates(
    const BenchOptions &options, const LevelShape &level, std::size_t count,
    std::size_t ciphertext_bytes)
{
    std::vector<gpu::GpuCiphertextData> aggregates;
    aggregates.reserve(count);
    for (std::size_t index = 0; index < count; ++index)
    {
        gpu::GpuCiphertextData aggregate =
            make_gather_aggregate_ciphertext(options, level, options.root_device);
        const std::size_t expected_bytes = checked_mul(
            ciphertext_bytes,
            options.devices.size(),
            "gather aggregate byte count overflow");
        if (ciphertext_payload_bytes(aggregate) != expected_bytes)
        {
            throw std::logic_error("gather aggregate ciphertext shape mismatch");
        }
        aggregates.push_back(std::move(aggregate));
    }
    return aggregates;
}

CaseShape make_case_shape(
    const BenchOptions &options, const LevelShape &level, std::size_t count)
{
    CiphertextBatch batch =
        make_ciphertext_batch(options, level, count, options.root_device);
    return CaseShape{
        batch.ciphertext_bytes,
        batch.total_bytes,
        batch.words_per_ciphertext,
    };
}

GpuWord deterministic_word(
    int device, std::size_t ciphertext_index, std::size_t word_index)
{
    std::uint32_t value = 0x9e3779b9u;
    value ^= static_cast<std::uint32_t>(device + 1) * 0x85ebca6bu;
    value += static_cast<std::uint32_t>(ciphertext_index + 1) * 0x27d4eb2du;
    value += static_cast<std::uint32_t>(word_index) * 0xc2b2ae35u;
    value ^= static_cast<std::uint32_t>(word_index >> 32);
    return static_cast<GpuWord>(value);
}

void fill_ciphertext(
    gpu::GpuCiphertextData &ciphertext, int pattern_device,
    std::size_t ciphertext_index)
{
    gpu::GpuFieldData &field = single_field(ciphertext);
    std::vector<GpuWord> host(field.size());
    for (std::size_t index = 0; index < host.size(); ++index)
    {
        host[index] = deterministic_word(pattern_device, ciphertext_index, index);
    }
    field.buffer.copy_from_host(host.data(), host.size());
}

void fill_batch(CiphertextBatch &batch, int pattern_device)
{
    for (std::size_t index = 0; index < batch.ciphertexts.size(); ++index)
    {
        fill_ciphertext(batch.ciphertexts[index], pattern_device, index);
    }
}

void memset_ciphertext(gpu::GpuCiphertextData &ciphertext, int value)
{
    gpu::GpuFieldData &field = single_field(ciphertext);
    check_cuda(cudaSetDevice(field.device_id), "cudaSetDevice memset ciphertext");
    check_cuda(
        cudaMemset(field.data(), value, field_bytes(field)),
        "cudaMemset ciphertext");
}

void memset_batch(CiphertextBatch &batch, int value)
{
    for (gpu::GpuCiphertextData &ciphertext : batch.ciphertexts)
    {
        memset_ciphertext(ciphertext, value);
    }
}

std::vector<void *> mutable_rank_ptrs(
    std::vector<CiphertextBatch> &batches, std::size_t ciphertext_index)
{
    std::vector<void *> result;
    result.reserve(batches.size());
    for (CiphertextBatch &batch : batches)
    {
        if (ciphertext_index >= batch.ciphertexts.size())
        {
            throw std::out_of_range("ciphertext index is out of range");
        }
        result.push_back(ciphertext_data(batch.ciphertexts[ciphertext_index]));
    }
    return result;
}

std::vector<const void *> const_rank_ptrs(
    const std::vector<CiphertextBatch> &batches, std::size_t ciphertext_index)
{
    std::vector<const void *> result;
    result.reserve(batches.size());
    for (const CiphertextBatch &batch : batches)
    {
        if (ciphertext_index >= batch.ciphertexts.size())
        {
            throw std::out_of_range("ciphertext index is out of range");
        }
        result.push_back(ciphertext_data(batch.ciphertexts[ciphertext_index]));
    }
    return result;
}

void synchronize_devices(const std::vector<int> &devices)
{
    for (const int device : devices)
    {
        check_cuda(cudaSetDevice(device), "cudaSetDevice synchronize");
        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    }
}

std::vector<std::size_t> sample_indices(std::size_t word_count)
{
    if (word_count == 0)
    {
        throw std::invalid_argument("cannot sample an empty payload");
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

GpuWord read_device_word(const void *device_data, int device, std::size_t word_index)
{
    GpuWord value = 0;
    check_cuda(cudaSetDevice(device), "cudaSetDevice read");
    check_cuda(
        cudaMemcpy(
            &value,
            static_cast<const GpuWord *>(device_data) + word_index,
            sizeof(GpuWord),
            cudaMemcpyDeviceToHost),
        "cudaMemcpy read");
    return value;
}

void expect_word(
    const void *device_data, int device, std::size_t word_index,
    GpuWord expected, const char *context)
{
    const GpuWord actual = read_device_word(device_data, device, word_index);
    if (actual != expected)
    {
        std::ostringstream stream;
        stream << context << " validation failed on device " << device
               << " at word " << word_index
               << ": expected 0x" << std::hex << expected
               << " got 0x" << actual;
        throw std::runtime_error(stream.str());
    }
}

void validate_broadcast(
    const std::vector<CiphertextBatch> &batches, const BenchOptions &options)
{
    const int root_rank = rank_for_device(options.devices, options.root_device);
    const std::vector<std::size_t> samples =
        sample_indices(batches[static_cast<std::size_t>(root_rank)].words_per_ciphertext);

    for (std::size_t rank = 0; rank < batches.size(); ++rank)
    {
        if (rank == static_cast<std::size_t>(root_rank))
        {
            continue;
        }
        const CiphertextBatch &batch = batches[rank];
        for (std::size_t ciphertext_index = 0;
             ciphertext_index < batch.ciphertexts.size();
             ++ciphertext_index)
        {
            const gpu::GpuCiphertextData &ciphertext =
                batch.ciphertexts[ciphertext_index];
            for (const std::size_t word_index : samples)
            {
                expect_word(
                    ciphertext_data(ciphertext),
                    ciphertext_device(ciphertext),
                    word_index,
                    deterministic_word(options.root_device, ciphertext_index, word_index),
                    "broadcast");
            }
        }
    }
}

void validate_gather(
    const std::vector<gpu::GpuCiphertextData> &aggregates,
    const BenchOptions &options, std::size_t words_per_ciphertext,
    bool include_root)
{
    const int root_rank = rank_for_device(options.devices, options.root_device);
    const std::vector<std::size_t> samples = sample_indices(words_per_ciphertext);
    for (std::size_t rank = 0; rank < options.devices.size(); ++rank)
    {
        if (rank == static_cast<std::size_t>(root_rank) && !include_root)
        {
            continue;
        }
        for (std::size_t ciphertext_index = 0;
             ciphertext_index < aggregates.size();
             ++ciphertext_index)
        {
            const gpu::GpuCiphertextData &aggregate = aggregates[ciphertext_index];
            for (const std::size_t word_index : samples)
            {
                const std::size_t aggregate_index = checked_add(
                    checked_mul(
                        rank,
                        words_per_ciphertext,
                        "gather validation offset overflow"),
                    word_index,
                    "gather validation offset overflow");
                expect_word(
                    ciphertext_data(aggregate),
                    ciphertext_device(aggregate),
                    aggregate_index,
                    deterministic_word(
                        options.devices[rank], ciphertext_index, word_index),
                    "gather");
            }
        }
    }
}

void validate_sendrecv(
    const CiphertextBatch &destination, const BenchOptions &options)
{
    const std::vector<std::size_t> samples =
        sample_indices(destination.words_per_ciphertext);
    for (std::size_t ciphertext_index = 0;
         ciphertext_index < destination.ciphertexts.size();
         ++ciphertext_index)
    {
        const gpu::GpuCiphertextData &ciphertext =
            destination.ciphertexts[ciphertext_index];
        for (const std::size_t word_index : samples)
        {
            expect_word(
                ciphertext_data(ciphertext),
                ciphertext_device(ciphertext),
                word_index,
                deterministic_word(options.root_device, ciphertext_index, word_index),
                "sendrecv");
        }
    }
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

template <typename Operation, typename Synchronize>
Stats measure(
    int warmup_iterations, int iterations,
    Operation &&operation, Synchronize &&synchronize)
{
    for (int iter = 0; iter < warmup_iterations; ++iter)
    {
        operation();
        synchronize();
    }

    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(iterations));
    for (int iter = 0; iter < iterations; ++iter)
    {
        synchronize();
        const auto start = std::chrono::steady_clock::now();
        operation();
        synchronize();
        const auto stop = std::chrono::steady_clock::now();
        const auto elapsed =
            std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
                stop - start);
        samples.push_back(elapsed.count());
    }
    return summarize(samples);
}

void enable_peer_accesses(const std::vector<int> &devices)
{
    for (const int destination : devices)
    {
        check_cuda(cudaSetDevice(destination), "cudaSetDevice enable peer access");
        for (const int source : devices)
        {
            if (source == destination)
            {
                continue;
            }

            int can_access = 0;
            check_cuda(
                cudaDeviceCanAccessPeer(&can_access, destination, source),
                "cudaDeviceCanAccessPeer");
            if (!can_access)
            {
                continue;
            }

            const cudaError_t status = cudaDeviceEnablePeerAccess(source, 0);
            if (status == cudaSuccess)
            {
                continue;
            }
            if (status == cudaErrorPeerAccessAlreadyEnabled)
            {
                (void)cudaGetLastError();
                continue;
            }
            check_cuda(status, "cudaDeviceEnablePeerAccess");
        }
    }
}

Stats run_cuda_peer_broadcast(
    const BenchOptions &options, const LevelShape &level, std::size_t count)
{
    std::vector<CiphertextBatch> batches =
        make_rank_batches(options, level, count);
    const int root_rank = rank_for_device(options.devices, options.root_device);
    for (std::size_t rank = 0; rank < batches.size(); ++rank)
    {
        if (rank == static_cast<std::size_t>(root_rank))
        {
            fill_batch(batches[rank], options.root_device);
        }
        else
        {
            memset_batch(batches[rank], 0xa5);
        }
    }

    const std::size_t bytes =
        batches[static_cast<std::size_t>(root_rank)].ciphertext_bytes;
    auto operation = [&]() {
        for (std::size_t rank = 0; rank < batches.size(); ++rank)
        {
            if (rank == static_cast<std::size_t>(root_rank))
            {
                continue;
            }
            for (std::size_t ciphertext_index = 0;
                 ciphertext_index < batches[rank].ciphertexts.size();
                 ++ciphertext_index)
            {
                check_cuda(
                    cudaMemcpyPeer(
                        ciphertext_data(batches[rank].ciphertexts[ciphertext_index]),
                        options.devices[rank],
                        ciphertext_data(
                            batches[static_cast<std::size_t>(root_rank)]
                                .ciphertexts[ciphertext_index]),
                        options.root_device,
                        bytes),
                    "cudaMemcpyPeer broadcast");
            }
        }
    };
    auto synchronize = [&]() { synchronize_devices(options.devices); };

    Stats stats = measure(
        options.warmup_iterations, options.iterations, operation, synchronize);
    validate_broadcast(batches, options);
    return stats;
}

Stats run_cuda_peer_gather(
    const BenchOptions &options, const LevelShape &level, std::size_t count)
{
    std::vector<CiphertextBatch> send_batches =
        make_rank_batches(options, level, count);
    for (std::size_t rank = 0; rank < send_batches.size(); ++rank)
    {
        fill_batch(send_batches[rank], options.devices[rank]);
    }

    const std::size_t bytes = send_batches.front().ciphertext_bytes;
    std::vector<gpu::GpuCiphertextData> aggregates =
        make_gather_aggregates(options, level, count, bytes);
    for (gpu::GpuCiphertextData &aggregate : aggregates)
    {
        memset_ciphertext(aggregate, 0x5a);
    }

    const int root_rank = rank_for_device(options.devices, options.root_device);
    auto operation = [&]() {
        for (std::size_t rank = 0; rank < send_batches.size(); ++rank)
        {
            if (rank == static_cast<std::size_t>(root_rank))
            {
                continue;
            }
            for (std::size_t ciphertext_index = 0;
                 ciphertext_index < send_batches[rank].ciphertexts.size();
                 ++ciphertext_index)
            {
                check_cuda(
                    cudaMemcpyPeer(
                        static_cast<unsigned char *>(
                            ciphertext_data(aggregates[ciphertext_index])) +
                            rank * bytes,
                        options.root_device,
                        ciphertext_data(
                            send_batches[rank].ciphertexts[ciphertext_index]),
                        options.devices[rank],
                        bytes),
                    "cudaMemcpyPeer gather");
            }
        }
    };
    auto synchronize = [&]() { synchronize_devices(options.devices); };

    Stats stats = measure(
        options.warmup_iterations, options.iterations, operation, synchronize);
    validate_gather(
        aggregates, options, send_batches.front().words_per_ciphertext, false);
    return stats;
}

Stats run_nccl_broadcast(
    const BenchOptions &options, const LevelShape &level, std::size_t count,
    NcclComm &comm)
{
    std::vector<CiphertextBatch> batches =
        make_rank_batches(options, level, count);
    const int root_rank = rank_for_device(options.devices, options.root_device);
    for (std::size_t rank = 0; rank < batches.size(); ++rank)
    {
        if (rank == static_cast<std::size_t>(root_rank))
        {
            fill_batch(batches[rank], options.root_device);
        }
        else
        {
            memset_batch(batches[rank], 0xa5);
        }
    }

    std::vector<std::vector<void *>> ptrs_by_ciphertext;
    ptrs_by_ciphertext.reserve(count);
    for (std::size_t ciphertext_index = 0; ciphertext_index < count; ++ciphertext_index)
    {
        ptrs_by_ciphertext.push_back(
            mutable_rank_ptrs(batches, ciphertext_index));
    }

    const std::size_t bytes =
        batches[static_cast<std::size_t>(root_rank)].ciphertext_bytes;
    auto operation = [&]() {
        for (std::vector<void *> &ptrs : ptrs_by_ciphertext)
        {
            comm.broadcast(ptrs, bytes, root_rank);
        }
    };
    auto synchronize = [&]() { comm.synchronize_streams(); };

    Stats stats = measure(
        options.warmup_iterations, options.iterations, operation, synchronize);
    validate_broadcast(batches, options);
    return stats;
}

Stats run_nccl_gather(
    const BenchOptions &options, const LevelShape &level, std::size_t count,
    NcclComm &comm)
{
    std::vector<CiphertextBatch> send_batches =
        make_rank_batches(options, level, count);
    for (std::size_t rank = 0; rank < send_batches.size(); ++rank)
    {
        fill_batch(send_batches[rank], options.devices[rank]);
    }

    const std::size_t bytes = send_batches.front().ciphertext_bytes;
    std::vector<gpu::GpuCiphertextData> aggregates =
        make_gather_aggregates(options, level, count, bytes);
    for (gpu::GpuCiphertextData &aggregate : aggregates)
    {
        memset_ciphertext(aggregate, 0x5a);
    }

    std::vector<std::vector<const void *>> ptrs_by_ciphertext;
    ptrs_by_ciphertext.reserve(count);
    for (std::size_t ciphertext_index = 0; ciphertext_index < count; ++ciphertext_index)
    {
        ptrs_by_ciphertext.push_back(
            const_rank_ptrs(send_batches, ciphertext_index));
    }

    const int root_rank = rank_for_device(options.devices, options.root_device);
    auto operation = [&]() {
        for (std::size_t ciphertext_index = 0;
             ciphertext_index < ptrs_by_ciphertext.size();
             ++ciphertext_index)
        {
            comm.gather(
                ptrs_by_ciphertext[ciphertext_index],
                ciphertext_data(aggregates[ciphertext_index]),
                bytes,
                root_rank);
        }
    };
    auto synchronize = [&]() { comm.synchronize_streams(); };

    Stats stats = measure(
        options.warmup_iterations, options.iterations, operation, synchronize);
    validate_gather(
        aggregates, options, send_batches.front().words_per_ciphertext, true);
    return stats;
}

Stats run_nccl_sendrecv(
    const BenchOptions &options, const LevelShape &level, std::size_t count,
    NcclComm &comm)
{
    CiphertextBatch source =
        make_ciphertext_batch(options, level, count, options.root_device);
    CiphertextBatch destination =
        make_ciphertext_batch(options, level, count, options.target_device);
    fill_batch(source, options.root_device);
    memset_batch(destination, 0xa5);

    const int source_rank = rank_for_device(options.devices, options.root_device);
    const int destination_rank = rank_for_device(options.devices, options.target_device);
    const std::size_t bytes = source.ciphertext_bytes;

    auto operation = [&]() {
        for (std::size_t ciphertext_index = 0;
             ciphertext_index < source.ciphertexts.size();
             ++ciphertext_index)
        {
            comm.send_recv(
                ciphertext_data(source.ciphertexts[ciphertext_index]),
                ciphertext_data(destination.ciphertexts[ciphertext_index]),
                bytes,
                source_rank,
                destination_rank);
        }
    };
    auto synchronize = [&]() { comm.synchronize_streams(); };

    Stats stats = measure(
        options.warmup_iterations, options.iterations, operation, synchronize);
    validate_sendrecv(destination, options);
    return stats;
}

bool requires_nccl(Mode mode)
{
    return mode == Mode::NcclBroadcast ||
           mode == Mode::NcclGather ||
           mode == Mode::NcclSendRecv;
}

Stats run_mode(
    Mode mode, const BenchOptions &options, const LevelShape &level,
    std::size_t count, NcclComm *comm)
{
    switch (mode)
    {
    case Mode::CudaPeerBroadcast:
        return run_cuda_peer_broadcast(options, level, count);
    case Mode::CudaPeerGather:
        return run_cuda_peer_gather(options, level, count);
    case Mode::NcclBroadcast:
        if (comm == nullptr)
        {
            throw std::logic_error("NCCL communicator is not initialized");
        }
        return run_nccl_broadcast(options, level, count, *comm);
    case Mode::NcclGather:
        if (comm == nullptr)
        {
            throw std::logic_error("NCCL communicator is not initialized");
        }
        return run_nccl_gather(options, level, count, *comm);
    case Mode::NcclSendRecv:
        if (comm == nullptr)
        {
            throw std::logic_error("NCCL communicator is not initialized");
        }
        return run_nccl_sendrecv(options, level, count, *comm);
    }
    throw std::logic_error("unhandled benchmark mode");
}

std::optional<double> speedup_for_mode(
    Mode mode, const std::vector<ModeResult> &results)
{
    const auto find_stats = [&](Mode wanted) -> std::optional<Stats> {
        for (const ModeResult &result : results)
        {
            if (result.mode == wanted)
            {
                return result.stats;
            }
        }
        return std::nullopt;
    };

    if (mode == Mode::CudaPeerBroadcast || mode == Mode::CudaPeerGather)
    {
        return 1.0;
    }
    if (mode == Mode::NcclBroadcast)
    {
        const std::optional<Stats> baseline = find_stats(Mode::CudaPeerBroadcast);
        const std::optional<Stats> current = find_stats(mode);
        if (baseline && current)
        {
            return baseline->average_ms / current->average_ms;
        }
    }
    if (mode == Mode::NcclGather)
    {
        const std::optional<Stats> baseline = find_stats(Mode::CudaPeerGather);
        const std::optional<Stats> current = find_stats(mode);
        if (baseline && current)
        {
            return baseline->average_ms / current->average_ms;
        }
    }
    return std::nullopt;
}

std::size_t transfer_bytes_for_mode(
    Mode mode, std::size_t total_bytes, std::size_t rank_count)
{
    if (mode == Mode::NcclSendRecv)
    {
        return total_bytes;
    }
    return checked_mul(
        total_bytes,
        rank_count - 1,
        "effective transfer byte count overflow");
}

void print_header(const BenchOptions &options, const NcclComm *comm)
{
    std::cout << "Poseidon mgpu NCCL CKKS ciphertext transfer bench\n";
    std::cout << "devices=" << join_devices(options.devices)
              << " root=" << options.root_device
              << " target=" << options.target_device
              << " iterations=" << options.iterations
              << " warmup=" << options.warmup_iterations
              << " degree=" << options.degree
              << " components=" << options.component_count
              << " p_count=" << options.p_count;
    if (comm != nullptr)
    {
        std::cout << " nccl_gather="
                  << (comm->has_native_gather() ? "native" : "sendrecv_fallback");
    }
    std::cout << "\n\n";

    std::cout
        << std::left
        << std::setw(22) << "mode"
        << std::setw(20) << "devices"
        << std::setw(8) << "root"
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
        << std::setw(24) << "speedup_vs_cuda_peer"
        << "\n";
}

void print_result(
    const BenchOptions &options, const LevelShape &level, std::size_t count,
    const CaseShape &shape, const ModeResult &result,
    const std::vector<ModeResult> &case_results)
{
    const std::size_t transfer_bytes =
        transfer_bytes_for_mode(result.mode, shape.total_bytes, options.devices.size());
    const double gbps =
        (static_cast<double>(transfer_bytes) / 1.0e9) /
        (result.stats.average_ms / 1000.0);
    const std::optional<double> speedup =
        speedup_for_mode(result.mode, case_results);

    std::ostringstream speedup_stream;
    if (speedup)
    {
        speedup_stream << std::fixed << std::setprecision(3) << *speedup;
    }
    else
    {
        speedup_stream << "n/a";
    }

    std::cout
        << std::left
        << std::setw(22) << mode_name(result.mode)
        << std::setw(20) << join_devices(options.devices)
        << std::setw(8) << options.root_device
        << std::setw(8) << count
        << std::setw(10) << level.name
        << std::setw(10) << level.q_count
        << std::setw(14) << shape.ciphertext_bytes
        << std::setw(14) << shape.total_bytes
        << std::setw(14) << transfer_bytes
        << std::setw(14) << std::fixed << std::setprecision(4) << result.stats.average_ms
        << std::setw(14) << std::fixed << std::setprecision(4) << result.stats.min_ms
        << std::setw(14) << std::fixed << std::setprecision(4) << result.stats.max_ms
        << std::setw(14) << std::fixed << std::setprecision(3) << gbps
        << std::setw(24) << speedup_stream.str()
        << "\n";
}

}  // namespace

int main(int argc, char **argv)
{
    try
    {
        BenchOptions options = parse_options(argc, argv);
        validate_options(options);
        enable_peer_accesses(options.devices);

        RmmPoolScope rmm_scope(options.devices);

        const bool any_nccl = std::any_of(
            options.modes.begin(),
            options.modes.end(),
            [](Mode mode) { return requires_nccl(mode); });
        std::unique_ptr<NcclComm> comm;
        if (any_nccl)
        {
            comm = std::make_unique<NcclComm>(options.devices);
        }

        print_header(options, comm.get());

        const std::vector<std::size_t> counts{ 1, 5, 10 };
        const std::vector<LevelShape> levels{
            { "L4", 4 },
            { "L8", 8 },
            { "L12", 12 },
            { "L16", 16 },
            { "L20", 20 },
        };

        for (const std::size_t count : counts)
        {
            for (const LevelShape &level : levels)
            {
                const CaseShape shape = make_case_shape(options, level, count);

                std::vector<ModeResult> case_results;
                case_results.reserve(options.modes.size());
                for (const Mode mode : options.modes)
                {
                    case_results.push_back(ModeResult{
                        mode,
                        run_mode(mode, options, level, count, comm.get()),
                    });
                }

                for (const ModeResult &result : case_results)
                {
                    print_result(options, level, count, shape, result, case_results);
                }
            }
        }
    }
    catch (const std::exception &ex)
    {
        std::cerr << "poseidon_mgpu_nccl_transfer_bench: " << ex.what() << '\n';
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
