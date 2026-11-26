#include "poseidon/gpu/gpu_parameter.h"

#include "poseidon/basics/util/ntt.h"
#include "poseidon/basics/util/rns.h"
#include "poseidon/basics/util/uintarithsmallmod.h"
#include "poseidon/poseidon_context.h"
#include "poseidon/util/rns_tool_qp.h"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <utility>

namespace poseidon
{
namespace gpu
{
namespace
{

constexpr int kDefaultPrecomputedFusedMatrixStages = 0;
constexpr int kMaxPrecomputedFusedMatrixStages = 4;
constexpr const char *kFusedMatrixCacheDirEnv =
    "POSEIDON_NTT_FUSED_MATRIX_CACHE_DIR";
constexpr const char *kFusedMatrixProgressEnv =
    "POSEIDON_NTT_FUSED_MATRIX_PROGRESS";
constexpr const char *kFusedMatrixFp64TablesEnv =
    "POSEIDON_NTT_FUSED_MATRIX_FP64_TABLES";
constexpr const char *kFusedMatrixMaxLevelsEnv =
    "POSEIDON_NTT_FUSED_MATRIX_MAX_LEVELS";
constexpr const char *kSkipMontgomeryTablesEnv =
    "POSEIDON_GPU_SKIP_MONTGOMERY_TABLES";

std::string format_parms_id_for_error(const parms_id_type &parms_id)
{
    std::ostringstream stream;
    stream << std::hex << std::setfill('0');
    for (std::size_t i = 0; i < parms_id.size(); ++i)
    {
        if (i != 0)
        {
            stream << ':';
        }
        stream << std::setw(16) << parms_id[i];
    }
    return stream.str();
}

std::string format_known_levels_for_error(
    const std::vector<GpuLevelInfo> &levels)
{
    std::ostringstream stream;
    for (std::size_t i = 0; i < levels.size(); ++i)
    {
        const auto &level = levels[i];
        if (i != 0)
        {
            stream << "; ";
        }
        stream << "idx=" << i
               << ",q=" << level.q_count
               << ",p=" << level.p_count
               << ",parms_id=" << format_parms_id_for_error(level.parms_id);
    }
    return stream.str();
}
constexpr std::uint64_t kFusedMatrixCacheMagic = 0x314d41544e445350ULL;
constexpr std::uint32_t kFusedMatrixCacheVersion = 1;

GpuWord checked_gpu_word(std::uint64_t value, const char *what)
{
    if (value > std::numeric_limits<GpuWord>::max())
    {
        throw std::invalid_argument(what);
    }
    return static_cast<GpuWord>(value);
}

std::vector<Modulus> copy_rns_base_moduli(const util::RNSBase &base)
{
    std::vector<Modulus> result;
    result.reserve(base.size());
    for (std::size_t i = 0; i < base.size(); ++i)
    {
        result.push_back(base.base()[i]);
    }
    return result;
}

GpuWide barrett_ratio_64(const Modulus &modulus)
{
    const auto value = modulus.value();
    if (value == 0)
    {
        throw std::invalid_argument("GpuParameterData cannot build Barrett constant for zero modulus");
    }

    const auto numerator = static_cast<unsigned __int128>(1) << 64;
    return static_cast<GpuWide>(numerator / value);
}

GpuWord montgomery_r_mod(const Modulus &modulus)
{
    const auto value = modulus.value();
    if (value == 0 || value > std::numeric_limits<GpuWord>::max())
    {
        throw std::invalid_argument("GpuParameterData cannot build Montgomery R for invalid modulus");
    }
    const auto r = static_cast<unsigned __int128>(1) << 32;
    return static_cast<GpuWord>(r % value);
}

GpuWord montgomery_neg_inverse_u32(const Modulus &modulus)
{
    const auto value = checked_gpu_word(
        modulus.value(),
        "GpuParameterData only supports Montgomery moduli that fit in GpuWord");
    if ((value & 1U) == 0U)
    {
        throw std::invalid_argument("GpuParameterData Montgomery modulus must be odd");
    }

    GpuWord inverse = 1;
    for (int i = 0; i < 5; ++i)
    {
        inverse *= static_cast<GpuWord>(2U - value * inverse);
    }
    return static_cast<GpuWord>(0U - inverse);
}

GpuWord multiply_mod_host(
    GpuWord left,
    GpuWord right,
    const Modulus &modulus)
{
    return static_cast<GpuWord>(
        (static_cast<GpuWide>(left) * static_cast<GpuWide>(right)) %
        static_cast<GpuWide>(modulus.value()));
}

GpuWord multiply_mod_host(
    GpuWord left,
    GpuWord right,
    GpuWord modulus)
{
    return static_cast<GpuWord>(
        (static_cast<GpuWide>(left) * static_cast<GpuWide>(right)) %
        static_cast<GpuWide>(modulus));
}

template <typename T>
std::vector<T> concatenate_vectors(
    const std::vector<T> &first,
    const std::vector<T> &second)
{
    std::vector<T> result;
    result.reserve(first.size() + second.size());
    result.insert(result.end(), first.begin(), first.end());
    result.insert(result.end(), second.begin(), second.end());
    return result;
}

template <typename T>
DeviceVector<T> copy_to_device_vector_async(
    const std::vector<T> &host,
    int device_id,
    cudaStream_t stream)
{
    DeviceVector<T> result(host.size(), device_id);
    if (!host.empty())
    {
        gpu_check_cuda(
            cudaMemcpyAsync(
                result.data(),
                host.data(),
                host.size() * sizeof(T),
                cudaMemcpyHostToDevice,
                stream),
            "GpuParameterData cudaMemcpyAsync HostToDevice");
    }
    return result;
}

DeviceVector<GpuWord> slice_master_roots_d2d_async(
    const DeviceVector<GpuWord> &master,
    std::size_t max_q_used,
    std::size_t q_count,
    std::size_t p_count,
    std::size_t degree,
    int device_id,
    cudaStream_t stream)
{
    const std::size_t total_elems = (q_count + p_count) * degree;
    DeviceVector<GpuWord> result(total_elems, device_id);
    if (q_count > 0 && degree > 0)
    {
        gpu_check_cuda(
            cudaMemcpyAsync(
                result.data(),
                master.data(),
                q_count * degree * sizeof(GpuWord),
                cudaMemcpyDeviceToDevice,
                stream),
            "GpuParameterData cudaMemcpyAsync D2D Q roots");
    }
    if (p_count > 0 && degree > 0)
    {
        gpu_check_cuda(
            cudaMemcpyAsync(
                result.data() + q_count * degree,
                master.data() + max_q_used * degree,
                p_count * degree * sizeof(GpuWord),
                cudaMemcpyDeviceToDevice,
                stream),
            "GpuParameterData cudaMemcpyAsync D2D P roots");
    }
    return result;
}

struct StreamSyncOnExit
{
    int device_id = 0;
    cudaStream_t stream = nullptr;
    bool synced = false;

    ~StreamSyncOnExit()
    {
        if (!synced && stream != nullptr)
        {
            (void)cudaSetDevice(device_id);
            (void)cudaStreamSynchronize(stream);
        }
    }
};

struct FusedNttMatrixTables
{
    std::size_t fusion_stages = 0;
    std::vector<GpuWord> stage_counts;
    std::vector<GpuWide> group_counts;
    std::vector<GpuWide> stage_offsets;
    std::vector<GpuWord> matrices;
};

struct FusedNttMatrixFp64SplitTables
{
    std::vector<double> lo;
    std::vector<double> hi;
};

FusedNttMatrixFp64SplitTables split_fused_matrices_for_fp64(
    const std::vector<GpuWord> &matrices)
{
    FusedNttMatrixFp64SplitTables result;
    result.lo.reserve(matrices.size());
    result.hi.reserve(matrices.size());
    for (GpuWord value : matrices)
    {
        /*B按照高低位进行存储，边于32*16的乘法保证精度*/
        result.lo.push_back(static_cast<double>(value & 0xffffu));
        result.hi.push_back(static_cast<double>(value >> 16));
    }
    return result;
}

bool env_enabled(const char *name)
{
    const char *raw = std::getenv(name);
    if (raw == nullptr || raw[0] == '\0')
    {
        return false;
    }
    const std::string value(raw);
    return value != "0" && value != "false" && value != "FALSE" &&
           value != "off" && value != "OFF";
}

bool fused_matrix_progress_enabled()
{
    return env_enabled(kFusedMatrixProgressEnv);
}

bool skip_montgomery_tables_enabled()
{
    if (!env_enabled(kSkipMontgomeryTablesEnv))
    {
        return false;
    }
    const char *algorithm = std::getenv("POSEIDON_NTT_ALGO");
    if (algorithm != nullptr)
    {
        const std::string value(algorithm);
        if (value == "montgomery" || value == "mont")
        {
            return false;
        }
    }
    return true;
}

bool is_fp64_tensor_ntt_algorithm(const std::string &value)
{
    return value == "tensor_fp64" || value == "fp64" ||
           value == "tam_fp64" || value == "neo" ||
           value == "tensor_neo_fp64";
}

bool fused_matrix_fp64_tables_enabled()
{
    const char *raw = std::getenv(kFusedMatrixFp64TablesEnv);
    if (raw != nullptr && raw[0] != '\0')
    {
        return env_enabled(kFusedMatrixFp64TablesEnv);
    }

    const char *algorithm = std::getenv("POSEIDON_NTT_ALGO");
    return algorithm != nullptr &&
           is_fp64_tensor_ntt_algorithm(std::string(algorithm));
}

bool fp64_tensor_ntt_algorithm_requested()
{
    const char *algorithm = std::getenv("POSEIDON_NTT_ALGO");
    return algorithm != nullptr &&
           is_fp64_tensor_ntt_algorithm(std::string(algorithm));
}

std::size_t read_fused_matrix_max_levels()
{
    const char *raw = std::getenv(kFusedMatrixMaxLevelsEnv);
    if (raw == nullptr || raw[0] == '\0')
    {
        return std::numeric_limits<std::size_t>::max();
    }

    char *end = nullptr;
    errno = 0;
    const unsigned long long value = std::strtoull(raw, &end, 10);
    if (errno != 0 || end == raw || *end != '\0')
    {
        throw std::invalid_argument(
            "POSEIDON_NTT_FUSED_MATRIX_MAX_LEVELS must be a non-negative integer");
    }

    return static_cast<std::size_t>(value);
}

void print_progress_bar(
    const std::string &label,
    std::size_t current,
    std::size_t total)
{
    if (!fused_matrix_progress_enabled() || total == 0)
    {
        return;
    }

    constexpr std::size_t width = 30;
    const std::size_t clamped = std::min(current, total);
    const std::size_t filled = clamped * width / total;
    std::cout << "\r" << label << " [";
    for (std::size_t i = 0; i < width; ++i)
    {
        std::cout << (i < filled ? '#' : '.');
    }
    std::cout << "] " << clamped << "/" << total << std::flush;
    if (clamped == total)
    {
        std::cout << "\n";
    }
}

bool should_update_progress(std::size_t current, std::size_t total)
{
    if (!fused_matrix_progress_enabled())
    {
        return false;
    }
    const std::size_t step = std::max<std::size_t>(total / 50, 1);
    return current == 1 || current == total || current % step == 0;
}

void print_fused_matrix_message(const std::string &message)
{
    if (fused_matrix_progress_enabled())
    {
        std::cout << message << "\n";
    }
}

std::string read_fused_matrix_cache_dir()
{
    const char *raw = std::getenv(kFusedMatrixCacheDirEnv);
    return raw == nullptr ? std::string{} : std::string(raw);
}

void ensure_directory_exists(const std::string &path)
{
    if (path.empty())
    {
        return;
    }
    if (::mkdir(path.c_str(), 0755) != 0 && errno != EEXIST)
    {
        throw std::runtime_error(
            "GpuParameterData failed to create fused NTT matrix cache directory: " +
            path);
    }
}

void hash_combine_u64(std::uint64_t &hash, std::uint64_t value)
{
    constexpr std::uint64_t kFnvPrime = 1099511628211ULL;
    for (int i = 0; i < 8; ++i)
    {
        hash ^= static_cast<unsigned char>((value >> (i * 8)) & 0xffU);
        hash *= kFnvPrime;
    }
}

std::uint64_t fused_matrix_cache_hash(
    bool inverse,
    std::size_t degree,
    int fusion_stages,
    const std::vector<std::uint64_t> &moduli)
{
    std::uint64_t hash = 1469598103934665603ULL;
    hash_combine_u64(hash, inverse ? 1 : 0);
    hash_combine_u64(hash, static_cast<std::uint64_t>(degree));
    hash_combine_u64(hash, static_cast<std::uint64_t>(fusion_stages));
    hash_combine_u64(hash, static_cast<std::uint64_t>(moduli.size()));
    for (const auto modulus : moduli)
    {
        hash_combine_u64(hash, modulus);
    }
    return hash;
}

std::vector<std::uint64_t> collect_table_moduli(
    const std::vector<const util::NTTTables *> &tables)
{
    std::vector<std::uint64_t> result;
    result.reserve(tables.size());
    for (const auto *table : tables)
    {
        if (table == nullptr)
        {
            throw std::invalid_argument("GpuParameterData requires CPU NTT tables");
        }
        result.push_back(table->modulus().value());
    }
    return result;
}

std::string fused_matrix_cache_path(
    const std::string &cache_dir,
    bool inverse,
    std::size_t degree,
    int fusion_stages,
    const std::vector<std::uint64_t> &moduli)
{
    std::ostringstream stream;
    stream << cache_dir;
    if (!cache_dir.empty() && cache_dir.back() != '/')
    {
        stream << '/';
    }
    stream << "poseidon_tam_v" << kFusedMatrixCacheVersion
           << (inverse ? "_intt" : "_ntt")
           << "_n" << degree
           << "_f" << fusion_stages
           << "_l" << moduli.size()
           << "_" << std::hex << std::setw(16) << std::setfill('0')
           << fused_matrix_cache_hash(inverse, degree, fusion_stages, moduli)
           << ".bin";
    return stream.str();
}

template <typename T>
void write_scalar(std::ofstream &stream, T value)
{
    stream.write(reinterpret_cast<const char *>(&value), sizeof(T));
    if (!stream)
    {
        throw std::runtime_error("GpuParameterData failed to write fused NTT matrix cache");
    }
}

template <typename T>
bool read_scalar(std::ifstream &stream, T &value)
{
    stream.read(reinterpret_cast<char *>(&value), sizeof(T));
    return static_cast<bool>(stream);
}

template <typename T>
void write_vector(std::ofstream &stream, const std::vector<T> &values)
{
    if (!values.empty())
    {
        stream.write(
            reinterpret_cast<const char *>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(T)));
        if (!stream)
        {
            throw std::runtime_error(
                "GpuParameterData failed to write fused NTT matrix cache");
        }
    }
}

template <typename T>
bool read_vector(std::ifstream &stream, std::vector<T> &values, std::uint64_t size)
{
    if (size > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
    {
        return false;
    }
    values.resize(static_cast<std::size_t>(size));
    if (!values.empty())
    {
        stream.read(
            reinterpret_cast<char *>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(T)));
    }
    return static_cast<bool>(stream);
}

void save_fused_matrix_tables_to_cache(
    const std::string &path,
    bool inverse,
    std::size_t degree,
    int fusion_stages,
    const std::vector<std::uint64_t> &moduli,
    const FusedNttMatrixTables &tables)
{
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream)
    {
        throw std::runtime_error(
            "GpuParameterData failed to open fused NTT matrix cache for writing: " +
            path);
    }

    write_scalar(stream, kFusedMatrixCacheMagic);
    write_scalar(stream, kFusedMatrixCacheVersion);
    write_scalar(stream, static_cast<std::uint32_t>(inverse ? 1 : 0));
    write_scalar(stream, static_cast<std::uint64_t>(degree));
    write_scalar(stream, static_cast<std::uint64_t>(fusion_stages));
    write_scalar(stream, static_cast<std::uint64_t>(moduli.size()));
    write_scalar(stream, static_cast<std::uint64_t>(tables.stage_counts.size()));
    write_scalar(stream, static_cast<std::uint64_t>(tables.group_counts.size()));
    write_scalar(stream, static_cast<std::uint64_t>(tables.stage_offsets.size()));
    write_scalar(stream, static_cast<std::uint64_t>(tables.matrices.size()));
    write_vector(stream, moduli);
    write_vector(stream, tables.stage_counts);
    write_vector(stream, tables.group_counts);
    write_vector(stream, tables.stage_offsets);
    write_vector(stream, tables.matrices);
}

bool load_fused_matrix_tables_from_cache(
    const std::string &path,
    bool inverse,
    std::size_t degree,
    int fusion_stages,
    const std::vector<std::uint64_t> &moduli,
    FusedNttMatrixTables &tables)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
    {
        return false;
    }

    std::uint64_t magic = 0;
    std::uint32_t version = 0;
    std::uint32_t inverse_value = 0;
    std::uint64_t cached_degree = 0;
    std::uint64_t cached_fusion_stages = 0;
    std::uint64_t limb_count = 0;
    std::uint64_t stage_count_size = 0;
    std::uint64_t group_count_size = 0;
    std::uint64_t stage_offset_size = 0;
    std::uint64_t matrix_size = 0;
    if (!read_scalar(stream, magic) ||
        !read_scalar(stream, version) ||
        !read_scalar(stream, inverse_value) ||
        !read_scalar(stream, cached_degree) ||
        !read_scalar(stream, cached_fusion_stages) ||
        !read_scalar(stream, limb_count) ||
        !read_scalar(stream, stage_count_size) ||
        !read_scalar(stream, group_count_size) ||
        !read_scalar(stream, stage_offset_size) ||
        !read_scalar(stream, matrix_size))
    {
        return false;
    }

    if (magic != kFusedMatrixCacheMagic ||
        version != kFusedMatrixCacheVersion ||
        inverse_value != static_cast<std::uint32_t>(inverse ? 1 : 0) ||
        cached_degree != static_cast<std::uint64_t>(degree) ||
        cached_fusion_stages != static_cast<std::uint64_t>(fusion_stages) ||
        limb_count != static_cast<std::uint64_t>(moduli.size()))
    {
        return false;
    }

    std::vector<std::uint64_t> cached_moduli;
    if (!read_vector(stream, cached_moduli, limb_count) ||
        cached_moduli != moduli)
    {
        return false;
    }

    FusedNttMatrixTables loaded;
    loaded.fusion_stages = static_cast<std::size_t>(fusion_stages);
    if (!read_vector(stream, loaded.stage_counts, stage_count_size) ||
        !read_vector(stream, loaded.group_counts, group_count_size) ||
        !read_vector(stream, loaded.stage_offsets, stage_offset_size) ||
        !read_vector(stream, loaded.matrices, matrix_size))
    {
        return false;
    }

    tables = std::move(loaded);
    return true;
}

int read_precomputed_fused_matrix_stages()
{
    const char *raw = std::getenv("POSEIDON_NTT_FUSED_MATRIX_STAGES");
    if (raw == nullptr || raw[0] == '\0')
    {
        const char *algorithm = std::getenv("POSEIDON_NTT_ALGO");
        if (algorithm != nullptr)
        {
            const std::string value(algorithm);
            if (value == "tensor" || value == "tensor_core" ||
                value == "tam_tensor" || value == "matrix" ||
                value == "tensor_fp64" || value == "fp64" ||
                value == "tam_fp64" || value == "neo" ||
                value == "tensor_neo_fp64")
            {
                return kMaxPrecomputedFusedMatrixStages;
            }
        }
        return kDefaultPrecomputedFusedMatrixStages;
    }

    char *end = nullptr;
    errno = 0;
    const long value = std::strtol(raw, &end, 10);
    if (errno != 0 || end == raw || *end != '\0')
    {
        return kDefaultPrecomputedFusedMatrixStages;
    }
    if (value <= 0)
    {
        return 0;
    }
    if (value > kMaxPrecomputedFusedMatrixStages)
    {
        return kMaxPrecomputedFusedMatrixStages;
    }
    return static_cast<int>(value);
}

std::size_t log2_power_of_two(std::size_t value)
{
    if (value == 0 || (value & (value - 1)) != 0)
    {
        throw std::invalid_argument(
            "GpuParameterData fused NTT matrices require a power-of-two degree");
    }

    std::size_t result = 0;
    while (value > 1)
    {
        value >>= 1;
        ++result;
    }
    return result;
}

std::vector<const util::NTTTables *> collect_rns_ntt_table_pointers(
    const util::NTTTables *q_tables,
    std::size_t q_count,
    const util::NTTTables *p_tables,
    std::size_t p_count)
{
    if ((q_count != 0 && q_tables == nullptr) ||
        (p_count != 0 && p_tables == nullptr))
    {
        throw std::invalid_argument("GpuParameterData requires CPU NTT tables");
    }

    std::vector<const util::NTTTables *> result;
    result.reserve(q_count + p_count);
    for (std::size_t i = 0; i < q_count; ++i)
    {
        result.push_back(q_tables + i);
    }
    for (std::size_t i = 0; i < p_count; ++i)
    {
        result.push_back(p_tables + i);
    }
    return result;
}

constexpr std::size_t kZeroExponent =
    std::numeric_limits<std::size_t>::max();

bool is_zero_exponent(std::size_t exponent)
{
    return exponent == kZeroExponent;
}

std::size_t root_order(std::size_t degree)
{
    return degree << 1;
}

std::size_t normalize_exponent(std::size_t exponent, std::size_t degree)
{
    return exponent % root_order(degree);
}

std::size_t negate_exponent(std::size_t exponent, std::size_t degree)
{
    return is_zero_exponent(exponent)
        ? kZeroExponent
        : normalize_exponent(exponent + degree, degree);
}

std::size_t multiply_exponents(
    std::size_t left,
    std::size_t right,
    std::size_t degree)
{
    return is_zero_exponent(left)
        ? kZeroExponent
        : normalize_exponent(left + right, degree);
}

std::size_t merge_disjoint_terms(
    std::size_t left,
    std::size_t right,
    const char *what)
{
    if (is_zero_exponent(left))
    {
        return right;
    }
    if (is_zero_exponent(right))
    {
        return left;
    }
    throw std::logic_error(what);
}

std::size_t forward_root_index_to_exponent(
    std::size_t root_index,
    int log_degree)
{
    if (root_index == 0)
    {
        return 0;
    }
    return util::reverse_bits(root_index, log_degree);
}

std::size_t inverse_root_index_to_exponent(
    std::size_t root_index,
    std::size_t degree,
    int log_degree)
{
    if (root_index == 0)
    {
        return 0;
    }

    const std::size_t inverse_power =
        util::reverse_bits(root_index - 1, log_degree) + 1;
    return normalize_exponent(root_order(degree) - inverse_power, degree);
}

void apply_forward_exponent_butterfly(
    std::vector<std::size_t> &matrix,
    std::size_t matrix_size,
    std::size_t x_row,
    std::size_t y_row,
    std::size_t root_exponent,
    std::size_t degree)
{
    for (std::size_t column = 0; column < matrix_size; ++column)
    {
        const auto x = matrix[x_row * matrix_size + column];
        const auto y = matrix[y_row * matrix_size + column];
        const auto root_y =
            multiply_exponents(y, root_exponent, degree);
        const auto neg_root_y = negate_exponent(root_y, degree);

        matrix[x_row * matrix_size + column] =
            merge_disjoint_terms(
                x,
                root_y,
                "Forward fused NTT exponent matrix needs a non-monomial sum");
        matrix[y_row * matrix_size + column] =
            merge_disjoint_terms(
                x,
                neg_root_y,
                "Forward fused NTT exponent matrix needs a non-monomial sum");
    }
}

void apply_inverse_exponent_butterfly(
    std::vector<std::size_t> &matrix,
    std::size_t matrix_size,
    std::size_t x_row,
    std::size_t y_row,
    std::size_t root_exponent,
    std::size_t degree)
{
    for (std::size_t column = 0; column < matrix_size; ++column)
    {
        const auto x = matrix[x_row * matrix_size + column];
        const auto y = matrix[y_row * matrix_size + column];
        const auto root_x =
            multiply_exponents(x, root_exponent, degree);
        const auto root_y =
            multiply_exponents(y, root_exponent, degree);

        matrix[x_row * matrix_size + column] =
            merge_disjoint_terms(
                x,
                y,
                "Inverse fused NTT exponent matrix needs a non-monomial sum");
        matrix[y_row * matrix_size + column] =
            merge_disjoint_terms(
                root_x,
                negate_exponent(root_y, degree),
                "Inverse fused NTT exponent matrix needs a non-monomial sum");
    }
}

std::vector<std::size_t> identity_exponent_matrix(std::size_t matrix_size)
{
    std::vector<std::size_t> matrix(
        matrix_size * matrix_size,
        kZeroExponent);
    for (std::size_t row = 0; row < matrix_size; ++row)
    {
        matrix[row * matrix_size + row] = 0;
    }
    return matrix;
}

void append_forward_fused_exponent_matrix(
    std::vector<std::size_t> &destination,
    std::size_t degree,
    int log_degree,
    std::size_t m,
    std::size_t outer_group,
    int stage_count)
{
    const std::size_t matrix_size =
        static_cast<std::size_t>(1) << stage_count;
    auto matrix = identity_exponent_matrix(matrix_size);

    for (int stage = 0; stage < stage_count; ++stage)
    {
        const std::size_t local_stride =
            static_cast<std::size_t>(1) << (stage_count - 1 - stage);
        const std::size_t stage_m = m << stage;
        const std::size_t stage_group_base = outer_group << stage;

        for (std::size_t block = 0; block < matrix_size;
             block += (local_stride << 1))
        {
            const std::size_t block_group =
                block / (local_stride << 1);
            const std::size_t root_index =
                stage_m + stage_group_base + block_group;
            if (root_index >= degree)
            {
                throw std::out_of_range(
                    "GpuParameterData forward fused NTT root index out of range");
            }

            const std::size_t root_exponent =
                forward_root_index_to_exponent(root_index, log_degree);
            for (std::size_t offset = 0; offset < local_stride; ++offset)
            {
                apply_forward_exponent_butterfly(
                    matrix,
                    matrix_size,
                    block + offset,
                    block + offset + local_stride,
                    root_exponent,
                    degree);
            }
        }
    }

    destination.insert(destination.end(), matrix.begin(), matrix.end());
}

void append_inverse_fused_exponent_matrix(
    std::vector<std::size_t> &destination,
    std::size_t degree,
    int log_degree,
    std::size_t m,
    std::size_t outer_group,
    int stage_count)
{
    const std::size_t matrix_size =
        static_cast<std::size_t>(1) << stage_count;
    auto matrix = identity_exponent_matrix(matrix_size);

    for (int stage = 0; stage < stage_count; ++stage)
    {
        const std::size_t local_stride =
            static_cast<std::size_t>(1) << stage;
        const std::size_t stage_m = m >> stage;
        const std::size_t stage_root_base =
            degree - (stage_m << 1) + 1;
        const std::size_t stage_group_base =
            outer_group << (stage_count - 1 - stage);

        for (std::size_t block = 0; block < matrix_size;
             block += (local_stride << 1))
        {
            const std::size_t block_group =
                block / (local_stride << 1);
            const std::size_t root_index =
                stage_root_base + stage_group_base + block_group;
            if (root_index >= degree)
            {
                throw std::out_of_range(
                    "GpuParameterData inverse fused NTT root index out of range");
            }

            const std::size_t root_exponent =
                inverse_root_index_to_exponent(
                    root_index,
                    degree,
                    log_degree);
            for (std::size_t offset = 0; offset < local_stride; ++offset)
            {
                apply_inverse_exponent_butterfly(
                    matrix,
                    matrix_size,
                    block + offset,
                    block + offset + local_stride,
                    root_exponent,
                    degree);
            }
        }
    }

    destination.insert(destination.end(), matrix.begin(), matrix.end());
}

std::vector<GpuWord> build_root_power_lookup(
    const util::NTTTables &table,
    std::size_t degree,
    int log_degree)
{
    const auto &modulus = table.modulus();
    const auto *roots = table.get_from_root_powers();
    std::vector<GpuWord> lookup(root_order(degree), 0);

    lookup[0] = checked_gpu_word(
        static_cast<std::uint64_t>(1) % modulus.value(),
        "GpuParameterData fused NTT matrix identity value does not fit GpuWord");
    for (std::size_t exponent = 1; exponent < degree; ++exponent)
    {
        const std::size_t root_index =
            util::reverse_bits(exponent, log_degree);
        lookup[exponent] = checked_gpu_word(
            roots[root_index].operand,
            "GpuParameterData fused NTT root value does not fit GpuWord");
    }

    for (std::size_t exponent = degree; exponent < root_order(degree);
         ++exponent)
    {
        const GpuWord positive = lookup[exponent - degree];
        lookup[exponent] = positive == 0
            ? 0
            : checked_gpu_word(
                  modulus.value() - positive,
                  "GpuParameterData fused NTT negative root does not fit GpuWord");
    }
    return lookup;
}

std::vector<std::vector<GpuWord>> build_root_power_lookups(
    const std::vector<const util::NTTTables *> &tables,
    std::size_t degree,
    int log_degree)
{
    std::vector<std::vector<GpuWord>> lookups;
    lookups.reserve(tables.size());
    for (const auto *table : tables)
    {
        lookups.push_back(build_root_power_lookup(*table, degree, log_degree));
    }
    return lookups;
}

void append_materialized_exponent_matrix(
    std::vector<GpuWord> &destination,
    const std::size_t *exponents,
    std::size_t matrix_elements,
    const std::vector<GpuWord> &root_lookup)
{
    for (std::size_t i = 0; i < matrix_elements; ++i)
    {
        destination.push_back(
            is_zero_exponent(exponents[i]) ? 0 : root_lookup[exponents[i]]);
    }
}

void validate_fused_matrix_ntt_tables(
    const std::vector<const util::NTTTables *> &tables,
    std::size_t degree)
{
    for (const auto *table : tables)
    {
        if (table == nullptr)
        {
            throw std::invalid_argument("GpuParameterData requires CPU NTT tables");
        }
        if (table->coeff_count() != degree)
        {
            throw std::invalid_argument(
                "GpuParameterData fused NTT table degree mismatch");
        }
    }
}

void append_forward_fused_matrix_stage(
    FusedNttMatrixTables &result,
    const std::vector<const util::NTTTables *> &tables,
    const std::vector<std::vector<GpuWord>> &root_lookups,
    std::size_t degree,
    int log_degree,
    std::size_t m,
    int stage_count)
{
    const std::size_t group_count = m;
    const std::size_t matrix_size =
        static_cast<std::size_t>(1) << stage_count;
    const std::size_t matrix_elements = matrix_size * matrix_size;
    result.stage_counts.push_back(static_cast<GpuWord>(stage_count));
    result.group_counts.push_back(static_cast<GpuWide>(group_count));
    result.matrices.reserve(
        result.matrices.size() +
        tables.size() * group_count * matrix_elements);

    std::vector<std::size_t> exponent_matrices;
    exponent_matrices.reserve(group_count * matrix_elements);
    const std::string exponent_label =
        "  forward TAM exponent stage=" + std::to_string(stage_count);
    for (std::size_t outer_group = 0; outer_group < group_count;
         ++outer_group)
    {
        append_forward_fused_exponent_matrix(
            exponent_matrices,
            degree,
            log_degree,
            m,
            outer_group,
            stage_count);
        if (should_update_progress(outer_group + 1, group_count))
        {
            print_progress_bar(exponent_label, outer_group + 1, group_count);
        }
    }

    const std::size_t materialize_total = tables.size() * group_count;
    std::size_t materialize_done = 0;
    const std::string materialize_label =
        "  forward TAM materialize stage=" + std::to_string(stage_count);
    for (std::size_t table_index = 0; table_index < tables.size();
         ++table_index)
    {
        const auto &root_lookup = root_lookups[table_index];
        for (std::size_t outer_group = 0; outer_group < group_count;
             ++outer_group)
        {
            append_materialized_exponent_matrix(
                result.matrices,
                exponent_matrices.data() + outer_group * matrix_elements,
                matrix_elements,
                root_lookup);
            ++materialize_done;
            if (should_update_progress(materialize_done, materialize_total))
            {
                print_progress_bar(
                    materialize_label,
                    materialize_done,
                    materialize_total);
            }
        }
    }
    result.stage_offsets.push_back(static_cast<GpuWide>(result.matrices.size()));
}

void append_inverse_fused_matrix_stage(
    FusedNttMatrixTables &result,
    const std::vector<const util::NTTTables *> &tables,
    const std::vector<std::vector<GpuWord>> &root_lookups,
    std::size_t degree,
    int log_degree,
    std::size_t m,
    std::size_t gap,
    int stage_count)
{
    const std::size_t group_count = (degree >> stage_count) / gap;
    const std::size_t matrix_size =
        static_cast<std::size_t>(1) << stage_count;
    const std::size_t matrix_elements = matrix_size * matrix_size;
    result.stage_counts.push_back(static_cast<GpuWord>(stage_count));
    result.group_counts.push_back(static_cast<GpuWide>(group_count));
    result.matrices.reserve(
        result.matrices.size() +
        tables.size() * group_count * matrix_elements);

    std::vector<std::size_t> exponent_matrices;
    exponent_matrices.reserve(group_count * matrix_elements);
    const std::string exponent_label =
        "  inverse TAM exponent stage=" + std::to_string(stage_count);
    for (std::size_t outer_group = 0; outer_group < group_count;
         ++outer_group)
    {
        append_inverse_fused_exponent_matrix(
            exponent_matrices,
            degree,
            log_degree,
            m,
            outer_group,
            stage_count);
        if (should_update_progress(outer_group + 1, group_count))
        {
            print_progress_bar(exponent_label, outer_group + 1, group_count);
        }
    }

    const std::size_t materialize_total = tables.size() * group_count;
    std::size_t materialize_done = 0;
    const std::string materialize_label =
        "  inverse TAM materialize stage=" + std::to_string(stage_count);
    for (std::size_t table_index = 0; table_index < tables.size();
         ++table_index)
    {
        const auto &root_lookup = root_lookups[table_index];
        for (std::size_t outer_group = 0; outer_group < group_count;
             ++outer_group)
        {
            append_materialized_exponent_matrix(
                result.matrices,
                exponent_matrices.data() + outer_group * matrix_elements,
                matrix_elements,
                root_lookup);
            ++materialize_done;
            if (should_update_progress(materialize_done, materialize_total))
            {
                print_progress_bar(
                    materialize_label,
                    materialize_done,
                    materialize_total);
            }
        }
    }
    result.stage_offsets.push_back(static_cast<GpuWide>(result.matrices.size()));
}

FusedNttMatrixTables build_forward_fused_matrix_tables(
    const std::vector<const util::NTTTables *> &tables,
    std::size_t degree,
    int fusion_stages)
{
    if (fusion_stages <= 0)
    {
        return {};
    }
    if (fusion_stages > kMaxPrecomputedFusedMatrixStages)
    {
        throw std::invalid_argument(
            "GpuParameterData fused NTT matrix stage count is too large");
    }

    validate_fused_matrix_ntt_tables(tables, degree);
    const std::size_t log_degree_size = log2_power_of_two(degree);
    const int log_degree =
        static_cast<int>(log_degree_size);
    const auto root_lookups =
        build_root_power_lookups(tables, degree, log_degree);

    FusedNttMatrixTables result;
    result.fusion_stages = static_cast<std::size_t>(fusion_stages);
    result.stage_offsets.push_back(0);

    std::size_t stages_done = 0;
    std::size_t m = 1;
    while (stages_done < log_degree_size)
    {
        const std::size_t remaining = log_degree_size - stages_done;
        int stage_count = 1;
        if (fusion_stages > 1)
        {
            stage_count = static_cast<int>(remaining % fusion_stages);
            if (stage_count == 0)
            {
                stage_count = fusion_stages;
            }
        }

        append_forward_fused_matrix_stage(
            result,
            tables,
            root_lookups,
            degree,
            log_degree,
            m,
            stage_count);

        stages_done += static_cast<std::size_t>(stage_count);
        m <<= stage_count;
    }

    return result;
}

FusedNttMatrixTables build_inverse_fused_matrix_tables(
    const std::vector<const util::NTTTables *> &tables,
    std::size_t degree,
    int fusion_stages)
{
    if (fusion_stages <= 0)
    {
        return {};
    }
    if (fusion_stages > kMaxPrecomputedFusedMatrixStages)
    {
        throw std::invalid_argument(
            "GpuParameterData fused INTT matrix stage count is too large");
    }

    validate_fused_matrix_ntt_tables(tables, degree);
    const std::size_t log_degree_size = log2_power_of_two(degree);
    const int log_degree =
        static_cast<int>(log_degree_size);
    const auto root_lookups =
        build_root_power_lookups(tables, degree, log_degree);

    FusedNttMatrixTables result;
    result.fusion_stages = static_cast<std::size_t>(fusion_stages);
    result.stage_offsets.push_back(0);

    std::size_t stages_done = 0;
    std::size_t m = degree >> 1;
    std::size_t gap = 1;
    while (stages_done < log_degree_size)
    {
        const std::size_t remaining = log_degree_size - stages_done;
        const int stage_count = fusion_stages > 1
            ? static_cast<int>(
                  std::min<std::size_t>(remaining, fusion_stages))
            : 1;

        append_inverse_fused_matrix_stage(
            result,
            tables,
            root_lookups,
            degree,
            log_degree,
            m,
            gap,
            stage_count);

        stages_done += static_cast<std::size_t>(stage_count);
        m >>= stage_count;
        gap <<= stage_count;
    }

    return result;
}

FusedNttMatrixTables load_or_build_fused_matrix_tables(
    const std::vector<const util::NTTTables *> &tables,
    std::size_t degree,
    int fusion_stages,
    bool inverse)
{
    if (fusion_stages <= 0)
    {
        return {};
    }

    const auto moduli = collect_table_moduli(tables);
    const auto cache_dir = read_fused_matrix_cache_dir();
    std::string cache_path;
    if (!cache_dir.empty())
    {
        ensure_directory_exists(cache_dir);
        cache_path = fused_matrix_cache_path(
            cache_dir,
            inverse,
            degree,
            fusion_stages,
            moduli);

        FusedNttMatrixTables cached;
        if (load_fused_matrix_tables_from_cache(
                cache_path,
                inverse,
                degree,
                fusion_stages,
                moduli,
                cached))
        {
            print_fused_matrix_message(
                std::string("[TAM cache] loaded ") +
                (inverse ? "INTT" : "NTT") + " matrices: " + cache_path);
            return cached;
        }

        print_fused_matrix_message(
            std::string("[TAM cache] miss ") +
            (inverse ? "INTT" : "NTT") + " matrices: " + cache_path);
    }

    print_fused_matrix_message(
        std::string("[TAM build] generating ") +
        (inverse ? "INTT" : "NTT") +
        " matrices, degree=" + std::to_string(degree) +
        ", fusion=" + std::to_string(fusion_stages) +
        ", limbs=" + std::to_string(moduli.size()));

    auto result = inverse
        ? build_inverse_fused_matrix_tables(tables, degree, fusion_stages)
        : build_forward_fused_matrix_tables(tables, degree, fusion_stages);

    if (!cache_path.empty())
    {
        save_fused_matrix_tables_to_cache(
            cache_path,
            inverse,
            degree,
            fusion_stages,
            moduli,
            result);
        print_fused_matrix_message(
            std::string("[TAM cache] saved ") +
            (inverse ? "INTT" : "NTT") + " matrices: " + cache_path);
    }

    return result;
}

std::vector<GpuWord> copy_inv_q_last_mod_q_operands(
    const util::RNSTool *rns_tool,
    std::size_t q_count)
{
    if (q_count < 2)
    {
        return {};
    }
    if (rns_tool == nullptr || rns_tool->inv_q_last_mod_q() == nullptr)
    {
        throw std::invalid_argument("GpuParameterData requires RNSTool rescale constants");
    }

    std::vector<GpuWord> result(q_count - 1);
    const auto *inv_q_last_mod_q = rns_tool->inv_q_last_mod_q();
    for (std::size_t i = 0; i < q_count - 1; ++i)
    {
        result[i] = checked_gpu_word(
            inv_q_last_mod_q[i].operand,
            "GpuParameterData only supports inv_q_last_mod_q constants that fit in GpuWord");
    }
    return result;
}

std::vector<GpuWord> compute_half_q_last_mod_q(
    const std::vector<Modulus> &q,
    GpuWord half_q_last)
{
    if (q.size() < 2)
    {
        return {};
    }

    std::vector<GpuWord> result(q.size() - 1);
    for (std::size_t i = 0; i < q.size() - 1; ++i)
    {
        result[i] = checked_gpu_word(
            static_cast<std::uint64_t>(half_q_last) % q[i].value(),
            "GpuParameterData only supports half_q_last_mod_q constants that fit in GpuWord");
    }
    return result;
}

struct RescaleX2Constants
{
    GpuWord q_second_last = 0;
    GpuWide product = 0;
    GpuWide half_product = 0;
    GpuWord inv_q_last_mod_q_second_last = 0;
    std::vector<GpuWord> product_mod_q;
    std::vector<GpuWord> inv_product_mod_q;
};

RescaleX2Constants compute_rescale_x2_constants(
    const std::vector<Modulus> &q)
{
    RescaleX2Constants result;
    if (q.size() < 3)
    {
        return result;
    }

    const std::uint64_t q_second_last = q[q.size() - 2].value();
    const std::uint64_t q_last = q.back().value();
    result.q_second_last = checked_gpu_word(
        q_second_last,
        "GpuParameterData rescale_x2 q_second_last does not fit GpuWord");
    result.product =
        static_cast<GpuWide>(q_second_last) * static_cast<GpuWide>(q_last);
    result.half_product = result.product >> 1;

    std::uint64_t inverse = 0;
    if (!util::try_invert_uint_mod(
            q_last % q_second_last,
            q[q.size() - 2],
            inverse))
    {
        throw std::invalid_argument(
            "GpuParameterData cannot invert q_last modulo q_second_last");
    }
    result.inv_q_last_mod_q_second_last = checked_gpu_word(
        inverse,
        "GpuParameterData rescale_x2 CRT inverse does not fit GpuWord");

    const std::size_t retained_count = q.size() - 2;
    result.product_mod_q.resize(retained_count);
    result.inv_product_mod_q.resize(retained_count);
    for (std::size_t i = 0; i < retained_count; ++i)
    {
        const std::uint64_t qi = q[i].value();
        const std::uint64_t product_mod_q =
            static_cast<std::uint64_t>(
                static_cast<unsigned __int128>(result.product) % qi);
        result.product_mod_q[i] = checked_gpu_word(
            product_mod_q,
            "GpuParameterData rescale_x2 product residue does not fit GpuWord");

        inverse = 0;
        if (!util::try_invert_uint_mod(product_mod_q, q[i], inverse))
        {
            throw std::invalid_argument(
                "GpuParameterData cannot invert rescale_x2 product");
        }
        result.inv_product_mod_q[i] = checked_gpu_word(
            inverse,
            "GpuParameterData rescale_x2 product inverse does not fit GpuWord");
    }
    return result;
}

std::vector<GpuWord> compute_master_rescale_q_inv_mod_q(
    const std::vector<Modulus> &q,
    std::size_t count)
{
    std::vector<GpuWord> result(count * count, 0);
    for (std::size_t i = 0; i < count; ++i)
    {
        for (std::size_t j = 0; j < count; ++j)
        {
            if (i == j)
            {
                continue;
            }
            std::uint64_t inverse = 0;
            if (!util::try_invert_uint_mod(
                    q[i].value() % q[j].value(),
                    q[j],
                    inverse))
            {
                throw std::invalid_argument(
                    "GpuParameterData cannot invert one q prime modulo another");
            }
            result[i * count + j] = checked_gpu_word(
                inverse,
                "GpuParameterData pairwise q inverse does not fit GpuWord");
        }
    }
    return result;
}

std::vector<GpuWord> slice_rescale_q_inv_mod_q(
    const std::vector<GpuWord> &master,
    std::size_t master_q_count,
    std::size_t q_count)
{
    std::vector<GpuWord> result(q_count * q_count, 0);
    for (std::size_t i = 0; i < q_count; ++i)
    {
        std::memcpy(
            result.data() + i * q_count,
            master.data() + i * master_q_count,
            q_count * sizeof(GpuWord));
    }
    return result;
}

std::vector<GpuWord> copy_mod_operand_array(
    const util::MultiplyUIntModOperand *operands,
    std::size_t count,
    const char *what)
{
    std::vector<GpuWord> result(count);
    for (std::size_t i = 0; i < count; ++i)
    {
        result[i] = checked_gpu_word(operands[i].operand, what);
    }
    return result;
}

void append_base_converter_matrix_padded(
    const util::BaseConverter &converter,
    std::size_t padded_input_size,
    std::vector<GpuWord> &destination,
    const char *what)
{
    for (std::size_t row = 0; row < converter.obase_size(); ++row)
    {
        const auto *row_values = converter.base_change_matrix_row(row);
        for (std::size_t col = 0; col < padded_input_size; ++col)
        {
            const std::uint64_t value =
                col < converter.ibase_size() ? row_values[col] : 0;
            destination.push_back(checked_gpu_word(value, what));
        }
    }
}

struct HybridKeySwitchHostTables
{
    std::size_t base_q_count = 0;
    std::size_t base_p_count = 0;
    std::size_t decomp_count = 0;
    std::vector<GpuWord> decomp_start;
    std::vector<GpuWord> decomp_end;
    std::vector<GpuWord> p_mod_q;
    std::vector<GpuWord> inv_p_mod_q;
    std::vector<GpuWord> q_matrix_offsets;
    std::vector<GpuWord> p_matrix_offsets;
    std::vector<GpuWord> q_matrices;
    std::vector<GpuWord> p_matrices;
    std::vector<GpuWord> moddown_p_to_q_matrix;
    std::vector<GpuWord> qi_inv_punctured;
    std::vector<GpuWord> p_inv_punctured;
};

HybridKeySwitchHostTables prepare_hybrid_key_switch_tables(
    const util::RNSToolQP *rns_qp)
{
    HybridKeySwitchHostTables out;
    if (rns_qp == nullptr || rns_qp->base_q() == nullptr || rns_qp->base_p() == nullptr)
    {
        return out;
    }

    const std::size_t base_q_size = rns_qp->base_q()->size();
    const std::size_t base_p_size = rns_qp->base_p()->size();
    const std::size_t decomp_count = rns_qp->decomp_count();
    if (base_q_size == 0 || base_p_size == 0 || decomp_count == 0)
    {
        return out;
    }
    if (base_q_size > std::numeric_limits<GpuWord>::max() ||
        base_p_size > std::numeric_limits<GpuWord>::max() ||
        decomp_count > std::numeric_limits<GpuWord>::max())
    {
        throw std::invalid_argument("GpuParameterData HYBRID table shape does not fit GpuWord");
    }

    out.base_q_count = base_q_size;
    out.base_p_count = base_p_size;
    out.decomp_count = decomp_count;
    out.decomp_start.resize(decomp_count);
    out.decomp_end.resize(decomp_count);
    out.q_matrix_offsets.resize(decomp_count + 1);
    out.p_matrix_offsets.resize(decomp_count + 1);
    out.qi_inv_punctured.assign(decomp_count * base_p_size, 0);

    for (std::size_t decomp_index = 0; decomp_index < decomp_count; ++decomp_index)
    {
        const auto &decomp = rns_qp->decomp(decomp_index);
        const std::size_t start = decomp.base_start_idx();
        const std::size_t end = decomp.base_last_idx();
        out.decomp_start[decomp_index] = checked_gpu_word(
            start,
            "GpuParameterData HYBRID decomp start does not fit GpuWord");
        out.decomp_end[decomp_index] = checked_gpu_word(
            end,
            "GpuParameterData HYBRID decomp end does not fit GpuWord");
        out.q_matrix_offsets[decomp_index] = checked_gpu_word(
            out.q_matrices.size(),
            "GpuParameterData HYBRID q matrix offset does not fit GpuWord");
        out.p_matrix_offsets[decomp_index] = checked_gpu_word(
            out.p_matrices.size(),
            "GpuParameterData HYBRID p matrix offset does not fit GpuWord");

        if (!decomp.has_base_conversion())
        {
            out.qi_inv_punctured[decomp_index * base_p_size] = 1;
            out.q_matrices.resize(out.q_matrices.size() + base_q_size * base_p_size, 0);
            out.p_matrices.resize(out.p_matrices.size() + base_p_size * base_p_size, 0);
            continue;
        }

        const auto &decomp_ibase = decomp.obase_p_conv()->ibase();
        const auto *decomp_inv_punctured =
            decomp_ibase.inv_punctured_prod_mod_base_array();
        for (std::size_t col = 0; col < decomp_ibase.size(); ++col)
        {
            out.qi_inv_punctured[decomp_index * base_p_size + col] = checked_gpu_word(
                decomp_inv_punctured[col].operand,
                "GpuParameterData HYBRID Q_i inv punctured constant does not fit GpuWord");
        }

        const auto &q_conv_map = decomp.single_obase_q_conv_map();
        for (std::size_t q_limb = 0; q_limb < base_q_size; ++q_limb)
        {
            const bool limb_in_decomp = q_limb >= start && q_limb < end;
            if (limb_in_decomp)
            {
                for (std::size_t col = 0; col < base_p_size; ++col)
                {
                    out.q_matrices.push_back(0);
                }
                continue;
            }

            const auto found = q_conv_map.find(q_limb);
            if (found == q_conv_map.end() || found->second.get() == nullptr)
            {
                throw std::invalid_argument("GpuParameterData missing HYBRID q conversion row");
            }
            append_base_converter_matrix_padded(
                *found->second,
                base_p_size,
                out.q_matrices,
                "GpuParameterData HYBRID q conversion constant does not fit GpuWord");
        }

        append_base_converter_matrix_padded(
            *decomp.obase_p_conv(),
            base_p_size,
            out.p_matrices,
            "GpuParameterData HYBRID p conversion constant does not fit GpuWord");
    }

    out.q_matrix_offsets[decomp_count] = checked_gpu_word(
        out.q_matrices.size(),
        "GpuParameterData HYBRID q matrix final offset does not fit GpuWord");
    out.p_matrix_offsets[decomp_count] = checked_gpu_word(
        out.p_matrices.size(),
        "GpuParameterData HYBRID p matrix final offset does not fit GpuWord");

    out.p_mod_q = copy_mod_operand_array(
        rns_qp->p_mod_qi(),
        base_q_size,
        "GpuParameterData HYBRID p_mod_q constant does not fit GpuWord");
    out.inv_p_mod_q = copy_mod_operand_array(
        rns_qp->p_inv_mod_qi(),
        base_q_size,
        "GpuParameterData HYBRID inv_p_mod_q constant does not fit GpuWord");
    out.p_inv_punctured = copy_mod_operand_array(
        rns_qp->base_p()->inv_punctured_prod_mod_base_array(),
        base_p_size,
        "GpuParameterData HYBRID P inv punctured constant does not fit GpuWord");
    append_base_converter_matrix_padded(
        rns_qp->base_p_to_q_conv(),
        base_p_size,
        out.moddown_p_to_q_matrix,
        "GpuParameterData HYBRID moddown P-to-Q conversion constant does not fit GpuWord");
    return out;
}

void upload_hybrid_key_switch_tables_async(
    const HybridKeySwitchHostTables &host,
    GpuParameterShard &shard,
    int device_id,
    cudaStream_t stream)
{
    if (host.base_q_count == 0 || host.base_p_count == 0 || host.decomp_count == 0)
    {
        return;
    }

    shard.hybrid_base_q_count = host.base_q_count;
    shard.hybrid_base_p_count = host.base_p_count;
    shard.hybrid_decomp_count = host.decomp_count;
    shard.hybrid_decomp_start =
        copy_to_device_vector_async(host.decomp_start, device_id, stream);
    shard.hybrid_decomp_end =
        copy_to_device_vector_async(host.decomp_end, device_id, stream);
    shard.hybrid_p_mod_q =
        copy_to_device_vector_async(host.p_mod_q, device_id, stream);
    shard.hybrid_inv_p_mod_q =
        copy_to_device_vector_async(host.inv_p_mod_q, device_id, stream);
    shard.hybrid_qi_inv_punctured =
        copy_to_device_vector_async(host.qi_inv_punctured, device_id, stream);
    shard.hybrid_p_inv_punctured =
        copy_to_device_vector_async(host.p_inv_punctured, device_id, stream);
    shard.hybrid_q_conv_matrix_offsets =
        copy_to_device_vector_async(host.q_matrix_offsets, device_id, stream);
    shard.hybrid_p_conv_matrix_offsets =
        copy_to_device_vector_async(host.p_matrix_offsets, device_id, stream);
    shard.hybrid_q_conv_matrices =
        copy_to_device_vector_async(host.q_matrices, device_id, stream);
    shard.hybrid_p_conv_matrices =
        copy_to_device_vector_async(host.p_matrices, device_id, stream);
    shard.hybrid_moddown_p_to_q_matrix =
        copy_to_device_vector_async(host.moddown_p_to_q_matrix, device_id, stream);
}

struct BootstrapRaiseHostTables
{
    std::size_t source_q_count = 0;
    std::size_t target_q_count = 0;
    std::vector<GpuWord> inv_punctured;
    std::vector<GpuWord> matrix;
};

BootstrapRaiseHostTables prepare_bootstrap_raise_tables(
    const std::vector<Modulus> &source_q,
    const std::vector<Modulus> &target_q,
    const util::RNSTool *rns_tool)
{
    BootstrapRaiseHostTables out;
    const std::size_t source_q_count = source_q.size();
    const std::size_t target_q_count = target_q.size();
    if (source_q_count == 0 ||
        target_q_count == 0 ||
        source_q_count > target_q_count)
    {
        return out;
    }
    if (source_q_count > std::numeric_limits<GpuWord>::max() ||
        target_q_count > std::numeric_limits<GpuWord>::max())
    {
        throw std::invalid_argument(
            "GpuParameterData bootstrap ModRaise table shape does not fit GpuWord");
    }

    out.source_q_count = source_q_count;
    out.target_q_count = target_q_count;

    if (source_q_count == target_q_count)
    {
        return out;
    }

    std::vector<Modulus> raise_q;
    raise_q.insert(
        raise_q.end(),
        target_q.begin() + source_q_count,
        target_q.end());

    auto pool = MemoryManager::GetPool(mm_prof_opt::mm_force_thread_local);
    const util::RNSBase *reusable_source_base = nullptr;
    if (rns_tool != nullptr &&
        rns_tool->base_q() != nullptr &&
        rns_tool->base_q()->size() == source_q_count)
    {
        reusable_source_base = rns_tool->base_q();
    }

    std::unique_ptr<util::RNSBase> owned_source_base;
    if (reusable_source_base == nullptr)
    {
        owned_source_base = std::make_unique<util::RNSBase>(source_q, pool);
        reusable_source_base = owned_source_base.get();
    }

    util::RNSBase raise_base(raise_q, pool);
    util::BaseConverter converter(*reusable_source_base, raise_base, pool);

    out.inv_punctured = copy_mod_operand_array(
        reusable_source_base->inv_punctured_prod_mod_base_array(),
        reusable_source_base->size(),
        "GpuParameterData bootstrap ModRaise inv punctured constant does not fit GpuWord");

    append_base_converter_matrix_padded(
        converter,
        source_q_count,
        out.matrix,
        "GpuParameterData bootstrap ModRaise conversion constant does not fit GpuWord");
    return out;
}

void upload_bootstrap_raise_tables_async(
    const BootstrapRaiseHostTables &host,
    GpuParameterShard &shard,
    int device_id,
    cudaStream_t stream)
{
    if (host.source_q_count == 0 || host.target_q_count == 0)
    {
        return;
    }

    shard.bootstrap_raise_source_q_count = host.source_q_count;
    shard.bootstrap_raise_target_q_count = host.target_q_count;
    if (host.source_q_count == host.target_q_count)
    {
        return;
    }

    shard.bootstrap_raise_inv_punctured =
        copy_to_device_vector_async(host.inv_punctured, device_id, stream);
    shard.bootstrap_raise_matrix =
        copy_to_device_vector_async(host.matrix, device_id, stream);
}

struct SelectedContextLevel
{
    std::shared_ptr<const CrtContext::ContextData> context_data;
    std::size_t level_index = 0;
    std::vector<Modulus> parameter_p;
};

struct HostLevelPrep
{
    parms_id_type parms_id{};
    std::size_t degree = 0;
    std::size_t q_count = 0;
    std::size_t p_count = 0;
    std::size_t parameter_p_count = 0;
    GpuWord q_last = 0;
    GpuWord half_q_last = 0;
    GpuWord q_second_last = 0;
    GpuWide q_last_two_product = 0;
    GpuWide half_q_last_two_product = 0;
    GpuWord inv_q_last_mod_q_second_last = 0;
    GpuWide q_second_last_modulus_constant = 0;

    std::vector<GpuWord> q_words;
    std::vector<GpuWord> p_words;
    std::vector<GpuWord> rns_words;
    std::vector<GpuWide> q_barrett_ratios;
    std::vector<GpuWide> p_barrett_ratios;
    std::vector<GpuWide> rns_barrett_ratios;
    std::vector<GpuWord> rns_inv_degree;
    std::vector<GpuWord> rns_montgomery_inv_degree;
    std::vector<GpuWord> rns_montgomery_neg_inv;
    std::vector<GpuWord> half_q_last_mod_q;
    std::vector<GpuWord> inv_q_last_mod_q;
    std::vector<GpuWord> q_last_two_product_mod_q;
    std::vector<GpuWord> inv_q_last_two_product_mod_q;
    std::vector<GpuWord> rescale_q_inv_mod_q;

    HybridKeySwitchHostTables hybrid;
    BootstrapRaiseHostTables bootstrap_raise;

    FusedNttMatrixTables fused_ntt;
    FusedNttMatrixTables fused_intt;
    FusedNttMatrixFp64SplitTables fused_ntt_fp64;
    FusedNttMatrixFp64SplitTables fused_intt_fp64;
};

}  // namespace

GpuParameterData::GpuParameterData(const PoseidonContext &context, int device_id)
{
    build_from_poseidon_context(context, device_id);
}

GpuParameterData::GpuParameterData(
    const PoseidonContext &context,
    int device_id,
    const std::vector<std::size_t> &active_q_counts,
    bool include_key_level)
{
    if (active_q_counts.empty())
    {
        throw std::invalid_argument(
            "GpuParameterData sparse construction requires active q levels");
    }
    build_from_poseidon_context(context, device_id, active_q_counts, include_key_level);
}

void GpuParameterData::build_from_poseidon_context(
    const PoseidonContext &context,
    int device_id)
{
    build_from_poseidon_context(context, device_id, {}, false);
}

void GpuParameterData::build_from_poseidon_context(
    const PoseidonContext &context,
    int device_id,
    const std::vector<std::size_t> &active_q_counts,
    bool include_key_level)
{
    levels_.clear();

    const std::set<std::size_t> requested_q_counts(
        active_q_counts.begin(), active_q_counts.end());
    if (requested_q_counts.size() != active_q_counts.size() ||
        requested_q_counts.count(0) != 0)
    {
        throw std::invalid_argument(
            "GpuParameterData active q levels must be unique and nonzero");
    }
    const bool sparse_q_only = !requested_q_counts.empty();
    std::set<std::size_t> built_q_counts;

    auto crt_context = context.crt_context();
    if (!crt_context)
    {
        throw std::invalid_argument("GpuParameterData requires a valid PoseidonContext");
    }
    auto first_context_data = crt_context->first_context_data();
    if (!first_context_data)
    {
        throw std::invalid_argument("GpuParameterData requires a first q-only context level");
    }
    const auto &first_q = first_context_data->parms().q();
    const auto &master_q_moduli = context.parameters_literal()->q();
    const auto &master_p_moduli = context.parameters_literal()->p();
    const auto *small_ntt_tables = crt_context->small_ntt_tables();
    const std::size_t p_ntt_table_offset = master_q_moduli.size();
    const int precomputed_fused_matrix_stages =
        read_precomputed_fused_matrix_stages();
    const bool precompute_fused_matrix_fp64_tables =
        precomputed_fused_matrix_stages > 0 &&
        fused_matrix_fp64_tables_enabled();
    const bool precompute_word_fused_matrix_tables =
        precomputed_fused_matrix_stages > 0 &&
        !fp64_tensor_ntt_algorithm_requested();
    const std::size_t fused_matrix_max_levels =
        read_fused_matrix_max_levels();
    const bool skip_montgomery = skip_montgomery_tables_enabled();

    std::vector<SelectedContextLevel> selected;
    std::size_t max_q_used = 0;
    std::size_t max_p_used = 0;
    std::size_t common_degree = 0;

    auto context_data = crt_context->key_context_data();
    if (!context_data)
    {
        context_data = crt_context->first_context_data();
    }
    std::size_t level_index = 0;
    while (context_data)
    {
        const auto &parms = context_data->parms();
        const auto &q = parms.q();
        const auto &p = parms.p();
        const bool is_key_level = !p.empty();

        if (sparse_q_only)
        {
            if (is_key_level)
            {
                if (!include_key_level)
                {
                    context_data = context_data->next_context_data();
                    ++level_index;
                    continue;
                }
            }
            else if (requested_q_counts.count(q.size()) == 0)
            {
                context_data = context_data->next_context_data();
                ++level_index;
                continue;
            }
        }

        const auto *rns_qp = context_data->qp_rns_tool();
        std::vector<Modulus> parameter_p = p;
        if (parameter_p.empty() && rns_qp != nullptr && rns_qp->base_p() != nullptr)
        {
            parameter_p = copy_rns_base_moduli(*rns_qp->base_p());
        }

        if (selected.empty())
        {
            common_degree = parms.degree();
        }
        else if (parms.degree() != common_degree)
        {
            throw std::invalid_argument("GpuParameterData requires uniform polynomial degree across levels");
        }
        if (q.size() > master_q_moduli.size() || parameter_p.size() > master_p_moduli.size())
        {
            throw std::invalid_argument("GpuParameterData level modulus count exceeds context parameters");
        }
        for (std::size_t i = 0; i < q.size(); ++i)
        {
            if (q[i] != master_q_moduli[i])
            {
                throw std::invalid_argument("GpuParameterData requires level q primes to match context prefix");
            }
        }
        for (std::size_t i = 0; i < parameter_p.size(); ++i)
        {
            if (parameter_p[i] != master_p_moduli[i])
            {
                throw std::invalid_argument("GpuParameterData requires level p primes to match context prefix");
            }
        }

        max_q_used = std::max(max_q_used, q.size());
        max_p_used = std::max(max_p_used, parameter_p.size());
        if (sparse_q_only && !is_key_level)
        {
            built_q_counts.insert(q.size());
        }
        selected.push_back(SelectedContextLevel{
            context_data,
            level_index,
            std::move(parameter_p)});

        context_data = context_data->next_context_data();
        ++level_index;
    }

    if (sparse_q_only && built_q_counts != requested_q_counts)
    {
        std::ostringstream stream;
        stream << "GpuParameterData sparse q level not found; requested=";
        for (auto q_count : requested_q_counts)
        {
            stream << q_count << ',';
        }
        stream << " built=";
        for (auto q_count : built_q_counts)
        {
            stream << q_count << ',';
        }
        throw std::out_of_range(stream.str());
    }

    if (selected.empty())
    {
        return;
    }

    const std::size_t total_primes = max_q_used + max_p_used;
    if (total_primes != 0 && small_ntt_tables == nullptr)
    {
        throw std::invalid_argument("GpuParameterData requires CPU NTT tables");
    }

    gpu_check_cuda(cudaSetDevice(device_id), "GpuParameterData cudaSetDevice");
    const cudaStream_t stream = gpu_execution_stream();

    // 1. Precompute per-prime master root tables ONCE across the unique Q and P primes
    //    in parallel, then upload ONE master GPU copy and release the host root buffers.
    std::vector<GpuWord> master_words(total_primes);
    std::vector<GpuWide> master_barrett(total_primes);
    std::vector<GpuWord> master_inv_degree(total_primes);
    std::vector<GpuWord> master_mont_inv_degree(skip_montgomery ? 0 : total_primes);
    std::vector<GpuWord> master_mont_neg_inv(skip_montgomery ? 0 : total_primes);

    DeviceVector<GpuWord> d_master_ntt;
    DeviceVector<GpuWord> d_master_intt;
    DeviceVector<GpuWord> d_master_fourstep_intt;
    DeviceVector<GpuWord> d_master_norm_intt;
    DeviceVector<GpuWord> d_master_mont_ntt;
    DeviceVector<GpuWord> d_master_mont_intt;
    DeviceVector<GpuWord> d_master_mont_norm_intt;

    if (total_primes > 0)
    {
        std::vector<std::size_t> fourstep_perm(common_degree, 0);
        if (common_degree > 0)
        {
            int log_degree = 0;
            for (std::size_t value = common_degree; value > 1; value >>= 1)
            {
                ++log_degree;
            }
            fourstep_perm[0] = 0;
            for (std::size_t cheddar_index = 1;
                 cheddar_index < common_degree;
                 ++cheddar_index)
            {
                const std::size_t inverse_power =
                    util::reverse_bits(cheddar_index, log_degree);
                fourstep_perm[cheddar_index] =
                    util::reverse_bits(inverse_power - 1, log_degree) + 1;
            }
        }

        const std::size_t master_root_elems = total_primes * common_degree;
        std::vector<GpuWord> master_ntt(master_root_elems);
        std::vector<GpuWord> master_intt(master_root_elems);
        std::vector<GpuWord> master_fourstep_intt(master_root_elems);
        std::vector<GpuWord> master_norm_intt(master_root_elems);
        std::vector<GpuWord> master_mont_ntt(skip_montgomery ? 0 : master_root_elems);
        std::vector<GpuWord> master_mont_intt(skip_montgomery ? 0 : master_root_elems);
        std::vector<GpuWord> master_mont_norm_intt(skip_montgomery ? 0 : master_root_elems);

        std::exception_ptr master_error;
        #pragma omp parallel for schedule(static)
        for (std::size_t idx = 0; idx < total_primes; ++idx)
        {
            try
            {
                const bool is_q = idx < max_q_used;
                const std::size_t p_idx = is_q ? 0 : (idx - max_q_used);
                const auto &modulus = is_q ? master_q_moduli[idx] : master_p_moduli[p_idx];
                const auto &table = is_q
                    ? small_ntt_tables[idx]
                    : small_ntt_tables[p_ntt_table_offset + p_idx];
                if (table.coeff_count() != common_degree)
                {
                    throw std::invalid_argument("GpuParameterData NTT table degree mismatch");
                }

                const GpuWord mod_word = checked_gpu_word(
                    modulus.value(),
                    is_q ? "GpuParameterData only supports q primes that fit in GpuWord"
                         : "GpuParameterData only supports p primes that fit in GpuWord");
                master_words[idx] = mod_word;
                master_barrett[idx] = barrett_ratio_64(modulus);
                const GpuWord inv_deg = checked_gpu_word(
                    table.inv_degree_modulo().operand,
                    "GpuParameterData only supports inverse degree constants that fit in GpuWord");
                master_inv_degree[idx] = inv_deg;

                const auto *fwd_roots = table.get_from_root_powers();
                const auto *inv_roots = table.get_from_inv_root_powers();
                const std::size_t off = idx * common_degree;
                GpuWord *ntt_dst = master_ntt.data() + off;
                GpuWord *intt_dst = master_intt.data() + off;
                GpuWord *fs_dst = master_fourstep_intt.data() + off;
                GpuWord *norm_dst = master_norm_intt.data() + off;

                for (std::size_t i = 0; i < common_degree; ++i)
                {
                    ntt_dst[i] = checked_gpu_word(
                        fwd_roots[i].operand,
                        "GpuParameterData only supports NTT roots that fit in GpuWord");
                    intt_dst[i] = checked_gpu_word(
                        inv_roots[i].operand,
                        "GpuParameterData only supports inverse NTT roots that fit in GpuWord");
                }
                for (std::size_t i = 0; i < common_degree; ++i)
                {
                    fs_dst[i] = intt_dst[fourstep_perm[i]];
                }
                if (common_degree > 0)
                {
                    std::memcpy(norm_dst, intt_dst, common_degree * sizeof(GpuWord));
                    norm_dst[common_degree - 1] = multiply_mod_host(
                        norm_dst[common_degree - 1],
                        inv_deg,
                        mod_word);
                }

                if (!skip_montgomery)
                {
                    const GpuWord r_mod = montgomery_r_mod(modulus);
                    master_mont_neg_inv[idx] = montgomery_neg_inverse_u32(modulus);
                    master_mont_inv_degree[idx] = multiply_mod_host(inv_deg, r_mod, modulus);
                    GpuWord *m_ntt_dst = master_mont_ntt.data() + off;
                    GpuWord *m_intt_dst = master_mont_intt.data() + off;
                    GpuWord *m_norm_dst = master_mont_norm_intt.data() + off;
                    for (std::size_t i = 0; i < common_degree; ++i)
                    {
                        m_ntt_dst[i] = multiply_mod_host(ntt_dst[i], r_mod, modulus);
                        m_intt_dst[i] = multiply_mod_host(intt_dst[i], r_mod, modulus);
                    }
                    if (common_degree > 0)
                    {
                        std::memcpy(m_norm_dst, m_intt_dst, common_degree * sizeof(GpuWord));
                        m_norm_dst[common_degree - 1] = multiply_mod_host(
                            norm_dst[common_degree - 1],
                            r_mod,
                            modulus);
                    }
                }
            }
            catch (...)
            {
                #pragma omp critical
                {
                    if (!master_error)
                    {
                        master_error = std::current_exception();
                    }
                }
            }
        }
        if (master_error)
        {
            std::rethrow_exception(master_error);
        }

        StreamSyncOnExit master_sync{device_id, stream, false};
        d_master_ntt = copy_to_device_vector_async(master_ntt, device_id, stream);
        d_master_intt = copy_to_device_vector_async(master_intt, device_id, stream);
        d_master_fourstep_intt =
            copy_to_device_vector_async(master_fourstep_intt, device_id, stream);
        d_master_norm_intt =
            copy_to_device_vector_async(master_norm_intt, device_id, stream);
        if (!skip_montgomery)
        {
            d_master_mont_ntt =
                copy_to_device_vector_async(master_mont_ntt, device_id, stream);
            d_master_mont_intt =
                copy_to_device_vector_async(master_mont_intt, device_id, stream);
            d_master_mont_norm_intt =
                copy_to_device_vector_async(master_mont_norm_intt, device_id, stream);
        }
        gpu_check_cuda(
            cudaStreamSynchronize(stream),
            "GpuParameterData master root tables H2D sync");
        master_sync.synced = true;
    }

    // 2. Precompute pairwise q^{-1} mod q ONCE and prepare all per-level CPU tables in parallel.
    const auto master_q_inv_mod_q =
        compute_master_rescale_q_inv_mod_q(master_q_moduli, max_q_used);

    std::vector<HostLevelPrep> preps(selected.size());
    std::exception_ptr prep_error;
    #pragma omp parallel for schedule(dynamic, 1)
    for (std::size_t idx = 0; idx < selected.size(); ++idx)
    {
        try
        {
            const auto &sel = selected[idx];
            const auto &cd = *sel.context_data;
            const auto &parms = cd.parms();
            const auto &q = parms.q();
            const auto &p = parms.p();
            const std::size_t q_count = q.size();
            const std::size_t param_p_count = sel.parameter_p.size();

            auto &prep = preps[idx];
            prep.parms_id = cd.parms_id();
            prep.degree = parms.degree();
            prep.q_count = q_count;
            prep.p_count = p.size();
            prep.parameter_p_count = param_p_count;

            if (!q.empty())
            {
                prep.q_last = checked_gpu_word(
                    q.back().value(),
                    "GpuParameterData only supports q_last that fits in GpuWord");
                prep.half_q_last = static_cast<GpuWord>(prep.q_last >> 1);
            }

            prep.q_words.assign(master_words.begin(), master_words.begin() + q_count);
            prep.p_words.assign(
                master_words.begin() + max_q_used,
                master_words.begin() + max_q_used + param_p_count);
            prep.rns_words = concatenate_vectors(prep.q_words, prep.p_words);

            prep.q_barrett_ratios.assign(
                master_barrett.begin(),
                master_barrett.begin() + q_count);
            prep.p_barrett_ratios.assign(
                master_barrett.begin() + max_q_used,
                master_barrett.begin() + max_q_used + param_p_count);
            prep.rns_barrett_ratios =
                concatenate_vectors(prep.q_barrett_ratios, prep.p_barrett_ratios);

            prep.rns_inv_degree.reserve(q_count + param_p_count);
            prep.rns_inv_degree.insert(
                prep.rns_inv_degree.end(),
                master_inv_degree.begin(),
                master_inv_degree.begin() + q_count);
            prep.rns_inv_degree.insert(
                prep.rns_inv_degree.end(),
                master_inv_degree.begin() + max_q_used,
                master_inv_degree.begin() + max_q_used + param_p_count);

            if (!skip_montgomery)
            {
                prep.rns_montgomery_inv_degree.reserve(q_count + param_p_count);
                prep.rns_montgomery_inv_degree.insert(
                    prep.rns_montgomery_inv_degree.end(),
                    master_mont_inv_degree.begin(),
                    master_mont_inv_degree.begin() + q_count);
                prep.rns_montgomery_inv_degree.insert(
                    prep.rns_montgomery_inv_degree.end(),
                    master_mont_inv_degree.begin() + max_q_used,
                    master_mont_inv_degree.begin() + max_q_used + param_p_count);

                prep.rns_montgomery_neg_inv.reserve(q_count + param_p_count);
                prep.rns_montgomery_neg_inv.insert(
                    prep.rns_montgomery_neg_inv.end(),
                    master_mont_neg_inv.begin(),
                    master_mont_neg_inv.begin() + q_count);
                prep.rns_montgomery_neg_inv.insert(
                    prep.rns_montgomery_neg_inv.end(),
                    master_mont_neg_inv.begin() + max_q_used,
                    master_mont_neg_inv.begin() + max_q_used + param_p_count);
            }

            prep.half_q_last_mod_q = compute_half_q_last_mod_q(q, prep.half_q_last);
            prep.inv_q_last_mod_q = copy_inv_q_last_mod_q_operands(cd.rns_tool(), q_count);

            auto rescale_x2 = compute_rescale_x2_constants(q);
            prep.q_second_last = rescale_x2.q_second_last;
            prep.q_last_two_product = rescale_x2.product;
            prep.half_q_last_two_product = rescale_x2.half_product;
            prep.inv_q_last_mod_q_second_last = rescale_x2.inv_q_last_mod_q_second_last;
            prep.q_last_two_product_mod_q = std::move(rescale_x2.product_mod_q);
            prep.inv_q_last_two_product_mod_q = std::move(rescale_x2.inv_product_mod_q);
            if (q_count >= 3)
            {
                prep.q_second_last_modulus_constant = prep.q_barrett_ratios[q_count - 2];
            }
            prep.rescale_q_inv_mod_q =
                slice_rescale_q_inv_mod_q(master_q_inv_mod_q, max_q_used, q_count);

            prep.hybrid = prepare_hybrid_key_switch_tables(cd.qp_rns_tool());
            if (prep.p_count == 0)
            {
                prep.bootstrap_raise =
                    prepare_bootstrap_raise_tables(q, first_q, cd.rns_tool());
            }
        }
        catch (...)
        {
            #pragma omp critical
            {
                if (!prep_error)
                {
                    prep_error = std::current_exception();
                }
            }
        }
    }
    if (prep_error)
    {
        std::rethrow_exception(prep_error);
    }

    // Optional experimental TAM fused matrices (default stages=0 => skipped).
    if (precomputed_fused_matrix_stages > 0)
    {
        const auto *p_ntt_tables = small_ntt_tables + p_ntt_table_offset;
        for (std::size_t idx = 0; idx < selected.size(); ++idx)
        {
            const auto &sel = selected[idx];
            auto &prep = preps[idx];
            const int level_fused_matrix_stages =
                sel.level_index < fused_matrix_max_levels
                    ? precomputed_fused_matrix_stages
                    : 0;
            if (level_fused_matrix_stages <= 0)
            {
                continue;
            }
            const auto rns_ntt_table_ptrs = collect_rns_ntt_table_pointers(
                small_ntt_tables,
                prep.q_count,
                p_ntt_tables,
                prep.parameter_p_count);
            prep.fused_ntt = load_or_build_fused_matrix_tables(
                rns_ntt_table_ptrs,
                prep.degree,
                level_fused_matrix_stages,
                false);
            prep.fused_intt = load_or_build_fused_matrix_tables(
                rns_ntt_table_ptrs,
                prep.degree,
                level_fused_matrix_stages,
                true);
            if (precompute_fused_matrix_fp64_tables)
            {
                prep.fused_ntt_fp64 =
                    split_fused_matrices_for_fp64(prep.fused_ntt.matrices);
                prep.fused_intt_fp64 =
                    split_fused_matrices_for_fp64(prep.fused_intt.matrices);
            }
        }
    }

    // 3. Allocate per-level GPU shards on the calling thread (preserving TLS stream & RMM),
    //    fill giant root tables via fast async HBM D2D from the master GPU copy,
    //    enqueue tiny scalar/base-conv async H2D copies, and sync ONCE at the end.
    levels_.reserve(selected.size());
    StreamSyncOnExit upload_sync{device_id, stream, false};

    for (std::size_t idx = 0; idx < selected.size(); ++idx)
    {
        const auto &prep = preps[idx];

        GpuLevelInfo level;
        level.parms_id = prep.parms_id;
        level.degree = prep.degree;
        level.q_count = prep.q_count;
        level.p_count = prep.p_count;

        GpuParameterShard shard;
        shard.device_id = device_id;
        shard.limb_begin = 0;
        shard.limb_count = prep.q_count + prep.parameter_p_count;
        shard.q_last = prep.q_last;
        shard.half_q_last = prep.half_q_last;
        shard.q_second_last = prep.q_second_last;
        shard.q_last_two_product = prep.q_last_two_product;
        shard.half_q_last_two_product = prep.half_q_last_two_product;
        shard.inv_q_last_mod_q_second_last = prep.inv_q_last_mod_q_second_last;
        shard.q_second_last_modulus_constant = prep.q_second_last_modulus_constant;

        shard.q_primes = copy_to_device_vector_async(prep.q_words, device_id, stream);
        shard.p_primes = copy_to_device_vector_async(prep.p_words, device_id, stream);
        shard.rns_primes = copy_to_device_vector_async(prep.rns_words, device_id, stream);
        shard.q_modulus_constants =
            copy_to_device_vector_async(prep.q_barrett_ratios, device_id, stream);
        shard.p_modulus_constants =
            copy_to_device_vector_async(prep.p_barrett_ratios, device_id, stream);
        shard.rns_modulus_constants =
            copy_to_device_vector_async(prep.rns_barrett_ratios, device_id, stream);
        shard.half_q_last_mod_q =
            copy_to_device_vector_async(prep.half_q_last_mod_q, device_id, stream);
        shard.inv_q_last_mod_q =
            copy_to_device_vector_async(prep.inv_q_last_mod_q, device_id, stream);
        shard.q_last_two_product_mod_q =
            copy_to_device_vector_async(prep.q_last_two_product_mod_q, device_id, stream);
        shard.inv_q_last_two_product_mod_q =
            copy_to_device_vector_async(prep.inv_q_last_two_product_mod_q, device_id, stream);
        shard.rescale_q_inv_mod_q =
            copy_to_device_vector_async(prep.rescale_q_inv_mod_q, device_id, stream);

        shard.ntt_tables = slice_master_roots_d2d_async(
            d_master_ntt, max_q_used, prep.q_count, prep.parameter_p_count,
            prep.degree, device_id, stream);
        shard.intt_tables = slice_master_roots_d2d_async(
            d_master_intt, max_q_used, prep.q_count, prep.parameter_p_count,
            prep.degree, device_id, stream);
        shard.fourstep_intt_tables = slice_master_roots_d2d_async(
            d_master_fourstep_intt, max_q_used, prep.q_count, prep.parameter_p_count,
            prep.degree, device_id, stream);
        shard.intt_tables_normalized = slice_master_roots_d2d_async(
            d_master_norm_intt, max_q_used, prep.q_count, prep.parameter_p_count,
            prep.degree, device_id, stream);
        shard.inv_degree_modulo =
            copy_to_device_vector_async(prep.rns_inv_degree, device_id, stream);

        if (!skip_montgomery)
        {
            shard.montgomery_ntt_tables = slice_master_roots_d2d_async(
                d_master_mont_ntt, max_q_used, prep.q_count, prep.parameter_p_count,
                prep.degree, device_id, stream);
            shard.montgomery_intt_tables = slice_master_roots_d2d_async(
                d_master_mont_intt, max_q_used, prep.q_count, prep.parameter_p_count,
                prep.degree, device_id, stream);
            shard.montgomery_intt_tables_normalized = slice_master_roots_d2d_async(
                d_master_mont_norm_intt, max_q_used, prep.q_count, prep.parameter_p_count,
                prep.degree, device_id, stream);
            shard.montgomery_inv_degree_modulo =
                copy_to_device_vector_async(prep.rns_montgomery_inv_degree, device_id, stream);
            shard.montgomery_neg_inv_modulo =
                copy_to_device_vector_async(prep.rns_montgomery_neg_inv, device_id, stream);
        }

        if (precomputed_fused_matrix_stages > 0)
        {
            shard.ntt_fused_matrix_fusion_stages = prep.fused_ntt.fusion_stages;
            shard.ntt_fused_matrix_stage_counts =
                copy_to_device_vector_async(prep.fused_ntt.stage_counts, device_id, stream);
            shard.ntt_fused_matrix_group_counts =
                copy_to_device_vector_async(prep.fused_ntt.group_counts, device_id, stream);
            shard.ntt_fused_matrix_stage_offsets =
                copy_to_device_vector_async(prep.fused_ntt.stage_offsets, device_id, stream);
            if (precompute_word_fused_matrix_tables)
            {
                shard.ntt_fused_matrices =
                    copy_to_device_vector_async(prep.fused_ntt.matrices, device_id, stream);
            }
            shard.ntt_fused_matrices_fp64_lo =
                copy_to_device_vector_async(prep.fused_ntt_fp64.lo, device_id, stream);
            shard.ntt_fused_matrices_fp64_hi =
                copy_to_device_vector_async(prep.fused_ntt_fp64.hi, device_id, stream);

            shard.intt_fused_matrix_fusion_stages = prep.fused_intt.fusion_stages;
            shard.intt_fused_matrix_stage_counts =
                copy_to_device_vector_async(prep.fused_intt.stage_counts, device_id, stream);
            shard.intt_fused_matrix_group_counts =
                copy_to_device_vector_async(prep.fused_intt.group_counts, device_id, stream);
            shard.intt_fused_matrix_stage_offsets =
                copy_to_device_vector_async(prep.fused_intt.stage_offsets, device_id, stream);
            if (precompute_word_fused_matrix_tables)
            {
                shard.intt_fused_matrices =
                    copy_to_device_vector_async(prep.fused_intt.matrices, device_id, stream);
            }
            shard.intt_fused_matrices_fp64_lo =
                copy_to_device_vector_async(prep.fused_intt_fp64.lo, device_id, stream);
            shard.intt_fused_matrices_fp64_hi =
                copy_to_device_vector_async(prep.fused_intt_fp64.hi, device_id, stream);
        }

        upload_hybrid_key_switch_tables_async(prep.hybrid, shard, device_id, stream);
        if (level.p_count == 0)
        {
            upload_bootstrap_raise_tables_async(
                prep.bootstrap_raise, shard, device_id, stream);
        }

        level.shards.push_back(std::move(shard));
        levels_.push_back(std::move(level));
    }

    gpu_check_cuda(
        cudaStreamSynchronize(stream),
        "GpuParameterData build_from_poseidon_context sync");
    upload_sync.synced = true;
}

const GpuLevelInfo &GpuParameterData::get_level(const parms_id_type &parms_id) const
{
    for (const auto &level : levels_)
    {
        if (level.parms_id == parms_id)
        {
            return level;
        }
    }

    throw std::out_of_range(
        "GpuParameterData level not found for parms_id=" +
        format_parms_id_for_error(parms_id) +
        "; known levels: " + format_known_levels_for_error(levels_));
}

const GpuLevelInfo &GpuParameterData::get_next_level(const parms_id_type &parms_id) const
{
    for (std::size_t i = 0; i < levels_.size(); ++i)
    {
        if (levels_[i].parms_id == parms_id)
        {
            if (i + 1 >= levels_.size())
            {
                throw std::out_of_range("GpuParameterData next level not found for parms_id");
            }
            return levels_[i + 1];
        }
    }

    throw std::out_of_range(
        "GpuParameterData level not found for parms_id=" +
        format_parms_id_for_error(parms_id) +
        "; known levels: " + format_known_levels_for_error(levels_));
}

const GpuLevelInfo &GpuParameterData::get_level_by_q_count(
    std::size_t q_count,
    std::size_t p_count) const
{
    for (const auto &level : levels_)
    {
        if (level.q_count == q_count && level.p_count == p_count)
        {
            return level;
        }
    }

    throw std::out_of_range("GpuParameterData level not found for q_count");
}

GpuLevelInfo GpuParameterData::take_level_by_q_count(std::size_t q_count)
{
    for (auto &level : levels_)
    {
        if (level.q_count == q_count && level.p_count == 0)
        {
            GpuLevelInfo result = std::move(level);
            level = GpuLevelInfo{};
            return result;
        }
    }
    throw std::out_of_range("GpuParameterData::take_level_by_q_count: level not found");
}

void GpuParameterData::adopt_level(GpuLevelInfo level)
{
    for (auto &existing : levels_)
    {
        if (existing.parms_id == level.parms_id && existing.p_count == level.p_count)
        {
            if (existing.degree != level.degree || existing.q_count != level.q_count ||
                level.shards.size() != existing.shards.size())
            {
                throw std::invalid_argument(
                    "GpuParameterData::adopt_level: level shape differs from the replaced level");
            }
            existing = std::move(level);
            return;
        }
    }
    throw std::out_of_range("GpuParameterData::adopt_level: no level with this parms_id");
}

void GpuParameterData::adopt_or_insert_level(GpuLevelInfo level)
{
    if (level.degree == 0 || level.q_count == 0 || level.shards.empty())
    {
        throw std::invalid_argument(
            "GpuParameterData::adopt_or_insert_level: invalid level");
    }
    for (auto &existing : levels_)
    {
        if (existing.parms_id == level.parms_id && existing.p_count == level.p_count)
        {
            if (existing.degree != level.degree || existing.q_count != level.q_count ||
                level.shards.size() != existing.shards.size())
            {
                throw std::invalid_argument(
                    "GpuParameterData::adopt_or_insert_level: level shape differs from the replaced level");
            }
            existing = std::move(level);
            return;
        }
    }

    auto comes_before = [](const GpuLevelInfo &lhs, const GpuLevelInfo &rhs) {
        const bool lhs_key = lhs.p_count != 0;
        const bool rhs_key = rhs.p_count != 0;
        if (lhs_key != rhs_key)
        {
            return lhs_key;
        }
        return lhs.q_count > rhs.q_count;
    };

    auto it = levels_.begin();
    while (it != levels_.end() && comes_before(*it, level))
    {
        ++it;
    }
    levels_.insert(it, std::move(level));
}

const GpuLevelInfo &GpuParameterData::get_first_q_level() const
{
    const GpuLevelInfo *best = nullptr;
    for (const auto &level : levels_)
    {
        if (level.p_count != 0)
        {
            continue;
        }
        if (best == nullptr || level.q_count > best->q_count)
        {
            best = &level;
        }
    }
    if (best == nullptr)
    {
        throw std::out_of_range("GpuParameterData first q-only level not found");
    }
    return *best;
}

bool GpuParameterData::empty() const
{
    return levels_.empty();
}

std::size_t GpuParameterData::allocated_bytes() const
{
    std::size_t bytes = 0;
    const auto add = [&bytes](const auto &vector)
    {
        constexpr std::size_t element_bytes = sizeof(*vector.data());
        const auto count = vector.size();
        if (count > (std::numeric_limits<std::size_t>::max() - bytes) / element_bytes)
        {
            throw std::overflow_error("GpuParameterData allocated byte count overflow");
        }
        bytes += count * element_bytes;
    };

    for (const auto &level : levels_)
    {
        for (const auto &shard : level.shards)
        {
            add(shard.q_primes);
            add(shard.p_primes);
            add(shard.rns_primes);
            add(shard.q_modulus_constants);
            add(shard.p_modulus_constants);
            add(shard.rns_modulus_constants);
            add(shard.half_q_last_mod_q);
            add(shard.inv_q_last_mod_q);
            add(shard.q_last_two_product_mod_q);
            add(shard.inv_q_last_two_product_mod_q);
            add(shard.rescale_q_inv_mod_q);
            add(shard.hybrid_decomp_start);
            add(shard.hybrid_decomp_end);
            add(shard.hybrid_p_mod_q);
            add(shard.hybrid_inv_p_mod_q);
            add(shard.hybrid_q_conv_matrix_offsets);
            add(shard.hybrid_p_conv_matrix_offsets);
            add(shard.hybrid_q_conv_matrices);
            add(shard.hybrid_p_conv_matrices);
            add(shard.hybrid_moddown_p_to_q_matrix);
            add(shard.hybrid_qi_inv_punctured);
            add(shard.hybrid_p_inv_punctured);
            add(shard.bootstrap_raise_inv_punctured);
            add(shard.bootstrap_raise_matrix);
            add(shard.ntt_tables);
            add(shard.intt_tables);
            add(shard.fourstep_intt_tables);
            add(shard.intt_tables_normalized);
            add(shard.inv_degree_modulo);
            add(shard.montgomery_ntt_tables);
            add(shard.montgomery_intt_tables);
            add(shard.montgomery_intt_tables_normalized);
            add(shard.montgomery_inv_degree_modulo);
            add(shard.montgomery_neg_inv_modulo);
            add(shard.ntt_fused_matrix_stage_counts);
            add(shard.ntt_fused_matrix_group_counts);
            add(shard.ntt_fused_matrix_stage_offsets);
            add(shard.ntt_fused_matrices);
            add(shard.ntt_fused_matrices_fp64_lo);
            add(shard.ntt_fused_matrices_fp64_hi);
            add(shard.intt_fused_matrix_stage_counts);
            add(shard.intt_fused_matrix_group_counts);
            add(shard.intt_fused_matrix_stage_offsets);
            add(shard.intt_fused_matrices);
            add(shard.intt_fused_matrices_fp64_lo);
            add(shard.intt_fused_matrices_fp64_hi);
        }
    }
    return bytes;
}

}  // namespace gpu
}  // namespace poseidon
