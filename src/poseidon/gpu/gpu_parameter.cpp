#include "poseidon/gpu/gpu_parameter.h"

#include "poseidon/basics/util/ntt.h"
#include "poseidon/poseidon_context.h"
#include "poseidon/util/rns_tool_qp.h"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
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

std::vector<GpuWord> copy_moduli_to_gpu_words(
    const std::vector<Modulus> &moduli,
    const char *what)
{
    std::vector<GpuWord> result;
    result.reserve(moduli.size());
    for (const auto &modulus : moduli)
    {
        result.push_back(checked_gpu_word(modulus.value(), what));
    }
    return result;
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

std::vector<GpuWide> copy_barrett_ratios(const std::vector<Modulus> &moduli)
{
    std::vector<GpuWide> result;
    result.reserve(moduli.size());
    for (const auto &modulus : moduli)
    {
        result.push_back(barrett_ratio_64(modulus));
    }
    return result;
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
DeviceVector<T> copy_to_device_vector(const std::vector<T> &host, int device_id)
{
    DeviceVector<T> result(host.size(), device_id);
    if (!host.empty())
    {
        result.copy_from_host(host.data(), host.size());
    }
    return result;
}

std::vector<GpuWord> copy_ntt_root_operands(
    const util::NTTTables *ntt_tables,
    std::size_t limb_count,
    std::size_t degree,
    bool inverse)
{
    if (limb_count != 0 && ntt_tables == nullptr)
    {
        throw std::invalid_argument("GpuParameterData requires CPU NTT tables");
    }

    std::vector<GpuWord> result(limb_count * degree);
    for (std::size_t limb = 0; limb < limb_count; ++limb)
    {
        const auto &table = ntt_tables[limb];
        if (table.coeff_count() != degree)
        {
            throw std::invalid_argument("GpuParameterData NTT table degree mismatch");
        }

        const auto *roots = inverse
            ? table.get_from_inv_root_powers()
            : table.get_from_root_powers();

        for (std::size_t i = 0; i < degree; ++i)
        {
            result[limb * degree + i] = checked_gpu_word(
                roots[i].operand,
                inverse
                    ? "GpuParameterData only supports inverse NTT roots that fit in GpuWord"
                    : "GpuParameterData only supports NTT roots that fit in GpuWord");
        }
    }
    return result;
}

std::vector<GpuWord> copy_inv_degree_operands(
    const util::NTTTables *ntt_tables,
    std::size_t limb_count)
{
    if (limb_count != 0 && ntt_tables == nullptr)
    {
        throw std::invalid_argument("GpuParameterData requires CPU NTT tables");
    }

    std::vector<GpuWord> result(limb_count);
    for (std::size_t limb = 0; limb < limb_count; ++limb)
    {
        result[limb] = checked_gpu_word(
            ntt_tables[limb].inv_degree_modulo().operand,
            "GpuParameterData only supports inverse degree constants that fit in GpuWord");
    }
    return result;
}

struct FusedNttMatrixTables
{
    std::size_t fusion_stages = 0;
    std::vector<GpuWord> stage_counts;
    std::vector<GpuWide> group_counts;
    std::vector<GpuWide> stage_offsets;
    std::vector<GpuWord> matrices;
};

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
                value == "tam_tensor" || value == "matrix")
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

void copy_hybrid_key_switch_tables(
    const util::RNSToolQP *rns_qp,
    GpuParameterShard &shard,
    int device_id)
{
    if (rns_qp == nullptr || rns_qp->base_q() == nullptr || rns_qp->base_p() == nullptr)
    {
        return;
    }

    const std::size_t base_q_size = rns_qp->base_q()->size();
    const std::size_t base_p_size = rns_qp->base_p()->size();
    const std::size_t decomp_count = rns_qp->decomp_count();
    if (base_q_size == 0 || base_p_size == 0 || decomp_count == 0)
    {
        return;
    }
    if (base_q_size > std::numeric_limits<GpuWord>::max() ||
        base_p_size > std::numeric_limits<GpuWord>::max() ||
        decomp_count > std::numeric_limits<GpuWord>::max())
    {
        throw std::invalid_argument("GpuParameterData HYBRID table shape does not fit GpuWord");
    }

    shard.hybrid_base_q_count = base_q_size;
    shard.hybrid_base_p_count = base_p_size;
    shard.hybrid_decomp_count = decomp_count;

    std::vector<GpuWord> decomp_start(decomp_count);
    std::vector<GpuWord> decomp_end(decomp_count);
    std::vector<GpuWord> q_matrix_offsets(decomp_count + 1);
    std::vector<GpuWord> p_matrix_offsets(decomp_count + 1);
    std::vector<GpuWord> q_matrices;
    std::vector<GpuWord> p_matrices;
    std::vector<GpuWord> moddown_p_to_q_matrix;
    std::vector<GpuWord> qi_inv_punctured(decomp_count * base_p_size, 0);

    for (std::size_t decomp_index = 0; decomp_index < decomp_count; ++decomp_index)
    {
        const auto &decomp = rns_qp->decomp(decomp_index);
        const std::size_t start = decomp.base_start_idx();
        const std::size_t end = decomp.base_last_idx();
        decomp_start[decomp_index] = checked_gpu_word(
            start,
            "GpuParameterData HYBRID decomp start does not fit GpuWord");
        decomp_end[decomp_index] = checked_gpu_word(
            end,
            "GpuParameterData HYBRID decomp end does not fit GpuWord");
        q_matrix_offsets[decomp_index] = checked_gpu_word(
            q_matrices.size(),
            "GpuParameterData HYBRID q matrix offset does not fit GpuWord");
        p_matrix_offsets[decomp_index] = checked_gpu_word(
            p_matrices.size(),
            "GpuParameterData HYBRID p matrix offset does not fit GpuWord");

        if (!decomp.has_base_conversion())
        {
            qi_inv_punctured[decomp_index * base_p_size] = 1;
            q_matrices.resize(q_matrices.size() + base_q_size * base_p_size, 0);
            p_matrices.resize(p_matrices.size() + base_p_size * base_p_size, 0);
            continue;
        }

        const auto &decomp_ibase = decomp.obase_p_conv()->ibase();
        const auto *decomp_inv_punctured =
            decomp_ibase.inv_punctured_prod_mod_base_array();
        for (std::size_t col = 0; col < decomp_ibase.size(); ++col)
        {
            qi_inv_punctured[decomp_index * base_p_size + col] = checked_gpu_word(
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
                    q_matrices.push_back(0);
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
                q_matrices,
                "GpuParameterData HYBRID q conversion constant does not fit GpuWord");
        }

        append_base_converter_matrix_padded(
            *decomp.obase_p_conv(),
            base_p_size,
            p_matrices,
            "GpuParameterData HYBRID p conversion constant does not fit GpuWord");
    }

    q_matrix_offsets[decomp_count] = checked_gpu_word(
        q_matrices.size(),
        "GpuParameterData HYBRID q matrix final offset does not fit GpuWord");
    p_matrix_offsets[decomp_count] = checked_gpu_word(
        p_matrices.size(),
        "GpuParameterData HYBRID p matrix final offset does not fit GpuWord");

    auto p_mod_q = copy_mod_operand_array(
        rns_qp->p_mod_qi(),
        base_q_size,
        "GpuParameterData HYBRID p_mod_q constant does not fit GpuWord");
    auto inv_p_mod_q = copy_mod_operand_array(
        rns_qp->p_inv_mod_qi(),
        base_q_size,
        "GpuParameterData HYBRID inv_p_mod_q constant does not fit GpuWord");
    auto p_inv_punctured = copy_mod_operand_array(
        rns_qp->base_p()->inv_punctured_prod_mod_base_array(),
        base_p_size,
        "GpuParameterData HYBRID P inv punctured constant does not fit GpuWord");
    append_base_converter_matrix_padded(
        rns_qp->base_p_to_q_conv(),
        base_p_size,
        moddown_p_to_q_matrix,
        "GpuParameterData HYBRID moddown P-to-Q conversion constant does not fit GpuWord");

    shard.hybrid_decomp_start = DeviceVector<GpuWord>(decomp_start.size(), device_id);
    shard.hybrid_decomp_end = DeviceVector<GpuWord>(decomp_end.size(), device_id);
    shard.hybrid_p_mod_q = DeviceVector<GpuWord>(p_mod_q.size(), device_id);
    shard.hybrid_inv_p_mod_q = DeviceVector<GpuWord>(inv_p_mod_q.size(), device_id);
    shard.hybrid_q_conv_matrix_offsets =
        DeviceVector<GpuWord>(q_matrix_offsets.size(), device_id);
    shard.hybrid_p_conv_matrix_offsets =
        DeviceVector<GpuWord>(p_matrix_offsets.size(), device_id);
    shard.hybrid_q_conv_matrices = DeviceVector<GpuWord>(q_matrices.size(), device_id);
    shard.hybrid_p_conv_matrices = DeviceVector<GpuWord>(p_matrices.size(), device_id);
    shard.hybrid_moddown_p_to_q_matrix =
        DeviceVector<GpuWord>(moddown_p_to_q_matrix.size(), device_id);
    shard.hybrid_qi_inv_punctured =
        DeviceVector<GpuWord>(qi_inv_punctured.size(), device_id);
    shard.hybrid_p_inv_punctured =
        DeviceVector<GpuWord>(p_inv_punctured.size(), device_id);

    shard.hybrid_decomp_start.copy_from_host(decomp_start.data(), decomp_start.size());
    shard.hybrid_decomp_end.copy_from_host(decomp_end.data(), decomp_end.size());
    shard.hybrid_p_mod_q.copy_from_host(p_mod_q.data(), p_mod_q.size());
    shard.hybrid_inv_p_mod_q.copy_from_host(inv_p_mod_q.data(), inv_p_mod_q.size());
    shard.hybrid_qi_inv_punctured.copy_from_host(
        qi_inv_punctured.data(),
        qi_inv_punctured.size());
    shard.hybrid_p_inv_punctured.copy_from_host(
        p_inv_punctured.data(),
        p_inv_punctured.size());
    shard.hybrid_q_conv_matrix_offsets.copy_from_host(
        q_matrix_offsets.data(),
        q_matrix_offsets.size());
    shard.hybrid_p_conv_matrix_offsets.copy_from_host(
        p_matrix_offsets.data(),
        p_matrix_offsets.size());
    if (!q_matrices.empty())
    {
        shard.hybrid_q_conv_matrices.copy_from_host(q_matrices.data(), q_matrices.size());
    }
    if (!p_matrices.empty())
    {
        shard.hybrid_p_conv_matrices.copy_from_host(p_matrices.data(), p_matrices.size());
    }
    if (!moddown_p_to_q_matrix.empty())
    {
        shard.hybrid_moddown_p_to_q_matrix.copy_from_host(
            moddown_p_to_q_matrix.data(),
            moddown_p_to_q_matrix.size());
    }
}

}  // namespace

GpuParameterData::GpuParameterData(const PoseidonContext &context, int device_id)
{
    build_from_poseidon_context(context, device_id);
}

void GpuParameterData::build_from_poseidon_context(
    const PoseidonContext &context,
    int device_id)
{
    levels_.clear();

    auto crt_context = context.crt_context();
    if (!crt_context)
    {
        throw std::invalid_argument("GpuParameterData requires a valid PoseidonContext");
    }
    const auto *small_ntt_tables = crt_context->small_ntt_tables();
    const std::size_t p_ntt_table_offset =
        context.parameters_literal()->q().size();
    const int precomputed_fused_matrix_stages =
        read_precomputed_fused_matrix_stages();

    auto context_data = crt_context->key_context_data();
    if (!context_data)
    {
        context_data = crt_context->first_context_data();
    }
    while (context_data)
    {
        const auto &parms = context_data->parms();
        const auto &q = parms.q();
        const auto &p = parms.p();
        const auto *rns_qp = context_data->qp_rns_tool();
        std::vector<Modulus> parameter_p = p;
        if (parameter_p.empty() && rns_qp != nullptr && rns_qp->base_p() != nullptr)
        {
            parameter_p = copy_rns_base_moduli(*rns_qp->base_p());
        }

        GpuLevelInfo level;
        level.parms_id = context_data->parms_id();
        level.degree = parms.degree();
        level.q_count = q.size();
        level.p_count = p.size();

        GpuParameterShard shard;
        shard.device_id = device_id;
        shard.limb_begin = 0;
        shard.limb_count = level.q_count + parameter_p.size();
        if (!q.empty())
        {
            shard.q_last = checked_gpu_word(
                q.back().value(),
                "GpuParameterData only supports q_last that fits in GpuWord");
            shard.half_q_last = static_cast<GpuWord>(shard.q_last >> 1);
        }

        auto q_words = copy_moduli_to_gpu_words(
            q,
            "GpuParameterData only supports q primes that fit in GpuWord");
        auto p_words = copy_moduli_to_gpu_words(
            parameter_p,
            "GpuParameterData only supports p primes that fit in GpuWord");
        auto q_barrett_ratios = copy_barrett_ratios(q);
        auto p_barrett_ratios = copy_barrett_ratios(parameter_p);
        auto rns_words = concatenate_vectors(q_words, p_words);
        auto rns_barrett_ratios =
            concatenate_vectors(q_barrett_ratios, p_barrett_ratios);

        const auto *p_ntt_tables = small_ntt_tables + p_ntt_table_offset;
        auto q_ntt_roots = copy_ntt_root_operands(
            small_ntt_tables,
            q.size(),
            level.degree,
            false);
        auto p_ntt_roots = copy_ntt_root_operands(
            p_ntt_tables,
            parameter_p.size(),
            level.degree,
            false);
        auto q_intt_roots = copy_ntt_root_operands(
            small_ntt_tables,
            q.size(),
            level.degree,
            true);
        auto p_intt_roots = copy_ntt_root_operands(
            p_ntt_tables,
            parameter_p.size(),
            level.degree,
            true);
        auto q_inv_degree = copy_inv_degree_operands(
            small_ntt_tables,
            q.size());
        auto p_inv_degree = copy_inv_degree_operands(
            p_ntt_tables,
            parameter_p.size());
        auto rns_ntt_roots =
            concatenate_vectors(q_ntt_roots, p_ntt_roots);
        auto rns_intt_roots =
            concatenate_vectors(q_intt_roots, p_intt_roots);
        auto rns_inv_degree =
            concatenate_vectors(q_inv_degree, p_inv_degree);
        const auto rns_ntt_table_ptrs = collect_rns_ntt_table_pointers(
            small_ntt_tables,
            q.size(),
            p_ntt_tables,
            parameter_p.size());
        const auto rns_fused_ntt_matrices =
            load_or_build_fused_matrix_tables(
                rns_ntt_table_ptrs,
                level.degree,
                precomputed_fused_matrix_stages,
                false);
        const auto rns_fused_intt_matrices =
            load_or_build_fused_matrix_tables(
                rns_ntt_table_ptrs,
                level.degree,
                precomputed_fused_matrix_stages,
                true);
        auto half_q_last_mod_q =
            compute_half_q_last_mod_q(q, shard.half_q_last);
        auto inv_q_last_mod_q = copy_inv_q_last_mod_q_operands(
            context_data->rns_tool(),
            q.size());

        shard.q_primes = DeviceVector<GpuWord>(q_words.size(), device_id);
        if (!q_words.empty())
        {
            shard.q_primes.copy_from_host(q_words.data(), q_words.size());
        }

        shard.p_primes = DeviceVector<GpuWord>(p_words.size(), device_id);
        if (!p_words.empty())
        {
            shard.p_primes.copy_from_host(p_words.data(), p_words.size());
        }

        shard.rns_primes = DeviceVector<GpuWord>(rns_words.size(), device_id);
        if (!rns_words.empty())
        {
            shard.rns_primes.copy_from_host(
                rns_words.data(),
                rns_words.size());
        }

        shard.q_modulus_constants =
            DeviceVector<GpuWide>(q_barrett_ratios.size(), device_id);
        if (!q_barrett_ratios.empty())
        {
            shard.q_modulus_constants.copy_from_host(
                q_barrett_ratios.data(),
                q_barrett_ratios.size());
        }

        shard.p_modulus_constants =
            DeviceVector<GpuWide>(p_barrett_ratios.size(), device_id);
        if (!p_barrett_ratios.empty())
        {
            shard.p_modulus_constants.copy_from_host(
                p_barrett_ratios.data(),
                p_barrett_ratios.size());
        }

        shard.rns_modulus_constants =
            DeviceVector<GpuWide>(rns_barrett_ratios.size(), device_id);
        if (!rns_barrett_ratios.empty())
        {
            shard.rns_modulus_constants.copy_from_host(
                rns_barrett_ratios.data(),
                rns_barrett_ratios.size());
        }

        shard.half_q_last_mod_q =
            DeviceVector<GpuWord>(half_q_last_mod_q.size(), device_id);
        if (!half_q_last_mod_q.empty())
        {
            shard.half_q_last_mod_q.copy_from_host(
                half_q_last_mod_q.data(),
                half_q_last_mod_q.size());
        }

        shard.inv_q_last_mod_q =
            DeviceVector<GpuWord>(inv_q_last_mod_q.size(), device_id);
        if (!inv_q_last_mod_q.empty())
        {
            shard.inv_q_last_mod_q.copy_from_host(
                inv_q_last_mod_q.data(),
                inv_q_last_mod_q.size());
        }

        shard.ntt_tables =
            DeviceVector<GpuWord>(rns_ntt_roots.size(), device_id);
        if (!rns_ntt_roots.empty())
        {
            shard.ntt_tables.copy_from_host(
                rns_ntt_roots.data(),
                rns_ntt_roots.size());
        }

        shard.intt_tables =
            DeviceVector<GpuWord>(rns_intt_roots.size(), device_id);
        if (!rns_intt_roots.empty())
        {
            shard.intt_tables.copy_from_host(
                rns_intt_roots.data(),
                rns_intt_roots.size());
        }

        shard.inv_degree_modulo =
            DeviceVector<GpuWord>(rns_inv_degree.size(), device_id);
        if (!rns_inv_degree.empty())
        {
            shard.inv_degree_modulo.copy_from_host(
                rns_inv_degree.data(),
                rns_inv_degree.size());
        }

        shard.ntt_fused_matrix_fusion_stages =
            rns_fused_ntt_matrices.fusion_stages;
        shard.ntt_fused_matrix_stage_counts =
            copy_to_device_vector(
                rns_fused_ntt_matrices.stage_counts,
                device_id);
        shard.ntt_fused_matrix_group_counts =
            copy_to_device_vector(
                rns_fused_ntt_matrices.group_counts,
                device_id);
        shard.ntt_fused_matrix_stage_offsets =
            copy_to_device_vector(
                rns_fused_ntt_matrices.stage_offsets,
                device_id);
        shard.ntt_fused_matrices =
            copy_to_device_vector(
                rns_fused_ntt_matrices.matrices,
                device_id);

        shard.intt_fused_matrix_fusion_stages =
            rns_fused_intt_matrices.fusion_stages;
        shard.intt_fused_matrix_stage_counts =
            copy_to_device_vector(
                rns_fused_intt_matrices.stage_counts,
                device_id);
        shard.intt_fused_matrix_group_counts =
            copy_to_device_vector(
                rns_fused_intt_matrices.group_counts,
                device_id);
        shard.intt_fused_matrix_stage_offsets =
            copy_to_device_vector(
                rns_fused_intt_matrices.stage_offsets,
                device_id);
        shard.intt_fused_matrices =
            copy_to_device_vector(
                rns_fused_intt_matrices.matrices,
                device_id);

        copy_hybrid_key_switch_tables(rns_qp, shard, device_id);

        level.shards.push_back(std::move(shard));
        levels_.push_back(std::move(level));

        context_data = context_data->next_context_data();
    }
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

    throw std::out_of_range("GpuParameterData level not found for parms_id");
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

    throw std::out_of_range("GpuParameterData level not found for parms_id");
}

bool GpuParameterData::empty() const
{
    return levels_.empty();
}

}  // namespace gpu
}  // namespace poseidon
