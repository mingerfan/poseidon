#include "keyload_build_info.h"
#include "poseidon/gpu/gpu_parameter.h"
#include "poseidon/gpu/gpu_uploader.h"
#include "poseidon/keygenerator.h"

#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>
#ifdef __linux__
#include <sched.h>
#include <sys/utsname.h>
#endif

namespace
{
using namespace poseidon;
using namespace poseidon::gpu;
using Clock = std::chrono::steady_clock;

// A small JSON writer keeps this executable independent of ckks-runtime/nlohmann.
struct Json
{
    using Object = std::map<std::string, Json>;
    using Array = std::vector<Json>;
    std::variant<std::nullptr_t, bool, std::int64_t, std::uint64_t, double, std::string, Object, Array> value;
    Json() : value(nullptr) {}
    Json(bool v) : value(v) {}
    Json(int v) : value(static_cast<std::int64_t>(v)) {}
    Json(std::int64_t v) : value(v) {}
    Json(std::uint64_t v) : value(v) {}
    Json(double v) : value(v) {}
    Json(const char *v) : value(std::string(v)) {}
    Json(std::string v) : value(std::move(v)) {}
    Json(Object v) : value(std::move(v)) {}
    Json(Array v) : value(std::move(v)) {}
    Json &operator[](const std::string &key) { return std::get<Object>(value)[key]; }

    static void quote(std::ostream &out, const std::string &text)
    {
        out << '"';
        for (unsigned char c : text)
        {
            switch (c)
            {
            case '"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (c < 32)
                    out << "\\u00" << "0123456789abcdef"[c >> 4]
                        << "0123456789abcdef"[c & 15];
                else out << c;
            }
        }
        out << '"';
    }
    void dump(std::ostream &out) const
    {
        std::visit([&](const auto &v) {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, std::nullptr_t>) out << "null";
            else if constexpr (std::is_same_v<T, bool>) out << (v ? "true" : "false");
            else if constexpr (std::is_same_v<T, std::string>) quote(out, v);
            else if constexpr (std::is_same_v<T, Object>)
            {
                out << '{'; bool first = true;
                for (const auto &[key, child] : v)
                {
                    if (!first) out << ',';
                    first = false; out << '\n'; quote(out, key); out << ": "; child.dump(out);
                }
                if (!v.empty()) out << '\n';
                out << '}';
            }
            else if constexpr (std::is_same_v<T, Array>)
            {
                out << '['; bool first = true;
                for (const auto &child : v)
                {
                    if (!first) out << ',';
                    first = false; child.dump(out);
                }
                out << ']';
            }
            else out << std::setprecision(17) << v;
        }, value);
    }
};

void require(bool ok, const std::string &message)
{
    if (!ok) throw std::runtime_error(message);
}
std::size_t mul(std::size_t a, std::size_t b)
{
    require(a == 0 || b <= std::numeric_limits<std::size_t>::max() / a,
            "byte/shape count overflow");
    return a * b;
}
std::size_t add(std::size_t a, std::size_t b)
{
    require(b <= std::numeric_limits<std::size_t>::max() - a, "byte/shape count overflow");
    return a + b;
}

struct Options
{
    std::size_t log_n = 16, q_count = 50, p_count = 25, rotation_count = 16;
    std::size_t bits = 28;
    int device = 0;
    std::string output = "gpu-keyload.json", parameters = "none", check = "sample";
    std::vector<std::size_t> parameter_q_counts{1, 9, 34, 50};
};
void usage(std::ostream &out)
{
    out << "Usage: poseidon_gpu_keyload [logN q_count p_count rotation_count output.json]\n"
        << "       [--logN N] [--q-count N] [--p-count N] [--rotation-count N] [--json PATH]\n"
        << "       [--bits N] [--device N] [--parameter-mode none|full|sparse|top]\n"
        << "       [--parameter-q-counts 1,9,34,50] [--check sample|full]\n"
        << "Defaults: 16 50 25 16 gpu-keyload.json; bits=28; device=0; parameter-mode=none; check=sample\n"
        << "Rotations are direct left steps 1..rotation_count. Checks cannot be disabled.\n"
        << "WARNING: defaults generate large keys. Set OMP_NUM_THREADS explicitly.\n";
}
std::uint64_t unsigned_arg(const std::string &s, const std::string &name)
{
    require(!s.empty() && std::all_of(s.begin(), s.end(), [](unsigned char c) {
        return c >= '0' && c <= '9';
    }), name + " must be an unsigned decimal integer");
    std::size_t consumed = 0;
    const auto result = std::stoull(s, &consumed);
    require(consumed == s.size(), "invalid " + name);
    return result;
}
Options parse(int argc, char **argv)
{
    Options o;
    std::vector<std::string> positional;
    for (int i = 1; i < argc; ++i)
    {
        const std::string flag = argv[i];
        if (flag.rfind("--", 0) != 0) { positional.push_back(flag); continue; }
        require(i + 1 < argc, "missing value for " + flag);
        const std::string value = argv[++i];
        if (flag == "--json") o.output = value;
        else if (flag == "--parameter-mode" || flag == "--gpu-parameters") o.parameters = value;
        else if (flag == "--parameter-q-counts")
        {
            o.parameter_q_counts.clear();
            std::size_t begin = 0;
            do
            {
                const auto end = value.find(',', begin);
                o.parameter_q_counts.push_back(unsigned_arg(value.substr(begin,
                    end == std::string::npos ? end : end - begin), flag));
                if (end == std::string::npos) break;
                begin = end + 1;
            } while (true);
        }
        else if (flag == "--check") o.check = value;
        else if (flag == "--logN") o.log_n = unsigned_arg(value, flag);
        else if (flag == "--q-count") o.q_count = unsigned_arg(value, flag);
        else if (flag == "--p-count") o.p_count = unsigned_arg(value, flag);
        else if (flag == "--rotation-count") o.rotation_count = unsigned_arg(value, flag);
        else if (flag == "--bits") o.bits = unsigned_arg(value, flag);
        else if (flag == "--device")
        {
            const auto device = unsigned_arg(value, flag);
            require(device <= static_cast<std::uint64_t>(std::numeric_limits<int>::max()),
                    "device index exceeds int range");
            o.device = static_cast<int>(device);
        }
        else throw std::runtime_error("unknown option " + flag);
    }
    require(positional.empty() || positional.size() == 5, "provide either zero or five positional arguments");
    if (!positional.empty())
    {
        o.log_n = unsigned_arg(positional[0], "logN");
        o.q_count = unsigned_arg(positional[1], "q_count");
        o.p_count = unsigned_arg(positional[2], "p_count");
        o.rotation_count = unsigned_arg(positional[3], "rotation_count");
        o.output = positional[4];
    }
    require(o.log_n >= 8 && o.log_n <= 17, "logN must be 8..17 (hwt=192, Poseidon degree limit)");
    require(o.q_count >= 1 && o.q_count <= POSEIDON_COEFF_MOD_COUNT_MAX,
            "q_count is outside Poseidon's supported range");
    require(o.p_count >= 1 && o.p_count <= o.q_count &&
                o.p_count <= POSEIDON_COEFF_MOD_COUNT_MAX - o.q_count,
            "need 1 <= p_count <= q_count and q_count + p_count <= 256");
    require(o.bits >= 2 && o.bits <= 31, "bits must be 2..31 for uint32 GPU residues");
    require(o.rotation_count < (std::size_t{1} << (o.log_n - 1)),
            "rotation_count must be less than N/2 (unique nonidentity left steps)");
    if (o.parameters == "all") o.parameters = "full"; // backwards-compatible spelling
    require(o.parameters == "none" || o.parameters == "top" || o.parameters == "full" || o.parameters == "sparse",
            "--parameter-mode must be none, full, sparse or top");
    std::sort(o.parameter_q_counts.begin(), o.parameter_q_counts.end());
    o.parameter_q_counts.erase(std::unique(o.parameter_q_counts.begin(), o.parameter_q_counts.end()),
                               o.parameter_q_counts.end());
    o.parameter_q_counts.erase(std::remove_if(o.parameter_q_counts.begin(), o.parameter_q_counts.end(),
        [&](std::size_t q) { return q == 0 || q > o.q_count; }), o.parameter_q_counts.end());
    require(o.parameters != "sparse" || !o.parameter_q_counts.empty(),
            "sparse parameter selection has no valid q counts after filtering to 1..Q");
    require(o.check == "sample" || o.check == "full", "--check must be sample or full");
    require(!o.output.empty(), "empty JSON output path");
    return o;
}

std::string read_line(const char *path)
{
    std::ifstream in(path);
    std::string value;
    if (!std::getline(in, value)) return "unavailable";
    return value;
}
Json environment()
{
    int observed_team = 1;
#pragma omp parallel
    {
#pragma omp single
        observed_team = omp_get_num_threads();
    }
    Json env(Json::Object{
        {"hardware_concurrency", static_cast<std::uint64_t>(std::thread::hardware_concurrency())},
        {"openmp_version", _OPENMP}, {"omp_max_threads", omp_get_max_threads()},
        {"omp_observed_probe_team", observed_team}, {"omp_num_procs", omp_get_num_procs()},
        {"omp_dynamic", omp_get_dynamic() != 0}, {"omp_thread_limit", omp_get_thread_limit()},
        {"omp_max_active_levels", omp_get_max_active_levels()},
        {"cgroup_cpu_max", read_line("/sys/fs/cgroup/cpu.max")},
        {"cgroup_v1_cpu_quota_us", read_line("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")},
        {"cgroup_v1_cpu_period_us", read_line("/sys/fs/cgroup/cpu/cpu.cfs_period_us")}
    });
    Json vars(Json::Object{});
    for (const char *name : {"OMP_NUM_THREADS", "OMP_DYNAMIC", "OMP_PROC_BIND", "OMP_PLACES",
             "OMP_THREAD_LIMIT", "OMP_SCHEDULE", "GOMP_CPU_AFFINITY", "KMP_AFFINITY",
             "CUDA_VISIBLE_DEVICES", "CUDA_MODULE_LOADING", "POSEIDON_GPU_KEY_VIEW_CACHE",
             "POSEIDON_NTT_ALGO", "POSEIDON_NTT_FUSED_MATRIX_STAGES",
             "POSEIDON_NTT_FUSED_MATRIX_CACHE_DIR", "POSEIDON_NTT_FUSED_MATRIX_MAX_LEVELS",
             "POSEIDON_NTT_FUSED_MATRIX_FP64_TABLES"})
    {
        const char *raw = std::getenv(name);
        vars[name] = raw ? Json(raw) : Json();
    }
    env["variables"] = std::move(vars);
#ifdef __linux__
    cpu_set_t affinity;
    CPU_ZERO(&affinity);
    if (sched_getaffinity(0, sizeof(affinity), &affinity) == 0)
        env["affinity_cpu_count"] = CPU_COUNT(&affinity);
    struct utsname info{};
    if (uname(&info) == 0)
        env["system"] = std::string(info.sysname) + " " + info.release + " " + info.machine;
#endif
    return env;
}
std::string utc_now()
{
    const auto now = std::time(nullptr);
    std::tm tm{};
#ifdef _WIN32
    gmtime_s(&tm, &now);
#else
    gmtime_r(&now, &tm);
#endif
    std::ostringstream out;
    out << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
    return out.str();
}

class Stages
{
public:
    explicit Stages(Json &report) : report_(report) {}
    template <class F> void run(const char *name, bool gpu, F &&f)
    {
        report_["current_stage"] = name;
        std::cerr << "[gpu_keyload] " << name << " begin" << std::endl;
        std::size_t free_before = 0, total_before = 0;
        // Drain and memory queries are OUTSIDE the measured region.
        if (gpu)
        {
            try
            {
                gpu_check_cuda(cudaDeviceSynchronize(), "pre-stage cudaDeviceSynchronize");
                gpu_check_cuda(cudaMemGetInfo(&free_before, &total_before), "pre-stage cudaMemGetInfo");
            }
            catch (...)
            {
                report_["stages"][name] = Json::Object{{"status", "failed"}, {"wall_ms", Json()},
                    {"failure_phase", "pre_drain_or_memory_query"}};
                std::cerr << "[gpu_keyload] " << name << " failed before timing" << std::endl;
                throw;
            }
            report_["stages"][name]["gpu_memory_before"] = Json::Object{
                {"free_bytes", free_before}, {"total_bytes", total_before}};
        }
        const auto start = Clock::now();
        try
        {
            f();
            if (gpu) gpu_check_cuda(cudaDeviceSynchronize(), "post-stage cudaDeviceSynchronize");
        }
        catch (...)
        {
            finish(name, start, "failed");
            throw;
        }
        finish(name, start, "ok");
        if (gpu)
        {
            std::size_t free_after = 0, total_after = 0;
            gpu_check_cuda(cudaMemGetInfo(&free_after, &total_after), "post-stage cudaMemGetInfo");
            report_["stages"][name]["gpu_memory_before"] = Json::Object{
                {"free_bytes", free_before}, {"total_bytes", total_before}};
            report_["stages"][name]["gpu_memory_after"] = Json::Object{
                {"free_bytes", free_after}, {"total_bytes", total_after}};
            report_["stages"][name]["device_used_delta_bytes"] =
                static_cast<std::int64_t>(free_before) - static_cast<std::int64_t>(free_after);
        }
    }
private:
    void finish(const char *name, Clock::time_point start, const char *status)
    {
        const double ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        report_["stages"][name]["status"] = status;
        report_["stages"][name]["wall_ms"] = ms;
        std::cerr << "[gpu_keyload] " << name << ' ' << status << " wall_ms="
                  << std::fixed << std::setprecision(3) << ms << std::endl;
    }
    Json &report_;
};

struct CpuStats
{
    std::size_t payload = 0, capacity = 0, keys = 0, slots = 0;
    std::size_t ciphertexts = 0, polys = 0, max_decomposition = 0;
};
std::size_t ciphertext_words(const Ciphertext &ct, bool capacity)
{
    return mul(mul(capacity ? ct.size_capacity() : ct.size(), ct.poly_modulus_degree()),
               ct.coeff_modulus_size());
}
CpuStats cpu_stats(const KSwitchKeys &key)
{
    CpuStats s;
    s.keys = key.size(); s.slots = key.data().size();
    for (const auto &digits : key.data())
    {
        s.max_decomposition = std::max(s.max_decomposition, digits.size());
        s.ciphertexts = add(s.ciphertexts, digits.size());
        for (const auto &pk : digits)
        {
            s.payload = add(s.payload, mul(ciphertext_words(pk.data(), false), sizeof(std::uint64_t)));
            s.capacity = add(s.capacity, mul(ciphertext_words(pk.data(), true), sizeof(std::uint64_t)));
            s.polys = add(s.polys, pk.data().size());
        }
    }
    return s;
}
Json cpu_json(const CpuStats &s)
{
    return Json::Object{{"payload_bytes", s.payload}, {"capacity_bytes", s.capacity},
        {"nonempty_key_count", s.keys}, {"key_slot_count", s.slots},
        {"decomposition_ciphertext_count", s.ciphertexts},
        {"max_decomposition_count", s.max_decomposition}, {"polynomial_count", s.polys}};
}
Json gpu_json(const GpuEvaluationKeyData &key)
{
    std::size_t fields = 0, shards = 0;
    std::set<std::size_t> keys;
    for (const auto &field : key.fields_) fields = add(fields, field.buffer.allocated_bytes());
    for (const auto &poly : key.polys_) shards = add(shards, poly.shards.size());
    for (const auto &meta : key.poly_metadata_) keys.insert(meta.key_index);
    const auto tables = add(add(key.galois_key_q0_ptrs.allocated_bytes(), key.galois_key_p0_ptrs.allocated_bytes()),
                            add(key.galois_key_q1_ptrs.allocated_bytes(), key.galois_key_p1_ptrs.allocated_bytes()));
    return Json::Object{{"field_payload_bytes", fields}, {"pointer_table_bytes", tables},
        {"total_allocated_bytes", add(fields, tables)}, {"field_count", key.fields_.size()},
        {"shard_count", shards}, {"polynomial_count", key.polys_.size()},
        {"uploaded_nonempty_key_count", keys.size()}, {"metadata_key_slot_count", key.meta.key_count},
        {"decomposition_count", key.meta.decomposition_count}, {"component_count", key.meta.component_count},
        {"storage_q_count", key.meta.q_count}, {"p_count", key.meta.p_count}, {"degree", key.meta.degree}};
}

// Checks operate on owners, not pointer tables/non-owning views. Every source
// key/digit/component must occur once, and every shard rectangle must be valid.
// Sample mode checks first/middle/last + one deterministic mixed coefficient in
// EVERY limb of EVERY shard; full mode checks every word with <=1 MiB scratch.
Json check_key(const KSwitchKeys &cpu, const GpuEvaluationKeyData &gpu, bool full)
{
    const auto expected = cpu_stats(cpu);
    require(gpu.meta.key_parms_id == cpu.parms_id(), "GPU key parms_id mismatch");
    require(gpu.meta.key_count == expected.slots && gpu.meta.decomposition_count == expected.max_decomposition,
            "GPU key slot/decomposition metadata mismatch");
    require(gpu.polys_.size() == expected.polys && gpu.poly_metadata_.size() == expected.polys,
            "GPU key polynomial coverage mismatch");
    std::set<std::tuple<std::size_t, std::size_t, std::size_t>> seen;
    std::size_t compared = 0, shards_checked = 0;
    std::vector<GpuWord> downloaded;
    constexpr std::size_t chunk_words = (1 << 20) / sizeof(GpuWord);
    for (std::size_t i = 0; i < gpu.polys_.size(); ++i)
    {
        const auto &poly = gpu.polys_[i];
        const auto &meta = gpu.poly_metadata_[i];
        require(meta.poly_id == poly.poly_id && poly.poly_id == i, "GPU polynomial ID mismatch");
        require(seen.emplace(meta.key_index, meta.decomposition_index, meta.component_index).second,
                "duplicate GPU key polynomial");
        const auto &ct = cpu.data().at(meta.key_index).at(meta.decomposition_index).data();
        require(meta.component_index < ct.size(), "GPU component out of CPU range");
        const auto *source = ct.data(meta.component_index);
        const auto source_q = ct.polys().empty() ? ct.coeff_modulus_size()
                                                : ct.polys().at(meta.component_index).rns_num_q();
        const auto source_p = ct.polys().empty() ? 0 : ct.polys().at(meta.component_index).rns_num_p();
        require(poly.degree == ct.poly_modulus_degree() && poly.q_count == source_q && poly.p_count == source_p,
                "GPU/CPU polynomial shape mismatch");
        require(gpu.meta.degree == poly.degree && gpu.meta.q_count == poly.q_count &&
                    gpu.meta.p_count == poly.p_count && gpu.meta.component_count == ct.size(),
                "GPU owner/poly shape mismatch");
        std::size_t covered = 0;
        for (std::size_t j = 0; j < poly.shards.size(); ++j)
        {
            const auto &s = poly.shards[j];
            const auto &field = gpu.fields_.at(s.field_index);
            const auto words = mul(s.limb_count, s.coeff_count);
            require(words != 0 && s.limb_begin <= poly.q_count + poly.p_count &&
                        s.limb_count <= poly.q_count + poly.p_count - s.limb_begin &&
                        s.coeff_begin <= poly.degree && s.coeff_count <= poly.degree - s.coeff_begin,
                    "invalid key shard rectangle");
            require(s.field_offset <= field.size() && words <= field.size() - s.field_offset,
                    "key shard exceeds owning field");
            for (std::size_t k = 0; k < j; ++k)
            {
                const auto &other = poly.shards[k];
                const bool limbs_overlap = s.limb_begin < other.limb_begin + other.limb_count &&
                                           other.limb_begin < s.limb_begin + s.limb_count;
                const bool coeffs_overlap = s.coeff_begin < other.coeff_begin + other.coeff_count &&
                                            other.coeff_begin < s.coeff_begin + s.coeff_count;
                require(!(limbs_overlap && coeffs_overlap), "overlapping logical key shards");
            }
            covered = add(covered, words);
            ++shards_checked;
            gpu_check_cuda(cudaSetDevice(field.device_id), "check cudaSetDevice");
            const auto compare_word = [&](std::size_t packed, GpuWord actual) {
                const auto limb = s.limb_begin + packed / s.coeff_count;
                const auto source_limb = limb < poly.q_count ? limb : source_q + limb - poly.q_count;
                const auto coeff = s.coeff_begin + packed % s.coeff_count;
                const auto value = source[mul(source_limb, poly.degree) + coeff];
                if (value != static_cast<std::uint64_t>(actual))
                {
                    std::ostringstream message;
                    message << "key mismatch key=" << meta.key_index << " digit=" << meta.decomposition_index
                            << " component=" << meta.component_index << " limb=" << limb << " coeff=" << coeff
                            << " CPU=" << value << " GPU=" << actual;
                    throw std::runtime_error(message.str());
                }
                ++compared;
            };
            if (full)
            {
                downloaded.resize(std::min(words, chunk_words));
                for (std::size_t offset = 0; offset < words; offset += downloaded.size())
                {
                    const auto count = std::min(downloaded.size(), words - offset);
                    gpu_check_cuda(cudaMemcpyAsync(downloaded.data(), field.data() + s.field_offset + offset,
                        mul(count, sizeof(GpuWord)), cudaMemcpyDeviceToHost, gpu_execution_stream()), "check full D2H");
                    gpu_check_cuda(cudaStreamSynchronize(gpu_execution_stream()), "check full synchronize");
                    for (std::size_t k = 0; k < count; ++k) compare_word(offset + k, downloaded[k]);
                }
            }
            else
            {
                std::vector<std::size_t> offsets;
                for (std::size_t limb = 0; limb < s.limb_count; ++limb)
                {
                    // Fixed, data-independent sampling; include every Q and P limb.
                    std::set<std::size_t> coeffs{0, s.coeff_count / 2, s.coeff_count - 1,
                        static_cast<std::size_t>((i * 2654435761ULL + limb * 2246822519ULL + j) % s.coeff_count)};
                    for (auto coeff : coeffs) offsets.push_back(mul(limb, s.coeff_count) + coeff);
                }
                downloaded.resize(offsets.size());
                for (std::size_t k = 0; k < offsets.size(); ++k)
                    gpu_check_cuda(cudaMemcpyAsync(&downloaded[k], field.data() + s.field_offset + offsets[k],
                        sizeof(GpuWord), cudaMemcpyDeviceToHost, gpu_execution_stream()), "check sample D2H");
                gpu_check_cuda(cudaStreamSynchronize(gpu_execution_stream()), "check sample synchronize");
                for (std::size_t k = 0; k < offsets.size(); ++k) compare_word(offsets[k], downloaded[k]);
            }
        }
        require(covered == mul(poly.q_count + poly.p_count, poly.degree), "incomplete key shard coverage");
    }
    return Json::Object{{"status", "ok"}, {"compared_words", compared},
        {"checked_polynomials", gpu.polys_.size()}, {"checked_shards", shards_checked},
        {"full_comparison", full}};
}

Json memory_snapshot()
{
    std::size_t free = 0, total = 0;
    gpu_check_cuda(cudaMemGetInfo(&free, &total), "cudaMemGetInfo");
    return Json::Object{{"free_bytes", free}, {"total_bytes", total}};
}
Json moduli_json(const std::vector<Modulus> &moduli)
{
    Json::Array a;
    for (const auto &m : moduli) a.emplace_back(m.value());
    return a;
}

std::string root_fingerprint(const DeviceVector<GpuWord> &table, bool full)
{
    // Cross-process comparison aid, NOT an independent CPU arithmetic oracle.
    // FNV-1a-style uint32-word hash, reported as hex to avoid JSON number rounding.
    std::uint64_t hash = 14695981039346656037ULL;
    const auto mix = [&](GpuWord word) { hash ^= word; hash *= 1099511628211ULL; };
    std::vector<GpuWord> scratch(std::min<std::size_t>(table.size(), (1 << 20) / sizeof(GpuWord)));
    const auto read = [&](std::size_t offset, std::size_t count) {
        gpu_check_cuda(cudaMemcpyAsync(scratch.data(), table.data() + offset, mul(count, sizeof(GpuWord)),
            cudaMemcpyDeviceToHost, gpu_execution_stream()), "parameter fingerprint D2H");
        gpu_check_cuda(cudaStreamSynchronize(gpu_execution_stream()), "parameter fingerprint synchronize");
        for (std::size_t i = 0; i < count; ++i) mix(scratch[i]);
    };
    if (full)
    {
        for (std::size_t offset = 0; offset < table.size(); offset += scratch.size())
            read(offset, std::min(scratch.size(), table.size() - offset));
    }
    else if (!table.empty())
    {
        const auto count = std::min<std::size_t>(16, table.size());
        for (auto offset : std::set<std::size_t>{0, (table.size() - count) / 2, table.size() - count})
            read(offset, count);
    }
    std::ostringstream out;
    out << std::hex << std::setfill('0') << std::setw(16) << hash;
    return out.str();
}
Json parameter_stats(const GpuParameterData &parameters, const PoseidonContext &context, bool full)
{
    using Member = DeviceVector<GpuWord> GpuParameterShard::*;
    const std::pair<const char *, Member> roots[]{
        {"ntt_tables", &GpuParameterShard::ntt_tables},
        {"intt_tables", &GpuParameterShard::intt_tables},
        {"fourstep_intt_tables", &GpuParameterShard::fourstep_intt_tables},
        {"intt_tables_normalized", &GpuParameterShard::intt_tables_normalized},
        {"montgomery_ntt_tables", &GpuParameterShard::montgomery_ntt_tables},
        {"montgomery_intt_tables", &GpuParameterShard::montgomery_intt_tables},
        {"montgomery_intt_tables_normalized", &GpuParameterShard::montgomery_intt_tables_normalized}};
    std::map<std::string, std::size_t> family_bytes;
    Json::Array levels;
    std::size_t total_roots = 0;
    for (auto cd = context.crt_context()->key_context_data(); cd; cd = cd->next_context_data())
    {
        const GpuLevelInfo *level = nullptr;
        try { level = &parameters.get_level(cd->parms_id()); }
        catch (const std::out_of_range &) { continue; } // intentionally absent in sparse mode
        Json::Array shards;
        for (const auto &shard : level->shards)
        {
            gpu_check_cuda(cudaSetDevice(shard.device_id), "parameter fingerprint cudaSetDevice");
            Json families(Json::Object{});
            std::size_t shard_roots = 0;
            for (const auto &[name, member] : roots)
            {
                const auto &table = shard.*member;
                const auto bytes = table.allocated_bytes();
                shard_roots = add(shard_roots, bytes);
                family_bytes[name] = add(family_bytes[name], bytes);
                families[name] = Json::Object{{"allocated_bytes", bytes},
                    {"word_count", table.size()}, {"fingerprint", root_fingerprint(table, full)}};
            }
            total_roots = add(total_roots, shard_roots);
            shards.emplace_back(Json::Object{{"device_id", shard.device_id},
                {"root_limb_count_including_parameter_p", shard.rns_primes.size()},
                {"seven_root_families_total_bytes", shard_roots}, {"root_families", std::move(families)}});
        }
        levels.emplace_back(Json::Object{{"q_count", level->q_count}, {"p_count", level->p_count},
            {"degree", level->degree}, {"shards", std::move(shards)}});
    }
    require(levels.size() == parameters.level_count(), "parameter enumeration missed a level");
    Json totals(Json::Object{});
    for (const auto &[name, bytes] : family_bytes) totals[name] = bytes;
    return Json::Object{{"level_count", parameters.level_count()},
        {"allocated_bytes", parameters.allocated_bytes()}, {"seven_root_families_total_bytes", total_roots},
        {"root_family_bytes", std::move(totals)}, {"levels", std::move(levels)},
        {"fingerprint_mode", full ? "full" : "first_middle_last_16_words"},
        {"fingerprint_algorithm", "64-bit FNV-1a-style uint32-word XOR then multiply, offset=14695981039346656037 prime=1099511628211"},
        {"fingerprint_scope", "cross-process full-vs-sparse root-table comparison; not an independent CPU arithmetic check"}};
}

void benchmark(const Options &o, Json &report)
{
    report["environment"] = environment();
    Json::Array sparse_q_json;
    for (auto q : o.parameter_q_counts) sparse_q_json.emplace_back(q);
    report["parameters"] = Json::Object{{"scheme", "ckks"}, {"sec_level", "none"},
        {"hamming_weight", 192}, {"logN", o.log_n}, {"N", std::size_t{1} << o.log_n},
        {"q_count", o.q_count}, {"p_count", o.p_count}, {"prime_bits", o.bits},
        {"rotation_count", o.rotation_count}, {"rotation_selection", "direct left steps 1..rotation_count"},
        {"gpu_parameter_mode", o.parameters}, {"sparse_q_counts_after_filtering", std::move(sparse_q_json)},
        {"check_mode", o.check},
        {"random_generator", "Poseidon default Blake2xbPRNGFactory, system-generated seeds"}};
    std::cerr << "[gpu_keyload] N=" << (std::size_t{1} << o.log_n) << " Q=" << o.q_count
              << " P=" << o.p_count << " rotations=" << o.rotation_count << " bits=" << o.bits
              << " OMP max_threads=" << omp_get_max_threads() << std::endl;
    Stages stages(report);
    // Direct cudaMalloc resource (no pool): resource lifetime encloses every GPU owner.
    std::unique_ptr<rmm::mr::cuda_memory_resource> resource;
    stages.run("cuda_init", false, [&] {
        gpu_check_cuda(cudaSetDevice(o.device), "cudaSetDevice");
        gpu_check_cuda(cudaFree(nullptr), "CUDA context initialization");
        gpu_check_cuda(cudaDeviceSynchronize(), "CUDA init synchronize");
        resource = std::make_unique<rmm::mr::cuda_memory_resource>();
        cudaDeviceProp prop{};
        gpu_check_cuda(cudaGetDeviceProperties(&prop, o.device), "cudaGetDeviceProperties");
        int driver = 0, runtime = 0;
        gpu_check_cuda(cudaDriverGetVersion(&driver), "cudaDriverGetVersion");
        gpu_check_cuda(cudaRuntimeGetVersion(&runtime), "cudaRuntimeGetVersion");
        char pci[64]{};
        gpu_check_cuda(cudaDeviceGetPCIBusId(pci, sizeof(pci), o.device), "cudaDeviceGetPCIBusId");
        report["cuda"] = Json::Object{{"device_id", o.device}, {"name", prop.name},
            {"compute_major", prop.major}, {"compute_minor", prop.minor}, {"pci_bus_id", pci},
            {"total_global_memory_bytes", static_cast<std::uint64_t>(prop.totalGlobalMem)},
            {"driver_version", driver}, {"runtime_version", runtime},
            {"memory_resource", "rmm::mr::cuda_memory_resource (no pool)"}};
    });
    ScopedGpuMemoryResource resource_scope(resource.get());
    report["gpu_memory"]["after_cuda_init"] = memory_snapshot();
    std::unique_ptr<ParametersLiteral> literal;
    std::unique_ptr<PoseidonContext> context;
    stages.run("cpu_context", false, [&] {
        literal = std::make_unique<ParametersLiteral>(CKKS, o.log_n, o.log_n - 1, o.bits,
            192, 0, Modulus(0), std::vector<Modulus>{}, std::vector<Modulus>{}, sec_level_type::none);
        literal->set_log_modulus(std::vector<std::uint32_t>(o.q_count, o.bits),
                                 std::vector<std::uint32_t>(o.p_count, o.bits));
        context = std::make_unique<PoseidonContext>(*literal, false);
    });
    const char *variant = context->key_switch_variant() == HYBRID ? "HYBRID" :
        context->key_switch_variant() == GHS ? "GHS" : context->key_switch_variant() == BV ? "BV" : "none";
    report["parameters"]["key_switch_variant"] = variant;
    report["parameters"]["q_moduli"] = moduli_json(literal->q());
    report["parameters"]["p_moduli"] = moduli_json(literal->p());

    std::unique_ptr<KeyGenerator> generator;
    PublicKey pk;
    RelinKeys relin;
    GaloisKeys galois;
    stages.run("ordinary_keygen", false, [&] {
        generator = std::make_unique<KeyGenerator>(*context); // includes secret generation
        generator->create_public_key(pk);
    });
    report["cpu_keys"]["secret"] = Json::Object{
        {"payload_bytes", mul(generator->secret_key().data().coeff_count(), sizeof(std::uint64_t))},
        {"capacity_bytes", mul(generator->secret_key().data().capacity(), sizeof(std::uint64_t))},
        {"key_count", 1}};
    report["cpu_keys"]["public"] = Json::Object{
        {"payload_bytes", mul(ciphertext_words(pk.data(), false), sizeof(std::uint64_t))},
        {"capacity_bytes", mul(ciphertext_words(pk.data(), true), sizeof(std::uint64_t))},
        {"key_count", 1}};
    stages.run("relin_generation", false, [&] { generator->create_relin_keys(relin); });
    report["cpu_keys"]["relin"] = cpu_json(cpu_stats(relin));
    std::vector<int> steps;
    for (std::size_t i = 0; i < o.rotation_count; ++i) steps.push_back(static_cast<int>(i + 1));
    Json::Array step_json;
    for (int step : steps) step_json.emplace_back(step);
    report["parameters"]["rotation_steps"] = std::move(step_json);
    stages.run("galois_generation", false, [&] {
        generator->create_galois_keys(steps, galois);
    });
    require(relin.size() == 1 && galois.size() == o.rotation_count, "generated key count mismatch");
    report["cpu_keys"]["galois"] = cpu_json(cpu_stats(galois));
    std::cerr << "[gpu_keyload] CPU eval-key payload bytes: relin=" << cpu_stats(relin).payload
              << " galois=" << cpu_stats(galois).payload << std::endl;

    GpuRelinKeysData gpu_relin;
    GpuGaloisKeysData gpu_galois;
    stages.run("relin_upload", true, [&] {
        gpu_relin = GpuUploader::upload_relin_keys(relin, o.device);
    });
    report["gpu_keys"]["relin"] = gpu_json(gpu_relin);
    stages.run("galois_upload", true, [&] {
        gpu_galois = GpuUploader::upload_galois_keys(galois, o.device);
    });
    report["gpu_keys"]["galois"] = gpu_json(gpu_galois);
    report["gpu_memory"]["after_key_uploads"] = memory_snapshot();

    // No sampling/download, view creation or parameter construction in upload times.
    stages.run("relin_check", true, [&] {
        report["checks"]["relin"] = check_key(relin, gpu_relin, o.check == "full");
    });
    stages.run("galois_check", true, [&] {
        report["checks"]["galois"] = check_key(galois, gpu_galois, o.check == "full");
    });
    report["checks"]["homomorphic_operation_check"] = "not performed (upload equality only)";
    GpuParameterData parameters;
    if (o.parameters != "none")
    {
        std::vector<std::size_t> selected = o.parameters == "top" ? std::vector<std::size_t>{o.q_count}
                                                                 : o.parameter_q_counts;
        Json::Array selected_json;
        if (o.parameters != "full")
            for (auto q : selected) selected_json.emplace_back(q);
        report["parameters"]["parameter_selected_q_counts"] = std::move(selected_json);
        stages.run("gpu_parameters_build_and_upload", true, [&] {
            if (o.parameters == "full") parameters.build_from_poseidon_context(*context, o.device);
            else parameters.build_from_poseidon_context(*context, o.device, selected);
        });
        stages.run("gpu_parameters_fingerprint", true, [&] {
            report["gpu_parameters"] = parameter_stats(parameters, *context, o.check == "full");
        });
        report["gpu_parameters"]["mode"] = o.parameters;
    }
    else
    {
        std::cerr << "[gpu_keyload] gpu_parameters_build_and_upload skipped (mode=none)" << std::endl;
        for (const char *name : {"gpu_parameters_build_and_upload", "gpu_parameters_fingerprint"})
            report["stages"][name] = Json::Object{{"status", "skipped"}, {"wall_ms", Json()}};
        report["gpu_parameters"] = Json::Object{{"mode", "none"}, {"level_count", 0},
            {"allocated_bytes", 0}, {"seven_root_families_total_bytes", 0}};
    }
    report["gpu_memory"]["after_parameter_upload"] = memory_snapshot();
    report["current_stage"] = "completed";
}
} // namespace

int main(int argc, char **argv)
{
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == "--help" || std::string(argv[i]) == "-h") { usage(std::cout); return 0; }
    Options options;
    try { options = parse(argc, argv); }
    catch (const std::exception &e)
    {
        std::cerr << "[gpu_keyload] argument error: " << e.what() << '\n';
        usage(std::cerr);
        return 2;
    }
    // Fail before expensive keygen if the destination is not writable. No directories are created.
    std::ofstream output(options.output, std::ios::out | std::ios::trunc);
    if (!output)
    {
        std::cerr << "[gpu_keyload] cannot open JSON output: " << options.output << std::endl;
        return 2;
    }
    Json report(Json::Object{
        {"schema_version", 1}, {"status", "running"}, {"error", Json()}, {"started_utc", utc_now()},
        {"current_stage", "metadata"}, {"stages", Json::Object{}},
        {"cpu_keys", Json::Object{}}, {"gpu_keys", Json::Object{}},
        {"checks", Json::Object{}}, {"gpu_memory", Json::Object{}},
        {"build", Json::Object{{"source_revision", KEYLOAD_SOURCE_REVISION},
            {"benchmark_source_sha256", KEYLOAD_BENCH_SOURCE_SHA256}, {"build_type", KEYLOAD_BUILD_CONFIGURATION},
            {"cxx_compiler", KEYLOAD_CXX_COMPILER}, {"cuda_architectures", KEYLOAD_CUDA_ARCHITECTURES},
            {"poseidon_use_openmp", KEYLOAD_POSEIDON_USE_OPENMP != 0}, {"poseidon_build_deps", KEYLOAD_POSEIDON_BUILD_DEPS != 0},
            {"poseidon_use_msgsl", KEYLOAD_POSEIDON_USE_MSGSL != 0}, {"poseidon_use_zlib", KEYLOAD_POSEIDON_USE_ZLIB != 0},
            {"poseidon_use_zstd", KEYLOAD_POSEIDON_USE_ZSTD != 0}}},
        {"timing_method", "steady_clock wall time; GPU stages drain before timing and synchronize before stop; single cold run; teardown excluded"},
        {"upload_scope", "full-level standard keys: allocation + CPU uint64-to-uint32 packing + host-to-device copies + synchronization; NOT pure PCIe DMA time"},
        {"gpu_parameter_scope", "host NTT/RNS-table construction, optional fused-matrix cache I/O, device allocation, upload and synchronization; NOT pure PCIe DMA time"},
        {"cpu_byte_scope", "uint64 coefficient payload/capacity only; excludes object/vector metadata, context, keygen scratch and Poseidon memory-pool retained blocks"},
        {"gpu_byte_scope", "live owning fields and device pointer tables; excludes CUDA context, host metadata and allocator overhead"}
    });
    for (const char *stage : {"cuda_init", "cpu_context", "ordinary_keygen", "relin_generation", "galois_generation",
            "relin_upload", "galois_upload", "relin_check", "galois_check", "gpu_parameters_build_and_upload",
            "gpu_parameters_fingerprint"})
        report["stages"][stage] = Json::Object{{"status", "not_run"}, {"wall_ms", Json()}};
    Json::Array args;
    for (int i = 0; i < argc; ++i) args.emplace_back(argv[i]);
    report["argv"] = std::move(args);
    const auto start = Clock::now();
    int result = 0;
    try
    {
        benchmark(options, report);
        report["status"] = "ok";
    }
    catch (const std::exception &e)
    {
        report["status"] = "error";
        report["error"] = e.what();
        std::cerr << "[gpu_keyload] ERROR: " << e.what() << std::endl;
        result = 1;
    }
    report["finished_utc"] = utc_now();
    report["total_wall_ms_including_teardown"] = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    report.dump(output);
    output << '\n';
    output.flush();
    if (!output)
    {
        std::cerr << "[gpu_keyload] JSON write failed: " << options.output << std::endl;
        return 2;
    }
    std::cerr << "[gpu_keyload] JSON written: " << options.output << std::endl;
    return result;
}
