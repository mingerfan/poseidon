// Independent IOGenerator key-load microbench. Never serialize a secret key.
#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits.h>
#include <memory>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <sched.h>
#include <sys/resource.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cuda_runtime_api.h>
#include "cerium/runtime/io/generator.h"
#include "cerium_bench_primes.h"

namespace CR = Cerium::Runtime;
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
using KeyMap = std::unordered_map<std::string, CR::IOGenerator::RnsPolynomialPair>;
constexpr std::size_t kN = 65536;
constexpr std::size_t kLimbBytes = kN * sizeof(CR::Limb::Element_t);
static_assert(sizeof(CR::Limb::Element_t) == 4, "Expected uint32 Cerium limbs");

namespace {
// Only async-signal-safe calls in the handler. SIGKILL/power loss cannot clean up.
char signal_directory[PATH_MAX] = {};
char signal_inputs[PATH_MAX] = {};
char signal_evalkeys[PATH_MAX] = {};
void interrupted(int signal) {
  if (signal_evalkeys[0]) ::unlink(signal_evalkeys);
  if (signal_inputs[0]) ::unlink(signal_inputs);
  if (signal_directory[0]) ::rmdir(signal_directory);
  constexpr char message[] = "cerium_keyload: interrupted; temporary files unlinked\n";
  (void)::write(STDERR_FILENO, message, sizeof(message) - 1);
  ::_exit(128 + signal);
}
void require(bool condition, const std::string &message) {
  if (!condition) throw std::runtime_error(message);
}
void cuda_check(cudaError_t status, const char *operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}
std::string quoted(const std::string &s) {
  std::ostringstream out;
  out << '"';
  for (unsigned char c : s) {
    if (c == '"' || c == '\\') out << '\\' << c;
    else if (c < 0x20) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << unsigned(c);
    else out << c;
  }
  out << '"';
  return out.str();
}
template <class F> double timed(const char *stage, F &&f) {
  std::cerr << "[stage] " << stage << " begin\n" << std::flush;
  const auto start = Clock::now();
  f();
  const double ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
  std::cerr << "[stage] " << stage << " end ms=" << std::fixed << std::setprecision(3)
            << ms << '\n' << std::flush;
  return ms;
}
struct Options {
  std::size_t q = 32, p = 16, k = 8, hwt = 192, ephemeral_hwt = 192;
  int device = 0;
  bool plan_only = false, cpu_only = false;
  std::string tmp_root = "/tmp";
};
std::size_t number(const std::string &value) {
  require(!value.empty() && std::all_of(value.begin(), value.end(), [](char c) {
    return c >= '0' && c <= '9';
  }), "Expected nonnegative integer option");
  std::size_t pos = 0;
  const auto result = std::stoull(value, &pos);
  require(pos == value.size() && result <= SIZE_MAX, "Integer option out of range");
  return static_cast<std::size_t>(result);
}
Options parse_options(int argc, char **argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help") {
      std::cout << "usage: cerium_keyload [--q 32] [--p 16] [--k 8] [--hwt 192]\n"
        "  [--ephemeral-hwt 192] [--device 0] [--tmp-root /tmp]\n"
        "  [--plan-only | --cpu-only]\n"
        "N=65536, bits=28; digits=ceil(Q/P), contiguous P-limb partitions.\n"
        "Rotations are ordinary left steps 1..K, plus one relin.\n"
        "--plan-only: metadata/size check, no keygen and no CUDA calls.\n"
        "--cpu-only: CPU round-trip test using malloc, no CUDA calls; NOT a GPU benchmark.\n";
      std::exit(0);
    }
    if (arg == "--plan-only") { o.plan_only = true; continue; }
    if (arg == "--cpu-only") { o.cpu_only = true; continue; }
    require(i + 1 < argc, "Missing option value for " + arg);
    const std::string value = argv[++i];
    if (arg == "--q") o.q = number(value);
    else if (arg == "--p") o.p = number(value);
    else if (arg == "--k") o.k = number(value);
    else if (arg == "--hwt") o.hwt = number(value);
    else if (arg == "--ephemeral-hwt") o.ephemeral_hwt = number(value);
    else if (arg == "--device") {
      const auto device = number(value);
      require(device <= INT_MAX, "device out of range");
      o.device = static_cast<int>(device);
    }
    else if (arg == "--tmp-root") o.tmp_root = value;
    else throw std::runtime_error("Unknown option " + arg);
  }
  require(o.q >= 1 && o.p >= 1 && o.q <= 64 && o.p <= 64 && o.q + o.p <= 64,
          "Require Q>=1, P>=1 and Q+P<=64");
  require(o.k < kN / 2, "Require 0<=K<32768 (unique nonidentity rotations)");
  require(o.hwt >= 1 && o.hwt <= kN && o.ephemeral_hwt >= 1 && o.ephemeral_hwt <= kN,
          "Require secret/ephemeral Hamming weights in [1,65536]");
  require(!(o.cpu_only && o.plan_only), "Choose at most one validation-only mode");
  return o;
}
struct ScratchFiles {
  std::string directory, inputs, evalkeys;
  bool cleaned = false;
  explicit ScratchFiles(const std::string &root) {
    const auto pattern = (fs::absolute(root) / "cerium-keyload.XXXXXX").string();
    require(pattern.size() + 32 < PATH_MAX, "Temporary path too long");
    std::vector<char> buffer(pattern.begin(), pattern.end());
    buffer.push_back('\0');
    char *created = ::mkdtemp(buffer.data());
    require(created != nullptr, "mkdtemp failed (check --tmp-root)");
    directory = created;
    inputs = directory + "/program_inputs";
    evalkeys = directory + "/evalkeys.bin";
    std::strcpy(signal_directory, directory.c_str());
    std::strcpy(signal_inputs, inputs.c_str());
    std::strcpy(signal_evalkeys, evalkeys.c_str());
    std::cerr << "[temporary] " << directory << " (0700; removed on exit)\n" << std::flush;
  }
  bool cleanup() noexcept {
    if (cleaned) return true;
    bool ok = true;
    for (const auto &file : {evalkeys, inputs}) {
      if (::unlink(file.c_str()) != 0 && errno != ENOENT) ok = false;
    }
    if (::rmdir(directory.c_str()) != 0 && errno != ENOENT) ok = false;
    if (ok) {
      cleaned = true;
      signal_directory[0] = signal_inputs[0] = signal_evalkeys[0] = 0;
    }
    return ok;
  }
  ~ScratchFiles() {
    if (!cleanup()) std::cerr << "[cleanup] WARNING: remove temporary directory " << directory << '\n';
  }
};
std::string indices(std::size_t begin, std::size_t end) {
  std::ostringstream s;
  s << '[';
  for (auto i = begin; i < end; ++i) { if (i != begin) s << ','; s << i; }
  s << ']';
  return s.str();
}
struct Inventory {
  std::vector<std::string> partitions;
  std::set<std::string> ids;
  std::size_t terms = 0, pairs = 0, payload = 0, file_bytes = 0;
};
Inventory write_inputs(const Options &o, const std::string &path) {
  Inventory inv;
  for (std::size_t begin = 0; begin < o.q; begin += o.p)
    inv.partitions.push_back(indices(begin, std::min(o.q, begin + o.p)));
  std::ofstream out(path);
  out.exceptions(std::ios::badbit | std::ios::failbit);
  out << "Evalkey Stream:\n";
  for (std::size_t key = 0; key <= o.k; ++key) {
    const std::size_t type = key == 0 ? 0 : 1; // Mul, ordinary Rot (not RotInv).
    for (const auto &partition : inv.partitions) {
      const auto prefix = std::to_string(type) + ':' + std::to_string(o.q) + ':' +
                          std::to_string(o.p) + ':' + std::to_string(key) + ':';
      require(inv.ids.insert(prefix + partition).second, "Duplicate planned key id");
      for (int c = 0; c < 2; ++c) {
        out << 'k' << inv.terms++ << " | K:" << prefix << 'c' << c << ':' << partition
            << " | " << indices(0, o.q + o.p) << '\n';
      }
    }
  }
  out << ";\n";
  out.close();
  inv.pairs = inv.ids.size();
  inv.payload = inv.pairs * 2 * (o.q + o.p) * kLimbBytes;
  // Cerium native format: NUL-terminated header, native size_t and uint64 IDs.
  inv.file_bytes = CR::IOGenerator::evalkey_header.size() + 1 + 4 * sizeof(std::size_t)
                   + (o.q + o.p) * sizeof(std::uint64_t) + inv.payload;
  for (const auto &id : inv.ids)
    inv.file_bytes += 2 * sizeof(std::size_t) + id.size() + 1
                      + 2 * (o.q + o.p) * sizeof(std::uint64_t);
  return inv;
}
std::vector<std::uint64_t> select_primes(const Options &o) {
  std::vector<std::uint64_t> result(kCeriumPrimes.begin(), kCeriumPrimes.begin() + o.q + o.p);
  std::set<std::uint64_t> unique;
  for (auto prime : result) {
    require(prime >= (1ULL << 27) && prime < (1ULL << 28) && prime % (2 * kN) == 1,
            "Prime is not a 28-bit N=65536 NTT modulus");
    require(unique.insert(prime).second, "Duplicate prime");
    for (std::uint64_t divisor = 2; divisor * divisor <= prime; ++divisor)
      require(prime % divisor != 0, "Composite modulus in public prime list");
  }
  return result;
}
void wipe(std::vector<std::int64_t> &data) {
  volatile std::int64_t *p = data.data();
  for (std::size_t i = 0; i < data.size(); ++i) p[i] = 0;
}
std::vector<std::int64_t> synthetic_key(std::size_t hwt, std::mt19937_64 &rng) {
  std::vector<std::size_t> positions(kN);
  std::iota(positions.begin(), positions.end(), 0);
  std::vector<std::int64_t> key(kN, 0);
  for (std::size_t i = 0; i < hwt; ++i) {
    std::uniform_int_distribution<std::size_t> choose(i, kN - 1);
    std::swap(positions[i], positions[choose(rng)]);
    key[positions[i]] = (rng() & 1) ? 1 : -1;
  }
  return key;
}
class HostPool final : public CR::MemoryManager {
public:
  void *base = nullptr;
  std::size_t bytes;
  bool pinned;
  HostPool(std::size_t n, bool pin) : bytes(n), pinned(pin) { initialize_memory_pool(n); }
private:
  std::unique_ptr<std::uint8_t, std::function<void(void *)>> allocate_memory(std::size_t n) override {
    if (pinned) cuda_check(cudaMallocHost(&base, n), "cudaMallocHost");
    else { base = std::malloc(n); require(base != nullptr, "malloc host pool failed"); }
    const bool pin = pinned;
    return {static_cast<std::uint8_t *>(base), [pin](void *p) {
      if (pin) (void)cudaFreeHost(p); else std::free(p);
    }};
  }
};
struct DeviceResources {
  void *pool = nullptr;
  cudaStream_t stream = nullptr;
  ~DeviceResources() {
    if (stream) (void)cudaStreamSynchronize(stream);
    if (pool) (void)cudaFree(pool);
    if (stream) (void)cudaStreamDestroy(stream);
  }
};
std::size_t audit_map(const KeyMap &keys, const Options &o, const Inventory &inv) {
  require(keys.size() == inv.pairs, "Actual keypair count disagrees with planned IDs");
  std::size_t bytes = 0;
  for (const auto &[id, pair] : keys) {
    require(inv.ids.count(id) == 1, "Unexpected actual key ID");
    for (const auto &poly : {pair.first, pair.second}) {
      require(poly != nullptr && poly->num_limbs() == o.q + o.p, "Wrong QP limb count");
      for (std::size_t i = 0; i < o.q + o.p; ++i) {
        const auto &limb = poly->at(i);
        require(limb->size() == kN && limb->rns_base_id() == i && limb->is_ntt_form(),
                "Wrong limb size, RNS ID, or NTT representation");
        bytes += limb->size() * sizeof(CR::Limb::Element_t);
      }
    }
  }
  require(bytes == inv.payload, "Actual payload bytes disagree with expected shape");
  return bytes;
}
void compare_maps(const KeyMap &original, const KeyMap &loaded) {
  require(original.size() == loaded.size(), "Round-trip keypair count mismatch");
  for (const auto &[id, pair] : original) {
    const auto &other = loaded.at(id);
    for (int c = 0; c < 2; ++c) {
      const auto &left = c == 0 ? pair.first : pair.second;
      const auto &right = c == 0 ? other.first : other.second;
      for (const auto &[limb_id, limb] : left->limb_index_map())
        require(std::memcmp(limb->data(), right->at(limb_id)->data(), kLimbBytes) == 0,
                "Serialization round-trip mismatch (values deliberately not printed)");
    }
  }
}
std::vector<CR::LimbPtr> pool_order(const KeyMap &keys, const HostPool &pool) {
  std::vector<CR::LimbPtr> limbs;
  for (const auto &[id, pair] : keys) {
    (void)id;
    for (const auto &poly : {pair.first, pair.second})
      for (const auto &[limb_id, limb] : poly->limb_index_map()) {
        (void)limb_id;
        limbs.push_back(limb);
      }
  }
  std::sort(limbs.begin(), limbs.end(), [](const auto &a, const auto &b) {
    return reinterpret_cast<std::uintptr_t>(a->data()) < reinterpret_cast<std::uintptr_t>(b->data());
  });
  auto *base = static_cast<const std::uint8_t *>(pool.base);
  require(limbs.size() * kLimbBytes == pool.bytes, "Host pool byte count mismatch");
  for (std::size_t i = 0; i < limbs.size(); ++i)
    require(static_cast<const void *>(limbs[i]->data()) == base + i * kLimbBytes,
            "Deserialized limbs are not exact disjoint slices of the single pool");
  return limbs;
}
std::string small_file(const char *path) {
  std::ifstream in(path);
  std::string text;
  std::getline(in, text);
  return text;
}
void write_shape(std::ostream &out, const Options &o, const Inventory &inv) {
  out << "\"n\":" << kN << ",\"q\":" << o.q << ",\"p\":" << o.p
      << ",\"bits\":28,\"context_prime_count\":" << o.q + o.p
      << ",\"digits\":" << inv.partitions.size() << ",\"rotation_count\":" << o.k
      << ",\"rotation_steps\":" << indices(1, o.k + 1)
      << ",\"rotation_type\":\"ordinary_Rot_left\",\"relin_count\":1,\"digit_partitions\":[";
  for (std::size_t i = 0; i < inv.partitions.size(); ++i) { if (i) out << ','; out << inv.partitions[i]; }
  out << "],\"q_rns_ids\":" << indices(0, o.q) << ",\"p_rns_ids\":" << indices(o.q, o.q + o.p)
      << ",\"metadata_term_count\":" << inv.terms << ",\"keypair_count\":" << inv.pairs
      << ",\"payload_bytes\":" << inv.payload << ",\"expected_file_bytes\":" << inv.file_bytes;
}
} // namespace

int main(int argc, char **argv) {
  // Cerium Logger binds cout.rdbuf() at construction; route its logs to stderr.
  // Keep a separate stdout stream solely for our JSON, never key coefficients.
  std::ostream json(std::cout.rdbuf());
  auto *saved_cout = std::cout.rdbuf(std::cerr.rdbuf());
  int exit_code = 0;
  try {
    // Help remains human-readable on stdout without creating key material.
    if (argc == 2 && std::string(argv[1]) == "--help") {
      std::cout.rdbuf(saved_cout);
      (void)parse_options(argc, argv);
    }
    const Options o = parse_options(argc, argv);
    ::umask(0077);
    struct rlimit no_core = {0, 0};
    require(::setrlimit(RLIMIT_CORE, &no_core) == 0, "Cannot set core dump limit");
    // Pipe-based core handlers may ignore RLIMIT_CORE; also mark nondumpable.
    require(::prctl(PR_SET_DUMPABLE, 0, 0, 0, 0) == 0, "Cannot disable process dumpability");
    struct sigaction action {};
    action.sa_handler = interrupted;
    ::sigemptyset(&action.sa_mask);
    for (int signal : {SIGINT, SIGTERM, SIGHUP, SIGABRT})
      require(::sigaction(signal, &action, nullptr) == 0, "Cannot install cleanup handler");
    const auto primes = select_primes(o);
    ScratchFiles scratch(o.tmp_root);
    const Inventory inv = write_inputs(o, scratch.inputs);
    if (o.plan_only) {
      require(scratch.cleanup(), "Temporary file cleanup failed");
      json << "{\"ok\":true,\"mode\":\"plan_only\",\"gpu_calls\":false,";
      write_shape(json, o, inv);
      json << ",\"temporary_directory\":" << quoted(scratch.directory)
           << ",\"temporary_files_removed\":true,\"keys_generated\":false}\n" << std::flush;
      std::cout.rdbuf(saved_cout);
      return 0;
    }
    double cuda_init_ms = 0;
    int runtime_version = 0, driver_version = 0;
    cudaDeviceProp device_prop {};
    DeviceResources device;
    if (!o.cpu_only) cuda_init_ms = timed("cuda_initialize (excluded from H2D)", [&] {
      cuda_check(cudaSetDevice(o.device), "cudaSetDevice");
      cuda_check(cudaFree(nullptr), "initialize CUDA context");
      cuda_check(cudaRuntimeGetVersion(&runtime_version), "cudaRuntimeGetVersion");
      cuda_check(cudaDriverGetVersion(&driver_version), "cudaDriverGetVersion");
      cuda_check(cudaGetDeviceProperties(&device_prop, o.device), "cudaGetDeviceProperties");
      cuda_check(cudaStreamCreateWithFlags(&device.stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
      cuda_check(cudaStreamSynchronize(device.stream), "initial cudaStreamSynchronize");
    });
    std::unique_ptr<CR::Context> context;
    const double context_ms = timed("Context", [&] { context = std::make_unique<CR::Context>(primes); });
    require(context->n() == kN, "Runtime N differs from fixed 65536");
    // Synthetic benchmark keys only: random_device-seeded MT19937, no saved/logged seed.
    std::random_device entropy;
    std::array<std::uint32_t, 8> entropy_words;
    for (auto &word : entropy_words) word = entropy();
    std::seed_seq seed(entropy_words.begin(), entropy_words.end());
    std::mt19937_64 rng(seed);
    auto secret = synthetic_key(o.hwt, rng);
    auto ephemeral = synthetic_key(o.ephemeral_hwt, rng);
    std::unique_ptr<CR::CKKSEncryptorContext> encryptor;
    const double encryptor_ms = timed("CKKSEncryptorContext", [&] {
      encryptor = std::make_unique<CR::CKKSEncryptorContext>(*context, secret, ephemeral);
    });
    wipe(secret);
    wipe(ephemeral);
    CR::IOGenerator generator(*context);
    // Check the actual public parser, IDs, component references and RNS lists.
    {
      std::ifstream metadata(scratch.inputs);
      std::string line;
      std::getline(metadata, line);
      std::set<std::string> parsed_ids;
      std::size_t terms = 0;
      while (std::getline(metadata, line) && line != ";") {
        auto parsed = generator.parse_evalkey(line);
        const auto &info = std::get<0>(parsed.second);
        require(inv.ids.count(info.id) == 1 && info.level == o.q && info.extension_size == o.p
                && info.ct_number == terms % 2 && info.rns_bases.size() == o.q + o.p,
                "Public EvalkeyInfo parser disagrees with generated metadata");
        parsed_ids.insert(info.id);
        ++terms;
      }
      require(parsed_ids == inv.ids && terms == inv.terms && line == ";", "Metadata inventory mismatch");
    }
    std::pair<KeyMap, std::size_t> generated;
    const double generate_ms = timed("generate_evalkeys", [&] {
      generated = generator.generate_evalkeys(scratch.inputs, *encryptor);
    });
    audit_map(generated.first, o, inv);
    require(generated.second * sizeof(CR::Limb::Element_t) == inv.payload,
            "IOGenerator element count disagrees with audited payload bytes");
    // Destroy the encryptor and its private SK copies outside all named timings.
    encryptor.reset();
    const double serialize_ms = timed("serialize_evalkeys (close/flush; no fsync)", [&] {
      generator.serialize_evalkeys(scratch.evalkeys, generated.first, generated.second);
    });
    const auto actual_file_bytes = fs::file_size(scratch.evalkeys);
    require(actual_file_bytes == inv.file_bytes, "Serialization file size mismatch (possible write failure)");
    std::shared_ptr<HostPool> host_pool;
    double host_alloc_ms = 0;
    std::size_t callback_count = 0;
    KeyMap loaded;
    const double deserialize_ms = timed("deserialize_evalkeys (includes host pool allocation)", [&] {
      loaded = CR::IOGenerator::deserialize_evalkeys(*context, scratch.evalkeys,
        [&](std::size_t bytes) -> CR::MemoryManagerHandle {
          require(++callback_count == 1 && bytes == inv.payload, "Unexpected pool request");
          const auto start = Clock::now();
          host_pool = std::make_shared<HostPool>(bytes, !o.cpu_only);
          host_alloc_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
          return host_pool;
        });
    });
    require(callback_count == 1 && host_pool != nullptr, "Deserializer did not create exactly one host pool");
    audit_map(loaded, o, inv);
    const double host_verify_ms = timed("verify serialized round-trip (exact; untimed for load)", [&] {
      compare_maps(generated.first, loaded);
    });
    generated.first.clear(); // Excluded from generation/serialization/deserialization timing.
    auto limbs = pool_order(loaded, *host_pool); // Prepare all descriptors outside H2D.
    double device_alloc_ms = 0, h2d_ms = 0, gpu_verify_ms = 0;
    if (!o.cpu_only) {
      device_alloc_ms = timed("cudaMalloc single device pool (excluded from H2D)", [&] {
        cuda_check(cudaMalloc(&device.pool, inv.payload), "cudaMalloc device pool");
      });
      h2d_ms = timed("H2D (per-limb cudaMemcpyAsync + final stream sync)", [&] {
        auto *dest = static_cast<std::uint8_t *>(device.pool);
        for (std::size_t i = 0; i < limbs.size(); ++i)
          cuda_check(cudaMemcpyAsync(dest + i * kLimbBytes, limbs[i]->data(), kLimbBytes,
                                     cudaMemcpyHostToDevice, device.stream), "cudaMemcpyAsync H2D");
        cuda_check(cudaStreamSynchronize(device.stream), "H2D final cudaStreamSynchronize");
      });
      // A separate ONE-limb pinned buffer; never overwrite the source host pool.
      gpu_verify_ms = timed("verify GPU readback (all limbs exact; excluded from H2D)", [&] {
        HostPool readback(kLimbBytes, true);
        const auto *src = static_cast<const std::uint8_t *>(device.pool);
        for (std::size_t i = 0; i < limbs.size(); ++i) {
          cuda_check(cudaMemcpyAsync(readback.base, src + i * kLimbBytes, kLimbBytes,
                                     cudaMemcpyDeviceToHost, device.stream), "cudaMemcpyAsync verify D2H");
          cuda_check(cudaStreamSynchronize(device.stream), "verify D2H sync");
          require(std::memcmp(readback.base, limbs[i]->data(), kLimbBytes) == 0,
                  "GPU readback mismatch (values deliberately not printed)");
        }
      });
    }
    require(scratch.cleanup(), "Temporary file cleanup failed");
    cpu_set_t affinity;
    CPU_ZERO(&affinity);
    const int cpu_count = ::sched_getaffinity(0, sizeof(affinity), &affinity) == 0 ? CPU_COUNT(&affinity) : -1;
    const char *omp = std::getenv("OMP_NUM_THREADS");
    json << std::fixed << std::setprecision(6)
         << "{\"ok\":true,\"mode\":" << quoted(o.cpu_only ? "cpu_validation_only" : "gpu_keyload")
         << ",\"gpu_calls\":" << (o.cpu_only ? "false" : "true") << ',';
    write_shape(json, o, inv);
    json << ",\"actual_keypair_count\":" << loaded.size()
         << ",\"actual_payload_bytes\":" << host_pool->bytes << ",\"actual_file_bytes\":" << actual_file_bytes
         << ",\"limb_count\":" << limbs.size() << ",\"limb_bytes\":" << kLimbBytes
         << ",\"secret_hwt\":" << o.hwt << ",\"ephemeral_hwt\":" << o.ephemeral_hwt
         << ",\"secret_sampling\":\"synthetic_sparse_ternary_random_device_seeded_mt19937_64\""
         << ",\"generator_threads\":64,\"cpu_affinity_count\":" << cpu_count
         << ",\"cgroup_cpu_max\":" << quoted(small_file("/sys/fs/cgroup/cpu.max"))
         << ",\"omp_num_threads\":" << (omp ? quoted(omp) : "null")
         << ",\"host_pool\":" << quoted(o.cpu_only ? "malloc_CPU_TEST_ONLY" : "cudaMallocHost")
         << ",\"host_pool_count\":1,\"device_pool_count\":" << (o.cpu_only ? 0 : 1)
         << ",\"serialize_fsync\":false,\"cache_condition\":\"immediate_reread_after_write_page_cache_likely_warm_no_eviction\""
         << ",\"h2d_timing\":\"wall_clock_memcpy_submissions_and_one_final_stream_sync_no_alloc_no_validation\""
         << ",\"h2d_warmup_runs\":0,\"h2d_iterations\":" << (o.cpu_only ? 0 : 1)
         << ",\"host_roundtrip_verified_before_h2d\":true"
         << ",\"timing_ms\":{\"Context\":" << context_ms << ",\"CKKSEncryptorContext\":" << encryptor_ms
         << ",\"generate_evalkeys\":" << generate_ms << ",\"serialize_evalkeys_no_fsync\":" << serialize_ms
         << ",\"deserialize_evalkeys_including_host_alloc\":" << deserialize_ms
         << ",\"host_pool_alloc_subset_of_deserialize\":" << host_alloc_ms
         << ",\"verify_host_roundtrip\":" << host_verify_ms
         << ",\"cuda_initialize\":" << (o.cpu_only ? "null" : std::to_string(cuda_init_ms))
         << ",\"device_pool_alloc\":" << (o.cpu_only ? "null" : std::to_string(device_alloc_ms))
         << ",\"h2d\":" << (o.cpu_only ? "null" : std::to_string(h2d_ms))
         << ",\"verify_gpu_readback\":" << (o.cpu_only ? "null" : std::to_string(gpu_verify_ms)) << '}'
         << ",\"h2d_GB_per_s\":" << (o.cpu_only ? "null" : std::to_string(inv.payload / (h2d_ms * 1.0e6)))
         << ",\"cuda_runtime_version\":" << runtime_version << ",\"cuda_driver_version\":" << driver_version
         << ",\"gpu_name\":" << (o.cpu_only ? "null" : quoted(device_prop.name))
         << ",\"verified_host_exact\":true,\"verified_gpu_exact\":" << (o.cpu_only ? "false" : "true")
         << ",\"temporary_directory\":" << quoted(scratch.directory)
         << ",\"temporary_files_removed\":true,\"secret_key_serialized\":false}\n" << std::flush;
  } catch (const std::exception &e) {
    std::cerr << "[error] " << e.what() << '\n' << std::flush;
    json << "{\"ok\":false,\"error\":" << quoted(e.what())
         << ",\"temporary_files_removed\":" << (signal_directory[0] == 0 ? "true" : "false")
         << ",\"secret_key_serialized\":false}\n" << std::flush;
    exit_code = 1;
  }
  std::cout.rdbuf(saved_cout);
  return exit_code;
}
