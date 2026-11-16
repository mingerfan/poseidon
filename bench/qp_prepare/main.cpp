// Exact public-constant regression probe; never generates or prints keys.
#include "poseidon/advance/homomorphic_linear_transform.h"
#include "poseidon/basics/util/ntt.h"
#include "poseidon/gpu/gpu_uploader.h"
#include <gmpxx.h>
#include <algorithm>
#include <bit>
#include <cstdlib>
#include <future>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

using namespace poseidon;
using namespace poseidon::gpu;
using Words = std::vector<GpuWord>;
static_assert(sizeof(unsigned long) >= sizeof(std::uint64_t), "GMP ui oracle needs 64-bit unsigned long");

static void require(bool ok, const std::string &message) {
    if (!ok) throw std::runtime_error(message);
}
struct Env {
    std::string name;
    std::optional<std::string> old;
    Env(const char *key, const char *value) : name(key) {
        if (const auto *v = std::getenv(key)) old = v;
        require(setenv(key, value, 1) == 0, "setenv failed");
    }
    ~Env() { if (old) setenv(name.c_str(), old->c_str(), 1); else unsetenv(name.c_str()); }
};
static std::size_t reversed(std::size_t i, std::size_t n) {
    std::size_t r = 0;
    for (; n > 1; n >>= 1, i >>= 1) r = (r << 1) | (i & 1);
    return r;
}
static mpz_class centered(const mpz_class &integer, const mpz_class &q) {
    mpz_class x;
    mpz_mod(x.get_mpz_t(), integer.get_mpz_t(), q.get_mpz_t());
    if (x > q / 2) x -= q;
    return x;
}
static void equal_words(const Words &got, const Words &want, const std::string &where) {
    require(got.size() == want.size(), where + ": word count");
    const auto mismatch = std::mismatch(got.begin(), got.end(), want.begin());
    if (mismatch.first != got.end()) {
        const auto i = mismatch.first - got.begin();
        throw std::runtime_error(where + ": word " + std::to_string(i) + " got " +
                                 std::to_string(*mismatch.first) + " expected " + std::to_string(*mismatch.second));
    }
}
static std::size_t oracle_period(const Words &words, std::size_t n) {
    for (std::size_t t = 1; t < n; t <<= 1) {
        bool matches = true;
        for (std::size_t limb = 0; matches && limb < words.size() / n; ++limb)
            for (std::size_t j = 0; j < n; ++j)
                if (words[limb*n+j] != words[limb*n+reversed(reversed(j,n)&(t-1),n)]) {
                    matches = false; break;
                }
        if (matches) return t;
    }
    return n;
}
struct Case {
    std::string name;
    Plaintext source;
    Words oracle;
    std::size_t period = 0;
};
// Oracle starts with actual signed integers, centers by exact GMP arithmetic,
// reduces independently into every prime, and uses the library's CPU NTT only.
static Case make_case(const std::string &name, const std::vector<mpz_class> &integers,
                      const PoseidonContext &context, parms_id_type id, bool lazy = false) {
    const auto level = context.crt_context()->get_context_data(id);
    const auto key = context.crt_context()->key_context_data();
    const auto &q = level->parms().q();
    const auto &p = key->parms().p();
    const auto n = integers.size();
    mpz_class product = 1;
    for (const auto &m : q) product *= m.value();
    std::vector<mpz_class> coefficients;
    for (const auto &v : integers) coefficients.push_back(centered(v, product));
    Case result;
    result.name = name;
    result.source.resize(q.size() * n);
    result.source.scale() = 1024.0; // small integer coefficients also represent signed fractions
    result.oracle.resize((q.size() + p.size()) * n);
    const auto *tables = context.crt_context()->small_ntt_tables();
    for (std::size_t limb = 0; limb < q.size() + p.size(); ++limb) {
        const auto &m = limb < q.size() ? q[limb] : p[limb - q.size()];
        std::vector<std::uint64_t> residues(n);
        for (std::size_t j = 0; j < n; ++j)
            residues[j] = mpz_fdiv_ui(coefficients[j].get_mpz_t(), m.value());
        const auto table = limb < q.size() ? limb : key->parms().q().size() + limb - q.size();
        util::ntt_negacyclic_harvey(residues.data(), tables[table]);
        for (std::size_t j = 0; j < n; ++j) {
            require(residues[j] < m.value(), "CPU NTT must be canonical");
            result.oracle[limb*n+j] = static_cast<GpuWord>(residues[j]);
            if (limb < q.size())
                result.source.data()[limb*n+j] = residues[j] + (lazy && j%3 != 0 ? m.value() : 0);
        }
        // This also establishes that lazy Q-NTT inputs are inside the original
        // inverse transform's [0,2q) contract and preserve exact coefficients.
        if (limb < q.size()) std::copy_n(result.source.data()+limb*n, n, residues.data());
        util::inverse_ntt_negacyclic_harvey(residues.data(), tables[table]);
        for (std::size_t j = 0; j < n; ++j)
            require(residues[j] == mpz_fdiv_ui(coefficients[j].get_mpz_t(), m.value()),
                    name + ": CPU inverse NTT/oracle mismatch");
    }
    result.source.parms_id() = id;
    result.period = oracle_period(result.oracle, n);
    return result;
}
static std::vector<Case> cases(const PoseidonContext &context, parms_id_type id) {
    const auto level = context.crt_context()->get_context_data(id);
    const auto &q = level->parms().q();
    const auto &p = context.crt_context()->key_context_data()->parms().p();
    const auto n = level->parms().degree();
    mpz_class product = 1, pair_product = q[0].value();
    for (const auto &m : q) product *= m.value();
    if (q.size() > 1) pair_product *= q[1].value();
    const mpz_class half = product/2;
    std::vector<Case> result;
    auto add = [&](const char *name, const std::vector<mpz_class> &v, bool lazy = false) {
        result.push_back(make_case(name, v, context, id, lazy));
    };
    std::vector<mpz_class> v(n);
    add("zero", v);
    for (std::size_t i = 0; i < n; ++i) v[i] = static_cast<long>(i%35)-17;
    add("signed_small", v);
    add("lazy_signed_small", v, true);
    v.assign(n, 0);
    v[0] = p[0].value(); v[1] = -v[0];
    v[2] = p[1].value(); v[3] = -v[2];
    v[4] = mpz_class(p[0].value()) * p[1].value(); v[5] = -v[4];
    v[6] = q[0].value(); v[7] = -v[6];
    // For Q>=2, +/-P0 are genuinely centered and their P0 residues must be 0.
    if (q.size() >= 2) require(centered(v[0], product) == v[0], "P-divisible sample not centered");
    add("p_divisible", v);
    v.assign(n, 0);
    std::vector<mpz_class> boundary = {half-1, half, half+1, half+2, -half, -half-1,
                                     product-1, product, product+1, -product-1};
    std::copy(boundary.begin(), boundary.end(), v.begin());
    add("full_Q_center_boundaries", v);
    v.assign(n, 0);
    v[0] = pair_product/2; v[1] = v[0]+1; v[2] = -v[1];
    v[3] = pair_product+1; v[4] = -v[3];
    v[5] = pair_product; v[6] = -pair_product;
    // Q=2 must re-center at Q=q0*q1. Q=4 must NOT re-center at q0*q1:
    // these samples are inside full Q/2, forcing the verified shortcut to fail.
    if (q.size() >= 3) require(centered(v[1], product) == v[1] && v[1] > pair_product/2,
                               "missing forced full-CRT sample");
    add("two_prime_center_or_CRT_fallback", v);
    add("lazy_CRT_fallback", v, true);
    v.assign(n, 0); v.back() = pair_product;
    // For Q4, all earlier Q residues of this last coefficient are zero;
    // only the LAST Q limb rejects the small candidate. +/-q0*q1 above
    // independently cover two-prime aliases.
    if (q.size() > 2) v.back() = product / q.back().value();
    add("last_coefficient_alias_fallback", v);
    v.assign(n, 0); v[0] = -7;
    add("period_1", v);
    require(result.back().period == 1, "constant polynomial period");
    for (std::size_t j = 1; j < 8; ++j) v[j*(n/8)] = static_cast<long>(j)-4;
    add("period_8", v);
    require(result.back().period == 8, "support-derived intermediate period");
    v.assign(n, 0); v[1] = 1;
    add("period_N", v);
    require(result.back().period == n, "monomial must be uncompressible");
    return result;
}
static LinearMatrixGroup group_for(const std::vector<Case> &tests, std::uint32_t log_n,
                                   std::uint32_t nq) {
    LinearMatrixGroup group;
    group.set_step(1); group.set_rescale_min_scale(1024.0); group.rescale_counts() = {1};
    group.data().emplace_back();
    auto &matrix = group.data().back();
    matrix.log_slots = log_n-1; matrix.n1 = 4; matrix.level = nq-1; matrix.scale = 1024.0;
    for (std::size_t i = 0; i < tests.size(); ++i) matrix.plain_vec.emplace(static_cast<int>(i), tests[i].source);
    return group;
}
static void check_meta(const GpuPlaintextMeta &meta, const Case &test, std::size_t n,
                       std::size_t nq, std::size_t np) {
    require(meta.parms_id == test.source.parms_id() && meta.is_ntt_form &&
            std::bit_cast<std::uint64_t>(meta.scale) == std::bit_cast<std::uint64_t>(test.source.scale()) &&
            meta.degree == n && meta.q_count == nq && meta.p_count == np, test.name + ": metadata");
}
using Snapshot = std::vector<std::uint64_t>;
static void append(Snapshot &s, const Words &words) { s.insert(s.end(), words.begin(), words.end()); }
static Snapshot check_gpu(const std::vector<Case> &tests, const LinearMatrixGroup &group,
                          const PoseidonContext &context, std::size_t n, std::size_t nq, std::size_t np) {
    Snapshot snapshot;
    for (const auto &test : tests) {
        auto full = GpuUploader::upload_plaintext_qp_exact(test.source, context, 0);
        check_meta(full.meta, test, n, nq, np);
        require(full.fields_.size() == 1 && full.fields_[0].size() == test.oracle.size(), "full QP owner shape");
        Words got(test.oracle.size());
        full.fields_[0].buffer.copy_to_host(got.data(), got.size());
        equal_words(got, test.oracle, test.name + " full Q/P");
        append(snapshot, got);
    }
    auto compressed = GpuUploader::upload_linear_matrix_group_qp(group, context, 0, 1, true);
    require(compressed.data().size() == 1, "single matrix result");
    const auto &matrix = compressed.data().front();
    require(matrix.plain_vec_qp.empty() && matrix.compressed_plain_vec_qp.size() == tests.size(), "compressed owners");
    for (std::size_t i = 0; i < tests.size(); ++i) {
        const auto &test = tests[i];
        const auto &plain = matrix.compressed_plain_vec_qp.at(static_cast<int>(i));
        check_meta(plain.meta, test, n, nq, np);
        require(plain.exact_device_reconstruction && plain.period == test.period &&
                plain.residues.size() == (nq+np)*plain.period, test.name + ": compressed metadata");
        Words compact(plain.residues.size()), expanded(test.oracle.size());
        plain.residues.copy_to_host(compact.data(), compact.size());
        for (std::size_t limb = 0; limb < nq+np; ++limb)
            for (std::size_t j = 0; j < n; ++j)
                expanded[limb*n+j] = compact[limb*plain.period +
                    (plain.period == n ? j : reversed(j,n)&(plain.period-1))];
        equal_words(expanded, test.oracle, test.name + " expanded Q/P");
        snapshot.push_back(plain.period); append(snapshot, compact);
    }
    const auto &plan = matrix.plan;
    require(plan.compressed_plaintexts && plan.log_slots == matrix.log_slots && plan.n1 == matrix.n1 &&
            plan.rescale_count == 1 && plan.terms.size() == tests.size(), "compressed plan metadata");
    std::vector<std::uint32_t> periods(plan.diagonal_periods.size());
    std::vector<const GpuWord *> qptr(plan.diagonal_q_ptrs.size()), pptr(plan.diagonal_p_ptrs.size());
    plan.diagonal_periods.copy_to_host(periods.data(), periods.size());
    plan.diagonal_q_ptrs.copy_to_host(qptr.data(), qptr.size());
    plan.diagonal_p_ptrs.copy_to_host(pptr.data(), pptr.size());
    for (std::size_t i = 0; i < plan.terms.size(); ++i) {
        const auto &term = plan.terms[i];
        const int diagonal = plan.giant_steps.at(term.giant_index) + plan.baby_steps.at(term.baby_index);
        const auto &plain = matrix.compressed_plain_vec_qp.at(diagonal);
        require(periods.at(i) == plain.period && qptr.at(i) == plain.residues.data() &&
                pptr.at(i) == plain.residues.data()+nq*plain.period, "plan owner/offset reference");
        snapshot.push_back(term.giant_index); snapshot.push_back(term.baby_index); snapshot.push_back(periods[i]);
    }
    return snapshot;
}
template<class F> static void expect_throw(F &&f, const char *name) {
    bool threw = false;
    try { f(); } catch (const std::exception &) { threw = true; }
    require(threw, std::string("missing exception: ") + name);
}
static void check_exceptions(const std::vector<Case> &tests, const LinearMatrixGroup &valid,
                             const PoseidonContext &context) {
    auto bad = tests.front().source;
    bad.parms_id() = parms_id_zero;
    expect_throw([&] { GpuUploader::upload_plaintext_qp_exact(bad, context, 0); }, "non-NTT");
    bad.parms_id() = tests.front().source.parms_id(); bad.parms_id()[0] ^= 123;
    expect_throw([&] { GpuUploader::upload_plaintext_qp_exact(bad, context, 0); }, "unknown parms_id");
    Plaintext short_plain(tests.front().source.coeff_count()-1);
    short_plain.parms_id() = tests.front().source.parms_id();
    expect_throw([&] { GpuUploader::upload_plaintext_qp_exact(short_plain, context, 0); }, "invalid shape");
    auto group = valid;
    // Earlier tasks are valid. With requested 4 workers this error occurs while
    // other host tasks can be in flight; returning must join all of them.
    group.data()[0].plain_vec.at(2).parms_id() = parms_id_zero;
    expect_throw([&] { GpuUploader::upload_linear_matrix_group_qp(group, context, 0, 1, true); }, "worker exception");
    const auto *mode = std::getenv("POSEIDON_GPU_QP_PREPARE");
    if (!mode || std::string(mode) == "optimized") {
        for (const char *setting : {"", "0", "abc", "65"}) {
            Env threads("POSEIDON_GPU_QP_PREPARE_THREADS", setting);
            expect_throw([&] { GpuUploader::upload_linear_matrix_group_qp(valid, context, 0, 1, true); }, "invalid worker count");
        }
    }
}
struct TestStream {
    cudaStream_t stream = nullptr;
    std::optional<ScopedGpuExecutionStream> override;
    explicit TestStream(bool enabled) {
        if (enabled) {
            gpu_check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create test stream");
            override.emplace(stream);
            require(gpu_execution_stream() == stream, "caller stream override not installed");
        }
    }
    void finish() {
        override.reset();
        if (stream) {
            const auto owned = stream; stream = nullptr;
            gpu_check_cuda(cudaStreamDestroy(owned), "destroy test stream");
        }
    }
    ~TestStream() { override.reset(); if (stream) (void)cudaStreamDestroy(stream); }
};
int main(int argc, char **argv) {
    try {
        bool cpu_only = false, compare = false;
        for (int i = 1; i < argc; ++i) {
            if (std::string(argv[i]) == "--cpu-only") cpu_only = true;
            else if (std::string(argv[i]) == "--compare") compare = true;
            else throw std::invalid_argument("usage: qp_prepare [--cpu-only] [--compare]");
        }
        if (!cpu_only) {
            gpu_check_cuda(cudaSetDevice(0), "set device");
            gpu_check_cuda(cudaFree(nullptr), "initialize CUDA");
        }
        TestStream stream(!cpu_only); // GPU work must stay on the caller's nondefault stream.
        std::size_t configurations = 0, checked = 0;
        for (std::uint32_t log_n : {10U, 12U}) for (std::uint32_t bits : {28U, 30U, 32U, 0U}) {
            ParametersLiteral params(CKKS, log_n, log_n-1, 10, 64, 0, 0, {}, {}, sec_level_type::none);
            params.set_log_modulus(bits ? std::vector<std::uint32_t>(4,bits) : std::vector<std::uint32_t>{32,30,28,32},
                                   bits ? std::vector<std::uint32_t>(bits == 30 ? 3 : 2,bits)
                                        : std::vector<std::uint32_t>{28,30,32});
            PoseidonContext context(params);
            if (bits == 32) {
                mpz_class pair = mpz_class(params.q()[0].value())*params.q()[1].value();
                require(pair > (mpz_class(1)<<64)*99/100, "32-bit pair must exercise product near 2^64");
            }
            for (std::uint32_t nq : {1U, 2U, 4U}) {
                const auto id = context.crt_context()->parms_id_map().at(nq-1);
                const auto tests = cases(context, id);
                auto group = group_for(tests, log_n, nq);
                const auto n = std::size_t(1)<<log_n, np = params.p().size();
                ++configurations; checked += tests.size();
                if (!cpu_only) {
                    auto run = [&] {
                        // Exception followed by a complete valid upload checks recovery.
                        check_exceptions(tests, group, context);
                        return check_gpu(tests, group, context, n, nq, np);
                    };
                    if (compare) {
                        Env mode("POSEIDON_GPU_QP_PREPARE", "reference");
                        const auto reference = run();
                        for (const char *workers : {"1", "4"}) {
                            Env optimized("POSEIDON_GPU_QP_PREPARE", "optimized");
                            Env threads("POSEIDON_GPU_QP_PREPARE_THREADS", workers);
                            require(run() == reference, "reference/optimized full words, compact words or plans differ");
                            // Only the optimized path accepts the wider lazy range.
                            // Its oracle is canonical full CRT/GMP, NOT a reference
                            // INTT called outside the original [0,2q) contract.
                            auto wide_lazy = tests;
                            const auto &q = context.crt_context()->get_context_data(id)->parms().q();
                            for (auto &test : wide_lazy)
                                for (std::size_t limb = 0; limb < nq; ++limb)
                                    for (std::size_t i = 0; i < n; ++i)
                                        test.source.data()[limb*n+i] += 2*q[limb].value();
                            // Optimized VERIFY_REFERENCE must canonicalize before
                            // passing wide-lazy input to the old INTT oracle.
                            Env verify_reference("POSEIDON_GPU_QP_VERIFY_REFERENCE", "1");
                            const auto wide_group = group_for(wide_lazy, log_n, nq);
                            require(check_gpu(wide_lazy, wide_group, context, n, nq, np) == reference,
                                    "optimized [2q,4q) lazy canonicalization differs");
                            if (log_n == 10 && bits == 28 && nq == 1 && std::string(workers) == "4") {
                                // Force the same std::system_error path as OS thread
                                // exhaustion: all work must finish on the caller.
                                std::cerr << "QP_REGRESSION_ASYNC_FAILURE_BEGIN\n" << std::flush;
                                Env inject("POSEIDON_GPU_QP_TEST_ASYNC_FAILURE", "1");
                                require(run() == reference, "serial fallback after thread launch failure differs");
                                std::cerr << "QP_REGRESSION_ASYNC_FAILURE_END\n" << std::flush;
                            }
                            if (log_n == 10 && bits == 28 && nq == 2 && std::string(workers) == "4") {
                                // Concurrent profile callers share the process-wide
                                // extra-worker budget, while retaining distinct streams.
                                auto parallel_check = [&] {
                                    gpu_check_cuda(cudaSetDevice(0), "set concurrent device");
                                    TestStream local_stream(true);
                                    auto result = check_gpu(tests, group, context, n, nq, np);
                                    local_stream.finish();
                                    return result;
                                };
                                auto one = std::async(std::launch::async, parallel_check);
                                auto two = std::async(std::launch::async, parallel_check);
                                require(one.get() == reference && two.get() == reference,
                                        "concurrent QP preparation differs");
                            }
                        }
                    } else run();
                }
                std::cout << "PASS " << (cpu_only ? "CPU oracle only" : "GPU exact")
                          << " N=" << n << " Q=" << nq << " P=" << np << " bits=" << bits << '\n';
            }
        }
        stream.finish();
        std::cout << "PASS configurations=" << configurations << " cases=" << checked
                  << " gpu_executed=" << (cpu_only ? "false" : "true")
                  << " caller_nondefault_stream=" << (cpu_only ? "false" : "true") << '\n';
        return 0;
    } catch (const std::exception &e) { std::cerr << "FAIL: " << e.what() << '\n'; return 1; }
}
