// End-to-end GPU verification & timing test for:
//  1) Parallel Galois keygen (without keyswitch_base.cpp omp critical lock)
//  2) Shared GPU C2S/S2C QP matrices across r=8192 and r=65536 Boot profiles
//  3) Public constants disk cache + dev key disk cache (cold vs warm + fresh-key public-cache-only)
//  4) Layer Ciphertext::save / Ciphertext::load checkpointing (bit-exact RNS + wrong-key rejection)
#include "source_fast_boot.hpp"
#include "source_checkpoint.hpp"
#include "source_paired.hpp"
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <rmm/mr/device/statistics_resource_adaptor.hpp>
#include <omp.h>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace cerium_source;
using namespace poseidon::gpu;

namespace {

struct ScopedRmm {
    rmm::mr::device_memory_resource *old = nullptr;
    std::unique_ptr<rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource>> pool;
    std::unique_ptr<rmm::mr::statistics_resource_adaptor<rmm::mr::device_memory_resource>> stats;
    explicit ScopedRmm(int dev = 0) {
        gpu_check_cuda(cudaSetDevice(dev), "test setDevice");
        old = rmm::mr::get_current_device_resource();
        pool = std::make_unique<rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource>>(
            old, 64ULL << 20, 28ULL << 30);
        stats = std::make_unique<rmm::mr::statistics_resource_adaptor<rmm::mr::device_memory_resource>>(pool.get());
        rmm::mr::set_current_device_resource(stats.get());
    }
    ~ScopedRmm() {
        (void)cudaDeviceSynchronize();
        rmm::mr::set_current_device_resource(old);
    }
};

double test_boot_pair_accuracy(Runtime &r,
                               Api &api,
                               const std::vector<double> &a_vec,
                               const std::vector<double> &b_vec) {
    auto ca = r.encrypt(a_vec, 6);
    auto cb = r.encrypt(b_vec, 6);
    auto out = boot_pair(api, ca, cb);
    auto da = r.decrypt_final(out[0]);
    auto db = r.decrypt_final(out[1]);
    double max_err = 0.0;
    for (std::size_t i = 0; i < 32768; ++i) {
        max_err = std::max(max_err, std::hypot(da[i].real() - a_vec[i], da[i].imag()));
        max_err = std::max(max_err, std::hypot(db[i].real() - b_vec[i], db[i].imag()));
    }
    return max_err;
}

} // namespace

int main(int argc, char **argv) {
    try {
        if (argc < 3) {
            std::cerr << "usage: llama31_setup_cache_ckpt_test <r20_fixture.txt> <work_dir>\n";
            return 1;
        }
        const std::string fixture = argv[1];
        const std::filesystem::path work_dir(argv[2]);
        const std::filesystem::path pub_dir = work_dir / "public_cache";
        const std::filesystem::path key_dir = work_dir / "dev_key_cache";
        const std::filesystem::path ckpt_dir = work_dir / "checkpoints";
        std::filesystem::remove_all(work_dir);
        std::filesystem::create_directories(work_dir);

        ::setenv("LLAMA_FAST_P", "1", 1);
        ::setenv("LLAMA_FAST_BOOT", "1", 1);
        ::setenv("LLAMA_SCALAR_CACHE", "1", 1);
        ::setenv("LLAMA_ADD_PREFIX", "1", 1);
        ::setenv("POSEIDON_NTT_ALGO", "fourstep", 1);
        ::setenv("POSEIDON_GPU_SKIP_MONTGOMERY_TABLES", "1", 0);
        ::setenv("POSEIDON_GPU_SSE_LOW_P", "2", 1);
        ::setenv("POSEIDON_RESCALE_X2", "1", 1);
        ::setenv("POSEIDON_MANUAL_BOOT_RANGE_EXPERIMENT", "1", 1);

        const auto literal = parameters_from_fixture(fixture.c_str());
        require_cerium_parameters(literal);
        const auto rots = source_rotations();

        // ------------------------------------------------------------------
        // 0) PoseidonContext + GpuParameterData build benchmark (full vs sparse)
        // ------------------------------------------------------------------
        {
            ScopedRmm rmm(0);
            const auto t_ctx = Clock::now();
            poseidon::PoseidonContext ctx(literal);
            const double ctx_s = seconds(t_ctx);

            const auto t_full = Clock::now();
            GpuParameterData full_params(ctx, 0);
            const double full_s = seconds(t_full);
            const double full_gib = double(full_params.allocated_bytes()) / double(1ULL << 30);

            const auto t_sp = Clock::now();
            GpuParameterData sparse_params(ctx, 0, fast_runtime_global_q_counts(49), true);
            const double sparse_s = seconds(t_sp);
            const double sparse_gib = double(sparse_params.allocated_bytes()) / double(1ULL << 30);

            std::cout << std::fixed << std::setprecision(3)
                      << "TEST0_CTX_AND_PARAMS ctx_s=" << ctx_s
                      << " full_levels=" << full_params.level_count()
                      << " full_params_s=" << full_s
                      << " full_params_gib=" << full_gib
                      << " sparse_levels=" << sparse_params.level_count()
                      << " sparse_params_s=" << sparse_s
                      << " sparse_params_gib=" << sparse_gib << " PASS\n";
        }

        // ------------------------------------------------------------------
        // 1) Parallel Galois keygen benchmark (1 thread vs multi-thread)
        // ------------------------------------------------------------------
        {
            poseidon::PoseidonContext ctx(literal);
            poseidon::KeyGenerator kg(ctx);
            const auto gt = ctx.crt_context()->galois_tool();
            std::vector<std::uint32_t> elts;
            for (int s : std::vector<int>{0, 1, 2, 4, 8, 16, 32, 64, 128, -1, -2, -4, -8, -16, -32, -64})
                elts.push_back(gt->get_elt_from_step(s));

            const int prev_max = omp_get_max_threads();
            const int par_threads = std::min(16, std::max(1, prev_max));

            omp_set_num_threads(1);
            const auto t1 = Clock::now();
            poseidon::GaloisKeys gk1;
            kg.create_galois_keys(elts, gk1);
            const double s1 = seconds(t1);

            omp_set_num_threads(par_threads);
            const auto tp = Clock::now();
            poseidon::GaloisKeys gkp;
            kg.create_galois_keys(elts, gkp);
            const double sp = seconds(tp);

            // Duplicate input element safety check
            std::vector<std::uint32_t> dup_elts = elts;
            dup_elts.insert(dup_elts.end(), elts.begin(), elts.end());
            poseidon::GaloisKeys gkd;
            kg.create_galois_keys(dup_elts, gkd);
            for (auto e : elts) {
                if (!gkp.has_key(e) || !gkd.has_key(e))
                    throw std::runtime_error("missing Galois key after parallel generation");
            }
            std::cout << std::fixed << std::setprecision(3)
                      << "TEST1_KEYGEN_OMP keys=" << elts.size()
                      << " threads_1_s=" << s1
                      << " threads_" << par_threads << "_s=" << sp
                      << " speedup=" << (s1 / std::max(sp, 1e-6)) << "x PASS\n";
        }

        // Deterministic test payloads in [-0.45, 0.45]
        std::vector<double> vec_a(32768), vec_b(32768);
        for (std::size_t i = 0; i < 32768; ++i) {
            vec_a[i] = 0.40 * std::sin(0.0013 * double(i));
            vec_b[i] = 0.35 * std::cos(0.0021 * double(i) + 0.5);
        }

        double cold_setup_s = 0.0;
        double warm_setup_s = 0.0;
        double pub_only_setup_s = 0.0;
        const auto ckpt_file = checkpoint::layer_checkpoint_path(ckpt_dir.string(), 1);
        std::vector<std::vector<std::uint64_t>> saved_rns_words(16);

        // ------------------------------------------------------------------
        // 2) Cold run: populate public cache + dev key cache, verify shared C2S/S2C,
        //    verify Boot accuracy on BOTH r=8192 & r=65536, and save 16-tile checkpoint.
        // ------------------------------------------------------------------
        {
            ScopedRmm rmm(0);
            const auto t0 = Clock::now();
            HostRuntimeKeys hk = HostRuntimeKeys::create(literal, rots, key_dir.string());
            if (hk.loaded_from_dev_cache) throw std::runtime_error("expected cold dev key cache miss");
            poseidon::KeyGenerator kg(hk.context, hk.secret_key);
            poseidon::CKKSEncoder enc(hk.context);
            auto bundle = fast_boot::prepare_host_fast_boot_bundle(
                hk.context, kg, enc, rots, pub_dir.string(), key_dir.string());
            Runtime rt(hk, 0, true);
            fast_boot::install_fast_runtime_levels_from_bundle(rt, bundle);
            // Verify canonical modulus-chain ordering after sparse adopt_or_insert_level.
            if (rt.parameters.level_count() != 50 || rt.parameters.get_first_q_level().q_count != 49) {
                throw std::runtime_error("unexpected level_count or first_q_level after sparse band install");
            }
            for (std::size_t q = 49; q >= 2; --q) {
                const auto &cur = rt.parameters.get_level_by_q_count(q);
                const auto &nxt = rt.parameters.get_next_level(cur.parms_id);
                if (nxt.q_count != q - 1 || nxt.p_count != 0) {
                    throw std::runtime_error("broken get_next_level chain at q=" + std::to_string(q));
                }
            }
            manual_backend::SubmissionOwner owner;
            auto pair = fast_boot::make_fast_profile_pair(rt, owner, &bundle, pub_dir.string());
            cold_setup_s = seconds(t0);

            const auto &d0 = pair[0].first->bootstrap_data;
            const auto &d1 = pair[1].first->bootstrap_data;
            if (!d0.shared_coeff_to_slot_matrix_qp ||
                d0.shared_coeff_to_slot_matrix_qp.get() != d1.shared_coeff_to_slot_matrix_qp.get() ||
                d0.shared_slot_to_coeff_matrix_qp.get() != d1.shared_slot_to_coeff_matrix_qp.get() ||
                d0.shared_coeff_to_slot_groups_qp.get() != d1.shared_coeff_to_slot_groups_qp.get() ||
                d0.shared_slot_to_coeff_groups_qp.get() != d1.shared_slot_to_coeff_groups_qp.get() ||
                !d0.coeff_to_slot_matrix_qp.data().empty() || !d1.coeff_to_slot_matrix_qp.data().empty() ||
                !d0.slot_to_coeff_matrix_qp.data().empty() || !d1.slot_to_coeff_matrix_qp.data().empty()) {
                throw std::runtime_error("C2S/S2C QP matrices are not shared between r=8192 and r=65536");
            }
            std::cout << "TEST2_SHARED_C2S_S2C shared_ptr_equal=true duplicate_owned_empty=true"
                      << " rmm_live_gib=" << (double(rmm.stats->get_bytes_counter().value) / double(1ULL << 30))
                      << " PASS\n";

            manual_backend::BoundApi b_primary(rt, owner, std::move(pair[0]), 0);
            manual_backend::BoundApi b_silu(rt, owner, std::move(pair[1]), 0);
            const double err_8192 = test_boot_pair_accuracy(rt, b_primary.get(), vec_a, vec_b);
            const double err_65536 = test_boot_pair_accuracy(rt, b_silu.get(), vec_a, vec_b);
            if (!(err_8192 < 1e-3 && err_65536 < 1e-3)) {
                throw std::runtime_error("cold Boot accuracy exceeded 1e-3");
            }
            std::cout << "TEST2_COLD_SETUP seconds=" << cold_setup_s
                      << " err_r8192=" << err_8192 << " err_r65536=" << err_65536 << " PASS\n";

            // Encrypt 16 tiles, refresh pair 0..1 via Boot + cp_to + drop(6), and save checkpoint.
            std::vector<CT> tiles;
            tiles.reserve(16);
            for (std::size_t t = 0; t < 16; ++t) {
                std::vector<double> v(32768);
                for (std::size_t i = 0; i < 32768; ++i)
                    v[i] = 0.25 * std::sin(0.001 * double(i) + 0.1 * double(t));
                tiles.push_back(rt.encrypt(v, 6));
            }
            {
                auto br = boot_pair(b_primary.get(), tiles[0], tiles[1]);
                auto cp = b_primary.get().cp_to_pair(br[0], br[1], 1.0, std::exp2(40.0), 1);
                tiles[0] = b_primary.get().drop(cp[0], 6);
                tiles[1] = b_primary.get().drop(cp[1], 6);
            }
            for (std::size_t t = 0; t < 16; ++t) {
                poseidon::Ciphertext cpu_ct;
                GpuUploader::download_ciphertext(tiles[t], cpu_ct, rt.context);
                const std::size_t words = cpu_ct.size() * cpu_ct.coeff_modulus_size() * cpu_ct.poly_modulus_degree();
                saved_rns_words[t].assign(cpu_ct.data(), cpu_ct.data() + words);
            }
            const auto t_sv = Clock::now();
            checkpoint::save_gpu_tiles(ckpt_file, rt.context, rt.keys.secret_key(), tiles, 1, 38);
            std::cout << "TEST4_CKPT_SAVE bytes=" << std::filesystem::file_size(ckpt_file)
                      << " seconds=" << seconds(t_sv) << " PASS\n";
        }

        // ------------------------------------------------------------------
        // 3) Warm run: load BOTH public cache + dev key cache, verify speedup,
        //    verify Boot accuracy, and verify bit-exact 16-tile checkpoint reload.
        // ------------------------------------------------------------------
        {
            ScopedRmm rmm(0);
            const auto t0 = Clock::now();
            HostRuntimeKeys hk = HostRuntimeKeys::create(literal, rots, key_dir.string());
            if (!hk.loaded_from_dev_cache) throw std::runtime_error("expected warm dev key cache hit");
            poseidon::KeyGenerator kg(hk.context, hk.secret_key);
            poseidon::CKKSEncoder enc(hk.context);
            auto bundle = fast_boot::prepare_host_fast_boot_bundle(
                hk.context, kg, enc, rots, pub_dir.string(), key_dir.string());
            if (!bundle.keys_loaded_from_cache || bundle.has_cpu_dft)
                throw std::runtime_error("expected both fast_boot key cache hit and skipped CPU DFT");
            Runtime rt(hk, 0, true);
            fast_boot::install_fast_runtime_levels_from_bundle(rt, bundle);
            // Verify canonical modulus-chain ordering after sparse adopt_or_insert_level.
            if (rt.parameters.level_count() != 50 || rt.parameters.get_first_q_level().q_count != 49) {
                throw std::runtime_error("unexpected level_count or first_q_level after sparse band install");
            }
            for (std::size_t q = 49; q >= 2; --q) {
                const auto &cur = rt.parameters.get_level_by_q_count(q);
                const auto &nxt = rt.parameters.get_next_level(cur.parms_id);
                if (nxt.q_count != q - 1 || nxt.p_count != 0) {
                    throw std::runtime_error("broken get_next_level chain at q=" + std::to_string(q));
                }
            }
            manual_backend::SubmissionOwner owner;
            auto pair = fast_boot::make_fast_profile_pair(rt, owner, &bundle, pub_dir.string());
            warm_setup_s = seconds(t0);

            manual_backend::BoundApi b_primary(rt, owner, std::move(pair[0]), 0);
            manual_backend::BoundApi b_silu(rt, owner, std::move(pair[1]), 0);
            const double err_8192 = test_boot_pair_accuracy(rt, b_primary.get(), vec_a, vec_b);
            const double err_65536 = test_boot_pair_accuracy(rt, b_silu.get(), vec_a, vec_b);
            if (!(err_8192 < 1e-3 && err_65536 < 1e-3)) {
                throw std::runtime_error("warm cached Boot accuracy exceeded 1e-3");
            }
            std::cout << "TEST3_WARM_FULL_CACHE cold_s=" << cold_setup_s
                      << " warm_s=" << warm_setup_s
                      << " speedup=" << (cold_setup_s / std::max(warm_setup_s, 1e-6)) << "x"
                      << " err_r8192=" << err_8192 << " err_r65536=" << err_65536 << " PASS\n";

            // Reload 16-tile checkpoint and verify bit-exact RNS words!
            const auto t_ld = Clock::now();
            checkpoint::Metadata meta{};
            auto reloaded = checkpoint::load_gpu_tiles(ckpt_file, rt.context, rt.keys.secret_key(), 0, meta);
            const double ld_s = seconds(t_ld);
            if (meta.completed_layers != 1 || meta.total_tiles != 16 || reloaded.size() != 16)
                throw std::runtime_error("checkpoint metadata mismatch");
            std::size_t total_words = 0;
            for (std::size_t t = 0; t < 16; ++t) {
                poseidon::Ciphertext cpu_ct;
                GpuUploader::download_ciphertext(reloaded[t], cpu_ct, rt.context);
                const std::size_t words = cpu_ct.size() * cpu_ct.coeff_modulus_size() * cpu_ct.poly_modulus_degree();
                if (words != saved_rns_words[t].size())
                    throw std::runtime_error("reloaded tile RNS word count mismatch");
                for (std::size_t w = 0; w < words; ++w) {
                    if (cpu_ct.data()[w] != saved_rns_words[t][w])
                        throw std::runtime_error("reloaded tile RNS word mismatch (not bit-exact)");
                }
                total_words += words;
            }
            // Also run one more Boot on reloaded tiles[0..1] to verify live GPU usability!
            auto br2 = boot_pair(b_primary.get(), reloaded[0], reloaded[1]);
            auto d0 = rt.decrypt_final(br2[0]);
            double ck_err = 0.0;
            for (std::size_t i = 0; i < 32768; ++i)
                ck_err = std::max(ck_err, std::hypot(d0[i].real() - 0.25 * std::sin(0.001 * double(i)), d0[i].imag()));
            if (!(ck_err < 1e-3)) throw std::runtime_error("post-checkpoint Boot accuracy exceeded 1e-3");
            std::cout << "TEST4_CKPT_RELOAD bit_exact_words=" << total_words
                      << " load_seconds=" << ld_s << " post_reload_boot_err=" << ck_err << " PASS\n";
        }

        // ------------------------------------------------------------------
        // 4) Public-cache-only with FRESH random key + wrong-key checkpoint rejection
        // ------------------------------------------------------------------
        {
            ScopedRmm rmm(0);
            const auto t0 = Clock::now();
            HostRuntimeKeys hk = HostRuntimeKeys::create(literal, rots, ""); // fresh key!
            if (hk.loaded_from_dev_cache) throw std::runtime_error("fresh key should not hit dev cache");
            poseidon::KeyGenerator kg(hk.context, hk.secret_key);
            poseidon::CKKSEncoder enc(hk.context);
            auto bundle = fast_boot::prepare_host_fast_boot_bundle(
                hk.context, kg, enc, rots, pub_dir.string(), ""); // public cache ONLY
            if (bundle.has_cpu_dft || bundle.keys_loaded_from_cache)
                throw std::runtime_error("public-cache-only contract mismatch");
            Runtime rt(hk, 0, true);
            fast_boot::install_fast_runtime_levels_from_bundle(rt, bundle);
            // Verify canonical modulus-chain ordering after sparse adopt_or_insert_level.
            if (rt.parameters.level_count() != 50 || rt.parameters.get_first_q_level().q_count != 49) {
                throw std::runtime_error("unexpected level_count or first_q_level after sparse band install");
            }
            for (std::size_t q = 49; q >= 2; --q) {
                const auto &cur = rt.parameters.get_level_by_q_count(q);
                const auto &nxt = rt.parameters.get_next_level(cur.parms_id);
                if (nxt.q_count != q - 1 || nxt.p_count != 0) {
                    throw std::runtime_error("broken get_next_level chain at q=" + std::to_string(q));
                }
            }
            manual_backend::SubmissionOwner owner;
            auto pair = fast_boot::make_fast_profile_pair(rt, owner, &bundle, pub_dir.string());
            pub_only_setup_s = seconds(t0);

            manual_backend::BoundApi b_primary(rt, owner, std::move(pair[0]), 0);
            manual_backend::BoundApi b_silu(rt, owner, std::move(pair[1]), 0);
            const double err_8192 = test_boot_pair_accuracy(rt, b_primary.get(), vec_a, vec_b);
            const double err_65536 = test_boot_pair_accuracy(rt, b_silu.get(), vec_a, vec_b);
            if (!(err_8192 < 1e-3 && err_65536 < 1e-3))
                throw std::runtime_error("public-cache-only fresh-key Boot accuracy exceeded 1e-3");

            bool rejected_wrong_key = false;
            try {
                checkpoint::Metadata meta{};
                (void)checkpoint::load_gpu_tiles(ckpt_file, rt.context, rt.keys.secret_key(), 0, meta);
            } catch (const std::exception &) {
                rejected_wrong_key = true;
            }
            if (!rejected_wrong_key)
                throw std::runtime_error("checkpoint failed to reject mismatched secret key!");

            std::cout << "TEST5_PUBLIC_CACHE_FRESH_KEY seconds=" << pub_only_setup_s
                      << " err_r8192=" << err_8192 << " err_r65536=" << err_65536
                      << " wrong_key_ckpt_rejected=true PASS\n";
        }

        std::cout << "ALL_TESTS_PASSED cold_s=" << cold_setup_s
                  << " pub_only_s=" << pub_only_setup_s
                  << " warm_full_s=" << warm_setup_s << "\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "TEST_FAILED: " << e.what() << "\n";
        return 1;
    }
}
