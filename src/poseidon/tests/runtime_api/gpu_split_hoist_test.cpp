// Standalone exact split-hoist regression; no bootstrap/RuntimePlan fixture.
// Usage: gpu_split_hoist_test [--small-only|--large-only] [--extended]
// Default: N4096/N65536, Q4/P2 and Q10/P9, each at full Q and Q3.
// --extended adds Q13/P10 and Q28/P25. Exit 77 means CUDA is unavailable.
#include "poseidon/ckks_encoder.h"
#include "poseidon/encryptor.h"
#include "poseidon/evaluator/software/evaluator_ckks_software.h"
#include "poseidon/gpu/gpu_evaluator.h"
#include "poseidon/gpu/gpu_uploader.h"
#include "poseidon/keygenerator.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace poseidon;
using namespace poseidon::gpu;
namespace
{
void require(bool condition, const std::string &message)
{
    if (!condition) throw std::runtime_error(message);
}

template <typename Function>
void rejects(Function &&function, const std::string &label)
{
    bool rejected = false;
    try { function(); }
    catch (const std::invalid_argument &) { rejected = true; }
    require(rejected, label + ": expected invalid_argument");
}

struct TestStream
{
    cudaStream_t value{};
    TestStream()
    {
        gpu_check_cuda(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking),
                       "create split-hoist test stream");
    }
    ~TestStream()
    {
        cudaStreamSynchronize(value);
        cudaStreamDestroy(value);
    }
};

ParametersLiteral parameters(unsigned log_n, std::size_t q, std::size_t p)
{
    ParametersLiteral result{CKKS, log_n, log_n - 1, 44, 64, 0, 0, {}, {}, sec_level_type::none};
    std::vector<std::uint32_t> q_bits(q), p_bits(p);
    for (std::size_t i = 0; i < q; ++i) q_bits[i] = 29 + i % 4;
    for (std::size_t i = 0; i < p; ++i) p_bits[i] = 29 + (i + 1) % 4;
    result.set_log_modulus(q_bits, p_bits);
    return result;
}

bool same_meta(const GpuCiphertextMeta &a, const GpuCiphertextMeta &b)
{
    return a.parms_id == b.parms_id && a.scale == b.scale &&
           a.correction_factor == b.correction_factor && a.is_ntt_form == b.is_ntt_form &&
           a.degree == b.degree && a.q_count == b.q_count && a.p_count == b.p_count &&
           a.component_count == b.component_count;
}

void exact(const GpuCiphertextData &a, const GpuCiphertextData &b,
           const PoseidonContext &context, const std::string &label)
{
    require(same_meta(a.meta, b.meta), label + ": GPU metadata changed");
    Ciphertext left, right;
    GpuUploader::download_ciphertext(a, left, context);
    GpuUploader::download_ciphertext(b, right, context);
    require(left.parms_id() == right.parms_id() && left.scale() == right.scale() &&
                left.size() == right.size() && left.is_ntt_form() == right.is_ntt_form() &&
                left.poly_modulus_degree() == right.poly_modulus_degree() &&
                left.coeff_modulus_size() == right.coeff_modulus_size(),
            label + ": downloaded metadata changed");
    const auto words = left.size() * left.poly_modulus_degree() * left.coeff_modulus_size();
    const auto mismatch = std::mismatch(left.data(), left.data() + words, right.data());
    require(mismatch.first == left.data() + words,
            label + ": RNS residue mismatch at word " +
                std::to_string(mismatch.first - left.data()));
}

GpuCiphertextData separate_component_fields(const GpuCiphertextData &source)
{
    GpuCiphertextData result;
    result.meta = source.meta;
    result.polys_ = source.polys_;
    const auto view = source.make_const_view();
    const auto words = source.meta.q_count * source.meta.degree;
    for (std::size_t component = 0; component < source.size(); ++component)
    {
        const auto &shard = view.polys[component].shards.front();
        result.fields_.emplace_back(shard.device_id, words);
        gpu_check_cuda(cudaMemcpyAsync(result.fields_.back().data(), shard.ptr,
            words * sizeof(GpuWord), cudaMemcpyDeviceToDevice, gpu_execution_stream()),
            "copy separate component field");
        result.polys_[component].shards.front().field_index = component;
        result.polys_[component].shards.front().field_offset = 0;
    }
    return result;
}

std::vector<GpuWord> readback(const DeviceVector<GpuWord> &buffer)
{
    std::vector<GpuWord> words(buffer.size());
    buffer.copy_to_host(words.data(), words.size());
    return words;
}

struct HoistSnapshot
{
    parms_id_type parms_id;
    int device;
    std::size_t degree, q, p, dnum, bytes;
    const GpuWord *intt_ptr, *q_ptr, *p_ptr, *source0, *source1;
    GpuCiphertextMeta source_meta;
    parms_id_type key_parms_id;
    GpuGaloisKeyFormat key_format;
    std::vector<GpuWord> intt, digits_q, digits_p;

    explicit HoistSnapshot(const GpuHoistedDecomposition &h)
        : parms_id(h.parms_id), device(h.device_id), degree(h.degree), q(h.q_count),
          p(h.p_count), dnum(h.dnum), bytes(h.allocated_bytes()),
          intt_ptr(h.source_intt_q.data()), q_ptr(h.digits_q_ntt.data()),
          p_ptr(h.digits_p_ntt.data()), source0(h.rotation_source_c0),
          source1(h.rotation_source_c1), source_meta(h.rotation_source_meta),
          key_parms_id(h.rotation_key_parms_id), key_format(h.rotation_key_format),
          intt(readback(h.source_intt_q)), digits_q(readback(h.digits_q_ntt)),
          digits_p(readback(h.digits_p_ntt))
    {}

    void unchanged(const GpuHoistedDecomposition &h, const std::string &label) const
    {
        require(parms_id == h.parms_id && device == h.device_id && degree == h.degree &&
                    q == h.q_count && p == h.p_count && dnum == h.dnum &&
                    bytes == h.allocated_bytes() && intt_ptr == h.source_intt_q.data() &&
                    q_ptr == h.digits_q_ntt.data() && p_ptr == h.digits_p_ntt.data() &&
                    source0 == h.rotation_source_c0 && source1 == h.rotation_source_c1 &&
                    same_meta(source_meta, h.rotation_source_meta) &&
                    key_parms_id == h.rotation_key_parms_id && key_format == h.rotation_key_format,
                label + ": hoist allocation/metadata changed");
        require(intt == readback(h.source_intt_q) && digits_q == readback(h.digits_q_ntt) &&
                    digits_p == readback(h.digits_p_ntt), label + ": hoist contents changed");
    }
};

void allocation_accounting()
{
    GpuHoistedDecomposition h;
    require(h.allocated_bytes() == 0, "empty hoist allocation accounting");
    // Deliberately unrelated to logical shape: count actual retained buffers.
    h.source_intt_q.allocate(17, 0);
    h.digits_q_ntt.allocate(31, 0);
    h.digits_p_ntt.allocate(13, 0);
    require(h.allocated_bytes() == (17 + 31 + 13) * sizeof(GpuWord),
            "hoist byte accounting omitted a retained buffer/capacity");
    h.source_intt_q.release();
    require(h.allocated_bytes() == (31 + 13) * sizeof(GpuWord),
            "hoist byte accounting includes released storage");
}

void split_case(unsigned log_n, std::size_t q, std::size_t p)
{
    PoseidonContext context(parameters(log_n, q, p));
    KeyGenerator generator(context);
    require(!generator.secret_key().data().is_zero(), "test requires a nonzero secret");
    PublicKey public_key;
    GaloisKeys rotations;
    const std::vector<int> steps{1, -2, 3};
    generator.create_public_key(public_key);
    generator.create_galois_keys(steps, rotations);
    CKKSEncoder encoder(context);
    Encryptor encryptor(context, public_key);
    EvaluatorCkksSoftware cpu(context);
    std::vector<std::complex<double>> values(encoder.slot_count());
    for (std::size_t i = 0; i < values.size(); ++i)
        values[i] = {0.2 * std::sin(0.017 * i), 0.1 * std::cos(0.013 * i)};
    Plaintext plain;
    encoder.encode(values, std::ldexp(1., 44), plain);
    Ciphertext source, other_source;
    encryptor.encrypt(plain, source);
    encryptor.encrypt(plain, other_source);
    GpuParameterData gpu_parameters(context, 0, {q, 3});
    GpuEvaluator evaluator(gpu_parameters);
    if (log_n == 12 && q == 4 && p == 2)
    {
        // Also guard switch_key's level-override dispatch: its parent need not
        // own a key. The key's algebra is immaterial to this exact A/B check.
        RelinKeys host_switch;
        generator.create_relin_keys(host_switch);
        auto selected = std::make_shared<GpuEvaluationKeyData>(
            GpuUploader::upload_relin_keys(host_switch, 0));
        GpuEvaluationKeyData dispatcher;
        dispatcher.level_keys = std::make_shared<const std::map<std::size_t,
            std::shared_ptr<const GpuEvaluationKeyData>>>(
                std::map<std::size_t, std::shared_ptr<const GpuEvaluationKeyData>>{
                    {q, selected}, {3, selected}});
        for (const auto active_q : {q, std::size_t{3}})
        {
            Ciphertext input = source;
            cpu.drop_modulus(input, input, context.crt_context()->parms_id_map().at(active_q - 1));
            auto gpu = GpuUploader::upload_ciphertext(input, 0);
            GpuCiphertextData direct, dispatched;
            evaluator.switch_key(gpu, *selected, direct);
            evaluator.switch_key(gpu, dispatcher, dispatched);
            const auto label = "switch_key Q=" + std::to_string(active_q);
            exact(direct, dispatched, context, label + " active-level override");

            // The active-Q view is cached by the successful direct call.
            // Reject incompatible selected-owner metadata before consulting a
            // cached view, including when only the dispatch parent's override
            // owns any key storage. Mutations below do not move device buffers.
            const auto valid_meta = selected->meta;
            const auto rejects_layout = [&](const std::string &kind) {
                rejects([&] { evaluator.switch_key(gpu, *selected, dispatched); },
                        label + " direct " + kind);
                rejects([&] { evaluator.switch_key(gpu, dispatcher, dispatched); },
                        label + " selected override " + kind);
            };
            selected->meta.galois_format = GpuGaloisKeyFormat::InversePreRotated;
            rejects_layout("inverse-pre-rotated Galois format");
            selected->meta = valid_meta;
            selected->meta.component_count = 1;
            rejects_layout("wrong key component count");
            selected->meta = valid_meta;
            for (const auto invalid_count : {std::size_t{0}, std::size_t{2}})
            {
                selected->meta.key_count = invalid_count;
                rejects_layout("wrong key_count=" + std::to_string(invalid_count));
            }
            selected->meta = valid_meta;
            require(!selected->fields_.empty(), label + ": selected key has no fields");
            const auto valid_device = selected->fields_.back().device_id;
            selected->fields_.back().device_id = valid_device + 1;
            rejects_layout("wrong key device");
            selected->fields_.back().device_id = valid_device;

            evaluator.switch_key(gpu, dispatcher, dispatched);
            exact(direct, dispatched, context, label + " override after layout rejections");
        }
        std::cout << "PASS switch_key active-level override all_RNS_exact=true layout_negatives=true\n";
    }
    // Reuse scratch across formats and levels, while each public H owns itself.
    GpuDoubleHoistWorkspace reference_workspace, workspace;
    for (const bool pre_rotated : {false, true})
    {
        auto keys_owner = std::make_shared<GpuGaloisKeysData>(
            pre_rotated ? GpuUploader::upload_double_hoist_galois_keys(rotations, 0)
                        : GpuUploader::upload_galois_keys(rotations, 0));
        auto &keys = *keys_owner;
        for (const auto active_q : {q, std::size_t{3}})
        {
            Ciphertext input = source, other_input = other_source;
            const auto id = context.crt_context()->parms_id_map().at(active_q - 1);
            cpu.drop_modulus(input, input, id);
            cpu.drop_modulus(other_input, other_input, id);
            auto gpu = GpuUploader::upload_ciphertext(input, 0);
            auto gpu_other = GpuUploader::upload_ciphertext(other_input, 0);
            auto untouched = GpuUploader::upload_ciphertext(input, 0);
            const auto label = "N=" + std::to_string(std::size_t{1} << log_n) + " Q=" +
                std::to_string(active_q) + " P=" + std::to_string(p) +
                (pre_rotated ? " inverse-pre-rotated" : " standard");
            std::vector<GpuCiphertextData> reference;
            evaluator.rotate_many_hoisted(gpu, steps, keys, reference_workspace, reference);
            GpuHoistedDecomposition h, other_h;
            evaluator.rotation_hoist(gpu, keys, h, workspace);
            const HoistSnapshot captured(h);
            const auto dnum = (active_q + p - 1) / p;
            require(h.dnum == dnum && h.p_count == p && h.q_count == active_q,
                    label + ": wrong active hoist shape");
            require(h.allocated_bytes() ==
                        (active_q + dnum * (active_q + p)) * h.degree * sizeof(GpuWord),
                    label + ": hoist logical/allocated bytes disagree for fresh result");
            std::vector<GpuCiphertextData> split(steps.size());
            evaluator.rotate_from_hoist(gpu, h, steps[0], keys, workspace, split[0]);
            exact(reference[0], split[0], context, label + " first");

            // Another prefix and the original batched API can reuse all scratch
            // without changing an externally owned H or previously returned output.
            evaluator.rotation_hoist(gpu_other, keys, other_h, workspace);
            GpuCiphertextData interleaved;
            evaluator.rotate_from_hoist(gpu_other, other_h, steps[1], keys, workspace, interleaved);
            std::vector<GpuCiphertextData> other_reference;
            evaluator.rotate_many_hoisted(gpu_other, {steps[1]}, keys, workspace, other_reference);
            exact(other_reference[0], interleaved, context, label + " other source");
            for (std::size_t i = 1; i < steps.size(); ++i)
                evaluator.rotate_from_hoist(gpu, h, steps[i], keys, workspace, split[i]);
            for (std::size_t i = 0; i < steps.size(); ++i)
                exact(reference[i], split[i], context, label + " split " + std::to_string(i));
            GpuCiphertextData repeated;
            evaluator.rotate_from_hoist(gpu, h, steps[0], keys, workspace, repeated);
            exact(reference[0], repeated, context, label + " repeat");
            {
                GpuDoubleHoistWorkspace independent_workspace;
                evaluator.rotate_from_hoist(gpu, h, steps[2], keys, independent_workspace, repeated);
                exact(reference[2], repeated, context, label + " independent suffix workspace");
            }
            {
                // An empty parent key is legal when its active-level override
                // supplies the real key; validation must inspect the selected key.
                GpuGaloisKeysData dispatcher;
                dispatcher.level_keys = std::make_shared<const std::map<std::size_t,
                    std::shared_ptr<const GpuEvaluationKeyData>>>(
                        std::map<std::size_t, std::shared_ptr<const GpuEvaluationKeyData>>{
                            {active_q, keys_owner}});
                GpuHoistedDecomposition dispatched_h;
                evaluator.rotation_hoist(gpu, dispatcher, dispatched_h, workspace);
                evaluator.rotate_from_hoist(gpu, dispatched_h, steps[1], dispatcher, workspace, repeated);
                exact(reference[1], repeated, context, label + " active-level key dispatch");
            }
            captured.unchanged(h, label);
            exact(untouched, gpu, context, label + " source immutable");

            {
                // One full shard per component need not mean one owning field.
                auto separated = separate_component_fields(gpu);
                std::vector<GpuCiphertextData> separated_reference;
                evaluator.rotate_many_hoisted(separated, {steps[2]}, keys,
                                             reference_workspace, separated_reference);
                GpuHoistedDecomposition separated_h;
                evaluator.rotation_hoist(separated, keys, separated_h, workspace);
                evaluator.rotate_from_hoist(separated, separated_h, steps[2], keys, workspace, repeated);
                exact(separated_reference[0], repeated, context, label + " separate component fields");
            }
            // Existing batched/BSGS operations own and refill these scratch
            // members. A public immutable H must not be placed there.
            rejects([&] { evaluator.rotation_hoist(gpu, keys, workspace.source_hoist, workspace); },
                    label + " scratch-owned prefix H");
            rejects([&] { evaluator.rotate_from_hoist(gpu, workspace.source_hoist, 1, keys, workspace, repeated); },
                    label + " scratch-owned suffix H");

            rejects([&] { evaluator.rotate_from_hoist(gpu, h, 0, keys, workspace, repeated); },
                    label + " zero step");
            rejects([&] { evaluator.rotate_from_hoist(gpu_other, h, 1, keys, workspace, repeated); },
                    label + " different source allocation");
            rejects([&] { evaluator.rotate_from_hoist(gpu, h, 1, keys, workspace, gpu); },
                    label + " in-place destination");
            GpuHoistedDecomposition empty;
            rejects([&] { evaluator.rotate_from_hoist(gpu, empty, 1, keys, workspace, repeated); },
                    label + " empty H");
            const auto old_scale = gpu.meta.scale;
            gpu.meta.scale *= 2;
            rejects([&] { evaluator.rotate_from_hoist(gpu, h, 1, keys, workspace, repeated); },
                    label + " source metadata version");
            gpu.meta.scale = old_scale;
            const auto old_p = keys.meta.p_count;
            keys.meta.p_count = old_p + 1;
            rejects([&] { evaluator.rotate_from_hoist(gpu, h, 1, keys, workspace, repeated); },
                    label + " mismatched key P");
            rejects([&] { evaluator.rotation_hoist(gpu, keys, empty, workspace); },
                    label + " mismatched prefix key P");
            keys.meta.p_count = old_p;
            const auto old_format = keys.meta.galois_format;
            keys.meta.galois_format = pre_rotated ? GpuGaloisKeyFormat::Standard
                                                : GpuGaloisKeyFormat::InversePreRotated;
            rejects([&] { evaluator.rotate_from_hoist(gpu, h, 1, keys, workspace, repeated); },
                    label + " mismatched key format");
            keys.meta.galois_format = old_format;
            if (active_q == q)
            {
                Ciphertext lower = input;
                cpu.drop_modulus(lower, lower, context.crt_context()->parms_id_map().at(2));
                auto gpu_lower = GpuUploader::upload_ciphertext(lower, 0);
                rejects([&] { evaluator.rotate_from_hoist(gpu_lower, h, 1, keys, workspace, repeated); },
                        label + " mismatched level");
            }
            // Failed suffixes must not poison H or its later consumers.
            evaluator.rotate_from_hoist(gpu, h, steps[1], keys, workspace, repeated);
            exact(reference[1], repeated, context, label + " after rejection");
            captured.unchanged(h, label + " after rejection");
            std::cout << "PASS split-hoist " << label << " dnum=" << h.dnum
                      << " bytes=" << h.allocated_bytes() << " all_RNS_exact=true\n";
        }
    }
}
}  // namespace

int main(int argc, char **argv)
{
    try
    {
        bool small = true, large = true, extended = false;
        for (int i = 1; i < argc; ++i)
        {
            const std::string arg(argv[i]);
            if (arg == "--small-only") large = false;
            else if (arg == "--large-only") small = false;
            else if (arg == "--extended") extended = true;
            else throw std::invalid_argument("unknown argument: " + arg);
        }
        require(small || large, "choose only one of --small-only/--large-only");
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return 77;
        gpu_check_cuda(cudaSetDevice(0), "select split-hoist test device");
        TestStream stream;
        ScopedGpuExecutionStream execution_stream(stream.value);
        allocation_accounting();
        std::vector<std::pair<std::size_t, std::size_t>> shapes{{4, 2}, {10, 9}};
        if (extended) { shapes.emplace_back(13, 10); shapes.emplace_back(28, 25); }
        for (const auto log_n : {12U, 16U})
            if ((log_n == 12 && small) || (log_n == 16 && large))
                for (const auto &[q, p] : shapes) split_case(log_n, q, p);
        std::cout << "GPU split-hoist regressions passed\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }
}
