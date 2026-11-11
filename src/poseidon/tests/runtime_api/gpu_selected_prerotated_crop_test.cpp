// Standalone regression: compile/link with poseidon_gpu; not registered by this task.
#include "poseidon/gpu/gpu_uploader.h"
#include "poseidon/keygenerator.h"
#include "poseidon/poseidon_context.h"

#include <cuda_runtime_api.h>
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace poseidon;
using namespace poseidon::gpu;

namespace
{
void require(bool value, const char *message)
{
    if (!value) throw std::runtime_error(message);
}

std::vector<GpuWord> read_words(const GpuWord *pointer, std::size_t count)
{
    std::vector<GpuWord> result(count);
    gpu_check_cuda(cudaMemcpyAsync(result.data(), pointer, count * sizeof(GpuWord),
                                   cudaMemcpyDeviceToHost, gpu_execution_stream()), "read key");
    gpu_check_cuda(cudaStreamSynchronize(gpu_execution_stream()), "read key sync");
    return result;
}

void check(const GpuEvaluationKeyData &full, const GpuEvaluationKeyData &cropped,
           std::size_t q, std::size_t index)
{
    const auto n = cropped.meta.degree;
    const auto p = cropped.meta.p_count;
    const auto digits = 1 + (q - 1) / p;
    require(cropped.meta.q_count == q && cropped.meta.decomposition_count == digits,
            "cropped key metadata is wrong");
    require(cropped.meta.key_parms_id == full.meta.key_parms_id,
            "cropped key context changed");
    require(cropped.polys_.size() == 2 * digits, "unexpected selected key storage");
    std::size_t bytes = 0;
    for (const auto &field : cropped.fields_) bytes += field.buffer.allocated_bytes();
    require(bytes == 2 * digits * (q + p) * n * sizeof(GpuWord),
            "cropped coefficient allocation is not physical");

    const auto cv = cropped.make_const_view();
    const auto fv = full.make_const_view();
    std::vector<const GpuWord *> tables[4];
    const DeviceVector<const GpuWord *> *device_tables[] = {
        &cropped.galois_key_q0_ptrs, &cropped.galois_key_p0_ptrs,
        &cropped.galois_key_q1_ptrs, &cropped.galois_key_p1_ptrs};
    for (int i = 0; i < 4; ++i)
    {
        tables[i].resize(cropped.meta.key_count * digits);
        require(device_tables[i]->size() == tables[i].size(), "wrong pointer table size");
        device_tables[i]->copy_to_host(tables[i].data(), tables[i].size());
    }
    for (std::size_t flat = 0; flat < tables[0].size(); ++flat)
        if (flat / digits != index)
            for (auto &table : tables) require(table[flat] == nullptr, "unselected key table is non-null");

    for (const auto &meta : cropped.poly_metadata_)
    {
        require(meta.key_index == index && meta.decomposition_index < digits && meta.component_index < 2,
                "unexpected key polynomial");
        const auto found = std::find_if(full.poly_metadata_.begin(), full.poly_metadata_.end(),
            [&](const auto &m) { return m.key_index == meta.key_index &&
                m.decomposition_index == meta.decomposition_index && m.component_index == meta.component_index; });
        require(found != full.poly_metadata_.end(), "source key polynomial missing");
        const auto *cp = cv.polys.at(meta.poly_id).shards.front().ptr;
        const auto *fp = fv.polys.at(found->poly_id).shards.front().ptr;
        const auto actual = read_words(cp, (q + p) * n);
        const auto expected_q = read_words(fp, q * n);
        const auto expected_p = read_words(fp + full.meta.q_count * n, p * n);
        require(std::equal(expected_q.begin(), expected_q.end(), actual.begin()), "rotated Q prefix mismatch");
        require(std::equal(expected_p.begin(), expected_p.end(), actual.begin() + q * n), "rotated P mismatch");
        const auto flat = index * digits + meta.decomposition_index;
        const auto component = meta.component_index;
        require(tables[2 * component][flat] == cp &&
                tables[2 * component + 1][flat] == cp + q * n,
                "table not based on cropped owner");
    }
}
}

int main()
{
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return 77;
    try
    {
        cudaStream_t stream{};
        gpu_check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream");
        {
            ScopedGpuExecutionStream scope(stream);
            ParametersLiteral params{CKKS, 12, 11, 44, 64, 0, 0, {}, {}, sec_level_type::none};
            params.set_log_modulus({29, 30, 31, 32}, {30, 31});
            PoseidonContext context(params);
            KeyGenerator generator(context);
            GaloisKeys keys;
            generator.create_galois_keys(std::vector<int>{1, 3}, keys);
            const auto elt = context.crt_context()->galois_tool()->get_elt_from_step(3);
            const auto index = GaloisKeys::get_index(elt);
            const auto full = GpuUploader::upload_double_hoist_galois_keys(keys, 0);
            for (std::size_t q : {std::size_t{1}, std::size_t{3}, std::size_t{4}})
            {
                auto cropped = GpuUploader::upload_double_hoist_galois_keys(keys, 0, q, {elt});
                check(full, cropped, q, index);
                auto copy = cropped;
                check(full, copy, q, index);
            }
            for (std::size_t q : {std::size_t{0}, std::size_t{5}})
            {
                bool rejected = false;
                try { (void)GpuUploader::upload_double_hoist_galois_keys(keys, 0, q, {elt}); }
                catch (const std::invalid_argument &) { rejected = true; }
                require(rejected, "invalid Q was accepted");
            }
        }
        gpu_check_cuda(cudaStreamDestroy(stream), "destroy stream");
        std::cout << "PASS selected pre-rotated key crop and copy\n";
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
