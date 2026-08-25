#include "poseidon/gpu/gpu_ciphertext.h"
#include "poseidon/parameters_literal.h"
#include "poseidon/poseidon_context.h"
#include "poseidon/runtime_api/poseidon_gpu_api.h"

#include <cuda_runtime_api.h>
#include <mpi.h>

#include <cmath>
#include <cstdint>
#include <algorithm>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{

std::vector<int> parse_counts(const std::string &encoded)
{
    std::vector<int> result;
    std::size_t begin = 0;
    while (begin <= encoded.size())
    {
        const std::size_t end = encoded.find('x', begin);
        const std::string token = encoded.substr(
            begin, end == std::string::npos ? std::string::npos : end - begin);
        if (token.empty())
        {
            throw std::invalid_argument("device-counts contains an empty entry");
        }
        std::size_t consumed = 0;
        const int value = std::stoi(token, &consumed);
        if (consumed != token.size() || value <= 0)
        {
            throw std::invalid_argument("device-counts entries must be positive");
        }
        result.push_back(value);
        if (end == std::string::npos)
        {
            break;
        }
        begin = end + 1;
    }
    return result;
}

std::vector<int> parse_rank_to_node(const std::string &encoded)
{
    std::vector<int> result;
    std::size_t begin = 0;
    while (begin <= encoded.size())
    {
        const std::size_t end = encoded.find('x', begin);
        const std::string token = encoded.substr(
            begin, end == std::string::npos ? std::string::npos : end - begin);
        if (token.empty())
        {
            throw std::invalid_argument("rank-to-node contains an empty entry");
        }
        std::size_t consumed = 0;
        const int value = std::stoi(token, &consumed);
        if (consumed != token.size() || value < 0)
        {
            throw std::invalid_argument(
                "rank-to-node entries must be nonnegative integers");
        }
        result.push_back(value);
        if (end == std::string::npos)
        {
            break;
        }
        begin = end + 1;
    }
    return result;
}

void check_cuda(cudaError_t status, const char *what)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error(std::string(what) + ": " +
                                 cudaGetErrorString(status));
    }
}

poseidon::PoseidonContext make_context()
{
    poseidon::ParametersLiteral parameters(
        CKKS, 12, 11, 30, 0, 0, poseidon::Modulus(0), {}, {},
        poseidon::sec_level_type::none);
    parameters.set_log_modulus(std::vector<std::uint32_t>(7, 30),
                               std::vector<std::uint32_t>(2, 30));
    return poseidon::PoseidonContext(parameters);
}

} // namespace

int main(int argc, char **argv)
{
    int provided = MPI_THREAD_SINGLE;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS)
    {
        return 2;
    }

    int rank = 0;
    int world = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);

    try
    {
        std::string counts_text;
        std::string rank_to_node_text;
        for (int index = 1; index < argc; ++index)
        {
            const std::string argument = argv[index];
            if (argument == "--device-counts" && index + 1 < argc)
            {
                counts_text = argv[index + 1];
                ++index;
            }
            else if (argument == "--rank-to-node" && index + 1 < argc)
            {
                rank_to_node_text = argv[index + 1];
                ++index;
            }
            else
            {
                throw std::invalid_argument(
                    "usage: --device-counts 1x4 [--rank-to-node 0x1]");
            }
        }
        if (counts_text.empty())
        {
            throw std::invalid_argument("usage: --device-counts 1x1x1x1");
        }
        const std::vector<int> counts = parse_counts(counts_text);
        if (static_cast<int>(counts.size()) != world || world < 2)
        {
            throw std::invalid_argument(
                "transfer test requires at least two MPI ranks and matching device-counts");
        }
        std::vector<int> rank_to_node;
        if (!rank_to_node_text.empty())
        {
            rank_to_node = parse_rank_to_node(rank_to_node_text);
        }
        if (!rank_to_node.empty() &&
            static_cast<int>(rank_to_node.size()) != world)
        {
            throw std::invalid_argument(
                "rank-to-node length must match MPI world size");
        }

        MPI_Comm local_comm = MPI_COMM_NULL;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                            MPI_INFO_NULL, &local_comm);
        int local_rank = 0;
        MPI_Comm_rank(local_comm, &local_rank);
        int local_device_offset = 0;
        const int local_device_count = counts[rank];
        MPI_Exscan(&local_device_count, &local_device_offset, 1, MPI_INT,
                   MPI_SUM, local_comm);
        if (local_rank == 0)
        {
            local_device_offset = 0;
        }
        MPI_Comm_free(&local_comm);

        std::vector<int> local_devices(static_cast<std::size_t>(counts[rank]));
        for (int index = 0; index < counts[rank]; ++index)
        {
            local_devices[static_cast<std::size_t>(index)] =
                local_device_offset + index;
        }

        const auto context = make_context();
        poseidon::runtime_api::GpuProcessTopology topology;
        topology.device_counts = counts;
        topology.rank_to_node = rank_to_node;
        poseidon::runtime_api::PoseidonGpuApi api(
            "poseidon-gpu-mpi-transfer-test", context, MPI_COMM_WORLD,
            local_devices, topology);

        fhegpu::TargetConfig target;
        target.target_id = "poseidon-ckks-gpu";
        target.world_size = world;
        target.device_counts = counts;
        target.capability_version = 1;
        fhegpu::OperatorSpec operator_spec;
        operator_spec.id = "poseidon-gpu-mpi-transfer-test";
        operator_spec.version = 1;
        operator_spec.status = "test";
        operator_spec.target_id = target.target_id;
        operator_spec.rescale_mode = fhegpu::RescaleMode::Lazy;
        operator_spec.context_id = "poseidon-gpu-mpi-transfer-test";
        operator_spec.poly_degree = context.parameters_literal()->degree();
        for (const auto &modulus : context.parameters_literal()->q())
        {
            operator_spec.rns_moduli_log2.push_back(modulus.bit_count());
        }
        operator_spec.max_modulus_log2 = *std::max_element(
            operator_spec.rns_moduli_log2.begin(),
            operator_spec.rns_moduli_log2.end());
        operator_spec.default_scale_log2 =
            static_cast<int>(context.parameters_literal()->log_scale());
        operator_spec.level_lower_bound = 0;
        operator_spec.level_upper_bound =
            static_cast<int>(operator_spec.rns_moduli_log2.size() - 1);
        fhegpu::OperatorSupport rescale_support;
        rescale_support.supported = true;
        rescale_support.max_levels_per_op = 4;
        operator_spec.operators.emplace(fhegpu::ComputeKind::Rescale,
                                         std::move(rescale_support));
        fhegpu::PlanRequirements requirements;
        api.preflight("sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                      false, target, operator_spec, requirements);

        constexpr std::size_t degree = 4096;
        const std::size_t q_count = context.parameters_literal()->q().size();
        const auto parms_id = context.crt_context()->parms_id_map().at(q_count - 1);
        std::optional<poseidon::runtime_api::PoseidonGpuValue> source_value;
        if (rank == 0)
        {
            auto source = poseidon::gpu::GpuCiphertextData::allocate_single_device(
                degree, q_count, 2, local_devices.front());
            source.meta.parms_id = parms_id;
            source.meta.scale = std::ldexp(1.0, 30);
            source.meta.is_ntt_form = true;
            std::vector<poseidon::gpu::GpuWord> pattern(source.fields_.front().size());
            for (std::size_t index = 0; index < pattern.size(); ++index)
            {
                pattern[index] = static_cast<poseidon::gpu::GpuWord>(
                    (index * 17U + 23U) & 0xffffffffU);
            }
            check_cuda(cudaSetDevice(local_devices.front()), "cudaSetDevice source");
            check_cuda(cudaMemcpy(source.fields_.front().data(), pattern.data(),
                                  pattern.size() * sizeof(pattern.front()),
                                  cudaMemcpyHostToDevice),
                       "cudaMemcpy source pattern");
            source_value.emplace(
                poseidon::runtime_api::PoseidonGpuValue::from_device_ciphertext(
                    std::move(source)));
        }

        fhegpu::CommAction action;
        action.id = 7001;
        action.kind = fhegpu::CommKind::Transfer;
        action.hint = fhegpu::CommHint::PointToPoint;
        action.inputs = {1};
        action.outputs = {2};
        action.sources = {{fhegpu::PlaceKind::Device, 0, 0}};
        action.destinations = {{fhegpu::PlaceKind::Device, 1, 0}};
        action.output_types = {fhegpu::ValueKind::Ciphertext};

        std::vector<poseidon::runtime_api::PoseidonGpuValue> local_inputs;
        if (rank == 0)
        {
            local_inputs.push_back(std::move(*source_value));
        }
        const fhegpu::ValueDesc output_desc{
            2, fhegpu::ValueKind::Ciphertext,
            {fhegpu::PlaceKind::Device, 1, 0},
            "poseidon-gpu-mpi-transfer-test",
            static_cast<int>(q_count - 1), 30, true, 2};
        auto handle = api.communicate_async(action, local_inputs, {output_desc});
        auto outputs = api.wait(handle);

        if (rank == 1)
        {
            if (outputs.size() != 1)
            {
                throw std::runtime_error("destination rank returned wrong output count");
            }
            const auto &received = outputs.front().device_ciphertext();
            if (received.fields_.size() != 1 || received.fields_.front().size() == 0)
            {
                throw std::runtime_error("destination ciphertext field is empty");
            }
            std::vector<poseidon::gpu::GpuWord> host(received.fields_.front().size());
            check_cuda(cudaSetDevice(local_devices.front()), "cudaSetDevice destination");
            check_cuda(cudaMemcpy(host.data(), received.fields_.front().data(),
                                  host.size() * sizeof(host.front()),
                                  cudaMemcpyDeviceToHost),
                       "cudaMemcpy destination pattern");
            for (std::size_t index = 0; index < host.size(); ++index)
            {
                const auto expected = static_cast<poseidon::gpu::GpuWord>(
                    (index * 17U + 23U) & 0xffffffffU);
                if (host[index] != expected)
                {
                    throw std::runtime_error(
                        "destination ciphertext payload mismatch at word " +
                        std::to_string(index));
                }
            }
        }
        else if (!outputs.empty())
        {
            throw std::runtime_error("non-destination rank returned an output");
        }

        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0)
        {
            std::cout << "Poseidon GPU MPI transfer passed: world=" << world
                      << " device_counts=" << counts_text;
            if (!rank_to_node_text.empty())
            {
                std::cout << " rank_to_node=" << rank_to_node_text;
            }
            std::cout << '\n';
        }
    }
    catch (const std::exception &error)
    {
        std::cerr << "[rank " << rank << "] Poseidon GPU MPI transfer failed: "
                  << error.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Finalize();
    return 0;
}
