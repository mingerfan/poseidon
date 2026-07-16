#pragma once

#include "poseidon/ciphertext.h"
#include "poseidon/gpu/gpu_ciphertext.h"
#include "poseidon/gpu/gpu_key.h"
#include "poseidon/gpu/gpu_plaintext.h"
#include "poseidon/plaintext.h"
#include "poseidon/poseidon_context.h"
#include "runtime/operator_spec.hpp"
#include "runtime/plan.hpp"

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <variant>
#include <vector>

namespace poseidon
{
class CKKSEncoder;
class GaloisKeys;
class RelinKeys;

namespace gpu
{
class GpuEvaluator;
class GpuParameterData;
} // namespace gpu

namespace runtime_api
{

class PoseidonGpuValue
{
public:
    static PoseidonGpuValue from_host_plaintext(Plaintext value);
    static PoseidonGpuValue from_host_ciphertext(Ciphertext value);
    static PoseidonGpuValue from_device_plaintext(gpu::GpuPlaintextData value);
    static PoseidonGpuValue from_device_ciphertext(gpu::GpuCiphertextData value);

    fhegpu::ValueKind kind() const;
    fhegpu::PlaceKind place_kind() const;

    const Plaintext &host_plaintext() const;
    const Ciphertext &host_ciphertext() const;
    const gpu::GpuPlaintextData &device_plaintext() const;
    const gpu::GpuCiphertextData &device_ciphertext() const;

private:
    using Storage =
        std::variant<std::shared_ptr<Plaintext>, std::shared_ptr<Ciphertext>,
                     std::shared_ptr<gpu::GpuPlaintextData>,
                     std::shared_ptr<gpu::GpuCiphertextData>>;

    explicit PoseidonGpuValue(Storage storage);

    Storage storage_;
};

class PoseidonGpuApi
{
public:
    using Value = PoseidonGpuValue;

    struct CommHandle
    {
        std::vector<Value> outputs;
        bool waited = false;
    };

    PoseidonGpuApi(std::string context_id, PoseidonContext context, int cuda_device_id,
                   std::shared_ptr<const RelinKeys> relin_keys = {},
                   std::shared_ptr<const GaloisKeys> galois_keys = {});
    ~PoseidonGpuApi();

    PoseidonGpuApi(const PoseidonGpuApi &) = delete;
    PoseidonGpuApi &operator=(const PoseidonGpuApi &) = delete;

    std::string name() const;
    Value encode_plaintext(const fhegpu::ValueDesc &output_desc,
                           const std::vector<double> &slots);
    Value compute(const fhegpu::ComputeOp &op, const std::vector<Value> &inputs);
    CommHandle communicate_async(const fhegpu::CommAction &action,
                                 const std::vector<Value> &local_inputs);
    std::vector<Value> wait(CommHandle &handle);
    void synchronize(Value &value);
    void preflight(std::string_view plan_source_sha256, bool skip_artifact_digest_checks,
                   const fhegpu::TargetConfig &target,
                   const fhegpu::OperatorSpec &operator_spec,
                   const fhegpu::PlanRequirements &requirements);
    [[noreturn]] void abort_all(int exit_code, const std::string &reason);
    void validate_value(const Value &value, const fhegpu::ValueDesc &expected) const;

private:
    void synchronize_device() const;
    const gpu::GpuRelinKeysData &relin_keys_for(std::size_t q_count);
    const gpu::GpuGaloisKeysData &galois_keys_for(std::size_t q_count);

    std::string context_id_;
    PoseidonContext context_;
    int cuda_device_id_ = 0;
    std::unique_ptr<CKKSEncoder> encoder_;
    std::unique_ptr<gpu::GpuParameterData> gpu_parameters_;
    std::unique_ptr<gpu::GpuEvaluator> evaluator_;
    std::shared_ptr<const RelinKeys> relin_keys_;
    std::shared_ptr<const GaloisKeys> galois_keys_;
    std::unordered_map<std::size_t, std::unique_ptr<gpu::GpuRelinKeysData>>
        gpu_relin_keys_by_q_count_;
    std::unordered_map<std::size_t, std::unique_ptr<gpu::GpuGaloisKeysData>>
        gpu_galois_keys_by_q_count_;
    std::optional<int> max_rescale_levels_per_op_;
};

} // namespace runtime_api
} // namespace poseidon
