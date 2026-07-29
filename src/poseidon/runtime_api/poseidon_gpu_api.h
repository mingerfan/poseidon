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
#include <variant>
#include <vector>

namespace poseidon
{
class CKKSEncoder;
class Decryptor;
class Encryptor;
class GaloisKeys;
class PublicKey;
class RelinKeys;
class SecretKey;

namespace gpu
{
class GpuEvaluator;
class GpuParameterData;
} // namespace gpu

namespace runtime_api
{

namespace communication
{
class CudaLocalTransfer;
} // namespace communication

class PoseidonGpuValue
{
public:
    PoseidonGpuValue(const PoseidonGpuValue &) = default;
    PoseidonGpuValue(PoseidonGpuValue &&) noexcept = default;
    PoseidonGpuValue &operator=(const PoseidonGpuValue &other);
    PoseidonGpuValue &operator=(PoseidonGpuValue &&other) noexcept;

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
    class Completion;

    using Storage =
        std::variant<std::shared_ptr<Plaintext>, std::shared_ptr<Ciphertext>,
                     std::shared_ptr<gpu::GpuPlaintextData>,
                     std::shared_ptr<gpu::GpuCiphertextData>>;

    explicit PoseidonGpuValue(Storage storage);

    Storage storage_;
    std::shared_ptr<Completion> completion_;

    friend class PoseidonGpuApi;
};

class PoseidonGpuApi
{
public:
    using Value = PoseidonGpuValue;

    struct CommHandle
    {
        CommHandle();
        ~CommHandle();

        CommHandle(const CommHandle &) = delete;
        CommHandle &operator=(const CommHandle &) = delete;
        CommHandle(CommHandle &&) noexcept;
        CommHandle &operator=(CommHandle &&) noexcept;

    private:
        struct State;
        std::unique_ptr<State> state_;

        friend class PoseidonGpuApi;
    };

    PoseidonGpuApi(std::string context_id, PoseidonContext context, int cuda_device_id,
                   std::shared_ptr<const RelinKeys> relin_keys = {},
                   std::shared_ptr<const GaloisKeys> galois_keys = {},
                   std::shared_ptr<const PublicKey> boot_public_key = {},
                   std::shared_ptr<const SecretKey> boot_secret_key = {});
    PoseidonGpuApi(std::string context_id, PoseidonContext context,
                   std::vector<int> cuda_device_ids,
                   std::shared_ptr<const RelinKeys> relin_keys = {},
                   std::shared_ptr<const GaloisKeys> galois_keys = {},
                   std::shared_ptr<const PublicKey> boot_public_key = {},
                   std::shared_ptr<const SecretKey> boot_secret_key = {});
    ~PoseidonGpuApi();

    PoseidonGpuApi(const PoseidonGpuApi &) = delete;
    PoseidonGpuApi &operator=(const PoseidonGpuApi &) = delete;

    std::string name() const;
    int local_device_count() const noexcept;
    int cuda_device_id(int logical_device_index) const;
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
    struct DeviceState;

    DeviceState &device_state(const fhegpu::Place &place, const char *where);
    const DeviceState &device_state(const fhegpu::Place &place,
                                    const char *where) const;
    DeviceState &device_state(int logical_device_index);
    const DeviceState &device_state(int logical_device_index) const;
    void synchronize_device(int cuda_device_id) const;
    void synchronize_all_devices() const;
    std::size_t q_count_for_level(int level) const;
    void materialize_relin_keys(DeviceState &device, std::size_t q_count);
    void materialize_galois_keys(DeviceState &device, std::size_t q_count);
    const gpu::GpuRelinKeysData &relin_keys_for(DeviceState &device,
                                                std::size_t q_count);
    const gpu::GpuGaloisKeysData &galois_keys_for(DeviceState &device,
                                                  std::size_t q_count);

    std::string context_id_;
    PoseidonContext context_;
    std::unique_ptr<CKKSEncoder> encoder_;
    std::unique_ptr<Encryptor> boot_encryptor_;
    std::unique_ptr<Decryptor> boot_decryptor_;
    std::vector<std::shared_ptr<DeviceState>> devices_;
    std::unique_ptr<communication::CudaLocalTransfer> cuda_transfer_;
    std::shared_ptr<const RelinKeys> relin_keys_;
    std::shared_ptr<const GaloisKeys> galois_keys_;
    std::optional<int> max_rescale_levels_per_op_;
};

} // namespace runtime_api
} // namespace poseidon
