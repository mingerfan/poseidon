#pragma once
#include "runtime/waterline.hpp"

#include "poseidon/ciphertext.h"
#include "poseidon/gpu/gpu_ciphertext.h"
#include "poseidon/gpu/gpu_key.h"
#include "poseidon/gpu/gpu_plaintext.h"
#include "poseidon/plaintext.h"
#include "poseidon/poseidon_context.h"
#include "runtime/operator_spec.hpp"
#include "runtime/plan.hpp"

#include <cstdint>
#include <atomic>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#ifdef POSEIDON_RUNTIME_GPU_NCCL
#include <mpi.h>
#endif

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
struct GpuBootstrapData;
struct GpuBootstrapProfile;
class GpuEvaluator;
class GpuParameterData;
} // namespace gpu

namespace runtime_api
{

fhegpu::BootProfile make_native_boot_profile(
    const gpu::GpuBootstrapProfile &profile);

namespace communication
{
class CudaLocalTransfer;
class CudaTransferRequest;
#ifdef POSEIDON_RUNTIME_GPU_NCCL
class NcclMpiTransport;
#endif
} // namespace communication

struct GpuProcessTopology
{
    std::vector<int> device_counts;
    std::vector<int> rank_to_node;
};

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
    gpu::GpuPlaintextData &device_plaintext();
    gpu::GpuCiphertextData &device_ciphertext();

private:
    class ReadyEvent;

    using Storage =
        std::variant<std::shared_ptr<Plaintext>, std::shared_ptr<Ciphertext>,
                     std::shared_ptr<gpu::GpuPlaintextData>,
                     std::shared_ptr<gpu::GpuCiphertextData>>;

    explicit PoseidonGpuValue(Storage storage);

    Storage storage_;
    std::shared_ptr<ReadyEvent> ready_;

    friend class PoseidonGpuApi;
};

class PoseidonGpuApi
{
public:
    using Value = PoseidonGpuValue;
    // Run cross-rank actions on one ordered issuer thread per MPI rank.
    static constexpr bool background_communication_issuer = true;

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
#ifdef POSEIDON_RUNTIME_GPU_NCCL
    PoseidonGpuApi(std::string context_id, PoseidonContext context,
                   MPI_Comm control_comm, std::vector<int> cuda_device_ids,
                   GpuProcessTopology topology,
                   std::shared_ptr<const RelinKeys> relin_keys = {},
                   std::shared_ptr<const GaloisKeys> galois_keys = {},
                   std::shared_ptr<const PublicKey> boot_public_key = {},
                   std::shared_ptr<const SecretKey> boot_secret_key = {});
#endif
    ~PoseidonGpuApi();

    PoseidonGpuApi(const PoseidonGpuApi &) = delete;
    PoseidonGpuApi &operator=(const PoseidonGpuApi &) = delete;

    /**
     * @brief Install one fully uploaded native-bootstrap profile on a device.
     *
     * Bootstrap constants and keys remain full single-device objects. The
     * RuntimePlan selects the profile and device; this method performs no
     * placement or cross-device movement.
     */
    void configure_native_bootstrap(
        std::string operator_profile,
        int logical_device_index,
        gpu::GpuBootstrapData bootstrap_data,
        gpu::GpuRelinKeysData relin_keys,
        gpu::GpuGaloisKeysData galois_keys);

    /**
     * @brief Install a profile while sharing immutable evaluation keys.
     *
     * Shared keys must belong to the selected CUDA device. This overload
     * avoids duplicating large key allocations when one device exposes
     * several native-bootstrap profiles.
     */
    void configure_native_bootstrap(
        std::string operator_profile,
        int logical_device_index,
        gpu::GpuBootstrapData bootstrap_data,
        std::shared_ptr<const gpu::GpuRelinKeysData> relin_keys,
        std::shared_ptr<const gpu::GpuGaloisKeysData> galois_keys);

    /** Install the complete result returned by GpuBootstrapProfileBuilder. */
    void configure_native_bootstrap(
        int logical_device_index,
        gpu::GpuBootstrapProfile profile);

    std::string name() const;
    int mpi_rank() const noexcept;
    int mpi_world_size() const noexcept;
    int local_device_count() const noexcept;
    int cuda_device_id(int logical_device_index) const;
    int nccl_rank(int logical_device_index) const;
    Value encode_plaintext(const fhegpu::ValueDesc &output_desc,
                           const std::vector<double> &slots);
    fhegpu::ResourceUsage value_resources(const fhegpu::ValueDesc &desc) const;
    fhegpu::PrepareResources prepare_resources(const fhegpu::ValueDesc &desc, fhegpu::PrepareBackend backend);
    fhegpu::ResourceUsage instruction_workspace(const fhegpu::Instruction &instruction) const;
    void validate_waterline_resources(const fhegpu::RuntimePlan &plan);
    std::vector<std::uint64_t> waterline_allocator_capacity() const;
    fhegpu::ResourceUsage waterline_fixed_resources() const;
    struct DeviceMeasurements {
        std::uint64_t allocated_live=0,allocated_peak=0,pool_reserved=0,pool_capacity=0;
        std::uint64_t gpu_encode_calls=0,gpu_raw_h2d_bytes=0,gpu_raw_h2d_nanoseconds=0,gpu_encode_nanoseconds=0;
        // Completed CUDA intervals around foreground input dependencies;
        // excludes CPU scheduling waits and same-stream ordering.
        std::uint64_t consumer_wait_calls=0,consumer_wait_nanoseconds=0;
        bool managed_pool=false;
    };
    struct WaterlineMeasurements {
        std::uint64_t cpu_prepare_calls=0,cpu_encode_nanoseconds=0,cpu_plain_h2d_bytes=0;
        std::uint64_t transport_h2d_calls=0,transport_h2d_bytes=0,transport_h2d_nanoseconds=0;
        std::vector<DeviceMeasurements> devices;
    };
    WaterlineMeasurements waterline_measurements();
    Value prepare_plaintext_async(const fhegpu::ValueDesc &desc, const std::vector<double> &slots, fhegpu::PrepareBackend backend);
    bool is_ready(Value &value);
    Value compute(const fhegpu::ComputeOp &op, const std::vector<Value> &inputs);
    bool supports_reuse(const fhegpu::ComputeOp &op) const;
    Value compute_reuse(const fhegpu::ComputeOp &op, Value input,
                        const std::vector<Value> &other_inputs);
    CommHandle communicate_async(const fhegpu::CommAction &action,
                                 const std::vector<Value> &local_inputs,
                                 const std::vector<fhegpu::ValueDesc> &output_descs);
    // One slot per action output; each value is moved out exactly once.
    // Present device values carry their CUDA dependency.
    std::vector<std::optional<Value>> posted_outputs(CommHandle &handle);
    std::vector<std::optional<Value>> wait(CommHandle &handle);
    void collect_completed();
    void drain();
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
#ifdef POSEIDON_RUNTIME_GPU_NCCL
    int nccl_rank_for_place(const fhegpu::Place &place) const;
    CommHandle communicate_distributed(const fhegpu::CommAction &action,
                                       const std::vector<Value> &local_inputs,
                                       const std::vector<fhegpu::ValueDesc> &output_descs);
#endif
    struct InFlightWork;
    void wait_on_compute_inputs(DeviceState &device, const std::vector<Value> &inputs,
                                const Value *first = nullptr);
    void retain_in_flight(std::shared_ptr<PoseidonGpuValue::ReadyEvent> completion,
                          const std::vector<Value> &values,
                          const std::vector<std::shared_ptr<void>> &resources = {});
    void retain_communication(CommHandle::State &state,
                              const std::vector<Value> &local_inputs);
    void synchronize_device(int cuda_device_id) const;
    void synchronize_all_devices() const;
    std::size_t q_count_for_level(int level) const;
    void materialize_relin_keys(DeviceState &device, std::size_t q_count);
    void materialize_galois_keys(
        DeviceState &device, std::size_t q_count,
        const std::set<std::uint32_t> &galois_elts);
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
    std::optional<int> waterline_rescale_workspace_levels_;
    std::vector<std::uint64_t> waterline_allocator_capacity_;
#ifdef POSEIDON_RUNTIME_GPU_NCCL
    std::unique_ptr<communication::NcclMpiTransport> nccl_transport_;
#endif
    int mpi_rank_ = 0;
    int mpi_world_size_ = 1;
    std::vector<int> device_counts_;
    std::vector<int> rank_to_node_;
    std::vector<InFlightWork> in_flight_work_;
    std::mutex in_flight_mutex_;
    std::atomic<std::uint64_t> cpu_prepare_calls_{0},cpu_encode_nanoseconds_{0},cpu_plain_h2d_bytes_{0};
};

} // namespace runtime_api
} // namespace poseidon
