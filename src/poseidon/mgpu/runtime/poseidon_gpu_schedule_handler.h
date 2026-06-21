#pragma once

#include "poseidon/ciphertext.h"
#include "poseidon/gpu/gpu_ciphertext.h"
#include "poseidon/gpu/gpu_evaluator.h"
#include "poseidon/gpu/gpu_key.h"
#include "poseidon/gpu/gpu_parameter.h"
#include "poseidon/gpu/gpu_plaintext.h"
#include "poseidon/mgpu/runtime/hevm_io_binding.h"
#include "poseidon/mgpu/runtime/hevm_plaintext_encoding.h"
#include "poseidon/mgpu/runtime/schedule_interpreter.h"
#include "poseidon/plaintext.h"
#include "poseidon/poseidon_context.h"

#include <memory>
#include <unordered_map>
#include <vector>

namespace poseidon
{

class GaloisKeys;
class RelinKeys;

namespace mgpu
{

struct PoseidonGpuDeviceContext
{
    int device_id = 0;
    std::shared_ptr<gpu::GpuParameterData> parameters;
    std::shared_ptr<gpu::GpuEvaluator> evaluator;
    std::shared_ptr<gpu::GpuRelinKeysData> relin_keys;
    std::shared_ptr<gpu::GpuGaloisKeysData> galois_keys;
};

class PoseidonGpuScheduleHandler final : public ScheduleOpHandler
{
public:
    explicit PoseidonGpuScheduleHandler(
        const PoseidonContext &context,
        std::vector<PoseidonGpuDeviceContext> devices = {});

    void add_device(PoseidonGpuDeviceContext device);
    void upload_keys_for_device(
        int device_id, const RelinKeys *relin_keys, const GaloisKeys *galois_keys);

    void bind_plain_upload(ValueId id, std::shared_ptr<const Plaintext> plaintext);
    void bind_cipher_upload(ValueId id, std::shared_ptr<const Ciphertext> ciphertext);

    bool has_plain_download(ValueId id) const;
    bool has_cipher_download(ValueId id) const;
    std::shared_ptr<Plaintext> plain_download(ValueId id) const;
    std::shared_ptr<Ciphertext> cipher_download(ValueId id) const;

    void execute(const MgpuOp &op, MgpuObjectStore &object_store) override;

private:
    struct UploadBinding
    {
        MgpuValueKind kind = MgpuValueKind::Ciphertext;
        std::shared_ptr<const void> object;
    };

    const PoseidonContext &context_;
    std::unordered_map<int, PoseidonGpuDeviceContext> devices_;
    std::unordered_map<ValueId, UploadBinding> uploads_;
    std::unordered_map<ValueId, std::shared_ptr<Plaintext>> plain_downloads_;
    std::unordered_map<ValueId, std::shared_ptr<Ciphertext>> cipher_downloads_;

    PoseidonGpuDeviceContext &device_context(int device_id);
    const PoseidonGpuDeviceContext &device_context(int device_id) const;
    const gpu::GpuEvaluator &evaluator(int device_id);

    void execute_upload_plain(const MgpuOp &op, MgpuObjectStore &object_store);
    void execute_upload_cipher(const MgpuOp &op, MgpuObjectStore &object_store);
    void execute_download(const MgpuOp &op, MgpuObjectStore &object_store);
    void execute_cipher_binary(const MgpuOp &op, MgpuObjectStore &object_store);
    void execute_cipher_plain_binary(const MgpuOp &op, MgpuObjectStore &object_store);
    void execute_cipher_unary(const MgpuOp &op, MgpuObjectStore &object_store);

    std::shared_ptr<gpu::GpuCiphertextData> cipher_object(
        const MgpuObjectStore &object_store, ValueId id) const;
    std::shared_ptr<gpu::GpuPlaintextData> plain_object(
        const MgpuObjectStore &object_store, ValueId id) const;
};

void bind_hevm_cipher_inputs(
    PoseidonGpuScheduleHandler &handler, const HevmIoBindingPlan &plan,
    const std::vector<std::shared_ptr<const Ciphertext>> &cipher_inputs);

void bind_hevm_encoded_plain_inputs(
    PoseidonGpuScheduleHandler &handler,
    const std::vector<HevmEncodedPlaintext> &plaintexts);

std::vector<std::shared_ptr<Ciphertext>> collect_hevm_results(
    const PoseidonGpuScheduleHandler &handler, const HevmIoBindingPlan &plan);

}  // namespace mgpu
}  // namespace poseidon
