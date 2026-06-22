#include "poseidon/mgpu/runtime/poseidon_gpu_execution_backend.h"

#include "poseidon/gpu/gpu_uploader.h"
#include "poseidon/key/galoiskeys.h"
#include "poseidon/key/relinkeys.h"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace poseidon::mgpu
{
namespace
{

ValueId output_id(const MgpuOp &op)
{
    if (op.outputs.size() != 1)
    {
        throw std::invalid_argument("operation must have exactly one output");
    }
    return op.outputs[0].id;
}

ValueId input_id(const MgpuOp &op, std::size_t index)
{
    if (index >= op.inputs.size())
    {
        throw std::invalid_argument("operation input index is out of range");
    }
    return op.inputs[index].id;
}

std::string value_name(ValueId id)
{
    std::ostringstream stream;
    stream << '%' << id;
    return stream.str();
}

void ensure_full_single_device_ciphertext(
    const gpu::GpuCiphertextData &ciphertext, int expected_device)
{
    if (ciphertext.fields_.size() != 1)
    {
        throw std::invalid_argument(
            "PoseidonGpuExecutionBackend V1 requires full single-device ciphertexts");
    }
    if (ciphertext.fields_[0].device_id != expected_device)
    {
        std::ostringstream stream;
        stream << "ciphertext is on device " << ciphertext.fields_[0].device_id
               << ", expected device " << expected_device;
        throw std::invalid_argument(stream.str());
    }
}

void ensure_full_single_device_plaintext(
    const gpu::GpuPlaintextData &plaintext, int expected_device)
{
    if (plaintext.fields_.size() != 1)
    {
        throw std::invalid_argument(
            "PoseidonGpuExecutionBackend V1 requires full single-device plaintexts");
    }
    if (plaintext.fields_[0].device_id != expected_device)
    {
        std::ostringstream stream;
        stream << "plaintext is on device " << plaintext.fields_[0].device_id
               << ", expected device " << expected_device;
        throw std::invalid_argument(stream.str());
    }
}

int require_int_attr(const MgpuOp &op, const char *name)
{
    const auto iter = op.integer_attributes.find(name);
    if (iter == op.integer_attributes.end())
    {
        throw std::invalid_argument(std::string("missing integer attribute '") + name + "'");
    }
    if (iter->second < static_cast<std::int64_t>(std::numeric_limits<int>::min()) ||
        iter->second > static_cast<std::int64_t>(std::numeric_limits<int>::max()))
    {
        throw std::out_of_range(std::string("integer attribute out of int range: ") + name);
    }
    return static_cast<int>(iter->second);
}

void validate_hevm_index_range(std::uint64_t index, std::size_t count, const char *what)
{
    if (index >= count)
    {
        std::ostringstream stream;
        stream << "HEVM " << what << " index " << index
               << " is outside object count " << count;
        throw std::invalid_argument(stream.str());
    }
}

std::shared_ptr<gpu::GpuCiphertextData> make_cipher_handle(
    gpu::GpuCiphertextData &&ciphertext)
{
    return std::make_shared<gpu::GpuCiphertextData>(std::move(ciphertext));
}

std::shared_ptr<gpu::GpuPlaintextData> make_plain_handle(
    gpu::GpuPlaintextData &&plaintext)
{
    return std::make_shared<gpu::GpuPlaintextData>(std::move(plaintext));
}

}  // namespace

PoseidonGpuExecutionBackend::PoseidonGpuExecutionBackend(
    const PoseidonContext &context,
    std::vector<PoseidonGpuDeviceContext> devices)
    : context_(context)
{
    for (auto &device : devices)
    {
        add_device(std::move(device));
    }
}

void PoseidonGpuExecutionBackend::add_device(PoseidonGpuDeviceContext device)
{
    if (device.device_id < 0)
    {
        throw std::invalid_argument("device id must be non-negative");
    }
    if (device.parameters == nullptr)
    {
        device.parameters = std::make_shared<gpu::GpuParameterData>(context_, device.device_id);
    }
    if (device.evaluator == nullptr)
    {
        device.evaluator = std::make_shared<gpu::GpuEvaluator>(*device.parameters);
    }

    const auto [_, inserted] = devices_.emplace(device.device_id, std::move(device));
    if (!inserted)
    {
        throw std::invalid_argument("duplicate Poseidon GPU device context");
    }
}

void PoseidonGpuExecutionBackend::upload_keys_for_device(
    int device_id, const RelinKeys *relin_keys, const GaloisKeys *galois_keys)
{
    PoseidonGpuDeviceContext &device = device_context(device_id);
    if (relin_keys != nullptr)
    {
        device.relin_keys = std::make_shared<gpu::GpuRelinKeysData>(
            gpu::GpuUploader::upload_relin_keys(*relin_keys, device_id));
    }
    if (galois_keys != nullptr)
    {
        device.galois_keys = std::make_shared<gpu::GpuGaloisKeysData>(
            gpu::GpuUploader::upload_galois_keys(*galois_keys, device_id));
    }
}

void PoseidonGpuExecutionBackend::bind_plain_upload(
    ValueId id, std::shared_ptr<const Plaintext> plaintext)
{
    if (id == 0 || plaintext == nullptr)
    {
        throw std::invalid_argument("invalid plaintext upload binding");
    }
    const auto [_, inserted] =
        uploads_.emplace(id, UploadBinding{ MgpuValueKind::Plaintext, std::move(plaintext) });
    if (!inserted)
    {
        throw std::invalid_argument("duplicate upload binding for " + value_name(id));
    }
}

void PoseidonGpuExecutionBackend::bind_cipher_upload(
    ValueId id, std::shared_ptr<const Ciphertext> ciphertext)
{
    if (id == 0 || ciphertext == nullptr)
    {
        throw std::invalid_argument("invalid ciphertext upload binding");
    }
    const auto [_, inserted] =
        uploads_.emplace(id, UploadBinding{ MgpuValueKind::Ciphertext, std::move(ciphertext) });
    if (!inserted)
    {
        throw std::invalid_argument("duplicate upload binding for " + value_name(id));
    }
}

bool PoseidonGpuExecutionBackend::has_plain_download(ValueId id) const
{
    return plain_downloads_.find(id) != plain_downloads_.end();
}

bool PoseidonGpuExecutionBackend::has_cipher_download(ValueId id) const
{
    return cipher_downloads_.find(id) != cipher_downloads_.end();
}

std::shared_ptr<Plaintext> PoseidonGpuExecutionBackend::plain_download(ValueId id) const
{
    const auto iter = plain_downloads_.find(id);
    if (iter == plain_downloads_.end())
    {
        throw std::out_of_range("missing plaintext download for " + value_name(id));
    }
    return iter->second;
}

std::shared_ptr<Ciphertext> PoseidonGpuExecutionBackend::cipher_download(ValueId id) const
{
    const auto iter = cipher_downloads_.find(id);
    if (iter == cipher_downloads_.end())
    {
        throw std::out_of_range("missing ciphertext download for " + value_name(id));
    }
    return iter->second;
}

void PoseidonGpuExecutionBackend::execute(const MgpuOp &op, MgpuObjectStore &object_store)
{
    switch (op.kind)
    {
    case MgpuOpKind::UploadPlain:
        execute_upload_plain(op, object_store);
        return;
    case MgpuOpKind::UploadCipher:
        execute_upload_cipher(op, object_store);
        return;
    case MgpuOpKind::Download:
        execute_download(op, object_store);
        return;
    case MgpuOpKind::Add:
    case MgpuOpKind::Sub:
    case MgpuOpKind::Multiply:
        execute_cipher_binary(op, object_store);
        return;
    case MgpuOpKind::AddPlain:
    case MgpuOpKind::MultiplyPlain:
        execute_cipher_plain_binary(op, object_store);
        return;
    case MgpuOpKind::Negate:
    case MgpuOpKind::Relinearize:
    case MgpuOpKind::Rescale:
    case MgpuOpKind::Rotate:
        execute_cipher_unary(op, object_store);
        return;
    case MgpuOpKind::CopyPlain:
    case MgpuOpKind::CopyCipher:
        throw std::runtime_error(
            "copy ops must be handled by the multi-GPU communication layer");
    case MgpuOpKind::BootstrapFallback:
        throw std::runtime_error("Poseidon GPU bootstrap fallback is not implemented");
    }
}

PoseidonGpuDeviceContext &PoseidonGpuExecutionBackend::device_context(int device_id)
{
    const auto iter = devices_.find(device_id);
    if (iter == devices_.end())
    {
        add_device(PoseidonGpuDeviceContext{ device_id });
        return devices_.at(device_id);
    }
    return iter->second;
}

const PoseidonGpuDeviceContext &PoseidonGpuExecutionBackend::device_context(int device_id) const
{
    const auto iter = devices_.find(device_id);
    if (iter == devices_.end())
    {
        throw std::out_of_range("missing Poseidon GPU device context");
    }
    return iter->second;
}

const gpu::GpuEvaluator &PoseidonGpuExecutionBackend::evaluator(int device_id)
{
    const PoseidonGpuDeviceContext &device = device_context(device_id);
    if (device.evaluator == nullptr)
    {
        throw std::runtime_error("Poseidon GPU evaluator is not initialized");
    }
    return *device.evaluator;
}

void PoseidonGpuExecutionBackend::execute_upload_plain(
    const MgpuOp &op, MgpuObjectStore &object_store)
{
    const ValueId id = output_id(op);
    const auto iter = uploads_.find(id);
    if (iter == uploads_.end() || iter->second.kind != MgpuValueKind::Plaintext)
    {
        throw std::runtime_error("missing plaintext upload binding for " + value_name(id));
    }
    device_context(op.device_id);
    const auto plaintext = std::static_pointer_cast<const Plaintext>(iter->second.object);
    auto gpu_plaintext =
        make_plain_handle(gpu::GpuUploader::upload_plaintext(*plaintext, op.device_id));
    ensure_full_single_device_plaintext(*gpu_plaintext, op.device_id);
    object_store.define(id, MgpuValueKind::Plaintext, op.device_id, std::move(gpu_plaintext));
}

void PoseidonGpuExecutionBackend::execute_upload_cipher(
    const MgpuOp &op, MgpuObjectStore &object_store)
{
    const ValueId id = output_id(op);
    const auto iter = uploads_.find(id);
    if (iter == uploads_.end() || iter->second.kind != MgpuValueKind::Ciphertext)
    {
        throw std::runtime_error("missing ciphertext upload binding for " + value_name(id));
    }
    device_context(op.device_id);
    const auto ciphertext = std::static_pointer_cast<const Ciphertext>(iter->second.object);
    auto gpu_ciphertext = make_cipher_handle(
        gpu::GpuUploader::upload_ciphertext(*ciphertext, op.device_id));
    ensure_full_single_device_ciphertext(*gpu_ciphertext, op.device_id);
    object_store.define(id, MgpuValueKind::Ciphertext, op.device_id, std::move(gpu_ciphertext));
}

void PoseidonGpuExecutionBackend::execute_download(
    const MgpuOp &op, MgpuObjectStore &object_store)
{
    const ValueId id = input_id(op, 0);
    const MgpuObjectMetadata &metadata = object_store.at(id);
    if (metadata.kind == MgpuValueKind::Plaintext)
    {
        auto plaintext = std::make_shared<Plaintext>();
        gpu::GpuUploader::download_plaintext(*plain_object(object_store, id), *plaintext, context_);
        plain_downloads_[id] = std::move(plaintext);
        return;
    }

    auto ciphertext = std::make_shared<Ciphertext>();
    gpu::GpuUploader::download_ciphertext(*cipher_object(object_store, id), *ciphertext, context_);
    cipher_downloads_[id] = std::move(ciphertext);
}

void PoseidonGpuExecutionBackend::execute_cipher_binary(
    const MgpuOp &op, MgpuObjectStore &object_store)
{
    auto left = cipher_object(object_store, input_id(op, 0));
    auto right = cipher_object(object_store, input_id(op, 1));
    ensure_full_single_device_ciphertext(*left, op.device_id);
    ensure_full_single_device_ciphertext(*right, op.device_id);

    gpu::GpuCiphertextData result;
    switch (op.kind)
    {
    case MgpuOpKind::Add:
        evaluator(op.device_id).add(*left, *right, result);
        break;
    case MgpuOpKind::Sub:
        evaluator(op.device_id).sub(*left, *right, result);
        break;
    case MgpuOpKind::Multiply:
        evaluator(op.device_id).multiply(*left, *right, result);
        break;
    default:
        throw std::logic_error("unexpected cipher binary op");
    }

    object_store.define(
        output_id(op), MgpuValueKind::Ciphertext, op.device_id,
        make_cipher_handle(std::move(result)));
}

void PoseidonGpuExecutionBackend::execute_cipher_plain_binary(
    const MgpuOp &op, MgpuObjectStore &object_store)
{
    auto cipher = cipher_object(object_store, input_id(op, 0));
    auto plain = plain_object(object_store, input_id(op, 1));
    ensure_full_single_device_ciphertext(*cipher, op.device_id);
    ensure_full_single_device_plaintext(*plain, op.device_id);

    gpu::GpuCiphertextData result;
    switch (op.kind)
    {
    case MgpuOpKind::AddPlain:
        evaluator(op.device_id).add_plain(*cipher, *plain, result);
        break;
    case MgpuOpKind::MultiplyPlain:
        evaluator(op.device_id).multiply_plain(*cipher, *plain, result);
        break;
    default:
        throw std::logic_error("unexpected cipher/plain binary op");
    }

    object_store.define(
        output_id(op), MgpuValueKind::Ciphertext, op.device_id,
        make_cipher_handle(std::move(result)));
}

void PoseidonGpuExecutionBackend::execute_cipher_unary(
    const MgpuOp &op, MgpuObjectStore &object_store)
{
    auto cipher = cipher_object(object_store, input_id(op, 0));
    ensure_full_single_device_ciphertext(*cipher, op.device_id);

    gpu::GpuCiphertextData result;
    switch (op.kind)
    {
    case MgpuOpKind::Negate:
        evaluator(op.device_id).negate(*cipher, result);
        break;
    case MgpuOpKind::Rescale:
        evaluator(op.device_id).rescale(*cipher, result);
        break;
    case MgpuOpKind::Relinearize: {
        const auto &device = device_context(op.device_id);
        if (device.relin_keys == nullptr)
        {
            throw std::runtime_error("relinearize requires uploaded relin keys");
        }
        evaluator(op.device_id).relinearize(*cipher, *device.relin_keys, result);
        break;
    }
    case MgpuOpKind::Rotate: {
        const auto &device = device_context(op.device_id);
        if (device.galois_keys == nullptr)
        {
            throw std::runtime_error("rotate requires uploaded galois keys");
        }
        evaluator(op.device_id)
            .rotate(*cipher, require_int_attr(op, "rotate_step"), *device.galois_keys, result);
        break;
    }
    default:
        throw std::logic_error("unexpected cipher unary op");
    }

    object_store.define(
        output_id(op), MgpuValueKind::Ciphertext, op.device_id,
        make_cipher_handle(std::move(result)));
}

std::shared_ptr<gpu::GpuCiphertextData> PoseidonGpuExecutionBackend::cipher_object(
    const MgpuObjectStore &object_store, ValueId id) const
{
    const MgpuObjectMetadata &metadata = object_store.at(id);
    if (metadata.kind != MgpuValueKind::Ciphertext)
    {
        throw std::invalid_argument(value_name(id) + " is not a ciphertext");
    }
    if (metadata.object == nullptr)
    {
        throw std::invalid_argument(value_name(id) + " has no GPU ciphertext handle");
    }
    return std::static_pointer_cast<gpu::GpuCiphertextData>(metadata.object);
}

std::shared_ptr<gpu::GpuPlaintextData> PoseidonGpuExecutionBackend::plain_object(
    const MgpuObjectStore &object_store, ValueId id) const
{
    const MgpuObjectMetadata &metadata = object_store.at(id);
    if (metadata.kind != MgpuValueKind::Plaintext)
    {
        throw std::invalid_argument(value_name(id) + " is not a plaintext");
    }
    if (metadata.object == nullptr)
    {
        throw std::invalid_argument(value_name(id) + " has no GPU plaintext handle");
    }
    return std::static_pointer_cast<gpu::GpuPlaintextData>(metadata.object);
}

void bind_hevm_cipher_inputs(
    PoseidonGpuExecutionBackend &backend, const HevmIoBindingPlan &plan,
    const std::vector<std::shared_ptr<const Ciphertext>> &cipher_inputs)
{
    if (cipher_inputs.size() != plan.cipher_inputs.size())
    {
        std::ostringstream stream;
        stream << "HEVM cipher input object count " << cipher_inputs.size()
               << " does not match schedule input count " << plan.cipher_inputs.size();
        throw std::invalid_argument(stream.str());
    }

    for (const HevmCipherInputSlot &slot : plan.cipher_inputs)
    {
        validate_hevm_index_range(slot.index, cipher_inputs.size(), "cipher input");
        const auto &ciphertext = cipher_inputs[static_cast<std::size_t>(slot.index)];
        if (ciphertext == nullptr)
        {
            std::ostringstream stream;
            stream << "HEVM cipher input object at index " << slot.index
                   << " must not be null";
            throw std::invalid_argument(stream.str());
        }
        backend.bind_cipher_upload(slot.value_id, ciphertext);
    }
}

void bind_hevm_encoded_plain_inputs(
    PoseidonGpuExecutionBackend &backend,
    const std::vector<HevmEncodedPlaintext> &plaintexts)
{
    for (const HevmEncodedPlaintext &plaintext : plaintexts)
    {
        if (plaintext.plaintext == nullptr)
        {
            throw std::invalid_argument("encoded HEVM plaintext must not be null");
        }
        backend.bind_plain_upload(plaintext.value_id, plaintext.plaintext);
    }
}

std::vector<std::shared_ptr<Ciphertext>> collect_hevm_results(
    const PoseidonGpuExecutionBackend &backend, const HevmIoBindingPlan &plan)
{
    std::vector<std::shared_ptr<Ciphertext>> results(plan.results.size());
    for (const HevmResultSlot &slot : plan.results)
    {
        validate_hevm_index_range(slot.index, results.size(), "result");
        if (!backend.has_cipher_download(slot.value_id))
        {
            throw std::out_of_range(
                "missing HEVM ciphertext result for " + value_name(slot.value_id));
        }
        results[static_cast<std::size_t>(slot.index)] =
            backend.cipher_download(slot.value_id);
    }
    return results;
}

}  // namespace poseidon::mgpu
