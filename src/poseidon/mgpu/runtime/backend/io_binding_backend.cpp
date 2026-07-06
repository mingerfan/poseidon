#include "poseidon/mgpu/runtime/backend/io_binding_backend.h"

#include <sstream>
#include <stdexcept>
#include <utility>

namespace poseidon::mgpu
{
namespace
{

ValueId single_output_id(const MgpuOp &op)
{
    if (op.outputs.size() != 1)
    {
        std::ostringstream stream;
        stream << "expected exactly one output for " << to_string(op.kind);
        throw std::invalid_argument(stream.str());
    }
    return op.outputs[0].id;
}

ValueId single_input_id(const MgpuOp &op)
{
    if (op.inputs.size() != 1)
    {
        std::ostringstream stream;
        stream << "expected exactly one input for " << to_string(op.kind);
        throw std::invalid_argument(stream.str());
    }
    return op.inputs[0].id;
}

std::string value_name(ValueId id)
{
    std::ostringstream stream;
    stream << '%' << id;
    return stream.str();
}

}  // namespace

IoBindingExecutionBackend::IoBindingExecutionBackend(ScheduleExecutionBackend *fallback)
    : fallback_(fallback)
{
}

void IoBindingExecutionBackend::bind_upload(
    ValueId id, MgpuValueKind kind, std::shared_ptr<void> object)
{
    if (id == 0)
    {
        throw std::invalid_argument("upload binding value id 0 is reserved");
    }
    if (object == nullptr)
    {
        throw std::invalid_argument("upload binding object must not be null");
    }

    const auto [_, inserted] =
        uploads_.emplace(id, BoundScheduleObject{ kind, std::move(object) });
    if (!inserted)
    {
        throw std::invalid_argument("duplicate upload binding for " + value_name(id));
    }
}

void IoBindingExecutionBackend::bind_plain_upload(ValueId id, std::shared_ptr<void> object)
{
    bind_upload(id, MgpuValueKind::Plaintext, std::move(object));
}

void IoBindingExecutionBackend::bind_cipher_upload(ValueId id, std::shared_ptr<void> object)
{
    bind_upload(id, MgpuValueKind::Ciphertext, std::move(object));
}

bool IoBindingExecutionBackend::has_download(ValueId id) const
{
    return downloads_.find(id) != downloads_.end();
}

std::shared_ptr<void> IoBindingExecutionBackend::downloaded_object(ValueId id) const
{
    const auto iter = downloads_.find(id);
    if (iter == downloads_.end())
    {
        throw std::out_of_range("missing download for " + value_name(id));
    }
    return iter->second;
}

const std::unordered_map<ValueId, std::shared_ptr<void>> &
IoBindingExecutionBackend::downloads() const noexcept
{
    return downloads_;
}

void IoBindingExecutionBackend::clear_downloads() noexcept
{
    downloads_.clear();
}

void IoBindingExecutionBackend::execute(const MgpuOp &op, MgpuObjectStore &object_store)
{
    switch (op.kind)
    {
    case MgpuOpKind::UploadPlain:
        execute_upload(op, object_store, MgpuValueKind::Plaintext);
        return;
    case MgpuOpKind::UploadCipher:
        execute_upload(op, object_store, MgpuValueKind::Ciphertext);
        return;
    case MgpuOpKind::Download:
        execute_download(op, object_store);
        return;
    default:
        if (fallback_ != nullptr)
        {
            fallback_->execute(op, object_store);
        }
        return;
    }
}

void IoBindingExecutionBackend::execute_upload(
    const MgpuOp &op, MgpuObjectStore &object_store, MgpuValueKind expected_kind)
{
    const ValueId output_id = single_output_id(op);
    const auto iter = uploads_.find(output_id);
    if (iter == uploads_.end())
    {
        throw std::runtime_error("missing upload binding for " + value_name(output_id));
    }

    if (iter->second.kind != expected_kind)
    {
        std::ostringstream stream;
        stream << "upload binding for " << value_name(output_id) << " is "
               << to_string(iter->second.kind) << ", expected " << to_string(expected_kind);
        throw std::runtime_error(stream.str());
    }

    object_store.define(output_id, expected_kind, op.device_id, iter->second.object);
}

void IoBindingExecutionBackend::execute_download(
    const MgpuOp &op, MgpuObjectStore &object_store)
{
    const ValueId input_id = single_input_id(op);
    const MgpuObjectMetadata &metadata = object_store.at(input_id);
    if (metadata.object == nullptr)
    {
        throw std::runtime_error(
            "download input " + value_name(input_id) + " has no object handle");
    }

    downloads_[input_id] = metadata.object;
}

}  // namespace poseidon::mgpu
