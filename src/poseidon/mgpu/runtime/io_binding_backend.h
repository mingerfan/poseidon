#pragma once

#include "poseidon/mgpu/runtime/sequential_schedule_executor.h"

#include <memory>
#include <unordered_map>

namespace poseidon::mgpu
{

struct BoundScheduleObject
{
    MgpuValueKind kind = MgpuValueKind::Ciphertext;
    std::shared_ptr<void> object;
};

class IoBindingExecutionBackend final : public ScheduleExecutionBackend
{
public:
    explicit IoBindingExecutionBackend(ScheduleExecutionBackend *fallback = nullptr);

    void bind_upload(ValueId id, MgpuValueKind kind, std::shared_ptr<void> object);
    void bind_plain_upload(ValueId id, std::shared_ptr<void> object);
    void bind_cipher_upload(ValueId id, std::shared_ptr<void> object);

    bool has_download(ValueId id) const;
    std::shared_ptr<void> downloaded_object(ValueId id) const;
    const std::unordered_map<ValueId, std::shared_ptr<void>> &downloads() const noexcept;
    void clear_downloads() noexcept;

    void execute(const MgpuOp &op, MgpuObjectStore &object_store) override;

private:
    void execute_upload(
        const MgpuOp &op, MgpuObjectStore &object_store, MgpuValueKind expected_kind);
    void execute_download(const MgpuOp &op, MgpuObjectStore &object_store);

    ScheduleExecutionBackend *fallback_ = nullptr;
    std::unordered_map<ValueId, BoundScheduleObject> uploads_;
    std::unordered_map<ValueId, std::shared_ptr<void>> downloads_;
};

}  // namespace poseidon::mgpu
