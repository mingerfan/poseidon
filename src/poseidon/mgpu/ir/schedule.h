#pragma once

#include <cstdint>
#include <iosfwd>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace poseidon::mgpu
{

using ValueId = std::uint64_t;

enum class MgpuValueKind
{
    Plaintext,
    Ciphertext
};

enum class MgpuOpKind
{
    UploadPlain,
    UploadCipher,
    CopyPlain,
    CopyCipher,
    Add,
    Sub,
    MultiplyPlain,
    Multiply,
    Relinearize,
    Rescale,
    Rotate,
    BootstrapFallback,
    Download
};

struct MgpuValueRef
{
    ValueId id = 0;
};

struct MgpuOp
{
    MgpuOpKind kind = MgpuOpKind::Download;
    int device_id = 0;
    std::vector<MgpuValueRef> inputs;
    std::vector<MgpuValueRef> outputs;
    std::string debug_name;
};

struct MgpuSchedule
{
    std::vector<MgpuOp> ops;
};

const char *to_string(MgpuValueKind kind) noexcept;
const char *to_string(MgpuOpKind kind) noexcept;
std::optional<MgpuOpKind> mgpu_op_kind_from_string(std::string_view name) noexcept;

bool is_upload_op(MgpuOpKind kind) noexcept;
bool is_copy_op(MgpuOpKind kind) noexcept;
bool is_download_op(MgpuOpKind kind) noexcept;
bool is_compute_op(MgpuOpKind kind) noexcept;

std::string dump_schedule(const MgpuSchedule &schedule);
void dump_schedule(std::ostream &stream, const MgpuSchedule &schedule);

}  // namespace poseidon::mgpu
