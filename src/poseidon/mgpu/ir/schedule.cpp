#include "poseidon/mgpu/ir/schedule.h"

#include <ostream>
#include <sstream>

namespace poseidon::mgpu
{
namespace
{

void write_value_list(std::ostream &stream, const std::vector<MgpuValueRef> &values)
{
    stream << '[';
    for (std::size_t i = 0; i < values.size(); ++i)
    {
        if (i > 0)
        {
            stream << ", ";
        }
        stream << '%' << values[i].id;
    }
    stream << ']';
}

}  // namespace

const char *to_string(MgpuValueKind kind) noexcept
{
    switch (kind)
    {
    case MgpuValueKind::Plaintext:
        return "plaintext";
    case MgpuValueKind::Ciphertext:
        return "ciphertext";
    }
    return "unknown";
}

const char *to_string(MgpuOpKind kind) noexcept
{
    switch (kind)
    {
    case MgpuOpKind::UploadPlain:
        return "upload_plain";
    case MgpuOpKind::UploadCipher:
        return "upload_cipher";
    case MgpuOpKind::CopyPlain:
        return "copy_plain";
    case MgpuOpKind::CopyCipher:
        return "copy_cipher";
    case MgpuOpKind::Add:
        return "add";
    case MgpuOpKind::Sub:
        return "sub";
    case MgpuOpKind::MultiplyPlain:
        return "multiply_plain";
    case MgpuOpKind::Multiply:
        return "multiply";
    case MgpuOpKind::Relinearize:
        return "relinearize";
    case MgpuOpKind::Rescale:
        return "rescale";
    case MgpuOpKind::Rotate:
        return "rotate";
    case MgpuOpKind::BootstrapFallback:
        return "bootstrap_fallback";
    case MgpuOpKind::Download:
        return "download";
    }
    return "unknown";
}

std::optional<MgpuOpKind> mgpu_op_kind_from_string(std::string_view name) noexcept
{
    if (name == "upload_plain")
    {
        return MgpuOpKind::UploadPlain;
    }
    if (name == "upload_cipher")
    {
        return MgpuOpKind::UploadCipher;
    }
    if (name == "copy_plain")
    {
        return MgpuOpKind::CopyPlain;
    }
    if (name == "copy_cipher")
    {
        return MgpuOpKind::CopyCipher;
    }
    if (name == "add")
    {
        return MgpuOpKind::Add;
    }
    if (name == "sub")
    {
        return MgpuOpKind::Sub;
    }
    if (name == "multiply_plain")
    {
        return MgpuOpKind::MultiplyPlain;
    }
    if (name == "multiply")
    {
        return MgpuOpKind::Multiply;
    }
    if (name == "relinearize")
    {
        return MgpuOpKind::Relinearize;
    }
    if (name == "rescale")
    {
        return MgpuOpKind::Rescale;
    }
    if (name == "rotate")
    {
        return MgpuOpKind::Rotate;
    }
    if (name == "bootstrap_fallback")
    {
        return MgpuOpKind::BootstrapFallback;
    }
    if (name == "download")
    {
        return MgpuOpKind::Download;
    }
    return std::nullopt;
}

bool is_upload_op(MgpuOpKind kind) noexcept
{
    return kind == MgpuOpKind::UploadPlain || kind == MgpuOpKind::UploadCipher;
}

bool is_copy_op(MgpuOpKind kind) noexcept
{
    return kind == MgpuOpKind::CopyPlain || kind == MgpuOpKind::CopyCipher;
}

bool is_download_op(MgpuOpKind kind) noexcept
{
    return kind == MgpuOpKind::Download;
}

bool is_compute_op(MgpuOpKind kind) noexcept
{
    return !is_upload_op(kind) && !is_copy_op(kind) && !is_download_op(kind);
}

std::string dump_schedule(const MgpuSchedule &schedule)
{
    std::ostringstream stream;
    dump_schedule(stream, schedule);
    return stream.str();
}

void dump_schedule(std::ostream &stream, const MgpuSchedule &schedule)
{
    stream << "mgpu.schedule {\n";
    for (std::size_t i = 0; i < schedule.ops.size(); ++i)
    {
        const MgpuOp &op = schedule.ops[i];
        stream << "  #" << i << " ";

        if (!op.outputs.empty())
        {
            write_value_list(stream, op.outputs);
            stream << " = ";
        }

        stream << "mgpu." << to_string(op.kind) << " device=" << op.device_id;

        if (!op.inputs.empty())
        {
            stream << " inputs=";
            write_value_list(stream, op.inputs);
        }

        if (!op.debug_name.empty())
        {
            stream << " name=\"" << op.debug_name << '"';
        }

        stream << '\n';
    }
    stream << "}\n";
}

}  // namespace poseidon::mgpu
