#include "poseidon/mgpu/ir/schedule_summary.h"

#include "poseidon/util/json.h"

#include <sstream>
#include <stdexcept>

namespace poseidon::mgpu
{
namespace
{

using Json = nlohmann::json;

constexpr MgpuOpKindOrder kOpKindOrder{
    MgpuOpKind::UploadCipher,
    MgpuOpKind::UploadPlain,
    MgpuOpKind::CopyCipher,
    MgpuOpKind::CopyPlain,
    MgpuOpKind::Add,
    MgpuOpKind::AddPlain,
    MgpuOpKind::Sub,
    MgpuOpKind::Negate,
    MgpuOpKind::Multiply,
    MgpuOpKind::MultiplyPlain,
    MgpuOpKind::Relinearize,
    MgpuOpKind::Rescale,
    MgpuOpKind::Rotate,
    MgpuOpKind::BootstrapFallback,
    MgpuOpKind::Download,
};

void increment_kind_count(std::vector<MgpuOpKindCount> &counts, MgpuOpKind kind)
{
    for (MgpuOpKindCount &count : counts)
    {
        if (count.kind == kind)
        {
            ++count.count;
            return;
        }
    }
}

Json op_counts_to_json(const std::vector<MgpuOpKindCount> &counts)
{
    Json result = Json::object();
    for (const MgpuOpKindCount &count : counts)
    {
        result[to_string(count.kind)] = count.count;
    }
    return result;
}

Json device_counts_to_json(const std::vector<MgpuDeviceOpCount> &counts)
{
    Json result = Json::array();
    for (const MgpuDeviceOpCount &count : counts)
    {
        result.push_back(Json{
            { "device_id", count.device_id },
            { "ops", count.count },
        });
    }
    return result;
}

}  // namespace

const MgpuOpKindOrder &ordered_mgpu_op_kinds() noexcept
{
    return kOpKindOrder;
}

MgpuScheduleSummary summarize_schedule(const MgpuSchedule &schedule, int device_count)
{
    if (device_count < 0)
    {
        throw std::invalid_argument("device_count must be non-negative");
    }

    MgpuScheduleSummary summary;
    summary.total_ops = schedule.ops.size();
    summary.op_counts.reserve(kOpKindOrder.size());
    for (const MgpuOpKind kind : kOpKindOrder)
    {
        summary.op_counts.push_back(MgpuOpKindCount{ kind, 0 });
    }

    summary.device_op_counts.reserve(static_cast<std::size_t>(device_count));
    for (int device = 0; device < device_count; ++device)
    {
        summary.device_op_counts.push_back(MgpuDeviceOpCount{ device, 0 });
    }

    for (const MgpuOp &op : schedule.ops)
    {
        increment_kind_count(summary.op_counts, op.kind);
        if (is_upload_op(op.kind))
        {
            ++summary.upload_ops;
        }
        else if (is_copy_op(op.kind))
        {
            ++summary.copy_ops;
        }
        else if (is_download_op(op.kind))
        {
            ++summary.download_ops;
        }
        else
        {
            ++summary.compute_ops;
        }

        if (op.device_id < 0)
        {
            ++summary.unassigned_device_ops;
        }
        else if (op.device_id < device_count)
        {
            ++summary.device_op_counts[static_cast<std::size_t>(op.device_id)].count;
        }
        else
        {
            ++summary.invalid_device_ops;
        }
    }

    return summary;
}

std::string dump_schedule_summary(const MgpuScheduleSummary &summary)
{
    std::ostringstream stream;
    dump_schedule_summary(stream, summary);
    return stream.str();
}

void dump_schedule_summary(std::ostream &stream, const MgpuScheduleSummary &summary)
{
    stream << "schedule_ops: " << summary.total_ops << '\n';
    stream << "op_counts:\n";
    for (const MgpuOpKindCount &count : summary.op_counts)
    {
        if (count.count > 0)
        {
            stream << "  " << to_string(count.kind) << ": " << count.count << '\n';
        }
    }

    stream << "device_op_counts:\n";
    for (const MgpuDeviceOpCount &count : summary.device_op_counts)
    {
        stream << "  device " << count.device_id << ": " << count.count << '\n';
    }
    if (summary.unassigned_device_ops > 0)
    {
        stream << "  unassigned: " << summary.unassigned_device_ops << '\n';
    }
    if (summary.invalid_device_ops > 0)
    {
        stream << "  invalid: " << summary.invalid_device_ops << '\n';
    }
}

std::string schedule_summary_to_json(const MgpuScheduleSummary &summary, int indent)
{
    Json root;
    root["version"] = 1;
    root["total_ops"] = summary.total_ops;
    root["category_counts"] = Json{
        { "upload", summary.upload_ops },
        { "copy", summary.copy_ops },
        { "compute", summary.compute_ops },
        { "download", summary.download_ops },
    };
    root["op_counts"] = op_counts_to_json(summary.op_counts);
    root["device_op_counts"] = device_counts_to_json(summary.device_op_counts);
    root["unassigned_device_ops"] = summary.unassigned_device_ops;
    root["invalid_device_ops"] = summary.invalid_device_ops;
    return root.dump(indent);
}

}  // namespace poseidon::mgpu
