#pragma once

#include "poseidon/mgpu/ir/schedule.h"

#include <array>
#include <cstddef>
#include <iosfwd>
#include <string>
#include <vector>

namespace poseidon::mgpu
{

using MgpuOpKindOrder = std::array<MgpuOpKind, 15>;

struct MgpuOpKindCount
{
    MgpuOpKind kind = MgpuOpKind::Download;
    std::size_t count = 0;
};

struct MgpuDeviceOpCount
{
    int device_id = 0;
    std::size_t count = 0;
};

struct MgpuScheduleSummary
{
    std::size_t total_ops = 0;
    std::size_t upload_ops = 0;
    std::size_t copy_ops = 0;
    std::size_t compute_ops = 0;
    std::size_t download_ops = 0;
    std::size_t unassigned_device_ops = 0;
    std::size_t invalid_device_ops = 0;
    std::vector<MgpuOpKindCount> op_counts;
    std::vector<MgpuDeviceOpCount> device_op_counts;
};

const MgpuOpKindOrder &ordered_mgpu_op_kinds() noexcept;

MgpuScheduleSummary summarize_schedule(const MgpuSchedule &schedule, int device_count);

std::string dump_schedule_summary(const MgpuScheduleSummary &summary);
void dump_schedule_summary(std::ostream &stream, const MgpuScheduleSummary &summary);

std::string schedule_summary_to_json(const MgpuScheduleSummary &summary, int indent = 2);

}  // namespace poseidon::mgpu
