#include "poseidon/mgpu/compiler/scheduler/static_scheduler.h"

#include "poseidon/mgpu/compiler/copy_insertion.h"
#include "poseidon/mgpu/compiler/static_placement.h"

#include <algorithm>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace poseidon::mgpu
{
namespace
{

struct ValueInfo
{
    MgpuValueKind kind = MgpuValueKind::Ciphertext;
    MgpuOp source_upload;
    bool has_source_upload = false;
    int producer_node = -1;
};

struct ResidentValue
{
    int device_id = 0;
    ValueId physical_id = 0;
    double available_time = 0.0;
};

struct ComputeNode
{
    std::size_t op_index = 0;
    MgpuOp op;
    std::vector<int> successors;
    int indegree = 0;
    bool scheduled = false;
};

struct CandidateCopy
{
    ValueId logical_id = 0;
    ValueId source_physical_id = 0;
    int source_device = 0;
    int destination_device = 0;
    MgpuValueKind kind = MgpuValueKind::Ciphertext;
    double ready_time = 0.0;
    bool needed = false;
};

struct DeviceCandidate
{
    int device_id = 0;
    double input_ready_time = 0.0;
    double start_time = 0.0;
    double finish_time = 0.0;
    std::vector<CandidateCopy> copies;
    bool valid = false;
};

struct ParsedSchedule
{
    std::unordered_map<ValueId, ValueInfo> values;
    std::vector<ComputeNode> compute_nodes;
    std::vector<MgpuOp> cipher_uploads;
    std::vector<MgpuOp> plain_uploads;
    std::vector<MgpuOp> downloads;
    ValueId next_value_id = 1;
};

struct GreedyState
{
    MgpuSchedule output;
    StaticSchedulePreflight preflight;
    std::unordered_map<ValueId, std::vector<ResidentValue>> resident;
    std::unordered_map<ValueId, bool> source_plain_emitted;
    std::vector<double> gpu_available;
    ValueId next_value_id = 1;
};

void add_diagnostic(
    StaticSchedulingResult &result, std::size_t op_index, std::string message)
{
    result.diagnostics.push_back(
        StaticSchedulingDiagnostic{ op_index, std::move(message) });
}

bool is_valid_device(int device_id, int device_count)
{
    return device_id >= 0 && device_id < device_count;
}

MgpuValueRef value_ref(ValueId id)
{
    return MgpuValueRef{ id };
}

ValueId max_value_id(const MgpuSchedule &schedule)
{
    ValueId max_id = 0;
    for (const MgpuOp &op : schedule.ops)
    {
        for (const MgpuValueRef &input : op.inputs)
        {
            max_id = std::max(max_id, input.id);
        }
        for (const MgpuValueRef &output : op.outputs)
        {
            max_id = std::max(max_id, output.id);
        }
    }
    return max_id;
}

bool output_kind(MgpuOpKind kind, MgpuValueKind &output)
{
    switch (kind)
    {
    case MgpuOpKind::UploadPlain:
    case MgpuOpKind::CopyPlain:
        output = MgpuValueKind::Plaintext;
        return true;
    case MgpuOpKind::UploadCipher:
    case MgpuOpKind::CopyCipher:
    case MgpuOpKind::Add:
    case MgpuOpKind::AddPlain:
    case MgpuOpKind::Sub:
    case MgpuOpKind::MultiplyPlain:
    case MgpuOpKind::Multiply:
    case MgpuOpKind::Negate:
    case MgpuOpKind::Relinearize:
    case MgpuOpKind::Rescale:
    case MgpuOpKind::Rotate:
    case MgpuOpKind::BootstrapFallback:
        output = MgpuValueKind::Ciphertext;
        return true;
    case MgpuOpKind::Download:
        return false;
    }
    return false;
}

bool define_value(
    StaticSchedulingResult &result, ParsedSchedule &parsed, std::size_t op_index,
    ValueId id, MgpuValueKind kind, const MgpuOp *source_upload, int producer_node)
{
    if (id == 0)
    {
        add_diagnostic(result, op_index, "output value id 0 is reserved");
        return false;
    }

    ValueInfo info;
    info.kind = kind;
    info.producer_node = producer_node;
    if (source_upload != nullptr)
    {
        info.source_upload = *source_upload;
        info.has_source_upload = true;
    }

    const auto [_, inserted] = parsed.values.emplace(id, std::move(info));
    if (!inserted)
    {
        std::ostringstream stream;
        stream << "duplicate output value %" << id;
        add_diagnostic(result, op_index, stream.str());
        return false;
    }
    return true;
}

std::vector<int> compute_devices(const StaticSchedulerOptions &options)
{
    if (!options.compute_devices.empty())
    {
        return options.compute_devices;
    }

    std::vector<int> result;
    result.reserve(static_cast<std::size_t>(options.device_count));
    for (int device = 0; device < options.device_count; ++device)
    {
        result.push_back(device);
    }
    return result;
}

bool validate_scheduler_options(
    StaticSchedulingResult &result, const StaticSchedulerOptions &options)
{
    bool ok = true;
    if (options.device_count <= 0)
    {
        add_diagnostic(result, 0, "device_count must be positive");
        return false;
    }
    if (!is_valid_device(options.default_device, options.device_count))
    {
        add_diagnostic(result, 0, "default_device must be in [0, device_count)");
        ok = false;
    }
    if (options.upload_device.has_value() &&
        !is_valid_device(*options.upload_device, options.device_count))
    {
        add_diagnostic(result, 0, "upload_device must be in [0, device_count)");
        ok = false;
    }
    if (options.download_device.has_value() &&
        !is_valid_device(*options.download_device, options.device_count))
    {
        add_diagnostic(result, 0, "download_device must be in [0, device_count)");
        ok = false;
    }

    for (std::size_t i = 0; i < options.compute_devices.size(); ++i)
    {
        const int device = options.compute_devices[i];
        if (!is_valid_device(device, options.device_count))
        {
            std::ostringstream stream;
            stream << "invalid compute device " << device;
            add_diagnostic(result, 0, stream.str());
            ok = false;
        }
        for (std::size_t j = 0; j < i; ++j)
        {
            if (options.compute_devices[j] == device)
            {
                std::ostringstream stream;
                stream << "duplicate compute device " << device;
                add_diagnostic(result, 0, stream.str());
                ok = false;
                break;
            }
        }
    }
    return ok;
}

std::size_t latency_index(const MgpuOp &op)
{
    const auto level = op.integer_attributes.find("level");
    if (level != op.integer_attributes.end() && level->second >= 0)
    {
        return static_cast<std::size_t>(level->second);
    }
    const auto target_level = op.integer_attributes.find("target_level");
    if (target_level != op.integer_attributes.end() && target_level->second >= 0)
    {
        return static_cast<std::size_t>(target_level->second);
    }
    return 0;
}

double runtime_for(const MgpuOp &op, const LatencyTable &latency_table)
{
    return latency_table.latency_for(op.kind, latency_index(op));
}

bool parse_schedule_for_greedy(
    StaticSchedulingResult &result, const MgpuSchedule &schedule,
    ParsedSchedule &parsed)
{
    parsed.next_value_id = max_value_id(schedule) + 1;

    for (std::size_t op_index = 0; op_index < schedule.ops.size(); ++op_index)
    {
        const MgpuOp &op = schedule.ops[op_index];
        if (is_copy_op(op.kind))
        {
            add_diagnostic(
                result, op_index,
                "greedy_ready scheduler expects copy-free input schedules");
            continue;
        }

        if (op.kind == MgpuOpKind::BootstrapFallback)
        {
            add_diagnostic(
                result, op_index,
                "greedy_ready scheduler does not support bootstrap_fallback");
            continue;
        }

        if (op.kind == MgpuOpKind::UploadCipher || op.kind == MgpuOpKind::UploadPlain)
        {
            if (op.outputs.size() != 1)
            {
                add_diagnostic(result, op_index, "upload op must have exactly one output");
                continue;
            }

            const MgpuValueKind kind = op.kind == MgpuOpKind::UploadPlain
                                           ? MgpuValueKind::Plaintext
                                           : MgpuValueKind::Ciphertext;
            if (define_value(
                    result, parsed, op_index, op.outputs[0].id, kind, &op, -1))
            {
                if (kind == MgpuValueKind::Ciphertext)
                {
                    parsed.cipher_uploads.push_back(op);
                }
                else
                {
                    parsed.plain_uploads.push_back(op);
                }
            }
            continue;
        }

        if (op.kind == MgpuOpKind::Download)
        {
            parsed.downloads.push_back(op);
            continue;
        }

        MgpuValueKind kind = MgpuValueKind::Ciphertext;
        if (!output_kind(op.kind, kind) || op.outputs.size() != 1)
        {
            add_diagnostic(result, op_index, "compute op must have exactly one output");
            continue;
        }

        const int node_index = static_cast<int>(parsed.compute_nodes.size());
        ComputeNode node;
        node.op_index = op_index;
        node.op = op;
        parsed.compute_nodes.push_back(std::move(node));
        define_value(result, parsed, op_index, op.outputs[0].id, kind, nullptr, node_index);
    }

    if (!result.ok())
    {
        return false;
    }

    for (std::size_t node_index = 0; node_index < parsed.compute_nodes.size(); ++node_index)
    {
        ComputeNode &node = parsed.compute_nodes[node_index];
        for (const MgpuValueRef &input : node.op.inputs)
        {
            const auto value_iter = parsed.values.find(input.id);
            if (value_iter == parsed.values.end())
            {
                std::ostringstream stream;
                stream << "unknown input value %" << input.id;
                add_diagnostic(result, node.op_index, stream.str());
                continue;
            }
            if (value_iter->second.producer_node >= 0)
            {
                ++node.indegree;
                parsed.compute_nodes[
                    static_cast<std::size_t>(value_iter->second.producer_node)]
                    .successors.push_back(static_cast<int>(node_index));
            }
        }
    }

    for (const MgpuOp &download : parsed.downloads)
    {
        if (download.inputs.size() != 1)
        {
            add_diagnostic(result, 0, "download op must have exactly one input");
            continue;
        }
        if (parsed.values.find(download.inputs[0].id) == parsed.values.end())
        {
            std::ostringstream stream;
            stream << "unknown download input value %" << download.inputs[0].id;
            add_diagnostic(result, 0, stream.str());
        }
    }

    return result.ok();
}

ResidentValue *find_resident(
    std::vector<ResidentValue> &residents, int device_id)
{
    for (ResidentValue &resident : residents)
    {
        if (resident.device_id == device_id)
        {
            return &resident;
        }
    }
    return nullptr;
}

const ResidentValue *find_resident(
    const std::vector<ResidentValue> &residents, int device_id)
{
    for (const ResidentValue &resident : residents)
    {
        if (resident.device_id == device_id)
        {
            return &resident;
        }
    }
    return nullptr;
}

void add_or_update_resident(
    GreedyState &state, ValueId logical_id, int device_id, ValueId physical_id,
    double available_time)
{
    std::vector<ResidentValue> &residents = state.resident[logical_id];
    ResidentValue *existing = find_resident(residents, device_id);
    if (existing == nullptr)
    {
        residents.push_back(ResidentValue{ device_id, physical_id, available_time });
        return;
    }
    if (available_time < existing->available_time)
    {
        existing->physical_id = physical_id;
        existing->available_time = available_time;
    }
}

void emit_plain_upload(
    GreedyState &state, const ValueInfo &info, ValueId logical_id, int device_id,
    double available_time)
{
    ValueId physical_id = logical_id;
    if (state.source_plain_emitted[logical_id])
    {
        physical_id = state.next_value_id++;
    }
    state.source_plain_emitted[logical_id] = true;

    MgpuOp upload = info.source_upload;
    upload.device_id = device_id;
    upload.outputs = { value_ref(physical_id) };
    upload.debug_name = upload.debug_name.empty() ? "plain_preload" : upload.debug_name;
    state.output.ops.push_back(std::move(upload));
    add_or_update_resident(state, logical_id, device_id, physical_id, available_time);
    ++state.preflight.plaintext_preload_count;
}

bool ensure_plain_resident(
    StaticSchedulingResult &result, GreedyState &state, const ParsedSchedule &parsed,
    ValueId logical_id, int device_id)
{
    const auto value_iter = parsed.values.find(logical_id);
    if (value_iter == parsed.values.end() || !value_iter->second.has_source_upload)
    {
        std::ostringstream stream;
        stream << "plaintext value %" << logical_id << " has no upload source";
        add_diagnostic(result, 0, stream.str());
        return false;
    }

    const auto resident_iter = state.resident.find(logical_id);
    if (resident_iter != state.resident.end() &&
        find_resident(resident_iter->second, device_id) != nullptr)
    {
        return true;
    }

    emit_plain_upload(state, value_iter->second, logical_id, device_id, 0.0);
    return true;
}

DeviceCandidate evaluate_on_device(
    StaticSchedulingResult &result, GreedyState &state, const ParsedSchedule &parsed,
    const ComputeNode &node, int device_id, const StaticSchedulerOptions &options)
{
    DeviceCandidate candidate;
    candidate.device_id = device_id;
    candidate.valid = true;

    for (const MgpuValueRef &input : node.op.inputs)
    {
        const auto value_iter = parsed.values.find(input.id);
        if (value_iter == parsed.values.end())
        {
            candidate.valid = false;
            continue;
        }

        if (value_iter->second.kind == MgpuValueKind::Plaintext)
        {
            candidate.input_ready_time = std::max(candidate.input_ready_time, 0.0);
            continue;
        }

        const auto resident_iter = state.resident.find(input.id);
        if (resident_iter == state.resident.end() || resident_iter->second.empty())
        {
            std::ostringstream stream;
            stream << "ciphertext value %" << input.id << " is not resident anywhere";
            add_diagnostic(result, node.op_index, stream.str());
            candidate.valid = false;
            continue;
        }

        const ResidentValue *best = nullptr;
        double best_ready = std::numeric_limits<double>::infinity();
        for (const ResidentValue &resident : resident_iter->second)
        {
            const double ready =
                resident.device_id == device_id
                    ? resident.available_time
                    : resident.available_time + options.greedy_ready.default_copy_latency;
            if (ready < best_ready)
            {
                best = &resident;
                best_ready = ready;
            }
        }
        candidate.input_ready_time = std::max(candidate.input_ready_time, best_ready);
        if (best != nullptr && best->device_id != device_id)
        {
            candidate.copies.push_back(CandidateCopy{
                input.id,
                best->physical_id,
                best->device_id,
                device_id,
                MgpuValueKind::Ciphertext,
                best_ready,
                true,
            });
        }
    }

    const double runtime = runtime_for(node.op, options.latency_table);
    candidate.start_time = std::max(
        state.gpu_available[static_cast<std::size_t>(device_id)],
        candidate.input_ready_time);
    candidate.finish_time = candidate.start_time + runtime;
    return candidate;
}

DeviceCandidate choose_device_for_node(
    StaticSchedulingResult &result, GreedyState &state, const ParsedSchedule &parsed,
    const ComputeNode &node, const StaticSchedulerOptions &options,
    const std::vector<int> &allowed_devices)
{
    std::vector<int> candidates = allowed_devices;
    if (options.preserve_existing_devices &&
        node.op.device_id != kUnassignedDevice)
    {
        candidates = { node.op.device_id };
    }

    DeviceCandidate best;
    for (const int device : candidates)
    {
        DeviceCandidate candidate =
            evaluate_on_device(result, state, parsed, node, device, options);
        if (!candidate.valid)
        {
            continue;
        }
        if (!best.valid || candidate.finish_time < best.finish_time ||
            (candidate.finish_time == best.finish_time &&
             candidate.device_id < best.device_id))
        {
            best = std::move(candidate);
        }
    }
    return best;
}

bool commit_node(
    StaticSchedulingResult &result, GreedyState &state, const ParsedSchedule &parsed,
    const ComputeNode &node, const DeviceCandidate &candidate)
{
    MgpuOp op = node.op;
    op.device_id = candidate.device_id;

    for (const CandidateCopy &copy : candidate.copies)
    {
        const ValueId copied_id = state.next_value_id++;
        MgpuOp copy_op;
        copy_op.kind = MgpuOpKind::CopyCipher;
        copy_op.device_id = copy.destination_device;
        copy_op.inputs = { value_ref(copy.source_physical_id) };
        copy_op.outputs = { value_ref(copied_id) };
        copy_op.debug_name = "greedy_ready_copy";
        state.output.ops.push_back(copy_op);
        add_or_update_resident(
            state, copy.logical_id, copy.destination_device, copied_id,
            copy.ready_time);
        ++state.preflight.ciphertext_copy_count;
    }

    for (MgpuValueRef &input : op.inputs)
    {
        const auto value_iter = parsed.values.find(input.id);
        if (value_iter == parsed.values.end())
        {
            continue;
        }
        if (value_iter->second.kind == MgpuValueKind::Plaintext)
        {
            if (!ensure_plain_resident(result, state, parsed, input.id, op.device_id))
            {
                return false;
            }
        }

        const auto resident_iter = state.resident.find(input.id);
        if (resident_iter == state.resident.end())
        {
            std::ostringstream stream;
            stream << "input value %" << input.id << " is not resident on device "
                   << op.device_id;
            add_diagnostic(result, node.op_index, stream.str());
            return false;
        }
        const ResidentValue *resident = find_resident(resident_iter->second, op.device_id);
        if (resident == nullptr)
        {
            std::ostringstream stream;
            stream << "input value %" << input.id << " is not resident on device "
                   << op.device_id;
            add_diagnostic(result, node.op_index, stream.str());
            return false;
        }
        input.id = resident->physical_id;
    }

    state.output.ops.push_back(op);
    state.gpu_available[static_cast<std::size_t>(op.device_id)] = candidate.finish_time;
    add_or_update_resident(
        state, node.op.outputs[0].id, op.device_id, op.outputs[0].id,
        candidate.finish_time);
    return true;
}

double compute_runtime_sum(
    const std::vector<ComputeNode> &nodes, const LatencyTable &latency_table)
{
    double total = 0.0;
    for (const ComputeNode &node : nodes)
    {
        total += runtime_for(node.op, latency_table);
    }
    return total;
}

double compute_critical_path(
    const std::vector<ComputeNode> &nodes, const LatencyTable &latency_table)
{
    std::vector<double> longest(nodes.size(), 0.0);
    double result = 0.0;
    for (std::size_t i = 0; i < nodes.size(); ++i)
    {
        const double finish = longest[i] + runtime_for(nodes[i].op, latency_table);
        result = std::max(result, finish);
        for (const int successor : nodes[i].successors)
        {
            longest[static_cast<std::size_t>(successor)] =
                std::max(longest[static_cast<std::size_t>(successor)], finish);
        }
    }
    return result;
}

bool materialize_download_input(
    StaticSchedulingResult &result, GreedyState &state, const ParsedSchedule &parsed,
    MgpuOp &download, int device_id, const StaticSchedulerOptions &options)
{
    ValueId logical_id = download.inputs[0].id;
    const auto value_iter = parsed.values.find(logical_id);
    if (value_iter == parsed.values.end())
    {
        std::ostringstream stream;
        stream << "unknown download input value %" << logical_id;
        add_diagnostic(result, 0, stream.str());
        return false;
    }

    if (value_iter->second.kind == MgpuValueKind::Plaintext)
    {
        if (!ensure_plain_resident(result, state, parsed, logical_id, device_id))
        {
            return false;
        }
    }

    std::vector<ResidentValue> &residents = state.resident[logical_id];
    const ResidentValue *resident = find_resident(residents, device_id);
    if (resident == nullptr)
    {
        const ResidentValue *best = nullptr;
        for (const ResidentValue &candidate : residents)
        {
            if (best == nullptr || candidate.available_time < best->available_time)
            {
                best = &candidate;
            }
        }
        if (best == nullptr)
        {
            add_diagnostic(result, 0, "download input is not resident anywhere");
            return false;
        }

        const ValueId copied_id = state.next_value_id++;
        MgpuOp copy_op;
        copy_op.kind = value_iter->second.kind == MgpuValueKind::Plaintext
                           ? MgpuOpKind::CopyPlain
                           : MgpuOpKind::CopyCipher;
        copy_op.device_id = device_id;
        copy_op.inputs = { value_ref(best->physical_id) };
        copy_op.outputs = { value_ref(copied_id) };
        copy_op.debug_name = "greedy_ready_download_copy";
        state.output.ops.push_back(copy_op);
        add_or_update_resident(
            state, logical_id, device_id, copied_id,
            best->available_time + options.greedy_ready.default_copy_latency);
        resident = find_resident(state.resident[logical_id], device_id);
        if (copy_op.kind == MgpuOpKind::CopyCipher)
        {
            ++state.preflight.ciphertext_copy_count;
        }
    }

    download.inputs[0].id = resident->physical_id;
    return true;
}

StaticSchedulingResult run_single_device(
    const MgpuSchedule &schedule, const StaticSchedulerOptions &options)
{
    StaticSchedulingResult result;
    if (!validate_scheduler_options(result, options))
    {
        return result;
    }

    StaticPlacementOptions placement;
    placement.device_count = options.device_count;
    placement.default_device = options.default_device;
    placement.policy = StaticPlacementPolicy::SingleDevice;
    placement.preserve_existing_devices = options.preserve_existing_devices;
    placement.upload_device = options.upload_device;
    placement.download_device = options.download_device;

    const StaticPlacementResult placement_result =
        place_static_schedule(schedule, placement);
    for (const StaticPlacementDiagnostic &diagnostic : placement_result.diagnostics)
    {
        add_diagnostic(result, diagnostic.op_index, diagnostic.message);
    }
    if (!placement_result.ok())
    {
        return result;
    }

    const CopyInsertionResult copy_result =
        insert_required_copies(placement_result.schedule, options.copy_insertion);
    result.schedule = copy_result.schedule;
    for (const CopyInsertionDiagnostic &diagnostic : copy_result.diagnostics)
    {
        add_diagnostic(result, diagnostic.op_index, diagnostic.message);
    }

    result.preflight.compute_time_per_gpu.assign(
        static_cast<std::size_t>(options.device_count), 0.0);
    for (const MgpuOp &op : result.schedule.ops)
    {
        if (is_compute_op(op.kind))
        {
            const double runtime = runtime_for(op, options.latency_table);
            result.preflight.compute_time_per_gpu[static_cast<std::size_t>(op.device_id)] +=
                runtime;
            result.preflight.estimated_makespan += runtime;
        }
        else if (op.kind == MgpuOpKind::CopyCipher)
        {
            ++result.preflight.ciphertext_copy_count;
        }
        else if (op.kind == MgpuOpKind::UploadPlain)
        {
            ++result.preflight.plaintext_preload_count;
        }
    }
    result.preflight.single_device_baseline = result.preflight.estimated_makespan;
    result.preflight.estimated_speedup = 1.0;
    result.preflight.critical_path_lower_bound = result.preflight.estimated_makespan;
    result.preflight.work_lower_bound = result.preflight.estimated_makespan;
    return result;
}

StaticSchedulingResult run_greedy_ready(
    const MgpuSchedule &schedule, const StaticSchedulerOptions &options)
{
    StaticSchedulingResult result;
    if (!validate_scheduler_options(result, options))
    {
        return result;
    }

    ParsedSchedule parsed;
    if (!parse_schedule_for_greedy(result, schedule, parsed))
    {
        return result;
    }

    GreedyState state;
    state.next_value_id = std::max(options.copy_insertion.next_value_id, parsed.next_value_id);
    state.gpu_available.assign(static_cast<std::size_t>(options.device_count), 0.0);
    state.preflight.compute_time_per_gpu.assign(
        static_cast<std::size_t>(options.device_count), 0.0);

    for (MgpuOp upload : parsed.cipher_uploads)
    {
        if (!options.preserve_existing_devices || upload.device_id == kUnassignedDevice)
        {
            upload.device_id = options.upload_device.value_or(options.default_device);
        }
        state.output.ops.push_back(upload);
        add_or_update_resident(
            state, upload.outputs[0].id, upload.device_id, upload.outputs[0].id, 0.0);
    }

    const std::vector<int> allowed_devices = compute_devices(options);
    const double total_work = compute_runtime_sum(parsed.compute_nodes, options.latency_table);
    state.preflight.single_device_baseline = total_work;
    state.preflight.work_lower_bound =
        allowed_devices.empty() ? total_work : total_work / allowed_devices.size();
    state.preflight.critical_path_lower_bound =
        compute_critical_path(parsed.compute_nodes, options.latency_table);

    std::vector<int> ready;
    for (std::size_t i = 0; i < parsed.compute_nodes.size(); ++i)
    {
        if (parsed.compute_nodes[i].indegree == 0)
        {
            ready.push_back(static_cast<int>(i));
        }
    }

    std::size_t scheduled_count = 0;
    while (!ready.empty())
    {
        int best_node = -1;
        DeviceCandidate best_candidate;
        std::size_t best_ready_index = 0;

        for (std::size_t i = 0; i < ready.size(); ++i)
        {
            const int node_index = ready[i];
            const ComputeNode &node = parsed.compute_nodes[static_cast<std::size_t>(node_index)];
            DeviceCandidate candidate = choose_device_for_node(
                result, state, parsed, node, options, allowed_devices);
            if (!candidate.valid)
            {
                continue;
            }
            if (!best_candidate.valid ||
                candidate.finish_time < best_candidate.finish_time ||
                (candidate.finish_time == best_candidate.finish_time &&
                 node.op_index < parsed.compute_nodes[static_cast<std::size_t>(best_node)].op_index))
            {
                best_node = node_index;
                best_candidate = std::move(candidate);
                best_ready_index = i;
            }
        }

        if (best_node < 0)
        {
            add_diagnostic(result, 0, "no valid ready op placement found");
            return result;
        }

        ComputeNode &node = parsed.compute_nodes[static_cast<std::size_t>(best_node)];
        if (!commit_node(result, state, parsed, node, best_candidate))
        {
            return result;
        }
        state.preflight.compute_time_per_gpu[static_cast<std::size_t>(best_candidate.device_id)] +=
            runtime_for(node.op, options.latency_table);
        node.scheduled = true;
        ++scheduled_count;
        ready.erase(ready.begin() + static_cast<std::ptrdiff_t>(best_ready_index));

        for (const int successor : node.successors)
        {
            ComputeNode &successor_node =
                parsed.compute_nodes[static_cast<std::size_t>(successor)];
            --successor_node.indegree;
            if (successor_node.indegree == 0)
            {
                ready.push_back(successor);
            }
        }
    }

    if (scheduled_count != parsed.compute_nodes.size())
    {
        add_diagnostic(result, 0, "schedule graph contains unscheduled compute ops");
        return result;
    }

    for (MgpuOp download : parsed.downloads)
    {
        if (!download.inputs.empty())
        {
            int device_id = options.download_device.value_or(kUnassignedDevice);
            if (options.preserve_existing_devices &&
                download.device_id != kUnassignedDevice)
            {
                device_id = download.device_id;
            }
            if (device_id == kUnassignedDevice)
            {
                const auto resident_iter = state.resident.find(download.inputs[0].id);
                if (resident_iter != state.resident.end() && !resident_iter->second.empty())
                {
                    device_id = resident_iter->second.front().device_id;
                }
                else
                {
                    device_id = options.default_device;
                }
            }
            download.device_id = device_id;
            if (!materialize_download_input(result, state, parsed, download, device_id, options))
            {
                return result;
            }
        }
        state.output.ops.push_back(std::move(download));
    }

    for (const MgpuOp &plain_upload : parsed.plain_uploads)
    {
        if (!state.source_plain_emitted[plain_upload.outputs[0].id])
        {
            const auto value_iter = parsed.values.find(plain_upload.outputs[0].id);
            if (value_iter != parsed.values.end())
            {
                emit_plain_upload(
                    state, value_iter->second, plain_upload.outputs[0].id,
                    options.upload_device.value_or(options.default_device), 0.0);
            }
        }
    }

    state.preflight.estimated_makespan = 0.0;
    for (const double available : state.gpu_available)
    {
        state.preflight.estimated_makespan =
            std::max(state.preflight.estimated_makespan, available);
    }
    if (state.preflight.estimated_makespan > 0.0)
    {
        state.preflight.estimated_speedup =
            state.preflight.single_device_baseline / state.preflight.estimated_makespan;
    }

    std::size_t used_compute_devices = 0;
    for (const double compute_time : state.preflight.compute_time_per_gpu)
    {
        if (compute_time > 0.0)
        {
            ++used_compute_devices;
        }
    }
    state.preflight.parallelism_found =
        used_compute_devices > 1 && state.preflight.estimated_speedup > 1.0;

    result.schedule = std::move(state.output);
    result.preflight = std::move(state.preflight);
    return result;
}

StaticSchedulingResult run_not_implemented(
    StaticSchedulerKind kind, const MgpuSchedule &schedule)
{
    StaticSchedulingResult result;
    result.schedule = schedule;
    add_diagnostic(
        result, 0,
        std::string("scheduler '") + to_string(kind) + "' is not implemented yet");
    return result;
}

}  // namespace

const char *to_string(StaticSchedulerKind kind) noexcept
{
    switch (kind)
    {
    case StaticSchedulerKind::SingleDevice:
        return "single_device";
    case StaticSchedulerKind::GreedyReady:
        return "greedy_ready";
    case StaticSchedulerKind::ValueAwareHeft:
        return "value_aware_heft";
    case StaticSchedulerKind::ValueAwarePeft:
        return "value_aware_peft";
    }
    return "unknown";
}

std::optional<StaticSchedulerKind> static_scheduler_kind_from_string(
    std::string_view name) noexcept
{
    if (name == "single_device")
    {
        return StaticSchedulerKind::SingleDevice;
    }
    if (name == "greedy_ready")
    {
        return StaticSchedulerKind::GreedyReady;
    }
    if (name == "value_aware_heft")
    {
        return StaticSchedulerKind::ValueAwareHeft;
    }
    if (name == "value_aware_peft")
    {
        return StaticSchedulerKind::ValueAwarePeft;
    }
    return std::nullopt;
}

std::string StaticSchedulingResult::format_diagnostics() const
{
    std::ostringstream stream;
    for (std::size_t i = 0; i < diagnostics.size(); ++i)
    {
        if (i > 0)
        {
            stream << '\n';
        }
        stream << "op #" << diagnostics[i].op_index << ": "
               << diagnostics[i].message;
    }
    return stream.str();
}

StaticSchedulingResult schedule_static(
    const MgpuSchedule &schedule, const StaticSchedulerOptions &options)
{
    switch (options.kind)
    {
    case StaticSchedulerKind::SingleDevice:
        return run_single_device(schedule, options);
    case StaticSchedulerKind::GreedyReady:
        return run_greedy_ready(schedule, options);
    case StaticSchedulerKind::ValueAwareHeft:
    case StaticSchedulerKind::ValueAwarePeft:
        return run_not_implemented(options.kind, schedule);
    }
    return run_not_implemented(options.kind, schedule);
}

}  // namespace poseidon::mgpu
