#include "poseidon/mgpu/runtime/poseidon_gpu_schedule_preflight.h"

#include "poseidon/util/json.h"

#include <algorithm>
#include <ostream>
#include <sstream>
#include <utility>

namespace poseidon::mgpu
{
namespace
{

using Json = nlohmann::json;

void add_diagnostic(
    PoseidonGpuSchedulePreflightResult &result, std::size_t op_index,
    std::string message)
{
    result.diagnostics.push_back(
        PoseidonGpuSchedulePreflightDiagnostic{ op_index, std::move(message) });
}

void add_device(PoseidonGpuSchedulePreflightResult &result, int device_id)
{
    if (device_id < 0)
    {
        return;
    }
    if (std::find(result.devices.begin(), result.devices.end(), device_id) ==
        result.devices.end())
    {
        result.devices.push_back(device_id);
    }
}

Json diagnostics_to_json(
    const std::vector<PoseidonGpuSchedulePreflightDiagnostic> &diagnostics)
{
    Json result = Json::array();
    for (const PoseidonGpuSchedulePreflightDiagnostic &diagnostic : diagnostics)
    {
        result.push_back(Json{
            { "op_index", diagnostic.op_index },
            { "message", diagnostic.message },
        });
    }
    return result;
}

Json devices_to_json(const std::vector<int> &devices)
{
    Json result = Json::array();
    for (const int device : devices)
    {
        result.push_back(device);
    }
    return result;
}

}  // namespace

std::string PoseidonGpuSchedulePreflightResult::format_diagnostics() const
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

PoseidonGpuSchedulePreflightResult preflight_poseidon_gpu_schedule(
    const MgpuSchedule &schedule,
    const PoseidonGpuSchedulePreflightOptions &options)
{
    PoseidonGpuSchedulePreflightResult result;
    if (options.device_count <= 0)
    {
        add_diagnostic(result, 0, "device_count must be positive");
    }

    for (std::size_t op_index = 0; op_index < schedule.ops.size(); ++op_index)
    {
        const MgpuOp &op = schedule.ops[op_index];
        add_device(result, op.device_id);
        if (op.device_id < 0)
        {
            add_diagnostic(result, op_index, "operation has an unassigned device");
        }
        else if (options.device_count > 0 && op.device_id >= options.device_count)
        {
            std::ostringstream stream;
            stream << "operation device " << op.device_id
                   << " is outside device_count " << options.device_count;
            add_diagnostic(result, op_index, stream.str());
        }

        switch (op.kind)
        {
        case MgpuOpKind::UploadPlain:
        case MgpuOpKind::UploadCipher:
        case MgpuOpKind::Download:
        case MgpuOpKind::Add:
        case MgpuOpKind::AddPlain:
        case MgpuOpKind::Sub:
        case MgpuOpKind::Negate:
        case MgpuOpKind::MultiplyPlain:
        case MgpuOpKind::Multiply:
        case MgpuOpKind::Rescale:
            break;
        case MgpuOpKind::CopyPlain:
        case MgpuOpKind::CopyCipher:
            result.requires_comm = true;
            if (!options.copy_ops_have_comm)
            {
                add_diagnostic(
                    result, op_index,
                    "copy op requires executing through the mgpu communication layer");
            }
            break;
        case MgpuOpKind::Relinearize:
            result.requires_relin_keys = true;
            if (!options.relin_keys_available)
            {
                add_diagnostic(
                    result, op_index,
                    "relinearize requires RelinKeys to be uploaded for schedule devices");
            }
            break;
        case MgpuOpKind::Rotate:
            result.requires_galois_keys = true;
            if (!options.galois_keys_available)
            {
                add_diagnostic(
                    result, op_index,
                    "rotate requires GaloisKeys to be uploaded for schedule devices");
            }
            break;
        case MgpuOpKind::BootstrapFallback:
            add_diagnostic(
                result, op_index,
                "Poseidon GPU bootstrap fallback is not implemented");
            break;
        }
    }

    return result;
}

std::string dump_poseidon_gpu_schedule_preflight(
    const PoseidonGpuSchedulePreflightResult &result)
{
    std::ostringstream stream;
    dump_poseidon_gpu_schedule_preflight(stream, result);
    return stream.str();
}

void dump_poseidon_gpu_schedule_preflight(
    std::ostream &stream, const PoseidonGpuSchedulePreflightResult &result)
{
    stream << "poseidon_gpu_preflight:\n";
    stream << "  status: " << (result.ok() ? "ok" : "error") << '\n';
    stream << "  requires_comm: " << (result.requires_comm ? "true" : "false") << '\n';
    stream << "  requires_relin_keys: "
           << (result.requires_relin_keys ? "true" : "false") << '\n';
    stream << "  requires_galois_keys: "
           << (result.requires_galois_keys ? "true" : "false") << '\n';
    stream << "  devices: [";
    for (std::size_t i = 0; i < result.devices.size(); ++i)
    {
        if (i > 0)
        {
            stream << ", ";
        }
        stream << result.devices[i];
    }
    stream << "]\n";

    if (!result.diagnostics.empty())
    {
        stream << "  diagnostics:\n";
        for (const PoseidonGpuSchedulePreflightDiagnostic &diagnostic :
             result.diagnostics)
        {
            stream << "    op #" << diagnostic.op_index << ": "
                   << diagnostic.message << '\n';
        }
    }
}

std::string poseidon_gpu_schedule_preflight_to_json(
    const PoseidonGpuSchedulePreflightResult &result, int indent)
{
    Json root;
    root["version"] = 1;
    root["ok"] = result.ok();
    root["requires_comm"] = result.requires_comm;
    root["requires_relin_keys"] = result.requires_relin_keys;
    root["requires_galois_keys"] = result.requires_galois_keys;
    root["devices"] = devices_to_json(result.devices);
    root["diagnostics"] = diagnostics_to_json(result.diagnostics);
    return root.dump(indent);
}

}  // namespace poseidon::mgpu
