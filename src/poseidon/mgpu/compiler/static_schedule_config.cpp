#include "poseidon/mgpu/compiler/static_schedule_config.h"

#include "poseidon/util/json.h"

#include <cstdint>
#include <optional>
#include <sstream>
#include <utility>

namespace poseidon::mgpu
{
namespace
{

using Json = nlohmann::json;

void add_diagnostic(
    StaticScheduleExecutionConfigParseResult &result, std::string path,
    std::string message)
{
    result.diagnostics.push_back(
        StaticScheduleExecutionConfigDiagnostic{ std::move(path), std::move(message) });
}

bool read_int(
    StaticScheduleExecutionConfigParseResult &result, const Json &object,
    const char *field, int &value, const std::string &path, bool required = false)
{
    const auto iter = object.find(field);
    if (iter == object.end())
    {
        if (required)
        {
            add_diagnostic(result, path, std::string("missing field '") + field + "'");
        }
        return false;
    }

    const std::string item_path = path + "/" + field;
    if (!iter->is_number_integer() && !iter->is_number_unsigned())
    {
        add_diagnostic(result, item_path, "expected an integer");
        return false;
    }

    try
    {
        value = iter->get<int>();
    }
    catch (const std::exception &ex)
    {
        add_diagnostic(result, item_path, ex.what());
        return false;
    }
    return true;
}

bool read_bool(
    StaticScheduleExecutionConfigParseResult &result, const Json &object,
    const char *field, bool &value, const std::string &path)
{
    const auto iter = object.find(field);
    if (iter == object.end())
    {
        return false;
    }

    const std::string item_path = path + "/" + field;
    if (!iter->is_boolean())
    {
        add_diagnostic(result, item_path, "expected a boolean");
        return false;
    }

    value = iter->get<bool>();
    return true;
}

bool read_optional_int(
    StaticScheduleExecutionConfigParseResult &result, const Json &object,
    const char *field, std::optional<int> &value, const std::string &path)
{
    const auto iter = object.find(field);
    if (iter == object.end())
    {
        return false;
    }

    const std::string item_path = path + "/" + field;
    if (iter->is_null())
    {
        value.reset();
        return true;
    }

    if (!iter->is_number_integer() && !iter->is_number_unsigned())
    {
        add_diagnostic(result, item_path, "expected an integer or null");
        return false;
    }

    try
    {
        value = iter->get<int>();
    }
    catch (const std::exception &ex)
    {
        add_diagnostic(result, item_path, ex.what());
        return false;
    }
    return true;
}

bool read_int_array(
    StaticScheduleExecutionConfigParseResult &result, const Json &object,
    const char *field, std::vector<int> &values, const std::string &path)
{
    const auto iter = object.find(field);
    if (iter == object.end())
    {
        return false;
    }

    const std::string array_path = path + "/" + field;
    if (!iter->is_array())
    {
        add_diagnostic(result, array_path, "expected an array");
        return false;
    }

    values.clear();
    values.reserve(iter->size());
    bool ok = true;
    for (std::size_t i = 0; i < iter->size(); ++i)
    {
        const Json &item = (*iter)[i];
        const std::string item_path = array_path + "/" + std::to_string(i);
        if (!item.is_number_integer() && !item.is_number_unsigned())
        {
            add_diagnostic(result, item_path, "expected an integer");
            ok = false;
            continue;
        }
        try
        {
            values.push_back(item.get<int>());
        }
        catch (const std::exception &ex)
        {
            add_diagnostic(result, item_path, ex.what());
            ok = false;
        }
    }
    return ok;
}

bool is_valid_device(int device_id, int device_count)
{
    return device_id >= 0 && device_id < device_count;
}

void parse_placement_policy(
    StaticScheduleExecutionConfigParseResult &result, const Json &placement,
    StaticScheduleExecutionConfig &config)
{
    const auto iter = placement.find("policy");
    if (iter == placement.end())
    {
        return;
    }

    if (!iter->is_string())
    {
        add_diagnostic(result, "/placement/policy", "expected a string");
        return;
    }

    const std::string policy = iter->get<std::string>();
    if (policy == "single_device")
    {
        config.pipeline.placement.policy = StaticPlacementPolicy::SingleDevice;
        return;
    }
    if (policy == "round_robin_compute")
    {
        config.pipeline.placement.policy = StaticPlacementPolicy::RoundRobinCompute;
        return;
    }

    add_diagnostic(result, "/placement/policy", "unknown placement policy '" + policy + "'");
}

void parse_placement(
    StaticScheduleExecutionConfigParseResult &result, const Json &root,
    StaticScheduleExecutionConfig &config)
{
    const auto iter = root.find("placement");
    if (iter == root.end())
    {
        return;
    }
    if (!iter->is_object())
    {
        add_diagnostic(result, "/placement", "expected an object");
        return;
    }

    const Json &placement = *iter;
    parse_placement_policy(result, placement, config);
    read_int(
        result, placement, "default_device", config.pipeline.placement.default_device,
        "/placement");
    read_optional_int(
        result, placement, "upload_device", config.pipeline.placement.upload_device,
        "/placement");
    read_optional_int(
        result, placement, "download_device", config.pipeline.placement.download_device,
        "/placement");
    if (read_int_array(
            result, placement, "compute_devices",
            config.pipeline.placement.compute_devices, "/placement") &&
        !config.pipeline.placement.compute_devices.empty() &&
        placement.find("policy") == placement.end())
    {
        config.pipeline.placement.policy = StaticPlacementPolicy::RoundRobinCompute;
    }
}

void parse_topology(
    StaticScheduleExecutionConfigParseResult &result, const Json &root,
    StaticScheduleExecutionConfig &config)
{
    const auto iter = root.find("topology");
    if (iter == root.end())
    {
        return;
    }
    if (!iter->is_object())
    {
        add_diagnostic(result, "/topology", "expected an object");
        return;
    }

    const Json &topology = *iter;
    read_int(result, topology, "nodes", config.node_count, "/topology");
    read_int(
        result, topology, "devices_per_node", config.devices_per_node,
        "/topology");
}

void parse_preflight(
    StaticScheduleExecutionConfigParseResult &result, const Json &root,
    StaticScheduleExecutionConfig &config)
{
    const auto iter = root.find("preflight");
    if (iter == root.end())
    {
        return;
    }
    if (!iter->is_object())
    {
        add_diagnostic(result, "/preflight", "expected an object");
        return;
    }

    const Json &preflight = *iter;
    read_bool(result, preflight, "opcode_summary", config.opcode_summary, "/preflight");
    read_bool(
        result, preflight, "poseidon_gpu_preflight",
        config.poseidon_gpu_preflight, "/preflight");
    read_bool(
        result, preflight, "comm_available",
        config.preflight_comm_available, "/preflight");
    read_bool(
        result, preflight, "relin_keys",
        config.preflight_relin_keys_available, "/preflight");
    read_bool(
        result, preflight, "galois_keys",
        config.preflight_galois_keys_available, "/preflight");
    read_bool(
        result, preflight, "communication_plan",
        config.communication_plan, "/preflight");
    read_bool(
        result, preflight, "communication_execution",
        config.communication_execution_preflight, "/preflight");
    read_bool(result, preflight, "require_ready", config.require_ready, "/preflight");
}

void parse_execution_backends(
    StaticScheduleExecutionConfigParseResult &result, const Json &root,
    StaticScheduleExecutionConfig &config)
{
    const auto iter = root.find("execution_backends");
    if (iter == root.end())
    {
        return;
    }
    if (!iter->is_object())
    {
        add_diagnostic(result, "/execution_backends", "expected an object");
        return;
    }

    const Json &backends = *iter;
    read_bool(
        result, backends, "same_device",
        config.communication_execution.same_device_available,
        "/execution_backends");
    read_bool(
        result, backends, "cuda_peer",
        config.communication_execution.cuda_peer_available,
        "/execution_backends");
    read_bool(
        result, backends, "inter_node",
        config.communication_execution.inter_node_available,
        "/execution_backends");
}

void apply_require_ready_implied_checks(StaticScheduleExecutionConfig &config)
{
    if (!config.require_ready)
    {
        return;
    }

    config.opcode_summary = true;
    config.poseidon_gpu_preflight = true;
    config.communication_plan = true;
    config.communication_execution_preflight = true;
}

void validate_config(StaticScheduleExecutionConfigParseResult &result)
{
    StaticScheduleExecutionConfig &config = result.config;
    if (config.pipeline.device_count <= 0)
    {
        add_diagnostic(result, "/device_count", "device_count must be positive");
    }
    if (config.node_count <= 0)
    {
        add_diagnostic(result, "/topology/nodes", "nodes must be positive");
    }
    if (config.devices_per_node < 0)
    {
        add_diagnostic(
            result, "/topology/devices_per_node",
            "devices_per_node must be non-negative");
    }
    if (!is_valid_device(
            config.pipeline.placement.default_device, config.pipeline.device_count))
    {
        add_diagnostic(
            result, "/placement/default_device",
            "default_device must be in [0, device_count)");
    }
    if (config.pipeline.placement.upload_device.has_value() &&
        !is_valid_device(
            *config.pipeline.placement.upload_device, config.pipeline.device_count))
    {
        add_diagnostic(
            result, "/placement/upload_device",
            "upload_device must be in [0, device_count)");
    }
    if (config.pipeline.placement.download_device.has_value() &&
        !is_valid_device(
            *config.pipeline.placement.download_device, config.pipeline.device_count))
    {
        add_diagnostic(
            result, "/placement/download_device",
            "download_device must be in [0, device_count)");
    }

    for (std::size_t i = 0; i < config.pipeline.placement.compute_devices.size(); ++i)
    {
        const int device = config.pipeline.placement.compute_devices[i];
        if (!is_valid_device(device, config.pipeline.device_count))
        {
            add_diagnostic(
                result,
                "/placement/compute_devices/" + std::to_string(i),
                "compute device must be in [0, device_count)");
            continue;
        }
        for (std::size_t j = 0; j < i; ++j)
        {
            if (config.pipeline.placement.compute_devices[j] == device)
            {
                add_diagnostic(
                    result,
                    "/placement/compute_devices/" + std::to_string(i),
                    "duplicate compute device");
                break;
            }
        }
    }

    if (config.communication_plan || config.communication_execution_preflight)
    {
        const int devices_per_node =
            config.devices_per_node == 0 ? config.pipeline.device_count
                                         : config.devices_per_node;
        if (devices_per_node <= 0)
        {
            add_diagnostic(
                result, "/topology/devices_per_node",
                "topology devices_per_node must be positive");
        }
        else if (
            config.node_count * devices_per_node < config.pipeline.device_count)
        {
            add_diagnostic(
                result, "/topology",
                "topology has fewer logical devices than device_count");
        }
    }
}

Json int_vector_to_json(const std::vector<int> &values)
{
    Json result = Json::array();
    for (const int value : values)
    {
        result.push_back(value);
    }
    return result;
}

Json placement_to_json(const StaticPlacementOptions &placement)
{
    Json result;
    result["policy"] = to_string(placement.policy);
    result["default_device"] = placement.default_device;
    result["compute_devices"] = int_vector_to_json(placement.compute_devices);
    result["upload_device"] = placement.upload_device.has_value()
                                  ? Json(*placement.upload_device)
                                  : Json(nullptr);
    result["download_device"] = placement.download_device.has_value()
                                    ? Json(*placement.download_device)
                                    : Json(nullptr);
    return result;
}

}  // namespace

std::string StaticScheduleExecutionConfigParseResult::format_diagnostics() const
{
    std::ostringstream stream;
    for (std::size_t i = 0; i < diagnostics.size(); ++i)
    {
        if (i > 0)
        {
            stream << '\n';
        }
        stream << diagnostics[i].path << ": " << diagnostics[i].message;
    }
    return stream.str();
}

StaticScheduleExecutionConfigParseResult parse_static_schedule_execution_config_json(
    std::string_view text)
{
    StaticScheduleExecutionConfigParseResult result;

    Json root;
    try
    {
        root = Json::parse(text.begin(), text.end());
    }
    catch (const std::exception &ex)
    {
        add_diagnostic(result, "/", ex.what());
        return result;
    }

    if (!root.is_object())
    {
        add_diagnostic(result, "/", "expected an object");
        return result;
    }

    const auto version_iter = root.find("version");
    if (version_iter != root.end())
    {
        if (!version_iter->is_number_integer() && !version_iter->is_number_unsigned())
        {
            add_diagnostic(result, "/version", "expected an integer");
        }
        else if (
            (version_iter->is_number_unsigned() &&
             version_iter->get<std::uint64_t>() != 1) ||
            (!version_iter->is_number_unsigned() && version_iter->get<long long>() != 1))
        {
            add_diagnostic(result, "/version", "unsupported config JSON version");
        }
    }

    read_int(result, root, "device_count", result.config.pipeline.device_count, "/");
    read_bool(
        result, root, "emit_debug_dump", result.config.pipeline.emit_debug_dump, "/");
    parse_placement(result, root, result.config);
    parse_topology(result, root, result.config);
    parse_preflight(result, root, result.config);
    parse_execution_backends(result, root, result.config);
    apply_require_ready_implied_checks(result.config);
    validate_config(result);
    return result;
}

std::string static_schedule_execution_config_to_json(
    const StaticScheduleExecutionConfig &config, int indent)
{
    Json root;
    root["version"] = 1;
    root["device_count"] = config.pipeline.device_count;
    root["emit_debug_dump"] = config.pipeline.emit_debug_dump;
    root["placement"] = placement_to_json(config.pipeline.placement);
    root["topology"] = Json{
        { "nodes", config.node_count },
        { "devices_per_node", config.devices_per_node },
    };
    root["preflight"] = Json{
        { "opcode_summary", config.opcode_summary },
        { "poseidon_gpu_preflight", config.poseidon_gpu_preflight },
        { "comm_available", config.preflight_comm_available },
        { "relin_keys", config.preflight_relin_keys_available },
        { "galois_keys", config.preflight_galois_keys_available },
        { "communication_plan", config.communication_plan },
        { "communication_execution", config.communication_execution_preflight },
        { "require_ready", config.require_ready },
    };
    root["execution_backends"] = Json{
        { "same_device", config.communication_execution.same_device_available },
        { "cuda_peer", config.communication_execution.cuda_peer_available },
        { "inter_node", config.communication_execution.inter_node_available },
    };
    return root.dump(indent);
}

MgpuTopology make_static_schedule_execution_topology(
    const StaticScheduleExecutionConfig &config)
{
    if (config.node_count == 1 && config.devices_per_node == 0)
    {
        return make_single_node_topology(config.pipeline.device_count);
    }

    const int devices_per_node =
        config.devices_per_node == 0 ? config.pipeline.device_count
                                     : config.devices_per_node;
    return make_uniform_cluster_topology(config.node_count, devices_per_node);
}

}  // namespace poseidon::mgpu
