#include "poseidon/mgpu/ir/schedule_json.h"

#include "poseidon/util/json.h"

#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace poseidon::mgpu
{
namespace
{

using Json = nlohmann::json;

void add_diagnostic(
    ScheduleJsonParseResult &result, std::string path, std::string message)
{
    result.diagnostics.push_back(
        ScheduleJsonDiagnostic{ std::move(path), std::move(message) });
}

bool get_object_field(
    ScheduleJsonParseResult &result, const Json &object, const char *field,
    const Json *&value, const std::string &path)
{
    const auto iter = object.find(field);
    if (iter == object.end())
    {
        add_diagnostic(result, path, std::string("missing field '") + field + "'");
        return false;
    }
    value = &(*iter);
    return true;
}

bool get_device(const Json &object, int &device)
{
    const auto device_iter = object.find("device");
    if (device_iter != object.end())
    {
        device = device_iter->get<int>();
        return true;
    }

    const auto device_id_iter = object.find("device_id");
    if (device_id_iter != object.end())
    {
        device = device_id_iter->get<int>();
        return true;
    }

    return false;
}

bool parse_value_refs(
    ScheduleJsonParseResult &result, const Json &array, std::vector<MgpuValueRef> &refs,
    const std::string &path)
{
    if (!array.is_array())
    {
        add_diagnostic(result, path, "expected an array");
        return false;
    }

    bool ok = true;
    refs.reserve(array.size());
    for (std::size_t i = 0; i < array.size(); ++i)
    {
        const Json &item = array[i];
        const std::string item_path = path + "/" + std::to_string(i);
        if (!item.is_number_integer() && !item.is_number_unsigned())
        {
            add_diagnostic(result, item_path, "expected an integer value id");
            ok = false;
            continue;
        }

        if (item.is_number_unsigned())
        {
            refs.push_back(MgpuValueRef{ item.get<std::uint64_t>() });
            continue;
        }

        const auto signed_id = item.get<long long>();
        if (signed_id < 0)
        {
            add_diagnostic(result, item_path, "value id must be non-negative");
            ok = false;
            continue;
        }

        refs.push_back(MgpuValueRef{ static_cast<ValueId>(signed_id) });
    }
    return ok;
}

bool parse_integer_attributes(
    ScheduleJsonParseResult &result, const Json &object,
    std::unordered_map<std::string, std::int64_t> &attributes, const std::string &path)
{
    if (!object.is_object())
    {
        add_diagnostic(result, path, "expected an object");
        return false;
    }

    bool ok = true;
    for (auto iter = object.begin(); iter != object.end(); ++iter)
    {
        const std::string item_path = path + "/" + iter.key();
        if (!iter.value().is_number_integer() && !iter.value().is_number_unsigned())
        {
            add_diagnostic(result, item_path, "expected an integer attribute");
            ok = false;
            continue;
        }

        if (iter.value().is_number_unsigned())
        {
            const std::uint64_t unsigned_value = iter.value().get<std::uint64_t>();
            if (unsigned_value > static_cast<std::uint64_t>(
                                     std::numeric_limits<std::int64_t>::max()))
            {
                add_diagnostic(result, item_path, "integer attribute exceeds int64_t");
                ok = false;
                continue;
            }
            attributes.emplace(iter.key(), static_cast<std::int64_t>(unsigned_value));
            continue;
        }

        attributes.emplace(iter.key(), iter.value().get<std::int64_t>());
    }
    return ok;
}

void parse_op(ScheduleJsonParseResult &result, const Json &op_json, std::size_t op_index)
{
    const std::string op_path = "/ops/" + std::to_string(op_index);
    if (!op_json.is_object())
    {
        add_diagnostic(result, op_path, "expected an object");
        return;
    }

    const Json *kind_json = nullptr;
    if (!get_object_field(result, op_json, "kind", kind_json, op_path))
    {
        return;
    }
    if (!kind_json->is_string())
    {
        add_diagnostic(result, op_path + "/kind", "expected a string");
        return;
    }

    const std::string kind_name = kind_json->get<std::string>();
    const std::optional<MgpuOpKind> kind = mgpu_op_kind_from_string(kind_name);
    if (!kind.has_value())
    {
        add_diagnostic(result, op_path + "/kind", "unknown op kind '" + kind_name + "'");
        return;
    }

    int device = 0;
    try
    {
        if (!get_device(op_json, device))
        {
            add_diagnostic(result, op_path, "missing field 'device'");
            return;
        }
    }
    catch (const std::exception &ex)
    {
        add_diagnostic(result, op_path + "/device", ex.what());
        return;
    }

    MgpuOp op;
    op.kind = *kind;
    op.device_id = device;

    const auto inputs_iter = op_json.find("inputs");
    if (inputs_iter != op_json.end())
    {
        parse_value_refs(result, *inputs_iter, op.inputs, op_path + "/inputs");
    }

    const auto outputs_iter = op_json.find("outputs");
    if (outputs_iter != op_json.end())
    {
        parse_value_refs(result, *outputs_iter, op.outputs, op_path + "/outputs");
    }

    const auto name_iter = op_json.find("name");
    if (name_iter != op_json.end())
    {
        if (name_iter->is_string())
        {
            op.debug_name = name_iter->get<std::string>();
        }
        else
        {
            add_diagnostic(result, op_path + "/name", "expected a string");
        }
    }

    const auto debug_name_iter = op_json.find("debug_name");
    if (debug_name_iter != op_json.end())
    {
        if (debug_name_iter->is_string())
        {
            op.debug_name = debug_name_iter->get<std::string>();
        }
        else
        {
            add_diagnostic(result, op_path + "/debug_name", "expected a string");
        }
    }

    const auto attrs_iter = op_json.find("attrs");
    if (attrs_iter != op_json.end())
    {
        parse_integer_attributes(result, *attrs_iter, op.integer_attributes, op_path + "/attrs");
    }

    result.schedule.ops.push_back(std::move(op));
}

Json value_refs_to_json(const std::vector<MgpuValueRef> &refs)
{
    Json result = Json::array();
    for (const MgpuValueRef &ref : refs)
    {
        result.push_back(ref.id);
    }
    return result;
}

}  // namespace

std::string ScheduleJsonParseResult::format_diagnostics() const
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

ScheduleJsonParseResult parse_schedule_json(std::string_view text)
{
    ScheduleJsonParseResult result;

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
            add_diagnostic(result, "/version", "unsupported schedule JSON version");
        }
    }

    const Json *ops_json = nullptr;
    if (!get_object_field(result, root, "ops", ops_json, "/"))
    {
        return result;
    }
    if (!ops_json->is_array())
    {
        add_diagnostic(result, "/ops", "expected an array");
        return result;
    }

    for (std::size_t i = 0; i < ops_json->size(); ++i)
    {
        parse_op(result, (*ops_json)[i], i);
    }

    return result;
}

std::string schedule_to_json(const MgpuSchedule &schedule, int indent)
{
    Json root;
    root["version"] = 1;
    root["ops"] = Json::array();

    for (const MgpuOp &op : schedule.ops)
    {
        Json op_json;
        op_json["kind"] = to_string(op.kind);
        op_json["device"] = op.device_id;
        op_json["inputs"] = value_refs_to_json(op.inputs);
        op_json["outputs"] = value_refs_to_json(op.outputs);
        if (!op.debug_name.empty())
        {
            op_json["name"] = op.debug_name;
        }
        if (!op.integer_attributes.empty())
        {
            Json attrs_json = Json::object();
            for (const auto &item : op.integer_attributes)
            {
                attrs_json[item.first] = item.second;
            }
            op_json["attrs"] = std::move(attrs_json);
        }
        root["ops"].push_back(std::move(op_json));
    }

    return root.dump(indent);
}

}  // namespace poseidon::mgpu
