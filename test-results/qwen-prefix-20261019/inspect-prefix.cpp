#include "runtime/plan_reader.hpp"
#include <nlohmann/json.hpp>
#include <iostream>
#include <stdexcept>
#include <unordered_map>

using namespace fhegpu;
using Json = nlohmann::json;

void count(Json &object, const std::string &key) {
    if (!object.contains(key)) object[key] = 0;
    object[key] = object[key].get<std::uint64_t>() + 1;
}
std::string place(const Place &p) {
    return p.kind == PlaceKind::Host ? "Host" : "GPU" + std::to_string(p.index);
}
int main(int argc, char **argv) {
    try {
        if (argc != 3) throw std::runtime_error("usage: inspect-prefix PLAN POLY_DEGREE");
        const auto degree = std::stoull(argv[2]);
        const auto loaded = RuntimePlanReader::read_file(argv[1]);
        Json result = Json::object();
        std::unordered_map<ValueId, const ValueDesc *> values;
        for (const auto &value : loaded.plan.values) values.emplace(value.id, &value);
        for (const auto &instruction : loaded.plan.execution) {
            if (const auto *op = std::get_if<ComputeOp>(&instruction.body)) {
                count(result["compute_by_kind"], to_string(op->kind));
                count(result["compute_by_place_and_kind"][place(op->place)], to_string(op->kind));
            } else if (const auto *op = std::get_if<CommAction>(&instruction.body)) {
                if (op->sources.size() != 1) throw std::runtime_error("expected one communication source");
                count(result["communication_records_by_kind"], to_string(op->kind));
                count(result["communication_records_by_hint"], to_string(op->hint));
                const auto &value = *values.at(op->inputs.at(0));
                if (value.level < 0 || value.components < 1) throw std::runtime_error("invalid transport shape");
                // Poseidon maps level to q_count-1 (crt_context.cpp), and the
                // GPU backend packs uint32 GpuWord (gpu_memory.h).
                const std::uint64_t bytes = degree * (value.level + 1) * value.components * 4;
                for (const auto &destination : op->destinations) {
                    const auto edge = place(op->sources.front()) + " -> " + place(destination);
                    count(result["communication_edges"], edge);
                    if (!result["logical_gpu_payload_bytes"].contains(edge)) result["logical_gpu_payload_bytes"][edge] = 0;
                    result["logical_gpu_payload_bytes"][edge] = result["logical_gpu_payload_bytes"][edge].get<std::uint64_t>() + bytes;
                }
            } else if (const auto *op = std::get_if<PreparePlaintextOp>(&instruction.body)) {
                const auto &value = *values.at(op->output);
                if (value.level < 0) throw std::runtime_error("invalid plaintext level");
                const auto device = place(value.place);
                const std::uint64_t bytes = degree * (value.level + 1) * 4;
                if (!result["all_prepared_value_bytes_by_device"].contains(device)) result["all_prepared_value_bytes_by_device"][device] = 0;
                result["all_prepared_value_bytes_by_device"][device] = result["all_prepared_value_bytes_by_device"][device].get<std::uint64_t>() + bytes;
            }
        }
        result["execution_records"] = loaded.plan.execution.size();
        result["payload_scope"] = "Logical GPU packed payload per edge: degree*(level+1)*components*4. Excludes extra HostStaged transactions, keys, initialization, validation and internal CPU Prepare uploads.";
        result["prepared_value_scope"] = "Sum of all Prepare output buffers if kept simultaneously, not a peak; no content deduplication. GPU plaintext degree*(level+1)*4.";
        std::cout << result.dump(2) << '\n';
    } catch (const std::exception &error) {
        std::cerr << "inspect-prefix: " << error.what() << '\n';
        return 1;
    }
}
