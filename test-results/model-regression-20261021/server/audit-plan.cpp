#include "api/mock_api.hpp"
#include "runtime/operator_spec_reader.hpp"
#include "runtime/plan_reader.hpp"
#include "runtime/runtime.hpp"
#include "runtime/verifier.hpp"
#include <nlohmann/json.hpp>
#include <cmath>
#include <atomic>
#include <fstream>
#include <iostream>
#include <unordered_map>

using namespace fhegpu;
using Json = nlohmann::json;

class RoundedOneApi : public MockVecApi {
public:
  RoundedOneApi(std::shared_ptr<MockCluster> cluster, bool rounded, size_t slots)
      : MockVecApi(0, std::move(cluster)), rounded_(rounded), slots_(slots) {}
  Value encode_plaintext(const ValueDesc &desc, const std::vector<double> &slots) {
    auto value = MockVecApi::encode_plaintext(desc, slots);
    // A uniform full-slot 1 has only the constant polynomial coefficient.
    // Isolate its integer rounding; this is a diagnostic, not a CKKS simulator.
    if (rounded_ && slots.size() == slots_ && desc.scale_log2 < 30 &&
        std::all_of(slots.begin(), slots.end(), [](double x) { return x == 1; })) {
      const auto delta = std::exp2(desc.scale_log2);
      const auto gain = std::round(delta)/delta;
      ++rounded_encodings;
      auto payload = value.materialize();
      for (auto &x : payload.slots) x *= gain;
      value = VecValue::ready(std::move(payload));
    }
    return value;
  }
  std::atomic<size_t> rounded_encodings{0};
  ResourceUsage value_resources(const ValueDesc &desc) {
    ResourceUsage result(1);
    const auto bytes = slots_*sizeof(double);
    if (desc.place.kind == PlaceKind::Host) result.host = bytes;
    else if (desc.kind == ValueKind::Plaintext) result.plain.at(0) = bytes;
    else result.front.at(0) = bytes;
    return result;
  }
  PrepareResources prepare_resources(const ValueDesc &desc, PrepareBackend backend) {
    auto resident = value_resources(desc), peak = resident;
    peak.host += slots_*sizeof(double);
    return {peak, resident, backend == PrepareBackend::Cpu};
  }
  ResourceUsage instruction_workspace(const Instruction &) { return ResourceUsage(1); }
  Value prepare_plaintext_async(const ValueDesc &desc, const std::vector<double> &slots,
                                PrepareBackend backend) {
    if (backend != PrepareBackend::Cpu) throw std::runtime_error("FP64 replay requires CPU preparation");
    return encode_plaintext(desc, slots);
  }
  bool is_ready(Value &value) { return value.is_ready(); }
private:
  bool rounded_;
  size_t slots_;
};

int main(int argc, char **argv) {
  try {
    if (argc != 7) throw std::runtime_error("PLAN SPEC BUNDLE FIXTURE REPORT MODE(audit|execute)");
    auto loaded = RuntimePlanReader::read_file(argv[1]);
    auto spec = OperatorSpecReader::read_file(argv[2]);
    auto &plan = loaded.plan;
    PlanVerifier::verify(plan, spec, false);
    std::unordered_map<ValueId, ValueDesc> values;
    for (const auto &value : plan.values) values.emplace(value.id, value);
    Json report{{"verifier_passed", true}, {"format_version", plan.format_version},
                {"boots", 0}, {"rescale_q", 0}, {"modswitch_q", 0},
                {"compute_operations", Json::object()}, {"outputs", Json::array()}};
    for (const auto &instruction : plan.execution) {
      auto op = std::get_if<ComputeOp>(&instruction.body);
      if (!op) continue;
      auto name = to_string(op->kind);
      report["compute_operations"][name] = report["compute_operations"].value(name, 0) + 1;
      if (op->kind == ComputeKind::Boot) report["boots"] = report["boots"].get<int>() + 1;
      if (op->kind == ComputeKind::Rescale || op->kind == ComputeKind::ModSwitch) {
        auto key = op->kind == ComputeKind::Rescale ? "rescale_q" : "modswitch_q";
        report[key] = report[key].get<int>() + values.at(op->inputs.at(0)).level-values.at(op->output).level;
      }
    }
    for (auto id : plan.final_outputs) report["outputs"].push_back({
      {"value_id", id}, {"level", values.at(id).level}, {"scale_log2", values.at(id).scale_log2}});
    if (std::string(argv[6]) == "execute" || std::string(argv[6]) == "rounded") {
      Json fixture;
      { std::ifstream stream(argv[4]); stream >> fixture; }
      std::unordered_map<ValueId, VecValue> inputs;
      const auto slots = spec.spec.poly_degree / 2;
      for (size_t i = 0; i < plan.external_inputs.size(); ++i) {
        const auto &desc = values.at(plan.external_inputs[i]);
        auto numbers = fixture.at("inputs").at(i).at("values").get<std::vector<double>>();
        numbers.resize(slots, 0.0);
        inputs.emplace(desc.id, make_cipher(std::move(numbers), desc.context, spec.spec.poly_degree,
                                           desc.level, desc.scale_log2, desc.ntt, desc.components));
      }
      auto cluster = std::make_shared<MockCluster>(MockClusterConfig{});
      RoundedOneApi api(cluster, std::string(argv[6]) == "rounded", slots);
      // Arithmetic/levels are unchanged; this replay encodes FP64 vectors on CPU.
      if (plan.waterline) plan.waterline->backend = PrepareBackend::Cpu;
      SequentialRuntime<RoundedOneApi> runtime(0, 1, plan.target.device_counts.at(0), api,
                                               DeviceExecutionMode::PerDeviceWorkers, 1);
      RuntimeResources resources{spec, std::filesystem::path(argv[3]), false};
      const auto artifact = runtime.run(loaded, resources, inputs);
      size_t failures = 0;
      double maximum = 0;
      for (size_t i = 0; i < plan.final_outputs.size(); ++i) {
        auto actual = artifact.values.at(plan.final_outputs[i]).value.materialize().slots;
        auto expected = fixture.at("expected_outputs").at(i).at("values").get<std::vector<double>>();
        const size_t checked = fixture.value("output_encoding", "zero-padded-full-slots") ==
                                   "actual-logits-first-10-slots" ? expected.size() : slots;
        expected.resize(slots, 0.0);
        double error = 0;
        for (size_t j = 0; j < checked; ++j) {
          const auto delta = std::abs(actual.at(j)-expected[j]);
          if (!std::isfinite(actual.at(j)) || delta > 0.001+0.0001*std::abs(expected[j])) ++failures;
          error = std::max(error, delta);
        }
        report["outputs"][i]["max_abs_error"] = error;
        report["outputs"][i]["checked_slots"] = checked;
        if (checked == 10) report["outputs"][i]["logits"] = std::vector<double>(actual.begin(), actual.begin()+10);
        maximum = std::max(maximum, error);
      }
      report["clear_runtime"] = {{"passed", failures == 0}, {"bad_slots", failures},
        {"max_abs_error", maximum}, {"scope", "FP64 MockVecApi; ideal identity Boot; no CKKS noise"}};
      report["rounded_one_encodings"] = api.rounded_encodings.load();
    } else if (std::string(argv[6]) != "audit") throw std::runtime_error("unknown mode");
    std::ofstream out(argv[5]); out << report.dump(2) << '\n';
    std::cout << report.dump(2) << '\n';
    return 0;
  } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
