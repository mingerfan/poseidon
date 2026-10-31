#include "runtime/operator_spec_reader.hpp"
#include "runtime/plan_reader.hpp"
#include "runtime/verifier.hpp"
#include "runtime/utils/sha256.hpp"
#include <nlohmann/json.hpp>
#include <cmath>
#include <iostream>
#include <map>
#include <set>

int main(int argc, char **argv) {
  if (argc < 3) return 2;
  auto spec = fhegpu::OperatorSpecReader::read_file(argv[1]);
  std::set<std::string> reference;
  for (int arg = 2; arg < argc; ++arg) {
    auto loaded = fhegpu::RuntimePlanReader::read_file(argv[arg]);
    const auto &plan = loaded.plan;
    auto requirements = fhegpu::PlanVerifier::verify(plan, spec);
    std::map<fhegpu::ValueId, const fhegpu::ValueDesc *> values;
    for (const auto &v : plan.values) values[v.id] = &v;
    std::set<std::string> contents;
    std::map<std::string, unsigned> counts;
    const char *names[] = {"add_cc", "add_cp", "sub_cc", "sub_cp", "mul_cc", "mul_cp", "negate", "rotate", "rescale", "mod_switch", "relinearize", "boot"};
    nlohmann::json boots = nlohmann::json::array();
    for (auto phase : {&plan.initialization, &plan.execution, &plan.finalization})
      for (const auto &i : *phase) {
        if (auto encode = std::get_if<fhegpu::EncodeOp>(&i.body)) {
          if (auto bundle = std::get_if<fhegpu::BundleEncodePayload>(&encode->payload)) contents.insert(bundle->content);
        }
        if (auto op = std::get_if<fhegpu::ComputeOp>(&i.body)) {
          std::string name = names[int(op->kind)]; ++counts[name];
          if (name == "boot") {
            const auto &before = *values.at(op->inputs.at(0));
            const auto &after = *values.at(op->output);
            if (before.level != 5 || std::abs(before.scale_log2 - 40) > 1e-8 || after.level != 30 || after.scale_log2 != 40) return 3;
            boots.push_back({{"input_q", before.level + 1}, {"input_s", before.scale_log2}, {"output_q", after.level + 1}, {"output_s", after.scale_log2}});
          }
        }
      }
    std::string joined;
    for (const auto &content : contents) joined += content + "\n";
    const auto &out = *values.at(plan.final_outputs.at(0));
    nlohmann::json report = {{"plan", argv[arg]}, {"verifier_passed", true}, {"counts", counts},
      {"values", plan.values.size()}, {"boots", boots}, {"output_q", out.level + 1},
      {"output_s", out.scale_log2}, {"content_count", contents.size()},
      {"content_set_sha256", fhegpu::sha256_hex(joined)}};
    if (arg == 2) reference = contents;
    else report["same_payload_contents_as_first_plan"] = contents == reference;
    nlohmann::json inputs = nlohmann::json::array();
    for (auto id : plan.external_inputs) inputs.push_back({{"q", values.at(id)->level + 1}, {"s", values.at(id)->scale_log2}});
    report["inputs"] = inputs;
    std::cout << report.dump() << '\n';
  }
}
