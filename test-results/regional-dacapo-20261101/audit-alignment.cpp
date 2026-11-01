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
    std::map<fhegpu::ValueId, fhegpu::ValueId> aliases;
    std::map<fhegpu::ValueId, const fhegpu::ComputeOp *> producers;
    auto original = [&](fhegpu::ValueId id) {
      while (aliases.count(id)) id = aliases.at(id);
      return id;
    };
    nlohmann::json boots = nlohmann::json::array();
    for (auto phase : {&plan.initialization, &plan.execution, &plan.finalization})
      for (const auto &i : *phase) {
        if (auto encode = std::get_if<fhegpu::EncodeOp>(&i.body)) {
          if (auto bundle = std::get_if<fhegpu::BundleEncodePayload>(&encode->payload)) contents.insert(bundle->content);
        }
        if (auto transfer = std::get_if<fhegpu::CommAction>(&i.body))
          for (unsigned j = 0; j < transfer->outputs.size(); ++j)
            aliases[transfer->outputs[j]] = original(transfer->inputs.at(j));
        if (auto op = std::get_if<fhegpu::ComputeOp>(&i.body)) {
          producers[op->output] = op;
          std::string name = names[int(op->kind)]; ++counts[name];
          if (name == "boot") {
            const auto &before = *values.at(op->inputs.at(0));
            const auto &after = *values.at(op->output);
            if (before.level != 5 || std::abs(before.scale_log2 - 40) > 1e-8 || after.level != 30 || after.scale_log2 != 40) return 3;
            auto input = original(op->inputs.at(0));
            auto producer = producers.find(input);
            nlohmann::json alignment = nullptr;
            if (producer != producers.end() && producer->second->kind == fhegpu::ComputeKind::ModSwitch) {
              const auto &pre = *values.at(producer->second->inputs.at(0));
              alignment = {{"before_q", pre.level + 1}, {"before_s", pre.scale_log2},
                           {"after_q", before.level + 1}, {"after_s", before.scale_log2},
                           {"discarded_q", pre.level - before.level}};
            }
            boots.push_back({{"immediate_modswitch", alignment},{"input_q", before.level + 1}, {"input_s", before.scale_log2}, {"output_q", after.level + 1}, {"output_s", after.scale_log2}});
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
