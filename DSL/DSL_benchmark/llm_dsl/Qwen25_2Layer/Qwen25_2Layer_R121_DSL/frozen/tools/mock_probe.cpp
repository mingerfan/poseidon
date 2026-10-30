// Composite subgraph against independently checked arithmetic
// fixture. MockVec evaluates decoded arithmetic; it cannot validate CKKS noise.
#include "api/mock_api.hpp"
#include "runtime/json_plan_reader.hpp"
#include "runtime/operator_spec_reader.hpp"
#include "runtime/runtime.hpp"
#include <nlohmann/json.hpp>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sys/resource.h>
using namespace fhegpu;
using Json = nlohmann::json;
std::uint64_t peak_rss() {
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage)) throw std::runtime_error("getrusage failed");
    return std::uint64_t(usage.ru_maxrss)*1024;
}

class RangedMock : public MockVecApi {
public:
 using MockVecApi::MockVecApi;
 Json ranges=Json::array();
 std::filesystem::path directory;
 bool diagnostics=false;
 Value compute(const ComputeOp &op,const std::vector<Value> &inputs) {
   if(op.kind==ComputeKind::Boot){
     const auto values=inputs.at(0).materialize().slots;
     double maxabs=0,minv=values[0],maxv=values[0],prefix=0,tail=0;
     std::size_t large=0;
     for(std::size_t i=0;i<values.size();++i){
       maxabs=std::max(maxabs,std::abs(values[i]));minv=std::min(minv,values[i]);maxv=std::max(maxv,values[i]);
       if(i<1792)prefix=std::max(prefix,std::abs(values[i]));else tail=std::max(tail,std::abs(values[i]));
       if(std::abs(values[i])>1)++large;
     }
     if(diagnostics){
      std::ofstream f(directory/("boot-"+std::to_string(op.output)+"-reference.bin"),std::ios::binary);
      f.write(reinterpret_cast<const char*>(values.data()),values.size()*sizeof(double));
      if(!f)throw std::runtime_error("Cannot save diagnostic Mock reference");
     }
     ranges.push_back({{"input_id",op.inputs.at(0)},{"output_id",op.output},
       {"min",minv},{"max",maxv},{"max_abs",maxabs},{"prefix_1792_max_abs",prefix},
       {"tail_max_abs",tail},{"slots_abs_gt_one",large}});

   }
   return MockVecApi::compute(op,inputs);
 }
};

int main(int argc, char **argv) {
    if (argc != 6) { std::cerr << "PLAN SPEC BUNDLE FIXTURE REPORT\n"; return 2; }
    Json report = {{"mode", "plaintext_mock_runtime"}, {"ckks_executed", false},
                   {"retention", "FinalOnly"}, {"status", "loading"}};
    auto save = [&] {
        report["peak_rss_bytes"] = peak_rss();
        std::ofstream output(argv[5]);
        if (!output) throw std::runtime_error("Cannot write report");
        output << report.dump(2) << '\n';
    };
    try {
        auto plan = RuntimePlanJsonReader::read_file(argv[1]);
        auto spec = OperatorSpecReader::read_file(argv[2]);
        PlanVerifier::verify(plan.plan, spec, false);
        Json fixture; std::ifstream input(argv[4]); input >> fixture;
        if (plan.plan.external_inputs.size() != 2 || plan.plan.final_outputs.size() != 7 ||
            fixture.at("inputs").size() != 2 || fixture.at("expected").size() != 7)
            throw std::runtime_error("Expected two full blocks, two inputs and seven outputs");
        std::unordered_map<ValueId, VecValue> inputs;
        for (std::size_t i=0; i<plan.plan.external_inputs.size(); ++i) {
            auto id = plan.plan.external_inputs[i];
            const ValueDesc *desc = nullptr;
            for (const auto &value : plan.plan.values) if (value.id == id) desc = &value;
            if (!desc) throw std::runtime_error("Missing input descriptor");
            inputs.emplace(id, make_cipher(fixture.at("inputs").at(i).get<std::vector<double>>(),
                desc->context, spec.spec.poly_degree, desc->level, desc->scale_log2,
                desc->ntt, desc->components));
        }
        auto cluster = std::make_shared<MockCluster>(MockClusterConfig{});
        RangedMock api(0, cluster); api.directory=std::filesystem::path(argv[5]).parent_path();
        api.diagnostics=fixture.value("short_test",false);
        SequentialRuntime<RangedMock> runtime(0, 1, 0, api);
        RuntimeResources resources{spec, std::filesystem::path(argv[3]), false};
        report["status"] = "executing"; save();
        const auto start = std::chrono::steady_clock::now();
        auto artifact = runtime.run(plan, resources, inputs, DiffMode::FinalOnly);
        report["runtime_seconds"] = std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
  report["runtime_timing"]={{"encode_calls",artifact.timing.encode_calls},{"bundle_read_calls",artifact.timing.bundle_read_calls},{"bundle_read_bytes",artifact.timing.bundle_read_bytes},{"raw_peak_bytes",artifact.timing.raw_peak_bytes},{"bundle_resident_bytes",artifact.timing.bundle_resident_bytes},{"encode_nanoseconds",artifact.timing.encode_nanoseconds},{"bundle_read_nanoseconds",artifact.timing.bundle_read_nanoseconds},{"initialization_nanoseconds",artifact.timing.initialization_nanoseconds},{"online_execution_nanoseconds",artifact.timing.online_execution_nanoseconds},{"boot_calls",artifact.timing.boot_calls}};
        if(artifact.values.size()!=7) throw std::runtime_error("FinalOnly scope mismatch");
        report["retained_values"] = artifact.values.size(); report["boot_ranges"]=api.ranges;
        double error=0, tail=0;
        report["outputs"] = Json::array();
        for (std::size_t i=0; i<plan.plan.final_outputs.size(); ++i) {
            const auto actual = artifact.values.at(plan.plan.final_outputs[i]).value.materialize().slots;
            const auto expected = fixture.at("expected").at(i).get<std::vector<double>>();
            if (actual.size() != 32768 || expected.size()>actual.size()) throw std::runtime_error("Output shape mismatch");
            double row_error=0, row_tail=0;
            for (std::size_t j=0; j<actual.size(); ++j) {
                if (!std::isfinite(actual[j])) throw std::runtime_error("Nonfinite output");
                if (j<expected.size()) row_error=std::max(row_error, std::abs(actual[j]-expected[j]));
                else row_tail=std::max(row_tail, std::abs(actual[j]));
            }
            error=std::max(error,row_error); tail=std::max(tail,row_tail);
            report["outputs"].push_back({{"name", fixture.at("output_names").at(i)},
                {"max_abs_error", row_error}, {"max_abs_tail", row_tail}});
        }
        const bool passed=error<1e-5 && tail<1e-10;
        report["max_abs_error"]=error; report["max_abs_tail"]=tail;
        report["threshold"]=1e-5; report["tail_threshold"]=1e-10;
        report["passed"]=passed; report["status"]=passed?"passed":"failed_precision";
        save(); std::cout << report.dump() << '\n'; return passed?0:1;
    } catch (const std::exception &error) {
        report["status"]="failed"; report["error"]=error.what(); save();
        std::cerr << error.what() << '\n'; return 1;
    }
}
