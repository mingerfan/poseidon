#include "runtime/binary_plan_codec.hpp"
#include "runtime/operator_spec_reader.hpp"
#include "runtime/plan_reader.hpp"
#include "runtime/verifier.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

using namespace fhegpu;
using Json = nlohmann::json;

std::vector<ValueId> inputs(const Instruction &i) {
    if (const auto *p=std::get_if<ComputeOp>(&i.body)) return p->inputs;
    if (const auto *p=std::get_if<CommAction>(&i.body)) return p->inputs;
    if (const auto *p=std::get_if<ReleaseOp>(&i.body)) return {p->value};
    return {};
}

int main(int argc,char **argv) {
 try {
    if(argc!=6)throw std::runtime_error("usage: make-prefix SOURCE_PLAN SPEC FIXTURE OUTPUT_DIR EXECUTION_LIMIT");
    const auto limit=std::stoull(argv[5]);
    if(!limit)throw std::runtime_error("execution limit must be positive");
    const std::filesystem::path out=argv[4];
    if(!std::filesystem::create_directory(out))throw std::runtime_error("output directory must not exist");
    auto loaded=RuntimePlanReader::read_file(argv[1]);
    auto &source=loaded.plan;
    const auto spec=OperatorSpecReader::read_file(argv[2]);
    Json fixture;{std::ifstream stream(argv[3]);stream>>fixture;}
    if(fixture.at("expected_outputs").size()!=source.final_outputs.size())
        throw std::runtime_error("full fixture output order does not match source plan");
    std::unordered_map<ValueId,EncodeOp> encodes;
    std::unordered_map<ValueId,PreparePlaintextOp> preparations;
    for(const auto *phase:{&source.initialization,&source.execution,&source.finalization})
        for(const auto &i:*phase) {
            if(const auto *op=std::get_if<EncodeOp>(&i.body))encodes.emplace(op->output,*op);
            if(const auto *op=std::get_if<PreparePlaintextOp>(&i.body))
                preparations.emplace(op->output,*op);
        }
    std::unordered_set<std::size_t> converted_transfers;
    for(const auto *phase:{&source.initialization,&source.execution,&source.finalization})
        for(const auto &i:*phase)if(const auto *op=std::get_if<CommAction>(&i.body)) {
            if(op->inputs.size()!=1 || !encodes.count(op->inputs[0]))continue;
            for(std::size_t j=0;j<op->outputs.size();++j) {
                if(op->destinations[j].kind!=PlaceKind::Device || op->output_types[j]!=ValueKind::Plaintext)
                    throw std::runtime_error("constant upload is not a Device plaintext");
                if(!preparations.emplace(op->outputs[j],PreparePlaintextOp{encodes.at(op->inputs[0]).payload,op->outputs[j]}).second)
                    throw std::runtime_error("duplicate plaintext upload output");
            }
            converted_transfers.insert(i.ordinal);
        }
    RuntimePlan prefix;
    prefix.format_version=4;prefix.plan_id=source.plan_id;
    prefix.target=source.target;prefix.plaintext_bundle=source.plaintext_bundle;
    prefix.external_inputs=source.external_inputs;
    const auto devices=prefix.target.device_counts.at(0);
    WaterlineConfig water;
    water.device_capacity_bytes.assign(devices,31ULL<<30);
    water.device_reserve_bytes.assign(devices,6ULL<<30);
    water.device_fractions.assign(devices,.5);
    water.host_budget_bytes=64ULL<<30;water.pinned_budget_bytes=1ULL<<30;
    water.raw_budget_bytes=64ULL<<20;water.lookahead_groups=128;water.backend=PrepareBackend::Auto;
    if(source.waterline)water=*source.waterline;
    water.lookahead_groups=128;water.backend=PrepareBackend::Auto;
    prefix.waterline=water;
    std::unordered_set<ValueId> live(prefix.external_inputs.begin(),prefix.external_inputs.end());
    std::unordered_set<ValueId> touched=live,prepared;
    const auto apply=[&](const Instruction &i) {
        for(auto id:inputs(i)) {
            if(!live.count(id))throw std::runtime_error("prefix input unavailable: "+std::to_string(id)+" source ordinal "+std::to_string(i.ordinal));
            touched.insert(id);
        }
        const auto define=[&](ValueId id) {if(!live.insert(id).second)throw std::runtime_error("duplicate live output");touched.insert(id);};
        if(const auto *op=std::get_if<ComputeOp>(&i.body)) {
            if(op->reuse_input)live.erase(op->inputs.at(0));
            define(op->output);
        } else if(const auto *op=std::get_if<CommAction>(&i.body))for(auto id:op->outputs)define(id);
        else if(const auto *op=std::get_if<ReleaseOp>(&i.body))live.erase(op->value);
        else if(const auto *op=std::get_if<PreparePlaintextOp>(&i.body))define(op->output);
        else if(!std::holds_alternative<FenceOp>(i.body))throw std::runtime_error("unexpected prefix instruction");
    };
    const auto skip=[&](const Instruction &i) {
        if(std::holds_alternative<EncodeOp>(i.body) || std::holds_alternative<PreparePlaintextOp>(i.body) || converted_transfers.count(i.ordinal))return true;
        const auto *release=std::get_if<ReleaseOp>(&i.body);
        return release && encodes.count(release->value);
    };
    for(const auto &i:source.initialization)if(!skip(i)) {
        for(auto id:inputs(i))if(preparations.count(id))throw std::runtime_error("eager initialization has a constant consumer");
        apply(i);prefix.initialization.push_back(i);
    }
    std::size_t source_kept=0,source_first=0,source_last=0;
    for(const auto &i:source.execution) {
        if(skip(i))continue;
        std::vector<ValueId> needed;
        for(auto id:inputs(i))if(encodes.count(id))throw std::runtime_error("Host plaintext has an unsupported non-upload consumer");
        for(auto id:inputs(i))if(preparations.count(id) && !prepared.count(id) && std::find(needed.begin(),needed.end(),id)==needed.end())needed.push_back(id);
        if(prefix.execution.size()+needed.size()+1>limit)break;
        for(auto id:needed) {
            Instruction prep{i.ordinal,preparations.at(id)};
            apply(prep);prefix.execution.push_back(std::move(prep));prepared.insert(id);
        }
        apply(i);prefix.execution.push_back(i);
        if(!source_kept)source_first=i.ordinal;
        source_last=i.ordinal;++source_kept;
    }
    // Only already-computable original output downloads are allowed after the
    // prefix. No further model arithmetic is introduced to obtain a checkpoint.
    for(const auto &i:source.finalization)if(const auto *op=std::get_if<CommAction>(&i.body)) {
        bool ready=true;for(auto id:op->inputs)ready&=live.count(id)!=0;
        if(ready) {apply(i);prefix.finalization.push_back(i);}
    }
    Json rows=Json::array(),indices=Json::array();
    for(std::size_t index=0;index<source.final_outputs.size();++index) {
        const auto id=source.final_outputs[index];
        if(live.count(id)) {
            prefix.final_outputs.push_back(id);
            auto row=fixture.at("expected_outputs").at(index);row["value_id"]=id;
            rows.push_back(std::move(row));indices.push_back(index);
        }
    }
    if(prefix.final_outputs.empty())throw std::runtime_error("prefix reaches no reference-backed Qwen output; performance-only mode must be explicit");
    std::unordered_set<ValueId> keep(prefix.final_outputs.begin(),prefix.final_outputs.end());
    keep.insert(prefix.external_inputs.begin(),prefix.external_inputs.end());
    std::vector<ValueId> cleanup;
    for(auto id:live)if(!keep.count(id))cleanup.push_back(id);
    std::sort(cleanup.begin(),cleanup.end());
    for(auto id:cleanup) {Instruction i{0,ReleaseOp{id}};apply(i);prefix.finalization.push_back(std::move(i));}
    for(const auto &value:source.values)if(touched.count(value.id))prefix.values.push_back(value);
    std::size_t ordinal=0;
    for(auto *phase:{&prefix.initialization,&prefix.execution,&prefix.finalization})
        for(auto &i:*phase)i.ordinal=ordinal++;
    PlanVerifier::verify(prefix,spec,false);
    Json counts=Json::object(),per_device=Json::object();
    const char *tags[]={"encode","compute","communication","release","fence","prepare"};
    for(const auto &i:prefix.execution) {
        const std::string tag=tags[i.body.index()];
        counts[tag]=counts.value(tag,std::uint64_t(0))+1;
        if(const auto *op=std::get_if<ComputeOp>(&i.body)) {
            const auto device=op->place.kind==PlaceKind::Host?"host":std::to_string(op->place.index);
            if(!per_device.contains(device))per_device[device]=0;
            per_device[device]=per_device[device].get<std::uint64_t>()+1;
        }
    }
    const auto plan_path=out/"prefix.runtime-plan.bin";
    {std::ofstream stream(plan_path,std::ios::binary);stream.exceptions(std::ios::badbit|std::ios::failbit);binary_io::write_plan(stream,prefix);}
    fixture["expected_outputs"]=std::move(rows);
    fixture["model"]["complete_qwen"]=false;
    fixture["model"]["full_model_ckks_validated"]=false;
    fixture["model"]["model_completeness"]="Bounded execution prefix of the complete pretrained 28-layer graph; not a complete forward.";
    fixture["model"]["prefix_execution_instructions"]=prefix.execution.size();
    fixture["model"]["checkpoint_output_count"]=prefix.final_outputs.size();
    {std::ofstream stream(out/"fixture.json");stream.exceptions(std::ios::badbit|std::ios::failbit);stream<<fixture.dump(2)<<'\n';}
    Json summary{{"source_plan",argv[1]},{"source_waterline",source.waterline.has_value()},{"source_fixture",argv[3]},
        {"limit",limit},{"limit_scope","execution records including first-use Prepare; excludes input initialization and output downloads/Release cleanup"},
        {"execution_instructions",prefix.execution.size()},{"execution_counts",counts},
        {"compute_by_place",per_device},{"initialization_instructions",prefix.initialization.size()},
        {"finalization_instructions",prefix.finalization.size()},{"cleanup_releases",cleanup.size()},
        {"source_execution_records_kept",source_kept},{"source_first_ordinal",source_first},{"source_last_ordinal",source_last},
        {"checkpoint_original_indices",indices},{"values",prefix.values.size()},
        {"devices",devices},{"verifier_passed",true},{"config",binary_io::metadata(prefix).at("waterline")}};
    {std::ofstream stream(out/"prefix-summary.json");stream.exceptions(std::ios::badbit|std::ios::failbit);stream<<summary.dump(2)<<'\n';}
    std::cout<<summary.dump(2)<<'\n';
    return 0;
 } catch(const std::exception &error) {std::cerr<<"make-prefix: "<<error.what()<<'\n';return 1;}
}
