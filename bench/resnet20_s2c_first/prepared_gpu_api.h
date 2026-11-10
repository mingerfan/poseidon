#pragma once

#include "prepared_api_graph.h"
#include "poseidon/gpu/gpu_evaluator.h"
#include "runtime/runtime.hpp"
#include <cmath>
#include <functional>
#include <memory>
#include <unordered_map>

namespace poseidon::benchmark::s2c_first {

// A single-device backend for the real SequentialRuntime. It dispatches only
// explicitly registered prepared calls; ordinary arithmetic/communications are
// not silently emulated. The registry owns no secret key or encryptor.
class PreparedGpuApi {
public:
    using CT=poseidon::gpu::GpuCiphertextData;
    using Value=std::shared_ptr<CT>;
    using Callback=std::function<CT(const PreparedApiTask &,std::vector<CT>)>;
    struct CommHandle {};

    PreparedGpuApi(const PreparedApiGraph &graph,const fhegpu::RuntimePlan &plan,
        const poseidon::PoseidonContext &context,poseidon::gpu::GpuEvaluator &eval,Callback callback)
        :graph_(graph),context_(context),eval_(eval),callback_(std::move(callback)) {
        dispatch_order_.reserve(graph.tasks.size());
        std::unordered_map<fhegpu::ValueId,int> producers;
        std::vector<bool> seen(graph.tasks.size());
        const auto scan=[&](const std::vector<fhegpu::Instruction> &phase) {
            for(const auto &instruction:phase) {
                const auto *op=std::get_if<fhegpu::ComputeOp>(&instruction.body);
                if(!op) {
                    if(!std::holds_alternative<fhegpu::ReleaseOp>(instruction.body) &&
                        !std::holds_alternative<fhegpu::FenceOp>(instruction.body))
                        throw std::runtime_error("prepared GPU backend only accepts ApiCall/Release/Fence");
                    continue;
                }
                if(op->kind!=fhegpu::ComputeKind::ApiCall || op->reuse_input)
                    throw std::runtime_error("expected an ApiCall without reuse_input");
                const auto &attrs=std::get<fhegpu::ApiCallAttrs>(op->attrs);
                const auto &task=graph.tasks.at(attrs.call_id);
                if(seen.at(attrs.call_id) || task.callee!=attrs.callee ||
                    op->place.kind!=fhegpu::PlaceKind::Device ||
                    op->place.rank!=0 || op->place.index!=0)
                    throw std::runtime_error("RuntimePlan does not match the single-GPU API registry");
                std::vector<int> inputs;
                for(auto id:op->inputs) {inputs.push_back(producers.at(id));++remaining_[id];}
                if(inputs!=task.inputs)
                    throw std::runtime_error("RuntimePlan API dependencies differ from the manual graph");
                producers.emplace(op->output,int(attrs.call_id));seen[attrs.call_id]=true;
            }
        };
        scan(plan.initialization);scan(plan.execution);scan(plan.finalization);
        for(bool present:seen)if(!present)throw std::runtime_error("RuntimePlan is missing a prepared API call");
        if(!plan.external_inputs.empty() || plan.final_outputs.size()!=1 ||
            producers.at(plan.final_outputs.front())!=int(graph.tasks.size()-1))
            throw std::runtime_error("RuntimePlan must produce the manual encrypted logits");
    }

    std::string name() const {return "PreparedResnet20GpuApi";}
    Value compute(const fhegpu::ComputeOp &op,const std::vector<Value> &inputs) {
        if(op.kind!=fhegpu::ComputeKind::ApiCall)
            throw std::runtime_error("unsupported operation in prepared GPU API");
        const auto &attrs=std::get<fhegpu::ApiCallAttrs>(op.attrs);
        const auto &task=graph_.tasks.at(attrs.call_id);
        std::vector<CT> arguments;arguments.reserve(inputs.size());
        for(std::size_t i=0;i<inputs.size();++i) {
            // Runtime values are shared handles. Move storage only at the last
            // submitted use, otherwise preserve the branch with one GPU copy.
            // For the explicit clone/fork this yields exactly one copy total,
            // whichever branch the compiler schedules first.
            if(--remaining_.at(op.inputs[i])==0)arguments.push_back(std::move(*inputs[i]));
            else {
                CT copy;eval_.drop_modulus(*inputs[i],copy,inputs[i]->meta.parms_id);
                arguments.push_back(std::move(copy));
            }
        }
        dispatch_order_.push_back(attrs.call_id);
        ++calls_;if(task.callee.rfind("bootstrap.",0)==0)++boots_;
        return std::make_shared<CT>(callback_(task,std::move(arguments)));
    }
    void validate_value(const Value &value,const fhegpu::ValueDesc &expected) const {
        if(!value || value->empty() || expected.kind!=fhegpu::ValueKind::Ciphertext ||
            value->meta.degree!=context_.parameters_literal()->degree() ||
            value->meta.parms_id!=context_.crt_context()->parms_id_map().at(expected.level) ||
            value->meta.q_count!=std::size_t(expected.level+1) ||
            value->meta.component_count!=std::size_t(expected.components) ||
            value->meta.is_ntt_form!=expected.ntt || !std::isfinite(value->meta.scale) ||
            value->meta.scale<=0 || std::abs(std::log2(value->meta.scale)-expected.scale_log2)>1e-7)
            throw std::runtime_error("prepared API output differs from physical RuntimePlan metadata");
    }
    void preflight(std::string_view,bool,const fhegpu::TargetConfig &target,
        const fhegpu::OperatorSpec &spec,const fhegpu::PlanRequirements &) {
        if(target.world_size!=1 || target.device_counts!=std::vector<int>{1})
            throw std::runtime_error("prepared GPU backend requires exactly one GPU");
        std::vector<std::uint64_t> moduli;
        for(const auto &q:context_.parameters_literal()->q())moduli.push_back(q.value());
        if(spec.poly_degree!=context_.parameters_literal()->degree() || spec.rns_moduli!=moduli)
            throw std::runtime_error("OperatorSpec physical context differs from prepared GPU resources");
    }
    void drain() {poseidon::gpu::gpu_check_cuda(cudaDeviceSynchronize(),"prepared runtime drain");}
    void synchronize(Value &) {drain();}
    void collect_completed() {}
    [[noreturn]] void abort_all(int,const std::string &message) {throw std::runtime_error(message);}
    Value encode_plaintext(const fhegpu::ValueDesc &,const std::vector<double> &) {
        throw std::runtime_error("prepared constants must be uploaded before runtime execution");
    }
    CommHandle communicate_async(const fhegpu::CommAction &,const std::vector<Value> &,
        const std::vector<fhegpu::ValueDesc> &) {
        throw std::runtime_error("single-GPU prepared API plan must not communicate");
    }
    std::vector<std::optional<Value>> wait(CommHandle &) {
        throw std::runtime_error("single-GPU prepared API plan must not communicate");
    }
    std::size_t calls() const {return calls_;}
    std::size_t boot_calls() const {return boots_;}
    const std::vector<std::uint64_t> &dispatch_order() const {return dispatch_order_;}
private:
    const PreparedApiGraph &graph_;
    const poseidon::PoseidonContext &context_;
    poseidon::gpu::GpuEvaluator &eval_;
    Callback callback_;
    std::unordered_map<fhegpu::ValueId,std::size_t> remaining_;
    std::size_t calls_=0,boots_=0;
    std::vector<std::uint64_t> dispatch_order_;
};

} // namespace poseidon::benchmark::s2c_first
