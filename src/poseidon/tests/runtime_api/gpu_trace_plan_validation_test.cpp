// Standalone GPU regression for TracedPlanApi::validate_plan. Exit 77 means
// CUDA is unavailable; select a physical GPU with CUDA_VISIBLE_DEVICES.
#define POSEIDON_S2C_FIRST_ENABLE_TIMING
#include "../../../../bench/resnet20_s2c_first/traced_gpu_evaluator.h"
#include "poseidon/ckks_encoder.h"
#include "poseidon/encryptor.h"
#include "poseidon/gpu/gpu_uploader.h"
#include "poseidon/keygenerator.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

using namespace poseidon;
using namespace poseidon::gpu;

namespace {
void require(bool condition, const std::string &message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename Function>
void rejects(Function &&function, const std::string &label) {
    try { function(); }
    catch (const std::exception &) { return; }
    throw std::runtime_error(label + ": validate_plan unexpectedly accepted the plan");
}

ParametersLiteral parameters() {
    ParametersLiteral result{CKKS, 12, 11, 44, 64, 0, 0, {}, {}, sec_level_type::none};
    result.set_log_modulus({29, 30, 31, 32}, {30, 31});
    return result;
}

// Construct descriptors and operations from the actual capture, not assumptions
// about SSA IDs or closure names. Plan IDs intentionally differ from trace IDs.
fhegpu::RuntimePlan make_plan(const ApiTraceRecorder &recorder) {
    using namespace fhegpu;
    const Place device{PlaceKind::Device, 0, 0};
    RuntimePlan plan;
    plan.format_version = 8;
    plan.target = {"gpu-trace-test", 1, {1}, 1,
                   {"gpu-trace-test", 1, "sha256:" + std::string(64, '0')}};
    std::unordered_map<ApiTraceRecorder::ValueId, ValueId> ids;
    const auto append_value = [&](ApiTraceRecorder::ValueId trace_id) {
        const auto &value = recorder.value(trace_id);
        require(!value.buffer, "fixture unexpectedly contains a buffer");
        const auto id = static_cast<ValueId>(ids.size());
        require(ids.emplace(trace_id, id).second, "duplicate recorded value");
        const auto &meta = value.meta;
        plan.values.push_back({id, ValueKind::Ciphertext, device, "gpu-trace-context",
                               static_cast<int>(meta.q_count - 1), std::log2(meta.scale),
                               meta.is_ntt_form, static_cast<int>(meta.component_count), 0, {}});
    };
    for (auto id : recorder.input_ids()) {
        append_value(id);
        plan.external_inputs.push_back(ids.at(id));
    }
    for (std::size_t i = 0; i < recorder.nodes().size(); ++i) {
        const auto &node = recorder.nodes()[i];
        std::vector<ValueId> inputs;
        for (auto id : node.inputs) inputs.push_back(ids.at(id));
        append_value(node.output);
        plan.execution.push_back({i, ComputeOp{ComputeKind::ApiCall, std::move(inputs),
            ids.at(node.output), device, ApiCallAttrs{node.callee, i}}});
    }
    plan.final_outputs = {ids.at(recorder.final_value_id())};
    return plan;
}

void run_test() {
    PoseidonContext context(parameters());
    KeyGenerator generator(context);
    require(!generator.secret_key().data().is_zero(), "test requires a nonzero secret key");
    PublicKey public_key;
    generator.create_public_key(public_key);
    CKKSEncoder encoder(context);
    Encryptor encryptor(context, public_key);
    std::vector<std::complex<double>> values(encoder.slot_count(), {0.125, -0.0625});
    Plaintext plain;
    encoder.encode(values, std::ldexp(1., 44), plain);
    Ciphertext host_a, host_b;
    encryptor.encrypt(plain, host_a);
    encryptor.encrypt(plain, host_b);
    auto a = std::make_shared<GpuCiphertextData>(GpuUploader::upload_ciphertext(host_a, 0));
    auto b = std::make_shared<GpuCiphertextData>(GpuUploader::upload_ciphertext(host_b, 0));
    GpuParameterData gpu_parameters(context, 0, {4});
    GpuEvaluator evaluator(gpu_parameters);
    ApiTraceRecorder recorder(context, evaluator);
    recorder.register_input(a);
    recorder.register_input(b);
    GpuCiphertextData left, right, out;
    {
        ScopedApiTrace scope(recorder);
        trace_gpu(evaluator).add(*a, *a, left);
        trace_gpu(evaluator).add(*b, *b, right);
        trace_gpu(evaluator).add(left, right, out);
        recorder.finish(out);
    }
    require(recorder.nodes().size() == 3 && recorder.input_ids().size() == 2,
            "unexpected capture shape");
    const auto plan = make_plan(recorder);
    TracedPlanApi api(recorder, plan, context, evaluator);
    api.validate_plan(plan);

    // Independent branches can be scheduled in either topological order.
    auto reordered = plan;
    std::swap(reordered.execution[0], reordered.execution[1]);
    for (std::size_t i = 0; i < reordered.execution.size(); ++i)
        reordered.execution[i].ordinal = i;
    api.validate_plan(reordered);

    auto wrong_final = plan;
    wrong_final.final_outputs[0] = std::get<fhegpu::ComputeOp>(plan.execution[0].body).output;
    rejects([&] { api.validate_plan(wrong_final); }, "intermediate final output");

    auto missing = plan;
    missing.execution.pop_back();
    rejects([&] { api.validate_plan(missing); }, "missing call");

    auto bad_signature = plan;
    std::get<fhegpu::ApiCallAttrs>(std::get<fhegpu::ComputeOp>(bad_signature.execution[0].body).attrs)
        .callee += ".different";
    rejects([&] { api.validate_plan(bad_signature); }, "changed closure signature");

    auto bad_input = plan;
    std::swap(bad_input.external_inputs[0], bad_input.external_inputs[1]);
    rejects([&] { api.validate_plan(bad_input); }, "changed external input mapping");

    auto bad_dependency = plan;
    auto &last = std::get<fhegpu::ComputeOp>(bad_dependency.execution.back().body);
    std::swap(last.inputs[0], last.inputs[1]);
    rejects([&] { api.validate_plan(bad_dependency); }, "changed dependency order");

    // This graph is independently valid for the recorder, but swaps the plan
    // IDs assigned to the two branch outputs. It specifically exercises the
    // cross-plan binding comparison rather than just constructor validation.
    auto rebound = plan;
    auto &left_op = std::get<fhegpu::ComputeOp>(rebound.execution[0].body);
    auto &right_op = std::get<fhegpu::ComputeOp>(rebound.execution[1].body);
    auto &join_op = std::get<fhegpu::ComputeOp>(rebound.execution[2].body);
    std::swap(left_op.output, right_op.output);
    std::swap(join_op.inputs[0], join_op.inputs[1]);
    TracedPlanApi independently_valid(recorder, rebound, context, evaluator);
    independently_valid.validate_plan(rebound);
    rejects([&] { api.validate_plan(rebound); }, "valid graph with changed value bindings");
    api.validate_plan(plan); // A failed validation must not corrupt the original backend.
    gpu_check_cuda(cudaDeviceSynchronize(), "synchronize trace plan test");
    std::cout << "PASS GPU trace plan validation and cross-plan binding regression\n";
}
} // namespace

int main() {
    try {
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return 77;
        gpu_check_cuda(cudaSetDevice(0), "select trace plan test device");
        run_test();
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }
}
