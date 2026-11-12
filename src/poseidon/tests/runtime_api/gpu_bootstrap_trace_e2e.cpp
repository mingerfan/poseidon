// Native Boot versus expanded API capture and SequentialRuntime replay.
// All keys, profile data, evaluator and workspace outlive capture and replay.
#define POSEIDON_S2C_FIRST_ENABLE_TIMING
#include "../../../../bench/resnet20_s2c_first/traced_gpu_evaluator.h"
#include "poseidon/ckks_encoder.h"
#include "poseidon/encryptor.h"
#include "poseidon/gpu/gpu_bootstrap_profile.h"
#include "poseidon/gpu/gpu_uploader.h"
#include "poseidon/keygenerator.h"

#include <cuda_runtime_api.h>
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace poseidon;
using namespace poseidon::gpu;
namespace {
using Id = ApiTraceRecorder::ValueId;
const std::string kContext = "bootstrap-trace-test";
const std::string kDigest = "sha256:" + std::string(64, '0');

void require(bool condition, const std::string &message) {
    if (!condition) throw std::runtime_error(message);
}

void exact(const GpuCiphertextData &expected, const GpuCiphertextData &actual,
           const PoseidonContext &context, const std::string &label) {
    gpu_check_cuda(cudaDeviceSynchronize(), "synchronize Boot comparison");
    require(api_trace_detail::same_meta(expected.meta, actual.meta), label + ": physical metadata differs");
    Ciphertext a, b;
    GpuUploader::download_ciphertext(expected, a, context);
    GpuUploader::download_ciphertext(actual, b, context);
    require(a.parms_id() == b.parms_id() && a.scale() == b.scale() && a.size() == b.size() &&
            a.is_ntt_form() == b.is_ntt_form() && a.poly_modulus_degree() == b.poly_modulus_degree() &&
            a.coeff_modulus_size() == b.coeff_modulus_size(), label + ": host metadata differs");
    const auto words = a.size() * a.poly_modulus_degree() * a.coeff_modulus_size();
    const auto mismatch = std::mismatch(a.data(), a.data() + words, b.data());
    require(mismatch.first == a.data() + words,
            label + ": RNS mismatch at word " + std::to_string(mismatch.first - a.data()));
}

// Test the graph, not merely its opcode histogram: all calls must originate in
// the registered source, both EvalMods must be independent but share a non-root
// ancestor, and the final value must depend on both branches.
void check_graph(const ApiTraceRecorder &recorder, std::size_t boot_count = 1) {
    const auto &nodes = recorder.nodes();
    require(recorder.input_ids().size() == 1 && nodes.size() >= 6 * boot_count, "Boot was not expanded");
    const Id root = recorder.input_ids().front();
    std::unordered_map<Id, std::unordered_set<Id>> ancestors;
    ancestors[root].insert(root);
    std::vector<Id> evalmods;
    std::size_t raises = 0, linears = 0;
    for (const auto &node : nodes) {
        require(node.opcode != "bootstrap", "opaque bootstrap in expanded capture");
        require(!node.inputs.empty(), "stage has no ciphertext dependencies");
        std::unordered_set<Id> parents;
        for (const auto input : node.inputs) {
            require(ancestors.count(input), "stage uses an undefined or forward SSA value");
            const auto &set = ancestors.at(input);
            parents.insert(set.begin(), set.end());
        }
        require(parents.count(root), "stage is not rooted in Boot input");
        if (node.opcode.find("eval_mod_high_precision") != std::string::npos)
            evalmods.push_back(node.output);
        raises += node.opcode.find("bootstrap_raise") != std::string::npos;
        linears += node.opcode.find("bootstrap_linear") != std::string::npos;
        parents.insert(node.output);
        require(ancestors.emplace(node.output, std::move(parents)).second, "duplicate SSA output");
    }
    require(raises == boot_count && linears >= boot_count && evalmods.size() == 2 * boot_count,
            "missing raise, linear layer, or two EvalMod stages per Boot");
    const auto &result = ancestors.at(recorder.final_value_id());
    for (std::size_t i = 0; i < evalmods.size(); i += 2) {
        const auto &left = ancestors.at(evalmods[i]);
        const auto &right = ancestors.at(evalmods[i + 1]);
        require(!left.count(evalmods[i + 1]) && !right.count(evalmods[i]), "EvalMod branches are not independent");
        bool shared = false;
        for (Id id : left) if (id != root && right.count(id)) shared = true;
        require(shared, "EvalMod branches have no common split predecessor");
        require(result.count(evalmods[i]) && result.count(evalmods[i + 1]),
                "final output ignores an EvalMod branch");
    }
}

fhegpu::RuntimePlan make_plan(const ApiTraceRecorder &recorder, bool reverse_ready = false) {
    using namespace fhegpu;
    const Place device{PlaceKind::Device, 0, 0};
    RuntimePlan plan;
    plan.format_version = 8;
    plan.target = {kContext, 1, {1}, 1, {kContext, 1, kDigest}};
    std::unordered_map<Id, ValueId> ids;
    const auto append = [&](Id trace_id) {
        const auto &v = recorder.value(trace_id);
        const auto id = static_cast<ValueId>(ids.size());
        require(ids.emplace(trace_id, id).second, "duplicate trace value");
        if (v.buffer) {
            plan.values.push_back({id, ValueKind::Buffer, device, kContext, 0, 0, false, 0,
                                   v.hoist.bytes, v.hoist.layout});
        } else {
            const auto &m = v.meta;
            plan.values.push_back({id, ValueKind::Ciphertext, device, kContext,
                                   int(m.q_count - 1), std::log2(m.scale), m.is_ntt_form,
                                   int(m.component_count), 0, {}});
        }
    };
    for (auto id : recorder.input_ids()) {
        append(id);
        plan.external_inputs.push_back(ids.at(id));
    }
    for (const auto &node : recorder.nodes()) append(node.output);
    plan.final_outputs = {ids.at(recorder.final_value_id())};
    std::unordered_map<Id, std::size_t> remaining;
    for (const auto &node : recorder.nodes())
        for (Id id : node.inputs) ++remaining[id];
    std::unordered_set<Id> available(recorder.input_ids().begin(), recorder.input_ids().end());
    const std::unordered_set<Id> external(recorder.input_ids().begin(), recorder.input_ids().end());
    std::vector<bool> done(recorder.nodes().size());
    for (std::size_t completed = 0; completed < done.size(); ++completed) {
        std::size_t selected = done.size();
        for (std::size_t k = 0; k < done.size(); ++k) {
            const std::size_t i = reverse_ready ? done.size() - 1 - k : k;
            if (done[i]) continue;
            const auto &node = recorder.nodes()[i];
            if (std::all_of(node.inputs.begin(), node.inputs.end(),
                            [&](Id id) { return available.count(id) != 0; })) {
                selected = i;
                break;
            }
        }
        require(selected != done.size(), "trace graph has a dependency cycle");
        const auto &node = recorder.nodes()[selected];
        std::vector<ValueId> inputs;
        for (Id id : node.inputs) inputs.push_back(ids.at(id));
        plan.execution.push_back({plan.execution.size(), ComputeOp{ComputeKind::ApiCall, std::move(inputs),
            ids.at(node.output), device, ApiCallAttrs{node.callee, selected}}});
        available.insert(node.output);
        done[selected] = true;
        for (Id id : node.inputs) {
            require(remaining.at(id) > 0, "bad trace use count");
            if (--remaining[id] == 0 && !external.count(id) && id != recorder.final_value_id())
                plan.execution.push_back({plan.execution.size(), ReleaseOp{ids.at(id)}});
        }
        if (!remaining[node.output] && node.output != recorder.final_value_id())
            plan.execution.push_back({plan.execution.size(), ReleaseOp{ids.at(node.output)}});
    }
    return plan;
}

fhegpu::LoadedOperatorSpec make_spec(const PoseidonContext &context) {
    fhegpu::OperatorSpec spec;
    spec.id = kContext; spec.version = 1; spec.status = "test";
    spec.target_id = kContext; spec.context_id = kContext;
    const auto &parameters = *context.parameters_literal();
    spec.poly_degree = parameters.degree();
    for (const auto &q : parameters.q()) {
        spec.rns_moduli.push_back(q.value());
        spec.rns_moduli_log2.push_back(q.bit_count());
    }
    spec.max_modulus_log2 = *std::max_element(spec.rns_moduli_log2.begin(), spec.rns_moduli_log2.end());
    spec.default_scale_log2 = int(parameters.log_scale());
    spec.level_lower_bound = 0; spec.level_upper_bound = int(parameters.q().size() - 1);
    return {std::move(spec), kDigest};
}

void run_test() {
    // Same native profile and chain as poseidon_gpu_bootstrap_e2e, but a smaller
    // ring makes this a practical regression rather than a production benchmark.
    const int log_n = std::getenv("POSEIDON_TRACE_BOOT_LOG_N")
        ? std::stoi(std::getenv("POSEIDON_TRACE_BOOT_LOG_N")) : 12;
    require(log_n >= 12 && log_n <= 16, "POSEIDON_TRACE_BOOT_LOG_N must be in [12,16]");
    ParametersLiteral literal(CKKS, log_n, log_n - 1, 40, 5, 1, Modulus(0), {}, {}, sec_level_type::none);
    literal.set_log_modulus(std::vector<std::uint32_t>(50, 30), std::vector<std::uint32_t>(8, 30));
    PoseidonContext context(literal);
    KeyGenerator generator(context);
    require(!generator.secret_key().data().is_zero(), "fixture needs nonzero secret key");
    PublicKey public_key;
    generator.create_public_key(public_key);
    CKKSEncoder encoder(context);
    Encryptor encryptor(context, public_key);
    GpuBootstrapProfileConfig config;
    config.eval_mod_log_scale = 45;
    config.eval_mod_sine_degree = 58;
    config.eval_mod_generation_degree = 59;
    config.eval_mod_truncate_degree.reset();
    config.eval_mod_double_angle = 2;
    auto profile = GpuBootstrapProfileBuilder::build(context, generator, 0, config);
    require(bool(profile.relin_keys) && bool(profile.galois_keys), "profile missing keys");
    std::vector<std::complex<double>> message(encoder.slot_count());
    for (std::size_t i = 0; i < message.size(); ++i)
        message[i] = {double(i % 17 + 1) / 32., double(i % 11 + 1) / 64.};
    Plaintext plain;
    encoder.encode(message, context.crt_context()->parms_id_map().at(profile.input_level_max),
                   std::ldexp(1., 40), plain);
    Ciphertext host;
    encryptor.encrypt(plain, host);
    GpuParameterData parameters(context, 0);
    for (auto &level : profile.level_parameters) parameters.adopt_level(std::move(level));
    GpuEvaluator evaluator(parameters);
    GpuBootstrapWorkspace workspace;
    auto input = std::make_shared<GpuCiphertextData>(GpuUploader::upload_ciphertext(host, 0));
    GpuCiphertextData reference;
    evaluator.bootstrap(*input, profile.bootstrap_data, *profile.relin_keys, *profile.galois_keys,
                        workspace, reference);
    gpu_check_cuda(cudaDeviceSynchronize(), "complete fused reference and warm workspace");
    const auto spec = make_spec(context);
    for (bool alias : {false, true}) {
        // Fresh upload protects the identical oracle input from in-place writes.
        auto source = std::make_shared<GpuCiphertextData>(GpuUploader::upload_ciphertext(host, 0));
        ApiTraceRecorder recorder(context, evaluator);
        recorder.register_input(source);
        GpuCiphertextData output;
        {
            ScopedApiTrace capture(recorder);
            if (alias) {
                // Registered inputs are immutable. Create an explicit traced
                // copy first, then exercise aliasing on that intermediate.
                trace_gpu(evaluator).drop_modulus(*source, output, source->meta.parms_id);
                trace_gpu(evaluator).bootstrap(output, profile.bootstrap_data, *profile.relin_keys,
                                               *profile.galois_keys, workspace, output);
            } else {
                trace_gpu(evaluator).bootstrap(*source, profile.bootstrap_data, *profile.relin_keys,
                                               *profile.galois_keys, workspace, output);
            }
            recorder.finish(output);
        }
        check_graph(recorder);
        exact(reference, output, context, alias ? "capture intermediate alias" : "capture");
        exact(*input, *source, context, "capture preserved external input");
        const auto plan = make_plan(recorder);
        TracedPlanApi api(recorder, plan, context, evaluator);
        fhegpu::SequentialRuntime<TracedPlanApi> runtime(0, 1, 1, api);
        const auto artifact = runtime.run(fhegpu::LoadedRuntimePlan{plan, kDigest},
                                          fhegpu::RuntimeResources{spec, std::nullopt, false},
                                          api.external_inputs());
        exact(reference, TracedPlanApi::result_ciphertext(artifact.values.at(plan.final_outputs.front()).value),
              context, alias ? "replay alias" : "replay");
        require(api.calls() == recorder.nodes().size(),
                "replay did not dispatch exactly the expanded nodes");
        std::cout << "PASS native expanded Boot " << (alias ? "alias" : "nonalias")
                  << " nodes=" << recorder.nodes().size() << '\n';
    }

    // Reject mutation of a registered input before any GPU work, including a
    // metadata-only mutation whose shape would otherwise still pass preflight.
    {
        auto protected_input = std::make_shared<GpuCiphertextData>(GpuUploader::upload_ciphertext(host, 0));
        ApiTraceRecorder rejected(context, evaluator);
        rejected.register_input(protected_input);
        ScopedApiTrace capture(rejected);
        bool boot_rejected = false, scale_rejected = false;
        try {
            trace_gpu(evaluator).bootstrap(*protected_input, profile.bootstrap_data, *profile.relin_keys,
                                           *profile.galois_keys, workspace, *protected_input);
        } catch (const std::runtime_error &) { boot_rejected = true; }
        try { trace_gpu(evaluator).set_scale(*protected_input, protected_input->meta.scale); }
        catch (const std::runtime_error &) { scale_rejected = true; }
        require(boot_rejected && scale_rejected && rejected.nodes().empty(),
                "registered input overwrite was not rejected before recording");
        exact(*input, *protected_input, context, "rejected input overwrite");
    }

    // Keep the first Boot result alive across another complete Boot on the
    // same workspace, then consume both; no workspace concurrency is assumed.
    GpuCiphertextData expected_sum;
    evaluator.add(reference, reference, expected_sum);
    auto source = std::make_shared<GpuCiphertextData>(GpuUploader::upload_ciphertext(host, 0));
    ApiTraceRecorder recorder(context, evaluator);
    recorder.register_input(source);
    GpuCiphertextData first, second, sum;
    {
        ScopedApiTrace capture(recorder);
        trace_gpu(evaluator).bootstrap(*source, profile.bootstrap_data, *profile.relin_keys,
                                       *profile.galois_keys, workspace, first);
        trace_gpu(evaluator).bootstrap(*source, profile.bootstrap_data, *profile.relin_keys,
                                       *profile.galois_keys, workspace, second);
        trace_gpu(evaluator).add(first, second, sum);
        recorder.finish(sum);
    }
    check_graph(recorder, 2);
    exact(expected_sum, sum, context, "two Boot capture");
    for (bool reverse_ready : {false, true}) {
        const auto plan = make_plan(recorder, reverse_ready);
        if (reverse_ready) {
            std::vector<std::uint64_t> call_order;
            for (const auto &instruction : plan.execution)
                if (const auto *op = std::get_if<fhegpu::ComputeOp>(&instruction.body))
                    call_order.push_back(std::get<fhegpu::ApiCallAttrs>(op->attrs).call_id);
            require(!std::is_sorted(call_order.begin(), call_order.end()),
                    "reverse-ready scheduler did not reorder any calls");
        }
        TracedPlanApi api(recorder, plan, context, evaluator);
        fhegpu::SequentialRuntime<TracedPlanApi> runtime(0, 1, 1, api);
        const auto artifact = runtime.run(fhegpu::LoadedRuntimePlan{plan, kDigest},
                                          fhegpu::RuntimeResources{spec, std::nullopt, false},
                                          api.external_inputs());
        exact(expected_sum, TracedPlanApi::result_ciphertext(artifact.values.at(plan.final_outputs.front()).value),
              context, reverse_ready ? "two Boot reordered replay" : "two Boot replay");
        require(api.calls() == recorder.nodes().size(), "two Boot replay missed calls");
    }
    std::cout << "PASS two live Boot results, shared workspace, nodes=" << recorder.nodes().size() << '\n';
}
} // namespace

int main() {
    try {
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
            std::cerr << "[SKIP] GPU native Boot needs CUDA device 0\n";
            return 77;
        }
        gpu_check_cuda(cudaSetDevice(0), "select Boot trace device");
        run_test();
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }
}
