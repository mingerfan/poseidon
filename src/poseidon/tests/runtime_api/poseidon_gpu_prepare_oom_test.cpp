#include "poseidon/ckks_encoder.h"
#include "poseidon/decryptor.h"
#include "poseidon/encryptor.h"
#include "poseidon/gpu/gpu_uploader.h"
#include "poseidon/keygenerator.h"
#include "poseidon/runtime_api/communication/cuda_local_transfer.h"
#include "poseidon/runtime_api/poseidon_gpu_api.h"
#include "runtime/runtime.hpp"

#include <rmm/mr/device/per_device_resource.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>

namespace {
using poseidon::runtime_api::PoseidonGpuApi;
using poseidon::runtime_api::PoseidonGpuValue;
using Pinned = poseidon::runtime_api::communication::PinnedHostBuffer;
const std::string context_id = "gpu-prepare-oom-test";
const std::string spec_sha = "sha256:0000000000000000000000000000000000000000000000000000000000000000";
const std::string plan_sha = "sha256:1111111111111111111111111111111111111111111111111111111111111111";

void require(bool condition, const std::string &message) {
    if (!condition) throw std::runtime_error(message);
}
void check(cudaError_t status, const char *message) {
    poseidon::gpu::gpu_check_cuda(status, message);
}

// The callback only holds the actual encoder stream. The watchdog releases
// it after observing the real allocation failure; it never calls CUDA.
struct AllocationGate {
    std::mutex mutex;
    std::condition_variable condition;
    bool entered = false, failed = false, released = false;
    bool timed_out = false, observed_unfinished_run = false;
    std::atomic<bool> run_finished{false};
    std::thread watchdog;
    AllocationGate() : watchdog([this] {
        std::unique_lock<std::mutex> lock(mutex);
        if (!condition.wait_for(lock, std::chrono::seconds(3), [&] { return failed || released; }))
            timed_out = true;
        if (failed && !released) {
            condition.wait_for(lock, std::chrono::milliseconds(50), [&] { return released; });
            observed_unfinished_run = !run_finished.load();
        }
        released = true;
        condition.notify_all();
    }) {}
    ~AllocationGate() {
        { std::lock_guard<std::mutex> lock(mutex); released = true; condition.notify_all(); }
        if (watchdog.joinable()) watchdog.join();
    }
    static void CUDART_CB wait(void *opaque) {
        auto &gate = *static_cast<AllocationGate *>(opaque);
        std::unique_lock<std::mutex> lock(gate.mutex);
        gate.entered = true; gate.condition.notify_all();
        gate.condition.wait(lock, [&] { return gate.released; });
    }
};

// Public RMM hook: output storage is allocated normally and gates its stream;
// the next allocation (GPU Encode Scratch.raw) fails while that stream is busy.
class FailingAllocationResource final : public rmm::mr::device_memory_resource {
public:
    FailingAllocationResource(rmm::mr::device_memory_resource *upstream, AllocationGate &gate)
        : upstream_(upstream), gate_(gate) {}
    std::size_t calls() const { return calls_; }
    std::size_t failures() const { return failures_; }
    std::size_t live_bytes() const { return live_bytes_; }
    void arm() { inject_ = true; }
    void disable_failure() { inject_ = false; }
private:
    void *do_allocate(std::size_t bytes, rmm::cuda_stream_view stream) override {
        const auto call = inject_ ? ++calls_ : 0;
        if (inject_ && call == 2) {
            { std::lock_guard<std::mutex> lock(gate_.mutex);
              require(gate_.entered && !gate_.released, "allocation failed before producer stream was in flight");
              gate_.failed = true; gate_.condition.notify_all(); }
            ++failures_;
            throw std::bad_alloc{};
        }
        auto *pointer = upstream_->allocate(bytes, stream);
        live_bytes_ += bytes;
        if (inject_ && call == 1) {
            check(cudaLaunchHostFunc(stream.value(), AllocationGate::wait, &gate_), "gate encoder allocation stream");
            std::unique_lock<std::mutex> lock(gate_.mutex);
            require(gate_.condition.wait_for(lock, std::chrono::seconds(1), [&] { return gate_.entered; }),
                    "encoder allocation stream gate never entered");
        }
        return pointer;
    }
    void do_deallocate(void *pointer, std::size_t bytes, rmm::cuda_stream_view stream) override {
        upstream_->deallocate(pointer, bytes, stream);
        live_bytes_ -= bytes;
    }
    bool do_is_equal(const rmm::mr::device_memory_resource &other) const noexcept override {
        return this == &other;
    }
    rmm::mr::device_memory_resource *upstream_;
    AllocationGate &gate_;
    std::atomic<std::size_t> calls_{0}, failures_{0}, live_bytes_{0};
    bool inject_ = false;
};

class ObservedApi : public PoseidonGpuApi {
public:
    using PoseidonGpuApi::PoseidonGpuApi;
    std::function<void()> before_prepare;
    std::atomic<unsigned> gpu_attempts{0}, cpu_attempts{0}, compute_calls{0}, drain_calls{0};
    Value prepare_plaintext_async(const fhegpu::ValueDesc &desc, const std::vector<double> &slots,
                                 fhegpu::PrepareBackend backend) {
        if (backend == fhegpu::PrepareBackend::Gpu) ++gpu_attempts;
        else if (backend == fhegpu::PrepareBackend::Cpu) ++cpu_attempts;
        if (before_prepare) before_prepare();
        return PoseidonGpuApi::prepare_plaintext_async(desc, slots, backend);
    }
    Value compute(const fhegpu::ComputeOp &op, const std::vector<Value> &inputs) {
        ++compute_calls; return PoseidonGpuApi::compute(op, inputs);
    }
    void drain() { ++drain_calls; PoseidonGpuApi::drain(); }
};

fhegpu::LoadedOperatorSpec make_spec(const poseidon::PoseidonContext &context) {
    fhegpu::OperatorSpec spec;
    spec.id = "gpu-prepare-oom"; spec.version = 1; spec.status = "test";
    spec.target_id = "poseidon-ckks-gpu"; spec.context_id = context_id;
    spec.rescale_mode = fhegpu::RescaleMode::Lazy;
    spec.poly_degree = context.parameters_literal()->degree();
    for (const auto &q : context.parameters_literal()->q()) spec.rns_moduli_log2.push_back(q.bit_count());
    spec.max_modulus_log2 = *std::max_element(spec.rns_moduli_log2.begin(), spec.rns_moduli_log2.end());
    spec.default_scale_log2 = 40; spec.level_lower_bound = 0; spec.level_upper_bound = 6;
    fhegpu::OperatorSupport support; support.supported = true;
    spec.operators.emplace(fhegpu::ComputeKind::AddCP, support);
    support.max_levels_per_op = 4;
    spec.operators.emplace(fhegpu::ComputeKind::Rescale, support);
    return {std::move(spec), spec_sha};
}

fhegpu::RuntimePlan make_plan(const fhegpu::LoadedOperatorSpec &spec, const std::vector<double> &weights) {
    fhegpu::RuntimePlan plan; plan.format_version = 4; plan.plan_id = 491;
    plan.target.target_id = "poseidon-ckks-gpu"; plan.target.world_size = 1;
    plan.target.device_counts = {1}; plan.target.capability_version = 1;
    plan.target.operator_spec = {spec.spec.id, spec.spec.version, spec.source_sha256};
    const fhegpu::Place device{fhegpu::PlaceKind::Device, 0, 0};
    const fhegpu::Place host{fhegpu::PlaceKind::Host, 0, 0};
    plan.values = {{0, fhegpu::ValueKind::Ciphertext, host, context_id, 6, 40, true, 2},
                   {1, fhegpu::ValueKind::Ciphertext, device, context_id, 6, 40, true, 2},
                   {2, fhegpu::ValueKind::Plaintext, device, context_id, 6, 40, true, 1},
                   {3, fhegpu::ValueKind::Ciphertext, device, context_id, 6, 40, true, 2},
                   {4, fhegpu::ValueKind::Ciphertext, host, context_id, 6, 40, true, 2}};
    plan.external_inputs = {0}; plan.final_outputs = {4};
    fhegpu::CommAction upload{1, fhegpu::CommKind::Transfer, fhegpu::CommHint::PointToPoint,
                              {0}, {1}, {host}, {device}, {fhegpu::ValueKind::Ciphertext}};
    fhegpu::CommAction download{2, fhegpu::CommKind::Transfer, fhegpu::CommHint::PointToPoint,
                                {3}, {4}, {device}, {host}, {fhegpu::ValueKind::Ciphertext}};
    plan.initialization = {{0, upload}};
    plan.execution = {{1, fhegpu::PreparePlaintextOp{fhegpu::InlineEncodePayload{weights}, 2}},
                      {2, fhegpu::ComputeOp{fhegpu::ComputeKind::AddCP, {1, 2}, 3, device, {}}},
                      {3, fhegpu::ReleaseOp{1}}, {4, fhegpu::ReleaseOp{2}}};
    plan.finalization = {{5, download}, {6, fhegpu::ReleaseOp{3}}};
    fhegpu::WaterlineConfig config; config.backend = fhegpu::PrepareBackend::Gpu;
    config.device_capacity_bytes = {1ULL << 30}; config.device_reserve_bytes = {256ULL << 20};
    config.device_fractions = {.25}; config.host_budget_bytes = 48ULL << 20;
    config.pinned_budget_bytes = 8ULL << 20; config.raw_budget_bytes = 2ULL << 20;
    config.lookahead_groups = 2; plan.waterline = config;
    return plan;
}
}

int main() {
    try {
        setenv("POSEIDON_GPU_RUNTIME_MAXIMUM_POOL_MB", "512", 1);
        // The fixed resources plus the 64 MiB margin must already be reserved
        // before the intended second-allocation failure is injected.
        setenv("POSEIDON_GPU_RUNTIME_INITIAL_POOL_MB", "128", 1);
        int count = 0; check(cudaGetDeviceCount(&count), "OOM test CUDA devices");
        require(count == 1, "OOM test requires one explicitly selected GPU");
        check(cudaSetDevice(0), "OOM test device");
        poseidon::ParametersLiteral literal(CKKS, 12, 11, 40, 0, 0, poseidon::Modulus(0), {}, {},
                                            poseidon::sec_level_type::none);
        literal.set_log_modulus(std::vector<std::uint32_t>(7, 30), std::vector<std::uint32_t>(2, 30));
        poseidon::PoseidonContext context(literal);
        poseidon::KeyGenerator generator(context);
        poseidon::PublicKey public_key; generator.create_public_key(public_key);
        poseidon::CKKSEncoder encoder(context);
        poseidon::Encryptor encryptor(context, public_key);
        poseidon::Decryptor decryptor(context, generator.secret_key());
        ObservedApi api(context_id, context, 0);
        const auto spec = make_spec(context);
        std::vector<double> input(encoder.slot_count()), weights(encoder.slot_count());
        for (std::size_t i = 0; i < input.size(); ++i) {
            input[i] = .25 * std::sin(i * .071); weights[i] = .125 * std::cos(i * .037);
        }
        poseidon::Plaintext plain; encoder.encode(input, std::ldexp(1., 40), plain);
        poseidon::Ciphertext cipher; encryptor.encrypt(plain, cipher);
        const std::unordered_map<fhegpu::ValueId, PoseidonGpuValue> inputs{
            {0, PoseidonGpuValue::from_host_ciphertext(cipher)}};
        const auto plan = make_plan(spec, weights);

        // A complete normal run warms preflight tables, encoder, and producer.
        // The injected run must pass the same complete resource precheck.
        {
            fhegpu::SequentialRuntime<ObservedApi> warm(0, 1, 1, api, fhegpu::DeviceExecutionMode::PerDeviceWorkers);
            auto artifact = warm.run({plan, plan_sha}, {spec, std::nullopt, false}, inputs);
        }
        api.collect_completed(); api.drain();
        require(Pinned::live_bytes() == 0, "warm run retained pinned storage");
        const auto baseline = api.waterline_measurements().devices.at(0).allocated_live;
        const auto gpu_before = api.gpu_attempts.load(), cpu_before = api.cpu_attempts.load();
        const auto compute_before = api.compute_calls.load(), drain_before = api.drain_calls.load();
        const auto pinned_allocations = Pinned::allocation_calls(), pinned_frees = Pinned::free_calls();
        auto *upstream = rmm::mr::get_current_device_resource();
        AllocationGate gate;
        FailingAllocationResource failing(upstream, gate);
        struct RestoreResource {
            rmm::mr::device_memory_resource *resource;
            ~RestoreResource() { rmm::mr::set_current_device_resource(resource); }
        } restore{upstream};
        rmm::mr::set_current_device_resource(&failing);
        api.before_prepare = [&] { failing.arm(); };
        std::string failure;
        fhegpu::SequentialRuntime<ObservedApi> runtime(0, 1, 1, api, fhegpu::DeviceExecutionMode::PerDeviceWorkers);
        {
            try { (void)runtime.run({plan, plan_sha}, {spec, std::nullopt, false}, inputs); }
            catch (const std::exception &error) { failure = error.what(); }
            gate.run_finished = true;
            require(failure.find("bad_alloc") != std::string::npos, "real allocation failure was not propagated: " + failure);
            require(api.gpu_attempts - gpu_before == 1 && api.cpu_attempts == cpu_before,
                    "failed GPU Prepare silently retried or fell back to CPU");
            require(api.compute_calls == compute_before, "consumer ran after failed Prepare");
            require(failing.calls() == 2 && failing.failures() == 1, "failure did not occur on the second real allocation");
            { std::lock_guard<std::mutex> lock(gate.mutex);
              require(gate.entered && gate.failed && gate.released && !gate.timed_out && gate.observed_unfinished_run,
                      "producer gate did not prove safe waiting during an in-flight allocation failure"); }
            const auto after_failure = api.waterline_measurements().devices.at(0).allocated_live;
            std::cout << "OOM propagated=true gpu_attempts=1 cpu_attempts=0 consumer_calls=0 real_allocation_calls=2"
                      << " allocator_failures=1 stream_gate_inflight=true run_waited_for_gate=true"
                      << " fixed_bytes=" << baseline
                      << " allocated_live_after_failed_run=" << after_failure
                      << " injected_dynamic_live_after_failed_run=" << failing.live_bytes()
                      << " pinned_live_after_failed_run=" << Pinned::live_bytes()
                      << " automatic_drain_calls=" << api.drain_calls - drain_before << '\n';
            require(api.drain_calls - drain_before == 1, "failed run did not automatically drain exactly once");
            require(failing.live_bytes() == 0, "failed run returned while retaining initial GPU input or dynamic allocations");
            require(after_failure == baseline, "failed run returned with RMM memory beyond fixed resources");
            require(Pinned::live_bytes() == 0 && Pinned::allocation_calls() - pinned_allocations ==
                    Pinned::free_calls() - pinned_frees, "failed run returned while retaining pinned source storage");
        }

        // Keep the failed runtime alive: only this explicit next run may retry.
        api.before_prepare = {};
        failing.disable_failure();
        double maximum_error = 0;
        {
            auto artifact = runtime.run({plan, plan_sha}, {spec, std::nullopt, false}, inputs);
            const auto &actual = artifact.values.at(4).value.host_ciphertext();
            poseidon::Plaintext decrypted; decryptor.decrypt(actual, decrypted);
            std::vector<std::complex<double>> slots; encoder.decode(decrypted, slots);
            require(slots.size() == input.size(), "explicit next run changed slot layout");
            for (std::size_t i = 0; i < slots.size(); ++i) {
                const auto error = std::abs(slots[i] - std::complex<double>(input[i] + weights[i], 0));
                require(std::isfinite(error) && error < 1e-4, "explicit next run failed independent dense numerical check");
                maximum_error = std::max(maximum_error, error);
            }
        }
        api.drain(); api.collect_completed();
        require(failing.live_bytes() == 0 && Pinned::live_bytes() == 0,
                "explicit next run retained dynamic or pinned storage");
        require(api.waterline_measurements().devices.at(0).allocated_live == baseline,
                "explicit next run leaked RMM memory beyond fixed resources");
        require(api.gpu_attempts - gpu_before == 2 && api.cpu_attempts == cpu_before,
                "next run made extra Prepare attempts or CPU fallback");
        std::cout << "[PASS] real CUDA GPU Prepare allocation failure: propagated=true automatic_drain=true"
                  << " dynamic_live=0 pinned_live=0 implicit_retry=false cpu_fallback=false"
                  << " runtime_kept_alive=true explicit_same_runtime_next_run=true complex_slots=" << input.size()
                  << " max_abs_error=" << maximum_error << '\n';
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "[FAIL] real CUDA GPU Prepare allocation failure: " << error.what() << '\n'; return 1;
    }
}
