#include "poseidon/ckks_encoder.h"
#include "poseidon/decryptor.h"
#include "poseidon/encryptor.h"
#include "poseidon/evaluator/software/evaluator_ckks_software.h"
#include "poseidon/gpu/gpu_uploader.h"
#include "poseidon/gpu/gpu_evaluator.h"
#include "poseidon/keygenerator.h"
#include "poseidon/runtime_api/poseidon_gpu_api.h"
#include "poseidon/runtime_api/communication/cuda_local_transfer.h"
#include "runtime/runtime.hpp"
#include "runtime/utils/sha256.hpp"

#include <cuda_runtime_api.h>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/limiting_resource_adaptor.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <rmm/mr/device/statistics_resource_adaptor.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <condition_variable>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace
{

using poseidon::runtime_api::PoseidonGpuApi;
using poseidon::runtime_api::PoseidonGpuValue;

constexpr int kDeviceId = 0;
int kDefaultScaleLog2 = 80;
const std::string kContextId = "poseidon-gpu-api-test-context";
const std::string kOperatorSpecSha =
    "sha256:0000000000000000000000000000000000000000000000000000000000000000";
const std::string kPlanSha =
    "sha256:1111111111111111111111111111111111111111111111111111111111111111";

int tests_run = 0;

struct CudaHostGate
{
    std::mutex mutex;
    std::condition_variable condition;
    bool entered = false;
    bool release = false;
    bool post_finished = false;
    bool watchdog_released = false;
};

void CUDART_CB wait_for_cuda_host_gate(void *opaque)
{
    auto &gate = *static_cast<CudaHostGate *>(opaque);
    std::unique_lock<std::mutex> lock(gate.mutex);
    gate.entered = true;
    gate.condition.notify_all();
    gate.condition.wait(lock, [&] { return gate.release; });
}

class CudaGateWatchdog
{
public:
    explicit CudaGateWatchdog(CudaHostGate &gate)
        : gate_(gate), thread_([this] {
              std::unique_lock<std::mutex> lock(gate_.mutex);
              if (!gate_.condition.wait_for(
                      lock, std::chrono::seconds(2),
                      [&] { return gate_.post_finished; }))
              {
                  gate_.watchdog_released = true;
              }
              gate_.release = true;
              gate_.condition.notify_all();
          })
    {}

    ~CudaGateWatchdog()
    {
        {
            std::lock_guard<std::mutex> lock(gate_.mutex);
            gate_.release = true;
            gate_.condition.notify_all();
        }
        if (thread_.joinable())
        {
            thread_.join();
        }
    }

    bool finish_post()
    {
        std::lock_guard<std::mutex> lock(gate_.mutex);
        const bool returned_before_release = !gate_.release;
        gate_.post_finished = true;
        gate_.condition.notify_all();
        return returned_before_release && !gate_.watchdog_released;
    }

private:
    CudaHostGate &gate_;
    std::thread thread_;
};

void require(bool condition, const std::string &message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}

void run_test(const char *name, const std::function<void()> &test)
{
    test();
    ++tests_run;
    std::cout << "[PASS] " << name << '\n';
}

template <class Function>
void require_rejected(Function &&function, const std::string &needle)
{
    try
    {
        function();
    }
    catch (const std::exception &error)
    {
        require(std::string(error.what()).find(needle) != std::string::npos,
                "unexpected rejection: " + std::string(error.what()));
        return;
    }
    throw std::runtime_error("expected rejection containing: " + needle);
}

class RmmPoolScope
{
public:
    explicit RmmPoolScope(int device_id)
        : device_id_(select_device(device_id)), pool_(&upstream_, 1 << 20, std::nullopt),
          accounting_(&pool_, std::numeric_limits<std::size_t>::max()), statistics_(&accounting_)
    {
        previous_ = rmm::mr::get_current_device_resource();
        rmm::mr::set_current_device_resource(&statistics_);
    }

    RmmPoolScope(const RmmPoolScope &) = delete;
    RmmPoolScope &operator=(const RmmPoolScope &) = delete;

    std::size_t allocated_bytes() const noexcept
    {
        return accounting_.get_allocated_bytes();
    }

    std::size_t total_allocated_bytes() const noexcept
    {
        return statistics_.get_bytes_counter().total;
    }

    ~RmmPoolScope()
    {
        (void)cudaSetDevice(device_id_);
        rmm::mr::set_current_device_resource(previous_);
    }

private:
    static int select_device(int device_id)
    {
        const cudaError_t status = cudaSetDevice(device_id);
        if (status != cudaSuccess)
            throw std::runtime_error(std::string("cudaSetDevice failed: ") +
                                     cudaGetErrorString(status));
        return device_id;
    }

    int device_id_ = 0;
    rmm::mr::cuda_memory_resource upstream_;
    rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> pool_;
    rmm::mr::limiting_resource_adaptor<rmm::mr::device_memory_resource> accounting_;
    rmm::mr::statistics_resource_adaptor<rmm::mr::device_memory_resource> statistics_;
    rmm::mr::device_memory_resource *previous_ = nullptr;
};

// Inject a gate into the actual stream supplied by a public RMM allocation.
// The warmed GPU encoder allocates its output before submitting H2D/FFT/NTT.
class GatedAllocationResource final : public rmm::mr::device_memory_resource
{
public:
    GatedAllocationResource(rmm::mr::device_memory_resource *upstream,CudaHostGate &gate)
        :upstream_(upstream),gate_(gate) {}
    cudaStream_t injected_stream() const {return injected_stream_;}
private:
    void *do_allocate(std::size_t bytes,rmm::cuda_stream_view stream) override {
        auto *pointer=upstream_->allocate(bytes,stream);
        if(!injected_) {
            injected_=true;
            injected_stream_=stream.value();
            require(cudaLaunchHostFunc(stream.value(),wait_for_cuda_host_gate,&gate_)==cudaSuccess,
                    "failed to gate cached GPU Prepare allocation stream");
        }
        return pointer;
    }
    void do_deallocate(void *pointer,std::size_t bytes,rmm::cuda_stream_view stream) override {
        upstream_->deallocate(pointer,bytes,stream);
    }
    bool do_is_equal(const rmm::mr::device_memory_resource &other) const noexcept override {return this==&other;}
    rmm::mr::device_memory_resource *upstream_;
    CudaHostGate &gate_;
    bool injected_=false;
    cudaStream_t injected_stream_=nullptr;
};

poseidon::ParametersLiteral make_parameters()
{
    poseidon::ParametersLiteral parameters(
        CKKS,
        /*log_n=*/std::getenv("POSEIDON_WATERLINE_LOG_N") ? std::stoi(std::getenv("POSEIDON_WATERLINE_LOG_N")) : 12,
        /*log_slots=*/std::getenv("POSEIDON_WATERLINE_LOG_N") ? std::stoi(std::getenv("POSEIDON_WATERLINE_LOG_N"))-1 : 11,
        /*log_scale=*/kDefaultScaleLog2,
        /*hamming_weight=*/0,
        /*q0_level=*/0,
        poseidon::Modulus(0),
        {},
        {},
        poseidon::sec_level_type::none);
    parameters.set_log_modulus(std::vector<std::uint32_t>(std::getenv("POSEIDON_WATERLINE_Q_COUNT") ? std::stoi(std::getenv("POSEIDON_WATERLINE_Q_COUNT")) : 10, 30),
                               std::vector<std::uint32_t>(2, 30));
    return parameters;
}

void test_internal_memory_pool_lifetime(
    const poseidon::PoseidonContext &context)
{
    require(cudaSetDevice(kDeviceId) == cudaSuccess,
            "failed to select CUDA device for memory-pool lifetime test");
    auto *previous_resource = rmm::mr::get_current_device_resource();
    rmm::mr::device_memory_resource *pool_resource = nullptr;
    std::optional<PoseidonGpuValue> escaped_value;

    {
        PoseidonGpuApi api(kContextId, context, kDeviceId);
        pool_resource = rmm::mr::get_current_device_resource();
        require(pool_resource != previous_resource,
                "Poseidon GPU Api did not install its internal RMM pool");

        auto device_plaintext =
            poseidon::gpu::GpuPlaintextData::allocate_single_device(
                16, 1, kDeviceId);
        escaped_value.emplace(PoseidonGpuValue::from_device_plaintext(
            std::move(device_plaintext)));
    }

    require(rmm::mr::get_current_device_resource() == pool_resource,
            "internal RMM pool did not outlive an allocated GPU value");
    *escaped_value =
        PoseidonGpuValue::from_host_plaintext(poseidon::Plaintext{});
    require(rmm::mr::get_current_device_resource() == previous_resource,
            "internal RMM pool did not restore the previous resource");
}

fhegpu::LoadedOperatorSpec make_operator_spec(const poseidon::PoseidonContext &context)
{
    fhegpu::OperatorSpec spec;
    spec.id = "poseidon-gpu-api-test";
    spec.version = 1;
    spec.status = "test";
    spec.target_id = "poseidon-ckks-gpu";
    spec.rescale_mode = fhegpu::RescaleMode::Lazy;
    spec.context_id = kContextId;

    const auto parameters = context.parameters_literal();
    spec.poly_degree = parameters->degree();
    for (const auto &modulus : parameters->q())
    {
        spec.rns_moduli_log2.push_back(modulus.bit_count());
    }
    spec.max_modulus_log2 =
        *std::max_element(spec.rns_moduli_log2.begin(), spec.rns_moduli_log2.end());
    spec.default_scale_log2 = static_cast<int>(parameters->log_scale());
    spec.level_lower_bound = 0;
    spec.level_upper_bound = static_cast<int>(parameters->q().size() - 1);

    for (const auto kind : {fhegpu::ComputeKind::AddCC, fhegpu::ComputeKind::AddCP,
                            fhegpu::ComputeKind::SubCC, fhegpu::ComputeKind::SubCP,
                            fhegpu::ComputeKind::MulCC, fhegpu::ComputeKind::MulCP,
                            fhegpu::ComputeKind::Negate, fhegpu::ComputeKind::Rotate,
                            fhegpu::ComputeKind::Relinearize})
    {
        fhegpu::OperatorSupport support;
        support.supported = true;
        spec.operators.emplace(kind, std::move(support));
    }

    fhegpu::OperatorSupport rescale;
    rescale.supported = true;
    rescale.max_levels_per_op = 8;
    spec.operators.emplace(fhegpu::ComputeKind::Rescale, std::move(rescale));

    fhegpu::OperatorSupport unsupported;
    unsupported.supported = false;
    fhegpu::OperatorSupport mod_switch;
    mod_switch.supported = true;
    spec.operators.emplace(fhegpu::ComputeKind::ModSwitch, std::move(mod_switch));
    spec.operators.emplace(fhegpu::ComputeKind::Boot, std::move(unsupported));

    return {std::move(spec), kOperatorSpecSha};
}

fhegpu::LoadedOperatorSpec make_boot_operator_spec(
    const fhegpu::LoadedOperatorSpec &loaded_spec)
{
    auto result = loaded_spec;
    result.spec.operators.at(fhegpu::ComputeKind::Boot).supported = true;
    fhegpu::BootProfile profile;
    profile.profile_id = "poseidon-gpu-host-boot-test";
    profile.implementation = fhegpu::BootImplementation::DecryptReencrypt;
    profile.input_level_min = result.spec.level_lower_bound;
    profile.input_level_max = result.spec.level_upper_bound;
    profile.input_components = 2;
    profile.output_level = result.spec.level_upper_bound;
    profile.output_scale_log2 = result.spec.default_scale_log2;
    profile.output_components = 2;
    profile.needs_secret_key = true;
    profile.needs_host_compute = true;
    result.spec.boot_profiles = {std::move(profile)};
    return result;
}

fhegpu::TargetConfig make_target(const fhegpu::LoadedOperatorSpec &loaded_spec,
                                 int local_device_count = 1)
{
    fhegpu::TargetConfig target;
    target.target_id = "poseidon-ckks-gpu";
    target.world_size = 1;
    target.device_counts = {local_device_count};
    target.capability_version = 1;
    target.operator_spec =
        {loaded_spec.spec.id, loaded_spec.spec.version, loaded_spec.source_sha256};
    return target;
}

void test_waterline_reserved_pool_capacity(const poseidon::PoseidonContext &context,
                                           const fhegpu::LoadedOperatorSpec &spec)
{
    struct RestoreEnvironment {
        std::optional<std::string> initial,maximum;
        RestoreEnvironment() {
            if(const auto *value=std::getenv("POSEIDON_GPU_RUNTIME_INITIAL_POOL_MB"))initial=value;
            if(const auto *value=std::getenv("POSEIDON_GPU_RUNTIME_MAXIMUM_POOL_MB"))maximum=value;
        }
        ~RestoreEnvironment() {
            if(initial)setenv("POSEIDON_GPU_RUNTIME_INITIAL_POOL_MB",initial->c_str(),1);
            else unsetenv("POSEIDON_GPU_RUNTIME_INITIAL_POOL_MB");
            if(maximum)setenv("POSEIDON_GPU_RUNTIME_MAXIMUM_POOL_MB",maximum->c_str(),1);
            else unsetenv("POSEIDON_GPU_RUNTIME_MAXIMUM_POOL_MB");
        }
    } restore;
    setenv("POSEIDON_GPU_RUNTIME_MAXIMUM_POOL_MB","256",1);
    for(bool small_initial:{false,true}) {
        if(small_initial)setenv("POSEIDON_GPU_RUNTIME_INITIAL_POOL_MB","128",1);
        else unsetenv("POSEIDON_GPU_RUNTIME_INITIAL_POOL_MB");
        PoseidonGpuApi api(kContextId,context,kDeviceId);
        fhegpu::RuntimePlan plan;plan.format_version=4;plan.target=make_target(spec);
        fhegpu::WaterlineConfig config;config.backend=fhegpu::PrepareBackend::Cpu;
        config.device_capacity_bytes={4ULL<<30};config.device_reserve_bytes={2ULL<<30};
        config.device_fractions={.5};config.host_budget_bytes=64ULL<<20;
        config.pinned_budget_bytes=16ULL<<20;config.raw_budget_bytes=8ULL<<20;plan.waterline=config;
        api.validate_waterline_resources(plan);
        const auto measured=api.waterline_measurements().devices.at(0);
        const auto expected_reserved=(small_initial?128ULL:256ULL)<<20;
        require(measured.managed_pool && measured.pool_capacity==256ULL<<20 &&
                    measured.pool_reserved==expected_reserved,
                "managed pool did not physically reserve its default maximum or explicit initial size");
        // Empty plan has no retained workspace: only setup allocations and the
        // existing 64 MiB margin are removed from actual reserved pool bytes.
        require(measured.pool_reserved>measured.allocated_live+(64ULL<<20),
                "reserved pool test context does not fit its fixed setup margin");
        const auto expected_graph=measured.pool_reserved-measured.allocated_live-(64ULL<<20);
        require(api.waterline_allocator_capacity().at(0)==expected_graph,
                "Waterline counted unreserved future pool growth as available graph capacity");
        std::cout<<"WATERLINE_RESERVED_POOL explicit_initial="<<small_initial
                 <<" reserved="<<measured.pool_reserved<<" maximum="<<measured.pool_capacity
                 <<" graph_capacity="<<expected_graph<<'\n';
    }
}

fhegpu::Place host_place()
{
    return {fhegpu::PlaceKind::Host, 0, 0};
}

fhegpu::Place device_place(int index = 0)
{
    return {fhegpu::PlaceKind::Device, 0, index};
}

fhegpu::CommAction transfer_action(fhegpu::TransferId id, fhegpu::ValueKind kind,
                                   const fhegpu::Place &source,
                                   const fhegpu::Place &destination)
{
    fhegpu::CommAction action;
    action.id = id;
    action.kind = fhegpu::CommKind::Transfer;
    action.inputs = {0};
    action.outputs = {1};
    action.sources = {source};
    action.destinations = {destination};
    action.output_types = {kind};
    return action;
}

fhegpu::CommAction replicate_action(
    fhegpu::TransferId id, fhegpu::ValueKind kind, const fhegpu::Place &source,
    std::vector<fhegpu::Place> destinations, std::vector<fhegpu::ValueId> outputs)
{
    fhegpu::CommAction action;
    action.id = id;
    action.kind = fhegpu::CommKind::Replicate;
    action.hint = fhegpu::CommHint::Broadcast;
    action.inputs = {0};
    action.outputs = std::move(outputs);
    action.sources = {source};
    action.destinations = std::move(destinations);
    action.output_types.assign(action.outputs.size(), kind);
    return action;
}

PoseidonGpuValue transfer_value(PoseidonGpuApi &api, fhegpu::TransferId id,
                                const PoseidonGpuValue &input, fhegpu::ValueKind kind,
                                const fhegpu::Place &source,
                                const fhegpu::Place &destination)
{
    if (kind != fhegpu::ValueKind::Ciphertext)
    {
        throw std::invalid_argument(
            "test transfer descriptor helper supports ciphertext only");
    }
    int level = 0;
    int scale_log2 = 0;
    bool ntt = true;
    int components = 2;
    if (input.place_kind() == fhegpu::PlaceKind::Host)
    {
        const auto &cipher = input.host_ciphertext();
        level = static_cast<int>(cipher.level());
        scale_log2 = static_cast<int>(std::llround(std::log2(cipher.scale())));
        ntt = cipher.is_ntt_form();
        components = static_cast<int>(cipher.size());
    }
    else
    {
        const auto &cipher = input.device_ciphertext();
        level = static_cast<int>(cipher.meta.q_count - 1);
        scale_log2 =
            static_cast<int>(std::llround(std::log2(cipher.meta.scale)));
        ntt = cipher.meta.is_ntt_form;
        components = static_cast<int>(cipher.meta.component_count);
    }
    const fhegpu::ValueDesc output_desc{
        1, kind, destination, kContextId, level, scale_log2, ntt, components};
    auto handle = api.communicate_async(
        transfer_action(id, kind, source, destination), {input}, {output_desc});
    auto outputs = api.wait(handle);
    require(outputs.size() == 1, "Transfer returned the wrong output count");
    require_rejected([&] { (void)api.wait(handle); }, "already waited");
    require(outputs.front().has_value(), "Transfer omitted its output");
    return std::move(*outputs.front());
}

void test_transfer_posts_completion_event(
    const poseidon::PoseidonContext &context,
    poseidon::KeyGenerator &key_generator)
{
    PoseidonGpuApi api(kContextId, context, kDeviceId);
    poseidon::PublicKey public_key;
    key_generator.create_public_key(public_key);
    poseidon::Encryptor encryptor(context, public_key);
    poseidon::CKKSEncoder encoder(context);

    poseidon::Plaintext plain;
    const std::vector<double> values{1.0, -2.0, 3.0, 4.0};
    encoder.encode(values,
                   std::ldexp(1.0, kDefaultScaleLog2), plain);
    poseidon::Ciphertext cipher;
    encryptor.encrypt(plain, cipher);
    const int level = static_cast<int>(cipher.level());
    const int components = static_cast<int>(cipher.size());
    const fhegpu::ValueDesc output_desc{
        1, fhegpu::ValueKind::Ciphertext, device_place(), kContextId,
        level, kDefaultScaleLog2, cipher.is_ntt_form(), components};

    auto warmup_device = transfer_value(
        api, 39, PoseidonGpuValue::from_host_ciphertext(cipher),
        fhegpu::ValueKind::Ciphertext, host_place(), device_place());
    fhegpu::ComputeOp negate;
    negate.kind = fhegpu::ComputeKind::Negate;
    negate.inputs = {1};
    negate.output = 2;
    negate.place = device_place();
    auto warmup = api.compute(negate, {warmup_device});
    api.synchronize(warmup);

    CudaHostGate gate;
    require(cudaSetDevice(kDeviceId) == cudaSuccess,
            "failed to select CUDA device for asynchronous upload test");
    require(cudaLaunchHostFunc(
                cudaStreamPerThread, wait_for_cuda_host_gate, &gate) ==
                cudaSuccess,
            "failed to enqueue CUDA upload gate");
    {
        std::unique_lock<std::mutex> lock(gate.mutex);
        require(gate.condition.wait_for(
                    lock, std::chrono::seconds(5),
                    [&] { return gate.entered; }),
                "CUDA upload gate did not start");
    }

    CudaGateWatchdog watchdog(gate);
    auto handle = api.communicate_async(
        transfer_action(40, fhegpu::ValueKind::Ciphertext, host_place(),
                        device_place()),
        {PoseidonGpuValue::from_host_ciphertext(std::move(cipher))},
        {output_desc});
    auto posted = api.posted_outputs(handle);
    require(posted.size() == 1 && posted.front().has_value(),
            "H2D transfer did not publish its device output");
    auto second_post = api.posted_outputs(handle);
    require(second_post.size() == 1 && !second_post.front(),
            "H2D output was published twice");

    auto consumed = api.compute(negate, {*posted.front()});
    require(watchdog.finish_post(),
            "posted H2D output or consumer compute waited for CUDA completion");
    api.synchronize(consumed);

    auto completed = api.wait(handle);
    require(completed.size() == 1 && !completed.front(),
            "wait delivered the already posted H2D output again");

    const fhegpu::ValueDesc host_desc{
        1, fhegpu::ValueKind::Ciphertext, host_place(), kContextId,
        level, kDefaultScaleLog2, true, components};
    auto download = api.communicate_async(
        transfer_action(41, fhegpu::ValueKind::Ciphertext, device_place(),
                        host_place()),
        {consumed}, {host_desc});
    auto host_posted = api.posted_outputs(download);
    require(host_posted.size() == 1 && !host_posted.front(),
            "D2H transfer published Host data before completion");
    auto host_completed = api.wait(download);
    require(host_completed.size() == 1 && host_completed.front(),
            "completed D2H transfer returned the wrong output count");
}


void test_async_memory_cleanup(const poseidon::PoseidonContext &context,
                               poseidon::KeyGenerator &key_generator,
                               const RmmPoolScope &pool)
{
    PoseidonGpuApi api(kContextId, context, kDeviceId);
    poseidon::PublicKey public_key;
    key_generator.create_public_key(public_key);
    poseidon::Encryptor encryptor(context, public_key);
    poseidon::CKKSEncoder encoder(context);
    poseidon::Plaintext plain;
    encoder.encode(std::vector<double>{1.0, -2.0},
                   std::ldexp(1.0, kDefaultScaleLog2), plain);
    poseidon::Ciphertext cipher;
    encryptor.encrypt(plain, cipher);
    fhegpu::ComputeOp negate;
    negate.kind = fhegpu::ComputeKind::Negate;
    negate.place = device_place();
    negate.inputs = {1};
    negate.output = 2;
    // Materialize evaluator tables before comparing live object bytes.
    {
        auto input = transfer_value(api, 80,
            PoseidonGpuValue::from_host_ciphertext(cipher),
            fhegpu::ValueKind::Ciphertext, host_place(), device_place());
        auto output = api.compute(negate, {input});
        api.drain();
    }
    api.collect_completed();
    const auto baseline = pool.allocated_bytes();
    for (int iteration = 0; iteration < 3; ++iteration)
    {
        std::optional<PoseidonGpuValue> input(transfer_value(api, 81 + iteration,
            PoseidonGpuValue::from_host_ciphertext(cipher),
            fhegpu::ValueKind::Ciphertext, host_place(), device_place()));
        const auto input_bytes = pool.allocated_bytes();
        CudaHostGate gate;
        require(cudaSetDevice(kDeviceId) == cudaSuccess, "cudaSetDevice failed");
        require(cudaLaunchHostFunc(cudaStreamPerThread, wait_for_cuda_host_gate,
                                   &gate) == cudaSuccess, "compute gate failed");
        CudaGateWatchdog watchdog(gate);
        std::optional<PoseidonGpuValue> output(api.compute(negate, {*input}));
        const auto submitted_bytes = pool.allocated_bytes();
        require(submitted_bytes > input_bytes, "Negate did not allocate an output");
        input.reset();
        output.reset();
        api.collect_completed();
        require(pool.allocated_bytes() == submitted_bytes,
                "unfinished computation lost its input or output storage");
        require(watchdog.finish_post(), "collect_completed waited for computation");
        api.drain();
        require(pool.allocated_bytes() == baseline,
                "completed computation still retains GPU object storage");
    }
    // A posted transfer result may be discarded while its copy is pending.
    const fhegpu::ValueDesc desc{
        1, fhegpu::ValueKind::Ciphertext, device_place(), kContextId,
        static_cast<int>(cipher.level()), kDefaultScaleLog2,
        cipher.is_ntt_form(), static_cast<int>(cipher.size())};
    CudaHostGate gate;
    require(cudaSetDevice(kDeviceId) == cudaSuccess, "cudaSetDevice failed");
    require(cudaLaunchHostFunc(cudaStreamPerThread, wait_for_cuda_host_gate,
                               &gate) == cudaSuccess, "transfer gate failed");
    CudaGateWatchdog watchdog(gate);
    auto handle = api.communicate_async(
        transfer_action(85, fhegpu::ValueKind::Ciphertext, host_place(), device_place()),
        {PoseidonGpuValue::from_host_ciphertext(cipher)}, {desc});
    auto posted = api.posted_outputs(handle);
    require(posted.front().has_value(), "upload was not posted");
    const auto pending_bytes = pool.allocated_bytes();
    posted.clear();
    api.collect_completed();
    require(pool.allocated_bytes() == pending_bytes && pending_bytes > baseline,
            "unfinished transfer lost its output storage");
    require(watchdog.finish_post(), "collect_completed waited for transfer");
    auto completed = api.wait(handle);
    require(completed.size() == 1 && !completed.front(), "upload was delivered twice");
    require(pool.allocated_bytes() == baseline,
            "communication handle retains a delivered GPU result");
}

void test_is_ready_reclaims_completed_temporaries(
    const poseidon::PoseidonContext &context,poseidon::KeyGenerator &key_generator,
    const fhegpu::LoadedOperatorSpec &spec,const RmmPoolScope &pool)
{
    PoseidonGpuApi api(kContextId,context,kDeviceId);
    api.preflight(kPlanSha,false,make_target(spec),spec.spec,{});
    const auto n=context.parameters_literal()->degree();
    const int level=static_cast<int>(context.parameters_literal()->q().size()-1);
    fhegpu::RuntimePlan plan;plan.format_version=4;plan.target=make_target(spec);
    fhegpu::WaterlineConfig config;config.backend=fhegpu::PrepareBackend::Cpu;
    config.device_capacity_bytes={4ULL<<30};config.device_reserve_bytes={2ULL<<30};
    config.device_fractions={.5};config.host_budget_bytes=64ULL<<20;
    config.pinned_budget_bytes=16ULL<<20;config.raw_budget_bytes=8ULL<<20;plan.waterline=config;
    // Use the actual foreground stream, rather than assuming PTDS is where
    // the Rescale allocation or kernels will be submitted.
    api.validate_waterline_resources(plan);
    poseidon::PublicKey public_key;key_generator.create_public_key(public_key);
    poseidon::Encryptor encryptor(context,public_key);
    poseidon::Decryptor decryptor(context,key_generator.secret_key());
    poseidon::CKKSEncoder encoder(context);
    std::vector<double> slots(n/2);
    for(std::size_t i=0;i<slots.size();++i)slots[i]=.125*std::sin(i*.071);
    poseidon::Plaintext plain;encoder.encode(slots,std::ldexp(1.,160),plain);
    poseidon::Ciphertext cipher;encryptor.encrypt(plain,cipher);
    auto input=transfer_value(api,181,PoseidonGpuValue::from_host_ciphertext(cipher),
        fhegpu::ValueKind::Ciphertext,host_place(),device_place());
    fhegpu::ComputeOp rescale;rescale.kind=fhegpu::ComputeKind::Rescale;
    rescale.place=device_place();rescale.attrs=fhegpu::RescaleAttrs{level-5,40};
    {auto warm=api.compute(rescale,{input});api.drain();}
    api.collect_completed();
    const auto run_gated=[&](const char *path,const std::function<PoseidonGpuValue()> &submit,
                            const std::function<void(const PoseidonGpuValue &)> &validate) {
        const auto baseline=pool.allocated_bytes();
        const auto pinned_before=poseidon::runtime_api::communication::PinnedHostBuffer::live_bytes();
        CudaHostGate gate;CudaGateWatchdog watchdog(gate);
        auto *upstream=rmm::mr::get_current_device_resource();
        GatedAllocationResource resource(upstream,gate);
        struct RestoreResource {
            rmm::mr::device_memory_resource *original;
            ~RestoreResource(){rmm::mr::set_current_device_resource(original);}
        } restore{upstream};
        rmm::mr::set_current_device_resource(&resource);
        std::optional<PoseidonGpuValue> output;
        try {
            output.emplace(submit());
            const auto output_bytes=(output->kind()==fhegpu::ValueKind::Ciphertext
                ?output->device_ciphertext().fields_.front().size()
                :output->device_plaintext().fields_.front().size())*sizeof(poseidon::gpu::GpuWord);
            const auto pending=pool.allocated_bytes();
            require(pending>baseline+output_bytes,"Ready test did not retain real temporary GPU storage");
            require(!api.is_ready(*output),"Ready test gate did not keep the submitted work unfinished");
            require(watchdog.finish_post(),"Ready test submission or query waited for the actual allocation stream");
            // Complete CUDA work without entering any API synchronization or
            // collection path. The next call must itself settle the retainer.
            require(cudaStreamSynchronize(resource.injected_stream())==cudaSuccess,
                    "Ready test external stream synchronization failed");
            require(pool.allocated_bytes()==pending,"external CUDA completion unexpectedly released API temporaries");
            require(api.is_ready(*output),"completed Ready did not become usable after reclaim");
            const auto ready_bytes=pool.allocated_bytes();
            require(ready_bytes==baseline+output_bytes,
                    std::string("is_ready(true) still retains completed GPU temporaries: ")+path);
            require(poseidon::runtime_api::communication::PinnedHostBuffer::live_bytes()==pinned_before,
                    std::string("is_ready(true) still retains its completed pinned source: ")+path);
            validate(*output);
            output.reset();
            require(pool.allocated_bytes()==baseline,"Ready test output deletion retained GPU allocation");
            std::cout<<"IS_READY_RECLAIM path="<<path<<" pending_bytes="<<pending
                     <<" ready_bytes="<<ready_bytes<<" output_bytes="<<output_bytes
                     <<" baseline_bytes="<<baseline<<" pinned_live="<<pinned_before<<'\n';
        } catch(...) {
            // Negative controls must also destroy every allocation while its
            // public resource wrapper is still alive.
            (void)watchdog.finish_post();(void)cudaDeviceSynchronize();
            api.drain();output.reset();throw;
        }
    };
    run_gated("Rescale",[&]{return api.compute(rescale,{input});},[&](const PoseidonGpuValue &value) {
        require(value.device_ciphertext().meta.scale==std::ldexp(1.,40) &&
                    value.device_ciphertext().meta.q_count==static_cast<std::size_t>(level-4),
                "Ready test Rescale changed planned level or scale");
        poseidon::Ciphertext host;poseidon::gpu::GpuUploader::download_ciphertext(value.device_ciphertext(),host,context);
        poseidon::Plaintext decoded;decryptor.decrypt(host,decoded);
        std::vector<std::complex<double>> actual;encoder.decode(decoded,actual);
        require(actual.size()==slots.size(),"Ready test Rescale changed slot count");
        for(std::size_t i=0;i<slots.size();++i)
            require(std::abs(actual[i]-slots[i])<1e-4,"Ready test Rescale differs from dense input");
    });
    fhegpu::ValueDesc plaintext{182,fhegpu::ValueKind::Plaintext,device_place(),kContextId,level,80,true,1};
    api.prepare_resources(plaintext,fhegpu::PrepareBackend::Gpu);
    {auto warm=api.prepare_plaintext_async(plaintext,slots,fhegpu::PrepareBackend::Gpu);api.drain();}
    run_gated("GPU_Prepare",[&]{return api.prepare_plaintext_async(plaintext,slots,fhegpu::PrepareBackend::Gpu);},
        [&](const PoseidonGpuValue &value) {
            poseidon::Plaintext host;poseidon::gpu::GpuUploader::download_plaintext(value.device_plaintext(),host,context);
            std::vector<std::complex<double>> actual;encoder.decode(host,actual);
            require(actual.size()==slots.size(),"Ready test Prepare changed slot count");
            for(std::size_t i=0;i<slots.size();++i)
                require(std::abs(actual[i]-slots[i])<1e-4,"Ready test Prepare differs from dense slots");
        });
    api.drain();
}

void test_rescale_workspace_uses_validated_plan(
    poseidon::PoseidonContext &context,poseidon::KeyGenerator &key_generator,
    const fhegpu::LoadedOperatorSpec &original_spec,const RmmPoolScope &pool)
{
    auto spec=original_spec;
    spec.spec.operators.at(fhegpu::ComputeKind::Rescale).max_levels_per_op=50;
    const auto degree=context.parameters_literal()->degree();
    const int level=static_cast<int>(context.parameters_literal()->q().size()-1);
    const auto make_plan=[&](bool include_six,std::optional<int> requested_input_level=std::nullopt) {
        const int input_level=requested_input_level.value_or(level);
        fhegpu::RuntimePlan plan;plan.format_version=4;plan.target=make_target(spec);
        plan.values.push_back({0,fhegpu::ValueKind::Ciphertext,host_place(),kContextId,input_level,160,true,3});
        plan.values.push_back({1,fhegpu::ValueKind::Ciphertext,device_place(),kContextId,input_level,160,true,3});
        plan.external_inputs={0};
        plan.initialization.push_back({0,fhegpu::CommAction{0,fhegpu::CommKind::Transfer,
            fhegpu::CommHint::PointToPoint,{0},{1},{host_place()},{device_place()},
            {fhegpu::ValueKind::Ciphertext}}});
        for(int drops:{5,6}) {
            if(drops==6 && !include_six)continue;
            const auto id=static_cast<fhegpu::ValueId>(plan.values.size());
            plan.values.push_back({id,fhegpu::ValueKind::Ciphertext,device_place(),kContextId,input_level-drops,40,true,3});
            plan.execution.push_back({id-1,fhegpu::ComputeOp{fhegpu::ComputeKind::Rescale,{1},id,
                device_place(),fhegpu::RescaleAttrs{input_level-drops,40}}});
            plan.final_outputs.push_back(id);
        }
        fhegpu::WaterlineConfig config;config.backend=fhegpu::PrepareBackend::Cpu;
        config.device_capacity_bytes={4ULL<<30};config.device_reserve_bytes={2ULL<<30};
        config.device_fractions={.5};config.host_budget_bytes=64ULL<<20;
        config.pinned_budget_bytes=16ULL<<20;config.raw_budget_bytes=8ULL<<20;plan.waterline=config;
        fhegpu::PlanVerifier::verify(plan,spec,false);
        return plan;
    };
    auto five=make_plan(false),six=make_plan(true),low_five=make_plan(false,level-2);
    PoseidonGpuApi api(kContextId,context,kDeviceId);
    api.preflight(kPlanSha,false,five.target,spec.spec,{});
    require_rejected([&]{(void)api.instruction_workspace(five.execution.front());},"completed plan validation");
    api.validate_waterline_resources(five);
    const auto five_bytes=api.instruction_workspace(five.execution.front()).front.at(0);
    {
        // The same legal five-drop graph with an independently lower supported
        // maximum must receive the same budget, rather than charging capability.
        PoseidonGpuApi smaller_capability(kContextId,context,kDeviceId);
        smaller_capability.preflight(kPlanSha,false,make_target(original_spec),original_spec.spec,{});
        smaller_capability.validate_waterline_resources(five);
        require(smaller_capability.instruction_workspace(five.execution.front()).front.at(0)==five_bytes,
                "Rescale workspace follows capability maximum instead of the validated graph");
    }
    api.validate_waterline_resources(six);
    const auto six_bytes=api.instruction_workspace(six.execution.front()).front.at(0);
    require(six_bytes>five_bytes &&
                api.instruction_workspace(six.execution.back()).front.at(0)==six_bytes,
            "whole-plan six-drop maximum or full-Q clamp is incorrect for the top-level consumers");
    auto negative_target=six.execution.front();
    std::get<fhegpu::RescaleAttrs>(std::get<fhegpu::ComputeOp>(negative_target.body).attrs).target_level=-1;
    require_rejected([&]{(void)api.instruction_workspace(negative_target);},"target level is negative");

    poseidon::PublicKey public_key;key_generator.create_public_key(public_key);
    poseidon::Encryptor encryptor(context,public_key);
    poseidon::Decryptor decryptor(context,key_generator.secret_key());
    poseidon::CKKSEncoder encoder(context);poseidon::EvaluatorCkksSoftware evaluator(context);
    std::vector<double> slots(degree/2);
    for(std::size_t i=0;i<slots.size();++i)slots[i]=.125*std::sin(i*.017);
    poseidon::Plaintext plain;encoder.encode(slots,std::ldexp(1.,80),plain);
    poseidon::Ciphertext cipher,product;encryptor.encrypt(plain,cipher);
    evaluator.multiply(cipher,cipher,product);
    require(product.size()==3 && product.scale()==std::ldexp(1.,160),
            "Rescale workspace fixture did not exercise all three supported components");
    auto input=transfer_value(api,183,PoseidonGpuValue::from_host_ciphertext(std::move(product)),
        fhegpu::ValueKind::Ciphertext,host_place(),device_place());
    {auto warm=api.compute(std::get<fhegpu::ComputeOp>(six.execution.front().body),{input});api.drain();}
    const auto check_actual=[&](const fhegpu::Instruction &instruction,const PoseidonGpuValue &source,
                                int input_level,int plan_max_drop) {
        const auto &op=std::get<fhegpu::ComputeOp>(instruction.body);
        const auto before=pool.total_allocated_bytes();
        auto output=api.compute(op,{source});
        const auto allocated=pool.total_allocated_bytes()-before;
        const auto output_bytes=output.device_ciphertext().fields_.front().size()*sizeof(poseidon::gpu::GpuWord);
        const auto bound=api.instruction_workspace(instruction).front.at(0);
        // Cumulative actual GPU allocations, excluding the separately charged
        // output, upper-bound live temporary storage without mirroring its formula.
        require(allocated>=output_bytes && allocated-output_bytes<=bound,
                "validated Rescale workspace does not cover actual three-component GPU allocations");
        api.drain();poseidon::Ciphertext host;
        poseidon::gpu::GpuUploader::download_ciphertext(output.device_ciphertext(),host,context);
        poseidon::Plaintext decoded;decryptor.decrypt(host,decoded);
        std::vector<std::complex<double>> actual;encoder.decode(decoded,actual);
        require(actual.size()==slots.size(),"Rescale workspace changed slot count");
        double max_error=0;
        for(std::size_t i=0;i<slots.size();++i) {
            const auto error=std::abs(actual[i]-slots[i]*slots[i]);
            require(error<1e-4,"Rescale workspace case differs from independent dense square");
            max_error=std::max(max_error,error);
        }
        std::cout<<"RESCALE_PLAN_WORKSPACE capability=50 input_level="<<input_level
                 <<" input_q="<<source.device_ciphertext().meta.q_count<<" actual_drop="
                 <<input_level-std::get<fhegpu::RescaleAttrs>(op.attrs).target_level
                 <<" plan_max_drop="<<plan_max_drop<<" components=3 transient_allocation_total="<<allocated-output_bytes
                 <<" output_allocation="<<output_bytes<<" workspace_bound="<<bound
                 <<" slots="<<actual.size()<<" max_abs_error="<<max_error<<'\n';
    };
    for(const auto &instruction:six.execution)check_actual(instruction,input,level,6);
    api.validate_waterline_resources(five);
    require(api.instruction_workspace(five.execution.front()).front.at(0)==five_bytes,
            "smaller revalidated Rescale graph retained a stale larger workspace maximum");

    api.validate_waterline_resources(low_five);
    const auto low_bytes=api.instruction_workspace(low_five.execution.front()).front.at(0);
    require(low_bytes<five_bytes,
            "lower-input-level five-drop graph retained the full-Q Rescale workspace");
    fhegpu::ComputeOp mod_switch;
    mod_switch.kind=fhegpu::ComputeKind::ModSwitch;mod_switch.place=device_place();
    mod_switch.attrs=fhegpu::ModSwitchAttrs{level-2};
    auto low_input=api.compute(mod_switch,{input});
    api.validate_value(low_input,low_five.values.at(1));
    require(low_input.device_ciphertext().meta.scale==std::ldexp(1.,160) &&
                low_input.device_ciphertext().meta.q_count+2==input.device_ciphertext().meta.q_count,
            "lower-level fixture did not preserve scale while dropping two Q primes");
    api.drain();
    check_actual(low_five.execution.front(),low_input,level-2,5);
    api.validate_waterline_resources(five);
    require(api.instruction_workspace(five.execution.front()).front.at(0)==five_bytes,
            "top-level graph did not restore its budget after the lower-level graph");
    auto invalid=six;
    std::get<fhegpu::RescaleAttrs>(std::get<fhegpu::ComputeOp>(invalid.execution.back().body).attrs).target_level=level;
    require_rejected([&]{api.validate_waterline_resources(invalid);},"level range is unsupported");
    require_rejected([&]{(void)api.instruction_workspace(five.execution.front());},"completed plan validation");
    api.validate_waterline_resources(five);
    require(api.instruction_workspace(five.execution.front()).front.at(0)==five_bytes,
            "successful validation after a rejected plan did not restore the correct Rescale budget");
    api.drain();
}

void test_transfer_staging_cleanup()
{
    namespace comm = poseidon::runtime_api::communication;
    comm::CudaLocalTransfer transfer({kDeviceId});
    auto buffer = poseidon::gpu::GpuPlaintextData::allocate_single_device(16, 1, kDeviceId);
    auto staging = std::make_shared<comm::PinnedHostBuffer>(16 * sizeof(poseidon::gpu::GpuWord));
    std::weak_ptr<comm::PinnedHostBuffer> observer = staging;
    CudaHostGate gate;
    require(cudaSetDevice(kDeviceId) == cudaSuccess, "cudaSetDevice failed");
    require(cudaLaunchHostFunc(cudaStreamPerThread, wait_for_cuda_host_gate,
                               &gate) == cudaSuccess, "staging gate failed");
    CudaGateWatchdog watchdog(gate);
    auto request = transfer.copy_host_to_device_async(
        staging, buffer.fields_.front().data(), staging->size(), kDeviceId);
    staging.reset();
    require(!request.collect_completed() && !observer.expired(),
            "unfinished upload released its pinned staging buffer");
    require(watchdog.finish_post(), "transfer collection waited for upload");
    require(cudaEventSynchronize(request.completion_event()) == cudaSuccess,
            "upload event synchronization failed");
    require(request.collect_completed() && observer.expired(),
            "completed upload retained pinned staging through its request");
    // The dependency event remains usable after its buffer is collected.
    request.wait();
    require(cudaStreamWaitEvent(cudaStreamPerThread, request.completion_event(), 0) == cudaSuccess,
            "collected transfer lost its dependency event");
}


fhegpu::RuntimePlan make_add_plain_plan(const fhegpu::LoadedOperatorSpec &loaded_spec,
                                        int level)
{
    const auto host = host_place();
    const auto device = device_place();
    fhegpu::RuntimePlan plan;
    plan.plan_id = 1;
    plan.target = make_target(loaded_spec);
    plan.values = {
        {0, fhegpu::ValueKind::Ciphertext, host, kContextId, level, kDefaultScaleLog2, true, 2},
        {1, fhegpu::ValueKind::Plaintext, host, kContextId, level, kDefaultScaleLog2, true, 1},
        {2, fhegpu::ValueKind::Ciphertext, device, kContextId, level, kDefaultScaleLog2, true, 2},
        {3, fhegpu::ValueKind::Plaintext, device, kContextId, level, kDefaultScaleLog2, true, 1},
        {4, fhegpu::ValueKind::Ciphertext, device, kContextId, level, kDefaultScaleLog2, true, 2},
        {5, fhegpu::ValueKind::Ciphertext, host, kContextId, level, kDefaultScaleLog2, true, 2},
    };
    plan.external_inputs = {0};

    auto upload_cipher = transfer_action(1, fhegpu::ValueKind::Ciphertext, host, device);
    upload_cipher.inputs = {0};
    upload_cipher.outputs = {2};
    auto upload_plain = transfer_action(2, fhegpu::ValueKind::Plaintext, host, device);
    upload_plain.inputs = {1};
    upload_plain.outputs = {3};
    auto download_cipher = transfer_action(3, fhegpu::ValueKind::Ciphertext, device, host);
    download_cipher.inputs = {4};
    download_cipher.outputs = {5};

    plan.initialization = {
        {0, fhegpu::EncodeOp{fhegpu::InlineEncodePayload{{0.5, -1.0, 2.0, 3.0}}, 1}},
        {1, std::move(upload_cipher)},
        {2, std::move(upload_plain)},
    };
    plan.execution = {
        {3, fhegpu::ComputeOp{fhegpu::ComputeKind::AddCP, {2, 3}, 4, device, {}}},
    };
    plan.finalization = {
        {4, std::move(download_cipher)},
    };
    plan.final_outputs = {5};
    return plan;
}

fhegpu::RuntimePlan make_add_cipher_plan(const fhegpu::LoadedOperatorSpec &loaded_spec,
                                         int level)
{
    const auto host = host_place();
    const auto device = device_place();
    fhegpu::RuntimePlan plan;
    plan.plan_id = 2;
    plan.target = make_target(loaded_spec);
    plan.values = {
        {0, fhegpu::ValueKind::Ciphertext, host, kContextId, level, kDefaultScaleLog2, true, 2},
        {1, fhegpu::ValueKind::Ciphertext, host, kContextId, level, kDefaultScaleLog2, true, 2},
        {2, fhegpu::ValueKind::Ciphertext, device, kContextId, level, kDefaultScaleLog2, true, 2},
        {3, fhegpu::ValueKind::Ciphertext, device, kContextId, level, kDefaultScaleLog2, true, 2},
        {4, fhegpu::ValueKind::Ciphertext, device, kContextId, level, kDefaultScaleLog2, true, 2},
        {5, fhegpu::ValueKind::Ciphertext, host, kContextId, level, kDefaultScaleLog2, true, 2},
    };
    plan.external_inputs = {0, 1};

    auto upload_left = transfer_action(4, fhegpu::ValueKind::Ciphertext, host, device);
    upload_left.inputs = {0};
    upload_left.outputs = {2};
    auto upload_right = transfer_action(5, fhegpu::ValueKind::Ciphertext, host, device);
    upload_right.inputs = {1};
    upload_right.outputs = {3};
    auto download_result = transfer_action(6, fhegpu::ValueKind::Ciphertext, device, host);
    download_result.inputs = {4};
    download_result.outputs = {5};

    plan.initialization = {
        {0, std::move(upload_left)},
        {1, std::move(upload_right)},
    };
    plan.execution = {
        {2, fhegpu::ComputeOp{fhegpu::ComputeKind::AddCC, {2, 3}, 4, device, {}}},
    };
    plan.finalization = {
        {3, std::move(download_result)},
    };
    plan.final_outputs = {5};
    return plan;
}

fhegpu::RuntimePlan make_multiply_plain_plan(
    const fhegpu::LoadedOperatorSpec &loaded_spec, int level)
{
    const auto host = host_place();
    const auto device = device_place();
    const int output_scale_log2 = kDefaultScaleLog2 * 2;
    fhegpu::RuntimePlan plan;
    plan.plan_id = 3;
    plan.target = make_target(loaded_spec);
    plan.values = {
        {0, fhegpu::ValueKind::Ciphertext, host, kContextId, level, kDefaultScaleLog2, true, 2},
        {1, fhegpu::ValueKind::Plaintext, host, kContextId, level, kDefaultScaleLog2, true, 1},
        {2, fhegpu::ValueKind::Ciphertext, device, kContextId, level, kDefaultScaleLog2, true, 2},
        {3, fhegpu::ValueKind::Plaintext, device, kContextId, level, kDefaultScaleLog2, true, 1},
        {4, fhegpu::ValueKind::Ciphertext, device, kContextId, level, output_scale_log2, true, 2},
        {5, fhegpu::ValueKind::Ciphertext, host, kContextId, level, output_scale_log2, true, 2},
    };
    plan.external_inputs = {0};

    auto upload_cipher = transfer_action(7, fhegpu::ValueKind::Ciphertext, host, device);
    upload_cipher.inputs = {0};
    upload_cipher.outputs = {2};
    auto upload_plain = transfer_action(8, fhegpu::ValueKind::Plaintext, host, device);
    upload_plain.inputs = {1};
    upload_plain.outputs = {3};
    auto download_result = transfer_action(9, fhegpu::ValueKind::Ciphertext, device, host);
    download_result.inputs = {4};
    download_result.outputs = {5};

    plan.initialization = {
        {0, fhegpu::EncodeOp{fhegpu::InlineEncodePayload{{0.5, 1.5, -1.0, 2.0}}, 1}},
        {1, std::move(upload_cipher)},
        {2, std::move(upload_plain)},
    };
    plan.execution = {
        {3, fhegpu::ComputeOp{fhegpu::ComputeKind::MulCP, {2, 3}, 4, device, {}}},
    };
    plan.finalization = {
        {4, std::move(download_result)},
    };
    plan.final_outputs = {5};
    return plan;
}

fhegpu::RuntimePlan make_two_gpu_plan(const fhegpu::LoadedOperatorSpec &loaded_spec,
                                      int level)
{
    const auto host = host_place();
    const auto device0 = device_place(0);
    const auto device1 = device_place(1);
    fhegpu::RuntimePlan plan;
    plan.plan_id = 4;
    plan.target = make_target(loaded_spec, 2);
    plan.values = {
        {0, fhegpu::ValueKind::Ciphertext, host, kContextId, level, kDefaultScaleLog2, true, 2},
        {1, fhegpu::ValueKind::Plaintext, host, kContextId, level, kDefaultScaleLog2, true, 1},
        {2, fhegpu::ValueKind::Ciphertext, device0, kContextId, level, kDefaultScaleLog2, true, 2},
        {3, fhegpu::ValueKind::Ciphertext, device1, kContextId, level, kDefaultScaleLog2, true, 2},
        {4, fhegpu::ValueKind::Plaintext, device0, kContextId, level, kDefaultScaleLog2, true, 1},
        {5, fhegpu::ValueKind::Plaintext, device1, kContextId, level, kDefaultScaleLog2, true, 1},
        {6, fhegpu::ValueKind::Ciphertext, device0, kContextId, level, kDefaultScaleLog2, true, 2},
        {7, fhegpu::ValueKind::Ciphertext, device1, kContextId, level, kDefaultScaleLog2, true, 2},
        {8, fhegpu::ValueKind::Ciphertext, device0, kContextId, level, kDefaultScaleLog2, true, 2},
        {9, fhegpu::ValueKind::Ciphertext, device0, kContextId, level, kDefaultScaleLog2, true, 2},
        {10, fhegpu::ValueKind::Ciphertext, host, kContextId, level, kDefaultScaleLog2, true, 2},
    };
    plan.external_inputs = {0};

    auto replicate_cipher = replicate_action(
        30, fhegpu::ValueKind::Ciphertext, host, {device0, device1}, {2, 3});
    auto upload_plain =
        transfer_action(31, fhegpu::ValueKind::Plaintext, host, device0);
    upload_plain.inputs = {1};
    upload_plain.outputs = {4};
    auto copy_plain =
        transfer_action(32, fhegpu::ValueKind::Plaintext, device0, device1);
    copy_plain.hint = fhegpu::CommHint::HostStaged;
    copy_plain.inputs = {4};
    copy_plain.outputs = {5};
    auto return_cipher =
        transfer_action(33, fhegpu::ValueKind::Ciphertext, device1, device0);
    return_cipher.hint = fhegpu::CommHint::PointToPoint;
    return_cipher.inputs = {7};
    return_cipher.outputs = {8};
    auto download_result =
        transfer_action(34, fhegpu::ValueKind::Ciphertext, device0, host);
    download_result.inputs = {9};
    download_result.outputs = {10};

    plan.initialization = {
        {0, fhegpu::EncodeOp{fhegpu::InlineEncodePayload{{0.5, -1.0, 2.0, 3.0}}, 1}},
        {1, std::move(replicate_cipher)},
        {2, std::move(upload_plain)},
        {3, std::move(copy_plain)},
    };
    plan.execution = {
        {4, fhegpu::ComputeOp{fhegpu::ComputeKind::AddCP, {2, 4}, 6, device0, {}}},
        {5, fhegpu::ComputeOp{fhegpu::ComputeKind::AddCP, {3, 5}, 7, device1, {}}},
        {6, std::move(return_cipher)},
        {7, fhegpu::ComputeOp{fhegpu::ComputeKind::AddCC, {6, 8}, 9, device0, {}}},
    };
    plan.finalization = {
        {8, std::move(download_result)},
    };
    plan.final_outputs = {10};
    return plan;
}

void test_device_mapping_rejections(const poseidon::PoseidonContext &context,
                                    int visible_device_count)
{
    require_rejected(
        [&] {
            PoseidonGpuApi api(kContextId, context, std::vector<int>{});
        },
        "at least one CUDA device");
    require_rejected(
        [&] {
            PoseidonGpuApi api(kContextId, context,
                               std::vector<int>{kDeviceId, kDeviceId});
        },
        "unique");
    require_rejected(
        [&] {
            PoseidonGpuApi api(kContextId, context,
                               std::vector<int>{visible_device_count});
        },
        "unavailable");
}

void test_preflight_rejections(PoseidonGpuApi &api,
                               const fhegpu::LoadedOperatorSpec &loaded_spec)
{
    require_rejected(
        [&] {
            api.configure_native_bootstrap(
                "missing-native-boot-keys", 0,
                poseidon::gpu::GpuBootstrapData{},
                std::shared_ptr<const poseidon::gpu::GpuRelinKeysData>{},
                std::shared_ptr<const poseidon::gpu::GpuGaloisKeysData>{});
        },
        "evaluation keys");

    const auto target = make_target(loaded_spec);
    const fhegpu::PlanRequirements valid_requirements{
        {fhegpu::RequiredCapability::Encode, fhegpu::RequiredCapability::Transfer,
         fhegpu::RequiredCapability::Replicate}, {}};
    api.preflight(kPlanSha, false, target, loaded_spec.spec, valid_requirements);
    require_rejected([&] { static_cast<void>(api.cuda_device_id(1)); },
                     "logical device index");

    fhegpu::PlanRequirements remote_requirements;
    remote_requirements.keys = {
        {fhegpu::KeyKind::Galois,
         {fhegpu::PlaceKind::Device, 1, 0}, 1, std::nullopt},
    };
    api.preflight(kPlanSha, false, target, loaded_spec.spec,
                  remote_requirements);

    auto placeholder = loaded_spec.spec;
    placeholder.status = "placeholder";
    require_rejected(
        [&] { api.preflight(kPlanSha, false, target, placeholder, valid_requirements); },
        "placeholder");

    auto multi_device = target;
    multi_device.device_counts = {2};
    require_rejected(
        [&] { api.preflight(kPlanSha, false, multi_device, loaded_spec.spec, valid_requirements); },
        "device count");

    auto shallow_rescale = loaded_spec.spec;
    shallow_rescale.operators.at(fhegpu::ComputeKind::Rescale).max_levels_per_op = 2;
    require_rejected(
        [&] { api.preflight(kPlanSha, false, target, shallow_rescale, valid_requirements); },
        "four rescale levels");

    auto unsupported_boot = loaded_spec.spec;
    unsupported_boot.operators.at(fhegpu::ComputeKind::Boot).supported = true;
    require_rejected(
        [&] {
            api.preflight(kPlanSha, false, target, unsupported_boot, valid_requirements);
        },
        "Boot profiles");

    auto native_boot = loaded_spec.spec;
    native_boot.operators.at(fhegpu::ComputeKind::Boot).supported = true;
    fhegpu::BootProfile native_profile;
    native_profile.profile_id = "unconfigured-native-boot";
    native_profile.implementation = fhegpu::BootImplementation::Native;
    native_profile.input_level_min = native_boot.level_lower_bound;
    native_profile.input_level_max = native_boot.level_upper_bound;
    native_profile.input_components = 2;
    native_profile.output_level = native_boot.level_upper_bound;
    native_profile.output_scale_log2 = native_boot.default_scale_log2;
    native_profile.output_components = 2;
    native_boot.boot_profiles = {std::move(native_profile)};
    api.preflight(kPlanSha, false, target, native_boot, valid_requirements);
    const fhegpu::PlanRequirements native_boot_requirement{
        {fhegpu::RequiredCapability::BootNative}, {}};
    require_rejected(
        [&] {
            api.preflight(kPlanSha, false, target, native_boot,
                          native_boot_requirement);
        },
        "boot_native");

    const fhegpu::PlanRequirements host_compute{
        {fhegpu::RequiredCapability::HostCompute}, {}};
    require_rejected(
        [&] { api.preflight(kPlanSha, false, target, loaded_spec.spec, host_compute); },
        "host_compute");

    fhegpu::PlanRequirements missing_key_level;
    missing_key_level.keys = {
        {fhegpu::KeyKind::Galois, device_place(), 1, std::nullopt},
    };
    require_rejected(
        [&] {
            api.preflight(kPlanSha, false, target, loaded_spec.spec,
                          missing_key_level);
        },
        "has no level");
}

void test_selected_galois_key_upload(
    const poseidon::PoseidonContext &context,
    const poseidon::GaloisKeys &galois_keys)
{
    const std::size_t q_count = context.parameters_literal()->q().size();
    const std::uint32_t selected_galois_elt =
        context.crt_context()->galois_tool()->get_elt_from_step(2);
    const std::size_t selected_key_index =
        poseidon::GaloisKeys::get_index(selected_galois_elt);

    auto all_keys = poseidon::gpu::GpuUploader::upload_galois_keys(
        galois_keys, kDeviceId, q_count);
    auto selected_keys = poseidon::gpu::GpuUploader::upload_galois_keys(
        galois_keys, kDeviceId, q_count, {selected_galois_elt});

    require(selected_keys.polys_.size() < all_keys.polys_.size(),
            "selected Galois-key upload did not reduce GPU key data");
    require(!selected_keys.poly_metadata_.empty(),
            "selected Galois-key upload produced no GPU key data");
    for (const auto &mapping : selected_keys.poly_metadata_)
    {
        require(mapping.key_index == selected_key_index,
                "selected Galois-key upload included an unrequested key");
    }
}

void test_application_gpu_extensions(
    const poseidon::PoseidonContext &context,
    poseidon::KeyGenerator &key_generator,
    const poseidon::GaloisKeys &galois_keys)
{
    using namespace poseidon::gpu;
    poseidon::PublicKey public_key;
    key_generator.create_public_key(public_key);
    poseidon::CKKSEncoder encoder(context);
    poseidon::Encryptor encryptor(context, public_key);
    poseidon::Plaintext input_plain;
    encoder.encode(std::vector<double>{0.25, -0.5, 0.75, 1.0},
                   std::ldexp(1.0, kDefaultScaleLog2), input_plain);
    poseidon::Ciphertext input;
    encryptor.encrypt(input_plain, input);
    auto source = GpuUploader::upload_ciphertext(input, kDeviceId);
    auto unchanged = GpuUploader::upload_ciphertext(input, kDeviceId);
    const auto source_q = source.meta.q_count;
    const auto target_q = source_q - 2;
    const auto target = context.crt_context()->parms_id_map().at(target_q - 1);
    GpuParameterData parameters(context, kDeviceId, {source_q, target_q});
    require(parameters.level_count() == 2, "sparse parameter levels were not retained");
    GpuEvaluator evaluator(parameters);

    const auto exact = [&](const GpuCiphertextData &actual,
                           const GpuCiphertextData &expected) {
        poseidon::Ciphertext a, b;
        GpuUploader::download_ciphertext(actual, a, context);
        GpuUploader::download_ciphertext(expected, b, context);
        require(a.parms_id() == b.parms_id() && a.size() == b.size() &&
                    a.scale() == b.scale() && a.is_ntt_form() == b.is_ntt_form(),
                "application extension changed ciphertext metadata");
        const auto words = a.size() * a.poly_modulus_degree() * a.coeff_modulus_size();
        require(std::equal(a.data(), a.data() + words, b.data()),
                "application extension changed RNS residues");
    };

    GpuCiphertextData dropped;
    evaluator.drop_modulus(source, dropped, target);
    poseidon::Plaintext weight;
    encoder.encode(std::vector<double>{0.5, 0.25, -0.5, 0.75}, target,
                   std::ldexp(1.0, kDefaultScaleLog2), weight);
    auto plaintext = GpuUploader::upload_plaintext(weight, kDeviceId);
    GpuCiphertextData expected, actual, doubled;
    evaluator.multiply_plain(dropped, plaintext, expected);
    evaluator.multiply_plain_q_prefix(source, plaintext, actual);
    exact(actual, expected);
    evaluator.add(expected, expected, doubled);
    evaluator.multiply_plain_sum_q_prefix({&source, &source}, {&plaintext, &plaintext}, actual);
    exact(actual, doubled);
    evaluator.multiply(dropped, dropped, expected);
    evaluator.multiply_q_prefix(source, source, actual, target);
    exact(actual, expected);
    evaluator.add(dropped, dropped, doubled);
    evaluator.sub_plain(doubled, plaintext, expected);
    evaluator.double_sub_plain_q_prefix(dropped, plaintext, actual);
    exact(actual, expected);
    exact(source, unchanged);

    auto rotations = GpuUploader::upload_double_hoist_galois_keys(galois_keys, kDeviceId);
    GpuDoubleHoistWorkspace workspace;
    std::vector<GpuCiphertextData> rotated;
    evaluator.rotate_many_hoisted(source, {1, 2}, rotations, workspace, rotated);
    require(rotated.size() == 2, "hoisted rotation batch size changed");
    for (std::size_t i = 0; i < rotated.size(); ++i)
    {
        evaluator.rotate(source, static_cast<int>(i + 1), rotations, expected);
        exact(rotated[i], expected);
    }
    exact(source, unchanged);
    const auto *allocation = source.fields_.front().data();
    evaluator.drop_modulus_inplace(source, target);
    require(source.fields_.front().data() == allocation,
            "inplace modulus drop replaced its allocation");
    exact(source, dropped);
}

void test_host_decrypt_reencrypt_boot(
    PoseidonGpuApi &api, const poseidon::PoseidonContext &context,
    const poseidon::PublicKey &public_key, const poseidon::SecretKey &secret_key,
    const fhegpu::LoadedOperatorSpec &loaded_spec)
{
    const auto boot_spec = make_boot_operator_spec(loaded_spec);
    fhegpu::PlanRequirements requirements;
    requirements.capabilities = {
        fhegpu::RequiredCapability::HostCompute,
        fhegpu::RequiredCapability::BootDecryptReencrypt,
    };
    requirements.keys = {
        {fhegpu::KeyKind::Secret, host_place(), std::nullopt},
    };
    api.preflight(kPlanSha, false, make_target(boot_spec), boot_spec.spec, requirements);

    constexpr int input_level = 2;
    const int output_level = boot_spec.spec.level_upper_bound;
    const std::vector<double> expected{1.25, -2.5, 3.75, 4.5};
    poseidon::CKKSEncoder encoder(context);
    poseidon::Plaintext input_plain;
    encoder.encode(expected,
                   context.crt_context()->parms_id_map().at(input_level),
                   std::ldexp(1.0, kDefaultScaleLog2), input_plain);
    poseidon::Encryptor encryptor(context, public_key);
    poseidon::Ciphertext input_cipher;
    encryptor.encrypt(input_plain, input_cipher);

    fhegpu::ComputeOp boot;
    boot.kind = fhegpu::ComputeKind::Boot;
    boot.place = host_place();
    boot.attrs = fhegpu::BootAttrs{
        output_level, kDefaultScaleLog2, 2,
        "poseidon-gpu-host-boot-test",
        fhegpu::BootImplementation::DecryptReencrypt,
    };
    auto host_input = PoseidonGpuValue::from_host_ciphertext(std::move(input_cipher));
    auto refreshed = api.compute(boot, {host_input});
    auto runtime_boot = boot;
    runtime_boot.inputs = {50}; runtime_boot.output = 51;
    fhegpu::RuntimePlan boot_plan;
    boot_plan.format_version = 3;
    boot_plan.target = make_target(boot_spec);
    boot_plan.values = {
        {50, fhegpu::ValueKind::Ciphertext, host_place(), kContextId, input_level, kDefaultScaleLog2, true, 2},
        {51, fhegpu::ValueKind::Ciphertext, host_place(), kContextId, output_level, kDefaultScaleLog2, true, 2},
        {52, fhegpu::ValueKind::Plaintext, host_place(), kContextId, output_level, kDefaultScaleLog2, true, 1}};
    boot_plan.external_inputs = {50}; boot_plan.final_outputs = {51};
    boot_plan.execution = {{0, fhegpu::EncodeOp{fhegpu::InlineEncodePayload{{1.0}}, 52}},
                          {1, fhegpu::ReleaseOp{52}}, {2, runtime_boot}, {3, fhegpu::FenceOp{}}};
    fhegpu::SequentialRuntime<PoseidonGpuApi> boot_runtime(0, 1, 1, api,
        fhegpu::DeviceExecutionMode::PerDeviceWorkers);
    const auto boot_result = boot_runtime.run({boot_plan, kPlanSha}, {boot_spec, std::nullopt, false}, {{50, host_input}});
    refreshed = boot_result.values.at(51).value;
    api.validate_value(
        refreshed,
        {50, fhegpu::ValueKind::Ciphertext, host_place(), kContextId,
         output_level, kDefaultScaleLog2, true, 2});

    poseidon::Decryptor decryptor(context, secret_key);
    poseidon::Plaintext output_plain;
    decryptor.decrypt(refreshed.host_ciphertext(), output_plain);
    std::vector<std::complex<double>> actual;
    encoder.decode(output_plain, actual);
    for (std::size_t i = 0; i < expected.size(); ++i)
    {
        require(std::abs(actual[i].real() - expected[i]) < 1e-4,
                "Host decrypt_reencrypt Boot result mismatch at slot " +
                    std::to_string(i));
        require(std::abs(actual[i].imag()) < 1e-4,
                "Host decrypt_reencrypt Boot imaginary mismatch at slot " +
                    std::to_string(i));
    }

    auto native = boot;
    native.attrs = fhegpu::BootAttrs{
        output_level, kDefaultScaleLog2, 2,
        "poseidon-gpu-host-boot-test", fhegpu::BootImplementation::Native,
    };
    require_rejected(
        [&] { (void)api.compute(native, {refreshed}); }, "native Boot");
}

void test_runtime_drains_unreturned_work(
    const poseidon::PoseidonContext &context,
    poseidon::KeyGenerator &key_generator,
    const fhegpu::LoadedOperatorSpec &spec,
    const RmmPoolScope &pool)
{
    class ObservedApi : public PoseidonGpuApi
    {
    public:
        using PoseidonGpuApi::PoseidonGpuApi;
        bool drained = false;
        CudaHostGate *gate = nullptr;
        Value compute(const fhegpu::ComputeOp &op, const std::vector<Value> &inputs)
        {
            if (gate)
            {
                require(cudaSetDevice(kDeviceId) == cudaSuccess, "cudaSetDevice failed");
                require(cudaLaunchHostFunc(cudaStreamPerThread, wait_for_cuda_host_gate,
                                           gate) == cudaSuccess, "runtime gate failed");
            }
            return PoseidonGpuApi::compute(op, inputs);
        }
        void drain()
        {
            PoseidonGpuApi::drain();
            drained = true;
        }
    };
    ObservedApi api(kContextId, context, kDeviceId);
    poseidon::PublicKey public_key;
    key_generator.create_public_key(public_key);
    poseidon::Encryptor encryptor(context, public_key);
    poseidon::CKKSEncoder encoder(context);
    poseidon::Plaintext plain;
    encoder.encode(std::vector<double>{1.0, -2.0},
                   std::ldexp(1.0, kDefaultScaleLog2), plain);
    poseidon::Ciphertext cipher;
    encryptor.encrypt(plain, cipher);
    const int level = static_cast<int>(cipher.level());
    const int components = static_cast<int>(cipher.size());
    fhegpu::RuntimePlan plan;
    plan.plan_id = 90;
    plan.target = make_target(spec);
    plan.values = {
        {0, fhegpu::ValueKind::Ciphertext, host_place(), kContextId, level,
         kDefaultScaleLog2, true, components},
        {1, fhegpu::ValueKind::Ciphertext, device_place(), kContextId, level,
         kDefaultScaleLog2, true, components},
        {2, fhegpu::ValueKind::Ciphertext, device_place(), kContextId, level,
         kDefaultScaleLog2, true, components}};
    plan.external_inputs = {0};
    plan.initialization = {{0, transfer_action(
        91, fhegpu::ValueKind::Ciphertext, host_place(), device_place())}};
    plan.final_outputs = {0}; // Device work has no dependency on this Host output.
    plan.execution = {{1, fhegpu::ComputeOp{
        fhegpu::ComputeKind::Negate, {1}, 2, device_place(), {}}}};
    const auto baseline = pool.allocated_bytes();
    auto host = PoseidonGpuValue::from_host_ciphertext(cipher);
    {
        auto device = transfer_value(api, 90, host, fhegpu::ValueKind::Ciphertext,
                                     host_place(), device_place());
        {
            auto warmup = api.compute(std::get<fhegpu::ComputeOp>(plan.execution.front().body), {device});
            api.drain();
        }
    }
    for (auto mode : {fhegpu::DeviceExecutionMode::Sequential,
                      fhegpu::DeviceExecutionMode::PerDeviceWorkers})
    {
        fhegpu::SequentialRuntime<ObservedApi> runtime(0, 1, 1, api, mode);
        for (int iteration = 0; iteration < 3; ++iteration)
        {
            CudaHostGate gate;
            CudaGateWatchdog watchdog(gate);
            api.gate = &gate;
            api.drained = false;
            auto artifact = runtime.run({plan, kPlanSha}, {spec, std::nullopt, false},
                                        {{0, host}});
            api.gate = nullptr;
            require(api.drained, "Runtime did not drain submitted work");
            require(!watchdog.finish_post(),
                    "Runtime returned before unrelated Device work completed");
            require(cudaStreamQuery(cudaStreamPerThread) == cudaSuccess,
                    "Runtime left an unreturned computation running");
            require(artifact.values.size() == 1 && artifact.values.count(0) == 1,
                    "Runtime final output changed");
        }
    }
    require(pool.allocated_bytes() == baseline,
            "Runtime retained intermediate objects between completed runs");
}

void test_runtime_release(const poseidon::PoseidonContext &context,
                          poseidon::KeyGenerator &key_generator,
                          const fhegpu::LoadedOperatorSpec &spec,
                          const RmmPoolScope &pool)
{
    class ObservedApi : public PoseidonGpuApi
    {
    public:
        using PoseidonGpuApi::PoseidonGpuApi;
        const RmmPoolScope *pool = nullptr;
        CudaHostGate *gate = nullptr;
        CudaGateWatchdog *watchdog = nullptr;
        bool gate_transfer = false;
        int releases_before_unblock = 0;
        int observed_releases = 0;
        std::size_t pending_bytes = 0;
        std::optional<std::size_t> expected_drain_bytes;
        void enqueue_gate()
        {
            require(cudaSetDevice(kDeviceId) == cudaSuccess, "cudaSetDevice failed");
            require(cudaLaunchHostFunc(cudaStreamPerThread, wait_for_cuda_host_gate,
                                       gate) == cudaSuccess, "Release gate failed");
        }
        Value compute(const fhegpu::ComputeOp &op, const std::vector<Value> &inputs)
        {
            if (gate && !gate_transfer) enqueue_gate();
            auto result = PoseidonGpuApi::compute(op, inputs);
            if (gate && !gate_transfer) pending_bytes = pool->allocated_bytes();
            return result;
        }
        CommHandle communicate_async(const fhegpu::CommAction &action,
                                      const std::vector<Value> &inputs,
                                      const std::vector<fhegpu::ValueDesc> &descs)
        {
            if (gate && gate_transfer) enqueue_gate();
            auto result = PoseidonGpuApi::communicate_async(action, inputs, descs);
            if (gate && gate_transfer) pending_bytes = pool->allocated_bytes();
            return result;
        }
        void collect_completed()
        {
            PoseidonGpuApi::collect_completed();
            if (pending_bytes && observed_releases < releases_before_unblock)
            {
                require(pool->allocated_bytes() == pending_bytes,
                        "Release reclaimed storage while GPU work was still reading it");
                if (++observed_releases == releases_before_unblock)
                    require(watchdog->finish_post(), "Release waited for unfinished GPU work");
            }
        }
        void drain()
        {
            PoseidonGpuApi::drain();
            if (expected_drain_bytes)
                require(pool->allocated_bytes() == *expected_drain_bytes,
                        "Release kept GPU storage until Runtime teardown");
        }
    };
    ObservedApi api(kContextId, context, kDeviceId);
    api.pool = &pool;
    poseidon::PublicKey public_key;
    key_generator.create_public_key(public_key);
    poseidon::Encryptor encryptor(context, public_key);
    poseidon::Decryptor decryptor(context, key_generator.secret_key());
    poseidon::CKKSEncoder encoder(context);
    poseidon::Plaintext plain;
    const std::vector<double> input{1.0, 2.0, 3.0, 4.0};
    encoder.encode(input, std::ldexp(1.0, kDefaultScaleLog2), plain);
    poseidon::Ciphertext cipher;
    encryptor.encrypt(plain, cipher);
    const auto caller_input = PoseidonGpuValue::from_host_ciphertext(cipher);
    const int level = static_cast<int>(cipher.level());
    const auto ordinary = make_add_plain_plan(spec, level);
    const fhegpu::RuntimeResources resources{spec, std::nullopt, false};
    // Warm evaluator tables before observing active object bytes.
    {
        fhegpu::SequentialRuntime<ObservedApi> runtime(0, 1, 1, api);
        auto warmup = runtime.run({ordinary, kPlanSha}, resources, {{0, caller_input}});
    }
    const auto baseline = pool.allocated_bytes();
    for (bool transfer_only : {false, true})
    {
        auto plan = ordinary;
        plan.format_version = 2;
        if (transfer_only)
        {
            plan.values = {ordinary.values.at(0), ordinary.values.at(2)};
            plan.initialization = {{0, ordinary.initialization.at(1).body},
                                   {1, fhegpu::ReleaseOp{2}}};
            plan.execution.clear();
            plan.finalization.clear();
            plan.final_outputs = {0};
        }
        else
        {
            plan.initialization.push_back({0, fhegpu::ReleaseOp{0}});
            plan.initialization.push_back({0, fhegpu::ReleaseOp{1}});
            plan.execution.push_back({0, fhegpu::ReleaseOp{2}});
            plan.execution.push_back({0, fhegpu::ReleaseOp{3}});
            plan.finalization.push_back({0, fhegpu::ReleaseOp{4}});
            std::size_t ordinal = 0;
            for (auto *phase : {&plan.initialization, &plan.execution, &plan.finalization})
                for (auto &instruction : *phase) instruction.ordinal = ordinal++;
        }
        for (auto mode : {fhegpu::DeviceExecutionMode::Sequential,
                          fhegpu::DeviceExecutionMode::PerDeviceWorkers})
        {
            fhegpu::SequentialRuntime<ObservedApi> runtime(0, 1, 1, api, mode);
            for (int iteration = 0; iteration < 3; ++iteration)
            {
                CudaHostGate gate;
                CudaGateWatchdog watchdog(gate);
                api.gate = &gate;
                api.watchdog = &watchdog;
                api.gate_transfer = transfer_only;
                api.releases_before_unblock = transfer_only ? 1 : 2;
                api.observed_releases = 0;
                api.pending_bytes = 0;
                api.expected_drain_bytes = baseline;
                const auto artifact = runtime.run({plan, kPlanSha}, resources, {{0, caller_input}});
                api.gate = nullptr;
                api.watchdog = nullptr;
                require(api.observed_releases == api.releases_before_unblock,
                        "Runtime did not collect completions at Release");
                poseidon::Plaintext result_plain;
                decryptor.decrypt(artifact.values.at(transfer_only ? 0 : 5).value.host_ciphertext(),
                                  result_plain);
                std::vector<std::complex<double>> result;
                encoder.decode(result_plain, result);
                const std::vector<double> expected = transfer_only ? input : std::vector<double>{1.5, 1, 5, 7};
                for (std::size_t i = 0; i < expected.size(); ++i)
                    require(std::abs(result[i].real() - expected[i]) < 1e-4 &&
                            std::abs(result[i].imag()) < 1e-4, "Release changed the final result");
                decryptor.decrypt(caller_input.host_ciphertext(), result_plain);
                encoder.decode(result_plain, result);
                for (std::size_t i = 0; i < input.size(); ++i)
                    require(std::abs(result[i].real() - input[i]) < 1e-4,
                            "Release changed caller-owned input");
            }
        }
    }
}

void test_runtime_add_plain(PoseidonGpuApi &api, const poseidon::PoseidonContext &context,
                            poseidon::KeyGenerator &key_generator,
                            const fhegpu::LoadedOperatorSpec &loaded_spec)
{
    poseidon::PublicKey public_key;
    key_generator.create_public_key(public_key);
    poseidon::Encryptor encryptor(context, public_key);
    poseidon::Decryptor decryptor(context, key_generator.secret_key());
    poseidon::CKKSEncoder encoder(context);

    const std::vector<double> input{1.0, 2.0, 3.0, 4.0};
    poseidon::Plaintext input_plain;
    encoder.encode(input, std::ldexp(1.0, kDefaultScaleLog2), input_plain);
    poseidon::Ciphertext input_cipher;
    encryptor.encrypt(input_plain, input_cipher);

    const int level = static_cast<int>(context.parameters_literal()->q().size() - 1);
    const auto plan = make_add_plain_plan(loaded_spec, level);
    const fhegpu::LoadedRuntimePlan loaded_plan{plan, kPlanSha};
    const fhegpu::RuntimeResources resources{loaded_spec, std::nullopt, false};

    fhegpu::SequentialRuntime<PoseidonGpuApi> runtime(0, 1, 1, api);
    std::unordered_map<fhegpu::ValueId, PoseidonGpuValue> inputs;
    inputs.emplace(0, PoseidonGpuValue::from_host_ciphertext(std::move(input_cipher)));
    const auto artifact = runtime.run(loaded_plan, resources, inputs);

    poseidon::Plaintext result_plain;
    decryptor.decrypt(artifact.values.at(5).value.host_ciphertext(), result_plain);
    std::vector<std::complex<double>> result;
    encoder.decode(result_plain, result);

    const std::vector<double> expected{1.5, 1.0, 5.0, 7.0};
    for (std::size_t i = 0; i < expected.size(); ++i)
    {
        require(std::abs(result[i].real() - expected[i]) < 1e-4,
                "unexpected AddCP result at slot " + std::to_string(i));
        require(std::abs(result[i].imag()) < 1e-4,
                "unexpected AddCP imaginary part at slot " + std::to_string(i));
    }
}

// Compare eager encoding with online uploads, then V3 online encoding/Fence.
void test_runtime_explicit_encode_fence(const poseidon::PoseidonContext &context,
                                   poseidon::KeyGenerator &key_generator,
                                   const fhegpu::LoadedOperatorSpec &spec)
{
    PoseidonGpuApi api(kContextId, context, kDeviceId);
    poseidon::PublicKey public_key;
    key_generator.create_public_key(public_key);
    poseidon::CKKSEncoder encoder(context);
    poseidon::Encryptor encryptor(context, public_key);
    poseidon::Decryptor decryptor(context, key_generator.secret_key());
    poseidon::Plaintext plain;
    encoder.encode(std::vector<double>{1, 2, 3, 4}, std::ldexp(1.0, kDefaultScaleLog2), plain);
    poseidon::Ciphertext cipher;
    encryptor.encrypt(plain, cipher);
    const int level = static_cast<int>(cipher.level());
    const auto base = make_add_plain_plan(spec, level);
    const auto input = PoseidonGpuValue::from_host_ciphertext(cipher);
    for (bool online_encode : {false, true})
    for (auto mode : {fhegpu::DeviceExecutionMode::Sequential,
                      fhegpu::DeviceExecutionMode::PerDeviceWorkers}) {
        auto plan = base;
        plan.format_version = online_encode ? 3 : 2;
        plan.values = {base.values[0], base.values[2], base.values[5]};
        plan.initialization = {base.initialization[1]};
        plan.execution.clear();
        fhegpu::ValueId previous = 2;
        for (int round = 0; round < 3; ++round) {
            const fhegpu::ValueId weight = 10 + round * 3, uploaded = weight + 1, output = weight + 2;
            for (auto entry : {base.values[1], base.values[3], base.values[4]}) {
                entry.id = entry.kind == fhegpu::ValueKind::Ciphertext ? output :
                    entry.place.kind == fhegpu::PlaceKind::Host ? weight : uploaded;
                plan.values.push_back(entry);
            }
            auto encoding = std::get<fhegpu::EncodeOp>(base.initialization[0].body);
            encoding.output = weight;
            (online_encode ? plan.execution : plan.initialization).push_back({0, encoding});
            auto upload = std::get<fhegpu::CommAction>(base.initialization[2].body);
            upload.id = 10 + round; upload.inputs = {weight}; upload.outputs = {uploaded};
            plan.execution.push_back({0, upload});
            plan.execution.push_back({0, fhegpu::ReleaseOp{weight}});
            plan.execution.push_back({0, fhegpu::ComputeOp{fhegpu::ComputeKind::AddCP,
                {previous, uploaded}, output, device_place(), {}}});
            plan.execution.push_back({0, fhegpu::ReleaseOp{uploaded}});
            plan.execution.push_back({0, fhegpu::ReleaseOp{previous}});
            if (online_encode) plan.execution.push_back({0, fhegpu::FenceOp{}});
            previous = output;
        }
        std::get<fhegpu::CommAction>(plan.finalization[0].body).inputs = {previous};
        plan.finalization.push_back({0, fhegpu::ReleaseOp{previous}});
        std::size_t ordinal = 0;
        for (auto *phase : {&plan.initialization, &plan.execution, &plan.finalization})
            for (auto &instruction : *phase) instruction.ordinal = ordinal++;
        fhegpu::SequentialRuntime<PoseidonGpuApi> runtime(0, 1, 1, api, mode);
        auto result = runtime.run({plan, kPlanSha}, {spec, std::nullopt, false}, {{0, input}});
        const auto &output = result.values.at(5).value;
        api.validate_value(output, base.values[5]);
        poseidon::Plaintext decoded;
        decryptor.decrypt(output.host_ciphertext(), decoded);
        std::vector<std::complex<double>> slots;
        encoder.decode(decoded, slots);
        const std::vector<double> expected{2.5, -1, 9, 13};
        for (std::size_t i = 0; i < expected.size(); ++i)
            require(std::abs(slots[i] - expected[i]) < 1e-4,
                    "explicit Encode/Fence plaintext numerical mismatch");
    }
}

void test_runtime_add_ciphertexts(PoseidonGpuApi &api,
                                  const poseidon::PoseidonContext &context,
                                  poseidon::KeyGenerator &key_generator,
                                  const fhegpu::LoadedOperatorSpec &loaded_spec)
{
    poseidon::PublicKey public_key;
    key_generator.create_public_key(public_key);
    poseidon::Encryptor encryptor(context, public_key);
    poseidon::Decryptor decryptor(context, key_generator.secret_key());
    poseidon::CKKSEncoder encoder(context);

    const std::vector<double> left{1.0, -2.0, 3.0, 4.0};
    const std::vector<double> right{0.5, 1.5, -1.0, 2.0};
    poseidon::Plaintext left_plain;
    poseidon::Plaintext right_plain;
    encoder.encode(left, std::ldexp(1.0, kDefaultScaleLog2), left_plain);
    encoder.encode(right, std::ldexp(1.0, kDefaultScaleLog2), right_plain);
    poseidon::Ciphertext left_cipher;
    poseidon::Ciphertext right_cipher;
    encryptor.encrypt(left_plain, left_cipher);
    encryptor.encrypt(right_plain, right_cipher);

    const int level = static_cast<int>(context.parameters_literal()->q().size() - 1);
    const fhegpu::LoadedRuntimePlan loaded_plan{make_add_cipher_plan(loaded_spec, level),
                                                kPlanSha};
    const fhegpu::RuntimeResources resources{loaded_spec, std::nullopt, false};
    fhegpu::SequentialRuntime<PoseidonGpuApi> runtime(0, 1, 1, api);
    std::unordered_map<fhegpu::ValueId, PoseidonGpuValue> inputs;
    inputs.emplace(0, PoseidonGpuValue::from_host_ciphertext(std::move(left_cipher)));
    inputs.emplace(1, PoseidonGpuValue::from_host_ciphertext(std::move(right_cipher)));
    const auto artifact = runtime.run(loaded_plan, resources, inputs);

    poseidon::Plaintext result_plain;
    decryptor.decrypt(artifact.values.at(5).value.host_ciphertext(), result_plain);
    std::vector<std::complex<double>> result;
    encoder.decode(result_plain, result);
    for (std::size_t i = 0; i < left.size(); ++i)
    {
        const double expected = left[i] + right[i];
        require(std::abs(result[i].real() - expected) < 1e-4,
                "unexpected AddCC result at slot " + std::to_string(i));
        require(std::abs(result[i].imag()) < 1e-4,
                "unexpected AddCC imaginary part at slot " + std::to_string(i));
    }
}

void test_runtime_multiply_plain(PoseidonGpuApi &api,
                                 const poseidon::PoseidonContext &context,
                                 poseidon::KeyGenerator &key_generator,
                                 const fhegpu::LoadedOperatorSpec &loaded_spec)
{
    poseidon::PublicKey public_key;
    key_generator.create_public_key(public_key);
    poseidon::Encryptor encryptor(context, public_key);
    poseidon::Decryptor decryptor(context, key_generator.secret_key());
    poseidon::CKKSEncoder encoder(context);

    const std::vector<double> input{1.0, -2.0, 3.0, 4.0};
    const std::vector<double> multiplier{0.5, 1.5, -1.0, 2.0};
    poseidon::Plaintext input_plain;
    encoder.encode(input, std::ldexp(1.0, kDefaultScaleLog2), input_plain);
    poseidon::Ciphertext input_cipher;
    encryptor.encrypt(input_plain, input_cipher);

    const int level = static_cast<int>(context.parameters_literal()->q().size() - 1);
    const fhegpu::LoadedRuntimePlan loaded_plan{
        make_multiply_plain_plan(loaded_spec, level), kPlanSha};
    const fhegpu::RuntimeResources resources{loaded_spec, std::nullopt, false};
    fhegpu::SequentialRuntime<PoseidonGpuApi> runtime(0, 1, 1, api);
    std::unordered_map<fhegpu::ValueId, PoseidonGpuValue> inputs;
    inputs.emplace(0, PoseidonGpuValue::from_host_ciphertext(std::move(input_cipher)));
    const auto artifact = runtime.run(loaded_plan, resources, inputs);

    const auto &result_cipher = artifact.values.at(5).value.host_ciphertext();
    require(std::abs(std::log2(result_cipher.scale()) - kDefaultScaleLog2 * 2) < 1e-6,
            "MulCP output scale is incorrect");
    require(result_cipher.level() == static_cast<std::size_t>(level),
            "MulCP unexpectedly changed level");
    require(result_cipher.size() == 2, "MulCP unexpectedly changed components");

    poseidon::Plaintext result_plain;
    decryptor.decrypt(result_cipher, result_plain);
    std::vector<std::complex<double>> result;
    encoder.decode(result_plain, result);
    for (std::size_t i = 0; i < input.size(); ++i)
    {
        const double expected = input[i] * multiplier[i];
        require(std::abs(result[i].real() - expected) < 1e-4,
                "unexpected MulCP result at slot " + std::to_string(i));
        require(std::abs(result[i].imag()) < 1e-4,
                "unexpected MulCP imaginary part at slot " + std::to_string(i));
    }
}

void test_rescale_and_value_validation(PoseidonGpuApi &api,
                                       poseidon::PoseidonContext &context,
                                       poseidon::KeyGenerator &key_generator,
                                       const fhegpu::LoadedOperatorSpec &loaded_spec,
                                       const RmmPoolScope &rmm_pool)
{
    api.preflight(kPlanSha, false, make_target(loaded_spec), loaded_spec.spec, {});

    poseidon::PublicKey public_key;
    key_generator.create_public_key(public_key);
    poseidon::Encryptor encryptor(context, public_key);
    poseidon::Decryptor decryptor(context, key_generator.secret_key());
    poseidon::CKKSEncoder encoder(context);
    poseidon::EvaluatorCkksSoftware cpu_evaluator(context);

    constexpr int input_scale_log2 = 240;
    poseidon::Plaintext plain;
    encoder.encode(std::vector<double>{1.0, -2.0, 3.0, 4.0},
                   std::ldexp(1.0, input_scale_log2), plain);
    poseidon::Ciphertext cipher;
    encryptor.encrypt(plain, cipher);

    poseidon::Ciphertext expected_lazy = cipher;
    for (int dropped = 0; dropped < 5; ++dropped)
    {
        poseidon::Ciphertext next;
        cpu_evaluator.rescale(expected_lazy, next);
        expected_lazy = std::move(next);
    }
    const int input_level = static_cast<int>(cipher.level());
    auto device_value = transfer_value(
        api, 10, PoseidonGpuValue::from_host_ciphertext(cipher),
        fhegpu::ValueKind::Ciphertext, host_place(), device_place());
    api.validate_value(
        device_value,
        {10, fhegpu::ValueKind::Ciphertext, device_place(), kContextId, input_level,
         input_scale_log2, true, 2});

    fhegpu::ComputeOp negate;
    negate.kind = fhegpu::ComputeKind::Negate;
    negate.place = device_place();
    auto double_negated = api.compute(negate, {api.compute(negate, {device_value})});
    const std::size_t bytes_before_synchronize = rmm_pool.allocated_bytes();
    api.synchronize(double_negated);
    const std::size_t bytes_after_synchronize = rmm_pool.allocated_bytes();
    require(bytes_after_synchronize == bytes_before_synchronize,
            "synchronize unexpectedly released input storage");
    api.synchronize(double_negated);
    require(rmm_pool.allocated_bytes() == bytes_after_synchronize,
            "repeated synchronize changed completed input storage");
    api.validate_value(
        double_negated,
        {15, fhegpu::ValueKind::Ciphertext, device_place(), kContextId, input_level,
         input_scale_log2, true, 2});
    auto double_negated_host =
        transfer_value(api, 11, double_negated, fhegpu::ValueKind::Ciphertext,
                       device_place(), host_place());
    poseidon::Plaintext double_negated_plain;
    decryptor.decrypt(double_negated_host.host_ciphertext(), double_negated_plain);
    std::vector<std::complex<double>> double_negated_slots;
    encoder.decode(double_negated_plain, double_negated_slots);
    const std::vector<double> original_slots{1.0, -2.0, 3.0, 4.0};
    for (std::size_t i = 0; i < original_slots.size(); ++i)
    {
        require(std::abs(double_negated_slots[i].real() - original_slots[i]) < 1e-4,
                "asynchronous double Negate result mismatch");
    }

    fhegpu::ComputeOp ordinary;
    ordinary.kind = fhegpu::ComputeKind::Rescale;
    ordinary.place = device_place();
    ordinary.attrs = fhegpu::RescaleAttrs{input_level - 2, input_scale_log2 - 30};
    auto ordinary_result = api.compute(ordinary, {device_value});
    api.validate_value(
        ordinary_result,
        {11, fhegpu::ValueKind::Ciphertext, device_place(), kContextId, input_level - 2,
         input_scale_log2 - 30, true, 2});

    fhegpu::ComputeOp lazy = ordinary;
    lazy.attrs = fhegpu::RescaleAttrs{input_level - 4, input_scale_log2 - 120};
    require_rejected([&]{(void)api.compute(lazy,{device_value});},"modulus/precision space");
    lazy.attrs = fhegpu::RescaleAttrs{input_level - 5, input_scale_log2 - 120};
    auto lazy_result = api.compute(lazy, {device_value});
    api.validate_value(
        lazy_result,
        {12, fhegpu::ValueKind::Ciphertext, device_place(), kContextId, input_level - 5,
         input_scale_log2 - 120, true, 2});

    auto lazy_host =
        transfer_value(api, 12, lazy_result, fhegpu::ValueKind::Ciphertext,
                       device_place(), host_place());
    poseidon::Plaintext lazy_plain;
    poseidon::Plaintext expected_lazy_plain;
    decryptor.decrypt(lazy_host.host_ciphertext(), lazy_plain);
    decryptor.decrypt(expected_lazy, expected_lazy_plain);
    std::vector<std::complex<double>> lazy_slots;
    std::vector<std::complex<double>> expected_lazy_slots;
    encoder.decode(lazy_plain, lazy_slots);
    encoder.decode(expected_lazy_plain, expected_lazy_slots);
    for (std::size_t i = 0; i < lazy_slots.size(); ++i)
    {
        require(std::abs(lazy_slots[i].real() - expected_lazy_slots[i].real()) < 1e-4,
                "GPU/CPU lazy Rescale mismatch at slot " + std::to_string(i) +
                    ": CPU " + std::to_string(expected_lazy_slots[i].real()) +
                    ", GPU " + std::to_string(lazy_slots[i].real()));
        const auto dense_expected=i<original_slots.size()?original_slots[i]:0.0;
        require(std::abs(lazy_slots[i]-dense_expected) < 1e-4,
                "lazy Rescale result mismatch at slot " + std::to_string(i) +
                    ": expected " + std::to_string(dense_expected) +
                    ", got " + std::to_string(lazy_slots[i].real()));
    }

    fhegpu::ComputeOp mod_switch;
    mod_switch.kind = fhegpu::ComputeKind::ModSwitch;
    mod_switch.place = device_place();
    mod_switch.attrs = fhegpu::ModSwitchAttrs{input_level - 1};
    auto mod_switched = api.compute(mod_switch, {device_value});
    api.validate_value(
        mod_switched,
        {13, fhegpu::ValueKind::Ciphertext, device_place(), kContextId,
         input_level - 1, input_scale_log2, true, 2});

    auto mod_switched_host =
        transfer_value(api, 14, mod_switched, fhegpu::ValueKind::Ciphertext,
                       device_place(), host_place());
    poseidon::Plaintext mod_switched_plain;
    decryptor.decrypt(mod_switched_host.host_ciphertext(), mod_switched_plain);
    std::vector<std::complex<double>> mod_switched_slots;
    encoder.decode(mod_switched_plain, mod_switched_slots);
    for (std::size_t i = 0; i < original_slots.size(); ++i)
    {
        require(std::abs(mod_switched_slots[i].real() - original_slots[i]) < 1e-4,
                "ModSwitch result mismatch at slot " + std::to_string(i));
    }

    auto wrong_scale = poseidon::gpu::GpuUploader::upload_ciphertext(cipher, kDeviceId);
    wrong_scale.meta.scale *= 2.0;
    require_rejected(
        [&] {
            api.validate_value(
                PoseidonGpuValue::from_device_ciphertext(std::move(wrong_scale)),
                {13, fhegpu::ValueKind::Ciphertext, device_place(), kContextId, input_level,
                 input_scale_log2, true, 2});
        },
        "metadata");

    auto wrong_device = poseidon::gpu::GpuUploader::upload_ciphertext(cipher, kDeviceId);
    wrong_device.fields_.front().device_id = kDeviceId + 1;
    require_rejected(
        [&] {
            api.validate_value(
                PoseidonGpuValue::from_device_ciphertext(std::move(wrong_device)),
                {14, fhegpu::ValueKind::Ciphertext, device_place(), kContextId, input_level,
                 input_scale_log2, true, 2});
        },
        "configured CUDA device");
}

void test_multiply_relinearize_rescale_rotate(
    PoseidonGpuApi &api, poseidon::PoseidonContext &context,
    poseidon::KeyGenerator &key_generator, const poseidon::RelinKeys &relin_keys,
    const poseidon::GaloisKeys &galois_keys,
    const fhegpu::LoadedOperatorSpec &loaded_spec,
    const RmmPoolScope &rmm_pool)
{
    const auto device = device_place();
    const int full_level =
        static_cast<int>(context.parameters_literal()->q().size() - 1);
    fhegpu::PlanRequirements requirements;
    requirements.keys = {
        {fhegpu::KeyKind::Relin, device, std::nullopt, full_level},
        {fhegpu::KeyKind::Galois, device, 3, full_level - 1},
    };
    const std::size_t before_preflight = rmm_pool.allocated_bytes();
    api.preflight(kPlanSha, false, make_target(loaded_spec), loaded_spec.spec,
                  requirements);
    const std::size_t after_preflight = rmm_pool.allocated_bytes();
    require(after_preflight > before_preflight,
            "GPU preflight did not materialize level-specific evaluation keys");
    api.preflight(kPlanSha, false, make_target(loaded_spec), loaded_spec.spec,
                  requirements);
    require(rmm_pool.allocated_bytes() == after_preflight,
            "GPU preflight materialized duplicate evaluation-key caches");

    poseidon::PublicKey public_key;
    key_generator.create_public_key(public_key);
    poseidon::Encryptor encryptor(context, public_key);
    poseidon::Decryptor decryptor(context, key_generator.secret_key());
    poseidon::CKKSEncoder encoder(context);
    poseidon::EvaluatorCkksSoftware cpu_evaluator(context);

    poseidon::Plaintext plain0;
    poseidon::Plaintext plain1;
    encoder.encode(std::vector<double>{1.0, -2.0, 3.0, 4.5},
                   std::ldexp(1.0, kDefaultScaleLog2), plain0);
    encoder.encode(std::vector<double>{0.5, 1.5, -1.0, 2.0},
                   std::ldexp(1.0, kDefaultScaleLog2), plain1);
    poseidon::Ciphertext cipher0;
    poseidon::Ciphertext cipher1;
    encryptor.encrypt(plain0, cipher0);
    encryptor.encrypt(plain1, cipher1);

    poseidon::Ciphertext expected_multiply;
    poseidon::Ciphertext expected_relinearize;
    poseidon::Ciphertext expected_rescale;
    poseidon::Ciphertext expected_rotate;
    cpu_evaluator.multiply(cipher0, cipher1, expected_multiply);
    cpu_evaluator.relinearize(expected_multiply, expected_relinearize, relin_keys);
    expected_rescale=expected_relinearize;
    for(int dropped=0;dropped<4;++dropped) {
        poseidon::Ciphertext next;
        cpu_evaluator.rescale(expected_rescale,next);expected_rescale=std::move(next);
    }
    poseidon::Ciphertext expected_rotate_once;
    cpu_evaluator.rotate(expected_rescale, expected_rotate_once, 1, galois_keys);
    cpu_evaluator.rotate(expected_rotate_once, expected_rotate, 2, galois_keys);

    auto gpu0 = transfer_value(api, 20, PoseidonGpuValue::from_host_ciphertext(cipher0),
                               fhegpu::ValueKind::Ciphertext, host_place(), device);
    auto gpu1 = transfer_value(api, 21, PoseidonGpuValue::from_host_ciphertext(cipher1),
                               fhegpu::ValueKind::Ciphertext, host_place(), device);

    fhegpu::ComputeOp multiply;
    multiply.kind = fhegpu::ComputeKind::MulCC;
    multiply.place = device;
    auto multiplied = api.compute(multiply, {gpu0, gpu1});

    fhegpu::ComputeOp relinearize;
    relinearize.kind = fhegpu::ComputeKind::Relinearize;
    relinearize.place = device;
    auto relinearized = api.compute(relinearize, {multiplied});

    fhegpu::ComputeOp rescale;
    rescale.kind = fhegpu::ComputeKind::Rescale;
    rescale.place = device;
    rescale.attrs = fhegpu::RescaleAttrs{static_cast<int>(cipher0.level()) - 4,
                                        kDefaultScaleLog2};
    auto rescaled = api.compute(rescale, {relinearized});

    fhegpu::ComputeOp rotate;
    rotate.kind = fhegpu::ComputeKind::Rotate;
    rotate.place = device;
    rotate.attrs = fhegpu::RotateAttrs{3};
    auto rotated = api.compute(rotate, {rescaled});
    auto rotated_replay = api.compute(rotate, {rescaled});
    auto downloaded = transfer_value(api, 22, rotated, fhegpu::ValueKind::Ciphertext,
                                     device, host_place());
    auto downloaded_replay = transfer_value(
        api, 23, rotated_replay, fhegpu::ValueKind::Ciphertext,
        device, host_place());

    poseidon::Plaintext expected_plain;
    poseidon::Plaintext actual_plain;
    poseidon::Plaintext replay_plain;
    decryptor.decrypt(expected_rotate, expected_plain);
    decryptor.decrypt(downloaded.host_ciphertext(), actual_plain);
    decryptor.decrypt(downloaded_replay.host_ciphertext(), replay_plain);
    std::vector<std::complex<double>> expected;
    std::vector<std::complex<double>> actual;
    std::vector<std::complex<double>> replay;
    encoder.decode(expected_plain, expected);
    encoder.decode(actual_plain, actual);
    encoder.decode(replay_plain, replay);
    const std::vector<double> dense_product{.5,-3,-3,9};
    for (std::size_t i = 0; i < actual.size(); ++i)
    {
        const auto original_index=(i+3)%actual.size();
        const auto dense_expected=original_index<dense_product.size()?dense_product[original_index]:0.0;
        require(std::abs(actual[i]-dense_expected)<1e-4,"normalized MulCC/Rescale/Rotate differs from dense arithmetic");
        require(std::abs(actual[i].real() - expected[i].real()) < 1e-4,
                "GPU chain result mismatch at slot " + std::to_string(i));
        require(std::abs(actual[i].imag() - expected[i].imag()) < 1e-4,
                "GPU chain imaginary mismatch at slot " + std::to_string(i));
        require(std::abs(replay[i].real() - expected[i].real()) < 1e-4,
                "GPU chain replay mismatch at slot " + std::to_string(i));
        require(std::abs(replay[i].imag() - expected[i].imag()) < 1e-4,
                "GPU chain replay imaginary mismatch at slot " +
                    std::to_string(i));
    }
}

void test_runtime_reuse(const poseidon::PoseidonContext &context,
                        poseidon::KeyGenerator &keys,
                        const fhegpu::LoadedOperatorSpec &spec,
                        const RmmPoolScope &pool)
{
    auto rotations = std::make_shared<poseidon::GaloisKeys>();
    keys.create_galois_keys(std::vector<int>{1, 2, -1, -2}, *rotations);
    poseidon::PublicKey public_key; keys.create_public_key(public_key);
    poseidon::CKKSEncoder encoder(context);
    poseidon::Encryptor encryptor(context, public_key);
    poseidon::Decryptor decryptor(context, keys.secret_key());
    poseidon::Plaintext plain;
    encoder.encode(std::vector<double>{1, -2, 3, 4, 5, 6}, std::ldexp(1.0, kDefaultScaleLog2), plain);
    poseidon::Ciphertext cipher; encryptor.encrypt(plain, cipher);
    std::unordered_map<fhegpu::ValueId, PoseidonGpuValue> inputs;
    inputs.emplace(0, PoseidonGpuValue::from_host_ciphertext(cipher));
    const auto decode = [&](const poseidon::Ciphertext &value) {
        poseidon::Plaintext plaintext; decryptor.decrypt(value, plaintext);
        std::vector<std::complex<double>> slots; encoder.decode(plaintext, slots); return slots;
    };
    const auto original = decode(cipher);
    class ObservedApi : public PoseidonGpuApi {
    public:
        using PoseidonGpuApi::PoseidonGpuApi;
        const RmmPoolScope *pool = nullptr;
        std::vector<const void *> pointers;
        int reuses = 0;
        Value compute_reuse(const fhegpu::ComputeOp &op, Value input, const std::vector<Value> &others) {
            std::vector<const void *> before;
            for (const auto &field : input.device_ciphertext().fields_) before.push_back(field.data());
            if (pointers.empty()) pointers = before;
            require(before == pointers, "GPU reuse chain changed ciphertext storage");
            // Completion collection may free old work during this call. Count
            // allocations independently of those changes in active bytes.
            const auto allocated = pool->total_allocated_bytes();
            auto output = PoseidonGpuApi::compute_reuse(op, std::move(input), others);
            std::vector<const void *> after;
            for (const auto &field : output.device_ciphertext().fields_) after.push_back(field.data());
            require(after == before, "GPU reuse allocated another final ciphertext");
            if (op.kind != fhegpu::ComputeKind::Rotate)
                require(pool->total_allocated_bytes() == allocated, "AddCP/SubCP reuse allocated ciphertext storage");
            ++reuses; return output;
        }
    };
    for (int variant = 0; variant < 4; ++variant) {
        auto plan = make_add_plain_plan(spec, static_cast<int>(cipher.level()));
        plan.format_version = 2;
        for (fhegpu::ValueId id = 6; id <= 8; ++id) {
            auto desc = plan.values.at(4); desc.id = id; plan.values.push_back(desc);
        }
        const std::vector<int> steps = variant == 2 ? std::vector<int>{1, 2, -3} : std::vector<int>{1, 3, -3};
        for (int i = 0; i < 3; ++i) {
            const auto kind = variant == 0 ? fhegpu::ComputeKind::AddCP :
                variant == 1 ? fhegpu::ComputeKind::SubCP : fhegpu::ComputeKind::Rotate;
            fhegpu::ComputeOp op{kind, {static_cast<fhegpu::ValueId>(i == 0 ? 4 : i + 5)},
                static_cast<fhegpu::ValueId>(i + 6), device_place(), {}};
            if (variant < 2) op.inputs.push_back(3);
            else op.attrs = fhegpu::RotateAttrs{steps[i]};
            op.reuse_input = 0; plan.execution.push_back({0, op});
        }
        std::get<fhegpu::CommAction>(plan.finalization.front().body).inputs = {8};
        plan.initialization.push_back({0, fhegpu::ReleaseOp{0}});
        plan.initialization.push_back({0, fhegpu::ReleaseOp{1}});
        plan.execution.insert(plan.execution.begin() + 1, {0, fhegpu::ReleaseOp{2}});
        plan.execution.push_back({0, fhegpu::ReleaseOp{3}});
        plan.finalization.push_back({0, fhegpu::ReleaseOp{8}});
        std::uint64_t ordinal = 0;
        for (auto *phase : {&plan.initialization, &plan.execution, &plan.finalization})
            for (auto &instruction : *phase) instruction.ordinal = ordinal++;
        auto baseline = plan;
        for (auto *phase : {&baseline.initialization, &baseline.execution, &baseline.finalization})
            for (auto &instruction : *phase)
                if (auto *op = std::get_if<fhegpu::ComputeOp>(&instruction.body)) op->reuse_input.reset();
        for (auto mode : {fhegpu::DeviceExecutionMode::Sequential, fhegpu::DeviceExecutionMode::PerDeviceWorkers}) {
            ObservedApi api(kContextId, context, kDeviceId, {}, rotations); api.pool = &pool;
            fhegpu::SequentialRuntime<ObservedApi> runtime(0, 1, 1, api, mode);
            const auto expected = decode(runtime.run({baseline, kPlanSha}, {spec, {}, false}, inputs).values.at(5).value.host_ciphertext());
            const auto bytes = pool.allocated_bytes();
            for (int iteration = 0; iteration < 3; ++iteration) {
                api.pointers.clear();
                const auto actual = decode(runtime.run({plan, kPlanSha}, {spec, {}, false}, inputs).values.at(5).value.host_ciphertext());
                for (std::size_t i = 0; i < actual.size(); ++i)
                    require(std::abs(actual[i] - expected[i]) < 1e-4, "GPU reuse differs from ordinary result");
                require(pool.allocated_bytes() == bytes, "GPU reuse retained storage between runs");
                require(decode(inputs.at(0).host_ciphertext()) == original, "GPU reuse changed caller input");
            }
            require(api.reuses == 9, "GPU reuse chain did not dispatch three operations per run");
        }
    }
    // Exercise zero rotation directly; RuntimePlan keeps its nonzero-step rule.
    {
        ObservedApi api(kContextId, context, kDeviceId, {}, rotations); api.pool = &pool;
        auto uploaded = transfer_value(api, 77, inputs.at(0), fhegpu::ValueKind::Ciphertext, host_place(), device_place());
        fhegpu::ComputeOp zero{fhegpu::ComputeKind::Rotate, {0}, 1, device_place(), fhegpu::RotateAttrs{0}};
        zero.reuse_input = 0;
        auto result = api.compute_reuse(zero, std::move(uploaded), {});
        api.synchronize(result);
        auto host = transfer_value(api, 78, result, fhegpu::ValueKind::Ciphertext, device_place(), host_place());
        require(decode(host.host_ciphertext()) == original, "GPU zero rotation changed result");
        api.drain();
    }
    // A consumer on another CUDA stream must observe the new completion event.
    {
        PoseidonGpuApi api(kContextId, context, kDeviceId, {}, rotations);
        const auto baseline = pool.allocated_bytes();
        {
            auto uploaded = transfer_value(api, 79, inputs.at(0), fhegpu::ValueKind::Ciphertext, host_place(), device_place());
            const fhegpu::ValueDesc weight_desc{1, fhegpu::ValueKind::Plaintext, device_place(), kContextId,
                static_cast<int>(cipher.level()), kDefaultScaleLog2, true, 1};
            auto weight_handle = api.communicate_async(transfer_action(80, fhegpu::ValueKind::Plaintext, host_place(), device_place()),
                {PoseidonGpuValue::from_host_plaintext(plain)}, {weight_desc});
            auto weight = std::move(*api.wait(weight_handle).front());
            api.drain();
            CudaHostGate gate;
            CudaGateWatchdog watchdog(gate);
            require(cudaLaunchHostFunc(cudaStreamPerThread, wait_for_cuda_host_gate, &gate) == cudaSuccess, "reuse gate failed");
            fhegpu::ComputeOp add{fhegpu::ComputeKind::AddCP, {0, 1}, 2, device_place(), {}};
            add.reuse_input = 0;
            auto output = api.compute_reuse(add, std::move(uploaded), {weight});
            const auto pending = pool.allocated_bytes();
            auto consumer = std::async(std::launch::async, [&] {
                require(cudaSetDevice(kDeviceId) == cudaSuccess, "consumer device selection failed");
                auto host = transfer_value(api, 81, output, fhegpu::ValueKind::Ciphertext, device_place(), host_place());
                return decode(host.host_ciphertext());
            });
            require(consumer.wait_for(std::chrono::milliseconds(50)) == std::future_status::timeout,
                    "reuse consumer observed the old ready event");
            api.collect_completed();
            require(pool.allocated_bytes() == pending, "unfinished reuse lost its retained storage");
            require(watchdog.finish_post(), "reuse submission waited for completion");
            const auto actual = consumer.get();
            for (std::size_t i = 0; i < actual.size(); ++i)
                require(std::abs(actual[i] - original[i] * 2.0) < 1e-4, "gated reuse result mismatch");
            api.drain();
        }
        require(pool.allocated_bytes() == baseline, "gated reuse retained storage after completion");
    }
    // Check the inverse-pre-rotated key branch also writes into the original block.
    {
        poseidon::gpu::GpuParameterData parameters(context, kDeviceId);
        poseidon::gpu::GpuEvaluator evaluator(parameters);
        auto pre_rotated = poseidon::gpu::GpuUploader::upload_double_hoist_galois_keys(*rotations, kDeviceId);
        poseidon::Plaintext precise_plain;
        encoder.encode(std::vector<double>{1, -2, 3, 4, 5, 6}, std::ldexp(1.0, 40), precise_plain);
        poseidon::Ciphertext precise_cipher; encryptor.encrypt(precise_plain, precise_cipher);
        auto value = poseidon::gpu::GpuUploader::upload_ciphertext(precise_cipher, kDeviceId);
        const auto *allocation = value.fields_.front().data();
        poseidon::gpu::GpuRotateWorkspace workspace;
        poseidon::Ciphertext expected = precise_cipher;
        auto cpu_context = context;
        poseidon::EvaluatorCkksSoftware cpu(cpu_context);
        for (int step : {1, -1, 2}) {
            cpu.rotate_inplace(expected, step, *rotations);
            poseidon::gpu::GpuCiphertextData ordinary;
            evaluator.rotate(value, step, pre_rotated, ordinary);
            evaluator.rotate_inplace(value, step, pre_rotated, workspace);
            require(value.fields_.front().data() == allocation, "pre-rotated reuse changed final allocation");
            require(cudaStreamSynchronize(cudaStreamPerThread) == cudaSuccess, "rotation synchronization failed");
            poseidon::Ciphertext normal, reused;
            poseidon::gpu::GpuUploader::download_ciphertext(ordinary, normal, context);
            poseidon::gpu::GpuUploader::download_ciphertext(value, reused, context);
            const auto normal_slots = decode(normal), reused_slots = decode(reused);
            double maximum = 0;
            for (std::size_t i = 0; i < reused_slots.size(); ++i)
                maximum = std::max(maximum, std::abs(normal_slots[i] - reused_slots[i]));
            require(maximum < 1e-4, "pre-rotated inplace differs from ordinary step " + std::to_string(step) + " error=" + std::to_string(maximum));
        }
        require(cudaStreamSynchronize(cudaStreamPerThread) == cudaSuccess, "rotation synchronization failed");
        poseidon::Ciphertext actual;
        poseidon::gpu::GpuUploader::download_ciphertext(value, actual, context);
        const auto expected_slots = decode(expected); const auto actual_slots = decode(actual);
        for (std::size_t i = 0; i < actual_slots.size(); ++i)
            require(std::abs(actual_slots[i] - expected_slots[i]) < 1e-4, "pre-rotated inplace numerical mismatch slot=" + std::to_string(i) + " error=" + std::to_string(std::abs(actual_slots[i] - expected_slots[i])));
    }
}

void test_two_gpu_runtime_plan(const poseidon::PoseidonContext &context,
                               poseidon::KeyGenerator &key_generator,
                               const fhegpu::LoadedOperatorSpec &loaded_spec,
                               int second_device_id = 1)
{
    RmmPoolScope second_device_pool(second_device_id);
    PoseidonGpuApi api(kContextId, context,
                       std::vector<int>{second_device_id, kDeviceId});
    require(api.local_device_count() == 2, "two-GPU Api device count is incorrect");
    require(api.cuda_device_id(0) == second_device_id &&
                api.cuda_device_id(1) == kDeviceId,
            "logical CUDA device mapping is incorrect");

    poseidon::PublicKey public_key;
    key_generator.create_public_key(public_key);
    poseidon::Encryptor encryptor(context, public_key);
    poseidon::Decryptor decryptor(context, key_generator.secret_key());
    poseidon::CKKSEncoder encoder(context);

    const std::vector<double> input{1.0, -2.0, 3.0, 4.0};
    const std::vector<double> addend{0.5, -1.0, 2.0, 3.0};
    poseidon::Plaintext input_plain;
    encoder.encode(input, std::ldexp(1.0, kDefaultScaleLog2), input_plain);
    poseidon::Ciphertext input_cipher;
    encryptor.encrypt(input_plain, input_cipher);

    const int level = static_cast<int>(context.parameters_literal()->q().size() - 1);
    const auto ordinary = make_two_gpu_plan(loaded_spec, level);
    auto release_plan = ordinary;
    release_plan.format_version = 2;
    // Move Replicate online so per-device workers split its two sends.
    release_plan.initialization.erase(release_plan.initialization.begin() + 1);
    release_plan.initialization.push_back({0, fhegpu::ReleaseOp{1}});
    release_plan.execution = {
        {0, ordinary.initialization.at(1).body}, {0, fhegpu::ReleaseOp{0}},
        {0, ordinary.execution.at(0).body}, {0, fhegpu::ReleaseOp{2}}, {0, fhegpu::ReleaseOp{4}},
        {0, ordinary.execution.at(1).body}, {0, fhegpu::ReleaseOp{3}}, {0, fhegpu::ReleaseOp{5}},
        {0, ordinary.execution.at(2).body}, {0, fhegpu::ReleaseOp{7}},
        {0, ordinary.execution.at(3).body}, {0, fhegpu::ReleaseOp{6}}, {0, fhegpu::ReleaseOp{8}}};
    release_plan.finalization.push_back({0, fhegpu::ReleaseOp{9}});
    std::size_t ordinal = 0;
    for (auto *phase : {&release_plan.initialization, &release_plan.execution, &release_plan.finalization})
        for (auto &instruction : *phase) instruction.ordinal = ordinal++;
    auto explicit_fence = ordinary;
    explicit_fence.format_version = 3;
    auto second_weight = ordinary.values[1]; second_weight.id = 11;
    explicit_fence.values.push_back(second_weight);
    auto second_encode = std::get<fhegpu::EncodeOp>(ordinary.initialization[0].body);
    second_encode.output = 11;
    auto second_upload = std::get<fhegpu::CommAction>(ordinary.initialization[2].body);
    second_upload.id = 40; second_upload.inputs = {11}; second_upload.outputs = {5};
    second_upload.destinations = {device_place(1)};
    explicit_fence.initialization = {ordinary.initialization[1]};
    explicit_fence.execution = {
        {0, ordinary.initialization[0].body}, {0, ordinary.initialization[2].body},
        {0, fhegpu::ReleaseOp{1}}, {0, ordinary.execution[0].body},
        {0, fhegpu::ReleaseOp{4}}, {0, fhegpu::FenceOp{}},
        {0, second_encode}, {0, second_upload}, {0, fhegpu::ReleaseOp{11}},
        {0, ordinary.execution[1].body}, {0, fhegpu::ReleaseOp{5}},
        {0, fhegpu::FenceOp{}}, {0, ordinary.execution[2].body}, {0, ordinary.execution[3].body}};
    ordinal = 0;
    for (auto *phase : {&explicit_fence.initialization, &explicit_fence.execution, &explicit_fence.finalization})
        for (auto &instruction : *phase) instruction.ordinal = ordinal++;
    const fhegpu::RuntimeResources resources{loaded_spec, std::nullopt, false};
    std::unordered_map<fhegpu::ValueId, PoseidonGpuValue> inputs;
    inputs.emplace(0, PoseidonGpuValue::from_host_ciphertext(std::move(input_cipher)));
    for (const auto &plan : {ordinary, release_plan, explicit_fence})
    {
        for (auto mode : {fhegpu::DeviceExecutionMode::Sequential,
                          fhegpu::DeviceExecutionMode::PerDeviceWorkers})
        {
            fhegpu::SequentialRuntime<PoseidonGpuApi> runtime(0, 1, 2, api, mode);
            for (int iteration = 0; iteration < 3; ++iteration)
            {
                const auto artifact = runtime.run({plan, kPlanSha}, resources, inputs);
                poseidon::Plaintext result_plain;
                decryptor.decrypt(artifact.values.at(10).value.host_ciphertext(), result_plain);
                std::vector<std::complex<double>> result;
                encoder.decode(result_plain, result);
                for (std::size_t i = 0; i < input.size(); ++i)
                {
                    const double expected = 2.0 * (input[i] + addend[i]);
                    require(std::abs(result[i].real() - expected) < 1e-4,
                            "unexpected two-GPU result at slot " + std::to_string(i));
                    require(std::abs(result[i].imag()) < 1e-4,
                            "unexpected two-GPU imaginary part at slot " + std::to_string(i));
                }
            }
        }
    }
}

void test_gpu_plaintext_wide_coefficients(const poseidon::PoseidonContext &context)
{
    PoseidonGpuApi api(kContextId,context,kDeviceId);
    poseidon::CKKSEncoder encoder(context);
    const int level=static_cast<int>(context.parameters_literal()->q().size()-1);
    const auto parms=context.crt_context()->parms_id_map().at(static_cast<std::uint32_t>(level));
    const auto n=context.parameters_literal()->degree();
    for(int scale:{40,60,80,120}) {
        fhegpu::ValueDesc desc{999,fhegpu::ValueKind::Plaintext,device_place(),kContextId,level,scale,true,1};
        const std::vector<double> input(n/2,std::ldexp(1.,20));
        poseidon::Plaintext reference;encoder.encode(input,parms,std::ldexp(1.,scale),reference);
        auto value=api.prepare_plaintext_async(desc,input,fhegpu::PrepareBackend::Gpu);
        api.synchronize(value);poseidon::Plaintext actual;
        poseidon::gpu::GpuUploader::download_plaintext(value.device_plaintext(),actual,context);
        require(reference.coeff_count()==actual.coeff_count(),"wide coefficient shape changed");
        for(std::size_t i=0;i<reference.coeff_count();++i)require(reference.data()[i]==actual.data()[i],"wide constant RNS differs from CPU Encode");
        std::cout<<"PASS complete GPU Encode wide constant scale="<<scale<<" coefficient_bits="<<scale+21<<'\n';
        api.drain();
    }
}

class DelayedWaterlineApi : public PoseidonGpuApi
{
public:
    using PoseidonGpuApi::PoseidonGpuApi;
    std::atomic<bool> delayed{false},waiting{false};
    std::atomic<unsigned> peers_while_waiting{0};
    std::atomic<unsigned> host_staged_posts{0},host_staged_pinned_observed{0};
    void reset_delay() {
        std::lock_guard<std::mutex> lock(delay_mutex_);
        delayed=false;waiting=true;peers_while_waiting=0;peer_devices_.clear();
        host_staged_posts=0;host_staged_pinned_observed=0;
    }
    CommHandle communicate_async(const fhegpu::CommAction &action,const std::vector<Value> &inputs,
                                 const std::vector<fhegpu::ValueDesc> &outputs) {
        const bool staged=action.hint==fhegpu::CommHint::HostStaged;
        if(staged) {
            require(action.sources.size()==1 && action.destinations.size()==1,"HostStaged ring must transfer one object");
            const auto source=cuda_device_id(action.sources.front().index);
            const auto destination=cuda_device_id(action.destinations.front().index);
            require(poseidon::runtime_api::communication::CudaLocalTransfer::select_route(
                destination,source,poseidon::runtime_api::communication::CudaTransferRoute::HostStaged)==
                poseidon::runtime_api::communication::CudaTransferRoute::HostStaged,"ring did not resolve HostStaged route");
        }
        auto handle=PoseidonGpuApi::communicate_async(action,inputs,outputs);
        if(staged) {
            ++host_staged_posts;
            if(poseidon::runtime_api::communication::PinnedHostBuffer::live_bytes()>0)++host_staged_pinned_observed;
        }
        return handle;
    }
    Value compute(const fhegpu::ComputeOp &op,const std::vector<Value> &inputs) {
        if(local_device_count()==1 || op.place.kind!=fhegpu::PlaceKind::Device)
            return PoseidonGpuApi::compute(op,inputs);
        const bool first_gpu0=op.place.index==0 && !delayed.exchange(true);
        if(first_gpu0) {
            std::unique_lock<std::mutex> lock(delay_mutex_);
            if(!delay_condition_.wait_for(lock,std::chrono::seconds(5),[&]{
                return peer_devices_.size()==static_cast<std::size_t>(local_device_count()-1);
            }))throw std::runtime_error("other GPUs did not complete during GPU0 delay");
            lock.unlock();std::this_thread::sleep_for(std::chrono::milliseconds(80));
        }
        auto result=PoseidonGpuApi::compute(op,inputs);
        if(first_gpu0 || (op.place.index!=0 && waiting.load())) {
            // The test observes true device completion, including peers that
            // finish before GPU0's worker enters its first compute call.
            synchronize(result);
            std::lock_guard<std::mutex> lock(delay_mutex_);
            if(first_gpu0)waiting=false;
            else if(waiting.load() && peer_devices_.insert(op.place.index).second) {
                ++peers_while_waiting;delay_condition_.notify_all();
            }
        }
        return result;
    }
private:
    std::mutex delay_mutex_;
    std::condition_variable delay_condition_;
    std::set<int> peer_devices_;
};

void test_gpu_raw_pinned_cache(const poseidon::PoseidonContext &context,
                              const fhegpu::LoadedOperatorSpec &spec,int device_count)
{
    using Pinned=poseidon::runtime_api::communication::PinnedHostBuffer;
    std::vector<int> ids(device_count);for(int d=0;d<device_count;++d)ids[d]=d;
    PoseidonGpuApi api(kContextId,context,ids);
    const auto n=context.parameters_literal()->degree();
    const int level=static_cast<int>(context.parameters_literal()->q().size()-1);
    std::vector<double> slots(n/2);for(std::size_t i=0;i<slots.size();++i)slots[i]=.125*std::sin(i*.071);
    fhegpu::RuntimePlan plan;plan.format_version=4;plan.target=make_target(spec,device_count);
    for(int d=0;d<device_count;++d) {
        plan.values.push_back({static_cast<fhegpu::ValueId>(d),fhegpu::ValueKind::Plaintext,device_place(d),kContextId,level,kDefaultScaleLog2,true,1});
        plan.execution.push_back({static_cast<std::uint64_t>(d),fhegpu::PreparePlaintextOp{fhegpu::InlineEncodePayload{slots},static_cast<fhegpu::ValueId>(d)}});
    }
    fhegpu::WaterlineConfig config;config.backend=fhegpu::PrepareBackend::Gpu;
    config.device_capacity_bytes.assign(device_count,4ULL<<30);config.device_reserve_bytes.assign(device_count,2ULL<<30);
    config.device_fractions.assign(device_count,.02);config.host_budget_bytes=64ULL<<20;
    config.pinned_budget_bytes=16ULL<<20;config.raw_budget_bytes=8ULL<<20;plan.waterline=config;
    api.validate_waterline_resources(plan);
    const auto fixed=api.waterline_fixed_resources();
    require(fixed.host==n*4*device_count && fixed.pinned==fixed.host && fixed.raw==0,
            "GPU raw cache fixed Host/pinned cost is incomplete");
    require(Pinned::live_bytes()==0,"GPU cache test starts with pending pinned buffers");
    const auto allocations= Pinned::allocation_calls(),frees=Pinned::free_calls();
    poseidon::CKKSEncoder encoder(context);double maximum=0;
    for(int step=0;step<10;++step) {
        slots[0]=.125+step*.01;
        for(int d=0;d<device_count;++d) {
            const auto cost=api.prepare_resources(plan.values[d],fhegpu::PrepareBackend::Gpu);
            require(cost.peak.pinned==0 && cost.peak.host==n*8,
                    "GPU cached Prepare omitted coexisting packed byte storage and decoded slots");
            auto value=api.prepare_plaintext_async(plan.values[d],slots,fhegpu::PrepareBackend::Gpu);
            api.synchronize(value);api.collect_completed();
            if(step==9) {
                poseidon::Plaintext host;poseidon::gpu::GpuUploader::download_plaintext(value.device_plaintext(),host,context);
                std::vector<std::complex<double>> actual;encoder.decode(host,actual);
                require(actual.size()==slots.size(),"GPU cache changed slot layout");
                for(std::size_t i=0;i<actual.size();++i)maximum=std::max(maximum,std::abs(actual[i]-slots[i]));
            }
        }
    }
    require(maximum<1e-4,"reused GPU raw pinned buffer encoded stale or invalid slots");
    require(Pinned::allocation_calls()-allocations==static_cast<std::size_t>(device_count),"ten GPU prepares did not share one pinned allocation per device");
    require(Pinned::free_calls()==frees && Pinned::live_bytes()==fixed.pinned,"cached pinned storage was released before drain");
    auto cpu_plan=plan;cpu_plan.waterline->backend=fhegpu::PrepareBackend::Cpu;
    require_rejected([&]{api.validate_waterline_resources(cpu_plan);},"drain cached GPU Prepare");
    require(api.waterline_fixed_resources().pinned==fixed.pinned,"rejected reconfiguration lost fixed pinned accounting");
    // A deterministic pending-input counterexample, using only public RMM
    // hooks: the actual encoder stream is blocked before its raw H2D.
    require(cudaSetDevice(0)==cudaSuccess,"cache gate device selection failed");
    {
        CudaHostGate gate;CudaGateWatchdog watchdog(gate);
        auto *upstream=rmm::mr::get_current_device_resource();
        GatedAllocationResource resource(upstream,gate);
        struct RestoreResource {
            rmm::mr::device_memory_resource *original;
            ~RestoreResource(){rmm::mr::set_current_device_resource(original);}
        } restore{upstream};
        rmm::mr::set_current_device_resource(&resource);
        auto pending=api.prepare_plaintext_async(plan.values[0],slots,fhegpu::PrepareBackend::Gpu);
        require(!api.is_ready(pending),"GPU Prepare gate did not leave an unfinished encode");
        require_rejected([&]{(void)api.prepare_plaintext_async(plan.values[0],slots,fhegpu::PrepareBackend::Gpu);},"still in use");
        require_rejected([&]{api.validate_waterline_resources(cpu_plan);},"drain cached GPU Prepare");
        require(watchdog.finish_post(),"cached GPU Prepare submission waited for the allocation gate");
        api.synchronize(pending);api.collect_completed();
        require_rejected([&]{api.validate_waterline_resources(cpu_plan);},"drain cached GPU Prepare");
    }
    api.drain();
    require(Pinned::free_calls()-frees==static_cast<std::size_t>(device_count) && Pinned::live_bytes()==0,
            "drain did not reclaim all cached GPU raw pinned buffers");
    require(api.waterline_fixed_resources().pinned==fixed.pinned,"drain changed the whole-run fixed reservation");
    api.validate_waterline_resources(cpu_plan);
    require(api.waterline_fixed_resources().host==0 && api.waterline_fixed_resources().pinned==0,
            "CPU-only reconfiguration retained GPU pinned reservation");
    std::cout<<"GPU_RAW_PINNED_CACHE devices="<<device_count<<" sequential_prepares_per_device=10"
             <<" allocations="<<Pinned::allocation_calls()-allocations<<" frees="<<Pinned::free_calls()-frees
             <<" fixed_host="<<fixed.host<<" fixed_pinned="<<fixed.pinned<<" pinned_live="<<Pinned::live_bytes()
             <<" max_abs_including_imaginary="<<maximum<<" overlap_rejected=true ready_reconfigure_requires_drain=true\n";
}

void test_gpu_packed_prepare_host_budget(const poseidon::PoseidonContext &context,
                                         const fhegpu::LoadedOperatorSpec &spec)
{
    using Pinned=poseidon::runtime_api::communication::PinnedHostBuffer;
    PoseidonGpuApi api(kContextId,context,kDeviceId);
    const auto n=context.parameters_literal()->degree();
    const auto level=static_cast<int>(context.parameters_literal()->q().size()-1);
    const std::uint64_t fixed_host=4*n,read_host=8*n;
    const std::uint64_t tight_budget=10*n,sufficient_budget=fixed_host+read_host;
    const auto directory=std::filesystem::temp_directory_path()/
        ("poseidon-gpu-packed-host-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    require(std::filesystem::create_directory(directory),"cannot create packed Host-budget fixture directory");
    struct RemoveFixture {
        std::filesystem::path path;
        ~RemoveFixture(){std::error_code error;std::filesystem::remove_all(path,error);}
    } cleanup{directory};
    std::vector<double> first(n/2),second(n/2);
    for(std::size_t i=0;i<first.size();++i) {
        first[i]=.125*std::cos(i*.137);
        second[i]=.0625*std::sin(i*.191);
    }
    std::ofstream pack(directory/"data.bin",std::ios::binary);
    const auto write_payload=[&](const std::vector<double> &values) {
        std::string bytes(values.size()*8,'\0');
        for(std::size_t i=0;i<values.size();++i) {
            std::uint64_t bits;std::memcpy(&bits,&values[i],sizeof(bits));
            for(int b=0;b<8;++b)bytes[i*8+b]=static_cast<char>((bits>>(b*8))&255);
        }
        pack.write(bytes.data(),static_cast<std::streamsize>(bytes.size()));
        return "sha256:"+fhegpu::sha256_hex(bytes);
    };
    const auto first_content=write_payload(first),second_content=write_payload(second);
    pack.close();require(bool(pack),"cannot write packed Host-budget fixture");
    const std::uint64_t payload_bytes=first.size()*sizeof(double);
    std::ostringstream manifest;
    manifest<<"{\"bundle_format_version\":2,\"bundle_id\":\"gpu-packed-host\",\"version\":1,\"pack_byte_length\":"
            <<2*payload_bytes<<",\"blobs\":[{\"content\":\""<<first_content<<"\",\"offset\":0,\"byte_length\":"
            <<payload_bytes<<"},{\"content\":\""<<second_content<<"\",\"offset\":"<<payload_bytes
            <<",\"byte_length\":"<<payload_bytes<<"}]}";
    std::ofstream manifest_file(directory/"manifest.json");manifest_file<<manifest.str();manifest_file.close();
    require(bool(manifest_file),"cannot write packed Host-budget manifest");
    const fhegpu::PlaintextBundleRef reference{"gpu-packed-host",1,"sha256:"+fhegpu::sha256_hex(manifest.str())};
    auto loader=fhegpu::PlaintextBundleLoader::open(directory,reference,{first_content,second_content},n/2,false);
    require(loader.resident_bytes()==0,"packed Host-budget fixture unexpectedly uses resident mode");
    fhegpu::RuntimePlan plan;plan.format_version=4;plan.target=make_target(spec);
    plan.values={{0,fhegpu::ValueKind::Plaintext,device_place(),kContextId,level,kDefaultScaleLog2,true,1}};
    plan.execution={{0,fhegpu::PreparePlaintextOp{fhegpu::BundleEncodePayload{first_content},0}}};
    fhegpu::WaterlineConfig config;config.backend=fhegpu::PrepareBackend::Gpu;
    config.device_capacity_bytes={4ULL<<30};config.device_reserve_bytes={2ULL<<30};config.device_fractions={.02};
    config.host_budget_bytes=sufficient_budget;config.pinned_budget_bytes=fixed_host;config.raw_budget_bytes=read_host;
    plan.waterline=config;api.validate_waterline_resources(plan);
    const auto fixed=api.waterline_fixed_resources();
    require(fixed.host==fixed_host && fixed.pinned==fixed_host,"packed cache fixed reservation is incomplete");
    require(Pinned::live_bytes()==0,"packed Host-budget test starts with pending pinned storage");
    const auto allocations=Pinned::allocation_calls(),frees=Pinned::free_calls();
    {
        auto slots=loader.read(first_content);
        auto warm=api.prepare_plaintext_async(plan.values[0],slots,fhegpu::PrepareBackend::Gpu);
        api.synchronize(warm);api.collect_completed();
    }
    require(Pinned::allocation_calls()-allocations==1 && Pinned::live_bytes()==fixed_host,
            "first packed Prepare did not leave one persistent pinned cache");
    const auto cost=api.prepare_resources(plan.values[0],fhegpu::PrepareBackend::Gpu);
    auto tight_config=config;tight_config.host_budget_bytes=tight_budget;
    fhegpu::WaterlineLedger tight(tight_config);tight.reserve(fixed);
    std::cout<<"GPU_PACKED_PREPARE_HOST_BOUND N="<<n<<" fixed_host="<<fixed.host<<" dynamic_host="<<cost.peak.host
             <<" tight_budget="<<tight_budget<<" complete_budget="<<sufficient_budget
             <<" warm_cache_live="<<Pinned::live_bytes()<<" tight_admitted="<<tight.fits(cost.peak)<<'\n';
    require(!tight.fits(cost.peak),"cached GPU packed-read Host boundary incorrectly admitted the tight budget");
    fhegpu::WaterlineLedger sufficient(config);sufficient.reserve(fixed);
    require(sufficient.fits(cost.peak),"complete cached GPU packed-read budget was rejected");
    sufficient.reserve(cost.peak);
    std::optional<PoseidonGpuValue> encoded;
    {
        // The loader owns byte storage while decode_slots allocates its result.
        // The first Prepare's pinned cache remains live throughout this read.
        require(Pinned::live_bytes()==fixed_host,"warm GPU cache disappeared before the second packed read");
        auto slots=loader.read(second_content);
        require(slots==second,"second packed payload was decoded incorrectly");
        encoded.emplace(api.prepare_plaintext_async(plan.values[0],slots,fhegpu::PrepareBackend::Gpu));
        api.synchronize(*encoded);api.collect_completed();
    }
    auto temporary=cost.peak;temporary-=cost.resident;sufficient.release(temporary);
    poseidon::Plaintext host;poseidon::gpu::GpuUploader::download_plaintext(encoded->device_plaintext(),host,context);
    poseidon::CKKSEncoder encoder(context);std::vector<std::complex<double>> actual;encoder.decode(host,actual);
    require(actual.size()==second.size(),"GPU packed-read boundary changed slot count");
    double maximum=0;
    for(std::size_t i=0;i<actual.size();++i) {
        require(std::isfinite(actual[i].real()) && std::isfinite(actual[i].imag()),"GPU packed-read returned nonfinite slots");
        maximum=std::max(maximum,std::abs(actual[i]-second[i]));
    }
    require(maximum<1e-4,"GPU packed-read positive boundary numerical mismatch");
    encoded.reset();sufficient.release(cost.resident);api.drain();
    require(Pinned::allocation_calls()-allocations==1 && Pinned::free_calls()-frees==1 && Pinned::live_bytes()==0,
            "packed positive case did not reuse and drain exactly one GPU pinned cache");
    require(sufficient.peak.host==sufficient_budget && sufficient.used.host==fixed.host,
            "packed positive case did not charge the complete read peak or release temporary Host credit");
    std::cout<<"GPU_PACKED_PREPARE_HOST_BOUND positive_budget="<<sufficient_budget<<" resident_bytes="<<loader.resident_bytes()
             <<" packed_reads=2 second_read_with_cache=true allocations=1 frees=1 slots_checked="<<actual.size()
             <<" max_abs_including_imaginary="<<maximum<<" scope=API_Ledger_boundary_not_full_Runtime\n";
}

void test_cpu_prepare_host_budget(const poseidon::PoseidonContext &context,
                                  poseidon::KeyGenerator &key_generator,
                                  const fhegpu::LoadedOperatorSpec &spec)
{
    PoseidonGpuApi api(kContextId,context,kDeviceId);
    poseidon::PublicKey public_key;key_generator.create_public_key(public_key);
    poseidon::Encryptor encryptor(context,public_key);
    poseidon::Decryptor decryptor(context,key_generator.secret_key());
    poseidon::CKKSEncoder encoder(context);
    const auto n=context.parameters_literal()->degree();
    const auto q=context.parameters_literal()->q().size();
    const std::uint64_t bytes=n*q*sizeof(poseidon::gpu::GpuWord);
    const std::uint64_t old_host_bound=2*bytes+40*n;
    const std::uint64_t complete_host_bound=old_host_bound+bytes;
    // Retain six legal Host Encode outputs after H2D and release them before
    // D2H. This makes Prepare, rather than IO staging, determine the boundary.
    const std::uint64_t prepare_host_resident=16*bytes;
    const std::uint64_t tight_host_budget=prepare_host_resident+old_host_bound+bytes/2;
    const std::uint64_t sufficient_host_budget=prepare_host_resident+complete_host_bound;
    std::vector<double> input(encoder.slot_count()),weight(encoder.slot_count());
    for(std::size_t i=0;i<input.size();++i) {
        input[i]=std::sin(double(i)*.073)*.25;
        weight[i]=std::cos(double(i)*.137)*.125;
    }
    poseidon::Plaintext plain;encoder.encode(input,std::ldexp(1.,kDefaultScaleLog2),plain);
    poseidon::Ciphertext cipher;encryptor.encrypt(plain,cipher);
    const int level=static_cast<int>(cipher.level());
    fhegpu::RuntimePlan plan;plan.format_version=4;plan.plan_id=402;plan.target=make_target(spec);
    plan.values={
        {0,fhegpu::ValueKind::Ciphertext,host_place(),kContextId,level,kDefaultScaleLog2,true,2},
        {1,fhegpu::ValueKind::Ciphertext,device_place(),kContextId,level,kDefaultScaleLog2,true,2},
        {2,fhegpu::ValueKind::Plaintext,device_place(),kContextId,level,kDefaultScaleLog2,true,1},
        {3,fhegpu::ValueKind::Ciphertext,device_place(),kContextId,level,kDefaultScaleLog2,true,2}};
    plan.external_inputs={0};plan.final_outputs={10};
    auto upload=transfer_action(1,fhegpu::ValueKind::Ciphertext,host_place(),device_place());
    upload.inputs={0};upload.outputs={1};plan.initialization={{0,upload}};
    for(fhegpu::ValueId id=4;id<10;++id) {
        plan.values.push_back({id,fhegpu::ValueKind::Plaintext,host_place(),kContextId,level,kDefaultScaleLog2,true,1});
        plan.execution.push_back({id-3,fhegpu::EncodeOp{fhegpu::InlineEncodePayload{weight},id}});
    }
    plan.values.push_back({10,fhegpu::ValueKind::Ciphertext,host_place(),kContextId,level,kDefaultScaleLog2,true,2});
    plan.execution.push_back({7,fhegpu::PreparePlaintextOp{fhegpu::InlineEncodePayload{weight},2}});
    plan.execution.push_back({8,fhegpu::ComputeOp{fhegpu::ComputeKind::AddCP,{1,2},3,device_place(),{}}});
    plan.execution.push_back({9,fhegpu::ReleaseOp{1}});
    plan.execution.push_back({10,fhegpu::ReleaseOp{2}});
    for(fhegpu::ValueId id=4;id<10;++id)plan.execution.push_back({id+7,fhegpu::ReleaseOp{id}});
    auto download=transfer_action(2,fhegpu::ValueKind::Ciphertext,device_place(),host_place());
    download.inputs={3};download.outputs={10};plan.finalization={{17,download},{18,fhegpu::ReleaseOp{3}}};
    fhegpu::WaterlineConfig config;config.backend=fhegpu::PrepareBackend::Cpu;
    config.device_capacity_bytes={4ULL<<30};config.device_reserve_bytes={2ULL<<30};config.device_fractions={.02};
    config.host_budget_bytes=tight_host_budget;config.pinned_budget_bytes=3*bytes;config.raw_budget_bytes=8*n;
    config.lookahead_groups=1;plan.waterline=config;
    const std::unordered_map<fhegpu::ValueId,PoseidonGpuValue> inputs{{0,PoseidonGpuValue::from_host_ciphertext(cipher)}};
    const auto before=api.waterline_measurements().cpu_prepare_calls;
    std::string rejection;
    try {
        fhegpu::SequentialRuntime<PoseidonGpuApi> runtime(0,1,1,api,fhegpu::DeviceExecutionMode::PerDeviceWorkers);
        (void)runtime.run({plan,kPlanSha},{spec,std::nullopt,false},inputs);
    } catch(const std::exception &error) {rejection=error.what();}
    const auto rejected_calls=api.waterline_measurements().cpu_prepare_calls-before;
    std::cout<<"CPU_PREPARE_HOST_BOUND N="<<n<<" Q="<<q<<" old_bound="<<old_host_bound
             <<" complete_bound="<<complete_host_bound<<" prepare_host_resident="<<prepare_host_resident
             <<" required_total="<<sufficient_host_budget<<" tight_budget="<<tight_host_budget
             <<" cpu_prepare_calls_delta="<<rejected_calls<<" rejection="<<rejection<<'\n';
    require(rejection.find("single-group baseline infeasible")!=std::string::npos,
            "CPU Prepare Host budget did not reject before work submission: "+rejection);
    require(rejection.find("instruction #8")!=std::string::npos,
            "CPU Prepare boundary was masked by initialization/finalization workspace");
    require(rejection.find("Host="+std::to_string(sufficient_host_budget))!=std::string::npos,
            "CPU Prepare rejection did not include coexisting pinned storage in Host peak");
    require(rejected_calls==0,"CPU Prepare started before its complete Host budget fit");

    config.host_budget_bytes=sufficient_host_budget;plan.waterline=config;
    const auto pinned_before=poseidon::runtime_api::communication::PinnedHostBuffer::live_bytes();
    require(pinned_before==0,"CPU Prepare boundary test starts with pending pinned storage");
    std::ostringstream diagnostics;
    struct RestoreStderr {
        std::streambuf *original;
        ~RestoreStderr() {std::cerr.rdbuf(original);}
    };
    std::optional<fhegpu::RunArtifact<PoseidonGpuValue>> artifact;
    {
        RestoreStderr restore{std::cerr.rdbuf(diagnostics.rdbuf())};
        fhegpu::SequentialRuntime<PoseidonGpuApi> runtime(0,1,1,api,fhegpu::DeviceExecutionMode::PerDeviceWorkers);
        artifact.emplace(runtime.run({plan,kPlanSha},{spec,std::nullopt,false},inputs));
    }
    std::cerr<<diagnostics.str();
    require(diagnostics.str().find("remaining_front=0 remaining_plain=0")!=std::string::npos,
            "CPU Prepare retained dynamic foreground/plaintext credit after completion");
    require(diagnostics.str().find("host_peak="+std::to_string(sufficient_host_budget))!=std::string::npos,
            "CPU Prepare positive boundary did not charge its complete Host peak");
    require(api.waterline_measurements().cpu_prepare_calls-before==1,"CPU Prepare positive boundary did not encode exactly once");
    require(artifact->timing.encode_calls==7,"CPU Prepare pressure graph did not encode six Host values and one prepared plaintext");
    require(poseidon::runtime_api::communication::PinnedHostBuffer::live_bytes()==pinned_before,
            "CPU Prepare pinned upload outlived its Ready event");
    poseidon::Plaintext decrypted;decryptor.decrypt(artifact->values.at(10).value.host_ciphertext(),decrypted);
    std::vector<std::complex<double>> slots;encoder.decode(decrypted,slots);
    require(slots.size()==input.size(),"CPU Prepare positive boundary changed slot layout");
    double maximum=0;
    for(std::size_t i=0;i<slots.size();++i) {
        require(std::isfinite(slots[i].real()) && std::isfinite(slots[i].imag()),"CPU Prepare returned nonfinite slots");
        maximum=std::max({maximum,std::abs(slots[i].real()-input[i]-weight[i]),std::abs(slots[i].imag())});
    }
    require(maximum<1e-4,"CPU Prepare positive boundary numerical mismatch");
    std::cout<<"CPU_PREPARE_HOST_BOUND positive_budget="<<sufficient_host_budget<<" pinned_budget="<<config.pinned_budget_bytes
             <<" slots_checked="<<slots.size()<<" max_abs_including_imaginary="<<maximum
             <<" retained_external_final_host="<<8*bytes
             <<" remaining_dynamic_front=0 remaining_plain=0 pinned_live="<<pinned_before<<'\n';
}

void test_waterline_runtime(const poseidon::PoseidonContext &context,
                           poseidon::KeyGenerator &key_generator,
                           const fhegpu::LoadedOperatorSpec &spec,int device_count)
{
    const auto *host_staged_env=std::getenv("POSEIDON_WATERLINE_HOST_STAGED");
    const std::string host_staged_setting=host_staged_env?host_staged_env:"0";
    require(host_staged_setting=="0" || host_staged_setting=="1","POSEIDON_WATERLINE_HOST_STAGED must be 0 or 1");
    const bool host_staged=host_staged_setting=="1";
    std::vector<int> ids(device_count);for(int d=0;d<device_count;++d)ids[d]=d;
    DelayedWaterlineApi api(kContextId,context,ids);
    poseidon::PublicKey public_key;key_generator.create_public_key(public_key);
    poseidon::Encryptor encryptor(context,public_key);poseidon::Decryptor decryptor(context,key_generator.secret_key());
    poseidon::CKKSEncoder encoder(context);poseidon::Plaintext plain;
    const std::vector<double> input{1,-2,3,4};
    std::vector<double> weight(context.parameters_literal()->degree()/2);
    for(std::size_t i=0;i<weight.size();++i)weight[i]=std::sin(double(i)*.137)*.125;
    weight[0]=.5;weight[1]=-1;weight[2]=2;weight[3]=3;
    encoder.encode(input,std::ldexp(1.,kDefaultScaleLog2),plain);poseidon::Ciphertext cipher;encryptor.encrypt(plain,cipher);
    const int level=static_cast<int>(cipher.level());
    fhegpu::RuntimePlan plan;plan.format_version=4;plan.plan_id=401;plan.target=make_target(spec,device_count);
    const auto add_value=[&](fhegpu::ValueKind kind,fhegpu::Place place) {
        const auto id=plan.values.size();plan.values.push_back({id,kind,place,kContextId,level,kDefaultScaleLog2,true,kind==fhegpu::ValueKind::Plaintext?1:2});return id;
    };
    const auto arg=add_value(fhegpu::ValueKind::Ciphertext,host_place());plan.external_inputs={arg};
    std::vector<fhegpu::ValueId> current;
    std::uint64_t transfer_id=1;
    for(int d=0;d<device_count;++d) {
        current.push_back(add_value(fhegpu::ValueKind::Ciphertext,device_place(d)));
        auto initial=transfer_action(transfer_id++,fhegpu::ValueKind::Ciphertext,host_place(),device_place(d));
        initial.inputs={arg};initial.outputs={current.back()};plan.initialization.push_back({0,initial});
    }
    std::vector<fhegpu::ValueId> weights;
    for(int d=0;d<device_count;++d) {
        auto w=add_value(fhegpu::ValueKind::Plaintext,device_place(d));weights.push_back(w);
        plan.execution.push_back({0,fhegpu::PreparePlaintextOp{fhegpu::InlineEncodePayload{weight},w}});
    }
    const int rounds=5;
    for(int r=0;r<rounds;++r) {
      for(int d=0;d<device_count;++d) {
        const auto out=add_value(fhegpu::ValueKind::Ciphertext,device_place(d));
        plan.execution.push_back({0,fhegpu::ComputeOp{fhegpu::ComputeKind::AddCP,{current[d],weights[d]},out,device_place(d),{}}});
        plan.execution.push_back({0,fhegpu::ReleaseOp{current[d]}});
        if(r==rounds-1)plan.execution.push_back({0,fhegpu::ReleaseOp{weights[d]}});
        current[d]=out;
      }
      if(device_count>1) {
        std::vector<fhegpu::ValueId> moved(device_count);
        for(int d=0;d<device_count;++d) {
            const int next=(d+1)%device_count;moved[next]=add_value(fhegpu::ValueKind::Ciphertext,device_place(next));
            auto action=transfer_action(transfer_id++,fhegpu::ValueKind::Ciphertext,device_place(d),device_place(next));
            if(host_staged)action.hint=fhegpu::CommHint::HostStaged;
            action.inputs={current[d]};action.outputs={moved[next]};plan.execution.push_back({0,action});
        }
        for(auto id:current)plan.execution.push_back({0,fhegpu::ReleaseOp{id}});
        current=std::move(moved);
      }
    }
    for(int d=0;d<device_count;++d) {
        const auto result=add_value(fhegpu::ValueKind::Ciphertext,host_place());
        auto download=transfer_action(transfer_id++,fhegpu::ValueKind::Ciphertext,device_place(d),host_place());
        download.inputs={current[d]};download.outputs={result};
        plan.finalization.push_back({0,download});plan.finalization.push_back({0,fhegpu::ReleaseOp{current[d]}});
        plan.final_outputs.push_back(result);
    }
    std::size_t ordinal=0;for(auto *phase:{&plan.initialization,&plan.execution,&plan.finalization})for(auto &i:*phase)i.ordinal=ordinal++;
    fhegpu::WaterlineConfig config;config.device_capacity_bytes.assign(device_count,4ULL<<30);config.device_reserve_bytes.assign(device_count,2ULL<<30);
    config.host_budget_bytes=512ULL<<20;config.pinned_budget_bytes=64ULL<<20;config.raw_budget_bytes=32ULL<<20;
    if(host_staged) {
        const std::uint64_t n=context.parameters_literal()->degree(),q=context.parameters_literal()->q().size();
        const std::uint64_t bytes=n*q*sizeof(poseidon::gpu::GpuWord);
        const auto host_cipher=4*bytes;
        const auto cpu_prepare_host=3*bytes+40*n;
        // At most two full-context communication workspace reservations fit
        // simultaneously. Leave Host room for outputs and two CPU prepares.
        config.pinned_budget_bytes=6*bytes;
        config.host_budget_bytes=std::max(host_cipher*(device_count+1)+18*bytes,
                                          host_cipher+2*cpu_prepare_host);
        config.raw_budget_bytes=16*n;
    }
    config.device_fractions.assign(device_count,.02);config.lookahead_groups=8;
    std::unordered_map<fhegpu::ValueId,PoseidonGpuValue> inputs{{arg,PoseidonGpuValue::from_host_ciphertext(cipher)}};
    for(auto backend:{fhegpu::PrepareBackend::Cpu,fhegpu::PrepareBackend::Gpu,fhegpu::PrepareBackend::Auto}) {
        config.backend=backend;plan.waterline=config;
        std::cout<<"WATERLINE mode="<<(host_staged?"HostStaged":"Auto")<<" backend="<<int(backend)
                 <<" devices="<<device_count<<" host_budget="<<config.host_budget_bytes
                 <<" pinned_budget="<<config.pinned_budget_bytes<<" raw_budget="<<config.raw_budget_bytes<<'\n'<<std::flush;
        api.reset_delay();
        const auto begin=std::chrono::steady_clock::now();
        fhegpu::SequentialRuntime<DelayedWaterlineApi> runtime(0,1,device_count,api,fhegpu::DeviceExecutionMode::PerDeviceWorkers);
        auto artifact=runtime.run({plan,kPlanSha},{spec,std::nullopt,false},inputs);
        const auto measurements=api.waterline_measurements();
        std::cout<<"WATERLINE measurements cumulative_cpu_calls="<<measurements.cpu_prepare_calls
                 <<" cpu_encode_ms="<<measurements.cpu_encode_nanoseconds/1e6
                 <<" cpu_plain_h2d_bytes="<<measurements.cpu_plain_h2d_bytes
                 <<" transport_h2d_bytes="<<measurements.transport_h2d_bytes
                 <<" transport_h2d_ms="<<measurements.transport_h2d_nanoseconds/1e6<<'\n';
        for(std::size_t d=0;d<measurements.devices.size();++d) {
            const auto &m=measurements.devices[d];
            std::cout<<"WATERLINE measurements GPU"<<d<<" allocated_peak="<<m.allocated_peak<<" allocated_live="<<m.allocated_live
                     <<" pool_reserved="<<m.pool_reserved<<" pool_cap="<<m.pool_capacity<<" gpu_encode_calls="<<m.gpu_encode_calls
                     <<" gpu_encode_ms="<<m.gpu_encode_nanoseconds/1e6<<" gpu_raw_h2d_bytes="<<m.gpu_raw_h2d_bytes
                     <<" consumer_wait_calls="<<m.consumer_wait_calls<<" consumer_wait_ms="<<m.consumer_wait_nanoseconds/1e6
                     <<" gpu_raw_h2d_ms="<<m.gpu_raw_h2d_nanoseconds/1e6<<'\n';
        }
        double maximum=0;
        for(auto result:plan.final_outputs) {
            poseidon::Plaintext decoded;decryptor.decrypt(artifact.values.at(result).value.host_ciphertext(),decoded);
            std::vector<std::complex<double>> slots;encoder.decode(decoded,slots);
            for(std::size_t i=0;i<slots.size();++i) {
                const double expected=(i<input.size()?input[i]:0)+rounds*weight[i];
                maximum=std::max(maximum,std::abs(slots[i]-expected));
            }
            if(maximum>=1e-4) {
                const auto &actual=artifact.values.at(result).value.host_ciphertext();
                std::cout<<"Waterline failure output="<<result<<" components="<<actual.size()
                         <<" level="<<actual.level()<<" scale="<<actual.scale()<<'\n';
                for(std::size_t i=0;i<std::min<std::size_t>(4,slots.size());++i) {
                    const double expected=(i<input.size()?input[i]:0)+rounds*weight[i];
                    std::cout<<"Waterline failure slot="<<i<<" real="<<slots[i].real()
                             <<" imag="<<slots[i].imag()<<" expected="<<expected<<'\n';
                }
            }
        }
        if(device_count>1)require(api.peers_while_waiting.load()==static_cast<unsigned>(device_count-1),"not all peers completed before delayed GPU0");
        const auto expected_staged_posts=host_staged && device_count>1?rounds*device_count:0;
        require(api.host_staged_posts.load()==static_cast<unsigned>(expected_staged_posts),"HostStaged ring posts were omitted");
        if(expected_staged_posts)require(api.host_staged_pinned_observed.load()>0,"HostStaged ring did not materialize pinned staging");
        std::cout << "Waterline diagnostic max=" << maximum << " outputs="<<plan.final_outputs.size()
                  <<" peer_devices_completed_before_gpu0="<<api.peers_while_waiting.load()<<'\n';
        std::cout<<"WATERLINE HostStaged posts="<<api.host_staged_posts.load()
                 <<" posts_observed_with_pinned="<<api.host_staged_pinned_observed.load()
                 <<" expected_posts="<<expected_staged_posts<<'\n';
        require(maximum<1e-4,"Waterline plaintext consumer numerical mismatch");
        require(artifact.timing.encode_calls==static_cast<std::size_t>(device_count),"shared Prepare was encoded more than once");
        std::cout<<"WATERLINE backend="<<int(backend)<<" devices="<<device_count<<" max_abs_error="<<maximum
                 <<" runtime_ms="<<std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count()
                 <<" forward_ms="<<artifact.timing.online_execution_nanoseconds/1e6<<" pinned_live="
                 <<poseidon::runtime_api::communication::PinnedHostBuffer::live_bytes()<<'\n';
        require(poseidon::runtime_api::communication::PinnedHostBuffer::live_bytes()==0,
                "Waterline ring retained completed pinned staging");
        std::cout<<"WATERLINE mode="<<(host_staged?"HostStaged":"Auto")
                 <<" physical_pinned_peak_process="<<poseidon::runtime_api::communication::PinnedHostBuffer::peak_bytes()
                 <<" pinned_budget="<<config.pinned_budget_bytes<<'\n';
    }
}


} // namespace

int main()
{
    int device_count = 0;
    const cudaError_t cuda_status = cudaGetDeviceCount(&device_count);
    if (cuda_status != cudaSuccess || device_count <= kDeviceId)
    {
        std::cerr << "[FAIL] Poseidon GPU Api tests require CUDA device 0: "
                  << cudaGetErrorString(cuda_status) << '\n';
        return 1;
    }

    try
    {
        if(std::getenv("POSEIDON_WATERLINE_ONLY") && !std::getenv("POSEIDON_GPU_RUNTIME_MAXIMUM_POOL_MB"))
            setenv("POSEIDON_GPU_RUNTIME_MAXIMUM_POOL_MB","3520",1);
        if (std::getenv("POSEIDON_WATERLINE_SCALE")) kDefaultScaleLog2=std::stoi(std::getenv("POSEIDON_WATERLINE_SCALE"));
        const auto parameters = make_parameters();
        poseidon::PoseidonContext context(parameters);
        run_test("internal RMM pool lifetime",
                 [&] { test_internal_memory_pool_lifetime(context); });

        poseidon::KeyGenerator key_generator(context);
        auto relin_keys = std::make_shared<poseidon::RelinKeys>();
        auto galois_keys = std::make_shared<poseidon::GaloisKeys>();
        key_generator.create_relin_keys(*relin_keys);
        key_generator.create_galois_keys(std::vector<int>{1, 2}, *galois_keys);
        auto boot_public_key = std::make_shared<poseidon::PublicKey>();
        auto boot_secret_key =
            std::make_shared<poseidon::SecretKey>(key_generator.secret_key());
        key_generator.create_public_key(*boot_public_key);

        const auto loaded_spec = make_operator_spec(context);
        if (std::getenv("POSEIDON_WATERLINE_ONLY")) {
            test_gpu_plaintext_wide_coefficients(context);
            run_test("GPU cached packed-read complete Host budget negative/positive boundary",[&] {
                test_gpu_packed_prepare_host_budget(context,loaded_spec);
            });
            test_gpu_raw_pinned_cache(context,loaded_spec,device_count);
            run_test("CPU Prepare complete Host budget negative/positive boundary",[&] {
                test_cpu_prepare_host_budget(context,key_generator,loaded_spec);
            });
            test_waterline_runtime(context,key_generator,loaded_spec,device_count);
            return 0;
        }
        run_test("Waterline counts only physically reserved pool capacity",[&] {
            test_waterline_reserved_pool_capacity(context,loaded_spec);
        });
        RmmPoolScope rmm_pool(kDeviceId);
        run_test("GPU cached packed-read complete Host budget negative/positive boundary",[&] {
            test_gpu_packed_prepare_host_budget(context,loaded_spec);
        });
        run_test("CPU Prepare complete Host budget negative/positive boundary",[&] {
            test_cpu_prepare_host_budget(context,key_generator,loaded_spec);
        });
        run_test("CPU/GPU/auto Waterline through encrypted consumers",[&] {
            test_gpu_raw_pinned_cache(context,loaded_spec,1);
            test_waterline_runtime(context,key_generator,loaded_spec,1);
        });
        run_test("selected Galois-key upload",
                 [&] { test_selected_galois_key_upload(context, *galois_keys); });
        run_test("application Q-prefix, sparse levels and hoisted rotations",
                 [&] { test_application_gpu_extensions(context, key_generator, *galois_keys); });
        run_test("GPU transfer publishes completion events",
                 [&] {
                     test_transfer_posts_completion_event(context, key_generator);
                 });
        run_test("async computation and transfer storage cleanup",
                 [&] { test_async_memory_cleanup(context, key_generator, rmm_pool); });
        run_test("is_ready settles completed Rescale and GPU Prepare storage",[&] {
            test_is_ready_reclaims_completed_temporaries(context,key_generator,loaded_spec,rmm_pool);
        });
        run_test("Rescale workspace follows the validated graph and covers actual GPU allocations",[&] {
            test_rescale_workspace_uses_validated_plan(context,key_generator,loaded_spec,rmm_pool);
        });
        run_test("completed transfer releases pinned staging", test_transfer_staging_cleanup);
        run_test("Runtime drains work outside final output dependencies",
                 [&] { test_runtime_drains_unreturned_work(
                     context, key_generator, loaded_spec, rmm_pool); });
        run_test("Runtime Release during unfinished computation and transfer",
                 [&] { test_runtime_release(context, key_generator, loaded_spec, rmm_pool); });
        run_test("online Encode/upload and explicit Fence in both execution modes",
                 [&] { test_runtime_explicit_encode_fence(context, key_generator, loaded_spec); });
        run_test("GPU AddCP/SubCP/Rotate reuse chains",
                 [&] { test_runtime_reuse(context, key_generator, loaded_spec, rmm_pool); });
        run_test("device mapping rejects invalid CUDA device lists",
                 [&] { test_device_mapping_rejections(context, device_count); });
        {
            PoseidonGpuApi api(kContextId, context, kDeviceId, relin_keys, galois_keys);
            require(api.local_device_count() == 1 && api.cuda_device_id(0) == kDeviceId,
                    "single-GPU constructor mapping is incorrect");

            run_test("preflight rejects unsupported configurations",
                     [&] { test_preflight_rejections(api, loaded_spec); });
            run_test("Runtime Encode/Transfer/AddCP/Transfer",
                     [&] { test_runtime_add_plain(api, context, key_generator, loaded_spec); });
            run_test("Runtime Transfer/AddCC/Transfer",
                     [&] {
                         test_runtime_add_ciphertexts(api, context, key_generator, loaded_spec);
                     });
            run_test("Runtime Encode/Transfer/MulCP/Transfer",
                     [&] {
                         test_runtime_multiply_plain(api, context, key_generator, loaded_spec);
                     });
            run_test("ordinary/lazy Rescale and Value validation",
                     [&] {
                         test_rescale_and_value_validation(api, context, key_generator,
                                                           loaded_spec, rmm_pool);
                     });
            run_test("MulCC/Relinearize/Rescale/composite Rotate",
                     [&] {
                         test_multiply_relinearize_rescale_rotate(
                             api, context, key_generator, *relin_keys, *galois_keys,
                             loaded_spec, rmm_pool);
                     });
        }

        {
            PoseidonGpuApi api(kContextId, context, kDeviceId, relin_keys,
                               galois_keys, boot_public_key, boot_secret_key);
            run_test("Host decrypt/re-encrypt Boot",
                     [&] {
                         test_host_decrypt_reencrypt_boot(
                             api, context, *boot_public_key, *boot_secret_key,
                             loaded_spec);
                     });
        }

        if (device_count >= 2)
        {
            run_test("two-GPU RuntimePlan mapping/Replicate/Transfer",
                     [&] { test_two_gpu_runtime_plan(context, key_generator, loaded_spec); });
        }
        else
        {
            std::cout << "[SKIP] two-GPU RuntimePlan requires two CUDA devices\n";
        }

        if (device_count >= 4)
            run_test("GPU0/GPU3 explicit Encode/Fence shared weight and ciphertext",
                [&] { test_two_gpu_runtime_plan(context, key_generator, loaded_spec, 3); });
        std::cout << tests_run << " Poseidon GPU Runtime Api tests passed\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }
}
