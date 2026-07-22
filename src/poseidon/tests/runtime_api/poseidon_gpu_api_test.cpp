#include "poseidon/ckks_encoder.h"
#include "poseidon/decryptor.h"
#include "poseidon/encryptor.h"
#include "poseidon/evaluator/software/evaluator_ckks_software.h"
#include "poseidon/gpu/gpu_uploader.h"
#include "poseidon/keygenerator.h"
#include "poseidon/runtime_api/poseidon_gpu_api.h"
#include "runtime/runtime.hpp"

#include <cuda_runtime_api.h>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/device/limiting_resource_adaptor.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <condition_variable>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
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
constexpr int kDefaultScaleLog2 = 30;
const std::string kContextId = "poseidon-gpu-api-test-context";
const std::string kOperatorSpecSha =
    "sha256:0000000000000000000000000000000000000000000000000000000000000000";
const std::string kPlanSha =
    "sha256:1111111111111111111111111111111111111111111111111111111111111111";

int tests_run = 0;

struct CudaHostGate
{
    ~CudaHostGate()
    {
        {
            std::lock_guard<std::mutex> lock(mutex);
            release = true;
            condition.notify_all();
        }
        (void)cudaSetDevice(kDeviceId);
        (void)cudaDeviceSynchronize();
    }

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
        : device_id_(device_id), pool_(&upstream_, 1 << 20, std::nullopt),
          accounting_(&pool_, std::numeric_limits<std::size_t>::max())
    {
        const cudaError_t status = cudaSetDevice(device_id_);
        if (status != cudaSuccess)
        {
            throw std::runtime_error(std::string("cudaSetDevice failed: ") +
                                     cudaGetErrorString(status));
        }
        previous_ = rmm::mr::get_current_device_resource();
        rmm::mr::set_current_device_resource(&accounting_);
    }

    RmmPoolScope(const RmmPoolScope &) = delete;
    RmmPoolScope &operator=(const RmmPoolScope &) = delete;

    std::size_t allocated_bytes() const noexcept
    {
        return accounting_.get_allocated_bytes();
    }

    ~RmmPoolScope()
    {
        (void)cudaSetDevice(device_id_);
        rmm::mr::set_current_device_resource(previous_);
    }

private:
    int device_id_ = 0;
    rmm::mr::cuda_memory_resource upstream_;
    rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> pool_;
    rmm::mr::limiting_resource_adaptor<rmm::mr::device_memory_resource> accounting_;
    rmm::mr::device_memory_resource *previous_ = nullptr;
};

poseidon::ParametersLiteral make_parameters()
{
    poseidon::ParametersLiteral parameters(
        CKKS,
        /*log_n=*/12,
        /*log_slots=*/11,
        /*log_scale=*/kDefaultScaleLog2,
        /*hamming_weight=*/0,
        /*q0_level=*/0,
        poseidon::Modulus(0),
        {},
        {},
        poseidon::sec_level_type::none);
    parameters.set_log_modulus(std::vector<std::uint32_t>(7, 30),
                               std::vector<std::uint32_t>(2, 30));
    return parameters;
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
    rescale.max_levels_per_op = 4;
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
    auto handle = api.communicate_async(transfer_action(id, kind, source, destination), {input});
    auto outputs = api.wait(handle);
    require(outputs.size() == 1, "Transfer returned the wrong output count");
    require_rejected([&] { (void)api.wait(handle); }, "already waited");
    return std::move(outputs.front());
}

void test_transfer_post_does_not_wait_for_producer(
    PoseidonGpuApi &api, const poseidon::PoseidonContext &context,
    poseidon::KeyGenerator &key_generator)
{
    poseidon::PublicKey public_key;
    key_generator.create_public_key(public_key);
    poseidon::Encryptor encryptor(context, public_key);
    poseidon::Decryptor decryptor(context, key_generator.secret_key());
    poseidon::CKKSEncoder encoder(context);

    const std::vector<double> expected{1.0, -2.0, 3.0, 4.0};
    poseidon::Plaintext plain;
    encoder.encode(expected, std::ldexp(1.0, kDefaultScaleLog2), plain);
    poseidon::Ciphertext cipher;
    encryptor.encrypt(plain, cipher);
    auto device_value = transfer_value(
        api, 40, PoseidonGpuValue::from_host_ciphertext(std::move(cipher)),
        fhegpu::ValueKind::Ciphertext, host_place(), device_place());

    fhegpu::ComputeOp negate;
    negate.kind = fhegpu::ComputeKind::Negate;
    negate.place = device_place();
    {
        auto warmup = api.compute(negate, {device_value});
        api.synchronize(warmup);
    }

    CudaHostGate gate;
    require(cudaSetDevice(kDeviceId) == cudaSuccess,
            "failed to select CUDA device for asynchronous Transfer test");
    require(cudaLaunchHostFunc(nullptr, wait_for_cuda_host_gate, &gate) == cudaSuccess,
            "failed to enqueue CUDA producer gate");
    {
        std::unique_lock<std::mutex> lock(gate.mutex);
        require(gate.condition.wait_for(lock, std::chrono::seconds(5),
                                        [&] { return gate.entered; }),
                "CUDA producer gate did not start");
    }

    CudaGateWatchdog watchdog(gate);
    auto produced = api.compute(negate, {device_value});
    auto handle = api.communicate_async(
        transfer_action(41, fhegpu::ValueKind::Ciphertext, device_place(), host_place()),
        {produced});
    require(watchdog.finish_post(),
            "communicate_async blocked on an unfinished GPU producer");

    auto outputs = api.wait(handle);
    require(outputs.size() == 1, "asynchronous Transfer returned wrong output count");
    poseidon::Plaintext actual_plain;
    decryptor.decrypt(outputs.front().host_ciphertext(), actual_plain);
    std::vector<std::complex<double>> actual;
    encoder.decode(actual_plain, actual);
    for (std::size_t index = 0; index < expected.size(); ++index)
    {
        require(std::abs(actual[index].real() + expected[index]) < 1e-4,
                "asynchronous Transfer result mismatch at slot " +
                    std::to_string(index));
    }
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
    constexpr int output_scale_log2 = kDefaultScaleLog2 * 2;
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
    const auto target = make_target(loaded_spec);
    const fhegpu::PlanRequirements valid_requirements{
        {fhegpu::RequiredCapability::Encode, fhegpu::RequiredCapability::Transfer,
         fhegpu::RequiredCapability::Replicate}, {}};
    api.preflight(kPlanSha, false, target, loaded_spec.spec, valid_requirements);
    require_rejected([&] { static_cast<void>(api.cuda_device_id(1)); },
                     "logical device index");

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

    const fhegpu::PlanRequirements host_compute{
        {fhegpu::RequiredCapability::HostCompute}, {}};
    require_rejected(
        [&] { api.preflight(kPlanSha, false, target, loaded_spec.spec, host_compute); },
        "host_compute");
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
    auto refreshed = api.compute(
        boot, {PoseidonGpuValue::from_host_ciphertext(std::move(input_cipher))});
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

    constexpr int input_scale_log2 = 140;
    poseidon::Plaintext plain;
    encoder.encode(std::vector<double>{1.0, -2.0, 3.0, 4.0},
                   std::ldexp(1.0, input_scale_log2), plain);
    poseidon::Ciphertext cipher;
    encryptor.encrypt(plain, cipher);

    poseidon::Ciphertext expected_lazy = cipher;
    for (int dropped = 0; dropped < 4; ++dropped)
    {
        poseidon::Ciphertext next;
        cpu_evaluator.rescale(expected_lazy, next);
        expected_lazy = std::move(next);
    }
    expected_lazy.scale() = std::ldexp(1.0, input_scale_log2 - 120);
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
    require(bytes_after_synchronize < bytes_before_synchronize,
            "synchronize did not release completed input storage");
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
    ordinary.attrs = fhegpu::RescaleAttrs{input_level - 1, input_scale_log2 - 30};
    auto ordinary_result = api.compute(ordinary, {device_value});
    api.validate_value(
        ordinary_result,
        {11, fhegpu::ValueKind::Ciphertext, device_place(), kContextId, input_level - 1,
         input_scale_log2 - 30, true, 2});

    fhegpu::ComputeOp lazy = ordinary;
    lazy.attrs = fhegpu::RescaleAttrs{input_level - 4, input_scale_log2 - 120};
    auto lazy_result = api.compute(lazy, {device_value});
    api.validate_value(
        lazy_result,
        {12, fhegpu::ValueKind::Ciphertext, device_place(), kContextId, input_level - 4,
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
    for (std::size_t i = 0; i < original_slots.size(); ++i)
    {
        require(std::abs(lazy_slots[i].real() - expected_lazy_slots[i].real()) < 1e-4,
                "GPU/CPU lazy Rescale mismatch at slot " + std::to_string(i) +
                    ": CPU " + std::to_string(expected_lazy_slots[i].real()) +
                    ", GPU " + std::to_string(lazy_slots[i].real()));
        require(std::abs(lazy_slots[i].real() - original_slots[i]) < 5e-3,
                "lazy Rescale result mismatch at slot " + std::to_string(i) +
                    ": expected " + std::to_string(original_slots[i]) +
                    ", got " + std::to_string(lazy_slots[i].real()));
    }

    fhegpu::ComputeOp mod_switch;
    mod_switch.kind = fhegpu::ComputeKind::ModSwitch;
    mod_switch.place = device_place();
    mod_switch.attrs = fhegpu::ModSwitchAttrs{input_level - 2};
    auto mod_switched = api.compute(mod_switch, {device_value});
    api.validate_value(
        mod_switched,
        {13, fhegpu::ValueKind::Ciphertext, device_place(), kContextId,
         input_level - 2, input_scale_log2, true, 2});

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
    const fhegpu::LoadedOperatorSpec &loaded_spec)
{
    const auto device = device_place();
    fhegpu::PlanRequirements requirements;
    requirements.keys = {
        {fhegpu::KeyKind::Relin, device, std::nullopt},
        {fhegpu::KeyKind::Galois, device, 3},
    };
    api.preflight(kPlanSha, false, make_target(loaded_spec), loaded_spec.spec, requirements);

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
    cpu_evaluator.rescale(expected_relinearize, expected_rescale);
    expected_rescale.scale() = std::ldexp(1.0, kDefaultScaleLog2);
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
    rescale.attrs = fhegpu::RescaleAttrs{static_cast<int>(cipher0.level()) - 1,
                                        kDefaultScaleLog2};
    auto rescaled = api.compute(rescale, {relinearized});

    fhegpu::ComputeOp rotate;
    rotate.kind = fhegpu::ComputeKind::Rotate;
    rotate.place = device;
    rotate.attrs = fhegpu::RotateAttrs{3};
    auto rotated = api.compute(rotate, {rescaled});
    auto downloaded = transfer_value(api, 22, rotated, fhegpu::ValueKind::Ciphertext,
                                     device, host_place());

    poseidon::Plaintext expected_plain;
    poseidon::Plaintext actual_plain;
    decryptor.decrypt(expected_rotate, expected_plain);
    decryptor.decrypt(downloaded.host_ciphertext(), actual_plain);
    std::vector<std::complex<double>> expected;
    std::vector<std::complex<double>> actual;
    encoder.decode(expected_plain, expected);
    encoder.decode(actual_plain, actual);
    for (std::size_t i = 0; i < 4; ++i)
    {
        require(std::abs(actual[i].real() - expected[i].real()) < 1e-4,
                "GPU chain result mismatch at slot " + std::to_string(i));
        require(std::abs(actual[i].imag() - expected[i].imag()) < 1e-4,
                "GPU chain imaginary mismatch at slot " + std::to_string(i));
    }
}

void test_two_gpu_runtime_plan(const poseidon::PoseidonContext &context,
                               poseidon::KeyGenerator &key_generator,
                               const fhegpu::LoadedOperatorSpec &loaded_spec)
{
    constexpr int second_device_id = 1;
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
    const fhegpu::LoadedRuntimePlan loaded_plan{
        make_two_gpu_plan(loaded_spec, level), kPlanSha};
    const fhegpu::RuntimeResources resources{loaded_spec, std::nullopt, false};
    fhegpu::SequentialRuntime<PoseidonGpuApi> runtime(0, 1, 2, api);
    std::unordered_map<fhegpu::ValueId, PoseidonGpuValue> inputs;
    inputs.emplace(0, PoseidonGpuValue::from_host_ciphertext(std::move(input_cipher)));
    const auto artifact = runtime.run(loaded_plan, resources, inputs);

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
        RmmPoolScope rmm_pool(kDeviceId);
        const auto parameters = make_parameters();
        poseidon::PoseidonContext context(parameters);
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
            run_test("Transfer post does not wait for unfinished producer",
                     [&] {
                         test_transfer_post_does_not_wait_for_producer(
                             api, context, key_generator);
                     });
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
                             loaded_spec);
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

        std::cout << tests_run << " Poseidon GPU Runtime Api tests passed\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }
}
