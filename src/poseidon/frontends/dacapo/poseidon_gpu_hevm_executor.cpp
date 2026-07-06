#include "poseidon/frontends/dacapo/poseidon_gpu_hevm_executor.h"

#include "poseidon/mgpu/runtime/backend/copy_dispatching_backend.h"
#include "poseidon/mgpu/runtime/backend/poseidon_gpu_execution_backend.h"
#include "poseidon/mgpu/runtime/preflight/poseidon_gpu_schedule_preflight.h"
#include "poseidon/mgpu/runtime/executor/sequential_schedule_executor.h"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace poseidon::mgpu
{
namespace
{

void add_error(
    PoseidonGpuHevmExecutionResult &result, std::size_t op_index, std::string message)
{
    result.execution.errors.push_back(
        ScheduleExecutionError{ op_index, std::move(message) });
}

void add_preflight_errors(
    PoseidonGpuHevmExecutionResult &result,
    const PoseidonGpuSchedulePreflightResult &preflight)
{
    for (const PoseidonGpuSchedulePreflightDiagnostic &diagnostic :
         preflight.diagnostics)
    {
        add_error(result, diagnostic.op_index, "preflight: " + diagnostic.message);
    }
}

std::vector<int> schedule_devices(const MgpuSchedule &schedule)
{
    std::vector<int> devices;
    for (const MgpuOp &op : schedule.ops)
    {
        if (op.device_id < 0)
        {
            continue;
        }
        if (std::find(devices.begin(), devices.end(), op.device_id) == devices.end())
        {
            devices.push_back(op.device_id);
        }
    }
    return devices;
}

void upload_keys_for_schedule_devices(
    PoseidonGpuExecutionBackend &backend, const MgpuSchedule &schedule,
    const PoseidonGpuHevmExecutionOptions &options)
{
    if (options.relin_keys == nullptr && options.galois_keys == nullptr)
    {
        return;
    }

    for (const int device_id : schedule_devices(schedule))
    {
        backend.upload_keys_for_device(device_id, options.relin_keys, options.galois_keys);
    }
}

std::string value_name(ValueId id)
{
    std::ostringstream stream;
    stream << '%' << id;
    return stream.str();
}

void validate_hevm_index_range(std::uint64_t index, std::size_t count, const char *what)
{
    if (index >= count)
    {
        std::ostringstream stream;
        stream << "HEVM " << what << " index " << index
               << " is outside object count " << count;
        throw std::invalid_argument(stream.str());
    }
}

void bind_hevm_cipher_inputs(
    PoseidonGpuExecutionBackend &backend, const HevmIoBindingPlan &plan,
    const std::vector<std::shared_ptr<const Ciphertext>> &cipher_inputs)
{
    if (cipher_inputs.size() != plan.cipher_inputs.size())
    {
        std::ostringstream stream;
        stream << "HEVM cipher input object count " << cipher_inputs.size()
               << " does not match schedule input count " << plan.cipher_inputs.size();
        throw std::invalid_argument(stream.str());
    }

    for (const HevmCipherInputSlot &slot : plan.cipher_inputs)
    {
        validate_hevm_index_range(slot.index, cipher_inputs.size(), "cipher input");
        const auto &ciphertext = cipher_inputs[static_cast<std::size_t>(slot.index)];
        if (ciphertext == nullptr)
        {
            std::ostringstream stream;
            stream << "HEVM cipher input object at index " << slot.index
                   << " must not be null";
            throw std::invalid_argument(stream.str());
        }
        backend.bind_cipher_upload(slot.value_id, ciphertext);
    }
}

void bind_hevm_encoded_plain_inputs(
    PoseidonGpuExecutionBackend &backend,
    const std::vector<HevmEncodedPlaintext> &plaintexts)
{
    for (const HevmEncodedPlaintext &plaintext : plaintexts)
    {
        if (plaintext.plaintext == nullptr)
        {
            throw std::invalid_argument("encoded HEVM plaintext must not be null");
        }
        backend.bind_plain_upload(plaintext.value_id, plaintext.plaintext);
    }
}

std::vector<std::shared_ptr<Ciphertext>> collect_hevm_results(
    const PoseidonGpuExecutionBackend &backend, const HevmIoBindingPlan &plan)
{
    std::vector<std::shared_ptr<Ciphertext>> results(plan.results.size());
    for (const HevmResultSlot &slot : plan.results)
    {
        validate_hevm_index_range(slot.index, results.size(), "result");
        if (!backend.has_cipher_download(slot.value_id))
        {
            throw std::out_of_range(
                "missing HEVM ciphertext result for " + value_name(slot.value_id));
        }
        results[static_cast<std::size_t>(slot.index)] =
            backend.cipher_download(slot.value_id);
    }
    return results;
}

void bind_hevm_plan_inputs(
    PoseidonGpuExecutionBackend &backend, const HevmStaticExecutionPlan &plan,
    const std::vector<std::shared_ptr<const Ciphertext>> &cipher_inputs,
    const PoseidonGpuHevmExecutionOptions &options)
{
    bind_hevm_cipher_inputs(backend, plan.io_plan, cipher_inputs);
    bind_hevm_encoded_plain_inputs(backend, plan.encoded_plaintexts);
    upload_keys_for_schedule_devices(backend, plan.schedule, options);
}

}  // namespace

std::string PoseidonGpuHevmExecutionResult::format_errors() const
{
    return execution.format_errors();
}

PoseidonGpuHevmExecutionResult execute_hevm_static_plan_with_poseidon_gpu(
    const PoseidonContext &context, const HevmStaticExecutionPlan &plan,
    const std::vector<std::shared_ptr<const Ciphertext>> &cipher_inputs,
    const PoseidonGpuHevmExecutionOptions &options)
{
    PoseidonGpuHevmExecutionResult result;
    if (options.device_count <= 0)
    {
        add_error(result, 0, "device_count must be positive");
        return result;
    }

    const PoseidonGpuSchedulePreflightResult preflight =
        preflight_poseidon_gpu_schedule(
            plan.schedule,
            PoseidonGpuSchedulePreflightOptions{
                options.device_count,
                /*copy_ops_have_comm=*/false,
                options.relin_keys != nullptr,
                options.galois_keys != nullptr,
            });
    if (!preflight.ok())
    {
        add_preflight_errors(result, preflight);
        return result;
    }

    try
    {
        PoseidonGpuExecutionBackend backend(context);
        bind_hevm_plan_inputs(backend, plan, cipher_inputs, options);
        SequentialScheduleExecutor executor(
            SequentialScheduleExecutorOptions{ options.device_count });
        result.execution = executor.run(plan.schedule, backend);
        if (!result.execution.ok())
        {
            return result;
        }

        result.results = collect_hevm_results(backend, plan.io_plan);
    }
    catch (const std::exception &ex)
    {
        add_error(result, plan.schedule.ops.size(), ex.what());
    }
    catch (...)
    {
        add_error(result, plan.schedule.ops.size(), "unknown HEVM GPU execution error");
    }

    return result;
}

PoseidonGpuHevmExecutionResult execute_hevm_static_plan_with_poseidon_gpu(
    const PoseidonContext &context, const HevmStaticExecutionPlan &plan,
    const std::vector<std::shared_ptr<const Ciphertext>> &cipher_inputs, GpuComm &comm,
    const PoseidonGpuHevmExecutionOptions &options)
{
    PoseidonGpuHevmExecutionResult result;
    if (options.device_count <= 0)
    {
        add_error(result, 0, "device_count must be positive");
        return result;
    }

    const PoseidonGpuSchedulePreflightResult preflight =
        preflight_poseidon_gpu_schedule(
            plan.schedule,
            PoseidonGpuSchedulePreflightOptions{
                options.device_count,
                /*copy_ops_have_comm=*/true,
                options.relin_keys != nullptr,
                options.galois_keys != nullptr,
            });
    if (!preflight.ok())
    {
        add_preflight_errors(result, preflight);
        return result;
    }

    try
    {
        PoseidonGpuExecutionBackend backend(context);
        bind_hevm_plan_inputs(backend, plan, cipher_inputs, options);
        CopyDispatchingExecutionBackend copy_backend(comm, &backend);
        SequentialScheduleExecutor executor(
            SequentialScheduleExecutorOptions{ options.device_count });
        result.execution = executor.run(plan.schedule, copy_backend);
        if (!result.execution.ok())
        {
            return result;
        }

        result.results = collect_hevm_results(backend, plan.io_plan);
    }
    catch (const std::exception &ex)
    {
        add_error(result, plan.schedule.ops.size(), ex.what());
    }
    catch (...)
    {
        add_error(result, plan.schedule.ops.size(), "unknown HEVM GPU execution error");
    }

    return result;
}

}  // namespace poseidon::mgpu
