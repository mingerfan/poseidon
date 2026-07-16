#include "poseidon/runtime_api/poseidon_gpu_api.h"

#include "poseidon/ckks_encoder.h"
#include "poseidon/gpu/gpu_evaluator.h"
#include "poseidon/gpu/gpu_memory.h"
#include "poseidon/gpu/gpu_parameter.h"
#include "poseidon/gpu/gpu_uploader.h"
#include "poseidon/key/galoiskeys.h"
#include "poseidon/key/relinkeys.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

#include <cuda_runtime_api.h>

namespace poseidon::runtime_api
{
namespace
{

double exact_scale(int scale_log2)
{
    const double result = std::ldexp(1.0, scale_log2);
    if (!(result > 0.0) || !std::isfinite(result))
    {
        throw std::invalid_argument("scale_log2 is outside the supported double range");
    }
    return result;
}

void require_host_place(const fhegpu::Place &place, const char *where)
{
    if (place.kind != fhegpu::PlaceKind::Host || place.rank != 0 || place.index != 0)
    {
        throw std::invalid_argument(std::string(where) + " requires Host(rank=0,index=0)");
    }
}

void require_device_place(const fhegpu::Place &place, const char *where)
{
    if (place.kind != fhegpu::PlaceKind::Device || place.rank != 0 || place.index != 0)
    {
        throw std::invalid_argument(std::string(where) +
                                    " requires Device(rank=0,index=0)");
    }
}

const gpu::GpuCiphertextData &require_ciphertext(const std::vector<PoseidonGpuValue> &inputs,
                                                 std::size_t index)
{
    if (index >= inputs.size())
    {
        throw std::invalid_argument("missing GPU ciphertext input");
    }
    return inputs[index].device_ciphertext();
}

const gpu::GpuPlaintextData &require_plaintext(const std::vector<PoseidonGpuValue> &inputs,
                                               std::size_t index)
{
    if (index >= inputs.size())
    {
        throw std::invalid_argument("missing GPU plaintext input");
    }
    return inputs[index].device_plaintext();
}

bool gpu_compute_supported(fhegpu::ComputeKind kind)
{
    return kind != fhegpu::ComputeKind::ModSwitch && kind != fhegpu::ComputeKind::Boot;
}

template <class GpuValue>
void require_single_full_shard(const GpuValue &value, std::size_t component_count,
                               int cuda_device_id, const char *where)
{
    if (value.fields_.size() != 1 || value.fields_.front().device_id != cuda_device_id ||
        value.fields_.front().buffer.device_id() != cuda_device_id)
    {
        throw std::runtime_error(std::string(where) +
                                 " must have one field on the configured CUDA device");
    }

    const std::size_t limb_count = value.meta.q_count + value.meta.p_count;
    const std::size_t component_words = value.meta.degree * limb_count;
    if (value.fields_.front().size() != component_words * component_count)
    {
        throw std::runtime_error(std::string(where) + " field size does not match metadata");
    }
}

void require_full_poly(const gpu::GpuRNSPoly &poly, std::size_t component,
                       std::size_t degree, std::size_t q_count, std::size_t p_count,
                       const char *where)
{
    const std::size_t limb_count = q_count + p_count;
    const std::size_t component_words = degree * limb_count;
    if (poly.poly_id != component || poly.degree != degree || poly.q_count != q_count ||
        poly.p_count != p_count || poly.shards.size() != 1)
    {
        throw std::runtime_error(std::string(where) + " polynomial metadata is invalid");
    }

    const auto &shard = poly.shards.front();
    if (shard.field_index != 0 || shard.field_offset != component * component_words ||
        shard.limb_begin != 0 || shard.limb_count != limb_count || shard.coeff_begin != 0 ||
        shard.coeff_count != degree)
    {
        throw std::runtime_error(std::string(where) + " must use one complete shard");
    }
}

} // namespace

PoseidonGpuValue::PoseidonGpuValue(Storage storage) : storage_(std::move(storage)) {}

PoseidonGpuValue PoseidonGpuValue::from_host_plaintext(Plaintext value)
{
    return PoseidonGpuValue(std::make_shared<Plaintext>(std::move(value)));
}

PoseidonGpuValue PoseidonGpuValue::from_host_ciphertext(Ciphertext value)
{
    return PoseidonGpuValue(std::make_shared<Ciphertext>(std::move(value)));
}

PoseidonGpuValue PoseidonGpuValue::from_device_plaintext(gpu::GpuPlaintextData value)
{
    return PoseidonGpuValue(std::make_shared<gpu::GpuPlaintextData>(std::move(value)));
}

PoseidonGpuValue PoseidonGpuValue::from_device_ciphertext(gpu::GpuCiphertextData value)
{
    return PoseidonGpuValue(std::make_shared<gpu::GpuCiphertextData>(std::move(value)));
}

fhegpu::ValueKind PoseidonGpuValue::kind() const
{
    return std::holds_alternative<std::shared_ptr<Plaintext>>(storage_) ||
                   std::holds_alternative<std::shared_ptr<gpu::GpuPlaintextData>>(storage_)
               ? fhegpu::ValueKind::Plaintext
               : fhegpu::ValueKind::Ciphertext;
}

fhegpu::PlaceKind PoseidonGpuValue::place_kind() const
{
    return std::holds_alternative<std::shared_ptr<Plaintext>>(storage_) ||
                   std::holds_alternative<std::shared_ptr<Ciphertext>>(storage_)
               ? fhegpu::PlaceKind::Host
               : fhegpu::PlaceKind::Device;
}

const Plaintext &PoseidonGpuValue::host_plaintext() const
{
    const auto *value = std::get_if<std::shared_ptr<Plaintext>>(&storage_);
    if (value == nullptr || *value == nullptr)
    {
        throw std::invalid_argument("Poseidon GPU Api value is not a Host plaintext");
    }
    return **value;
}

const Ciphertext &PoseidonGpuValue::host_ciphertext() const
{
    const auto *value = std::get_if<std::shared_ptr<Ciphertext>>(&storage_);
    if (value == nullptr || *value == nullptr)
    {
        throw std::invalid_argument("Poseidon GPU Api value is not a Host ciphertext");
    }
    return **value;
}

const gpu::GpuPlaintextData &PoseidonGpuValue::device_plaintext() const
{
    const auto *value = std::get_if<std::shared_ptr<gpu::GpuPlaintextData>>(&storage_);
    if (value == nullptr || *value == nullptr)
    {
        throw std::invalid_argument("Poseidon GPU Api value is not a Device plaintext");
    }
    return **value;
}

const gpu::GpuCiphertextData &PoseidonGpuValue::device_ciphertext() const
{
    const auto *value = std::get_if<std::shared_ptr<gpu::GpuCiphertextData>>(&storage_);
    if (value == nullptr || *value == nullptr)
    {
        throw std::invalid_argument("Poseidon GPU Api value is not a Device ciphertext");
    }
    return **value;
}

PoseidonGpuApi::PoseidonGpuApi(std::string context_id, PoseidonContext context,
                               int cuda_device_id,
                               std::shared_ptr<const RelinKeys> relin_keys,
                               std::shared_ptr<const GaloisKeys> galois_keys)
    : context_id_(std::move(context_id)), context_(std::move(context)),
      cuda_device_id_(cuda_device_id), relin_keys_(std::move(relin_keys)),
      galois_keys_(std::move(galois_keys))
{
    if (context_id_.empty())
    {
        throw std::invalid_argument("Poseidon GPU Api context id is empty");
    }
    if (context_.parameters_literal()->scheme() != CKKS)
    {
        throw std::invalid_argument("Poseidon GPU Api requires a CKKS context");
    }

    int device_count = 0;
    gpu::gpu_check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (cuda_device_id_ < 0 || cuda_device_id_ >= device_count)
    {
        throw std::invalid_argument("Poseidon GPU Api CUDA device id is unavailable");
    }
    gpu::gpu_check_cuda(cudaSetDevice(cuda_device_id_), "cudaSetDevice");

    const auto parameters = context_.parameters_literal();
    for (const auto &modulus : parameters->q())
    {
        if (modulus.bit_count() > std::numeric_limits<gpu::GpuWord>::digits)
        {
            throw std::invalid_argument("Poseidon GPU Api q modulus does not fit in GpuWord");
        }
    }
    for (const auto &modulus : parameters->p())
    {
        if (modulus.bit_count() > std::numeric_limits<gpu::GpuWord>::digits)
        {
            throw std::invalid_argument("Poseidon GPU Api p modulus does not fit in GpuWord");
        }
    }

    encoder_ = std::make_unique<CKKSEncoder>(context_);
    gpu_parameters_ = std::make_unique<gpu::GpuParameterData>(context_, cuda_device_id_);
    evaluator_ = std::make_unique<gpu::GpuEvaluator>(*gpu_parameters_);
    const std::size_t full_q_count = parameters->q().size();
    if (relin_keys_ != nullptr)
    {
        gpu_relin_keys_by_q_count_.emplace(
            full_q_count,
            std::make_unique<gpu::GpuRelinKeysData>(
                gpu::GpuUploader::upload_relin_keys(
                    *relin_keys_, cuda_device_id_, full_q_count)));
    }
    if (galois_keys_ != nullptr)
    {
        gpu_galois_keys_by_q_count_.emplace(
            full_q_count,
            std::make_unique<gpu::GpuGaloisKeysData>(
                gpu::GpuUploader::upload_galois_keys(
                    *galois_keys_, cuda_device_id_, full_q_count)));
    }
    synchronize_device();
}

PoseidonGpuApi::~PoseidonGpuApi() = default;

std::string PoseidonGpuApi::name() const
{
    return "PoseidonGpuApi";
}

PoseidonGpuApi::Value PoseidonGpuApi::encode_plaintext(
    const fhegpu::ValueDesc &output_desc, const std::vector<double> &slots)
{
    require_host_place(output_desc.place, "Poseidon GPU Encode");
    if (output_desc.kind != fhegpu::ValueKind::Plaintext || output_desc.components != 1)
    {
        throw std::invalid_argument("Poseidon GPU Encode output must be a Host plaintext");
    }
    if (output_desc.context != context_id_ || !output_desc.ntt)
    {
        throw std::invalid_argument("Poseidon GPU Encode metadata does not match context");
    }

    const auto parms_id =
        context_.crt_context()->parms_id_map().at(static_cast<std::uint32_t>(output_desc.level));
    Plaintext output;
    encoder_->encode(slots, parms_id, exact_scale(output_desc.scale_log2), output);
    return Value::from_host_plaintext(std::move(output));
}

PoseidonGpuApi::Value PoseidonGpuApi::compute(const fhegpu::ComputeOp &op,
                                              const std::vector<Value> &inputs)
{
    require_device_place(op.place, "Poseidon GPU compute");
    gpu::GpuCiphertextData output;

    switch (op.kind)
    {
    case fhegpu::ComputeKind::AddCC:
        evaluator_->add(require_ciphertext(inputs, 0), require_ciphertext(inputs, 1), output);
        break;
    case fhegpu::ComputeKind::AddCP:
        evaluator_->add_plain(require_ciphertext(inputs, 0), require_plaintext(inputs, 1), output);
        break;
    case fhegpu::ComputeKind::SubCC:
        evaluator_->sub(require_ciphertext(inputs, 0), require_ciphertext(inputs, 1), output);
        break;
    case fhegpu::ComputeKind::SubCP:
        evaluator_->sub_plain(require_ciphertext(inputs, 0), require_plaintext(inputs, 1), output);
        break;
    case fhegpu::ComputeKind::MulCC:
        evaluator_->multiply(require_ciphertext(inputs, 0), require_ciphertext(inputs, 1), output);
        break;
    case fhegpu::ComputeKind::MulCP:
        evaluator_->multiply_plain(require_ciphertext(inputs, 0), require_plaintext(inputs, 1),
                                   output);
        break;
    case fhegpu::ComputeKind::Negate:
        evaluator_->negate(require_ciphertext(inputs, 0), output);
        break;
    case fhegpu::ComputeKind::Rotate:
    {
        const auto &input = require_ciphertext(inputs, 0);
        if (galois_keys_ == nullptr)
        {
            throw std::runtime_error("Poseidon GPU Rotate requires GaloisKeys");
        }
        evaluator_->rotate(input, std::get<fhegpu::RotateAttrs>(op.attrs).steps,
                           galois_keys_for(input.meta.q_count), output);
        break;
    }
    case fhegpu::ComputeKind::Rescale:
    {
        if (!max_rescale_levels_per_op_)
        {
            throw std::runtime_error("Poseidon GPU Api preflight was not completed");
        }
        const auto attrs = std::get<fhegpu::RescaleAttrs>(op.attrs);
        const auto &input = require_ciphertext(inputs, 0);
        const auto context_data = context_.crt_context()->get_context_data(input.meta.parms_id);
        if (context_data == nullptr)
        {
            throw std::invalid_argument("Poseidon GPU Rescale input has an unknown parms_id");
        }
        const int input_level = static_cast<int>(context_data->level());
        const int drop_count = input_level - attrs.target_level;
        if (drop_count <= 0 || drop_count > *max_rescale_levels_per_op_)
        {
            throw std::invalid_argument("Poseidon GPU Rescale target level is unsupported");
        }

        std::vector<std::unique_ptr<gpu::GpuCiphertextData>> intermediates;
        intermediates.reserve(static_cast<std::size_t>(drop_count));
        const gpu::GpuCiphertextData *source = &input;
        for (int dropped = 0; dropped < drop_count; ++dropped)
        {
            auto next = std::make_unique<gpu::GpuCiphertextData>();
            evaluator_->rescale(*source, *next);
            source = next.get();
            intermediates.push_back(std::move(next));
        }
        synchronize_device();
        intermediates.back()->meta.scale = exact_scale(attrs.target_scale_log2);
        output = std::move(*intermediates.back());
        break;
    }
    case fhegpu::ComputeKind::Relinearize:
    {
        const auto &input = require_ciphertext(inputs, 0);
        if (relin_keys_ == nullptr)
        {
            throw std::runtime_error("Poseidon GPU Relinearize requires RelinKeys");
        }
        evaluator_->relinearize(input, relin_keys_for(input.meta.q_count), output);
        break;
    }
    case fhegpu::ComputeKind::ModSwitch:
        throw std::runtime_error("Poseidon GPU ModSwitch is not implemented");
    case fhegpu::ComputeKind::Boot:
        throw std::runtime_error("Poseidon GPU Boot is not implemented");
    }

    synchronize_device();
    return Value::from_device_ciphertext(std::move(output));
}

PoseidonGpuApi::CommHandle PoseidonGpuApi::communicate_async(
    const fhegpu::CommAction &action, const std::vector<Value> &local_inputs)
{
    if (action.kind != fhegpu::CommKind::Transfer)
    {
        throw std::runtime_error("Poseidon GPU Api does not support Replicate");
    }
    if (action.inputs.size() != 1 || action.outputs.size() != 1 ||
        action.sources.size() != 1 || action.destinations.size() != 1 ||
        action.output_types.size() != 1 || local_inputs.size() != 1)
    {
        throw std::invalid_argument("Poseidon GPU Transfer requires one input and one output");
    }

    const auto &source_place = action.sources.front();
    const auto &destination_place = action.destinations.front();
    const auto &input = local_inputs.front();
    if (input.kind() != action.output_types.front())
    {
        throw std::invalid_argument("Poseidon GPU Transfer value kind mismatch");
    }

    CommHandle handle;
    if (source_place.kind == fhegpu::PlaceKind::Host &&
        destination_place.kind == fhegpu::PlaceKind::Device)
    {
        require_host_place(source_place, "Poseidon GPU Transfer source");
        require_device_place(destination_place, "Poseidon GPU Transfer destination");
        if (input.place_kind() != fhegpu::PlaceKind::Host)
        {
            throw std::invalid_argument("Poseidon GPU Transfer expected a Host input");
        }
        if (input.kind() == fhegpu::ValueKind::Plaintext)
        {
            handle.outputs.push_back(Value::from_device_plaintext(
                gpu::GpuUploader::upload_plaintext(input.host_plaintext(), cuda_device_id_)));
        }
        else
        {
            handle.outputs.push_back(Value::from_device_ciphertext(
                gpu::GpuUploader::upload_ciphertext(input.host_ciphertext(), cuda_device_id_)));
        }
        synchronize_device();
        return handle;
    }

    if (source_place.kind == fhegpu::PlaceKind::Device &&
        destination_place.kind == fhegpu::PlaceKind::Host)
    {
        require_device_place(source_place, "Poseidon GPU Transfer source");
        require_host_place(destination_place, "Poseidon GPU Transfer destination");
        if (input.place_kind() != fhegpu::PlaceKind::Device)
        {
            throw std::invalid_argument("Poseidon GPU Transfer expected a Device input");
        }
        if (input.kind() == fhegpu::ValueKind::Plaintext)
        {
            Plaintext output;
            gpu::GpuUploader::download_plaintext(input.device_plaintext(), output, context_);
            handle.outputs.push_back(Value::from_host_plaintext(std::move(output)));
        }
        else
        {
            Ciphertext output;
            gpu::GpuUploader::download_ciphertext(input.device_ciphertext(), output, context_);
            handle.outputs.push_back(Value::from_host_ciphertext(std::move(output)));
        }
        synchronize_device();
        return handle;
    }

    throw std::invalid_argument("Poseidon GPU Transfer only supports Host/Device conversion");
}

std::vector<PoseidonGpuApi::Value> PoseidonGpuApi::wait(CommHandle &handle)
{
    if (handle.waited)
    {
        throw std::runtime_error("Poseidon GPU Transfer handle was already waited");
    }
    handle.waited = true;
    return std::move(handle.outputs);
}

void PoseidonGpuApi::synchronize(Value &value)
{
    if (value.place_kind() == fhegpu::PlaceKind::Device)
    {
        synchronize_device();
    }
}

void PoseidonGpuApi::preflight(std::string_view plan_source_sha256,
                               bool skip_artifact_digest_checks,
                               const fhegpu::TargetConfig &target,
                               const fhegpu::OperatorSpec &operator_spec,
                               const fhegpu::PlanRequirements &requirements)
{
    static_cast<void>(skip_artifact_digest_checks);
    if (plan_source_sha256.size() != 71 || plan_source_sha256.substr(0, 7) != "sha256:")
    {
        throw std::invalid_argument("invalid RuntimePlan source SHA-256");
    }
    if (target.target_id != "poseidon-ckks-gpu" || target.capability_version != 1)
    {
        throw std::invalid_argument("Poseidon GPU Api target is unsupported");
    }
    if (target.world_size != 1 || target.device_counts != std::vector<int>{1})
    {
        throw std::invalid_argument(
            "Poseidon GPU Api currently supports one process and one logical device");
    }

    const auto parameters = context_.parameters_literal();
    std::vector<int> modulus_bits;
    modulus_bits.reserve(parameters->q().size());
    for (const auto &modulus : parameters->q())
    {
        modulus_bits.push_back(modulus.bit_count());
    }

    if (operator_spec.status == "placeholder")
    {
        throw std::invalid_argument("Poseidon GPU Api rejects placeholder OperatorSpec");
    }
    if (operator_spec.target_id != target.target_id ||
        operator_spec.context_id != context_id_ ||
        operator_spec.poly_degree != parameters->degree() ||
        operator_spec.rns_moduli_log2 != modulus_bits ||
        operator_spec.max_modulus_log2 !=
            *std::max_element(modulus_bits.begin(), modulus_bits.end()) ||
        operator_spec.default_scale_log2 != static_cast<int>(parameters->log_scale()) ||
        operator_spec.rescale_mode != fhegpu::RescaleMode::Lazy ||
        operator_spec.level_lower_bound < 0 ||
        operator_spec.level_upper_bound >= static_cast<int>(modulus_bits.size()) ||
        operator_spec.level_lower_bound > operator_spec.level_upper_bound)
    {
        throw std::invalid_argument("OperatorSpec parameters do not match Poseidon GPU context");
    }

    for (const auto &[kind, support] : operator_spec.operators)
    {
        if (support.supported && !gpu_compute_supported(kind))
        {
            throw std::invalid_argument("OperatorSpec enables an unsupported Poseidon GPU op: " +
                                        fhegpu::to_string(kind));
        }
    }
    if (!operator_spec.boot_profiles.empty())
    {
        throw std::invalid_argument("Poseidon GPU Api does not support Boot profiles");
    }

    const auto rescale = operator_spec.operators.find(fhegpu::ComputeKind::Rescale);
    if (rescale == operator_spec.operators.end() || !rescale->second.supported ||
        !rescale->second.max_levels_per_op || *rescale->second.max_levels_per_op < 4)
    {
        throw std::invalid_argument(
            "Poseidon GPU OperatorSpec must support at least four rescale levels per op");
    }
    max_rescale_levels_per_op_ = *rescale->second.max_levels_per_op;

    for (const auto capability : requirements.capabilities)
    {
        if (capability != fhegpu::RequiredCapability::Encode &&
            capability != fhegpu::RequiredCapability::Transfer)
        {
            throw std::runtime_error("Poseidon GPU Api lacks required capability: " +
                                     fhegpu::to_string(capability));
        }
    }

    for (const auto &key : requirements.keys)
    {
        require_device_place(key.place, "Poseidon GPU key");
        if (key.kind == fhegpu::KeyKind::Relin)
        {
            if (relin_keys_ == nullptr || gpu_relin_keys_by_q_count_.empty() ||
                !relin_keys_->has_key(2))
            {
                throw std::runtime_error("Poseidon GPU Api lacks RelinKeys");
            }
        }
        else if (key.kind == fhegpu::KeyKind::Galois)
        {
            if (galois_keys_ == nullptr || gpu_galois_keys_by_q_count_.empty() ||
                !key.rotation_step)
            {
                throw std::runtime_error("Poseidon GPU Api lacks GaloisKeys");
            }
            const auto galois_elt =
                context_.crt_context()->galois_tool()->get_elt_from_step(*key.rotation_step);
            if (!galois_keys_->has_key(galois_elt))
            {
                throw std::runtime_error("Poseidon GPU Api lacks the required rotation key");
            }
        }
        else
        {
            throw std::runtime_error("Poseidon GPU Api does not support secret-key operations");
        }
    }

    synchronize_device();
}

[[noreturn]] void PoseidonGpuApi::abort_all(int, const std::string &reason)
{
    throw std::runtime_error(reason);
}

void PoseidonGpuApi::validate_value(const Value &value,
                                    const fhegpu::ValueDesc &expected) const
{
    if (expected.context != context_id_ || value.kind() != expected.kind ||
        value.place_kind() != expected.place.kind)
    {
        throw std::runtime_error(
            "Poseidon GPU value kind, place, or context does not match ValueDesc");
    }

    int actual_level = 0;
    double actual_scale = 0.0;
    bool actual_ntt = false;
    int actual_components = 0;

    if (expected.place.kind == fhegpu::PlaceKind::Host)
    {
        require_host_place(expected.place, "Poseidon GPU Host value");
        if (expected.kind == fhegpu::ValueKind::Plaintext)
        {
            const auto &plain = value.host_plaintext();
            const auto context_data = context_.crt_context()->get_context_data(plain.parms_id());
            if (context_data == nullptr)
            {
                throw std::runtime_error("Poseidon Host plaintext has an unknown parms_id");
            }
            actual_level = static_cast<int>(context_data->level());
            actual_scale = plain.scale();
            actual_ntt = plain.is_ntt_form();
            actual_components = 1;
        }
        else
        {
            const auto &cipher = value.host_ciphertext();
            if (cipher.poly_modulus_degree() != context_.parameters_literal()->degree())
            {
                throw std::runtime_error(
                    "Poseidon Host ciphertext polynomial degree does not match context");
            }
            actual_level = static_cast<int>(cipher.level());
            actual_scale = cipher.scale();
            actual_ntt = cipher.is_ntt_form();
            actual_components = static_cast<int>(cipher.size());
        }
    }
    else
    {
        require_device_place(expected.place, "Poseidon GPU Device value");
        if (expected.kind == fhegpu::ValueKind::Plaintext)
        {
            const auto &plain = value.device_plaintext();
            const auto context_data = context_.crt_context()->get_context_data(plain.meta.parms_id);
            if (context_data == nullptr)
            {
                throw std::runtime_error("Poseidon GPU plaintext has an unknown parms_id");
            }
            if (plain.meta.degree != context_.parameters_literal()->degree() ||
                plain.meta.q_count != context_data->parms().q().size() ||
                plain.meta.p_count != 0 || plain.poly_.degree != plain.meta.degree ||
                plain.poly_.q_count != plain.meta.q_count || plain.poly_.p_count != 0)
            {
                throw std::runtime_error("Poseidon GPU plaintext shape does not match context");
            }
            require_single_full_shard(plain, 1, cuda_device_id_, "Poseidon GPU plaintext");
            require_full_poly(plain.poly_, 0, plain.meta.degree, plain.meta.q_count, 0,
                              "Poseidon GPU plaintext");
            actual_level = static_cast<int>(context_data->level());
            actual_scale = plain.meta.scale;
            actual_ntt = plain.meta.is_ntt_form;
            actual_components = 1;
        }
        else
        {
            const auto &cipher = value.device_ciphertext();
            const auto context_data =
                context_.crt_context()->get_context_data(cipher.meta.parms_id);
            if (context_data == nullptr)
            {
                throw std::runtime_error("Poseidon GPU ciphertext has an unknown parms_id");
            }
            if (cipher.meta.degree != context_.parameters_literal()->degree() ||
                cipher.meta.q_count != context_data->parms().q().size() ||
                cipher.meta.p_count != 0 ||
                cipher.meta.component_count != cipher.polys_.size())
            {
                throw std::runtime_error("Poseidon GPU ciphertext shape does not match context");
            }
            require_single_full_shard(cipher, cipher.meta.component_count, cuda_device_id_,
                                      "Poseidon GPU ciphertext");
            for (std::size_t component = 0; component < cipher.polys_.size(); ++component)
            {
                require_full_poly(cipher.polys_[component], component, cipher.meta.degree,
                                  cipher.meta.q_count, 0, "Poseidon GPU ciphertext");
            }
            actual_level = static_cast<int>(context_data->level());
            actual_scale = cipher.meta.scale;
            actual_ntt = cipher.meta.is_ntt_form;
            actual_components = static_cast<int>(cipher.meta.component_count);
        }
    }

    if (!(actual_scale > 0.0) || !std::isfinite(actual_scale))
    {
        throw std::runtime_error("Poseidon GPU Api value scale is invalid");
    }
    if (actual_level != expected.level ||
        std::abs(std::log2(actual_scale) - expected.scale_log2) > 1e-6 ||
        actual_ntt != expected.ntt || actual_components != expected.components)
    {
        throw std::runtime_error("Poseidon GPU Api value metadata does not match ValueDesc " +
                                 std::to_string(expected.id));
    }
}

void PoseidonGpuApi::synchronize_device() const
{
    gpu::gpu_check_cuda(cudaSetDevice(cuda_device_id_), "cudaSetDevice");
    gpu::gpu_check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
}

const gpu::GpuRelinKeysData &PoseidonGpuApi::relin_keys_for(std::size_t q_count)
{
    const auto existing = gpu_relin_keys_by_q_count_.find(q_count);
    if (existing != gpu_relin_keys_by_q_count_.end())
    {
        return *existing->second;
    }
    if (relin_keys_ == nullptr)
    {
        throw std::runtime_error("Poseidon GPU Relinearize requires RelinKeys");
    }

    auto keys = std::make_unique<gpu::GpuRelinKeysData>(
        gpu::GpuUploader::upload_relin_keys(*relin_keys_, cuda_device_id_, q_count));
    const auto [inserted, ok] =
        gpu_relin_keys_by_q_count_.emplace(q_count, std::move(keys));
    if (!ok)
    {
        throw std::logic_error("Poseidon GPU RelinKeys cache insertion failed");
    }
    synchronize_device();
    return *inserted->second;
}

const gpu::GpuGaloisKeysData &PoseidonGpuApi::galois_keys_for(std::size_t q_count)
{
    const auto existing = gpu_galois_keys_by_q_count_.find(q_count);
    if (existing != gpu_galois_keys_by_q_count_.end())
    {
        return *existing->second;
    }
    if (galois_keys_ == nullptr)
    {
        throw std::runtime_error("Poseidon GPU Rotate requires GaloisKeys");
    }

    auto keys = std::make_unique<gpu::GpuGaloisKeysData>(
        gpu::GpuUploader::upload_galois_keys(*galois_keys_, cuda_device_id_, q_count));
    const auto [inserted, ok] =
        gpu_galois_keys_by_q_count_.emplace(q_count, std::move(keys));
    if (!ok)
    {
        throw std::logic_error("Poseidon GPU GaloisKeys cache insertion failed");
    }
    synchronize_device();
    return *inserted->second;
}

} // namespace poseidon::runtime_api
