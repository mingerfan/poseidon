#include "poseidon/runtime_api/poseidon_cpu_api.h"

#include "poseidon/ckks_encoder.h"
#include "poseidon/evaluator/software/evaluator_ckks_software.h"
#include "poseidon/key/galoiskeys.h"
#include "poseidon/key/relinkeys.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace poseidon::runtime_api
{
namespace
{

const Ciphertext &require_ciphertext(const std::vector<PoseidonCpuValue> &inputs, std::size_t index)
{
    if (index >= inputs.size())
    {
        throw std::invalid_argument("missing ciphertext input");
    }
    return inputs[index].ciphertext();
}

const Plaintext &require_plaintext(const std::vector<PoseidonCpuValue> &inputs, std::size_t index)
{
    if (index >= inputs.size())
    {
        throw std::invalid_argument("missing plaintext input");
    }
    return inputs[index].plaintext();
}

double exact_scale(int scale_log2)
{
    return std::ldexp(1.0, scale_log2);
}

void require_host_place(const fhegpu::Place &place, const char *where)
{
    if (place.kind != fhegpu::PlaceKind::Host || place.rank != 0 || place.index != 0)
    {
        throw std::invalid_argument(std::string(where) + " requires Host(rank=0,index=0)");
    }
}

} // namespace

PoseidonCpuValue::PoseidonCpuValue(Storage storage) : storage_(std::move(storage)) {}

PoseidonCpuValue PoseidonCpuValue::from_plaintext(Plaintext value)
{
    return PoseidonCpuValue(std::make_shared<Plaintext>(std::move(value)));
}

PoseidonCpuValue PoseidonCpuValue::from_ciphertext(Ciphertext value)
{
    return PoseidonCpuValue(std::make_shared<Ciphertext>(std::move(value)));
}

fhegpu::ValueKind PoseidonCpuValue::kind() const
{
    return std::holds_alternative<std::shared_ptr<Plaintext>>(storage_)
               ? fhegpu::ValueKind::Plaintext
               : fhegpu::ValueKind::Ciphertext;
}

const Plaintext &PoseidonCpuValue::plaintext() const
{
    const auto *value = std::get_if<std::shared_ptr<Plaintext>>(&storage_);
    if (value == nullptr || *value == nullptr)
    {
        throw std::invalid_argument("Poseidon CPU value is not a plaintext");
    }
    return **value;
}

const Ciphertext &PoseidonCpuValue::ciphertext() const
{
    const auto *value = std::get_if<std::shared_ptr<Ciphertext>>(&storage_);
    if (value == nullptr || *value == nullptr)
    {
        throw std::invalid_argument("Poseidon CPU value is not a ciphertext");
    }
    return **value;
}

PoseidonCpuApi::PoseidonCpuApi(std::string context_id, PoseidonContext context,
                               std::shared_ptr<const RelinKeys> relin_keys,
                               std::shared_ptr<const GaloisKeys> galois_keys)
    : context_id_(std::move(context_id)), context_(std::move(context)),
      encoder_(std::make_unique<CKKSEncoder>(context_)),
      evaluator_(std::make_unique<EvaluatorCkksSoftware>(context_)),
      relin_keys_(std::move(relin_keys)), galois_keys_(std::move(galois_keys))
{
    if (context_id_.empty())
    {
        throw std::invalid_argument("Poseidon CPU Api context id is empty");
    }
    if (context_.parameters_literal()->scheme() != CKKS)
    {
        throw std::invalid_argument("Poseidon CPU Api requires a CKKS context");
    }
}

PoseidonCpuApi::~PoseidonCpuApi() = default;

std::string PoseidonCpuApi::name() const
{
    return "PoseidonCpuApi";
}

PoseidonCpuApi::Value PoseidonCpuApi::encode_plaintext(const fhegpu::ValueDesc &output_desc,
                                                       const std::vector<double> &slots)
{
    require_host_place(output_desc.place, "Poseidon CPU Encode");
    if (output_desc.kind != fhegpu::ValueKind::Plaintext || output_desc.components != 1)
    {
        throw std::invalid_argument("Poseidon CPU Encode output must be plaintext");
    }
    if (output_desc.context != context_id_ || !output_desc.ntt)
    {
        throw std::invalid_argument("Poseidon CPU Encode metadata does not match context");
    }

    const auto parms_id =
        context_.crt_context()->parms_id_map().at(static_cast<std::uint32_t>(output_desc.level));
    Plaintext output;
    encoder_->encode(slots, parms_id, exact_scale(output_desc.scale_log2), output);
    return Value::from_plaintext(std::move(output));
}

PoseidonCpuApi::Value PoseidonCpuApi::compute(const fhegpu::ComputeOp &op,
                                              const std::vector<Value> &inputs)
{
    require_host_place(op.place, "Poseidon CPU compute");
    Ciphertext output;

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
        output = require_ciphertext(inputs, 0);
        for (auto &poly : output.polys())
        {
            poly.negate();
        }
        break;
    case fhegpu::ComputeKind::Rotate:
        if (galois_keys_ == nullptr)
        {
            throw std::runtime_error("Poseidon CPU Rotate requires GaloisKeys");
        }
        evaluator_->rotate(require_ciphertext(inputs, 0), output,
                           std::get<fhegpu::RotateAttrs>(op.attrs).steps, *galois_keys_);
        break;
    case fhegpu::ComputeKind::Rescale:
    {
        const auto attrs = std::get<fhegpu::RescaleAttrs>(op.attrs);
        const auto &input = require_ciphertext(inputs, 0);
        if (input.level() != static_cast<std::size_t>(attrs.target_level + 1))
        {
            throw std::invalid_argument("Poseidon CPU Rescale supports one level per operation");
        }
        evaluator_->rescale(input, output);
        output.scale() = exact_scale(attrs.target_scale_log2);
        break;
    }
    case fhegpu::ComputeKind::ModSwitch:
    {
        const auto attrs = std::get<fhegpu::ModSwitchAttrs>(op.attrs);
        evaluator_->drop_modulus(require_ciphertext(inputs, 0), output,
                                 static_cast<std::uint32_t>(attrs.target_level));
        break;
    }
    case fhegpu::ComputeKind::Relinearize:
        if (relin_keys_ == nullptr)
        {
            throw std::runtime_error("Poseidon CPU Relinearize requires RelinKeys");
        }
        evaluator_->relinearize(require_ciphertext(inputs, 0), output, *relin_keys_);
        break;
    case fhegpu::ComputeKind::Boot:
        throw std::runtime_error("Poseidon CPU Boot is not implemented");
    }

    return Value::from_ciphertext(std::move(output));
}

PoseidonCpuApi::CommHandle PoseidonCpuApi::communicate_async(const fhegpu::CommAction &,
                                                             const std::vector<Value> &)
{
    throw std::runtime_error("Poseidon CPU Api does not support communication");
}

std::vector<PoseidonCpuApi::Value> PoseidonCpuApi::wait(CommHandle &)
{
    throw std::runtime_error("Poseidon CPU Api has no communication handle to wait for");
}

void PoseidonCpuApi::synchronize(Value &) {}

void PoseidonCpuApi::preflight(std::string_view plan_source_sha256,
                               bool skip_artifact_digest_checks, const fhegpu::TargetConfig &target,
                               const fhegpu::OperatorSpec &operator_spec,
                               const fhegpu::PlanRequirements &requirements)
{
    static_cast<void>(skip_artifact_digest_checks);
    if (plan_source_sha256.size() != 71 || plan_source_sha256.substr(0, 7) != "sha256:")
    {
        throw std::invalid_argument("invalid RuntimePlan source SHA-256");
    }
    if (target.target_id != "poseidon-ckks-cpu" || target.capability_version != 1)
    {
        throw std::invalid_argument("Poseidon CPU Api target is unsupported");
    }
    if (target.world_size != 1 || target.device_counts != std::vector<int>{0})
    {
        throw std::invalid_argument("Poseidon CPU Api currently supports one "
                                    "process and Host compute only");
    }

    const auto parameters = context_.parameters_literal();
    if (operator_spec.context_id != context_id_ ||
        operator_spec.poly_degree != parameters->degree())
    {
        throw std::invalid_argument("OperatorSpec context does not match Poseidon CPU context");
    }

    std::vector<int> modulus_bits;
    modulus_bits.reserve(parameters->q().size());
    for (const auto &modulus : parameters->q())
    {
        modulus_bits.push_back(modulus.bit_count());
    }
    if (operator_spec.rns_moduli_log2 != modulus_bits ||
        operator_spec.max_modulus_log2 !=
            *std::max_element(modulus_bits.begin(), modulus_bits.end()) ||
        operator_spec.default_scale_log2 != static_cast<int>(parameters->log_scale()) ||
        operator_spec.rescale_mode != fhegpu::RescaleMode::Eager ||
        operator_spec.level_lower_bound < 0 ||
        operator_spec.level_upper_bound >= static_cast<int>(modulus_bits.size()) ||
        operator_spec.level_lower_bound > operator_spec.level_upper_bound)
    {
        throw std::invalid_argument("OperatorSpec parameters do not match Poseidon CPU context");
    }

    for (const auto capability : requirements.capabilities)
    {
        if (capability != fhegpu::RequiredCapability::Encode &&
            capability != fhegpu::RequiredCapability::HostCompute)
        {
            throw std::runtime_error("Poseidon CPU Api lacks required capability: " +
                                     fhegpu::to_string(capability));
        }
    }

    for (const auto &key : requirements.keys)
    {
        require_host_place(key.place, "Poseidon CPU key");
        if (key.kind == fhegpu::KeyKind::Relin)
        {
            if (relin_keys_ == nullptr || !relin_keys_->has_key(2))
            {
                throw std::runtime_error("Poseidon CPU Api lacks RelinKeys");
            }
        }
        else if (key.kind == fhegpu::KeyKind::Galois)
        {
            if (galois_keys_ == nullptr || !key.rotation_step)
            {
                throw std::runtime_error("Poseidon CPU Api lacks GaloisKeys");
            }
            const auto galois_elt =
                context_.crt_context()->galois_tool()->get_elt_from_step(*key.rotation_step);
            if (!galois_keys_->has_key(galois_elt))
            {
                throw std::runtime_error("Poseidon CPU Api lacks the required rotation key");
            }
        }
        else
        {
            throw std::runtime_error("Poseidon CPU Api does not support secret-key operations");
        }
    }
}

[[noreturn]] void PoseidonCpuApi::abort_all(int, const std::string &reason)
{
    throw std::runtime_error(reason);
}

void PoseidonCpuApi::validate_value(const Value &value, const fhegpu::ValueDesc &expected) const
{
    require_host_place(expected.place, "Poseidon CPU value");
    if (expected.context != context_id_ || value.kind() != expected.kind)
    {
        throw std::runtime_error("Poseidon CPU value kind or context does not match ValueDesc");
    }

    int actual_level = 0;
    double actual_scale = 0.0;
    bool actual_ntt = false;
    int actual_components = 0;
    if (expected.kind == fhegpu::ValueKind::Plaintext)
    {
        const auto &plain = value.plaintext();
        const auto context_data = context_.crt_context()->get_context_data(plain.parms_id());
        if (context_data == nullptr)
        {
            throw std::runtime_error("Poseidon plaintext has an unknown parms_id");
        }
        actual_level = static_cast<int>(context_data->level());
        actual_scale = plain.scale();
        actual_ntt = plain.is_ntt_form();
        actual_components = 1;
    }
    else
    {
        const auto &cipher = value.ciphertext();
        if (cipher.poly_modulus_degree() != context_.parameters_literal()->degree())
        {
            throw std::runtime_error(
                "Poseidon ciphertext polynomial degree does not match context");
        }
        actual_level = static_cast<int>(cipher.level());
        actual_scale = cipher.scale();
        actual_ntt = cipher.is_ntt_form();
        actual_components = static_cast<int>(cipher.size());
    }

    if (!(actual_scale > 0.0))
    {
        throw std::runtime_error("Poseidon CPU value scale is not positive");
    }
    const double actual_scale_log2 = std::log2(actual_scale);
    if (actual_level != expected.level ||
        std::abs(actual_scale_log2 - expected.scale_log2) > 1e-6 || actual_ntt != expected.ntt ||
        actual_components != expected.components)
    {
        throw std::runtime_error("Poseidon CPU value metadata does not match ValueDesc " +
                                 std::to_string(expected.id));
    }
}

} // namespace poseidon::runtime_api
