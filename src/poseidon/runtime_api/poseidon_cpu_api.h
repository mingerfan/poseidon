#pragma once

#include "poseidon/ciphertext.h"
#include "poseidon/plaintext.h"
#include "poseidon/poseidon_context.h"
#include "runtime/operator_spec.hpp"
#include "runtime/plan.hpp"

#include <memory>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace poseidon
{
class CKKSEncoder;
class EvaluatorCkksBase;
class GaloisKeys;
class RelinKeys;

namespace runtime_api
{

class PoseidonCpuValue
{
public:
    static PoseidonCpuValue from_plaintext(Plaintext value);
    static PoseidonCpuValue from_ciphertext(Ciphertext value);

    fhegpu::ValueKind kind() const;
    const Plaintext &plaintext() const;
    const Ciphertext &ciphertext() const;

private:
    using Storage = std::variant<std::shared_ptr<Plaintext>, std::shared_ptr<Ciphertext>>;

    explicit PoseidonCpuValue(Storage storage);

    Storage storage_;
};

class PoseidonCpuApi
{
public:
    using Value = PoseidonCpuValue;
    struct CommHandle
    {
    };

    PoseidonCpuApi(std::string context_id, PoseidonContext context,
                   std::shared_ptr<const RelinKeys> relin_keys = {},
                   std::shared_ptr<const GaloisKeys> galois_keys = {});
    ~PoseidonCpuApi();

    PoseidonCpuApi(const PoseidonCpuApi &) = delete;
    PoseidonCpuApi &operator=(const PoseidonCpuApi &) = delete;

    std::string name() const;
    Value encode_plaintext(const fhegpu::ValueDesc &output_desc, const std::vector<double> &slots);
    Value compute(const fhegpu::ComputeOp &op, const std::vector<Value> &inputs);
    CommHandle communicate_async(const fhegpu::CommAction &action,
                                 const std::vector<Value> &local_inputs);
    std::vector<Value> wait(CommHandle &handle);
    void synchronize(Value &value);
    void preflight(std::string_view plan_source_sha256, bool skip_artifact_digest_checks,
                   const fhegpu::TargetConfig &target, const fhegpu::OperatorSpec &operator_spec,
                   const fhegpu::PlanRequirements &requirements);
    [[noreturn]] void abort_all(int exit_code, const std::string &reason);
    void validate_value(const Value &value, const fhegpu::ValueDesc &expected) const;

private:
    std::string context_id_;
    PoseidonContext context_;
    std::unique_ptr<CKKSEncoder> encoder_;
    std::unique_ptr<EvaluatorCkksBase> evaluator_;
    std::shared_ptr<const RelinKeys> relin_keys_;
    std::shared_ptr<const GaloisKeys> galois_keys_;
};

} // namespace runtime_api
} // namespace poseidon
