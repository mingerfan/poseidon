#pragma once

#include "poseidon/poseidon_context.h"
#include "runtime/plan.hpp"

#include <cmath>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace poseidon::test {

// Real-number reference for the manual arcsine_degree=0 circuit. The
// integer-period contract removes wraps; the intended map is a sine,
// not an exact identity. Polynomial fitting and HE rounding remain errors.
inline double manual_bootstrap_reference(const PoseidonContext &context, double value) {
    const auto parameters = context.parameters_literal();
    const double q0 = context.crt_context()->q0();
    const double qdiff = q0 / std::exp2(std::round(std::log2(q0)));
    double log_scale = 40 + 3 * 45;
    for (std::size_t index = 2; index < 6; ++index)
        log_scale -= std::log2(static_cast<double>(parameters->q().at(index).value()));
    const double s2c_scale = std::exp2(log_scale);
    const double target = std::exp2(std::round(std::log2(q0 / 32)));
    const double prepared = s2c_scale * std::round(target / s2c_scale);
    const double two_pi = 2 * std::acos(-1.0);
    return 32 * qdiff / two_pi * std::sin(two_pi * value * prepared / q0);
}

// Evaluate the scalar arithmetic test plan over clear real numbers.
// Rescale/ModSwitch preserve values; each Boot applies the map above.
inline std::vector<double> manual_plan_reference(const fhegpu::RuntimePlan &plan,
    const PoseidonContext &context, const std::vector<double> &input, bool sine_boot = true) {
    using namespace fhegpu;
    std::unordered_map<ValueId, std::vector<double>> values;
    values.emplace(plan.external_inputs.at(0), input);
    auto phase = [&](const std::vector<Instruction> &instructions) {
        for (const auto &instruction : instructions) {
            if (const auto *encode = std::get_if<EncodeOp>(&instruction.body)) {
                const auto &payload = std::get<InlineEncodePayload>(encode->payload).values;
                if (payload.size() != 1 && payload.size() != input.size())
                    throw std::runtime_error("manual reference requires a scalar or full-slot plaintext");
                std::vector<double> result(input.size());
                for (std::size_t i = 0; i < result.size(); ++i)
                    result[i] = payload.at(payload.size() == 1 ? 0 : i);
                values[encode->output] = std::move(result);
            } else if (const auto *comm = std::get_if<CommAction>(&instruction.body)) {
                for (auto output : comm->outputs) values[output] = values.at(comm->inputs.at(0));
            } else if (const auto *op = std::get_if<ComputeOp>(&instruction.body)) {
                auto result = values.at(op->inputs.at(0));
                const std::vector<double> *rhs = op->inputs.size() == 2 ? &values.at(op->inputs.at(1)) : nullptr;
                for (std::size_t i = 0; i < result.size(); ++i) {
                    switch (op->kind) {
                    case ComputeKind::AddCC: case ComputeKind::AddCP: result[i] += rhs->at(i); break;
                    case ComputeKind::SubCC: case ComputeKind::SubCP: result[i] -= rhs->at(i); break;
                    case ComputeKind::MulCC: case ComputeKind::MulCP: result[i] *= rhs->at(i); break;
                    case ComputeKind::Negate: result[i] = -result[i]; break;
                    case ComputeKind::Boot:
                        if (sine_boot) result[i] = manual_bootstrap_reference(context, result[i]);
                        break;
                    case ComputeKind::Rescale: case ComputeKind::ModSwitch: case ComputeKind::Relinearize: break;
                    default: throw std::runtime_error("unsupported operation in manual scalar reference");
                    }
                }
                values[op->output] = std::move(result);
            }
        }
    };
    phase(plan.initialization);
    phase(plan.execution);
    phase(plan.finalization);
    return values.at(plan.final_outputs.at(0));
}

} // namespace poseidon::test
