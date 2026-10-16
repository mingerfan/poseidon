#include "gpu_relu.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace poseidon::benchmark::resnet50_gpu
{
namespace
{

using DeviceCiphertext = GpuCkksRuntime::DeviceCiphertext;
constexpr int kEntryQ = 31;

struct Node
{
    std::string type;
    int work = 0;
    int output = 0;
    int degree = 0;
    int split = 0;
    double pre = 0.0;
    double scale = 0.0;
    std::vector<double> coefficients;
    std::unique_ptr<Node> quotient;
    std::unique_ptr<Node> remainder;
};

struct Basis
{
    int degree = 0;
    int work = 0;
    int output = 0;
    double pre = 0.0;
    double scale = 0.0;
};

struct Stage
{
    int degree = 0;
    int start = 0;
    int end = 0;
    double input_scale = 0.0;
    double output_scale = 0.0;
    std::vector<Basis> basis;
    std::unique_ptr<Node> tree;
};

struct Fixture
{
    std::vector<Stage> stages;
    int tail_work = 0;
    int tail_drop = 0;
};

std::unique_ptr<Node> read_node(std::istream &stream)
{
    auto node = std::make_unique<Node>();
    stream >> node->type >> node->work >> node->output >> node->pre >> node->scale;
    if (node->type == "leaf")
    {
        stream >> node->degree;
        if (node->degree < 1 || node->degree > 27 || node->degree % 2 == 0)
        {
            throw std::runtime_error("invalid S2C-first ReLU leaf degree");
        }
        node->coefficients.resize((node->degree + 1) / 2);
        for (double &coefficient : node->coefficients)
        {
            stream >> coefficient;
        }
    }
    else if (node->type == "combine")
    {
        stream >> node->split;
        node->quotient = read_node(stream);
        node->remainder = read_node(stream);
    }
    else
    {
        throw std::runtime_error("invalid S2C-first ReLU node type");
    }
    if (!stream)
    {
        throw std::runtime_error("truncated S2C-first ReLU fixture");
    }
    return node;
}

const Fixture &fixture()
{
    static Fixture value;
    static std::once_flag loaded;
    std::call_once(loaded, []() {
        std::ifstream stream(POSEIDON_RESNET20_RELU_FIXTURE);
        if (!stream)
        {
            throw std::runtime_error(
                "failed to open S2C-first ReLU fixture");
        }
        std::string magic;
        int q_count = 0;
        int p_count = 0;
        stream >> magic >> q_count >> p_count;
        if (magic != "RELU_PRECISION_V1" || q_count != 50 || p_count != 25)
        {
            throw std::runtime_error("invalid S2C-first ReLU fixture header");
        }
        for (int index = 0; index < q_count + p_count; ++index)
        {
            std::uint64_t modulus = 0;
            stream >> modulus;
        }
        int stage_count = 0;
        stream >> stage_count;
        if (!stream || stage_count != 3)
        {
            throw std::runtime_error("S2C-first ReLU requires three stages");
        }
        value.stages.resize(stage_count);
        for (auto &stage : value.stages)
        {
            int basis_count = 0;
            stream >> stage.degree >> stage.start >> stage.end >>
                stage.input_scale >> stage.output_scale >> basis_count;
            if (!stream || basis_count < 1 || basis_count > 32)
            {
                throw std::runtime_error("invalid S2C-first ReLU stage");
            }
            stage.basis.resize(basis_count);
            for (auto &basis : stage.basis)
            {
                stream >> basis.degree >> basis.work >> basis.output >>
                    basis.pre >> basis.scale;
            }
            stage.tree = read_node(stream);
        }
        stream >> value.tail_work >> value.tail_drop;
        if (!stream || value.tail_work + value.tail_drop != 22)
        {
            throw std::runtime_error("invalid S2C-first ReLU tail");
        }
    });
    return value;
}

DeviceCiphertext drop(
    const GpuCkksRuntime &runtime,
    const DeviceCiphertext &input,
    int work)
{
    const int target_q_count = kEntryQ - work;
    if (target_q_count <= 0 || input.meta.q_count < target_q_count)
    {
        throw std::runtime_error("invalid S2C-first ReLU drop target");
    }
    return runtime.drop_to_q_count(input, static_cast<std::size_t>(target_q_count));
}

DeviceCiphertext rescale_to(
    const GpuCkksRuntime &runtime,
    const DeviceCiphertext &input,
    int output_work,
    double output_scale)
{
    const int target_q_count = kEntryQ - output_work;
    if (target_q_count <= 0 || input.meta.q_count < target_q_count)
    {
        throw std::runtime_error(
            "invalid S2C-first ReLU rescale target: output_work=" +
            std::to_string(output_work) + " target_q=" +
            std::to_string(target_q_count) + " current_q=" +
            std::to_string(input.meta.q_count) + " log2_scale=" +
            std::to_string(std::log2(input.meta.scale)));
    }
    const std::size_t drop_count =
        input.meta.q_count - static_cast<std::size_t>(target_q_count);
    DeviceCiphertext result = drop_count == 0
        ? runtime.drop_to_q_count(input, input.meta.q_count)
        : runtime.rescale(input, static_cast<std::uint32_t>(drop_count));
    const double actual_scale = std::log2(result.meta.scale);
    if (!std::isfinite(actual_scale) ||
        std::abs(actual_scale - output_scale) > 1.0e-6)
    {
        throw std::runtime_error(
            "S2C-first ReLU rescale scale mismatch: expected=" +
            std::to_string(output_scale) + " actual=" +
            std::to_string(actual_scale));
    }
    return result;
}

DeviceCiphertext multiply(
    const GpuCkksRuntime &runtime,
    const DeviceCiphertext &left,
    const DeviceCiphertext &right)
{
    return runtime.multiply_relinearize(left, right);
}

DeviceCiphertext scalar(
    const GpuCkksRuntime &runtime,
    double value,
    const DeviceCiphertext &source,
    double scale)
{
    return runtime.multiply_plain_scalar(source, value, scale);
}

DeviceCiphertext evaluate_node(
    const GpuCkksRuntime &runtime,
    const Node &node,
    const std::map<int, DeviceCiphertext> &basis)
{
    DeviceCiphertext output;
    if (node.type == "leaf")
    {
        std::optional<DeviceCiphertext> accumulator;
        for (std::size_t index = 0; index < node.coefficients.size(); ++index)
        {
            DeviceCiphertext term =
                drop(runtime, basis.at(static_cast<int>(2 * index + 1)), node.work);
            DeviceCiphertext product = scalar(
                runtime, node.coefficients[index], term,
                std::exp2(node.pre) / term.meta.scale);
            if (!accumulator)
            {
                accumulator = std::move(product);
            }
            else
            {
                if (std::abs(std::log2(product.meta.scale) -
                             std::log2(accumulator->meta.scale)) > 1.0e-6)
                {
                    throw std::runtime_error(
                        "S2C-first ReLU leaf alignment scale mismatch");
                }
                product.meta.scale = accumulator->meta.scale;
                accumulator = runtime.add_aligned(*accumulator, product);
            }
        }
        if (!accumulator)
        {
            throw std::runtime_error("empty S2C-first ReLU leaf");
        }
        output = std::move(*accumulator);
    }
    else
    {
        DeviceCiphertext quotient = evaluate_node(runtime, *node.quotient, basis);
        DeviceCiphertext remainder = evaluate_node(runtime, *node.remainder, basis);
        DeviceCiphertext ratio =
            drop(runtime, basis.at(node.split), node.work);
        quotient = drop(runtime, std::move(quotient), node.work);
        remainder = drop(runtime, std::move(remainder), node.work);
        DeviceCiphertext product = multiply(runtime, ratio, quotient);
        if (std::abs(std::log2(remainder.meta.scale) -
                     std::log2(product.meta.scale)) > 1.0e-6)
        {
            throw std::runtime_error(
                "S2C-first ReLU combine alignment scale mismatch");
        }
        remainder.meta.scale = product.meta.scale;
        output = runtime.add_aligned(product, remainder);
    }
    if (std::abs(std::log2(output.meta.scale) - node.pre) > 1.0e-6)
    {
        throw std::runtime_error("S2C-first ReLU node scale mismatch");
    }
    return rescale_to(
        runtime, std::move(output), node.output, node.scale);
}

DeviceCiphertext evaluate_stage(
    const GpuCkksRuntime &runtime,
    const Stage &stage,
    DeviceCiphertext input)
{
    if (std::abs(std::log2(input.meta.scale) - stage.input_scale) > 1.0e-8)
    {
        throw std::runtime_error("S2C-first ReLU stage input scale mismatch");
    }
    std::map<int, DeviceCiphertext> basis;
    basis.emplace(1, drop(runtime, std::move(input), stage.start));
    static std::atomic<int> basis_debug_budget{6};
    const bool basis_debug =
        std::getenv("POSEIDON_RELU_DEBUG") != nullptr &&
        basis_debug_budget.fetch_sub(1) > 0;
    const auto dump = [&](const char *label, const DeviceCiphertext &value) {
        if (!basis_debug)
        {
            return;
        }
        const auto decoded = runtime.decrypt(value);
        std::cout << "RELU_BASIS_DEBUG " << label
                  << " q=" << value.meta.q_count
                  << " log2_scale=" << std::log2(value.meta.scale)
                  << " v0=" << decoded[0].real() << '\n';
    };
    for (const auto &spec : stage.basis)
    {
        int left = 1;
        while (2 * left < spec.degree)
        {
            left *= 2;
        }
        const int right = spec.degree - left;
        const int difference = left - right;
        DeviceCiphertext left_value =
            drop(runtime, basis.at(left), spec.work);
        DeviceCiphertext right_value =
            drop(runtime, basis.at(right), spec.work);
        DeviceCiphertext product = multiply(runtime, left_value, right_value);
        dump(("product_degree" + std::to_string(spec.degree)).c_str(), product);
        DeviceCiphertext twice = runtime.add(product, product);
        DeviceCiphertext corrected;
        if (difference == 0)
        {
            DeviceCiphertext one = runtime.encrypt_constant(1.0);
            one = runtime.drop_to_q_count(
                one, twice.meta.q_count);
            DeviceCiphertext scaled_one = scalar(
                runtime, 1.0, one, twice.meta.scale / one.meta.scale);
            corrected = runtime.sub_aligned(twice, scaled_one);
        }
        else
        {
            DeviceCiphertext difference_value =
                drop(runtime, basis.at(difference), spec.work);
            DeviceCiphertext aligned = scalar(
                runtime, 1.0, difference_value,
                twice.meta.scale / difference_value.meta.scale);
            if (std::abs(std::log2(aligned.meta.scale) -
                         std::log2(twice.meta.scale)) > 1.0e-6)
            {
                throw std::runtime_error(
                    "S2C-first ReLU basis alignment scale mismatch");
            }
            aligned.meta.scale = twice.meta.scale;
            corrected = runtime.sub_aligned(twice, aligned);
        }
        if (std::abs(std::log2(corrected.meta.scale) - spec.pre) > 1.0e-6)
        {
            throw std::runtime_error("S2C-first ReLU basis scale mismatch");
        }
        dump(("corrected_degree" + std::to_string(spec.degree)).c_str(), corrected);
        basis.emplace(
            spec.degree,
            rescale_to(
                runtime, std::move(corrected), spec.output, spec.scale));
        dump(("basis_degree" + std::to_string(spec.degree)).c_str(),
             basis.at(spec.degree));
    }
    return evaluate_node(runtime, *stage.tree, basis);
}

long double chebyshev(int degree, long double value)
{
    long double previous = 1.0L;
    long double current = value;
    if (degree == 0)
    {
        return previous;
    }
    for (int index = 2; index <= degree; ++index)
    {
        const long double next = 2.0L * value * current - previous;
        previous = current;
        current = next;
    }
    return current;
}

long double evaluate_plain_node(const Node &node, long double value)
{
    if (node.type == "leaf")
    {
        long double result = 0.0L;
        for (std::size_t index = 0; index < node.coefficients.size(); ++index)
        {
            result += node.coefficients[index] *
                chebyshev(static_cast<int>(2 * index + 1), value);
        }
        return result;
    }
    return chebyshev(node.split, value) *
            evaluate_plain_node(*node.quotient, value) +
        evaluate_plain_node(*node.remainder, value);
}

}  // namespace

GpuCkksRuntime::DeviceCiphertext polynomial_relu(
    const GpuCkksRuntime::DeviceCiphertext &input,
    const GpuCkksRuntime &runtime,
    const GpuReluConfig &)
{
    if (input.meta.q_count != static_cast<std::size_t>(kEntryQ) ||
        std::abs(std::log2(input.meta.scale) - 40.0) > 1.0e-8)
    {
        throw std::invalid_argument(
            "S2C-first ReLU requires a Q31/scale40 input");
    }
    DeviceCiphertext original =
        runtime.drop_to_q_count(input, input.meta.q_count);
    DeviceCiphertext current =
        runtime.drop_to_q_count(input, input.meta.q_count);
    static std::atomic<int> debug_budget{3};
    const bool debug =
        std::getenv("POSEIDON_RELU_DEBUG") != nullptr &&
        debug_budget.fetch_sub(1) > 0;
    const auto dump = [&](const char *label, const DeviceCiphertext &value) {
        if (!debug)
        {
            return;
        }
        const auto decoded = runtime.decrypt(value);
        std::cout << "RELU_DEBUG " << label
                  << " q=" << value.meta.q_count
                  << " log2_scale=" << std::log2(value.meta.scale);
        for (std::size_t index = 0;
             index < std::min<std::size_t>(decoded.size(), 4);
             ++index)
        {
            std::cout << " v" << index << "=" << decoded[index].real();
        }
        std::cout << '\n';
    };
    dump("input", current);
    const auto &plan = fixture();
    for (std::size_t stage_index = 0;
         stage_index < plan.stages.size();
         ++stage_index)
    {
        current = evaluate_stage(
            runtime, plan.stages[stage_index], std::move(current));
        dump(("stage" + std::to_string(stage_index + 1)).c_str(), current);
    }
    DeviceCiphertext shifted = runtime.add_plain_scalar(current, 0.5);
    DeviceCiphertext left = drop(runtime, original, plan.tail_work);
    DeviceCiphertext right = drop(runtime, std::move(shifted), plan.tail_work);
    DeviceCiphertext product = multiply(runtime, left, right);
    DeviceCiphertext result = rescale_to(
        runtime, std::move(product),
        plan.tail_work + plan.tail_drop, 40.0);
    dump("output", result);
    return result;
}

GpuMultiplexedTensor polynomial_relu(
    const GpuMultiplexedTensor &input,
    const GpuCkksRuntime &runtime,
    const GpuReluConfig &config)
{
    GpuMultiplexedTensor output;
    output.h = input.h;
    output.w = input.w;
    output.c = input.c;
    output.k = input.k;
    output.pages_per_cipher = input.pages_per_cipher;
    output.page_size = input.page_size;
    output.slot_count = input.slot_count;
    output.packs.reserve(input.packs.size());
    for (const auto &pack : input.packs)
    {
        output.packs.push_back(polynomial_relu(pack, runtime, config));
    }
    return output;
}

double polynomial_relu_reference(double input, const GpuReluConfig &)
{
    const auto &plan = fixture();
    long double value = input;
    for (const auto &stage : plan.stages)
    {
        value = evaluate_plain_node(*stage.tree, value);
    }
    return static_cast<double>(input * (value + 0.5L));
}

}  // namespace poseidon::benchmark::resnet50_gpu
