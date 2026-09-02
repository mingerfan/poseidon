#include "gpu_relu.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace poseidon::benchmark::resnet50_gpu
{
namespace
{

enum class EvalType
{
    none,
    odd_baby,
};

struct Tree
{
    int depth = 0;
    EvalType type = EvalType::none;
    std::vector<int> nodes{ -1, 0 };
    int m = 0;
    int l = 0;

    explicit Tree(EvalType eval_type = EvalType::none) : type(eval_type) {}

    void clear()
    {
        depth = 0;
        type = EvalType::none;
        nodes = { -1, 0 };
        m = 0;
        l = 0;
    }

    void merge(const Tree &left, const Tree &right, int split)
    {
        if (left.type != right.type)
        {
            throw std::invalid_argument("ReLU tree types do not match");
        }
        clear();
        type = left.type;
        depth = std::max(left.depth, right.depth) + 1;
        nodes.assign(std::size_t{1} << (depth + 1), -1);
        nodes[0] = -1;
        nodes[1] = split;
        for (int index = 1; index <= (1 << (left.depth + 1)) - 1; ++index)
        {
            const int offset = 1 << static_cast<int>(std::log2(index));
            nodes[index + offset] = left.nodes[index];
        }
        for (int index = 1; index <= (1 << (right.depth + 1)) - 1; ++index)
        {
            const int offset = 1 << static_cast<int>(std::log2(index));
            nodes[index + 2 * offset] = right.nodes[index];
        }
    }
};

int power_of_two(int exponent)
{
    if (exponent < 0 || exponent >= 30)
    {
        throw std::invalid_argument("invalid ReLU power-of-two exponent");
    }
    return 1 << exponent;
}

int population_count(int value)
{
    int count = 0;
    while (value > 0)
    {
        count += value & 1;
        value >>= 1;
    }
    return count;
}

void build_odd_baby_tree(int degree, Tree &tree)
{
    const int depth = static_cast<int>(
        std::ceil(std::log2(static_cast<double>(degree))));
    int best_total = 10000;
    int best_m = 0;
    int best_l = 0;
    Tree best_tree(EvalType::odd_baby);

    for (int l = 1; power_of_two(l) - 1 <= degree; ++l)
    {
        for (int m = 1; power_of_two(m - 1) < degree; ++m)
        {
            std::vector<std::vector<int>> costs(
                degree + 1, std::vector<int>(depth + 1, 0));
            std::vector<std::vector<Tree>> trees(
                degree + 1,
                std::vector<Tree>(depth + 1, Tree(EvalType::odd_baby)));
            costs[1][1] = 0;
            for (int odd_degree = 3; odd_degree <= degree; odd_degree += 2)
            {
                costs[odd_degree][1] = 10000;
            }
            for (int level = 2; level <= depth; ++level)
            {
                for (int odd_degree = 1; odd_degree <= degree; odd_degree += 2)
                {
                    if (odd_degree <= power_of_two(l) - 1 &&
                        odd_degree <= power_of_two(level - 1))
                    {
                        costs[odd_degree][level] = 0;
                        continue;
                    }
                    int best = 10000;
                    Tree candidate_tree;
                    for (int split_log = 1;
                         split_log <= m - 1 &&
                         power_of_two(split_log) < odd_degree &&
                         split_log < level;
                         ++split_log)
                    {
                        const int split = power_of_two(split_log);
                        const int candidate =
                            costs[odd_degree - split][level - 1] +
                            costs[split - 1][level] + 1;
                        if (candidate < best)
                        {
                            best = candidate;
                            candidate_tree.merge(
                                trees[split - 1][level],
                                trees[odd_degree - split][level - 1],
                                split);
                        }
                    }
                    costs[odd_degree][level] = best;
                    trees[odd_degree][level] = std::move(candidate_tree);
                }
            }
            const int total = costs[degree][depth] + power_of_two(l - 1) + m - 2;
            if (total < best_total)
            {
                best_total = total;
                best_tree = trees[degree][depth];
                best_m = m;
                best_l = l;
            }
        }
    }
    tree = std::move(best_tree);
    tree.type = EvalType::odd_baby;
    tree.m = best_m;
    tree.l = best_l;
}

int coefficient_count(int degree, const Tree &tree)
{
    const int tree_size = 1 << (tree.depth + 1);
    std::vector<int> decomposed_degree(tree_size, -1);
    decomposed_degree[1] = degree;
    for (int level = 1; level <= tree.depth; ++level)
    {
        for (int node = 1 << level; node < (1 << (level + 1)); ++node)
        {
            decomposed_degree[node] = (node % 2 == 0)
                ? tree.nodes[node / 2] - 1
                : decomposed_degree[node / 2] - tree.nodes[node / 2];
        }
    }
    int count = 0;
    for (int node = 1; node < tree_size; ++node)
    {
        if (tree.nodes[node] == 0)
        {
            count += decomposed_degree[node] + 1;
        }
    }
    return count;
}

std::vector<Tree> make_trees()
{
    std::vector<Tree> result(3, Tree(EvalType::odd_baby));
    build_odd_baby_tree(15, result[0]);
    build_odd_baby_tree(15, result[1]);
    build_odd_baby_tree(27, result[2]);
    return result;
}

std::vector<std::vector<double>> load_coefficients(
    const GpuReluConfig &config,
    const std::vector<Tree> &trees)
{
    const std::filesystem::path path =
        std::filesystem::path(POSEIDON_GPU_RESNET50_SOURCE_DIR) /
        "relu_param" /
        ("d" + std::to_string(config.alpha) + ".txt");
    std::ifstream stream(path);
    if (!stream)
    {
        throw std::runtime_error("failed to open GPU ReLU coefficients: " + path.string());
    }
    const std::vector<int> degrees{ 15, 15, 27 };
    std::vector<std::vector<double>> result;
    for (std::size_t component = 0; component < degrees.size(); ++component)
    {
        const int count = coefficient_count(degrees[component], trees[component]);
        std::vector<double> coefficients(count);
        for (double &coefficient : coefficients)
        {
            if (!(stream >> coefficient))
            {
                throw std::runtime_error("GPU ReLU coefficient file is truncated");
            }
        }
        result.push_back(std::move(coefficients));
    }
    for (double &coefficient : result[0])
    {
        coefficient /= 2.0;
    }
    for (double &coefficient : result[1])
    {
        coefficient /= config.scaled_value;
    }
    for (double &coefficient : result[2])
    {
        coefficient *= 0.5;
    }
    return result;
}

GpuCkksRuntime::DeviceCiphertext evaluate_chebyshev_basis(
    const GpuCkksRuntime::DeviceCiphertext &left,
    const GpuCkksRuntime::DeviceCiphertext &right,
    const GpuCkksRuntime::DeviceCiphertext &difference,
    const GpuCkksRuntime &runtime)
{
    auto product = runtime.multiply_relinearize_rescale(left, right);
    auto doubled = runtime.add(product, product);
    return runtime.sub_aligned(doubled, difference);
}

GpuCkksRuntime::DeviceCiphertext evaluate_component(
    const GpuCkksRuntime::DeviceCiphertext &input,
    int degree,
    const std::vector<double> &coefficients,
    const Tree &tree,
    const GpuCkksRuntime &runtime)
{
    const int total_depth = static_cast<int>(
        std::ceil(std::log2(static_cast<double>(degree + 1))));
    const int tree_size = 1 << (tree.depth + 1);
    std::vector<int> decomposed_degree(tree_size, -1);
    std::vector<int> coefficient_start(tree_size, -1);
    decomposed_degree[1] = degree;
    int next_coefficient = 1;
    for (int level = 1; level <= tree.depth; ++level)
    {
        for (int node = 1 << level; node < (1 << (level + 1)); ++node)
        {
            decomposed_degree[node] = (node % 2 == 0)
                ? tree.nodes[node / 2] - 1
                : decomposed_degree[node / 2] - tree.nodes[node / 2];
        }
    }
    for (int node = 1; node < tree_size; ++node)
    {
        if (tree.nodes[node] == 0)
        {
            coefficient_start[node] = next_coefficient;
            next_coefficient += decomposed_degree[node] + 1;
        }
    }

    std::vector<std::unique_ptr<GpuCkksRuntime::DeviceCiphertext>> basis(128);
    std::vector<std::unique_ptr<GpuCkksRuntime::DeviceCiphertext>> partial(128);
    auto one = runtime.encrypt_constant(1.0);
    basis[0] = std::make_unique<GpuCkksRuntime::DeviceCiphertext>(
        runtime.drop_to_q_count(one, input.meta.q_count));
    basis[1] = std::make_unique<GpuCkksRuntime::DeviceCiphertext>(
        runtime.drop_to_q_count(input, input.meta.q_count));

    for (int stage = 1; stage <= total_depth; ++stage)
    {
        for (int node = 1; node < tree_size; ++node)
        {
            if (tree.nodes[node] != 0 ||
                total_depth + 1 - population_count(node) != stage)
            {
                continue;
            }
            int coefficient_index = coefficient_start[node];
            auto accumulator = runtime.multiply_plain_scalar_rescale(
                *basis[1], coefficients.at(coefficient_index));
            coefficient_index += 2;
            for (int chebyshev_degree = 3;
                 chebyshev_degree <= decomposed_degree[node];
                 chebyshev_degree += 2)
            {
                if (!basis[chebyshev_degree])
                {
                    throw std::runtime_error("GPU ReLU is missing a Chebyshev basis");
                }
                auto term = runtime.multiply_plain_scalar_rescale(
                    *basis[chebyshev_degree],
                    coefficients.at(coefficient_index));
                accumulator = runtime.add_aligned(accumulator, term);
                coefficient_index += 2;
            }
            partial[node] = std::make_unique<GpuCkksRuntime::DeviceCiphertext>(
                std::move(accumulator));
        }

        for (int node = 1; node < tree_size; ++node)
        {
            if (tree.nodes[node] <= 0 || node % 2 == 0 ||
                total_depth + 1 - population_count(node) != stage)
            {
                continue;
            }
            int walk = node;
            auto accumulator = runtime.multiply_relinearize_rescale(
                *basis[tree.nodes[walk]], *partial[2 * walk + 1]);
            walk *= 2;
            while (tree.nodes[walk] != 0)
            {
                auto term = runtime.multiply_relinearize_rescale(
                    *basis[tree.nodes[walk]], *partial[2 * walk + 1]);
                accumulator = runtime.add_aligned(accumulator, term);
                walk *= 2;
            }
            accumulator = runtime.add_aligned(accumulator, *partial[walk]);
            partial[node] = std::make_unique<GpuCkksRuntime::DeviceCiphertext>(
                std::move(accumulator));
        }

        if (stage <= tree.m - 1)
        {
            const int chebyshev_degree = power_of_two(stage);
            basis[chebyshev_degree] =
                std::make_unique<GpuCkksRuntime::DeviceCiphertext>(
                    evaluate_chebyshev_basis(
                        *basis[power_of_two(stage - 1)],
                        *basis[power_of_two(stage - 1)],
                        *basis[0],
                        runtime));
        }
        if (stage <= tree.l)
        {
            for (int chebyshev_degree = power_of_two(stage - 1) + 1;
                 chebyshev_degree <= power_of_two(stage) - 1;
                 chebyshev_degree += 2)
            {
                basis[chebyshev_degree] =
                    std::make_unique<GpuCkksRuntime::DeviceCiphertext>(
                        evaluate_chebyshev_basis(
                            *basis[power_of_two(stage - 1)],
                            *basis[chebyshev_degree - power_of_two(stage - 1)],
                            *basis[power_of_two(stage) - chebyshev_degree],
                            runtime));
            }
        }
    }
    if (!partial[1])
    {
        throw std::runtime_error("GPU ReLU tree did not produce a root ciphertext");
    }
    return std::move(*partial[1]);
}

double evaluate_component_plain(
    double input,
    int degree,
    const std::vector<double> &coefficients,
    const Tree &tree)
{
    const int tree_size = 1 << (tree.depth + 1);
    std::vector<int> decomposed_degree(tree_size, -1);
    std::vector<int> coefficient_start(tree_size, -1);
    decomposed_degree[1] = degree;
    int next_coefficient = 1;
    for (int level = 1; level <= tree.depth; ++level)
    {
        for (int node = 1 << level; node < (1 << (level + 1)); ++node)
        {
            decomposed_degree[node] = (node % 2 == 0)
                ? tree.nodes[node / 2] - 1
                : decomposed_degree[node / 2] - tree.nodes[node / 2];
        }
    }
    for (int node = 1; node < tree_size; ++node)
    {
        if (tree.nodes[node] == 0)
        {
            coefficient_start[node] = next_coefficient;
            next_coefficient += decomposed_degree[node] + 1;
        }
    }
    std::vector<double> chebyshev(degree + 1, 0.0);
    chebyshev[0] = 1.0;
    chebyshev[1] = input;
    for (int index = 2; index <= degree; ++index)
    {
        chebyshev[index] = 2.0 * input * chebyshev[index - 1] - chebyshev[index - 2];
    }
    std::function<double(int)> evaluate = [&](int node) {
        if (tree.nodes[node] == 0)
        {
            int coefficient_index = coefficient_start[node];
            double result = chebyshev[1] * coefficients.at(coefficient_index);
            coefficient_index += 2;
            for (int index = 3; index <= decomposed_degree[node]; index += 2)
            {
                result += chebyshev[index] * coefficients.at(coefficient_index);
                coefficient_index += 2;
            }
            return result;
        }
        return chebyshev[tree.nodes[node]] * evaluate(2 * node + 1) +
               evaluate(2 * node);
    };
    return evaluate(1);
}

}  // namespace

GpuCkksRuntime::DeviceCiphertext polynomial_relu(
    const GpuCkksRuntime::DeviceCiphertext &input,
    const GpuCkksRuntime &runtime,
    const GpuReluConfig &config)
{
    const std::vector<int> degrees{ 15, 15, 27 };
    const auto trees = make_trees();
    const auto coefficients = load_coefficients(config, trees);
    auto step = runtime.drop_to_q_count(input, input.meta.q_count);
    for (std::size_t component = 0; component < degrees.size(); ++component)
    {
        step = evaluate_component(
            step, degrees[component], coefficients[component], trees[component], runtime);
        if (std::getenv("POSEIDON_GPU_RELU_DEBUG") != nullptr)
        {
            const auto decoded = runtime.decrypt(step);
            std::cerr << "GPU ReLU component " << component
                      << " q=" << step.meta.q_count
                      << " log2_scale=" << std::log2(step.meta.scale)
                      << " slots=";
            for (std::size_t index = 0; index < 9; ++index)
            {
                std::cerr << (index == 0 ? "" : ",") << decoded[index].real();
            }
            std::cerr << '\n';
        }
    }
    step = runtime.add_plain_scalar(step, 0.5);
    return runtime.multiply_relinearize_rescale(input, step);
}

GpuMultiplexedTensor polynomial_relu(
    const GpuMultiplexedTensor &input,
    const GpuCkksRuntime &runtime,
    const GpuReluConfig &config)
{
    input.validate();
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

double polynomial_relu_reference(double input, const GpuReluConfig &config)
{
    const std::vector<int> degrees{ 15, 15, 27 };
    static int cached_alpha = -1;
    static double cached_scaled_value = 0.0;
    static std::vector<Tree> cached_trees;
    static std::vector<std::vector<double>> cached_coefficients;
    if (cached_alpha != config.alpha ||
        cached_scaled_value != config.scaled_value ||
        cached_trees.empty() || cached_coefficients.empty())
    {
        cached_trees = make_trees();
        cached_coefficients = load_coefficients(config, cached_trees);
        cached_alpha = config.alpha;
        cached_scaled_value = config.scaled_value;
    }
    double step = input;
    for (std::size_t component = 0; component < degrees.size(); ++component)
    {
        step = evaluate_component_plain(
            step, degrees[component], cached_coefficients[component],
            cached_trees[component]);
        if (std::getenv("POSEIDON_GPU_RELU_DEBUG") != nullptr)
        {
            std::cerr << "Plain ReLU input=" << input
                      << " component=" << component
                      << " value=" << step << '\n';
        }
    }
    return input * (step + 0.5);
}

}  // namespace poseidon::benchmark::resnet50_gpu
