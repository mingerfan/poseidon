#include "functional_wide_linear.h"
#include "functional_wide_polynomial.h"
#include "gpu_activity_timing.h"
#include "original_conv_reference.h"
#include "poseidon/advance/homomorphic_dft.h"
#include "poseidon/encryptor.h"
#include "poseidon/factory/poseidon_factory.h"
#include "poseidon/keygenerator.h"
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>

namespace
{
using namespace poseidon;
using namespace poseidon::gpu;
using probe::Buffer;
using probe::Layer;
using probe::WideValue;
using Complex = std::complex<double>;
using Diagonals = std::map<int, std::vector<double>>;
constexpr int log_n = 15;
constexpr int slots = 1 << (log_n - 1);
constexpr int pitch = 32;
constexpr int depth = 2;
constexpr double scale = 1099511627776.0;

struct WidePool
{
    rmm::mr::cuda_memory_resource upstream;
    rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> pool;
    rmm::mr::device_memory_resource *previous;
    explicit WidePool(std::size_t gib)
        : pool(&upstream, std::size_t{1} << 20, gib << 30),
          previous(rmm::mr::get_current_device_resource())
    {
        rmm::mr::set_current_device_resource(&pool);
    }
    ~WidePool() { rmm::mr::set_current_device_resource(previous); }
};

struct Fit
{
    double phase;
    std::vector<Complex> harmonic;
    std::vector<Complex> carrier;

    double evaluate(double input) const
    {
        if (!std::isfinite(input) || std::abs(input) > 1)
            throw std::runtime_error("plain activation exceeds Trident bound");
        const Complex phase_point = std::polar(1.0, 2.0 * std::acos(-1.0) * phase * input);
        Complex sum = 0;
        for (auto it = harmonic.rbegin(); it != harmonic.rend(); ++it)
            sum = sum * phase_point + *it;
        return 2 * sum.real();
    }
};

Fit load_fit(const std::filesystem::path &directory, unsigned degree)
{
    Fit fit;
    std::ifstream harmonic(directory / ("relu_d" + std::to_string(degree) + ".txt"));
    std::string magic, extra;
    unsigned file_degree = 0;
    double approximation_error = 0, constant = 0;
    if (!(harmonic >> magic >> fit.phase >> file_degree >> approximation_error >> constant) ||
        magic != "RELU_TRIG_V1" || file_degree != degree || !std::isfinite(fit.phase) ||
        fit.phase <= 0 || fit.phase >= 0.5 || !std::isfinite(approximation_error) ||
        approximation_error <= 0 || !std::isfinite(constant))
        throw std::invalid_argument("invalid frozen harmonic header");

    fit.harmonic.emplace_back(constant / 2, 0);
    for (unsigned k = 0; k < degree; ++k)
    {
        double re = 0, im = 0;
        if (!(harmonic >> re >> im) || !std::isfinite(re) || !std::isfinite(im))
            throw std::invalid_argument("invalid frozen harmonic coefficient");
        fit.harmonic.emplace_back(re / 2, -im / 2);
    }
    if (harmonic >> extra)
        throw std::invalid_argument("unexpected trailing harmonic coefficient");

    std::ifstream carrier(directory / "carrier_k12_d31.txt");
    unsigned bound = 0, squares = 0;
    if (!(carrier >> magic >> bound >> squares >> file_degree) ||
        magic != "COMPLEX_CARRIER_V1" || bound != 12 || squares != 3 || file_degree != 31)
        throw std::invalid_argument("invalid frozen carrier header");
    for (unsigned k = 0; k <= file_degree; ++k)
    {
        double re = 0, im = 0;
        if (!(carrier >> re >> im) || !std::isfinite(re) || !std::isfinite(im))
            throw std::invalid_argument("invalid frozen carrier coefficient");
        fit.carrier.emplace_back(re, im);
    }
    if (carrier >> extra)
        throw std::invalid_argument("unexpected trailing carrier coefficient");
    return fit;
}

// Keep the physical stride of the original image in every stage. When
// resolution halves, additional channels occupy the freed column phases.
int slot_index(int channel, int row, int col, int side)
{
    const int spatial_stride = pitch / side;
    return (channel % 16) * pitch * pitch +
           spatial_stride * (row * pitch + col) + channel / 16;
}

void diagonal_entry(Diagonals &diagonals, int source, int target, double value)
{
    const int step = (source - target + slots) & (slots - 1);
    auto &row = diagonals.try_emplace(step, slots, 0.0).first->second;
    row.at(target) += value;
}

class RotationKeys
{
public:
    RotationKeys(KeyGenerator &generator, const util::GaloisTool &tool)
        : generator_(generator), tool_(tool)
    {}

    const GaloisKeys &get(const std::set<int> &steps)
    {
        std::vector<int> missing;
        for (int step : steps)
            if (!keys_.has_key(tool_.get_elt_from_step(step)))
                missing.push_back(step);
        if (!missing.empty())
        {
            GaloisKeys generated;
            generator_.create_galois_keys(missing, generated);
            keys_.parms_id() = generated.parms_id();
            keys_.data().resize(std::max(keys_.data().size(), generated.data().size()));
            for (int step : missing)
            {
                const auto index = GaloisKeys::get_index(tool_.get_elt_from_step(step));
                keys_.data().at(index) = std::move(generated.data().at(index));
            }
        }
        return keys_;
    }

private:
    KeyGenerator &generator_;
    const util::GaloisTool &tool_;
    GaloisKeys keys_;
};

class Network
{
public:
    Network(const ParametersLiteral &params, CKKSEncoder &encoder, KeyGenerator &keygen,
            KeyGenerator &app_keygen, const RelinKeys &relin, const Fit &fit)
        : params_(params), encoder_(encoder), fit_(fit),
          galois_(*encoder.context().crt_context()->galois_tool()),
          bootstrap_keys_(keygen, galois_), application_keys_(app_keygen, galois_),
          polynomial_(params, encoder, relin, 0),
          raise_single_(params.log_n(), {params.q().front()}, 0),
          raise_full_(params.log_n(), params.q(), 0)
    {
        const auto top = params_.q().size() - 1;
        const double gain = q0() / scale * fit_.phase;
        HomomorphicDFTMatrixLiteral stc(decode, log_n, log_n - 1, depth, {1, 1},
                                      true, 1.0, false, 1);
        auto diagonals = stc.gen_matrices();
        for (auto &[rotation, column] : diagonals.back())
            for (auto &entry : column)
                entry *= gain;

        stc_.data().resize(depth);
        for (std::size_t i = 0; i < stc_.data().size(); ++i)
        {
            const auto level = depth - i;
            gen_linear_transform_bsgs(stc_.data()[i], stc_.rot_index(), encoder_,
                diagonals.at(i), level, prime(level), 1, log_n - 1);
        }
        const char *ratio_env = std::getenv("POSEIDON_CTS_LOG_BSGS_RATIO");
        const std::string ratio_text = ratio_env ? ratio_env : "2";
        if (ratio_text != "0" && ratio_text != "1" && ratio_text != "2" && ratio_text != "3")
            throw std::invalid_argument("CtS log BSGS ratio must be 0, 1, 2, or 3");
        const auto cts_ratio = static_cast<unsigned>(std::stoul(ratio_text));
        std::cout << "CTS log_bsgs_ratio=" << cts_ratio << '\n';
        HomomorphicDFTMatrixLiteral cts(encode, log_n, log_n - 1, top, {1, 1},
                                      true, 1.0 / 12, false, cts_ratio);
        cts.create(cts_, encoder_, 1);
        std::set<int> steps{0};
        steps.insert(stc_.rot_index().begin(), stc_.rot_index().end());
        steps.insert(cts_.rot_index().begin(), cts_.rot_index().end());
        std::cout << "bootstrap_key_setup steps=" << steps.size() << '\n';
        bootstrap_keys_.get(steps);
    }

    WideValue run(WideValue input, const ApplicationNetworkFixture &fixture)
    {
        activations_ = 0;
        linear_index_ = 0;
        tag("convolution");
        auto value = activate(conv(input, fixture.stem), "stem");
        for (const auto &block : fixture.blocks)
        {
            const std::string id = "stage" + std::to_string(block.stage) +
                                   ".block" + std::to_string(block.block);
            tag("convolution");
            auto branch = activate(conv(value, block.conv1), id + ".conv1");
            tag("convolution");
            auto after_conv = conv(branch, block.conv2);
            tag("residual");
            auto skip = block.option_a_shortcut ? shortcut(value, block.conv1)
                                                : drop_to(value, after_conv.level);
            layer(after_conv.level).add(skip.residues, after_conv.residues);
            value = activate(std::move(after_conv), id + ".residual");
        }
        if (activations_ != 19)
            throw std::runtime_error("expected exactly 19 fused activations");
        tag("head");
        return head(std::move(value), fixture);
    }

    std::size_t activations() const { return activations_; }
    void quiet(bool enabled) { quiet_ = enabled; }
    void set_activity(poseidon::benchmark::s2c_first::GpuActivityTiming *activity)
    {
        activity_ = activity;
        if (activity) { cpu_label_ = "online"; cpu_begin_ = std::chrono::steady_clock::now(); }
    }
    void finish_cpu_stage()
    {
        if (!cpu_label_.empty())
        {
            cpu_ms_[cpu_label_] += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - cpu_begin_).count();
            cpu_label_.clear();
        }
    }
    void print_cpu_stages() const
    {
        for (const auto &[label, ms] : cpu_ms_)
            std::cout << "CPU_STAGE name=" << label << " submit_wall_ms=" << ms << '\n';
    }
    void print_resident_bytes() const
    {
        std::size_t plains = 0, keys = 0;
        for (const auto &entry : layers_)
        {
            const auto [p, k] = entry.second->resident_bytes();
            plains += p;
            keys += k;
        }
        std::cout << "BENCHMARK resident_plain_bytes=" << plains
                  << " resident_rotation_key_bytes=" << keys << '\n';
    }

private:
    void tag(const char *label)
    {
        if (activity_)
        {
            finish_cpu_stage();
            activity_->stage("", false);
            activity_->stage(label, true);
            cpu_label_ = label;
            cpu_begin_ = std::chrono::steady_clock::now();
        }
    }
    double prime(std::size_t level) const { return static_cast<double>(params_.q().at(level).value()); }
    double q0() const { return prime(0); }

    Layer &layer(std::size_t level)
    {
        auto &entry = layers_[level];
        if (!entry)
            entry = std::make_unique<Layer>(params_, level + 1);
        return *entry;
    }

    WideValue drop_to(const WideValue &input, std::size_t level)
    {
        if (level > input.level)
            throw std::invalid_argument("cannot increase Q by dropping limbs");
        const auto words = (level + 1) * params_.degree();
        const auto original_words = input.residues.size() / 2;
        Buffer out(2 * words, 0);
        for (std::size_t c = 0; c < 2; ++c)
            gpu_check_cuda(cudaMemcpyAsync(out.data() + c * words,
                input.residues.data() + c * original_words, words * sizeof(std::uint64_t),
                cudaMemcpyDeviceToDevice, gpu_execution_stream()), "ResNet20 drop Q");
        return {std::move(out), level, input.scale};
    }

    void bias(WideValue &value, const std::vector<double> &coefficients)
    {
        Plaintext encoded;
        encoder_.encode(coefficients,
            encoder_.context().crt_context()->parms_id_map().at(value.level),
            value.scale, encoded);
        layer(value.level).add_plain(encoded, value.residues);
    }

    WideValue diagonal(const WideValue &input, const MatrixPlain &matrix, RotationKeys &bank,
                       bool stable_plaintexts = true, bool trace_cts = false)
    {
        const auto [index, giants, babies] = bsgs_index(matrix.plain_vec, slots, matrix.n1);
        std::set<int> steps(giants.begin(), giants.end());
        steps.insert(babies.begin(), babies.end());
        steps.erase(0);
        const auto &keys = bank.get(steps);
        std::function<void(const char *)> trace;
        if (trace_cts && activity_) trace = [this](const char *label) { tag(label); };
        const int hoist = trace_cts && std::getenv("POSEIDON_WIDE_HOIST_DIGITS") ? 2 :
                          trace_cts && std::getenv("POSEIDON_WIDE_HOIST") ? 1 : 0;
        auto result = layer(input.level).transform(input.residues, matrix, index, babies, keys, galois_, stable_plaintexts, trace, hoist);
        return {std::move(result), input.level - 1, input.scale * matrix.scale / prime(input.level)};
    }

    WideValue linear(const WideValue &input, Diagonals &diagonals)
    {
        auto [entry, inserted] = linear_cache_.try_emplace(linear_index_++);
        if (inserted)
        {
            std::vector<int> rotations;
            gen_linear_transform_bsgs(entry->second, rotations, encoder_, diagonals, input.level,
                                     prime(input.level), 2, log_n - 1);
        }
        return diagonal(input, entry->second, application_keys_);
    }

    WideValue activate(WideValue input, const std::string &site)
    {
        tag("stc");
        auto value = drop_to(input, depth);
        for (const auto &matrix : stc_.data())
            value = diagonal(value, matrix, bootstrap_keys_);

        tag("modraise");
        value = {probe::modraise(value.residues, params_, 0, &raise_single_, &raise_full_), params_.q().size() - 1, q0()};
        tag("cts");
        for (const auto &matrix : cts_.data())
            value = diagonal(value, matrix, bootstrap_keys_, true, true);
        tag("cts_projection");
        const auto &keys = bootstrap_keys_.get({0});
        layer(value.level).project_real(value.residues, keys, galois_);

        tag("carrier");
        value = polynomial_.evaluate(value, fit_.carrier, q0(), true);
        tag("carrier_squares");
        for (int j = 0; j < 3; ++j)
            value = polynomial_.square(value);
        tag("harmonic");
        value = polynomial_.evaluate(value, fit_.harmonic, scale, false);
        tag("projection_out");
        layer(value.level).project_real(value.residues, keys, galois_);
        if (value.level != 4 || !std::isfinite(value.scale) || std::abs(value.scale / scale - 1) > 1e-4)
            throw std::runtime_error(site + " unexpected activation output");
        ++activations_;
        if (!quiet_)
        {
            gpu_check_cuda(cudaStreamSynchronize(gpu_execution_stream()), "activation completion");
            std::cout << "ACTIVATION " << activations_ << " " << site << " L4 scale_bits="
                      << std::log2(value.scale) << '\n';
        }
        return value;
    }

    WideValue conv(const WideValue &input, const ApplicationLayerFixture &config)
    {
        auto existing = conv_cache_.find(&config);
        if (existing != conv_cache_.end())
        {
            auto result = diagonal(input, existing->second.first, application_keys_);
            layer(result.level).add_plain(existing->second.second, result.residues, true);
            return result;
        }
        const int side = config.input_h;
        const int output_side = side / config.stride;
        Diagonals diagonals;
        std::vector<double> offsets(slots);
        for (int oc = 0; oc < config.output_channels; ++oc)
        for (int r = 0; r < output_side; ++r)
        for (int c = 0; c < output_side; ++c)
        {
            const int target = slot_index(oc, r, c, output_side);
            offsets[target] = config.bn_bias.at(oc);
            for (int ic = 0; ic < config.input_channels; ++ic)
            for (int kh = 0; kh < 3; ++kh)
            for (int kw = 0; kw < 3; ++kw)
            {
                const int ih = config.stride * r + kh - 1;
                const int iw = config.stride * c + kw - 1;
                if (ih >= 0 && ih < side && iw >= 0 && iw < side)
                {
                    const int weight = ((oc * config.input_channels + ic) * 3 + kh) * 3 + kw;
                    diagonal_entry(diagonals, slot_index(ic, ih, iw, side), target,
                        config.weights.at(weight) * config.bn_scale.at(oc));
                }
            }
        }
        auto &entry = conv_cache_[&config];
        std::vector<int> rotations;
        gen_linear_transform_bsgs(entry.first, rotations, encoder_, diagonals, input.level,
                                 prime(input.level), 2, log_n - 1);
        auto result = diagonal(input, entry.first, application_keys_);
        encoder_.encode(offsets, encoder_.context().crt_context()->parms_id_map().at(result.level),
                        result.scale, entry.second);
        layer(result.level).add_plain(entry.second, result.residues, true);
        return result;
    }

    WideValue shortcut(const WideValue &input, const ApplicationLayerFixture &config)
    {
        auto cached = linear_cache_.find(linear_index_);
        if (cached != linear_cache_.end())
        {
            ++linear_index_;
            return diagonal(input, cached->second, application_keys_);
        }
        Diagonals diagonals;
        const int side = config.input_h / 2;
        for (int ch = 0; ch < config.input_channels; ++ch)
        for (int r = 0; r < side; ++r)
        for (int c = 0; c < side; ++c)
            diagonal_entry(diagonals, slot_index(ch, 2 * r, 2 * c, 2 * side),
                slot_index(ch + config.input_channels / 2, r, c, side), 1.0);
        return linear(input, diagonals);
    }

    WideValue head(WideValue value, const ApplicationNetworkFixture &fixture)
    {
        const std::set<int> steps{4, 8, 16, 128, 256, 512};
        const auto &keys = application_keys_.get(steps);
        Buffer rotated(value.residues.size(), 0);
        auto &operator_at_level = layer(value.level);
        for (int step : steps)
        {
            operator_at_level.rotate(value.residues, step, keys, galois_, rotated);
            operator_at_level.add(rotated, value.residues);
        }
        Diagonals average;
        if (!linear_cache_.count(linear_index_))
            average.emplace(0, std::vector<double>(slots, 40.0 / 64));
        value = linear(value, average);
        Diagonals fc;
        if (!linear_cache_.count(linear_index_))
            for (int out = 0; out < 10; ++out)
            for (int feature = 0; feature < 64; ++feature)
                diagonal_entry(fc, slot_index(feature, 0, 0, 8), out,
                    fixture.linear_weight.at(out * 64 + feature));
        auto result = linear(value, fc);
        if (!head_bias_)
        {
            auto logits_bias = fixture.linear_bias;
            logits_bias.resize(slots);
            head_bias_ = std::make_unique<Plaintext>();
            encoder_.encode(logits_bias,
                encoder_.context().crt_context()->parms_id_map().at(result.level),
                result.scale, *head_bias_);
        }
        layer(result.level).add_plain(*head_bias_, result.residues, true);
        return result;
    }

    const ParametersLiteral &params_;
    CKKSEncoder &encoder_;
    const Fit &fit_;
    const util::GaloisTool &galois_;
    RotationKeys bootstrap_keys_, application_keys_;
    probe::WidePolynomial polynomial_;
    GpuWideNtt raise_single_, raise_full_;
    LinearMatrixGroup stc_, cts_;
    std::map<std::size_t, std::unique_ptr<Layer>> layers_;
    std::map<const ApplicationLayerFixture *, std::pair<MatrixPlain, Plaintext>> conv_cache_;
    std::map<std::size_t, MatrixPlain> linear_cache_;
    std::unique_ptr<Plaintext> head_bias_;
    std::size_t linear_index_ = 0;
    std::size_t activations_ = 0;
    bool quiet_ = false;
    poseidon::benchmark::s2c_first::GpuActivityTiming *activity_ = nullptr;
    std::string cpu_label_;
    std::chrono::steady_clock::time_point cpu_begin_;
    std::map<std::string, double> cpu_ms_;
};
} // namespace

int main(int argc, char **argv)
try
{
    if ((argc != 4 && argc != 5) || (std::string(argv[1]) != "15" && std::string(argv[1]) != "31"))
        throw std::invalid_argument("usage: resnet20_wide_functional 15|31 image_id coefficients_dir [--benchmark|--batch=count]");
    const std::string option = argc == 5 ? argv[4] : "";
    const bool benchmark = option == "--benchmark";
    int batch_count = 1;
    if (option.rfind("--batch=", 0) == 0)
        batch_count = std::stoi(option.substr(8));
    else if (!option.empty() && !benchmark)
        throw std::invalid_argument("unknown option");
    std::unique_ptr<WidePool> memory_pool;
    if (const char *raw_pool = std::getenv("POSEIDON_WIDE_POOL_GIB"))
    {
        std::size_t used = 0;
        const auto gib = std::stoul(raw_pool, &used);
        if (used != std::string(raw_pool).size() || gib < 1 || gib > 64)
            throw std::invalid_argument("POSEIDON_WIDE_POOL_GIB must be 1..64");
        memory_pool = std::make_unique<WidePool>(gib);
        std::cout << "WIDE_POOL max_gib=" << gib << '\n';
    }
    const unsigned degree = std::stoul(argv[1]);
    const int image_id = std::stoi(argv[2]);
    if (image_id < 0 || batch_count < 1 || image_id + batch_count > 1000)
        throw std::invalid_argument("image_id outside bundled test set");
    std::cout << std::unitbuf << std::setprecision(10);
    const auto fit = load_fit(argv[3], degree);
    double max_plain_activation = 0;
    const auto plain_activation = [&](double value) {
        max_plain_activation = std::max(max_plain_activation, std::abs(value));
        return fit.evaluate(value);
    };
    auto fixture = load_application_network_fixture(image_id,
        plain_activation);

    std::vector<std::uint32_t> q_bits{45, 40, 40, 40, 40};
    q_bits.resize(degree == 15 ? 19 : 20, 45);
    ParametersLiteral params(CKKS, log_n, log_n - 1, 40, 32, 0, Modulus(0),
                             {}, {}, sec_level_type::none);
    params.set_log_modulus(q_bits, {51});
    auto factory = PoseidonFactory::get_instance();
    factory->set_device_type(DEVICE_SOFTWARE);
    auto context = factory->create_poseidon_context(params);
    CKKSEncoder encoder(context);
    KeyGenerator keys(context);
    RelinKeys relin;
    keys.create_relin_keys(relin);

    // Application rotations never need more than five Q limbs. Derive the
    // lower-level key context from the same secret's Q prefix and its P limb.
    auto app_params = params;
    constexpr std::size_t app_q = 5;
    app_params.set_modulus({params.q().begin(), params.q().begin() + app_q}, params.p());
    auto app_context = factory->create_poseidon_context(app_params);
    SecretKey app_secret;
    const auto ring = params.degree();
    app_secret.data().resize(app_context, app_context.crt_context()->key_parms_id(), (app_q + 1) * ring);
    const auto *secret = keys.secret_key().data().data();
    std::copy_n(secret, app_q * ring, app_secret.data().data());
    std::copy_n(secret + params.q().size() * ring, ring, app_secret.data().data() + app_q * ring);
    KeyGenerator app_keys(app_context, app_secret);

    PublicKey public_key;
    keys.create_public_key(public_key);
    Encryptor encryptor(context, public_key);
    Plaintext encoded;
    std::vector<double> image(slots);
    std::copy(fixture.image.begin(), fixture.image.end(), image.begin());
    auto evaluator = factory->create_ckks_evaluator(context);
    encoder.encode(image, context.crt_context()->parms_id_map().at(4), scale, encoded);
    Ciphertext encrypted;
    encryptor.encrypt(encoded, encrypted);
    const auto input = [&] {
        Buffer input_gpu(2 * app_q * ring, 0);
        input_gpu.copy_from_host(encrypted.data(0), input_gpu.size());
        return input_gpu;
    };
    std::cout << "image=" << image_id << " degree=" << degree << " N=" << ring
              << " Q=" << params.q().size() << " P=1\n";

    Network network(params, encoder, keys, app_keys, relin, fit);
    WideValue output;
    if (benchmark)
    {
        std::cout << "BENCHMARK warmup=begin\n";
        {
            auto warmup = network.run({input(), 4, scale}, fixture);
            gpu_check_cuda(cudaStreamSynchronize(gpu_execution_stream()), "benchmark warmup");
        }
        std::size_t free_bytes = 0, total_bytes = 0;
        gpu_check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes), "benchmark memory");
        network.print_resident_bytes();
        std::cout << "BENCHMARK memory_free_bytes=" << free_bytes
                  << " total_bytes=" << total_bytes << '\n';
        std::cout << "BENCHMARK warmup=complete\n";

        auto resident_input = input();
        gpu_check_cuda(cudaStreamSynchronize(gpu_execution_stream()), "benchmark input residency");
        network.quiet(true);
        poseidon::benchmark::s2c_first::GpuActivityTiming activity;
        activity.begin_continuous();
        activity.stage("online", true);
        cudaEvent_t begin = nullptr, end = nullptr;
        gpu_check_cuda(cudaEventCreate(&begin), "benchmark begin event");
        gpu_check_cuda(cudaEventCreate(&end), "benchmark end event");
        const auto wall_begin = std::chrono::steady_clock::now();
        gpu_check_cuda(cudaEventRecord(begin, gpu_execution_stream()), "benchmark begin");
        network.set_activity(&activity);
        output = network.run({std::move(resident_input), 4, scale}, fixture);
        network.finish_cpu_stage();
        gpu_check_cuda(cudaEventRecord(end, gpu_execution_stream()), "benchmark end");
        const auto enqueue_end = std::chrono::steady_clock::now();
        gpu_check_cuda(cudaEventSynchronize(end), "benchmark completion");
        const auto wall_end = std::chrono::steady_clock::now();
        float cuda_ms = 0;
        gpu_check_cuda(cudaEventElapsedTime(&cuda_ms, begin, end), "benchmark elapsed");
        gpu_check_cuda(cudaEventDestroy(begin), "benchmark begin cleanup");
        gpu_check_cuda(cudaEventDestroy(end), "benchmark end cleanup");
        activity.stage("online", false);
        network.set_activity(nullptr);
        const auto timing = activity.finish();
        timing.print();
        network.print_cpu_stages();
        for (const auto &[label, stage] : timing.stages)
            std::cout << "GPU_STAGE name=" << label << " ms=" << stage.gpu_ms << '\n';
        const auto wall_ms = std::chrono::duration<double, std::milli>(wall_end - wall_begin).count();
        std::cout << "ONLINE_TIMING mode=warm_resident wall_ms=" << wall_ms
                  << " cuda_event_ms=" << cuda_ms
                  << " host_enqueue_ms=" << std::chrono::duration<double, std::milli>(enqueue_end - wall_begin).count()
                  << " final_wait_ms=" << std::chrono::duration<double, std::milli>(wall_end - enqueue_end).count()
                  << " input_upload_included=false"
                  << " public_material_upload_included=false\n";
    }
    else
        output = network.run({input(), 4, scale}, fixture);
    Decryptor decryptor(context, keys.secret_key());
    probe::WidePolynomial observer(params, encoder, relin, 0);
    const auto check_output = [&]() {
    const auto decoded = observer.decode(output, decryptor);
    std::vector<double> logits;
    double max_error = 0;
    for (std::size_t i = 0; i < fixture.reference_logits.size(); ++i)
    {
        if (!std::isfinite(decoded.at(i).real()) || !std::isfinite(decoded[i].imag()))
            throw std::runtime_error("nonfinite logit");
        logits.push_back(decoded[i].real());
        max_error = std::max(max_error, std::abs(logits.back() - fixture.reference_logits[i]));
        std::cout << "LOGIT " << i << " gpu=" << logits.back()
                  << " plain=" << fixture.reference_logits[i] << '\n';
    }
    const auto prediction = std::max_element(logits.begin(), logits.end()) - logits.begin();
    const auto reference = std::max_element(fixture.reference_logits.begin(),
        fixture.reference_logits.end()) - fixture.reference_logits.begin();
    auto sorted_logits = fixture.reference_logits;
    std::partial_sort(sorted_logits.begin(), sorted_logits.begin() + 2, sorted_logits.end(), std::greater<double>());
    const bool pass = max_error <= 0.01 && prediction == reference;
    std::cout << "NETWORK image=" << fixture.image_id << " activations=" << network.activations() << " L" << output.level
              << " predicted=" << prediction << " plain=" << reference << " label=" << fixture.true_label
              << " scale=" << output.scale << " max_logit_error=" << max_error
              << " plain_margin=" << sorted_logits[0] - sorted_logits[1]
              << " plain_max_abs_activation=" << max_plain_activation
              << " result=" << (pass ? "PASS" : "FAIL") << '\n';
    return pass;
    };
    bool all_pass = check_output();
    network.quiet(true);
    for (int index = image_id + 1; index < image_id + batch_count; ++index)
    {
        try
        {
            max_plain_activation = 0;
            auto next = load_application_network_fixture(index, plain_activation);
            // Keep the original model objects alive because GPU caches refer to them.
            fixture.image_id = next.image_id;
            fixture.true_label = next.true_label;
            fixture.image = std::move(next.image);
            fixture.reference_logits = std::move(next.reference_logits);
            std::fill(image.begin(), image.end(), 0);
            std::copy(fixture.image.begin(), fixture.image.end(), image.begin());
            encoder.encode(image, context.crt_context()->parms_id_map().at(4), scale, encoded);
            encryptor.encrypt(encoded, encrypted);
            output = network.run({input(), 4, scale}, fixture);
            all_pass = check_output() && all_pass;
        }
        catch (const std::exception &error)
        {
            std::cerr << "NETWORK image=" << index << " result=FAIL reason=" << error.what() << '\n';
            all_pass = false;
        }
    }
    return all_pass ? 0 : 1;
}
catch (const std::exception &error)
{
    std::cerr << error.what() << '\n';
    return 1;
}
