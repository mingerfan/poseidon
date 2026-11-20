#include "functional_wide_polynomial.h"
#include "functional_wide_linear.h"
#include "poseidon/advance/bootstrapper.h"
#include "poseidon/advance/homomorphic_dft.h"
#include "poseidon/encryptor.h"
#include "poseidon/factory/poseidon_factory.h"
#include "poseidon/keygenerator.h"

#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <string>

using namespace poseidon;
using namespace poseidon::gpu;

namespace
{
using Word = std::uint64_t;
using Buffer = DeviceVector<Word>;
using probe::Layer;
using probe::modraise;
constexpr int device_id = 0;
constexpr int depth = 2;
constexpr int carrier_bound = 12;
constexpr double scale = 1099511627776.0;

void compare(const std::string &stage, const Buffer &device, const Ciphertext &oracle)
{
    const std::size_t words = oracle.coeff_modulus_size() * oracle.poly_modulus_degree();
    const std::size_t size = 2 * words;
    if (device.size() != size || oracle.size() != 2 || !oracle.is_ntt_form())
        throw std::runtime_error(stage + " shape mismatch");
    std::vector<Word> host(size);
    device.copy_to_host(host.data(), size);
    for (std::size_t component = 0; component < 2; ++component)
    {
        const auto first = host.begin() + component * words;
        const auto last = first + words;
        const auto diff = std::mismatch(first, last, oracle.data(component));
        if (diff.first != last)
            throw std::runtime_error(stage + " mismatch at component " +
                std::to_string(component) + " word " + std::to_string(diff.first - first));
    }
    std::cout << stage << " L" << oracle.level() << " scale_bits=" << std::log2(oracle.scale())
              << " residues=" << size << " match=EXACT\n";
}

void run_dft(const std::string &name, const LinearMatrixGroup &matrices,
             const ParametersLiteral &params, EvaluatorCkksBase &cpu, KeyGenerator &keygen,
             const util::GaloisTool &galois, Ciphertext &reference, Buffer &device)
{
    std::size_t stage = 0;
    for (const auto &matrix : matrices.data())
    {
        const auto [index, giants, babies] =
            bsgs_index(matrix.plain_vec, 1 << matrix.log_slots, matrix.n1);
        std::set<int> unique(giants.begin(), giants.end());
        unique.insert(babies.begin(), babies.end());
        unique.erase(0);
        const std::string label = name + " stage " + std::to_string(++stage);
        const auto prior_level = reference.level();
        if (prior_level != matrix.level || prior_level == 0)
            throw std::runtime_error(label + " matrix at unexpected level");

        GaloisKeys keys;
        std::cout << label << " prepare keys=" << unique.size()
                  << " diagonals=" << matrix.plain_vec.size() << " n1=" << matrix.n1 << '\n';
        keygen.create_galois_keys(std::vector<int>(unique.begin(), unique.end()), keys);
        Layer layer(params, prior_level + 1);
        device = layer.transform(device, matrix, index, babies, keys, galois);
        cpu.multiply_by_diag_matrix_bsgs(reference, matrix, reference, keys);
        if (reference.level() != prior_level - 1)
            throw std::runtime_error(label + " rescale did not drop exactly one modulus");
        compare(label, device, reference);
    }
}

using Complex = std::complex<double>;

std::vector<Complex> load_carrier(const char *path)
{
    std::ifstream file(path);
    std::string magic, extra;
    unsigned bound = 0, doublings = 0, degree = 0;
    if (!(file >> magic >> bound >> doublings >> degree) ||
        magic != "COMPLEX_CARRIER_V1" || bound != carrier_bound ||
        doublings != 3 || degree != 31)
        throw std::invalid_argument("unexpected Trident carrier header");

    std::vector<Complex> coefficients(degree + 1);
    for (auto &coefficient : coefficients)
    {
        double real = 0, imag = 0;
        if (!(file >> real >> imag) || !std::isfinite(real) || !std::isfinite(imag))
            throw std::invalid_argument("invalid carrier coefficient");
        coefficient = {real, imag};
    }
    if (file >> extra)
        throw std::invalid_argument("unexpected trailing carrier coefficient");
    return coefficients;
}

void check_slots(const std::string &stage, const probe::WideValue &value,
                 const std::vector<Complex> &actual, const std::vector<Complex> &expected,
                 std::size_t level, double scale, double tolerance)
{
    if (actual.size() != expected.size() || value.level != level ||
        !std::isfinite(value.scale) || std::abs(value.scale / scale - 1) > 1e-4)
        throw std::runtime_error(stage + " level, scale or slot-count mismatch");
    double error = 0;
    for (std::size_t i = 0; i < actual.size(); ++i)
    {
        const double difference = std::abs(actual[i] - expected[i]);
        if (!std::isfinite(difference))
            throw std::runtime_error(stage + " nonfinite decoded slot");
        error = std::max(error, difference);
    }
    std::cout << stage << " L" << value.level << " scale_bits=" << std::log2(value.scale)
              << " slots=" << actual.size() << " max_error=" << error
              << " tolerance=" << tolerance << '\n';
    if (error > tolerance)
        throw std::runtime_error(stage + " exceeds decoded tolerance");
}

void run_nonlinear(const ParametersLiteral &params, CKKSEncoder &encoder,
                   KeyGenerator &keygen, const util::GaloisTool &galois,
                   const GaloisKeys &projection_key, const std::vector<Complex> &carrier,
                   const std::vector<Complex> &harmonic, const std::vector<double> &input_slots,
                   double phase, double fit_error, probe::WideValue value)
{
    RelinKeys relin;
    keygen.create_relin_keys(relin);
    Decryptor decryptor(encoder.context(), keygen.secret_key());
    probe::WidePolynomial polynomial(params, encoder, relin, device_id);
    const auto cts_slots = polynomial.decode(value, decryptor);
    const auto start_level = value.level;
    const double q0 = value.scale;
    double modular_error = 0;
    double max_cts = 0;
    std::vector<Complex> expected(cts_slots.size());
    for (std::size_t i = 0; i < cts_slots.size(); ++i)
    {
        const auto x = cts_slots[i];
        const double offset = carrier_bound * x.real() - phase * input_slots.at(i);
        if (!std::isfinite(offset) || !std::isfinite(x.imag()))
            throw std::runtime_error("CtS nonfinite slot");
        modular_error = std::max(modular_error, std::abs(offset - std::round(offset)));
        max_cts = std::max(max_cts, std::abs(x));
        Complex t0 = 1, t1 = x;
        expected[i] = carrier[0] + carrier[1] * x;
        for (std::size_t k = 2; k < carrier.size(); ++k)
        {
            const Complex next = 2.0 * x * t1 - t0;
            expected[i] += carrier[k] * next;
            t0 = t1;
            t1 = next;
        }
    }
    std::cout << "CtS modular_error=" << modular_error << " max_abs=" << max_cts << '\n';
    if (modular_error > 1e-5 || max_cts > 1)
        throw std::runtime_error("CtS domain or phase recovery failed");

    value = polynomial.evaluate(value, carrier, q0, true);
    check_slots("carrier C31", value, polynomial.decode(value, decryptor), expected,
                start_level - 5, q0, 1e-6);
    for (std::size_t j = 1; j <= 3; ++j)
    {
        value = polynomial.square(value);
        for (auto &z : expected)
            z *= z;
        check_slots("periodic square " + std::to_string(j), value,
                    polynomial.decode(value, decryptor), expected, start_level - 5 - j,
                    q0, 1e-6);
    }

    const auto periodic = polynomial.decode(value, decryptor);
    for (std::size_t i = 0; i < periodic.size(); ++i)
    {
        expected[i] = 0;
        for (auto coefficient = harmonic.rbegin(); coefficient != harmonic.rend(); ++coefficient)
            expected[i] = expected[i] * periodic[i] + *coefficient;
    }
    value = polynomial.evaluate(value, harmonic, scale, false);
    check_slots("harmonic", value, polynomial.decode(value, decryptor), expected, 4, scale, 1e-6);

    Layer projection(params, value.level + 1);
    projection.project_real(value.residues, projection_key, galois);
    double approximation_error = 0;
    for (std::size_t i = 0; i < expected.size(); ++i)
    {
        const Complex phase_point = std::polar(1.0, 2 * std::acos(-1.0) * phase * input_slots[i]);
        Complex sum = 0;
        for (auto coefficient = harmonic.rbegin(); coefficient != harmonic.rend(); ++coefficient)
            sum = sum * phase_point + *coefficient;
        expected[i] = 2 * sum.real();
        approximation_error = std::max(approximation_error,
            std::abs(expected[i].real() - std::max(0.0, input_slots[i])));
    }
    check_slots("fused output vs frozen fit", value, polynomial.decode(value, decryptor),
                expected, 4, scale, std::max(1e-6, fit_error * 0.1));
    std::cout << "frozen fit vs ReLU max_error=" << approximation_error << " PASS\n";
}
} // namespace

int main(int argc, char **argv)
try
{
    if ((argc != 3 && argc != 4) || (std::string(argv[1]) != "15" && std::string(argv[1]) != "31"))
        throw std::invalid_argument("usage: functional_wide_dft_probe 15|31 trident-relu-coefficients [trident-carrier]");
    const bool d15 = std::string(argv[1]) == "15";
    std::ifstream coefficients(argv[2]);
    std::string magic;
    unsigned harmonic_degree = 0;
    double phase = 0;
    if (!(coefficients >> magic >> phase >> harmonic_degree) || magic != "RELU_TRIG_V1" ||
        harmonic_degree != (d15 ? 15U : 31U) || !std::isfinite(phase) || phase <= 0 || phase >= 0.5)
        throw std::invalid_argument("unexpected Trident coefficient header");

    double fit_error = 0;
    std::vector<Complex> harmonic;
    std::vector<Complex> carrier;
    if (argc == 4)
    {
        double constant = 0;
        if (!(coefficients >> fit_error >> constant) || !std::isfinite(fit_error) ||
            fit_error <= 0 || !std::isfinite(constant))
            throw std::invalid_argument("invalid Trident harmonic header");
        harmonic.emplace_back(constant / 2, 0);
        for (unsigned k = 0; k < harmonic_degree; ++k)
        {
            double cosine = 0, sine = 0;
            if (!(coefficients >> cosine >> sine) || !std::isfinite(cosine) || !std::isfinite(sine))
                throw std::invalid_argument("invalid Trident harmonic coefficient");
            harmonic.emplace_back(cosine / 2, -sine / 2);
        }
        std::string extra;
        if (coefficients >> extra)
            throw std::invalid_argument("unexpected trailing harmonic coefficient");
        carrier = load_carrier(argv[3]);
    }

    std::vector<std::uint32_t> q_bits{45, 40, 40, 40, 40};
    q_bits.resize(d15 ? 19 : 20, 45);
    ParametersLiteral params(CKKS, 15, 14, 40, 32, 0, Modulus(0), {}, {}, sec_level_type::none);
    params.set_log_modulus(q_bits, {51});
    auto factory = PoseidonFactory::get_instance();
    factory->set_device_type(DEVICE_SOFTWARE);
    auto context = factory->create_poseidon_context(params);
    auto cpu = factory->create_ckks_evaluator(context);
    CKKSEncoder encoder(context);
    KeyGenerator keygen(context);
    PublicKey key;
    keygen.create_public_key(key);
    const auto galois = context.crt_context()->galois_tool();

    std::cout << std::unitbuf << "D" << harmonic_degree
              << " N=" << params.degree() << " Q=" << q_bits.size() << " P=1 phase=" << phase << '\n';
    std::vector<double> values(encoder.slot_count());
    for (std::size_t i = 0; i < values.size(); ++i)
        values[i] = std::sin(i * 0.031) / 2;
    if (!carrier.empty())
    {
        for (std::size_t i = 0; i < values.size(); ++i)
            values[i] = -1.0 + 2.0 * i / (values.size() - 1);
        values[values.size() / 2] = 0;
        values[1] = -1e-8;
        values[2] = 1e-8;
        std::mt19937 random(1339);
        std::shuffle(values.begin(), values.end(), random);
    }
    Plaintext plain;
    encoder.encode(values, scale, plain);
    Ciphertext reference;
    Encryptor encryptor(context, key);
    encryptor.encrypt(plain, reference);
    cpu->drop_modulus(reference, reference, depth);
    Buffer device(2 * params.degree() * (depth + 1), device_id);
    device.copy_from_host(reference.data(0), device.size());

    const double q0 = static_cast<double>(params.q().front().value());
    HomomorphicDFTMatrixLiteral stc_literal(decode, params.log_n(), params.log_slots(),
        depth, std::vector<std::uint32_t>(depth, 1), true, 1.0, false, 1);
    auto stc_diagonals = stc_literal.gen_matrices();
    for (auto &[rotation, diagonal] : stc_diagonals.back())
        for (auto &entry : diagonal)
            entry *= q0 / scale * phase;
    LinearMatrixGroup stc;
    stc.data().resize(depth);
    for (std::size_t j = 0; j < stc_diagonals.size(); ++j)
    {
        const auto level = depth - j;
        gen_linear_transform_bsgs(stc.data().at(j), stc.rot_index(), encoder, stc_diagonals[j],
            level, static_cast<double>(params.q()[level].value()), 1, params.log_slots());
    }
    run_dft("StC", stc, params, *cpu, keygen, *galois, reference, device);

    Bootstrapper bootstrap(context, *cpu, encoder, params.log_slots(), carrier_bound, scale, scale);
    Ciphertext raised;
    bootstrap.mod_raise(reference, raised);
    raised.scale() = q0;
    reference = std::move(raised);
    device = modraise(device, params);
    compare("ModRaise", device, reference);

    HomomorphicDFTMatrixLiteral cts_literal(encode, params.log_n(), params.log_slots(),
        q_bits.size() - 1, std::vector<std::uint32_t>(depth, 1), true,
        1.0 / carrier_bound, false, 2);
    LinearMatrixGroup cts;
    cts_literal.create(cts, encoder, 1);
    run_dft("CtS", cts, params, *cpu, keygen, *galois, reference, device);
    GaloisKeys projection_key;
    keygen.create_galois_keys(std::vector<int>{0}, projection_key);
    Layer projection(params, reference.level() + 1);
    projection.project_real(device, projection_key, *galois);
    Ciphertext conjugate;
    cpu->conjugate(reference, projection_key, conjugate);
    cpu->add(reference, conjugate, reference);
    compare("CtS real projection", device, reference);
    if (!carrier.empty())
        run_nonlinear(params, encoder, keygen, *galois, projection_key, carrier, harmonic,
                      values, phase, fit_error,
                      {std::move(device), reference.level(), reference.scale()});
    return 0;
}
catch (const std::exception &error)
{
    std::cerr << error.what() << '\n';
    return 1;
}
