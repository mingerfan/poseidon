#include "poseidon/gpu/gpu_wide_ntt.h"
#include "poseidon/advance/bootstrapper.h"
#include "poseidon/encryptor.h"
#include "poseidon/factory/poseidon_factory.h"
#include "poseidon/keygenerator.h"
#include "poseidon/parameters_literal.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <string>

using namespace poseidon;
using namespace poseidon::gpu;

namespace
{
void check_modraise(const ParametersLiteral &literal)
{
    auto factory = PoseidonFactory::get_instance();
    factory->set_device_type(DEVICE_SOFTWARE);
    auto context = factory->create_poseidon_context(literal);
    auto evaluator = factory->create_ckks_evaluator(context);
    CKKSEncoder encoder(context);
    KeyGenerator keys(context);
    PublicKey pk;
    keys.create_public_key(pk);
    Encryptor encryptor(context, pk);

    const double scale = std::exp2(40);
    std::vector<double> slots(encoder.slot_count());
    for (std::size_t i = 0; i < slots.size(); ++i)
        slots[i] = std::sin(i * 0.031) / 2;
    Plaintext plaintext;
    encoder.encode(slots, scale, plaintext);
    Ciphertext input, q0, reference;
    encryptor.encrypt(plaintext, input);
    evaluator->drop_modulus(input, q0, 0);
    Bootstrapper bootstrap(context, *evaluator, encoder, literal.log_n() - 1, 12,
                            scale, scale);
    bootstrap.mod_raise(q0, reference);

    const auto degree = literal.degree();
    GpuWideNtt q0_ntt(literal.log_n(), {literal.q().front()}, 0);
    GpuWideNtt full_ntt(literal.log_n(), literal.q(), 0);
    DeviceVector<std::uint64_t> source(degree, 0), raised(degree * literal.q().size(), 0);
    std::vector<std::uint64_t> result(raised.size());
    for (std::size_t component = 0; component < q0.size(); ++component)
    {
        source.copy_from_host(q0.data(component), degree);
        q0_ntt.inverse(source);
        full_ntt.extend_centered_q0(source, raised);
        full_ntt.forward(raised);
        raised.copy_to_host(result.data(), result.size());
        const auto diff = std::mismatch(result.begin(), result.end(), reference.data(component));
        if (diff.first != result.end())
            throw std::runtime_error("ModRaise CPU mismatch at component " +
                std::to_string(component) + " word " + std::to_string(diff.first - result.begin()));
    }
    std::cout << "ModRaise q0->Q" << literal.q().size() << " components=" << q0.size()
              << " residues=" << result.size() * q0.size() << " match=EXACT\n";
}
} // namespace

int main(int argc, char **argv)
try
{
    if (argc != 2 || (std::string(argv[1]) != "15" && std::string(argv[1]) != "31"))
        throw std::invalid_argument("usage: functional_wide_ntt_probe 15|31");
    const bool d15 = std::string(argv[1]) == "15";
    constexpr int log_degree = 15;
    constexpr std::size_t degree = std::size_t{1} << log_degree;
    std::vector<std::uint32_t> q_bits{45, 40, 40, 40, 40};
    q_bits.resize(d15 ? 19 : 20, 45);
    ParametersLiteral literal(CKKS, log_degree, log_degree - 1, 40, 32, 0,
                              Modulus(0), {}, {}, sec_level_type::none);
    literal.set_log_modulus(q_bits, {51});
    auto moduli = literal.q();
    moduli.insert(moduli.end(), literal.p().begin(), literal.p().end());

    GpuWideNtt ntt(log_degree, moduli, 0);
    DeviceVector<std::uint64_t> device(moduli.size() * degree, 0);
    std::mt19937_64 random(0x933);
    std::vector<std::uint64_t> original(device.size()), cpu(device.size()), got(device.size());
    for (std::size_t limb = 0; limb < moduli.size(); ++limb)
    {
        const auto p = moduli[limb].value();
        auto *row = original.data() + limb * degree;
        for (std::size_t i = 0; i < degree; ++i)
            row[i] = random() % p;
        row[0] = 0;
        row[1] = 1;
        row[2] = p - 1;
        row[3] = p / 2;
    }

    cpu = original;
    for (std::size_t limb = 0; limb < moduli.size(); ++limb)
    {
        const util::NTTTables tables(log_degree, moduli[limb]);
        util::ntt_negacyclic_harvey(cpu.data() + limb * degree, tables);
    }

    device.copy_from_host(original.data(), original.size());
    ntt.forward(device);
    gpu_check_cuda(cudaStreamSynchronize(gpu_execution_stream()), "forward synchronize");
    device.copy_to_host(got.data(), got.size());
    const auto forward_diff = std::mismatch(cpu.begin(), cpu.end(), got.begin());
    if (forward_diff.first != cpu.end())
        throw std::runtime_error("forward CPU mismatch at word " +
                                 std::to_string(forward_diff.first - cpu.begin()));

    ntt.inverse(device);
    gpu_check_cuda(cudaStreamSynchronize(gpu_execution_stream()), "inverse synchronize");
    device.copy_to_host(got.data(), got.size());
    const auto inverse_diff = std::mismatch(original.begin(), original.end(), got.begin());
    if (inverse_diff.first != original.end())
        throw std::runtime_error("inverse round-trip mismatch at word " +
                                 std::to_string(inverse_diff.first - original.begin()));

    std::cout << "D" << argv[1] << " N=" << degree << " Q=" << q_bits.size()
              << " P=1 residues=" << got.size() << " forward=EXACT inverse=EXACT\n";
    check_modraise(literal);
    return 0;
}
catch (const std::exception &error)
{
    std::cerr << error.what() << '\n';
    return 1;
}
