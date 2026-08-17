#include "poseidon/runtime_api/poseidon_cpu_api.h"

#include "poseidon/ckks_encoder.h"
#include "poseidon/decryptor.h"
#include "poseidon/encryptor.h"
#include "poseidon/evaluator/software/evaluator_ckks_software.h"
#include "poseidon/key/galoiskeys.h"
#include "poseidon/key/relinkeys.h"
#include "poseidon/runtime_api/rotation_key_basis.h"
#include "runtime/utils/sha256.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string_view>
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

std::vector<int> available_rotation_steps(const PoseidonContext &context,
                                          const GaloisKeys &keys, int requested_step)
{
    const std::size_t slot_count = context.parameters_literal()->slot();
    const int normalized = normalize_rotation_step(requested_step, slot_count);
    if (normalized == 0)
    {
        return {};
    }

    const auto galois_tool = context.crt_context()->galois_tool();
    if (keys.has_key(galois_tool->get_elt_from_step(normalized)))
    {
        return {normalized};
    }

    auto steps = decompose_rotation_step(normalized, slot_count);
    for (int step : steps)
    {
        if (!keys.has_key(galois_tool->get_elt_from_step(step)))
        {
            throw std::runtime_error("Poseidon CPU Api lacks a binary rotation key");
        }
    }
    return steps;
}

double exact_scale(int scale_log2)
{
    return std::ldexp(1.0, scale_log2);
}

#if defined(POSEIDON_RUNTIME_CPU_MPI)
using TraceClock = std::chrono::steady_clock;

std::uint64_t elapsed_nanoseconds(TraceClock::time_point start, TraceClock::time_point finish)
{
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count());
}

std::string trace_path_for_rank(std::string path, int rank)
{
    const std::string rank_text = std::to_string(rank);
    std::size_t offset = 0;
    bool replaced = false;
    while ((offset = path.find("%r", offset)) != std::string::npos)
    {
        path.replace(offset, 2, rank_text);
        offset += rank_text.size();
        replaced = true;
    }
    if (!replaced)
    {
        path += ".rank" + rank_text + ".csv";
    }
    return path;
}

void check_mpi(int code, const char *operation)
{
    if (code == MPI_SUCCESS)
    {
        return;
    }

    char message[MPI_MAX_ERROR_STRING];
    int length = 0;
    MPI_Error_string(code, message, &length);
    throw std::runtime_error(std::string(operation) + " failed: " +
                             std::string(message, static_cast<std::size_t>(length)));
}

std::vector<poseidon_byte> serialize_value(const PoseidonCpuValue &value)
{
    const auto mode = compr_mode_type::none;
    const std::streamoff size = value.kind() == fhegpu::ValueKind::Plaintext
                                    ? value.plaintext().save_size(mode)
                                    : value.ciphertext().save_size(mode);
    if (size <= 0 || static_cast<std::uint64_t>(size) >
                         static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
    {
        throw std::runtime_error("Poseidon CPU MPI serialized value exceeds MPI count range");
    }

    std::vector<poseidon_byte> bytes(static_cast<std::size_t>(size));
    const std::streamoff written = value.kind() == fhegpu::ValueKind::Plaintext
                                       ? value.plaintext().save(bytes.data(), bytes.size(), mode)
                                       : value.ciphertext().save(bytes.data(), bytes.size(), mode);
    if (written != size)
    {
        throw std::runtime_error("Poseidon CPU MPI serialization size mismatch");
    }
    return bytes;
}

PoseidonCpuValue deserialize_value(const PoseidonContext &context, fhegpu::ValueKind kind,
                                   const std::vector<poseidon_byte> &bytes)
{
    if (kind == fhegpu::ValueKind::Plaintext)
    {
        Plaintext value;
        if (value.load(context, bytes.data(), bytes.size()) !=
            static_cast<std::streamoff>(bytes.size()))
        {
            throw std::runtime_error("Poseidon CPU MPI plaintext load size mismatch");
        }
        // Plaintext::load restores the bytes but does not rebuild its RNSPoly view.
        value.resize(context, value.parms_id(), value.coeff_count());
        return PoseidonCpuValue::from_plaintext(std::move(value));
    }

    Ciphertext value;
    if (value.load(context, bytes.data(), bytes.size()) !=
        static_cast<std::streamoff>(bytes.size()))
    {
        throw std::runtime_error("Poseidon CPU MPI ciphertext load size mismatch");
    }
    return PoseidonCpuValue::from_ciphertext(std::move(value));
}

std::string context_sha256(const PoseidonContext &context)
{
    const auto parameters = context.parameters_literal();
    std::string bytes;
    const auto append_u64 = [&](std::uint64_t value) {
        for (int shift = 56; shift >= 0; shift -= 8)
        {
            bytes.push_back(static_cast<char>((value >> shift) & 0xffU));
        }
    };
    append_u64(static_cast<std::uint64_t>(parameters->scheme()));
    append_u64(parameters->degree());
    append_u64(parameters->slot());
    append_u64(parameters->log_scale());
    append_u64(parameters->hamming_weight());
    append_u64(parameters->q0_level());
    append_u64(parameters->plain_modulus().value());
    append_u64(static_cast<std::uint64_t>(parameters->sec_level()));
    append_u64(parameters->q().size());
    for (const auto &modulus : parameters->q())
    {
        append_u64(modulus.value());
    }
    append_u64(parameters->p().size());
    for (const auto &modulus : parameters->p())
    {
        append_u64(modulus.value());
    }
    return fhegpu::sha256_hex(bytes);
}

void require_same_string(MPI_Comm communicator, int world_size, std::string_view local,
                         const char *what)
{
    const std::uint64_t local_size = local.size();
    std::vector<std::uint64_t> sizes(static_cast<std::size_t>(world_size));
    check_mpi(MPI_Allgather(&local_size, 1, MPI_UINT64_T, sizes.data(), 1, MPI_UINT64_T,
                            communicator),
              "MPI_Allgather(string size)");
    for (std::uint64_t size : sizes)
    {
        if (size != local_size)
        {
            throw std::runtime_error(std::string(what) + " length mismatch across MPI ranks");
        }
    }
    if (local_size > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
    {
        throw std::runtime_error(std::string(what) + " exceeds MPI count range");
    }

    std::vector<char> all(static_cast<std::size_t>(world_size) * local.size());
    check_mpi(MPI_Allgather(local.data(), static_cast<int>(local.size()), MPI_CHAR, all.data(),
                            static_cast<int>(local.size()), MPI_CHAR, communicator),
              "MPI_Allgather(string)");
    for (int rank = 0; rank < world_size; ++rank)
    {
        const char *remote = all.data() + static_cast<std::size_t>(rank) * local.size();
        if (!std::equal(local.begin(), local.end(), remote))
        {
            throw std::runtime_error(std::string(what) + " mismatch across MPI ranks");
        }
    }
}
#endif

} // namespace

#if defined(POSEIDON_RUNTIME_CPU_MPI)
struct PoseidonCpuApi::MpiState
{
    MPI_Comm communicator = MPI_COMM_NULL;
    int tag_upper_bound = -1;
    std::shared_ptr<std::ofstream> trace;

    ~MpiState()
    {
        int finalized = 0;
        if (communicator != MPI_COMM_NULL && MPI_Finalized(&finalized) == MPI_SUCCESS && !finalized)
        {
            MPI_Comm_free(&communicator);
        }
    }
};

struct PoseidonCpuApi::CommState
{
    struct Send
    {
        std::array<MPI_Request, 2> requests{MPI_REQUEST_NULL, MPI_REQUEST_NULL};
    };
    struct Receive
    {
        std::size_t slot = 0;
        int source_rank = 0;
        fhegpu::ValueKind kind = fhegpu::ValueKind::Ciphertext;
        std::uint64_t size = 0;
        MPI_Request size_request = MPI_REQUEST_NULL;
    };

    fhegpu::TransferId id = 0;
    std::uint64_t send_size = 0;
    std::vector<poseidon_byte> send_bytes;
    std::vector<Send> sends;
    std::vector<Receive> receives;
    std::shared_ptr<std::ofstream> trace;
    std::uint64_t serialize_nanoseconds = 0;
    std::uint64_t post_nanoseconds = 0;
    std::uint64_t size_wait_nanoseconds = 0;
    std::uint64_t body_wait_nanoseconds = 0;
    std::uint64_t deserialize_nanoseconds = 0;
    std::uint64_t wait_nanoseconds = 0;
    std::uint64_t request_lifetime_nanoseconds = 0;
    TraceClock::time_point requests_posted_at{};
    bool waited = false;
};
#else
struct PoseidonCpuApi::MpiState
{
};

struct PoseidonCpuApi::CommState
{
    bool waited = false;
};
#endif

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
                               std::shared_ptr<const GaloisKeys> galois_keys,
                               std::shared_ptr<const PublicKey> boot_public_key,
                               std::shared_ptr<const SecretKey> boot_secret_key)
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
    if (static_cast<bool>(boot_public_key) != static_cast<bool>(boot_secret_key))
    {
        throw std::invalid_argument(
            "Poseidon CPU decrypt_reencrypt Boot requires public and secret keys");
    }
    if (boot_public_key)
    {
        boot_encryptor_ = std::make_unique<Encryptor>(context_, *boot_public_key);
        boot_decryptor_ = std::make_unique<Decryptor>(context_, *boot_secret_key);
    }
}

#if defined(POSEIDON_RUNTIME_CPU_MPI)
PoseidonCpuApi::PoseidonCpuApi(std::string context_id, PoseidonContext context,
                               MPI_Comm communicator,
                               std::shared_ptr<const RelinKeys> relin_keys,
                               std::shared_ptr<const GaloisKeys> galois_keys,
                               std::shared_ptr<const PublicKey> boot_public_key,
                               std::shared_ptr<const SecretKey> boot_secret_key)
    : PoseidonCpuApi(std::move(context_id), std::move(context), std::move(relin_keys),
                     std::move(galois_keys), std::move(boot_public_key),
                     std::move(boot_secret_key))
{
    if (communicator == MPI_COMM_NULL)
    {
        throw std::invalid_argument("Poseidon CPU MPI communicator is null");
    }
    int initialized = 0;
    check_mpi(MPI_Initialized(&initialized), "MPI_Initialized");
    if (!initialized)
    {
        throw std::runtime_error("Poseidon CPU MPI mode requires MPI_Init_thread");
    }
    int finalized = 0;
    check_mpi(MPI_Finalized(&finalized), "MPI_Finalized");
    if (finalized)
    {
        throw std::runtime_error("Poseidon CPU MPI mode cannot use finalized MPI");
    }
    int thread_level = MPI_THREAD_SINGLE;
    check_mpi(MPI_Query_thread(&thread_level), "MPI_Query_thread");
    if (thread_level < MPI_THREAD_FUNNELED)
    {
        throw std::runtime_error("Poseidon CPU MPI mode requires MPI_THREAD_FUNNELED");
    }

    auto state = std::make_unique<MpiState>();
    check_mpi(MPI_Comm_dup(communicator, &state->communicator), "MPI_Comm_dup");
    check_mpi(MPI_Comm_set_errhandler(state->communicator, MPI_ERRORS_RETURN),
              "MPI_Comm_set_errhandler");
    check_mpi(MPI_Comm_rank(state->communicator, &rank_), "MPI_Comm_rank");
    check_mpi(MPI_Comm_size(state->communicator, &world_size_), "MPI_Comm_size");
    int *tag_limit = nullptr;
    int present = 0;
    check_mpi(MPI_Comm_get_attr(state->communicator, MPI_TAG_UB, &tag_limit, &present),
              "MPI_Comm_get_attr(MPI_TAG_UB)");
    if (!present || tag_limit == nullptr)
    {
        throw std::runtime_error("Poseidon CPU MPI communicator has no MPI_TAG_UB");
    }
    state->tag_upper_bound = *tag_limit;
    if (const char *trace_path = std::getenv("POSEIDON_MPI_TRACE");
        trace_path != nullptr && trace_path[0] != '\0' && std::string_view(trace_path) != "0")
    {
        const std::string base_path = std::string_view(trace_path) == "1"
                                          ? "/tmp/poseidon-mpi-transfer"
                                          : std::string(trace_path);
        state->trace = std::make_shared<std::ofstream>(
            trace_path_for_rank(base_path, rank_), std::ios::out | std::ios::trunc);
        if (!*state->trace)
        {
            throw std::runtime_error("cannot open Poseidon CPU MPI trace file");
        }
        *state->trace
            << "rank,transfer_id,role,peer_count,payload_bytes,wire_bytes,"
               "serialize_ns,post_ns,request_lifetime_ns,wait_ns,size_wait_ns,"
               "body_wait_ns,deserialize_ns\n";
        state->trace->flush();
    }
    mpi_ = std::move(state);
}
#endif

PoseidonCpuApi::~PoseidonCpuApi() = default;

std::string PoseidonCpuApi::name() const
{
    return "PoseidonCpuApi";
}

int PoseidonCpuApi::rank() const noexcept
{
    return rank_;
}

int PoseidonCpuApi::world_size() const noexcept
{
    return world_size_;
}

void PoseidonCpuApi::require_local_host_place(const fhegpu::Place &place, const char *where) const
{
    if (place.kind != fhegpu::PlaceKind::Host || place.rank != rank_ || place.index != 0)
    {
        throw std::invalid_argument(std::string(where) + " requires Host(rank=" +
                                    std::to_string(rank_) + ",index=0)");
    }
}

#if defined(POSEIDON_RUNTIME_CPU_MPI)
int PoseidonCpuApi::mpi_tag(fhegpu::TransferId id, int part) const
{
    if (mpi_ == nullptr || part < 0 || part > 1 || mpi_->tag_upper_bound < 1)
    {
        throw std::runtime_error("invalid Poseidon CPU MPI tag state");
    }
    const std::uint64_t upper_bound = static_cast<std::uint64_t>(mpi_->tag_upper_bound);
    if (id > (upper_bound - static_cast<std::uint64_t>(part)) / 2ULL)
    {
        throw std::runtime_error("Poseidon CPU TransferId exceeds MPI_TAG_UB");
    }
    return static_cast<int>(id * 2ULL + static_cast<std::uint64_t>(part));
}
#endif

PoseidonCpuApi::Value PoseidonCpuApi::encode_plaintext(const fhegpu::ValueDesc &output_desc,
                                                       const std::vector<double> &slots)
{
    require_local_host_place(output_desc.place, "Poseidon CPU Encode");
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
    require_local_host_place(op.place, "Poseidon CPU compute");
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
    {
        if (galois_keys_ == nullptr)
        {
            throw std::runtime_error("Poseidon CPU Rotate requires GaloisKeys");
        }
        output = require_ciphertext(inputs, 0);
        const auto steps = available_rotation_steps(
            context_, *galois_keys_, std::get<fhegpu::RotateAttrs>(op.attrs).steps);
        for (int step : steps)
        {
            Ciphertext next;
            evaluator_->rotate(output, next, step, *galois_keys_);
            output = std::move(next);
        }
        break;
    }
    case fhegpu::ComputeKind::Rescale:
    {
        const auto attrs = std::get<fhegpu::RescaleAttrs>(op.attrs);
        const auto &input = require_ciphertext(inputs, 0);
        if (attrs.target_level < 0 ||
            input.level() <= static_cast<std::size_t>(attrs.target_level))
        {
            throw std::invalid_argument("Poseidon CPU Rescale target level is invalid");
        }
        output = input;
        while (output.level() > static_cast<std::size_t>(attrs.target_level))
        {
            Ciphertext next;
            evaluator_->rescale(output, next);
            output = std::move(next);
        }
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
    {
        const auto attrs = std::get<fhegpu::BootAttrs>(op.attrs);
        if (attrs.implementation != fhegpu::BootImplementation::DecryptReencrypt)
        {
            throw std::runtime_error("Poseidon CPU native Boot is not implemented");
        }
        if (!boot_encryptor_ || !boot_decryptor_)
        {
            throw std::runtime_error("Poseidon CPU decrypt_reencrypt Boot has no keys");
        }
        Plaintext decrypted;
        boot_decryptor_->decrypt(require_ciphertext(inputs, 0), decrypted);
        std::vector<std::complex<double>> slots;
        encoder_->decode(decrypted, slots);
        Plaintext refreshed;
        const auto parms_id = context_.crt_context()->parms_id_map().at(
            static_cast<std::uint32_t>(attrs.target_level));
        encoder_->encode(slots, parms_id, exact_scale(attrs.target_scale_log2), refreshed);
        boot_encryptor_->encrypt(refreshed, output);
        break;
    }
    }

    return Value::from_ciphertext(std::move(output));
}

PoseidonCpuApi::CommHandle PoseidonCpuApi::communicate_async(
    const fhegpu::CommAction &action, const std::vector<Value> &local_inputs)
{
#if defined(POSEIDON_RUNTIME_CPU_MPI)
    if (mpi_ != nullptr)
    {
        if (action.inputs.size() != 1 || action.sources.size() != 1 ||
            action.outputs.size() != action.destinations.size() ||
            action.outputs.size() != action.output_types.size())
        {
            throw std::invalid_argument("Poseidon CPU communication mapping is invalid");
        }
        if (action.kind == fhegpu::CommKind::Transfer && action.outputs.size() != 1)
        {
            throw std::invalid_argument("Poseidon CPU Transfer requires one output");
        }
        if (action.kind == fhegpu::CommKind::Replicate && action.outputs.size() < 2)
        {
            throw std::invalid_argument("Poseidon CPU Replicate requires at least two outputs");
        }

        const auto &source = action.sources.front();
        if (source.kind != fhegpu::PlaceKind::Host || source.index != 0 || source.rank < 0 ||
            source.rank >= world_size_)
        {
            throw std::invalid_argument("Poseidon CPU communication source is not a valid Host");
        }
        const bool source_local = source.rank == rank_;
        if (source_local && local_inputs.size() != 1)
        {
            throw std::invalid_argument("Poseidon CPU MPI sender requires one local input");
        }
        if (!source_local && !local_inputs.empty())
        {
            throw std::invalid_argument("Poseidon CPU MPI non-source supplied a local input");
        }

        auto state = std::make_shared<CommState>();
        state->id = action.id;
        state->trace = mpi_->trace;
        state->receives.reserve(action.destinations.size());
        if (source_local)
        {
            const auto serialize_start = TraceClock::now();
            state->send_bytes = serialize_value(local_inputs.front());
            state->serialize_nanoseconds =
                elapsed_nanoseconds(serialize_start, TraceClock::now());
            state->send_size = state->send_bytes.size();
            state->sends.reserve(action.destinations.size());
        }

        const auto post_start = TraceClock::now();
        for (std::size_t slot = 0; slot < action.destinations.size(); ++slot)
        {
            const auto &destination = action.destinations[slot];
            if (destination.kind != fhegpu::PlaceKind::Host || destination.index != 0 ||
                destination.rank < 0 || destination.rank >= world_size_ ||
                destination.rank == source.rank)
            {
                throw std::invalid_argument(
                    "Poseidon CPU communication destination is not a valid remote Host");
            }
            if (source_local && local_inputs.front().kind() != action.output_types[slot])
            {
                throw std::invalid_argument("Poseidon CPU communication value kind mismatch");
            }

            if (source_local)
            {
                state->sends.emplace_back();
                auto &send = state->sends.back();
                check_mpi(MPI_Isend(&state->send_size, 1, MPI_UINT64_T, destination.rank,
                                    mpi_tag(action.id, 0), mpi_->communicator,
                                    &send.requests[0]),
                          "MPI_Isend(Poseidon value size)");
                check_mpi(MPI_Isend(state->send_bytes.data(),
                                    static_cast<int>(state->send_bytes.size()), MPI_BYTE,
                                    destination.rank, mpi_tag(action.id, 1), mpi_->communicator,
                                    &send.requests[1]),
                          "MPI_Isend(Poseidon value)");
            }
            else if (destination.rank == rank_)
            {
                state->receives.push_back(
                    CommState::Receive{slot, source.rank, action.output_types[slot], 0,
                                       MPI_REQUEST_NULL});
                auto &receive = state->receives.back();
                check_mpi(MPI_Irecv(&receive.size, 1, MPI_UINT64_T, receive.source_rank,
                                    mpi_tag(action.id, 0), mpi_->communicator,
                                    &receive.size_request),
                          "MPI_Irecv(Poseidon value size)");
            }
        }
        state->requests_posted_at = TraceClock::now();
        state->post_nanoseconds = elapsed_nanoseconds(post_start, state->requests_posted_at);
        return CommHandle{std::move(state)};
    }
#else
    static_cast<void>(action);
    static_cast<void>(local_inputs);
#endif
    throw std::runtime_error("Poseidon CPU Api does not support communication");
}

std::vector<PoseidonCpuApi::Value> PoseidonCpuApi::wait(CommHandle &handle)
{
    if (handle.state == nullptr)
    {
        throw std::runtime_error("Poseidon CPU Api has no communication handle to wait for");
    }
    if (handle.state->waited)
    {
        throw std::runtime_error("Poseidon CPU communication handle was waited twice");
    }
    handle.state->waited = true;

#if defined(POSEIDON_RUNTIME_CPU_MPI)
    if (mpi_ != nullptr)
    {
        const auto wait_start = TraceClock::now();
        for (auto &send : handle.state->sends)
        {
            const auto body_wait_start = TraceClock::now();
            check_mpi(MPI_Waitall(static_cast<int>(send.requests.size()), send.requests.data(),
                                  MPI_STATUSES_IGNORE),
                      "MPI_Waitall(Poseidon send)");
            handle.state->body_wait_nanoseconds +=
                elapsed_nanoseconds(body_wait_start, TraceClock::now());
        }

        std::vector<std::pair<std::size_t, Value>> completed;
        completed.reserve(handle.state->receives.size());
        for (auto &receive : handle.state->receives)
        {
            const auto size_wait_start = TraceClock::now();
            check_mpi(MPI_Wait(&receive.size_request, MPI_STATUS_IGNORE),
                      "MPI_Wait(Poseidon value size)");
            handle.state->size_wait_nanoseconds +=
                elapsed_nanoseconds(size_wait_start, TraceClock::now());
            if (receive.size == 0 || receive.size >
                                         static_cast<std::uint64_t>(
                                             std::numeric_limits<int>::max()))
            {
                throw std::runtime_error("invalid Poseidon CPU MPI serialized value size");
            }

            std::vector<poseidon_byte> bytes(static_cast<std::size_t>(receive.size));
            MPI_Request request = MPI_REQUEST_NULL;
            check_mpi(MPI_Irecv(bytes.data(), static_cast<int>(bytes.size()), MPI_BYTE,
                                receive.source_rank, mpi_tag(handle.state->id, 1),
                                mpi_->communicator, &request),
                      "MPI_Irecv(Poseidon value)");
            const auto body_wait_start = TraceClock::now();
            check_mpi(MPI_Wait(&request, MPI_STATUS_IGNORE), "MPI_Wait(Poseidon value)");
            handle.state->body_wait_nanoseconds +=
                elapsed_nanoseconds(body_wait_start, TraceClock::now());
            const auto deserialize_start = TraceClock::now();
            completed.emplace_back(receive.slot,
                                   deserialize_value(context_, receive.kind, bytes));
            handle.state->deserialize_nanoseconds +=
                elapsed_nanoseconds(deserialize_start, TraceClock::now());
        }
        std::sort(completed.begin(), completed.end(),
                  [](const auto &left, const auto &right) { return left.first < right.first; });

        std::vector<Value> outputs;
        outputs.reserve(completed.size());
        for (auto &entry : completed)
        {
            outputs.push_back(std::move(entry.second));
        }
        const auto wait_finish = TraceClock::now();
        handle.state->wait_nanoseconds = elapsed_nanoseconds(wait_start, wait_finish);
        handle.state->request_lifetime_nanoseconds =
            elapsed_nanoseconds(handle.state->requests_posted_at, wait_finish);
        if (handle.state->trace)
        {
            const bool sending = !handle.state->sends.empty();
            const std::size_t peer_count = sending ? handle.state->sends.size()
                                                   : handle.state->receives.size();
            std::uint64_t payload_bytes = handle.state->send_size;
            if (!sending)
            {
                for (const auto &receive : handle.state->receives)
                {
                    payload_bytes += receive.size;
                }
            }
            const std::uint64_t wire_bytes =
                sending ? payload_bytes * static_cast<std::uint64_t>(peer_count)
                        : payload_bytes;
            *handle.state->trace
                << rank_ << ',' << handle.state->id << ','
                << (sending ? "send" : "receive") << ',' << peer_count << ','
                << payload_bytes << ',' << wire_bytes << ','
                << handle.state->serialize_nanoseconds << ','
                << handle.state->post_nanoseconds << ','
                << handle.state->request_lifetime_nanoseconds << ','
                << handle.state->wait_nanoseconds << ','
                << handle.state->size_wait_nanoseconds << ','
                << handle.state->body_wait_nanoseconds << ','
                << handle.state->deserialize_nanoseconds << '\n';
            handle.state->trace->flush();
        }
        return outputs;
    }
#endif
    throw std::runtime_error("Poseidon CPU Api has no MPI communication state");
}

void PoseidonCpuApi::synchronize(Value &) {}

void PoseidonCpuApi::preflight(std::string_view plan_source_sha256,
                               bool skip_artifact_digest_checks, const fhegpu::TargetConfig &target,
                               const fhegpu::OperatorSpec &operator_spec,
                               const fhegpu::PlanRequirements &requirements)
{
    if (plan_source_sha256.size() != 71 || plan_source_sha256.substr(0, 7) != "sha256:")
    {
        throw std::invalid_argument("invalid RuntimePlan source SHA-256");
    }
    if (target.target_id != "poseidon-ckks-cpu" || target.capability_version != 1)
    {
        throw std::invalid_argument("Poseidon CPU Api target is unsupported");
    }
    if (target.world_size != world_size_ ||
        target.device_counts.size() != static_cast<std::size_t>(world_size_) ||
        !std::all_of(target.device_counts.begin(), target.device_counts.end(),
                     [](int count) { return count == 0; }))
    {
        throw std::invalid_argument(
            mpi_ == nullptr
                ? "Poseidon CPU local mode supports one process and Host compute only"
                : "Poseidon CPU MPI target does not match the communicator");
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
        operator_spec.level_lower_bound < 0 ||
        operator_spec.level_upper_bound >= static_cast<int>(modulus_bits.size()) ||
        operator_spec.level_lower_bound > operator_spec.level_upper_bound)
    {
        throw std::invalid_argument("OperatorSpec parameters do not match Poseidon CPU context");
    }

#if defined(POSEIDON_RUNTIME_CPU_MPI)
    if (mpi_ != nullptr)
    {
        const int local_skip = skip_artifact_digest_checks ? 1 : 0;
        std::vector<int> skip_values(static_cast<std::size_t>(world_size_));
        check_mpi(MPI_Allgather(&local_skip, 1, MPI_INT, skip_values.data(), 1, MPI_INT,
                                mpi_->communicator),
                  "MPI_Allgather(skip_artifact_digest_checks)");
        for (int value : skip_values)
        {
            if (value != local_skip)
            {
                throw std::runtime_error(
                    "skip_artifact_digest_checks mismatch across MPI ranks");
            }
        }
        if (!skip_artifact_digest_checks)
        {
            require_same_string(mpi_->communicator, world_size_, plan_source_sha256,
                                "RuntimePlan source SHA-256");
        }
        const std::string context_digest = context_sha256(context_);
        require_same_string(mpi_->communicator, world_size_, context_digest,
                            "Poseidon CPU context SHA-256");
    }
#else
    static_cast<void>(skip_artifact_digest_checks);
#endif

    for (const auto capability : requirements.capabilities)
    {
        const bool local = capability == fhegpu::RequiredCapability::Encode ||
                           capability == fhegpu::RequiredCapability::HostCompute;
        const bool boot = capability == fhegpu::RequiredCapability::BootDecryptReencrypt &&
                          boot_encryptor_ != nullptr && boot_decryptor_ != nullptr;
        const bool communication = mpi_ != nullptr &&
                                   (capability == fhegpu::RequiredCapability::Transfer ||
                                    capability == fhegpu::RequiredCapability::Replicate);
        if (!local && !communication && !boot)
        {
            throw std::runtime_error("Poseidon CPU Api lacks required capability: " +
                                     fhegpu::to_string(capability));
        }
    }

    for (const auto &key : requirements.keys)
    {
        if (key.place.kind != fhegpu::PlaceKind::Host || key.place.index != 0 ||
            key.place.rank < 0 || key.place.rank >= world_size_)
        {
            throw std::invalid_argument("Poseidon CPU key requires a valid Host place");
        }
        if (key.place.rank != rank_)
        {
            continue;
        }
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
            (void)available_rotation_steps(context_, *galois_keys_, *key.rotation_step);
        }
        else if (key.kind == fhegpu::KeyKind::Secret)
        {
            if (!boot_decryptor_)
            {
                throw std::runtime_error("Poseidon CPU Api lacks decrypt_reencrypt Boot keys");
            }
        }
        else
        {
            throw std::runtime_error("Poseidon CPU Api does not support secret-key operations");
        }
    }
}

[[noreturn]] void PoseidonCpuApi::abort_all(int exit_code, const std::string &reason)
{
#if defined(POSEIDON_RUNTIME_CPU_MPI)
    if (mpi_ != nullptr)
    {
        std::fprintf(stderr, "[rank %d] Poseidon CPU MPI abort: %s\n", rank_, reason.c_str());
        std::fflush(stderr);
        MPI_Abort(mpi_->communicator, exit_code);
        std::abort();
    }
#else
    static_cast<void>(exit_code);
#endif
    throw std::runtime_error(reason);
}

void PoseidonCpuApi::validate_value(const Value &value, const fhegpu::ValueDesc &expected) const
{
    require_local_host_place(expected.place, "Poseidon CPU value");
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
        const auto context_data = context_.crt_context()->get_context_data(cipher.parms_id());
        if (context_data == nullptr)
        {
            throw std::runtime_error("Poseidon ciphertext has an unknown parms_id");
        }
        if (cipher.poly_modulus_degree() != context_.parameters_literal()->degree())
        {
            throw std::runtime_error(
                "Poseidon ciphertext polynomial degree does not match context");
        }
        actual_level = static_cast<int>(context_data->level());
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
