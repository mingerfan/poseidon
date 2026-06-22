#include "poseidon/frontends/dacapo/dacapo_schedule_pipeline.h"
#include "poseidon/batchencoder.h"
#include "poseidon/ciphertext.h"
#include "poseidon/ckks_encoder.h"
#include "poseidon/encryptor.h"
#include "poseidon/factory/poseidon_factory.h"
#include "poseidon/keygenerator.h"
#include "poseidon/mgpu/comm/gpu_comm.h"
#include "poseidon/frontends/dacapo/dacapo_constants.h"
#include "poseidon/mgpu/compiler/static_schedule_pipeline.h"
#include "poseidon/frontends/dacapo/hevm_static_execution_plan.h"
#include "poseidon/frontends/dacapo/poseidon_gpu_hevm_executor.h"
#include "poseidon/mgpu/runtime/poseidon_gpu_execution_backend.h"
#include "poseidon/mgpu/runtime/sequential_schedule_executor.h"
#include "poseidon/parameters_literal.h"
#include "poseidon/plaintext.h"
#include "poseidon/poseidon_context.h"
#include "poseidon/tests/frontends/dacapo/hevm_test_utils.h"

#include <cuda_runtime_api.h>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace poseidon;
using namespace poseidon::mgpu;

namespace
{

constexpr int kSkip = 77;

MgpuValueRef value(ValueId id)
{
    return MgpuValueRef{ id };
}

MgpuOp op(
    MgpuOpKind kind, int device_id, std::vector<MgpuValueRef> inputs,
    std::vector<MgpuValueRef> outputs)
{
    return MgpuOp{ kind, device_id, std::move(inputs), std::move(outputs), {} };
}

MgpuOp op_with_attr(
    MgpuOpKind kind, int device_id, std::vector<MgpuValueRef> inputs,
    std::vector<MgpuValueRef> outputs, const std::string &attr_name, std::int64_t attr_value)
{
    MgpuOp result{ kind, device_id, std::move(inputs), std::move(outputs), {} };
    result.integer_attributes.emplace(attr_name, attr_value);
    return result;
}

void require(bool condition, const std::string &message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}

void require_contains(const std::string &text, const std::string &needle)
{
    if (text.find(needle) == std::string::npos)
    {
        throw std::runtime_error("expected text to contain: " + needle + "\ntext:\n" + text);
    }
}

void require_plaintexts_equal(const Plaintext &expected, const Plaintext &actual)
{
    require(actual.coeff_count() == expected.coeff_count(), "plaintext coeff_count mismatch");
    require(actual.parms_id() == expected.parms_id(), "plaintext parms_id mismatch");
    require(actual.scale() == expected.scale(), "plaintext scale mismatch");
    for (std::size_t i = 0; i < expected.coeff_count(); ++i)
    {
        if (actual.data()[i] != expected.data()[i])
        {
            std::ostringstream stream;
            stream << "plaintext coefficient mismatch at " << i;
            throw std::runtime_error(stream.str());
        }
    }
}

void require_ciphertexts_equal(const Ciphertext &expected, const Ciphertext &actual)
{
    require(actual.parms_id() == expected.parms_id(), "ciphertext parms_id mismatch");
    require(actual.size() == expected.size(), "ciphertext size mismatch");
    require(
        actual.poly_modulus_degree() == expected.poly_modulus_degree(),
        "ciphertext degree mismatch");
    require(
        actual.coeff_modulus_size() == expected.coeff_modulus_size(),
        "ciphertext coeff modulus size mismatch");
    require(actual.is_ntt_form() == expected.is_ntt_form(), "ciphertext NTT form mismatch");
    require(actual.scale() == expected.scale(), "ciphertext scale mismatch");
    require(
        actual.correction_factor() == expected.correction_factor(),
        "ciphertext correction factor mismatch");

    const auto word_count =
        expected.size() * expected.poly_modulus_degree() * expected.coeff_modulus_size();
    for (std::size_t i = 0; i < word_count; ++i)
    {
        if (actual.data()[i] != expected.data()[i])
        {
            std::ostringstream stream;
            stream << "ciphertext word mismatch at " << i;
            throw std::runtime_error(stream.str());
        }
    }
}

void insert_same_device_cipher_copy_before_multiply_plain(HevmStaticExecutionPlan &plan)
{
    require(!plan.io_plan.cipher_inputs.empty(), "expected HEVM cipher input");
    const ValueId source_value_id = plan.io_plan.cipher_inputs[0].value_id;
    constexpr ValueId copied_value_id = 100000;

    MgpuOp copy_op{
        MgpuOpKind::CopyCipher,
        plan.io_plan.cipher_inputs[0].device_id,
        { value(source_value_id) },
        { value(copied_value_id) },
        "same_device_hevm_copy",
    };

    auto insert_pos = plan.schedule.ops.begin();
    for (; insert_pos != plan.schedule.ops.end(); ++insert_pos)
    {
        if (insert_pos->kind == MgpuOpKind::UploadPlain)
        {
            break;
        }
    }
    plan.schedule.ops.insert(insert_pos, std::move(copy_op));

    for (MgpuOp &op : plan.schedule.ops)
    {
        if (op.kind == MgpuOpKind::MultiplyPlain && !op.inputs.empty() &&
            op.inputs[0].id == source_value_id)
        {
            op.inputs[0].id = copied_value_id;
            return;
        }
    }

    throw std::runtime_error("expected HEVM multiply_plain op");
}

void append_i64(std::string &output, std::int64_t value)
{
    const auto bits = static_cast<std::uint64_t>(value);
    for (int i = 0; i < 8; ++i)
    {
        output.push_back(static_cast<char>((bits >> (8 * i)) & 0xFF));
    }
}

void append_double(std::string &output, double value)
{
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    for (int i = 0; i < 8; ++i)
    {
        output.push_back(static_cast<char>((bits >> (8 * i)) & 0xFF));
    }
}

std::string make_hevm_constant_file()
{
    std::string output;
    append_i64(output, 1);
    append_i64(output, 4);
    append_double(output, 0.5);
    append_double(output, -1.0);
    append_double(output, 2.0);
    append_double(output, 3.5);
    return output;
}

class RmmPoolScope
{
public:
    explicit RmmPoolScope(int device_id)
        : device_id_(device_id), pool_(&upstream_, 1 << 20, std::nullopt)
    {
        const cudaError_t status = cudaSetDevice(device_id_);
        if (status != cudaSuccess)
        {
            throw std::runtime_error(
                std::string("cudaSetDevice failed: ") + cudaGetErrorString(status));
        }

        previous_ = rmm::mr::get_current_device_resource();
        rmm::mr::set_current_device_resource(&pool_);
    }

    RmmPoolScope(const RmmPoolScope &) = delete;
    RmmPoolScope &operator=(const RmmPoolScope &) = delete;

    ~RmmPoolScope()
    {
        try
        {
            (void)cudaSetDevice(device_id_);
            rmm::mr::set_current_device_resource(previous_);
        }
        catch (...)
        {}
    }

private:
    int device_id_ = 0;
    rmm::mr::cuda_memory_resource upstream_;
    rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> pool_;
    rmm::mr::device_memory_resource *previous_ = nullptr;
};

int check_cuda_runtime()
{
    int device_count = 0;
    const cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status != cudaSuccess || device_count == 0)
    {
        std::cout << "Skipping GPU runtime smoke test: no CUDA device is available\n";
        return kSkip;
    }
    return EXIT_SUCCESS;
}

ParametersLiteral make_gpu_test_parameters()
{
    return ParametersLiteral(
        BFV,
        /*log_n=*/12,
        /*log_slots=*/12,
        /*log_scale=*/0,
        /*hamming_weight=*/0,
        /*q0_level=*/0,
        Modulus(65537),
        std::vector<Modulus>{
            Modulus(786433),
            Modulus(1032193),
        },
        std::vector<Modulus>{
            Modulus(1179649),
        },
        sec_level_type::none);
}

ParametersLiteral make_gpu_ckks_test_parameters()
{
    ParametersLiteral parms(
        CKKS,
        /*log_n=*/12,
        /*log_slots=*/11,
        /*log_scale=*/20,
        /*hamming_weight=*/0,
        /*q0_level=*/0,
        Modulus(0),
        std::vector<Modulus>{},
        std::vector<Modulus>{},
        sec_level_type::none);
    parms.set_log_modulus(std::vector<std::uint32_t>(3, 30), {});
    return parms;
}

ParametersLiteral make_gpu_ckks_keyswitch_test_parameters()
{
    ParametersLiteral parms(
        CKKS,
        /*log_n=*/12,
        /*log_slots=*/11,
        /*log_scale=*/20,
        /*hamming_weight=*/0,
        /*q0_level=*/0,
        Modulus(0),
        std::vector<Modulus>{},
        std::vector<Modulus>{},
        sec_level_type::none);
    parms.set_log_modulus(std::vector<std::uint32_t>(3, 30), std::vector<std::uint32_t>(2, 30));
    return parms;
}

void run_gpu_runtime_smoke()
{
    constexpr int device_id = 0;
    RmmPoolScope rmm_scope(device_id);

    const ParametersLiteral parms = make_gpu_test_parameters();
    PoseidonContext context(parms);
    auto cpu_evaluator = PoseidonFactory::get_instance()->create_bfv_evaluator(context);

    KeyGenerator keygen(context);
    PublicKey public_key;
    keygen.create_public_key(public_key);

    BatchEncoder encoder(context);
    Plaintext plain0;
    Plaintext plain1;
    encoder.encode(std::vector<std::uint64_t>{ 1, 2, 3, 4, 5, 6, 7, 8 }, plain0);
    encoder.encode(std::vector<std::uint64_t>{ 8, 7, 6, 5, 4, 3, 2, 1 }, plain1);

    Encryptor encryptor(context, public_key, keygen.secret_key());
    Ciphertext cipher0;
    Ciphertext cipher1;
    encryptor.encrypt(plain0, cipher0);
    encryptor.encrypt(plain1, cipher1);

    Ciphertext expected_sum;
    cpu_evaluator->add(cipher0, cipher1, expected_sum);

    MgpuSchedule schedule;
    schedule.ops.push_back(op(MgpuOpKind::UploadCipher, device_id, {}, { value(1) }));
    schedule.ops.push_back(op(MgpuOpKind::UploadCipher, device_id, {}, { value(2) }));
    schedule.ops.push_back(op(MgpuOpKind::UploadPlain, device_id, {}, { value(3) }));
    schedule.ops.push_back(op(MgpuOpKind::Add, device_id, { value(1), value(2) }, { value(4) }));
    schedule.ops.push_back(op(MgpuOpKind::Download, device_id, { value(1) }, {}));
    schedule.ops.push_back(op(MgpuOpKind::Download, device_id, { value(3) }, {}));
    schedule.ops.push_back(op(MgpuOpKind::Download, device_id, { value(4) }, {}));

    PoseidonGpuExecutionBackend backend(context);
    backend.bind_cipher_upload(1, std::make_shared<Ciphertext>(cipher0));
    backend.bind_cipher_upload(2, std::make_shared<Ciphertext>(cipher1));
    backend.bind_plain_upload(3, std::make_shared<Plaintext>(plain0));

    SequentialScheduleExecutor executor(SequentialScheduleExecutorOptions{ 1 });
    const ScheduleExecutionResult result = executor.run(schedule, backend);
    require(result.ok(), "GPU runtime schedule failed:\n" + result.format_errors());
    require(backend.has_cipher_download(1), "missing ciphertext download");
    require(backend.has_plain_download(3), "missing plaintext download");
    require(backend.has_cipher_download(4), "missing add output download");
    require_ciphertexts_equal(cipher0, *backend.cipher_download(1));
    require_plaintexts_equal(plain0, *backend.plain_download(3));
    require_ciphertexts_equal(expected_sum, *backend.cipher_download(4));
}

void run_gpu_runtime_multiply_plain_smoke()
{
    constexpr int device_id = 0;
    RmmPoolScope rmm_scope(device_id);

    const ParametersLiteral parms = make_gpu_ckks_test_parameters();
    PoseidonContext context(parms);
    auto cpu_evaluator = PoseidonFactory::get_instance()->create_ckks_evaluator(context);

    KeyGenerator keygen(context);
    PublicKey public_key;
    keygen.create_public_key(public_key);

    CKKSEncoder encoder(context);
    Plaintext plain0;
    Plaintext plain1;
    encoder.encode(std::vector<double>{ 1.0, 2.0, 3.0, 4.0 }, parms.scale(), plain0);
    encoder.encode(std::vector<double>{ 0.5, -1.0, 2.0, 3.5 }, parms.scale(), plain1);

    Encryptor encryptor(context, public_key, keygen.secret_key());
    Ciphertext cipher0;
    encryptor.encrypt(plain0, cipher0);

    Ciphertext expected_product;
    cpu_evaluator->multiply_plain(cipher0, plain1, expected_product);

    MgpuSchedule schedule;
    schedule.ops.push_back(op(MgpuOpKind::UploadCipher, device_id, {}, { value(10) }));
    schedule.ops.push_back(op(MgpuOpKind::UploadPlain, device_id, {}, { value(11) }));
    schedule.ops.push_back(
        op(MgpuOpKind::MultiplyPlain, device_id, { value(10), value(11) }, { value(12) }));
    schedule.ops.push_back(op(MgpuOpKind::Download, device_id, { value(12) }, {}));

    PoseidonGpuExecutionBackend backend(context);
    backend.bind_cipher_upload(10, std::make_shared<Ciphertext>(cipher0));
    backend.bind_plain_upload(11, std::make_shared<Plaintext>(plain1));

    SequentialScheduleExecutor executor(SequentialScheduleExecutorOptions{ 1 });
    const ScheduleExecutionResult result = executor.run(schedule, backend);
    require(result.ok(), "GPU runtime multiply_plain schedule failed:\n" + result.format_errors());
    require(backend.has_cipher_download(12), "missing multiply_plain output download");
    require_ciphertexts_equal(expected_product, *backend.cipher_download(12));
}

void run_gpu_runtime_hevm_constants_smoke()
{
    constexpr int device_id = 0;
    RmmPoolScope rmm_scope(device_id);

    const ParametersLiteral parms = make_gpu_ckks_test_parameters();
    PoseidonContext context(parms);
    auto cpu_evaluator = PoseidonFactory::get_instance()->create_ckks_evaluator(context);

    const std::string hevm = test::make_hevm_binary(
        1, 1, 2, 1, { 1 },
        {
            test::HevmOpRecord{ 0, 0, 0, test::make_hevm_encode_attr(2, 20) },
            test::HevmOpRecord{ 9, 1, 0, 0 },
        },
        test::HevmConfigMetadata{
            { 20 },
            { 2 },
            { 40 },
            { 2 },
            2,
        });

    StaticSchedulePipelineOptions options;
    options.device_count = 1;
    const StaticSchedulePipelineResult pipeline = prepare_dacapo_static_schedule(
        hevm, DacapoAdapterOptions{ DacapoInputFormat::HevmBinary }, options);
    require(pipeline.ok(), "HEVM GPU pipeline failed:\n" + pipeline.format_diagnostics());

    const DacapoConstantParseResult constants =
        parse_dacapo_constant_file(make_hevm_constant_file());
    require(constants.ok(), "HEVM constants parse failed:\n" + constants.format_diagnostics());

    DacapoHevmArtifactResult artifacts;
    artifacts.schedule = pipeline.schedule;
    artifacts.constants = constants.table;
    artifacts.debug_dump = pipeline.debug_dump;

    const HevmStaticExecutionPlanResult execution_plan =
        prepare_hevm_static_execution_plan(context, artifacts);
    require(
        execution_plan.ok(),
        "HEVM static execution plan failed:\n" + execution_plan.format_diagnostics());
    require(
        execution_plan.plan.encoded_plaintexts.size() == 1,
        "expected one HEVM plaintext upload");

    CKKSEncoder encoder(context);
    Plaintext input_plain;
    encoder.encode(std::vector<double>{ 1.0, 2.0, 3.0, 4.0 }, std::ldexp(1.0, 20), input_plain);

    KeyGenerator keygen(context);
    PublicKey public_key;
    keygen.create_public_key(public_key);
    Encryptor encryptor(context, public_key, keygen.secret_key());
    Ciphertext input_cipher;
    encryptor.encrypt(input_plain, input_cipher);

    Ciphertext expected_product;
    cpu_evaluator->multiply_plain(
        input_cipher, *execution_plan.plan.encoded_plaintexts[0].plaintext,
        expected_product);

    HevmStaticExecutionPlan plan_with_same_device_copy = execution_plan.plan;
    insert_same_device_cipher_copy_before_multiply_plain(plan_with_same_device_copy);

    SameDeviceGpuComm comm;
    const PoseidonGpuHevmExecutionResult gpu_result =
        execute_hevm_static_plan_with_poseidon_gpu(
            context, plan_with_same_device_copy,
            std::vector<std::shared_ptr<const Ciphertext>>{
                std::make_shared<Ciphertext>(input_cipher),
            },
            comm,
            PoseidonGpuHevmExecutionOptions{ 1 });

    require(
        gpu_result.ok(),
        "GPU runtime HEVM constants schedule failed:\n" + gpu_result.format_errors());
    require(gpu_result.results.size() == 1, "expected one HEVM result slot");
    require(gpu_result.results[0] != nullptr, "missing HEVM result ciphertext");
    require_ciphertexts_equal(expected_product, *gpu_result.results[0]);
}

void run_gpu_runtime_hevm_preflight_failures()
{
    const ParametersLiteral parms = make_gpu_ckks_test_parameters();
    PoseidonContext context(parms);

    const std::string hevm = test::make_hevm_binary(
        1, 1, 2, 1, { 1 },
        {
            test::HevmOpRecord{ 0, 0, 0, test::make_hevm_encode_attr(2, 20) },
            test::HevmOpRecord{ 9, 1, 0, 0 },
        },
        test::HevmConfigMetadata{
            { 20 },
            { 2 },
            { 40 },
            { 2 },
            2,
        });

    StaticSchedulePipelineOptions options;
    options.device_count = 1;
    const StaticSchedulePipelineResult pipeline = prepare_dacapo_static_schedule(
        hevm, DacapoAdapterOptions{ DacapoInputFormat::HevmBinary }, options);
    require(pipeline.ok(), "HEVM GPU pipeline failed:\n" + pipeline.format_diagnostics());

    const DacapoConstantParseResult constants =
        parse_dacapo_constant_file(make_hevm_constant_file());
    require(constants.ok(), "HEVM constants parse failed:\n" + constants.format_diagnostics());

    DacapoHevmArtifactResult artifacts;
    artifacts.schedule = pipeline.schedule;
    artifacts.constants = constants.table;

    const HevmStaticExecutionPlanResult execution_plan =
        prepare_hevm_static_execution_plan(context, artifacts);
    require(
        execution_plan.ok(),
        "HEVM static execution plan failed:\n" + execution_plan.format_diagnostics());

    HevmStaticExecutionPlan plan_with_same_device_copy = execution_plan.plan;
    insert_same_device_cipher_copy_before_multiply_plain(plan_with_same_device_copy);
    const PoseidonGpuHevmExecutionResult copy_without_comm =
        execute_hevm_static_plan_with_poseidon_gpu(
            context, plan_with_same_device_copy,
            std::vector<std::shared_ptr<const Ciphertext>>{},
            PoseidonGpuHevmExecutionOptions{ 1 });
    require(!copy_without_comm.ok(), "copy without comm should fail preflight");
    require_contains(copy_without_comm.format_errors(), "preflight");
    require_contains(copy_without_comm.format_errors(), "communication layer");

    HevmStaticExecutionPlan plan_with_rotate = execution_plan.plan;
    plan_with_rotate.schedule.ops.insert(
        plan_with_rotate.schedule.ops.end() - 1,
        op_with_attr(
            MgpuOpKind::Rotate, 0, { value(1) }, { value(100) },
            "rotate_step", 1));
    plan_with_rotate.schedule.ops.back().inputs[0] = value(100);
    plan_with_rotate.io_plan.results[0].value_id = 100;

    const PoseidonGpuHevmExecutionResult rotate_without_keys =
        execute_hevm_static_plan_with_poseidon_gpu(
            context, plan_with_rotate, std::vector<std::shared_ptr<const Ciphertext>>{},
            PoseidonGpuHevmExecutionOptions{ 1 });
    require(!rotate_without_keys.ok(), "rotate without keys should fail preflight");
    require_contains(rotate_without_keys.format_errors(), "preflight");
    require_contains(rotate_without_keys.format_errors(), "GaloisKeys");
}

void run_gpu_runtime_add_plain_rescale_smoke()
{
    constexpr int device_id = 0;
    RmmPoolScope rmm_scope(device_id);

    const ParametersLiteral parms = make_gpu_ckks_test_parameters();
    PoseidonContext context(parms);
    auto cpu_evaluator = PoseidonFactory::get_instance()->create_ckks_evaluator(context);

    KeyGenerator keygen(context);
    PublicKey public_key;
    keygen.create_public_key(public_key);

    CKKSEncoder encoder(context);
    Plaintext input_plain;
    Plaintext bias_plain;
    encoder.encode(std::vector<double>{ 1.25, -2.0, 3.5, 4.0 }, parms.scale(), input_plain);
    encoder.encode(std::vector<double>{ 0.25, 0.5, -1.5, 2.0 }, parms.scale(), bias_plain);

    Encryptor encryptor(context, public_key, keygen.secret_key());
    Ciphertext input_cipher;
    encryptor.encrypt(input_plain, input_cipher);

    Ciphertext expected_add_plain;
    Ciphertext expected_negate;
    Ciphertext expected_rescale;
    cpu_evaluator->add_plain(input_cipher, bias_plain, expected_add_plain);
    cpu_evaluator->sub(input_cipher, input_cipher, expected_negate);
    cpu_evaluator->sub(expected_negate, input_cipher, expected_negate);
    cpu_evaluator->rescale(expected_add_plain, expected_rescale);

    MgpuSchedule schedule;
    schedule.ops.push_back(op(MgpuOpKind::UploadCipher, device_id, {}, { value(20) }));
    schedule.ops.push_back(op(MgpuOpKind::UploadPlain, device_id, {}, { value(21) }));
    schedule.ops.push_back(
        op(MgpuOpKind::AddPlain, device_id, { value(20), value(21) }, { value(22) }));
    schedule.ops.push_back(op(MgpuOpKind::Rescale, device_id, { value(22) }, { value(23) }));
    schedule.ops.push_back(op(MgpuOpKind::Negate, device_id, { value(20) }, { value(24) }));
    schedule.ops.push_back(op(MgpuOpKind::Download, device_id, { value(22) }, {}));
    schedule.ops.push_back(op(MgpuOpKind::Download, device_id, { value(23) }, {}));
    schedule.ops.push_back(op(MgpuOpKind::Download, device_id, { value(24) }, {}));

    PoseidonGpuExecutionBackend backend(context);
    backend.bind_cipher_upload(20, std::make_shared<Ciphertext>(input_cipher));
    backend.bind_plain_upload(21, std::make_shared<Plaintext>(bias_plain));

    SequentialScheduleExecutor executor(SequentialScheduleExecutorOptions{ 1 });
    const ScheduleExecutionResult result = executor.run(schedule, backend);
    require(
        result.ok(),
        "GPU runtime add_plain/rescale schedule failed:\n" + result.format_errors());
    require(backend.has_cipher_download(22), "missing add_plain output download");
    require(backend.has_cipher_download(23), "missing rescale output download");
    require(backend.has_cipher_download(24), "missing negate output download");
    require_ciphertexts_equal(expected_add_plain, *backend.cipher_download(22));
    require_ciphertexts_equal(expected_rescale, *backend.cipher_download(23));
    require_ciphertexts_equal(expected_negate, *backend.cipher_download(24));
}

void run_gpu_runtime_rotate_smoke()
{
    constexpr int device_id = 0;
    constexpr int rotate_step = 1;
    RmmPoolScope rmm_scope(device_id);

    const ParametersLiteral parms = make_gpu_ckks_keyswitch_test_parameters();
    PoseidonContext context(parms);
    auto cpu_evaluator = PoseidonFactory::get_instance()->create_ckks_evaluator(context);

    KeyGenerator keygen(context);
    PublicKey public_key;
    keygen.create_public_key(public_key);
    GaloisKeys galois_keys;
    keygen.create_galois_keys(std::vector<int>{ rotate_step }, galois_keys);

    CKKSEncoder encoder(context);
    Plaintext plain;
    encoder.encode(std::vector<double>{ 1.0, 2.0, 3.0, 4.0 }, parms.scale(), plain);

    Encryptor encryptor(context, public_key, keygen.secret_key());
    Ciphertext cipher;
    encryptor.encrypt(plain, cipher);

    Ciphertext expected_rotate;
    cpu_evaluator->rotate(cipher, expected_rotate, rotate_step, galois_keys);

    MgpuSchedule schedule;
    schedule.ops.push_back(op(MgpuOpKind::UploadCipher, device_id, {}, { value(30) }));
    schedule.ops.push_back(op_with_attr(
        MgpuOpKind::Rotate, device_id, { value(30) }, { value(31) },
        "rotate_step", rotate_step));
    schedule.ops.push_back(op(MgpuOpKind::Download, device_id, { value(31) }, {}));

    PoseidonGpuExecutionBackend backend(context);
    backend.bind_cipher_upload(30, std::make_shared<Ciphertext>(cipher));
    backend.upload_keys_for_device(device_id, nullptr, &galois_keys);

    SequentialScheduleExecutor executor(SequentialScheduleExecutorOptions{ 1 });
    const ScheduleExecutionResult result = executor.run(schedule, backend);
    require(result.ok(), "GPU runtime rotate schedule failed:\n" + result.format_errors());
    require(backend.has_cipher_download(31), "missing rotate output download");
    require_ciphertexts_equal(expected_rotate, *backend.cipher_download(31));
}

void run_gpu_runtime_multiply_relinearize_rescale_smoke()
{
    constexpr int device_id = 0;
    RmmPoolScope rmm_scope(device_id);

    const ParametersLiteral parms = make_gpu_ckks_keyswitch_test_parameters();
    PoseidonContext context(parms);
    auto cpu_evaluator = PoseidonFactory::get_instance()->create_ckks_evaluator(context);

    KeyGenerator keygen(context);
    PublicKey public_key;
    keygen.create_public_key(public_key);
    RelinKeys relin_keys;
    keygen.create_relin_keys(relin_keys);

    CKKSEncoder encoder(context);
    Plaintext plain0;
    Plaintext plain1;
    encoder.encode(std::vector<double>{ 1.0, -2.0, 3.0, 4.5 }, parms.scale(), plain0);
    encoder.encode(std::vector<double>{ 0.5, 1.5, -1.0, 2.0 }, parms.scale(), plain1);

    Encryptor encryptor(context, public_key, keygen.secret_key());
    Ciphertext cipher0;
    Ciphertext cipher1;
    encryptor.encrypt(plain0, cipher0);
    encryptor.encrypt(plain1, cipher1);

    Ciphertext expected_multiply;
    Ciphertext expected_relinearize;
    Ciphertext expected_rescale;
    cpu_evaluator->multiply(cipher0, cipher1, expected_multiply);
    cpu_evaluator->relinearize(expected_multiply, expected_relinearize, relin_keys);
    cpu_evaluator->rescale(expected_relinearize, expected_rescale);

    MgpuSchedule schedule;
    schedule.ops.push_back(op(MgpuOpKind::UploadCipher, device_id, {}, { value(40) }));
    schedule.ops.push_back(op(MgpuOpKind::UploadCipher, device_id, {}, { value(41) }));
    schedule.ops.push_back(
        op(MgpuOpKind::Multiply, device_id, { value(40), value(41) }, { value(42) }));
    schedule.ops.push_back(op(MgpuOpKind::Relinearize, device_id, { value(42) }, { value(43) }));
    schedule.ops.push_back(op(MgpuOpKind::Rescale, device_id, { value(43) }, { value(44) }));
    schedule.ops.push_back(op(MgpuOpKind::Download, device_id, { value(42) }, {}));
    schedule.ops.push_back(op(MgpuOpKind::Download, device_id, { value(43) }, {}));
    schedule.ops.push_back(op(MgpuOpKind::Download, device_id, { value(44) }, {}));

    PoseidonGpuExecutionBackend backend(context);
    backend.bind_cipher_upload(40, std::make_shared<Ciphertext>(cipher0));
    backend.bind_cipher_upload(41, std::make_shared<Ciphertext>(cipher1));
    backend.upload_keys_for_device(device_id, &relin_keys, nullptr);

    SequentialScheduleExecutor executor(SequentialScheduleExecutorOptions{ 1 });
    const ScheduleExecutionResult result = executor.run(schedule, backend);
    require(
        result.ok(),
        "GPU runtime multiply/relinearize/rescale schedule failed:\n" +
            result.format_errors());
    require(backend.has_cipher_download(42), "missing multiply output download");
    require(backend.has_cipher_download(43), "missing relinearize output download");
    require(backend.has_cipher_download(44), "missing relin/rescale output download");
    require_ciphertexts_equal(expected_multiply, *backend.cipher_download(42));
    require_ciphertexts_equal(expected_relinearize, *backend.cipher_download(43));
    require_ciphertexts_equal(expected_rescale, *backend.cipher_download(44));
}

}  // namespace

int main()
{
    const int cuda_status = check_cuda_runtime();
    if (cuda_status != EXIT_SUCCESS)
    {
        return cuda_status;
    }

    try
    {
        run_gpu_runtime_smoke();
        run_gpu_runtime_multiply_plain_smoke();
        run_gpu_runtime_hevm_constants_smoke();
        run_gpu_runtime_hevm_preflight_failures();
        run_gpu_runtime_add_plain_rescale_smoke();
        run_gpu_runtime_rotate_smoke();
        run_gpu_runtime_multiply_relinearize_rescale_smoke();
    }
    catch (const std::exception &ex)
    {
        std::cerr << "mgpu GPU runtime smoke test failed: " << ex.what() << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "mgpu GPU runtime smoke tests passed\n";
    return EXIT_SUCCESS;
}
