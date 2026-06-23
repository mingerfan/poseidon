#include "poseidon/mgpu/comm/gpu_comm.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace poseidon::mgpu;

namespace
{

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

GpuObjectCopyRequest make_valid_request()
{
    static int source = 1;
    static int destination = 0;
    GpuObjectCopyRequest request;
    request.source_id = 1;
    request.destination_id = 2;
    request.kind = MgpuValueKind::Ciphertext;
    request.buffers.push_back(GpuObjectBufferCopy{
        &source,
        &destination,
        sizeof(int),
        0,
        1,
    });
    return request;
}

void test_valid_full_object_copy_request()
{
    const GpuObjectCopyValidationResult result =
        validate_full_object_copy_request(make_valid_request());
    require(result.ok(), "valid object copy request should pass:\n" + result.format_errors());
}

void test_rejects_reserved_ids()
{
    GpuObjectCopyRequest request = make_valid_request();
    request.source_id = 0;
    request.destination_id = 0;

    const GpuObjectCopyValidationResult result =
        validate_full_object_copy_request(request);
    require(!result.ok(), "reserved ids should fail");
    require_contains(result.format_errors(), "source value id 0 is reserved");
    require_contains(result.format_errors(), "destination value id 0 is reserved");
}

void test_rejects_multi_buffer_copy()
{
    GpuObjectCopyRequest request = make_valid_request();
    request.buffers.push_back(request.buffers[0]);

    const GpuObjectCopyValidationResult result =
        validate_full_object_copy_request(request);
    require(!result.ok(), "multi-buffer copy should fail in V1");
    require_contains(result.format_errors(), "requires exactly one buffer");
}

void test_rejects_empty_buffer()
{
    GpuObjectCopyRequest request = make_valid_request();
    request.buffers.clear();

    const GpuObjectCopyValidationResult result =
        validate_full_object_copy_request(request);
    require(!result.ok(), "empty buffer list should fail");
    require_contains(result.format_errors(), "got 0");
}

void test_rejects_invalid_buffer_fields()
{
    GpuObjectCopyRequest request = make_valid_request();
    request.buffers[0].source = nullptr;
    request.buffers[0].destination = nullptr;
    request.buffers[0].bytes = 0;
    request.buffers[0].source_device = -1;
    request.buffers[0].destination_device = -2;

    const GpuObjectCopyValidationResult result =
        validate_full_object_copy_request(request);
    require(!result.ok(), "invalid buffer fields should fail");
    require_contains(result.format_errors(), "buffer must be non-empty");
    require_contains(result.format_errors(), "source pointer is null");
    require_contains(result.format_errors(), "destination pointer is null");
    require_contains(result.format_errors(), "source device must be non-negative");
    require_contains(result.format_errors(), "destination device must be non-negative");
}

}  // namespace

int main()
{
    try
    {
        test_valid_full_object_copy_request();
        test_rejects_reserved_ids();
        test_rejects_multi_buffer_copy();
        test_rejects_empty_buffer();
        test_rejects_invalid_buffer_fields();
    }
    catch (const std::exception &ex)
    {
        std::cerr << "mgpu object copy test failed: " << ex.what() << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "mgpu object copy tests passed\n";
    return EXIT_SUCCESS;
}
