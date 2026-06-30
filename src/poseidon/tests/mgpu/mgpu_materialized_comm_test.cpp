#include "poseidon/mgpu/comm/gpu_comm.h"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

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

class VectorCopyMaterializer final : public GpuObjectCopyMaterializer
{
public:
    MaterializedGpuObjectCopy materialize_copy(const GpuCommCopyRequest &request) override
    {
        requests.push_back(request);

        auto source = std::static_pointer_cast<std::vector<int>>(request.source_object);
        auto destination = std::make_shared<std::vector<int>>(source->size(), 0);

        MaterializedGpuObjectCopy result;
        result.destination_object = destination;
        result.object_copy.source_id = request.source_id;
        result.object_copy.destination_id = request.destination_id;
        result.object_copy.kind = request.kind;
        result.object_copy.buffers.push_back(GpuObjectBufferCopy{
            source->data(),
            destination->data(),
            source->size() * sizeof(int),
            request.source_device,
            request.destination_device,
        });
        return result;
    }

    std::vector<GpuCommCopyRequest> requests;
};

class InvalidMaterializer final : public GpuObjectCopyMaterializer
{
public:
    MaterializedGpuObjectCopy materialize_copy(const GpuCommCopyRequest &request) override
    {
        MaterializedGpuObjectCopy result;
        result.destination_object = std::make_shared<int>(0);
        result.object_copy.source_id = request.source_id;
        result.object_copy.destination_id = request.destination_id;
        result.object_copy.kind = request.kind;
        return result;
    }
};

class MemcpyBackend final : public GpuObjectCopyBackend
{
public:
    void copy_object(const GpuObjectCopyRequest &request) override
    {
        requests.push_back(request);
        const GpuObjectBufferCopy &buffer = request.buffers[0];
        std::memcpy(buffer.destination, buffer.source, buffer.bytes);
    }

    std::vector<GpuObjectCopyRequest> requests;
};

void test_materialized_comm_copies_object_buffer()
{
    VectorCopyMaterializer materializer;
    MemcpyBackend backend;
    MaterializedGpuComm comm(materializer, backend);

    auto source = std::make_shared<std::vector<int>>(std::initializer_list<int>{ 1, 2, 3 });
    const std::shared_ptr<void> copied = comm.copy(GpuCommCopyRequest{
        10,
        11,
        MgpuValueKind::Ciphertext,
        0,
        1,
        source,
    });

    require(materializer.requests.size() == 1, "materializer request count mismatch");
    require(materializer.requests[0].source_id == 10, "materializer source id mismatch");
    require(materializer.requests[0].destination_id == 11, "materializer destination id mismatch");
    require(materializer.requests[0].source_object == source, "materializer source object mismatch");
    require(backend.requests.size() == 1, "backend request count mismatch");
    require(backend.requests[0].source_id == 10, "backend source id mismatch");
    require(backend.requests[0].destination_id == 11, "backend destination id mismatch");
    require(backend.requests[0].buffers.size() == 1, "backend should receive one full buffer");

    const auto copied_vector = std::static_pointer_cast<std::vector<int>>(copied);
    require(*copied_vector == *source, "copied vector mismatch");
    require(copied_vector != source, "copy should return a destination object");
}

void test_materialized_comm_repeated_copies_object_buffers()
{
    VectorCopyMaterializer materializer;
    MemcpyBackend backend;
    MaterializedGpuComm comm(materializer, backend);

    auto source0 = std::make_shared<std::vector<int>>(
        std::initializer_list<int>{ 1, 2, 3 });
    auto source1 = std::make_shared<std::vector<int>>(
        std::initializer_list<int>{ 4, 5, 6, 7 });

    std::vector<std::shared_ptr<void>> copied;
    copied.push_back(comm.copy(GpuCommCopyRequest{
        10,
        20,
        MgpuValueKind::Ciphertext,
        0,
        1,
        source0,
    }));
    copied.push_back(comm.copy(GpuCommCopyRequest{
        11,
        21,
        MgpuValueKind::Ciphertext,
        0,
        1,
        source1,
    }));

    require(copied.size() == 2, "repeated copy result count mismatch");
    require(materializer.requests.size() == 2, "repeated materializer request count mismatch");
    require(backend.requests.size() == 2, "repeated backend request count mismatch");

    require(
        *std::static_pointer_cast<std::vector<int>>(copied[0]) == *source0,
        "first copied vector mismatch");
    require(
        *std::static_pointer_cast<std::vector<int>>(copied[1]) == *source1,
        "second copied vector mismatch");
}

void test_materialized_comm_rejects_invalid_object_copy()
{
    InvalidMaterializer materializer;
    MemcpyBackend backend;
    MaterializedGpuComm comm(materializer, backend);

    bool failed = false;
    try
    {
        (void)comm.copy(GpuCommCopyRequest{
            1,
            2,
            MgpuValueKind::Ciphertext,
            0,
            1,
            std::make_shared<int>(7),
        });
    }
    catch (const std::invalid_argument &ex)
    {
        failed = true;
        require_contains(ex.what(), "requires exactly one buffer");
    }

    require(failed, "invalid materialized copy should fail");
    require(backend.requests.empty(), "backend should not run for invalid copy request");
}

void test_materialized_comm_rejects_null_source_before_materializer()
{
    VectorCopyMaterializer materializer;
    MemcpyBackend backend;
    MaterializedGpuComm comm(materializer, backend);

    bool failed = false;
    try
    {
        (void)comm.copy(GpuCommCopyRequest{
            3,
            4,
            MgpuValueKind::Ciphertext,
            0,
            1,
            nullptr,
        });
    }
    catch (const std::invalid_argument &ex)
    {
        failed = true;
        require_contains(ex.what(), "source object is null");
    }

    require(failed, "null source object should fail");
    require(
        materializer.requests.empty(),
        "null source should fail before materialization");
    require(backend.requests.empty(), "backend should not run for null source");
}

}  // namespace

int main()
{
    try
    {
        test_materialized_comm_copies_object_buffer();
        test_materialized_comm_repeated_copies_object_buffers();
        test_materialized_comm_rejects_invalid_object_copy();
        test_materialized_comm_rejects_null_source_before_materializer();
    }
    catch (const std::exception &ex)
    {
        std::cerr << "mgpu materialized comm test failed: " << ex.what() << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "mgpu materialized comm tests passed\n";
    return EXIT_SUCCESS;
}
