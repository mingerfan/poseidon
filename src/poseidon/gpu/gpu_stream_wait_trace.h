#pragma once

#include <cuda_runtime_api.h>

namespace poseidon::gpu
{

void gpu_stream_wait_event(
    cudaStream_t consumer_stream, cudaEvent_t dependency_event,
    int consumer_device, const char *trace_name, const char *error_context);

} // namespace poseidon::gpu
