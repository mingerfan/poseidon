// Optional operator-boundary instrumentation. Events record one stream's span
// (including host starvation), NOT summed kernel busy time. No intermediate
// synchronizations: results are collected only at the block's existing end.
#pragma once
#include "poseidon/gpu/gpu_memory.h"
#include <cuda_runtime_api.h>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
namespace cerium_source {
inline bool operator_timing_enabled() {
    const char *v=std::getenv("LLAMA_OPERATOR_TIMING");return v&&std::string(v)=="1";
}
class OperatorTiming {
    struct Boundary {std::string label;cudaEvent_t event;std::chrono::steady_clock::time_point wall;std::uint64_t boots;};
    bool enabled_=operator_timing_enabled();std::vector<Boundary> boundaries_;
public:
    ~OperatorTiming(){for(auto &b:boundaries_)cudaEventDestroy(b.event);}
    void mark(const std::string &label,std::uint64_t boots) {
        if(!enabled_)return;
        cudaEvent_t event;poseidon::gpu::gpu_check_cuda(cudaEventCreate(&event),"operator event create");
        poseidon::gpu::gpu_check_cuda(cudaEventRecord(event,poseidon::gpu::gpu_execution_stream()),"operator event record");
        boundaries_.push_back({label,event,std::chrono::steady_clock::now(),boots});
    }
    void finish(std::size_t layer) {
        if(!enabled_||boundaries_.size()<2)return;
        poseidon::gpu::gpu_check_cuda(cudaEventSynchronize(boundaries_.back().event),"operator events ready");
        for(std::size_t i=0;i+1<boundaries_.size();++i) {
            float ms=0;const auto &a=boundaries_[i],&b=boundaries_[i+1];
            poseidon::gpu::gpu_check_cuda(cudaEventElapsedTime(&ms,a.event,b.event),"operator event elapsed");
            const auto wall=std::chrono::duration<double>(b.wall-a.wall).count();
            std::cout<<"SOURCE_OPERATOR {\"layer\":"<<layer<<",\"stage\":\""<<a.label
                <<"\",\"event_span_ms\":"<<ms<<",\"host_submission_seconds\":"<<wall
                <<",\"boot_calls\":"<<b.boots-a.boots<<"}"<<std::endl;
        }
    }
};
} // namespace cerium_source
