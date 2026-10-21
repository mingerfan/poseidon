// CPU-side CUDA API counters only. No CUDA events, synchronization, or CUPTI.
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <nlohmann/json.hpp>
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unordered_map>

namespace {
using Json=nlohmann::json;
std::uint64_t now() {
    timespec t{};
    if(clock_gettime(CLOCK_MONOTONIC,&t))std::abort();
    return std::uint64_t(t.tv_sec)*1000000000+t.tv_nsec;
}
const std::uint64_t origin=now();
const std::uint64_t begin=origin+std::stoull(std::getenv("CUDA_PROBE_DELAY_SECONDS"))*1000000000;
const std::uint64_t end=begin+std::stoull(std::getenv("CUDA_PROBE_DURATION_SECONDS"))*1000000000;
const char *output=std::getenv("CUDA_PROBE_OUTPUT");
const char *names[]={"cudaSetDevice","cudaGetDevice","cudaLaunchKernel","cuLaunchKernel",
    "cudaEventQuery","cudaEventRecord","cudaEventCreateWithFlags","cudaEventDestroy",
    "cudaEventElapsedTime","cudaStreamWaitEvent","cudaEventSynchronize",
    "cudaStreamSynchronize","cudaDeviceSynchronize","cudaMemcpyAsync",
    "cudaMemcpyPeerAsync","cudaMemcpy","cudaMalloc","cudaFree","cudaMemsetAsync",
    "cudaLaunchKernel_ptsz","cudaMemcpyAsync_ptsz","cudaEventRecord_ptsz",
    "cudaStreamSynchronize_ptsz","cudaStreamWaitEvent_ptsz","cudaMemsetAsync_ptsz",
    "cudaMallocHost","cudaFreeHost"};
constexpr int api_count=sizeof(names)/sizeof(*names);
struct Stat {
    std::uint64_t count=0,total=0,maximum=0,errors=0;
    std::array<std::uint64_t,32> log2_ns{};
    void add(std::uint64_t ns,int error) {
        ++count;total+=ns;if(ns>maximum)maximum=ns;
        if(error && error!=int(cudaErrorNotReady))++errors;
        const auto bin=ns?std::min(31,63-__builtin_clzll(ns)):0;
        ++log2_ns[bin];
    }
};
struct Thread {
    int device=-1;
    bool dumped=false;
    std::uint64_t calls=0,first=0,last=0;
    std::uint64_t random=0x9e3779b97f4a7c15ULL;
    std::array<std::array<Stat,5>,api_count> stats{};
    std::array<std::uint64_t,5> last_launch{};
    std::array<Stat,5> launch_intervals{};
    std::unordered_map<std::uintptr_t,std::uint64_t> sampled_setdevice_callers;
    void dump() {
        if(dumped || !calls)return;
        dumped=true;
        Json rows=Json::array(),gaps=Json::array(),callers=Json::array();
        for(int a=0;a<api_count;++a)for(int d=0;d<5;++d) {
            const auto &s=stats[a][d];if(!s.count)continue;
            rows.push_back({{"api",names[a]},{"device",d-1},{"count",s.count},
                {"total_ns",s.total},{"maximum_ns",s.maximum},{"errors_except_not_ready",s.errors},
                {"log2_ns",s.log2_ns}});
        }
        for(int d=0;d<5;++d)if(launch_intervals[d].count) {
            const auto &s=launch_intervals[d];
            gaps.push_back({{"device",d-1},{"count",s.count},{"total_ns",s.total},
                {"maximum_ns",s.maximum},{"log2_ns",s.log2_ns}});
        }
        for(const auto &[address,count]:sampled_setdevice_callers) {
            Dl_info info{};
            if(!dladdr(reinterpret_cast<void *>(address),&info))std::abort();
            callers.push_back({{"object",info.dli_fname},
                {"offset",address-reinterpret_cast<std::uintptr_t>(info.dli_fbase)},
                {"symbol",info.dli_sname?info.dli_sname:""},{"sample_count",count}});
        }
        Json data={{"pid",getpid()},{"tid",syscall(SYS_gettid)},
            {"window_begin_ns",begin},{"window_end_ns",end},{"first_api_start_ns",first},
            {"last_api_end_ns",last},{"rows",rows},{"runtime_launch_start_intervals",gaps},
            {"setdevice_caller_sample_probability",1.0/256},{"setdevice_callers",callers}};
        const std::string line=data.dump()+"\n";
        const int fd=open(output,O_WRONLY|O_CREAT|O_APPEND,0600);
        if(fd<0 || write(fd,line.data(),line.size())!=ssize_t(line.size()) || close(fd))std::abort();
    }
    ~Thread(){dump();}
};
thread_local Thread state;
template<class F> F real(const char *name) {
    const auto p=dlsym(RTLD_NEXT,name);
    if(!p) {std::fprintf(stderr,"CUDA probe missing symbol %s\n",name);std::abort();}
    return reinterpret_cast<F>(p);
}
template<class F> auto call(int api,void *caller,F function) {
    const auto start=now();
    const auto result=function();
    const auto stop=now();
    if(start>=begin && start<end) {
        const int device=state.device;
        if(device < -1 || device>3)std::abort();
        state.stats[api][device+1].add(stop-start,int(result));
        if(!state.calls++)state.first=start;
        state.last=stop;
        if(api==2 || api==19) {
            auto &last=state.last_launch[device+1];
            if(last)state.launch_intervals[device+1].add(start-last,0);
            last=start;
        }
        if(api==0) {
            state.random^=state.random<<13;state.random^=state.random>>7;state.random^=state.random<<17;
            if((state.random&255)==0)
                ++state.sampled_setdevice_callers[reinterpret_cast<std::uintptr_t>(caller)];
        }
    }
    if(stop>=end)state.dump();
    return result;
}
}

#define WRAP(ID,NAME,ARGS,CALL) \
extern "C" cudaError_t NAME ARGS { \
    static const auto f=real<decltype(&NAME)>(#NAME); \
    return call(ID,__builtin_return_address(0),[&]{return f CALL;}); \
}
extern "C" cudaError_t cudaSetDevice(int device) {
    static const auto f=real<decltype(&cudaSetDevice)>("cudaSetDevice");
    // Attribute SetDevice time to its requested destination; other APIs use the
    // most recently observed successful SetDevice/GetDevice on this thread.
    const auto previous=state.device;state.device=device;
    const auto r=call(0,__builtin_return_address(0),[&]{return f(device);});
    if(r!=cudaSuccess)state.device=previous;
    return r;
}
extern "C" cudaError_t cudaGetDevice(int *device) {
    static const auto f=real<decltype(&cudaGetDevice)>("cudaGetDevice");
    const auto r=call(1,__builtin_return_address(0),[&]{return f(device);});
    if(r==cudaSuccess)state.device=*device;
    return r;
}
WRAP(2,cudaLaunchKernel,(const void *fptr,dim3 grid,dim3 block,void **args,size_t shared,cudaStream_t stream),(fptr,grid,block,args,shared,stream))
extern "C" CUresult CUDAAPI cuLaunchKernel(CUfunction fn,unsigned gx,unsigned gy,unsigned gz,
    unsigned bx,unsigned by,unsigned bz,unsigned shared,CUstream stream,void **params,void **extra) {
    static const auto f=real<decltype(&cuLaunchKernel)>("cuLaunchKernel");
    return call(3,__builtin_return_address(0),[&]{return f(fn,gx,gy,gz,bx,by,bz,shared,stream,params,extra);});
}
WRAP(4,cudaEventQuery,(cudaEvent_t event),(event))
WRAP(5,cudaEventRecord,(cudaEvent_t event,cudaStream_t stream),(event,stream))
WRAP(6,cudaEventCreateWithFlags,(cudaEvent_t *event,unsigned flags),(event,flags))
WRAP(7,cudaEventDestroy,(cudaEvent_t event),(event))
WRAP(8,cudaEventElapsedTime,(float *ms,cudaEvent_t start,cudaEvent_t stop),(ms,start,stop))
WRAP(9,cudaStreamWaitEvent,(cudaStream_t stream,cudaEvent_t event,unsigned flags),(stream,event,flags))
WRAP(10,cudaEventSynchronize,(cudaEvent_t event),(event))
WRAP(11,cudaStreamSynchronize,(cudaStream_t stream),(stream))
WRAP(12,cudaDeviceSynchronize,(void),())
WRAP(13,cudaMemcpyAsync,(void *dst,const void *src,size_t bytes,cudaMemcpyKind kind,cudaStream_t stream),(dst,src,bytes,kind,stream))
WRAP(14,cudaMemcpyPeerAsync,(void *dst,int dd,const void *src,int sd,size_t bytes,cudaStream_t stream),(dst,dd,src,sd,bytes,stream))
WRAP(15,cudaMemcpy,(void *dst,const void *src,size_t bytes,cudaMemcpyKind kind),(dst,src,bytes,kind))
WRAP(16,cudaMalloc,(void **ptr,size_t bytes),(ptr,bytes))
WRAP(17,cudaFree,(void *ptr),(ptr))
WRAP(18,cudaMemsetAsync,(void *ptr,int value,size_t bytes,cudaStream_t stream),(ptr,value,bytes,stream))
WRAP(19,cudaLaunchKernel_ptsz,(const void *fptr,dim3 grid,dim3 block,void **args,size_t shared,cudaStream_t stream),(fptr,grid,block,args,shared,stream))
WRAP(20,cudaMemcpyAsync_ptsz,(void *dst,const void *src,size_t bytes,cudaMemcpyKind kind,cudaStream_t stream),(dst,src,bytes,kind,stream))
WRAP(21,cudaEventRecord_ptsz,(cudaEvent_t event,cudaStream_t stream),(event,stream))
WRAP(22,cudaStreamSynchronize_ptsz,(cudaStream_t stream),(stream))
WRAP(23,cudaStreamWaitEvent_ptsz,(cudaStream_t stream,cudaEvent_t event,unsigned flags),(stream,event,flags))
WRAP(24,cudaMemsetAsync_ptsz,(void *ptr,int value,size_t bytes,cudaStream_t stream),(ptr,value,bytes,stream))
WRAP(25,cudaMallocHost,(void **ptr,size_t bytes),(ptr,bytes))
WRAP(26,cudaFreeHost,(void *ptr),(ptr))
