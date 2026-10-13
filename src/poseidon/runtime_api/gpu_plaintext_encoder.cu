#include "gpu_plaintext_encoder.h"
#include "poseidon/gpu/kernels/gpu_ntt_kernels.h"
#include "poseidon/util/defines.h"
#include <cmath>
#include <cstdio>
#include <exception>
#include <limits>
#include <stdexcept>
#include <vector>
namespace poseidon::runtime_api {
using namespace gpu;
namespace {
void fft_check(cufftResult status,const char *name) {
    if(status!=CUFFT_SUCCESS)throw std::runtime_error(std::string(name)+": cuFFT error "+std::to_string(status));
}
__global__ void permute(const double *input,double2 *work,const std::uint32_t *indices,std::size_t n) {
    const auto i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n/2)return;
    work[indices[i]]=make_double2(input[i],0);work[indices[n/2+i]]=make_double2(input[i],0);
}
__global__ void expand(const double2 *fft,const double2 *twists,GpuWord *rns,const GpuWord *primes,
    std::size_t n,std::size_t limbs,double fix) {
    const auto i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n)return;
    const auto x=fft[i],w=twists[i];const double c=round((x.x*w.x-x.y*w.y)*fix);
    const double magnitude=fabs(c);
    int exponent=0;
    const double fraction=frexp(magnitude,&exponent);
    const auto significant=static_cast<unsigned long long>(ldexp(fraction,53));
    for(std::size_t l=0;l<limbs;++l) {
        const GpuWord q=primes[l];GpuWord v;
        if(magnitude<0x1p64) v=static_cast<unsigned long long>(magnitude)%q;
        else {
            // A rounded FP64 integer is an exact 53-bit significand times a
            // power of two. Reduce that representation without an overflowing cast.
            unsigned shift=static_cast<unsigned>(exponent-53);
            unsigned long long power=1,base=2;
            while(shift) {if(shift&1)power=power*base%q;base=base*base%q;shift>>=1;}
            v=(significant%q)*power%q;
        }
        rns[l*n+i]=c<0 && v?q-v:v;
    }
}
struct Scratch {
    DeviceVector<double> raw;
    DeviceVector<double2> fft;
    DeviceVector<GpuWord> rns;
    DeviceVector<std::uint8_t> fft_work;
    Scratch(std::size_t degree,std::size_t limbs,std::size_t bytes,int device)
        :raw(degree/2,device),fft(degree,device),rns(degree*limbs,device),fft_work(bytes,device) {}
};
struct TimingEvent {
    cudaEvent_t value=nullptr;
    TimingEvent() {gpu_check_cuda(cudaEventCreateWithFlags(&value,cudaEventDefault),"GPU Encode timing event create");}
    ~TimingEvent() noexcept {
        const auto status=cudaEventDestroy(value);
        if(status!=cudaSuccess) {
            std::fprintf(stderr,"GPU Encode timing event destroy: %s\n",cudaGetErrorString(status));
            std::terminate();
        }
    }
    TimingEvent(const TimingEvent &)=delete;
    TimingEvent &operator=(const TimingEvent &)=delete;
};
std::uint64_t event_nanoseconds(cudaEvent_t begin,cudaEvent_t end) {
    float milliseconds=0;
    gpu_check_cuda(cudaEventElapsedTime(&milliseconds,begin,end),"GPU Encode timing elapsed");
    const double nanoseconds=static_cast<double>(milliseconds)*1e6;
    if(!std::isfinite(nanoseconds) || nanoseconds<0)throw std::runtime_error("invalid GPU Encode event duration");
    if(nanoseconds>=static_cast<double>(std::numeric_limits<std::uint64_t>::max()))
        throw std::overflow_error("GPU Encode event duration overflow");
    return static_cast<std::uint64_t>(nanoseconds);
}
void add_timing(std::uint64_t &total,std::uint64_t delta) {
    if(delta>std::numeric_limits<std::uint64_t>::max()-total)throw std::overflow_error("GPU Encode timing counter overflow");
    total+=delta;
}
}
struct GpuPlaintextEncoder::TimingEvents {
    TimingEvent raw_begin,raw_end,encode_end;
    std::uint64_t raw_h2d_bytes;
    explicit TimingEvents(std::uint64_t bytes):raw_h2d_bytes(bytes) {}
};
GpuPlaintextEncoder::GpuPlaintextEncoder(std::size_t degree,int device)
    :degree_(degree),device_(device) {
    if(degree<2 || degree>(1U<<20) || (degree&(degree-1)))throw std::invalid_argument("unsupported GPU Encode degree");
    gpu_check_cuda(cudaSetDevice(device),"GPU Encode device");
    int least=0,greatest=0;gpu_check_cuda(cudaDeviceGetStreamPriorityRange(&least,&greatest),"GPU Encode priorities");
    gpu_check_cuda(cudaStreamCreateWithPriority(&stream_,cudaStreamNonBlocking,least),"GPU Encode stream");
    try {
        ScopedGpuExecutionStream scope(stream_);
        fft_check(cufftCreate(&plan_),"GPU Encode plan");
        fft_check(cufftSetAutoAllocation(plan_,0),"GPU Encode explicit workspace");
        fft_check(cufftMakePlan1d(plan_,int(degree),CUFFT_Z2Z,1,&fft_work_bytes_),"GPU Encode plan geometry");
        fft_check(cufftSetStream(plan_,stream_),"GPU Encode plan stream");
        std::vector<std::uint32_t> indices(degree);std::vector<double2> twists(degree);
        std::size_t pos=1;
        for(std::size_t i=0;i<degree/2;++i) {indices[i]=(pos-1)/2;indices[degree/2+i]=(2*degree-pos-1)/2;pos=pos*5%(2*degree);}
        const double pi=std::acos(-1.0);
        for(std::size_t i=0;i<degree;++i)twists[i]=make_double2(std::cos(pi*i/degree),-std::sin(pi*i/degree));
        indices_=DeviceVector<std::uint32_t>(degree,device);twists_=DeviceVector<double2>(degree,device);
        indices_.copy_from_host(indices.data(),degree);twists_.copy_from_host(twists.data(),degree);
    } catch(...) {if(plan_)cufftDestroy(plan_);cudaStreamDestroy(stream_);throw;}
}
GpuPlaintextEncoder::~GpuPlaintextEncoder() {
    (void)cudaSetDevice(device_);(void)cudaStreamSynchronize(stream_);
    timing_events_.clear();
    if(plan_)cufftDestroy(plan_);if(stream_)cudaStreamDestroy(stream_);
}
std::shared_ptr<void> GpuPlaintextEncoder::submit(const double *slots,double scale,
    const GpuParameterShard &parameters,GpuPlaintextData &output) {
    if(parameters.device_id!=device_ || parameters.limb_begin!=0 || parameters.limb_count<output.meta.q_count ||
        output.meta.degree!=degree_ || !(scale>0) || !std::isfinite(scale))throw std::invalid_argument("GPU Encode parameter/place mismatch");
    gpu_check_cuda(cudaSetDevice(device_),"GPU Encode submit device");ScopedGpuExecutionStream scope(stream_);
    const auto limbs=output.meta.q_count;auto scratch=std::make_shared<Scratch>(degree_,limbs,fft_work_bytes_,device_);
    fft_check(cufftSetWorkArea(plan_,scratch->fft_work.data()),"GPU Encode FFT workspace");
    auto timing=std::make_unique<TimingEvents>(degree_*4);
    gpu_check_cuda(cudaEventRecord(timing->raw_begin.value,stream_),"GPU Encode raw H2D start event");
    gpu_check_cuda(cudaMemcpyAsync(scratch->raw.data(),slots,degree_*4,cudaMemcpyHostToDevice,stream_),"GPU Encode raw H2D");
    gpu_check_cuda(cudaEventRecord(timing->raw_end.value,stream_),"GPU Encode raw H2D end event");
    permute<<<(degree_/2+255)/256,256,0,stream_>>>(scratch->raw.data(),scratch->fft.data(),indices_.data(),degree_);
    gpu_check_cuda(cudaGetLastError(),"GPU Encode permutation");
    fft_check(cufftExecZ2Z(plan_,scratch->fft.data(),scratch->fft.data(),CUFFT_FORWARD),"GPU Encode FFT");
    expand<<<(degree_+255)/256,256,0,stream_>>>(scratch->fft.data(),twists_.data(),scratch->rns.data(),parameters.rns_primes.data(),degree_,limbs,scale/degree_);
    gpu_check_cuda(cudaGetLastError(),"GPU Encode RNS");
    auto out=output.make_view().poly.shards.front();
    GpuConstPolyShardView in{device_,scratch->rns.data(),0,limbs,0,degree_};
    kernel::launch_forward_ntt_plaintext_cuda(out,in,parameters,degree_);
    gpu_check_cuda(cudaEventRecord(timing->encode_end.value,stream_),"GPU Encode completion timing event");
    {std::lock_guard<std::mutex> lock(timing_mutex_);timing_events_.push_back(std::move(timing));}
    return scratch;
}
GpuPlaintextEncoder::CompletedTiming GpuPlaintextEncoder::completed_timing(bool wait) {
    std::lock_guard<std::mutex> lock(timing_mutex_);
    gpu_check_cuda(cudaSetDevice(device_),"GPU Encode timing device");
    for(auto it=timing_events_.begin();it!=timing_events_.end();) {
        const auto &events=**it;
        const auto status=wait?cudaEventSynchronize(events.encode_end.value):cudaEventQuery(events.encode_end.value);
        if(!wait && status==cudaErrorNotReady) {++it;continue;}
        gpu_check_cuda(status,"GPU Encode timing completion");
        auto next=timing_;
        add_timing(next.calls,1);
        add_timing(next.raw_h2d_bytes,events.raw_h2d_bytes);
        add_timing(next.raw_h2d_nanoseconds,event_nanoseconds(events.raw_begin.value,events.raw_end.value));
        add_timing(next.encode_nanoseconds,event_nanoseconds(events.raw_end.value,events.encode_end.value));
        timing_=next;
        it=timing_events_.erase(it);
    }
    return timing_;
}
}
