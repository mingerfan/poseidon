// Independent, lossless plaintext-format experiment. No runtime API changes.
#include "gpu_encode_experiment.h"
#include "gpu_plaintext_compression_kernels.h"
#include "poseidon/basics/util/croots.h"
#include "poseidon/ckks_encoder.h"
#include "json.hpp"
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <numeric>
#include <random>

using namespace poseidon;
using namespace poseidon::gpu;
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
using Backend = GpuEncodeExperiment::Backend;
namespace {
class Pool {
    rmm::mr::device_memory_resource *previous_ = rmm::mr::get_current_device_resource();
    rmm::mr::cuda_memory_resource upstream_;
    rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> pool_{&upstream_,64ULL<<20};
public:
    Pool() { rmm::mr::set_current_device_resource(&pool_); }
    ~Pool() { rmm::mr::set_current_device_resource(previous_); }
};
template<class T> struct Pinned {
    T *data = nullptr;
    std::size_t bytes;
    explicit Pinned(const std::vector<T> &values) : bytes(values.size()*sizeof(T)) {
        gpu_check_cuda(cudaMallocHost(&data,bytes),"allocate pinned payload");
        std::memcpy(data,values.data(),bytes);
    }
    ~Pinned() { if (data) cudaFreeHost(data); }
};
struct Event {
    cudaEvent_t value{};
    Event() { gpu_check_cuda(cudaEventCreate(&value),"create event"); }
    ~Event() { cudaEventDestroy(value); }
    void record() { gpu_check_cuda(cudaEventRecord(value,gpu_execution_stream()),"record event"); }
};
double elapsed(const Event &a,const Event &b) {
    float ms;
    gpu_check_cuda(cudaEventElapsedTime(&ms,a.value,b.value),"elapsed event");
    return ms;
}
Json stats(const std::vector<double> &values) {
    auto sorted=values;
    std::sort(sorted.begin(),sorted.end());
    return {{"median_ms",(sorted[(sorted.size()-1)/2]+sorted[sorted.size()/2])/2},
        {"p95_ms",sorted[std::size_t(std::ceil(sorted.size()*.95))-1]},
        {"samples_ms",values}};
}
unsigned width(std::uint64_t value) {
    unsigned bits=1;
    while(value>>bits && bits<63) ++bits;
    if(value>>63) throw std::invalid_argument("payload requires more than 63 bits");
    return bits;
}
std::vector<std::uint64_t> pack(const std::vector<std::uint64_t> &values,unsigned bits) {
    std::vector<std::uint64_t> result((values.size()*bits+63)/64+1,0);
    for(std::size_t i=0;i<values.size();++i) {
        if(values[i]>>bits) throw std::runtime_error("bitpack overflow");
        const auto offset=i*bits;
        const unsigned shift=offset%64;
        result[offset/64]|=values[i]<<shift;
        if(shift+bits>64) result[offset/64+1]|=values[i]>>(64-shift);
    }
    return result;
}
std::vector<double> coefficients(const std::vector<double> &slots,std::size_t n,
    std::size_t batch,const std::vector<std::uint32_t> &indices,const std::vector<double2> &twists) {
    const int logn=util::get_power_of_two(n);
    std::vector<std::complex<double>> roots(n),work(n);
    for(std::size_t i=1;i<n;++i) {
        const auto r=twists[util::reverse_bits(i-1,logn)+1];
        roots[i]={r.x,r.y};
    }
    util::FFTHandler fft{util::ComplexArith{}};
    const double normalization=1./n;
    std::vector<double> result(n*batch);
    for(std::size_t b=0;b<batch;++b) {
        for(std::size_t i=0;i<n/2;++i) {
            work[util::reverse_bits(indices[i],logn)]={slots[b*n/2+i],0};
            work[util::reverse_bits(indices[n/2+i],logn)]={slots[b*n/2+i],0};
        }
        fft.transform_from_rev(work.data(),logn,roots.data(),&normalization);
        for(std::size_t i=0;i<n;++i) result[b*n+i]=work[i].real();
    }
    return result;
}
std::size_t replication_group(const std::vector<GpuWord> &values,std::size_t n) {
    for(std::size_t group=n;group>1;group/=2) {
        bool equal=true;
        for(std::size_t i=0;i<values.size() && equal;++i)
            equal=values[i]==values[i/group*group];
        if(equal) return group;
    }
    return 1;
}
struct Mode {
    std::string name;
    const void *host;
    void *device;
    std::size_t bytes;
    std::function<void()> prepare,ntt;
};
Json measure(const Mode &mode,bool upload,int repeat,double warmup_ms) {
    Event begin,uploaded,prepared,end;
    std::vector<double> total,transfer,preparation,ntt,wall;
    double warmed=0;
    int warm_iterations=0;
    while(total.size()<std::size_t(repeat)) {
        const bool sample=warmed>=warmup_ms;
        const auto start=Clock::now();
        begin.record();
        if(upload) gpu_check_cuda(cudaMemcpyAsync(mode.device,mode.host,mode.bytes,
            cudaMemcpyHostToDevice,gpu_execution_stream()),"upload compressed payload");
        uploaded.record();
        mode.prepare();
        prepared.record();
        mode.ntt();
        end.record();
        gpu_check_cuda(cudaEventSynchronize(end.value),"finish materialization");
        const auto finish=Clock::now();
        if(sample) {
            total.push_back(elapsed(begin,end));
            transfer.push_back(elapsed(begin,uploaded));
            preparation.push_back(elapsed(uploaded,prepared));
            ntt.push_back(elapsed(prepared,end));
            wall.push_back(std::chrono::duration<double,std::milli>(finish-start).count());
        } else {
            warmed+=elapsed(begin,end);
            ++warm_iterations;
        }
    }
    return {{"total",stats(total)},{"h2d",stats(transfer)},
        {"unpack_or_rns",stats(preparation)},{"ntt",stats(ntt)},
        {"wall",stats(wall)},{"warmup_gpu_ms",warmed},{"warmup_iterations",warm_iterations}};
}
template<class T> void dump(Json &files,const std::string &directory,
    const std::string &name,const std::vector<T> &values,const std::string &dtype) {
    if(directory=="-") return;
    const auto path=std::filesystem::path(directory)/(name+".bin");
    std::ofstream output(path,std::ios::binary|std::ios::trunc);
    output.write(reinterpret_cast<const char*>(values.data()),values.size()*sizeof(T));
    if(!output) throw std::runtime_error("cannot write compression corpus");
    files.push_back({{"file",path.string()},{"representation",name},{"dtype",dtype},
        {"bytes",values.size()*sizeof(T)}});
}
} // namespace

int main(int argc,char **argv)
try {
    if(argc!=8) throw std::invalid_argument(
        "usage: compression_bench Q BATCH SCALE_LOG2 REPEAT SEED REPORT.json DUMP_DIR_OR_DASH");
    const std::size_t n=65536,limbs=std::stoul(argv[1]),batch=std::stoul(argv[2]);
    const int scale_log2=std::stoi(argv[3]),repeat=std::stoi(argv[4]);
    const auto seed=std::stoull(argv[5]);
    if(!limbs||limbs>64||!batch||batch>32||repeat<1||scale_log2<20||scale_log2>60||
        scale_log2+1>=int(limbs*30)) throw std::invalid_argument("unsupported benchmark geometry");
    const double scale=std::ldexp(1.,scale_log2);
    const char *warm_env=std::getenv("POSEIDON_COMPRESSION_WARMUP_MS");
    const double warmup_ms=warm_env?std::stod(warm_env):100.;
    if(!std::isfinite(warmup_ms)||warmup_ms<0) throw std::invalid_argument("invalid warmup");
    const std::string directory=argv[7];
    if(directory!="-") std::filesystem::create_directories(directory);
    gpu_check_cuda(cudaSetDevice(0),"select GPU");
    Pool pool;
    cudaDeviceProp props{};
    gpu_check_cuda(cudaGetDeviceProperties(&props,0),"GPU properties");
    setenv("POSEIDON_NTT_ALGO","tensor",1);
    setenv("POSEIDON_NTT_FUSION_STAGES","4",1);
    setenv("POSEIDON_NTT_FUSED_MATRIX_STAGES","4",1);
    setenv("POSEIDON_NTT_FUSED_MATRIX_MAX_LEVELS","1",1);
    setenv("POSEIDON_NTT_FUSED_MATRIX_FP64_TABLES","0",1);
    ParametersLiteral parameters(CKKS,16,15,scale_log2,0,0,Modulus(0),{},{},sec_level_type::none);
    parameters.set_log_modulus(std::vector<std::uint32_t>(limbs,30),{});
    PoseidonContext context(parameters);
    GpuParameterData gpu_parameters(context,0);
    const auto &level=gpu_parameters.get_first_q_level();
    if(level.q_count!=limbs||level.shards.size()!=1) throw std::runtime_error("unexpected RNS layout");
    const auto &shard=level.shards.front();
    std::vector<std::uint32_t> indices(n);
    std::vector<double2> twists(n);
    util::ComplexRoots roots(2*n,MemoryManager::GetPool());
    std::size_t position=1;
    for(std::size_t i=0;i<n/2;++i) {
        indices[i]=(position-1)/2;
        indices[n/2+i]=(2*n-position-1)/2;
        position=position*5%(2*n);
    }
    for(std::size_t i=0;i<n;++i) {
        const auto r=std::conj(roots.get_root(i));
        twists[i]=make_double2(r.real(),r.imag());
    }
    GpuEncodeExperiment encoder(n,limbs,batch,indices,twists);
    std::vector<std::pair<Backend,std::string>> backends{{Backend::cuda,"cuda_ntt"}};
#ifdef POSEIDON_ENCODE_TILELANG
    if(props.major*10+props.minor<POSEIDON_ENCODE_TILELANG_SM)
        throw std::runtime_error("GPU older than configured TileLang target");
    encoder.initialize_tilelang(shard);
    backends.emplace_back(Backend::tilelang_tensor,"tilelang_tensor_ntt");
#endif
    CKKSEncoder cpu(context);
    Json report{{"gpu",props.name},{"degree",n},{"q_limbs",limbs},{"batch",batch},
        {"scale_log2",scale_log2},{"repeat",repeat},{"seed",seed},{"modulus_bits",30},
        {"gpu_word_bytes",sizeof(GpuWord)},{"cpu_word_bytes",sizeof(std::uint64_t)},
        {"scope","synthetic real slots; lossless; no compiler/runtime integration or concurrent foreground"},
        {"tensor_scope","CUDA unpack/RNS then TileLang INT8 Tensor NTT; not Tensor entropy decoding"},
        {"payload_size_scope","payload only; pack includes one uint64 guard word; format metadata excluded"},
        {"cases",Json::array()}};
    std::mt19937_64 rng(seed),order_rng(seed+1);
    std::normal_distribution<double> normal(0.,.02);
    for(const std::string pattern:{"dense","periodic1024","prefix896","scalar"}) {
        std::vector<double> slots(n/2*batch),base(1024);
        for(std::size_t b=0;b<batch;++b) {
            for(auto &x:base) x=normal(rng);
            for(std::size_t i=0;i<n/2;++i)
                slots[b*n/2+i]=pattern=="scalar"?.125:pattern=="periodic1024"?base[i%1024]:
                    pattern=="prefix896"?(i<896?normal(rng):0.):normal(rng);
        }
        auto fft=coefficients(slots,n,batch,indices,twists);
        std::vector<std::int64_t> integers(fft.size());
        std::vector<std::uint64_t> codes(fft.size());
        std::uint64_t max_code=0;
        for(std::size_t i=0;i<fft.size();++i) {
            const double x=std::round(fft[i]*scale);
            if(!std::isfinite(x)||std::abs(x)>=0x1p62) throw std::runtime_error("coefficient exceeds compact prototype");
            integers[i]=std::int64_t(x);
            codes[i]=x<0?std::uint64_t(-integers[i])*2-1:std::uint64_t(integers[i])*2;
            max_code=std::max(max_code,codes[i]);
        }
        const auto bits=width(max_code);
        auto packed=pack(codes,bits);
        std::vector<std::uint64_t> encoded64(n*limbs*batch);
        std::vector<GpuWord> reference(encoded64.size());
        for(std::size_t b=0;b<batch;++b) {
            Plaintext plain;
            cpu.encode(std::vector<double>(slots.begin()+b*n/2,slots.begin()+(b+1)*n/2),
                level.parms_id,scale,plain);
            for(std::size_t i=0;i<n*limbs;++i) {
                const auto value=plain.data()[i];
                if(value>std::numeric_limits<GpuWord>::max()) throw std::runtime_error("residue overflows GPU word");
                encoded64[b*n*limbs+i]=value;
                reference[b*n*limbs+i]=value;
            }
        }
        auto packed_ntt=pack(encoded64,30);
        const auto group=replication_group(reference,n);
        std::vector<GpuWord> reduced;
        for(std::size_t i=0;i<reference.size();i+=group) reduced.push_back(reference[i]);
        Pinned<double> host_fft(fft);
        Pinned<std::uint64_t> host_packed(packed),host_packed_ntt(packed_ntt);
        Pinned<GpuWord> host_encoded(reference),host_reduced(reduced);
        DeviceVector<double> dev_fft(fft.size(),0);
        DeviceVector<std::uint64_t> dev_packed(packed.size(),0),dev_packed_ntt(packed_ntt.size(),0);
        DeviceVector<GpuWord> dev_reduced(reduced.size(),0);
        dev_fft.copy_from_host(fft.data(),fft.size());
        dev_packed.copy_from_host(packed.data(),packed.size());
        dev_packed_ntt.copy_from_host(packed_ntt.data(),packed_ntt.size());
        dev_reduced.copy_from_host(reduced.data(),reduced.size());
        encoder.prepare_coefficients(dev_fft.data(),scale,shard,true);
        std::vector<Mode> modes;
        const auto noop=[]{};
        modes.push_back({"encoded_uint32_upload",host_encoded.data,encoder.output.data(),host_encoded.bytes,noop,noop});
        modes.push_back({"ntt_bitpack30_cuda",host_packed_ntt.data,dev_packed_ntt.data(),host_packed_ntt.bytes,[&]{
            gpu_check_cuda(launch_unpack_residues(dev_packed_ntt.data(),encoder.output.data(),
                reference.size(),30,gpu_execution_stream()),"unpack NTT residues");
        },noop});
        if(group>1) modes.push_back({"ntt_replication_cuda",host_reduced.data,dev_reduced.data(),host_reduced.bytes,[&]{
            gpu_check_cuda(launch_replicate_ntt(dev_reduced.data(),encoder.output.data(),
                reference.size(),n,group,gpu_execution_stream()),"replicate NTT groups");
        },noop});
        for(const auto &[backend,name]:backends) {
            modes.push_back({"fft_fp64_"+name,host_fft.data,dev_fft.data(),host_fft.bytes,
                [&]{encoder.prepare_coefficients(dev_fft.data(),scale,shard,true);},
                [&,backend]{encoder.ntt(shard,backend,true);}});
            modes.push_back({"integer_bitpack_"+name,host_packed.data,dev_packed.data(),host_packed.bytes,
                [&]{encoder.prepare_packed_coefficients(dev_packed.data(),bits,shard);},
                [&,backend]{encoder.ntt(shard,backend,true);}});
        }
        std::shuffle(modes.begin(),modes.end(),order_rng);
        Json row{{"pattern",pattern},{"coefficient_bits",bits},{"replication_group",group},
            {"slots_bytes",slots.size()*8},{"fft_fp64_bytes",fft.size()*8},
            {"integer_int64_bytes",integers.size()*8},{"integer_bitpacked_bytes",packed.size()*8},
            {"encoded_cpu_uint64_bytes",encoded64.size()*8},{"encoded_gpu_uint32_bytes",reference.size()*4},
            {"ntt_bitpacked_bytes",packed_ntt.size()*8},{"ntt_replicated_payload_bytes",reduced.size()*4},
            {"gpu_to_integer_compression_ratio",double(reference.size()*4)/(packed.size()*8)},
            {"files",Json::array()},{"modes",Json::array()}};
        dump(row["files"],directory,pattern+"-slots-fp64",slots,"float64");
        dump(row["files"],directory,pattern+"-fft-fp64",fft,"float64");
        dump(row["files"],directory,pattern+"-integer-int64",integers,"int64");
        dump(row["files"],directory,pattern+"-integer-bitpacked",packed,"packed-zigzag");
        dump(row["files"],directory,pattern+"-ntt-cpu-uint64",encoded64,"uint64");
        dump(row["files"],directory,pattern+"-ntt-gpu-uint32",reference,"uint32");
        dump(row["files"],directory,pattern+"-ntt-bitpack30",packed_ntt,"packed30");
        dump(row["files"],directory,pattern+"-ntt-replication",reduced,"uint32");
        gpu_check_cuda(cudaStreamSynchronize(gpu_execution_stream()),"finish setup");
        for(const auto &mode:modes) {
            // Validate before warmup/timing; all formats must reconstruct every
            // NTT residue bit-for-bit, not just match decoded slot tolerance.
            gpu_check_cuda(cudaMemcpyAsync(mode.device,mode.host,mode.bytes,cudaMemcpyHostToDevice,
                gpu_execution_stream()),"validation upload");
            mode.prepare();mode.ntt();encoder.check_result();
            std::vector<GpuWord> actual(reference.size());
            encoder.output.copy_to_host(actual.data(),actual.size());
            if(actual!=reference) throw std::runtime_error(pattern+"/"+mode.name+": CPU NTT mismatch");
            Json measured{{"name",mode.name},{"uploaded_bytes",mode.bytes},{"exact_cpu_ntt_match",true}};
            if(mode.name!="encoded_uint32_upload") measured["device_only"]=measure(mode,false,repeat,warmup_ms);
            measured["with_h2d"]=measure(mode,true,repeat,warmup_ms);
            if(mode.name=="ntt_bitpack30_cuda"||mode.name=="ntt_replication_cuda")
                measured["decompressed_output_gb_per_second"]=reference.size()*4/
                    (measured["device_only"]["unpack_or_rns"]["median_ms"].get<double>()*1e6);
            row["modes"].push_back(std::move(measured));
        }
        std::cout<<pattern<<": integer bits="<<bits<<", NTT replication="<<group
            <<", all modes exact"<<std::endl;
        report["cases"].push_back(std::move(row));
    }
    std::ofstream output(argv[6]);
    output<<report.dump(2)<<'\n';
    if(!output) throw std::runtime_error("cannot write benchmark report");
    return 0;
} catch(const std::exception &error) {
    std::cerr<<error.what()<<std::endl;
    return 1;
}
