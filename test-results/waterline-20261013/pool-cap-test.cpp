#include "poseidon/runtime_api/poseidon_gpu_api.cpp"
#include <iostream>

using namespace poseidon::runtime_api;

void check(bool ok, const char *what) {
    if (!ok) throw std::runtime_error(what);
}
void rejects(const char *value, const char *message_fragment) {
    setenv("POSEIDON_GPU_RUNTIME_MAXIMUM_POOL_MB",value,1);
    try { (void)maximum_device_pool_size(0); }
    catch (const std::invalid_argument &error) {
        check(std::string(error.what()).find(message_fragment)!=std::string::npos,
              "unexpected pool parse error");
        std::cout<<"reject value="<<value<<" error="<<error.what()<<'\n';
        return;
    }
    throw std::runtime_error(std::string("accepted invalid cap: ")+value);
}
int main() {
    try {
        int visible=0;
        poseidon::gpu::gpu_check_cuda(cudaGetDeviceCount(&visible),"visible devices");
        check(visible>0,"no visible CUDA devices");
        poseidon::gpu::gpu_check_cuda(cudaSetDevice(0),"set device");
        poseidon::gpu::gpu_check_cuda(cudaFree(nullptr),"warm context");
        (void)rmm::mr::get_current_device_resource();
        std::size_t free=0,total=0;
        poseidon::gpu::gpu_check_cuda(cudaMemGetInfo(&free,&total),"query free");
        unsetenv("POSEIDON_GPU_RUNTIME_MAXIMUM_POOL_MB");
        auto default_cap=maximum_device_pool_size(0);
        check(default_cap<=free-(64ULL<<20),"default headroom missing");
        check(default_cap%256==0,"default alignment missing");
        setenv("POSEIDON_GPU_RUNTIME_MAXIMUM_POOL_MB","64",1);
        check(maximum_device_pool_size(0)==(64ULL<<20),"scalar cap mismatch");
        std::string visible_caps;
        for (int ordinal=0;ordinal<visible;++ordinal) {
            if (ordinal) visible_caps+=',';
            visible_caps+=std::to_string(64+ordinal);
        }
        setenv("POSEIDON_GPU_RUNTIME_MAXIMUM_POOL_MB",visible_caps.c_str(),1);
        check(maximum_device_pool_size(0)==(64ULL<<20),"ordinal zero cap mismatch");
        if (visible>1) {
            for (int ordinal=1;ordinal<visible;++ordinal) {
                poseidon::gpu::gpu_check_cuda(cudaSetDevice(ordinal),"set ordinal");
                check(maximum_device_pool_size(ordinal)==(std::size_t(64+ordinal)<<20),
                      "CUDA ordinal list indexing mismatch");
            }
            poseidon::gpu::gpu_check_cuda(cudaSetDevice(0),"restore ordinal");
        }
        rejects("63","integers >=64");
        rejects("-64","integers >=64");
        rejects("64x","integers >=64");
        rejects("64,","integers >=64");
        rejects(",64","integers >=64");
        rejects("64,,64","integers >=64");
        rejects("18446744073709551615","integers >=64");
        std::string invalid_count="64";
        for (int i=0;i<visible;++i) invalid_count+=",64";
        rejects(invalid_count.c_str(),"one entry per visible CUDA ordinal");
        const auto oversized_mb=free/(1ULL<<20)+1;
        rejects(std::to_string(oversized_mb).c_str(),"64 MiB setup headroom");
        setenv("POSEIDON_GPU_RUNTIME_INITIAL_POOL_MB","64",1);
        setenv("POSEIDON_GPU_RUNTIME_MAXIMUM_POOL_MB","128",1);
        void *outside=nullptr;
        poseidon::gpu::gpu_check_cuda(cudaMalloc(&outside,32ULL<<20),"outside setup allocation");
        poseidon::gpu::gpu_check_cuda(cudaMemGetInfo(&free,&total),"prepool query");
        auto *old=rmm::mr::get_current_device_resource();
        std::size_t baseline=0,reserved=0,cap=0;
        {
            DeviceMemoryPool pool(0);
            baseline=pool.initial_external_bytes();
            reserved=pool.reserved_bytes();
            cap=pool.capacity_bytes();
            check(baseline==total-free,"pool initial baseline differs from preallocation driver query");
            check(cap==(128ULL<<20),"pool cap mismatch");
            check(reserved==(64ULL<<20),"initial reserve mismatch");
            auto *allocation=pool.resource()->allocate(8ULL<<20);
            check(pool.bytes_counter().value==(8ULL<<20),"statistics live bytes mismatch");
            pool.resource()->deallocate(allocation,8ULL<<20);
            check(pool.bytes_counter().value==0,"live bytes not retired");
        }
        check(rmm::mr::get_current_device_resource()==old,"resource not restored after pool destruction");
        poseidon::gpu::gpu_check_cuda(cudaFree(outside),"outside setup release");
        std::cout<<"visible_cuda_devices="<<visible<<" default_cap_bytes="<<default_cap
                 <<" captured_initial_external_bytes="<<baseline
                 <<" artificial_prepool_outside_bytes="<<(32ULL<<20)
                 <<" initial_reserved_bytes="<<reserved<<" configured_cap_bytes="<<cap<<'\n';
        std::cout<<"PASS scalar/list parsing, invalid count/value/range rejection, free-64MiB limit, prepool baseline and pool statistics/lifetime\n";
        if(visible==1)std::cout<<"LIMITATION valid multi-ordinal list selection requires server all-GPU run; local one visible CUDA device\n";
    } catch(const std::exception &error) {
        std::cerr<<"FAIL "<<error.what()<<'\n';
        return 1;
    }
}
