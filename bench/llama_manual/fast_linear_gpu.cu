// Resident-weight GPU encoder for Cerium periodic (period-256) diagonals.
// Replaces the per-matrix CPU path (weight gather -> 512-point inverse DWT ->
// rounding -> 512-point NTT per limb) with one kernel. The DWT replicates the
// host DWTHandler::transform_from_rev butterfly order with non-contracted IEEE
// double operations, so coefficients match the CPU encoder (x86-64, no FMA).
#include "fast_linear_gpu.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdlib>
#include <type_traits>

namespace fast_linear {
namespace {
__device__ __forceinline__ double2 cadd(double2 a,double2 b){return make_double2(__dadd_rn(a.x,b.x),__dadd_rn(a.y,b.y));}
__device__ __forceinline__ double2 csub(double2 a,double2 b){return make_double2(__dsub_rn(a.x,b.x),__dsub_rn(a.y,b.y));}
__device__ __forceinline__ double2 cmul(double2 a,double2 r){
    return make_double2(__dsub_rn(__dmul_rn(a.x,r.x),__dmul_rn(a.y,r.y)),
                        __dadd_rn(__dmul_rn(a.x,r.y),__dmul_rn(a.y,r.x)));
}
__device__ __forceinline__ double2 cscale(double2 a,double s){return make_double2(__dmul_rn(a.x,s),__dmul_rn(a.y,s));}
__device__ __forceinline__ uint32_t mul_shoup(uint32_t a,uint32_t b,uint32_t b_shoup,uint32_t q){
    const uint32_t quotient=__umulhi(a,b_shoup);
    uint64_t r=uint64_t(a)*b-uint64_t(quotient)*q;
    if(r>=q)r-=q;
    return uint32_t(r);
}

// grid: (diagonal position in giant order, input tile); block: 256 threads.
__global__ void encode_diagonals_kernel(EncodeArgs args){
    __shared__ double2 work[512];
    __shared__ long long coeff[512];
    __shared__ uint32_t a[512];
    const int tid=threadIdx.x;
    const int position=blockIdx.x;
    const int it=args.it_begin+blockIdx.y;
    const int d=args.diagonals[position];
    const int gs=(((d+32768)%32768)/16)*16;
    for(int i=tid;i<512;i+=256)work[i]=make_double2(0.,0.);
    __syncthreads();
    {
        const int k=tid;
        const int f=(k-gs+32768)%256,half=f/128,column=f%128,row=column+d;
        double v=0.;
        if(row>=0&&row<128) {
            const long long r=(long long)args.ot*256+((half+args.lane)%2)*128+column;
            const long long c=(long long)it*256+half*128+row;
            v=args.weights[r*args.columns+c];
        }
        work[args.slot_map[k]]=make_double2(v,0.);
        work[args.slot_map[k+256]]=make_double2(v,0.);
    }
    __syncthreads();
    // transform_from_rev(work, log_n=9, inverse_roots, &fix)
    int root_base=1;
    int gap=1;
    for(int m=256;m>1;m>>=1) {
        const int i=tid/gap,j=tid%gap;
        const double2 r=args.inverse_roots[root_base+i];
        const int x=i*2*gap+j,y=x+gap;
        const double2 u=work[x],v=work[y];
        __syncthreads();
        work[x]=cadd(u,v);
        work[y]=cmul(csub(u,v),r);
        __syncthreads();
        root_base+=m;gap<<=1;
    }
    {
        const double2 r=args.inverse_roots[root_base];
        const double2 scaled_r=cscale(r,args.fix);
        const int x=tid,y=tid+256;
        const double2 u=work[x],v=work[y];
        __syncthreads();
        work[x]=cscale(cadd(u,v),args.fix);
        work[y]=cmul(csub(u,v),scaled_r);
        __syncthreads();
    }
    for(int i=tid;i<512;i+=256) {
        const double value=round(work[i].x);
        if(!(fabs(value)<args.coefficient_limit)){atomicExch(args.overflow,1);coeff[i]=0;}
        else coeff[i]=(long long)value;
    }
    __syncthreads();
    const int limbs=args.nq+args.np;
    uint32_t *out=args.outputs[blockIdx.y]+(uint64_t)position*limbs*512;
    for(int limb=0;limb<limbs;++limb) {
        const uint64_t table=limb<args.nq?uint64_t(limb):uint64_t(args.full_q)+limb-args.nq;
        const uint32_t q=args.moduli[table];const uint64_t table_offset=table*512;
        for(int i=tid;i<512;i+=256) {
            const long long v=coeff[i];
            const uint64_t magnitude=v<0?uint64_t(0)-uint64_t(v):uint64_t(v);
            const uint32_t r=uint32_t(magnitude%q);
            a[i]=v<0&&r?q-r:r;
        }
        __syncthreads();
        for(int m=1,g=256;m<512;m*=2,g/=2) {
            const int group=tid/g,j=group*(2*g)+(tid%g);
            const uint64_t root_index=table_offset+m+group;
            const uint32_t u=a[j],v=mul_shoup(a[j+g],args.ntt_roots[root_index],args.ntt_shoup[root_index],q);
            uint64_t sum=uint64_t(u)+v;if(sum>=q)sum-=q;
            a[j]=uint32_t(sum);a[j+g]=u>=v?u-v:q-(v-u);
            __syncthreads();
        }
        for(int i=tid;i<512;i+=256)out[limb*512+i]=a[args.permutation[table_offset+i]];
        __syncthreads();
    }
}
} // namespace

cudaError_t launch_encode_diagonals(const EncodeArgs &args,int diagonal_count,int input_tiles,cudaStream_t stream){
    if(!args.weights||!args.diagonals||!args.slot_map||!args.inverse_roots||!args.moduli||!args.ntt_roots||
       !args.ntt_shoup||!args.permutation||!args.outputs||!args.overflow||diagonal_count<=0||input_tiles<=0||
       args.nq<=0||args.np<0||args.full_q<args.nq)
        return cudaErrorInvalidValue;
    encode_diagonals_kernel<<<dim3(diagonal_count,input_tiles),256,0,stream>>>(args);
    return cudaGetLastError();
}

namespace {
__device__ __forceinline__ uint32_t reduce_u64_mod(uint64_t value,uint32_t q,uint64_t barrett_ratio) {
    const uint64_t quotient=__umul64hi(value,barrett_ratio);
    uint64_t r=value-quotient*uint64_t(q);
    if(r>=q)r-=q;
    if(r>=q)r-=q;
    return uint32_t(r);
}

// grid: (256 slots, nin, 1); block: 256 threads.
// Encodes slot = g*16+b into [it][slot][limb][512][batch_count], skipping slot 128 (d=-128).
__global__ void batch_encode_q_diagonals_kernel(BatchEncodeQArgs args) {
    __shared__ union {
        double2 work[512];
        struct {
            uint32_t a[8][512];
        } ntt;
    } smem;
    const int tid=threadIdx.x;
    const int slot=blockIdx.x;
    if(slot==128)return;
    const int d=(slot<128)?slot:(slot-256);
    const int gs=(slot>>4)*16;
    const int it=blockIdx.y;
    const int B=args.batch_count;
    long long my_c[8][2];
    for(int b=0;b<B;++b) {
        const int ot=args.ots[b],lane=args.lanes[b];
        const int f=(tid-gs+256)%256,half=f/128,column=f%128,row=column+d;
        double v=0.;
        if(row>=0&&row<128) {
            const long long r=(long long)ot*256+((half+lane)%2)*128+column;
            const long long raw_c=(long long)it*256+half*128+row;
            const long long c=args.col_perm?static_cast<long long>(args.col_perm[raw_c]):raw_c;
            if(args.weights_bf16) {
                const uint32_t bits=(uint32_t(args.weights_bf16[r*args.columns+c]))<<16;
                v=static_cast<double>(__uint_as_float(bits));
            } else {
                v=args.weights[r*args.columns+c];
            }
        }
        smem.work[args.slot_map[tid]]=make_double2(v,0.);
        smem.work[args.slot_map[tid+256]]=make_double2(v,0.);
        __syncthreads();
        int root_base=1,gap=1;
        for(int m=256;m>1;m>>=1) {
            const int i=tid/gap,j=tid%gap,x=i*2*gap+j,y=x+gap;
            const double2 r=args.inverse_roots[root_base+i];
            const double2 u=smem.work[x],wv=smem.work[y];
            __syncthreads();
            smem.work[x]=cadd(u,wv);
            smem.work[y]=cmul(csub(u,wv),r);
            __syncthreads();
            root_base+=m;gap<<=1;
        }
        {
            const double2 u=smem.work[tid],wv=smem.work[tid+256];
            const double2 scaled_r=cscale(args.inverse_roots[root_base],args.fix);
            __syncthreads();
            smem.work[tid]=cscale(cadd(u,wv),args.fix);
            smem.work[tid+256]=cmul(csub(u,wv),scaled_r);
            __syncthreads();
        }
        auto extract_coeff=[&](int idx)->long long {
            const double value=round(smem.work[idx].x);
            if(!(fabs(value)<args.coefficient_limit)){atomicExch(args.overflow,1);return 0;}
            return (long long)value;
        };
        my_c[b][0]=extract_coeff(tid);
        my_c[b][1]=extract_coeff(tid+256);
        __syncthreads();
    }
    uint32_t *out_slot=args.encoded_q+(size_t(it)*256+size_t(slot))*size_t(args.nq)*512ULL*size_t(B);
    const size_t out_limb_stride=512ULL*size_t(B);
    const bool use_vec4=(B==8)&&((uintptr_t(out_slot)&15ULL)==0ULL);
    const int warp=tid>>5;
    const int lane=tid&31;
    for(int limb=0;limb<args.nq;++limb) {
        const uint32_t q=args.moduli[limb];
        const size_t table_offset=size_t(limb)*512;
        auto reduce_coeff=[&](long long v)->uint32_t {
            const uint64_t mag=v<0?uint64_t(0)-uint64_t(v):uint64_t(v);
            const uint32_t rem=uint32_t(mag%q);
            return v<0&&rem?q-rem:rem;
        };
        for(int b=0;b<B;++b) {
            smem.ntt.a[b][tid]=reduce_coeff(my_c[b][0]);
            smem.ntt.a[b][tid+256]=reduce_coeff(my_c[b][1]);
        }
        __syncthreads();
        if(warp<B) {
            uint32_t *a=smem.ntt.a[warp];
            for(int m=1,g=256;m<512;m*=2,g/=2) {
                #pragma unroll
                for(int step=0;step<8;++step) {
                    const int p=step*32+lane;
                    const int group=p/g,j=group*(2*g)+(p%g);
                    const size_t root_index=table_offset+m+group;
                    const uint32_t u=a[j];
                    const uint32_t v=mul_shoup(a[j+g],args.ntt_roots[root_index],args.ntt_shoup[root_index],q);
                    uint64_t sum=uint64_t(u)+v;if(sum>=q)sum-=q;
                    a[j]=uint32_t(sum);
                    a[j+g]=u>=v?u-v:q-(v-u);
                }
                __syncwarp();
            }
        }
        __syncthreads();
        uint32_t *out_limb=out_slot+size_t(limb)*out_limb_stride;
        if(use_vec4) {
            uint4 *dst4=reinterpret_cast<uint4*>(out_limb);
            #pragma unroll
            for(int k=0;k<4;++k) {
                const int i=k*256+tid;
                const int idx=i>>1;
                const int h=(i&1)<<2;
                const uint32_t perm=args.permutation[table_offset+idx];
                dst4[i]=make_uint4(
                    smem.ntt.a[h+0][perm],
                    smem.ntt.a[h+1][perm],
                    smem.ntt.a[h+2][perm],
                    smem.ntt.a[h+3][perm]);
            }
        } else {
            for(size_t i=size_t(tid);i<out_limb_stride;i+=256) {
                const int b=int(i%size_t(B));
                const int idx=int(i/size_t(B));
                out_limb[i]=smem.ntt.a[b][args.permutation[table_offset+idx]];
            }
        }
        __syncthreads();
    }
}

__device__ __forceinline__ void mac96(uint32_t &w0,uint32_t &w1,uint32_t &w2,uint32_t a,uint32_t b) {
    asm("mad.lo.cc.u32 %0, %3, %4, %0;\n\t"
        "madc.hi.cc.u32 %1, %3, %4, %1;\n\t"
        "addc.u32 %2, %2, 0;\n\t"
        : "+&r"(w0), "+&r"(w1), "+&r"(w2) : "r"(a), "r"(b));
}

__device__ __forceinline__ void mac96_vec4(
    uint32_t *w0,uint32_t *w1,uint32_t *w2,uint32_t baby,const uint4 v) {
    mac96(w0[0],w1[0],w2[0],baby,v.x);
    mac96(w0[1],w1[1],w2[1],baby,v.y);
    mac96(w0[2],w1[2],w2[2],baby,v.z);
    mac96(w0[3],w1[3],w2[3],baby,v.w);
}

template<int G_TILE,int B_FIXED,int BLOCK_THREADS>
__global__ __launch_bounds__(BLOCK_THREADS)
void batched_compact_mac_q_kernel(BatchMacQArgs args) {
    const int g0=int(blockIdx.x)*G_TILE;
    const int coeff_block=int(blockIdx.y);
    const int coeff=coeff_block*BLOCK_THREADS+int(threadIdx.x);
    const int l=int(blockIdx.z)%args.nq;
    const int c=int(blockIdx.z)/args.nq;
    const int B=(B_FIXED>0)?B_FIXED:args.batch_count;
    const uint32_t ridx=(BLOCK_THREADS<=128)
        ?((__brev(uint32_t(coeff_block)*uint32_t(BLOCK_THREADS))>>16)&511U)
        :((__brev(uint32_t(coeff))>>16)&511U);
    const uint32_t q=args.q_primes[l];
    const uint64_t ratio=args.q_barrett[l];
    constexpr int B_CAP=(B_FIXED>0)?B_FIXED:8;
    uint32_t w0[G_TILE][B_CAP],w1[G_TILE][B_CAP],w2[G_TILE][B_CAP];
    #pragma unroll
    for(int gt=0;gt<G_TILE;++gt)
        #pragma unroll
        for(int k=0;k<B_CAP;++k){w0[gt][k]=0;w1[gt][k]=0;w2[gt][k]=0;}
    const size_t nq=size_t(args.nq),nin=size_t(args.nin),Bs=size_t(B);
    const size_t baby_it_stride=16ULL*2*nq*65536ULL;
    const size_t baby_b_stride=2ULL*nq*65536ULL;
    const size_t baby_off=size_t(c)*nq*65536ULL+size_t(l)*65536ULL+size_t(coeff);
    const size_t pt_it_stride=256ULL*nq*512ULL*Bs;
    const size_t pt_slot_stride=nq*512ULL*Bs;
    const size_t pt_off=(size_t(l)*512ULL+size_t(ridx))*Bs;
    for(size_t it=0;it<nin;++it) {
        const uint32_t *baby_base=args.babies_q+it*baby_it_stride+baby_off;
        const uint32_t *pt_base_it=args.encoded_q+it*pt_it_stride+pt_off;
        for(int b=0;b<16;++b) {
            const uint32_t baby=baby_base[size_t(b)*baby_b_stride];
            #pragma unroll
            for(int gt=0;gt<G_TILE;++gt) {
                const int g=g0+gt;
                if(g==8&&b==0)continue;
                const uint32_t *pt_ptr=pt_base_it+size_t(g*16+b)*pt_slot_stride;
                if constexpr(B_FIXED==8) {
                    const uint4 *vptr=reinterpret_cast<const uint4*>(pt_ptr);
                    mac96_vec4(&w0[gt][0],&w1[gt][0],&w2[gt][0],baby,__ldg(vptr));
                    mac96_vec4(&w0[gt][4],&w1[gt][4],&w2[gt][4],baby,__ldg(vptr+1));
                } else if constexpr(B_FIXED==4) {
                    const uint4 v0=__ldg(reinterpret_cast<const uint4*>(pt_ptr));
                    mac96_vec4(&w0[gt][0],&w1[gt][0],&w2[gt][0],baby,v0);
                } else {
                    for(int k=0;k<B;++k)mac96(w0[gt][k],w1[gt][k],w2[gt][k],baby,pt_ptr[k]);
                }
            }
        }
    }
    const uint32_t two32=reduce_u64_mod(1ULL<<32,q,ratio);
    const uint32_t two64=reduce_u64_mod(uint64_t(two32)*uint64_t(two32),q,ratio);
    const size_t out_k_stride=16ULL*2*nq*65536ULL;
    const size_t out_g_stride=2ULL*nq*65536ULL;
    #pragma unroll
    for(int gt=0;gt<G_TILE;++gt) {
        const int g=g0+gt;
        if constexpr(B_FIXED>0) {
            #pragma unroll
            for(int k=0;k<B_FIXED;++k) {
                const uint64_t lo=(uint64_t(w1[gt][k])<<32)|uint64_t(w0[gt][k]);
                const uint32_t r0=reduce_u64_mod(lo,q,ratio);
                const uint32_t r1=reduce_u64_mod(uint64_t(w2[gt][k])*uint64_t(two64),q,ratio);
                uint64_t res=uint64_t(r0)+uint64_t(r1);if(res>=q)res-=q;
                args.out_q_batch[size_t(k)*out_k_stride+size_t(g)*out_g_stride+baby_off]=uint32_t(res);
            }
        } else {
            for(int k=0;k<B;++k) {
                const uint64_t lo=(uint64_t(w1[gt][k])<<32)|uint64_t(w0[gt][k]);
                const uint32_t r0=reduce_u64_mod(lo,q,ratio);
                const uint32_t r1=reduce_u64_mod(uint64_t(w2[gt][k])*uint64_t(two64),q,ratio);
                uint64_t res=uint64_t(r0)+uint64_t(r1);if(res>=q)res-=q;
                args.out_q_batch[size_t(k)*out_k_stride+size_t(g)*out_g_stride+baby_off]=uint32_t(res);
            }
        }
    }
}
} // namespace

cudaError_t launch_batch_encode_q_diagonals(const BatchEncodeQArgs &args,cudaStream_t stream) {
    if((!args.weights&&!args.weights_bf16)||!args.slot_map||!args.inverse_roots||!args.moduli||!args.ntt_roots||
       !args.ntt_shoup||!args.permutation||!args.encoded_q||!args.overflow||
       args.nq<=0||args.nin<=0||args.batch_count<=0||args.batch_count>8)
        return cudaErrorInvalidValue;
    batch_encode_q_diagonals_kernel<<<dim3(256,args.nin),256,0,stream>>>(args);
    return cudaGetLastError();
}
cudaError_t launch_batched_compact_mac_q(const BatchMacQArgs &args,cudaStream_t stream) {
    if(!args.babies_q||!args.encoded_q||!args.out_q_batch||!args.q_primes||!args.q_barrett||
       args.nq<=0||args.nin<=0||args.batch_count<=0||args.batch_count>8)
        return cudaErrorInvalidValue;
    int gtile=1;
    if(const char *gt_env=std::getenv("LLAMA_FAST_MAC_GTILE")) {
        if(gt_env[0]=='2'&&gt_env[1]=='\0')gtile=2;
    }
    const char *bt_env=std::getenv("LLAMA_FAST_MAC_THREADS");
    const int bthreads=(bt_env&&bt_env[0]=='1'&&bt_env[1]=='2'&&bt_env[2]=='8'&&bt_env[3]=='\0')?128:256;
    const bool aligned16=(uintptr_t(args.encoded_q)&15ULL)==0ULL;
    auto launch=[&](auto bt_tag,auto gt_tag){
        constexpr int BT=decltype(bt_tag)::value,GT=decltype(gt_tag)::value;
        const dim3 grid(16/GT,65536/BT,2*args.nq);
        if(aligned16&&args.batch_count==8)batched_compact_mac_q_kernel<GT,8,BT><<<grid,BT,0,stream>>>(args);
        else if(aligned16&&args.batch_count==4)batched_compact_mac_q_kernel<GT,4,BT><<<grid,BT,0,stream>>>(args);
        else batched_compact_mac_q_kernel<GT,0,BT><<<grid,BT,0,stream>>>(args);
    };
    if(bthreads==128) {
        if(gtile==2)launch(std::integral_constant<int,128>{},std::integral_constant<int,2>{});
        else launch(std::integral_constant<int,128>{},std::integral_constant<int,1>{});
    } else {
        if(gtile==2)launch(std::integral_constant<int,256>{},std::integral_constant<int,2>{});
        else launch(std::integral_constant<int,256>{},std::integral_constant<int,1>{});
    }
    return cudaGetLastError();
}
} // namespace fast_linear
