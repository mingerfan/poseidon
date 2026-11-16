/*
 * LD_PRELOAD regression guard for --help, no arguments, and --check-only.
 *
 * No CUDA headers or libraries are needed by this DSO. Every intercepted entry
 * increments the counter, reports to stderr, and terminates with status 99
 * BEFORE forwarding anything. _Exit avoids core dumps and CUDA teardown calls.
 * Exported stubs deliberately ignore the ABI arguments and never return; they
 * are not implementations of CUDA and must never be used for real GPU work.
 *
 * __cudaRegisterFatBinary[End], __cudaUnregisterFatBinary, __cudaRegisterFunction,
 * __cudaRegisterVar, etc. are NOT replaced: static host-side fatbinary
 * registration is allowed to proceed in the real runtime. Device/runtime API
 * queries (even error/version queries) are conservatively forbidden below.
 * This is an ELF symbol-interposition regression tripwire, not a security
 * sandbox: statically linked, hidden, or explicitly handle-resolved symbols can
 * bypass LD_PRELOAD. Audit imported device symbols when dependencies change.
 *
 * Successful normal exit emits:
 *   BERT_NO_DEVICE_GUARD finished device_calls=0
 * A violation emits the offending API plus device_calls=1 and exits 99.
 * Both a loaded and a finished marker should be required by CPU gate tests.
 */
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define EXPORTED __attribute__((visibility("default")))
static _Atomic unsigned int device_calls;

EXPORTED unsigned int bert_no_device_guard_device_calls(void)
{
    return atomic_load_explicit(&device_calls, memory_order_relaxed);
}

static void report(const char *state, const char *api, unsigned int count)
{
    char line[512];
    int length = snprintf(line, sizeof(line),
                          "BERT_NO_DEVICE_GUARD %s%s%s device_calls=%u\n",
                          state, api ? " api=" : "", api ? api : "", count);
    if (length > 0) {
        size_t remaining = (size_t)length;
        if (remaining >= sizeof(line))
            remaining = sizeof(line) - 1;
        const char *cursor = line;
        while (remaining) {
            ssize_t written = write(STDERR_FILENO, cursor, remaining);
            if (written <= 0)
                break;
            cursor += written;
            remaining -= (size_t)written;
        }
    }
}

__attribute__((constructor)) static void guard_loaded(void)
{
    report("loaded", NULL, bert_no_device_guard_device_calls());
}

__attribute__((destructor)) static void guard_finished(void)
{
    report("finished", NULL, bert_no_device_guard_device_calls());
}

static _Noreturn void blocked(const char *api)
{
    unsigned int count = atomic_fetch_add_explicit(&device_calls, 1,
                                                  memory_order_relaxed) + 1;
    report("blocked", api, count);
    _Exit(99);
}

#define GUARD(name) EXPORTED _Noreturn void name(void) { blocked(#name); }
/* Include both per-thread stream ABI spellings, even if this toolkit only
 * exports one of them. Extra exported trap names have no runtime dependency. */
#define RUNTIME_STREAM(name) GUARD(name) GUARD(name##_ptsz) GUARD(name##_ptds)
#define DRIVER(name) RUNTIME_STREAM(name) RUNTIME_STREAM(name##_v2) RUNTIME_STREAM(name##_v3)

/* Runtime initialization, device selection/properties, and diagnostics. */
GUARD(cudaInitDevice)
GUARD(cudaSetDevice)
GUARD(cudaGetDevice)
GUARD(cudaGetDeviceCount)
GUARD(cudaGetDeviceProperties)
GUARD(cudaGetDeviceProperties_v2)
GUARD(cudaDeviceGetAttribute)
GUARD(cudaDeviceGetByPCIBusId)
GUARD(cudaDeviceGetPCIBusId)
GUARD(cudaChooseDevice)
GUARD(cudaSetValidDevices)
GUARD(cudaSetDeviceFlags)
GUARD(cudaGetDeviceFlags)
GUARD(cudaDeviceReset)
GUARD(cudaDeviceSynchronize)
GUARD(cudaDeviceSetLimit)
GUARD(cudaDeviceGetLimit)
GUARD(cudaDeviceSetCacheConfig)
GUARD(cudaDeviceGetCacheConfig)
GUARD(cudaDeviceSetSharedMemConfig)
GUARD(cudaDeviceGetSharedMemConfig)
GUARD(cudaDeviceGetStreamPriorityRange)
GUARD(cudaDeviceCanAccessPeer)
GUARD(cudaDeviceEnablePeerAccess)
GUARD(cudaDeviceDisablePeerAccess)
GUARD(cudaDeviceGetP2PAttribute)
GUARD(cudaGetLastError)
GUARD(cudaPeekAtLastError)
GUARD(cudaGetErrorName)
GUARD(cudaGetErrorString)
GUARD(cudaRuntimeGetVersion)
GUARD(cudaDriverGetVersion)
GUARD(cudaGetDriverEntryPoint)
GUARD(cudaGetDriverEntryPointByVersion)
GUARD(cudaGetExportTable)
GUARD(cudaProfilerStart)
GUARD(cudaProfilerStop)

/* Runtime allocation, transfers, memory pools, and host-pinned memory. */
GUARD(cudaMalloc)
GUARD(cudaMallocPitch)
GUARD(cudaMalloc3D)
GUARD(cudaMallocArray)
GUARD(cudaMalloc3DArray)
GUARD(cudaMallocMipmappedArray)
GUARD(cudaMallocManaged)
GUARD(cudaMallocHost)
GUARD(cudaHostAlloc)
GUARD(cudaHostRegister)
GUARD(cudaHostUnregister)
GUARD(cudaHostGetDevicePointer)
GUARD(cudaHostGetFlags)
GUARD(cudaFree)
GUARD(cudaFreeHost)
GUARD(cudaFreeArray)
GUARD(cudaFreeMipmappedArray)
GUARD(cudaMemGetInfo)
GUARD(cudaPointerGetAttributes)
GUARD(cudaMemAdvise)
RUNTIME_STREAM(cudaMemPrefetchAsync)
RUNTIME_STREAM(cudaMemRangeGetAttribute)
RUNTIME_STREAM(cudaMemRangeGetAttributes)
RUNTIME_STREAM(cudaMemcpy)
RUNTIME_STREAM(cudaMemcpyAsync)
RUNTIME_STREAM(cudaMemcpy2D)
RUNTIME_STREAM(cudaMemcpy2DAsync)
RUNTIME_STREAM(cudaMemcpy3D)
RUNTIME_STREAM(cudaMemcpy3DAsync)
RUNTIME_STREAM(cudaMemcpyPeer)
RUNTIME_STREAM(cudaMemcpyPeerAsync)
RUNTIME_STREAM(cudaMemcpy3DPeer)
RUNTIME_STREAM(cudaMemcpy3DPeerAsync)
RUNTIME_STREAM(cudaMemcpyToSymbol)
RUNTIME_STREAM(cudaMemcpyToSymbolAsync)
RUNTIME_STREAM(cudaMemcpyFromSymbol)
RUNTIME_STREAM(cudaMemcpyFromSymbolAsync)
RUNTIME_STREAM(cudaMemcpyToArray)
RUNTIME_STREAM(cudaMemcpyFromArray)
RUNTIME_STREAM(cudaMemset)
RUNTIME_STREAM(cudaMemsetAsync)
RUNTIME_STREAM(cudaMemset2D)
RUNTIME_STREAM(cudaMemset2DAsync)
RUNTIME_STREAM(cudaMemset3D)
RUNTIME_STREAM(cudaMemset3DAsync)
RUNTIME_STREAM(cudaMallocAsync)
RUNTIME_STREAM(cudaMallocFromPoolAsync)
RUNTIME_STREAM(cudaFreeAsync)
GUARD(cudaDeviceGetDefaultMemPool)
GUARD(cudaDeviceGetMemPool)
GUARD(cudaDeviceSetMemPool)
GUARD(cudaMemPoolCreate)
GUARD(cudaMemPoolDestroy)
GUARD(cudaMemPoolTrimTo)
GUARD(cudaMemPoolSetAttribute)
GUARD(cudaMemPoolGetAttribute)
GUARD(cudaMemPoolSetAccess)
GUARD(cudaMemPoolGetAccess)
GUARD(cudaMemPoolExportToShareableHandle)
GUARD(cudaMemPoolImportFromShareableHandle)
GUARD(cudaMemPoolExportPointer)
GUARD(cudaMemPoolImportPointer)
GUARD(cudaIpcGetMemHandle)
GUARD(cudaIpcOpenMemHandle)
GUARD(cudaIpcCloseMemHandle)

/* Runtime streams, events, launches, graphs, and kernel lookup. */
GUARD(cudaStreamCreate)
GUARD(cudaStreamCreateWithFlags)
GUARD(cudaStreamCreateWithPriority)
GUARD(cudaStreamDestroy)
RUNTIME_STREAM(cudaStreamSynchronize)
RUNTIME_STREAM(cudaStreamQuery)
RUNTIME_STREAM(cudaStreamWaitEvent)
RUNTIME_STREAM(cudaStreamGetFlags)
RUNTIME_STREAM(cudaStreamGetPriority)
RUNTIME_STREAM(cudaStreamGetDevice)
RUNTIME_STREAM(cudaStreamGetId)
RUNTIME_STREAM(cudaStreamAddCallback)
RUNTIME_STREAM(cudaStreamBeginCapture)
RUNTIME_STREAM(cudaStreamEndCapture)
RUNTIME_STREAM(cudaStreamIsCapturing)
RUNTIME_STREAM(cudaStreamGetCaptureInfo)
GUARD(cudaEventCreate)
GUARD(cudaEventCreateWithFlags)
GUARD(cudaEventDestroy)
RUNTIME_STREAM(cudaEventRecord)
RUNTIME_STREAM(cudaEventRecordWithFlags)
GUARD(cudaEventQuery)
GUARD(cudaEventSynchronize)
GUARD(cudaEventElapsedTime)
GUARD(cudaIpcGetEventHandle)
GUARD(cudaIpcOpenEventHandle)
RUNTIME_STREAM(cudaLaunch)
RUNTIME_STREAM(cudaLaunchKernel)
RUNTIME_STREAM(cudaLaunchKernelExC)
RUNTIME_STREAM(cudaLaunchCooperativeKernel)
RUNTIME_STREAM(cudaLaunchCooperativeKernelMultiDevice)
RUNTIME_STREAM(cudaLaunchHostFunc)
GUARD(cudaConfigureCall)
GUARD(cudaSetupArgument)
GUARD(cudaFuncGetAttributes)
GUARD(cudaFuncSetAttribute)
GUARD(cudaFuncSetCacheConfig)
GUARD(cudaFuncSetSharedMemConfig)
GUARD(cudaGetSymbolAddress)
GUARD(cudaGetSymbolSize)
GUARD(cudaOccupancyMaxActiveBlocksPerMultiprocessor)
GUARD(cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags)
GUARD(cudaOccupancyAvailableDynamicSMemPerBlock)
GUARD(cudaGraphCreate)
GUARD(cudaGraphDestroy)
GUARD(cudaGraphInstantiate)
GUARD(cudaGraphInstantiateWithFlags)
GUARD(cudaGraphExecDestroy)
RUNTIME_STREAM(cudaGraphUpload)
RUNTIME_STREAM(cudaGraphLaunch)
GUARD(cudaGraphExecUpdate)
GUARD(cudaCreateTextureObject)
GUARD(cudaDestroyTextureObject)
GUARD(cudaCreateSurfaceObject)
GUARD(cudaDestroySurfaceObject)
GUARD(cudaImportExternalMemory)
GUARD(cudaDestroyExternalMemory)
GUARD(cudaImportExternalSemaphore)
GUARD(cudaDestroyExternalSemaphore)
RUNTIME_STREAM(cudaSignalExternalSemaphoresAsync)
RUNTIME_STREAM(cudaWaitExternalSemaphoresAsync)
GUARD(__cudaGetKernel)
RUNTIME_STREAM(__cudaLaunchKernel)
RUNTIME_STREAM(__cudaLaunchKernelExC)
GUARD(__cudaPushCallConfiguration)
GUARD(__cudaPopCallConfiguration)

/* cuFFT entry points directly referenced by the imported mature archives.
 * Planning can initialize a device internally, so reject it as well as exec. */
GUARD(cufftCreate)
GUARD(cufftDestroy)
GUARD(cufftExecZ2Z)
GUARD(cufftMakePlan1d)
GUARD(cufftSetAutoAllocation)
GUARD(cufftSetStream)
GUARD(cufftSetWorkArea)

/* Driver APIs as well as dispatch lookup, to catch direct/indirect bypasses
 * used by allocators. Loading the driver or querying devices is forbidden. */
DRIVER(cuInit)
DRIVER(cuDriverGetVersion)
DRIVER(cuGetProcAddress)
DRIVER(cuGetExportTable)
DRIVER(cuGetErrorName)
DRIVER(cuGetErrorString)
DRIVER(cuDeviceGet)
DRIVER(cuDeviceGetCount)
DRIVER(cuDeviceGetName)
DRIVER(cuDeviceGetAttribute)
DRIVER(cuDeviceTotalMem)
DRIVER(cuDevicePrimaryCtxRetain)
DRIVER(cuDevicePrimaryCtxRelease)
DRIVER(cuDevicePrimaryCtxReset)
DRIVER(cuDevicePrimaryCtxSetFlags)
DRIVER(cuDevicePrimaryCtxGetState)
DRIVER(cuCtxCreate)
DRIVER(cuCtxDestroy)
DRIVER(cuCtxSetCurrent)
DRIVER(cuCtxGetCurrent)
DRIVER(cuCtxPushCurrent)
DRIVER(cuCtxPopCurrent)
DRIVER(cuCtxGetDevice)
DRIVER(cuCtxSynchronize)
DRIVER(cuMemGetInfo)
DRIVER(cuMemAlloc)
DRIVER(cuMemAllocPitch)
DRIVER(cuMemAllocManaged)
DRIVER(cuMemAllocAsync)
DRIVER(cuMemAllocFromPoolAsync)
DRIVER(cuMemFree)
DRIVER(cuMemFreeAsync)
DRIVER(cuMemAllocHost)
DRIVER(cuMemHostAlloc)
DRIVER(cuMemFreeHost)
DRIVER(cuMemHostRegister)
DRIVER(cuMemHostUnregister)
DRIVER(cuMemHostGetDevicePointer)
DRIVER(cuMemCreate)
DRIVER(cuMemRelease)
DRIVER(cuMemAddressReserve)
DRIVER(cuMemAddressFree)
DRIVER(cuMemMap)
DRIVER(cuMemUnmap)
DRIVER(cuMemSetAccess)
DRIVER(cuMemGetAccess)
DRIVER(cuMemPoolCreate)
DRIVER(cuMemPoolDestroy)
DRIVER(cuMemPoolTrimTo)
DRIVER(cuMemPoolGetAttribute)
DRIVER(cuMemPoolSetAttribute)
DRIVER(cuMemcpy)
DRIVER(cuMemcpyAsync)
DRIVER(cuMemcpyHtoD)
DRIVER(cuMemcpyHtoDAsync)
DRIVER(cuMemcpyDtoH)
DRIVER(cuMemcpyDtoHAsync)
DRIVER(cuMemcpyDtoD)
DRIVER(cuMemcpyDtoDAsync)
DRIVER(cuMemcpyPeer)
DRIVER(cuMemcpyPeerAsync)
DRIVER(cuMemsetD8)
DRIVER(cuMemsetD8Async)
DRIVER(cuMemsetD32)
DRIVER(cuMemsetD32Async)
DRIVER(cuStreamCreate)
DRIVER(cuStreamCreateWithPriority)
DRIVER(cuStreamDestroy)
DRIVER(cuStreamSynchronize)
DRIVER(cuStreamQuery)
DRIVER(cuStreamWaitEvent)
DRIVER(cuEventCreate)
DRIVER(cuEventDestroy)
DRIVER(cuEventRecord)
DRIVER(cuEventSynchronize)
DRIVER(cuEventElapsedTime)
DRIVER(cuModuleLoad)
DRIVER(cuModuleLoadData)
DRIVER(cuModuleLoadDataEx)
DRIVER(cuModuleUnload)
DRIVER(cuModuleGetFunction)
DRIVER(cuLibraryLoadData)
DRIVER(cuLibraryLoadFromFile)
DRIVER(cuLibraryUnload)
DRIVER(cuLibraryGetKernel)
DRIVER(cuLaunchKernel)
DRIVER(cuLaunchKernelEx)
DRIVER(cuLaunchCooperativeKernel)
DRIVER(cuLaunchHostFunc)
DRIVER(cuGraphLaunch)
DRIVER(cuGraphUpload)
