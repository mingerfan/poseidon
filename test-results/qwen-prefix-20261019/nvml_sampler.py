"""Persistent NVML observer; ABI checked against the server CUDA 12.2 nvml.h."""
import ctypes as c
import time


class Utilization(c.Structure):
    _fields_ = [('gpu', c.c_uint), ('memory', c.c_uint)]


class Memory(c.Structure):
    _fields_ = [('total', c.c_ulonglong), ('free', c.c_ulonglong), ('used', c.c_ulonglong)]


class ProcessV2(c.Structure):
    _fields_ = [('pid', c.c_uint), ('usedGpuMemory', c.c_ulonglong),
                ('gpuInstanceId', c.c_uint), ('computeInstanceId', c.c_uint)]


class Observer:
    def __init__(self, indices=(0, 1, 2, 3)):
        self.lib = c.CDLL('libnvidia-ml.so.1')
        self.lib.nvmlErrorString.argtypes = [c.c_int]
        self.lib.nvmlErrorString.restype = c.c_char_p
        signatures = {
            'nvmlInit_v2': [], 'nvmlShutdown': [],
            'nvmlDeviceGetHandleByIndex_v2': [c.c_uint, c.POINTER(c.c_void_p)],
            'nvmlDeviceGetUUID': [c.c_void_p, c.c_char_p, c.c_uint],
            'nvmlDeviceGetUtilizationRates': [c.c_void_p, c.POINTER(Utilization)],
            'nvmlDeviceGetMemoryInfo': [c.c_void_p, c.POINTER(Memory)],
            'nvmlDeviceGetPowerUsage': [c.c_void_p, c.POINTER(c.c_uint)],
            'nvmlDeviceGetClockInfo': [c.c_void_p, c.c_uint, c.POINTER(c.c_uint)],
            'nvmlDeviceGetComputeRunningProcesses_v2': [c.c_void_p, c.POINTER(c.c_uint), c.POINTER(ProcessV2)],
        }
        for name, arguments in signatures.items():
            function = getattr(self.lib, name)
            function.argtypes, function.restype = arguments, c.c_int
        self.call('nvmlInit_v2')
        self.devices = []
        for index in indices:
            handle = c.c_void_p()
            self.call('nvmlDeviceGetHandleByIndex_v2', index, c.byref(handle))
            uuid = c.create_string_buffer(96)
            self.call('nvmlDeviceGetUUID', handle, uuid, len(uuid))
            self.devices.append((index, handle, uuid.value.decode()))

    def call(self, name, *arguments):
        result = getattr(self.lib, name)(*arguments)
        if result:
            raise RuntimeError(f'{name}: NVML {result}: {self.lib.nvmlErrorString(result).decode()}')

    def applications(self):
        result = []
        for _, handle, uuid in self.devices:
            count = c.c_uint(128)
            processes = (ProcessV2 * count.value)()
            self.call('nvmlDeviceGetComputeRunningProcesses_v2', handle, c.byref(count), processes)
            result.extend(dict(uuid=uuid, pid=p.pid, memory_mib=p.usedGpuMemory // 2**20)
                          for p in processes[:count.value])
        return result

    def sample(self):
        begin = time.monotonic()
        devices = []
        for index, handle, uuid in self.devices:
            util, memory = Utilization(), Memory()
            power, clock = c.c_uint(), c.c_uint()
            self.call('nvmlDeviceGetUtilizationRates', handle, c.byref(util))
            self.call('nvmlDeviceGetMemoryInfo', handle, c.byref(memory))
            self.call('nvmlDeviceGetPowerUsage', handle, c.byref(power))
            self.call('nvmlDeviceGetClockInfo', handle, 1, c.byref(clock))
            devices.append(dict(index=index, uuid=uuid, gpu_util=util.gpu, memory_util=util.memory,
                memory_mib=memory.used // 2**20, power_w=power.value / 1000, clock_mhz=clock.value))
        apps = self.applications()
        return dict(monotonic=time.monotonic(), query_seconds=time.monotonic()-begin,
                    gpu=devices, applications=apps)

    def close(self):
        self.call('nvmlShutdown')
