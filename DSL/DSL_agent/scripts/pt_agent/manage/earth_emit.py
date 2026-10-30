"""Generic Earth MLIR, without importing or executing the Hecate frontend."""

import ctypes
from dataclasses import replace
import math
from pathlib import Path
import struct
import time

import numpy as np

from ..backends.base import Backend
from .check import check
from .greedy import timed_stage, write_timings
from .ir import Function, Node, Type, shared_type
from .streams import output_path, text_writer


def type_text(t):
    """include/hecate/Dialect/Earth/IR/EarthOpsTypes.td: ci/pl assembly."""
    return f'tensor<1x!earth.{"ci" if t.cipher else "pl"}<{t.scale} * {t.level}>>'


class RMSCalculator:
    """EarthOps.td:145–157, S1 §8: platform-selected sequential accumulation."""

    def __init__(self, accumulation='fma'):
        if accumulation not in ('fma', 'separate'):
            raise ValueError('rms_accumulation must be fma or separate')
        self.accumulation = accumulation
        self.fma = None
        if accumulation == 'fma':
            try:
                self.library = ctypes.CDLL('libm.so.6')
                self.fma = self.library.fma
                self.fma.argtypes = [ctypes.c_double]*3
                self.fma.restype = ctypes.c_double
            except (OSError, AttributeError) as exc:
                raise RuntimeError('rms_accumulation=fma requires system libm fma') from exc
        self.cache = {}
        self.hits = self.misses = 0

    def __call__(self, data):
        """EarthOps.td:147–152: array order, then division and correctly rounded sqrt."""
        data = np.asarray(data, dtype='<f8').reshape(-1)
        if not len(data):
            raise ValueError('RMS requires a nonempty constant')
        # S1 §8 explicitly permits value-based reuse. This key stores the payload
        # itself; it performs no file-integrity or digest verification.
        key = data.tobytes()
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        total = 0.0
        if self.accumulation == 'fma':
            for value in data:
                d = float(value)
                total = self.fma(d, d, total)
        else:
            for value in data:
                d = float(value)
                total += d*d
        value = math.sqrt(total/len(data))
        self.cache[key] = value
        return value


def rms(data, accumulation='fma'):
    """EarthOps.td:145–157; the caller chooses the tested build's accumulation."""
    return RMSCalculator(accumulation)(data)


def float_text(value):
    """MLIR f64 attributes use the APFloat bit pattern, preserving every bit."""
    return '0x'+struct.pack('>d', float(value)).hex().upper()


def constant_table(function):
    """Carry Backend.const names through private encodings and management."""
    def constants():
        return (n for n in function.nodes if n.kind=='constant')
    if any(n.const_key is None for n in constants()):
        if any(n.const_key is not None for n in constants()):
            raise ValueError('Mixed named and unnamed Earth constant payloads')
        # Legacy full-text readers have no NameLoc key. Reuse the existing
        # Backend.const identity once when emitting those legacy graphs.
        backend = Backend(len(next(constants()).data))
        backend._keep_const_data = False
        for n in constants():
            n.const_key = backend.const(n.data).name
    table = {}
    for n in constants():
        table.setdefault(n.const_key,n.data)
    return table


def constant_sizes(function, calculator=None):
    """U4 §7.6.3: estimate private encodings before writing managed IR."""
    table = constant_table(function)
    calculator = calculator or RMSCalculator(function.profile['rms_accumulation'])
    combinations = set()
    count = 0
    for n in (n for n in function.nodes if n.kind=='constant'):
        count += 1
        value = n.attrs['rms_var'] if 'rms_var' in n.attrs else calculator(n.data)
        combinations.add((n.const_key,n.type,struct.pack('>d',value)))
    slots = len(next(iter(table.values()))) if table else 0
    return dict(constant_ops=count,distinct_payloads=len(table),
                encoding_combinations=len(combinations),estimated_cst_bytes=len(combinations)*slots*8)


def _emit(function, path, *, metrics=None, unmanaged=False, compressed=False):
    """tools/frontend.cpp:147–150 and compiler debug-printer generic syntax."""
    started = time.monotonic()
    if not unmanaged:
        check(function)
    path = Path(path)
    types = function.types()
    calculator = RMSCalculator(function.profile['rms_accumulation'])
    payloads = constant_table(function)
    sizes = constant_sizes(function,calculator)
    if metrics is not None:
        metrics.update(sizes)
    if not unmanaged and sizes['estimated_cst_bytes']>8*1024**3:
        raise ValueError('Estimated .cst exceeds 8 GiB: '+str(sizes))

    def value(a):
        return f'%arg{-a-1}' if a < 0 else f'%v{a}'

    # S1 §11.3: structural locations never inherit a file name or <stdin>.
    inputs = ', '.join(f'%arg{i}: {type_text(t)} loc(unknown)' for i, t in enumerate(function.inputs))
    results = ', '.join(type_text(t) for t in function.result_types)
    a = function.attrs
    attrs = ', '.join((f'arg_scale = array<i64: {", ".join(map(str, a["arg_scale"]))}>',
                       f'init_level = {a["init_level"]} : i64',
                       f'res_scale = array<i64: {", ".join(map(str, a["res_scale"]))}>')) if a else ''
    suffix = f' attributes {{{attrs}}}' if attrs else ''
    header = f'module {{\n  func.func @_hecate_{function.name}({inputs}) -> ({results}){suffix} {{\n'
    footer = ('    return '+', '.join(value(v) for v in function.outputs)+' : '+results+
              ' loc(unknown)\n  } loc(unknown)\n} loc(unknown)\n')

    def alias(key,data, *, measure=False):
        payload = '' if measure else np.asarray(data,dtype='<f8').tobytes().hex().upper()
        return f'#pt{key} = dense<"0x{payload}"> : tensor<{len(data)}xf64>\n'

    def line(n, *, measure=False):
        args = ', '.join(value(v) for v in n.args)
        props = ''
        if n.kind == 'constant':
            data = np.asarray(n.data, dtype='<f8')
            rms_value = 0 if measure else n.attrs['rms_var'] if 'rms_var' in n.attrs else calculator(data)
            attr = (f' {{rms_var = {float_text(rms_value)} : f64, '
                    f'value = #pt{n.const_key}}}')
        else:
            attr = ''
            if n.attrs:
                entries = []
                for key, v in n.attrs.items():
                    entries.append(f'{key} = array<i64: {", ".join(map(str, v))}>' if key == 'offset'
                                   else f'{key} = {v} : i64')
                props = ' <{'+', '.join(entries)+'}>'
        operands = ', '.join(type_text(types[v]) for v in n.args)
        # PT-DSL ownership is diagnostic location data, not an operation attribute.
        line = max(1, n.lineno)
        loc = f' loc("ptdsl-op-{n.op_id}":{line}:0)'
        if n.kind=='constant':
            loc = f' loc("ptdsl-const-{n.const_key}"("ptdsl-op-{n.op_id}":{line}:0))'
        return (f'    %v{n.id} = "earth.{n.kind}"({args}){props}{attr} : '
                f'({operands}) -> {type_text(n.type)}{loc}\n')

    # The dense payload has exactly 16 ASCII hex characters per float64. Size
    # this text without materializing it or calculating RMS a second time.
    size = (len(header)+len(footer)+sum(len(line(n,measure=True)) for n in function.nodes)+
            sum(len(alias(key,data,measure=True))+len(data)*16 for key,data in payloads.items()))
    path = output_path(path,size,compressed=compressed)
    with text_writer(path) as stream:
        for key,data in payloads.items():
            stream.write(alias(key,data))
        stream.write(header)
        for n in function.nodes:
            stream.write(line(n))
        stream.write(footer)
    if metrics is not None:
        metrics.update(seconds=time.monotonic()-started, rms_accumulation=calculator.accumulation,
                       rms_cache_hits=calculator.hits, rms_cache_misses=calculator.misses,
                       uncompressed_bytes=size, path=str(path))
    return path


def emit(function, path, *, metrics=None, compressed=False):
    try:
        with timed_stage("managed_emit", nodes=len(function.nodes)):
            return _emit(function, path, metrics=metrics, compressed=compressed)
    finally:
        write_timings(Path(path).parent/"trace-stages.json")


def emit_raw(recorded, path, profile, *, name='golden', metrics=None, compressed=False):
    """frontend.cpp:84–106,165–249: builder types precede save() preprocessing."""
    allowed = {'constant','add','sub','mul','negate','rotate'}
    nodes = []
    for n in recorded.nodes:
        if n.kind not in allowed:
            raise ValueError('unsupported unmanaged Earth operation '+n.kind)
        nodes.append(replace(n,args=list(n.args),attrs=dict(n.attrs),type=shared_type(n.kind!='constant',0,0)))
    inputs = [shared_type(True,0,0) for _ in recorded._inputs]
    outputs = [v.nid for v in recorded._outputs]
    f = Function(name,inputs,nodes,outputs,[shared_type(True,0,0) for _ in outputs],{},profile)
    return _emit(f,path,metrics=metrics,unmanaged=True,compressed=compressed)


def read(path, profile, *, locations=None):
    """Compatibility entry; one parser handles stage 0 and managed Earth."""
    from .mlir_read import read as parse_module
    return parse_module(path, profile, locations=locations)
