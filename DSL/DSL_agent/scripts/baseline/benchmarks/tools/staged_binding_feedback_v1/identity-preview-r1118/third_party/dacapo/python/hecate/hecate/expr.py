import ctypes
import weakref
import re
import inspect
from subprocess import Popen
from collections.abc import Iterable
import torch

import os
import time
import numpy as np
import numpy.ctypeslib as npcl

hecate_dir = os.environ["HECATE"]
hecateBuild = hecate_dir+"/build"
heaan_keyset = "/heaan_keyset"
libpath = hecateBuild + "/lib/"
lt = ctypes.CDLL(libpath+"libHecateFrontend.so")
os.environ['PATH'] = libpath + os.pathsep + os.environ['PATH']





"""Object Creation"""
lt.createConstant.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double), ctypes.c_size_t, ctypes.c_char_p,
        ctypes.c_size_t
        ]
lt.createFunc.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, 
        ctypes.POINTER(ctypes.c_int), ctypes.c_size_t, ctypes.c_char_p,
        ctypes.c_size_t
        ]
lt.initFunc.argtypes = [
        ctypes.c_void_p, ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t), ctypes.c_size_t
        ]
lt.createConstant.restype = ctypes.c_size_t
lt.createConstant.restype = ctypes.c_size_t
lt.createFunc.restype = ctypes.c_size_t
lt.createCall.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t), ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t), ctypes.c_size_t,
        ctypes.c_char_p, ctypes.c_size_t]
lt.createCall.restype = ctypes.c_size_t

"""Immediate arguments"""
lt.createRotation.argtypes = [
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int64, ctypes.c_char_p, ctypes.c_size_t
        ]
lt.createRotation.restype = ctypes.c_size_t


"""Unary Operation"""
lt.createUnary.argtypes = [
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_char_p,
        ctypes.c_size_t
        ]
lt.createUnary.restype = ctypes.c_size_t
"""Binary Operation"""
lt.createBinary.argtypes = [
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
        ctypes.c_char_p, ctypes.c_size_t
        ]
lt.createBinary.restype = ctypes.c_size_t
"""Return"""
lt.setOutput.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t), 
        ctypes.c_size_t]
"""compile"""
lt.save.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
lt.save.restype = ctypes.c_char_p

lt.init.restype = ctypes.c_void_p
lt.finalize.argtypes = [ctypes.c_void_p]
"""Context Generation"""
ctxt = lt.init()
import sys
# weakref.finalize(sys.modules[__name__], lt.finalize, ctxt)

"""OpCode Tables"""
toUnary = {
        "bootstrap": 0,  # Bootstrap
        }
toBinary = {
        "add": 6,  # Addition
        "sub": 7,  # Subtraction
        "mul": 8,  # Multiplication
        }
toInnerUnary = {
        "neg": 13,  # Negation
        }



def save(dirs="", cst_dirs=""):
    (frame, filename, line_number, function_name, lines,
            index) = getProperFrame()

    start = time.perf_counter()
    proc_start = time.process_time()
    if dirs=="" :
        dirs = os.getcwd()
    if cst_dirs=="" :
        cst_dirs = os.getcwd()
    
    [func.eval() for func in funcList]
    name = filename.split("/")[-1].split(".")[0].encode('utf-8')
    name = (dirs +"/"+ filename.split("/")[-1].split(".")[0]+ ".mlir" ).encode('utf-8') 
    cst_name = (cst_dirs).encode('utf-8') 
    name = lt.save(ctxt, cst_name, name ).decode('utf-8')


    return name 

"""Set UnaryOperators"""


def unaryFactory(name, opcode):
    def unaryMethod(self):
        (frame, filename, line_number, function_name, lines,
                index) = inspect.stack()[1]
        def apply(value):
            return Expr(lt.createUnary(ctxt, opcode, value.obj,
                filename.encode('utf-8'), line_number))

        if isinstance(self, Expr):
            return apply(self)
        if isinstance(self, np.ndarray):
            if self.dtype != object or not self.flags.writeable:
                raise TypeError("Unary expression arrays must be writable object arrays")
            values = list(self.flat)
        elif isinstance(self, (list, tuple)):
            values = list(self)
        else:
            raise TypeError("Unary expression requires Expr, list, tuple or object ndarray")
        # Validate every cell before emitting IR or mutating shared storage.
        if not all(isinstance(value, Expr) for value in values):
            raise TypeError("Unary expression container cells must be Expr values")
        results = [apply(value) for value in values]
        if isinstance(self, tuple):
            return tuple(results)
        if isinstance(self, np.ndarray):
            self.flat[:] = results
        else:
            self[:] = results
        return self

    globals()[name] = unaryMethod
[unaryFactory(name, opcode) for name, opcode in toUnary.items()]
"""Metaclasses"""


class hecateMetaBase(type):
    def __new__(cls, name, bases, attrs):
        newcls = super().__new__(cls, name, bases, attrs)

        def raiser():
            raise Exception("Copying Hecate object is forbidden")
        setattr(newcls, "__copy__", raiser)
        setattr(newcls, "__deepcopy__", raiser)
        return newcls


class hecateMetaBinary(hecateMetaBase):
    def __new__(cls, name, bases, attrs):
        newcls = super().__new__(cls, name, bases, attrs)

        def binaryFactory(cls, name, opcode):
            def binaryMethod(self, other):
                (frame, filename, line_number, function_name, lines,
                        index) = inspect.stack()[1]
                tmp = resolveType(other)
                return Expr(
                        lt.createBinary(ctxt, opcode, self.obj, tmp.obj,
                            filename.encode('utf-8'), line_number))

            def binaryReverseMethod(self, other):
                (frame, filename, line_number, function_name, lines,
                        index) = inspect.stack()[1]
                tmp = resolveType(other)
                return Expr(
                        lt.createBinary(ctxt, opcode, tmp.obj, self.obj,
                            filename.encode('utf-8'), line_number))

            def binaryInplaceMethod(self, other):
                (frame, filename, line_number, function_name, lines,
                        index) = inspect.stack()[1]
                tmp = resolveType(other)
                self.obj = lt.createBinary(ctxt, opcode, self.obj, tmp.obj,
                        filename.encode('utf-8'),
                        line_number)

            setattr(cls, f"__{name}__", binaryMethod)
            setattr(cls, f"__r{name}__", binaryReverseMethod)
            # Augmented assignment rebinds to a new Expr, as before, but must
            # preserve lhs/rhs order. The reverse method made x -= y emit y-x.
            setattr(cls, f"__i{name}__", binaryMethod)


        def innerFactory(cls, name, opcode):
            def innerMethod(self):
                (frame, filename, line_number, function_name, lines,
                        index) = inspect.stack()[1]
                return Expr(
                        lt.createUnary(ctxt, opcode, self.obj,
                            filename.encode('utf-8'), line_number))

            setattr(cls, f"__{name}__", innerMethod)

        [ 
            binaryFactory(newcls, name, opcode)
            for name, opcode in toBinary.items()
        ]
        [
            innerFactory(newcls, name, opcode)
            for name, opcode in toInnerUnary.items()
        ]

        def rotate(self, offset) :
            (frame, filename, line_number, function_name, lines,
                        index) = inspect.stack()[1]
            return Expr(lt.createRotation(ctxt, self.obj, offset, 
                filename.encode('utf-8'), line_number))
        setattr(newcls, "rotate", rotate)
        return newcls

"""Helper Functions"""

def getProperFrame():
    (frame, thisname, line_number, function_name, lines,
            index) = inspect.stack()[0]
    for (frame, filename, line_number, function_name, lines,
            index) in inspect.stack():
        if thisname != filename:
            return (frame, filename, line_number, function_name, lines, index)

def recType(li):
    if isinstance(li, int) or isinstance(li, float):
        return []
    elif isinstance(li, list):
        tys = [recType(x) for x in li]
        if all(i == tys[0] for i in tys):
            return [len(li)] + tys[0]
        else:
            raise Exception("Cannot create compatiable type")
    else:
        raise Exception("Cannot create compatiable type")


def flatten(L):
    for l in L:
        if isinstance(l, list):
            yield from flatten(l)
        else:
            yield l


def resolveType(other):
    if isinstance(other, Expr):
        return other
    elif isinstance(other, int):
        return Plain(np.array([other], dtype=np.float64)) #Plain([other])
    elif isinstance(other, float):
        return Plain(np.array([other], dtype=np.float64)) #Plain([other])
    elif isinstance(other, list):
        return Plain(np.array(other, dtype=np.float64)) #Plain(other)
    elif isinstance(other, torch.Tensor) :
        return Plain( torch.flatten(other).tolist() )
    elif isinstance(other, np.ndarray):
        return Plain(other)
    else:
        print(type(other))
        raise Exception("Cannot create compatiable type")
    """Binary-operatable expression"""


class Expr(metaclass=hecateMetaBinary):
    def __init__(self, obj):
        self.obj = obj

class Plain(Expr):
    def __init__(self, data, scale = 40):
        # carr = (ctypes.c_double *
        #         len(data))(*[float(x) for x in flatten(data)])
        if not isinstance(data, np.ndarray) :
            data = np.array(data, dtype=np.float64)
        if isinstance(data, np.ndarray) :
            data = np.array(data.tolist(), dtype=np.float64)
         
        carr = npcl.as_ctypes(data) 
        (frame, filename, line_number, function_name, lines,
                index) = getProperFrame()
        super().__init__(
                lt.createConstant(ctxt, carr, len(data), filename.encode('utf-8'),
                    line_number))

class Empty :
    def __init__ (self) : 
        pass
    def __add__(self, other) :
        return resolveType(other)
    def __radd__(self, other) :
        return resolveType(other)
    def __iadd__(self, other) :
        return resolveType(other)
    def __sub__(self, other) :
        return resolveType(other)
    def __rsub__(self, other) :
        return resolveType(other)
    def __isub__(self, other) :
        return resolveType(other)


"""Function Decorator"""
funcList = []
_active_traces = []
_tracing_failed = False


def func(param):
    def generateFunc(func):
        global funcList
        (frame, filename, line_number, function_name, lines,
                index) = getProperFrame()
        a = Func(func, param, filename, line_number)
        funcList.append(a)
        return a

    return generateFunc


"""Function object"""
from collections.abc import Iterable

class Func(metaclass=hecateMetaBase):
    def __init__(self, fun, paramstr, filename, line_number):
        name = fun.__name__
        self.fun = fun

        if not isinstance(paramstr, str):
            raise TypeError("Function parameter signature must be a string")
        arg = [a.strip() for a in paramstr.split(",")] if paramstr.strip() else []
        if any(a not in ("c", "p") for a in arg):
            raise ValueError("Function parameters must be c or p")
        inputs = [ a== "c" for a in arg]

        self.inputlen = len(inputs)
        inspect.signature(fun).bind(*([None] * self.inputlen))
        self.evaluation_state = "new"
        inputTys = (ctypes.c_int * len(inputs))(*inputs)
        self.obj = lt.createFunc(ctxt, name.encode('utf-8'), inputTys,
                len(inputs), filename.encode('utf-8'),
                line_number)

    def eval(self):
        global _tracing_failed
        if _tracing_failed:
            raise ValueError("Native tracing context previously failed; use a fresh process")
        if self.evaluation_state == "done":
            return
        if self.evaluation_state != "new":
            raise ValueError("Recursive or previously failed decorated function tracing")
        self.evaluation_state = "active"
        _active_traces.append(self)
        try:
            inputarr = (ctypes.c_size_t * self.inputlen)()
            lt.initFunc(ctxt, self.obj, inputarr, self.inputlen)
            inputs = [Expr(x) for x in inputarr[:self.inputlen]]
            returns = self.fun(*inputs)
            if _tracing_failed:
                raise ValueError("Nested native tracing failed; use a fresh process")
            if isinstance(returns, Expr):
                self.return_kind, self.return_shape = "scalar", None
                values = [returns]
            elif isinstance(returns, np.ndarray) and returns.dtype == object:
                self.return_kind, self.return_shape = "array", returns.shape
                values = list(returns.flat)
            elif isinstance(returns, (list, tuple)):
                self.return_kind, self.return_shape = type(returns).__name__, None
                values = list(returns)
            else:
                raise TypeError("Decorated functions return Expr or a flat expression container")
            if not all(isinstance(value, Expr) for value in values):
                raise TypeError("Function output cells must be Expr values")
            outputs = [x.obj for x in values]
            self.outputlen = len(outputs)
            outputvec = (ctypes.c_size_t * len(outputs))(*outputs)
            lt.setOutput(ctxt, self.obj, outputvec, len(outputs))
            self.evaluation_state = "done"
        except BaseException:
            self.evaluation_state = "failed"
            _tracing_failed = True
            raise
        finally:
            _active_traces.pop()

    def __call__(self, *args):
        if len(args) != self.inputlen:
            raise TypeError("Decorated function argument count mismatch")
        if not _active_traces or _tracing_failed:
            raise ValueError("Decorated function calls require an active, non-failed trace")
        (frame, filename, line_number, function_name, lines,
                index) = getProperFrame()
        tmps = [resolveType(arg) for arg in args]
        argarr = (ctypes.c_size_t * len(args))(*[tmp.obj for tmp in tmps])
        self.eval()  # Also handles forward definitions; restores caller insertion point.
        outputs = (ctypes.c_size_t * self.outputlen)()
        count = lt.createCall(ctxt, self.obj, argarr, len(args), outputs,
                             self.outputlen, filename.encode('utf-8'), line_number)
        if count != self.outputlen:
            raise ValueError("Native function call type/body/insertion-point mismatch")
        values = [Expr(value) for value in outputs]
        if self.return_kind == "scalar":
            return values[0]
        if self.return_kind == "array":
            result = np.empty(self.return_shape, dtype=object)
            result.flat[:] = values
            return result
        return tuple(values) if self.return_kind == "tuple" else values


def removeCtxt() :
    lt.removeCtxt()
