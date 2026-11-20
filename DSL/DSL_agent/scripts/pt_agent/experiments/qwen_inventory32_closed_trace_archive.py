"""Pinned inventory32 family for the original closed-component archive observer.

Only producer/module identities change. Every original gate, gzip worker,
consumer check, resource limit, shared observer lock and terminal protocol runs
from unchanged code objects in a private namespace.
"""
from types import FunctionType

from . import qwen_closed_component_trace_archive as base

MODULE='pt_agent.experiments.qwen_inventory32_closed_trace_archive'
ROUND='QWEN-INVENTORY32-CLOSED-COMPONENT-TRACE-ARCHIVE'
MODE='inventory32-accepted-component-two-MLIR-gzip-overlap-observer'
WATCH_PID,WATCH_START=219446,5038402
PREFIX='qwen-retained13-component-inventory32-stagecut-schema-'
CASE_NAMES=dict(source=PREFIX+'recovery-20261117',
    probe=PREFIX+'frozen-layer0-native-abi-probe-20261117',
    remaining=PREFIX+'frozen-remaining23-head-native-compile-20261117',
    kernel=PREFIX+'frozen-first-terminal-native-kernels-20261117',
    watch='deadline-qwen-retained-inventory32-stagecut-schema-serial-batch-20261117')


def _namespace():
    namespace=vars(base).copy()
    namespace.update(MODULE=MODULE,ROUND=ROUND,MODE=MODE,WATCH_PID=WATCH_PID,
        WATCH_START=WATCH_START,PREFIX=PREFIX,CASE_NAMES=CASE_NAMES.copy())
    for name,function in vars(base).items():
        if isinstance(function,FunctionType) and function.__globals__ is vars(base):
            cloned=FunctionType(function.__code__,namespace,function.__name__,
                function.__defaults__,function.__closure__)
            cloned.__kwdefaults__=(None if function.__kwdefaults__ is None else function.__kwdefaults__.copy())
            cloned.__annotations__=function.__annotations__.copy()
            cloned.__dict__.update(function.__dict__)
            namespace[name]=cloned
    return namespace


def parser():return _namespace()['parser']()


def main(argv=None):return _namespace()['main'](argv)


if __name__=='__main__':raise SystemExit(main())
