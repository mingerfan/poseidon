"""Wait for the recovered comparison; reuse its original isolated report policy."""
import copy
from types import FunctionType

from . import bert_three_hour_report_waiter as original
from . import bert_recovery_frozen_report as report

MODULE='pt_agent.experiments.bert_recovery_report_waiter'


def namespace():
    source=vars(original);values=dict(source,report=report)
    for name,fn in source.items():
        if isinstance(fn,FunctionType) and fn.__globals__ is source:
            clone=FunctionType(fn.__code__,values,fn.__name__,fn.__defaults__,fn.__closure__)
            clone.__kwdefaults__=copy.copy(fn.__kwdefaults__);values[name]=clone
    return values


def main(argv=None):return namespace()['main'](argv)


if __name__=='__main__':raise SystemExit(main())
