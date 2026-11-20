"""Original RSS12 VGG pipeline with a private numeric-literal JSON decoder.

Keep the original check_constants code, inventory, float64 arrays, independent
math check, exact array equality and output records. The original emitter's
numeric nested-array grammar is decoded without building a Python syntax tree.
"""
import argparse
import json
from pathlib import Path
import re
import resource
import shutil
import signal
import sys
from types import SimpleNamespace

from . import heir_cnn_math_check as math_check
from . import heir_vgg_host_cpu_overlap_rss12 as rss12

host=rss12.host
original=host.original
ROUND='HEIR-VGG-HOST-CPU-OVERLAP-RSS12-JSON'
MODE=rss12.MODE+'-strict-numeric-json-literals'
MODULE='pt_agent.experiments.heir_vgg_host_cpu_overlap_rss12_json'
OUT_PREFIX='heir-vgg16-fused-pool-conv-host-cpu-overlap-rss12-json-'
SCOPE=(rss12.SCOPE+' Private original-checker code uses strict numeric JSON '
    'literal decoding in place of Python AST construction; every original '
    'constant inventory/shape/float64/numerical equality criterion is retained.')
NUMERIC_ARRAY_GRAMMAR=re.compile(r'[\[\]0-9+\-., eE\t\r\n]+')


def _reject_constant(value):
    raise ValueError('Finite original emitted numeric literal required: '+value)


def numeric_json_literal(text):
    if not isinstance(text,str)or NUMERIC_ARRAY_GRAMMAR.fullmatch(text)is None:
        raise ValueError('Original finite-number nested-array literal grammar required')
    return json.loads(text,parse_constant=_reject_constant)


def constant_checker():
    namespace=dict(vars(math_check),ast=SimpleNamespace(literal_eval=numeric_json_literal))
    return host.overlap._clone_plain(math_check.check_constants,namespace)


def _runtime():
    api=original._runtime();api['check_constants']=constant_checker();return api


def numeric_worker():
    return host.overlap._clone_plain(original.worker,dict(vars(original),_runtime=_runtime))


def worker(args):return numeric_worker()(args)


def _scope():
    scope=rss12._scope()
    numeric=SimpleNamespace(**dict(vars(original),MODULE=MODULE,worker=worker,_runtime=_runtime))
    base=SimpleNamespace(**dict(vars(scope['base']),original=numeric))
    scope.update(original=numeric,base=base,ROUND=ROUND,MODE=MODE,SCOPE=SCOPE,OUT_PREFIX=OUT_PREFIX)
    for name in('worker_command','runtime_record'):
        scope[name]=host.overlap._clone_plain(getattr(host,name),scope)
    scope['_namespace']=lambda args:_supervised_namespace(args,scope)
    return scope


def _supervised_namespace(args,scope):
    namespace=rss12._supervised_namespace(args,scope)
    previous_snapshot=namespace['_snapshot'];previous_durable=namespace['durable_json']
    def snapshot(out):
        rows=previous_snapshot(out);target=out/'source'/'heir-vgg-host-cpu-overlap-rss12-json.py'
        shutil.copyfile(__file__,target)
        rows.append(dict(original=str(Path(__file__).resolve()),snapshot=str(target),bytes=target.stat().st_size))
        return rows
    def durable(path,result):
        if Path(path).name=='result.json'and result.get('round')==ROUND:
            result.update(constant_parser_variant='private-numeric-JSON-original-checker-code',
                original_constant_check_code_unchanged=True,original_math_and_array_equality_unchanged=True,
                resource_variant='aggregate-RSS12GiB; original-AS24/hostavailable2GiB',
                prior_RSS10_failure_preserved=True)
        return previous_durable(path,result)
    namespace.update(_snapshot=snapshot,durable_json=durable)
    return namespace


def _namespace(args):return _supervised_namespace(args,_scope())


def execute(args):return host.overlap._clone_plain(host.execute,_scope())(args)


def parser():
    command=host.parser();command.description=__doc__;return command


def worker_parser():
    command=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    command.add_argument('--worker',action='store_true',required=True)
    command.add_argument('--out',type=Path,required=True)
    command.add_argument('--memory-gib',type=int,choices=(24,),required=True)
    for name in original.SOURCE_FIELDS:command.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    return command


def worker_main(argv):
    args=worker_parser().parse_args(argv)
    resource.setrlimit(resource.RLIMIT_AS,(24*original.GIB,24*original.GIB))
    resource.setrlimit(resource.RLIMIT_FSIZE,(16*original.GIB,16*original.GIB))
    signal.alarm(original.WORKER_SECONDS)
    try:return worker(args)
    finally:signal.alarm(0)


def main(argv=None):
    argv=list(sys.argv[1:]if argv is None else argv)
    result=worker_main(argv)if'--worker'in argv else execute(parser().parse_args(argv))
    print(json.dumps({key:result.get(key)for key in('status','error','strict_math_passed',
        'model_compiled','model_compiled_accepted','constant_parser_variant')}))
    return int(result['status']not in('ok','prepared'))


if __name__=='__main__':raise SystemExit(main())
