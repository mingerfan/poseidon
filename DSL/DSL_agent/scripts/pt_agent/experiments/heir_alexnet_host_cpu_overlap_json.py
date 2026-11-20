"""Original Alex host pipeline with private JSON decoding for both literal paths.

Keep the original31-constant mask/inventory/float64 checks. A trusted source AST
copy changes one local checker import binding; private ast.literal_eval decodes
the same emitter numeric syntax using the reviewed common JSON helper.
"""
import ast
import copy
import inspect
import json
from pathlib import Path
import shutil
import textwrap
from types import CodeType,FunctionType,SimpleNamespace

from . import heir_alexnet_host_cpu_overlap as base
from . import heir_vgg_host_cpu_overlap_rss12_json as common_json

alex=base.alex
host=base.host
ROUND='HEIR-ALEX-HOST-CPU-OVERLAP-JSON'
MODE=base.MODE+'-strict-numeric-json-literals'
MODULE='pt_agent.experiments.heir_alexnet_host_cpu_overlap_json'
OUT_PREFIX='heir-alexnet-wide-pool-mask-host-cpu-overlap-json-'
SCOPE=(base.SCOPE+' Original31-constant checker with two private JSON bindings: '
    'mask literal decoding and the common checker local import. All original '
    'inventory, mask mathematics, row order and exactfloat64 comparison retained.')


def checker_tree():
    original=ast.parse(textwrap.dedent(inspect.getsource(alex.check_constants)))
    tree=copy.deepcopy(original);function=tree.body[0]
    locations=[i for i,node in enumerate(function.body)if isinstance(node,ast.ImportFrom)
        and node.level==1 and node.module=='heir_cnn_math_check']
    if len(locations)!=1:raise ValueError('Exactly one original Alex common-checker import seam required')
    index=locations[0];node=function.body[index]
    if [(a.name,a.asname)for a in node.names]!=[('check_constants','original'),('check_math',None)]:
        raise ValueError('Preserve the original Alex checker import aliases')
    function.body[index]=ast.copy_location(ast.ImportFrom(module=node.module,
        names=[ast.alias(name='check_math',asname=None)],level=1),node)
    function.body.insert(index+1,ast.copy_location(ast.Assign(targets=[ast.Name(id='original',ctx=ast.Store())],
        value=ast.Name(id='_json_common_checker',ctx=ast.Load())),node))
    # Remove precisely the approved import/assignment delta for a complete
    # structural comparison; every other original AST node must be identical.
    restored=copy.deepcopy(tree);restored.body[0].body[index:index+2]=[copy.deepcopy(original.body[0].body[index])]
    if ast.dump(restored,include_attributes=False)!=ast.dump(original,include_attributes=False):
        raise ValueError('Unexpected original checker AST change outside the import binding')
    return ast.fix_missing_locations(tree)


def constant_checker():
    tree=checker_tree();compiled=compile(tree,'<private-original-Alex-checker-JSON-import>','exec')
    code=[value for value in compiled.co_consts if isinstance(value,CodeType)and value.co_name=='check_constants']
    if len(code)!=1:raise ValueError('Unique trusted original Alex checker code required')
    namespace=dict(vars(alex),ast=SimpleNamespace(literal_eval=common_json.numeric_json_literal),
        _json_common_checker=common_json.constant_checker())
    function=FunctionType(code[0],namespace,alex.check_constants.__name__,alex.check_constants.__defaults__)
    function.__kwdefaults__=alex.check_constants.__kwdefaults__;return function


def _runtime():
    api=alex._runtime();api['check_constants']=constant_checker();return api


def numeric_worker():return host.overlap._clone_plain(alex.worker,dict(vars(alex),_runtime=_runtime))


def worker(args):return numeric_worker()(args)


def _scope():
    scope=base._scope();numeric=SimpleNamespace(**dict(vars(scope['original']),MODULE=MODULE,worker=worker,_runtime=_runtime))
    scope.update(original=numeric,base=SimpleNamespace(**dict(vars(scope['base']),original=numeric)),
        ROUND=ROUND,MODE=MODE,SCOPE=SCOPE,OUT_PREFIX=OUT_PREFIX)
    for name in('worker_command','runtime_record'):scope[name]=host.overlap._clone_plain(getattr(host,name),scope)
    scope['_namespace']=lambda args:_supervised_namespace(args,scope)
    return scope


def _supervised_namespace(args,scope):
    namespace=base._supervised_namespace(args,scope)
    previous_snapshot=namespace['_snapshot'];previous_durable=namespace['durable_json']
    def snapshot(out):
        rows=previous_snapshot(out)
        for path,label in((Path(__file__),'heir-alexnet-host-cpu-overlap-json.py'),
                (Path(common_json.__file__),'private-common-numeric-json-checker.py')):
            target=out/'source'/label;shutil.copyfile(path,target)
            rows.append(dict(original=str(path.resolve()),snapshot=str(target),bytes=target.stat().st_size))
        return rows
    def durable(path,result):
        if Path(path).name=='result.json'and result.get('round')==ROUND:
            result.update(constant_parser_variant='private-Alex-mask-and-common-numeric-JSON',
                original_checker_AST_except_one_import_binding_unchanged=True,
                original_math_array_equality_and_31_constant_order_unchanged=True,
                original_worker_module=MODULE,original_Alex_worker_code_unchanged=True)
        return previous_durable(path,result)
    namespace.update(_snapshot=snapshot,durable_json=durable);return namespace


def _namespace(args):return _supervised_namespace(args,_scope())


def execute(args):return host.overlap._clone_plain(host.execute,_scope())(args)


def parser():
    command=base.parser();command.description=__doc__;return command


def main(argv=None):
    import sys
    argv=list(sys.argv[1:]if argv is None else argv)
    if'--worker'in argv:
        namespace=dict(vars(common_json),worker=worker)
        result=host.overlap._clone_plain(common_json.worker_main,namespace)(argv)
    else:result=execute(parser().parse_args(argv))
    print(json.dumps({key:result.get(key)for key in('status','error','strict_math_passed',
        'model_compiled','model_compiled_accepted','constant_parser_variant')}))
    return int(result['status']not in('ok','prepared'))


if __name__=='__main__':raise SystemExit(main())
