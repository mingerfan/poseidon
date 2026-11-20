"""Private lifetime adaptation for the original Qwen Agent cost schedule.

Only original function dependencies and public-vector ownership change. The
original planner, arithmetic, candidate, partitions and feasibility policy run
unchanged; compact rectangular scheduling is outside this adapter.
"""
import ast
from contextlib import contextmanager
import inspect
import textwrap
from types import FunctionType,ModuleType


VARIANT='qwen-agent-original-schedule-ephemeral-cost-v1'


def _estimate_delegate(original,namespace):
    tree=ast.parse(textwrap.dedent(inspect.getsource(original)))
    found=[]
    class Dependencies(ast.NodeTransformer):
        def visit_ImportFrom(self,node):
            names=tuple(alias.name for alias in node.names)
            if node.level==1 and node.module=='plan'and names==('plan_program','verify_program'):
                found.append('plan');return None
            if node.level==1 and node.module=='backends.cost_dag'and names==('CostDag',):
                found.append('CostDag');return None
            return node
    tree=Dependencies().visit(tree)
    if sorted(found)!=['CostDag','plan']:raise ValueError('Original cost local dependencies changed')
    ast.fix_missing_locations(tree)
    exec(compile(tree,inspect.getsourcefile(original)or __file__,'exec'),namespace)
    delegated=namespace[original.__name__]
    delegated.__defaults__=original.__defaults__;delegated.__kwdefaults__=original.__kwdefaults__
    return delegated


def private_cost_namespace(cost_module=None,planner=None):
    from . import qwen_bridge_agent as bridge
    if cost_module is None:
        from .. import cost as cost_module
    if planner is None:
        from .qwen_prefill_compile import _planner_namespace
        planner=_planner_namespace()
    namespace=bridge._clone_namespace(cost_module,dict(CostDag=planner['CostDag'],
        CountBackend=planner['CountBackend'],plan_program=planner['plan_program'],verify_program=planner['verify_program']))
    namespace['estimate']=_estimate_delegate(cost_module.estimate,namespace)
    return namespace


@contextmanager
def original_schedule_scope():
    from .qwen_prefill_compile import _projection_backend_scope
    from ..kernels.streaming_linear import streaming_transformer_projections
    with _projection_backend_scope(),streaming_transformer_projections():yield


def _replace_closure(function,name,value):
    names=function.__code__.co_freevars
    if names.count(name)!=1:raise ValueError('Original bridge evaluator closure changed')
    def cell(item):return (lambda:item).__closure__[0]
    closure=list(function.__closure__);closure[names.index(name)]=cell(value)
    delegated=FunctionType(function.__code__,function.__globals__,function.__name__,function.__defaults__,tuple(closure))
    delegated.__kwdefaults__=function.__kwdefaults__
    return delegated


def adapt_agent_functions(functions,costs=None,scope=None):
    """Preserve the bridge candidate gate around its private original evaluator."""
    from . import qwen_bridge_agent as bridge
    callback=functions['evaluate']
    original=inspect.getclosurevars(callback).nonlocals.get('original_evaluate')
    if (not isinstance(original,FunctionType)or original.__name__!='evaluate'
            or original.__globals__.get('__name__')!='pt_agent.planners.agent_tools'):
        raise ValueError('Require the original private bridge Agent evaluator')
    costs=private_cost_namespace()if costs is None else costs
    module=ModuleType(original.__module__);module.__dict__.update(original.__globals__)
    tools=bridge._clone_namespace(module,{name:costs[name]for name in
        ('estimate','estimate_normalized','estimate_fast','estimate_normalized_fast')})
    evaluate=tools['evaluate'];scope=original_schedule_scope if scope is None else scope
    def measured(source,ctx,**kwargs):
        with scope():return evaluate(source,ctx,**kwargs)
    adapted=dict(functions)
    # Agent helpers resolve evaluate/feedback_message through their globals.
    # Rebind that private namespace together; copying just its dictionary
    # leaves agent_step and _finish_iteration looking up the old callbacks.
    for name,function in functions.items():
        if isinstance(function,FunctionType)and function.__globals__ is functions:
            cloned=FunctionType(function.__code__,adapted,function.__name__,function.__defaults__,function.__closure__)
            cloned.__kwdefaults__=function.__kwdefaults__
            cloned.__annotations__=function.__annotations__
            cloned.__dict__.update(function.__dict__)
            adapted[name]=cloned
    adapted['evaluate']=_replace_closure(callback,'original_evaluate',measured)
    return adapted
