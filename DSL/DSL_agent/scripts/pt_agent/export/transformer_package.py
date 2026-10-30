"""Copy U4 arithmetic and serialized polynomial coefficients into P2 core."""

import ast
import inspect

from ..kernels import transformer as tr,approx
from ..kernels.poly import cheb_fit,minimax_coeffs


def entry(attrs,op):
    result={k:v for k,v in attrs.items() if v is not None}
    result.update(op_id=op.id,lineno=op.lineno)
    method=attrs['method']
    if method=='cheb':coefficients=cheb_fit(attrs['fn'],attrs['deg'],*attrs['domain'])
    elif method=='minimax_comp':coefficients=minimax_coeffs(tuple(attrs['degrees']))
    elif method=='exact':return result
    else:coefficients=approx.fitted_coefficients(attrs)
    result['coefficients']=([v.tolist() for v in coefficients] if method in ('composite','minimax_comp') else coefficients.tolist())
    return result


def points(program):
    result={}
    for op in program.ops:
        if op.kind in ('layernorm','rmsnorm'):
            result[op.outputs[0]+'::rsqrt']=entry(tr.approximation(op.attrs,'rsqrt','rsqrt'),op)
        elif op.kind=='softmax':
            for role in ('exp','inv'):
                result[op.outputs[0]+'::'+role]=entry(tr.approximation(op.attrs,role,role),op)
            if op.attrs.get('steps',0):
                result[op.outputs[0]+'::step_inv']=entry(tr.approximation(op.attrs,'step_inv','inv'),op)
            if op.attrs.get('final_inv_method') is not None:
                result[op.outputs[0]+'::final_inv']=entry(tr.approximation(op.attrs,'final_inv','inv'),op)
    return result


def core_texts():
    from .. import transformer_lower
    from ..config import LINEAR_TALL_MIN_RATIO
    text=inspect.getsource(tr).replace('from .poly import act_cheb, act_minimax','from .cheb import cheb_eval')
    tree=ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):
            # Standalone packages evaluate the complete saved coefficients;
            # fitting and mathematical exact mode remain external.
            node.body=[ast.Raise(ast.Call(ast.Name('KernelError',ast.Load()),
                          [ast.Constant('exact mode is external to P2')],[]),None)
                       if isinstance(child,ast.If) and isinstance(child.test,ast.Name) and child.test.id=='exact' else child
                       for child in node.body]
            if node.name=='apply':
                node.body=ast.parse('''
if a['method']=='exact':return value*value
if a['method']=='cheb':
 lo,hi=a['domain']
 return cheb_eval(b,value*(2/(hi-lo))-(hi+lo)/(hi-lo),a['coefficients'])
if a['method']=='minimax_comp':
 y=value*(1/a['domain'][1])
 for c in a['coefficients']:y=cheb_eval(b,y,c)
 return value*(y+0.5)
return approx.evaluate(b,value,a['fn'],a['method'],a['domain'],a['coefficients'],iters=a.get('iters') or 0)
''').body
            node.body=[child for child in node.body if not (isinstance(child,ast.ImportFrom) and child.level==2)]
    # The exact-only MockExpr references are unreachable: the exact branches
    # above raise before any use. No Mock implementation is shipped.
    text=ast.unparse(ast.fix_missing_locations(tree))+'\n'
    dispatch=inspect.getsource(transformer_lower)
    dispatch=dispatch.replace('from .config import LINEAR_TALL_MIN_RATIO',f'LINEAR_TALL_MIN_RATIO = {LINEAR_TALL_MIN_RATIO}')
    dispatch=dispatch.replace('from .kernels import transformer as tr','from . import transformer as tr')
    dispatch=dispatch.replace('from .kernels.cnn import','from .cnn import').replace('from .kernels.layout import','from .layout import')
    return {'transformer.py':text,'token_dispatch.py':dispatch}
