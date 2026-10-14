"""Read-only candidate arithmetic diagnostics. Never execute source or replace a backend."""
import ast,math
from pathlib import Path
import importlib.util
OLD=Path(__file__).resolve().parent.parent/"closed_live_source_audit_v1/audit.py"
spec=importlib.util.spec_from_file_location("retained_straight_line_audit",OLD)
legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
Unsupported=legacy.Unsupported
difference=legacy.difference

def evaluate(source,inputs,constants,period,*,entry=None):
    if type(source) is not str or len(source.encode())>65536:
        raise Unsupported("source diagnostic budget")
    if type(period) is not int or not 1<=period<=16384:
        raise Unsupported("period diagnostic budget")
    try:tree=ast.parse(source)
    except (SyntaxError,RecursionError,ValueError):raise Unsupported("source parse")
    if sum(1 for _ in ast.walk(tree))>4096 or not 1<=len(tree.body)<=16 or any(type(n) is not ast.FunctionDef for n in tree.body):
        raise Unsupported("registered straight-line function declarations required")
    functions={}
    for function in tree.body:
        a=function.args
        if function.name in functions or a.posonlyargs or a.vararg or a.kwonlyargs or a.kwarg or a.defaults or a.kw_defaults:
            raise Unsupported("positional helper signature required")
        if function.returns is not None or any(x.annotation is not None for x in a.args):
            raise Unsupported("plain helper signature required")
        for d in function.decorator_list:
            if not (type(d) is ast.Call and not d.keywords and len(d.args)==1 and
                    type(d.func) is ast.Attribute and d.func.attr=="func" and
                    type(d.func.value) is ast.Name and d.func.value.id=="hc" and
                    type(d.args[0]) is ast.Constant and type(d.args[0].value) is str and
                    d.args[0].value.split(",")==["c"]*len(a.args)):
                raise Unsupported("registered hc.func signature required")
        if any(type(s) not in (ast.Assign,ast.Return) for s in function.body):
            raise Unsupported("straight-line assignment and return required")
        functions[function.name]=function
    if entry is None:entry=next(iter(functions)) if len(functions)==1 else "golden"
    if entry not in functions:raise Unsupported("entry function required")
    public={k:legacy.numeric(v,period) for k,v in constants.items()}
    if any(name in functions for name in public):raise Unsupported("public/helper name conflict")
    if type(inputs) is not list or any(type(v) is not list or len(v)!=period for v in inputs):
        raise Unsupported("input period mismatch")
    operations={ast.Add:lambda a,b:a+b,ast.Sub:lambda a,b:a-b,ast.Mult:lambda a,b:a*b}
    visited=work=calls=0
    stack=[]
    def charge(cost=1):
        nonlocal visited,work
        visited+=1;work+=cost
        if visited>32768 or work>1048576:raise Unsupported("arithmetic diagnostic work budget")
    def vector_arithmetic(left,right,op):
        charge(period)
        try:return legacy.arithmetic(left,right,op,period)
        except (OverflowError,TypeError,ZeroDivisionError):raise Unsupported("finite diagnostic arithmetic")
    def expression(node,symbols):
        charge()
        if type(node) is ast.Name:
            if node.id not in symbols:raise Unsupported("unbound diagnostic symbol")
            return symbols[node.id]
        if type(node) is ast.Constant:return legacy.finite(node.value)
        if type(node) is ast.UnaryOp and type(node.op) in (ast.UAdd,ast.USub):
            value=expression(node.operand,symbols)
            return value if type(node.op) is ast.UAdd else vector_arithmetic(0,value,operations[ast.Sub])
        if type(node) is ast.BinOp and type(node.op) in operations:
            return vector_arithmetic(expression(node.left,symbols),expression(node.right,symbols),operations[type(node.op)])
        if type(node) in (ast.Tuple,ast.List):return [expression(n,symbols) for n in node.elts]
        if type(node) is ast.Call and not node.keywords:
            if type(node.func) is ast.Name and node.func.id in functions:
                if node.func.id in symbols:raise Unsupported("shadowed helper call")
                return call(node.func.id,[expression(n,symbols) for n in node.args])
            if type(node.func) is ast.Attribute and node.func.attr=="rotate" and len(node.args)==1:
                try:step=ast.literal_eval(node.args[0])
                except (ValueError,TypeError,RecursionError):raise Unsupported("static integer rotation required")
                if type(step) is not int or abs(step)>4096:raise Unsupported("static integer rotation required")
                value=expression(node.func.value,symbols)
                if type(value) is not tuple or len(value)!=period:raise Unsupported("rotation vector required")
                charge(period);return tuple(value[(i+step)%period] for i in range(period))
        raise Unsupported("unsupported diagnostic expression")
    def call(name,args):
        nonlocal calls
        calls+=1
        if calls>256 or len(stack)>=16 or name in stack:raise Unsupported("helper recursion or call budget")
        function=functions[name]
        if len(function.args.args)!=len(args):raise Unsupported("helper arity mismatch")
        symbols=dict(public)
        symbols.update((arg.arg,value) for arg,value in zip(function.args.args,args))
        stack.append(name)
        try:
            returned=False;result=None
            for statement in function.body:
                if returned:raise Unsupported("statement after return")
                if type(statement) is ast.Assign and len(statement.targets)==1 and type(statement.targets[0]) is ast.Name:
                    symbols[statement.targets[0].id]=expression(statement.value,symbols)
                elif type(statement) is ast.Return and statement.value is not None:
                    result=expression(statement.value,symbols);returned=True
                else:raise Unsupported("unsupported diagnostic statement")
            if not returned:raise Unsupported("helper return required")
            return result
        finally:stack.pop()
    result=call(entry,[legacy.numeric(v,period) for v in inputs])
    outputs=result if type(result) is list else [result]
    if not outputs or any(type(v) is not tuple or len(v)!=period for v in outputs):
        raise Unsupported("vector output ports required")
    return outputs
