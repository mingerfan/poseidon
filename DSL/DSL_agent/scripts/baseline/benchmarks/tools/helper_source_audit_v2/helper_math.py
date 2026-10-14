"""Bounded mathematical diagnosis for already validated decorated DSL functions.

No candidate import/eval/exec, frontend, compiler, FHE runtime or Provider call.
CKKS noise, level/scale and rescale are outside this interpreter's scope.
"""
import ast,sys
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])
sys.path.insert(0,str(ROOT/"scripts/baseline/benchmarks/tools/closed_live_source_audit_v1"))
from audit import Unsupported,finite,numeric,arithmetic

def evaluate(source,inputs,constants,period):
    if type(period) is not int or not 1<=period<=4096:raise Unsupported("Diagnostic period budget")
    tree=ast.parse(source)
    if len(list(ast.walk(tree)))>4096:raise Unsupported("AST diagnostic budget")
    if not 1<=len(tree.body)<=32 or any(type(n) is not ast.FunctionDef for n in tree.body):
        raise Unsupported("Decorated functions only")
    functions={f.name:f for f in tree.body}
    if len(functions)!=len(tree.body) or "golden" not in functions:raise Unsupported("Unique functions and golden required")
    calls={}
    for name,function in functions.items():
        args=function.args
        if (args.posonlyargs or args.vararg or args.kwarg or args.kwonlyargs or args.defaults or args.kw_defaults):
            raise Unsupported("Positional fixed function parameters only")
        if len({a.arg for a in args.args})!=len(args.args):raise Unsupported("Unique function parameters")
        if len(function.decorator_list)!=1:raise Unsupported("Fixed hc.func declaration")
        declaration=function.decorator_list[0]
        if not (type(declaration) is ast.Call and type(declaration.func) is ast.Attribute
            and type(declaration.func.value) is ast.Name and declaration.func.value.id=="hc"
            and declaration.func.attr=="func" and len(declaration.args)==1 and not declaration.keywords
            and type(declaration.args[0]) is ast.Constant and type(declaration.args[0].value) is str):
            raise Unsupported("Fixed hc.func declaration")
        types=declaration.args[0].value.split(",")
        if any(t not in ("c","p") for t in types) or len(types)!=len(args.args):
            raise Unsupported("Scalar Expr declarations only")
        calls[name]={n.func.id for n in ast.walk(function) if type(n) is ast.Call and type(n.func) is ast.Name}
        if not calls[name]<=functions.keys():raise Unsupported("Unregistered function call")
    visited=set()
    def check(name,stack=()):
        if name in stack:raise Unsupported("Recursive helper")
        if len(stack)>=32:raise Unsupported("Helper depth budget")
        if name in visited:return
        for target in calls[name]:check(target,(*stack,name))
        visited.add(name)
    for name in functions:check(name)
    globals_={name:numeric(value,period) for name,value in constants.items()}
    ops={ast.Add:lambda a,b:a+b,ast.Sub:lambda a,b:a-b,ast.Mult:lambda a,b:a*b}
    steps=0
    def tick():
        nonlocal steps
        steps+=1
        if steps>65536:raise Unsupported("Expanded diagnostic work budget")
    def invoke(name,values,stack=()):
        tick()
        if name in stack or len(stack)>=32:raise Unsupported("Recursive or deep helper")
        function=functions[name]
        if len(values)!=len(function.args.args):raise Unsupported("Function arity mismatch")
        locals_={arg.arg:value for arg,value in zip(function.args.args,values)}
        def expression(node):
            tick()
            if type(node) is ast.Name:
                if node.id in locals_:return locals_[node.id]
                if node.id in globals_:return globals_[node.id]
                raise Unsupported("Unbound symbol: "+node.id)
            if type(node) is ast.Constant:return finite(node.value)
            if type(node) is ast.UnaryOp and type(node.op) is ast.USub:
                return arithmetic(0.,expression(node.operand),ops[ast.Sub],period)
            if type(node) is ast.BinOp and type(node.op) in ops:
                return arithmetic(expression(node.left),expression(node.right),ops[type(node.op)],period)
            if type(node) is ast.Call and type(node.func) is ast.Name:
                if node.keywords or any(type(n) is ast.Starred for n in node.args):raise Unsupported("Fixed helper arguments")
                return invoke(node.func.id,[expression(n) for n in node.args],(*stack,name))
            if (type(node) is ast.Call and type(node.func) is ast.Attribute and node.func.attr=="rotate"
                and len(node.args)==1 and not node.keywords):
                step=ast.literal_eval(node.args[0])
                if type(step) is not int or abs(step)>4096:raise Unsupported("Static integer rotation")
                vector=expression(node.func.value)
                if type(vector) is not tuple or len(vector)!=period:raise Unsupported("Rotation vector")
                return tuple(vector[(i+step)%period] for i in range(period))
            raise Unsupported("Unsupported expression: "+type(node).__name__)
        returned=False;value=None
        for statement in function.body:
            tick()
            if returned:raise Unsupported("Statement after return")
            if type(statement) is ast.Assign and len(statement.targets)==1 and type(statement.targets[0]) is ast.Name:
                locals_[statement.targets[0].id]=expression(statement.value)
            elif type(statement) is ast.Return:
                if type(statement.value) in (ast.List,ast.Tuple):
                    value=[expression(n) for n in statement.value.elts]
                else:value=expression(statement.value)
                returned=True
            else:raise Unsupported("Unsupported statement: "+type(statement).__name__)
        if not returned:raise Unsupported("Return required")
        return value
    values=[]
    for value in inputs:
        if type(value) is not list or len(value)!=period:raise Unsupported("Input period mismatch")
        values.append(numeric(value,period))
    result=invoke("golden",values)
    outputs=result if type(result) is list else [result]
    if not outputs or any(type(v) is not tuple or len(v)!=period for v in outputs):
        raise Unsupported("Vector outputs required")
    return outputs
