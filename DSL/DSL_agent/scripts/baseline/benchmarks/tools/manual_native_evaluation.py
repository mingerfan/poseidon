"""Small test-only AST interpreter for arithmetic helper probes; never reference or FHE."""
import ast
import numpy as np

def evaluate(source,request,inputs):
    if len(source.encode())>65536:raise ValueError("Source limit")
    tree=ast.parse(source)
    if sum(1 for _ in ast.walk(tree))>4096:raise ValueError("AST limit")
    functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    if len(functions)!=len(tree.body) or "golden" not in functions:raise ValueError("Function-only probe")
    period=request["layout"]["input_slot_period"]
    public={k:np.asarray(v,dtype=np.float64) for k,v in request["public_constants"].items()}
    provided={}
    for b in request["layout"]["inputs"]:
        x=np.asarray(inputs[b["name"]],dtype=np.float64).reshape(-1)
        if len(x)>period:raise ValueError("Probe needs single-block layout")
        provided[b["dsl_name"]]=np.pad(x,(0,period-len(x)))
    provided["zero_ct"]=np.zeros(period)
    fuel=[0]
    def call(name,args,depth):
        if depth>16:raise ValueError("Helper depth")
        fn=functions[name]
        if len(args)!=len(fn.args.args):raise ValueError("Helper arity")
        env=dict(public);env.update({p.arg:v for p,v in zip(fn.args.args,args)})
        def expr(n):
            fuel[0]+=1
            if fuel[0]>20000:raise ValueError("Probe work budget")
            if isinstance(n,ast.Name):return env[n.id]
            if isinstance(n,ast.Constant) and type(n.value) in (int,float):return n.value
            if isinstance(n,(ast.List,ast.Tuple)):return [expr(x) for x in n.elts]
            if isinstance(n,ast.UnaryOp) and isinstance(n.op,ast.USub):return -expr(n.operand)
            if isinstance(n,ast.BinOp):
                a,b=expr(n.left),expr(n.right)
                if isinstance(n.op,ast.Add):return a+b
                if isinstance(n.op,ast.Sub):return a-b
                if isinstance(n.op,ast.Mult):return a*b
            if isinstance(n,ast.Call) and not n.keywords:
                if isinstance(n.func,ast.Name) and n.func.id in functions:
                    return call(n.func.id,[expr(x) for x in n.args],depth+1)
                if isinstance(n.func,ast.Attribute) and n.func.attr=="rotate" and len(n.args)==1:
                    return np.roll(expr(n.func.value),-int(expr(n.args[0])))
            raise NotImplementedError("Arithmetic-helper probe syntax")
        for stmt in fn.body:
            if isinstance(stmt,ast.Assign) and len(stmt.targets)==1 and isinstance(stmt.targets[0],ast.Name):
                env[stmt.targets[0].id]=expr(stmt.value)
            elif isinstance(stmt,ast.Return):return expr(stmt.value)
            else:raise NotImplementedError("Arithmetic-helper probe statement")
        raise ValueError("Missing return")
    fn=functions["golden"]
    result=call("golden",[provided[p.arg] for p in fn.args.args],0)
    return result if isinstance(result,list) else [result]
