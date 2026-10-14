"""Developer-only exact factor witness. Never applied to Provider responses."""
import ast,copy,math
from fractions import Fraction

def compact(source,request):
    def scalar(n):
        if type(n)is ast.Name:
            value=request["public_constants"].get(n.id)
        elif type(n)is ast.Constant:value=n.value
        else:return None
        return float(value)if type(value)in(int,float)and value!=0 and math.isfinite(value)and math.frexp(abs(value))[0]==.5 else None
    tree=ast.parse(source);merges=0
    for fn in tree.body:
        assert type(fn)is ast.FunctionDef
        definitions={}
        for statement in fn.body:
            if type(statement)is not ast.Assign:continue
            assert len(statement.targets)==1 and type(statement.targets[0])is ast.Name
            v=statement.value
            if type(v)is ast.BinOp and type(v.op)is ast.Mult and type(v.left)is ast.Name:
                prior=definitions.get(v.left.id)
                if type(prior)is ast.BinOp and type(prior.op)is ast.Mult:
                    a,b=scalar(prior.right),scalar(v.right)
                    if a is not None and b is not None:
                        product=a*b
                        if (product!=0 and math.isfinite(product)and abs(product)<=1024 and
                            Fraction.from_float(product)==Fraction.from_float(a)*Fraction.from_float(b)):
                            statement.value=ast.BinOp(left=copy.deepcopy(prior.left),op=ast.Mult(),right=ast.Constant(value=product))
                            merges+=1
            definitions[statement.targets[0].id]=statement.value
    assert merges>0
    return ast.unparse(ast.fix_missing_locations(tree))+"\n",merges
