"""Distinct final expressions for the trusted v9 rule baseline only."""
import ast,hashlib,json
from benchmark_graph import require
VERSION="capacity-rule-distinct-outputs-v1"

def distinct_outputs(source):
    tree=ast.parse(source)
    functions=[n for n in tree.body if type(n) is ast.FunctionDef and n.name=="golden"]
    require(len(functions)==1,"Rule output function")
    fn=functions[0]
    require(fn.body and type(fn.body[-1]) is ast.Return and type(fn.body[-1].value) in (ast.List,ast.Tuple),
            "Rule output list")
    require(any(a.arg=="zero_ct" for a in fn.args.args),"Bound encrypted zero input")
    targets={}
    for node in fn.body[:-1]:
        if type(node) is ast.Assign and len(node.targets)==1 and type(node.targets[0]) is ast.Name:
            targets.setdefault(node.targets[0].id,[]).append(node.value)
    aliases={k:v[0] for k,v in targets.items() if len(v)==1}
    cached={};work=0
    def key(node,trail=()):
        nonlocal work
        work+=1;require(work<=30000,"Rule output identity work bound")
        if type(node) is ast.Name and type(node.ctx) is ast.Load and node.id in aliases:
            require(node.id not in trail,"Rule output alias cycle")
            if node.id not in cached:cached[node.id]=key(aliases[node.id],trail+(node.id,))
            return cached[node.id]
        if isinstance(node,ast.AST):
            fields=[type(node).__name__]+[(name,key(value,trail)) for name,value in ast.iter_fields(node)]
            return hashlib.sha256(json.dumps(fields,separators=(",",":"),allow_nan=False).encode()).hexdigest()
        if type(node) is list:return [key(v,trail) for v in node]
        return [type(node).__name__,repr(node)]
    values=fn.body[-1].value.elts;seen=set();additions=[]
    for index,value in enumerate(values):
        identity=key(value);count=0
        while identity in seen:
            value=ast.BinOp(left=value,op=ast.Add(),right=ast.Name(id="zero_ct",ctx=ast.Load()))
            count+=1;require(count<=len(values),"Rule output identity budget")
            identity=key(value)
        seen.add(identity);values[index]=value
        if count:additions.append(dict(output_chunk=index,encrypted_zero_additions=count))
    if not additions:return source,None
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)+"\n",dict(version=VERSION,scope="rule_baseline_only",bindings=additions)
