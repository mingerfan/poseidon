"""Local compaction for caller-attested deterministic rule source only.

This staging helper is not registered on a Provider candidate/repair path.
It parses source as data and never executes candidate code. Real request/native
checks and backend qualification remain mandatory in the controlled caller.
"""
import ast
import copy
import hashlib
import json
import math
import re
from pathlib import Path

VERSION="poseidon-trusted-bound-mask-compaction-v1"
FORMAT="poseidon-trusted-bound-mask-compaction-record-v1"
SOURCE_KIND="deterministic_rule_baseline"

def require(condition,message):
    if not condition:raise ValueError(message)

def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode("utf8")

def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()

def implementation_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

def _functions(tree,public,input_names):
    require(type(tree) is ast.Module and 1<=len(tree.body)<=17 and
            all(type(fn) is ast.FunctionDef for fn in tree.body),"Trusted rule functions required")
    functions={fn.name:fn for fn in tree.body}
    require(len(functions)==len(tree.body) and "golden" in functions,"Unique trusted golden entry")
    for fn in functions.values():
        require(fn.name not in public,"Public binding function shadow")
        args=fn.args
        require(not args.posonlyargs and not args.vararg and not args.kwarg and
                not args.kwonlyargs and not args.defaults and not args.kw_defaults and
                fn.returns is None and all(a.annotation is None for a in args.args),
                "Trusted fixed scalar signature")
        names=[a.arg for a in args.args]
        require(len(names)==len(set(names)) and not set(names)&public,"Public binding argument shadow")
        require(len(fn.decorator_list)==1,"Trusted rule decorator")
        call=fn.decorator_list[0]
        require(type(call) is ast.Call and type(call.func) is ast.Attribute and
                type(call.func.value) is ast.Name and call.func.value.id=="hc" and
                call.func.attr=="func" and len(call.args)==1 and not call.keywords and
                type(call.args[0]) is ast.Constant and type(call.args[0].value) is str,
                "Trusted rule decorator")
        kinds=call.args[0].value.split(",")
        require(len(kinds)==len(names) and all(k in ("c","p") for k in kinds),
                "Trusted rule scalar types")
        if fn.name=="golden":
            require(names==input_names and all(k=="c" for k in kinds),"Trusted golden layout binding")
        require(fn.body and type(fn.body[-1]) is ast.Return and
                all(type(s) is ast.Assign and len(s.targets)==1 and type(s.targets[0]) is ast.Name
                    for s in fn.body[:-1]),"Trusted rule scalar statements")
        for node in ast.walk(fn):
            require(not isinstance(node,(ast.Global,ast.Nonlocal)),"Public global/nonlocal shadow")
            if isinstance(node,ast.arg):
                require(node.arg not in public,"Public binding nested argument shadow")
            require(not isinstance(node,(ast.Lambda,ast.NamedExpr,ast.ListComp,ast.SetComp,ast.DictComp,
                                         ast.GeneratorExp,ast.Await,ast.Yield,ast.YieldFrom)),
                    "Trusted rule scalar expression grammar")
            if isinstance(node,ast.Name) and isinstance(node.ctx,(ast.Store,ast.Del)):
                require(node.id not in public,"Public binding assignment shadow")
        occupied=set(names)|public
        for statement in fn.body[:-1]:
            target=statement.targets[0].id
            require(target not in occupied,"Trusted rule single assignment")
            occupied.add(target)
    return functions

def compact(source,request,*,source_kind):
    require(type(source_kind) is str and source_kind==SOURCE_KIND,"Rule-only source authority required")
    require(type(source) is str and 0<len(source.encode("utf8"))<=65536,"Trusted rule source budget")
    require(type(request) is dict and type(request.get("request_id")) is str and
            0<len(request["request_id"])<=256 and type(request.get("layout")) is dict and
            type(request.get("public_constants")) is dict,"Trusted immutable request binding")
    layout=request["layout"];public=request["public_constants"];period=layout.get("input_slot_period")
    require(type(period) is int and 1<=period<=16384 and period&(period-1)==0,
            "Trusted power-of-two period")
    require(all(type(n) is str and re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*",n) for n in public),
            "Trusted public symbol bindings")
    inputs=layout.get("inputs")
    require(type(inputs) is list and all(type(v) is dict and type(v.get("dsl_name")) is str for v in inputs),
            "Trusted physical input binding")
    input_names=[v["dsl_name"] for v in inputs]+["zero_ct"]
    tree=ast.parse(source)
    require(len(list(ast.walk(tree)))<=4096,"Trusted source AST budget")
    _functions(tree,set(public),input_names)
    masks={name for name,value in public.items() if type(value) is list and len(value)==period and
           all(type(v) in (int,float) and math.isfinite(v) and v in (0,1) for v in value)}
    used=set()
    class Fold(ast.NodeTransformer):
        removed=0
        def visit_BinOp(self,node):
            node=self.generic_visit(node)
            if (type(node.op) is ast.Mult and type(node.right) is ast.Name and node.right.id in masks and
                type(node.left) is ast.BinOp and type(node.left.op) is ast.Mult and
                type(node.left.right) is ast.Name and node.left.right.id==node.right.id):
                self.removed+=1;used.add(node.right.id)
                return ast.copy_location(node.left,node)
            return node
    fold=Fold();modified=fold.visit(copy.deepcopy(tree))
    result=ast.unparse(ast.fix_missing_locations(modified))+"\n" if fold.removed else source
    require(len(result.encode("utf8"))<=65536 and
            len(list(ast.walk(ast.parse(result))))<=len(list(ast.walk(tree))),
            "Compaction cannot enlarge source AST")
    static=dict(source_kind=SOURCE_KIND,request_id=request["request_id"],request_sha256=digest(request),
                layout_sha256=digest(layout),public_constants_sha256=digest(public),
                input_source_sha256=hashlib.sha256(source.encode("utf8")).hexdigest(),
                implementation_sha256=implementation_sha256())
    record=dict(format=FORMAT,version=VERSION,source_kind=SOURCE_KIND,applied=fold.removed>0,
                removed=fold.removed,bound_names=sorted(used),static_binding=static,
                static_binding_sha256=digest(static),
                output_source_sha256=hashlib.sha256(result.encode("utf8")).hexdigest(),
                request_changed=False,model_changed=False,reference_changed=False,
                provider_candidate_rewritten=False,new_constants=0,new_rotations=0,new_zero_ciphertexts=0,
                depth_is_capacity_proof=False)
    record["binding"]=digest(record)
    return result,record
