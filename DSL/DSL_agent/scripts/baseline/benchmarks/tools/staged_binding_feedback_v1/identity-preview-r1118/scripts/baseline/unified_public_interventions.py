"""Bounded control/state interventions, never evaluation of candidate Python."""
import ast
import copy

def add_one(value):
    return ast.BinOp(left=copy.deepcopy(value),op=ast.Add(),right=ast.Constant(1))

def replacements(target,feature,tree):
    out=[]
    names={n.id for n in ast.walk(tree) if type(n) is ast.Name}|{n.arg for n in ast.walk(tree) if type(n) is ast.arg};i=0
    while "discardProbe"+str(i) in names:i+=1
    discard=ast.Name(id="discardProbe"+str(i),ctx=ast.Store())
    def assign(value):return ast.Assign(targets=[copy.deepcopy(discard)],value=value)
    def simple_index(n):
        return all(type(x) in (ast.Constant,ast.Tuple,ast.Slice,ast.UnaryOp,ast.USub,ast.UAdd,ast.Load) for x in ast.walk(n))
    # Resource counters denote interpreter events, never user variables named
    # after a counter. Perturb the event's result/write while preserving type,
    # shape and argument evaluation; malformed interventions cannot earn credit.
    if type(target) is ast.Call and feature in ("call.values", "counter.mapping_views"):
        if type(target.func) is ast.Attribute and target.func.attr == "values":
            cell = discard.id
            out.append(ast.parse("["+cell+"+1 for "+cell+" in ("+ast.unparse(target)+")]", mode="eval").body)
            return out
        if type(target.func) is ast.Attribute and target.func.attr == "items":
            cell = discard.id
            out.append(ast.parse("[("+cell+"[0],"+cell+"[1]+1) for "+cell+" in ("+ast.unparse(target)+")]", mode="eval").body)
            return out
    if feature == "counter.sort_key_calls" and type(target) is ast.Call:
        if type(target.func) is ast.Name and target.func.id == "sorted":
            for j, keyword in enumerate(target.keywords):
                if keyword.arg == "key" and type(keyword.value) is ast.Lambda:
                    changed = copy.deepcopy(target)
                    key = changed.keywords[j].value
                    key.body = ast.UnaryOp(op=ast.USub(), operand=key.body)
                    out.append(changed)
            return out
    if feature == "counter.sort_key_calls" and type(target) is ast.Lambda:
        changed = copy.deepcopy(target)
        changed.body = ast.UnaryOp(op=ast.USub(), operand=changed.body)
        out.append(changed)
        return out
    if feature in ("counter.mapping_writes", "counter.slice_writes", "slice.write") and type(target) is ast.Assign:
        if len(target.targets) == 1 and type(target.targets[0]) is ast.Subscript:
            changed = copy.deepcopy(target)
            changed.value = add_one(target.value)
            out.append(changed)
            changed = copy.deepcopy(target)
            cell = discard.id
            changed.value = ast.parse("["+cell+"+1 for "+cell+" in ("+ast.unparse(target.value)+")]", mode="eval").body
            out.append(changed)
    if feature == "counter.mapping_writes" and type(target) is ast.Expr and type(target.value) is ast.Call:
        return [ast.Expr(value=value) for value in replacements(target.value,feature,tree)]
    if feature == "counter.mapping_writes" and type(target) is ast.Call:
        if type(target.func) is ast.Attribute and target.func.attr == "update":
            for j, arg in enumerate(target.args):
                if type(arg) is not ast.Dict:continue
                changed = copy.deepcopy(target)
                # Keep all keys and evaluate each original value exactly once,
                # in the same order. No wholesale deletion producing missing keys.
                changed.args[j].values = [add_one(v) for v in changed.args[j].values]
                out.append(changed)
            return out
    if feature == "counter.starred_expansions" and type(target) is ast.Call:
        feature = "expansion.star"
    # Result interventions retain symbolic cells and evaluate the call once.
    # They do not change arguments or credit an unused call's side effects.
    if type(target) is ast.Call and feature in ("call.enumerate","call.items"):
        cell=discard.id
        text="[("+cell+"[0],"+cell+"[1]+1) for "+cell+" in ("+ast.unparse(target)+")]"
        out.append(ast.parse(text,mode="eval").body)
        return out
    if (type(target) is ast.Call and feature=="counter.sorted_calls" and
            type(target.func) is ast.Name and target.func.id=="sorted"):
        out.append(ast.parse("list(reversed("+ast.unparse(target)+"))",mode="eval").body)
        return out
    if feature.startswith('lambda.'):
        if type(target) is not ast.Lambda:return []
        field={'lambda.posonly':'posonlyargs','lambda.kwonly':'kwonlyargs',
               'lambda.vararg':'vararg','lambda.kwarg':'kwarg'}.get(feature)
        parameters=getattr(target.args,field,None) if field else None
        parameters=parameters if type(parameters) is list else [parameters] if parameters else []
        for parameter in parameters[:8]:
            name=parameter.arg
            if field=='vararg':
                texts=['tuple(['+discard.id+'+1 for '+discard.id+' in '+name+'])',
                       'tuple(['+discard.id+'[1]+('+discard.id+'[0]+1) for '+discard.id+' in enumerate('+name+')])',
                       'tuple(['+discard.id+'*2 for '+discard.id+' in '+name+'])']
                values=[ast.parse(text,mode='eval').body for text in texts]
            elif field=='kwarg':
                texts=['{'+discard.id+':'+name+'['+discard.id+']+1 for '+discard.id+' in '+name+'}',
                       '{'+discard.id+'[1]:'+name+'['+discard.id+'[1]]+('+discard.id+'[0]+1) for '+discard.id+' in enumerate('+name+')}',
                       '{'+discard.id+':'+name+'['+discard.id+']*2 for '+discard.id+' in '+name+'}']
                values=[ast.parse(text,mode='eval').body for text in texts]
            else:values=[add_one(ast.Name(id=name,ctx=ast.Load())),
                         ast.BinOp(left=ast.Name(id=name,ctx=ast.Load()),op=ast.Mult(),right=ast.Constant(2))]
            for value in values:
                changed=copy.deepcopy(target)
                # Shadow only this already-bound value inside the expression.
                # Original defaults/arguments still evaluate once, in original order.
                # No whole-body result perturbation: an ignored parameter gives no credit.
                inner=ast.Lambda(args=ast.arguments(posonlyargs=[],args=[ast.arg(arg=name)],
                    vararg=None,kwonlyargs=[],kw_defaults=[],kwarg=None,defaults=[]),body=copy.deepcopy(target.body))
                changed.body=ast.Call(func=inner,args=[value],keywords=[])
                out.append(changed)
        return out
    if feature.startswith("signature.") and type(target) is not ast.FunctionDef:
        # Lambda parameters need a separate binding intervention. Never credit
        # changing the lambda body when the parameter itself was ignored.
        return []
    if type(target) is ast.FunctionDef and feature.startswith("signature."):
        # Perturb the bound value at function entry, before the body can overwrite it.
        # Editing arbitrary Name loads could falsely credit a later local assignment.
        field={"signature.posonly":"posonlyargs","signature.kwonly":"kwonlyargs",
               "signature.vararg":"vararg","signature.kwarg":"kwarg"}.get(feature)
        parameters=getattr(target.args,field,None) if field else None
        parameters=parameters if type(parameters) is list else [parameters] if parameters else []
        for parameter in parameters[:8]:
            name=parameter.arg
            if field=="vararg":
                value=ast.parse("tuple(["+discard.id+"+1 for "+discard.id+" in "+name+"])",mode="eval").body
            elif field=="kwarg":
                # One comprehension variable; no collision with another temporary.
                value=ast.parse("{"+discard.id+":"+name+"["+discard.id+"]+1 for "+discard.id+" in "+name+"}",mode="eval").body
            else:value=add_one(ast.Name(id=name,ctx=ast.Load()))
            changed=copy.deepcopy(target)
            changed.body.insert(0,ast.Assign(targets=[ast.Name(id=name,ctx=ast.Store())],value=value))
            out.append(changed)
    elif type(target) is ast.Call and feature in ("expansion.star","expansion.kwstar"):
        # Only expansion contents change; original values/side effects are evaluated
        # once by the comprehension's iterable, in the original argument position.
        if feature=="expansion.star":
            for j,arg in enumerate(target.args):
                if type(arg) is not ast.Starred:continue
                changed=copy.deepcopy(target)
                changed.args[j].value=ast.ListComp(elt=add_one(ast.Name(id=discard.id,ctx=ast.Load())),
                    generators=[ast.comprehension(target=copy.deepcopy(discard),
                                 iter=copy.deepcopy(arg.value),ifs=[],is_async=0)])
                out.append(changed)
                # zip(*rows) expands sequence arguments, not scalar cells.
                # Each original iterable is evaluated once; only its values
                # change. This is a bounded rank-two alternative.
                changed = copy.deepcopy(target)
                row = discard.id
                cell = row+"Cell"
                while cell in names:cell += "Cell"
                changed.args[j].value = ast.parse("[["+cell+"+1 for "+cell+" in "+row+"] for "+row+
                    " in ("+ast.unparse(arg.value)+")]",mode="eval").body
                out.append(changed)
        else:
            for j,keyword in enumerate(target.keywords):
                if keyword.arg is not None:continue
                changed=copy.deepcopy(target)
                # items() evaluates the expansion mapping once. The tuple carries
                # both key and value, preserving order without re-reading the map.
                changed.keywords[j].value=ast.DictComp(
                    key=ast.Subscript(value=ast.Name(id=discard.id,ctx=ast.Load()),slice=ast.Constant(0),ctx=ast.Load()),
                    value=add_one(ast.Subscript(value=ast.Name(id=discard.id,ctx=ast.Load()),slice=ast.Constant(1),ctx=ast.Load())),
                    generators=[ast.comprehension(target=copy.deepcopy(discard),
                        iter=ast.Call(func=ast.Attribute(value=copy.deepcopy(keyword.value),attr="items",ctx=ast.Load()),args=[],keywords=[]),
                        ifs=[],is_async=0)])
                out.append(changed)
    elif type(target) in (ast.FunctionDef,ast.Lambda):
        if feature=="counter.default_evaluations":
            for field in ("defaults","kw_defaults"):
                for j,value in enumerate(getattr(target.args,field)):
                    if value is None:continue
                    changed=copy.deepcopy(target)
                    getattr(changed.args,field)[j]=add_one(value)
                    out.append(changed)
        elif type(target) is ast.Lambda:
            changed=copy.deepcopy(target);changed.body=add_one(target.body);out.append(changed)
        else:
            class Returns(ast.NodeTransformer):
                def visit_Return(self,node):
                    return ast.copy_location(ast.Return(value=add_one(node.value)),node) if node.value is not None else node
                def visit_FunctionDef(self,node):return node
                def visit_Lambda(self,node):return node
            changed=copy.deepcopy(target)
            changed.body=[Returns().visit(n) for n in changed.body];out.append(changed)
    elif type(target) in (ast.Break,ast.Continue):out.append(ast.Pass())
    elif type(target) in (ast.For,ast.While):
        changed=copy.deepcopy(target)
        if feature in ("event.for_else","event.while_else"):
            changed.orelse=[ast.Pass()]
        elif type(target) is ast.For:
            changed.iter=ast.Subscript(value=changed.iter,slice=ast.Slice(upper=ast.UnaryOp(op=ast.USub(),operand=ast.Constant(1))),ctx=ast.Load())
        else:
            changed.test=ast.Subscript(value=ast.List(elts=[changed.test,ast.Constant(False)],ctx=ast.Load()),slice=ast.Constant(1),ctx=ast.Load())
        out.append(changed)
    elif type(target) is ast.AugAssign:
        changed=copy.deepcopy(target);changed.op=ast.Sub() if type(target.op) is ast.Add else ast.Add();out.append(changed)
    elif type(target) is ast.Assign and len(target.targets)==1:
        dst=target.targets[0]
        if type(dst) is ast.Subscript and type(dst.value) is ast.Name and simple_index(dst.slice):
            # RHS still runs once; only the write to a side-effect-free target is suppressed.
            out.append(assign(copy.deepcopy(target.value)))
        elif type(dst) is ast.Name and feature=="counter.nonlocal_writes":
            changed=copy.deepcopy(target);changed.value=add_one(target.value);out.append(changed)
    elif type(target) is ast.Expr and type(target.value) is ast.Call:
        call=target.value
        if type(call.func) is ast.Attribute and call.func.attr in ("clear","update") and type(call.func.value) is ast.Name and not any(type(a) is ast.Starred for a in call.args) and all(k.arg is not None for k in call.keywords):
            # Preserve receiver/argument evaluation order exactly once, skip only method mutation.
            out.append(assign(ast.List(elts=[copy.deepcopy(call.func.value),*copy.deepcopy(call.args),
                *[copy.deepcopy(k.value) for k in call.keywords]],ctx=ast.Load())))
    return out
