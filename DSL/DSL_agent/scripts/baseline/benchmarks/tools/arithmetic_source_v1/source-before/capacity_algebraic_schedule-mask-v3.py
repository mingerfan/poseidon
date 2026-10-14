"""Bounded algebraic scheduling of trusted capacity DSL, never Agent candidates."""
import ast
import copy
from benchmark_graph import require
from unified_graph_scheduling import Schedule
VERSION="capacity-algebraic-schedule-v3"
NEUTRAL_VERSION="capacity-bound-identities-v1"

def remove_neutral_factors(source,request):
    """Remove exact bound ones from trusted baseline DSL; never Provider source.

    Only complete-period vectors are identities. A short or padded vector is
    still a mask. No public products, new constants, rotations or zero
    ciphertexts are introduced, and no helper is expanded.
    """
    period=request["layout"]["input_slot_period"]
    constants=request["public_constants"]
    scalar={name for name,value in constants.items()
            if type(value) in (int,float) and value==1}
    vectors={name for name,value in constants.items()
             if type(value) is list and len(value)==period and
             all(type(v) in (int,float) and v==1 for v in value)}
    identities=scalar|vectors
    record=dict(version=NEUTRAL_VERSION,applied=False,removed_multiplications=0,
        scalar_multiplications=0,vector_multiplications=0,bound_names=[],
        request_changed=False,reference_changed=False,provider_candidate_rewritten=False)
    if not identities:return source,record
    require(len(source.encode())<=65536,"Rule source budget")
    tree=ast.parse(source)
    require(tree.body and all(isinstance(f,ast.FunctionDef) for f in tree.body) and
            any(f.name=="golden" for f in tree.body),"Trusted scalar functions")
    # The generated helpers have ciphertext formals and SSA locals. Reject
    # shadowing instead of treating a local variable as a bound public value.
    for fn in tree.body:
        args=fn.args.posonlyargs+fn.args.args+fn.args.kwonlyargs
        args+=([fn.args.vararg] if fn.args.vararg else [])+([fn.args.kwarg] if fn.args.kwarg else [])
        require(not any(a.arg in constants for a in args) and fn.name not in constants and
                not any(isinstance(n,ast.Name) and isinstance(n.ctx,(ast.Store,ast.Del)) and
                        n.id in constants for n in ast.walk(fn)),
                "Trusted public identity binding shadowed")
    used=set()
    class Remove(ast.NodeTransformer):
        def visit_BinOp(self,node):
            node=self.generic_visit(node)
            if (isinstance(node.op,ast.Mult) and isinstance(node.right,ast.Name) and
                    node.right.id in identities and
                    not(isinstance(node.left,ast.Name) and node.left.id in constants)):
                name=node.right.id;used.add(name)
                record["removed_multiplications"]+=1
                record["scalar_multiplications" if name in scalar else "vector_multiplications"]+=1
                return ast.copy_location(node.left,node)
            return node
    tree=Remove().visit(tree)
    if not used:return source,record
    ast.fix_missing_locations(tree)
    result=ast.unparse(tree)+"\n"
    require(len(result.encode())<=65536,"Rule source budget")
    record.update(applied=True,bound_names=sorted(used))
    return result,record


def scalar_helpers(source):
    tree=ast.parse(source)
    functions={f.name:f for f in tree.body if isinstance(f,ast.FunctionDef)}
    require(len(functions)==len(tree.body) and "golden" in functions, "Trusted scalar functions")
    entry=functions["golden"];body=[];count=0;stack=[]
    def assign(value):
        nonlocal count
        require(count<1024,"Capacity scalar expansion budget")
        name="v"+str(count);count+=1
        body.append(ast.Assign(targets=[ast.Name(id=name,ctx=ast.Store())],value=value))
        return ast.Name(id=name,ctx=ast.Load())
    def expr(value,env,depth=0):
        require(depth<64,"Capacity scalar expression depth")
        if isinstance(value,ast.Name):return copy.deepcopy(env.get(value.id,value))
        if (isinstance(value,ast.Call) and isinstance(value.func,ast.Name)
                and value.func.id in functions):
            fn=functions[value.func.id]
            require(fn.name!="golden" and fn.name not in stack,"Capacity recursive scalar helper")
            require(not value.keywords and len(value.args)==len(fn.args.args),"Capacity scalar helper arguments")
            args=[expr(v,env,depth+1) for v in value.args]
            require(not fn.args.vararg and not fn.args.kwarg and not fn.args.kwonlyargs and not fn.args.defaults,
                    "Capacity static scalar signature")
            local={a.arg:v for a,v in zip(fn.args.args,args)}
            stack.append(fn.name)
            result=statements(fn,local,depth+1)
            stack.pop()
            return result
        result=copy.deepcopy(value)
        for field,child in ast.iter_fields(result):
            if isinstance(child,ast.AST):setattr(result,field,expr(child,env,depth+1))
            elif isinstance(child,list):
                setattr(result,field,[expr(v,env,depth+1) if isinstance(v,ast.AST) else v for v in child])
        return result
    def statements(fn,env,depth):
        require(fn.body and isinstance(fn.body[-1],ast.Return),"Capacity scalar return")
        for s in fn.body[:-1]:
            require(isinstance(s,ast.Assign) and len(s.targets)==1 and isinstance(s.targets[0],ast.Name),
                    "Capacity scalar assignments")
            env[s.targets[0].id]=assign(expr(s.value,env,depth))
        return expr(fn.body[-1].value,env,depth)
    returned=statements(entry,{a.arg:ast.Name(id=a.arg,ctx=ast.Load()) for a in entry.args.args},0)
    require(isinstance(returned,ast.List),"Capacity scalar output list")
    entry=copy.deepcopy(entry);entry.body=body+[ast.Return(value=returned)]
    result=ast.Module(body=[entry],type_ignores=[]);ast.fix_missing_locations(result)
    return ast.unparse(result)+"\n",len(functions)-1

def has_squared_feedback(graph):
    """Identify y*(public_offset + term(square(y))) by graph dependencies."""
    producers={out:node for node in graph["nodes"] for out in node["outputs"]}
    for node in graph["nodes"]:
        if node["op"]!="multiply":continue
        for y,correction in (node["inputs"],list(reversed(node["inputs"]))):
            add=producers.get(correction)
            if add is None or add["op"] not in ("add","subtract"):continue
            if not any(r in graph["constants"] for r in add["inputs"]):continue
            todo=list(add["inputs"]);seen=set()
            while todo and len(seen)<64:
                value=todo.pop()
                if value in seen:continue
                seen.add(value);part=producers.get(value)
                if part is None:continue
                if part["op"]=="square" and part["inputs"]==[y]:return True
                if part["op"] in ("add","subtract","multiply"):todo.extend(part["inputs"])
    return False

class BalancedSchedule(Schedule):
    def __init__(self,request):
        super().__init__(request,binary_masks=True,neutral_factors=True);self.balanced_products=0;self.distributed_offsets=0
        self.nested_mask_removals=0;self.disjoint_mask_removals=0
    def factor(self,n,p,nesting=0):
        """Coalesce adjacent exact bound masks at the same slot coordinates.

        Only the trusted rule baseline is scheduled. Requests, bound constants
        and Provider candidates are preserved. Disjoint products require the
        existing trusted-client encrypted-zero declaration.
        """
        require(not n.plain and p.plain,"Ciphertext/plaintext factor types")
        require(nesting<256,"Trusted schedule recursion budget")
        key=(n.index,p.index)
        if key in self.factors:return self.factors[key]
        if (self.mask(p) and not self.identity(p) and n.op=="multiply" and
                n.inputs[1] is not p and self.mask(n.inputs[1])):
            previous=self.request["public_constants"][n.inputs[1].value]
            target=self.request["public_constants"][p.value]
            # Both masks are complete-period, exact numeric 0/1 vectors.
            # Rotations cannot cross this rule.
            if all(a<=b for a,b in zip(previous,target)):
                result=n;self.nested_mask_removals+=1
            elif all(b<=a for a,b in zip(previous,target)):
                result=self.factor(n.inputs[0],p,nesting+1);self.nested_mask_removals+=1
            elif self.trusted_zero and all(not(a and b)for a,b in zip(previous,target)):
                result=self.leaf("zero_ct");self.disjoint_mask_removals+=1
            else:return super().factor(n,p,nesting)
            self.visit_count+=1
            require(self.visit_count<=8192,"Trusted schedule work budget")
            self.factors[key]=result
            return result
        return super().factor(n,p,nesting)

    def balanced_product(self,left,right):
        factors=[];todo=[right,left]
        while todo:
            n=todo.pop()
            if n.op=="multiply" and all(not c.plain for c in n.inputs):
                todo.extend(reversed(n.inputs))
            else:factors.append(n)
            if len(factors)+len(todo)>16:return self.node("multiply",(left,right))
        if len(factors)<3:return self.node("multiply",(left,right))
        old=self.node("multiply",(left,right))
        while len(factors)>1:
            factors.sort(key=lambda n:(n.depth,n.index))
            a,b=factors[:2];factors=factors[2:]+[self.node("multiply",(a,b))]
        result=factors[0]
        if result is not old:self.balanced_products+=1
        return result if result.depth<=old.depth else old
    def product(self,left,right):
        if right.plain or self.encrypted_public(left) is not None or self.encrypted_public(right) is not None:
            return super().product(left,right)
        def repeated_factor(value,other):
            todo=[value];seen=set()
            while todo and len(seen)<64:
                n=todo.pop()
                if n is other:return True
                if n.index in seen:continue
                seen.add(n.index)
                if n.op=="multiply":todo.extend(n.inputs)
            return False
        for value,other in ((right,left),(left,right)):
            if (value.op in ("add","subtract") and not value.inputs[0].plain and value.inputs[1].plain
                    and repeated_factor(value.inputs[0],other)):
                self.distributed_offsets+=1
                first=self.balanced_product(other,value.inputs[0])
                second=self.factor(other,value.inputs[1])
                return self.addition(value.op,first,second)
        return self.balanced_product(left,right)

def schedule(source,request):
    flattened,helpers=scalar_helpers(source)
    old=Schedule(request);previous=old.parse(flattened,rewrite=False)
    new=BalancedSchedule(request);outputs=new.parse(flattened,rewrite=True)
    old_depth=max(n.depth for n in previous);depth=max(n.depth for n in outputs)
    record=dict(version=VERSION,applied=False,original_depth_bound=old_depth,scheduled_depth_bound=depth,
        scalar_helpers=helpers,balanced_products=new.balanced_products,distributed_public_offsets=new.distributed_offsets,
        reused_encrypted_public_values=new.encrypted_public_products,
        removed_nested_bound_masks=new.nested_mask_removals,
        removed_disjoint_bound_masks=new.disjoint_mask_removals,
        removed_identity_factors=new.neutral_factor_removals,removed_duplicate_masks=new.idempotent_masks,
        moved_plaintext_factors=new.moves,distributed_bound_masks=new.distributions,
        hoisted_power_of_two_factors=new.power_two_hoists,factored_common_ciphertext_products=new.common_products,
        request_changed=False,reference_changed=False,
        provider_candidate_rewritten=False,depth_is_capacity_proof=False)
    changes=(new.balanced_products,new.distributed_offsets,new.encrypted_public_products,
             new.neutral_factor_removals,new.idempotent_masks,new.moves,new.distributions,
             new.power_two_hoists,new.common_products,new.nested_mask_removals,new.disjoint_mask_removals)
    if depth>old_depth or not any(changes):
        record["reason"]="No non-increasing trusted multiplication schedule"
        return source,record
    result,operations=new.source(flattened,outputs)
    record.update(applied=True,emitted_operations=operations)
    return result,record
