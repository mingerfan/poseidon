"""Bounded typed native-function core, enabled only by its distinct opt-in contract.

Candidate Python is parsed as data, never eval/exec/compiled into Python code.
This staged core preserves native call boundaries rather than stripping decorators.
Existing v22 construction remains separate until its composition is verified.
"""
import ast
import copy
import hashlib
import math

from hecate_contract import IDENTIFIER, rotation_literal
from seal_artifact_gate import require
import native_array_core as storage

CONTRACT = 'decorated-functions-core-v1'
ROTATIONS = (-3, -2, -1, 1, 2, 3)
LIMIT = 4096


def literal(node):
    sign = 1
    if type(node) is ast.UnaryOp and type(node.op) in (ast.USub, ast.UAdd):
        sign = -1 if type(node.op) is ast.USub else 1
        node = node.operand
    if type(node) is ast.Constant and type(node.value) in (int, float):
        require(abs(node.value) <= 1024 and math.isfinite(node.value), 'Numeric literal outside finite bound')
        return sign*node.value
    return None


def cells(value):
    return value[1] if type(value) is tuple else (value,)


def _analyze(source, constants, expected_outputs, input_names, *, arrays=False, starred_calls=False, array_arithmetic=False, public_loops=False, scalar_augmented=False, array_mutation=False, slot_period=None):
    packed = slot_period is not None
    if packed:
        from packed_input_abi import rotations
        allowed_rotations = rotations(slot_period)
    else:
        allowed_rotations = ROTATIONS
    if array_mutation:
        import native_array_alias as storage
    else:
        import native_array_core as storage
    scalar_augmented = scalar_augmented or array_mutation
    public_loops = public_loops or scalar_augmented
    array_arithmetic = array_arithmetic or public_loops
    starred_calls = starred_calls or array_arithmetic
    arrays = arrays or starred_calls
    require(type(source) is str and len(source.encode()) <= 65536, 'Source size limit')
    require(type(expected_outputs) is int and 1 <= expected_outputs <= (16 if packed else 4), 'Output count limit')
    require(type(input_names) in (list, tuple) and 1 <= len(input_names) <= 5 and
            all(type(n) is str and IDENTIFIER.fullmatch(n) and n not in ('hc','golden') for n in input_names) and
            len(set(input_names)) == len(input_names), 'Invalid input names')
    require(type(constants) is dict and len(constants) <= (256 if packed else 128), 'Constant registry limit')
    reserved = {'hc','np','object'} if arrays else {'hc'}
    if public_loops: reserved.add('range')
    require(not (set(input_names) & reserved), 'Reserved native input name')
    for name, value in constants.items():
        require(type(name) is str and IDENTIFIER.fullmatch(name) and name not in reserved | {'golden'} and
                name not in input_names, 'Invalid constant name')
        data = value if type(value) is list else [value]
        require(len(data) in (1,slot_period if packed else 4) and all(type(v) in (int,float) and abs(v) <= 1024 and math.isfinite(v)
                for v in data), 'Constants must be finite scalar or selected-period real data')
    try:
        tree = ast.parse(source, type_comments=True)
    except (SyntaxError, RecursionError, MemoryError) as error:
        raise ValueError('Invalid native function syntax') from error
    require(len(list(ast.walk(tree))) <= LIMIT, 'AST node limit')
    loop_plan = None
    if public_loops:
        from native_public_loops import expand
        tree,loop_plan = expand(tree,constants,scalar_augmented=scalar_augmented)
    require(1 <= len(tree.body) <= 17 and all(type(n) is ast.FunctionDef for n in tree.body),
            'Only native function declarations allowed')
    nodes = {n.name:n for n in tree.body}
    require(len(nodes) == len(tree.body) and 'golden' in nodes, 'Duplicate or missing function')
    signatures = {}
    for name, node in nodes.items():
        require(IDENTIFIER.fullmatch(name) and name not in reserved and name not in constants and
                name not in input_names, 'Invalid or colliding function name')
        a = node.args
        require(not a.posonlyargs and not a.vararg and not a.kwarg and not a.kwonlyargs and
                not a.defaults and not a.kw_defaults and not node.returns and not node.type_comment and
                all(p.annotation is None for p in a.args), 'Native core parameters must be unannotated positional names')
        names = [p.arg for p in a.args]
        require(len(names) <= 16 and len(set(names)) == len(names) and
                all(IDENTIFIER.fullmatch(n) and n not in nodes and n not in reserved and n not in constants for n in names),
                'Invalid or colliding parameter name')
        require(len(node.decorator_list) == 1, 'Exactly one hc.func decorator required')
        d = node.decorator_list[0]
        require(type(d) is ast.Call and type(d.func) is ast.Attribute and d.func.attr == 'func' and
                type(d.func.value) is ast.Name and d.func.value.id == 'hc' and not d.keywords and
                len(d.args) == 1 and type(d.args[0]) is ast.Constant and type(d.args[0].value) is str,
                'Only literal hc.func signature allowed')
        signature = d.args[0].value
        kinds = tuple(signature.split(',')) if signature else ()
        require(len(kinds) == len(names) and all(k in ('c','p') for k in kinds), 'c/p signature mismatch')
        require(node.body and type(node.body[-1]) is ast.Return, 'Final return required')
        signatures[name] = dict(parameters=names, kinds=kinds, signature=signature)
    require(signatures['golden']['parameters'] == list(input_names) and
            signatures['golden']['kinds'] == ('c',)*len(input_names), 'Golden input ABI mismatch')
    active = set(); summaries = {}; order = []; augmentations = []; mutations = []; mutation_cost = 0

    def analyze(name):
        nonlocal mutation_cost
        require(name not in active, 'Recursive native call graph')
        if name in summaries: return summaries[name]
        require(len(active) < 17, 'Call depth limit')
        active.add(name)
        symbols = {n:'p' for n in constants}
        symbols.update(zip(signatures[name]['parameters'], signatures[name]['kinds']))
        edges = []; rotations = set(); cost = len(constants)

        def spend(amount=1):
            nonlocal cost
            cost += amount
            require(cost <= LIMIT, 'Expanded native IR resource limit')

        def expression(n, depth=0):
            require(depth <= 64, 'Expression depth limit')
            if literal(n) is not None:
                spend(); return 'p'
            if type(n) is ast.Name:
                require(type(n.ctx) is ast.Load and n.id in symbols, 'Undefined value')
                return symbols[n.id]
            if type(n) in (ast.List, ast.Tuple):
                require(type(n.ctx) is ast.Load and len(n.elts) <= 16, 'Flat container limit')
                items = tuple(expression(v, depth+1) for v in n.elts)
                require(arrays or all(v in ('c','p') for v in items), 'Return/storage containers must be flat scalar cells')
                return ('list' if type(n) is ast.List else 'tuple', items)
            if type(n) is ast.Subscript:
                value = expression(n.value, depth+1)
                if arrays and storage.is_array(value):
                    require(type(n.ctx) is ast.Load, 'Native array index must be a read')
                    spend()
                    return storage.subscript_type(value,n.slice)
                require(type(value) is tuple and type(n.ctx) is ast.Load, 'Indexing requires a result container')
                index = rotation_literal(n.slice)
                require(-len(value[1]) <= index < len(value[1]), 'Result index out of bounds')
                return value[1][index]
            if type(n) is ast.UnaryOp and type(n.op) is ast.USub:
                operand = expression(n.operand, depth+1)
                if array_arithmetic and storage.is_array(operand):
                    value = storage.arithmetic_type(operand)
                    spend(max(1,len(value[1])) if storage.is_array(value) else 1); return value
                require(operand == 'c', 'Native negation requires ciphertext')
                spend(); return 'c'
            if type(n) is ast.BinOp and type(n.op) in (ast.Add,ast.Sub,ast.Mult):
                a, b = expression(n.left,depth+1), expression(n.right,depth+1)
                if array_arithmetic and (storage.is_array(a) or storage.is_array(b)):
                    value = storage.arithmetic_type(a,b)
                    spend(max(1,len(value[1])) if storage.is_array(value) else 1); return value
                require(a in ('c','p') and b in ('c','p') and 'c' in (a,b),
                        'Native core arithmetic requires a ciphertext operand')
                spend(); return 'c'
            if type(n) is ast.Call:
                if arrays and storage.constructor(n):
                    storage.check_constructor(n)
                    value = storage.array_type(expression(n.args[0],depth+1))
                    spend(max(1,len(value[1])))
                    return value
                require(not n.keywords and (all(type(v) is not ast.Starred for v in n.args)
                        or starred_calls and type(n.func) is ast.Name), 'Positional calls only')
                if type(n.func) is ast.Name:
                    target = n.func.id
                    require(target in nodes and target != 'golden', 'Unknown or forbidden callee')
                    values = []
                    for argument in n.args:
                        if type(argument) is ast.Starred:
                            value = expression(argument.value, depth+1)
                            if storage.is_array(value):
                                require(value[2], 'Cannot unpack a zero-dimensional native array')
                                expanded = tuple(storage.subscript_type(value,ast.Constant(i)) for i in range(value[2][0]))
                            else:
                                require(type(value) is tuple and value[0] in ('list','tuple'),
                                        'Starred native arguments require list/tuple or array storage')
                                expanded = value[1]
                            require(all(v in ('c','p') for v in expanded),
                                    'Starred native arguments must yield Expr cells; no implicit flattening')
                            values.extend(expanded)
                        else:
                            values.append(expression(argument, depth+1))
                        require(len(values) <= 16, 'Expanded native argument count bound')
                    values = tuple(values)
                    require(values == signatures[target]['kinds'], 'Native call argument type/count mismatch')
                    callee = analyze(target); edges.append(target); rotations.update(callee['rotation_steps'])
                    spend(callee['expanded_cost']+1)
                    return storage.fresh(callee['result']) if array_mutation else callee['result']
                if arrays and type(n.func) is ast.Attribute and n.func.attr in storage.METHODS:
                    value = expression(n.func.value,depth+1)
                    require(storage.is_array(value),'Native storage method requires an object array')
                    spend()
                    return storage.method_type(value,n)
                require(type(n.func) is ast.Attribute and n.func.attr == 'rotate' and len(n.args) == 1,
                        'Only declared functions or ciphertext.rotate allowed')
                require(expression(n.func.value,depth+1) == 'c', 'Rotation requires ciphertext')
                step = rotation_literal(n.args[0]); require(step in allowed_rotations, 'Unsupported rotation step')
                rotations.add(step); spend(); return 'c'
            if arrays and type(n) is ast.Attribute and n.attr == 'T':
                value = expression(n.value,depth+1)
                require(storage.is_array(value),'Native transpose requires object storage')
                spend()
                return storage.result_type(value,storage.grid(value).T)
            raise ValueError('Unsupported native core expression: '+type(n).__name__)

        def assign(target, value):
            if type(target) in (ast.List, ast.Tuple):
                if arrays and storage.is_array(value):
                    require(value[2] and len(target.elts) == value[2][0], 'Native array unpacking first-axis mismatch')
                    value = ('tuple',tuple(storage.subscript_type(value,ast.Constant(i)) for i in range(value[2][0])))
                require(type(value) is tuple and len(target.elts) == len(value[1]) and
                        all(type(n) is ast.Name for n in target.elts), 'Flat unpacking mismatch')
                require(len({n.id for n in target.elts}) == len(target.elts), 'Duplicate unpacking target')
                for n,v in zip(target.elts,value[1]): assign(n,v)
                return
            require(type(target) is ast.Name and type(target.ctx) is ast.Store and
                    IDENTIFIER.fullmatch(target.id) and target.id not in constants and target.id not in nodes and
                    target.id not in reserved | {'zero_ct'}, 'Invalid assignment target')
            symbols[target.id] = value

        for statement in nodes[name].body[:-1]:
            if scalar_augmented and type(statement) is ast.AugAssign:
                require(type(statement.target) is ast.Name and type(statement.op) in (ast.Add,ast.Sub,ast.Mult),
                        'Only scalar name +=, -= and *= are supported')
                left=symbols.get(statement.target.id)
                right=expression(statement.value)
                if array_mutation and storage.is_array(left):
                    before=storage.bindings(symbols,left)
                    mutation_cost+=2*sum(1+len(r['kinds']) for r in before)
                    require(mutation_cost<=LIMIT,'Native mutation observation resource bound')
                    storage.inplace_type(left,right)
                    mutations.append(dict(function=name,target=statement.target.id,
                        operation={ast.Add:'add',ast.Sub:'subtract',ast.Mult:'multiply'}[type(statement.op)],
                        span=[statement.lineno,statement.col_offset,statement.end_lineno,statement.end_col_offset],
                        before=before,after=storage.bindings(symbols,left),same_array_object=True,
                        original_exprs_unchanged=True,changed_cells=storage.changed_cells(symbols,left)))
                    spend(max(1,len(left[1]))); assign(statement.target,left)
                    continue
                require(left in ('c','p') and right in ('c','p') and 'c' in (left,right),
                        'Augmented operands must be scalar Exprs with a ciphertext operand; array mutation is not enabled')
                augmentations.append(dict(function=name,target=statement.target.id,
                    operation={ast.Add:'add',ast.Sub:'subtract',ast.Mult:'multiply'}[type(statement.op)],
                    left_kind=left,right_kind=right,
                    span=[statement.lineno,statement.col_offset,statement.end_lineno,statement.end_col_offset]))
                spend(); assign(statement.target,'c')
            elif type(statement) is ast.Assign:
                require(len(statement.targets) == 1 and statement.type_comment is None, 'Single assignment required')
                assign(statement.targets[0], expression(statement.value))
            else:
                require(type(statement) is ast.Expr and type(statement.value) is ast.Call and
                        type(statement.value.func) is ast.Name, 'Only assignments and helper calls precede return')
                expression(statement.value)
        result = expression(nodes[name].body[-1].value)
        if array_mutation: result=storage.freeze(result)
        require(all(v in ('c','p') for v in cells(result)), 'Native returns require flat Expr cells or object arrays')
        summaries[name] = dict(signatures[name], result=result, calls=edges,
                               rotation_steps=sorted(rotations), expanded_cost=cost)
        active.remove(name); order.append(name)
        return summaries[name]

    for name in nodes: analyze(name)  # Reject invalid or recursive unused definitions too.
    outputs = cells(summaries['golden']['result'])
    require(len(outputs) == expected_outputs and all(v == 'c' for v in outputs), 'Golden ciphertext outputs mismatch')
    require(sum(v['expanded_cost'] for v in summaries.values()) <= LIMIT, 'Total native module resource limit')
    plan = dict(schema=1, contract='decorated-functions-core-v5' if public_loops else 'decorated-functions-core-v4' if array_arithmetic else 'decorated-functions-core-v3' if starred_calls else 'decorated-functions-core-v2' if arrays else CONTRACT, source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        functions=summaries, topological_order=order, rotation_steps=summaries['golden']['rotation_steps'],
        output_ciphertexts=len(outputs), compilation_checked=False, encrypted_correctness_checked=False,
        agent_contract_enabled=True, agent_generation_validated=False)
    if public_loops: plan['public_loops'] = loop_plan
    if scalar_augmented:
        plan['contract'] = 'decorated-functions-core-v6'
        plan['scalar_augmented'] = dict(schema=1,sites=augmentations,array_mutation_enabled=False)
    if array_mutation:
        plan['contract']='decorated-functions-core-v7'
        plan['scalar_augmented']['array_mutation_enabled']=True
        plan['array_mutation']=dict(schema=1,sites=mutations,subscript_writes_enabled=False)
    if packed:
        plan['packed_binding']=dict(slot_period=slot_period,rotation_steps=list(allowed_rotations),
                                    output_limit=16,constant_limit=256)
    return nodes, plan


def validate(source, constants, expected_outputs=1, *, input_names=('x',), arrays=False, starred_calls=False, array_arithmetic=False, public_loops=False, scalar_augmented=False, array_mutation=False, slot_period=None):
    return _analyze(source, constants, expected_outputs, input_names, arrays=arrays,starred_calls=starred_calls,array_arithmetic=array_arithmetic,public_loops=public_loops,scalar_augmented=scalar_augmented,array_mutation=array_mutation,slot_period=slot_period)[1]


def register(source, constants, frontend, expected_outputs=1, *, input_names=('x',), observe=None, arrays=False, observe_storage=None, starred_calls=False, array_arithmetic=False, public_loops=False, observe_starred=None, scalar_augmented=False, observe_augmented=None, array_mutation=False, observe_mutation=None, slot_period=None):
    """Trusted harness only: validate all code before touching the supplied native frontend.

    Each wrapper is trusted Python; candidate functions remain inert AST nodes.
    The caller owns sandboxing, input manifests, frontend lifetime and hc.save.
    """
    arrays = arrays or starred_calls or array_arithmetic or public_loops or scalar_augmented or array_mutation
    nodes, plan = _analyze(source, constants, expected_outputs, input_names, arrays=arrays,starred_calls=starred_calls,array_arithmetic=array_arithmetic,public_loops=public_loops,scalar_augmented=scalar_augmented,array_mutation=array_mutation,slot_period=slot_period)
    public = copy.deepcopy(constants); functions = {}

    def make_body(node):
        parameters = tuple(plan['functions'][node.name]['parameters'])
        def body(*args):
            star_origins = {}
            symbols = {n:frontend.resolveType(frontend.np.asarray(v if type(v) is list else [v], dtype=float))
                       for n,v in public.items()}
            symbols.update(zip(parameters, args))

            def storage_event(op,n,value,before=None):
                if observe_storage is not None:
                    from native_array_exercises import trace_record
                    observe_storage(trace_record(op,n,node.name,value,before))
                return value

            def expression(n):
                number = literal(n)
                if number is not None: return frontend.resolveType(number)
                if type(n) is ast.Name: return symbols[n.id]
                if type(n) in (ast.List,ast.Tuple):
                    values = [expression(v) for v in n.elts]
                    return values if type(n) is ast.List else tuple(values)
                if type(n) is ast.Subscript:
                    value = expression(n.value)
                    key = storage.index(n.slice) if arrays and type(value) is frontend.np.ndarray else rotation_literal(n.slice)
                    result = value[key]
                    return storage_event('index',n,result,value) if arrays and type(value) is frontend.np.ndarray else result
                if arrays and type(n) is ast.Attribute:
                    value = expression(n.value)
                    return storage_event('transpose',n,value.T,value)
                if type(n) is ast.UnaryOp: return -expression(n.operand)
                if type(n) is ast.BinOp:
                    a,b = expression(n.left), expression(n.right)
                    if type(n.op) is ast.Add: return a+b
                    if type(n.op) is ast.Sub: return a-b
                    return a*b
                if type(n) is ast.Call:
                    if arrays and storage.constructor(n):
                        return storage_event('array',n,frontend.np.array(expression(n.args[0]),dtype=object))
                    if type(n.func) is ast.Name:
                        arguments = []
                        segments = []
                        for argument in n.args:
                            value = expression(argument.value if type(argument) is ast.Starred else argument)
                            if observe_starred is not None:
                                from native_star_exercises import segment
                                segments.append(segment(argument,value,star_origins))
                            if type(argument) is ast.Starred:
                                # Python's first-axis iteration; never ravel/flatten arrays.
                                arguments.extend(value)
                            else:
                                arguments.append(value)
                        result = functions[n.func.id](*arguments)
                        if observe_starred is not None and any(type(a) is ast.Starred for a in n.args):
                            from native_star_exercises import trace_record
                            observe_starred(trace_record(n,node.name,segments))
                        # Trusted observer only, after a successful actual native call.
                        if observe is not None:
                            observe(dict(caller=node.name, callee=n.func.id,
                                span=[n.lineno,n.col_offset,n.end_lineno,n.end_col_offset]))
                        return storage_event('call',n,result) if arrays and type(result) is frontend.np.ndarray else result
                    if arrays and n.func.attr in storage.METHODS:
                        value = expression(n.func.value)
                        if observe_starred is not None:
                            star_origins[id(n)] = list(value.shape)
                        return storage_event(n.func.attr,n,storage.apply_method(value,n),value)
                    return expression(n.func.value).rotate(rotation_literal(n.args[0]))
                raise ValueError('Unexpected checked native AST')

            def assign(target, value):
                if type(target) is ast.Name: symbols[target.id] = value
                else:
                    if arrays and type(value) is frontend.np.ndarray:
                        storage_event('unpack',target,value,value)
                    for n,v in zip(target.elts,value): symbols[n.id] = v

            for statement in node.body[:-1]:
                if type(statement) is ast.AugAssign:
                    # Array __iop__ mutates storage; Expr __iop__ rebinds the name.
                    value=symbols[statement.target.id]
                    if array_mutation and type(value) is frontend.np.ndarray:
                        from native_array_alias import runtime_bindings
                        previous=value; shape=value.shape
                        before=runtime_bindings(symbols,value,frontend)
                        old=[(cell,cell.obj) for cell in value.flat]
                        old_bindings={name:list(v.flat) for name,v in symbols.items() if type(v) is frontend.np.ndarray}
                        other=expression(statement.value)
                        if type(statement.op) is ast.Add:value += other
                        elif type(statement.op) is ast.Sub:value -= other
                        else:value *= other
                        require(value is previous and value.shape==shape and all(cell.obj==obj for cell,obj in old),
                                'Unexpected native inplace array/Expr alias behavior')
                        symbols[statement.target.id]=value
                        if observe_mutation is not None:
                            observe_mutation(dict(function=node.name,target=statement.target.id,
                                operation={ast.Add:'add',ast.Sub:'subtract',ast.Mult:'multiply'}[type(statement.op)],
                                span=[statement.lineno,statement.col_offset,statement.end_lineno,statement.end_col_offset],
                                before=before,after=runtime_bindings(symbols,value,frontend),
                                same_array_object=True,original_exprs_unchanged=True,
                                changed_cells=[dict(name=name,indices=[i for i,(a,b) in enumerate(zip(saved,symbols[name].flat))
                                                                      if a is not b])
                                               for name,saved in sorted(old_bindings.items())]))
                        continue
                    previous=value; previous_obj=value.obj
                    other=expression(statement.value)
                    if type(statement.op) is ast.Add: value += other
                    elif type(statement.op) is ast.Sub: value -= other
                    else: value *= other
                    require(value is not previous and previous.obj == previous_obj,
                            'Native scalar augmented operator changed an aliased Expr')
                    symbols[statement.target.id]=value
                    if observe_augmented is not None:
                        observe_augmented(dict(function=node.name,target=statement.target.id,
                            operation={ast.Add:'add',ast.Sub:'subtract',ast.Mult:'multiply'}[type(statement.op)],
                            span=[statement.lineno,statement.col_offset,statement.end_lineno,statement.end_col_offset],
                            fresh_expr=True,original_expr_unchanged=True))
                elif type(statement) is ast.Assign: assign(statement.targets[0], expression(statement.value))
                else: expression(statement.value)
            result = expression(node.body[-1].value)
            return storage_event('return',node.body[-1],result) if arrays and type(result) is frontend.np.ndarray else result
        body.__name__ = node.name
        return body

    for name,node in nodes.items():
        functions[name] = frontend.func(plan['functions'][name]['signature'])(make_body(node))
    return dict(functions), plan
