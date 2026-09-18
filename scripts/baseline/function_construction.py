"""Lexically scoped, bounded interpretation of Hecate construction functions.

Never execute candidate Python. Containers hold immutable symbolic value IDs;
public control flow is evaluated here, then a flat function is type-checked by
the existing arithmetic contract. No ciphertext-dependent branching or indexing.

The preceding public_construction module is a frozen compatibility engine for
request v7. This version adds fresh function frames, return unwinding, bounded
recursion and higher-order calls between locally declared trusted-AST functions.
No Python function object from candidate code is ever created or executed.
"""
import ast
import copy
from dataclasses import dataclass
import hashlib
import json
import operator

from seal_artifact_gate import require


@dataclass(frozen=True)
class Value:
    name: str
    kind: str


@dataclass(frozen=True)
class Function:
    definition: object
    local_names: frozenset
    environment: object = None
    nonlocal_names: frozenset = frozenset()
    defaults: tuple = ()
    keyword_defaults: tuple = ()


class Returned(Exception):
    def __init__(self, value):
        self.value = value


class LoopBreak(Exception):
    pass


class LoopContinue(Exception):
    pass


@dataclass
class PublicIterator:
    iterator: object


@dataclass(eq=False)
class MappingView:
    mapping: dict
    kind: str

    def native(self):
        return getattr(self.mapping, self.kind)()


UNBOUND = object()


INTEGER_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
               ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod}
COMPARISONS = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt,
               ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge}
ALLOWED = {ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return,
           ast.Name, ast.Load, ast.Store, ast.Assign, ast.AugAssign, ast.For,
           ast.If, ast.IfExp, ast.Expr, ast.Call, ast.Attribute, ast.List, ast.Tuple,
           ast.Subscript, ast.Constant, ast.BinOp, ast.UnaryOp, ast.Compare,
           ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod, ast.USub, ast.UAdd,
           *COMPARISONS}


def normalize(source, constants, expected_outputs=1, *, input_names=('x',), closures=False,
              call_binding=False, public_iteration=False, function_literals=False, public_sequences=False,
              public_numbers=False, public_control=False, public_strings=False, public_polynomial=False,
              object_arrays=False, public_mappings=False, observe=None, object_arithmetic=False,
              scalar_conversion=False, object_unary=False, _unary_probe=None):
    # Trusted diagnostic only; never exposed to candidate code or request flags.
    if _unary_probe is not None:
        require(object_unary and type(_unary_probe) is dict and
                set(_unary_probe)=={'span','kind'} and
                type(_unary_probe['span']) is list and len(_unary_probe['span'])==4 and
                all(type(v) is int and 0 <= v <= 10000000 for v in _unary_probe['span']) and
                _unary_probe['kind'] in ('cipher','plain','number','bool'),
                'Invalid trusted unary diagnostic probe')
    scalar_conversion = scalar_conversion or object_unary
    object_arithmetic = object_arithmetic or scalar_conversion
    if scalar_conversion:
        import scalar_conversion as scalars
    public_mappings = public_mappings or object_arithmetic
    object_arrays = object_arrays or public_mappings
    public_polynomial = public_polynomial or object_arrays
    if object_arrays:
        import object_arrays as objects
    public_strings = public_strings or public_polynomial
    if public_polynomial:
        import public_polynomial as polynomial
    public_control = public_control or public_strings
    if public_strings:
        import public_strings as strings
    public_numbers = public_numbers or public_control
    public_sequences = public_sequences or public_numbers
    if public_numbers:
        import public_numeric as numeric
    function_literals = function_literals or public_sequences
    public_iteration = public_iteration or function_literals
    call_binding = call_binding or public_iteration
    closures = closures or call_binding
    if call_binding:
        from construction_calls import parameter_nodes, bind
    from hecate_contract import validate_function, IDENTIFIER, CONTRACT_ROTATIONS
    require(type(expected_outputs) is int and 1 <= expected_outputs <= 4, 'Invalid result count')
    require(type(input_names) in (list, tuple) and 1 <= len(input_names) <= 5 and
            all(type(n) is str for n in input_names), 'Invalid input names')
    require(type(source) is str and len(source.encode()) <= 65536, 'Source size limit')
    try:
        tree = ast.parse(source)
    except (SyntaxError, RecursionError, MemoryError) as error:
        raise ValueError('Invalid construction syntax or parser limit') from error
    nodes = list(ast.walk(tree))
    require(len(nodes) <= 4096, 'Construction AST node limit')
    require(1 <= len(tree.body) <= 17 and all(type(n) is ast.FunctionDef for n in tree.body),
            'Only golden and at most 16 top-level helper definitions are allowed')
    all_definitions = [n for n in nodes if type(n) is ast.FunctionDef]
    lambdas = [n for n in nodes if type(n) is ast.Lambda] if function_literals else []
    require(len(all_definitions) + len(lambdas) <= 17, 'Function/lambda declaration limit')
    require(len(all_definitions) <= 17, 'At most 17 total function definitions')
    if not closures:
        require(len(all_definitions) == len(tree.body), 'Nested definitions and closures are not yet supported')
    definitions = {n.name: n for n in tree.body}
    require(len(definitions) == len(tree.body) and 'golden' in definitions, 'Duplicate or missing function name')
    fn = definitions.pop('golden')
    require(type(constants) is dict, 'Invalid constants manifest')
    forbidden = {'hc', 'golden', 'zero_ct', 'range', 'len'}
    builtin_names = {'range', 'len'}
    if public_mappings:
        builtin_names.add('dict')
        forbidden.add('dict')
    if object_arrays:
        forbidden.update(('Empty','object'))
        builtin_names.add('Empty')
    if public_iteration:
        builtin_names.update(('enumerate', 'zip', 'reversed', 'iter', 'next', 'list', 'tuple'))
        forbidden.update(builtin_names)
    if function_literals:
        builtin_names.add('sorted')
        forbidden.add('sorted')
    if public_numbers:
        builtin_names.update(('float','int','pow'))
        forbidden.update(('np','float','int','pow'))
        require(not set(constants) & forbidden, 'Numeric namespace collides with public constants')

    def numeric_attribute(node):
        return public_numbers and type(node) is ast.Attribute and (
            (type(node.value) is ast.Name and node.value.id == 'np' and node.attr in ('array','asarray'))
            or node.attr in ('reshape','flatten','shape') or (scalar_conversion and node.attr=='item'))

    def string_method(node):
        return public_strings and type(node) is ast.Attribute and node.attr in strings.METHODS

    def mapping_method(node):
        return public_mappings and type(node) is ast.Attribute and node.attr in (
            'copy','get','setdefault','pop','popitem','clear','update','keys','values','items')

    def object_attribute(node):
        return object_arrays and type(node) is ast.Attribute and (
            (type(node.value) is ast.Name and node.value.id == 'np' and
             (node.attr in ('empty','full','concatenate','object_') or
              object_unary and node.attr in ('negative','positive'))) or
            (type(node.value) is ast.Name and node.value.id == 'hc' and node.attr == 'Empty') or
            node.attr in ('copy','transpose','T','size','ndim'))

    def public_np_path(node):
        if not public_polynomial or type(node) is not ast.Attribute:
            return None
        if type(node.value) is ast.Name and node.value.id == 'np':
            return node.attr if node.attr in (*polynomial.UFUNCS,'double','float64','polynomial') else None
        if node.attr == 'Chebyshev' and public_np_path(node.value) == 'polynomial':
            return 'Chebyshev'
        return None
    functions = {}
    helpers = [n for n in all_definitions if n is not fn] if closures else list(definitions.values())
    helpers += lambdas
    for definition in helpers:
        anonymous = type(definition) is ast.Lambda
        if not anonymous:
            name = definition.name
            require(IDENTIFIER.fullmatch(name) and name not in forbidden and
                    (definition not in tree.body or (name not in constants and name not in input_names)),
                    'Invalid or colliding helper name')
        args = definition.args
        names = [a.arg for a in args.args]
        parameters = parameter_nodes(args) if call_binding else args.args
        names = [a.arg for a in parameters]
        signature_supported = call_binding or (not args.posonlyargs and not args.vararg and not args.kwarg
                and not args.kwonlyargs and not args.defaults and not args.kw_defaults)
        require((anonymous or (not definition.decorator_list and definition.returns is None and definition.type_comment is None))
                and signature_supported and all(a.annotation is None for a in parameters)
                and len(names) <= 16 and len(set(names)) == len(names)
                and all(IDENTIFIER.fullmatch(n) and n not in forbidden for n in names),
                'Helpers require unannotated positional parameters without defaults or decorators')
        locals_ = frozenset(names) | frozenset(n.id for n in ast.walk(definition)
                     if type(n) is ast.Name and type(n.ctx) is ast.Store)
        if definition in tree.body:
            functions[name] = Function(definition, locals_)
    callable_names = set(functions) | {n.id for n in nodes if type(n) is ast.Name and type(n.ctx) is ast.Store}
    callable_names.update(a.arg for n in all_definitions for a in n.args.args)
    callable_names.update(n.name for n in all_definitions)
    if call_binding:
        callable_names.update(a.arg for n in [*all_definitions, *lambdas] for a in parameter_nodes(n.args))
    for node in nodes:
        require(type(node) in ALLOWED or (closures and type(node) is ast.Nonlocal) or
                (call_binding and type(node) in (ast.keyword, ast.Starred, ast.Dict)) or
                (public_iteration and type(node) in (ast.ListComp, ast.DictComp, ast.comprehension)) or
                (function_literals and type(node) is ast.Lambda) or
                (public_sequences and type(node) is ast.Slice) or
                (public_numbers and type(node) in (ast.Div,ast.Pow)) or
                (public_control and type(node) in (ast.While,ast.Break,ast.Continue,ast.Pass,
                    ast.BoolOp,ast.And,ast.Or,ast.Not,ast.In,ast.NotIn)),
                'Unsupported construction syntax: '+type(node).__name__)
        if type(node) is ast.comprehension:
            require(not node.is_async, 'Async comprehensions are not supported')
        if closures and type(node) is ast.Attribute:
            require(numeric_attribute(node) or object_attribute(node) or public_np_path(node) is not None or
                    (public_polynomial and node.attr in ('coef','domain','window')) or
                    any(type(call) is ast.Call and call.func is node and
                        (node.attr in ('rotate', 'append') or string_method(node) or mapping_method(node) or call in fn.decorator_list)
                        for call in nodes), 'Only permitted method/decorator attributes are allowed')
        if type(node) is ast.Call:
            require(call_binding or not node.keywords, 'No keyword calls in construction syntax')
            valid = ((type(node.func) is ast.Name and node.func.id in callable_names | builtin_names) or
                     (type(node.func) is ast.Attribute and node.func.attr in ('rotate', 'append')) or
                     (function_literals and type(node.func) in (ast.Lambda, ast.Call, ast.Subscript, ast.IfExp)) or
                     numeric_attribute(node.func) or object_attribute(node.func) or string_method(node.func) or mapping_method(node.func) or
                     (public_polynomial and public_np_path(node.func) in (*polynomial.UFUNCS,'Chebyshev')) or
                     node in fn.decorator_list)
            require(valid, 'Unsupported construction call')
    # Reuse the existing exact decorator/signature/manifest checks, including zero ABI.
    header = copy.deepcopy(fn)
    header.body = [ast.Return(value=ast.List(elts=[ast.Name(id=input_names[0], ctx=ast.Load())
                   for _ in range(expected_outputs)], ctx=ast.Load()))] if input_names else []
    validate_function(ast.unparse(ast.fix_missing_locations(ast.Module(body=[header], type_ignores=[]))),
                      constants, expected_outputs, contract='hecate-function-v5', input_names=input_names)
    globals_ = {name: Value(name, 'plain') for name in constants}
    globals_.update(functions)
    bindings = dict(globals_)
    bindings.update({name: Value(name, 'cipher') for name in input_names})
    scopes = {}
    if closures:
        from lexical_scope import Frame, analyze_scopes, UNBOUND as CELL_UNBOUND
        scopes = analyze_scopes(tree, call_binding=call_binding, public_iteration=public_iteration,
                                function_literals=function_literals)
        module = Frame(module=True)
        module.update({name: Value(name, 'plain') for name in constants})
        functions = {name: Function(definition, scopes[id(definition)][0], module,
                                  scopes[id(definition)][1]) for name, definition in definitions.items()}
        if not call_binding:
            module.update(functions)
        bindings = Frame(scopes[id(fn)][0], module, scopes[id(fn)][1])
        bindings.update({name: Value(name, 'cipher') for name in input_names})
        golden_frame = bindings
    reserved = set(constants) | forbidden | set(functions)
    frame_locals = None
    call_depth = 0
    taken = {n.id for n in nodes if type(n) is ast.Name} | set(globals_) | set(input_names)
    emitted = []
    derived = {}
    derived_names = {}
    counters = dict(steps=0, loop_iterations=0, public_branches=0, container_writes=0,
                    helper_calls=0, max_call_depth=0)
    if closures:
        counters.update(closure_instances=0, nonlocal_writes=0)
    if call_binding:
        counters.update(default_evaluations=0, keyword_arguments=0, starred_expansions=0)
    if public_iteration:
        counters.update(comprehensions=0, comprehension_iterations=0, iterator_advances=0)
    if function_literals:
        counters.update(lambda_instances=0, sorted_calls=0, sort_key_calls=0)
    if public_sequences:
        counters.update(sequence_slices=0, sequence_operations=0, slice_writes=0)
    if public_numbers:
        counters.update(public_numeric_operations=0, public_array_operations=0)
    if public_control:
        counters.update(while_iterations=0,loop_breaks=0,loop_continues=0,
                        membership_tests=0,short_circuits=0)
    if public_strings:
        counters.update(public_string_calls=0)
    if public_polynomial:
        counters.update(public_polynomial_operations=0,public_numpy_functions=0)
    if object_arrays:
        counters.update(object_array_operations=0,empty_operations=0)
    if object_arithmetic:
        counters.update(object_elementwise_operations=0,object_inplace_operations=0)
    if object_unary:
        counters.update(object_unary_operations=0)
    if scalar_conversion:
        counters.update(scalar_conversions=0,legacy_array_scalar_conversions=0,array_item_calls=0)
    if public_mappings:
        counters.update(mapping_calls=0,mapping_writes=0,mapping_views=0)

    def instantiate(definition, depth):
        if observe is not None:
            observe(definition)
        locals_, nonlocals = scopes[id(definition)]
        defaults, kwdefaults = (), ()
        if call_binding:
            defaults = tuple(expression(n, depth+1) for n in definition.args.defaults)
            kwdefaults = tuple((a.arg, expression(n, depth+1)) for a, n in
                               zip(definition.args.kwonlyargs, definition.args.kw_defaults) if n is not None)
            counters['default_evaluations'] += len(defaults) + len(kwdefaults)
        return Function(definition, locals_, bindings, nonlocals, defaults, kwdefaults)

    def invoke(function, values, depth, keywords=None):
        nonlocal bindings, frame_locals, call_depth
        definition = function.definition
        names = [a.arg for a in definition.args.args]
        if call_binding:
            arguments = bind(definition.args, function.defaults, function.keyword_defaults, values, keywords or {})
        else:
            require(len(values) == len(names), 'Helper positional argument count mismatch')
            arguments = dict(zip(names, values))
        counters['helper_calls'] += 1
        require(counters['helper_calls'] <= 128 and call_depth < 16, 'Helper call/depth resource limit')
        previous, previous_locals = bindings, frame_locals
        if closures:
            bindings = Frame(function.local_names, function.environment, function.nonlocal_names)
        else:
            bindings = dict(globals_)  # Lexical module globals, never the caller's locals.
            bindings.update({name: UNBOUND for name in function.local_names})
        bindings.update(arguments)  # Lists/defaults shared by reference; Expr IDs immutable.
        frame_locals = function.local_names
        call_depth += 1
        counters['max_call_depth'] = max(counters['max_call_depth'], call_depth)
        try:
            if type(definition) is ast.Lambda:
                return expression(definition.body, depth+1)
            block(definition.body, depth + 1)
        except Returned as returned:
            return returned.value
        finally:
            bindings, frame_locals = previous, previous_locals
            call_depth -= 1
        return None

    def tick(depth=0):
        counters['steps'] += 1
        require(counters['steps'] <= 4096 and depth <= 32, 'Construction expansion resource limit')

    def integer(value):
        require(type(value) is int and abs(value) <= 1048576, 'Bounded public integer required')
        return value

    def condition(value):
        if public_mappings and type(value) is MappingView:
            return bool(value.mapping)
        if object_arrays and type(value) is objects.Empty:
            return True
        if public_control:
            require(type(value) in (type(None),bool,int,float,str,list,tuple,dict,Function,PublicIterator),
                    'Truth value requires public construction data; cipher/array truth is unsupported')
            if type(value) in (Function,PublicIterator):
                return True
            return bool(value)
        if public_numbers and type(value) is float:
            return bool(numeric.number(value))
        require(type(value) in (int, bool), 'Branch condition must be public, never ciphertext')
        return bool(value)

    def mapping_key(value, depth=0):
        if not public_control:
            require(type(value) is str,'Construction mapping requires string keys')
            return value
        tick(depth)
        require(type(value) in (type(None),bool,int,float,str,tuple),
                'Mapping keys must be immutable public values, never cipher/array/function')
        if type(value) in (int,float):
            numeric.number(value)
        elif type(value) is tuple:
            for child in value:
                mapping_key(child,depth+1)
        return value

    def public_equal(left, right, depth=0):
        tick(depth)
        safe=(type(None),bool,int,float,str,list,tuple,dict)
        require(type(left) in safe and type(right) in safe,
                'Equality requires public data, never cipher/array/function')
        if type(left) in (list,tuple) and type(right) is type(left):
            return len(left) == len(right) and all(public_equal(a,b,depth+1) for a,b in zip(left,right))
        if type(left) is dict and type(right) is dict:
            return len(left) == len(right) and all(k in right and public_equal(v,right[k],depth+1)
                                                  for k,v in left.items())
        if type(left) in (list,tuple,dict) or type(right) in (list,tuple,dict):
            return False
        return left == right

    def public_compare(op, left, right, depth):
        tick(depth)
        if type(op) in (ast.In,ast.NotIn):
            counters['membership_tests'] += 1
            if public_mappings and type(right) is MappingView:
                if right.kind == 'keys':
                    found = mapping_key(left,depth+1) in right.mapping
                elif right.kind == 'items':
                    found = (type(left) is tuple and len(left) == 2 and
                             mapping_key(left[0],depth+1) in right.mapping and
                             public_equal(left[1],right.mapping[left[0]],depth+1))
                else:
                    found = any(public_equal(left,item,depth+1) for item in right.mapping.values())
            elif type(right) is dict:
                found = mapping_key(left,depth+1) in right
            elif type(right) is str:
                require(type(left) is str,'String membership requires public string')
                found = left in right
            else:
                require(type(right) in (list,tuple,PublicIterator),'Membership requires public container/iterator')
                found = any(public_equal(left,value,depth+1) for value in public_items(right,depth+1))
            return not found if type(op) is ast.NotIn else found
        if type(op) in (ast.Eq,ast.NotEq):
            equal = public_equal(left,right,depth+1)
            return not equal if type(op) is ast.NotEq else equal
        if type(left) in (list,tuple) and type(right) is type(left):
            for a,b in zip(left,right):
                if not public_equal(a,b,depth+1):
                    return public_compare(op,a,b,depth+1)
            return COMPARISONS[type(op)](len(left),len(right))
        require((type(left) in (bool,int,float) and type(right) in (bool,int,float)) or
                (type(left) is str and type(right) is str), 'Ordering requires comparable public scalars/sequences')
        return COMPARISONS[type(op)](left,right)

    def emit(node):
        require(len(emitted) < 256, 'Expanded ciphertext operation budget exceeded')
        number = len(emitted)
        name = 'constructed' + str(number)
        while name in taken:
            number += 1
            name = 'constructed' + str(number)
        taken.add(name)
        emitted.append(ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())], value=node))
        return Value(name, 'cipher')

    def ref(value):
        if public_numbers and type(value) is not Value:
            if type(value) in (list,tuple):
                value = numeric.array(public_data(value))
            data = numeric.encoding(value)
            key = json.dumps(data,allow_nan=False,separators=(',',':'))
            if key not in derived_names:
                require(len(constants)+len(derived) < 128, 'Derived constant count limit')
                number = len(derived)
                name = 'derived'+str(number)
                while name in taken:
                    number += 1
                    name = 'derived'+str(number)
                taken.add(name)
                derived[name] = data
                derived_names[key] = name
            value = Value(derived_names[key],'plain')
        require(type(value) is Value, 'Arithmetic literals/containers are not ciphertext operands; use named constants')
        return ast.Name(id=value.name, ctx=ast.Load())

    def binary(op, left, right):
        if object_arithmetic and any(type(v) is objects.ObjectArray for v in (left,right)):
            # Expr on the LEFT resolves the entire ndarray as Plain; it does
            # not dispatch a NumPy object ufunc. Do not invent broadcasting.
            require(type(left) not in (Value, objects.Empty),
                    'Symbolic/Empty left operand does not use object-array broadcasting')
            require(not (type(right) is Value and right.kind == 'plain'),
                    'Explicit Plain/object-array dispatch requires separate validation')
            counters['object_array_operations'] += 1
            counters['object_elementwise_operations'] += 1
            return objects.elementwise(op,left,right,binary)
        if object_arrays and any(type(v) is objects.Empty for v in (left,right)):
            counters['empty_operations'] += 1
            def resolve(value):
                if type(value) is Value:
                    return value
                def converted(item, depth=0):
                    tick(depth)
                    if type(item) is list:
                        require(len(item) <= 128,'Empty public conversion length limit')
                        return [converted(child,depth+1) for child in item]
                    return int(item) if type(item) is bool else numeric.number(item)
                return Value(ref(converted(value)).id,'plain')
            return objects.empty_binary(op,left,right,resolve)
        if public_polynomial and any(type(v) is polynomial.Polynomial for v in (left,right)):
            counters['public_polynomial_operations'] += 1
            left = left if type(left) is polynomial.Polynomial else public_data(left)
            right = right if type(right) is polynomial.Polynomial else public_data(right)
            return polynomial.binary(op,left,right)
        if public_numbers:
            ciphertext = any(type(v) is Value and v.kind == 'cipher' for v in (left,right))
            if ciphertext:
                require(type(op) in (ast.Add,ast.Sub,ast.Mult),
                        'Hecate ciphertext operators are +, -, *; no invented division/power lowering')
                return emit(ast.BinOp(left=ref(left),op=op,right=ref(right)))
            # Preserve Python list/tuple/string + and repetition, NOT array math.
            sequence_operation = (type(op) is ast.Add and type(left) in (list,tuple,str)
                                  and type(right) is type(left)) or (
                                  type(op) is ast.Mult and (type(left) in (list,tuple,str)
                                                           or type(right) in (list,tuple,str)))
            if any(type(v) in (numeric.Array,Value) for v in (left,right)):
                sequence_operation = False
            if not sequence_operation:
                counters['public_numeric_operations'] += 1
                return numeric.binary(op,public_data(left),public_data(right))
        if public_sequences:
            sequence = (list, tuple, str)
            if type(op) is ast.Add and type(left) in sequence and type(right) is type(left):
                require(len(left) + len(right) <= 128, 'Public sequence length limit')
                counters['sequence_operations'] += 1
                return left + right
            if type(op) is ast.Mult and (type(left) in sequence or type(right) in sequence):
                value, count = (left, right) if type(left) in sequence else (right, left)
                require(type(count) in (int, bool), 'Sequence repetition requires a public integer')
                count = integer(int(count))
                require(len(value) * max(count, 0) <= 128, 'Public sequence length limit')
                counters['sequence_operations'] += 1
                return value * count  # Shallow references, not a deep copy.
        if type(left) is int and type(right) is int:
            require(type(op) in INTEGER_OPS, 'Unsupported public integer arithmetic')
            integer(left); integer(right)
            require(not (type(op) in (ast.FloorDiv, ast.Mod) and right == 0), 'Public division by zero')
            return integer(INTEGER_OPS[type(op)](left, right))
        require(type(op) in (ast.Add, ast.Sub, ast.Mult) and type(left) is Value and
                type(right) is Value and 'cipher' in (left.kind, right.kind),
                'Ciphertext arithmetic needs named cipher/plain values')
        return emit(ast.BinOp(left=ref(left), op=op, right=ref(right)))


    def object_unary_value(op, value, node):
        require(type(value) is objects.ObjectArray, 'np.negative/positive require explicit object arrays in this contract')
        counters['object_array_operations'] += 1
        counters['object_unary_operations'] += 1
        def cell(operation, item):
            if type(item) is Value:
                require(type(operation) is ast.USub, 'Hecate Expr has no unary positive operator')
                if item.kind == 'cipher':
                    return emit(ast.UnaryOp(op=ast.USub(),operand=ref(item)))
                # Existing public-only arithmetic folds known constants but
                # retains Plain typing: it must not become float-convertible.
                require(item.kind=='plain','Unknown symbolic cell type')
                negated=numeric.binary(ast.Mult(),public_data(item),-1)
                return Value(ref(negated).id,'plain')
            require(type(item) in (int,float,bool), 'Unary object cells cannot be Empty or None')
            val=int(item) if type(item) is bool else item
            return numeric.number(-val if type(operation) is ast.USub else +val)
        def probed_cell(operation,item):
            result=cell(operation,item)
            if _unary_probe is None or _unary_probe['span'] != [
                    node.lineno,node.col_offset,node.end_lineno,node.end_col_offset]:
                return result
            kind=item.kind if type(item) is Value else 'bool' if type(item) is bool else 'number'
            if kind != _unary_probe['kind']:return result
            if kind=='cipher':return emit(ast.UnaryOp(op=ast.USub(),operand=ref(result)))
            if kind=='plain':return Value(ref(numeric.binary(ast.Add(),public_data(result),1)).id,'plain')
            return numeric.number(result+1)
        result=objects.unary(op,value,probed_cell)
        if observe is not None:
            cells=list(value.data.flat)
            facts=dict(operator=type(op).__name__,shape=list(value.shape),
                syntax='call' if type(node) is ast.Call else 'operator',
                input_is_view=value.data.base is not None,
                boolean_cells=sum(type(v) is bool for v in cells),
                public_real_cells=sum(type(v) in (int,float) for v in cells),
                cipher_cells=sum(type(v) is Value and v.kind=='cipher' for v in cells),
                plain_cells=sum(type(v) is Value and v.kind=='plain' for v in cells),
                public_cells=sum(type(v) in (int,float,bool) for v in cells),
                result_shape=list(result.shape) if type(result) is objects.ObjectArray else None)
            observe(('object_unary',node,facts))
        return result

    def at(container, index):
        if call_binding and type(container) is dict:
            mapping_key(index)
            require(index in container, 'Missing construction mapping key')
            return index
        require(type(container) in ((list, tuple, str) if public_sequences else (list, tuple)),
                'Only public containers are indexable; ciphertext slots are not')
        if public_sequences and type(index) is slice:
            counters['sequence_slices'] += 1
            return index
        if public_sequences and type(index) is bool:
            index = int(index)
        index = integer(index)
        require(-len(container) <= index < len(container), 'Public container index out of bounds')
        return index

    def no_cycle(target, value):
        pending = [(value, 0)]
        while pending:
            item, depth = pending.pop()
            tick(depth)
            require(item is not target, 'Cyclic containers are not supported')
            if type(item) in (list, tuple):
                pending.extend((v, depth+1) for v in item)
            elif call_binding and type(item) is dict:
                pending.extend((v, depth+1) for v in item.values())
            elif public_mappings and type(item) is MappingView:
                pending.append((item.mapping,depth+1))

    def public_data(value, depth=0):
        tick(depth)
        if type(value) is Value:
            require(value.kind == 'plain', 'Ciphertext is not public numerical data')
            data = derived[value.name] if object_arrays and value.name in derived else constants[value.name]
            return numeric.array(data if type(data) is list else [data],floating=True)
        if type(value) in (list,tuple):
            return type(value)(public_data(v,depth+1) for v in value)
        return value

    def raw_iterator(value):
        if public_mappings and type(value) is MappingView:
            return iter(value.native())
        if object_arrays and type(value) is objects.ObjectArray:
            return objects.iterate(value)
        if public_numbers and (type(value) is numeric.Array or type(value) is Value):
            value = public_data(value)
            require(type(value) is numeric.Array and value.shape, 'Scalar array is not iterable')
            return iter(tuple(numeric.index(value,i) for i in range(value.shape[0])))
        require(type(value) in ((list, tuple, dict, PublicIterator, str) if public_sequences
                               else (list, tuple, dict, PublicIterator)),
                'Iteration requires a public container/iterator, never ciphertext or opaque constants')
        return value.iterator if type(value) is PublicIterator else iter(value)

    def advance(iterator, depth):
        tick(depth)
        counters['iterator_advances'] += 1
        try:
            return next(iterator)
        except RuntimeError as error:
            raise ValueError('Public iterator invalidated by mutation') from error

    def public_items(value, depth):
        iterator = raw_iterator(value)
        while True:
            try:
                item = advance(iterator, depth)
            except StopIteration:
                return
            yield item

    def materialize(value, depth):
        result = []
        for item in public_items(value, depth):
            require(len(result) < 128, 'Public iterator materialization length limit')
            result.append(item)
        return result

    def mapping_insert(target, key, value, depth):
        mapping_key(key,depth+1)
        require(key in target or len(target) < 128,'Mapping length limit')
        no_cycle(target,value)
        target[key] = value
        counters['mapping_writes'] += 1

    def mapping_update(target, values, keywords, depth):
        require(len(values) <= 1,'dict/update accepts at most one positional argument')
        if values:
            source = values[0]
            if type(source) is dict:
                for key in source:
                    tick(depth+1)
                    mapping_insert(target,key,source[key],depth+1)
            else:
                for item in public_items(source,depth+1):
                    pair = materialize(item,depth+1)
                    require(len(pair) == 2,'Mapping update entries must contain exactly two values')
                    mapping_insert(target,pair[0],pair[1],depth+1)
        for key,value in keywords.items():
            mapping_insert(target,key,value,depth+1)

    def mapping_invoke(name, receiver, values, keywords, depth):
        counters['mapping_calls'] += 1
        if name == 'copy' and type(receiver) is objects.ObjectArray:
            require(not values and not keywords,'Object copy takes no arguments')
            counters['object_array_operations'] += 1
            return objects.copy(receiver)
        require(type(receiver) is dict,'Mapping method receiver must be a local dict')
        if name == 'update':
            mapping_update(receiver,values,keywords,depth+1)
            return None
        require(not keywords,'Mapping method parameters are positional-only')
        if name in ('get','setdefault','pop'):
            require(1 <= len(values) <= 2,'Mapping key method takes key and optional default')
            key = mapping_key(values[0],depth+1)
            if name == 'get':
                return receiver.get(key,values[1] if len(values) == 2 else None)
            if name == 'setdefault':
                if key not in receiver:
                    mapping_insert(receiver,key,values[1] if len(values) == 2 else None,depth+1)
                return receiver[key]
            require(key in receiver or len(values) == 2,'Mapping pop missing key without default')
            if key in receiver:
                counters['mapping_writes'] += 1
                return receiver.pop(key)
            return values[1]
        require(not values,'Mapping method takes no arguments')
        if name == 'copy':
            return receiver.copy()
        if name in ('keys','values','items'):
            counters['mapping_views'] += 1
            return MappingView(receiver,name)
        if name == 'clear':
            receiver.clear()
            counters['mapping_writes'] += 1
            return None
        require(name == 'popitem' and receiver,'Mapping popitem requires nonempty dict')
        counters['mapping_writes'] += 1
        return receiver.popitem()

    def comprehension(node, depth):
        nonlocal bindings, frame_locals
        counters['comprehensions'] += 1
        # Python evaluates the outermost iterable in the surrounding scope.
        outer = raw_iterator(expression(node.generators[0].iter, depth+1))
        local_names = frozenset(n.id for generator in node.generators for n in ast.walk(generator.target)
                                if type(n) is ast.Name and type(n.ctx) is ast.Store)
        previous, previous_locals = bindings, frame_locals
        bindings = Frame(local_names, previous)
        frame_locals = local_names
        result = [] if type(node) is ast.ListComp else {}

        def expand(index, iterator, nesting):
            generator = node.generators[index]
            for value in public_items(PublicIterator(iterator), nesting):
                counters['comprehension_iterations'] += 1
                assign(generator.target, value, nesting+1)
                take = True
                for test in generator.ifs:
                    counters['public_branches'] += 1
                    if not condition(expression(test, nesting+1)):
                        take = False
                        break
                if not take:
                    continue
                if index+1 < len(node.generators):
                    inner = raw_iterator(expression(node.generators[index+1].iter, nesting+1))
                    expand(index+1, inner, nesting+1)
                elif type(node) is ast.ListComp:
                    require(len(result) < 128, 'Comprehension result length limit')
                    result.append(expression(node.elt, nesting+1))
                else:
                    key = expression(node.key, nesting+1)
                    mapping_key(key,nesting+1)
                    require(key in result or len(result) < 128,'Mapping comprehension length limit')
                    result[key] = expression(node.value, nesting+1)
        try:
            expand(0, outer, depth+1)
        finally:
            bindings, frame_locals = previous, previous_locals
        return result

    def expression(node, depth=0):
        if observe is not None:
            observe(node)
        tick(depth)
        if object_arrays and type(node) is ast.Name and node.id == 'object':
            return objects.OBJECT
        if object_arrays and type(node) is ast.Attribute:
            if type(node.value) is ast.Name and node.value.id == 'np' and node.attr == 'object_':
                return objects.OBJECT
            if node.attr in ('T','size','ndim'):
                value = expression(node.value,depth+1)
                require(type(value) is objects.ObjectArray,'Object array metadata requires object array')
                return (objects.transpose(value) if node.attr == 'T' else
                        value.data.size if node.attr == 'size' else len(value.shape))
        if public_polynomial and type(node) is ast.Attribute:
            if public_np_path(node) in ('double','float64'):
                return polynomial.FLOAT64
            if node.attr in ('coef','domain','window'):
                value = expression(node.value,depth+1)
                require(type(value) is polynomial.Polynomial,'Polynomial attribute requires public Chebyshev data')
                data = value.coefficients if node.attr == 'coef' else value.domain if node.attr == 'domain' else value.window
                return numeric.array(data,floating=True)
        if public_control and type(node) is ast.BoolOp:
            for item in node.values[:-1]:
                value = expression(item,depth+1)
                take = condition(value)
                if (type(node.op) is ast.And and not take) or (type(node.op) is ast.Or and take):
                    counters['short_circuits'] += 1
                    return value
            return expression(node.values[-1],depth+1)
        if public_control and type(node) is ast.UnaryOp and type(node.op) is ast.Not:
            return not condition(expression(node.operand,depth+1))
        if public_numbers and type(node) is ast.Attribute and node.attr == 'shape':
            value = public_data(expression(node.value,depth+1))
            require(type(value) is numeric.Array or (object_arrays and type(value) is objects.ObjectArray),
                    'shape requires an array')
            return value.shape
        if public_sequences and type(node) is ast.Slice:
            parts = [expression(part, depth+1) if part is not None else None
                     for part in (node.lower, node.upper, node.step)]
            for part in parts:
                require(part is None or type(part) in (int, bool), 'Slice bounds must be public integers or None')
                if part is not None:
                    integer(int(part))
            require(parts[2] is None or parts[2] != 0, 'Slice step cannot be zero')
            return slice(*parts)
        if function_literals and type(node) is ast.Lambda:
            counters['lambda_instances'] += 1
            counters['closure_instances'] += 1
            require(counters['closure_instances'] <= 128, 'Closure instance resource limit')
            return instantiate(node, depth+1)
        if public_iteration and type(node) in (ast.ListComp, ast.DictComp):
            return comprehension(node, depth)
        if type(node) is ast.Name:
            require(node.id in bindings and bindings[node.id] is not UNBOUND and
                    (not closures or bindings[node.id] is not CELL_UNBOUND),
                    'Undefined or unbound local construction value: '+node.id)
            return bindings[node.id]
        if type(node) is ast.Constant:
            if node.value is None:
                return None
            if call_binding and type(node.value) is str:
                require(len(node.value) <= 128, 'Public mapping key length limit')
                return node.value
            if public_numbers and type(node.value) is float:
                return numeric.number(node.value)
            require(type(node.value) in (int, bool), 'Only public integer/bool/None literals are allowed')
            return integer(node.value) if type(node.value) is int else node.value
        if type(node) in (ast.List, ast.Tuple):
            require(len(node.elts) <= 128, 'Public container length limit')
            values = [expression(n, depth+1) for n in node.elts]
            return values if type(node) is ast.List else tuple(values)
        if type(node) is ast.Subscript:
            container = expression(node.value, depth+1)
            if object_arrays and type(container) is objects.ObjectArray:
                counters['object_array_operations'] += 1
                return objects.get(container,expression(node.slice,depth+1))
            if public_numbers and (type(container) is numeric.Array or type(container) is Value):
                container = public_data(container)
                require(type(container) is numeric.Array, 'Indexing requires a public array')
                counters['public_array_operations'] += 1
                return numeric.index(container,expression(node.slice,depth+1))
            return container[at(container, expression(node.slice, depth+1))]
        if call_binding and type(node) is ast.Dict:
            require(len(node.keys) <= 128 and all(k is not None for k in node.keys),
                    'Bounded explicit mapping entries required')
            result = {}
            for key, value in zip(node.keys, node.values):
                key = expression(key, depth+1)
                mapping_key(key,depth+1)
                result[key] = expression(value, depth+1)
            return result
        if type(node) is ast.BinOp:
            left,right=expression(node.left,depth+1),expression(node.right,depth+1)
            result=binary(node.op,left,right)
            if object_arithmetic and observe is not None and any(type(v) is objects.ObjectArray for v in (left,right)):
                facts=objects.arithmetic_facts(left,right)
                facts['result_shape']=list(result.shape) if type(result) is objects.ObjectArray else []
                observe(('object_binary',node,facts))
            return result
        if type(node) is ast.UnaryOp and type(node.op) in (ast.USub, ast.UAdd):
            value = expression(node.operand, depth+1)
            if object_unary and type(value) is objects.ObjectArray:
                return object_unary_value(node.op,value,node)
            if public_polynomial and type(value) is polynomial.Polynomial:
                return binary(ast.Mult(),value,-1 if type(node.op) is ast.USub else 1)
            if public_numbers and not (type(value) is Value and value.kind == 'cipher'):
                counters['public_numeric_operations'] += 1
                return numeric.binary(ast.Mult(),public_data(value),-1 if type(node.op) is ast.USub else 1)
            if type(value) is int:
                return integer(-value if type(node.op) is ast.USub else value)
            require(type(node.op) is ast.USub and type(value) is Value and value.kind == 'cipher',
                    'Unary operation requires public integer or cipher negation')
            return emit(ast.UnaryOp(op=ast.USub(), operand=ref(value)))
        if type(node) is ast.Compare:
            if public_control:
                left = expression(node.left,depth+1)
                for op,item in zip(node.ops,node.comparators):
                    right = expression(item,depth+1)
                    if not public_compare(op,left,right,depth+1):
                        return False
                    left = right
                return True
            check_number = numeric.number if public_numbers else integer
            left = check_number(expression(node.left, depth+1))
            for op, item in zip(node.ops, node.comparators):
                right = check_number(expression(item, depth+1))
                if not COMPARISONS[type(op)](left, right):
                    return False
                left = right
            return True
        if type(node) is ast.IfExp:
            take = condition(expression(node.test, depth+1))
            counters['public_branches'] += 1
            return expression(node.body if take else node.orelse, depth+1)
        if type(node) is ast.Call:
            if mapping_method(node.func):
                receiver = expression(node.func.value,depth+1)
                values,keywords = [],{}
                for argument in node.args:
                    if type(argument) is ast.Starred:
                        values.extend(materialize(expression(argument.value,depth+1),depth+1))
                        counters['starred_expansions'] += 1
                    else:
                        values.append(expression(argument,depth+1))
                    require(len(values) <= 128,'Mapping argument count limit')
                for keyword in node.keywords:
                    value = expression(keyword.value,depth+1)
                    extra = {keyword.arg:value} if keyword.arg is not None else value
                    require(type(extra) is dict and all(type(k) is str for k in extra),
                            'Keyword unpacking requires string-keyed mapping')
                    for key,value in extra.items():
                        require(key not in keywords,'Duplicate mapping keyword')
                        keywords[key] = value
                    require(len(keywords) <= 128,'Mapping keyword count limit')
                counters['keyword_arguments'] += len(keywords)
                return mapping_invoke(node.func.attr,receiver,values,keywords,depth+1)
            empty_call = object_arrays and (
                type(node.func) is ast.Name and node.func.id == 'Empty' or
                type(node.func) is ast.Attribute and type(node.func.value) is ast.Name and
                node.func.value.id == 'hc' and node.func.attr == 'Empty')
            if empty_call:
                require(not node.args and not node.keywords,'Empty takes no arguments')
                counters['empty_operations'] += 1
                return objects.Empty()
            if object_attribute(node.func):
                attr = node.func.attr
                constructor = type(node.func.value) is ast.Name and node.func.value.id == 'np'
                receiver = None if constructor else expression(node.func.value,depth+1)
                require(all(type(n) is not ast.Starred for n in node.args),'Object calls require explicit arguments')
                values = [expression(n,depth+1) for n in node.args]
                keywords = {}
                for keyword in node.keywords:
                    require(keyword.arg is not None and keyword.arg not in keywords,'Explicit unique object keywords required')
                    keywords[keyword.arg] = expression(keyword.value,depth+1)
                counters['object_array_operations'] += 1
                if object_unary and constructor and attr in ('negative','positive'):
                    require(len(values)==1 and not keywords,'Unary object ufunc accepts one positional argument; no out/dtype/where')
                    return object_unary_value(ast.USub() if attr=='negative' else ast.UAdd(),values[0],node)
                if constructor and attr in ('full','empty'):
                    require(set(keywords) == {'dtype'} and keywords['dtype'] is objects.OBJECT and
                            len(values) == (2 if attr == 'full' else 1),
                            'Object full/empty require positional shape/fill and dtype=object')
                    return objects.full(*values) if attr == 'full' else objects.empty(*values)
                if constructor and attr == 'concatenate':
                    require(len(values) == 1 and set(keywords) <= {'axis'},'Concatenate requires arrays and optional axis')
                    return objects.concatenate(values[0],**keywords)
                require(type(receiver) is objects.ObjectArray and not keywords,'Object method requires object array; use array.transpose(...), not np.transpose(...)')
                if attr == 'copy':
                    require(not values,'Object copy takes no arguments')
                    return objects.copy(receiver)
                require(attr == 'transpose','Unsupported object array method')
                return objects.transpose(receiver,None if not values else values[0] if len(values) == 1 else tuple(values))
            if public_polynomial and public_np_path(node.func) in (*polynomial.UFUNCS,'Chebyshev'):
                path = public_np_path(node.func)
                require(all(type(n) is not ast.Starred for n in node.args),
                        'Public NumPy constructors/functions require explicit arguments')
                values = [public_data(expression(n,depth+1)) for n in node.args]
                keywords = {}
                for kw in node.keywords:
                    require(kw.arg is not None and kw.arg not in keywords,'Explicit unique NumPy keywords required')
                    keywords[kw.arg] = public_data(expression(kw.value,depth+1))
                if path == 'Chebyshev':
                    require(len(values) <= 4 and set(keywords) <= {'coef','domain','window','symbol'},
                            'Invalid Chebyshev constructor signature')
                    counters['public_polynomial_operations'] += 1
                    try:
                        return polynomial.create(*values,**keywords)
                    except TypeError as error:
                        raise ValueError('Invalid Chebyshev constructor binding') from error
                require(len(values) == 1 and not keywords,'Public numeric function requires one positional argument')
                counters['public_numpy_functions'] += 1
                return polynomial.ufunc(path,values[0])
            if string_method(node.func):
                # Resolve the receiver before argument side effects; only a
                # direct method call is admitted, never arbitrary attributes.
                receiver = expression(node.func.value,depth+1)
                require(type(receiver) is str,'String receiver must be public, never cipher/array')
                values = []
                for argument in node.args:
                    if type(argument) is ast.Starred:
                        values.extend(materialize(expression(argument.value,depth+1),depth+1))
                        counters['starred_expansions'] += 1
                    else:
                        values.append(expression(argument,depth+1))
                    require(len(values) <= 128,'String call argument limit')
                keywords = {}
                for keyword in node.keywords:
                    value = expression(keyword.value,depth+1)
                    extra = {keyword.arg:value} if keyword.arg is not None else value
                    require(type(extra) is dict and all(type(k) is str for k in extra),
                            'Keyword unpacking requires string-keyed public mapping')
                    for key,value in extra.items():
                        require(key not in keywords,'Duplicate keyword argument: '+key)
                        keywords[key] = value
                    require(len(keywords) <= 128,'String keyword argument limit')
                counters['keyword_arguments'] += len(keywords)
                if node.func.attr == 'join' and len(values) == 1 and not keywords:
                    values = [materialize(values[0],depth+1)]
                counters['public_string_calls'] += 1
                return strings.invoke(node.func.attr,receiver,values,keywords)
            if numeric_attribute(node.func):
                counters['public_array_operations'] += 1
                attr = node.func.attr
                constructor = type(node.func.value) is ast.Name and node.func.value.id == 'np'
                raw_receiver = None if constructor else expression(node.func.value,depth+1)
                if scalar_conversion and attr == 'item' and type(raw_receiver) is Value:
                    require(raw_receiver.kind=='plain' and raw_receiver.name in constants,
                            'item is an array operation, not a ciphertext/Plain Expr operation')
                receiver = None if constructor else public_data(raw_receiver)
                require(all(type(n) is not ast.Starred for n in node.args), 'Array calls use explicit arguments')
                values = [(expression(n,depth+1) if object_arrays else public_data(expression(n,depth+1)))
                          for n in node.args]
                if scalar_conversion and attr == 'item':
                    require(not constructor and not node.keywords,'item requires an array and positional indices')
                    counters['array_item_calls'] += 1
                    result = scalars.item(receiver,values)
                    if observe is not None:
                        observe(('scalar_item',node,scalars.item_facts(receiver,values,result)))
                    return result
                if constructor:
                    require(attr in ('array','asarray') and len(values) == 1 and
                            len(node.keywords) <= 1 and all(k.arg == 'dtype' for k in node.keywords),
                            'Only np.array/asarray(value, dtype="float64") are permitted')
                    dtype = expression(node.keywords[0].value,depth+1) if node.keywords else None
                    if object_arrays and (dtype is objects.OBJECT or
                            dtype is None and type(values[0]) is objects.ObjectArray):
                        counters['object_array_operations'] += 1
                        named_numeric = (scalar_conversion and type(values[0]) is Value and
                                         values[0].kind=='plain' and values[0].name in constants)
                        if scalar_conversion:
                            def object_input(value,level=0):
                                tick(level)
                                if type(value) is Value and value.kind=='plain' and value.name in constants:
                                    value=public_data(value)
                                if type(value) is numeric.Array:
                                    return scalars.object_from_numeric(value)
                                if type(value) in (tuple,list):
                                    return [object_input(v,level+1) for v in value]
                                return value
                            values[0]=object_input(values[0])
                        result = objects.array(values[0],copy=attr == 'array')
                        if scalar_conversion and observe is not None and named_numeric:
                            observe(('object_numeric_constructor',node,dict(shape=list(result.shape))))
                        return result
                    is_float64 = dtype == 'float64' or (public_polynomial and dtype is polynomial.FLOAT64)
                    require(dtype is None or is_float64, 'Only inferred real or explicit float64 array dtype')
                    return numeric.array(public_data(values[0]) if object_arrays else values[0],floating=is_float64)
                if object_arrays and type(receiver) is objects.ObjectArray:
                    require(not node.keywords,'Object reshape/flatten require positional arguments')
                    counters['object_array_operations'] += 1
                    if attr == 'flatten':
                        require(not values,'Object flatten takes no arguments')
                        return objects.flatten(receiver)
                    require(attr == 'reshape' and values,'Object reshape requires shape')
                    return objects.reshape(receiver,values[0] if len(values) == 1 else tuple(values))
                if object_arrays:
                    values = [public_data(value) for value in values]
                require(type(receiver) is numeric.Array and not node.keywords, 'Array method requires public array and positional arguments')
                if attr == 'flatten':
                    require(not values, 'Only C-order flatten() is supported')
                    return numeric.reshape(receiver,(len(receiver.values),))
                require(attr == 'reshape' and values, 'Only reshape/flatten are array methods')
                return numeric.reshape(receiver,values[0] if len(values) == 1 else tuple(values))
            if type(node.func) is ast.Name or (function_literals and type(node.func) is not ast.Attribute):
                # Resolve the callable before evaluating arguments, matching Python.
                named = type(node.func) is ast.Name
                function = bindings.get(node.func.id) if named else expression(node.func, depth+1)
                builtin = named and node.func.id in builtin_names and node.func.id not in bindings
                require(builtin or type(function) is Function, 'Call target is not a declared construction function')
                values = []
                for n in node.args:
                    if call_binding and type(n) is ast.Starred:
                        extra = expression(n.value, depth+1)
                        if public_iteration:
                            extra = materialize(extra, depth+1)
                        require(type(extra) in (list, tuple), 'Star arguments require public list/tuple')
                        values.extend(extra)
                        counters['starred_expansions'] += 1
                    else:
                        values.append(expression(n, depth+1))
                    require(len(values) <= 128, 'Call argument resource limit')
                keywords = {}
                for keyword in node.keywords:
                    value = expression(keyword.value, depth+1)
                    extra = {keyword.arg: value} if keyword.arg is not None else value
                    require(type(extra) is dict and all(type(k) is str for k in extra),
                            'Keyword unpacking requires string-keyed public mapping')
                    for key, value in extra.items():
                        require(key not in keywords, 'Duplicate keyword argument: '+key)
                        keywords[key] = value
                    require(len(keywords) <= 128, 'Call argument resource limit')
                if call_binding:
                    counters['keyword_arguments'] += len(keywords)
                if type(function) is Function:
                    return invoke(function, values, depth+1, keywords)
                if public_numbers and node.func.id in ('float','int','pow'):
                    require(not keywords, 'Numeric builtin accepts positional arguments only')
                    if node.func.id == 'pow':
                        require(len(values) == 2, 'Public pow requires two arguments')
                        return numeric.binary(ast.Pow(),public_data(values[0]),public_data(values[1]))
                    if scalar_conversion:
                        converted=[]
                        for value in values:
                            if type(value) is Value and value.kind=='plain' and value.name in constants:
                                value=public_data(value)
                            converted.append(value)
                        value,deprecated=scalars.cast(node.func.id,converted)
                        if observe is not None:
                            observe(('scalar_cast',node,scalars.cast_facts(node.func.id,converted,value,deprecated)))
                        counters['scalar_conversions'] += 1
                        counters['legacy_array_scalar_conversions'] += int(deprecated)
                        return value
                    require(len(values) == 1 and type(values[0]) in (str,int,float),
                            'Numeric conversion requires a public scalar/string')
                    try:
                        return numeric.number((float if node.func.id == 'float' else int)(values[0]))
                    except (OverflowError,ValueError) as error:
                        raise ValueError('Invalid public numeric conversion') from error
                if public_mappings and node.func.id == 'dict':
                    result = {}
                    mapping_update(result,values,keywords,depth+1)
                    counters['mapping_calls'] += 1
                    return result
                if function_literals and node.func.id == 'sorted':
                    require(len(values) == 1 and set(keywords) <= {'key', 'reverse'},
                            'sorted requires one iterable and optional key/reverse keywords')
                    key = keywords.get('key')
                    reverse = keywords.get('reverse', False)
                    require(key is None or type(key) is Function, 'Sort key must be a declared construction function')
                    require(type(reverse) in (bool, int), 'Sort reverse flag must be public')
                    counters['sorted_calls'] += 1
                    items = materialize(values[0], depth+1)
                    decorated = []
                    def public_key(value, nesting):
                        tick(nesting)
                        require(type(value) in ((int, bool, float, str, tuple, list) if public_numbers else
                                              (int, bool, str, tuple, list)),
                                'Sort keys must be public, never cipher/plain symbolic operands')
                        if type(value) in (tuple, list):
                            for child in value:
                                public_key(child, nesting+1)
                    for item in items:
                        order = invoke(key, [item], depth+1) if key is not None else item
                        if key is not None:
                            counters['sort_key_calls'] += 1
                        public_key(order, depth+1)
                        decorated.append((order, item))
                    # A later key callback may mutate a previously returned key.
                    # Revalidate shared key containers before native comparison.
                    for order, item in decorated:
                        public_key(order, depth+1)
                    try:
                        # Trusted implementation lambda, not a generated Python function.
                        ordered = sorted(decorated, key=lambda pair: pair[0], reverse=bool(reverse))
                    except TypeError as error:
                        raise ValueError('Incompatible public sort keys') from error
                    return [item for order, item in ordered]
                require(not keywords, 'Construction builtin does not accept keywords')
                if public_iteration:
                    name = node.func.id
                    if name == 'enumerate':
                        require(1 <= len(values) <= 2, 'enumerate needs iterable and optional public start')
                        start = integer(values[1]) if len(values) == 2 else 0
                        return PublicIterator(enumerate(raw_iterator(values[0]), start))
                    if name == 'zip':
                        return PublicIterator(zip(*(raw_iterator(v) for v in values)))
                    if name == 'reversed':
                        if public_mappings and len(values) == 1 and type(values[0]) is MappingView:
                            return PublicIterator(reversed(values[0].native()))
                        if public_numbers and len(values) == 1 and (type(values[0]) is numeric.Array or type(values[0]) is Value):
                            array = public_data(values[0])
                            require(type(array) is numeric.Array and array.shape, 'reversed requires nonscalar public array')
                            return PublicIterator(iter(tuple(numeric.index(array,i) for i in range(array.shape[0]-1,-1,-1))))
                        require(len(values) == 1 and type(values[0]) in
                                ((list, tuple, dict, str) if public_sequences else (list, tuple, dict)),
                                'reversed needs a public reversible container')
                        return PublicIterator(reversed(values[0]))
                    if name == 'iter':
                        require(len(values) == 1, 'iter needs one public iterable')
                        return values[0] if type(values[0]) is PublicIterator else PublicIterator(raw_iterator(values[0]))
                    if name == 'next':
                        require(1 <= len(values) <= 2 and type(values[0]) is PublicIterator,
                                'next needs a public iterator and optional default')
                        try:
                            return advance(values[0].iterator, depth+1)
                        except StopIteration:
                            require(len(values) == 2, 'Public iterator exhausted without default')
                            return values[1]
                    if name in ('list', 'tuple'):
                        require(len(values) <= 1, 'Container constructor takes at most one public iterable')
                        items = materialize(values[0], depth+1) if values else []
                        return items if name == 'list' else tuple(items)
                if node.func.id == 'range':
                    require(1 <= len(values) <= 3, 'range requires one to three public integers')
                    values = [integer(v) for v in values]
                    require(len(values) < 3 or values[2] != 0, 'range step cannot be zero')
                    result = range(*values)
                    require(len(result) <= 128, 'Public range length limit')
                    return tuple(result)
                if public_mappings and node.func.id == 'len' and len(values) == 1 and type(values[0]) is MappingView:
                    return len(values[0].mapping)
                if object_arrays and node.func.id == 'len' and len(values) == 1 and type(values[0]) is objects.ObjectArray:
                    require(bool(values[0].shape),'len requires nonscalar object array')
                    return values[0].shape[0]
                if public_numbers and node.func.id == 'len' and len(values) == 1 and (type(values[0]) is numeric.Array or type(values[0]) is Value):
                    array = public_data(values[0])
                    require(type(array) is numeric.Array and array.shape, 'len requires nonscalar public array')
                    return array.shape[0]
                require(node.func.id == 'len' and len(values) == 1 and
                        (type(values[0]) in (list, tuple) or (call_binding and type(values[0]) is dict)
                         or (public_sequences and type(values[0]) is str)),
                        'len requires one public container')
                return len(values[0])
            require(type(node.func) is ast.Attribute and node.func.attr == 'rotate' and len(node.args) == 1 and not node.keywords,
                    'Only rotate is an expression method')
            value = expression(node.func.value, depth+1)
            step = integer(expression(node.args[0], depth+1))
            require(type(value) is Value and value.kind == 'cipher' and step in CONTRACT_ROTATIONS['hecate-function-v5'],
                    'Rotation requires a ciphertext and a provisioned public step')
            return emit(ast.Call(func=ast.Attribute(value=ref(value), attr='rotate', ctx=ast.Load()),
                                 args=[ast.Constant(value=step)], keywords=[]))
        raise ValueError('Unsupported construction expression: '+type(node).__name__)

    def assign(target, value, depth=0):
        tick(depth)
        if type(target) is ast.Name:
            writable = target.id not in reserved if frame_locals is None else (
                target.id in frame_locals and target.id not in forbidden)
            if closures and target.id in bindings.nonlocals:
                owner = bindings.owner(target.id)
                writable = target.id not in forbidden and not (owner is golden_frame and target.id in constants)
                counters['nonlocal_writes'] += 1
            require(IDENTIFIER.fullmatch(target.id) and writable, 'Read-only or invalid binding')
            bindings[target.id] = value
        elif type(target) in (ast.List, ast.Tuple):
            if public_iteration:
                value = materialize(value, depth+1)
            require(type(value) in (list, tuple) and len(target.elts) == len(value), 'Unpacking length/type mismatch')
            for child, item in zip(target.elts, tuple(value)):
                assign(child, item, depth+1)
        elif type(target) is ast.Subscript:
            container = expression(target.value, depth+1)
            if object_arrays and type(container) is objects.ObjectArray:
                objects.put(container,expression(target.slice,depth+1),value)
                counters['container_writes'] += 1
                counters['object_array_operations'] += 1
                return
            if call_binding and type(container) is dict:
                key = expression(target.slice, depth+1)
                mapping_key(key,depth+1)
                require(key in container or len(container) < 128, 'Mapping length limit')
                no_cycle(container, value)
                container[key] = value
                counters['container_writes'] += 1
                return
            require(type(container) is list, 'Only a mutable local list supports item assignment')
            index = at(container, expression(target.slice, depth+1))
            if public_sequences and type(index) is slice:
                # Evaluate RHS before target (assign's caller), then materialize
                # before changing the target. Self/iterator assignments snapshot.
                replacement = materialize(value, depth+1)
                for item in replacement:
                    no_cycle(container, item)
                start, stop, step = index.indices(len(container))
                removed = len(range(start, stop, step))
                require(step == 1 or len(replacement) == removed,
                        'Extended slice assignment length mismatch')
                require(len(container) - removed + len(replacement) <= 128, 'Public sequence length limit')
                # Keep original None/negative boundaries; normalized -1 has a
                # different meaning if applied a second time as a Python slice.
                container[index] = replacement
                counters['slice_writes'] += 1
                counters['container_writes'] += 1
                return
            no_cycle(container, value)
            container[index] = value
            counters['container_writes'] += 1
        else:
            raise ValueError('Unsupported construction assignment target')

    def block(statements, depth=0):
        for statement in statements:
            if observe is not None:
                observe(statement)
            tick(depth)
            if public_control and type(statement) is ast.Pass:
                continue
            if public_control and type(statement) is ast.Break:
                counters['loop_breaks'] += 1
                raise LoopBreak()
            if public_control and type(statement) is ast.Continue:
                counters['loop_continues'] += 1
                raise LoopContinue()
            if public_control and type(statement) is ast.While:
                broken = False
                while True:
                    tick(depth+1)
                    counters['public_branches'] += 1
                    if not condition(expression(statement.test,depth+1)):
                        break
                    counters['loop_iterations'] += 1
                    counters['while_iterations'] += 1
                    try:
                        block(statement.body,depth+1)
                    except LoopContinue:
                        continue
                    except LoopBreak:
                        broken = True
                        break
                if not broken:
                    if observe is not None and statement.orelse:
                        observe(('while_else', statement))
                    block(statement.orelse,depth+1)
                continue
            if closures and type(statement) is ast.FunctionDef:
                counters['closure_instances'] += 1
                require(counters['closure_instances'] <= 128, 'Closure instance resource limit')
                assign(ast.Name(id=statement.name, ctx=ast.Store()),
                       instantiate(statement, depth+1), depth+1)
                continue
            if closures and type(statement) is ast.Nonlocal:
                continue  # Statically validated; Frame performs nearest enclosing writes.
            if type(statement) is ast.Return:
                raise Returned(expression(statement.value, depth+1) if statement.value is not None else None)
            if type(statement) is ast.Assign:
                require(len(statement.targets) == 1 and statement.type_comment is None, 'Single assignment target required')
                assign(statement.targets[0], expression(statement.value, depth+1), depth+1)
            elif type(statement) is ast.AugAssign:
                if object_arrays and type(statement.target) is ast.Subscript:
                    container = expression(statement.target.value,depth+1)
                    require(type(container) is objects.ObjectArray,'Augmented item writes require object array')
                    key = expression(statement.target.slice,depth+1)
                    left = objects.get(container,key)
                    require(object_arithmetic or type(left) is not objects.ObjectArray,
                            'Vectorized object-array arithmetic not yet verified')
                    right = expression(statement.value,depth+1)
                    # Python evaluates the target once, before RHS effects;
                    # do not re-resolve a rebound name or side-effecting index.
                    if object_arithmetic and type(left) is objects.ObjectArray:
                        facts=objects.arithmetic_facts(left,right) if observe is not None else None
                        require(not (type(right) is Value and right.kind == 'plain'),
                                'Explicit Plain/object-array dispatch requires separate validation')
                        result = objects.elementwise(statement.op,left,right,binary,inplace=True)
                        counters['object_inplace_operations'] += 1
                        if observe is not None:
                            observe(('object_inplace',statement,facts))
                    else:
                        result = binary(statement.op,left,right)
                    objects.put(container,key,result)
                    counters['container_writes'] += 1
                    counters['object_array_operations'] += 1
                    continue
                require(type(statement.target) is ast.Name, 'Augmented item assignment not yet supported')
                left = expression(ast.Name(id=statement.target.id, ctx=ast.Load()), depth+1)
                if public_numbers:
                    require(not (type(left) is numeric.Array or (type(left) is Value and left.kind == 'plain')),
                            'Public arrays are read-only; augmented array writes are not implemented')
                right = expression(statement.value, depth+1)
                if object_arithmetic and type(left) is objects.ObjectArray:
                    facts=objects.arithmetic_facts(left,right) if observe is not None else None
                    require(not (type(right) is Value and right.kind == 'plain'),
                            'Explicit Plain/object-array dispatch requires separate validation')
                    result = objects.elementwise(statement.op,left,right,binary,inplace=True)
                    counters['object_inplace_operations'] += 1
                    counters['object_array_operations'] += 1
                    if observe is not None:
                        observe(('object_inplace',statement,facts))
                    assign(statement.target,result,depth+1)
                elif public_sequences and type(left) is list and type(statement.op) in (ast.Add, ast.Mult):
                    if type(statement.op) is ast.Add:
                        # Direct self-extension snapshots; an iterator over the
                        # target instead observes each append (and hits the bound).
                        items = list(right) if right is left else right
                        for item in public_items(items, depth+1):
                            require(len(left) < 128, 'Public sequence length limit')
                            no_cycle(left, item)
                            left.append(item)
                    else:
                        # Compute the bounded shallow repeat before mutating aliases.
                        left[:] = binary(statement.op, left, right)
                    counters['container_writes'] += 1
                    assign(statement.target, left, depth+1)
                else:
                    assign(statement.target, binary(statement.op, left, right), depth+1)
            elif type(statement) is ast.For:
                require((public_control or not statement.orelse) and statement.type_comment is None,
                        'for-else requires public-control contract; type comments not supported')
                values = expression(statement.iter, depth+1)
                if public_iteration:
                    broken = False
                    for value in public_items(values, depth+1):
                        assign(statement.target, value, depth+1)
                        counters['loop_iterations'] += 1
                        try:
                            block(statement.body, depth+1)
                        except LoopContinue:
                            continue
                        except LoopBreak:
                            broken = True
                            break
                    if public_control and not broken:
                        if observe is not None and statement.orelse:
                            observe(('for_else', statement))
                        block(statement.orelse,depth+1)
                    continue
                require(type(values) in (list, tuple), 'Loop requires public range/container, never ciphertext')
                index = 0
                # Match Python list iterator behavior, including bounded list mutation.
                while index < len(values):
                    tick(depth+1)
                    assign(statement.target, values[index], depth+1)
                    counters['loop_iterations'] += 1
                    block(statement.body, depth+1)
                    index += 1
            elif type(statement) is ast.If:
                take = condition(expression(statement.test, depth+1))
                counters['public_branches'] += 1
                block(statement.body if take else statement.orelse, depth+1)
            elif type(statement) is ast.Expr:
                call = statement.value
                if type(call) is ast.Call and mapping_method(call.func):
                    expression(call,depth+1)
                    continue
                if type(call) is ast.Call and (type(call.func) is ast.Name or
                        (function_literals and type(call.func) is not ast.Attribute)):
                    require(type(call.func) is not ast.Name or call.func.id not in builtin_names,
                            'Unused builtin call is not a helper statement')
                    expression(call, depth+1)
                    continue
                require(type(call) is ast.Call and type(call.func) is ast.Attribute and call.func.attr == 'append'
                        and len(call.args) == 1 and not call.keywords, 'Only list.append may be a statement call')
                container = expression(call.func.value, depth+1)
                value = expression(call.args[0], depth+1)
                require(type(container) is list and len(container) < 128, 'append requires bounded mutable list')
                no_cycle(container, value)
                container.append(value)
                counters['container_writes'] += 1
            else:
                raise ValueError('Unsupported construction statement: '+type(statement).__name__)

    if call_binding:
        # Module definitions have observable order: defaults may call only
        # earlier bound helpers; the new function is bound AFTER its defaults.
        bindings = module
        for definition in tree.body:
            if definition is not fn:
                module[definition.name] = instantiate(definition, 0)
        bindings = golden_frame
    try:
        block(fn.body)
    except Returned as returned:
        result = returned.value
    else:
        raise ValueError('golden returned no ciphertext value')
    if object_arrays and type(result) is objects.ObjectArray:
        require(len(result.shape) == 1,'Object array returns must be one-dimensional; no implicit flattening')
        result = list(objects.iterate(result))
    returned = list(result) if type(result) in (list, tuple) else [result]
    require(len(returned) == expected_outputs and all(type(v) is Value and v.kind == 'cipher' for v in returned),
            'Return must contain the declared number of ciphertexts, not slots/public values')
    ret = ast.List(elts=[ref(v) for v in returned], ctx=ast.Load()) if type(result) in (list, tuple) else ref(result)
    header.body = emitted + [ast.Return(value=ret)]
    normalized = ast.unparse(ast.fix_missing_locations(ast.Module(body=[header], type_ignores=[])))+'\n'
    effective = dict(constants,**derived)
    checked = validate_function(normalized, effective, expected_outputs, contract='hecate-function-v5', input_names=input_names)
    metadata = dict(schema=16 if object_unary else 15 if scalar_conversion else 14 if object_arithmetic else 13 if public_mappings else 12 if object_arrays else 11 if public_polynomial else 10 if public_strings else 9 if public_control else 8 if public_numbers else 7 if public_sequences else 6 if function_literals else 5 if public_iteration else 4 if call_binding else 3 if closures else 2,
                    function_definitions=len(helpers) if closures else len(functions), **counters,
                    source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                    normalized_sha256=hashlib.sha256(normalized.encode()).hexdigest(),
                    candidate_python_executed=False, constants_changed=False)
    result = dict(source=normalized, check=checked, construction=metadata)
    if public_numbers:
        metadata.update(derived_constant_count=len(derived),
                        derived_constants_sha256=hashlib.sha256(json.dumps(derived,sort_keys=True,
                            separators=(',',':'),allow_nan=False).encode()).hexdigest(),
                        input_constants_changed=False)
        result.update(constants=effective,derived_constants=derived)
    return result
