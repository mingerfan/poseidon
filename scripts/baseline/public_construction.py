"""Bounded, data-only partial evaluation of public Hecate construction syntax.

Never execute candidate Python. Containers hold immutable symbolic value IDs;
public control flow is evaluated here, then a flat function is type-checked by
the existing arithmetic contract. No ciphertext-dependent branching or indexing.
"""
import ast
import copy
from dataclasses import dataclass
import hashlib
import operator

from seal_artifact_gate import require


@dataclass(frozen=True)
class Value:
    name: str
    kind: str


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


def normalize(source, constants, expected_outputs=1, *, input_names=('x',)):
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
    require(len(tree.body) == 1 and type(tree.body[0]) is ast.FunctionDef,
            'One golden function required; helper definitions are not yet supported')
    fn = tree.body[0]
    require(sum(type(n) is ast.FunctionDef for n in nodes) == 1, 'Nested functions not supported')
    for node in nodes:
        require(type(node) in ALLOWED, 'Unsupported construction syntax: '+type(node).__name__)
        if type(node) is ast.Call:
            require(not node.keywords, 'No keyword calls in construction syntax')
            valid = ((type(node.func) is ast.Name and node.func.id in ('range', 'len')) or
                     (type(node.func) is ast.Attribute and node.func.attr in ('rotate', 'append')) or
                     node in fn.decorator_list)
            require(valid, 'Unsupported construction call')
    # Reuse the existing exact decorator/signature/manifest checks, including zero ABI.
    header = copy.deepcopy(fn)
    header.body = [ast.Return(value=ast.List(elts=[ast.Name(id=input_names[0], ctx=ast.Load())
                   for _ in range(expected_outputs)], ctx=ast.Load()))] if input_names else []
    validate_function(ast.unparse(ast.fix_missing_locations(ast.Module(body=[header], type_ignores=[]))),
                      constants, expected_outputs, contract='hecate-function-v5', input_names=input_names)
    require(fn.body and type(fn.body[-1]) is ast.Return, 'Requires final return')
    require(sum(type(n) is ast.Return for n in nodes) == 1, 'Only a final top-level return is supported')
    bindings = {name: Value(name, 'plain') for name in constants}
    bindings.update({name: Value(name, 'cipher') for name in input_names})
    reserved = set(constants) | {'hc', 'golden', 'zero_ct', 'range', 'len'}
    taken = {n.id for n in nodes if type(n) is ast.Name} | set(bindings)
    emitted = []
    counters = dict(steps=0, loop_iterations=0, public_branches=0, container_writes=0)

    def tick(depth=0):
        counters['steps'] += 1
        require(counters['steps'] <= 4096 and depth <= 32, 'Construction expansion resource limit')

    def integer(value):
        require(type(value) is int and abs(value) <= 1048576, 'Bounded public integer required')
        return value

    def condition(value):
        require(type(value) in (int, bool), 'Branch condition must be public, never ciphertext')
        return bool(value)

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
        require(type(value) is Value, 'Arithmetic literals/containers are not ciphertext operands; use named constants')
        return ast.Name(id=value.name, ctx=ast.Load())

    def binary(op, left, right):
        if type(left) is int and type(right) is int:
            require(type(op) in INTEGER_OPS, 'Unsupported public integer arithmetic')
            integer(left); integer(right)
            require(not (type(op) in (ast.FloorDiv, ast.Mod) and right == 0), 'Public division by zero')
            return integer(INTEGER_OPS[type(op)](left, right))
        require(type(op) in (ast.Add, ast.Sub, ast.Mult) and type(left) is Value and
                type(right) is Value and 'cipher' in (left.kind, right.kind),
                'Ciphertext arithmetic needs named cipher/plain values')
        return emit(ast.BinOp(left=ref(left), op=op, right=ref(right)))

    def at(container, index):
        require(type(container) in (list, tuple), 'Only public containers are indexable; ciphertext slots are not')
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

    def expression(node, depth=0):
        tick(depth)
        if type(node) is ast.Name:
            require(node.id in bindings, 'Undefined construction value: '+node.id)
            return bindings[node.id]
        if type(node) is ast.Constant:
            require(type(node.value) in (int, bool), 'Only public integer/bool literals are allowed')
            return integer(node.value) if type(node.value) is int else node.value
        if type(node) in (ast.List, ast.Tuple):
            require(len(node.elts) <= 128, 'Public container length limit')
            values = [expression(n, depth+1) for n in node.elts]
            return values if type(node) is ast.List else tuple(values)
        if type(node) is ast.Subscript:
            container = expression(node.value, depth+1)
            return container[at(container, expression(node.slice, depth+1))]
        if type(node) is ast.BinOp:
            return binary(node.op, expression(node.left, depth+1), expression(node.right, depth+1))
        if type(node) is ast.UnaryOp and type(node.op) in (ast.USub, ast.UAdd):
            value = expression(node.operand, depth+1)
            if type(value) is int:
                return integer(-value if type(node.op) is ast.USub else value)
            require(type(node.op) is ast.USub and type(value) is Value and value.kind == 'cipher',
                    'Unary operation requires public integer or cipher negation')
            return emit(ast.UnaryOp(op=ast.USub(), operand=ref(value)))
        if type(node) is ast.Compare:
            left = integer(expression(node.left, depth+1))
            for op, item in zip(node.ops, node.comparators):
                right = integer(expression(item, depth+1))
                if not COMPARISONS[type(op)](left, right):
                    return False
                left = right
            return True
        if type(node) is ast.IfExp:
            take = condition(expression(node.test, depth+1))
            counters['public_branches'] += 1
            return expression(node.body if take else node.orelse, depth+1)
        if type(node) is ast.Call:
            if type(node.func) is ast.Name:
                values = [expression(n, depth+1) for n in node.args]
                if node.func.id == 'range':
                    require(1 <= len(values) <= 3, 'range requires one to three public integers')
                    values = [integer(v) for v in values]
                    require(len(values) < 3 or values[2] != 0, 'range step cannot be zero')
                    result = range(*values)
                    require(len(result) <= 128, 'Public range length limit')
                    return tuple(result)
                require(node.func.id == 'len' and len(values) == 1 and type(values[0]) in (list, tuple),
                        'len requires one public container')
                return len(values[0])
            require(type(node.func) is ast.Attribute and node.func.attr == 'rotate' and len(node.args) == 1,
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
            require(IDENTIFIER.fullmatch(target.id) and target.id not in reserved, 'Read-only or invalid binding')
            bindings[target.id] = value
        elif type(target) in (ast.List, ast.Tuple):
            require(type(value) in (list, tuple) and len(target.elts) == len(value), 'Unpacking length/type mismatch')
            for child, item in zip(target.elts, tuple(value)):
                assign(child, item, depth+1)
        elif type(target) is ast.Subscript:
            container = expression(target.value, depth+1)
            require(type(container) is list, 'Only a mutable local list supports item assignment')
            index = at(container, expression(target.slice, depth+1))
            no_cycle(container, value)
            container[index] = value
            counters['container_writes'] += 1
        else:
            raise ValueError('Unsupported construction assignment target')

    def block(statements, depth=0):
        for statement in statements:
            tick(depth)
            if type(statement) is ast.Assign:
                require(len(statement.targets) == 1 and statement.type_comment is None, 'Single assignment target required')
                assign(statement.targets[0], expression(statement.value, depth+1), depth+1)
            elif type(statement) is ast.AugAssign:
                require(type(statement.target) is ast.Name, 'Augmented item assignment not yet supported')
                left = expression(ast.Name(id=statement.target.id, ctx=ast.Load()), depth+1)
                assign(statement.target, binary(statement.op, left, expression(statement.value, depth+1)), depth+1)
            elif type(statement) is ast.For:
                require(not statement.orelse and statement.type_comment is None, 'for-else/type comments not supported')
                values = expression(statement.iter, depth+1)
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

    block(fn.body[:-1])
    result = expression(fn.body[-1].value)
    returned = list(result) if type(result) in (list, tuple) else [result]
    require(len(returned) == expected_outputs and all(type(v) is Value and v.kind == 'cipher' for v in returned),
            'Return must contain the declared number of ciphertexts, not slots/public values')
    ret = ast.List(elts=[ref(v) for v in returned], ctx=ast.Load()) if type(result) in (list, tuple) else ref(result)
    header.body = emitted + [ast.Return(value=ret)]
    normalized = ast.unparse(ast.fix_missing_locations(ast.Module(body=[header], type_ignores=[])))+'\n'
    checked = validate_function(normalized, constants, expected_outputs, contract='hecate-function-v5', input_names=input_names)
    metadata = dict(schema=1, **counters, source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                    normalized_sha256=hashlib.sha256(normalized.encode()).hexdigest(),
                    candidate_python_executed=False, constants_changed=False)
    return dict(source=normalized, check=checked, construction=metadata)
