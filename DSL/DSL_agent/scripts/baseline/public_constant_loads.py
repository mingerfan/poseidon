"""Opt-in compile-time loads from immutable public vectors, never candidate eval."""
import ast
import copy
import hashlib
import math
import operator
import struct
from dataclasses import dataclass
from benchmark_graph import require


@dataclass(frozen=True)
class Vector:
    name: str
    values: list


UNKNOWN = object()
INTEGER_BOUND = 1048576
BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
          ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod}


def lower(tree, constants, period):
    require(type(constants) is dict and len(constants) <= 256 and type(period) is int,
            'Public constant load context')
    globals_ = {}
    for name, value in constants.items():
        if type(value) is list:
            require(len(value) in (1, period) and all(type(x) in (int, float) and
                    math.isfinite(x) and abs(x) <= 1024 for x in value), 'Public constant load vector')
            globals_[name] = Vector(name, value)
    result = copy.deepcopy(tree)
    records = []

    for function in result.body:
        if type(function) is not ast.FunctionDef:
            continue
        environment = dict(globals_)
        pure_aliases = set()

        def integer(node, depth=0):
            require(depth <= 64, 'Public constant index depth')
            if type(node) is ast.Constant:
                value = node.value
            elif type(node) is ast.Name:
                value = environment.get(node.id, UNKNOWN)
            elif type(node) is ast.UnaryOp and type(node.op) in (ast.UAdd, ast.USub):
                value = integer(node.operand, depth + 1)
                value = value if type(node.op) is ast.UAdd else -value
            elif type(node) is ast.BinOp and type(node.op) in BINARY:
                try:
                    value = BINARY[type(node.op)](integer(node.left, depth + 1), integer(node.right, depth + 1))
                except (ZeroDivisionError, OverflowError) as error:
                    raise ValueError('Public constant index arithmetic') from error
            else:
                raise ValueError('Public constant index must be a static integer')
            require(type(value) is int and abs(value) <= INTEGER_BOUND, 'Public constant index integer bound')
            return value

        def reference(node, depth=0):
            require(depth <= 64, 'Public constant alias depth')
            if type(node) is ast.Name:
                return environment.get(node.id, UNKNOWN)
            if type(node) is ast.Constant and type(node.value) is int:
                return node.value
            if type(node) in (ast.Tuple, ast.List):
                require(len(node.elts) <= 16, 'Public constant alias container bound')
                items = tuple(reference(v, depth + 1) for v in node.elts)
                return UNKNOWN if any(v is UNKNOWN for v in items) else items
            if type(node) is ast.Subscript:
                base = reference(node.value, depth + 1)
                if type(base) is tuple:
                    index = integer(node.slice)
                    require(-len(base) <= index < len(base), 'Public constant alias index')
                    return base[index]
            return UNKNOWN

        def immutable(value):
            return type(value) is Vector or (type(value) is tuple and any(immutable(v) for v in value))

        class Loads(ast.NodeTransformer):
            def visit_Subscript(self, node):
                if type(node.ctx) is not ast.Load:
                    require(not immutable(reference(node.value)), 'Public constant vectors are readonly')
                    return node
                base = reference(node.value)
                if type(base) is Vector:
                    index = integer(node.slice)
                    require(-len(base.values) <= index < len(base.values), 'Public constant index out of bounds')
                    index %= len(base.values)
                    value = float(base.values[index])
                    require(len(records) < 4096, 'Public constant load work bound')
                    records.append(dict(function=function.name, line=node.lineno, column=node.col_offset,
                        constant=base.name, index=index,
                        binary64_sha256=hashlib.sha256(struct.pack('<d', value)).hexdigest()))
                    return ast.copy_location(ast.Constant(value=value), node)
                if type(base) is tuple:
                    index = integer(node.slice)
                    require(-len(base) <= index < len(base), 'Public constant alias index')
                    selected = base[index]
                    if type(selected) is Vector:
                        return ast.copy_location(ast.Name(id=selected.name, ctx=ast.Load()), node)
                return self.generic_visit(node)

        for statement in function.body:
            # Check stores against aliases before rewriting loads.
            for node in ast.walk(statement):
                if type(node) is ast.Subscript and type(node.ctx) is not ast.Load:
                    require(not immutable(reference(node.value)), 'Public constant vectors are readonly')
            if type(statement) is ast.Assign and len(statement.targets) == 1 and type(statement.targets[0]) is ast.Name:
                name = statement.targets[0].id
                require(name not in globals_, 'Public constant names cannot be assigned')
                value = reference(statement.value)
                statement.value = Loads().visit(statement.value)
                if value is UNKNOWN:
                    try:
                        value = integer(statement.value)
                    except ValueError:
                        value = UNKNOWN
                if value is not UNKNOWN:
                    pure_aliases.add(id(statement))
                if value is UNKNOWN:
                    environment.pop(name, None)
                else:
                    environment[name] = value
            else:
                if type(statement) is ast.AugAssign and type(statement.target) is ast.Name:
                    require(not immutable(environment.get(statement.target.id, UNKNOWN)), 'Public constant vectors are readonly')
                Loads().visit(statement)
                if type(statement) is ast.AugAssign and type(statement.target) is ast.Name:
                    environment.pop(statement.target.id, None)
        # Remove unused, side-effect-free public aliases after their reads fold.
        # This keeps the frontend from materializing an otherwise unused bank.
        live = set()
        kept = []
        for statement in reversed(function.body):
            target = (statement.targets[0].id if type(statement) is ast.Assign and
                      len(statement.targets) == 1 and type(statement.targets[0]) is ast.Name else None)
            if id(statement) in pure_aliases and target not in live:
                continue
            if target is not None:
                live.discard(target)
            live.update(n.id for n in ast.walk(statement) if type(n) is ast.Name and type(n.ctx) is ast.Load)
            kept.append(statement)
        function.body = list(reversed(kept))
    ast.fix_missing_locations(result)
    return result, dict(schema=1, policy='readonly-public-vector-scalar-load-v1',
                        source_modified=False, compiler_frontend_lowering=True, loads=records)
