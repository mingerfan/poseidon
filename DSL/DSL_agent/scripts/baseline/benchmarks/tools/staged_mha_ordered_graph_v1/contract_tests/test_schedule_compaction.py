"""Trusted scalar compaction controls; never run Provider, compiler or FHE.

The restricted interpreter below reads an AST. It does not use eval/exec, import
candidate modules, or accept arbitrary Python. These are rule-emitter compatibility
tests; candidate admission and immutable public bindings remain separate gates.
"""
import ast
from collections import Counter
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
BASE = HERE.parent.parents[2]
if str(BASE) not in sys.path:
    sys.path.append(str(BASE))
from unified_capacity_lowering import compact_scalar_source


class Value:
    def __init__(self, values, expression):
        self.values = tuple(values)
        self.expression = expression


class RestrictedMath:
    """Elementwise public arithmetic and cyclic rotation over finite vectors."""
    def __init__(self, source, public):
        tree = ast.parse(source)
        if not all(type(node) is ast.FunctionDef for node in tree.body):
            raise ValueError("Only trusted function declarations are interpreted")
        self.functions = {node.name: node for node in tree.body}
        if len(self.functions) != len(tree.body):
            raise ValueError("Duplicate function")
        self.public = {
            name: Value(value if type(value) in (list, tuple) else [value],
                        ("public", name))
            for name, value in public.items()
        }
        self.operations = Counter()
        self.active = set()

    def primitive(self, kind, left, right=None, amount=None):
        self.operations[kind] += 1
        if kind == "rotate":
            size = len(left.values)
            step = amount % size
            return Value(left.values[step:] + left.values[:step],
                         ("rotate", left.expression, amount))
        if kind == "negate":
            return Value([-item for item in left.values],
                         ("negate", left.expression))
        width = max(len(left.values), len(right.values))
        if len(left.values) not in (1, width) or len(right.values) not in (1, width):
            raise ValueError("Incompatible vector broadcast")
        a = left.values * width if len(left.values) == 1 else left.values
        b = right.values * width if len(right.values) == 1 else right.values
        op = {
            "add": lambda x, y: x + y,
            "subtract": lambda x, y: x - y,
            "multiply": lambda x, y: x * y,
        }[kind]
        return Value([op(x, y) for x, y in zip(a, b)],
                     (kind, left.expression, right.expression))

    def expression(self, node, environment):
        if type(node) is ast.Name:
            return environment[node.id]
        if type(node) is ast.Constant and type(node.value) in (int, float):
            return Value([node.value], ("literal", node.value))
        if type(node) is ast.UnaryOp and type(node.op) is ast.USub:
            return self.primitive("negate", self.expression(node.operand, environment))
        if type(node) is ast.BinOp and type(node.op) in (ast.Add, ast.Sub, ast.Mult):
            kind = {ast.Add: "add", ast.Sub: "subtract", ast.Mult: "multiply"}[type(node.op)]
            return self.primitive(kind, self.expression(node.left, environment),
                                  self.expression(node.right, environment))
        if type(node) in (ast.List, ast.Tuple):
            result = [self.expression(item, environment) for item in node.elts]
            return result if type(node) is ast.List else tuple(result)
        if type(node) is ast.Call and not node.keywords:
            if type(node.func) is ast.Name and node.func.id in self.functions:
                return self.call(node.func.id, [
                    self.expression(arg, environment) for arg in node.args])
            if type(node.func) is ast.Attribute and node.func.attr == "rotate" and len(node.args) == 1:
                amount = node.args[0]
                sign = 1
                if type(amount) is ast.UnaryOp and type(amount.op) is ast.USub:
                    sign, amount = -1, amount.operand
                if type(amount) is not ast.Constant or type(amount.value) is not int:
                    raise ValueError("Rotation requires a public integer")
                return self.primitive("rotate",
                                      self.expression(node.func.value, environment),
                                      amount=sign * amount.value)
        raise ValueError("Unsupported interpreter expression: " + type(node).__name__)

    def call(self, name, arguments):
        if name in self.active:
            raise ValueError("Recursive function")
        function = self.functions[name]
        if (function.args.vararg or function.args.kwarg or function.args.kwonlyargs
                or function.args.defaults or len(function.args.args) != len(arguments)):
            raise ValueError("Fixed formal arguments required")
        environment = dict(self.public)
        environment.update({formal.arg: value for formal, value in
                            zip(function.args.args, arguments)})
        self.active.add(name)
        try:
            for index, statement in enumerate(function.body):
                if type(statement) is ast.Assign and len(statement.targets) == 1 and type(statement.targets[0]) is ast.Name:
                    name = statement.targets[0].id
                    if name in environment:
                        raise ValueError("Interpreter expects immutable SSA and public bindings")
                    environment[name] = self.expression(statement.value, environment)
                elif type(statement) is ast.Return and index == len(function.body) - 1:
                    return self.expression(statement.value, environment)
                else:
                    raise ValueError("Only scalar assignments followed by return")
            raise ValueError("Missing return")
        finally:
            self.active.remove(function.name)

    def golden(self, inputs):
        function = self.functions["golden"]
        arguments = [
            Value(inputs[arg.arg], ("input", arg.arg)) for arg in function.args.args
        ]
        return self.call("golden", arguments)


def signature(value):
    if isinstance(value, Value):
        return ("value", value.values, value.expression)
    if type(value) is list:
        return ("list", tuple(signature(item) for item in value))
    if type(value) is tuple:
        return ("tuple", tuple(signature(item) for item in value))
    raise ValueError("Unknown interpreter result")


def depth(node):
    return 1 + max((depth(child) for child in ast.iter_child_nodes(node)), default=0)


class ScheduleCompaction(unittest.TestCase):
    PUBLIC = {"p0": .375, "p1": [-1., 2., -.5, 3.]}
    INPUTS = [
        {"x": [-.5, .25, 1.5, -2.], "y": [2., -.75, .125, 1.], "zero_ct": [0.] * 4},
        {"x": [0., 0., 0., 0.], "y": [-1., 3., -.5, 2.], "zero_ct": [0.] * 4},
        {"x": [.03125, -.125, .5, -.875], "y": [.75, -2., 1.25, .125], "zero_ct": [0.] * 4},
    ]

    def compare(self, source):
        compact = compact_scalar_source(source)
        for inputs in self.INPUTS:
            before = RestrictedMath(source, self.PUBLIC)
            after = RestrictedMath(compact, self.PUBLIC)
            self.assertEqual(signature(after.golden(inputs)), signature(before.golden(inputs)))
            self.assertEqual(after.operations, before.operations,
                             "Compaction must neither duplicate nor remove primitive work")
        return compact

    def test_single_use_scalar_assignments_fold(self):
        source = '''@hc.func("c,c,c")
def golden(x, y, zero_ct):
    v0 = x + p0
    v1 = v0 * y
    v2 = v1 - p1
    return [v2]
'''
        compact = self.compare(source)
        self.assertLess(sum(type(n) is ast.Assign for n in ast.walk(ast.parse(compact))), 3)

    def test_shared_square_is_not_duplicated(self):
        source = '''@hc.func("c,c,c")
def golden(x, y, zero_ct):
    v0 = x * x
    v1 = v0 + y
    v2 = v0 - y
    return [v1, v2]
'''
        compact = self.compare(source)
        self.assertEqual(sum(type(n) is ast.Mult for n in ast.walk(ast.parse(compact))), 1)
        self.assertTrue(any(type(n) is ast.Assign and type(n.value) is ast.BinOp
                            and type(n.value.op) is ast.Mult for n in ast.walk(ast.parse(compact))))

    def test_rotation_negation_and_operand_order_are_retained(self):
        source = '''@hc.func("c,c,c")
def golden(x, y, zero_ct):
    v0 = x.rotate(-1)
    v1 = -v0
    v2 = p1 - v1
    v3 = v2 * y
    v4 = x - v3
    v5 = v4.rotate(2)
    return [v5]
'''
        self.compare(source)

    def test_multiple_outputs_and_repeated_port_keep_shared_work(self):
        source = '''@hc.func("c,c,c")
def golden(x, y, zero_ct):
    v0 = x * y
    v1 = v0 + p0
    return [v1, v0, v1, zero_ct]
'''
        compact = self.compare(source)
        tree = ast.parse(compact)
        self.assertEqual(len(tree.body[-1].body[-1].value.elts), 4)
        self.assertEqual(sum(type(n) is ast.Mult for n in ast.walk(tree)), 1)

    def test_tuple_output_structure_preserved(self):
        source = '''@hc.func("c,c,c")
def golden(x, y, zero_ct):
    v0 = x + y
    v1 = v0 * p1
    return (v1, x, v1)
'''
        compact = self.compare(source)
        self.assertIs(type(ast.parse(compact).body[-1].body[-1].value), ast.Tuple)

    def test_single_use_typed_helper_is_inlined_once(self):
        source = '''@hc.func("c")
def helper(a):
    v0 = a * a
    v1 = v0 + v0
    return v1
@hc.func("c,c,c")
def golden(x, y, zero_ct):
    v0 = helper(x)
    v1 = v0 - y
    return [v1]
'''
        compact = self.compare(source)
        self.assertEqual(len(ast.parse(compact).body), 1)
        self.assertEqual(sum(type(n) is ast.Mult for n in ast.walk(ast.parse(compact))), 1)

    def test_repeated_typed_helper_calls_remain_independent(self):
        source = '''@hc.func("c")
def helper(a):
    v0 = a * a
    v1 = v0 + p0
    return v1
@hc.func("c,c,c")
def golden(x, y, zero_ct):
    v0 = helper(x)
    v1 = helper(y)
    v2 = v0 - v1
    return [v2]
'''
        compact = self.compare(source)
        tree = ast.parse(compact)
        self.assertEqual(len(tree.body), 2)
        self.assertEqual(sum(type(n) is ast.Call and type(n.func) is ast.Name
                             and n.func.id == "helper" for n in ast.walk(tree)), 2)
        self.assertEqual(ast.dump(tree.body[0].decorator_list[0]),
                         ast.dump(ast.parse(source).body[0].decorator_list[0]))

    def test_nested_single_use_helpers_preserve_public_binding(self):
        source = '''@hc.func("c")
def first(a):
    v0 = a + p0
    return v0
@hc.func("c")
def second(a):
    v0 = first(a)
    v1 = v0 * p1
    return v1
@hc.func("c,c,c")
def golden(x, y, zero_ct):
    v0 = second(x)
    v1 = y - v0
    return [v1]
'''
        compact = self.compare(source)
        self.assertEqual(len(ast.parse(compact).body), 1)
        targets = {n.id for node in ast.walk(ast.parse(compact)) if type(node) is ast.Assign
                   for n in node.targets if type(n) is ast.Name}
        self.assertTrue({"p0", "p1"}.isdisjoint(targets))
        self.assertTrue({"p0", "p1"} <= {n.id for n in ast.walk(ast.parse(compact))
                                       if type(n) is ast.Name and type(n.ctx) is ast.Load})

    def test_helper_local_renaming_avoids_existing_prefix_collision(self):
        source = '''@hc.func("c")
def helper(a):
    v0 = a * a
    v1 = v0 + v0
    return v1
@hc.func("c,c,c")
def golden(x, y, zero_ct):
    cap_inline0_v0 = y * y
    v0 = helper(x)
    v1 = v0 - cap_inline0_v0
    return [v1, cap_inline0_v0]
'''
        self.compare(source)

    def test_long_chain_respects_depth_24_and_retains_ssa_breaks(self):
        lines = ['@hc.func("c,c,c")', 'def golden(x, y, zero_ct):',
                 '    v0 = x + p0']
        lines += ["    v%d = v%d + p0" % (i, i - 1) for i in range(1, 60)]
        lines += ["    return [v59]"]
        compact = self.compare("\n".join(lines) + "\n")
        function = ast.parse(compact).body[0]
        assignments = [node for node in function.body if type(node) is ast.Assign]
        self.assertGreater(len(assignments), 0, "Depth boundary must preserve SSA breaks")
        self.assertLess(len(assignments), 60)
        self.assertTrue(all(depth(node) <= 24 for node in function.body))

    def test_non_scalar_helper_return_is_not_trusted_scalar_emission(self):
        source = '''@hc.func("c")
def helper(a):
    return [a, a]
@hc.func("c,c,c")
def golden(x, y, zero_ct):
    v0 = helper(x)
    return v0
'''
        with self.assertRaisesRegex(ValueError, "scalar return"):
            compact_scalar_source(source)

    def test_interpreter_refuses_dynamic_or_side_effecting_python(self):
        for source in [
            'def golden(x, y, zero_ct):\n    return x.__class__\n',
            'def golden(x, y, zero_ct):\n    return open("private")\n',
            'def golden(x, y, zero_ct):\n    return x.rotate(y)\n',
        ]:
            with self.assertRaises((ValueError, KeyError)):
                RestrictedMath(source, self.PUBLIC).golden(self.INPUTS[0])


if __name__ == "__main__":
    unittest.main()
