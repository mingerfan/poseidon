"""AST-only PT-DSL parser with static attributes and hygienic macro expansion."""

import ast
import re

from .errors import PTDSLError
from .ir import Header, Op, Param, Program, Region, WeightRef
from .ops_spec import OPS, RESERVED_NAMES


class _Tensor:
    def __init__(self, vid):
        self.vid = vid


class _Scope:
    def __init__(self, bindings, prefix=""):
        self.env = dict(bindings)
        self.counts = {name: 0 for name in bindings}
        self.prefix = prefix

    def next_id(self, name):
        index = self.counts.get(name, -1) + 1
        self.counts[name] = index
        return self.prefix + name + (f"__{index}" if index else "")


def _fail(code, node, msg):
    raise PTDSLError(code, getattr(node, "lineno", 0), msg)


def _integer(value):
    return type(value) is int


def _number(value):
    return type(value) in (int, float)


def _convert(value, spec, node, name):
    if value is None and not spec.required and spec.default is None:
        return None
    t = spec.type
    good = False
    if t == "int":
        good = _integer(value)
    elif t == "float":
        good = _number(value)
        if good:
            try:
                value = float(value)
            except OverflowError:
                good = False
    elif t == "str":
        good = type(value) is str
    elif t == "bool":
        good = type(value) is bool
    elif t == "weight":
        good = isinstance(value, WeightRef)
    elif t == "int_or_pair":
        if _integer(value):
            value = (value, value)
        good = type(value) is tuple and len(value) == 2 and all(_integer(x) for x in value)
    elif t in ("int_tuple", "axis"):
        if t == "axis" and _integer(value):
            value = (value,)
        good = type(value) is tuple and len(value) >= 1 and all(_integer(x) for x in value)
    elif t in ("float_tuple", "float_pair"):
        good = type(value) is tuple and len(value) >= 1 and all(_number(x) for x in value)
        if t == "float_pair":
            good = good and len(value) == 2
        if good:
            try:
                value = tuple(float(x) for x in value)
            except OverflowError:
                good = False
    if not good:
        _fail("E_ATTR_TYPE", node, f"{name} requires {t}")
    if t == "float_pair" and not value[0] < value[1]:
        _fail("E_ATTR_VALUE", node, f"{name} requires lo < hi")
    if spec.choices is not None and value not in spec.choices:
        _fail("E_ATTR_VALUE", node, f"{name} must be one of {spec.choices}")
    return value


class _Parser:
    def __init__(self, tree):
        self.tree = tree
        self.macros = {}
        self.ops = []
        self.regions = {}
        self.stack = []
        self.call_lines = []
        self.calls = 0

    def name(self, name, node, entry_param=False):
        if (not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name) or "__" in name
                or name in RESERVED_NAMES or name in self.macros
                or entry_param and re.fullmatch(r"v\d+", name)):
            _fail("E_NAME", node, f"Reserved or invalid name {name}")

    def arguments(self, fn, code):
        args = fn.args
        invalid = (args.posonlyargs or args.vararg or args.kwarg or args.kwonlyargs
                   or args.defaults or args.kw_defaults or fn.returns)
        if invalid:
            node = args.vararg or args.kwarg or (args.kwonlyargs[0] if args.kwonlyargs else fn)
            _fail(code, node, "Use named parameters without defaults, variadics or return annotations")
        if len({a.arg for a in args.args}) != len(args.args):
            _fail(code, fn, "Duplicate parameter")
        return args.args

    def literal(self, node, scope):
        if isinstance(node, ast.Constant) and (node.value is None or type(node.value) in (int, float, str, bool)):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            if isinstance(node.operand, ast.Constant) and _number(node.operand.value):
                return -node.operand.value
            _fail("E_EXPR", node, "A minus sign requires a numeric constant")
        if isinstance(node, ast.Name):
            if node.id == "W":
                _fail("E_WEIGHT_REF", node, 'Use W["weight.name"]')
            if node.id not in scope.env:
                _fail("E_UNDEFINED", node, f"Undefined name {node.id}")
            value = scope.env[node.id]
            if isinstance(value, _Tensor):
                _fail("E_EXPR", node, "A tensor cannot be an attribute")
            return value
        if isinstance(node, ast.Tuple):
            values = tuple(self.literal(x, scope) for x in node.elts)
            if not all(_number(x) for x in values):
                _fail("E_EXPR", node, "Tuple attributes contain only numbers")
            return values
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left, right = self.literal(node.left, scope), self.literal(node.right, scope)
            if type(left) is str and type(right) is str:
                return left + right
            _fail("E_EXPR", node, "Only string concatenation is allowed")
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == "W":
            try:
                value = self.literal(node.slice, scope)
            except PTDSLError:
                _fail("E_WEIGHT_REF", node, "Weight subscript must be a static string expression")
            if type(value) is not str:
                _fail("E_WEIGHT_REF", node, "Weight subscript must be a static string expression")
            return WeightRef(value)
        _fail("E_EXPR", node, "Expression is outside the PT-DSL whitelist")

    def header(self, fn):
        decorator = fn.decorator_list[0]
        if len(fn.decorator_list) != 1 or not (
                isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute)
                and isinstance(decorator.func.value, ast.Name)
                and decorator.func.value.id == "fhe" and decorator.func.attr == "program"
                and not decorator.args):
            _fail("E_HEADER", decorator, "Expected @fhe.program(keyword=value)")
        values = {}
        for kw in decorator.keywords:
            if kw.arg not in ("slots", "refresh", "placement", "l_max", "boot_out_level") or kw.arg in values:
                _fail("E_HEADER", decorator, "Unknown, expanded or duplicate header keyword")
            try:
                values[kw.arg] = self.literal(kw.value, _Scope({}))
            except PTDSLError:
                _fail("E_HEADER", decorator, "Header settings must be static")
        slots = values.get("slots")
        if not _integer(slots) or not 1024 <= slots <= 65536 or slots & (slots-1):
            _fail("E_HEADER", decorator, "slots must be a power of two in [1024,65536]")
        if values.get("refresh", "compiler") not in ("compiler", "agent"):
            _fail("E_HEADER", decorator, "refresh must be compiler or agent")
        if values.get("placement", "depth-dp") not in ("depth-dp", "optimized", "greedy"):
            _fail("E_HEADER", decorator, "Unsupported placement")
        for key in ("l_max", "boot_out_level"):
            if key in values and (not _integer(values[key]) or not 1 <= values[key] <= 64):
                _fail("E_HEADER", decorator, f"{key} must be in [1,64]")
        if values.get("refresh") == "agent" and (
                "l_max" not in values or "boot_out_level" not in values
                or values["boot_out_level"] > values["l_max"]):
            _fail("E_HEADER", decorator, "agent requires boot_out_level <= l_max")
        return Header(**values, lineno=decorator.lineno)

    def params(self, fn):
        result = []
        for arg in self.arguments(fn, "E_PARAM"):
            self.name(arg.arg, arg, entry_param=True)
            annotation = arg.annotation
            if not (isinstance(annotation, ast.Subscript) and isinstance(annotation.value, ast.Name)
                    and annotation.value.id in ("Cipher", "Plain")):
                _fail("E_PARAM", arg, "Expected Cipher[...] or Plain[...]")
            nodes = annotation.slice.elts if isinstance(annotation.slice, ast.Tuple) else [annotation.slice]
            if not nodes or any(not isinstance(n, ast.Constant) or not _integer(n.value) or n.value < 1 for n in nodes):
                _fail("E_PARAM", arg, "Tensor dimensions must be positive integer constants")
            result.append(Param(arg.arg, annotation.value.id.lower(), tuple(n.value for n in nodes), arg.lineno))
        if not any(p.kind == "cipher" for p in result):
            _fail("E_PARAM", fn, "At least one Cipher parameter is required")
        return tuple(result)

    def targets(self, statement):
        if len(statement.targets) != 1:
            _fail("E_STMT", statement, "Use one assignment target")
        target = statement.targets[0]
        nodes = target.elts if isinstance(target, ast.Tuple) else [target]
        if not nodes or not all(isinstance(node, ast.Name) for node in nodes):
            _fail("E_STMT", statement, "Targets are names or tuples of names")
        for node in nodes:
            self.name(node.id, node)
        return nodes, isinstance(target, ast.Tuple)

    def operator(self, call, statement, target_nodes, tuple_target, scope, region):
        name = call.func.id
        spec = OPS[name]
        inputs = []
        for node in call.args:
            if not isinstance(node, ast.Name):
                _fail("E_EXPR", node, "Operator inputs must be tensor names")
            if node.id not in scope.env:
                _fail("E_UNDEFINED", node, f"Undefined tensor {node.id}")
            value = scope.env[node.id]
            if not isinstance(value, _Tensor):
                _fail("E_MACRO", node, "A literal macro parameter cannot be a tensor input")
            inputs.append(value.vid)
        if (spec.inputs == "2+" and len(inputs) < 2
                or spec.inputs != "2+" and len(inputs) != spec.inputs):
            _fail("E_ARITY", statement, f"{name} requires {spec.inputs} inputs")
        supplied = {}
        for kw in call.keywords:
            if kw.arg is None:
                _fail("E_EXPR", statement, "Expanded keywords are forbidden")
            if kw.arg not in spec.attrs or kw.arg in supplied:
                _fail("E_ATTR_UNKNOWN", statement, f"{name}: unknown or duplicate attribute {kw.arg}")
            supplied[kw.arg] = self.literal(kw.value, scope)
        attrs = {}
        for key, field in spec.attrs.items():
            if field.required and key not in supplied:
                _fail("E_ATTR_MISSING", statement, f"{name}: missing {key}")
            attrs[key] = _convert(supplied.get(key, field.default), field, statement, key)
        if name == "avgpool" and attrs["stride"] is None:
            attrs["stride"] = attrs["k"]
        message = spec.validate(attrs)
        if message is not None:
            _fail("E_ATTR_VALUE", statement, message)
        expected = len(attrs["sizes"]) if spec.outputs == "split" else 1
        if (len(target_nodes) != expected or tuple_target != (spec.outputs == "split")):
            _fail("E_TARGET", statement, f"{name} requires {'a tuple of' if spec.outputs == 'split' else ''} {expected} target(s)")
        outputs = tuple(scope.next_id(node.id) for node in target_nodes)
        self.ops.append(Op(len(self.ops), name, tuple(inputs), outputs, attrs,
                           statement.lineno, region, tuple(self.stack), tuple(self.call_lines)))
        for node, vid in zip(target_nodes, outputs):
            scope.env[node.id] = _Tensor(vid)

    def macro(self, call, statement, targets, tuple_target, scope, region):
        name = call.func.id
        if tuple_target:
            _fail("E_TARGET", statement, "A macro call assigns one name")
        if name in self.stack or len(self.stack) >= 16:
            _fail("E_MACRO", statement, "Macro recursion or depth beyond 16")
        fn = self.macros[name]
        names = [arg.arg for arg in fn.args.args]
        if len(call.args) > len(names):
            _fail("E_MACRO", statement, "Too many macro arguments")
        nodes = dict(zip(names, call.args))
        for kw in call.keywords:
            if kw.arg not in names or kw.arg in nodes:
                _fail("E_MACRO", statement, "Unknown, duplicate or expanded macro argument")
            nodes[kw.arg] = kw.value
        if set(nodes) != set(names):
            _fail("E_MACRO", statement, "Missing macro argument")
        bindings = {}
        for key, node in nodes.items():
            if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == "W":
                _fail("E_MACRO", node, "Weight references are not macro arguments")
            if isinstance(node, ast.Name) and node.id in scope.env:
                bindings[key] = scope.env[node.id]
            else:
                value = self.literal(node, scope)
                if isinstance(value, WeightRef):
                    _fail("E_MACRO", node, "Weight references are not macro arguments")
                bindings[key] = value
        self.calls += 1
        local = _Scope(bindings, f"{name}__c{self.calls}__")
        self.stack.append(name)
        self.call_lines.append(statement.lineno)
        try:
            outputs = self.body(fn.body, local, region=region, macro=True)
        finally:
            self.call_lines.pop()
            self.stack.pop()
        scope.next_id(targets[0].id)
        scope.env[targets[0].id] = _Tensor(outputs[0])

    def with_region(self, statement, scope, region, macro):
        if macro or region is not None or len(statement.items) != 1:
            _fail("E_REGION", statement, "Regions are entry-only and cannot be nested")
        item = statement.items[0]
        call = item.context_expr
        if item.optional_vars is not None or not (
                isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "region"
                and len(call.args) == 1 and isinstance(call.args[0], ast.Constant)
                and type(call.args[0].value) is str and len(call.keywords) == 1
                and call.keywords[0].arg == "level_budget"):
            _fail("E_REGION", statement, "Expected region(string, level_budget=int)")
        budget = call.keywords[0].value
        name = call.args[0].value
        if (not isinstance(budget, ast.Constant) or not _integer(budget.value)
                or not 1 <= budget.value <= 64 or name in self.regions):
            _fail("E_REGION", statement, "Region names are unique and budgets are in [1,64]")
        self.regions[name] = Region(name, budget.value, statement.lineno)
        self.body(statement.body, scope, region=name, in_region=True)

    def body(self, statements, scope, *, region=None, macro=False, in_region=False):
        for index, statement in enumerate(statements):
            if isinstance(statement, ast.Assign):
                targets, tuple_target = self.targets(statement)
                call = statement.value
                if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
                    _fail("E_EXPR", statement, "Assignment requires a direct operator or macro call")
                if call.func.id in OPS:
                    self.operator(call, statement, targets, tuple_target, scope, region)
                elif call.func.id in self.macros:
                    self.macro(call, statement, targets, tuple_target, scope, region)
                else:
                    _fail("E_UNKNOWN_OP", statement, f"Unknown operator {call.func.id}")
            elif isinstance(statement, ast.With):
                self.with_region(statement, scope, region, macro)
            elif isinstance(statement, ast.Return):
                if in_region or index != len(statements)-1:
                    _fail("E_RETURN", statement, "Return must be the final function-level statement")
                value = statement.value
                nodes = value.elts if isinstance(value, ast.Tuple) and not macro else [value]
                if not nodes or any(not isinstance(node, ast.Name)
                        or not isinstance(scope.env.get(node.id), _Tensor) for node in nodes):
                    _fail("E_RETURN", statement, "Return defined tensor names; macros return one name")
                return tuple(scope.env[node.id].vid for node in nodes)
            else:
                _fail("E_STMT", statement, "Only call assignments, regions and return are allowed")
        if not in_region:
            _fail("E_RETURN", statements[-1], "Function must finish with return")
        return ()

    def macro_structure(self, fn):
        for index, statement in enumerate(fn.body):
            if isinstance(statement, ast.With):
                _fail("E_REGION", statement, "Regions are forbidden inside macros")
            if isinstance(statement, ast.Assign):
                self.targets(statement)
                call = statement.value
                if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
                    _fail("E_EXPR", statement, "Assignment requires a direct operator or macro call")
                if call.func.id not in OPS and call.func.id not in self.macros:
                    _fail("E_UNKNOWN_OP", statement, f"Unknown operator {call.func.id}")
            elif isinstance(statement, ast.Return):
                if index != len(fn.body)-1 or not isinstance(statement.value, ast.Name):
                    _fail("E_RETURN", statement, "A macro finishes by returning one tensor name")
            else:
                _fail("E_STMT", statement, "Forbidden macro statement")
        if not isinstance(fn.body[-1], ast.Return):
            _fail("E_RETURN", fn.body[-1], "Macro must finish with return")

    def run(self):
        for node in self.tree.body:
            if not isinstance(node, ast.FunctionDef):
                _fail("E_STMT", node, "Top level contains only function definitions")
        entries = [fn for fn in self.tree.body if fn.decorator_list]
        if not entries:
            raise PTDSLError("E_NO_ENTRY", 0, "Missing @fhe.program entry")
        if len(entries) > 1:
            _fail("E_MULTI_ENTRY", entries[1], "Multiple decorated entries")
        seen = set()
        for fn in self.tree.body:
            if fn.name in seen or fn is not entries[0] and fn.name in RESERVED_NAMES:
                _fail("E_MACRO", fn, "Duplicate function or reserved macro name")
            seen.add(fn.name)
        main = entries[0]
        self.macros = {fn.name: fn for fn in self.tree.body if fn is not main}
        for fn in self.macros.values():
            for arg in self.arguments(fn, "E_MACRO"):
                self.name(arg.arg, arg)
                if arg.annotation is not None:
                    _fail("E_MACRO", arg, "Macro parameters are unannotated")
            self.macro_structure(fn)
        header, params = self.header(main), self.params(main)
        outputs = self.body(main.body, _Scope({p.name: _Tensor(p.name) for p in params}))
        return Program(main.name, header, params, tuple(self.ops), outputs, self.regions)


def parse(text: str) -> Program:
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        raise PTDSLError("E_SYNTAX", exc.lineno or 0, exc.msg) from exc
    return _Parser(tree).run()
