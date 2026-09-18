"""Static checker for the initial generated Hecate FUNCTION fragment.

No eval/exec/import/tracing. This is a project-defined restricted subset of
the actual Hecate frontend, not an upstream DSL or an OS security sandbox.
Trusted harness owns imports, public arrays, hc.save, reference and parameters.
"""
import ast
import math
import re

from seal_artifact_gate import require
from packed_input_abi import CONTRACT as PACKED_CONTRACT, rotations as packed_rotations
from packed_input_abi import NATIVE_CONTRACT as PACKED_NATIVE_CONTRACT

IDENTIFIER = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,63}\Z")
RESERVED = {"hc", "golden", "x"}
CONTRACT_ROTATIONS = {"hecate-function-v0": (1, 2), "hecate-function-v1": (1, 2),
                      "hecate-function-v2": (-3, -2, -1, 1, 2, 3),
                      "hecate-function-v3": (-3, -2, -1, 1, 2, 3),
                      "hecate-function-v4": (-3, -2, -1, 1, 2, 3),
                      "hecate-function-v5": (-3, -2, -1, 1, 2, 3),
                      "hecate-function-v6": (-3, -2, -1, 1, 2, 3),
                      "hecate-function-v7": (-3, -2, -1, 1, 2, 3),
                      "hecate-function-v8": (-3, -2, -1, 1, 2, 3),
                      "hecate-function-v9": (-3, -2, -1, 1, 2, 3),
                      "hecate-function-v10": (-3, -2, -1, 1, 2, 3),
                      "hecate-function-v11": (-3, -2, -1, 1, 2, 3),
                      "hecate-function-v12": (-3, -2, -1, 1, 2, 3),
                      "hecate-function-v13": (-3, -2, -1, 1, 2, 3),
                      "hecate-function-v14": (-3, -2, -1, 1, 2, 3),
                      "hecate-function-v15": (-3, -2, -1, 1, 2, 3),
                      "hecate-function-v16": (-3, -2, -1, 1, 2, 3),
                      "hecate-function-v17": (-3, -2, -1, 1, 2, 3),
                      "hecate-function-v18": (-3, -2, -1, 1, 2, 3),
                      "hecate-function-v19": (-3, -2, -1, 1, 2, 3),
                      "hecate-function-v20": (-3, -2, -1, 1, 2, 3),
                      "hecate-function-v21": (-3, -2, -1, 1, 2, 3)}


def rotation_literal(node):
    """Signed integer literal only; never evaluate an arbitrary expression."""
    if type(node) is ast.Constant and type(node.value) is int:
        return node.value
    if type(node) is ast.UnaryOp and type(node.op) is ast.USub and type(node.operand) is ast.Constant and type(node.operand.value) is int:
        return -node.operand.value
    raise ValueError("Rotation step must be a signed integer literal")


CONTRACT_ROTATIONS['hecate-native-functions-v1'] = (-3,-2,-1,1,2,3)
CONTRACT_ROTATIONS['hecate-native-functions-v2'] = (-3,-2,-1,1,2,3)
CONTRACT_ROTATIONS['hecate-native-functions-v3'] = (-3,-2,-1,1,2,3)
CONTRACT_ROTATIONS['hecate-native-functions-v4'] = (-3,-2,-1,1,2,3)
CONTRACT_ROTATIONS['hecate-native-functions-v5'] = (-3,-2,-1,1,2,3)
CONTRACT_ROTATIONS['hecate-native-functions-v6'] = (-3,-2,-1,1,2,3)
CONTRACT_ROTATIONS['hecate-native-functions-v7'] = (-3,-2,-1,1,2,3)
CONTRACT_ROTATIONS[PACKED_CONTRACT] = packed_rotations(256)
CONTRACT_ROTATIONS[PACKED_NATIVE_CONTRACT] = packed_rotations(256)


def validate_function(source, public_constants, expected_outputs=1, *, contract="hecate-function-v0",
                      input_names=("x",), slot_period=4):
    """Return inferred layouts or raise ValueError. Constants are trusted data.

    A period-4 ciphertext is one packed vector, not four ciphertexts. A return
    list denotes ciphertexts, not slots. Reduction to a broadcast scalar is not
    inferred here: period remains conservative and output slot binding external.
    """
    require(type(contract) is str and contract in CONTRACT_ROTATIONS, "Unknown Hecate contract")
    packed=contract in (PACKED_CONTRACT,PACKED_NATIVE_CONTRACT)
    require(type(slot_period) is int and (packed or slot_period==4),'Slot period requires packed contract')
    allowed_rotations=packed_rotations(slot_period) if packed else CONTRACT_ROTATIONS[contract]
    if contract in ('hecate-native-functions-v1','hecate-native-functions-v2','hecate-native-functions-v3','hecate-native-functions-v4','hecate-native-functions-v5','hecate-native-functions-v6','hecate-native-functions-v7',PACKED_NATIVE_CONTRACT):
        from decorated_functions import validate
        plan = validate(source, public_constants, expected_outputs, input_names=input_names,
                        arrays=contract != 'hecate-native-functions-v1',starred_calls=contract in ('hecate-native-functions-v3','hecate-native-functions-v4'),
                        array_arithmetic=contract == 'hecate-native-functions-v4',public_loops=contract == 'hecate-native-functions-v5',
                        scalar_augmented=contract == 'hecate-native-functions-v6',array_mutation=contract in ('hecate-native-functions-v7',PACKED_NATIVE_CONTRACT),
                        slot_period=slot_period if contract==PACKED_NATIVE_CONTRACT else None)
        return dict(contract=contract, native_functions=plan,
                    inputs=[dict(name=n,kind='cipher',slot_period=slot_period) for n in input_names],
                    outputs=[dict(kind='cipher',slot_period=slot_period) for _ in range(expected_outputs)],
                    operator_counts={'expanded_native_work':plan['functions']['golden']['expanded_cost']},
                    rotation_steps=plan['rotation_steps'], syntax_type_layout_checked=True,
                    compilation_checked=False, encrypted_correctness_checked=False, safe_to_execute_untrusted=False)
    require(type(input_names) in (tuple, list) and all(type(n) is str and IDENTIFIER.fullmatch(n)
            and n not in ("hc", "golden") for n in input_names) and
            len(set(input_names)) == len(input_names), "Invalid encrypted input names")
    if contract in ('hecate-function-v6', 'hecate-function-v7', 'hecate-function-v8', 'hecate-function-v9', 'hecate-function-v10', 'hecate-function-v11', 'hecate-function-v12', 'hecate-function-v13', 'hecate-function-v14', 'hecate-function-v15', 'hecate-function-v16', 'hecate-function-v17', 'hecate-function-v18', 'hecate-function-v19', 'hecate-function-v20', 'hecate-function-v21'):
        if contract != 'hecate-function-v6':
            from function_construction import normalize
        else:
            from public_construction import normalize
        options = dict(closures=True) if contract == 'hecate-function-v8' else {}
        if contract == 'hecate-function-v9':
            options = dict(call_binding=True)
        if contract == 'hecate-function-v10':
            options = dict(public_iteration=True)
        if contract == 'hecate-function-v11':
            options = dict(function_literals=True)
        if contract == 'hecate-function-v12':
            options = dict(public_sequences=True)
        if contract == 'hecate-function-v13':
            options = dict(public_numbers=True)
        if contract == 'hecate-function-v14':
            options = dict(public_control=True)
        if contract == 'hecate-function-v15':
            options = dict(public_strings=True)
        if contract == 'hecate-function-v16':
            options = dict(public_polynomial=True)
        if contract == 'hecate-function-v17':
            options = dict(object_arrays=True)
        if contract == 'hecate-function-v18':
            options = dict(public_mappings=True)
        if contract == 'hecate-function-v19':
            options = dict(object_arithmetic=True)
        if contract == 'hecate-function-v20':
            options = dict(scalar_conversion=True)
        if contract == 'hecate-function-v21':
            options = dict(object_unary=True)
        result = normalize(source, public_constants, expected_outputs, input_names=input_names, **options)
        return dict(result['check'], contract=contract, public_construction=result['construction'])
    extended = contract == 'hecate-function-v5' or packed
    zero_input = contract == 'hecate-function-v4' or (extended and input_names[-1:] in (('zero_ct',), ['zero_ct']))
    valid_arity = (2 <= len(input_names) <= 5 and tuple(input_names) ==
                   (*('x', 'y', 'z', 't')[:len(input_names)-1], 'zero_ct')) if zero_input else (
                   2 <= len(input_names) <= 4 if contract == 'hecate-function-v3' else tuple(input_names) == ('x',))
    if extended and not zero_input:
        valid_arity = 1 <= len(input_names) <= 4 and tuple(input_names) == ('x', 'y', 'z', 't')[:len(input_names)]
    if packed:
        valid_arity=tuple(input_names) in (('x',),('x','zero_ct'))
    require(valid_arity,
            "Input signature incompatible with contract version")
    native_arithmetic = contract != "hecate-function-v0"
    require(isinstance(source, str) and len(source.encode()) <= 65536, "Source size limit")
    require(type(expected_outputs) is int and 1 <= expected_outputs <= (16 if packed else 4), "Invalid result count")
    require(type(public_constants) is dict and len(public_constants) <= (256 if packed else 128), "Invalid constants manifest")
    public = {}
    for name, value in public_constants.items():
        require(type(name) is str and bool(IDENTIFIER.fullmatch(name)) and name not in RESERVED and name not in input_names,
                "Invalid public constant name")
        values = value if type(value) is list else [value]
        require(len(values) in (1, slot_period), "Only scalar or declared-period public constants allowed")
        require(all(type(v) in (int, float) and abs(v) <= 1024 and math.isfinite(v) for v in values),
                "Public constants must be finite, bounded, real numbers")
        public[name] = ("plain", len(values))
    try:
        tree = ast.parse(source)
    except (SyntaxError, RecursionError, MemoryError) as error:
        raise ValueError("Invalid Python syntax or parser resource limit") from error
    nodes = list(ast.walk(tree))
    require(len(nodes) <= (12288 if packed else 4096), "AST node limit")
    require(len(tree.body) == 1 and type(tree.body[0]) is ast.FunctionDef, "Exactly one function; no module effects")
    function = tree.body[0]
    expected_signature = "def golden(" + ", ".join(input_names) + "):"
    expected_decorator = '@hc.func("' + ",".join(["c"] * len(input_names)) + '")'
    require(function.name == "golden" and function.returns is None and function.type_comment is None,
            "Requires unannotated golden function: " + expected_signature)
    args = function.args
    require([a.arg for a in args.args] == list(input_names) and all(a.annotation is None for a in args.args) and
            not args.posonlyargs and not args.vararg and not args.kwarg and not args.kwonlyargs and
            not args.defaults and not args.kw_defaults,
            "Encrypted parameter signature mismatch: " + expected_signature + " (no annotations or defaults)")
    require(len(function.decorator_list) == 1, "Requires exact Hecate decorator: " + expected_decorator)
    decorator = function.decorator_list[0]
    require(type(decorator) is ast.Call and not decorator.keywords and len(decorator.args) == 1 and
            type(decorator.args[0]) is ast.Constant and decorator.args[0].value == ",".join(["c"] * len(input_names)) and
            type(decorator.func) is ast.Attribute and decorator.func.attr == "func" and
            type(decorator.func.value) is ast.Name and decorator.func.value.id == "hc",
            "Requires exact encrypted Hecate decorator: " + expected_decorator + " (one string argument)")
    require(function.body and type(function.body[-1]) is ast.Return, "Requires final return")
    symbols = dict(public)
    symbols.update({name: ("cipher", slot_period) for name in input_names})
    counts = {"add": 0, "multiply": 0, "rotate": 0}
    if native_arithmetic:
        counts.update(subtract=0, negate=0)
    rotations = set()

    def expression(node, depth=0):
        require(depth <= 64, "Expression depth limit")
        if type(node) is ast.Name:
            require(node.id in symbols and type(node.ctx) is ast.Load, "Undefined or invalid value")
            return symbols[node.id]
        if native_arithmetic and type(node) is ast.UnaryOp and type(node.op) is ast.USub:
            operand = expression(node.operand, depth + 1)
            require(operand[0] == "cipher", "Unary negation requires ciphertext; public arithmetic is precomputed")
            counts["negate"] += 1
            return operand
        if type(node) is ast.BinOp and (type(node.op) in (ast.Add, ast.Mult) or
                                       native_arithmetic and type(node.op) is ast.Sub):
            left, right = expression(node.left, depth + 1), expression(node.right, depth + 1)
            require((left[0] == "cipher" and right[0] in ("cipher", "plain")) or
                    (extended and left[0] == "plain" and right[0] == "cipher"),
                    "Left operand must be ciphertext; precompute public-only arithmetic")
            counts[{ast.Add: "add", ast.Mult: "multiply", ast.Sub: "subtract"}[type(node.op)]] += 1
            return ("cipher", max(left[1], right[1]))
        if type(node) is ast.Call:
            require(not node.keywords and len(node.args) == 1 and type(node.func) is ast.Attribute and
                    node.func.attr == "rotate", "Only ciphertext.rotate(step) calls allowed")
            operand = expression(node.func.value, depth + 1)
            step = rotation_literal(node.args[0])
            require(operand[0] == "cipher" and step in allowed_rotations,
                    "Rotation requires ciphertext and a provisioned contract step")
            counts["rotate"] += 1
            rotations.add(step)
            return operand
        raise ValueError(f"Expression outside verified subset: {type(node).__name__}")

    for statement in function.body[:-1]:
        if extended and type(statement) is ast.AugAssign:
            require(type(statement.target) is ast.Name and statement.target.id in symbols and
                    symbols[statement.target.id][0] == 'cipher' and statement.target.id != 'zero_ct' and
                    type(statement.op) in (ast.Add, ast.Sub, ast.Mult),
                    'Augmented assignment requires an existing writable ciphertext name and +=/-=/*=')
            # Expr augmented operators return new values; existing aliases remain unchanged.
            name = statement.target.id
            symbols[name] = expression(ast.BinOp(left=ast.Name(id=name, ctx=ast.Load()),
                                                op=statement.op, right=statement.value))
            continue
        require(type(statement) is ast.Assign and len(statement.targets) == 1 and
                type(statement.targets[0]) is ast.Name and statement.type_comment is None,
                "Only straight-line single-name assignments allowed")
        name = statement.targets[0].id
        writable = (name not in public and name not in ('hc', 'golden', 'zero_ct')) if extended else (
                    name not in symbols and name not in RESERVED)
        require(bool(IDENTIFIER.fullmatch(name)) and writable,
                "No mutation, name rebinding or reserved names")
        value = expression(statement.value)
        require(value[0] == "cipher", "Only ciphertext temporaries allowed")
        symbols[name] = value
    returns = function.body[-1].value
    expressions = returns.elts if type(returns) is ast.List else [returns]
    require(len(expressions) == expected_outputs, "Output ciphertext count mismatch")
    output_types = [expression(node) for node in expressions]
    require(all(value[0] == "cipher" for value in output_types), "Outputs must be ciphertexts")
    result = dict(contract=contract, input={"kind": "cipher", "slot_period": slot_period},
                outputs=[dict(kind=k, slot_period=p) for k, p in output_types],
                operator_counts=counts, rotation_steps=sorted(rotations),
                syntax_type_layout_checked=True, compilation_checked=False,
                encrypted_correctness_checked=False, safe_to_execute_untrusted=False)
    if packed or contract in ("hecate-function-v3", "hecate-function-v4", "hecate-function-v5"):
        result.pop("input")
        result["inputs"] = [{"name": name, "kind": "cipher", "slot_period": slot_period} for name in input_names]
    if zero_input:
        result['auxiliary_encrypted_zero'] = 'zero_ct'
    return result
