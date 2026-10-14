"""Structural coverage of the *local* Hecate allowlist, not semantic proof.

Only inspect a function after the authoritative validator accepts it. Count
typed expressions and SSA dependencies, never search generated source strings.
Dead assignments are reported separately: they do not establish output coverage.
No candidate Python, Hecate tracing, model provider or filesystem access here.
"""
import ast
from collections import Counter

from hecate_contract import validate_function, rotation_literal


def feature_catalog():
    """Finite observation partitions, NOT all programs or all upstream Hecate."""
    features = {}
    for count in range(1, 5):
        features[f"inputs.{count}"] = "function_io"
        features[f"outputs.{count}"] = "function_io"
    features.update({"statement.assignment": "function_io", "statement.alias": "function_io",
                     "return.single": "function_io", "return.list": "function_io",
                     "expression.nested": "function_io", "negate.cipher": "negate"})
    for op in ("add", "subtract", "multiply"):
        for operand in ("cipher", "scalar", "length1", "length4"):
            features[f"{op}.{operand}"] = op
    for step in (-3, -2, -1, 1, 2, 3):
        features[f"rotate.{step:+d}"] = "rotation"
    return features


FEATURES = feature_catalog()
EXTENDED_FEATURES = dict(FEATURES, **{'statement.rebinding': 'function_io'})
for _op in ('add', 'subtract', 'multiply'):
    EXTENDED_FEATURES['statement.augmented.' + _op] = 'function_io'
    for _kind in ('scalar', 'length1', 'length4'):
        EXTENDED_FEATURES[_op + '.reverse.' + _kind] = _op


def analyze_source(source, constants, expected_outputs=1, *,
                   contract="hecate-function-v0", input_names=("x",)):
    check = validate_function(source, constants, expected_outputs,
                              contract=contract, input_names=input_names)
    if contract in ('hecate-native-functions-v1','hecate-native-functions-v2','hecate-native-functions-v3','hecate-native-functions-v4','hecate-native-functions-v5','hecate-native-functions-v6','hecate-native-functions-v7'):
        raise ValueError('Native function call coverage requires a dedicated call-graph audit; '
                         'single-function coverage cannot be reused')
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
        expanded = normalize(source, constants, expected_outputs, input_names=input_names, **options)
        result = analyze_source(expanded['source'], expanded.get('constants',constants), expected_outputs,
                                contract='hecate-function-v5', input_names=input_names)
        result.update(contract=contract, static_check=check, public_construction=expanded['construction'])
        result['interpretation'] = ('Ciphertext dependencies after public expansion; construction counters '
                                   'are not proof of output-reachable coverage or semantic equivalence.')
        return result
    fn = ast.parse(source).body[0]
    extended = contract == 'hecate-function-v5'
    # Versioned definition identities prevent x = ... from replacing earlier
    # dependencies. Aliases must continue to refer to the value they captured.
    bindings = {}
    symbols = {name: "cipher" for name in input_names}
    symbols.update({name: ("scalar" if type(value) is not list else f"length{len(value)}")
                    for name, value in constants.items()})
    all_counts, live_counts = Counter(), Counter()
    expressions = {}  # Per assignment; count each SSA definition once, not per use.

    def inspect(node, counts, dependencies):
        if type(node) is ast.Name:
            if node.id in bindings:
                dependencies.add(bindings[node.id])
            return symbols[node.id]
        if type(node) is ast.UnaryOp:
            inspect(node.operand, counts, dependencies)
            counts["negate.cipher"] += 1
        elif type(node) is ast.BinOp:
            left = inspect(node.left, counts, dependencies)
            right = inspect(node.right, counts, dependencies)
            op = {ast.Add: "add", ast.Sub: "subtract", ast.Mult: "multiply"}[type(node.op)]
            counts[f"{op}.{right}" if left == 'cipher' else f"{op}.reverse.{left}"] += 1
        elif type(node) is ast.Call:
            inspect(node.func.value, counts, dependencies)
            counts[f"rotate.{rotation_literal(node.args[0]):+d}"] += 1
        else:
            raise ValueError("Coverage analyzer drift from validated expression grammar")
        children = ([node.operand] if type(node) is ast.UnaryOp else
                    [node.left, node.right] if type(node) is ast.BinOp else [node.func.value])
        if any(type(child) is not ast.Name for child in children):
            counts["expression.nested"] += 1
        return "cipher"

    for index, statement in enumerate(fn.body[:-1]):
        augmented = type(statement) is ast.AugAssign
        name = statement.target.id if augmented else statement.targets[0].id
        node = (ast.BinOp(left=ast.Name(id=name, ctx=ast.Load()), op=statement.op, right=statement.value)
                if augmented else statement.value)
        identity = f'{name}@{index}' if extended else name
        counts, dependencies = Counter({"statement.assignment": 1}), set()
        if extended and name in symbols:
            counts['statement.rebinding'] += 1
        if augmented:
            op = {ast.Add: 'add', ast.Sub: 'subtract', ast.Mult: 'multiply'}[type(statement.op)]
            counts['statement.augmented.' + op] += 1
        if type(node) is ast.Name:
            counts["statement.alias"] += 1
        symbols[name] = inspect(node, counts, dependencies)
        expressions[identity] = (counts, dependencies)
        bindings[name] = identity
        all_counts.update(counts)
    returned = fn.body[-1].value
    results = returned.elts if type(returned) is ast.List else [returned]
    logical_count = len(input_names) - int('auxiliary_encrypted_zero' in check)
    live_counts.update({f"inputs.{logical_count}": 1, f"outputs.{len(results)}": 1,
                        "return.list" if type(returned) is ast.List else "return.single": 1})
    pending = set()
    for node in results:
        inspect(node, live_counts, pending)
    all_counts.update(live_counts)
    used = set()
    while pending:
        name = pending.pop()
        if name not in used:
            used.add(name)
            counts, dependencies = expressions[name]
            live_counts.update(counts)
            pending.update(dependencies - used)
    if not set(all_counts) <= set(EXTENDED_FEATURES if extended else FEATURES):
        raise ValueError("Coverage feature missing from catalog")
    return dict(contract=contract, static_check=check,
                present_counts=dict(sorted(all_counts.items())),
                output_dependency_counts=dict(sorted(live_counts.items())),
                dead_assignments=sorted(set(expressions) - used),
                auxiliary_encrypted_zero=('auxiliary_encrypted_zero' in check),
                semantic_equivalence_proven=False,
                interpretation="Syntactic output dependency, not proof that optimization retains each operation.")
