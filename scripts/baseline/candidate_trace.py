"""Trusted AST evaluator over real Hecate objects; NEVER execute candidate Python."""
import ast
import json
from pathlib import Path

from candidate_contract import strict_json, validate_candidate, request_input_names
from hecate_contract import rotation_literal


def load_frontend():
    # __init__.py also imports runner.py, whose default arguments read Path.home().
    # Use the same pinned expr.py frontend directly; no runtime/key imports needed.
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location("hecate_expr", "/frontend/hecate/expr.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # Fixed trusted dependency, never candidate source.
    return module


def evaluate_tree(source, constants, encrypted_input=None, *, encrypted_inputs=None):
    """Only call after validation. Operator meanings come from Hecate objects.

    This is frontend construction, NOT plaintext evaluation or a HEVM backend.
    """
    function = ast.parse(source).body[0]
    symbols = dict(constants)
    if encrypted_inputs is None:
        symbols["x"] = encrypted_input
    else:
        # The caller binds validated ordered arguments; never collapse inputs to x.
        names = [a.arg for a in function.args.args]
        if encrypted_input is not None or type(encrypted_inputs) is not dict or set(encrypted_inputs) != set(names):
            raise ValueError("Encrypted AST binding mismatch")
        if set(constants) & set(names):
            raise ValueError("Input shadows public constant")
        symbols.update(encrypted_inputs)

    def expression(node):
        if type(node) is ast.Name:
            return symbols[node.id]
        if type(node) is ast.UnaryOp and type(node.op) is ast.USub:
            return -expression(node.operand)
        if type(node) is ast.BinOp and type(node.op) in (ast.Add, ast.Mult, ast.Sub):
            a, b = expression(node.left), expression(node.right)
            if type(node.op) is ast.Sub:
                return a - b
            return a + b if type(node.op) is ast.Add else a * b
        if type(node) is ast.Call:
            # No getattr/eval: this is the one method in the validated grammar.
            return expression(node.func.value).rotate(rotation_literal(node.args[0]))
        raise ValueError("Unexpected validated AST expression")

    for statement in function.body[:-1]:
        if type(statement) is ast.AugAssign:
            name = statement.target.id
            value = symbols[name]
            other = expression(statement.value)
            # Use actual Expr augmented dispatch, not mutation of aliased objects.
            if type(statement.op) is ast.Add:
                value += other
            elif type(statement.op) is ast.Sub:
                value -= other
            elif type(statement.op) is ast.Mult:
                value *= other
            else:
                raise ValueError('Unexpected validated augmented operator')
            symbols[name] = value
        elif type(statement) is ast.Assign:
            symbols[statement.targets[0].id] = expression(statement.value)
        else:
            raise ValueError('Unexpected validated AST statement')
    result = function.body[-1].value
    return [expression(x) for x in result.elts] if type(result) is ast.List else expression(result)


def save_trace(hc, request, construction):
    """Both construction paths must emit the same driver evidence contract."""
    hc.save('/out', '/out')
    Path('/out/trace-evidence.json').write_text(json.dumps(dict(
        frontend='real_Hecate', candidate_python_executed=False,
        construction=construction, request_id=request['request_id'])))


def main():
    import resource  # Linux execution limits; pure AST dispatch also has host-side tests.
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (45, 50))
    resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024**2, 16 * 1024**2))
    payload = strict_json(Path("/payload.json").read_text())
    candidate, request = payload["candidate"], payload["request"]
    validate_candidate(candidate, request)
    from packed_input_abi import NATIVE_TASKS, NATIVE_EXERCISE_TASK
    packed_native = request['task'] in NATIVE_TASKS
    packed_exercise = request['task']==NATIVE_EXERCISE_TASK
    if packed_native or request['task'] in ('hecate-native-function-synthesis-v1', 'hecate-native-function-synthesis-v2', 'hecate-native-function-synthesis-v3', 'hecate-native-function-synthesis-v4', 'hecate-native-function-synthesis-v5', 'hecate-native-function-synthesis-v6', 'hecate-native-function-synthesis-v7', 'hecate-native-function-synthesis-v8', 'hecate-native-function-synthesis-v9', 'hecate-native-function-synthesis-v10'):
        from decorated_functions import register
        hc = load_frontend()
        calls = []
        storage_events = []
        star_events = []
        augmented_events = []
        mutation_events = []
        _, plan = register(candidate['hecate_source'], request['public_constants'], hc,
                           request['layout']['output_ciphertexts'], input_names=request_input_names(request),
                           observe=calls.append, arrays=request['task'] in ('hecate-native-function-synthesis-v3','hecate-native-function-synthesis-v4','hecate-native-function-synthesis-v5'),
                           starred_calls=request['task'] in ('hecate-native-function-synthesis-v5','hecate-native-function-synthesis-v6','hecate-native-function-synthesis-v8'),
                           array_arithmetic=request['task'] == 'hecate-native-function-synthesis-v6',
                           public_loops=request['task'] == 'hecate-native-function-synthesis-v7',
                           scalar_augmented=request['task'] == 'hecate-native-function-synthesis-v9',
                           array_mutation=packed_native or request['task'] == 'hecate-native-function-synthesis-v10',
                           slot_period=request['layout']['input_slot_period'] if packed_native else None,
                           observe_mutation=mutation_events.append if packed_native or request['task'] == 'hecate-native-function-synthesis-v10' else None,
                           observe_augmented=augmented_events.append if packed_native or request['task'] == 'hecate-native-function-synthesis-v9' else None,
                           observe_storage=storage_events.append if packed_exercise or request['task'] == 'hecate-native-function-synthesis-v4' else None,
                           observe_starred=star_events.append if packed_exercise or request['task'] == 'hecate-native-function-synthesis-v8' else None)
        Path('/out/native-function-plan.json').write_text(json.dumps(plan, sort_keys=True))
        save_trace(hc, request, 'validated_native_AST_to_Hecate_functions')
        Path('/out/native-call-events.json').write_text(json.dumps(calls, sort_keys=True))
        if packed_exercise or request['task'] == 'hecate-native-function-synthesis-v4':
            Path('/out/native-array-events.json').write_text(json.dumps(storage_events, sort_keys=True))
        if packed_exercise or request['task'] == 'hecate-native-function-synthesis-v8':
            Path('/out/native-star-events.json').write_text(json.dumps(star_events, sort_keys=True))
        if packed_native or request['task'] == 'hecate-native-function-synthesis-v9':
            Path('/out/native-augmented-events.json').write_text(json.dumps(augmented_events, sort_keys=True))
        if packed_native or request['task'] == 'hecate-native-function-synthesis-v10':
            Path('/out/native-array-mutation-events.json').write_text(json.dumps(mutation_events, sort_keys=True))
        return
    source = candidate['hecate_source']
    public_constants = request['public_constants']
    if request['task'] in ('hecate-function-synthesis-v7', 'hecate-function-synthesis-v8', 'hecate-function-synthesis-v9', 'hecate-function-synthesis-v10', 'hecate-function-synthesis-v11', 'hecate-function-synthesis-v12', 'hecate-function-synthesis-v13', 'hecate-function-synthesis-v14', 'hecate-function-synthesis-v15', 'hecate-function-synthesis-v16', 'hecate-function-synthesis-v17', 'hecate-function-synthesis-v18', 'hecate-function-synthesis-v19', 'hecate-function-synthesis-v20', 'hecate-function-synthesis-v21', 'hecate-function-synthesis-v22'):
        if request['task'] != 'hecate-function-synthesis-v7':
            from function_construction import normalize
        else:
            from public_construction import normalize
        options = dict(closures=True) if request['task'] == 'hecate-function-synthesis-v9' else {}
        if request['task'] == 'hecate-function-synthesis-v10':
            options = dict(call_binding=True)
        if request['task'] == 'hecate-function-synthesis-v11':
            options = dict(public_iteration=True)
        if request['task'] == 'hecate-function-synthesis-v12':
            options = dict(function_literals=True)
        if request['task'] == 'hecate-function-synthesis-v13':
            options = dict(public_sequences=True)
        if request['task'] == 'hecate-function-synthesis-v14':
            options = dict(public_numbers=True)
        if request['task'] == 'hecate-function-synthesis-v15':
            options = dict(public_control=True)
        if request['task'] == 'hecate-function-synthesis-v16':
            options = dict(public_strings=True)
        if request['task'] == 'hecate-function-synthesis-v17':
            options = dict(public_polynomial=True)
        if request['task'] == 'hecate-function-synthesis-v18':
            options = dict(object_arrays=True)
        if request['task'] == 'hecate-function-synthesis-v19':
            options = dict(public_mappings=True)
        if request['task'] == 'hecate-function-synthesis-v20':
            options = dict(object_arithmetic=True)
        if request['task'] == 'hecate-function-synthesis-v21':
            options = dict(scalar_conversion=True)
        if request['task'] == 'hecate-function-synthesis-v22':
            options = dict(object_unary=True)
        expanded = normalize(source, request['public_constants'], request['layout']['output_ciphertexts'],
                             input_names=request_input_names(request), **options)
        source = expanded['source']
        Path('/out/normalized-source.py').write_text(source)
        Path('/out/construction.json').write_text(json.dumps(expanded['construction']))
        if request['task'] in ('hecate-function-synthesis-v14','hecate-function-synthesis-v15', 'hecate-function-synthesis-v16', 'hecate-function-synthesis-v17', 'hecate-function-synthesis-v18', 'hecate-function-synthesis-v19', 'hecate-function-synthesis-v20', 'hecate-function-synthesis-v21', 'hecate-function-synthesis-v22'):
            public_constants = expanded['constants']
            Path('/out/derived-constants.json').write_text(json.dumps(expanded['derived_constants'],sort_keys=True,allow_nan=False))
    import numpy as np
    hc = load_frontend()
    constants = {k: np.asarray(v if type(v) is list else [v], dtype=np.float64)
                 for k, v in public_constants.items()}

    names = request_input_names(request)
    @hc.func(",".join(["c"] * len(names)))
    def golden(*inputs):
        # Plain on the left must use Hecate dispatch, not NumPy's object ufunc.
        # Construct constants inside the active frontend function.
        bound = ({k: hc.resolveType(v) for k, v in constants.items()}
                 if request['task'] in ('hecate-periodic-packed-synthesis-v1', 'hecate-chunked-input-synthesis-v1', 'hecate-function-synthesis-v6', 'hecate-function-synthesis-v7',
                                        'hecate-function-synthesis-v8', 'hecate-function-synthesis-v9',
                                        'hecate-function-synthesis-v10', 'hecate-function-synthesis-v11',
                                        'hecate-function-synthesis-v12', 'hecate-function-synthesis-v13',
                                        'hecate-function-synthesis-v14', 'hecate-function-synthesis-v15', 'hecate-function-synthesis-v16', 'hecate-function-synthesis-v17', 'hecate-function-synthesis-v18', 'hecate-function-synthesis-v19', 'hecate-function-synthesis-v20', 'hecate-function-synthesis-v21', 'hecate-function-synthesis-v22') else constants)
        return evaluate_tree(source, bound,
                             encrypted_inputs=dict(zip(names, inputs)))

    save_trace(hc, request, 'validated_AST_to_Hecate_objects')


if __name__ == "__main__":
    main()
