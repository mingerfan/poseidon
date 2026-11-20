"""One saved-math HEIR diagnostic with an explicit target capability.

Keep the original failed host program, strict train0..2 proof, parameters and
complete pipeline. One target override is added immediately after its original
Lattigo/CKKS annotation. The installed flag was accepted on an empty module;
its effect on the model's compact conversion path remains to be observed.
"""
import ast
import inspect
import json
from pathlib import Path
import shutil
from types import FunctionType, SimpleNamespace

from . import heir_saved_math_no_linear_fusion as base

MODULE = 'pt_agent.experiments.heir_saved_math_explicit_linear_transform'
ROUND = 'HEIR-SAVED-MATH-EXPLICIT-LINEAR-TRANSFORM'
MODE = 'closed-host-JSON-original-pipeline-explicit-target-native-diagnostic'
VARIANT = 'original-pipeline-plus-explicit-hasKernelLinearTransform-1-v1'
ANNOTATION_PASS = 'annotate-module{backend=lattigo scheme=ckks}'
OVERRIDE_PASS = 'apply-config-override{config=hasKernelLinearTransform=1}'
SCOPE = ('Native-only saved-math diagnostic with the original pipeline and '
    'one explicit target capability. The empty-module check established flag '
    'acceptance; the installed default, model conversion branch, target IR, Go '
    'code generation, quality, security and CKKS require separate observations.')
TOTAL_SECONDS, NATIVE_SECONDS = base.TOTAL_SECONDS, base.NATIVE_SECONDS
GIB, CASE_BYTES, RESERVE_BYTES = base.GIB, base.CASE_BYTES, base.RESERVE_BYTES
FAILURE_FLAGS, ZERO_FIELDS = base.FAILURE_FLAGS, base.ZERO_FIELDS


def add_explicit_linear_transform(pipeline):
    """Insert one complete target pass and preserve every original character."""
    if (not isinstance(pipeline, str)
            or pipeline.count('annotate-module') != 1
            or pipeline.count('linalg-fuse-linear-ops') != 1
            or 'apply-config-override' in pipeline
            or 'backend.config_override' in pipeline
            or 'hasKernelLinearTransform' in pipeline):
        raise ValueError('One original target annotation/fusion and no existing capability override are required')
    lines = pipeline.splitlines(keepends=True)
    fusion = [line for line in lines if line.strip() == 'linalg-fuse-linear-ops,']
    matches = [(index, line) for index, line in enumerate(lines)
        if line.strip() == ANNOTATION_PASS+',']
    if len(fusion) != 1 or len(matches) != 1:
        raise ValueError('Exact standalone original Lattigo/CKKS annotation and fusion entries are required')
    index, line = matches[0]
    newline = '\r\n' if line.endswith('\r\n') else '\n' if line.endswith('\n') else ''
    if not newline:
        raise ValueError('The original target annotation must retain its following pipeline line')
    indentation = line[:len(line)-len(line.lstrip())]
    return ''.join(lines[:index+1]+[indentation+OVERRIDE_PASS+','+newline]+lines[index+1:])


def _clone(function, namespace):
    if not isinstance(function, FunctionType) or hasattr(function, '__wrapped__'):
        raise TypeError('Only original undecorated ordinary functions are cloned')
    return FunctionType(function.__code__, namespace, function.__name__,
        function.__defaults__, function.__closure__)


def source_gate(args):
    namespace = dict(vars(base), remove_linear_fusion=add_explicit_linear_transform)
    checked = _clone(base.source_gate, namespace)(args)
    checked['candidate_configuration'] = dict(original_linear_fusion_retained=True,
        pass_added=OVERRIDE_PASS, added_count=1, immediately_after=ANNOTATION_PASS,
        other_pass_order_unchanged=True, original_failure_preserved=True,
        requested_target_capability=dict(hasKernelLinearTransform=1),
        installed_default_or_model_branch_verified=False)
    return checked


def native_command(args, out, checked):
    namespace = dict(vars(base), remove_linear_fusion=add_explicit_linear_transform)
    return _clone(base.native_command, namespace)(args, out, checked)


def _snapshot(out, args):
    rows = base._snapshot(out, args)
    source = Path(__file__).resolve();target = out/'source'/source.name
    shutil.copyfile(source, target)
    rows.append(dict(original=str(source), snapshot=str(target)))
    return rows


def durable_json(path, result):
    if Path(path).name == 'result.json' and result.get('round') == ROUND:
        if result.get('status') not in ('running', 'prepared', 'ok', 'error'):
            result['original_termination_status'] = result['status'];result['status'] = 'error'
        result['scope'] = SCOPE
        result['original_linear_fusion_retained'] = True
        result['requested_target_capability'] = dict(hasKernelLinearTransform=1)
        result['model_conversion_branch_verified'] = False
        result['model_specific_native_codegen_verified'] = False
        if result.get('status') != 'ok':result['model_compiled_accepted'] = False
    base.durable_json(path, result)


def _execute_tree():
    """Keep the trusted supervisor body apart from two recording sites."""
    tree = ast.parse(inspect.getsource(base.execute));function = tree.body[0]
    if not isinstance(function, ast.FunctionDef) or function.name != 'execute':
        raise ValueError('The original ordinary execute source shape changed')
    spans = [node for node in ast.walk(function) if isinstance(node, ast.Constant)
        and node.value == 'saved-math-no-linear-fusion-continuation']
    requests = [node for node in ast.walk(function) if isinstance(node, ast.Assign)
        and len(node.targets) == 1 and isinstance(node.targets[0], ast.Subscript)
        and isinstance(node.targets[0].value, ast.Name) and node.targets[0].value.id == 'result'
        and isinstance(node.targets[0].slice, ast.Constant) and node.targets[0].slice.value == 'native_request']
    if len(spans) != 1 or len(requests) != 1:
        raise ValueError('Exactly the original stage label and native request recording sites are required')
    call = requests[0].value
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name) or call.func.id != 'dict':
        raise ValueError('Original native request metadata shape changed')
    keywords = {node.arg:node for node in call.keywords}
    removed = keywords.get('pass_removed');count = keywords.get('removed_count')
    if (set(keywords) != {'parameters', 'source_mlir', 'pipeline', 'pass_removed', 'removed_count', 'other_pass_order_unchanged'}
            or not isinstance(removed.value, ast.Constant) or removed.value.value != 'linalg-fuse-linear-ops'
            or not isinstance(count.value, ast.Constant) or count.value.value != 1):
        raise ValueError('Original one-pass request recording changed')
    spans[0].value = 'saved-math-explicit-linear-transform-continuation'
    removed.arg = 'pass_added';removed.value.value = OVERRIDE_PASS
    count.arg = 'added_count'
    return ast.fix_missing_locations(tree)


def _closed_execute_tree():
    """Mask cancellation through monitor cleanup and durable finalization."""
    tree = _execute_tree()
    targets = [node for node in ast.walk(tree) if isinstance(node, ast.Try)
        and node.finalbody and ast.dump(node.finalbody[0], include_attributes=False)
        == ast.dump(ast.parse('stop.set()').body[0], include_attributes=False)]
    if len(targets) != 1:
        raise ValueError('One original stop/join/final-persistence block is required')
    target = targets[0]
    cancel = ast.parse('signal.setitimer(signal.ITIMER_REAL, 0)').body[0]
    restore = ast.parse('for sig, handler in old.items():\n    signal.signal(sig, handler)').body[0]
    same = lambda left, right: ast.dump(left, include_attributes=False) == ast.dump(right, include_attributes=False)
    if (sum(same(node, cancel) for node in target.finalbody) != 1
            or sum(same(node, restore) for node in target.finalbody) != 1):
        raise ValueError('Original signal timer and handler restoration sites changed')
    original = [node for node in target.finalbody if not same(node, cancel) and not same(node, restore)]
    prefix = ast.parse('for sig in old:\n    signal.signal(sig, signal.SIG_IGN)\nsignal.setitimer(signal.ITIMER_REAL, 0)').body
    suffix = ast.parse('if result["monitor_stopped"]:\n    for sig, handler in old.items():\n        signal.signal(sig, handler)').body
    # A live monitor or failed persistence leaves signals ignored until CLI exit.
    # The original completed/native facts and monitor-error policy remain intact.
    target.finalbody = prefix+original+suffix
    return ast.fix_missing_locations(tree)


def _namespace():
    namespace = dict(vars(base), MODULE=MODULE, ROUND=ROUND, MODE=MODE, VARIANT=VARIANT,
        source_gate=source_gate, native_command=native_command, _snapshot=_snapshot,
        durable_json=durable_json)
    namespace['original'] = SimpleNamespace(**dict(vars(base.original), durable_json=durable_json))
    namespace['_registry'] = _clone(base._registry, namespace)
    exec(compile(_closed_execute_tree(), str(Path(__file__).resolve())+'::original-execute', 'exec'), namespace)
    return namespace


def execute(args):
    return _namespace()['execute'](args)


def parser():
    command = base.parser();command.description = __doc__
    return command


def main(argv=None):
    args = parser().parse_args(argv)
    for key, value in vars(args).items():
        if isinstance(value, Path):setattr(args, key, value.expanduser().resolve())
    result = execute(args)
    print(json.dumps({key: result.get(key) for key in
        ('status', 'error', 'model_compiled', 'model_compiled_accepted')}))
    return int(result['status'] not in ('prepared', 'ok'))


if __name__ == '__main__':raise SystemExit(main())
