"""Saved original host mathematics with one official layout pass restored.

Retain the original linear fusion, model source, parameters and strict saved
three-training-input proof. The new native-only attempt adds the documented
layout-optimization pass immediately after layout-propagation. An earlier
default layout-optimization crash remains relevant; this entry claims no fix.
"""
import ast
import copy
import inspect
import json
from pathlib import Path
import shutil
import sys
from types import FunctionType, SimpleNamespace

from . import heir_saved_math_no_linear_fusion as base

MODULE = 'pt_agent.experiments.heir_saved_math_layout_16384'
ROUND = 'HEIR-SAVED-MATH-ORIGINAL-FUSION-LAYOUT-16384'
MODE = 'closed-host-JSON-original-fusion-one-layout-optimization-native-only'
VARIANT = 'original-linear-fusion-plus-one-layout-optimization-16384-v1'
LAYOUT_PASS = 'layout-optimization{min-slot-count=16384}'
PROPAGATION_PASS = 'layout-propagation{min-slot-count=16384}'
SCOPE = ('Native-only saved-math continuation of the original failed host case; '
    'all original passes including linear fusion are retained and one documented '
    'layout-optimization pass is restored. Earlier crash risk remains. Actual '
    'target IR, layout semantics, Go code generation, quality, security and CKKS '
    'acceptance require their own observations.')
TOTAL_SECONDS, NATIVE_SECONDS = base.TOTAL_SECONDS, base.NATIVE_SECONDS
GIB, CASE_BYTES, RESERVE_BYTES = base.GIB, base.CASE_BYTES, base.RESERVE_BYTES
FAILURE_FLAGS = base.FAILURE_FLAGS
ZERO_FIELDS = base.ZERO_FIELDS


def add_layout_optimization(pipeline):
    """Add one complete pass entry; preserve all original characters."""
    if (not isinstance(pipeline, str)
            or pipeline.count('linalg-fuse-linear-ops') != 1
            or pipeline.count('layout-propagation') != 1
            or 'layout-optimization' in pipeline):
        raise ValueError('Original fusion and one original propagation, with no layout optimization, are required')
    lines = pipeline.splitlines(keepends=True)
    fusion = [line for line in lines if line.strip() == 'linalg-fuse-linear-ops,']
    matches = [(index, line) for index, line in enumerate(lines)
        if line.strip() == PROPAGATION_PASS+',']
    if len(fusion) != 1 or len(matches) != 1:
        raise ValueError('Exact standalone original fusion and16384-slot propagation entries are required')
    index, line = matches[0]
    newline = '\r\n' if line.endswith('\r\n') else '\n' if line.endswith('\n') else ''
    if not newline:raise ValueError('The propagation entry must have its original following pipeline line')
    indentation = line[:len(line)-len(line.lstrip())]
    return ''.join(lines[:index+1]+[indentation+LAYOUT_PASS+','+newline]+lines[index+1:])


def _clone(function, namespace):
    if not isinstance(function, FunctionType) or hasattr(function, '__wrapped__'):
        raise TypeError('Only original undecorated ordinary functions are cloned')
    return FunctionType(function.__code__, namespace, function.__name__,
        function.__defaults__, function.__closure__)


def source_gate(args):
    # The old gate reads the original host source and saved proof. Its final
    # pure pass-shape check is rebound; no live host/BERT admission is called.
    namespace = dict(vars(base), remove_linear_fusion=add_layout_optimization)
    checked = _clone(base.source_gate, namespace)(args)
    checked['candidate_configuration'] = dict(original_linear_fusion_retained=True,
        pass_added=LAYOUT_PASS, added_count=1, immediately_after=PROPAGATION_PASS,
        other_pass_order_unchanged=True, original_failure_preserved=True,
        prior_default_layout_crash_risk_retained=True)
    return checked


def native_command(args, out, checked):
    namespace = dict(vars(base), remove_linear_fusion=add_layout_optimization)
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
        result['prior_default_layout_crash_risk_retained'] = True
        if result.get('status') != 'ok':result['model_compiled_accepted'] = False
    base.durable_json(path, result)


def _execute_tree():
    """Change two recording sites in the trusted original supervisor body."""
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
    spans[0].value = 'saved-math-original-fusion-layout-16384-continuation'
    removed.arg = 'pass_added';removed.value.value = LAYOUT_PASS
    count.arg = 'added_count'
    return ast.fix_missing_locations(tree)


def _namespace():
    namespace = dict(vars(base), MODULE=MODULE, ROUND=ROUND, MODE=MODE, VARIANT=VARIANT,
        source_gate=source_gate, native_command=native_command, _snapshot=_snapshot,
        durable_json=durable_json)
    # Only terminal persistence is redirected. Every arithmetic/native checker
    # and the original initializer contextmanager remain the original objects.
    namespace['original'] = SimpleNamespace(**dict(vars(base.original), durable_json=durable_json))
    namespace['_registry'] = _clone(base._registry, namespace)
    exec(compile(_execute_tree(), str(Path(__file__).resolve())+'::original-execute', 'exec'), namespace)
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
