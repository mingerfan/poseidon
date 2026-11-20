"""R16's CPU-only loading of three approved official Orbit Torch definitions."""

import ast
import builtins
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType


_PREFIX = 'pt_agent.experiments.r13_common_models:'
MODEL_RECORDS = {
    _PREFIX + 'resnet20_silu': dict(source_name='ResNetSiLU.py', factory='resnet20silu',
        definitions=('_weights_init', 'LambdaLayer', 'BasicBlock', 'ResNetSiLU', 'resnet20silu')),
    _PREFIX + 'vgg16_silu_avgpool': dict(source_name='VGG16SiLU.py', factory='vgg16silu',
        definitions=('BasicConv2d', 'VGG16SiLU', 'vgg16silu')),
    _PREFIX + 'alexnet_silu_avgpool': dict(source_name='AlexNetSiLU.py', factory='alexnetsilu',
        definitions=('AlexNetSiLU', 'alexnetsilu')),
}
_RECORDER = None


def _span(stage):
    return _RECORDER.span(stage) if _RECORDER is not None else nullcontext()


def load_definition(source, model) -> object:
    """Execute only the approved original class/function definitions, on CPU."""
    import torch

    if model not in MODEL_RECORDS:
        raise ValueError('Unknown approved original model: ' + str(model))
    source = Path(source)
    record = MODEL_RECORDS[model]
    parsed = ast.parse(source.read_text(), filename=str(source))
    selected = []
    for name in record['definitions']:
        found = [node for node in parsed.body
                 if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                 and node.name == name]
        if len(found) != 1 or isinstance(found[0], ast.AsyncFunctionDef):
            raise ValueError('Missing, duplicate or unexpected approved definition: ' + name)
        for child in ast.walk(found[0]):
            if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and child.decorator_list:
                raise ValueError('Unexpected decorator in approved definition: ' + name)
        selected.append(found[0])
    selected.sort(key=lambda node: node.lineno)
    module = ModuleType('pt_agent_r16_orbit_' + record['factory'])
    module.__file__ = str(source.resolve())
    module.__dict__.update(torch=torch, nn=torch.nn, F=torch.nn.functional,
                           init=torch.nn.init, use_cuda=False, __builtins__=builtins.__dict__)
    # These nodes are from the approved official source, never from an LLM response.
    code = compile(ast.Module(body=selected, type_ignores=[]), str(source), 'exec')
    exec(code, module.__dict__)
    return module


def _strict_load(module, state):
    """Keep every original state key and number; use float64 before loading."""
    import torch

    if not isinstance(state, dict) or any(not isinstance(key, str) for key in state):
        raise ValueError('The original state must be a dictionary of named tensors')
    target = module.state_dict()
    if set(target) != set(state):
        raise ValueError('Original state keys differ: missing=' + str(sorted(set(target)-set(state)))
                         + ', extra=' + str(sorted(set(state)-set(target))))
    converted, rows = {}, []
    for name, value in state.items():
        wanted = target[name]
        if (not isinstance(value, torch.Tensor) or value.device.type != 'cpu'
                or tuple(value.shape) != tuple(wanted.shape)):
            raise ValueError('Original state tensor or shape differs: ' + name)
        if not bool(torch.isfinite(value).all()):
            raise ValueError('Original state tensor contains nonfinite values: ' + name)
        if value.is_floating_point():
            if wanted.dtype != torch.float64:
                raise ValueError('The execution module must be double before strict loading: ' + name)
            numeric = value.detach().to(dtype=torch.float64)
            if not torch.equal(numeric.to(dtype=value.dtype), value):
                raise ValueError('Original floating values are not losslessly represented: ' + name)
        else:
            if (value.is_complex() or value.dtype != wanted.dtype
                    or not name.endswith('num_batches_tracked')):
                raise ValueError('Original integer BatchNorm dtype differs: ' + name)
            numeric = value.detach()
        converted[name] = numeric
        rows.append(dict(name=name, shape=list(value.shape), original_dtype=str(value.dtype),
                         target_dtype=str(wanted.dtype), finite=True, numerically_equal=False))
    module.load_state_dict(converted, strict=True)
    loaded = module.state_dict()
    for row in rows:
        name = row['name']
        row['numerically_equal'] = bool(torch.equal(loaded[name], converted[name]))
        if not row['numerically_equal']:
            raise ValueError('Strict loading changed an original state value: ' + name)
    return rows


def _validate_structure(module, model):
    """Inspect the original architecture without an additional forward pass."""
    import torch

    original = module.module if isinstance(module, torch.nn.DataParallel) else module
    errors, convolution_rows, pool_rows, linear_rows = [], [], [], []

    def conv(name, ci, co, kernel, stride=1, padding=1, bias=True):
        value = original.get_submodule(name)
        expected = dict(in_channels=ci, out_channels=co, kernel_size=(kernel, kernel),
                        stride=(stride, stride), padding=(padding, padding), dilation=(1, 1), groups=1)
        if not isinstance(value, torch.nn.Conv2d) or any(getattr(value, k, None) != v for k, v in expected.items()):
            errors.append('Original convolution differs: ' + name)
        if (getattr(value, 'bias', None) is not None) != bias:
            errors.append('Original convolution bias differs: ' + name)
        convolution_rows.append(dict(name=name, **expected, bias=bias))

    def bn(name, size, one_dimensional=False):
        value = original.get_submodule(name)
        kind = torch.nn.BatchNorm1d if one_dimensional else torch.nn.BatchNorm2d
        if (not isinstance(value, kind) or value.num_features != size or value.eps != 1e-5
                or value.momentum != 0.1 or not value.affine or not value.track_running_stats):
            errors.append('Original BatchNorm differs: ' + name)

    def pool(name, kernel, stride, padding=0):
        value = original.get_submodule(name)
        def pair(v):
            return (v, v) if isinstance(v, int) else tuple(v)
        if (not isinstance(value, torch.nn.AvgPool2d) or pair(value.kernel_size) != (kernel, kernel)
                or pair(value.stride) != (stride, stride) or pair(value.padding) != (padding, padding)
                or value.ceil_mode or not value.count_include_pad or value.divisor_override is not None):
            errors.append('Original average pooling differs: ' + name)
        pool_rows.append(dict(name=name, kernel=kernel, stride=stride, padding=padding))

    def linear(name, ni, no):
        value = original.get_submodule(name)
        if (not isinstance(value, torch.nn.Linear) or value.in_features != ni
                or value.out_features != no or value.bias is None):
            errors.append('Original linear head differs: ' + name)
        linear_rows.append(dict(name=name, in_features=ni, out_features=no))

    def silu(name):
        if not isinstance(original.get_submodule(name), torch.nn.SiLU):
            errors.append('Original SiLU differs: ' + name)

    try:
        if model == _PREFIX + 'resnet20_silu':
            conv('conv1', 3, 16, 3, bias=False)
            bn('bn1', 16)
            silu('mish')
            ci = 16
            for stage, co in enumerate((16, 32, 64), 1):
                layer = original.get_submodule('layer' + str(stage))
                if not isinstance(layer, torch.nn.Sequential) or len(layer) != 3:
                    errors.append('Original residual stage differs: layer' + str(stage))
                for index in range(3):
                    prefix = 'layer' + str(stage) + '.' + str(index)
                    stride = 2 if stage > 1 and index == 0 else 1
                    conv(prefix + '.conv1', ci, co, 3, stride=stride, bias=False)
                    conv(prefix + '.conv2', co, co, 3, bias=False)
                    bn(prefix + '.bn1', co)
                    bn(prefix + '.bn2', co)
                    silu(prefix + '.mish')
                    shortcut = original.get_submodule(prefix + '.shortcut')
                    if stride == 2:
                        if type(shortcut).__name__ != 'LambdaLayer' or list(shortcut.parameters()) or list(shortcut.buffers()):
                            errors.append('Original Option A shortcut differs: ' + prefix)
                    elif not isinstance(shortcut, torch.nn.Sequential) or len(shortcut):
                        errors.append('Original identity shortcut differs: ' + prefix)
                    ci = co
            linear('linear', 64, 10)
            expected_activations = 19
        elif model == _PREFIX + 'vgg16_silu_avgpool':
            ci = 3
            for stage, (co, count) in enumerate(((64, 2), (128, 2), (256, 3), (512, 3), (512, 3)), 1):
                for index in range(1, count + 1):
                    prefix = 'conv_' + str(stage) + '_' + str(index)
                    conv(prefix + '.Conv2d', ci, co, 3)
                    bn(prefix + '.bn', co)
                    silu(prefix + '.mish')
                    ci = co
                pool('avgpool_' + str(stage), 2, 2)
            for args in (('fc_1', 512, 256), ('fc_2', 256, 128), ('fc_3', 128, 10)):
                linear(*args)
            bn('bn_1', 128, True)
            silu('mish')
            if not isinstance(original.dp_1, torch.nn.Dropout):
                errors.append('Original unused Dropout differs')
            expected_activations = 15
        elif model == _PREFIX + 'alexnet_silu_avgpool':
            ci = 3
            for index, co in enumerate((96, 256, 384, 384, 256), 1):
                conv('Conv2d_' + str(index), ci, co, 5 if index == 2 else 3,
                     padding=2 if index == 2 else 1)
                bn('bn_' + str(index), co)
                ci = co
            for index in range(1, 4):
                pool('avgpool_' + str(index), 3, 2, 1)
            for args in (('fc_1', 4096, 2048), ('fc_2', 2048, 1024), ('fc_3', 1024, 10)):
                linear(*args)
            silu('mish')
            if not all(isinstance(getattr(original, name), torch.nn.Dropout) for name in ('dp_1', 'dp_2')):
                errors.append('Original unused Dropout differs')
            expected_activations = 7
        else:
            raise ValueError('Unknown approved original model: ' + str(model))
    except (AttributeError, KeyError, IndexError) as exc:
        errors.append('Original module structure is incomplete: ' + str(exc))
        expected_activations = None
    if any(child.training for child in original.modules()):
        errors.append('All original modules must remain in eval mode')
    return dict(passed=not errors, errors=errors, convolutions=convolution_rows,
                pools=pool_rows, linears=linear_rows, activation_occurrences=expected_activations,
                definition_body_preserved=True, use_cuda=False, evaluated_without_forward=True)


def build_model(model, source, state) -> tuple[object, dict]:
    import torch

    with _span('orbit_model_construct'):
        definition = load_definition(source, model)
        module = getattr(definition, MODEL_RECORDS[model]['factory'])()
        if model == _PREFIX + 'resnet20_silu':
            module = torch.nn.DataParallel(module)
    with _span('orbit_cpu_double'):
        module.cpu().double()
    with _span('orbit_constants'):
        extra_state = []
    with _span('orbit_strict_load'):
        rows = _strict_load(module, state)
    with _span('orbit_eval'):
        module.eval()
    structure = _validate_structure(module, model)
    return module, dict(passed=structure['passed'], strict_load=True, original_state=rows,
                        extra_state=extra_state, structure=structure,
                        state_key_prefix='module.' if isinstance(module, torch.nn.DataParallel) else '',
                        source=str(Path(source).resolve()), factory=MODEL_RECORDS[model]['factory'],
                        definition_names=list(MODEL_RECORDS[model]['definitions']), use_cuda=False,
                        scope='Original Torch definitions and three plaintext probes; full Orbit trace pending')
