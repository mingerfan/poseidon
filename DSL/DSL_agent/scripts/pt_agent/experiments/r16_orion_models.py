"""R16 plain Orion adapters for the three unchanged R13 model definitions.

The caller supplies the checkpoint state.  Importing this module does not import
Torch or Orion, load weights, initialize a scheme, or construct a model.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext


_PREFIX = "pt_agent.experiments.r13_common_models:"
RESNET = _PREFIX + "resnet20_silu"
VGG = _PREFIX + "vgg16_silu_avgpool"
ALEX = _PREFIX + "alexnet_silu_avgpool"
_MODELS = (RESNET, VGG, ALEX)
_RECORDER = None


def _span(stage):
    return _RECORDER.span(stage) if _RECORDER is not None else nullcontext()


def _runtime():
    import torch
    import orion.nn as on
    from orion.core import scheme

    return torch, on, scheme


def _require_model(model):
    if model not in _MODELS:
        raise ValueError(f"unsupported R16 Orion model: {model!r}")


def _extra_spec(model):
    """Return the complete, ordered implementation-constant inventory."""
    _require_model(model)
    if model == RESNET:
        return [
            {
                "name": "module.layer2.0.shortcut.weight",
                "shape": [32, 16, 1, 1],
                "role": "option_a_selector",
                "unused_by_plain_forward": False,
                "in_channels": 16,
            },
            {
                "name": "module.layer3.0.shortcut.weight",
                "shape": [64, 32, 1, 1],
                "role": "option_a_selector",
                "unused_by_plain_forward": False,
                "in_channels": 32,
            },
            {
                "name": "module._r16_avgpool.weight",
                "shape": [1, 1, 8, 8],
                "role": "avgpool_placeholder",
                "unused_by_plain_forward": True,
                "fill": 1.0 / 64.0,
            },
        ]
    size, pools = (2, 5) if model == VGG else (3, 3)
    return [
        {
            "name": f"avgpool_{index}.weight",
            "shape": [1, 1, size, size],
            "role": "avgpool_placeholder",
            "unused_by_plain_forward": True,
            "fill": 1.0 / (size * size),
        }
        for index in range(1, pools + 1)
    ]


def _option_a_weight(in_channels):
    """Create the fixed channel padding and stride-two selector weights."""
    if type(in_channels) is not int or in_channels <= 0 or in_channels % 2:
        raise ValueError("Option A requires a positive even input-channel count")
    torch, _, _ = _runtime()
    weight = torch.zeros(
        (2 * in_channels, in_channels, 1, 1), dtype=torch.float64,
        device="cpu", requires_grad=False,
    )
    for channel in range(in_channels):
        weight[in_channels // 2 + channel, channel, 0, 0] = 1.0
    return weight


def _extra_state(model):
    torch, _, _ = _runtime()
    extras = {}
    for item in _extra_spec(model):
        if item["role"] == "option_a_selector":
            tensor = _option_a_weight(item["in_channels"])
        else:
            tensor = torch.full(
                item["shape"], item["fill"], dtype=torch.float64,
                device="cpu", requires_grad=False,
            )
        extras[item["name"]] = tensor
    return extras


def _verify_extra_values(model, extras):
    torch, _, _ = _runtime()
    specs = _extra_spec(model)
    if set(extras) != {item["name"] for item in specs}:
        raise ValueError("extra state differs from the fixed R16 inventory")
    for item in specs:
        actual = extras[item["name"]]
        if item["role"] == "option_a_selector":
            expected = _option_a_weight(item["in_channels"])
        else:
            expected = torch.full(item["shape"], item["fill"], dtype=torch.float64, device="cpu")
        if (
            not isinstance(actual, torch.Tensor) or actual.dtype != torch.float64
            or actual.device.type != "cpu" or actual.requires_grad
            or list(actual.shape) != item["shape"] or not torch.equal(actual, expected)
        ):
            raise ValueError(f"implementation constant differs from R16: {item['name']}")


def _model_classes():
    """Define native classes lazily; callers may inspect or test small blocks."""
    torch, on, _ = _runtime()

    class BasicBlock(on.Module):
        expansion = 1

        def __init__(self, in_planes, planes, stride=1):
            super().__init__()
            self.conv1 = on.Conv2d(
                in_planes, planes, 3, stride=stride, padding=1, bias=False,
            )
            self.bn1 = on.BatchNorm2d(planes)
            self.conv2 = on.Conv2d(
                planes, planes, 3, stride=1, padding=1, bias=False,
            )
            self.bn2 = on.BatchNorm2d(planes)
            self.mish = on.SiLU()
            self.shortcut = torch.nn.Sequential()
            if stride != 1 or in_planes != planes:
                if stride != 2 or planes != 2 * in_planes or in_planes % 2:
                    raise ValueError("R16 only permits the original Option A transition")
                self.shortcut = on.Conv2d(
                    in_planes, planes, 1, stride=2, padding=0, bias=False,
                )
            self._r16_add = on.Add()

        def forward(self, x):
            out = self.mish(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out = self._r16_add(out, self.shortcut(x))
            out = self.mish(out)
            return out

    class ResNet(on.Module):
        def __init__(self):
            super().__init__()
            self.in_planes = 16
            self.conv1 = on.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
            self.bn1 = on.BatchNorm2d(16)
            self.layer1 = self._make_layer(16, 3, stride=1)
            self.layer2 = self._make_layer(32, 3, stride=2)
            self.layer3 = self._make_layer(64, 3, stride=2)
            self.linear = on.Linear(64, 10)
            self.mish = on.SiLU()
            self._r16_avgpool = on.AvgPool2d(8, stride=8)
            self._r16_flatten = on.Flatten()

        def _make_layer(self, planes, num_blocks, stride):
            layers = []
            for block_stride in [stride] + [1] * (num_blocks - 1):
                layers.append(BasicBlock(self.in_planes, planes, block_stride))
                self.in_planes = planes
            return torch.nn.Sequential(*layers)

        def forward(self, x):
            out = self.mish(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self._r16_avgpool(out)
            out = self._r16_flatten(out)
            out = self.linear(out)
            return out

    class BasicConv2d(on.Module):
        def __init__(self, inCH, outCh, ksize, padding=0, stride=1):
            super().__init__()
            self.Conv2d = on.Conv2d(
                inCH, outCh, ksize, stride=stride, padding=padding,
            )
            self.bn = on.BatchNorm2d(outCh)
            self.mish = on.SiLU()

        def forward(self, x):
            x = self.Conv2d(x)
            x = self.bn(x)
            x = self.mish(x)
            return x

    class VGG16(on.Module):
        def __init__(self):
            super().__init__()
            self.conv_1_1 = BasicConv2d(3, 64, 3, padding=(1, 1))
            self.conv_1_2 = BasicConv2d(64, 64, 3, padding=(1, 1))
            self.avgpool_1 = on.AvgPool2d((2, 2), stride=2)
            self.conv_2_1 = BasicConv2d(64, 128, 3, padding=(1, 1))
            self.conv_2_2 = BasicConv2d(128, 128, 3, padding=(1, 1))
            self.avgpool_2 = on.AvgPool2d((2, 2), stride=2)
            self.conv_3_1 = BasicConv2d(128, 256, 3, padding=(1, 1))
            self.conv_3_2 = BasicConv2d(256, 256, 3, padding=(1, 1))
            self.conv_3_3 = BasicConv2d(256, 256, 3, padding=(1, 1))
            self.avgpool_3 = on.AvgPool2d((2, 2), stride=2)
            self.conv_4_1 = BasicConv2d(256, 512, 3, padding=(1, 1))
            self.conv_4_2 = BasicConv2d(512, 512, 3, padding=(1, 1))
            self.conv_4_3 = BasicConv2d(512, 512, 3, padding=(1, 1))
            self.avgpool_4 = on.AvgPool2d((2, 2), stride=2)
            self.conv_5_1 = BasicConv2d(512, 512, 3, padding=(1, 1))
            self.conv_5_2 = BasicConv2d(512, 512, 3, padding=(1, 1))
            self.conv_5_3 = BasicConv2d(512, 512, 3, padding=(1, 1))
            self.avgpool_5 = on.AvgPool2d((2, 2), stride=2)
            self.fc_1 = on.Linear(512 * 1 * 1, 256)
            self.dp_1 = torch.nn.Dropout()
            self.fc_2 = on.Linear(256, 128)
            self.bn_1 = on.BatchNorm1d(128)
            self.fc_3 = on.Linear(128, 10)
            self.mish = on.SiLU()
            self._r16_flatten = on.Flatten()

        def forward(self, x):
            x = self.conv_1_1(x)
            x = self.conv_1_2(x)
            x = self.avgpool_1(x)
            x = self.conv_2_1(x)
            x = self.conv_2_2(x)
            x = self.avgpool_2(x)
            x = self.conv_3_1(x)
            x = self.conv_3_2(x)
            x = self.conv_3_3(x)
            x = self.avgpool_3(x)
            x = self.conv_4_1(x)
            x = self.conv_4_2(x)
            x = self.conv_4_3(x)
            x = self.avgpool_4(x)
            x = self.conv_5_1(x)
            x = self.conv_5_2(x)
            x = self.conv_5_3(x)
            x = self.avgpool_5(x)
            x = self._r16_flatten(x)
            x = self.fc_1(x)
            x = self.mish(x)
            x = self.fc_2(x)
            x = self.mish(x)
            x = self.bn_1(x)
            x = self.fc_3(x)
            return x

    class AlexNet(on.Module):
        def __init__(self):
            super().__init__()
            self.Conv2d_1 = on.Conv2d(3, 96, 3, padding=1)
            self.bn_1 = on.BatchNorm2d(96)
            self.avgpool_1 = on.AvgPool2d((3, 3), stride=2, padding=1)
            self.Conv2d_2 = on.Conv2d(96, 256, 5, padding=2)
            self.bn_2 = on.BatchNorm2d(256)
            self.avgpool_2 = on.AvgPool2d((3, 3), stride=2, padding=1)
            self.Conv2d_3 = on.Conv2d(256, 384, 3, padding=1)
            self.bn_3 = on.BatchNorm2d(384)
            self.Conv2d_4 = on.Conv2d(384, 384, 3, padding=1)
            self.bn_4 = on.BatchNorm2d(384)
            self.Conv2d_5 = on.Conv2d(384, 256, 3, padding=1)
            self.bn_5 = on.BatchNorm2d(256)
            self.avgpool_3 = on.AvgPool2d((3, 3), stride=2, padding=1)
            self.fc_1 = on.Linear(4 * 4 * 256, 2048)
            self.dp_1 = torch.nn.Dropout()
            self.fc_2 = on.Linear(2048, 1024)
            self.dp_2 = torch.nn.Dropout()
            self.fc_3 = on.Linear(1024, 10)
            self.mish = on.SiLU()
            self._r16_flatten = on.Flatten()

        def forward(self, x):
            x = self.Conv2d_1(x)
            x = self.bn_1(x)
            x = self.mish(x)
            x = self.avgpool_1(x)
            x = self.Conv2d_2(x)
            x = self.bn_2(x)
            x = self.mish(x)
            x = self.avgpool_2(x)
            x = self.Conv2d_3(x)
            x = self.bn_3(x)
            x = self.mish(x)
            x = self.Conv2d_4(x)
            x = self.bn_4(x)
            x = self.mish(x)
            x = self.Conv2d_5(x)
            x = self.bn_5(x)
            x = self.mish(x)
            x = self.avgpool_3(x)
            x = self._r16_flatten(x)
            x = self.mish(self.fc_1(x))
            x = self.mish(self.fc_2(x))
            x = self.fc_3(x)
            return x

    return {
        "BasicBlock": BasicBlock, "BasicConv2d": BasicConv2d,
        "ResNet": ResNet, "VGG16": VGG16, "AlexNet": AlexNet,
    }


def _construct_model(model):
    _require_model(model)
    torch, _, _ = _runtime()
    classes = _model_classes()
    if model == RESNET:
        return torch.nn.DataParallel(classes["ResNet"]())
    return classes["VGG16" if model == VGG else "AlexNet"]()


def _load_state(module, state, extras):
    """Strictly load a separate augmented state into an already double model."""
    torch, _, _ = _runtime()
    if not isinstance(state, Mapping) or not state:
        raise ValueError("original checkpoint state must be a nonempty mapping")
    if not isinstance(extras, Mapping):
        raise ValueError("extra state must be a mapping")
    if any(not isinstance(name, str) for name in state):
        raise ValueError("checkpoint keys must be strings")
    if set(state).intersection(extras):
        raise ValueError("implementation constants overlap original checkpoint keys")
    target = module.state_dict()
    expected = set(state) | set(extras)
    if set(target) != expected:
        missing = sorted(expected - set(target))
        unexpected = sorted(set(target) - expected)
        raise ValueError(f"exact state-key mismatch: missing={missing}, unexpected={unexpected}")

    augmented = {}
    original_inventory = []
    integer_dtypes = {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}
    for name, source in state.items():
        if not isinstance(source, torch.Tensor):
            raise ValueError(f"original state {name} is not a tensor")
        destination = target[name]
        if source.layout != torch.strided or list(source.shape) != list(destination.shape):
            raise ValueError(f"original state shape/layout mismatch: {name}")
        if destination.device.type != "cpu":
            raise ValueError(f"target state must be on CPU: {name}")
        if source.is_floating_point():
            if destination.dtype != torch.float64:
                raise ValueError(f"target floating state must already be float64: {name}")
            finite = bool(torch.isfinite(source).all().item())
            if not finite:
                raise ValueError(f"nonfinite original state: {name}")
            value = source.detach().to(device="cpu", dtype=torch.float64).clone()
            if not torch.equal(value.to(dtype=source.dtype), source.detach().cpu()):
                raise ValueError(f"original state is not losslessly represented: {name}")
        else:
            if source.dtype not in integer_dtypes or not name.endswith("num_batches_tracked"):
                raise ValueError(f"unexpected original nonfloating state: {name}")
            finite = bool(torch.isfinite(source).all().item())
            # double() leaves buffers integral; retain the checkpoint's exact
            # integer dtype even when the native constructor used torch.long.
            if name not in dict(module.named_buffers()):
                raise ValueError(f"integer BN state must be a registered buffer: {name}")
            if destination.is_floating_point():
                raise ValueError(f"integer BN target buffer became floating: {name}")
            if destination.dtype != source.dtype:
                owner_name, _, leaf = name.rpartition(".")
                owner = module.get_submodule(owner_name) if owner_name else module
                setattr(owner, leaf, destination.to(dtype=source.dtype))
            value = source.detach().cpu().clone()
        augmented[name] = value
        original_inventory.append({
            "name": name, "shape": list(source.shape),
            "original_dtype": str(source.dtype), "target_dtype": str(value.dtype),
            "finite": finite, "numerically_equal": None,
        })

    parameters = dict(module.named_parameters())
    extra_inventory = []
    for name, source in extras.items():
        if not isinstance(source, torch.Tensor):
            raise ValueError(f"implementation constant {name} is not a tensor")
        destination = target[name]
        if (
            source.dtype != torch.float64 or source.device.type != "cpu"
            or destination.dtype != torch.float64 or destination.device.type != "cpu"
            or list(source.shape) != list(destination.shape)
            or source.requires_grad or not bool(torch.isfinite(source).all().item())
        ):
            raise ValueError(f"invalid float64 implementation constant: {name}")
        if name not in parameters:
            raise ValueError(f"implementation constant is not a native parameter: {name}")
        parameters[name].requires_grad_(False)
        augmented[name] = source.detach().clone()
        selector = name.endswith("shortcut.weight")
        extra_inventory.append({
            "name": name, "shape": list(source.shape),
            "original_dtype": None, "target_dtype": str(source.dtype),
            "dtype": str(source.dtype), "finite": True,
            "numerically_equal": None, "requires_grad": False,
            "role": "option_a_selector" if selector else "avgpool_placeholder",
            "unused_by_plain_forward": not selector,
        })

    incompatible = module.load_state_dict(augmented, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError("strict state load returned incompatible keys")
    loaded = module.state_dict()
    for row in original_inventory + extra_inventory:
        name = row["name"]
        actual = loaded[name]
        expected_value = augmented[name]
        equal = actual.dtype == expected_value.dtype and torch.equal(actual, expected_value)
        if not equal:
            raise ValueError(f"loaded state differs from supplied value: {name}")
        row["numerically_equal"] = True
    for row in extra_inventory:
        if parameters[row["name"]].requires_grad:
            raise ValueError(f"implementation constant still requires gradients: {row['name']}")
    return {
        "strict_load": True, "original_state": original_inventory,
        "extra_state": extra_inventory, "original_keys_preserved": True,
        "original_state_count": len(original_inventory),
        "extra_state_count": len(extra_inventory),
    }


def _structure(model):
    common = {"input_shape": [1, 3, 32, 32], "output_shape": [1, 10],
              "activation": "SiLU", "activation_degree": 31,
              "activation_coefficients_fitted": False, "plain_he_mode": False}
    if model == RESNET:
        common.update({
            "name": "ResNet", "data_parallel": True, "channels": [16, 32, 64],
            "blocks": [3, 3, 3], "head": [64, 10],
            "block_order": ["conv1", "bn1", "mish", "conv2", "bn2", "add_shortcut", "mish"],
            "option_a": {"in_channels": [16, 32], "stride": 2, "kernel": 1,
                         "padding": 0, "bias": False, "trained_projection": False},
            "pool": {"name": "_r16_avgpool", "kernel": [8, 8], "stride": [8, 8]},
            "head_order": ["_r16_avgpool", "_r16_flatten", "linear"],
            "unused_modules": [],
        })
    elif model == VGG:
        common.update({
            "name": "VGG16", "data_parallel": False,
            "stage_channels": [[64, 64], [128, 128], [256, 256, 256],
                               [512, 512, 512], [512, 512, 512]],
            "basic_conv_order": ["Conv2d", "bn", "mish"],
            "convolution_bias": True, "pool_count": 5,
            "pool_kernel": [2, 2], "pool_stride": [2, 2], "pool_padding": [0, 0],
            "head": [512, 256, 128, 10],
            "head_order": ["_r16_flatten", "fc_1", "mish", "fc_2", "mish", "bn_1", "fc_3"],
            "unused_modules": ["dp_1"],
        })
    else:
        common.update({
            "name": "AlexNet", "data_parallel": False,
            "channels": [96, 256, 384, 384, 256],
            "convolution_kernels": [3, 5, 3, 3, 3],
            "convolution_bias": True, "pool_after_convolution": [1, 2, 5],
            "pool_kernel": [3, 3], "pool_stride": [2, 2], "pool_padding": [1, 1],
            "head": [4096, 2048, 1024, 10],
            "head_order": ["_r16_flatten", "fc_1", "mish", "fc_2", "mish", "fc_3"],
            "unused_modules": ["dp_1", "dp_2"],
        })
    return common


def _scheme_status(scheme):
    present = scheme is not None and hasattr(scheme, "backend") and hasattr(scheme, "traced")
    if not present or scheme.backend is not None or scheme.traced is not None:
        raise ValueError("R16 requires only Orion's uninitialized lightweight Scheme")
    return {
        "lightweight_scheme_present": True, "backend_is_none": True,
        "traced_is_none": True, "init_scheme_calls": 0,
        "backend_initializations": 0, "keygen_runs": 0,
        "fit_calls": 0, "compile_runs": 0,
    }


def build_model(model, state) -> tuple[object, dict]:
    """Build the fixed original model with native, uninitialized Orion ops."""
    _require_model(model)
    _, _, scheme = _runtime()
    _scheme_status(scheme)
    with _span("orion_model_construct"):
        module = _construct_model(model)
    # Keep this order: constructors may start with float32 parameters; all
    # implementation constants, particularly 1/9, are first made as float64.
    with _span("orion_cpu_double"):
        module.cpu().double()
    with _span("orion_constants"):
        extras = _extra_state(model)
        _verify_extra_values(model, extras)
    with _span("orion_strict_load"):
        report = _load_state(module, state, extras)
    specs = _extra_spec(model)
    rows = {row["name"]: row for row in report["extra_state"]}
    for item in specs:
        row = rows[item["name"]]
        if any(row[key] != item[key] for key in ("shape", "role", "unused_by_plain_forward")):
            raise ValueError(f"extra state inventory mismatch: {item['name']}")
        for key in ("fill", "in_channels"):
            if key in item:
                row[key] = item[key]
    with _span("orion_eval"):
        # Native on.Module.eval() returns None; never assign its return value.
        module.eval()
        all_he_mode_false = all(
            not item.he_mode for item in module.modules() if hasattr(item, "he_mode")
        )
        if not all_he_mode_false or any(item.training for item in module.modules()):
            raise ValueError("R16 model did not enter plain evaluation mode")
    report.update(_scheme_status(scheme))
    structure = _structure(model)
    structure["passed"] = True
    report.update({
        "passed": True, "model": model, "structure": structure,
        "state_key_prefix": "module." if model == RESNET else "",
        "all_he_mode_false": True, "all_training_false": True,
        "module_inventory": [
            {"name": name, "type": type(item).__name__,
             "class_module": type(item).__module__,
             "he_mode": bool(item.he_mode) if hasattr(item, "he_mode") else None,
             "training": bool(item.training)}
            for name, item in module.named_modules()
        ],
        "scope": "native_orion_plain_structure_and_state_only",
        "packing_verified": False, "fhe_activation_scheme_frozen": False,
    })
    return module, report
