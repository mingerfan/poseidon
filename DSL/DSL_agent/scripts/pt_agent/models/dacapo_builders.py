"""User-requested DaCapo CNN fixtures, imported from the local upstream tree."""

import importlib
import importlib.util
from contextlib import contextmanager
import sys
from types import ModuleType

from pt_agent.paths import DACAPO_UPSTREAM


VARIANTS = {
    'alexnet_silu_avgpool': ('AlexNet', 'alexnet', 'alexNet_silu_avgpool_model', False, False),
    'alexnet_relu_maxpool': ('AlexNet', 'alexnet', 'alexNet_relu_maxpool_model', True, True),
    'vgg16_silu_avgpool': ('VGG16', 'vgg16', 'vgg16_silu_avgpool_model', False, False),
    'vgg16_relu_maxpool': ('VGG16', 'vgg16', 'vgg16_relu_maxpool_model', True, True),
    'mobilenet_silu': ('MobileNet', 'mobilenet', 'mobileNet_silu_model', False, False),
    'mobilenet_relu': ('MobileNet', 'mobilenet', 'mobileNet_relu_model', True, False),
    'squeezenet_silu_avgpool': ('SqueezeNet', 'squeezenet', 'squeezeNet_silu_avgpool_model', False, False),
    'squeezenet_relu_maxpool': ('SqueezeNet', 'squeezenet', 'squeezeNet_relu_maxpool_model', True, True),
    'resnet20_relu': ('ResNet', 'resnet20', 'resnet20_relu_model', True, False),
}


TORCHVISION_ERROR = 'torchvision 未安装，此桩只用于满足未被使用的导入'


class _TorchvisionStub(ModuleType):
    def __getattr__(self, name):
        raise RuntimeError(TORCHVISION_ERROR)


@contextmanager
def _torchvision_imports():
    try:
        importlib.import_module('torchvision')
    except ModuleNotFoundError as exc:
        if exc.name != 'torchvision':
            raise
        names = ('torchvision', 'torchvision.datasets', 'torchvision.transforms')
        previous = {name: sys.modules.get(name) for name in names}
        stubs = {name: _TorchvisionStub(name) for name in names}
        # Import metadata and these two import bindings are the only attributes
        # supplied. Any use of datasets/transforms functionality fails explicitly.
        for stub in stubs.values():
            stub.__path__ = []
            stub.__file__ = '<unavailable torchvision>'
            stub.__spec__ = None
        stubs['torchvision'].datasets = stubs['torchvision.datasets']
        stubs['torchvision'].transforms = stubs['torchvision.transforms']
        sys.modules.update(stubs)
        try:
            yield
        finally:
            for name, before in previous.items():
                if before is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = before
    else:
        yield


def _load_definition(source, family):
    definition = importlib.util.spec_from_file_location('pt_agent_upstream_'+family, source)
    module = importlib.util.module_from_spec(definition)
    with _torchvision_imports():
        definition.loader.exec_module(module)
    return module


def _variant(module, relu, maxpool):
    import torch.nn as nn
    for name, child in list(module.named_children()):
        if relu and isinstance(child, nn.SiLU):
            setattr(module, name, nn.ReLU())
        elif maxpool and isinstance(child, nn.AvgPool2d):
            setattr(module, name, nn.MaxPool2d(child.kernel_size, child.stride, child.padding,
                                              dilation=1, ceil_mode=child.ceil_mode))
        else:
            _variant(child, relu, maxpool)


def _load(name):
    import torch
    family, factory, filename, relu, maxpool = VARIANTS[name]
    source = DACAPO_UPSTREAM/'python/poly/poly/models'/(family+'.py')
    module = _load_definition(source, family)
    model = getattr(module, factory)()
    # These are the two upstream fixture architectures requested by U3a;
    # their state_dict names and dimensions are identical.
    _variant(model, relu, maxpool)
    state = torch.load(DACAPO_UPSTREAM/'examples/data'/filename, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.eval()


def alexnet_silu_avgpool():
    return _load('alexnet_silu_avgpool')


def alexnet_relu_maxpool():
    return _load('alexnet_relu_maxpool')


def vgg16_silu_avgpool():
    return _load('vgg16_silu_avgpool')


def vgg16_relu_maxpool():
    return _load('vgg16_relu_maxpool')


def mobilenet_silu():
    return _load('mobilenet_silu')


def mobilenet_relu():
    return _load('mobilenet_relu')


def squeezenet_silu_avgpool():
    return _load('squeezenet_silu_avgpool')


def squeezenet_relu_maxpool():
    return _load('squeezenet_relu_maxpool')


def resnet20_relu():
    return _load('resnet20_relu')
